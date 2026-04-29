#include <metal_stdlib>
using namespace metal;

enum SortKey : uint {
    Brightness = 0,
    Hue        = 1,
    Saturation = 2,
    Red        = 3,
    Green      = 4,
    Blue       = 5,
};

// --- helpers ---
static inline float luminance(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));  // RGB -> greyscale
}

static inline float hue(float3 c) {
    float cmax  = max(c.r, max(c.g, c.b));
    float cmin  = min(c.r, min(c.g, c.b));
    float delta = cmax - cmin;
    if (delta < 1e-6) return 0.0;
    float h;
    if (cmax == c.r)      h = fmod((c.g - c.b) / delta, 6.0);
    else if (cmax == c.g) h = (c.b - c.r) / delta + 2.0;
    else                  h = (c.r - c.g) / delta + 4.0;
    h /= 6.0;
    if (h < 0.0) h += 1.0;
    return h;
}

static inline float saturation(float3 c) {
    float cmax  = max(c.r, max(c.g, c.b));
    float cmin  = min(c.r, min(c.g, c.b));
    if (cmax < 1e-6) return 0.0;
    return (cmax - cmin) / cmax;
}

static inline float sort_value(float3 rgb, uint key) {
    switch (SortKey(key)) {
        case SortKey::Brightness:  return luminance(rgb);
        case SortKey::Hue:         return hue(rgb);
        case SortKey::Saturation:  return saturation(rgb);
        case SortKey::Red:         return rgb.r;
        case SortKey::Green:       return rgb.g;
        case SortKey::Blue:        return rgb.b;
    }
    return luminance(rgb);
}

// --- parameters passed from CPU ---
struct Params {
    uint  width;
    uint  height;
    uint  sortKey;        // SortKey enum
    float lowerThreshold; // luminance low
    float upperThreshold; // luminance high
    uint  reverseSorting; // 0 = normal, 1 = reverse
    float gamma;          // output gamma
    uint  maxSpanLength;  // clamp span length (safety)
    uint  invertMask;     // 0/1
};

// --- create mask (0/1) ---

kernel void createMask(
    texture2d<float, access::read>  colorTex [[texture(0)]],
    texture2d<uint, access::write>  maskTex  [[texture(1)]],
    constant Params &params                  [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    float3 rgb = saturate(colorTex.read(gid).rgb);
    float l = luminance(rgb);
    bool inRange = (l >= params.lowerThreshold) && (l <= params.upperThreshold);
    uint m = inRange ? 1u : 0u;
    if (params.invertMask) m = 1u - m;
    maskTex.write(m, gid);
}

// --- sort-by scalars ---

kernel void buildSortKeys(
    texture2d<float, access::read>   colorTex   [[texture(0)]],
    texture2d<float, access::write> sortKeyTex  [[texture(1)]],
    constant Params &params                       [[buffer(0)]],
    uint2 gid                                     [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    float3 rgb = saturate(colorTex.read(gid).rgb);
    float v = sort_value(rgb, params.sortKey);
    sortKeyTex.write(float4(v, 0.0f, 0.0f, 0.0f), gid);
}

// --- span descriptor for indirect dispatch ---

struct SpanDescriptor {
    uint row;
    uint startX;
    uint length;
};

// --- identify spans and build span buffer ---

kernel void identifySpans(
    texture2d<uint, access::read>  maskTex     [[texture(0)]],
    constant Params &params                    [[buffer(0)]],
    device SpanDescriptor *spanBuffer          [[buffer(1)]],
    device atomic_uint *spanCount              [[buffer(2)]],
    uint2 gid                                  [[thread_position_in_grid]]
) {
    uint row = gid.y;
    if (gid.x != 0 || row >= params.height) return;

    uint pos = 0;
    uint spanStart = 0;
    uint spanLength = 0;
    uint spanLimit = max(1u, params.maxSpanLength);

    while (pos < params.width) {
        uint m = maskTex.read(uint2(pos, row)).x;
        pos += 1;

        if (m == 0 || spanLength >= spanLimit) {
            if (spanLength != 0) {
                uint outLen = (m == 1u) ? (spanLength + 1u) : spanLength;
                uint idx = atomic_fetch_add_explicit(spanCount, 1, memory_order_relaxed);
                spanBuffer[idx] = SpanDescriptor { row, spanStart, outLen };
            }
            spanStart = pos;
            spanLength = 0;
        } else {
            spanLength += 1;
        }
    }

    if (spanLength != 0 && spanStart < params.width) {
        uint idx = atomic_fetch_add_explicit(spanCount, 1, memory_order_relaxed);
        spanBuffer[idx] = SpanDescriptor { row, spanStart, spanLength };
    }
}

// --- prepare arguments from span count ---

struct IndirectArgs {
    uint threadgroupsX;
    uint threadgroupsY;
    uint threadgroupsZ;
};

kernel void prepareIndirectArgs(
    device atomic_uint *spanCount      [[buffer(0)]],
    device IndirectArgs *indirectArgs  [[buffer(1)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    uint count = atomic_load_explicit(spanCount, memory_order_relaxed);
    indirectArgs->threadgroupsX = count;
    indirectArgs->threadgroupsY = 1;
    indirectArgs->threadgroupsZ = 1;
}

// --- sort pixels within a span ---

constant uint TG_SIZE = 1024;
constant uint MAX_SPAN = 2048;  // 1024 threads x 2 elements each

kernel void pixelSortSpan(
    texture2d<float, access::read>  colorTex    [[texture(0)]],
    texture2d<float, access::read>  sortKeyTex  [[texture(1)]],
    texture2d<float, access::write> sortedTex   [[texture(2)]],
    constant Params &params                     [[buffer(0)]],
    device const SpanDescriptor *spanBuffer     [[buffer(1)]],
    uint tgid                                   [[threadgroup_position_in_grid]],
    uint tid                                    [[thread_index_in_threadgroup]]
) {
    threadgroup float  keys[MAX_SPAN];
    threadgroup ushort orig[MAX_SPAN];

    SpanDescriptor span = spanBuffer[tgid];
    uint row    = span.row;
    uint startX = span.startX;
    uint len    = min(span.length, params.width - startX);
    len = min(len, max(1u, params.maxSpanLength));
    len = min(len, MAX_SPAN);

    uint n = 1;
    while (n < len) n <<= 1;

    float sentinel = params.reverseSorting ? 2.0f : -2.0f;

    for (uint i = tid; i < n; i += TG_SIZE) {
        if (i < len) {
            keys[i] = sortKeyTex.read(uint2(startX + i, row)).x;
            orig[i] = ushort(i);
        } else {
            keys[i] = sentinel;
            orig[i] = ushort(i);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint halfN = n >> 1;
    for (uint k = 2; k <= n; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint t = tid; t < halfN; t += TG_SIZE) {
                uint lo = t + (t & ~(j - 1));  // insert 0-bit at position of j
                uint hi = lo + j;
                bool ascending = ((lo & k) == 0) == bool(params.reverseSorting);
                if ((ascending && keys[lo] > keys[hi]) ||
                    (!ascending && keys[lo] < keys[hi])) {
                    float  tmpK = keys[lo]; keys[lo] = keys[hi]; keys[hi] = tmpK;
                    ushort tmpI = orig[lo]; orig[lo] = orig[hi]; orig[hi] = tmpI;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < len; i += TG_SIZE) {
        float4 c = colorTex.read(uint2(startX + orig[i], row));
        sortedTex.write(c, uint2(startX + i, row));
    }
}

// --- composite sorted pixels onto original ---

kernel void composite(
    texture2d<uint, access::read>  maskTex     [[texture(0)]],
    texture2d<float, access::read> sortedTex   [[texture(1)]],
    texture2d<float, access::read> originalTex [[texture(2)]],
    texture2d<float, access::write> outTex     [[texture(3)]],
    constant Params &params                    [[buffer(0)]],
    uint2 gid                                  [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;

    float4 c = originalTex.read(gid);
    if (maskTex.read(gid).x == 1u) {
        float4 s = sortedTex.read(gid);
        c = pow(abs(s), float4(params.gamma));
    }
    outTex.write(c, gid);
}
