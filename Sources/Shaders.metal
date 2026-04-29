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

// --- prepare indirect dispatch arguments from span count ---

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

// --- sort pixels within a span (one thread per span, indirect dispatch) ---

constant uint MAX_LOCAL_SPAN = 2048;

kernel void pixelSortSpan(
    texture2d<float, access::read>  colorTex    [[texture(0)]],
    texture2d<float, access::write> sortedTex   [[texture(1)]],
    constant Params &params                     [[buffer(0)]],
    device const SpanDescriptor *spanBuffer     [[buffer(1)]],
    uint gid                                    [[thread_position_in_grid]]
) {
    SpanDescriptor span = spanBuffer[gid];
    uint row = span.row;
    uint x = span.startX;
    uint spanLength = min(span.length, params.width - x);
    spanLength = min(spanLength, max(1u, params.maxSpanLength));
    spanLength = min(spanLength, MAX_LOCAL_SPAN);

    float cache[MAX_LOCAL_SPAN];
    for (uint k = 0; k < spanLength; ++k) {
        float3 rgb = saturate(colorTex.read(uint2(x + k, row)).rgb);
        cache[k] = sort_value(rgb, params.sortKey);
    }

    float minValue = cache[0];
    float maxValue = cache[0];
    uint minIndex = 0;
    uint maxIndex = 0;

    uint steps = (spanLength / 2) + 1;
    for (uint i = 0; i < steps; ++i) {
        for (uint j = 1; j < spanLength; ++j) {
            float v = cache[j];
            if (v >= 0.0f && v <= 1.0f) {
                if (v < minValue) { minValue = v; minIndex = j; }
                if (maxValue < v) { maxValue = v; maxIndex = j; }
            }
        }

        uint dstMin = params.reverseSorting ? i : (spanLength - i - 1);
        uint dstMax = params.reverseSorting ? (spanLength - i - 1) : i;

        float4 cMin = colorTex.read(uint2(x + minIndex, row));
        float4 cMax = colorTex.read(uint2(x + maxIndex, row));

        sortedTex.write(cMin, uint2(x + dstMin, row));
        sortedTex.write(cMax, uint2(x + dstMax, row));

        cache[minIndex] = 2.0f;
        cache[maxIndex] = -2.0f;
        minValue = 1.0f;
        maxValue = -1.0f;
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
