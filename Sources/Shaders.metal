#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------
// Span-based pixel sorting (Unity Pixel-Sorting port)
//
// Pipeline:
// 1) createMask: mark pixels whose luminance is within thresholds
// 2) clearSpanBuffer
// 3) identifySpans: for each row, write span length at span start pixel
// 4) rgbToSortValue: compute per-pixel sort value (R/G/B/L/S/H)
// 5) pixelSortSpan: for each span start, sort pixels within the span
// 6) composite: apply sorted pixels only where mask==1
//
// This matches the semantics of the reference Unity compute shader:
// contiguous masked regions are sorted independently.
// --------------------------------------------------------------------

enum SortKey : uint {
    Brightness = 0,
    Hue        = 1,
    Saturation = 2,
    Red        = 3,
    Green      = 4,
    Blue       = 5,
};

// ---- helpers -------------------------------------------------------

static inline float luminance(float3 rgb) {
    return dot(rgb, float3(0.299, 0.587, 0.114));
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

// ---- parameters passed from the CPU side ---------------------------

struct Params {
    uint  width;
    uint  height;
    uint  sortKey;        // SortKey enum
    float lowerThreshold; // luminance low
    float upperThreshold; // luminance high
    uint  reverseSorting; // 0 = normal, 1 = reverse
    float gamma;          // output gamma (Unity applies pow(abs(sorted), gamma))
    uint  maxSpanLength;  // clamp span length (safety)
    uint  invertMask;     // 0/1
};

// ---- create mask (0/1) ---------------------------------------------

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

// ---- clear span buffer ---------------------------------------------

kernel void clearSpanBuffer(
    texture2d<uint, access::write> spanTex [[texture(0)]],
    constant Params &params                [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    spanTex.write(0u, gid);
}

// ---- identify spans (horizontal only) ------------------------------
// One thread per row: write span length at each span start.

kernel void identifySpans(
    texture2d<uint, access::read>  maskTex [[texture(0)]],
    texture2d<uint, access::write> spanTex [[texture(1)]],
    constant Params &params                [[buffer(0)]],
    uint2 gid                              [[thread_position_in_grid]]
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
            // Write at span start. Mirror Unity behavior: if we hit an unmasked pixel,
            // spanLength is current count; if we hit limit while still masked, include current pixel.
            if (spanLength != 0) {
                uint outLen = (m == 1u) ? (spanLength + 1u) : spanLength;
                spanTex.write(outLen, uint2(spanStart, row));
            }
            spanStart = pos;
            spanLength = 0;
        } else {
            spanLength += 1;
        }
    }

    if (spanLength != 0 && spanStart < params.width) {
        spanTex.write(spanLength, uint2(spanStart, row));
    }
}

// ---- per-pixel sort value ------------------------------------------

kernel void rgbToSortValue(
    texture2d<float, access::read>  colorTex [[texture(0)]],
    texture2d<half, access::write>  valTex   [[texture(1)]],
    constant Params &params                  [[buffer(0)]],
    uint2 gid                                [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;
    float3 rgb = saturate(colorTex.read(gid).rgb);
    float v = sort_value(rgb, params.sortKey);
    valTex.write(half(v), gid);
}

// ---- sort pixels within a span -------------------------------------
// One thread per pixel, but only span starts do work.
// Writes sorted pixels into `sortedTex` for the span region.

constant uint MAX_LOCAL_SPAN = 2048;

kernel void pixelSortSpan(
    texture2d<float, access::read>  colorTex  [[texture(0)]],
    texture2d<half, access::read>   valTex    [[texture(1)]],
    texture2d<uint, access::read>   spanTex   [[texture(2)]],
    texture2d<float, access::write> sortedTex [[texture(3)]],
    constant Params &params                   [[buffer(0)]],
    uint2 gid                                 [[thread_position_in_grid]]
) {
    if (gid.x >= params.width || gid.y >= params.height) return;

    uint spanLength = spanTex.read(gid).x;
    if (spanLength == 0) return;

    spanLength = min(spanLength, params.width - gid.x);
    spanLength = min(spanLength, max(1u, params.maxSpanLength));
    spanLength = min(spanLength, MAX_LOCAL_SPAN);

    // Cache sort values for this span.
    float cache[MAX_LOCAL_SPAN];
    for (uint k = 0; k < spanLength; ++k) {
        cache[k] = float(valTex.read(uint2(gid.x + k, gid.y)).x);
    }

    float minValue = cache[0];
    float maxValue = cache[0];
    uint minIndex = 0;
    uint maxIndex = 0;

    uint steps = (spanLength / 2) + 1;
    for (uint i = 0; i < steps; ++i) {
        for (uint j = 1; j < spanLength; ++j) {
            float v = cache[j];
            // Unity checks `v == saturate(v)` to ignore sentinels; equivalent is 0..1.
            if (v >= 0.0f && v <= 1.0f) {
                if (v < minValue) { minValue = v; minIndex = j; }
                if (maxValue < v) { maxValue = v; maxIndex = j; }
            }
        }

        uint dstMin = params.reverseSorting ? i : (spanLength - i - 1);
        uint dstMax = params.reverseSorting ? (spanLength - i - 1) : i;

        float4 cMin = colorTex.read(uint2(gid.x + minIndex, gid.y));
        float4 cMax = colorTex.read(uint2(gid.x + maxIndex, gid.y));

        sortedTex.write(cMin, uint2(gid.x + dstMin, gid.y));
        sortedTex.write(cMax, uint2(gid.x + dstMax, gid.y));

        cache[minIndex] = 2.0f;
        cache[maxIndex] = -2.0f;
        minValue = 1.0f;
        maxValue = -1.0f;
    }
}

// ---- composite sorted pixels onto original -------------------------

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
