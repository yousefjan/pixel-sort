#include <metal_stdlib>
using namespace metal;

// --------------------------------------------------------------------
// Bitonic-sort pixel-sorting kernel for glitch art
//
// Each row of the image is treated as an independent array.
// Pixels whose brightness falls within [lower, upper] form a "sortable
// mask."  The kernel runs successive bitonic merge passes over every
// row; masked-out pixels stay in place while masked-in pixels are
// compared-and-swapped by their sort key (brightness / hue / etc.).
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

static inline float brightness(float4 c) {
    return dot(c.rgb, float3(0.2126, 0.7152, 0.0722));
}

static inline float hue(float4 c) {
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

static inline float saturation(float4 c) {
    float cmax  = max(c.r, max(c.g, c.b));
    float cmin  = min(c.r, min(c.g, c.b));
    if (cmax < 1e-6) return 0.0;
    return (cmax - cmin) / cmax;
}

static inline float sort_value(float4 c, uint key) {
    switch (SortKey(key)) {
        case SortKey::Brightness:  return brightness(c);
        case SortKey::Hue:         return hue(c);
        case SortKey::Saturation:  return saturation(c);
        case SortKey::Red:         return c.r;
        case SortKey::Green:       return c.g;
        case SortKey::Blue:        return c.b;
    }
    return brightness(c);
}

// ---- parameters passed from the CPU side ---------------------------

struct Params {
    uint  width;          // image width  (= row length)
    uint  height;         // image height (= number of rows)
    uint  blockSize;      // bitonic block size  (power of 2, doubles each outer pass)
    uint  subBlockSize;   // comparison distance (power of 2, halves each inner pass)
    uint  sortKey;        // which channel to sort by (see SortKey enum)
    float lowerThreshold; // brightness lower bound for mask
    float upperThreshold; // brightness upper bound for mask
    uint  descending;     // 0 = ascending, 1 = descending
};

// ---- bitonic compare-and-swap kernel -------------------------------
//
// Dispatch with threads = (width/2) * height.
// Each thread handles one compare-swap pair in the current sub-pass.

kernel void bitonicSortStep(
    texture2d<float, access::read>  inputTexture  [[texture(0)]],
    texture2d<float, access::write> outputTexture [[texture(1)]],
    constant Params &params                       [[buffer(0)]],
    uint2 gid                                     [[thread_position_in_grid]]
) {
    uint pairIndex = gid.x;   // which pair within this row
    uint row       = gid.y;

    if (row >= params.height) return;
    if (pairIndex >= params.width / 2) return;

    // Determine the two indices to compare.
    uint blockSize    = params.blockSize;
    uint subBlockSize = params.subBlockSize;

    // Position within block and sub-block
    uint blockIndex = pairIndex / (subBlockSize / 2);
    uint offset     = pairIndex % (subBlockSize / 2);

    uint leftIdx  = blockIndex * subBlockSize + offset;
    uint rightIdx = leftIdx + subBlockSize / 2;

    if (leftIdx >= params.width || rightIdx >= params.width) {
        // Out of bounds — copy left pixel through unchanged.
        if (leftIdx < params.width) {
            outputTexture.write(inputTexture.read(uint2(leftIdx, row)), uint2(leftIdx, row));
        }
        return;
    }

    float4 leftPixel  = inputTexture.read(uint2(leftIdx, row));
    float4 rightPixel = inputTexture.read(uint2(rightIdx, row));

    // Threshold mask: only sort pixels whose brightness is in range.
    float leftBri  = brightness(leftPixel);
    float rightBri = brightness(rightPixel);
    bool leftIn    = leftBri  >= params.lowerThreshold && leftBri  <= params.upperThreshold;
    bool rightIn   = rightBri >= params.lowerThreshold && rightBri <= params.upperThreshold;

    if (leftIn && rightIn) {
        float leftVal  = sort_value(leftPixel,  params.sortKey);
        float rightVal = sort_value(rightPixel, params.sortKey);

        // Direction: ascending within even blocks, descending within odd
        // (standard bitonic pattern), then flip if user wants descending.
        bool ascending = ((leftIdx / blockSize) % 2 == 0);
        if (params.descending) ascending = !ascending;

        bool doSwap = ascending ? (leftVal > rightVal) : (leftVal < rightVal);
        if (doSwap) {
            float4 tmp = leftPixel;
            leftPixel  = rightPixel;
            rightPixel = tmp;
        }
    }

    outputTexture.write(leftPixel,  uint2(leftIdx,  row));
    outputTexture.write(rightPixel, uint2(rightIdx, row));
}

// ---- simple copy kernel for pixels not touched by a compare-swap ---
//
// After each step we need untouched pixels carried forward.  Instead of
// a separate pass we do a full-image copy first, then the sort step
// overwrites the pairs it touches.  This kernel does that copy.

kernel void copyTexture(
    texture2d<float, access::read>  src [[texture(0)]],
    texture2d<float, access::write> dst [[texture(1)]],
    uint2 gid                           [[thread_position_in_grid]]
) {
    if (gid.x >= src.get_width() || gid.y >= src.get_height()) return;
    dst.write(src.read(gid), gid);
}
