import AppKit
import ArgumentParser
import Compute
import Metal

@main
struct PixelSort: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "GPU-accelerated pixel sorting for glitch art"
    )

    @Argument(help: "Input image path")
    var input: String

    @Argument(help: "Output image path")
    var output: String

    @Option(name: .shortAndLong, help: "Sort key: brightness, hue, saturation, red, green, blue")
    var key: SortKeyOption = .brightness

    @Option(name: .shortAndLong, help: "Lower brightness threshold (0.0–1.0)")
    var lower: Float = 0.1

    @Option(name: .shortAndLong, help: "Upper brightness threshold (0.0–1.0)")
    var upper: Float = 0.9

    @Flag(name: .shortAndLong, help: "Sort descending (reverse)")
    var descending: Bool = false

    @Option(name: .shortAndLong, help: "Gamma applied to sorted pixels (Unity-style composite)")
    var gamma: Float = 1.0

    @Option(name: .shortAndLong, help: "Clamp maximum sortable span length (default: image width)")
    var maxSpan: Int?

    @Flag(name: .long, help: "Invert the threshold mask")
    var invertMask: Bool = false

    mutating func run() throws {
        let device = MTLCreateSystemDefaultDevice()!
        let compute = try Compute(device: device)

        // Load image into a Metal texture
        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: self.output)

        guard let nsImage = NSImage(contentsOf: inputURL),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else {
            throw ValidationError("Could not load image at \(input)")
        }

        let width = cgImage.width
        let height = cgImage.height

        let rgbaDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        rgbaDesc.usage = [.shaderRead, .shaderWrite]

        let texA = device.makeTexture(descriptor: rgbaDesc)!
        let texB = device.makeTexture(descriptor: rgbaDesc)!
        texA.label = "original"
        texB.label = "output"

        let sortedTex = device.makeTexture(descriptor: rgbaDesc)!
        sortedTex.label = "sorted"

        let maskDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r8Uint,
            width: width,
            height: height,
            mipmapped: false
        )
        maskDesc.usage = [.shaderRead, .shaderWrite]
        let maskTex = device.makeTexture(descriptor: maskDesc)!
        maskTex.label = "mask"

        let spanDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Uint,
            width: width,
            height: height,
            mipmapped: false
        )
        spanDesc.usage = [.shaderRead, .shaderWrite]
        let spanTex = device.makeTexture(descriptor: spanDesc)!
        spanTex.label = "spans"

        let valDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r16Float,
            width: width,
            height: height,
            mipmapped: false
        )
        valDesc.usage = [.shaderRead, .shaderWrite]
        let valTex = device.makeTexture(descriptor: valDesc)!
        valTex.label = "sortValues"

        // Upload pixel data
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: &pixelData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw ValidationError("Failed to create CGContext")
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        texA.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )

        // Load shaders
        let shaderSource = try String(contentsOf: Bundle.module.url(forResource: "Shaders", withExtension: "metal")!, encoding: .utf8)
        let library = ShaderLibrary.source(shaderSource)

        var createMaskPipeline = try compute.makePipeline(function: library.createMask)
        var clearSpanPipeline = try compute.makePipeline(function: library.clearSpanBuffer)
        var identifySpansPipeline = try compute.makePipeline(function: library.identifySpans)
        var rgbToValPipeline = try compute.makePipeline(function: library.rgbToSortValue)
        var pixelSortPipeline = try compute.makePipeline(function: library.pixelSortSpan)
        var compositePipeline = try compute.makePipeline(function: library.composite)

        var params = Params(
            width: UInt32(width),
            height: UInt32(height),
            sortKey: UInt32(key.metalValue),
            lowerThreshold: lower,
            upperThreshold: upper,
            reverseSorting: descending ? 1 : 0,
            gamma: gamma,
            maxSpanLength: UInt32(maxSpan ?? width),
            invertMask: invertMask ? 1 : 0
        )
        let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<Params>.stride, options: .storageModeShared)!

        // 1) Mask
        createMaskPipeline.arguments.colorTex = .texture(texA)
        createMaskPipeline.arguments.maskTex = .texture(maskTex)
        createMaskPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: createMaskPipeline, width: width, height: height)

        // 2) Clear span buffer
        clearSpanPipeline.arguments.spanTex = .texture(spanTex)
        clearSpanPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: clearSpanPipeline, width: width, height: height)

        // 3) Identify spans (1 thread per row: dispatch width=1)
        identifySpansPipeline.arguments.maskTex = .texture(maskTex)
        identifySpansPipeline.arguments.spanTex = .texture(spanTex)
        identifySpansPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: identifySpansPipeline, width: 1, height: height)

        // 4) Sort values
        rgbToValPipeline.arguments.colorTex = .texture(texA)
        rgbToValPipeline.arguments.valTex = .texture(valTex)
        rgbToValPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: rgbToValPipeline, width: width, height: height)

        // 5) Sort each span into sortedTex
        pixelSortPipeline.arguments.colorTex = .texture(texA)
        pixelSortPipeline.arguments.valTex = .texture(valTex)
        pixelSortPipeline.arguments.spanTex = .texture(spanTex)
        pixelSortPipeline.arguments.sortedTex = .texture(sortedTex)
        pixelSortPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: pixelSortPipeline, width: width, height: height)

        // 6) Composite only masked pixels into output texB
        compositePipeline.arguments.maskTex = .texture(maskTex)
        compositePipeline.arguments.sortedTex = .texture(sortedTex)
        compositePipeline.arguments.originalTex = .texture(texA)
        compositePipeline.arguments.outTex = .texture(texB)
        compositePipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: compositePipeline, width: width, height: height)

        // Read back from output
        var outputData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        texB.getBytes(
            &outputData,
            bytesPerRow: bytesPerRow,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )

        // Save output
        guard let outCtx = CGContext(
            data: &outputData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ),
        let outCGImage = outCtx.makeImage()
        else {
            throw ValidationError("Failed to create output image")
        }

        let nsOutImage = NSBitmapImageRep(cgImage: outCGImage)
        let ext = outputURL.pathExtension.lowercased()
        let fileType: NSBitmapImageRep.FileType = ext == "jpg" || ext == "jpeg" ? .jpeg : .png
        guard let data = nsOutImage.representation(using: fileType, properties: [:]) else {
            throw ValidationError("Failed to encode output image")
        }
        try data.write(to: outputURL)

        print("Pixel-sorted image saved to \(self.output) (\(width)×\(height))")
    }
}

// MARK: - Helpers

struct Params {
    var width: UInt32
    var height: UInt32
    var sortKey: UInt32
    var lowerThreshold: Float
    var upperThreshold: Float
    var reverseSorting: UInt32
    var gamma: Float
    var maxSpanLength: UInt32
    var invertMask: UInt32
}

enum SortKeyOption: String, ExpressibleByArgument, CaseIterable {
    case brightness, hue, saturation, red, green, blue

    var metalValue: Int {
        switch self {
        case .brightness:  return 0
        case .hue:         return 1
        case .saturation:  return 2
        case .red:         return 3
        case .green:       return 4
        case .blue:        return 5
        }
    }
}
