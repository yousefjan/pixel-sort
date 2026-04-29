import AppKit
import ArgumentParser
import Compute
import Metal

@main
struct PixelSort: ParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Metal-accelerated pixel sorting library"
    )

    @Argument(help: "Input image path")
    var input: String

    @Argument(help: "Output image path")
    var output: String

    @Option(name: .shortAndLong, help: "Sort key: brightness, hue, saturation, R, G, B")
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

        let inputURL = URL(fileURLWithPath: input)
        let outputURL = URL(fileURLWithPath: self.output)

        guard let nsImage = NSImage(contentsOf: inputURL),
            let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil)
        else {
            throw ValidationError("Could not load image")
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

        let sortKeyDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float,
            width: width,
            height: height,
            mipmapped: false
        )
        sortKeyDesc.usage = [.shaderRead, .shaderWrite]
        let sortKeyTex = device.makeTexture(descriptor: sortKeyDesc)!
        sortKeyTex.label = "sortKeys"

        // Span descriptor buffer (max one span per pixel is a safe upper bound)
        let maxSpans = width * height
        let spanBufferSize = maxSpans * MemoryLayout<SpanDescriptor>.stride
        let spanBuffer = device.makeBuffer(length: spanBufferSize, options: .storageModeShared)!
        spanBuffer.label = "spanBuffer"

        // Atomic counter for number of spans found
        let counterBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        counterBuffer.label = "spanCount"

        // Indirect dispatch arguments buffer (3 x uint32)
        let indirectArgsBuffer = device.makeBuffer(
            length: MemoryLayout<UInt32>.stride * 3, options: .storageModeShared)!
        indirectArgsBuffer.label = "indirectArgs"

        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard
            let ctx = CGContext(
                data: &pixelData,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            )
        else {
            throw ValidationError("Failed to create CGContext")
        }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        texA.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )

        let shaderSource = try String(
            contentsOf: Bundle.module.url(forResource: "Shaders", withExtension: "metal")!,
            encoding: .utf8)
        let library = ShaderLibrary.source(shaderSource)

        var createMaskPipeline = try compute.makePipeline(function: library.createMask)
        var buildSortKeysPipeline = try compute.makePipeline(function: library.buildSortKeys)
        var identifySpansPipeline = try compute.makePipeline(function: library.identifySpans)
        var prepareIndirectArgsPipeline = try compute.makePipeline(
            function: library.prepareIndirectArgs)
        let pixelSortPipeline = try compute.makePipeline(function: library.pixelSortSpan)
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
        let paramBuffer = device.makeBuffer(
            bytes: &params, length: MemoryLayout<Params>.stride, options: .storageModeShared)!

        counterBuffer.contents().assumingMemoryBound(to: UInt32.self).pointee = 0

        buildSortKeysPipeline.arguments.colorTex = .texture(texA)
        buildSortKeysPipeline.arguments.sortKeyTex = .texture(sortKeyTex)
        buildSortKeysPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: buildSortKeysPipeline, width: width, height: height)

        createMaskPipeline.arguments.colorTex = .texture(texA)
        createMaskPipeline.arguments.maskTex = .texture(maskTex)
        createMaskPipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: createMaskPipeline, width: width, height: height)

        identifySpansPipeline.arguments.maskTex = .texture(maskTex)
        identifySpansPipeline.arguments.params = .buffer(paramBuffer)
        identifySpansPipeline.arguments.spanBuffer = .buffer(spanBuffer)
        identifySpansPipeline.arguments.spanCount = .buffer(counterBuffer)
        try compute.run(pipeline: identifySpansPipeline, width: 1, height: height)

        prepareIndirectArgsPipeline.arguments.spanCount = .buffer(counterBuffer)
        prepareIndirectArgsPipeline.arguments.indirectArgs = .buffer(indirectArgsBuffer)
        try compute.run(pipeline: prepareIndirectArgsPipeline, width: 1, height: 1)

        // Sort each span into sortedTex (one threadgroup per span, bitonic sort)
        try compute.task(label: "pixelSort") { task in
            try task.run { dispatch in
                let enc = dispatch.commandEncoder
                enc.setComputePipelineState(pixelSortPipeline.computePipelineState)
                enc.setTexture(texA, index: 0)
                enc.setTexture(sortKeyTex, index: 1)
                enc.setTexture(sortedTex, index: 2)
                enc.setBuffer(paramBuffer, offset: 0, index: 0)
                enc.setBuffer(spanBuffer, offset: 0, index: 1)
                enc.dispatchThreadgroups(
                    indirectBuffer: indirectArgsBuffer,
                    indirectBufferOffset: 0,
                    threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1)
                )
            }
        }

        // composite only masked pixels into output texB
        compositePipeline.arguments.maskTex = .texture(maskTex)
        compositePipeline.arguments.sortedTex = .texture(sortedTex)
        compositePipeline.arguments.originalTex = .texture(texA)
        compositePipeline.arguments.outTex = .texture(texB)
        compositePipeline.arguments.params = .buffer(paramBuffer)
        try compute.run(pipeline: compositePipeline, width: width, height: height)

        var outputData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        texB.getBytes(
            &outputData,
            bytesPerRow: bytesPerRow,
            from: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0
        )

        guard
            let outCtx = CGContext(
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

struct SpanDescriptor {
    var row: UInt32
    var startX: UInt32
    var length: UInt32
}

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
        case .brightness: return 0
        case .hue: return 1
        case .saturation: return 2
        case .red: return 3
        case .green: return 4
        case .blue: return 5
        }
    }
}
