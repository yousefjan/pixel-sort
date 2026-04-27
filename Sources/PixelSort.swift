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

    @Flag(name: .shortAndLong, help: "Sort descending")
    var descending: Bool = false

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

        let desc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        desc.usage = [.shaderRead, .shaderWrite]

        let texA = device.makeTexture(descriptor: desc)!
        let texB = device.makeTexture(descriptor: desc)!
        texA.label = "texA"
        texB.label = "texB"

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

        var copyPipeline = try compute.makePipeline(function: library.copyTexture)
        var sortPipeline = try compute.makePipeline(function: library.bitonicSortStep)

        // Bitonic sort requires log2(nextPow2(width)) outer passes
        let n = nextPowerOf2(width)

        // Ping-pong between texA and texB
        var readTex = texA
        var writeTex = texB

        var blockSize: Int = 2
        while blockSize <= n {
            var subBlockSize = blockSize
            while subBlockSize >= 2 {
                // Copy readTex → writeTex so untouched pixels carry forward
                copyPipeline.arguments.src = .texture(readTex)
                copyPipeline.arguments.dst = .texture(writeTex)
                try compute.run(pipeline: copyPipeline, width: width, height: height)

                // Run the bitonic compare-swap step
                var params = BitonicParams(
                    width: UInt32(width),
                    height: UInt32(height),
                    blockSize: UInt32(blockSize),
                    subBlockSize: UInt32(subBlockSize),
                    sortKey: UInt32(key.metalValue),
                    lowerThreshold: lower,
                    upperThreshold: upper,
                    descending: descending ? 1 : 0
                )

                sortPipeline.arguments.inputTexture = .texture(readTex)
                sortPipeline.arguments.outputTexture = .texture(writeTex)

                let paramBuffer = device.makeBuffer(bytes: &params, length: MemoryLayout<BitonicParams>.stride, options: .storageModeShared)!
                sortPipeline.arguments.params = .buffer(paramBuffer)

                try compute.run(pipeline: sortPipeline, width: width / 2, height: height)

                // Swap
                let tmp = readTex
                readTex = writeTex
                writeTex = tmp

                subBlockSize /= 2
            }
            blockSize *= 2
        }

        // Read back from readTex (the last write destination after swap)
        var outputData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
        readTex.getBytes(
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

struct BitonicParams {
    var width: UInt32
    var height: UInt32
    var blockSize: UInt32
    var subBlockSize: UInt32
    var sortKey: UInt32
    var lowerThreshold: Float
    var upperThreshold: Float
    var descending: UInt32
}

func nextPowerOf2(_ n: Int) -> Int {
    var v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1
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
