// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "pixel-sort",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/schwa/Compute", from: "0.0.6"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
    ],
    targets: [
        .executableTarget(
            name: "pixel-sort",
            dependencies: [
                .product(name: "Compute", package: "Compute"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources",
            resources: [.copy("Shaders.metal")]
        ),
    ]
)
