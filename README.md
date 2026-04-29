```bash
swift build -c release
./.build/release/pixel-sort input.png output2.png --lower 0.2 --upper 0.8 --key brightness
```

Performance target: 2ms max for 1080p

Shader optimization bottlenecks:
- VRAM
- Shader code itself
    - Texture sampling (memory bandwidth): precalculate sorting value (SortKey)
    -
