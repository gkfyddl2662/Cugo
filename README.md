# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository currently contains the portable core primitives that future CPU reference and CUDA self-play engines will share:

- 48-card IDs packed into a 64-bit mask (`0..47`)
- month/slot extraction using shifts and masks
- popcount / first-set-card helpers with CPU and CUDA implementations
- deterministic `SplitMix64` usable from both host and device code
- CPU unit tests
- CUDA CPU/GPU differential validation when a CUDA compiler is available

Game-specific card metadata, dealing, legal actions, scoring, and the self-play state machine are intentionally not encoded yet. Those should be added only after the exact Shin Matgo ruleset is pinned down and covered by reference tests.

## Build

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCUGO_ENABLE_CUDA=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

If CMake cannot find a CUDA compiler, it still builds and runs the CPU core test and prints that the CUDA validation target is disabled.
