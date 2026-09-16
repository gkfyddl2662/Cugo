# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository now contains the first deterministic deal baseline shared by the CPU reference path and CUDA:

- 48 base cards packed into a 64-bit mask (`0..47`)
- month/slot extraction using shifts and masks
- popcount / first-set-card helpers with CPU and CUDA implementations
- CUDA `__fns` based n-th set-card selection
- deterministic `SplitMix64` usable from both host and device code
- unbiased bounded random sampling for drawing without replacement
- base-48 initial deal: 10 cards per player, 8 floor cards, 20-card stock
- fixed-seed CPU fingerprints and randomized deal invariants
- CUDA CPU/GPU differential validation over 65,536 deals
- one-thread-per-game CUDA deal throughput baseline with SoA output

The stock is intentionally represented as a remaining-card bitmask plus RNG state instead of a pre-shuffled per-game array. Future draws can therefore be sampled lazily without storing a 20-card order for every resident game.

### 48-card scope

Hangame's mobile Shin Matgo guide describes a 50-card game with 10 cards per player and 8 open floor cards. The current milestone deliberately models only the 48 standard cards. Bonus-card IDs and their rule effects will be added later, so `deal_base_48` must not be treated as the final exact 50-card game setup.

Scoring, legal actions, captures, Go/Stop decisions, and the self-play state machine are not implemented yet.

## Build and test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCUGO_ENABLE_CUDA=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

If CMake cannot find a CUDA compiler, it still builds and runs the CPU core test and prints that the CUDA validation targets are disabled.

On Windows, `scripts/windows_bootstrap.ps1` discovers the installed MSVC/CMake/Ninja toolchain, performs a clean CUDA build, and runs the correctness suite.

## CUDA deal benchmark

After a CUDA build:

```text
build\cugo_cuda_deal_test.exe --benchmark
```

The benchmark generates 1,048,576 independent base-48 deals per launch, sweeps 128/256/512 threads per block, and reports games/s, sampled cards/s, theoretical occupancy, registers/thread, and local bytes/thread.
