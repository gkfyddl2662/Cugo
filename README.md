# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository now contains the first deterministic resident-state baseline shared by the CPU reference path and CUDA:

- 48 base cards packed into a 64-bit mask (`0..47`)
- month/slot extraction using shifts and masks
- popcount / first-set-card helpers with CPU and CUDA implementations
- CUDA `__fns` based n-th set-card selection
- deterministic `SplitMix64` usable from both host and device code
- unbiased bounded random sampling for drawing without replacement
- base-48 initial deal: 10 cards per player, 8 floor cards, 20-card stock
- fixed-seed CPU fingerprints and randomized deal invariants
- CUDA CPU/GPU differential validation over 65,536 deals
- field-major SoA resident state for hand/floor/stock/RNG
- deterministic stock-draw transition shared by CPU and CUDA
- CUDA differential validation across 65,536 games x 20 stock transitions
- one-thread-per-game deal and resident-transition throughput baselines

The stock is represented as a remaining-card bitmask plus RNG state instead of a pre-shuffled per-game array. Each stock transition samples one rank from the remaining set, clears that bit, persists the RNG state, and emits the drawn card for a future resolve phase. This keeps the resident hot state compact and avoids a 20-card per-game stock array.

The measured RTX 5080 deal baseline reached full theoretical occupancy with 24 registers/thread and zero local memory per thread. 128 and 256 threads/block were effectively tied, while 512 threads/block was slower, so resident transition kernels currently use 256 threads/block as the working baseline.

### 48-card scope

Hangame's mobile Shin Matgo guide describes a 50-card game with 10 cards per player and 8 open floor cards. The current milestone deliberately models only the 48 standard cards. Bonus-card IDs and their rule effects will be added later, so `deal_base_48` must not be treated as the final exact 50-card game setup.

Scoring, legal actions, captures, Go/Stop decisions, and the full self-play state machine are not implemented yet.

## Build and test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCUGO_ENABLE_CUDA=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

If CMake cannot find a CUDA compiler, it still builds and runs the CPU core test and prints that the CUDA validation targets are disabled.

On Windows, `scripts/windows_bootstrap.ps1` discovers the installed MSVC/CMake/Ninja toolchain, performs a clean CUDA build, and runs the correctness suite.

## CUDA benchmarks

After a CUDA build:

```text
build\cugo_cuda_deal_test.exe --benchmark
build\cugo_cuda_state_test.exe --benchmark
```

The deal benchmark generates 1,048,576 independent base-48 deals per launch and sweeps 128/256/512 threads per block.

The resident-state benchmark keeps 1,048,576 game states in field-major SoA memory and executes the full 20-card stock sequence as 20 separate 256-thread transition kernels. It reports transitions/s, completed stock rounds/s, average transition-kernel time, theoretical occupancy, registers/thread, and local bytes/thread. The separate launches intentionally expose the current launch/phase-boundary cost before CUDA Graph or persistent scheduling is introduced.
