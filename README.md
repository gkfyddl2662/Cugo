# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository now contains a deterministic resident-state and turn-phase baseline shared by the CPU reference path and CUDA:

- 48 base cards packed into a 64-bit mask (`0..47`)
- month/slot extraction and 4-bit month masks
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
- CUDA Graph scheduling for repeated phase sequences
- `TurnState48` / `TurnStateSoA48` frame with `PLAY -> DRAW -> RESOLVE`
- pending played/drawn cards, captures, actor, turn index, and per-month ppuk ownership metadata
- CPU partition invariants that account for pending cards without losing or duplicating any of the 48 base cards
- CUDA differential validation for the turn frame over 65,536 games

The stock is represented as a remaining-card bitmask plus RNG state instead of a pre-shuffled per-game array. Each stock transition samples one rank from the remaining set, clears that bit, and persists the RNG state.

The turn frame deliberately delays capture mutation until the `RESOLVE` phase. The played card and the stock-drawn card are kept as explicit pending cards while the floor remains unchanged. This preserves enough context for exact Shin Matgo resolution of ppuk, ttadak, jjok, sweep, bombs, grenades, and ambiguous same-month choices instead of approximating those rules with a generic Go-Stop resolver.

### Measured RTX 5080 baseline

The measured deal and draw kernels use zero local memory and reach 100% theoretical occupancy. Longer block-size measurements showed 256 and 512 threads/block effectively tied, so 256 remains the working default to preserve headroom as the real resolver grows.

For the 21-node initialization + 20-draw sequence, CUDA Graph replay improved measured end-to-end wall time by about 1.11x-1.14x across three runs and reduced host submission time by roughly 65x-70x. CUDA Graph is therefore the current repeated-phase scheduling baseline; it can still be replaced later if real gameplay divergence makes another scheduler measurably better.

### 48-card scope

Hangame's mobile Shin Matgo guide describes a 50-card game with 10 cards per player and 8 open floor cards. The current milestone deliberately models only the 48 standard cards. Bonus-card IDs and their rule effects will be added later, so `deal_base_48` must not be treated as the final exact 50-card game setup.

The authoritative rule facts pinned so far, source URLs, and the intentionally deferred resolver details are documented in `docs/rules.md`.

Scoring, capture resolution, Go/Stop decisions, special play actions such as bombs/grenades, and the full self-play state machine are not implemented yet.

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

The deal benchmark generates 1,048,576 independent base-48 deals per launch, sweeps 128/256/512 threads per block, and uses 256 timed launches per block size to reduce short-run WDDM/clock noise. It reports average kernel time, games/s, sampled cards/s, theoretical occupancy, registers/thread, and local bytes/thread.

The resident-state benchmark keeps 1,048,576 game states in field-major SoA memory. It measures the 20 stock-draw kernels over a long window and compares manual host submission against CUDA Graph replay for the full 21-node initialization + 20-draw sequence. Graph setup cost is reported separately from replay.
