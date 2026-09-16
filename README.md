# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository contains a deterministic base-48 reference engine and CUDA differential path with:

- 48 standard cards packed into a 64-bit mask (`0..47`)
- month/slot extraction and 4-bit month masks
- a fixed engine slot convention for the 48 standard hwatu identities
- bright/animal/ribbon/pi metadata, including godori and ribbon-set masks
- fixed double-pi cards and optional Gukjin animal -> double-pi conversion
- host/device base scoring for bright, animal, ribbon, and pi groups
- host/device bit operations and deterministic RNG
- base-48 initial deal: 10 cards per player, 8 floor cards, 20-card stock
- field-major SoA resident state
- `PLAY -> DRAW -> RESOLVE` turn phases
- pending played/drawn cards so special capture rules can inspect the complete turn before mutating the floor
- normal unique capture, ppuk, jjok, ttadak, sweep detection, and ppuk ownership/capture metadata
- explicit `ResolveChoices` for ambiguous two-card same-month floor selections
- transactional resolve behavior: choice/error statuses never partially mutate state
- fixed CPU examples plus randomized card-partition/scoring invariants
- CUDA CPU/GPU differential validation over 65,536 deals, score masks, stock transitions, and turn resolves
- one-thread-per-game CUDA throughput baselines
- CUDA Graph as the current repeated-phase scheduling baseline

The stock is represented as a remaining-card bitmask plus RNG state instead of a pre-shuffled per-game array. This keeps the resident hot state compact and avoids storing a per-game stock array.

### Measured RTX 5080 baseline

The measured deal and draw kernels use zero local memory and reach 100% theoretical occupancy. Longer block-size measurements showed 256 and 512 threads/block effectively tied, so 256 remains the working default for the light phases.

The capture resolver currently uses 72 registers/thread and zero local memory. Across three 1,048,576-game x 256-iteration measurements, 512 threads/block was consistently the fastest tested resolver configuration despite 33.3% theoretical occupancy, reaching roughly 7.5-7.9 billion resolve attempts/s. Phase-specific block sizes are therefore allowed; 512 is the current `RESOLVE` performance baseline while 256 remains the light-phase baseline.

For the 21-node initialization + 20-draw sequence, CUDA Graph replay improved measured end-to-end wall time by about 1.11x-1.14x across three runs and reduced host submission time by roughly 65x-70x. CUDA Graph is therefore the current repeated-phase scheduling baseline; it can still be replaced later if real gameplay divergence makes another scheduler measurably better.

### Rule correctness boundary

Hangame's official guide is the source of truth. The pinned facts, implemented subset, internal card-ID convention, and intentionally deferred behavior are documented in `docs/rules.md`.

The complete mobile Shin Matgo ruleset uses 50 cards. The current engine deliberately models only the 48 standard cards, so bonus-card IDs and their rule effects still need to be added later.

The guide explicitly marks ppuk and jjok as having a last-card exception, but the current referenced page does not specify the replacement transition. Those exact final-stock-flip patterns return an unsupported status instead of silently applying a generic Go-Stop convention.

Pi stealing is not applied yet. The engine can now identify legal pi/double-pi cards and compute pi units, but the exact Hangame transfer-selection behavior is still deliberately deferred rather than guessed.

## Build and test

```bash
cmake -S . -B build -DBUILD_TESTING=ON -DCUGO_ENABLE_CUDA=ON
cmake --build build --config Release -j
ctest --test-dir build -C Release --output-on-failure
```

If CMake cannot find a CUDA compiler, it still builds and runs the CPU tests and prints that CUDA validation targets are disabled.

On Windows, `scripts/windows_bootstrap.ps1` discovers the installed MSVC/CMake/Ninja toolchain, performs a clean CUDA build, and runs the correctness suite.

## CUDA benchmarks

After a CUDA build:

```text
build\cugo_cuda_deal_test.exe --benchmark
build\cugo_cuda_state_test.exe --benchmark
build\cugo_cuda_turn_test.exe --benchmark
```

The deal benchmark generates 1,048,576 independent base-48 deals per launch, sweeps 128/256/512 threads per block, and uses 256 timed launches per block size to reduce short-run WDDM/clock noise.

The resident-state benchmark keeps 1,048,576 game states in field-major SoA memory. It measures the 20 stock-draw kernels over a long window and compares manual host submission against CUDA Graph replay for the full 21-node initialization + 20-draw sequence. Graph setup cost is reported separately from replay.

The turn-resolve benchmark prepares 1,048,576 deterministic first-turn states, times only the hot `load SoA -> resolve_turn -> status write -> store SoA` path, and sweeps 128/256/512 threads per block over 256 timed launches. It reports registers/thread, local bytes/thread, theoretical occupancy, games/s, average kernel time, and the resolved/choice-required status mix. Correctness-only invariant checks are deliberately excluded from the timed resolver kernel.
