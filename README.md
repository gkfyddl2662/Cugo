# Cugo

High-performance tabula-rasa self-play research project for Korean Go-Stop / Shin Matgo.

## Current milestone

The repository contains a deterministic CPU reference path and CUDA differential path with:

- 48 standard cards packed into IDs `0..47`
- Shin Matgo bonus cards at IDs `48` (2-pi) and `49` (3-pi)
- separate standard-deck and 50-card Shin Matgo masks inside one `uint64_t`
- a fixed engine slot convention for the 48 standard hwatu identities
- bright/animal/ribbon/pi metadata, including godori and ribbon-set masks
- fixed double-pi cards, 2-pi/3-pi bonus scoring, and optional Gukjin animal -> double-pi conversion
- host/device base scoring for bright, animal, ribbon, and pi groups
- deterministic base-48 and raw 50-card deal primitives
- raw 50-card layout: 10 cards per player, 8 initial floor cards, 22-card stock
- initial-floor bonus collection before the first player's turn
- hand-bonus replacement draws plus a Shin Matgo pi-steal event count
- stock bonus chaining that keeps flipped bonus cards pending until the eventual standard-card resolve
- field-major SoA resident state for the current base-48 turn engine
- `PLAY -> DRAW -> RESOLVE` turn phases
- normal unique capture, ppuk, jjok, ttadak, sweep detection, and ppuk ownership/capture metadata
- explicit `ResolveChoices` for ambiguous two-card same-month floor selections
- transactional resolve behavior: choice/error statuses never partially mutate state
- fixed CPU examples plus randomized card-partition/scoring/bonus invariants
- CUDA CPU/GPU differential validation over deals, score masks, bonus primitives, stock transitions, and turn resolves
- one-thread-per-game CUDA throughput baselines
- CUDA Graph as the current repeated-phase scheduling baseline

The 50-card work deliberately keeps bonus cards as physical cards in the same 64-bit mask rather than widening the state representation. `kFullDeckMask` remains the legacy 48-standard-card mask for base-48 regression tests; new exact-Shin-Matgo work uses `kShinMatgoDeckMask`.

### Bonus-rule integration boundary

Hangame's official guide states that two bonus cards are used in Shin Matgo, they count as 2-pi and 3-pi, an initial floor bonus is automatically taken before the first player starts, a hand bonus gives a replacement card before the stock flip and another play opportunity, and a stock-flipped bonus grants another flip. If a ppuk occurs after bonus flips, those bonus cards must be placed on the floor with the ppuk cards.

The current bonus primitives implement the physical 50-card deck, scoring, raw deal, initial-floor collection, hand replacement, and stock bonus chain. The stock-flip helper returns `pending_bonus_mask` instead of prematurely adding those cards to captured cards. This is intentional: the next resolver milestone must associate pending bonus cards with a specific ppuk stack so they can later be captured with that stack without ambiguity.

The official mode guide also states that playing a bonus card in Shin Matgo takes one opponent pi. The current hand-bonus primitive returns `pi_steal_count=1`; physical pi transfer is still deferred until the exact selection policy for multiple eligible opponent pi cards is pinned down.

### Measured RTX 5080 baseline

The measured deal and draw kernels use zero local memory and reach 100% theoretical occupancy. Longer block-size measurements showed 256 and 512 threads/block effectively tied, so 256 remains the working default for the light phases.

The capture resolver currently uses 72 registers/thread and zero local memory. Across three 1,048,576-game x 256-iteration measurements, 512 threads/block was consistently the fastest tested resolver configuration despite 33.3% theoretical occupancy, reaching roughly 7.5-7.9 billion resolve attempts/s. Phase-specific block sizes are therefore allowed; 512 is the current `RESOLVE` performance baseline while 256 remains the light-phase baseline.

For the 21-node initialization + 20-draw sequence, CUDA Graph replay improved measured end-to-end wall time by about 1.11x-1.14x across three runs and reduced host submission time by roughly 65x-70x. CUDA Graph is therefore the current repeated-phase scheduling baseline; it can still be replaced later if real gameplay divergence makes another scheduler measurably better.

### Rule correctness boundary

Hangame's official guide is the source of truth. The pinned facts, implemented subset, internal card-ID convention, and intentionally deferred behavior are documented in `docs/rules.md`.

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
