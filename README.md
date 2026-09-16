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
- stock bonus chaining until a standard card is flipped
- a bonus-aware `TurnState50` with `PLAY -> DRAW -> RESOLVE`
- pending stock bonuses kept transactional until the standard-card resolve is known
- per-bonus ppuk association metadata so floor bonuses stay attached to the ppuk month that owns them
- ppuk creation with pending bonuses and later capture of the ppuk plus its associated bonus cards
- normal unique capture, ppuk, jjok, ttadak, sweep detection, and ppuk ownership/capture metadata
- explicit resolve choices for ambiguous two-card same-month floor selections
- transactional resolve behavior: choice/error statuses never partially mutate state
- fixed CPU examples plus randomized card-partition/scoring/bonus/turn invariants
- CUDA CPU/GPU differential validation over deals, score masks, bonus primitives, base-48 turns, and 50-card bonus-aware turns
- one-thread-per-game CUDA throughput baselines
- CUDA Graph as the current repeated-phase scheduling baseline

The 50-card path keeps all physical cards in the same 64-bit mask. `kFullDeckMask` remains the legacy 48-standard-card mask for regression tests; exact Shin Matgo work uses `kShinMatgoDeckMask`.

### Bonus-aware turn integration

Hangame's official guide states that a stock-flipped bonus grants another flip and, if the eventual standard flip creates ppuk, the bonus card(s) go to the floor with that ppuk. `TurnState50` therefore carries a `pending_bonus_mask` during `RESOLVE` instead of capturing stock bonuses immediately.

If ppuk is created, each pending bonus is moved onto the floor and associated with that ppuk month through compact 12-bit metadata. When the three-card ppuk stack is later captured, its associated bonus card(s) are captured in the same transaction and the association metadata is cleared. This also allows more than one ppuk stack to exist without losing which stack owns a physical bonus card.

Playing a bonus card from hand remains a `PLAY`-phase action: the card is captured, one replacement card is drawn from stock, and the player receives another play opportunity. The action reports `pi_steal_count=1` for Shin Matgo, but physical opponent-pi transfer is still deferred until the exact selection policy for multiple eligible pi cards is pinned down.

The older `TurnState48` engine remains intact as a regression/performance reference while the 50-card path is brought to full rule parity.

### Measured RTX 5080 baseline

The measured deal and draw kernels use zero local memory and reach 100% theoretical occupancy. Longer block-size measurements showed 256 and 512 threads/block effectively tied, so 256 remains the working default for the light phases.

The base-48 capture resolver currently uses 72 registers/thread and zero local memory. Across three 1,048,576-game x 256-iteration measurements, 512 threads/block was consistently the fastest tested resolver configuration despite 33.3% theoretical occupancy, reaching roughly 7.5-7.9 billion resolve attempts/s. Phase-specific block sizes are therefore allowed; 512 is the current `RESOLVE` performance baseline while 256 remains the light-phase baseline.

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

The base-48 turn-resolve benchmark prepares 1,048,576 deterministic first-turn states, times only the hot `load SoA -> resolve_turn -> status write -> store SoA` path, and sweeps 128/256/512 threads per block over 256 timed launches. It reports registers/thread, local bytes/thread, theoretical occupancy, games/s, average kernel time, and the resolved/choice-required status mix. Correctness-only invariant checks are deliberately excluded from the timed resolver kernel.
