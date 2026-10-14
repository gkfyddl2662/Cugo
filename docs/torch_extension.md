# PyTorch CUDA extension baseline

This layer is the first real neural-network integration over the frozen `TorchEnv50` contract.
It lives entirely under `python/` and does not change the C++ engine or its CTest targets.

## GPU-resident API

The JIT extension exposes three operations:

- `create(seeds, first_players) -> states`
  - both inputs are CUDA `int64` vectors.
  - `states` is an opaque CUDA `uint8 [N, STATE_BYTES]` tensor containing `TorchEnv50` objects.
- `observe(states) -> features, legal, players, done, status`
  - `features`: CUDA `float32 [N, 720]` (`FEATURE_COUNT`).
  - `legal`: CUDA `bool [N, 179]` (`ACTION_COUNT`).
  - `players`: CUDA `uint8 [N]` decision player.
  - `done`: CUDA `bool [N]`.
  - `status`: CUDA `int16 [N]`.
- `step(states, actions) -> reward0, done, nagari, status, primary_committed`
  - mutates the opaque states in place.
  - `actions` is CUDA `int64 [N]` in the unified 179-action space.
  - `reward0` is from player 0's perspective and is non-zero on terminal settlement only.

Self-play also uses indexed variants (`observe_indexed`, `step_indexed`, `step_events_indexed`, `clone_indexed`, `branch_indexed`) and two oracle reads for the critic:

- `hands_indexed(states, env_ids) -> int64 [N, 2]`: both players' hand masks.
- `stock_indexed(states, env_ids) -> int64 [N]`: the next 8 stock draws, one card id per byte (0xFF once the stock is empty). Every draw takes the next uniform card of the stock mask from the state's SplitMix64 stream, so `(stock, rng_state)` fixes the order. Over 160k decisions of random play, every later prediction continued the earlier one (cards only left from its front) and no predicted card was ever in a hand.

The kernels use PyTorch's current CUDA stream. Game state never has to cross PCIe during self-play.

## JIT build

`python/cugo_torch50_ext.py` uses `torch.utils.cpp_extension.load()` with the repository `include/` directory and C++20/CUDA C++20 flags. The build cache is stored under `build/torch_extensions/cugo_torch50_ext_v6`; the version suffix changes with the feature contract or the exported operations, so a running trainer that loaded an older build keeps working while a new one compiles.

For the Windows CUDA 13.2 development machine, use a PyTorch CUDA 13.2 build. PyTorch 2.14.0 provides `cu132` wheels.

Example:

```powershell
py -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
py python/torch50_smoke.py --batch 32768 --warmup 1 --iters 3
```

The sequence-policy self-play (`python/cugo_seq_attention50.py`) uses a Triton kernel when Triton is importable (`py -m pip install --no-deps triton-windows==3.8.0.post29` on Windows; it falls back to plain matrix products otherwise). Triton writes its kernel cache to `%USERPROFILE%\.triton`; set `TRITON_CACHE_DIR` when that directory is not writable.

## Baseline network

`python/torch50_smoke.py` creates a deliberately small FP16 policy/value MLP:

- input: 720 floats
- trunk: 256 -> 256 with SiLU
- policy: 179 logits
- value: scalar

The policy logits are masked by the engine-generated 179-way legal mask before `argmax`. The script runs complete games until terminal or nagari and reports end-to-end games/s. This network is not a training architecture decision; it is a transport/inference baseline for measuring the CUDA-env/PyTorch boundary.
