# PyTorch CUDA extension baseline

This layer is the first real neural-network integration over the frozen `TorchEnv50` contract.
It lives entirely under `python/` and does not change the C++ engine or the 33 core CTest targets.

## GPU-resident API

The JIT extension exposes three operations:

- `create(seeds, first_players) -> states`
  - both inputs are CUDA `int64` vectors.
  - `states` is an opaque CUDA `uint8 [N, STATE_BYTES]` tensor containing `TorchEnv50` objects.
- `observe(states) -> features, legal, players, done, status`
  - `features`: CUDA `float32 [N, 496]`.
  - `legal`: CUDA `bool [N, 177]`.
  - `players`: CUDA `uint8 [N]` decision player.
  - `done`: CUDA `bool [N]`.
  - `status`: CUDA `int16 [N]`.
- `step(states, actions) -> reward0, done, nagari, status, primary_committed`
  - mutates the opaque states in place.
  - `actions` is CUDA `int64 [N]` in the unified 177-action space.
  - `reward0` is from player 0's perspective and is non-zero on terminal settlement only.

The kernels use PyTorch's current CUDA stream. Game state never has to cross PCIe during self-play.

## JIT build

`python/cugo_torch50_ext.py` uses `torch.utils.cpp_extension.load()` with the repository `include/` directory and C++20/CUDA C++20 flags. The build cache is stored under `build/torch_extensions/cugo_torch50_ext`.

For the Windows CUDA 13.2 development machine, use a PyTorch CUDA 13.2 build. PyTorch 2.14.0 provides `cu132` wheels.

Example:

```powershell
py -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu132
py python/torch50_smoke.py --batch 32768 --warmup 1 --iters 3
```

## Baseline network

`python/torch50_smoke.py` creates a deliberately small FP16 policy/value MLP:

- input: 496 floats
- trunk: 256 -> 256 with SiLU
- policy: 177 logits
- value: scalar

The policy logits are masked by the engine-generated 177-way legal mask before `argmax`. The script runs complete games until terminal or nagari and reports end-to-end games/s. This network is not a training architecture decision; it is a transport/inference baseline for measuring the CUDA-env/PyTorch boundary.
