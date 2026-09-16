# GPU self-play training baseline

This layer turns the GPU-resident `TorchEnv50` / PyTorch extension into an end-to-end learning loop. It is intentionally a **training infrastructure baseline**, not the final strongest-search algorithm.

## Data flow

Each iteration stays on CUDA:

1. `ext.create()` allocates a batch of `TorchEnv50` states.
2. `ext.observe_indexed()` encodes only active environments into the 496-float observation and 177-way legal mask.
3. `PolicyValueNet` produces policy logits and a scalar value for active rows.
4. A masked stochastic policy selects one legal unified action per active row.
5. `ext.step_indexed()` mutates only those active environment states on the current PyTorch CUDA stream.
6. Active decisions are appended to a bounded GPU trajectory buffer for the current rollout.
7. After each game finishes, its player-0 settlement reward is projected into the perspective of the player who made each recorded decision.
8. The resulting `(features, legal, action, value_target)` rows are CUDA-packed into a separate fixed-size GPU ring replay buffer.
9. Replay sampling uses a fused CUDA gather/unpack kernel and AdamW updates the policy/value network with CUDA AMP.

There is no environment-state PCIe round trip in this loop.

## Reward perspective

The engine returns terminal reward from player 0's perspective. Torch50 features are decision-player relative. Therefore a recorded player-1 decision uses the negated terminal reward as its value target.

Targets are divided by `--reward-scale` (default `32`) before training. This keeps large Shin Matgo settlement multipliers from immediately dominating the first optimizer experiments while preserving sign and relative magnitude.

## Trajectory and replay storage

Trajectory retention and replay history are separate capacities.

The per-iteration trajectory keeps the current rollout in the training contract used by the network:

- features as FP16 `[capacity, 496]`,
- legal masks as bool `[capacity, 177]`,
- actions and environment IDs as int64,
- decision players as uint8.

The RTX 5080 default trajectory capacity is 4,194,304 transitions, about 4,744 MiB of tensor storage. The measured 131,072-game self-play batch produces roughly 2.5-2.9 million transitions, so the complete rollout fits without tail truncation. If `--trajectory-capacity` is omitted it resolves to `min(4,194,304, replay_capacity)`, which keeps small-memory validation commands small. It can also be set explicitly.

`PackedGpuReplayBuffer` stores each historical transition in 137 bytes:

- the first 453 binary feature planes packed into 57 bytes,
- 26 scalar features as FP16 (52 bytes),
- the 17 fixed zero-padding features omitted,
- the 177-way legal mask packed into 23 bytes,
- selected action as uint8,
- value target as FP32.

Packing on add and indexed gather/unpack on sample are fused CUDA kernels. The RTX 5080 default replay capacity is 16,777,216 transitions, about 2,192 MiB. At the measured 2.5-2.9 million transitions per iteration this holds roughly five to six recent full rollouts while the transient trajectory remains capped at 4,194,304.

Each training iteration logs `retention_pct`, `replay_admission_pct`, and `replay_fill_pct`. `retention_pct=100` means the trajectory retained every generated transition. `replay_admission_pct=100` means every trajectory row was admitted by the replay add operation. With the tuned defaults both should remain 100%. If trajectory retention drops, increase `--trajectory-capacity` or reduce `--selfplay-batch`. If replay admission drops, increase `--replay-capacity` or reduce the trajectory capacity.

For smaller GPUs, override `--replay-capacity`, `--trajectory-capacity`, and `--selfplay-batch` explicitly.

## Baseline objective

The initial trainer uses a Monte-Carlo actor-critic objective:

- policy: `-advantage * log pi(action|state)`,
- value: Smooth L1 against scaled terminal settlement,
- entropy bonus for exploration,
- advantage clipping and gradient clipping for the bootstrap phase.

This is **not** intended as the final tabula-rasa algorithm. Its job is to validate trajectory ownership, reward perspective, replay writes/sampling, backpropagation, AMP, optimizer state, and checkpointing. The planned stronger search/training target can then reuse the same GPU plumbing.

## Commands

Small validation run with reduced GPU memory use:

```powershell
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 1 --selfplay-batch 4096 --trajectory-capacity 131072 --replay-capacity 131072 --updates-per-iter 2 --train-batch 4096
```

RTX 5080 baseline using the tuned defaults:

```powershell
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 100 --checkpoint-every 10
```

Checkpoints go under `build/checkpoints/torch50/`, which is already ignored by Git.
