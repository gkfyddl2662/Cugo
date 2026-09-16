# GPU self-play training baseline

This layer turns the GPU-resident `TorchEnv50` / PyTorch extension into an end-to-end learning loop. It is intentionally a **training infrastructure baseline**, not the final strongest-search algorithm.

## Data flow

Each iteration stays on CUDA:

1. `ext.create()` allocates a batch of `TorchEnv50` states.
2. `ext.observe_indexed()` encodes only active environments into the 496-float observation and 177-way legal mask.
3. `PolicyValueNet` produces policy logits and a scalar value for active rows.
4. A masked stochastic policy selects one legal unified action per active row.
5. `ext.step_indexed()` mutates only those active environment states on the current PyTorch CUDA stream.
6. Active decisions are appended to a GPU trajectory.
7. After each game finishes, its player-0 settlement reward is projected into the perspective of the player who made each recorded decision.
8. The resulting `(features, legal, action, value_target)` rows enter a fixed-size GPU ring replay buffer.
9. AdamW updates the policy/value network with CUDA AMP.

There is no environment-state PCIe round trip in this loop.

## Reward perspective

The engine returns terminal reward from player 0's perspective. Torch50 features are decision-player relative. Therefore a recorded player-1 decision uses the negated terminal reward as its value target.

Targets are divided by `--reward-scale` (default `32`) before training. This keeps large Shin Matgo settlement multipliers from immediately dominating the first optimizer experiments while preserving sign and relative magnitude.

## Replay storage

`GpuReplayBuffer` stores:

- features as FP16 `[capacity, 496]`,
- legal masks as bool `[capacity, 177]`,
- selected actions as int64,
- value targets as FP32.

The RTX 5080 / 16 GiB baseline defaults to 4,194,304 transitions, which uses about 4.6 GiB of tensor storage. The default self-play batch is 131,072 games. In the measured baseline this produces roughly 2.5-2.9 million transitions per iteration, so the complete rollout fits in replay and the ring retains part of earlier iterations instead of learning only from the tail of the newest rollout.

Each training iteration logs `retention_pct` and `replay_fill_pct`. `retention_pct=100` means every generated transition entered replay. If policy behavior changes enough that retention drops below 100%, increase replay capacity, reduce `--selfplay-batch`, or implement a bounded unbiased retention policy rather than relying on tail truncation.

For smaller GPUs, override `--replay-capacity` and `--selfplay-batch` explicitly.

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
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 1 --selfplay-batch 4096 --replay-capacity 131072 --updates-per-iter 2 --train-batch 4096
```

RTX 5080 baseline using the tuned defaults:

```powershell
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 100 --checkpoint-every 10
```

Checkpoints go under `build/checkpoints/torch50/`, which is already ignored by Git.
