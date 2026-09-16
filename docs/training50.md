# GPU self-play training baseline

This layer turns the GPU-resident `TorchEnv50` / PyTorch extension into an end-to-end learning loop. It is intentionally a **training infrastructure baseline**, not the final strongest-search algorithm.

## Data flow

Each iteration stays on CUDA:

1. `ext.create()` allocates a batch of `TorchEnv50` states.
2. `ext.observe()` produces the 496-float observation and 177-way legal mask.
3. `PolicyValueNet` produces policy logits and a scalar value.
4. A masked stochastic policy selects one legal unified action.
5. `ext.step()` mutates the environment state on the current PyTorch CUDA stream.
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

The default capacity of 262,144 transitions uses roughly 300 MiB. This is deliberately simple. Later search targets can replace or augment the selected-action field without changing the engine boundary.

## Baseline objective

The initial trainer uses a Monte-Carlo actor-critic objective:

- policy: `-advantage * log pi(action|state)`,
- value: Smooth L1 against scaled terminal settlement,
- entropy bonus for exploration,
- advantage clipping and gradient clipping for the bootstrap phase.

This is **not** intended as the final tabula-rasa algorithm. Its job is to validate trajectory ownership, reward perspective, replay writes/sampling, backpropagation, AMP, optimizer state, and checkpointing. The planned stronger search/training target can then reuse the same GPU plumbing.

## Commands

Small validation run:

```powershell
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 1 --selfplay-batch 4096 --updates-per-iter 2 --train-batch 4096
```

Longer baseline run:

```powershell
.\.venv-torch\Scripts\python.exe python\train50.py --iterations 100 --selfplay-batch 16384 --updates-per-iter 32 --train-batch 8192 --checkpoint-every 10
```

Checkpoints go under `build/checkpoints/torch50/`, which is already ignored by Git.
