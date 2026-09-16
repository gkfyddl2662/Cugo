# Torch-compatible training boundary for Shin Matgo 50

This layer freezes the engine-to-network contract without making the core engine depend on LibTorch.
`include/cugo/game/torch50.h` is ordinary C++20 / CUDA code and can be compiled by the existing CPU and CUDA validation targets.

## Unified action space

The policy head has 177 logits.

- `0..126`: the existing packed primary policy actions from `policy50.h`.
- `127..176`: post-chance physical-card choices for card ids `0..49`.

At any decision point only one part of the action space is active. A primary decision exposes only `0..126`; a resolve/pi post-chance decision exposes only `127..176`. `TorchActionMask50` stores the mask in three `uint64_t` words.

This means one network head can handle ordinary play, bonus play, shake/bomb/grenade/bomb-credit, Go/Stop, Chongtong, two-match resolve choices, and physical pi-transfer choices.

## Dense feature layout

`kTorch50FeatureCount = 496` floats. Features are always in `[0, 1]`. The first 479 are semantic and the final 17 are zero padding so the width is divisible by 16.

| Range | Size | Meaning |
| --- | ---: | --- |
| `0..299` | 300 | six 50-card planes: own hand, floor, own captured, opponent captured, pending public, unseen |
| `300..395` | 96 | eight 12-month planes: own/opponent shaken, ppuk, own/opponent ppuk, bonus-2/bonus-3 ppuk, pending Chongtong |
| `396..445` | 50 | already-selected physical pi cards during a multi-card pi choice |
| `446..452` | 7 | one-hot decision kind: none, primary, Go/Stop, Chongtong, resolve-played, resolve-drawn, pi-card |
| `453..478` | 26 | normalized scalar state and post-chance metadata |
| `479..495` | 17 | zero padding |

The scalar block contains turn/stock counts, both players' Go/base-score/bomb/credit/ppuk summaries, persistent Gukjin roles, requested pi count, primary action, revealed card, selected/legal choice counts, and compact hand/floor/unseen counts.

## Environment state

`TorchEnv50` keeps the authoritative `TerminalGameState50` plus `PostChancePending50`. `torch_step50()` accepts one unified action id and handles both primary actions and post-chance card selections transactionally.

The environment returns reward from player 0's perspective only when the game reaches a terminal settlement. Nagari returns zero. Training code can negate the reward for player 1 or use the observation's player-relative features.

## Correctness contract

The CPU test runs 8,192 full games through this environment and requires the rollout digest and post-chance choice counts to match the existing explicit post-chance driver. The CUDA test runs 65,536 full-game differential samples plus 4,096 CPU/GPU feature-and-mask differential samples.

No PyTorch dependency is required for these tests. The next integration layer can wrap this fixed 496-float / 177-action contract in a PyTorch CUDA extension without changing game semantics.
