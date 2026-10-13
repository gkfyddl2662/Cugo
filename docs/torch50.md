# Torch-compatible training boundary for Shin Matgo 50

This layer freezes the engine-to-network contract without making the core engine depend on LibTorch.
`include/cugo/game/torch50.h` is ordinary C++20 / CUDA code and can be compiled by the existing CPU and CUDA validation targets.

## Unified action space

The policy head has 179 logits: 177 legacy actions plus the Gukjin answers 177 (double pi) and 178 (animal).

- `0..126`: the existing packed primary policy actions from `policy50.h`.
- `127..176`: post-chance physical-card choices for card ids `0..49`.

At any decision point only one part of the action space is active. A primary decision exposes only `0..126`; a resolve/pi post-chance decision exposes only `127..176`. `TorchActionMask50` stores the mask in three `uint64_t` words.

This means one network head can handle ordinary play, bonus play, shake/bomb/grenade/bomb-credit, Go/Stop, Chongtong, two-match resolve choices, and physical pi-transfer choices.

## Dense feature layout

`kTorch50FeatureCount = 720` floats (contract v3, extension `cugo_torch50_ext_v4`). The first 479 are semantic, 479..490 is the neutral-ppuk month plane and 491..495 stay zero (the legacy 496-wide block). The Hangame block follows: 496..543 is the mission card plane, and 544..559 are 16 scalars (mission type one-hot, multipliers, Gukjin prompt and lock flags, last-turn flag, side payment, round multiplier, ppuk-chain flags). 560..709 are three public-history card planes and 710..719 are zero. Features are in `[0, 1]` except the side payment, which is clipped to `[-1, 1]`. New inputs are added where a zero weight leaves an old network's output unchanged; `python/migrate_ppo50.py` widens 560-feature checkpoints that way.

| Range | Size | Meaning |
| --- | ---: | --- |
| `0..299` | 300 | six 50-card planes: own hand, floor, own captured, opponent captured, pending public, unseen |
| `300..395` | 96 | eight 12-month planes: own/opponent shaken, ppuk, own/opponent ppuk, bonus-2/bonus-3 ppuk, pending Chongtong |
| `396..445` | 50 | already-selected physical pi cards during a multi-card pi choice |
| `446..452` | 7 | one-hot decision kind: none, primary, Go/Stop, Chongtong, resolve-played, resolve-drawn, pi-card |
| `453..478` | 26 | normalized scalar state and post-chance metadata |
| `479..490` | 12 | neutral ppuk months (initial-floor stacks without an owner) |
| `491..495` | 5 | zero padding |
| `496..559` | 64 | Hangame block: mission card plane (48) and 16 scalars |
| `560..709` | 150 | own played cards, opponent played cards, opponent cards known from a shake |
| `710..719` | 10 | zero padding |

The scalar block contains turn/stock counts, both players' Go/base-score/bomb/credit/ppuk summaries, persistent Gukjin roles, requested pi count, primary action, revealed card, selected/legal choice counts, and compact hand/floor/unseen counts.

## Environment state

`TorchEnv50` keeps the authoritative `TerminalGameState50` plus `PostChancePending50`. `torch_step50()` accepts one unified action id and handles both primary actions and post-chance card selections transactionally.

The environment returns reward from player 0's perspective only when the game reaches a terminal settlement. Nagari returns only the event payments already made (ppuk chain, first-turn ttadak). Training code can negate the reward for player 1 or use the observation's player-relative features.


## Event log

`torch_step_event50()` steps like `torch_step50()` and, when a primary action commits, also fills a `TorchEvent50`: the diff between the table before and after that action. `played` left the actor's hand, `drawn` entered it, `flipped` came off the stock, `gained0/1` and `lost0/1` changed each captured pile, `shown` was revealed by a shake, plus the actor and the action kind. `pack_torch_start50()` records the opening (both hands, floor, captured piles, mission and first player). The records are observer-independent; a policy must hide the opponent's `drawn` cards itself. `tests/torch50_test.cpp` replays start plus events and requires the hands, floor and piles to match the engine after every commit.

The extension exposes them as `step_events_indexed(states, env_ids, actions)` (the five `step_indexed` outputs plus `[N, 9]` int64 event words) and `start(states)` (`[N, 8]` int64). `python/cugo_seq50.py` turns them into token sequences; see `docs/seq50.md`.
## Correctness contract

The CPU test runs 8,192 full games through this environment and requires the rollout digest and post-chance choice counts to match the existing explicit post-chance driver. The CUDA test runs 65,536 full-game differential samples plus 4,096 CPU/GPU feature-and-mask differential samples.

No PyTorch dependency is required for these tests. The next integration layer can wrap this fixed 720-float / 179-action contract in a PyTorch CUDA extension without changing game semantics.
