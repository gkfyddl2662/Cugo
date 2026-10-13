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

`kTorch50FeatureCount = 720` floats (contract v4, extension `cugo_torch50_ext_v5`). Features are in `[0, 1]` except the side payment, which is clipped to `[-1, 1]`. New inputs go where a zero weight leaves an old network's output unchanged: `python/migrate_ppo50.py` widens 560-feature checkpoints, and `python/cugo_contract50.py` upgrades v3 checkpoints to v4 when they are loaded.

| Range | Size | Meaning |
| --- | ---: | --- |
| `0..299` | 300 | six 50-card planes: own hand, floor, own captured, opponent captured, pending public, unseen |
| `300..395` | 96 | eight 12-month planes: own/opponent shaken, ppuk, own/opponent ppuk, bonus-2/bonus-3 ppuk, pending Chongtong |
| `396..445` | 50 | floor cards the current floor choice may take (v3: selected pi) |
| `446` | 1 | the observer is the first player (v3: decision kind "none") |
| `447..451` | 5 | one-hot decision kind: primary, Go/Stop, Chongtong, resolve-played, resolve-drawn |
| `452` | 1 | opponent hand count / 10 (v3: decision kind "pi card") |
| `453..478` | 26 | normalized scalar state and post-chance metadata |
| `479..490` | 12 | neutral ppuk months (initial-floor stacks without an owner) |
| `491..495` | 5 | opponent pi count / 20; own and opponent bright / 5; own and opponent animal / 10 (v3: padding) |
| `496..559` | 64 | Hangame block: mission card plane (48) and 16 scalars |
| `560..709` | 150 | own played cards, opponent played cards, opponent cards known from a shake |
| `710..711` | 2 | own and opponent ribbon / 10 (v3: padding) |
| `712..719` | 8 | zero padding |

The scalar block contains turn/stock counts, both players' Go/base-score/bomb/credit/ppuk summaries, persistent Gukjin roles, primary action, revealed card, the legal choice count and compact hand/floor/unseen counts. Two of its slots changed in v4: `s[18]` (471) holds the final points stopping now would settle, as `log2(1 + points) / 12` (zero outside Go/Stop), and `s[21]` (474) the observer's pi count / 20.

### Contract v4

v3 left 54 columns permanently zero: Hangame steals pi automatically, so the pi-choice plane, its counters and the "pi card" decision kind never fired, and the "none" kind never marks a live decision. v4 puts facts a player reads off the table there (table above) and fixes two observations:

- `own_played` is complete at a floor choice on the opening play and after playing a bonus replacement (the engine now records the dealt hands and each replacement in `held0/held1` as they arrive).
- At a floor choice for the drawn card, the observation shows the table after the played card was resolved (its captures, the player's earlier floor choice and the updated base score) instead of the state before the turn.

The width and every other column are unchanged. A v3 network never received a non-zero value in the reused columns, so its input weights there were never trained; `cugo_contract50.upgrade_payload` zeroes them, which leaves the network's outputs exactly as they were on v3 inputs. Checkpoints written from v4 on carry `config.feature_contract = 4`.

## Environment state

`TorchEnv50` keeps the authoritative `TerminalGameState50` plus `PostChancePending50`. `torch_step50()` accepts one unified action id and handles both primary actions and post-chance card selections transactionally.

The environment returns reward from player 0's perspective only when the game reaches a terminal settlement. Nagari returns only the event payments already made (ppuk chain, first-turn ttadak). Training code can negate the reward for player 1 or use the observation's player-relative features.


## Event log

`torch_step_event50()` steps like `torch_step50()` and, when a primary action commits, also fills a `TorchEvent50`: the diff between the table before and after that action. `played` left the actor's hand, `drawn` entered it, `flipped` came off the stock, `gained0/1` and `lost0/1` changed each captured pile, `shown` was revealed by a shake, plus the actor and the action kind. `pack_torch_start50()` records the opening (both hands, floor, captured piles, mission and first player). The records are observer-independent; a policy must hide the opponent's `drawn` cards itself. `tests/torch50_test.cpp` replays start plus events and requires the hands, floor and piles to match the engine after every commit.

The extension exposes them as `step_events_indexed(states, env_ids, actions)` (the five `step_indexed` outputs plus `[N, 9]` int64 event words) and `start(states)` (`[N, 8]` int64). `python/cugo_seq50.py` turns them into token sequences; see `docs/seq50.md`.
## Correctness contract

The CPU test runs 8,192 full games through this environment and requires the rollout digest and post-chance choice counts to match the existing explicit post-chance driver. The CUDA test runs 65,536 full-game differential samples plus 4,096 CPU/GPU feature-and-mask differential samples.

No PyTorch dependency is required for these tests. The next integration layer can wrap this fixed 720-float / 179-action contract in a PyTorch CUDA extension without changing game semantics.
