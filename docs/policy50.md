# Packed policy I/O boundary

This layer is the GPU-facing boundary between the validated Shin Matgo environment and a future neural policy/value model.
It does not change game rules.

## Information-set observation

`PolicyObservation50` is always packed from the acting/deciding player's point of view.
It intentionally does **not** expose the opponent hand or the identity of cards remaining in stock.

The packed observation contains only information that is public or directly known to the player:

- own hand;
- floor;
- own and opponent captured cards;
- pending cards that have already become public;
- the unseen pool, computed as the known 50-card deck minus all visible/owned cards;
- stock **count**, but not stock card identities;
- relative Go state and base scores;
- bomb credits, bomb counts, shake months;
- ppuk ownership/streak/count metadata;
- bonus-to-ppuk associations;
- persistent Gukjin role;
- pending bonus-Chongtong months;
- turn index and current decision kind.

The unseen mask is therefore the union of hidden opponent-hand and stock cards. Swapping a hidden card between those two zones while keeping public state and counts unchanged must produce the same policy packet; the CPU test locks this invariant.

## 127-way primary action head

The primary policy head has exactly 127 actions and fits in two `uint64_t` words.

| Range | Meaning |
| --- | --- |
| 0..47 | play standard CardId 0..47 |
| 48..49 | play bonus 2-pi / 3-pi |
| 50..97 | shake and choose which standard card is played |
| 98..109 | bomb month 0..11 |
| 110..121 | grenade month 0..11 |
| 122 | consume bomb credit |
| 123..124 | Go / Stop |
| 125..126 | bonus-Chongtong win / continue |

A shake is card-indexed rather than month-indexed so the policy retains the choice of which one of the three same-month cards to play.

`legal_primary_actions50()` produces the 127-bit legal mask from player-visible state.
`primary_action_index50()` maps the existing `Action50` representation into this fixed head.
`decode_primary_action50()` maps a selected policy index back into an executable `Action50`.

## Current chance boundary

The existing full-game rollout API is transactional: a top-level action can also carry choices that become known only after the stock draw, such as a drawn-card floor match or a pi-transfer selection.
For this milestone, `decode_primary_action50()` uses the existing deterministic canonical helpers for those post-chance choices.

This is deliberately a temporary baseline, not the final strong-AI decision model.
The next policy milestone will split the environment at chance boundaries so that floor-match and pi-transfer choices become separate masked decision heads after the relevant stock card/event is visible.

## GPU representation

The packed representation keeps bitboards and small integer metadata instead of expanding directly to floats.
A later CUDA/PyTorch bridge can expand these packets to FP16/BF16 card tokens or planes in large batches without increasing the resident environment state.

`cugo_cuda_policy50_test` differentially checks 65,536 policy packets produced from deterministic mid-game snapshots.
`cugo_cuda_policy50_bench` measures packet-generation throughput, register count, local memory, and occupancy for 128/256/512 threads per block.
