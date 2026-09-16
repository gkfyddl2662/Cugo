# Hangame Shin Matgo rule lock

This file records only rules that have been pinned to Hangame's own public guide. It is the correctness boundary for the CPU reference engine and the CUDA differential implementation. Generic Go-Stop conventions must not silently fill missing behavior.

## Authoritative sources

- Basic rules: https://mgostop.hangame.com/guide/combine/02_01_rule.html
- Special capture/play situations: https://mgostop.hangame.com/guide/combine/02_02_rule.html
- Card groups, scoring groups, bonus cards, and bomb cards: https://mgostop.hangame.com/guide/combine/02_03_rule.html
- Hangame Shin Matgo vs. Match-Go mode differences: https://mgostop.hangame.com/guide/combine/04_03_game_mode.html

## Pinned base flow

- Mobile Shin Matgo uses 50 cards in the complete ruleset.
- Each player receives 10 cards and 8 cards start open on the floor.
- The players alternate turns.
- A player first plays a hand card. If it has the same month/pattern family as a floor card, the matching cards are taken.
- After the hand play, the top stock card is opened automatically and is resolved against the floor in the same way.
- Reaching at least 7 points allows a Go/Stop decision.
- The winner of a completed game leads the next game; the first game uses a separate first-player selection process.

The current engine intentionally uses only the 48 standard cards. With those 48 cards the temporary baseline layout is therefore 10 + 10 hands, 8 floor, and 20 stock. This is not the final exact 50-card setup.

## Pinned special situations relevant to RESOLVE

- Ppuk: a played card targets one floor card of the same month and the stock draw is also that month; all three remain on the floor instead of being captured. Last-card behavior is an exception in the guide.
- Capturing an opponent's ppuk stack steals one opponent pi; capturing one's own ppuk stack steals two opponent pi.
- First/second/third consecutive ppuk have Hangame-specific immediate money effects. Those economy effects are outside the current engine milestone but the ppuk ownership/history state must be representable.
- Ttadak: two same-month cards are already on the floor, the player plays that month, and the stock draw is also that month; all four are captured and one opponent pi is stolen.
- Sweep: taking all floor cards also steals one opponent pi.
- Bomb: three same-month hand cards can capture the fourth card from the floor in one play, steals one opponent pi, creates future bomb cards, and carries a score multiplier.
- Shake: showing three same-month hand cards when there is no floor match can create a score multiplier if the player later wins.
- Grenade / two-card bomb: two same-month hand cards can be used against the two same-month floor cards. Hangame's guide states that this steals pi but does not get the bomb score multiplier.
- Jjok: a hand play with no floor match followed by a same-month stock draw captures the pair and steals one opponent pi, except on the last card.

These cases are why the engine does not mutate the floor immediately in `PLAY`. `TurnState48` keeps `pending_played` and `pending_drawn` until `RESOLVE`, so the resolver can inspect the original floor plus both turn cards together.

## Floor topology metadata

A plain 48-bit floor mask is not enough to distinguish a natural three-card same-month floor configuration from a ppuk stack, nor can it tell whether a ppuk belongs to player 0 or player 1. `TurnState48` therefore reserves:

- `ppuk_months`: one bit per month that currently represents a ppuk triplet on the floor.
- `ppuk_owner1_months`: owner bit for those ppuk months; a clear owner bit means player 0, a set owner bit means player 1.

This metadata is initialized to zero and will be mutated only by the exact resolver.

## Bonus cards intentionally deferred

The official guide describes 2-pi and 3-pi bonus cards and special replacement/redraw behavior when they appear in the initial floor, a hand, or during stock flips. The current base-48 milestone deliberately excludes them. Adding bonus IDs and their transitions is required before calling the engine an exact 50-card Hangame Shin Matgo implementation.

## Not implemented yet

- Exact capture mutation for all RESOLVE cases
- Ambiguous same-month selection actions
- Bomb/grenade action encoding and bomb-card credits
- Captured-card category/scoring metadata
- Pi transfer
- Go/Stop and scoring state
- Missions and economy/betting effects
- Bonus cards

Each item should be added first to the deterministic CPU reference path, covered by fixed examples and randomized invariants, and only then mirrored into CUDA with CPU/GPU differential tests.
