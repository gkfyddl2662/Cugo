# Hangame Shin Matgo rule lock

This file records rules pinned to Hangame's public guide and explicit internal representation decisions made by Cugo. It is the correctness boundary for the CPU reference engine and the CUDA differential implementation. Generic Go-Stop conventions must not silently fill missing behavior.

## Authoritative sources

- Basic rules: https://mgostop.hangame.com/guide/combine/02_01_rule.html
- Special capture/play situations: https://mgostop.hangame.com/guide/combine/02_02_rule.html
- Card groups, scoring groups, bonus cards, and bomb cards: https://mgostop.hangame.com/guide/combine/02_03_rule.html
- Hangame Shin Matgo vs. Match-Go mode differences: https://mgostop.hangame.com/guide/combine/04_03_game_mode.html
- Hangame Shin Matgo vs. Go-Stop mode differences: https://mgostop.hangame.com/guide/combine/04_05_game_mode.html
- Legacy Hangame Shin Matgo card/art guide used to cross-check the 12 standard month groups: https://hangame-images.toastoven.net/hangame/pc/gostop/introduce/html/msduelgo/guide_msduelgo04_05.html

## Pinned base flow

- Mobile Shin Matgo uses 50 physical cards in the complete ruleset.
- Each player receives 10 cards and 8 cards start open on the floor, leaving 22 cards in the stock before bonus handling.
- The players alternate turns.
- A player first plays a hand card. If it has the same month/pattern family as a floor card, the matching cards are taken.
- After the hand play, the top stock card is opened automatically and is resolved against the floor in the same way.
- Reaching at least 7 points allows a Go/Stop decision.
- The winner of a completed game leads the next game; the first game uses a separate first-player selection process.

The older base-48 deal remains as a deterministic regression/reference fixture. New exact-Shin-Matgo work uses the 50-card deck.

## Stable physical card identity convention

`CardId = month * 4 + slot` for the 48 standard cards, with zero-based months. The month identity follows Hangame's Korean ordering (1 Songhak through 12 Bi). Slot ordering is an engine-internal convention and is frozen as follows:

| Month | slot 0 | slot 1 | slot 2 | slot 3 |
| --- | --- | --- | --- | --- |
| 1 | bright | hongdan | pi | pi |
| 2 | animal / godori | hongdan | pi | pi |
| 3 | bright | hongdan | pi | pi |
| 4 | animal / godori | chodan | pi | pi |
| 5 | animal | chodan | pi | pi |
| 6 | animal | cheongdan | pi | pi |
| 7 | animal | chodan | pi | pi |
| 8 | bright | animal / godori | pi | pi |
| 9 | animal / Gukjin | cheongdan | pi | pi |
| 10 | animal | cheongdan | pi | pi |
| 11 | bright | fixed double-pi | pi | pi |
| 12 | rain bright | animal | plain ribbon | fixed double-pi |

Special physical IDs are:

- `48`: 2-pi bonus card
- `49`: 3-pi bonus card

IDs `50..63` remain available for future nonstandard physical/synthetic cards if needed.

`kStandardDeckMask` covers IDs `0..47`. `kShinMatgoDeckMask` covers IDs `0..49`. The legacy name `kFullDeckMask` intentionally remains an alias of the 48-card standard mask so old deterministic tests do not silently change meaning.

## Pinned base scoring

`score_captured()` implements base group points without applying final win multipliers:

- Brights: 3 brights = 3 points, but a 3-bright set containing the rain bright = 2; 4 brights = 4; all 5 = 15.
- Animals: 5 cards = 1 point and each additional animal adds 1. Godori (months 2, 4, 8) adds 5 points. Seven or more animals sets the meongtta multiplier flag; the final multiplier itself is not applied in the base score.
- Ribbons: 5 cards = 1 point and each additional ribbon adds 1. Hongdan (1,2,3), chodan (4,5,7), and cheongdan (6,9,10) each add 3 points. The month-12 ribbon belongs to none of those sets.
- Pi: 10 pi units = 1 point and each additional unit adds 1. Plain pi contributes 1 unit. The fixed month-11/month-12 double-pi cards contribute 2. Gukjin contributes 2 only when `ScoreOptions::gukjin_as_double_pi` is true, in which case it no longer counts as an animal.
- The bonus cards contribute 2 and 3 pi units respectively.

`pi_card_mask()` exposes which captured physical cards currently count as pi under the same Gukjin option.

## Pinned bonus-card behavior

Hangame describes two bonus cards in Shin Matgo. The printed number determines whether the card contributes 2 or 3 pi units.

- The raw 50-card deal gives each player 10 cards and exposes 8 floor cards, leaving 22 cards in stock.
- If a bonus card is among the initial floor cards, it is automatically taken before the first player starts. `collect_initial_floor_bonuses()` moves those bonus cards to the first player's captured set. The referenced Hangame guide does not state that another floor card is dealt as a replacement, so Cugo does not invent that replacement step.
- If a player has a bonus card in hand, they may play it like a normal hand action. Before the normal stock flip, they receive one replacement card from the stock and get another opportunity to play a hand card. `play_hand_bonus()` performs this transaction and keeps hand size unchanged until that follow-up play.
- In Shin Matgo, playing a bonus card also takes one opponent pi. The primitive returns `pi_steal_count=1`; actual card transfer is deferred until the exact multiple-pi-card selection policy is sourced.
- If a bonus appears while flipping the stock, the player flips again. `draw_stock_with_bonus_chain()` therefore consumes consecutive bonus cards until it reaches a standard card and returns the encountered bonus cards as `pending_bonus_mask`.
- If that eventual standard flip produces ppuk, Hangame says the bonus card(s) must be placed on the floor together with the ppuk cards. Pending stock bonuses therefore must not be committed to captured cards before RESOLVE.

The main `TurnState48` resolver has not yet been widened to the 50-card bonus-aware turn state. The next resolver milestone must add explicit association between pending/floor bonus cards and the ppuk month that owns them; a plain floor bitmask is insufficient if more than one ppuk stack exists.

## Pinned special situations relevant to RESOLVE

- Ppuk: a played card targets one floor card of the same month and the stock draw is also that month; all three remain on the floor instead of being captured. Last-card behavior is an exception in the guide.
- Capturing an opponent's ppuk stack steals one opponent pi; capturing one's own ppuk stack steals two opponent pi.
- First/second/third consecutive ppuk have Hangame-specific immediate money effects. Those economy effects are outside the current engine milestone but the ppuk ownership/history state must be representable.
- Ttadak: two same-month cards are already on the floor, the player plays that month, and the stock draw is also that month; all four are captured and one opponent pi is stolen.
- Sweep / pansseuri: taking all floor cards also steals one opponent pi.
- Bomb: three same-month hand cards can capture the fourth card from the floor in one play, steals one opponent pi, creates future bomb cards, and carries a score multiplier.
- Shake: showing three same-month hand cards when there is no floor match can create a score multiplier if the player later wins.
- Grenade / two-card bomb: two same-month hand cards can be used against the two same-month floor cards. Hangame's guide states that this steals pi but does not get the bomb score multiplier.
- Jjok: a hand play with no floor match followed by a same-month stock draw captures the pair and steals one opponent pi, except on the last card.

## Current base-48 resolver behavior

The deterministic CPU/CUDA resolver currently implements card movement for this standard-card subset:

- normal unique same-month capture
- unmatched cards remaining on the floor
- ppuk creation and ppuk owner metadata
- capture of a three-card ppuk stack, with own/opponent ppuk capture counts returned as side-effect metadata
- jjok card movement
- ttadak card movement
- sweep detection after the complete turn is resolved
- explicit same-month floor choice when exactly two matching floor cards exist

When a two-card floor choice is required and the caller did not provide a valid `ResolveChoices` entry, the resolver returns `kChoiceRequired` without changing the game state. Invalid choices likewise leave the state unchanged.

The guide says ppuk and jjok have a last-card exception but does not describe the replacement transition on that page. The engine therefore returns `kUnsupportedLastCardSpecial` for those two final-stock-flip patterns instead of guessing.

## Not implemented yet

- 50-card bonus-aware `TurnState` integration and ppuk-bonus stack association
- Last-card ppuk/jjok exception transition
- Exact pi-steal card selection/transfer policy
- Bomb/grenade action encoding and bomb-card credits
- Go/Stop decision state and final score multipliers
- Missions and economy/betting effects

Each item should be added first to the deterministic CPU reference path, covered by fixed examples and randomized invariants, and only then mirrored into CUDA with CPU/GPU differential tests.
