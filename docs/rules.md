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
- If a player has a bonus card in hand, they may play it like a normal hand action. Before the normal stock flip, they receive one replacement card from the stock and get another opportunity to play a hand card. `play_bonus_for_turn50()` keeps the state in `PLAY`, captures the played bonus, and inserts the replacement into the active hand.
- In Shin Matgo, playing a bonus card also takes one opponent pi. The action returns `pi_steal_count=1`; actual card transfer is deferred until the exact multiple-pi-card selection policy is sourced.
- If a bonus appears while flipping the stock, the player flips again. `draw_for_turn50()` consumes consecutive stock bonuses until it reaches a standard card and stores those bonuses in `pending_bonus_mask`.
- If the eventual standard flip does not create ppuk, the pending stock bonuses are captured by the active player in the same resolve transaction.
- If the eventual standard flip creates ppuk, Hangame says the bonus card(s) must be placed on the floor together with the ppuk cards. `TurnState50` moves the pending bonuses onto the floor and associates each physical bonus with that ppuk month.
- When that ppuk stack is later captured, its associated bonus card(s) are captured with the standard ppuk cards and the association is cleared.

### Ppuk-bonus association representation

A plain floor bitmask cannot say which ppuk owns a floor bonus when more than one ppuk stack exists. `TurnState50` therefore stores:

- `ppuk_months`: one bit per ppuk month.
- `ppuk_owner1_months`: owner bit for those ppuk months.
- `bonus2_ppuk_months`: zero or one month bit identifying the ppuk that owns physical bonus ID 48.
- `bonus3_ppuk_months`: zero or one month bit identifying the ppuk that owns physical bonus ID 49.

A bonus may be on the floor only if its association map identifies an existing ppuk month. Each marked ppuk month must contain exactly three standard cards on the floor. The complete 50 physical cards must remain a disjoint partition across hands, floor, stock, captured piles, pending played/drawn cards, and `pending_bonus_mask`.

Choice/error resolution remains transactional: if a two-card floor selection is needed, `resolve_turn50()` leaves the complete state and pending bonus chain unchanged until a valid choice is supplied.

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

## Resolver paths

The older `TurnState48` deterministic CPU/CUDA resolver remains a standard-card regression/performance fixture.

The bonus-aware `TurnState50` path implements the same current standard-card subset plus:

- initial-floor bonus collection
- hand-bonus replacement action
- stock bonus chains
- pending bonus capture on ordinary resolve
- pending bonus placement/association on ppuk
- later ppuk capture including associated bonus cards
- 50-card state partition and ppuk-bonus association invariants

Both resolver paths currently implement normal unique same-month capture, unmatched floor placement, ppuk, capture of a three-card ppuk stack, jjok, ttadak, sweep detection, and explicit same-month floor choice when exactly two matching floor cards exist.

The guide says ppuk and jjok have a last-card exception but does not describe the replacement transition on that page. Both resolver paths therefore return an unsupported status for those exact final-stock-flip patterns instead of guessing.

## Not implemented yet

- Last-card ppuk/jjok exception transition
- Exact pi-steal card selection/transfer policy
- Bomb/grenade action encoding and bomb-card credits
- Go/Stop decision state and final score multipliers
- Missions and economy/betting effects

Each item should be added first to the deterministic CPU reference path, covered by fixed examples and randomized invariants, and only then mirrored into CUDA with CPU/GPU differential tests.
