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
- Pi: 10 pi units = 1 point and each additional unit adds 1. Plain pi contributes 1 unit. The fixed month-11/month-12 double-pi cards contribute 2. Gukjin contributes 2 only when its current role is double-pi, in which case it no longer counts as an animal.
- The bonus cards contribute 2 and 3 pi units respectively.

Hangame explicitly documents Gukjin as an animal card that can also be used as double-pi. The public guide does not specify a mandatory conversion moment. Cugo therefore exposes the role as explicit state rather than inventing an automatic conversion rule.

### Persistent Gukjin role representation

`TurnState50` stores one current role bit per player. To avoid adding another SoA field, these two flags are packed into the unused high bits of the existing `ppuk_owner1_months` word:

- bits `0..11`: existing ppuk-owner-by-month metadata
- bit `12`: player 0 currently treats captured Gukjin as double-pi
- bit `13`: player 1 currently treats captured Gukjin as double-pi
- bits `14..15`: reserved

`set_persistent_gukjin_role50()` changes the interpretation only when that player physically owns Gukjin in the captured pile. `persistent_score_player50()` derives the appropriate `ScoreOptions` from the state. `is_valid_persistent_turn_state50()` masks the role bits for the legacy base-state invariant and additionally rejects a double-pi role bit when the corresponding player does not own Gukjin.

If Gukjin is currently double-pi and is physically transferred by a pi-steal event, the current role follows the physical card to the new owner. This keeps physical ownership, pi eligibility, and scoring interpretation consistent. The new owner can later set the role back to animal explicitly. This is an engine-state convention; it does not claim that Hangame exposes the role-change timing as a separate UI action.

## Pinned bonus-card behavior

Hangame describes two bonus cards in Shin Matgo. The printed number determines whether the card contributes 2 or 3 pi units.

- The raw 50-card deal gives each player 10 cards and exposes 8 floor cards, leaving 22 cards in stock.
- If a bonus card is among the initial floor cards, it is automatically taken before the first player starts. `collect_initial_floor_bonuses()` moves those bonus cards to the first player's captured set. The referenced Hangame guide does not state that another floor card is dealt as a replacement, so Cugo does not invent that replacement step.
- If a player has a bonus card in hand, they may play it like a normal hand action. Before the normal stock flip, they receive one replacement card from the stock and get another opportunity to play a hand card. `play_bonus_for_turn50()` keeps the state in `PLAY`, captures the played bonus, and inserts the replacement into the active hand.
- In Shin Matgo, playing a bonus card also takes one opponent pi.
- If a bonus appears while flipping the stock, the player flips again. `draw_for_turn50()` consumes consecutive stock bonuses until it reaches a standard card and stores those bonuses in `pending_bonus_mask`.
- If the eventual standard flip does not create ppuk, the pending stock bonuses are captured by the active player in the same resolve transaction.
- If the eventual standard flip creates ppuk, Hangame says the bonus card(s) must be placed on the floor together with the ppuk cards. `TurnState50` moves the pending bonuses onto the floor and associates each physical bonus with that ppuk month.
- When that ppuk stack is later captured, its associated bonus card(s) are captured with the standard ppuk cards and the association is cleared.

### Ppuk-bonus association representation

A plain floor bitmask cannot say which ppuk owns a floor bonus when more than one ppuk stack exists. `TurnState50` therefore stores:

- `ppuk_months`: one bit per ppuk month.
- low 12 bits of `ppuk_owner1_months`: owner bit for those ppuk months.
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

## Pi-steal transfer boundary

The Hangame guide pins the number of opponent pi cards taken by the events above, and the Shin Matgo mode guide confirms that playing a bonus card also takes an opponent pi. The public guide does not specify the physical-card priority when the victim owns more eligible pi cards than must be transferred.

Cugo therefore does not silently encode an undocumented automatic priority. `pi_transfer.h` separates the pinned steal count from the provider-specific selection policy:

- `resolve_pi_steal_card_count()` converts resolver event metadata into the number of physical pi cards to take.
- `apply_pi_steal50()` moves physical cards between captured piles and, by default, derives the victim's Gukjin interpretation from `TurnState50`.
- If the victim has no more eligible physical pi cards than requested, every available pi card is transferred automatically.
- If the victim has more candidates than requested, the caller must provide a `PiTransferSelection` mask containing exactly the requested number of eligible physical cards.
- Missing selection returns `kSelectionRequired`; an illegal mask returns `kInvalidSelection`.
- `resolve_turn50_with_pi_transfer()` and `play_bonus_for_turn50_with_pi_transfer()` are transactional wrappers. Selection-required/invalid transfer leaves the entire resolve or hand-bonus action unchanged.
- A transferred double-pi or 3-pi bonus retains the full scoring value of that physical card. Gukjin is eligible only when the victim's persistent role marks it as double-pi.

Legacy overloads that accept an explicit `ScoreOptions` remain available for low-level differential/regression tests, but the normal 50-card state path uses the persistent player role.

This selection mask is an environment-policy input, not a documented Shin Matgo player action. Once Hangame's exact automatic priority is sourced or measured, that policy can supply the mask without changing the state representation.

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
- persistent Gukjin role and state-derived scoring/pi eligibility
- transactional pi-steal integration through `pi_transfer.h`

Both resolver paths currently implement normal unique same-month capture, unmatched floor placement, ppuk, capture of a three-card ppuk stack, jjok, ttadak, sweep detection, and explicit same-month floor choice when exactly two matching floor cards exist.

The guide says ppuk and jjok have a last-card exception but does not describe the replacement transition on that page. Both resolver paths therefore return an unsupported status for those exact final-stock-flip patterns instead of guessing.

## Not implemented yet

- Last-card ppuk/jjok exception transition
- Exact Hangame automatic pi-card priority when more candidates exist than must be stolen
- Exact Hangame UI/timing policy for changing Gukjin between animal and double-pi
- Bomb/grenade action encoding and bomb-card credits
- Go/Stop decision state and final score multipliers
- Missions and economy/betting effects

Each item should be added first to the deterministic CPU reference path, covered by fixed examples and randomized invariants, and only then mirrored into CUDA with CPU/GPU differential tests.
