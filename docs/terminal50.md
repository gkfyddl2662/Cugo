# Shin Matgo terminal-rule lock: Chongtong and three-ppuk

This note pins the terminal subset implemented by `include/cugo/game/terminal50.h`.
Hangame's public guide is authoritative where it is explicit. Where the public text stops
short of an implementation detail, Cugo records the narrow engine convention separately so
it can be replaced without changing capture logic.

## Authoritative Hangame facts

Primary source:

- https://mgostop.hangame.com/guide/combine/02_04_rule.html

Mode cross-check:

- https://mgostop.hangame.com/guide/combine/04_03_game_mode.html

Pinned behavior:

- Chongtong: receiving all four standard cards of one month ends the game at 10 points.
- A four-of-a-month on the initial floor also ends the game.
- If both players have Chongtong, the first player wins.
- If a replacement card received by playing a Shin Matgo bonus card creates Chongtong, the
  player may choose either Chongtong victory or `deulgo-chigi` (continue holding/playing it).
- Three ppuks by one player in the same game end the game with that player as winner whether
  the ppuks were consecutive or not.
- Nonconsecutive three-ppuk ends at 7 points.
- Three consecutive ppuks end at 49 points: the guide explicitly describes 7 + 14 + 21 for
  the consecutive-ppuk payments plus 7 points for the three-ppuk terminal result.

## Engine representation

`TerminalGameState50` wraps the already validated `SpecialGameState50` and adds only terminal
metadata:

- total ppuk count for each player;
- current ppuk streak for each player;
- pending bonus-replacement Chongtong actor/month mask;
- terminal winner, reason, and terminal point value.

No physical card representation changes. The 50-card mask remains unchanged.

A ppuk streak is tracked across that player's own completed turns. The opponent's intervening
turn does not reset it; a completed non-ppuk turn by the same player does. This matches the
public guide's first/second/third-turn consecutive-ppuk wording and is isolated in
`record_completed_turn_ppuk50()`.

The third ppuk has priority over a Go/Stop prompt that might otherwise be opened by the same
completed turn. When the third ppuk is recorded, any just-opened Go/Stop decision is cleared
and the terminal result wins the transition.

## Bonus-replacement Chongtong

The ordinary initial-hand Chongtong path is immediate. The special choice boundary is used
only when a successful bonus-card play draws a replacement standard card and that replacement
creates a new four-of-a-month in the current player's hand.

`play_bonus_for_terminal50_with_pi_transfer()` compares the hand's Chongtong month mask before
and after the replacement. New months open a `ChongtongAction50` choice:

- `kWin`: terminal 10-point Chongtong victory;
- `kContinue`: clear the pending choice and continue the game with the four cards still in hand.

The bonus play and pi-transfer remain transactional. If pi selection is unresolved, no
Chongtong choice is opened because the bonus transition itself has not committed.

## Floor-Chongtong convention

Hangame's public score page explicitly says a floor Chongtong ends the game, but the public text
shown there does not separately state the winner or repeat the point amount for that floor case.
Cugo currently resolves an initial floor Chongtong as a 10-point win for the first player. This
is an explicit engine convention, not presented as an additional Hangame quote. It is isolated
in `detect_initial_chongtong50()` and can be changed independently if a more specific Hangame
source is found.

For the extremely rare simultaneous collision of floor Chongtong with one or more hand
Chongtongs, the same first-player convention is used.

## Deliberately deferred

- money/economy settlement for first-ppuk and consecutive ppuk side payments;
- nagari carry-over;
- final pi-bak / gwang-bak / meongtta / dokbak settlement;
- provider-exact handling if Hangame publishes a different floor-Chongtong winner/point rule;
- last-card ppuk/jjok exceptions already deferred by the base resolver.
