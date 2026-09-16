# Shin Matgo special-play rule lock

This note pins the special-play subset implemented by `include/cugo/game/special50.h`.
Hangame's public guide remains the authority; cross-provider material is used only where the
Hangame text does not publish an implementation detail.

## Authoritative Hangame facts

Sources:

- https://mgostop.hangame.com/guide/combine/02_02_rule.html
- https://mgostop.hangame.com/guide/combine/02_03_rule.html
- https://mgostop.hangame.com/guide/combine/04_03_game_mode.html

Pinned behavior:

- 3-card bomb: if the player holds three cards of one month and the fourth is on the floor,
  all three hand cards are played together and all four standard cards are captured. The
  action steals one opponent pi card. A bomb contributes one 2x win multiplier.
- Shake: when the floor has no card of that month and the player holds three cards of the
  month, the player may reveal/shake the triplet when choosing one of those cards to play.
  Winning after that shake contributes one 2x win multiplier.
- Grenade / 2-card bomb: if two same-month cards are in hand and the other two are together
  on the floor, both hand cards may be played together and the four standard cards are
  captured. The action steals one opponent pi card. Hangame explicitly says this action does
  not receive the bomb score multiplier.
- After either bomb type, Hangame exposes synthetic bomb cards. Playing one consumes no
  physical hand card and only flips/resolves the stock.
- Normal Shin Matgo has the 2-card grenade; Hangame's separate Match-Go mode removes it.

## Bomb-card credit convention

Hangame's public text states that bomb cards are generated but does not state their exact
count in text. Cugo currently uses the standard hand-size compensation convention:

- a 3-card bomb creates two future draw-only bomb-card credits;
- a 2-card grenade creates one future draw-only bomb-card credit.

This is cross-checked against another major online Go-Stop provider's published rule guide,
which explicitly states two draw-only turns after a 3-card bomb and one after a 2-card bomb:
https://board-static.pmang.com/images/pmang/nabi/html/guide/gostop5/gostop5_2_2.html

This count is intentionally isolated in `special50.h` so it can be changed without altering
physical-card representation if a direct Hangame source proves a different count.

## Engine representation

`SpecialGameState50` wraps `GameState50` and adds only synthetic metadata:

- `shaken0/1`: one bit per month already shaken by each player;
- `bombs0/1`: number of multiplier-bearing 3-card bombs performed;
- `credits0/1`: remaining synthetic draw-only bomb cards.

The 50 physical cards remain entirely inside the nested `TurnState50`; synthetic bomb cards
never consume bits 50..63.

`legal_shake_months50()`, `legal_bomb_months50()`, and `legal_grenade_months50()` return compact
12-bit month masks. `SpecialAction50` encodes shake-play, bomb, grenade, and bomb-credit actions.

Bomb/grenade/bomb-credit execution is transactional across stock bonus chaining, draw-only
floor resolution, pi transfer, and Go/Stop decision opening. If a floor match choice or pi-card
selection is still required, the complete special action is rolled back.

`bomb_shake_multiplier50()` applies one factor of two for every recorded shake and every
3-card bomb. Grenades do not affect this multiplier. `go_bomb_shake_score50()` combines the
existing Go-adjusted score with only this bomb/shake multiplier; pi-bak, gwang-bak, meongtta,
dokbak, missions, and other final settlement effects remain deferred.

## Deliberately deferred

- chongtong / four-of-a-month start or bonus-replacement behavior;
- provider-exact bomb-card credit count if Hangame publishes a contrary value;
- final settlement interaction with pi-bak, gwang-bak, meongtta, dokbak, missions, and economy;
- last-card special exceptions already deferred by the base resolver.
