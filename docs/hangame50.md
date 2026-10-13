# Hangame Shin Matgo client rules (SingleServer)

The public guide pages in `rules.md` leave many details open. The Hangame Shin Matgo
client ships a local rule server (`SingleServer`, game type NDUELGO) that decides those
details. This file records what Cugo took from it. Where this file and an older doc
disagree, this file wins. Obfuscated method names are cited so each rule can be
re-checked against a fresh decompile. No client code or assets are stored in the
repository.

Tests: `tests/hangame_rules50_test.cpp` checks each rule below once; the CPU/CUDA
differential tests cover the same paths on the device.

## Turn numbering

Hangame counts turns 1..20, ten per player. `TurnState50::turn_index` is zero-based
and advances at the end of each turn. After a turn completes it therefore equals the
Hangame number of that turn. `kLastTurnIndex50 = 18` marks the start of turn 19, and
`is_last_turn50()` is true for turns 19 and 20.

## Deal

- **Initial floor bonuses** go to the first player. The floor is refilled from the
  stock, repeating while a refill card is itself a bonus
  (`refill_initial_floor_bonuses50`). Chongtong is judged on the refilled floor.
- **Dealt triples**: three cards of one month on the initial floor form a ppuk stack
  with no owner (`neutral_ppuk_months`). Capturing it steals one pi, like capturing the
  opponent's ppuk.
- **Mission** (`OKCHEMHJFHJ`): the type is uniform over easy / normal / hard /
  no-condition / triple. The multiplier is uniform over 2..4, and the card set is
  uniform within the type (`mission_cards50`). Cugo draws the mission from a hash of
  the deal so the stock order is unaffected.

## Last turns (19 and 20)

- No ppuk (LASTBBUCK). A played card matching one floor card captures it, then the
  drawn card is resolved on its own.
- No pi steals of any kind: jjok, ttadak, sweep, ppuk capture, bomb, or hand bonus. The
  jjok event is not raised.
- **Automatic stop**: once a player is eligible for Go/Stop after turn 19 or 20, they
  win at once. Go is not offered.

## Pi steals

- Steal count: one for each of jjok, ttadak, sweep, opponent or neutral ppuk capture;
  two for capturing your own ppuk; plus one for each bonus card captured
  (`resolve_pi_steal_card_count`).
- **Sweep** means the floor is empty after the turn. It does not depend on the floor
  before the turn.
- **Which pi** (`HJGDKDHNKLJ` / `DFHLKPIPPHC`) is chosen automatically, so no pi-card
  choice is ever offered (`hangame_pi_steal_selection50`):
  - Default: lowest value first (1, 2, then 3). On a tie, non-mission cards go before
    mission cards.
  - If the victim owns every mission card: all non-mission cards first, then by value.
  - Final tie-break: card id. Hangame's sort is unstable here, so this is an
    approximation.

## Two-card floor choice

When both candidates are plain pi of months 1..11, the choice is made automatically
(`JDAKOOAHHKI`, `auto_floor_choice50`). December and any mixed pair stay a real choice.
Hangame captures the card that reached the floor first; Cugo takes the lower id. Every
mission set holds both plain pi of a month or neither, so the two are interchangeable.

## Bomb, grenade, shake

- Bombs and grenades are forced. With three cards of a month in hand and one on the
  floor, or two in hand and two on the floor, a plain play of that month is illegal;
  the bomb or grenade must be used (`forced_special_months50`).
- **Four-card shake**: holding all four cards of a month, none on the floor, offers a
  shake. This can happen after a bonus-replacement Chongtong when the player continues.
  It counts like a normal shake (x2) and is tracked in `shaken4_0/1`.

## Ppuk chain (`PNFJMMBGKKM`)

| State | Next ppuk | Becomes |
|---|---|---|
| None | on turn ≤ 1 | First (pays 7) |
| None | later | One |
| First | on turn ≤ 3 | Second (pays 14) |
| First | later | Two |
| Second | on turn ≤ 5 | Third (pays 21, game ends) |
| Second | later | Three (game ends) |
| One | — | Two |
| Two | — | Three (game ends) |

Third or Three ends the game with a fixed 7 points. Turn numbers are Hangame's global
count, so the second player's first turn is turn 2 and can never start a "First" chain.

## Event payments

- First / second / third ppuk pay 7 / 14 / 21 at once.
- A ttadak on turn 1 pays 7.

These sit in `side_points0`, already multiplied by the round multiplier. They count
even when the game ends in nagari.

## Gukjin

- Gukjin is scored as an animal until decided. Each decision is one-time and locks
  the role (`decide_gukjin50`).
- **Prompt** (`IHDACFJKLOJ`): after their own turn, before Go/Stop, a player is asked
  when all of these hold:
  - they hold an undecided Gukjin;
  - they have at least 8 pi without it;
  - their score with Gukjin as double pi, plus their Go count, reaches 7.

  Torch actions 177 (pi) and 178 (animal) answer it.
- **Loser auto-decision** (`GFADGFKNBBD` / `OOBIDOMNFDO`): when the opponent reaches
  Go/Stop or wins, the loser's undecided Gukjin is decided for them:
  - with 6 or 7 pi, it becomes double pi (escaping pi-bak);
  - with 0 pi, it stays an animal (keeping the zero-pi exemption).
- A steal that finds too few pi locks the victim's Gukjin as an animal.
- A stolen Gukjin lands in the thief's pile as an undecided animal. The role does not
  follow the card.

## Settlement (`MCKLOLPMBOH`)

```
final = (base + go) × go multiplier × 2^(shakes + bombs) × meongtta × mission
        × gwang-bak × pi-bak × go-bak × round multiplier
```

- **Pi-bak** needs the winner to have at least 10 pi and the loser 1..7 pi. A loser
  with 0 pi is exempt.
- **Void game**: if the loser captured nothing at all, the payout is 0.
- **Mission**: the winner's mission multiplier.
  - Normal missions pay their multiplier when one player owns every mission card. The
    actor is credited first.
  - No-condition missions credit both players.
  - Triple missions pay 2^k, where k is the number of mission cards owned. Both
    players start at 2^0 = 1, as the server's EKGDFAHDPJD does with no cards.
- Chongtong (10) and three-ppuk (7) are fixed scores times the round multiplier.
- **Nagari carry-over**: `round_multiplier` scales the whole next game, side payments
  included. Training uses 1, which leaves the optimal policy unchanged because every
  payoff scales together.

## Differential check

`tools/hg_replay50.cpp` (`cugo_hg_replay50 <trace>`) replays games recorded from the
client's own SingleServer, run offline outside Unity against copies of its assemblies.
The deck order comes from the trace and stock draws are forced through
`g_forced_draws50`. At every turn notice it compares the actor, floor, hands, bomb
credits, captured cards, running score, Go count, ppuk state, shakes and mission
multiplier. At the end it compares the winner, the final score and side payments, and
a nagari against exhausted hands. Interchangeable cards (same pi value and mission
membership) are aligned first. Hangame also settles the loser's gukjin after a
fixed-score end, where it cannot matter, so the running score is not compared there.

The recording harness stays outside the repository because it loads the client's
binaries. Seat 0 plays a random client with four styles: random, capture-seeking,
always Go and always stop. Seat 1 is the server's AI. On 80,000 recorded games every
game replayed identically. That included 156 nagari, 2,143 Chongtong, 267 three-ppuk,
about 21k bomb dummies and about 42k gukjin choices.

## Shake answers

Hangame records a month as soon as its shake question is answered, yes or no, and does
not ask again (`_listShakeMonth`, `_listShake4Month`). Cugo keeps no such record and
needs none: a card of a shakeable month that is played without shaking lands on the
empty floor, so its owner can never again hold three of that month with none on the
floor. `test_declined_shake_never_returns` checks this over random games.

## Public information in the observation

A shake shows the month's cards to the opponent (`NoticeShakeToC` lists them), and
every card played from a hand is seen by both players. The observation carries the
cards each side has played, the opponent cards known from a shake that are still in
their hand (removed from the unseen plane), three- and four-card shakes alike, and
neutral ppuk stacks as their own plane. Contract v4 (`torch50.md`) adds what a
player reads off the table directly: who moved first, the opponent's hand size, each
side's pi/bright/animal/ribbon counts, the points stopping now would settle, and the
floor cards a floor choice may take. A bonus continuation after a hand-bonus
Chongtong does not show the four cards here; the client gives no sign that it does,
and the cards become public only through a later four-card shake.

## Known approximations

- Steal tie-break by card id, as noted above.
- The mission is drawn from a hash of the deal, not from Hangame's server RNG.
- The next game's first player and the carry of nagari between games live outside
  the single-game engine.
