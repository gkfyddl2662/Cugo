# Full-game rollout and Action50 rule lock

This layer joins the already validated 50-card deal, turn, pi-transfer, Go/Stop,
special-action, terminal, last-card, and settlement components into one complete
single-game environment path.

## Authoritative Hangame facts

Primary sources:

- https://mgostop.hangame.com/guide/combine/02_03_rule.html
- https://mgostop.hangame.com/guide/combine/02_04_rule.html

Hangame documents that a Nagari occurs when all hand cards have been used and
nobody has made 7 points, or when a player declared Go but failed to add another
point before the hand cards were exhausted. The next round is doubled if that
next round produces a score.

Cugo therefore reports a Nagari as a no-winner full-game outcome with
`next_round_multiplier = 2`. The multiplier is an output for the future match
state; it is deliberately not inserted into the current single-game state.

## Unified action boundary

`Action50` covers the player-facing decisions currently pinned by the engine:

- regular card play;
- hand bonus play;
- shake;
- three-card bomb;
- two-card grenade;
- synthetic bomb-credit draw;
- Go / Stop;
- bonus-replacement Chongtong win / continue.

Resolve-card and pi-transfer choices are fields of the same action, so the
entire action remains transactional. A required but missing choice never
partially commits the turn.

The exact Hangame UI timing for changing Gukjin between animal and double-pi is
still not stated by the public guide. That role therefore remains the explicit
persistent-state API in `gukjin_state.h` instead of being silently exposed at an
invented point in `Action50`.

## Canonical rollout policy

`rollout_canonical50()` is a deterministic correctness and throughput policy,
not an attempt to play strong Matgo. At a decision point it uses this stable
priority:

1. take an immediate bonus-Chongtong win;
2. on the first Go/Stop decision, Go; on the next eligible decision, Stop;
3. consume bomb credits;
4. play a hand bonus;
5. bomb;
6. grenade;
7. shake;
8. play the lowest CardId regular card.

When a floor or pi-transfer choice is necessary, the lowest valid physical card
IDs are selected. This makes CPU/GPU differential tests deterministic and gives
CUDA a complete full-game environment baseline before neural policy inference
is integrated.

## Rollout completion

A rollout ends as one of:

- normal/fixed terminal win, followed by `settle_terminal50()`;
- Nagari after both physical hands and synthetic bomb credits are exhausted;
- an explicit error/stall guard (test failure);
- the action safety cap (test failure).

The public Nagari wording is based on hand-card exhaustion, so leftover central
stock is not used as the Nagari predicate. This is important because the public
bonus-card guide says an opening floor bonus is automatically taken by the
first player but does not state that a replacement floor card is dealt.
