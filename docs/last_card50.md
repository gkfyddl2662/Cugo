# Shin Matgo last-card ppuk / jjok rule lock

This note pins the final-card exception implemented directly in `resolve_turn50()`.

## Authoritative Hangame fact

Primary source:

- https://mgostop.hangame.com/guide/combine/02_02_rule.html

Hangame explicitly states that both ppuk and jjok exclude the last card / final play from their
special-case rule. The public guide does not spell out the exact physical-card movement in a
separate sentence for that exception.

## Engine interpretation

Cugo preserves the already-established `stock == 0` predicate as the definition of the final
stock flip. This milestone deliberately does not redefine turn counting or stock exhaustion.

When a final stock flip would otherwise produce jjok:

- the played card and drawn card are captured normally;
- pending stock bonus cards are captured normally;
- the `JJOK` event bit is not set;
- therefore no jjok pi-steal is generated.

When a final stock flip would otherwise produce ppuk:

- the played card resolves against the one existing same-month floor card and captures that pair;
- the drawn same-month card then resolves against the updated floor and remains on the floor;
- pending stock bonus cards are captured normally;
- no ppuk metadata/ownership/bonus association is created;
- the `PPUK` event bit is not set.

This is the narrow ordinary two-step matching interpretation of Hangame's “last card excluded”
wording rather than a new provider-specific special rule.

## Cross-provider consistency check

Pmang's published Go-Stop guide separately states that last-hand-card cases do not receive the
extra pi steal for jjok, sweep, ttadak, and ppuk capture. This is not used as Hangame authority,
but it supports keeping the last-card exception free of a special pi-steal event:

- https://board-static.pmang.com/images/pmang/nabi/html/guide/gostop5/gostop5_2_2.html

## Deliberately unchanged

- non-final ppuk and jjok behavior;
- ttadak and sweep behavior, because Hangame's current public page does not mark those entries
  with the same explicit last-card exception;
- the existing final-turn predicate (`stock == 0`);
- bonus-to-ppuk association for non-final ppuk;
- terminal, Go/Stop, scoring, and settlement layers.
