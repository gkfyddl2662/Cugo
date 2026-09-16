# Shin Matgo terminal settlement rule lock

This note pins the final-point settlement implemented by `settlement50.h`.

## Authoritative Hangame facts

Primary source:

- https://mgostop.hangame.com/guide/combine/02_04_rule.html

For an ordinary Stop win, Hangame documents the following independent effects:

- every Go adds 1 point;
- from 3-Go onward, each additional Go doubles the final score again;
- every bomb and every shake doubles the final score;
- meongtta doubles the final score when the winner has at least seven animal cards;
- pi-bak doubles the final score when the winner has at least 10 pi and the loser has at most 7 pi;
- gwang-bak doubles the final score when the winner has at least three brights and the loser has no bright;
- dok-bak applies when the loser had already declared Go and the opponent Stops before that player can Go or Stop again.

The engine therefore computes ordinary Stop settlement as:

1. captured-card base score;
2. add Go points and apply the 3-Go-and-later multiplier;
3. apply the accumulated bomb/shake multiplier;
4. apply each qualifying x2 condition independently: meongtta, pi-bak, gwang-bak, dok-bak.

Pi counts use the same persistent card interpretation as normal scoring, including fixed double-pi,
bonus 2-pi/3-pi, and Gukjin when its owner has selected the double-pi role.

## Fixed special terminals

The public guide gives explicit fixed terminal values for Chongtong (10 points), ordinary three-ppuk
(7 points), and three consecutive ppuk (49 points), but it does not explicitly say to stack the
ordinary Go/bak/bomb/shake settlement multipliers on top of those immediate-win values.

Cugo therefore keeps those already-recorded `terminal_points` unchanged for:

- initial Chongtong;
- floor Chongtong (under the engine convention documented in `terminal50.md`);
- bonus-replacement Chongtong when the player chooses the immediate win;
- three-ppuk terminals.

This is an explicit conservative boundary, not a claim about undocumented Hangame behavior.

## Self-play reward

`settlement_reward50()` returns `+final_points` for the winner and `-final_points` for the loser,
which provides a zero-sum terminal reward without adding fields to the game state.
