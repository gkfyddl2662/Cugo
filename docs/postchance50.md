# Post-chance decision boundary

This layer exposes the player decisions that occur after a primary action has
already determined a public chance outcome. It does not change the Shin Matgo
rules or the existing transactional action implementation.

## Why this exists

The packed primary policy boundary has 127 actions, but the existing rollout
baseline still canonicalized two classes of choices internally:

- selecting one of two same-month floor cards,
- selecting which captured Pi card(s) to steal.

Those are player decisions and therefore must be visible to a learning policy.
`postchance50.h` turns them into explicit decision packets while preserving the
existing engine as the source of truth.

## Transaction model

The real `TerminalGameState50` is not modified while a post-chance decision is
pending.

1. Decode the chosen 127-way primary action without canonical choices.
2. Reproduce the public chance outcome in a staged copy.
3. If a floor-card or Pi-card decision is required, emit a packet.
4. Accumulate the selected card(s) in `PostChancePending50`.
5. When every required selection is present, call the existing
   `apply_action50()` once and commit atomically.

An invalid choice therefore leaves the base game state unchanged.

## Decision kinds

`PostChanceDecision50` uses the observation decision-kind values immediately
after the primary policy kinds:

- `4`: choose the floor card matched by the played card,
- `5`: choose the floor card matched by the stock-drawn card,
- `6`: choose one Pi card to transfer.

The Pi head is sequential. If two or more physical Pi cards must be selected,
the same 50-card head is invoked repeatedly and already selected cards are
removed from the legal mask. This avoids a combinatorial action space.

## Packet

`PostChancePacket50` contains:

- the normal information-safe `PolicyObservation50`, rebuilt from the staged
  public position and the original acting player's perspective,
- one 50-bit physical-card legal mask,
- the already selected Pi mask,
- the primary action index that led to this chance outcome,
- the revealed stock/replacement card,
- the requested Pi-card count and decision player.

The observation never adds the opponent hand or the remaining stock identity.
Only cards made public by the primary action/chance outcome are revealed.

## Network boundary

A minimal network can therefore use:

- primary policy head: 127 logits,
- post-chance card head: 50 logits,
- value head: scalar value.

The same 50-logit card head handles both floor matching and Pi transfer; the
post-chance decision kind tells the network which semantics are active.

## Validation

`cugo_postchance50_test` replays deterministic full games through the explicit
post-chance path and requires the final rollout digest to match the previous
canonical rollout exactly.

`cugo_cuda_postchance50_test` repeats that comparison for 65,536 games on CUDA
and requires non-zero coverage of both floor-card and Pi-card decisions.
