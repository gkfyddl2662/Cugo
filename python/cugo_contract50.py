"""Torch50 feature-contract versions and checkpoint upgrades.

Contract v4 keeps the 720-wide v3 layout but reuses columns v3 never set:

- 396..445: floor cards the current floor choice may take (v3: selected pi, but
  Hangame steals pi automatically, so no pi choice ever opened)
- 446 / 452: first-player bit / opponent hand count (v3: decision one-hot slots for
  "none" and "pi card", never live)
- 471 / 474: points stopping now would settle (log scale) / own pi count
  (v3: requested and selected pi counts, always zero)
- 491..495, 710..711: opponent pi, bright, animal and ribbon counts (v3 padding)

A v3 network saw zeros in all of these, so its input weights there were never
trained. Zeroing them keeps the network's output exactly as before; training then
learns to use the new inputs. The events-mode query also reads more columns in v4
(own hand, the new counts and the round multiplier); those enter with zero weights.

Checkpoints written before v4 carry no `feature_contract` entry and are upgraded on
load by `load_checkpoint`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CONTRACT = 4
STOP_POINTS_COLUMN = 471  # log2(1 + points stopping now would settle) / 12, at Go/Stop
V4_REUSED = [*range(396, 447), 452, 471, 474, *range(491, 496), 710, 711]
CONTEXT_V3 = [*range(200, 250), *range(396, 453), *range(471, 476), 552]
# Events-mode query columns: the v3 decision context, then own hand, table counts
# and the round multiplier.
CONTEXT_INDEX = [*CONTEXT_V3, *range(0, 50), *range(491, 496), 557, 710, 711]
_FEATURES = 720


def stop_points(features: torch.Tensor) -> torch.Tensor:
    """Exact settled points of stopping now from the fp32 features (0 off Go/Stop)."""
    return torch.round(torch.exp2(12.0 * features[:, STOP_POINTS_COLUMN].float()) - 1.0).long()


def checkpoint_contract(payload: dict[str, Any]) -> int:
    config = payload.get("config") or {}
    return int(config.get("feature_contract", 3))


def _is_seq(config: dict[str, Any]) -> bool:
    return str(config.get("algorithm", "")).startswith("seq-ppo")


def _events_widen(w: torch.Tensor) -> torch.Tensor:
    """[rows, |CONTEXT_V3| + rest] -> [rows, |CONTEXT_INDEX| + rest]; reused and new columns zero."""
    old = len(CONTEXT_V3)
    context = w[:, :old].clone()
    context[:, [i for i, c in enumerate(CONTEXT_V3) if c in V4_REUSED]] = 0
    extra = w.new_zeros(w.shape[0], len(CONTEXT_INDEX) - old)
    return torch.cat((context, extra, w[:, old:]), 1)


def _upgrade_state(state: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    out = dict(state)
    if _is_seq(config):
        key = "query_in.0.weight"
        if config.get("mode") == "events":
            out[key] = _events_widen(state[key])
            return out
    else:
        key = "inp.weight"
    w = state[key].clone()
    if w.shape[1] < _FEATURES:
        raise ValueError(f"{key} has {w.shape[1]} inputs; migrate 560-feature checkpoints first")
    w[:, V4_REUSED] = 0
    out[key] = w
    return out


def _upgrade_optimizer(optimizer: dict[str, Any], old_shapes: set[tuple[int, ...]]) -> dict[str, Any]:
    """Widen Adam moments of the events query weights the same way as the weights."""
    if not old_shapes:
        return optimizer
    state = {}
    for key, entry in optimizer["state"].items():
        entry = dict(entry)
        for name, value in entry.items():
            if torch.is_tensor(value) and tuple(value.shape) in old_shapes:
                entry[name] = _events_widen(value)
        state[key] = entry
    return {**optimizer, "state": state}


def upgrade_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Bring a training checkpoint to the current contract (no-op when already current)."""
    if checkpoint_contract(payload) >= CONTRACT:
        return payload
    config = dict(payload.get("config") or {})
    out = dict(payload)
    old_shapes = set()
    for role in ("actor", "actor_ema", "critic"):
        if role in payload:
            if _is_seq(config) and config.get("mode") == "events":
                old_shapes.add(tuple(payload[role]["query_in.0.weight"].shape))
            out[role] = _upgrade_state(payload[role], config)
    if "optimizer" in payload:
        out["optimizer"] = _upgrade_optimizer(payload["optimizer"], old_shapes)
    config["feature_contract"] = CONTRACT
    config["upgraded_from_contract"] = checkpoint_contract(payload)
    out["config"] = config
    return out


def load_checkpoint(path: Path | str, device: torch.device | str = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} did not contain a dictionary")
    return upgrade_payload(payload)
