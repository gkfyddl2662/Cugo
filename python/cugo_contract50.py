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
from torch import nn

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


# -- fitting a checkpoint onto networks that gained inputs, outputs or heads --

def _fit(old: torch.Tensor, new: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Old tensor reshaped onto the new one: equal shapes pass, a matrix that gained input
    columns (new oracle inputs, appended last) is zero-padded, anything else restarts at
    zero. Returns (tensor, whether it was kept whole)."""
    if old.shape == new.shape:
        return old, True
    if old.dim() == 2 and new.dim() == 2 and old.shape[0] == new.shape[0] and old.shape[1] < new.shape[1]:
        out = new.new_zeros(new.shape, dtype=old.dtype)
        out[:, :old.shape[1]] = old
        return out, True
    return torch.zeros_like(new, dtype=old.dtype), False


def fit_state(saved: dict, module: nn.Module) -> tuple[dict, list[str]]:
    """A checkpoint's state dict fitted to `module` (see _fit); names the tensors that restarted."""
    out, restarted = {}, []
    for name, tensor in module.state_dict().items():
        if name not in saved:
            out[name] = tensor
            restarted.append(name)
            continue
        out[name], kept = _fit(saved[name], tensor)
        if not kept:
            restarted.append(name)
    return out, restarted


def optimizer_names(actor: nn.Module, critic: nn.Module, belief: nn.Module | None = None) -> list[list[str]]:
    """Parameter names per optimizer group: the actor; the critic and the belief head."""
    rest = [f"critic.{n}" for n, _ in critic.named_parameters()]
    if belief is not None:
        rest += [f"belief.{n}" for n, _ in belief.named_parameters()]
    return [[f"actor.{n}" for n, _ in actor.named_parameters()], rest]


def fit_optimizer(payload: dict, actor: nn.Module, critic: nn.Module, belief: nn.Module | None = None) -> dict:
    """The checkpoint's Adam state laid out for this optimizer's groups, matched by
    parameter name. Checkpoints before the actor/critic split had one group in
    actor-then-critic order. New parameters start without state; reshaped ones as _fit
    says, with their step count reset when they restart."""
    saved = payload["optimizer"]
    names = payload.get("optimizer_names")
    if names is None:
        critic_names = [f"critic.{n}" for n in payload["critic"] if not n.startswith("support")]
        names = [[f"actor.{n}" for n, _ in actor.named_parameters()] + critic_names]
    old_ids = [i for group in saved["param_groups"] for i in group["params"]]
    flat = [name for group in names for name in group]
    by_name = {name: saved["state"].get(i) for name, i in zip(flat, old_ids)}
    current = {f"actor.{n}": p for n, p in actor.named_parameters()}
    current |= {f"critic.{n}": p for n, p in critic.named_parameters()}
    if belief is not None:
        current |= {f"belief.{n}": p for n, p in belief.named_parameters()}
    state, groups, k = {}, [], 0
    for g, group_names in enumerate(optimizer_names(actor, critic, belief)):
        ids = []
        for name in group_names:
            old = by_name.get(name)
            if old is not None:
                fitted, whole = {}, True
                for key, value in old.items():
                    if torch.is_tensor(value) and value.dim() > 0:
                        value, kept = _fit(value, current[name])
                        whole &= kept
                    fitted[key] = value
                if not whole and torch.is_tensor(fitted.get("step")):
                    fitted["step"] = torch.zeros_like(fitted["step"])
                state[k] = fitted
            ids.append(k)
            k += 1
        group = dict(saved["param_groups"][min(g, len(saved["param_groups"]) - 1)])
        group["params"] = ids
        groups.append(group)
    return {"state": state, "param_groups": groups}
