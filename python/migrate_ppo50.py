"""Widen a PPO checkpoint from the 496/177 Torch50 contract to the Hangame 560/179 one.

The new features (mission plane and scalars) are appended after the legacy 496, and
the two Gukjin answers after the legacy 177 actions, so zero-initialized weights keep
the migrated network's behaviour on every legacy decision exactly unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cugo_ppo_nets50 import ACTIONS, FEATURES, ORACLE_FEATURES

OLD_FEATURES = 496
OLD_ACTIONS = 177


def widen_actor(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = dict(state)
    w = state["inp.weight"]
    if w.shape[1] != OLD_FEATURES:
        raise ValueError(f"actor input has {w.shape[1]} columns, expected {OLD_FEATURES}")
    out["inp.weight"] = torch.cat([w, w.new_zeros(w.shape[0], FEATURES - OLD_FEATURES)], dim=1)
    ow, ob = state["out.weight"], state["out.bias"]
    if ow.shape[0] != OLD_ACTIONS:
        raise ValueError(f"actor output has {ow.shape[0]} rows, expected {OLD_ACTIONS}")
    out["out.weight"] = torch.cat([ow, ow.new_zeros(ACTIONS - OLD_ACTIONS, ow.shape[1])], dim=0)
    out["out.bias"] = torch.cat([ob, ob.new_zeros(ACTIONS - OLD_ACTIONS)], dim=0)
    return out


def widen_critic(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Critic input is [public features | opponent hand plane]; new features go between.
    out = dict(state)
    w = state["inp.weight"]
    if w.shape[1] != OLD_FEATURES + ORACLE_FEATURES:
        raise ValueError(f"critic input has {w.shape[1]} columns")
    zeros = w.new_zeros(w.shape[0], FEATURES - OLD_FEATURES)
    out["inp.weight"] = torch.cat([w[:, :OLD_FEATURES], zeros, w[:, OLD_FEATURES:]], dim=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = {k: v for k, v in payload.items() if k != "optimizer"}
    migrated["actor"] = widen_actor(payload["actor"])
    if "actor_ema" in payload:
        migrated["actor_ema"] = widen_actor(payload["actor_ema"])
    if "critic" in payload:
        migrated["critic"] = widen_critic(payload["critic"])
    config = dict(payload.get("config", {}))
    config["migrated_from"] = str(args.source)
    config["torch50_contract"] = f"{FEATURES}/{ACTIONS}"
    migrated["config"] = config
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.target)
    print(f"migrated {args.source} -> {args.target} features={FEATURES} actions={ACTIONS}")


if __name__ == "__main__":
    main()
