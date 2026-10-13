"""Widen a PPO checkpoint from the 560-feature Torch50 contract to the 720-feature one.

Contract v3 puts the neutral-ppuk month plane into legacy padding slots 479..490 and
appends three public-history card planes (own played, opponent played, opponent
cards known from a shake) at 560..709, padded to 720. Padding inputs were always zero,
so their weights were never trained; they are zeroed here together with the new
columns. The migrated network therefore behaves exactly as before until training
starts using the new inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cugo_ppo_nets50 import FEATURES, ORACLE_FEATURES

OLD_FEATURES = 560
PADDING = slice(479, 496)  # legacy padding, now partly the neutral-ppuk plane


def widen(w: torch.Tensor, oracle: bool) -> torch.Tensor:
    expected = OLD_FEATURES + (ORACLE_FEATURES if oracle else 0)
    if w.shape[1] != expected:
        raise ValueError(f"input has {w.shape[1]} columns, expected {expected}")
    public = w[:, :OLD_FEATURES].clone()
    public[:, PADDING] = 0
    parts = [public, w.new_zeros(w.shape[0], FEATURES - OLD_FEATURES)]
    if oracle:  # critic input is [public features | opponent hand plane]
        parts.append(w[:, OLD_FEATURES:])
    return torch.cat(parts, dim=1)


def widen_state(state: dict[str, torch.Tensor], oracle: bool) -> dict[str, torch.Tensor]:
    out = dict(state)
    out["inp.weight"] = widen(state["inp.weight"], oracle)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    payload = torch.load(args.source, map_location="cpu", weights_only=False)
    migrated = {k: v for k, v in payload.items() if k != "optimizer"}
    migrated["actor"] = widen_state(payload["actor"], oracle=False)
    if "actor_ema" in payload:
        migrated["actor_ema"] = widen_state(payload["actor_ema"], oracle=False)
    if "critic" in payload:
        migrated["critic"] = widen_state(payload["critic"], oracle=True)
    config = dict(payload.get("config", {}))
    config["migrated_from"] = str(args.source)
    config["torch50_contract"] = f"{FEATURES}"
    migrated["config"] = config
    args.target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(migrated, args.target)
    print(f"migrated {args.source} -> {args.target} features={FEATURES}")


if __name__ == "__main__":
    main()
