from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cugo_nfsp50 import AveragePolicyNet
from cugo_train50 import PolicyValueNet

class AverageTrainSummary:
    loss: float
    entropy: float
    top1_accuracy: float
    updates: int


def train_average_from_reservoir(
    model: AveragePolicyNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    reservoir: Any,
    batch_size: int,
    updates: int,
    grad_clip: float,
    use_amp: bool,
) -> AverageTrainSummary:
    if updates <= 0:
        return AverageTrainSummary(0.0, 0.0, 0.0, 0)
    if reservoir.size == 0:
        raise RuntimeError("cannot train average policy from an empty reservoir")

    metric_sums = torch.zeros(3, dtype=torch.float32, device=reservoir.device)
    model.train()
    for _ in range(updates):
        features, legal, actions, _targets = reservoir.sample(batch_size)
        optimizer.zero_grad(set_to_none=True)
        network_features = features if use_amp else features.float()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(network_features)
        masked_logits = logits.float().masked_fill(~legal, -1.0e30)
        loss = F.cross_entropy(masked_logits, actions)

        log_probs = F.log_softmax(masked_logits, dim=1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=1).mean()
        top1 = (masked_logits.argmax(dim=1) == actions).float().mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metric_sums.add_(
            torch.stack((loss.detach(), entropy.detach(), top1.detach()))
        )

    values = (metric_sums / float(updates)).tolist()
    return AverageTrainSummary(
        loss=float(values[0]),
        entropy=float(values[1]),
        top1_accuracy=float(values[2]),
        updates=updates,
    )


def save_nfsp_checkpoint(
    path: Path,
    br_model: PolicyValueNet,
    average_model: AveragePolicyNet,
    br_optimizer: torch.optim.Optimizer,
    average_optimizer: torch.optim.Optimizer,
    br_scaler: torch.amp.GradScaler,
    average_scaler: torch.amp.GradScaler,
    iteration: int,
    br_replay_size: int,
    average_reservoir_size: int,
    average_reservoir_seen: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": int(iteration),
            "br_model": br_model.state_dict(),
            "average_model": average_model.state_dict(),
            "br_optimizer": br_optimizer.state_dict(),
            "average_optimizer": average_optimizer.state_dict(),
            "br_scaler": br_scaler.state_dict(),
            "average_scaler": average_scaler.state_dict(),
            "br_replay_size": int(br_replay_size),
            "average_reservoir_size": int(average_reservoir_size),
            "average_reservoir_seen": int(average_reservoir_seen),
            "config": config,
        },
        path,
    )
