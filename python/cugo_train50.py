from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


SIGNED_GOLDEN = -7046029254386353131
SEED_BASE = 0x243F6A8885A308D3


class PolicyValueNet(nn.Module):
    def __init__(self, features: int, actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


def make_seeds(
    batch: int,
    offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(batch, dtype=torch.int64, device=device)
    seeds = index.mul(SIGNED_GOLDEN).add(SEED_BASE + offset)
    first = index.bitwise_and(1)
    return seeds, first


@dataclass
class SelfPlayBatch:
    features: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
    value_targets: torch.Tensor
    games: int
    terminal: int
    nagari: int
    decision_steps: int
    mean_abs_reward0: float

    @property
    def transitions(self) -> int:
        return int(self.actions.numel())


class GpuReplayBuffer:
    def __init__(
        self,
        capacity: int,
        feature_count: int,
        action_count: int,
        device: torch.device,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.feature_count = int(feature_count)
        self.action_count = int(action_count)
        self.device = device
        self.features = torch.empty(
            (capacity, feature_count), dtype=torch.float16, device=device
        )
        self.legal = torch.empty(
            (capacity, action_count), dtype=torch.bool, device=device
        )
        self.actions = torch.empty(capacity, dtype=torch.int64, device=device)
        self.value_targets = torch.empty(capacity, dtype=torch.float32, device=device)
        self.position = 0
        self.size = 0

    @property
    def storage_bytes(self) -> int:
        tensors = (self.features, self.legal, self.actions, self.value_targets)
        return sum(t.numel() * t.element_size() for t in tensors)

    def add(self, batch: SelfPlayBatch) -> None:
        count = batch.transitions
        if count == 0:
            return

        features = batch.features
        legal = batch.legal
        actions = batch.actions
        targets = batch.value_targets

        if count >= self.capacity:
            start = count - self.capacity
            features = features[start:]
            legal = legal[start:]
            actions = actions[start:]
            targets = targets[start:]
            count = self.capacity
            self.features.copy_(features)
            self.legal.copy_(legal)
            self.actions.copy_(actions)
            self.value_targets.copy_(targets)
            self.position = 0
            self.size = self.capacity
            return

        first = min(count, self.capacity - self.position)
        second = count - first
        end = self.position + first

        self.features[self.position:end].copy_(features[:first])
        self.legal[self.position:end].copy_(legal[:first])
        self.actions[self.position:end].copy_(actions[:first])
        self.value_targets[self.position:end].copy_(targets[:first])

        if second:
            self.features[:second].copy_(features[first:])
            self.legal[:second].copy_(legal[first:])
            self.actions[:second].copy_(actions[first:])
            self.value_targets[:second].copy_(targets[first:])

        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty replay buffer")
        index = torch.randint(
            self.size,
            (batch_size,),
            dtype=torch.int64,
            device=self.device,
        )
        return (
            self.features[index],
            self.legal[index],
            self.actions[index],
            self.value_targets[index],
        )


def _masked_actions(
    logits: torch.Tensor,
    legal: torch.Tensor,
    active: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits32 = logits.float()
    masked = logits32.masked_fill(~legal, -1.0e30)
    if temperature <= 0.0:
        actions = masked.argmax(dim=1)
    else:
        probabilities = torch.softmax(masked / temperature, dim=1)
        actions = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    return torch.where(active, actions, torch.zeros_like(actions)).to(torch.int64)


def collect_selfplay(
    ext: Any,
    model: nn.Module,
    batch: int,
    seed_offset: int,
    max_steps: int,
    temperature: float,
    reward_scale: float,
    use_amp: bool,
) -> SelfPlayBatch:
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")

    device = torch.device("cuda")
    seeds, first = make_seeds(batch, seed_offset, device)
    states = ext.create(seeds, first)
    final_reward0 = torch.zeros(batch, dtype=torch.float32, device=device)
    final_nagari = torch.zeros(batch, dtype=torch.bool, device=device)

    feature_chunks: list[torch.Tensor] = []
    legal_chunks: list[torch.Tensor] = []
    action_chunks: list[torch.Tensor] = []
    player_chunks: list[torch.Tensor] = []
    env_chunks: list[torch.Tensor] = []

    model.eval()
    decision_steps = 0
    with torch.inference_mode():
        for decision_steps in range(1, max_steps + 1):
            features, legal, players, observed_done, status = ext.observe(states)
            active = ~observed_done
            if not bool(active.any().item()):
                break
            if bool((status[active] != int(ext.STATUS_OK)).any().item()):
                bad = int((status[active] != int(ext.STATUS_OK)).sum().item())
                raise RuntimeError(f"observe returned non-OK status for {bad} active environments")
            if bool((legal[active].sum(dim=1) == 0).any().item()):
                raise RuntimeError("an active environment has no legal actions")

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits, _values = model(features)
            actions = _masked_actions(logits, legal, active, temperature)

            active_ids = torch.nonzero(active, as_tuple=False).squeeze(1)
            feature_chunks.append(features[active_ids].to(torch.float16))
            legal_chunks.append(legal[active_ids])
            action_chunks.append(actions[active_ids])
            player_chunks.append(players[active_ids])
            env_chunks.append(active_ids)

            reward0, new_done, nagari, step_status, _committed = ext.step(states, actions)
            if bool((step_status[active] != int(ext.STATUS_OK)).any().item()):
                bad = int((step_status[active] != int(ext.STATUS_OK)).sum().item())
                raise RuntimeError(f"step returned non-OK status for {bad} active environments")

            just_finished = active & new_done
            final_reward0 = torch.where(just_finished, reward0, final_reward0)
            final_nagari |= just_finished & nagari
            if bool(new_done.all().item()):
                break
        else:
            raise RuntimeError(f"not all games finished within {max_steps} decision steps")

    if not feature_chunks:
        raise RuntimeError("self-play produced no training transitions")

    features = torch.cat(feature_chunks, dim=0)
    legal = torch.cat(legal_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    players = torch.cat(player_chunks, dim=0)
    env_ids = torch.cat(env_chunks, dim=0)

    reward_for_transition = final_reward0[env_ids]
    signed_reward = torch.where(
        players == 0, reward_for_transition, -reward_for_transition
    )
    value_targets = signed_reward / reward_scale

    terminal = int((~final_nagari).sum().item())
    nagari_count = int(final_nagari.sum().item())
    return SelfPlayBatch(
        features=features,
        legal=legal,
        actions=actions,
        value_targets=value_targets,
        games=batch,
        terminal=terminal,
        nagari=nagari_count,
        decision_steps=decision_steps,
        mean_abs_reward0=float(final_reward0.abs().mean().item()),
    )


@dataclass
class TrainSummary:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    mean_abs_advantage: float
    updates: int


def train_from_replay(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    replay: GpuReplayBuffer,
    batch_size: int,
    updates: int,
    value_coef: float,
    entropy_coef: float,
    grad_clip: float,
    use_amp: bool,
) -> TrainSummary:
    if updates <= 0:
        return TrainSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_advantage = 0.0

    model.train()
    for _ in range(updates):
        features, legal, actions, targets = replay.sample(batch_size)
        optimizer.zero_grad(set_to_none=True)
        network_features = features if use_amp else features.float()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits, values = model(network_features)

        logits32 = logits.float()
        values32 = values.float()
        masked_logits = logits32.masked_fill(~legal, -1.0e30)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        probabilities = torch.softmax(masked_logits, dim=1)
        chosen_log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        advantage = (targets - values32.detach()).clamp(-4.0, 4.0)
        policy_loss = -(advantage * chosen_log_prob).mean()
        value_loss = F.smooth_l1_loss(values32, targets)
        entropy = -(probabilities * log_probs).sum(dim=1).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_policy += float(policy_loss.detach().item())
        total_value += float(value_loss.detach().item())
        total_entropy += float(entropy.detach().item())
        total_advantage += float(advantage.detach().abs().mean().item())

    inv = 1.0 / updates
    return TrainSummary(
        loss=total_loss * inv,
        policy_loss=total_policy * inv,
        value_loss=total_value * inv,
        entropy=total_entropy * inv,
        mean_abs_advantage=total_advantage * inv,
        updates=updates,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    iteration: int,
    replay_size: int,
    config: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "replay_size": replay_size,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
        },
        path,
    )
