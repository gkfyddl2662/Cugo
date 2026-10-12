"""Regularized PPO self-play for Shin Matgo 50.

On-policy PPO with an entropy bonus (MMD-style regularization toward uniform),
a critic that also sees the opponent's hidden hand (unbiased baseline, lower
variance), and an optional pool of frozen opponents. The same loop trains an
exploiter when every game pits the learner against one frozen target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cugo_ppo_nets50 import card_plane
from cugo_train50 import make_seeds

GOSTOP_KIND = 2  # decision-kind one-hot index for Go/Stop
DECISION_OFFSET = 446


@dataclass
class Rollout:
    features: torch.Tensor  # [N, 496] fp16
    legal: torch.Tensor  # [N, 177] bool
    actions: torch.Tensor  # [N] int64
    logp: torch.Tensor  # [N] fp32, behaviour log-prob
    opp_hand: torch.Tensor  # [N] int64 card mask
    returns: torch.Tensor  # [N] fp32, scaled final score for the acting player
    games: int
    terminal: int
    nagari: int
    decision_steps: int
    mean_abs_reward0: float
    learner_score: float  # mean scaled score of learner seats in pool games (nan if none)


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return (out[0] if isinstance(out, tuple) else out).float()


def _act(model: nn.Module, features: torch.Tensor, legal: torch.Tensor, greedy: bool):
    masked = _logits(model, features).masked_fill(~legal, -1.0e30)
    if greedy:
        actions = masked.argmax(dim=1)
    else:
        actions = torch.multinomial(torch.softmax(masked, dim=1), 1).squeeze(1)
    logp = torch.log_softmax(masked, dim=1).gather(1, actions.unsqueeze(1)).squeeze(1)
    return actions, logp


class RolloutStorage:
    """Fixed transition buffers reused every iteration.

    Variable-sized per-iteration allocations fragment PyTorch's cache; on Windows
    the overflow silently spills into shared system memory instead of failing.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = 0
        self.device = device
        self.size = 0
        self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        old = self.__dict__.get("tensors")
        d = self.device
        self.tensors = (
            torch.empty((capacity, 496), dtype=torch.float16, device=d),
            torch.empty((capacity, 177), dtype=torch.bool, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.float32, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
        )
        if old is not None:
            for new_t, old_t in zip(self.tensors, old):
                new_t[:self.size].copy_(old_t[:self.size])
        self.capacity = capacity

    def append(self, *parts: torch.Tensor) -> None:
        n = parts[0].shape[0]
        if self.size + n > self.capacity:
            self._allocate(int((self.size + n) * 1.25))
        for t, part in zip(self.tensors, parts):
            t[self.size:self.size + n].copy_(part)
        self.size += n

    def views(self) -> tuple[torch.Tensor, ...]:
        return tuple(t[:self.size] for t in self.tensors)


def collect(
    ext: Any,
    actor: nn.Module,
    pool: list[nn.Module],
    games: int,
    seed_offset: int,
    pool_fraction: float,
    pool_greedy: bool,
    reward_scale: float,
    storage: RolloutStorage,
    max_steps: int = 128,
) -> Rollout:
    device = torch.device("cuda")
    seeds, first = make_seeds(games, seed_offset, device)
    states = ext.create(seeds, first)

    opponent = torch.full((games,), -1, dtype=torch.int64, device=device)
    if pool and pool_fraction > 0.0:
        use_pool = torch.rand(games, device=device) < pool_fraction
        opponent = torch.where(
            use_pool, torch.randint(len(pool), (games,), device=device), opponent
        )
    learner_seat = torch.randint(2, (games,), device=device)

    final_reward0 = torch.zeros(games, device=device)
    final_nagari = torch.zeros(games, dtype=torch.bool, device=device)
    started = torch.zeros(games, dtype=torch.bool, device=device)
    active = torch.arange(games, device=device)
    storage.size = 0
    steps = 0

    actor.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(max_steps):
            features, legal, players, done, status = ext.observe_indexed(states, active)
            keep = ~done & (status == int(ext.STATUS_OK)) & legal.any(dim=1)
            if steps == 0:
                # Games resolved before any decision (initial Chongtong) carry no signal.
                rows = torch.nonzero(keep).squeeze(1)
                active, features, legal, players = (
                    t.index_select(0, rows) for t in (active, features, legal, players)
                )
                started.index_fill_(0, active, True)
            elif not bool(keep.all()):
                raise RuntimeError("active game has no legal action or bad status")
            if active.numel() == 0:
                break
            steps += 1

            player = players.to(torch.int64)
            opp = opponent.index_select(0, active)
            learner = (opp < 0) | (player == learner_seat.index_select(0, active))
            actions = torch.empty(active.numel(), dtype=torch.int64, device=device)

            rows = torch.nonzero(learner).squeeze(1)
            a, logp = _act(actor, features.index_select(0, rows), legal.index_select(0, rows), False)
            actions.index_copy_(0, rows, a)
            env_rows = active.index_select(0, rows)
            hands = ext.hands_indexed(states, env_rows)
            seat = player.index_select(0, rows)
            opp_hand = hands.gather(1, (1 - seat).unsqueeze(1)).squeeze(1)
            storage.append(features.index_select(0, rows), legal.index_select(0, rows),
                           a, logp, opp_hand, env_rows, seat)

            other = ~learner
            for k in range(len(pool)) if bool(other.any()) else ():
                rows = torch.nonzero(other & (opp == k)).squeeze(1)
                if rows.numel():
                    a, _ = _act(pool[k], features.index_select(0, rows),
                                legal.index_select(0, rows), pool_greedy)
                    actions.index_copy_(0, rows, a)

            reward0, new_done, nagari, step_status, _ = ext.step_indexed(states, active, actions)
            if not bool((step_status == int(ext.STATUS_OK)).all()):
                raise RuntimeError("step returned a non-OK status")
            final_reward0.index_copy_(0, active, torch.where(new_done, reward0, 0.0))
            final_nagari.index_copy_(0, active, new_done & nagari)
            active = active.index_select(0, torch.nonzero(~new_done).squeeze(1))
            if active.numel() == 0:
                break
        else:
            raise RuntimeError(f"games did not finish within {max_steps} steps")

    features, legal, actions, logp, opp_hand, env_ids, seat = storage.views()
    reward = final_reward0.index_select(0, env_ids)
    returns = torch.where(seat == 0, reward, -reward) / reward_scale

    pool_games = started & (opponent >= 0)
    learner_score = torch.where(learner_seat == 0, final_reward0, -final_reward0)[pool_games]
    return Rollout(
        features=features, legal=legal, actions=actions, logp=logp,
        opp_hand=opp_hand, returns=returns, games=games,
        terminal=int((started & ~final_nagari).sum()),
        nagari=int((started & final_nagari).sum()),
        decision_steps=steps,
        mean_abs_reward0=float(final_reward0[started].abs().mean()),
        learner_score=float(learner_score.mean()) if learner_score.numel() else float("nan"),
    )


@dataclass
class TrainStats:
    policy_loss: float
    value_loss: float
    entropy: float
    entropy_gostop: float
    kl: float
    clipfrac: float
    explained_var: float
    updates: int
    epochs: float


def _critic_input(features: torch.Tensor, opp_hand: torch.Tensor) -> torch.Tensor:
    return torch.cat((features, card_plane(opp_hand, features.dtype)), dim=1)


def ppo_update(
    actor: nn.Module,
    critic: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Rollout,
    epochs: int,
    minibatch: int,
    clip: float,
    entropy_coef: float,
    value_coef: float,
    target_kl: float,
    grad_clip: float,
) -> TrainStats:
    n = data.actions.numel()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        critic.eval()
        values = torch.cat([
            critic(_critic_input(data.features[i:i + minibatch], data.opp_hand[i:i + minibatch])).float()
            for i in range(0, n, minibatch)
        ])
    advantages = data.returns - values
    explained_var = 1.0 - float((data.returns - values).var() / data.returns.var().clamp_min(1e-8))
    gostop = data.features[:, DECISION_OFFSET + GOSTOP_KIND] > 0.5

    actor.train()
    critic.train()
    sums = torch.zeros(6, device=data.actions.device)
    gostop_entropy = torch.zeros(2, device=data.actions.device)
    updates = 0
    stop = False
    for epoch in range(epochs):
        perm = torch.randperm(n, device=data.actions.device)
        for i in range(0, n - minibatch + 1, minibatch):
            idx = perm[i:i + minibatch]
            feats = data.features[idx]
            legal = data.legal[idx]
            adv = advantages[idx]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = actor(feats).float()
                value = critic(_critic_input(feats, data.opp_hand[idx])).float()
            log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e30), dim=1)
            new_logp = log_probs.gather(1, data.actions[idx].unsqueeze(1)).squeeze(1)
            log_ratio = new_logp - data.logp[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            entropy_rows = -(log_probs.exp() * log_probs).sum(dim=1)
            entropy = entropy_rows.mean()
            value_loss = F.smooth_l1_loss(value, data.returns[idx])
            loss = policy_loss - entropy_coef * entropy + value_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], grad_clip)
            optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean()
                clipfrac = ((ratio - 1).abs() > clip).float().mean()
                sums += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach(),
                                     kl, clipfrac, torch.ones((), device=kl.device)))
                g = gostop[idx]
                gostop_entropy += torch.stack((entropy_rows[g].sum(), g.sum().float()))
            updates += 1
            if target_kl > 0 and updates % 8 == 0 and float(kl) > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break

    s = (sums / sums[5].clamp_min(1)).tolist()
    g_sum, g_count = gostop_entropy.tolist()
    return TrainStats(
        policy_loss=s[0], value_loss=s[1], entropy=s[2],
        entropy_gostop=g_sum / g_count if g_count else float("nan"),
        kl=s[3], clipfrac=s[4], explained_var=explained_var,
        updates=updates, epochs=updates / max(1, n // minibatch),
    )
