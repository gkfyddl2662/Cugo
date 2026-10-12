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

from cugo_ppo_nets50 import ACTIONS, FEATURES, card_plane
from cugo_train50 import make_seeds

GOSTOP_KIND = 2  # decision-kind one-hot index for Go/Stop
DECISION_OFFSET = 446


@dataclass
class Rollout:
    features: torch.Tensor  # [N, FEATURES] bf16
    legal: torch.Tensor  # [N, ACTIONS] bool
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
            torch.empty((capacity, FEATURES), dtype=torch.bfloat16, device=d),
            torch.empty((capacity, ACTIONS), dtype=torch.bool, device=d),
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

    # One pool opponent per iteration: a forward pass per pool member every step
    # made self-play slow down linearly as snapshots accumulated.
    opponent_model = pool[int(torch.randint(len(pool), ()))] if pool and pool_fraction > 0.0 else None
    opponent = torch.full((games,), -1, dtype=torch.int64, device=device)
    if opponent_model is not None:
        opponent.masked_fill_(torch.rand(games, device=device) < pool_fraction, 0)
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

            if opponent_model is not None:
                rows = torch.nonzero(~learner).squeeze(1)
                if rows.numel():
                    a, _ = _act(opponent_model, features.index_select(0, rows),
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


class PpoUpdater:
    """PPO minibatch step, optionally captured once as a CUDA graph and replayed.

    Per-update work is ~100 small kernels; eager launches left the GPU waiting on
    the CPU. All step inputs live in static buffers: the rollout storage itself,
    plus advantage/return copies and the minibatch index vector.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer,
                 storage: RolloutStorage, minibatch: int, clip: float, value_coef: float,
                 grad_clip: float, use_graph: bool = True) -> None:
        self.actor, self.critic, self.optimizer, self.storage = actor, critic, optimizer, storage
        self.minibatch, self.clip, self.value_coef, self.grad_clip = minibatch, clip, value_coef, grad_clip
        self.use_graph = use_graph
        d = storage.device
        self.params = [*actor.parameters(), *critic.parameters()]
        self.idx = torch.zeros(minibatch, dtype=torch.int64, device=d)
        self.entropy_coef = torch.zeros((), device=d)
        # policy, value, entropy, kl, clipfrac, count, gostop entropy sum, gostop count
        self.sums = torch.zeros(8, device=d)
        self.kl = torch.zeros((), device=d)
        self.graph: torch.cuda.CUDAGraph | None = None
        self._bind()

    def _bind(self) -> None:
        self.bound = tuple(t.data_ptr() for t in self.storage.tensors)
        self.adv = torch.zeros(self.storage.capacity, device=self.storage.device)
        self.ret = torch.zeros(self.storage.capacity, device=self.storage.device)
        self.graph = None

    def _step(self) -> None:
        features, legal_all, actions, logp_old, opp_hand, _, _ = self.storage.tensors
        idx = self.idx
        feats = features[idx]
        legal = legal_all[idx]
        adv = self.adv[idx]
        ret = self.ret[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            logits = self.actor(feats).float()
            value = self.critic(_critic_input(feats, opp_hand[idx])).float()
        log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e30), dim=1)
        new_logp = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
        log_ratio = new_logp - logp_old[idx]
        ratio = log_ratio.exp()
        policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
        entropy_rows = -(log_probs.exp() * log_probs).sum(dim=1)
        entropy = entropy_rows.mean()
        value_loss = F.smooth_l1_loss(value, ret)
        loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.grad_clip, foreach=True)
        self.optimizer.step()
        with torch.no_grad():
            kl = ((ratio - 1) - log_ratio).mean()
            self.kl.copy_(kl)
            g = (feats[:, DECISION_OFFSET + GOSTOP_KIND] > 0.5).float()
            self.sums += torch.stack((
                policy_loss.detach(), value_loss.detach(), entropy.detach(), kl,
                ((ratio - 1).abs() > self.clip).float().mean(), torch.ones_like(kl),
                (entropy_rows.detach() * g).sum(), g.sum()))

    def _run(self) -> None:
        if not self.use_graph:
            self.optimizer.zero_grad(set_to_none=True)
            self._step()
            return
        if self.graph is None:
            saved = self.sums.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):  # warm-up updates are real updates on the current minibatch
                    self.optimizer.zero_grad(set_to_none=True)
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.optimizer.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            self.sums.copy_(saved)
        self.graph.replay()

    def _advantages(self, values: torch.Tensor, returns: torch.Tensor, lam: float) -> torch.Tensor:
        """GAE(lambda), gamma=1, along each (game, seat) sequence of own decisions.

        Rewards arrive only at the end, so delta_t = V(h_next_own) - V(h_t) and the
        last decision bootstraps from the final score. lam=1 is the Monte-Carlo
        advantage R - V(h_t).
        """
        if lam >= 1.0:
            return returns - values
        n = values.numel()
        env_ids, seat = self.storage.tensors[5][:n], self.storage.tensors[6][:n]
        order = torch.sort(env_ids * 2 + seat, stable=True).indices  # storage is step-major
        key = (env_ids * 2 + seat)[order]
        has_next = torch.zeros(n, dtype=torch.bool, device=values.device)
        has_next[:-1] = key[1:] == key[:-1]
        v = values[order]
        delta = torch.where(has_next, v.roll(-1), returns[order]) - v
        pos = torch.arange(n, device=values.device)
        last = torch.where(has_next, n, pos).flip(0).cummin(0).values.flip(0)
        to_end = last - pos
        adv = torch.zeros_like(v)
        for d in range(int(to_end.max()) + 1):
            rows = torch.nonzero(to_end == d).squeeze(1)
            adv[rows] = delta[rows] + (lam * adv[rows + 1] if d else 0.0)
        out = torch.empty_like(adv)
        out[order] = adv
        return out

    def update(self, data: Rollout, epochs: int, entropy_coef: float, target_kl: float,
               gae_lambda: float = 1.0) -> TrainStats:
        if tuple(t.data_ptr() for t in self.storage.tensors) != self.bound:
            self._bind()
        n = data.actions.numel()
        chunk = 65536
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.critic.eval()
            values = torch.cat([
                self.critic(_critic_input(data.features[i:i + chunk], data.opp_hand[i:i + chunk])).float()
                for i in range(0, n, chunk)
            ])
        advantages = self._advantages(values, data.returns, gae_lambda)
        self.adv[:n].copy_(advantages)
        self.ret[:n].copy_(advantages + values)
        explained_var = 1.0 - float((data.returns - values).var() / data.returns.var().clamp_min(1e-8))

        self.actor.train()
        self.critic.train()
        self.entropy_coef.fill_(entropy_coef)
        self.sums.zero_()
        updates = 0
        stop = False
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.idx.device)
            for i in range(0, n - self.minibatch + 1, self.minibatch):
                self.idx.copy_(perm[i:i + self.minibatch])
                self._run()
                updates += 1
                if target_kl > 0 and updates % 8 == 0 and float(self.kl) > 1.5 * target_kl:
                    stop = True
                    break
            if stop:
                break

        s = self.sums.tolist()
        count = max(1.0, s[5])
        return TrainStats(
            policy_loss=s[0] / count, value_loss=s[1] / count, entropy=s[2] / count,
            entropy_gostop=s[6] / s[7] if s[7] else float("nan"),
            kl=s[3] / count, clipfrac=s[4] / count, explained_var=explained_var,
            updates=updates, epochs=updates / max(1, n // self.minibatch),
        )
