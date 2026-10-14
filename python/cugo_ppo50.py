"""Regularized PPO self-play for Shin Matgo 50.

On-policy PPO with an entropy bonus (MMD-style regularization toward uniform),
a critic that also sees the opponent's hidden hand (unbiased baseline, lower
variance), and an optional pool of frozen opponents. The same loop trains an
exploiter when every game pits the learner against one frozen target.

All-action mode (all_actions=True) branches legal actions of learner decisions
one step with ext.branch_indexed and scores the child with the oracle critic.
The branch replays the real stock, so the child already holds the flip that
action would get: Q(a) = +-V(child), or the final score when the action ends
the deal. Only actions that can move the policy are branched: those with
probability >= branch_min_prob, at decisions whose top action is below
branch_max_top, plus the sampled action there (a policy-gradient term scales
with the action's probability, so a 0.1% action's Q barely registers).

Two ways to learn from the Qs ("policy_target"):
- "pg": the clipped surrogate in expectation over the branched actions with
  A(a) = Q(a) - (their pi_old-weighted mean), plus the sampled action's
  (return - Q) control variate: unbiased whatever the critic's error.
  Decisions without branches fall back to the usual PPO term.
- "improve": move toward pi*(a) ~ pi_old(a) exp(A(a) / tau) by cross-entropy
  (MPO-style), tau chosen each iteration so KL(pi* || pi_old) averages
  improve_kl. Can raise a low-probability action that the Qs favour, which "pg"
  cannot; biased by critic error. Wants every action branched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cugo_ppo_nets50 import ACTIONS, FEATURES, card_plane, make_seeds, stock_planes, two_hot

GOSTOP_KIND = 2  # decision-kind one-hot index for Go/Stop
DECISION_OFFSET = 446


@dataclass
class Rollout:
    features: torch.Tensor  # [N, FEATURES] bf16
    legal: torch.Tensor  # [N, ACTIONS] bool
    actions: torch.Tensor  # [N] int64
    logp: torch.Tensor  # [N] fp32, behaviour log-prob
    opp_hand: torch.Tensor  # [N] int64 card mask
    stock: torch.Tensor  # [N] int64, next 8 stock draws (ext.stock_indexed)
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


def _act(model: nn.Module, features: torch.Tensor, legal: torch.Tensor, greedy: bool, full: bool = False):
    masked = _logits(model, features).masked_fill(~legal, -1.0e30)
    if greedy:
        actions = masked.argmax(dim=1)
    else:
        actions = torch.multinomial(torch.softmax(masked, dim=1), 1).squeeze(1)
    log_probs = torch.log_softmax(masked, dim=1)
    logp = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    return (actions, logp, log_probs) if full else (actions, logp)


Q_CHUNK = 65536  # children per critic pass (a 2048-wide hidden layer is 256 MB at this size)


def action_values(ext: Any, critic: nn.Module, states: torch.Tensor, env_rows: torch.Tensor,
                  seat: torch.Tensor, branch: torch.Tensor, reward_scale: float) -> torch.Tensor:
    """[R, ACTIONS] fp16 Q for the player to move of each action marked in branch (NaN elsewhere)."""
    rows, acts = torch.nonzero(branch, as_tuple=True)
    q = torch.full(branch.shape, float("nan"), dtype=torch.float16, device=branch.device)
    me = seat.index_select(0, rows)
    for i in range(0, rows.numel(), Q_CHUNK):
        r, a, m = rows[i:i + Q_CHUNK], acts[i:i + Q_CHUNK], me[i:i + Q_CHUNK]
        child, reward0, done = ext.branch_indexed(states, env_rows.index_select(0, r), a)[:3]
        ids = torch.arange(child.shape[0], device=child.device)
        features, _, players, _, _ = ext.observe_indexed(child, ids)
        mover = players.to(torch.int64)
        opp_hand = ext.hands_indexed(child, ids).gather(1, (1 - mover).unsqueeze(1)).squeeze(1)
        stock = ext.stock_indexed(child, ids) if critic.stock else None
        v = critic.value(critic(_critic_input(features, opp_hand, stock)))
        v = torch.where(mover == m, v, -v)  # zero-sum: the opponent's value is ours negated
        final = torch.where(m == 0, reward0, -reward0) / reward_scale
        q[r, a] = torch.where(done, final, v).to(torch.float16)
    return q


class RolloutStorage:
    """Fixed transition buffers reused every iteration.

    Variable-sized per-iteration allocations fragment PyTorch's cache; on Windows
    the overflow silently spills into shared system memory instead of failing.
    """

    def __init__(self, capacity: int, device: torch.device, all_actions: bool = False) -> None:
        self.capacity = 0
        self.device = device
        self.size = 0
        self.all_actions = all_actions
        self.extra: tuple[torch.Tensor, ...] = ()
        self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        old, old_extra = self.__dict__.get("tensors"), self.extra
        d = self.device
        if self.all_actions:  # Q of every action, behaviour log-probs of every action (fp16: 0.6 GB each)
            self.extra = (torch.empty((capacity, ACTIONS), dtype=torch.float16, device=d),
                          torch.empty((capacity, ACTIONS), dtype=torch.float16, device=d))
        self.tensors = (
            torch.empty((capacity, FEATURES), dtype=torch.bfloat16, device=d),
            torch.empty((capacity, ACTIONS), dtype=torch.bool, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.float32, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),
            torch.empty(capacity, dtype=torch.int64, device=d),  # stock (critic oracle input)
        )
        if old is not None:
            for new_t, old_t in zip((*self.tensors, *self.extra), (*old, *old_extra)):
                new_t[:self.size].copy_(old_t[:self.size])
        self.capacity = capacity

    def append(self, *parts: torch.Tensor) -> None:
        n = parts[0].shape[0]
        if self.size + n > self.capacity:
            self._allocate(int((self.size + n) * 1.25))
        for t, part in zip((*self.tensors, *self.extra), parts):
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
    critic: nn.Module | None = None,
    branch_min_prob: float = 0.0,
    branch_max_top: float = 2.0,
) -> Rollout:
    """Self-play games; with critic (and storage.all_actions) also every action's Q."""
    if storage.all_actions and critic is None:
        raise ValueError("all-action storage needs the critic")
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
    # Error checks accumulate on the GPU and are read once after the loop; each
    # per-step read would stall the CPU until the GPU caught up.
    bad = torch.zeros((), dtype=torch.int64, device=device)

    actor.eval()
    if critic is not None:
        critic.eval()
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
            else:
                bad += (~keep).sum()
            if active.numel() == 0:
                break
            steps += 1

            player = players.to(torch.int64)
            opp = opponent.index_select(0, active)
            learner = (opp < 0) | (player == learner_seat.index_select(0, active))
            actions = torch.empty(active.numel(), dtype=torch.int64, device=device)

            rows = torch.nonzero(learner).squeeze(1)
            own_legal = legal.index_select(0, rows)
            a, logp, log_probs = _act(actor, features.index_select(0, rows), own_legal, False, full=True)
            actions.index_copy_(0, rows, a)
            env_rows = active.index_select(0, rows)
            hands = ext.hands_indexed(states, env_rows)
            seat = player.index_select(0, rows)
            opp_hand = hands.gather(1, (1 - seat).unsqueeze(1)).squeeze(1)
            extra = ()
            if storage.all_actions:
                probs = log_probs.exp()
                open_ = probs.max(1).values < branch_max_top
                branch = own_legal & (probs >= branch_min_prob) & open_.unsqueeze(1)
                branch.scatter_(1, a.unsqueeze(1), branch.gather(1, a.unsqueeze(1)) | open_.unsqueeze(1))
                extra = (action_values(ext, critic, states, env_rows, seat, branch, reward_scale),
                         log_probs.clamp_min(-60.0).to(torch.float16))
            storage.append(features.index_select(0, rows), own_legal,
                           a, logp, opp_hand, env_rows, seat, ext.stock_indexed(states, env_rows), *extra)

            if opponent_model is not None:
                rows = torch.nonzero(~learner).squeeze(1)
                if rows.numel():
                    a, _ = _act(opponent_model, features.index_select(0, rows),
                                legal.index_select(0, rows), pool_greedy)
                    actions.index_copy_(0, rows, a)

            reward0, new_done, nagari, step_status, _ = ext.step_indexed(states, active, actions)
            bad += (step_status != int(ext.STATUS_OK)).sum()
            final_reward0.index_copy_(0, active, torch.where(new_done, reward0, 0.0))
            final_nagari.index_copy_(0, active, new_done & nagari)
            active = active.index_select(0, torch.nonzero(~new_done).squeeze(1))
            if active.numel() == 0:
                break
        else:
            raise RuntimeError(f"games did not finish within {max_steps} steps")
    if int(bad):
        raise RuntimeError("an active game had no legal action, a bad status or a failed step")

    features, legal, actions, logp, opp_hand, env_ids, seat, stock = storage.views()
    reward = final_reward0.index_select(0, env_ids)
    returns = torch.where(seat == 0, reward, -reward) / reward_scale

    pool_games = started & (opponent >= 0)
    learner_score = torch.where(learner_seat == 0, final_reward0, -final_reward0)[pool_games]
    return Rollout(
        features=features, legal=legal, actions=actions, logp=logp,
        opp_hand=opp_hand, stock=stock, returns=returns, games=games,
        terminal=int((started & ~final_nagari).sum()),
        nagari=int((started & final_nagari).sum()),
        decision_steps=steps,
        mean_abs_reward0=float(final_reward0[started].abs().mean()),
        learner_score=float(learner_score.mean()) if learner_score.numel() else float("nan"),
    )


def _branch_advantages(q_raw: torch.Tensor, old_logp: torch.Tensor, legal: torch.Tensor):
    """Branched mask, A(a) = Q(a) - pi_old-weighted mean Q over the branched actions (0 elsewhere), Q (0 for NaN)."""
    branched = ~torch.isnan(q_raw) & legal
    q = torch.where(branched, q_raw.float(), 0.0)
    p = old_logp.exp() * branched
    mean = (p * q).sum(1, keepdim=True) / p.sum(1, keepdim=True).clamp_min(1e-12)
    return branched, torch.where(branched, q - mean, 0.0), q


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
    belief_loss: float = float("nan")
    q_explained_var: float = float("nan")  # 1 - Var(return - Q(taken action)) / Var(return), branched rows
    inv_temp: float = float("nan")  # 1 / tau of the "improve" target


def _critic_input(features: torch.Tensor, opp_hand: torch.Tensor, stock: torch.Tensor | None = None) -> torch.Tensor:
    parts = [features, card_plane(opp_hand, features.dtype)]
    if stock is not None:
        parts.append(stock_planes(stock, features.dtype))
    return torch.cat(parts, dim=1)


def gae_advantages(values: torch.Tensor, returns: torch.Tensor, env_ids: torch.Tensor,
                   seat: torch.Tensor, lam: float) -> torch.Tensor:
    """GAE(lambda), gamma=1, along each (game, seat) sequence of own decisions.

    Rewards arrive only at the end, so delta_t = V(h_next_own) - V(h_t) and the
    last decision bootstraps from the final score. lam=1 is the Monte-Carlo
    advantage R - V(h_t). Rows must be in step-major order within each game.
    """
    if lam >= 1.0:
        return returns - values
    n = values.numel()
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


class PpoUpdater:
    """PPO minibatch step, optionally captured once as a CUDA graph and replayed.

    Per-update work is ~100 small kernels; eager launches left the GPU waiting on
    the CPU. All step inputs live in static buffers: the rollout storage itself,
    plus advantage/return copies and the minibatch index vector.
    """

    def __init__(self, actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer,
                 storage: RolloutStorage, minibatch: int, clip: float, value_coef: float,
                 grad_clip: float, use_graph: bool = True, belief: nn.Module | None = None,
                 belief_coef: float = 0.0, mc_coef: float = 1.0, policy_target: str = "pg",
                 improve_kl: float = 0.02) -> None:
        self.actor, self.critic, self.optimizer, self.storage = actor, critic, optimizer, storage
        self.minibatch, self.clip, self.value_coef, self.grad_clip = minibatch, clip, value_coef, grad_clip
        self.belief, self.belief_coef = belief, belief_coef
        self.mc_coef = mc_coef  # weight of the sampled action's (return - Q) term in all-action mode
        if policy_target not in ("pg", "improve"):
            raise ValueError("policy_target must be pg or improve")
        self.policy_target, self.improve_kl = policy_target, improve_kl
        self.use_graph = use_graph
        d = storage.device
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        # Gradients are clipped per network, so a critic loss on another scale (bins)
        # cannot shrink the actor's step.
        self.clip_groups = [list(actor.parameters()), list(critic.parameters())]
        if belief is not None:
            self.clip_groups.append(list(belief.parameters()))
        self.idx = torch.zeros(minibatch, dtype=torch.int64, device=d)
        self.entropy_coef = torch.zeros((), device=d)
        # policy, value, entropy, kl, clipfrac, count, gostop entropy sum, gostop count, belief
        self.sums = torch.zeros(9, device=d)
        self.kl = torch.zeros((), device=d)
        self.inv_temp = torch.zeros((), device=d)  # 1 / tau of the "improve" target
        self.graph: torch.cuda.CUDAGraph | None = None
        self._bind()

    def _bind(self) -> None:
        self.bound = tuple(t.data_ptr() for t in (*self.storage.tensors, *self.storage.extra))
        self.adv = torch.zeros(self.storage.capacity, device=self.storage.device)
        self.ret = torch.zeros(self.storage.capacity, device=self.storage.device)
        self.graph = None

    def _step(self) -> None:
        features, legal_all, actions, logp_old, opp_hand, _, _, stock = self.storage.tensors
        idx = self.idx
        feats = features[idx]
        legal = legal_all[idx]
        adv = self.adv[idx]
        ret = self.ret[idx]
        critic = self.critic
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            if self.belief is not None:
                logits, hidden = self.actor.forward_hidden(feats)
                belief_logits = self.belief(hidden).float()
            else:
                logits = self.actor(feats)
            logits = logits.float()
            value = critic(_critic_input(feats, opp_hand[idx], stock[idx] if critic.stock else None)).float()
        log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e30), dim=1)
        new_logp = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
        log_ratio = new_logp - logp_old[idx]
        ratio = log_ratio.exp()
        plain = -torch.min(ratio * adv, ratio.clamp(1 - self.clip, 1 + self.clip) * adv)
        if self.storage.all_actions:
            q_store, old_store = self.storage.extra
            old_all = old_store[idx].float()
            branched, adv_all, q = _branch_advantages(q_store[idx], old_all, legal)
            has = branched.any(1)
            if self.policy_target == "improve":
                target = torch.softmax((old_all + self.inv_temp * adv_all).masked_fill(~legal, -1.0e30), 1)
                rows = -(target * torch.where(legal, log_probs, 0.0)).sum(1)
            else:
                # Expected clipped surrogate over the branched actions under the behaviour policy,
                # plus the sampled action's (return - Q) term; the sampled action is always branched.
                p_old = old_all.exp() * branched
                r_all = (log_probs - old_all).clamp_max(20.0).exp()
                surr_all = torch.minimum(r_all * adv_all, r_all.clamp(1 - self.clip, 1 + self.clip) * adv_all)
                residual = ret - q.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
                surr_mc = torch.min(ratio * residual, ratio.clamp(1 - self.clip, 1 + self.clip) * residual)
                rows = -((p_old * surr_all).sum(1) + self.mc_coef * surr_mc)
            policy_loss = torch.where(has, rows, plain).mean()
        else:
            policy_loss = plain.mean()
        entropy_rows = -(log_probs.exp() * log_probs).sum(dim=1)
        entropy = entropy_rows.mean()
        if critic.value_bins:  # cross-entropy against the two-hot return (mean-preserving bins)
            value_loss = -(two_hot(ret, critic.support) * torch.log_softmax(value, -1)).sum(-1).mean()
        else:
            value_loss = F.smooth_l1_loss(value, ret)
        loss = policy_loss - self.entropy_coef * entropy + self.value_coef * value_loss
        if self.belief is not None:
            hand = card_plane(opp_hand[idx], torch.float32)
            belief_loss = F.binary_cross_entropy_with_logits(belief_logits, hand)
            loss = loss + self.belief_coef * belief_loss
        else:
            belief_loss = torch.zeros((), device=loss.device)
        loss.backward()
        for group in self.clip_groups:
            nn.utils.clip_grad_norm_(group, self.grad_clip, foreach=True)
        self.optimizer.step()
        with torch.no_grad():
            kl = ((ratio - 1) - log_ratio).mean()
            self.kl.copy_(kl)
            g = (feats[:, DECISION_OFFSET + GOSTOP_KIND] > 0.5).float()
            self.sums += torch.stack((
                policy_loss.detach(), value_loss.detach(), entropy.detach(), kl,
                ((ratio - 1).abs() > self.clip).float().mean(), torch.ones_like(kl),
                (entropy_rows.detach() * g).sum(), g.sum(), belief_loss.detach()))

    def _run(self) -> None:
        if not self.use_graph:
            self.optimizer.zero_grad(set_to_none=True)
            self._step()
            return
        if self.graph is None:
            saved = self.sums.clone()
            weights = [p.detach().clone() for p in self.params]
            moments = {p: {k: v.clone() for k, v in self.optimizer.state[p].items() if torch.is_tensor(v)}
                       for p in self.params if p in self.optimizer.state}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.optimizer.zero_grad(set_to_none=True)
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            # Undo the warm-up updates (in place: the graph keeps these tensors), so the
            # minibatch is applied once, by the replay below.
            with torch.no_grad():
                for p, w in zip(self.params, weights):
                    p.copy_(w)
                for p in self.params:
                    before = moments.get(p)
                    for k, v in self.optimizer.state[p].items():
                        if torch.is_tensor(v):
                            v.copy_(before[k]) if before is not None else v.zero_()
            self.optimizer.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            self.sums.copy_(saved)
        self.graph.replay()

    def _solve_inv_temp(self, n: int, legal_all: torch.Tensor, sample: int = 65536) -> float:
        """1/tau with mean KL(pi* || pi_old) = improve_kl over branched decisions (bisection in log space)."""
        q_store, old_store = self.storage.extra
        rows = torch.randint(n, (min(n, sample),), device=legal_all.device)
        legal = legal_all[rows]
        old = old_store[rows].float().masked_fill(~legal, -1.0e30)
        old = old - torch.logsumexp(old, 1, keepdim=True)
        branched, adv, _ = _branch_advantages(q_store[rows], old, legal)
        keep = branched.any(1)
        old, adv, legal = old[keep], adv[keep], legal[keep]

        def kl(beta: float) -> float:
            z = (old + beta * adv).masked_fill(~legal, -1.0e30)
            log_target = torch.log_softmax(z, 1)
            return float((log_target.exp() * torch.where(legal, log_target - old, 0.0)).sum(1).mean())

        lo, hi = -8.0, 12.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if kl(float(torch.exp(torch.tensor(mid)))) < self.improve_kl else (lo, mid)
        return float(torch.exp(torch.tensor(0.5 * (lo + hi))))

    def _advantages(self, values: torch.Tensor, returns: torch.Tensor, lam: float) -> torch.Tensor:
        n = values.numel()
        return gae_advantages(values, returns, self.storage.tensors[5][:n], self.storage.tensors[6][:n], lam)

    def update(self, data: Rollout, epochs: int, entropy_coef: float, target_kl: float,
               gae_lambda: float = 1.0) -> TrainStats:
        if tuple(t.data_ptr() for t in (*self.storage.tensors, *self.storage.extra)) != self.bound:
            self._bind()
        n = data.actions.numel()
        chunk = 65536
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.critic.eval()
            stock = self.critic.stock
            values = torch.cat([
                self.critic.value(self.critic(_critic_input(data.features[i:i + chunk], data.opp_hand[i:i + chunk],
                                                            data.stock[i:i + chunk] if stock else None)))
                for i in range(0, n, chunk)
            ])
        advantages = self._advantages(values, data.returns, gae_lambda)
        self.adv[:n].copy_(advantages)
        self.ret[:n].copy_(advantages + values)
        explained_var = 1.0 - float((data.returns - values).var() / data.returns.var().clamp_min(1e-8))
        q_explained_var = float("nan")
        if self.storage.all_actions:
            q_taken = self.storage.extra[0][:n].gather(1, data.actions.unsqueeze(1)).squeeze(1).float()
            ok = ~torch.isnan(q_taken)
            ret_b = data.returns[ok]
            q_explained_var = 1.0 - float((ret_b - q_taken[ok]).var() / ret_b.var().clamp_min(1e-8))
            if self.policy_target == "improve":
                self.inv_temp.fill_(self._solve_inv_temp(n, data.legal))

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
            updates=updates, epochs=updates / max(1, n // self.minibatch), belief_loss=s[8] / count,
            q_explained_var=q_explained_var,
            inv_temp=float(self.inv_temp) if self.policy_target == "improve" else float("nan"),
        )
