"""PPO self-play for the event-sequence policies (cugo_seq50.SeqNet).

Same objective as train_ppo50.py (clipped PPO, entropy bonus, oracle critic,
GAE, actor EMA, snapshot pool); only the inputs differ. Each iteration keeps the
event log of every game; a decision row stores how many events it saw, so the
update can rebuild exactly the sequence the policy acted on.

Self-play runs as replayed CUDA graphs over KV-cached policies
(cugo_seq_selfplay50); the update reads all minibatch shapes from the GPU once
per iteration, so its inner loop never waits on the host.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cugo_contract50 import CONTRACT, load_checkpoint
from cugo_eval50 import load_policy_checkpoint
from cugo_ppo50 import DECISION_OFFSET, GOSTOP_KIND, TrainStats
from cugo_ppo_nets50 import ACTIONS, FEATURES, make_seeds
from cugo_seq50 import MAX_EVENTS, History, HistBatch, policy_logits, seq_actor, seq_critic, stored_columns
from cugo_seq_selfplay50 import GameLog, WaveCollector, make_opponent_slot
from cugo_torch50_ext import load_extension
from league50 import match

EVAL_SEED_OFFSET = 900000000
EVAL_CHUNK = 8192


class SeqStorage:
    """Reused per-decision buffers (see RolloutStorage) plus the event-count column."""

    NAMES = ("features", "legal", "actions", "logp", "opp_hand", "env_ids", "seat", "count")

    def __init__(self, capacity: int, device: torch.device, columns: list[int] | None = None) -> None:
        # Events-mode policies read only a few feature columns; storing just those
        # saves ~1.7 GB at 65,536 games per iteration.
        self.device, self.capacity, self.size = device, 0, 0
        self.columns = None if columns is None else torch.tensor(columns, device=device)
        self.width = FEATURES if columns is None else len(columns)
        gostop = DECISION_OFFSET + GOSTOP_KIND
        self.gostop_column = gostop if columns is None else columns.index(gostop)
        self._allocate(capacity)

    def _allocate(self, capacity: int) -> None:
        d = self.device
        shapes = {"features": ((capacity, self.width), torch.bfloat16), "legal": ((capacity, ACTIONS), torch.bool),
                  "logp": ((capacity,), torch.float32)}
        for name in self.NAMES:
            shape, dtype = shapes.get(name, ((capacity,), torch.int64))
            new = torch.empty(shape, dtype=dtype, device=d)
            if self.size:
                new[:self.size].copy_(getattr(self, name)[:self.size])
            setattr(self, name, new)
        self.capacity = capacity

    def append(self, **parts: torch.Tensor) -> None:
        n = parts["actions"].shape[0]
        if self.size + n > self.capacity:
            # Rare: longer games as the policy learns to Go. Grow once, with headroom.
            self._allocate(int((self.size + n) * 1.25))
        for name, part in parts.items():
            getattr(self, name)[self.size:self.size + n].copy_(part)
        self.size += n


@dataclass
class SeqRollout:
    hist: History | GameLog
    returns: torch.Tensor
    games: int
    terminal: int
    nagari: int
    decision_steps: int
    mean_abs_reward0: float
    learner_score: float


def _act(model: nn.Module, features, legal, hist: HistBatch, greedy: bool):
    masked = policy_logits(model, features, legal, hist).masked_fill(~legal, -1.0e30)
    actions = masked.argmax(1) if greedy else torch.multinomial(torch.softmax(masked, 1), 1).squeeze(1)
    return actions, torch.log_softmax(masked, 1).gather(1, actions.unsqueeze(1)).squeeze(1)


def collect_eager(ext: Any, actor: nn.Module, pool: list[nn.Module], games: int, seed_offset: int,
            pool_fraction: float, pool_greedy: bool, reward_scale: float, storage: SeqStorage,
            max_steps: int = 128) -> SeqRollout:
    device = storage.device
    seeds, first = make_seeds(games, seed_offset, device)
    states = ext.create(seeds, first)
    hist = History(ext, states)

    opponent_model = pool[int(torch.randint(len(pool), ()))] if pool and pool_fraction > 0.0 else None
    vs_pool = torch.zeros(games, dtype=torch.bool, device=device)
    if opponent_model is not None:
        vs_pool = torch.rand(games, device=device) < pool_fraction
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
                rows = torch.nonzero(keep).squeeze(1)
                active, features, legal, players = (t.index_select(0, rows) for t in (active, features, legal, players))
                started.index_fill_(0, active, True)
            elif not bool(keep.all()):
                raise RuntimeError("active game has no legal action or bad status")
            if active.numel() == 0:
                break
            steps += 1

            player = players.to(torch.int64)
            learner = ~vs_pool.index_select(0, active) | (player == learner_seat.index_select(0, active))
            actions = torch.empty(active.numel(), dtype=torch.int64, device=device)

            rows = torch.nonzero(learner).squeeze(1)
            env_rows, seat = active.index_select(0, rows), player.index_select(0, rows)
            f, l = features.index_select(0, rows), legal.index_select(0, rows)
            hb = hist.batch(env_rows, seat)
            a, logp = _act(actor, f, l, hb, False)
            actions.index_copy_(0, rows, a)
            hands = ext.hands_indexed(states, env_rows)
            kept = f if storage.columns is None else f.index_select(1, storage.columns)
            storage.append(features=kept, legal=l, actions=a, logp=logp,
                           opp_hand=hands.gather(1, (1 - seat).unsqueeze(1)).squeeze(1),
                           env_ids=env_rows, seat=seat, count=hb.count)

            if opponent_model is not None:
                rows = torch.nonzero(~learner).squeeze(1)
                if rows.numel():
                    env_rows = active.index_select(0, rows)
                    a, _ = _act(opponent_model, features.index_select(0, rows), legal.index_select(0, rows),
                                hist.batch(env_rows, player.index_select(0, rows)), pool_greedy)
                    actions.index_copy_(0, rows, a)

            reward0, new_done, nagari, step_status, committed, events = ext.step_events_indexed(states, active, actions)
            if not bool((step_status == int(ext.STATUS_OK)).all()):
                raise RuntimeError("step returned a non-OK status")
            hist.record(active, committed, events)
            final_reward0.index_copy_(0, active, torch.where(new_done, reward0, 0.0))
            final_nagari.index_copy_(0, active, new_done & nagari)
            active = active.index_select(0, torch.nonzero(~new_done).squeeze(1))
            if active.numel() == 0:
                break
        else:
            raise RuntimeError(f"games did not finish within {max_steps} steps")
    hist.check()

    n = storage.size
    env_ids, seat = storage.env_ids[:n], storage.seat[:n]
    reward = final_reward0.index_select(0, env_ids)
    pool_games = started & vs_pool
    learner_score = torch.where(learner_seat == 0, final_reward0, -final_reward0)[pool_games]
    return SeqRollout(
        hist=hist, returns=torch.where(seat == 0, reward, -reward) / reward_scale, games=games,
        terminal=int((started & ~final_nagari).sum()), nagari=int((started & final_nagari).sum()),
        decision_steps=steps, mean_abs_reward0=float(final_reward0[started].abs().mean()),
        learner_score=float(learner_score.mean()) if learner_score.numel() else float("nan"),
    )


def rollout_from_waves(storage: SeqStorage, log: GameLog, out: dict, games: int,
                       reward_scale: float) -> SeqRollout:
    n = storage.size
    final_reward0, started = out["final_reward0"], out["started"]
    reward = final_reward0.index_select(0, storage.env_ids[:n])
    pool_games = started & out["vs_pool"]
    learner_score = torch.where(out["learner_seat"] == 0, final_reward0, -final_reward0)[pool_games]
    return SeqRollout(
        hist=log, returns=torch.where(storage.seat[:n] == 0, reward, -reward) / reward_scale, games=games,
        terminal=int((started & ~out["final_nagari"]).sum()), nagari=int((started & out["final_nagari"]).sum()),
        decision_steps=out["steps"], mean_abs_reward0=float(final_reward0[started].abs().mean()),
        learner_score=float(learner_score.mean()) if learner_score.numel() else float("nan"),
    )


class GameRows:
    """Decision rows grouped by (game, seat), in decision order: [S, Q] row ids, -1 = padding."""

    def __init__(self, storage: SeqStorage, hist: History | GameLog) -> None:
        n = storage.size
        key = storage.env_ids[:n] * 2 + storage.seat[:n]
        order = torch.sort(key, stable=True).indices  # storage is step-major, so decisions stay in order
        keys, inverse, counts = torch.unique_consecutive(key[order], return_inverse=True, return_counts=True)
        first = torch.cumsum(counts, 0) - counts
        within = torch.arange(n, device=key.device) - first[inverse]
        self.rows = torch.full((keys.numel(), int(counts.max())), -1, dtype=torch.int64, device=key.device)
        self.rows[inverse, within] = order
        self.games, self.seat, self.decisions = keys // 2, keys % 2, counts
        self.storage, self.hist = storage, hist

    def __len__(self) -> int:
        return self.rows.shape[0]

    def batches(self, decisions: int, shuffle: bool, bucket: int = 0) -> list[tuple[torch.Tensor, int, int]]:
        """Minibatches of whole sequences with their (events, decisions) padding sizes.

        All sizes come back in one device-to-host copy. bucket > 0 sorts each window
        of `bucket` minibatches by game length before splitting, which cuts padding.
        """
        size = max(1, decisions * len(self) // max(1, self.storage.size))
        dev = self.rows.device
        order = torch.randperm(len(self), device=dev) if shuffle else torch.arange(len(self), device=dev)
        events = self.hist.count.index_select(0, self.games)
        if bucket > 0:
            window = size * bucket
            parts = []
            for chunk in order.split(window):
                parts.append(chunk.index_select(0, torch.sort(events.index_select(0, chunk), stable=True).indices))
            order = torch.cat(parts)
        batches = list(order.split(size))
        if shuffle and bucket > 0:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        which = torch.cat([torch.full_like(b, i) for i, b in enumerate(batches)])
        flat = torch.cat(batches)
        limits = torch.zeros((len(batches), 2), dtype=torch.int64, device=dev)
        sizes = torch.stack((events.index_select(0, flat), self.decisions.index_select(0, flat)), 1)
        limits.scatter_reduce_(0, which.unsqueeze(1).expand(-1, 2), sizes, "amax")
        return [(b, max(1, e), max(1, q)) for b, (e, q) in zip(batches, limits.tolist())]

    def take(self, seqs: torch.Tensor, q: int) -> tuple[HistBatch, torch.Tensor, torch.Tensor]:
        """History of the chosen sequences, their [B, q] row ids (padding -> 0) and validity."""
        games = self.games.index_select(0, seqs)
        rows = self.rows.index_select(0, seqs)[:, :q]
        valid = rows >= 0
        h = self.hist
        return (HistBatch(h.events.index_select(0, games), h.count.index_select(0, games),
                          h.start.index_select(0, games), self.seat.index_select(0, seqs)),
                rows.clamp_min(0), valid)

    def advantages(self, values: torch.Tensor, returns: torch.Tensor, lam: float) -> torch.Tensor:
        """GAE(lambda), gamma=1, as a backward scan over each sequence's decision columns.

        Same result as cugo_ppo50.gae_advantages, without host round trips.
        """
        n = values.numel()
        valid = self.rows >= 0
        at = torch.where(valid, self.rows, n)
        v = torch.cat((values, values.new_zeros(1))).index_select(0, at.flatten()).view_as(at)
        r = torch.cat((returns, returns.new_zeros(1))).index_select(0, at.flatten()).view_as(at)
        has_next = torch.zeros_like(valid)
        has_next[:, :-1] = valid[:, 1:]
        next_v = torch.zeros_like(v)
        next_v[:, :-1] = v[:, 1:]
        delta = torch.where(has_next, next_v, r) - v
        adv = torch.zeros_like(v)
        running = torch.zeros_like(v[:, 0])
        for q in range(v.shape[1] - 1, -1, -1):
            running = delta[:, q] + lam * has_next[:, q] * running
            adv[:, q] = running
        out = values.new_zeros(n + 1)
        out.index_copy_(0, at.flatten(), adv.flatten())
        return out[:n]


def run_games(model: nn.Module, g: GameRows, hb: HistBatch, rows: torch.Tensor, oracle: bool,
              length: int) -> torch.Tensor:
    s = g.storage
    return model.forward_games(hb, s.features[rows], s.legal[rows], s.count[rows],
                               s.opp_hand[rows] if oracle else None, length=length)


def ppo_minibatch(actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer, params: list,
                  s: SeqStorage, hb: HistBatch, rows: torch.Tensor, valid: torch.Tensor, length: int,
                  adv_all: torch.Tensor, ret_all: torch.Tensor, gostop: torch.Tensor,
                  args: argparse.Namespace, entropy_coef: torch.Tensor, sums: torch.Tensor,
                  kl_out: torch.Tensor) -> None:
    """One PPO step on [B, q] decision rows (valid marks real ones); accumulates stats into sums."""
    w = valid.float()
    total = w.sum().clamp_min(1.0)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        logits = actor.forward_games(hb, s.features[rows], s.legal[rows], s.count[rows], length=length).float()
        value = critic.forward_games(hb, s.features[rows], s.legal[rows], s.count[rows], s.opp_hand[rows],
                                     length=length).float()
    legal, adv, ret = s.legal[rows], adv_all[rows], ret_all[rows]
    log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e30), dim=-1)
    log_ratio = (log_probs.gather(-1, s.actions[rows].unsqueeze(-1)).squeeze(-1) - s.logp[rows]) * w
    ratio = log_ratio.exp()
    policy_loss = -(torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv) * w).sum() / total
    entropy_rows = -(log_probs.exp() * log_probs).sum(dim=-1)
    entropy = (entropy_rows * w).sum() / total
    value_loss = (F.smooth_l1_loss(value, ret, reduction="none") * w).sum() / total
    loss = policy_loss - entropy_coef * entropy + args.value_coef * value_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(params, args.grad_clip, foreach=True)
    optimizer.step()
    with torch.no_grad():
        kl = (((ratio - 1) - log_ratio) * w).sum() / total
        kl_out.copy_(kl)
        gs = gostop[rows].float() * w
        sums += torch.stack((policy_loss.detach(), value_loss.detach(), entropy.detach(), kl,
                             ((((ratio - 1).abs() > args.clip).float()) * w).sum() / total,
                             torch.ones_like(kl), (entropy_rows.detach() * gs).sum(), gs.sum()))


class UpdateGraphs:
    """The PPO minibatch step captured as CUDA graphs and replayed.

    A step launches a few hundred small kernels; eagerly the CPU needs longer to
    launch them than the GPU needs to run them. Minibatches differ in their event
    length and decision count, so both are rounded up to a multiple of QUANTUM and
    each padded shape gets its own graph (padding is masked exactly like the padding
    inside a minibatch). All graphs share one memory pool: they never run at the same
    time and keep nothing between replays except the static buffers below, which
    each iteration's GameRows, advantages and returns are copied into.
    """

    QUANTUM = 4

    def __init__(self, actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer,
                 storage: SeqStorage, log: GameLog, args: argparse.Namespace, max_decisions: int = 64) -> None:
        self.actor, self.critic, self.optimizer, self.storage, self.log, self.args = (
            actor, critic, optimizer, storage, log, args)
        self.params = [*actor.parameters(), *critic.parameters()]
        dev = storage.device
        self.dummy = 2 * log.count.shape[0]  # one past the last (game, seat) sequence: all padding
        self.max_q = max_decisions
        self.rows = torch.full((self.dummy + 1, max_decisions), -1, dtype=torch.int64, device=dev)
        self.games = torch.zeros(self.dummy + 1, dtype=torch.int64, device=dev)
        self.seat = torch.zeros(self.dummy + 1, dtype=torch.int64, device=dev)
        self.batch = torch.zeros(0, dtype=torch.int64, device=dev)
        self.entropy_coef = torch.zeros((), device=dev)
        self.sums = torch.zeros(8, device=dev)
        self.kl = torch.zeros((), device=dev)
        self.graphs: dict[tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.pool = None
        self._bind()

    def _bind(self) -> None:
        s = self.storage
        self.bound = tuple(getattr(s, name).data_ptr() for name in s.NAMES)
        self.adv = torch.zeros(s.capacity, device=s.device)
        self.ret = torch.zeros(s.capacity, device=s.device)
        self.gostop = torch.zeros(s.capacity, dtype=torch.bool, device=s.device)
        self.graphs, self.pool = {}, None

    def load(self, g: GameRows, adv: torch.Tensor, ret: torch.Tensor, gostop: torch.Tensor,
             plan: list, entropy_coef: float) -> None:
        s = self.storage
        if tuple(getattr(s, name).data_ptr() for name in s.NAMES) != self.bound:
            self._bind()  # storage grew: every graph read the old buffers
        seqs, width = g.rows.shape
        if seqs > self.dummy or width > self.max_q:
            raise RuntimeError(f"{seqs} sequences x {width} decisions exceed the update graphs' buffers")
        n = adv.numel()
        self.rows.fill_(-1)
        self.rows[:seqs, :width] = g.rows
        self.games[:seqs] = g.games
        self.seat[:seqs] = g.seat
        self.adv[:n] = adv
        self.ret[:n] = ret
        self.gostop[:n] = gostop
        self.entropy_coef.fill_(entropy_coef)
        self.sums.zero_()
        largest = max(b.numel() for b, _, _ in plan)
        if largest > self.batch.numel():
            self.batch = torch.zeros(-(-largest // 64) * 64, dtype=torch.int64, device=s.device)
            self.graphs, self.pool = {}, None

    def _step(self, length: int, q: int) -> None:
        seqs = self.batch
        rows = self.rows.index_select(0, seqs)[:, :q]
        games = self.games.index_select(0, seqs)
        h = self.log
        hb = HistBatch(h.events.index_select(0, games), h.count.index_select(0, games),
                       h.start.index_select(0, games), self.seat.index_select(0, seqs))
        ppo_minibatch(self.actor, self.critic, self.optimizer, self.params, self.storage, hb, rows.clamp_min(0),
                      rows >= 0, length, self.adv, self.ret, self.gostop, self.args, self.entropy_coef,
                      self.sums, self.kl)

    def _capture(self, length: int, q: int) -> torch.cuda.CUDAGraph:
        # Warm up eagerly (cuBLAS, compiled kernels for this shape), then undo that
        # update in place so the minibatch is applied once, by the replay. The warm-up
        # runs on the current stream: the allocator caches blocks per stream, and a side
        # stream would hold a second copy of a step's activations.
        weights = [p.detach().clone() for p in self.params]
        moments = {p: {k: v.clone() for k, v in self.optimizer.state[p].items() if torch.is_tensor(v)}
                   for p in self.params if p in self.optimizer.state}
        saved = (self.sums.clone(), self.kl.clone())
        self._step(length, q)
        with torch.no_grad():
            for p, w in zip(self.params, weights):
                p.copy_(w)
            for p in self.params:
                before = moments.get(p)
                for k, v in self.optimizer.state[p].items():
                    if torch.is_tensor(v):
                        v.copy_(before[k]) if before is not None else v.zero_()
            self.sums.copy_(saved[0])
            self.kl.copy_(saved[1])
        self.optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        # The warm-up left autograd's gradient accumulators pointing at the default
        # stream; let the capture redirect them to its own stream.
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.graph(graph, pool=self.pool):
                self._step(length, q)
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        if self.pool is None:
            self.pool = graph.pool()
        self.graphs[(length, q)] = graph
        del weights, moments
        torch.cuda.empty_cache()
        return graph

    def run(self, seqs: torch.Tensor, length: int, q: int) -> None:
        quantum = self.QUANTUM
        length = min(MAX_EVENTS, -(-length // quantum) * quantum)
        q = min(self.max_q, -(-q // quantum) * quantum)
        self.batch.fill_(self.dummy)
        self.batch[:seqs.numel()].copy_(seqs)
        graph = self.graphs.get((length, q))
        if graph is None:
            graph = self._capture(length, q)
        graph.replay()


def update(actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer, storage: SeqStorage,
           data: SeqRollout, args: argparse.Namespace, entropy_coef: float,
           graphs: "UpdateGraphs | None" = None) -> TrainStats:
    n = storage.size
    s = storage
    g = GameRows(storage, data.hist)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        critic.eval()
        values = s.logp.new_zeros(n + 1)
        for seqs, length, q in g.batches(4 * args.minibatch, shuffle=False):
            hb, rows, valid = g.take(seqs, q)
            at = torch.where(valid, rows, n).flatten()
            values.index_copy_(0, at, run_games(critic, g, hb, rows, True, length).float().flatten())
        values = values[:n]
    adv_all = g.advantages(values, data.returns, args.gae_lambda)
    ret_all = adv_all + values
    explained_var = 1.0 - float((data.returns - values).var() / data.returns.var().clamp_min(1e-8))

    params = [*actor.parameters(), *critic.parameters()]
    gostop = s.features[:n, s.gostop_column] > 0.5
    actor.train()
    critic.train()
    plan = g.batches(args.minibatch, shuffle=True, bucket=args.length_bucket)
    if graphs is None:
        sums = torch.zeros(8, device=s.device)
        kl = torch.zeros((), device=s.device)
        coef = torch.full((), entropy_coef, device=s.device)
    else:
        graphs.load(g, adv_all, ret_all, gostop, plan, entropy_coef)
        sums, kl = graphs.sums, graphs.kl
    kl_host = torch.zeros(1, pin_memory=True)
    kl_ready: torch.cuda.Event | None = None
    updates, stop = 0, False
    for _ in range(args.epochs):
        for seqs, length, q in plan:
            if graphs is None:
                hb, rows, valid = g.take(seqs, q)
                ppo_minibatch(actor, critic, optimizer, params, s, hb, rows, valid, length, adv_all, ret_all,
                              gostop, args, coef, sums, kl)
            else:
                graphs.run(seqs, length, q)
            updates += 1
            # Early stop on KL without stalling the GPU: read the value copied one
            # update ago (the next update is already queued while we wait).
            if kl_ready is not None:
                kl_ready.synchronize()
                kl_ready = None
                if float(kl_host[0]) > 1.5 * args.target_kl:
                    stop = True
                    break
            if args.target_kl > 0 and updates % 8 == 0:
                kl_host.copy_(kl.detach().reshape(1), non_blocking=True)
                kl_ready = torch.cuda.Event()
                kl_ready.record()
        if stop:
            break

    t = sums.tolist()
    count = max(1.0, t[5])
    return TrainStats(policy_loss=t[0] / count, value_loss=t[1] / count, entropy=t[2] / count,
                      entropy_gostop=t[6] / t[7] if t[7] else float("nan"), kl=t[3] / count,
                      clipfrac=t[4] / count, explained_var=explained_var, updates=updates,
                      epochs=updates / max(1, len(plan)))


def frozen(model: nn.Module, args: argparse.Namespace, state: dict | None = None) -> nn.Module:
    """Eager, gradient-free copy of an actor (a fresh module, so compiled wrappers are not copied)."""
    copy_ = seq_actor(args.mode, args.d_model, args.layers, args.heads).to(next(model.parameters()).device)
    copy_.load_state_dict(model.state_dict() if state is None else state)
    copy_.eval()
    for p in copy_.parameters():
        p.requires_grad_(False)
    return copy_


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO self-play for event-sequence policies")
    parser.add_argument("--mode", choices=("hybrid", "events"), required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--games", type=int, default=65536)
    parser.add_argument("--storage-per-game", type=int, default=24, help="decision rows reserved per game")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch", type=int, default=16384, help="decisions per update (whole games)")
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.03)
    parser.add_argument("--entropy-final", type=float, default=None)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=80.0)
    parser.add_argument("--pool-fraction", type=float, default=0.2)
    parser.add_argument("--pool-every", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--init", type=Path, default=None, help="resume actor/critic/EMA/optimizer")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-pairs", type=int, default=32768)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.85)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the transformer stacks (dynamic shapes; needs triton)")
    parser.add_argument("--collector", choices=("graph", "eager"), default="graph",
                        help="graph: KV-cached CUDA-graph self-play; eager: reference implementation")
    parser.add_argument("--wave", type=int, default=4096, help="games per CUDA-graph wave")
    parser.add_argument("--update-graphs", action="store_true",
                        help="replay the PPO minibatch step as CUDA graphs (needs --collector graph)")
    parser.add_argument("--length-bucket", type=int, default=0,
                        help="sort each window of N minibatches by game length (0 = off)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ext = load_extension()
    storage = SeqStorage(args.games * args.storage_per_game, device, stored_columns(args.mode))

    actor = seq_actor(args.mode, args.d_model, args.layers, args.heads).to(device)
    critic = seq_critic(args.mode, args.d_model, args.layers, args.heads).to(device)
    ema_actor = frozen(actor, args)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr, eps=1e-5, fused=True,
                                 capturable=args.update_graphs)
    start_iteration = 0
    if args.init is not None:
        payload = load_checkpoint(args.init, device)
        actor.load_state_dict(payload["actor"])
        critic.load_state_dict(payload["critic"])
        ema_actor.load_state_dict(payload["actor_ema"])
        optimizer.load_state_dict(payload["optimizer"])
        for group in optimizer.param_groups:  # the checkpoint's setting would override ours
            group["capturable"] = args.update_graphs
        start_iteration = int(payload["iteration"])
    ema_params, actor_params = list(ema_actor.parameters()), list(actor.parameters())
    snapshots: list[nn.Module] = []
    if args.init is not None and args.pool_every > 0:
        # A snapshot taken at iteration k is the EMA actor saved in that iteration's checkpoint.
        for k in range(args.pool_every, start_iteration + 1, args.pool_every):
            path = args.init.parent / f"iter_{k:06d}.pt"
            if path.exists():
                snapshots.append(frozen(actor, args, load_checkpoint(path, device)["actor_ema"]))
        del snapshots[:-args.pool_size]
    if args.compile:
        # Only the encoder stack (every training forward goes through _encode); event and
        # query counts vary per batch. The self-play graph uses cached_step and stays eager.
        for model in (actor, critic):
            model._encode = torch.compile(model._encode, dynamic=True)
    reference = load_policy_checkpoint(args.reference, "auto", device).model if args.reference else None
    collector = None
    log = None
    if args.collector == "graph":
        factory = lambda: seq_actor(args.mode, args.d_model, args.layers, args.heads)  # noqa: E731
        collector = WaveCollector(ext, actor, make_opponent_slot(actor, factory), args.wave,
                                  round(args.wave * args.pool_fraction), stored_columns(args.mode))
        log = GameLog(args.games, device)
    if args.update_graphs and log is None:
        raise SystemExit("--update-graphs needs --collector graph (its event log stays at one address)")
    graphs = UpdateGraphs(actor, critic, optimizer, storage, log, args) if args.update_graphs else None

    algorithm = "seq-ppo-v1"
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print(f"algorithm={algorithm} mode={args.mode} d_model={args.d_model} layers={args.layers} heads={args.heads} "
          f"actor_params={sum(p.numel() for p in actor.parameters())} "
          f"critic_params={sum(p.numel() for p in critic.parameters())}")
    print(f"iterations={args.iterations} games_per_iter={args.games} lr={args.lr:g} epochs={args.epochs} "
          f"minibatch={args.minibatch} entropy={args.entropy:g} target_kl={args.target_kl:g} "
          f"reward_scale={args.reward_scale:g} gae_lambda={args.gae_lambda:g} pool_fraction={args.pool_fraction:g} "
          f"compile={int(args.compile)} start_iteration={start_iteration} pool={len(snapshots)} "
          f"collector={args.collector} wave={args.wave} length_bucket={args.length_bucket} "
          f"update_graphs={int(args.update_graphs)}")
    if args.reference:  # every invocation, so a resumed log records the reference it evaluates against
        print(f"reference={args.reference.as_posix()} reference_policy=auto")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"algorithm": algorithm, "feature_contract": CONTRACT}

    total_games = 0
    for iteration in range(start_iteration + 1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        final = args.entropy if args.entropy_final is None else args.entropy_final
        entropy_coef = args.entropy + (final - args.entropy) * frac

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        seed_offset = (args.seed * 10_000_000 + iteration) * args.games
        if collector is not None:
            pick = snapshots[int(torch.randint(len(snapshots), ()))].state_dict() if snapshots else None
            out = collector.collect(args.games, seed_offset, pick, storage, log)
            data = rollout_from_waves(storage, log, out, args.games, args.reward_scale)
        else:
            data = collect_eager(ext, actor, snapshots, args.games, seed_offset,
                                 args.pool_fraction if snapshots else 0.0, False, args.reward_scale, storage)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        stats = update(actor, critic, optimizer, storage, data, args, entropy_coef, graphs)
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, actor_params, 1.0 - args.ema_decay)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        total_games += data.games
        decision_steps = data.decision_steps
        del data

        print(f"iter={iteration} games={args.games} decisions={storage.size} decision_steps={decision_steps} "
              f"selfplay_s={t1 - t0:.4f} "
              f"games_per_s={args.games / (t1 - t0):.1f} pool={len(snapshots)} "
              f"peak_alloc_mib={torch.cuda.max_memory_allocated() / 2**20:.0f} "
              f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 2**20:.0f}", flush=True)
        print(f"iter={iteration} train_s={t2 - t1:.4f} ppo_policy={stats.policy_loss:.6f} "
              f"ppo_value={stats.value_loss:.6f} ppo_entropy={stats.entropy:.6f} "
              f"ppo_entropy_gostop={stats.entropy_gostop:.6f} ppo_kl={stats.kl:.6f} "
              f"ppo_clipfrac={stats.clipfrac:.4f} ppo_explained_var={stats.explained_var:.4f} "
              f"ppo_updates={stats.updates} ppo_epochs={stats.epochs:.2f} entropy_coef={entropy_coef:.5f}"
              + ("" if graphs is None else f" update_graphs={len(graphs.graphs)}"), flush=True)

        if args.pool_every > 0 and iteration % args.pool_every == 0:
            # The EMA is the release policy; the raw actor can swing between updates.
            snapshots.append(frozen(ema_actor, args))
            del snapshots[:-args.pool_size]

        if args.checkpoint_every > 0 and (iteration % args.checkpoint_every == 0 or iteration == args.iterations):
            path = args.checkpoint_dir / f"iter_{iteration:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"iteration": iteration, "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_ema": ema_actor.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config}, path)
            print(f"checkpoint={path}", flush=True)

        if reference is not None and args.eval_every > 0 and (iteration % args.eval_every == 0 or iteration == 1):
            # The evaluator replays whole histories next to the resident self-play buffers: hand it the
            # cached blocks and run it in chunks (same deal seeds as one big match).
            torch.cuda.empty_cache()
            r = match(ext, ema_actor, reference, args.eval_pairs, EVAL_CHUNK, EVAL_SEED_OFFSET, False)
            print(f"iter={iteration} eval_ref_score={r['score']:.4f} "
                  f"eval_ref_ci_low={r['lo']:.4f} eval_ref_ci_high={r['hi']:.4f} "
                  f"eval_ref_win_pct={r['win']:.3f}", flush=True)

    print(f"seq training PASS iterations={args.iterations} total_games={total_games}")


if __name__ == "__main__":
    main()
