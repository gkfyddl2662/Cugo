"""GPU-resident self-play for the event-sequence policies.

Games run in fixed-size waves. Every tensor a decision step touches has a fixed
shape (finished games stay in the batch and are masked), so one step - observe,
query the policy, sample, step the engine, append the event to the KV caches -
is captured once as a CUDA graph and replayed. The host only replays graphs,
checks every few steps whether the wave has finished, and compacts the recorded
decisions once per wave.

The policy reads its KV caches (cugo_seq50.SeqNet.cached_step): one new token per
event for each seat, one query token per decision, instead of re-encoding the
whole history every step.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from cugo_contract50 import stop_points
from cugo_ppo_nets50 import ACTIONS, make_seeds
from cugo_seq50 import EVENT_WORDS, MAX_EVENTS, START_WORDS, KVCache, SeqNet


class GameLog:
    """Event logs of every game in an iteration (the fields GameRows/HistBatch read)."""

    def __init__(self, games: int, device: torch.device) -> None:
        self.events = torch.zeros((games, MAX_EVENTS, EVENT_WORDS), dtype=torch.int64, device=device)
        self.count = torch.zeros(games, dtype=torch.int64, device=device)
        self.start = torch.zeros((games, START_WORDS), dtype=torch.int64, device=device)


class WaveCollector:
    """Self-play of `wave` games at a time with one CUDA graph per decision step.

    Pool games are the first `pool_games` of every wave: there the learner plays one
    seat and a frozen opponent (weights copied into a fixed slot) the other.
    """

    def __init__(self, ext: Any, actor: SeqNet, opponent: SeqNet, wave: int, pool_games: int,
                 columns: list[int] | None, max_steps: int = 64) -> None:
        # Steps count every decision, floor choices included (games so far took at most
        # 43); only committed actions become events, and those are checked against
        # MAX_EVENTS (the cache slots) after each wave.
        device = next(actor.parameters()).device
        self.ext, self.actor, self.opponent = ext, actor, opponent
        self.w, self.p, self.max_steps = wave, pool_games, max_steps
        w = wave
        self.ids = torch.arange(w, device=device)
        self.seats2 = torch.cat((torch.zeros(w, dtype=torch.int64, device=device),
                                 torch.ones(w, dtype=torch.int64, device=device)))
        seeds, first = make_seeds(w, 0, device)
        self.states = ext.create(seeds, first)
        self.events = torch.zeros((w, MAX_EVENTS, EVENT_WORDS), dtype=torch.int64, device=device)
        self.count = torch.zeros(w, dtype=torch.int64, device=device)
        self.start = torch.zeros((w, START_WORDS), dtype=torch.int64, device=device)
        self.learner_seat = torch.zeros(w, dtype=torch.int64, device=device)
        self.cache = KVCache(actor, 2 * w, device)  # row = seat * w + game
        self.opp_cache = KVCache(opponent, max(1, pool_games), device)  # row = pool game, opponent's seat
        self.columns = None if columns is None else torch.tensor(columns, device=device)
        width = actor.query_in[0].in_features - ACTIONS
        s = max_steps
        self.rec_features = torch.zeros((s, w, width), dtype=torch.bfloat16, device=device)
        self.rec_legal = torch.zeros((s, w, ACTIONS), dtype=torch.bool, device=device)
        self.rec_logp = torch.zeros((s, w), dtype=torch.float32, device=device)
        self.rec_long = {name: torch.zeros((s, w), dtype=torch.int64, device=device)
                         for name in ("actions", "opp_hand", "stock", "stop_points", "seat", "count")}
        self.rec_valid = torch.zeros((s, w), dtype=torch.bool, device=device)
        self.step_index = torch.zeros(1, dtype=torch.int64, device=device)
        self.final_reward0 = torch.zeros(w, device=device)
        self.final_nagari = torch.zeros(w, dtype=torch.bool, device=device)
        self.finished = torch.zeros(w, dtype=torch.bool, device=device)
        self.started = torch.zeros(w, dtype=torch.bool, device=device)
        self.bad = torch.zeros((), dtype=torch.int64, device=device)
        self.ok = int(ext.STATUS_OK)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.graph_pool: bool | None = None

    # -- one wave ---------------------------------------------------------------

    def _reset(self, seeds: torch.Tensor, first: torch.Tensor, learner_seat: torch.Tensor, pool: bool) -> None:
        w, p = self.w, self.p
        self.states.copy_(self.ext.create(seeds, first))
        self.start.copy_(self.ext.start(self.states))
        self.events.zero_()
        self.count.zero_()
        self.learner_seat.copy_(learner_seat)
        done = self.ext.observe_indexed(self.states, self.ids)[3]
        self.started.copy_(~done)
        self.finished.copy_(done)
        self.final_reward0.zero_()
        self.final_nagari.zero_()
        self.bad.zero_()
        self.step_index.zero_()
        self.rec_valid.zero_()
        self.cache.length.zero_()
        self.actor.cached_step(self.cache, self.actor.start_token(self.start.repeat(2, 1), self.seats2), 1)
        self.cache.length.fill_(1)
        if pool:
            self.opp_cache.length.zero_()
            opp_seat = 1 - self.learner_seat[:p]
            self.opponent.cached_step(self.opp_cache, self.opponent.start_token(self.start[:p], opp_seat), 1)
            self.opp_cache.length.fill_(1)

    def _step(self, pool: bool) -> None:
        w, p, ids = self.w, self.p, self.ids
        actor, opponent = self.actor, self.opponent
        features, legal, players, done, status = self.ext.observe_indexed(self.states, ids)
        active = ~done
        self.bad += (active & ((status != self.ok) | ~legal.any(1))).sum()
        player = players.long()

        # One query per game, on the mover's cache row (row = seat * w + game).
        hidden = actor.cached_step(self.cache, actor.query_token(features, legal), rows=(player & 1) * w + ids)
        logits = actor.readout(hidden).float()
        learner_move = torch.ones_like(active)
        if pool:
            opp_logits = opponent.readout(opponent.cached_step(
                self.opp_cache, opponent.query_token(features[:p], legal[:p]))).float()
            opp_move = player[:p] != self.learner_seat[:p]
            logits = torch.cat((torch.where(opp_move.unsqueeze(1), opp_logits, logits[:p]), logits[p:]))
            learner_move = torch.cat((~opp_move, learner_move[p:]))
        masked = logits.masked_fill(~legal, -1.0e30)
        # Gumbel-max sampling: same distribution as multinomial, and graph-capturable.
        gumbel = -torch.log(-torch.log(torch.rand_like(masked).clamp_(1e-20, 1.0)))
        actions = (masked + gumbel).argmax(1)
        logp = torch.log_softmax(masked, 1).gather(1, actions.unsqueeze(1)).squeeze(1)

        hands = self.ext.hands_indexed(self.states, ids)
        stock = self.ext.stock_indexed(self.states, ids)
        kept = features if self.columns is None else features.index_select(1, self.columns)
        at = self.step_index
        self.rec_features.index_copy_(0, at, kept.to(torch.bfloat16).unsqueeze(0))
        self.rec_legal.index_copy_(0, at, legal.unsqueeze(0))
        self.rec_logp.index_copy_(0, at, logp.unsqueeze(0))
        for name, value in (("actions", actions), ("seat", player), ("count", self.count),
                            ("opp_hand", hands.gather(1, (1 - player).unsqueeze(1)).squeeze(1)),
                            ("stock", stock), ("stop_points", stop_points(features))):
            self.rec_long[name].index_copy_(0, at, value.unsqueeze(0))
        self.rec_valid.index_copy_(0, at, (active & learner_move).unsqueeze(0))

        reward0, new_done, nagari, step_status, committed, events = self.ext.step_events_indexed(
            self.states, ids, actions)
        self.bad += (active & (step_status != self.ok)).sum()
        finished_now = active & new_done
        self.final_reward0.copy_(torch.where(finished_now, reward0, self.final_reward0))
        self.final_nagari |= finished_now & nagari
        self.finished |= finished_now

        commit = active & committed
        slot = self.count.clamp_max(MAX_EVENTS - 1)
        self.events.index_put_((ids, slot), torch.where(commit.unsqueeze(1), events, self.events[ids, slot]))
        actor.cached_step(self.cache, actor.event_token(events.repeat(2, 1), self.seats2, self.count.repeat(2)))
        self.cache.length += commit.repeat(2).long()
        if pool:
            opponent.cached_step(self.opp_cache, opponent.event_token(events[:p], 1 - self.learner_seat[:p],
                                                                      self.count[:p]))
            self.opp_cache.length += commit[:p].long()
        self.count += commit.long()
        self.step_index += 1

    def _capture(self, pool: bool) -> None:
        if self.graph is not None and self.graph_pool == pool:
            return
        self.graph = None
        seeds, first = make_seeds(self.w, 0, self.states.device)
        self._reset(seeds, first, torch.zeros_like(self.learner_seat), pool)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):  # warm-up (cuBLAS handles, Triton JIT, autocast) outside the capture
                self._step(pool)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step(pool)
        self.graph, self.graph_pool = graph, pool

    # -- one iteration ----------------------------------------------------------

    def collect(self, games: int, seed_offset: int, opponent_state: dict | None, storage: Any,
                log: GameLog) -> dict:
        """Play `games` games into storage/log; returns per-game outcome tensors."""
        if games % self.w:
            raise ValueError(f"games ({games}) must be a multiple of the wave size ({self.w})")
        device = self.states.device
        pool = opponent_state is not None and self.p > 0
        if pool:
            self.opponent.load_state_dict(opponent_state)  # in place: the graph keeps its pointers
        seeds, first = make_seeds(games, seed_offset, device)
        learner_seat = torch.randint(2, (games,), device=device)
        vs_pool = torch.zeros(games, dtype=torch.bool, device=device)
        final_reward0 = torch.zeros(games, device=device)
        final_nagari = torch.zeros(games, dtype=torch.bool, device=device)
        started = torch.zeros(games, dtype=torch.bool, device=device)
        storage.size = 0
        steps = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            self._capture(pool)
            for offset in range(0, games, self.w):
                part = slice(offset, offset + self.w)
                self._reset(seeds[part], first[part], learner_seat[part], pool)
                for k in range(self.max_steps):
                    self.graph.replay()
                    if (k + 1) % 4 == 0 and k + 1 >= 16 and bool(self.finished.all()):
                        steps = max(steps, k + 1)
                        break
                else:
                    if not bool(self.finished.all()):
                        raise RuntimeError(f"games did not finish within {self.max_steps} steps")
                    steps = self.max_steps
                if int(self.bad):
                    raise RuntimeError(f"{int(self.bad)} bad observe/step results in a wave")
                if int(self.count.max()) > MAX_EVENTS:
                    raise RuntimeError("event log overflow")

                at = torch.nonzero(self.rec_valid)  # step-major: decisions stay in game order
                s_idx, g_idx = at[:, 0], at[:, 1]
                storage.append(features=self.rec_features[s_idx, g_idx], legal=self.rec_legal[s_idx, g_idx],
                               actions=self.rec_long["actions"][s_idx, g_idx], logp=self.rec_logp[s_idx, g_idx],
                               opp_hand=self.rec_long["opp_hand"][s_idx, g_idx],
                               stock=self.rec_long["stock"][s_idx, g_idx],
                               stop_points=self.rec_long["stop_points"][s_idx, g_idx], env_ids=offset + g_idx,
                               seat=self.rec_long["seat"][s_idx, g_idx],
                               count=self.rec_long["count"][s_idx, g_idx])
                log.events[part] = self.events
                log.count[part] = self.count
                log.start[part] = self.start
                final_reward0[part] = self.final_reward0
                final_nagari[part] = self.final_nagari
                started[part] = self.started
                if pool:
                    vs_pool[offset:offset + self.p] = True
        return {"final_reward0": final_reward0, "final_nagari": final_nagari, "started": started,
                "learner_seat": learner_seat, "vs_pool": vs_pool, "steps": steps}


def make_opponent_slot(actor: nn.Module, factory) -> SeqNet:
    """A frozen module with the actor's shape whose weights each iteration's pool pick is copied into."""
    slot = factory().to(next(actor.parameters()).device).eval()
    for p in slot.parameters():
        p.requires_grad_(False)
    return slot
