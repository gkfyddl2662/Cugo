"""PPO self-play for the event-sequence policies (cugo_seq50.SeqNet).

Same objective as train_ppo50.py (clipped PPO, entropy bonus, oracle critic,
GAE, actor EMA, snapshot pool); only the inputs differ. Each iteration keeps the
event log of every game; a decision row stores how many events it saw, so the
update can rebuild exactly the sequence the policy acted on.
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

from cugo_eval50 import evaluate_paired, load_policy_checkpoint
from cugo_ppo50 import DECISION_OFFSET, GOSTOP_KIND, TrainStats, gae_advantages
from cugo_ppo_nets50 import ACTIONS, FEATURES, make_seeds
from cugo_seq50 import History, HistBatch, policy_logits, seq_actor, seq_critic, stored_columns
from cugo_torch50_ext import load_extension

EVAL_SEED_OFFSET = 900000000


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
    hist: History
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


def collect(ext: Any, actor: nn.Module, pool: list[nn.Module], games: int, seed_offset: int,
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


class GameRows:
    """Decision rows grouped by (game, seat), in decision order: [S, Q] row ids, -1 = padding."""

    def __init__(self, storage: SeqStorage, hist: History) -> None:
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

    def batches(self, decisions: int, shuffle: bool) -> list[torch.Tensor]:
        size = max(1, decisions * len(self) // max(1, self.storage.size))
        order = torch.randperm(len(self), device=self.rows.device) if shuffle else torch.arange(
            len(self), device=self.rows.device)
        return list(order.split(size))

    def take(self, seqs: torch.Tensor) -> tuple[HistBatch, torch.Tensor, torch.Tensor]:
        """History of the chosen sequences, their [B, Q] row ids (padding -> 0) and validity."""
        games = self.games.index_select(0, seqs)
        rows = self.rows.index_select(0, seqs)
        q = max(1, int(self.decisions.index_select(0, seqs).max()))
        rows = rows[:, :q]
        valid = rows >= 0
        h = self.hist
        return (HistBatch(h.events.index_select(0, games), h.count.index_select(0, games),
                          h.start.index_select(0, games), self.seat.index_select(0, seqs)),
                rows.clamp_min(0), valid)


def run_games(model: nn.Module, g: GameRows, hb: HistBatch, rows: torch.Tensor, oracle: bool) -> torch.Tensor:
    s = g.storage
    return model.forward_games(hb, s.features[rows], s.legal[rows], s.count[rows],
                               s.opp_hand[rows] if oracle else None)


def update(actor: nn.Module, critic: nn.Module, optimizer: torch.optim.Optimizer, storage: SeqStorage,
           data: SeqRollout, args: argparse.Namespace, entropy_coef: float) -> TrainStats:
    n = storage.size
    s = storage
    g = GameRows(storage, data.hist)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        critic.eval()
        values = torch.empty(n, device=s.device)
        for seqs in g.batches(4 * args.minibatch, shuffle=False):
            hb, rows, valid = g.take(seqs)
            values[rows[valid]] = run_games(critic, g, hb, rows, True)[valid].float()
    adv_all = gae_advantages(values, data.returns, s.env_ids[:n], s.seat[:n], args.gae_lambda)
    ret_all = adv_all + values
    explained_var = 1.0 - float((data.returns - values).var() / data.returns.var().clamp_min(1e-8))

    params = [*actor.parameters(), *critic.parameters()]
    actor.train()
    critic.train()
    sums = torch.zeros(8, device=s.device)
    updates, stop = 0, False
    for _ in range(args.epochs):
        for seqs in g.batches(args.minibatch, shuffle=True):
            hb, rows, valid = g.take(seqs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = run_games(actor, g, hb, rows, False)[valid].float()
                value = run_games(critic, g, hb, rows, True)[valid].float()
            idx = rows[valid]
            legal, adv, ret = s.legal[idx], adv_all[idx], ret_all[idx]
            log_probs = torch.log_softmax(logits.masked_fill(~legal, -1.0e30), dim=1)
            log_ratio = log_probs.gather(1, s.actions[idx].unsqueeze(1)).squeeze(1) - s.logp[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - args.clip, 1 + args.clip) * adv).mean()
            entropy_rows = -(log_probs.exp() * log_probs).sum(dim=1)
            value_loss = F.smooth_l1_loss(value, ret)
            loss = policy_loss - entropy_coef * entropy_rows.mean() + args.value_coef * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip, foreach=True)
            optimizer.step()
            updates += 1
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean()
                gs = (s.features[idx, s.gostop_column] > 0.5).float()
                sums += torch.stack((policy_loss.detach(), value_loss.detach(), entropy_rows.mean().detach(), kl,
                                     ((ratio - 1).abs() > args.clip).float().mean(), torch.ones_like(kl),
                                     (entropy_rows.detach() * gs).sum(), gs.sum()))
            if args.target_kl > 0 and updates % 8 == 0 and float(kl) > 1.5 * args.target_kl:
                stop = True
                break
        if stop:
            break

    t = sums.tolist()
    count = max(1.0, t[5])
    return TrainStats(policy_loss=t[0] / count, value_loss=t[1] / count, entropy=t[2] / count,
                      entropy_gostop=t[6] / t[7] if t[7] else float("nan"), kl=t[3] / count,
                      clipfrac=t[4] / count, explained_var=explained_var, updates=updates,
                      epochs=updates / max(1, len(g.batches(args.minibatch, shuffle=False))))


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
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr, eps=1e-5, fused=True)
    start_iteration = 0
    if args.init is not None:
        payload = torch.load(args.init, map_location=device, weights_only=False)
        actor.load_state_dict(payload["actor"])
        critic.load_state_dict(payload["critic"])
        ema_actor.load_state_dict(payload["actor_ema"])
        optimizer.load_state_dict(payload["optimizer"])
        start_iteration = int(payload["iteration"])
    ema_params, actor_params = list(ema_actor.parameters()), list(actor.parameters())
    snapshots: list[nn.Module] = []
    if args.init is not None and args.pool_every > 0:
        # A snapshot taken at iteration k is the actor saved in that iteration's checkpoint.
        for k in range(args.pool_every, start_iteration + 1, args.pool_every):
            path = args.init.parent / f"iter_{k:06d}.pt"
            if path.exists():
                snapshots.append(frozen(actor, args, torch.load(path, map_location=device,
                                                                 weights_only=False)["actor"]))
        del snapshots[:-args.pool_size]
    if args.compile:
        # Only the encoders: event/query counts vary per batch and are read on the host.
        for model in (actor, critic):
            model.encoder.compile(dynamic=True)
    reference = load_policy_checkpoint(args.reference, "auto", device).model if args.reference else None

    algorithm = "seq-ppo-v1"
    print(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")
    print(f"algorithm={algorithm} mode={args.mode} d_model={args.d_model} layers={args.layers} heads={args.heads} "
          f"actor_params={sum(p.numel() for p in actor.parameters())} "
          f"critic_params={sum(p.numel() for p in critic.parameters())}")
    print(f"iterations={args.iterations} games_per_iter={args.games} lr={args.lr:g} epochs={args.epochs} "
          f"minibatch={args.minibatch} entropy={args.entropy:g} target_kl={args.target_kl:g} "
          f"reward_scale={args.reward_scale:g} gae_lambda={args.gae_lambda:g} pool_fraction={args.pool_fraction:g} "
          f"compile={int(args.compile)} start_iteration={start_iteration} pool={len(snapshots)}")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()} | {"algorithm": algorithm}

    total_games = 0
    for iteration in range(start_iteration + 1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        final = args.entropy if args.entropy_final is None else args.entropy_final
        entropy_coef = args.entropy + (final - args.entropy) * frac

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data = collect(ext, actor, snapshots, args.games, (args.seed * 10_000_000 + iteration) * args.games,
                       args.pool_fraction if snapshots else 0.0, False, args.reward_scale, storage)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        stats = update(actor, critic, optimizer, storage, data, args, entropy_coef)
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, actor_params, 1.0 - args.ema_decay)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        total_games += data.games
        del data

        print(f"iter={iteration} games={args.games} decisions={storage.size} selfplay_s={t1 - t0:.4f} "
              f"games_per_s={args.games / (t1 - t0):.1f} pool={len(snapshots)} "
              f"peak_alloc_mib={torch.cuda.max_memory_allocated() / 2**20:.0f} "
              f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 2**20:.0f}", flush=True)
        print(f"iter={iteration} train_s={t2 - t1:.4f} ppo_policy={stats.policy_loss:.6f} "
              f"ppo_value={stats.value_loss:.6f} ppo_entropy={stats.entropy:.6f} "
              f"ppo_entropy_gostop={stats.entropy_gostop:.6f} ppo_kl={stats.kl:.6f} "
              f"ppo_clipfrac={stats.clipfrac:.4f} ppo_explained_var={stats.explained_var:.4f} "
              f"ppo_updates={stats.updates} ppo_epochs={stats.epochs:.2f} entropy_coef={entropy_coef:.5f}",
              flush=True)

        if args.pool_every > 0 and iteration % args.pool_every == 0:
            snapshots.append(frozen(actor, args))
            del snapshots[:-args.pool_size]

        if reference is not None and args.eval_every > 0 and (iteration % args.eval_every == 0 or iteration == 1):
            s = evaluate_paired(ext=ext, model_a=ema_actor, model_b=reference, pairs=args.eval_pairs,
                                seed_offset=EVAL_SEED_OFFSET, max_steps=128, use_amp=True)
            print(f"iter={iteration} eval_ref_score={s.mean_pair_score_a:.4f} "
                  f"eval_ref_ci_low={s.pair_score_ci95_low:.4f} eval_ref_ci_high={s.pair_score_ci95_high:.4f} "
                  f"eval_ref_win_pct={100.0 * s.a_wins / max(1, s.evaluated_games):.3f}", flush=True)

        if args.checkpoint_every > 0 and (iteration % args.checkpoint_every == 0 or iteration == args.iterations):
            path = args.checkpoint_dir / f"iter_{iteration:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"iteration": iteration, "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_ema": ema_actor.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": config}, path)
            print(f"checkpoint={path}", flush=True)

    print(f"seq training PASS iterations={args.iterations} total_games={total_games}")


if __name__ == "__main__":
    main()
