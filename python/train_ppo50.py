from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch

from cugo_contract50 import CONTRACT, fit_optimizer, fit_state, load_checkpoint, optimizer_names
from cugo_eval50 import load_policy_checkpoint
from cugo_ppo50 import PpoUpdater, RolloutStorage, collect
from cugo_ppo_nets50 import BeliefHead, OracleCritic, PpoActor
from cugo_torch50_ext import load_extension
from league50 import match

EVAL_SEED_OFFSET = 900000000
EVAL_CHUNK = 8192


def frozen(model: torch.nn.Module) -> torch.nn.Module:
    model = copy.deepcopy(model).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ppo(path: Path, device: torch.device) -> dict:
    return load_checkpoint(path, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regularized PPO self-play for Shin Matgo 50")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--games", type=int, default=65536, help="games per iteration")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--norm", choices=("layer", "none"), default="none")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=32768)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.05, help="entropy coef at start")
    parser.add_argument("--entropy-final", type=float, default=None, help="linear decay target")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.99,
                        help="per-iteration EMA of actor weights used for eval and release")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="1.0 = Monte-Carlo advantages")
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=32.0)
    parser.add_argument("--pool-fraction", type=float, default=0.2)
    parser.add_argument("--pool-every", type=int, default=50)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--pool-greedy", action="store_true")
    parser.add_argument("--opponent", type=Path, action="append", default=[],
                        help="frozen opponent checkpoint(s) added to the pool")
    parser.add_argument("--opponent-policy", default="auto")
    parser.add_argument("--exploiter", action="store_true",
                        help="train only against --opponent: every game, no self snapshots")
    parser.add_argument("--init", type=Path, default=None,
                        help="initialize actor/critic weights from a PPO checkpoint (a fresh run)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="continue a run from its checkpoint: weights, EMA, optimizer, iteration, "
                             "and the snapshot pool from the checkpoints beside it")
    parser.add_argument("--reference", type=Path, default=None, help="checkpoint for periodic paired eval")
    parser.add_argument("--reference-policy", default="auto")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-pairs", type=int, default=32768)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("build/checkpoints/ppo50"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-graph", action="store_true", help="disable CUDA-graph training step")
    parser.add_argument("--memory-fraction", type=float, default=0.80,
                        help="cap on PyTorch's share of dedicated VRAM (Windows pages GPU memory beyond ~12 GB of 16)")
    parser.add_argument("--value-bins", type=int, default=0,
                        help="distributional critic over N symlog-spaced bins (0 = smooth L1 regression)")
    parser.add_argument("--value-top", type=float, default=640.0, help="largest |scaled return| the bins reach")
    parser.add_argument("--critic-stock", action="store_true", help="the critic also sees the next 8 stock draws")
    parser.add_argument("--belief-coef", type=float, default=0.0,
                        help="weight of the actor's opponent-hand prediction loss (0 = no belief head)")
    parser.add_argument("--belief-width", type=int, default=512, help="hidden width of the belief head")
    parser.add_argument("--all-actions", action="store_true",
                        help="branch every legal action one step and learn from the critic's Q of each")
    parser.add_argument("--mc-coef", type=float, default=1.0,
                        help="all-action mode: weight of the sampled action's (return - Q) term (1 = unbiased)")
    parser.add_argument("--q-source", choices=("critic", "worlds"), default="critic",
                        help="all-action mode: Q from the oracle critic, or greedy play to the end in redealt worlds")
    parser.add_argument("--worlds", type=int, default=4, help="--q-source worlds: redealt worlds per decision")
    parser.add_argument("--policy-target", choices=("pg", "improve"), default="pg",
                        help="all-action mode: unbiased all-action gradient, or MPO-style pi_old*exp(A/tau) target")
    parser.add_argument("--improve-kl", type=float, default=0.02,
                        help="'improve': tau is set each iteration so KL(target || pi_old) averages this")
    parser.add_argument("--branch-min-prob", type=float, default=None,
                        help="all-action mode: branch actions with at least this probability "
                             "(default 0.01 for pg, 0 for improve)")
    parser.add_argument("--branch-max-top", type=float, default=None,
                        help="all-action mode: skip decisions whose top action has at least this probability "
                             "(default 0.99 for pg, never for improve)")
    parser.add_argument("--critic-warmup", type=int, default=0,
                        help="first N iterations of this invocation train only the critic and belief head")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    # Fail fast instead of spilling into shared system memory (Windows sysmem fallback).
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ext = load_extension()
    improve = args.policy_target == "improve"
    if args.branch_min_prob is None:
        args.branch_min_prob = 0.0 if improve else 0.01
    if args.branch_max_top is None:
        args.branch_max_top = 2.0 if improve else 0.99
    if args.all_actions and args.exploiter:
        raise SystemExit("--all-actions needs self-play: the critic only knows the learner's side in --exploiter")
    storage = RolloutStorage(args.games * 26, device, all_actions=args.all_actions)
    updater: PpoUpdater | None = None

    actor = PpoActor(args.hidden, args.blocks, args.norm).to(device)
    critic = OracleCritic(args.hidden, args.blocks, args.norm, stock=args.critic_stock,
                          value_bins=args.value_bins, value_top=args.value_top).to(device)
    belief = BeliefHead(args.hidden, args.belief_width).to(device) if args.belief_coef > 0 else None
    if args.init is not None and args.resume is not None:
        raise SystemExit("--init starts a fresh run from weights; --resume continues one. Pick one.")
    start_iteration = 0
    source = args.resume or args.init
    payload = load_ppo(source, device) if source is not None else None
    restarted: list[str] = []
    if payload is not None:
        actor.load_state_dict(payload["actor"])
        # A critic that gained inputs or bins keeps its trunk (see cugo_contract50.fit_state).
        critic_state, restarted = fit_state(payload["critic"], critic)
        critic.load_state_dict(critic_state)
        if belief is not None and "belief" in payload:
            belief.load_state_dict(payload["belief"])
    # Exponential moving average of actor weights: the release/eval policy. Greedy play
    # flips on near-tied actions, so single updates occasionally crashed raw checkpoints
    # (e.g. -12.5 pts at one eval); averaging ~1/(1-decay) iterations smooths that out.
    ema_actor = frozen(actor)
    if payload is not None and "actor_ema" in payload:
        ema_actor.load_state_dict(payload["actor_ema"])
    ema_params, actor_params = list(ema_actor.parameters()), list(actor.parameters())
    optimizer = torch.optim.Adam(
        [{"params": list(actor.parameters())},
         {"params": [*critic.parameters(), *(belief.parameters() if belief is not None else [])]}],
        lr=args.lr, eps=1e-5, fused=True, capturable=True)

    fixed_pool = [load_policy_checkpoint(p, args.opponent_policy, device).model for p in args.opponent]
    snapshots: list[torch.nn.Module] = []
    if args.resume is not None:
        optimizer.load_state_dict(fit_optimizer(payload, actor, critic, belief))
        start_iteration = int(payload["iteration"])
        if not args.exploiter and args.pool_every > 0:
            # A snapshot taken at iteration k is the EMA actor saved in that iteration's checkpoint.
            for k in range(args.pool_every, start_iteration + 1, args.pool_every):
                path = args.resume.parent / f"iter_{k:06d}.pt"
                if path.exists():
                    snapshot = PpoActor(args.hidden, args.blocks, args.norm).to(device)
                    snapshot.load_state_dict(load_ppo(path, device)["actor_ema"])
                    snapshots.append(frozen(snapshot).cpu())
            del snapshots[:-args.pool_size]
    # Learning rates live on the GPU: the captured update step reads them on every replay
    # (a float would be baked into the graph), so the warm-up switch needs no recapture.
    for group in optimizer.param_groups:
        group["capturable"] = True
        group["lr"] = torch.tensor(args.lr, device=device)
    warm_until = start_iteration + args.critic_warmup
    if args.critic_warmup > 0:
        optimizer.param_groups[0]["lr"].fill_(0.0)  # hold the actor while the critic catches up
    if args.exploiter and not fixed_pool:
        raise SystemExit("--exploiter needs at least one --opponent")
    pool_fraction = 1.0 if args.exploiter else args.pool_fraction
    reference = (load_policy_checkpoint(args.reference, args.reference_policy, device).model
                 if args.reference is not None else None)

    params = sum(p.numel() for p in actor.parameters())
    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    print(f"algorithm=ppo-oracle-critic-v1 iterations={args.iterations} games_per_iter={args.games} "
          f"exploiter={int(args.exploiter)} pool_fraction={pool_fraction:g} pool_size={args.pool_size}")
    print(f"network hidden={args.hidden} blocks={args.blocks} norm={args.norm} actor_params={params} "
          f"critic_params={sum(p.numel() for p in critic.parameters())}")
    print(f"lr={args.lr:g} epochs={args.epochs} minibatch={args.minibatch} clip={args.clip:g} "
          f"entropy={args.entropy:g} entropy_final={args.entropy_final if args.entropy_final is not None else args.entropy:g} "
          f"target_kl={args.target_kl:g} reward_scale={args.reward_scale:g} gae_lambda={args.gae_lambda:g}")
    print(f"value_bins={args.value_bins} value_top={args.value_top:g} critic_stock={int(args.critic_stock)} "
          f"belief_coef={args.belief_coef:g} critic_warmup={args.critic_warmup} "
          f"critic_restarted={','.join(restarted) or 'none'} memory_fraction={args.memory_fraction:g} "
          f"all_actions={int(args.all_actions)} q_source={args.q_source} worlds={args.worlds} "
          f"mc_coef={args.mc_coef:g} policy_target={args.policy_target} "
          f"improve_kl={args.improve_kl:g} branch_min_prob={args.branch_min_prob:g} "
          f"branch_max_top={args.branch_max_top:g}")
    if args.reference is not None:  # every invocation, so a resumed log records the reference it evaluates against
        print(f"reference={args.reference.as_posix()} reference_policy={args.reference_policy}")
    config = vars(args) | {"algorithm": "ppo-oracle-critic-v1", "feature_contract": CONTRACT}
    config = {k: (str(v) if isinstance(v, Path) or (isinstance(v, list) and v and isinstance(v[0], Path)) else v)
              for k, v in config.items()}

    total_games = 0
    for iteration in range(start_iteration + 1, args.iterations + 1):
        frac = (iteration - 1) / max(1, args.iterations - 1)
        final = args.entropy if args.entropy_final is None else args.entropy_final
        entropy_coef = args.entropy + (final - args.entropy) * frac
        pool = fixed_pool + snapshots
        # One opponent per iteration. Snapshots wait in host memory (16 x 140 MB at 2048x4
        # would not fit beside the all-action buffers); the chosen one visits the GPU.
        opponent = pool[int(torch.randint(len(pool), ()))].to(device) if pool and pool_fraction > 0 else None

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        data = collect(ext, actor, [opponent] if opponent is not None else [], args.games,
                       seed_offset=(args.seed * 10_000_000 + iteration) * args.games,
                       pool_fraction=pool_fraction if opponent is not None else 0.0, pool_greedy=args.pool_greedy,
                       reward_scale=args.reward_scale, storage=storage,
                       critic=critic if args.all_actions else None,
                       branch_min_prob=args.branch_min_prob, branch_max_top=args.branch_max_top,
                       q_source=args.q_source, worlds=args.worlds)
        if opponent is not None and not any(opponent is m for m in fixed_pool):
            opponent.cpu()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if args.critic_warmup > 0 and iteration == warm_until + 1:
            optimizer.param_groups[0]["lr"].fill_(args.lr)
        if updater is None:
            updater = PpoUpdater(actor, critic, optimizer, storage, args.minibatch, args.clip,
                                 args.value_coef, args.grad_clip, use_graph=not args.no_graph,
                                 belief=belief, belief_coef=args.belief_coef, mc_coef=args.mc_coef,
                                 policy_target=args.policy_target, improve_kl=args.improve_kl)
        stats = updater.update(data, args.epochs, entropy_coef, args.target_kl, args.gae_lambda)
        with torch.no_grad():
            torch._foreach_lerp_(ema_params, actor_params, 1.0 - args.ema_decay)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        total_games += data.games

        selfplay_s, train_s = t1 - t0, t2 - t1
        decisions = data.actions.numel()
        print(f"iter={iteration} games={data.games} decisions={decisions} terminal={data.terminal} "
              f"nagari={data.nagari} decision_steps={data.decision_steps} selfplay_s={selfplay_s:.4f} "
              f"games_per_s={data.games / selfplay_s:.1f} mean_abs_reward0={data.mean_abs_reward0:.4f} "
              f"pool={len(pool)} peak_alloc_mib={torch.cuda.max_memory_allocated() / 2**20:.0f} "
              f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 2**20:.0f}"
              + ("" if math.isnan(data.learner_score) else
                                     f" pool_learner_score={data.learner_score:.4f}"))
        print(f"iter={iteration} train_s={train_s:.4f} ppo_policy={stats.policy_loss:.6f} "
              f"ppo_value={stats.value_loss:.6f} ppo_entropy={stats.entropy:.6f} "
              f"ppo_entropy_gostop={stats.entropy_gostop:.6f} ppo_kl={stats.kl:.6f} "
              f"ppo_clipfrac={stats.clipfrac:.4f} ppo_explained_var={stats.explained_var:.4f} "
              f"ppo_updates={stats.updates} ppo_epochs={stats.epochs:.2f} entropy_coef={entropy_coef:.5f}"
              + (f" belief_loss={stats.belief_loss:.5f}" if belief is not None else "")
              + (f" q_explained_var={stats.q_explained_var:.4f}" if args.all_actions else "")
              + (f" improve_inv_temp={stats.inv_temp:.4g}" if args.all_actions and improve else "")
              + (" critic_warmup=1" if iteration <= warm_until else ""))

        if not args.exploiter and args.pool_every > 0 and iteration % args.pool_every == 0:
            # The EMA is the release policy; the raw actor can swing between updates.
            snapshots.append(frozen(ema_actor).cpu())
            del snapshots[:-args.pool_size]

        # Checkpoint first: a failed evaluation must not lose the iteration.
        if args.checkpoint_every > 0 and (iteration % args.checkpoint_every == 0 or iteration == args.iterations):
            path = args.checkpoint_dir / f"iter_{iteration:06d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"iteration": iteration, "actor": actor.state_dict(), "critic": critic.state_dict(),
                        "actor_ema": ema_actor.state_dict(), "optimizer": optimizer.state_dict(),
                        "optimizer_names": optimizer_names(actor, critic, belief), "config": config}
                       | ({"belief": belief.state_dict()} if belief is not None else {}), path)
            print(f"checkpoint={path}")

        if reference is not None and args.eval_every > 0 and (iteration % args.eval_every == 0 or iteration == 1):
            for tag, model in (("ref", ema_actor), ("raw", actor.eval())):
                r = match(ext, model, reference, args.eval_pairs, EVAL_CHUNK, EVAL_SEED_OFFSET, False)
                print(f"iter={iteration} eval_{tag}_score={r['score']:.4f} "
                      f"eval_{tag}_ci_low={r['lo']:.4f} eval_{tag}_ci_high={r['hi']:.4f} "
                      f"eval_{tag}_win_pct={r['win']:.3f}")

    print(f"ppo training PASS iterations={args.iterations} total_games={total_games}")


if __name__ == "__main__":
    main()
