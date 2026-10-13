"""Distill a PPO actor/critic into a larger network so it starts at the teacher's level.

States come from self-play by the teacher and, after a warm-up, increasingly by the
student itself (DAgger-style) so it also learns on the states it visits.
The output is a regular PPO checkpoint usable with train_ppo50.py --init.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cugo_contract50 import CONTRACT, load_checkpoint
from cugo_eval50 import evaluate_paired
from cugo_ppo50 import RolloutStorage, _critic_input, collect
from cugo_ppo_nets50 import OracleCritic, PpoActor
from cugo_torch50_ext import load_extension


def load_pair(path: Path, device: torch.device):
    payload = load_checkpoint(path, device)
    config = payload.get("config", {})
    shape = (int(config.get("hidden", 512)), int(config.get("blocks", 3)), config.get("norm", "layer"))
    actor, critic = PpoActor(*shape).to(device), OracleCritic(*shape).to(device)
    # Distill the release policy (the weight EMA) when the teacher has one.
    actor.load_state_dict(payload.get("actor_ema", payload["actor"]))
    critic.load_state_dict(payload["critic"])
    for model in (actor, critic):
        model.eval().requires_grad_(False)
    return actor, critic, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a PPO checkpoint into a larger network")
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--games", type=int, default=65536)
    parser.add_argument("--minibatch", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--student-play", type=float, default=0.5, help="share of iterations the student plays")
    parser.add_argument("--warmup", type=int, default=20, help="teacher-only iterations first")
    parser.add_argument("--eval-pairs", type=int, default=131072)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--memory-fraction", type=float, default=0.85,
                        help="cap on PyTorch's share of dedicated VRAM")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    ext = load_extension()
    teacher_actor, teacher_critic, teacher_config = load_pair(args.teacher, device)
    reward_scale = float(teacher_config.get("reward_scale", 32.0))
    actor = PpoActor(args.hidden, args.blocks, "none").to(device)
    critic = OracleCritic(args.hidden, args.blocks, "none").to(device)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=args.lr, fused=True)
    storage = RolloutStorage(args.games * 26, device)

    params = sum(p.numel() for p in actor.parameters())
    print(f"algorithm=ppo-distill-v1 iterations={args.iterations} teacher={args.teacher}")
    print(f"network hidden={args.hidden} blocks={args.blocks} norm=none actor_params={params}")

    for iteration in range(1, args.iterations + 1):
        student_plays = iteration > args.warmup and torch.rand(()) < args.student_play
        t0 = time.perf_counter()
        data = collect(ext, actor if student_plays else teacher_actor, [], args.games,
                       seed_offset=(args.seed * 10_000_000 + iteration) * args.games, pool_fraction=0.0,
                       pool_greedy=False, reward_scale=reward_scale, storage=storage)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        actor.train()
        critic.train()
        n = data.actions.numel()
        sums = torch.zeros(4, device=device)
        perm = torch.randperm(n, device=device)
        for i in range(0, n - args.minibatch + 1, args.minibatch):
            idx = perm[i:i + args.minibatch]
            feats, legal = data.features[idx], data.legal[idx]
            critic_in = _critic_input(feats, data.opp_hand[idx])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logp = torch.log_softmax(teacher_actor(feats).float().masked_fill(~legal, -1.0e30), dim=1)
                t_value = teacher_critic(critic_in).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s_logits = actor(feats).float()
                s_value = critic(critic_in).float()
            s_logp = torch.log_softmax(s_logits.masked_fill(~legal, -1.0e30), dim=1)
            kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=1).mean()
            value_loss = F.mse_loss(s_value, t_value)
            loss = kl + value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], 1.0)
            optimizer.step()
            with torch.no_grad():
                agree = (s_logp.argmax(1) == t_logp.argmax(1)).float().mean()
                sums += torch.stack((kl.detach(), value_loss.detach(), agree, torch.ones_like(agree)))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        kl_m, v_m, agree_m, count = sums.tolist()
        print(f"iter={iteration} games={data.games} decisions={n} selfplay_s={t1 - t0:.4f} "
              f"games_per_s={data.games / (t1 - t0):.1f} student_plays={int(student_plays)}")
        print(f"iter={iteration} train_s={t2 - t1:.4f} distill_kl={kl_m / count:.6f} "
              f"distill_value_mse={v_m / count:.6f} distill_top1_agree={agree_m / count:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    config = {"hidden": args.hidden, "blocks": args.blocks, "norm": "none", "reward_scale": reward_scale,
              "algorithm": "ppo-distill-v1", "teacher": str(args.teacher), "feature_contract": CONTRACT}
    # The student starts its PPO run with its EMA equal to the distilled weights.
    torch.save({"iteration": 0, "actor": actor.state_dict(), "actor_ema": actor.state_dict(),
                "critic": critic.state_dict(), "config": config}, args.out)
    print(f"checkpoint={args.out}")
    s = evaluate_paired(ext=ext, model_a=actor.eval(), model_b=teacher_actor, pairs=args.eval_pairs,
                        seed_offset=900000000, max_steps=128, use_amp=True)
    print(f"iter={args.iterations} eval_ref_score={s.mean_pair_score_a:.4f} "
          f"eval_ref_ci_low={s.pair_score_ci95_low:.4f} eval_ref_ci_high={s.pair_score_ci95_high:.4f}")
    print(f"distill training PASS iterations={args.iterations}")


if __name__ == "__main__":
    main()
