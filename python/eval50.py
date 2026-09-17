from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cugo_eval50 import evaluate_paired, load_policy_checkpoint
from cugo_torch50_ext import load_extension


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired seat-balanced Torch50 policy evaluator")
    parser.add_argument("--a-checkpoint", type=Path, required=True)
    parser.add_argument("--b-checkpoint", type=Path, required=True)
    parser.add_argument("--a-policy", choices=("auto", "average", "br", "legacy"), default="auto")
    parser.add_argument("--b-policy", choices=("auto", "average", "br", "legacy"), default="auto")
    parser.add_argument("--pairs", type=int, default=131072)
    parser.add_argument("--seed-offset", type=int, default=900000000)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    device = torch.device("cuda")
    use_amp = not args.no_amp
    ext = load_extension(verbose=args.verbose_build)
    if int(ext.FEATURE_COUNT) != 496 or int(ext.ACTION_COUNT) != 177:
        raise RuntimeError("extension constants do not match the frozen Torch50 contract")

    a = load_policy_checkpoint(args.a_checkpoint, args.a_policy, device)
    b = load_policy_checkpoint(args.b_checkpoint, args.b_policy, device)
    print(
        f"A checkpoint={a.checkpoint} policy={a.kind} iteration={a.iteration} "
        f"hidden={a.hidden}"
    )
    print(
        f"B checkpoint={b.checkpoint} policy={b.kind} iteration={b.iteration} "
        f"hidden={b.hidden}"
    )
    print(
        f"eval paired_seat_swap=True deterministic_argmax=True pairs={args.pairs} "
        f"games={args.pairs * 2} seed_offset={args.seed_offset} amp={use_amp}"
    )

    s = evaluate_paired(
        ext=ext,
        model_a=a.model,
        model_b=b.model,
        pairs=args.pairs,
        seed_offset=args.seed_offset,
        max_steps=args.max_steps,
        use_amp=use_amp,
    )
    print(
        f"RESULT games={s.games} evaluated_games={s.evaluated_games} "
        f"predecision_excluded={s.predecision_excluded} "
        f"A_wins={s.a_wins} B_wins={s.b_wins} draws={s.draws} "
        f"A_win_pct={100.0 * s.a_wins / s.evaluated_games:.4f} "
        f"B_win_pct={100.0 * s.b_wins / s.evaluated_games:.4f} "
        f"draw_pct={100.0 * s.draws / s.evaluated_games:.4f}"
    )
    print(
        f"SEATS A_as_p0={s.a_p0_wins}/{s.a_p0_losses}/{s.a_p0_draws} "
        f"A_as_p1={s.a_p1_wins}/{s.a_p1_losses}/{s.a_p1_draws} "
        f"mean_score_p0={s.mean_score_a_p0:.6f} mean_score_p1={s.mean_score_a_p1:.6f}"
    )
    print(
        f"PAIRS evaluated={s.evaluated_pairs}/{s.pairs} "
        f"A_win/loss/draw={s.pair_wins}/{s.pair_losses}/{s.pair_draws} "
        f"mean_score_A={s.mean_pair_score_a:.6f} std={s.pair_score_std:.6f} "
        f"se={s.pair_score_se:.6f} ci95=[{s.pair_score_ci95_low:.6f},{s.pair_score_ci95_high:.6f}]"
    )
    print(
        f"ENGINE terminal={s.terminal} nagari={s.nagari} decision_steps={s.decision_steps} "
        f"seconds={s.seconds:.6f} games_per_s={s.games_per_s:.1f}"
    )
    print("evaluation PASS")


if __name__ == "__main__":
    main()
