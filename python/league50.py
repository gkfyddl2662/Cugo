"""Round-robin paired evaluation of checkpoints; prints a score matrix.

    python python/league50.py ckpt_a.pt ckpt_b.pt ... [--pairs 32768] [--sample]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cugo_eval50 import evaluate_paired, load_policy_checkpoint
from cugo_torch50_ext import load_extension


def label(path: Path) -> str:
    return f"{path.parent.name}/{path.stem.replace('iter_', '')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin paired league")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--pairs", type=int, default=32768)
    parser.add_argument("--sample", action="store_true", help="both sides sample instead of argmax")
    parser.add_argument("--seed-offset", type=int, default=910000000)
    args = parser.parse_args()

    device = torch.device("cuda")
    ext = load_extension()
    models = [load_policy_checkpoint(p, "auto", device).model for p in args.checkpoints]
    names = [label(p) for p in args.checkpoints]
    n = len(models)
    score = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            s = evaluate_paired(ext=ext, model_a=models[i], model_b=models[j], pairs=args.pairs,
                                seed_offset=args.seed_offset, max_steps=128, use_amp=True,
                                sample_a=args.sample, sample_b=args.sample)
            score[i][j], score[j][i] = s.mean_pair_score_a, -s.mean_pair_score_a
            print(f"MATCH a={names[i]} b={names[j]} score={s.mean_pair_score_a:.4f} "
                  f"ci_low={s.pair_score_ci95_low:.4f} ci_high={s.pair_score_ci95_high:.4f} "
                  f"a_win_pct={100.0 * s.a_wins / max(1, s.evaluated_games):.3f}", flush=True)

    width = max(len(x) for x in names)
    print("\nrow vs column, points per game (row's perspective)")
    print(" " * width + "".join(f"{x[-8:]:>10}" for x in names) + f"{'mean':>10}")
    for i in range(n):
        mean = sum(score[i]) / max(1, n - 1)
        print(f"{names[i]:<{width}}" + "".join(f"{v:>10.2f}" for v in score[i]) + f"{mean:>10.2f}")


if __name__ == "__main__":
    main()
