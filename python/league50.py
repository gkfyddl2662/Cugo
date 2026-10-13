"""Round-robin paired evaluation of checkpoints; prints a score matrix.

    python python/league50.py ckpt_a.pt ckpt_b.pt ... [--pairs 32768] [--sample]

Large matches run in chunks of --chunk pairs and are pooled, so GPU memory stays
bounded. Each chunk continues the deal sequence of one big match, so the chunk size
does not change which deals are played (an even chunk keeps the seat alternation).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from cugo_eval50 import evaluate_paired, load_policy_checkpoint
from cugo_ppo_nets50 import SEED_BASE, SIGNED_GOLDEN
from cugo_torch50_ext import load_extension


def label(path: Path) -> str:
    return f"{path.parent.name}/{path.stem.replace('iter_', '')}"


def chunk_offset(seed_offset: int, start: int) -> int:
    """Seed offset whose first deal is deal `start` of a match at `seed_offset` (make_seeds wraps in int64)."""
    seed = (SEED_BASE + seed_offset + start * SIGNED_GOLDEN + 2 ** 63) % 2 ** 64 - 2 ** 63
    return seed - SEED_BASE


def match(ext, a, b, pairs: int, chunk: int, seed_offset: int, sample: bool) -> dict:
    if chunk % 2:
        raise ValueError("chunk must be even so every chunk starts with the same seat order")
    n = total = sq = wins = games = 0
    for start in range(0, pairs, chunk):
        size = min(chunk, pairs - start)
        s = evaluate_paired(ext=ext, model_a=a, model_b=b, pairs=size, seed_offset=chunk_offset(seed_offset, start),
                            max_steps=128, use_amp=True, sample_a=sample, sample_b=sample)
        k = s.evaluated_pairs
        n += k
        total += s.mean_pair_score_a * k
        sq += (s.pair_score_std ** 2) * (k - 1) + k * s.mean_pair_score_a ** 2
        wins += s.a_wins
        games += s.evaluated_games
    mean = total / n
    var = (sq - n * mean ** 2) / (n - 1)
    se = math.sqrt(var / n)
    return {"score": mean, "lo": mean - 1.96 * se, "hi": mean + 1.96 * se,
            "win": 100.0 * wins / games, "pairs": n, "games": games}


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-robin paired league")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--pairs", type=int, default=32768, help="deal pairs per match (2 games each)")
    parser.add_argument("--chunk", type=int, default=131072)
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
            r = match(ext, models[i], models[j], args.pairs, args.chunk, args.seed_offset, args.sample)
            score[i][j], score[j][i] = r["score"], -r["score"]
            print(f"MATCH a={names[i]} b={names[j]} score={r['score']:.4f} "
                  f"ci_low={r['lo']:.4f} ci_high={r['hi']:.4f} a_win_pct={r['win']:.3f} "
                  f"pairs={r['pairs']} games={r['games']}", flush=True)

    width = max(len(x) for x in names)
    print("\nrow vs column, points per game (row's perspective)")
    print(" " * width + "".join(f"{x[-8:]:>10}" for x in names) + f"{'mean':>10}")
    for i in range(n):
        mean = sum(score[i]) / max(1, n - 1)
        print(f"{names[i]:<{width}}" + "".join(f"{v:>10.2f}" for v in score[i]) + f"{mean:>10.2f}")


if __name__ == "__main__":
    main()
