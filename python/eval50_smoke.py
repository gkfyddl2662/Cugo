from __future__ import annotations

import torch

from cugo_eval50 import evaluate_paired
from cugo_nfsp50 import AveragePolicyNet, initialize_average_from_br
from cugo_torch50_ext import load_extension
from cugo_train50 import PolicyValueNet


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    device = torch.device("cuda")
    ext = load_extension(verbose=False)

    br = PolicyValueNet(496, 177, 256).to(device).eval()
    average = AveragePolicyNet(496, 177, 256).to(device).eval()
    initialize_average_from_br(average, br)

    same = evaluate_paired(
        ext=ext,
        model_a=average,
        model_b=average,
        pairs=32768,
        seed_offset=700001111,
        max_steps=128,
        use_amp=True,
    )
    if same.a_wins != same.b_wins:
        raise RuntimeError(f"identical-policy W/L mismatch {same.a_wins} != {same.b_wins}")
    if same.mean_pair_score_a != 0.0:
        raise RuntimeError(f"identical-policy paired score was {same.mean_pair_score_a}")
    if same.pair_wins != 0 or same.pair_losses != 0:
        raise RuntimeError("identical-policy paired outcomes did not cancel exactly")
    print(
        "IDENTICAL POLICY EXACT: PASS "
        f"games={same.games} evaluated={same.evaluated_games} excluded={same.predecision_excluded} "
        f"wins={same.a_wins} losses={same.b_wins} draws={same.draws} "
        f"paired_score={same.mean_pair_score_a:.6f} games_per_s={same.games_per_s:.1f}"
    )

    # BR and initialized average have identical policy/trunk weights, so their
    # greedy policies must also be exactly equivalent before either is trained.
    cross = evaluate_paired(
        ext=ext,
        model_a=br,
        model_b=average,
        pairs=32768,
        seed_offset=700002222,
        max_steps=128,
        use_amp=True,
    )
    if cross.a_wins != cross.b_wins or cross.mean_pair_score_a != 0.0:
        raise RuntimeError("initialized BR vs average did not cancel exactly")
    print(
        "BR/AVERAGE INIT EXACT: PASS "
        f"games={cross.games} wins={cross.a_wins} losses={cross.b_wins} "
        f"draws={cross.draws} paired_score={cross.mean_pair_score_a:.6f}"
    )
    print("EVAL50 CORE VALIDATION: PASS")


if __name__ == "__main__":
    main()
