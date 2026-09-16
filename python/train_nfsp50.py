from __future__ import annotations

import argparse
from pathlib import Path

from cugo_nfsp_runner50 import (
    DEFAULT_ANTICIPATORY,
    DEFAULT_AVERAGE_RESERVOIR_CAPACITY,
    DEFAULT_BR_REPLAY_CAPACITY,
    DEFAULT_SELFPLAY_BATCH,
    run_nfsp,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-resident anticipatory fictitious self-play (NFSP-style) baseline"
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--selfplay-batch", type=int, default=DEFAULT_SELFPLAY_BATCH)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--trajectory-capacity", type=int, default=None)
    parser.add_argument("--br-replay-capacity", type=int, default=DEFAULT_BR_REPLAY_CAPACITY)
    parser.add_argument(
        "--average-reservoir-capacity", type=int, default=DEFAULT_AVERAGE_RESERVOIR_CAPACITY
    )
    parser.add_argument("--train-batch", type=int, default=4096)
    parser.add_argument("--br-updates-per-iter", type=int, default=64)
    parser.add_argument("--average-updates-per-iter", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--anticipatory", type=float, default=DEFAULT_ANTICIPATORY)
    parser.add_argument("--br-temperature", type=float, default=1.0)
    parser.add_argument("--average-temperature", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=32.0)
    parser.add_argument("--br-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--average-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("build/checkpoints/nfsp50")
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    run_nfsp(parser.parse_args())


if __name__ == "__main__":
    main()
