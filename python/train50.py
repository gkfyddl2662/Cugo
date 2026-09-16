from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cugo_torch50_ext import load_extension
from cugo_train50 import (
    GpuReplayBuffer,
    PolicyValueNet,
    collect_selfplay,
    save_checkpoint,
    train_from_replay,
)


DEFAULT_SELFPLAY_BATCH = 131072
DEFAULT_REPLAY_CAPACITY = 1 << 22


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU-resident Shin Matgo self-play training baseline"
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--selfplay-batch", type=int, default=DEFAULT_SELFPLAY_BATCH)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=DEFAULT_REPLAY_CAPACITY)
    parser.add_argument("--train-batch", type=int, default=4096)
    parser.add_argument("--updates-per-iter", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=32.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("build/checkpoints/torch50"),
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.selfplay_batch <= 0 or args.train_batch <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.replay_capacity <= 0:
        raise SystemExit("--replay-capacity must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    use_amp = not args.no_amp
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(
        f"device={torch.cuda.get_device_name(0)} "
        f"capability={torch.cuda.get_device_capability(0)} amp={use_amp}"
    )

    ext = load_extension(verbose=args.verbose_build)
    feature_count = int(ext.FEATURE_COUNT)
    action_count = int(ext.ACTION_COUNT)
    if feature_count != 496 or action_count != 177:
        raise RuntimeError("extension constants do not match the frozen Torch50 contract")

    model = PolicyValueNet(feature_count, action_count, args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    replay = GpuReplayBuffer(
        args.replay_capacity,
        feature_count,
        action_count,
        device,
    )

    params = sum(p.numel() for p in model.parameters())
    replay_mib = replay.storage_bytes / (1024.0 * 1024.0)
    print(
        f"network hidden={args.hidden} params={params} "
        f"replay_capacity={args.replay_capacity} replay_storage_mib={replay_mib:.1f}"
    )
    print(
        "objective=monte_carlo_actor_critic "
        f"reward_scale={args.reward_scale:g} temperature={args.temperature:g}"
    )

    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)

    total_games = 0
    total_generated_transitions = 0
    total_retained_transitions = 0
    for iteration in range(1, args.iterations + 1):
        seed_offset = args.seed + (iteration - 1) * args.selfplay_batch

        torch.cuda.synchronize()
        collect_start = time.perf_counter()
        batch = collect_selfplay(
            ext=ext,
            model=model,
            batch=args.selfplay_batch,
            seed_offset=seed_offset,
            max_steps=args.max_steps,
            temperature=args.temperature,
            reward_scale=args.reward_scale,
            use_amp=use_amp,
            max_transitions=replay.capacity,
        )
        replay.add(batch)
        torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_start

        torch.cuda.synchronize()
        train_start = time.perf_counter()
        train = train_from_replay(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            replay=replay,
            batch_size=args.train_batch,
            updates=args.updates_per_iter,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start

        total_games += batch.games
        total_generated_transitions += batch.generated_transitions
        total_retained_transitions += batch.transitions
        games_per_s = batch.games / collect_seconds
        transitions_per_s = batch.generated_transitions / collect_seconds
        retention_pct = 100.0 * batch.transitions / batch.generated_transitions
        replay_fill_pct = 100.0 * replay.size / replay.capacity
        print(
            f"iter={iteration} games={batch.games} "
            f"transitions={batch.generated_transitions} retained={batch.transitions} "
            f"dropped={batch.dropped_transitions} retention_pct={retention_pct:.2f} "
            f"terminal={batch.terminal} nagari={batch.nagari} "
            f"decision_steps={batch.decision_steps} replay={replay.size} "
            f"replay_fill_pct={replay_fill_pct:.2f} "
            f"selfplay_s={collect_seconds:.4f} games_per_s={games_per_s:.1f} "
            f"transitions_per_s={transitions_per_s:.1f} "
            f"mean_abs_reward0={batch.mean_abs_reward0:.4f}"
        )
        print(
            f"iter={iteration} updates={train.updates} train_s={train_seconds:.4f} "
            f"loss={train.loss:.6f} policy={train.policy_loss:.6f} "
            f"value={train.value_loss:.6f} entropy={train.entropy:.6f} "
            f"mean_abs_advantage={train.mean_abs_advantage:.6f}"
        )

        if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint = args.checkpoint_dir / f"iter_{iteration:06d}.pt"
            save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scaler,
                iteration,
                replay.size,
                config,
            )
            print(f"checkpoint={checkpoint}")

    overall_retention_pct = (
        100.0 * total_retained_transitions / total_generated_transitions
    )
    print(
        f"training PASS iterations={args.iterations} total_games={total_games} "
        f"total_transitions={total_generated_transitions} "
        f"retained_transitions={total_retained_transitions} "
        f"retention_pct={overall_retention_pct:.2f} replay={replay.size}"
    )


if __name__ == "__main__":
    main()
