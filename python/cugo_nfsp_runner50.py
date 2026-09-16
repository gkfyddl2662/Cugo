from __future__ import annotations

import time

import torch

from cugo_nfsp50 import AveragePolicyNet, collect_nfsp_selfplay, initialize_average_from_br
from cugo_nfsp_buffers50 import nfsp_trajectory_storage_bytes
from cugo_nfsp_train50 import save_nfsp_checkpoint, train_average_from_reservoir
from cugo_replay50 import PackedGpuReplayBuffer, PackedGpuReservoirBuffer
from cugo_torch50_ext import load_extension
from cugo_train50 import PolicyValueNet, train_from_replay

DEFAULT_SELFPLAY_BATCH = 131072
DEFAULT_TRAJECTORY_CAPACITY = 1 << 20
DEFAULT_BR_REPLAY_CAPACITY = 1 << 22
DEFAULT_AVERAGE_RESERVOIR_CAPACITY = 1 << 24
DEFAULT_ANTICIPATORY = 0.10


def run_nfsp(args) -> None:
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.selfplay_batch <= 0 or args.train_batch <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.br_replay_capacity <= 0 or args.average_reservoir_capacity <= 0:
        raise SystemExit("replay/reservoir capacities must be positive")
    if not (0.0 < args.anticipatory <= 1.0):
        raise SystemExit("--anticipatory must be in (0, 1]")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    if args.trajectory_capacity is None:
        trajectory_capacity = min(
            DEFAULT_TRAJECTORY_CAPACITY,
            args.br_replay_capacity,
            args.average_reservoir_capacity,
        )
    else:
        trajectory_capacity = args.trajectory_capacity
    if trajectory_capacity <= 0:
        raise SystemExit("--trajectory-capacity must be positive")

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

    br_model = PolicyValueNet(feature_count, action_count, args.hidden).to(device)
    average_model = AveragePolicyNet(feature_count, action_count, args.hidden).to(device)
    initialize_average_from_br(average_model, br_model)

    br_optimizer = torch.optim.AdamW(
        br_model.parameters(),
        lr=args.br_learning_rate,
        weight_decay=args.weight_decay,
    )
    average_optimizer = torch.optim.AdamW(
        average_model.parameters(),
        lr=args.average_learning_rate,
        weight_decay=args.weight_decay,
    )
    br_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    average_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    br_replay = PackedGpuReplayBuffer(
        args.br_replay_capacity,
        feature_count,
        action_count,
        device,
    )
    average_reservoir = PackedGpuReservoirBuffer(
        args.average_reservoir_capacity,
        feature_count,
        action_count,
        device,
    )

    br_params = sum(p.numel() for p in br_model.parameters())
    average_params = sum(p.numel() for p in average_model.parameters())
    trajectory_mib = nfsp_trajectory_storage_bytes(
        trajectory_capacity, feature_count, action_count
    ) / (1024.0 * 1024.0)
    br_replay_mib = br_replay.storage_bytes / (1024.0 * 1024.0)
    average_reservoir_mib = average_reservoir.storage_bytes / (1024.0 * 1024.0)
    print(
        f"algorithm=nfsp-actor-critic-v1 anticipatory={args.anticipatory:g} "
        f"br_temperature={args.br_temperature:g} "
        f"average_temperature={args.average_temperature:g}"
    )
    print(
        f"network hidden={args.hidden} br_params={br_params} "
        f"average_params={average_params}"
    )
    print(
        f"trajectory_capacity={trajectory_capacity} "
        f"trajectory_storage_mib={trajectory_mib:.1f} "
        f"br_replay_capacity={br_replay.capacity} "
        f"br_replay_storage_mib={br_replay_mib:.1f} "
        f"average_reservoir_capacity={average_reservoir.capacity} "
        f"average_reservoir_storage_mib={average_reservoir_mib:.1f} "
        f"packed_bytes_per_transition={br_replay.bytes_per_transition}"
    )
    print(
        "br_objective=monte_carlo_actor_critic "
        "average_objective=supervised_historical_br_policy "
        f"reward_scale={args.reward_scale:g}"
    )

    config = vars(args).copy()
    config["checkpoint_dir"] = str(args.checkpoint_dir)
    config["trajectory_capacity_resolved"] = trajectory_capacity
    config["algorithm"] = "nfsp-actor-critic-v1"
    config["replay_format"] = br_replay.format_name
    config["reservoir_format"] = average_reservoir.format_name

    total_games = 0
    total_decisions = 0
    total_br_generated = 0
    total_br_retained = 0

    for iteration in range(1, args.iterations + 1):
        seed_offset = args.seed + (iteration - 1) * args.selfplay_batch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        collect_start = time.perf_counter()
        batch = collect_nfsp_selfplay(
            ext=ext,
            br_model=br_model,
            average_model=average_model,
            batch=args.selfplay_batch,
            seed_offset=seed_offset,
            max_steps=args.max_steps,
            anticipatory=args.anticipatory,
            br_temperature=args.br_temperature,
            average_temperature=args.average_temperature,
            reward_scale=args.reward_scale,
            use_amp=use_amp,
            max_br_transitions=trajectory_capacity,
        )
        torch.cuda.synchronize()
        collect_seconds = time.perf_counter() - collect_start

        add_start = time.perf_counter()
        br_replay.add(batch)
        average_reservoir.add(batch)
        torch.cuda.synchronize()
        add_seconds = time.perf_counter() - add_start
        peak_alloc_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)

        br_train_start = time.perf_counter()
        br_train = train_from_replay(
            model=br_model,
            optimizer=br_optimizer,
            scaler=br_scaler,
            replay=br_replay,
            batch_size=args.train_batch,
            updates=args.br_updates_per_iter,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )
        torch.cuda.synchronize()
        br_train_seconds = time.perf_counter() - br_train_start

        average_train_start = time.perf_counter()
        average_train = train_average_from_reservoir(
            model=average_model,
            optimizer=average_optimizer,
            scaler=average_scaler,
            reservoir=average_reservoir,
            batch_size=args.train_batch,
            updates=args.average_updates_per_iter,
            grad_clip=args.grad_clip,
            use_amp=use_amp,
        )
        torch.cuda.synchronize()
        average_train_seconds = time.perf_counter() - average_train_start

        total_games += batch.games
        total_decisions += batch.total_decisions
        total_br_generated += batch.generated_br_transitions
        total_br_retained += batch.transitions

        games_per_s = batch.games / collect_seconds
        decisions_per_s = batch.total_decisions / collect_seconds
        br_retention_pct = (
            100.0 * batch.transitions / batch.generated_br_transitions
            if batch.generated_br_transitions
            else 100.0
        )
        br_decision_pct = (
            100.0 * batch.generated_br_transitions / batch.total_decisions
            if batch.total_decisions
            else 0.0
        )
        br_mode_pct = 100.0 * batch.br_player_modes / (2.0 * batch.games)
        br_fill_pct = 100.0 * br_replay.size / br_replay.capacity
        average_fill_pct = 100.0 * average_reservoir.size / average_reservoir.capacity

        print(
            f"iter={iteration} games={batch.games} decisions={batch.total_decisions} "
            f"br_generated={batch.generated_br_transitions} "
            f"br_retained={batch.transitions} br_dropped={batch.dropped_transitions} "
            f"br_retention_pct={br_retention_pct:.2f} "
            f"br_decision_pct={br_decision_pct:.2f} br_mode_pct={br_mode_pct:.2f} "
            f"terminal={batch.terminal} nagari={batch.nagari} "
            f"decision_steps={batch.decision_steps} selfplay_s={collect_seconds:.4f} "
            f"games_per_s={games_per_s:.1f} decisions_per_s={decisions_per_s:.1f} "
            f"buffer_add_s={add_seconds:.4f} peak_alloc_mib={peak_alloc_mib:.1f} "
            f"peak_reserved_mib={peak_reserved_mib:.1f} "
            f"mean_abs_reward0={batch.mean_abs_reward0:.4f}"
        )
        print(
            f"iter={iteration} br_replay={br_replay.size} "
            f"br_replay_fill_pct={br_fill_pct:.2f} "
            f"average_reservoir={average_reservoir.size} "
            f"average_reservoir_fill_pct={average_fill_pct:.2f} "
            f"average_seen={average_reservoir.total_seen} "
            f"reservoir_candidates={average_reservoir.last_candidates} "
            f"reservoir_replacements={average_reservoir.last_replacements} "
            f"reservoir_collisions={average_reservoir.last_collisions}"
        )
        print(
            f"iter={iteration} br_updates={br_train.updates} "
            f"br_train_s={br_train_seconds:.4f} br_loss={br_train.loss:.6f} "
            f"br_policy={br_train.policy_loss:.6f} br_value={br_train.value_loss:.6f} "
            f"br_entropy={br_train.entropy:.6f} "
            f"br_mean_abs_advantage={br_train.mean_abs_advantage:.6f}"
        )
        print(
            f"iter={iteration} average_updates={average_train.updates} "
            f"average_train_s={average_train_seconds:.4f} "
            f"average_loss={average_train.loss:.6f} "
            f"average_entropy={average_train.entropy:.6f} "
            f"average_top1={average_train.top1_accuracy:.6f}"
        )

        if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            checkpoint = args.checkpoint_dir / f"iter_{iteration:06d}.pt"
            save_nfsp_checkpoint(
                checkpoint,
                br_model,
                average_model,
                br_optimizer,
                average_optimizer,
                br_scaler,
                average_scaler,
                iteration,
                br_replay.size,
                average_reservoir.size,
                average_reservoir.total_seen,
                config,
            )
            print(f"checkpoint={checkpoint}")

        del batch

    overall_retention_pct = (
        100.0 * total_br_retained / total_br_generated
        if total_br_generated
        else 100.0
    )
    overall_br_pct = (
        100.0 * total_br_generated / total_decisions if total_decisions else 0.0
    )
    print(
        f"nfsp training PASS iterations={args.iterations} total_games={total_games} "
        f"total_decisions={total_decisions} total_br_generated={total_br_generated} "
        f"total_br_retained={total_br_retained} "
        f"br_retention_pct={overall_retention_pct:.2f} "
        f"br_decision_pct={overall_br_pct:.2f} br_replay={br_replay.size} "
        f"average_reservoir={average_reservoir.size} "
        f"average_seen={average_reservoir.total_seen}"
    )

