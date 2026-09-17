from __future__ import annotations

from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, "python")

import cugo_torch50_ext as env_loader
import cugo_train50 as baseline
from cugo_nfsp50 import AveragePolicyNet, collect_nfsp_selfplay, initialize_average_from_br
from cugo_nfsp_train50 import AverageTrainSummary
from cugo_replay50 import PackedGpuReservoirBuffer


def exact_tensor(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise RuntimeError(f"{name}: shape {tuple(a.shape)} != {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise RuntimeError(f"{name}: dtype {a.dtype} != {b.dtype}")
    if torch.equal(a, b):
        return
    if a.dtype.is_floating_point:
        diff = float((a.float() - b.float()).abs().max().item())
        raise RuntimeError(f"{name}: max_abs_diff={diff}")
    count = int((a != b).sum().item())
    raise RuntimeError(f"{name}: mismatch_count={count}")


def compare_eta1(old: baseline.SelfPlayBatch, new, temperature: float) -> None:
    for name in ("features", "legal", "actions", "value_targets"):
        exact_tensor(
            f"eta1.temp_{temperature:g}.{name}",
            getattr(old, name),
            getattr(new, name),
        )

    scalar_pairs = (
        ("games", old.games, new.games),
        ("total_decisions", old.generated_transitions, new.total_decisions),
        (
            "generated_br_transitions",
            old.generated_transitions,
            new.generated_br_transitions,
        ),
        ("terminal", old.terminal, new.terminal),
        ("nagari", old.nagari, new.nagari),
        ("decision_steps", old.decision_steps, new.decision_steps),
        ("mean_abs_reward0", old.mean_abs_reward0, new.mean_abs_reward0),
    )
    for name, a, b in scalar_pairs:
        if a != b:
            raise RuntimeError(f"eta1.temp_{temperature:g}.{name}: {a} != {b}")

    expected_modes = 2 * new.games
    if new.br_player_modes != expected_modes:
        raise RuntimeError(
            f"eta1 BR mode count {new.br_player_modes} != {expected_modes}"
        )

    print(
        "ETA1 COLLECTOR EXACT: PASS "
        f"temperature={temperature:g} games={new.games} "
        f"decisions={new.total_decisions} transitions={new.transitions} "
        f"steps={new.decision_steps} terminal={new.terminal} nagari={new.nagari}"
    )


def make_synthetic(start: int, count: int, device: torch.device):
    ids = torch.arange(start, start + count, dtype=torch.int64, device=device)
    features = torch.zeros((count, 496), dtype=torch.float16, device=device)
    features[:, 453] = ids.remainder(1024).to(torch.float16)
    legal = torch.zeros((count, 177), dtype=torch.bool, device=device)
    legal[:, 0] = True
    actions = ids.remainder(177)
    targets = ids.to(torch.float32)
    return SimpleNamespace(
        transitions=count,
        features=features,
        legal=legal,
        actions=actions,
        value_targets=targets,
    )


def reservoir_exact(chunks: list[int], label: str) -> None:
    capacity = 64
    device = torch.device("cuda")
    reservoir = PackedGpuReservoirBuffer(capacity, 496, 177, device)
    expected: list[int] = []
    stream_offset = 0

    for count in chunks:
        batch = make_synthetic(stream_offset, count, device)
        rng = torch.cuda.get_rng_state()
        fill = min(count, capacity - len(expected))

        reservoir.add(batch)
        torch.cuda.synchronize()

        for src in range(fill):
            expected.append(stream_offset + src)

        remaining = count - fill
        if remaining:
            torch.cuda.set_rng_state(rng)
            positions = torch.arange(
                remaining, dtype=torch.float64, device=device
            ).add_(float(stream_offset + fill + 1))
            draws = torch.floor(
                torch.rand(remaining, dtype=torch.float64, device=device) * positions
            ).to(torch.int64)

            for local, dst in enumerate(draws.cpu().tolist()):
                if dst < capacity:
                    expected[int(dst)] = stream_offset + fill + local

        stream_offset += count
        if reservoir.total_seen != stream_offset:
            raise RuntimeError(
                f"{label}: total_seen={reservoir.total_seen} expected={stream_offset}"
            )
        expected_size = min(capacity, stream_offset)
        if reservoir.size != expected_size:
            raise RuntimeError(
                f"{label}: size={reservoir.size} expected={expected_size}"
            )

    index = torch.arange(reservoir.size, dtype=torch.int64, device=device)
    _features, _legal, actions, targets = reservoir._ext.gather_unpack(
        reservoir.feature_bits,
        reservoir.feature_scalars,
        reservoir.legal_bits,
        reservoir.actions,
        reservoir.value_targets,
        index,
    )
    expected_tensor = torch.tensor(expected, dtype=torch.float32, device=device)
    exact_tensor(f"{label}.targets", targets, expected_tensor)
    exact_tensor(
        f"{label}.actions",
        actions,
        expected_tensor.to(torch.int64).remainder(177),
    )
    print(
        "ALGORITHM-R EXACT: PASS "
        f"label={label} capacity={capacity} total_seen={reservoir.total_seen} "
        f"last_candidates={reservoir.last_candidates} "
        f"last_replacements={reservoir.last_replacements} "
        f"last_collisions={reservoir.last_collisions}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("torch=", torch.__version__)
    print("torch_cuda=", torch.version.cuda)
    print("device=", torch.cuda.get_device_name(0))
    print("capability=", torch.cuda.get_device_capability(0))

    summary = AverageTrainSummary(1.0, 2.0, 3.0, 4)
    print("AverageTrainSummary=", summary)

    env_loader._ensure_windows_build_env()
    ext = env_loader.load_extension(verbose=False)
    if int(ext.FEATURE_COUNT) != 496 or int(ext.ACTION_COUNT) != 177:
        raise RuntimeError("Torch50 contract mismatch")

    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    br_model = baseline.PolicyValueNet(496, 177, 256).cuda().eval()
    average_model = AveragePolicyNet(496, 177, 256).cuda().eval()
    initialize_average_from_br(average_model, br_model)

    for temperature in (0.0, 1.0):
        action_seed = 700000 + int(temperature * 1000)
        torch.cuda.manual_seed_all(action_seed)
        rng = torch.cuda.get_rng_state()

        old = baseline.collect_selfplay(
            ext=ext,
            model=br_model,
            batch=32768,
            seed_offset=99112233,
            max_steps=128,
            temperature=temperature,
            reward_scale=32.0,
            use_amp=True,
            max_transitions=1 << 20,
        )
        torch.cuda.synchronize()
        torch.cuda.set_rng_state(rng)
        new = collect_nfsp_selfplay(
            ext=ext,
            br_model=br_model,
            average_model=average_model,
            batch=32768,
            seed_offset=99112233,
            max_steps=128,
            anticipatory=1.0,
            br_temperature=temperature,
            average_temperature=temperature,
            reward_scale=32.0,
            use_amp=True,
            max_br_transitions=1 << 20,
        )
        torch.cuda.synchronize()
        compare_eta1(old, new, temperature)
        del old, new

    torch.cuda.manual_seed_all(424242)
    reservoir_exact([4096], "single_batch")
    torch.cuda.manual_seed_all(515151)
    reservoir_exact([13, 51, 200, 7, 400], "chunked")

    torch.cuda.manual_seed_all(888888)
    smoke = collect_nfsp_selfplay(
        ext=ext,
        br_model=br_model,
        average_model=average_model,
        batch=65536,
        seed_offset=555666777,
        max_steps=128,
        anticipatory=0.10,
        br_temperature=1.0,
        average_temperature=1.0,
        reward_scale=32.0,
        use_amp=True,
        max_br_transitions=1 << 20,
    )
    torch.cuda.synchronize()

    mode_pct = 100.0 * smoke.br_player_modes / (2.0 * smoke.games)
    br_pct = 100.0 * smoke.generated_br_transitions / smoke.total_decisions
    if not 7.0 <= mode_pct <= 13.0:
        raise RuntimeError(f"unexpected BR mode share {mode_pct:.3f}%")
    if not 5.0 <= br_pct <= 15.0:
        raise RuntimeError(f"unexpected BR decision share {br_pct:.3f}%")
    if smoke.dropped_transitions != 0:
        raise RuntimeError(
            f"BR trajectory unexpectedly dropped {smoke.dropped_transitions} rows"
        )
    if smoke.terminal + smoke.nagari != smoke.games:
        raise RuntimeError("terminal+nagari does not equal games")
    if smoke.transitions:
        chosen_legal = smoke.legal.gather(1, smoke.actions.unsqueeze(1)).squeeze(1)
        if not bool(chosen_legal.all().item()):
            raise RuntimeError("NFSP stored an illegal BR action")
    if not bool(torch.isfinite(smoke.value_targets).all().item()):
        raise RuntimeError("NFSP produced non-finite targets")

    print(
        "NFSP MIXED COLLECTOR: PASS "
        f"games={smoke.games} decisions={smoke.total_decisions} "
        f"br_generated={smoke.generated_br_transitions} "
        f"br_retained={smoke.transitions} br_mode_pct={mode_pct:.3f} "
        f"br_decision_pct={br_pct:.3f} steps={smoke.decision_steps} "
        f"terminal={smoke.terminal} nagari={smoke.nagari}"
    )
    print("NFSP CORE VALIDATION: PASS")


if __name__ == "__main__":
    main()
