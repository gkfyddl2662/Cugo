from __future__ import annotations

import argparse
import time

import torch
from torch import nn

from cugo_torch50_ext import load_extension


SIGNED_GOLDEN = -7046029254386353131
SEED_BASE = 0x243F6A8885A308D3


class PolicyValueNet(nn.Module):
    def __init__(self, features: int, actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.policy = nn.Linear(hidden, actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


def make_seeds(batch: int, offset: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(batch, dtype=torch.int64, device=device)
    seeds = index.mul(SIGNED_GOLDEN).add(SEED_BASE + offset)
    first = index.bitwise_and(1)
    return seeds, first


def run_games(ext, model: nn.Module, batch: int, seed_offset: int, max_steps: int) -> dict[str, float]:
    device = torch.device("cuda")
    seeds, first = make_seeds(batch, seed_offset, device)
    states = ext.create(seeds, first)
    done = torch.zeros(batch, dtype=torch.bool, device=device)
    final_reward = torch.zeros(batch, dtype=torch.float32, device=device)
    final_nagari = torch.zeros(batch, dtype=torch.bool, device=device)
    decision_steps = 0

    for decision_steps in range(1, max_steps + 1):
        features, legal, _players, observed_done, status = ext.observe(states)
        active = ~observed_done
        if not bool(active.any().item()):
            done = observed_done
            break
        if bool((status[active] != int(ext.STATUS_OK)).any().item()):
            raise RuntimeError("observe returned a non-OK status for an active environment")
        if bool((legal[active].sum(dim=1) == 0).any().item()):
            raise RuntimeError("an active environment has no legal actions")

        x = features.to(dtype=torch.float16)
        logits, _value = model(x)
        masked_logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        actions = masked_logits.argmax(dim=1).to(dtype=torch.int64)

        reward, new_done, nagari, step_status, _committed = ext.step(states, actions)
        if bool((step_status[active] != int(ext.STATUS_OK)).any().item()):
            bad = int((step_status[active] != int(ext.STATUS_OK)).sum().item())
            raise RuntimeError(f"step returned non-OK status for {bad} active environments")

        just_finished = active & new_done
        final_reward = torch.where(just_finished, reward, final_reward)
        final_nagari |= just_finished & nagari
        done = new_done
        if bool(done.all().item()):
            break
    else:
        raise RuntimeError(f"not all games finished within {max_steps} decision steps")

    terminal = int((done & ~final_nagari).sum().item())
    nagari_count = int(final_nagari.sum().item())
    return {
        "steps": float(decision_steps),
        "terminal": float(terminal),
        "nagari": float(nagari_count),
        "mean_abs_reward": float(final_reward.abs().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is unavailable")

    print(f"torch={torch.__version__} torch_cuda={torch.version.cuda}")
    print(f"device={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")

    ext = load_extension(verbose=args.verbose_build)
    print(
        f"extension state_bytes={ext.STATE_BYTES} features={ext.FEATURE_COUNT} "
        f"actions={ext.ACTION_COUNT} primary={ext.PRIMARY_ACTION_COUNT}"
    )
    if int(ext.FEATURE_COUNT) != 496 or int(ext.ACTION_COUNT) != 177:
        raise RuntimeError("extension constants do not match the frozen Torch50 contract")

    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)
    model = PolicyValueNet(int(ext.FEATURE_COUNT), int(ext.ACTION_COUNT), args.hidden)
    model = model.cuda().half().eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"network hidden={args.hidden} params={params} dtype=fp16")

    with torch.inference_mode():
        smoke_batch = min(args.batch, 4096)
        smoke = run_games(ext, model, smoke_batch, 0, args.max_steps)
        print(
            "smoke PASS "
            f"batch={smoke_batch} steps={int(smoke['steps'])} "
            f"terminal={int(smoke['terminal'])} nagari={int(smoke['nagari'])} "
            f"mean_abs_reward={smoke['mean_abs_reward']:.4f}"
        )

        for i in range(args.warmup):
            run_games(ext, model, args.batch, 1000000 + i * args.batch, args.max_steps)
        torch.cuda.synchronize()

        elapsed = []
        for i in range(args.iters):
            start = time.perf_counter()
            summary = run_games(
                ext,
                model,
                args.batch,
                2000000 + i * args.batch,
                args.max_steps,
            )
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            elapsed.append(seconds)
            print(
                f"run={i + 1} batch={args.batch} seconds={seconds:.6f} "
                f"games_per_s={args.batch / seconds:.3f} "
                f"steps={int(summary['steps'])} terminal={int(summary['terminal'])} "
                f"nagari={int(summary['nagari'])}"
            )

        mean_seconds = sum(elapsed) / len(elapsed)
        mean_gps = args.batch / mean_seconds
        best_gps = max(args.batch / seconds for seconds in elapsed)
        print(
            f"benchmark PASS mean_games_per_s={mean_gps:.3f} "
            f"best_games_per_s={best_gps:.3f} "
            f"mean_ms_per_batch={mean_seconds * 1000.0:.3f}"
        )


if __name__ == "__main__":
    main()
