from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import time

import torch
from torch import nn

from cugo_nfsp50 import AveragePolicyNet
from cugo_train50 import PolicyValueNet, make_seeds


@dataclass
class LoadedPolicy:
    model: nn.Module
    kind: str
    checkpoint: str
    iteration: int | None
    hidden: int


@dataclass
class EvalSummary:
    pairs: int
    games: int
    evaluated_pairs: int
    evaluated_games: int
    predecision_excluded: int
    a_wins: int
    b_wins: int
    draws: int
    a_p0_wins: int
    a_p0_losses: int
    a_p0_draws: int
    a_p1_wins: int
    a_p1_losses: int
    a_p1_draws: int
    pair_wins: int
    pair_losses: int
    pair_draws: int
    mean_score_a: float
    mean_score_a_p0: float
    mean_score_a_p1: float
    mean_pair_score_a: float
    pair_score_std: float
    pair_score_se: float
    pair_score_ci95_low: float
    pair_score_ci95_high: float
    terminal: int
    nagari: int
    decision_steps: int
    seconds: float

    @property
    def games_per_s(self) -> float:
        return self.games / self.seconds


def _checkpoint_hidden(payload: dict[str, Any], default: int = 256) -> int:
    config = payload.get("config")
    if isinstance(config, dict) and "hidden" in config:
        return int(config["hidden"])
    return default


def load_policy_checkpoint(
    path: Path,
    policy: str,
    device: torch.device,
) -> LoadedPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} did not contain a dictionary")

    hidden = _checkpoint_hidden(payload)
    iteration_value = payload.get("iteration")
    iteration = int(iteration_value) if iteration_value is not None else None

    if policy == "auto":
        if "average_model" in payload:
            policy = "average"
        elif "model" in payload:
            policy = "legacy"
        elif "br_model" in payload:
            policy = "br"
        else:
            raise ValueError(f"cannot infer policy kind from checkpoint {path}")

    if policy == "average":
        state = payload.get("average_model")
        if state is None:
            raise ValueError(f"checkpoint {path} has no average_model")
        model: nn.Module = AveragePolicyNet(496, 177, hidden)
    elif policy == "br":
        state = payload.get("br_model")
        if state is None:
            raise ValueError(f"checkpoint {path} has no br_model")
        model = PolicyValueNet(496, 177, hidden)
    elif policy == "legacy":
        state = payload.get("model")
        if state is None:
            raise ValueError(f"checkpoint {path} has no legacy model")
        model = PolicyValueNet(496, 177, hidden)
    else:
        raise ValueError("policy must be one of auto, average, br, legacy")

    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return LoadedPolicy(
        model=model,
        kind=policy,
        checkpoint=str(path),
        iteration=iteration,
        hidden=hidden,
    )


def _policy_logits(model: nn.Module, features: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(features)
    if isinstance(output, tuple):
        return output[0]
    return output


def _greedy_actions(
    model: nn.Module,
    features: torch.Tensor,
    legal: torch.Tensor,
    use_amp: bool,
) -> torch.Tensor:
    logits = _policy_logits(model, features, use_amp)
    masked = logits.float().masked_fill(~legal, -1.0e30)
    return masked.argmax(dim=1).to(torch.int64)


def evaluate_paired(
    ext: Any,
    model_a: nn.Module,
    model_b: nn.Module,
    pairs: int,
    seed_offset: int,
    max_steps: int,
    use_amp: bool,
) -> EvalSummary:
    if pairs <= 0:
        raise ValueError("pairs must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    device = torch.device("cuda")
    base_seeds, base_first = make_seeds(pairs, seed_offset, device)
    seeds = torch.cat((base_seeds, base_seeds), dim=0)
    first = torch.cat((base_first, base_first), dim=0)
    games = pairs * 2

    a_seat = torch.cat(
        (
            torch.zeros(pairs, dtype=torch.uint8, device=device),
            torch.ones(pairs, dtype=torch.uint8, device=device),
        ),
        dim=0,
    )

    states = ext.create(seeds, first)
    final_reward0 = torch.zeros(games, dtype=torch.float32, device=device)
    final_nagari = torch.zeros(games, dtype=torch.bool, device=device)
    predecision_done = torch.zeros(games, dtype=torch.bool, device=device)
    active_ids = torch.arange(games, dtype=torch.int64, device=device)

    observe_bad = torch.zeros((), dtype=torch.int64, device=device)
    legal_bad = torch.zeros((), dtype=torch.int64, device=device)
    step_bad = torch.zeros((), dtype=torch.int64, device=device)

    model_a.eval()
    model_b.eval()
    completed = False
    first_observe = True
    decision_steps = 0

    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(max_steps):
            if active_ids.numel() == 0:
                completed = True
                break

            features, legal, players, observed_done, status = ext.observe_indexed(
                states, active_ids
            )
            observe_ok = status == int(ext.STATUS_OK)
            has_legal = legal.any(dim=1)

            if first_observe:
                active = ~observed_done
                valid_active = active & observe_ok & has_legal
                initial_done_rows = torch.nonzero(observed_done, as_tuple=False).squeeze(1)
                if initial_done_rows.numel():
                    predecision_done.index_fill_(0, active_ids.index_select(0, initial_done_rows), True)
                observe_bad.add_((active & ~observe_ok).sum())
                legal_bad.add_((active & observe_ok & ~has_legal).sum())
                keep_rows = torch.nonzero(valid_active, as_tuple=False).squeeze(1)
                active_ids = active_ids.index_select(0, keep_rows)
                features = features.index_select(0, keep_rows)
                legal = legal.index_select(0, keep_rows)
                players = players.index_select(0, keep_rows)
                first_observe = False
                if active_ids.numel() == 0:
                    completed = True
                    break
            else:
                observe_bad.add_((~observe_ok).sum())
                legal_bad.add_((observe_ok & (observed_done | ~has_legal)).sum())

            decision_steps += 1
            current_a_seat = a_seat.index_select(0, active_ids).to(torch.int64)
            use_a = players.to(torch.int64) == current_a_seat
            a_rows = torch.nonzero(use_a, as_tuple=False).squeeze(1)
            b_rows = torch.nonzero(~use_a, as_tuple=False).squeeze(1)
            actions = torch.empty(active_ids.numel(), dtype=torch.int64, device=device)

            if a_rows.numel():
                actions.index_copy_(
                    0,
                    a_rows,
                    _greedy_actions(
                        model_a,
                        features.index_select(0, a_rows),
                        legal.index_select(0, a_rows),
                        use_amp,
                    ),
                )
            if b_rows.numel():
                actions.index_copy_(
                    0,
                    b_rows,
                    _greedy_actions(
                        model_b,
                        features.index_select(0, b_rows),
                        legal.index_select(0, b_rows),
                        use_amp,
                    ),
                )

            reward0, new_done, nagari, step_status, _committed = ext.step_indexed(
                states, active_ids, actions
            )
            step_ok = step_status == int(ext.STATUS_OK)
            step_bad.add_((~step_ok).sum())
            just_finished = step_ok & new_done
            final_reward0.index_copy_(
                0,
                active_ids,
                torch.where(just_finished, reward0, torch.zeros_like(reward0)),
            )
            final_nagari.index_copy_(0, active_ids, just_finished & nagari)

            keep_rows = torch.nonzero(step_ok & ~new_done, as_tuple=False).squeeze(1)
            active_ids = active_ids.index_select(0, keep_rows)
            if active_ids.numel() == 0:
                completed = True
                break

    torch.cuda.synchronize()
    seconds = time.perf_counter() - start

    counts = torch.stack((observe_bad, legal_bad, step_bad)).tolist()
    observe_bad_count, legal_bad_count, step_bad_count = map(int, counts)
    if observe_bad_count:
        raise RuntimeError(f"observe failed on {observe_bad_count} active visits")
    if legal_bad_count:
        raise RuntimeError(f"active state had invalid/no legal actions on {legal_bad_count} visits")
    if step_bad_count:
        raise RuntimeError(f"step failed on {step_bad_count} active visits")
    if not completed:
        raise RuntimeError(f"not all games finished within {max_steps} decision steps")

    score_a = torch.where(a_seat == 0, final_reward0, -final_reward0)
    p0_score = score_a[:pairs]
    p1_score = score_a[pairs:]
    pair_score = (p0_score + p1_score) * 0.5

    def _wdl(x: torch.Tensor) -> tuple[int, int, int]:
        return (
            int((x > 0).sum().item()),
            int((x < 0).sum().item()),
            int((x == 0).sum().item()),
        )

    p0_valid = ~predecision_done[:pairs]
    p1_valid = ~predecision_done[pairs:]
    pair_valid = p0_valid & p1_valid
    valid_score = score_a[~predecision_done]
    valid_p0_score = p0_score[p0_valid]
    valid_p1_score = p1_score[p1_valid]
    valid_pair_score = pair_score[pair_valid]

    a_wins, b_wins, draws = _wdl(valid_score)
    a_p0_wins, a_p0_losses, a_p0_draws = _wdl(valid_p0_score)
    a_p1_wins, a_p1_losses, a_p1_draws = _wdl(valid_p1_score)
    pair_wins, pair_losses, pair_draws = _wdl(valid_pair_score)

    evaluated_pairs = int(pair_valid.sum().item())
    evaluated_games = int((~predecision_done).sum().item())
    predecision_excluded = int(predecision_done.sum().item())
    mean_pair = float(valid_pair_score.mean().item()) if evaluated_pairs else 0.0
    if evaluated_pairs > 1:
        pair_std = float(valid_pair_score.std(unbiased=True).item())
        pair_se = pair_std / math.sqrt(float(evaluated_pairs))
    else:
        pair_std = 0.0
        pair_se = 0.0
    ci_half = 1.959963984540054 * pair_se

    return EvalSummary(
        pairs=pairs,
        games=games,
        evaluated_pairs=evaluated_pairs,
        evaluated_games=evaluated_games,
        predecision_excluded=predecision_excluded,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_p0_wins=a_p0_wins,
        a_p0_losses=a_p0_losses,
        a_p0_draws=a_p0_draws,
        a_p1_wins=a_p1_wins,
        a_p1_losses=a_p1_losses,
        a_p1_draws=a_p1_draws,
        pair_wins=pair_wins,
        pair_losses=pair_losses,
        pair_draws=pair_draws,
        mean_score_a=float(valid_score.mean().item()) if evaluated_games else 0.0,
        mean_score_a_p0=float(valid_p0_score.mean().item()) if valid_p0_score.numel() else 0.0,
        mean_score_a_p1=float(valid_p1_score.mean().item()) if valid_p1_score.numel() else 0.0,
        mean_pair_score_a=mean_pair,
        pair_score_std=pair_std,
        pair_score_se=pair_se,
        pair_score_ci95_low=mean_pair - ci_half,
        pair_score_ci95_high=mean_pair + ci_half,
        terminal=int((~final_nagari).sum().item()),
        nagari=int(final_nagari.sum().item()),
        decision_steps=decision_steps,
        seconds=seconds,
    )
