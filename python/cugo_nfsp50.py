from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cugo_nfsp_buffers50 import _NfspTrajectoryBuffer, nfsp_trajectory_storage_bytes
from cugo_train50 import PolicyValueNet, make_seeds


class AveragePolicyNet(nn.Module):
    def __init__(self, features: int, actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.policy = nn.Linear(hidden, actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.trunk(x))


def initialize_average_from_br(
    average_model: AveragePolicyNet,
    br_model: PolicyValueNet,
) -> None:
    average_model.trunk.load_state_dict(br_model.trunk.state_dict())
    average_model.policy.load_state_dict(br_model.policy.state_dict())


@dataclass
class NfspSelfPlayBatch:
    features: torch.Tensor
    legal: torch.Tensor
    actions: torch.Tensor
    value_targets: torch.Tensor
    games: int
    total_decisions: int
    generated_br_transitions: int
    terminal: int
    nagari: int
    decision_steps: int
    br_player_modes: int
    mean_abs_reward0: float

    @property
    def transitions(self) -> int:
        return int(self.actions.numel())

    @property
    def dropped_transitions(self) -> int:
        return self.generated_br_transitions - self.transitions



def _sample_actions(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    masked = logits.float().masked_fill(~legal, -1.0e30)
    if temperature <= 0.0:
        return masked.argmax(dim=1).to(torch.int64)
    probabilities = torch.softmax(masked / temperature, dim=1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(1).to(torch.int64)


def _sample_subset_actions(
    model: nn.Module,
    features: torch.Tensor,
    legal: torch.Tensor,
    rows: torch.Tensor,
    temperature: float,
    use_amp: bool,
    *,
    br_model: bool,
) -> torch.Tensor:
    subset_features = features.index_select(0, rows)
    subset_legal = legal.index_select(0, rows)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
        output = model(subset_features)
        logits = output[0] if br_model else output
    return _sample_actions(logits, subset_legal, temperature)


def collect_nfsp_selfplay(
    ext: Any,
    br_model: PolicyValueNet,
    average_model: AveragePolicyNet,
    batch: int,
    seed_offset: int,
    max_steps: int,
    anticipatory: float,
    br_temperature: float,
    average_temperature: float,
    reward_scale: float,
    use_amp: bool,
    max_br_transitions: int,
) -> NfspSelfPlayBatch:
    if batch <= 0:
        raise ValueError("batch must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not (0.0 < anticipatory <= 1.0):
        raise ValueError("anticipatory must be in (0, 1]")
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    if max_br_transitions <= 0:
        raise ValueError("max_br_transitions must be positive")

    device = torch.device("cuda")
    seeds, first = make_seeds(batch, seed_offset, device)
    states = ext.create(seeds, first)
    final_reward0 = torch.zeros(batch, dtype=torch.float32, device=device)
    final_nagari = torch.zeros(batch, dtype=torch.bool, device=device)
    active_ids = torch.arange(batch, dtype=torch.int64, device=device)

    if anticipatory >= 1.0:
        br_modes = torch.ones((batch, 2), dtype=torch.bool, device=device)
    else:
        br_modes = torch.rand((batch, 2), device=device) < anticipatory

    trajectory = _NfspTrajectoryBuffer(
        max_br_transitions,
        int(ext.FEATURE_COUNT),
        int(ext.ACTION_COUNT),
        device,
    )

    observe_bad = torch.zeros((), dtype=torch.int64, device=device)
    legal_bad = torch.zeros((), dtype=torch.int64, device=device)
    step_bad = torch.zeros((), dtype=torch.int64, device=device)

    br_model.eval()
    average_model.eval()
    total_decisions = 0
    generated_br_transitions = 0
    decision_steps = 0
    completed = False
    first_observe = True

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

            active_count = int(active_ids.numel())
            total_decisions += active_count
            decision_steps += 1

            player_index = players.to(torch.int64)
            use_br = br_modes[active_ids, player_index]
            active_actions = torch.empty(
                active_count, dtype=torch.int64, device=device
            )

            if anticipatory >= 1.0:
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=use_amp
                ):
                    br_logits, _ = br_model(features)
                active_actions.copy_(
                    _sample_actions(br_logits, legal, br_temperature)
                )
                br_rows = torch.arange(active_count, dtype=torch.int64, device=device)
            else:
                br_rows = torch.nonzero(use_br, as_tuple=False).squeeze(1)
                average_rows = torch.nonzero(~use_br, as_tuple=False).squeeze(1)
                if br_rows.numel():
                    active_actions.index_copy_(
                        0,
                        br_rows,
                        _sample_subset_actions(
                            br_model,
                            features,
                            legal,
                            br_rows,
                            br_temperature,
                            use_amp,
                            br_model=True,
                        ),
                    )
                if average_rows.numel():
                    active_actions.index_copy_(
                        0,
                        average_rows,
                        _sample_subset_actions(
                            average_model,
                            features,
                            legal,
                            average_rows,
                            average_temperature,
                            use_amp,
                            br_model=False,
                        ),
                    )

            br_count = int(br_rows.numel())
            generated_br_transitions += br_count
            if br_count:
                trajectory.append(
                    features.index_select(0, br_rows),
                    legal.index_select(0, br_rows),
                    active_actions.index_select(0, br_rows),
                    players.index_select(0, br_rows),
                    active_ids.index_select(0, br_rows),
                )

            reward0, new_done, nagari, step_status, _committed = ext.step_indexed(
                states, active_ids, active_actions
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

    error_counts = torch.stack((observe_bad, legal_bad, step_bad)).tolist()
    observe_bad_count, legal_bad_count, step_bad_count = map(int, error_counts)
    if observe_bad_count:
        raise RuntimeError(
            f"observe returned non-OK status for {observe_bad_count} active environment-visits"
        )
    if legal_bad_count:
        raise RuntimeError(
            f"an active environment had no legal actions on {legal_bad_count} visits"
        )
    if step_bad_count:
        raise RuntimeError(
            f"step returned non-OK status for {step_bad_count} active environment-visits"
        )
    if not completed:
        raise RuntimeError(f"not all games finished within {max_steps} decision steps")

    features, legal, actions, players, env_ids = trajectory.finish()
    if actions.numel():
        reward_for_transition = final_reward0[env_ids]
        signed_reward = torch.where(
            players == 0, reward_for_transition, -reward_for_transition
        )
        value_targets = signed_reward / reward_scale
    else:
        value_targets = torch.empty(0, dtype=torch.float32, device=device)

    terminal = int((~final_nagari).sum().item())
    nagari_count = int(final_nagari.sum().item())
    br_player_modes = int(br_modes.sum().item())
    return NfspSelfPlayBatch(
        features=features,
        legal=legal,
        actions=actions,
        value_targets=value_targets,
        games=batch,
        total_decisions=total_decisions,
        generated_br_transitions=generated_br_transitions,
        terminal=terminal,
        nagari=nagari_count,
        decision_steps=decision_steps,
        br_player_modes=br_player_modes,
        mean_abs_reward0=float(final_reward0.abs().mean().item()),
    )


