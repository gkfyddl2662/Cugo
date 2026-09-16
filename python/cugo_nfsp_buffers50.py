from __future__ import annotations

import torch


class _NfspTrajectoryBuffer:
    def __init__(
        self,
        capacity: int,
        feature_count: int,
        action_count: int,
        device: torch.device,
    ) -> None:
        if capacity <= 0:
            raise ValueError("trajectory capacity must be positive")
        self.capacity = int(capacity)
        self.features = torch.empty(
            (capacity, feature_count), dtype=torch.float16, device=device
        )
        self.legal = torch.empty(
            (capacity, action_count), dtype=torch.bool, device=device
        )
        self.actions = torch.empty(capacity, dtype=torch.int64, device=device)
        self.players = torch.empty(capacity, dtype=torch.uint8, device=device)
        self.env_ids = torch.empty(capacity, dtype=torch.int64, device=device)
        self.position = 0
        self.size = 0

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.features,
            self.legal,
            self.actions,
            self.players,
            self.env_ids,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def append(
        self,
        features: torch.Tensor,
        legal: torch.Tensor,
        actions: torch.Tensor,
        players: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        count = int(actions.numel())
        if count == 0:
            return

        if count >= self.capacity:
            start = count - self.capacity
            self.features.copy_(features[start:])
            self.legal.copy_(legal[start:])
            self.actions.copy_(actions[start:])
            self.players.copy_(players[start:])
            self.env_ids.copy_(env_ids[start:])
            self.position = 0
            self.size = self.capacity
            return

        first = min(count, self.capacity - self.position)
        second = count - first
        end = self.position + first

        self.features[self.position:end].copy_(features[:first])
        self.legal[self.position:end].copy_(legal[:first])
        self.actions[self.position:end].copy_(actions[:first])
        self.players[self.position:end].copy_(players[:first])
        self.env_ids[self.position:end].copy_(env_ids[:first])

        if second:
            self.features[:second].copy_(features[first:])
            self.legal[:second].copy_(legal[first:])
            self.actions[:second].copy_(actions[first:])
            self.players[:second].copy_(players[first:])
            self.env_ids[:second].copy_(env_ids[first:])

        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def finish(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.size == 0:
            return (
                self.features[:0],
                self.legal[:0],
                self.actions[:0],
                self.players[:0],
                self.env_ids[:0],
            )
        if self.size < self.capacity:
            end = self.size
            return (
                self.features[:end],
                self.legal[:end],
                self.actions[:end],
                self.players[:end],
                self.env_ids[:end],
            )
        if self.position == 0:
            return (
                self.features,
                self.legal,
                self.actions,
                self.players,
                self.env_ids,
            )

        position = self.position
        return (
            torch.cat((self.features[position:], self.features[:position]), dim=0),
            torch.cat((self.legal[position:], self.legal[:position]), dim=0),
            torch.cat((self.actions[position:], self.actions[:position]), dim=0),
            torch.cat((self.players[position:], self.players[:position]), dim=0),
            torch.cat((self.env_ids[position:], self.env_ids[:position]), dim=0),
        )


def nfsp_trajectory_storage_bytes(
    capacity: int,
    feature_count: int,
    action_count: int,
) -> int:
    return int(capacity) * (
        int(feature_count) * 2
        + int(action_count)
        + 8
        + 1
        + 8
    )

