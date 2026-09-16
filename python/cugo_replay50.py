from __future__ import annotations

from typing import Any

import torch

from cugo_replay50_ext import load_extension


FEATURE_COUNT = 496
FEATURE_BINARY_COUNT = 453
FEATURE_SCALAR_OFFSET = 453
FEATURE_SCALAR_COUNT = 26
FEATURE_PADDING_COUNT = 17
ACTION_COUNT = 177
FEATURE_PACK_BYTES = (FEATURE_BINARY_COUNT + 7) // 8
LEGAL_PACK_BYTES = (ACTION_COUNT + 7) // 8


class PackedGpuReplayBuffer:
    format_name = "packed-bitplanes-cuda-v2"

    def __init__(
        self,
        capacity: int,
        feature_count: int,
        action_count: int,
        device: torch.device,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if feature_count != FEATURE_COUNT:
            raise ValueError(
                f"packed replay requires feature_count={FEATURE_COUNT}, got {feature_count}"
            )
        if action_count != ACTION_COUNT:
            raise ValueError(
                f"packed replay requires action_count={ACTION_COUNT}, got {action_count}"
            )

        self.capacity = int(capacity)
        self.feature_count = int(feature_count)
        self.action_count = int(action_count)
        self.device = device
        self._ext = load_extension(verbose=False)

        if int(self._ext.FEATURE_COUNT) != FEATURE_COUNT:
            raise RuntimeError("replay CUDA extension feature contract mismatch")
        if int(self._ext.ACTION_COUNT) != ACTION_COUNT:
            raise RuntimeError("replay CUDA extension action contract mismatch")

        self.feature_bits = torch.empty(
            (capacity, FEATURE_PACK_BYTES), dtype=torch.uint8, device=device
        )
        self.feature_scalars = torch.empty(
            (capacity, FEATURE_SCALAR_COUNT), dtype=torch.float16, device=device
        )
        self.legal_bits = torch.empty(
            (capacity, LEGAL_PACK_BYTES), dtype=torch.uint8, device=device
        )
        self.actions = torch.empty(capacity, dtype=torch.uint8, device=device)
        self.value_targets = torch.empty(
            capacity, dtype=torch.float32, device=device
        )
        self.position = 0
        self.size = 0

    @property
    def storage_bytes(self) -> int:
        tensors = (
            self.feature_bits,
            self.feature_scalars,
            self.legal_bits,
            self.actions,
            self.value_targets,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    @property
    def bytes_per_transition(self) -> int:
        return self.storage_bytes // self.capacity

    def _write_segment(
        self,
        dst_start: int,
        features: torch.Tensor,
        legal: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        count = int(actions.numel())
        if count == 0:
            return

        self._ext.pack_into(
            self.feature_bits,
            self.feature_scalars,
            self.legal_bits,
            self.actions,
            self.value_targets,
            int(dst_start),
            features,
            legal,
            actions,
            targets,
        )

    def add(self, batch: Any) -> None:
        count = int(batch.transitions)
        if count == 0:
            return

        features = batch.features
        legal = batch.legal
        actions = batch.actions
        targets = batch.value_targets

        if count >= self.capacity:
            start = count - self.capacity
            self._write_segment(
                0,
                features[start:],
                legal[start:],
                actions[start:],
                targets[start:],
            )
            self.position = 0
            self.size = self.capacity
            return

        first = min(count, self.capacity - self.position)
        second = count - first

        self._write_segment(
            self.position,
            features[:first],
            legal[:first],
            actions[:first],
            targets[:first],
        )

        if second:
            self._write_segment(
                0,
                features[first:],
                legal[first:],
                actions[first:],
                targets[first:],
            )

        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty replay buffer")

        index = torch.randint(
            self.size,
            (batch_size,),
            dtype=torch.int64,
            device=self.device,
        )

        features, legal, actions, targets = self._ext.gather_unpack(
            self.feature_bits,
            self.feature_scalars,
            self.legal_bits,
            self.actions,
            self.value_targets,
            index,
        )
        return features, legal, actions, targets
