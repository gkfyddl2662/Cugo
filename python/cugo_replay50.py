from __future__ import annotations

from typing import Any

import torch


FEATURE_COUNT = 496
FEATURE_BINARY_COUNT = 453
FEATURE_SCALAR_OFFSET = 453
FEATURE_SCALAR_COUNT = 26
FEATURE_PADDING_COUNT = 17
ACTION_COUNT = 177
FEATURE_PACK_BYTES = (FEATURE_BINARY_COUNT + 7) // 8
LEGAL_PACK_BYTES = (ACTION_COUNT + 7) // 8
PACK_CHUNK_ROWS = 262144


class PackedGpuReplayBuffer:
    format_name = "packed-bitplanes-v1"

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

    @staticmethod
    def _pack_bits(source: torch.Tensor, bit_count: int) -> torch.Tensor:
        rows = source.size(0)
        packed_bytes = (bit_count + 7) // 8
        packed = torch.zeros(
            (rows, packed_bytes), dtype=torch.uint8, device=source.device
        )
        for bit in range(8):
            columns = source[:, bit:bit_count:8]
            width = columns.size(1)
            if width == 0:
                continue
            encoded = columns.ne(0).to(torch.uint8).bitwise_left_shift(bit)
            packed[:, :width].bitwise_or_(encoded)
        return packed

    @staticmethod
    def _unpack_bits(
        packed: torch.Tensor,
        bit_count: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
        unpacked = (
            packed.unsqueeze(-1)
            .bitwise_right_shift(shifts)
            .bitwise_and(1)
            .reshape(packed.size(0), -1)[:, :bit_count]
        )
        return unpacked.to(dtype)

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

        for src_start in range(0, count, PACK_CHUNK_ROWS):
            rows = min(PACK_CHUNK_ROWS, count - src_start)
            src_end = src_start + rows
            dst_chunk_start = dst_start + src_start
            dst_chunk_end = dst_chunk_start + rows

            feature_chunk = features[src_start:src_end]
            legal_chunk = legal[src_start:src_end]

            self.feature_bits[dst_chunk_start:dst_chunk_end].copy_(
                self._pack_bits(feature_chunk[:, :FEATURE_BINARY_COUNT], FEATURE_BINARY_COUNT)
            )
            self.feature_scalars[dst_chunk_start:dst_chunk_end].copy_(
                feature_chunk[
                    :,
                    FEATURE_SCALAR_OFFSET : FEATURE_SCALAR_OFFSET
                    + FEATURE_SCALAR_COUNT,
                ]
            )
            self.legal_bits[dst_chunk_start:dst_chunk_end].copy_(
                self._pack_bits(legal_chunk, ACTION_COUNT)
            )
            self.actions[dst_chunk_start:dst_chunk_end].copy_(
                actions[src_start:src_end]
            )
            self.value_targets[dst_chunk_start:dst_chunk_end].copy_(
                targets[src_start:src_end]
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

        packed_features = self.feature_bits.index_select(0, index)
        features = torch.zeros(
            (batch_size, FEATURE_COUNT), dtype=torch.float16, device=self.device
        )
        features[:, :FEATURE_BINARY_COUNT].copy_(
            self._unpack_bits(
                packed_features,
                FEATURE_BINARY_COUNT,
                torch.float16,
            )
        )
        features[
            :,
            FEATURE_SCALAR_OFFSET : FEATURE_SCALAR_OFFSET + FEATURE_SCALAR_COUNT,
        ].copy_(self.feature_scalars.index_select(0, index))

        legal = self._unpack_bits(
            self.legal_bits.index_select(0, index),
            ACTION_COUNT,
            torch.bool,
        )
        actions = self.actions.index_select(0, index).to(torch.int64)
        targets = self.value_targets.index_select(0, index)
        return features, legal, actions, targets
