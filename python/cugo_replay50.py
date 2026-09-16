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

    def _write_indexed(
        self,
        dst_index: torch.Tensor,
        features: torch.Tensor,
        legal: torch.Tensor,
        actions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        count = int(dst_index.numel())
        if count == 0:
            return
        if dst_index.dtype != torch.int64 or dst_index.device != self.device:
            raise ValueError("dst_index must be a CUDA int64 tensor on the replay device")

        packed_features = self._pack_bits(
            features[:, :FEATURE_BINARY_COUNT], FEATURE_BINARY_COUNT
        )
        packed_legal = self._pack_bits(legal, ACTION_COUNT)
        scalars = features[
            :,
            FEATURE_SCALAR_OFFSET : FEATURE_SCALAR_OFFSET + FEATURE_SCALAR_COUNT,
        ].to(torch.float16)

        self.feature_bits.index_copy_(0, dst_index, packed_features)
        self.feature_scalars.index_copy_(0, dst_index, scalars)
        self.legal_bits.index_copy_(0, dst_index, packed_legal)
        self.actions.index_copy_(0, dst_index, actions.to(torch.uint8))
        self.value_targets.index_copy_(0, dst_index, targets.to(torch.float32))

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


class PackedGpuReservoirBuffer(PackedGpuReplayBuffer):
    """Packed Algorithm-R reservoir for historical policy examples.

    Once full, each incoming item at global stream position t draws an integer
    uniformly from [0, t]. Items whose draw is below capacity replace that slot.
    Batched replacement collisions are resolved by keeping the latest incoming
    row, exactly matching sequential Algorithm-R update order for those draws.
    """

    format_name = "packed-bitplanes-cuda-v2-reservoir"

    def __init__(
        self,
        capacity: int,
        feature_count: int,
        action_count: int,
        device: torch.device,
    ) -> None:
        super().__init__(capacity, feature_count, action_count, device)
        self.total_seen = 0
        self.last_candidates = 0
        self.last_replacements = 0
        self.last_collisions = 0

    def add(self, batch: Any) -> None:
        count = int(batch.transitions)
        self.last_candidates = 0
        self.last_replacements = 0
        self.last_collisions = 0
        if count == 0:
            return

        features = batch.features
        legal = batch.legal
        actions = batch.actions
        targets = batch.value_targets

        fill = min(count, self.capacity - self.size)
        if fill:
            self._write_segment(
                self.size,
                features[:fill],
                legal[:fill],
                actions[:fill],
                targets[:fill],
            )
            self.size += fill
            self.total_seen += fill
            self.position = self.size % self.capacity

        remaining = count - fill
        if remaining == 0:
            return

        src_base = fill
        stream_positions = torch.arange(
            remaining, dtype=torch.float64, device=self.device
        ).add_(float(self.total_seen + 1))
        draws = torch.floor(
            torch.rand(remaining, dtype=torch.float64, device=self.device)
            * stream_positions
        ).to(torch.int64)
        candidate_mask = draws < self.capacity
        candidate_src = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
        candidate_dst = draws.index_select(0, candidate_src)
        candidate_count = int(candidate_src.numel())
        self.last_candidates = candidate_count

        if candidate_count:
            # Sort by (destination, source-order) and keep the last source for
            # each destination, matching sequential replacement semantics.
            key_scale = remaining + 1
            keys = candidate_dst.mul(key_scale).add(candidate_src)
            order = torch.argsort(keys)
            sorted_dst = candidate_dst.index_select(0, order)
            sorted_src = candidate_src.index_select(0, order)

            keep = torch.ones(
                candidate_count, dtype=torch.bool, device=self.device
            )
            if candidate_count > 1:
                keep[:-1] = sorted_dst[:-1] != sorted_dst[1:]
            keep_rows = torch.nonzero(keep, as_tuple=False).squeeze(1)
            unique_dst = sorted_dst.index_select(0, keep_rows)
            unique_src = sorted_src.index_select(0, keep_rows).add(src_base)

            self._write_indexed(
                unique_dst,
                features.index_select(0, unique_src),
                legal.index_select(0, unique_src),
                actions.index_select(0, unique_src),
                targets.index_select(0, unique_src),
            )
            replacement_count = int(unique_dst.numel())
            self.last_replacements = replacement_count
            self.last_collisions = candidate_count - replacement_count

        self.total_seen += remaining
        self.size = min(self.capacity, self.total_seen)
        self.position = 0 if self.size == self.capacity else self.size
