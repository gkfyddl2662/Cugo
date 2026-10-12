from __future__ import annotations

import torch
from torch import nn

FEATURES = 496
ACTIONS = 177
ORACLE_FEATURES = 50  # opponent hidden hand plane, critic only


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ResidualMlp(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: int, blocks: int) -> None:
        super().__init__()
        self.inp = nn.Linear(inputs, hidden)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden) for _ in range(blocks)))
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(x))
        return self.out(torch.nn.functional.silu(self.norm(h)))


class PpoActor(ResidualMlp):
    """Policy over the public 496-feature information state; returns logits."""

    def __init__(self, hidden: int = 512, blocks: int = 3) -> None:
        super().__init__(FEATURES, ACTIONS, hidden, blocks)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)


class OracleCritic(ResidualMlp):
    """Value of the full history: public features plus the opponent's hidden hand."""

    def __init__(self, hidden: int = 512, blocks: int = 3) -> None:
        super().__init__(FEATURES + ORACLE_FEATURES, 1, hidden, blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).squeeze(-1)


_BITS: dict[torch.device, torch.Tensor] = {}


def card_plane(masks: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Unpack int64 card masks [N] into a [N, 50] 0/1 plane."""
    bits = _BITS.get(masks.device)
    if bits is None:
        bits = _BITS[masks.device] = torch.arange(ORACLE_FEATURES, device=masks.device)
    return ((masks.unsqueeze(1) >> bits) & 1).to(dtype)
