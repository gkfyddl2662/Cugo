from __future__ import annotations

import torch
from torch import nn

FEATURES = 720  # 560 (legacy 496 + Hangame block) + public-history planes (150) + padding
ACTIONS = 179  # 177 legacy + Gukjin pi/animal
ORACLE_FEATURES = 50  # opponent hidden hand plane, critic only
STOCK_DRAWS = 8  # ext.stock_indexed packs the next 8 stock draws, one card id per byte
STOCK_FEATURES = STOCK_DRAWS * 50  # optional critic input: one-hot of each draw
SIGNED_GOLDEN = -7046029254386353131
SEED_BASE = 0x243F6A8885A308D3


def make_seeds(batch: int, offset: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(batch, dtype=torch.int64, device=device)
    return index.mul(SIGNED_GOLDEN).add(SEED_BASE + offset), index.bitwise_and(1)


def _norm(kind: str, hidden: int) -> nn.Module:
    # LayerNorm plus the fp32<->bf16 casts it forces cost ~45% of a training step;
    # "none" relies on zero-initialized residual branches for stability instead.
    return nn.LayerNorm(hidden) if kind == "layer" else nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, norm: str = "layer") -> None:
        super().__init__()
        self.norm = _norm(norm, hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.SiLU()
        if norm == "none":
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class ResidualMlp(nn.Module):
    def __init__(self, inputs: int, outputs: int, hidden: int, blocks: int, norm: str = "layer") -> None:
        super().__init__()
        self.inp = nn.Linear(inputs, hidden)
        self.blocks = nn.Sequential(*(ResidualBlock(hidden, norm) for _ in range(blocks)))
        self.norm = _norm(norm, hidden)
        self.out = nn.Linear(hidden, outputs)

    def forward_hidden(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Output and the trunk's final hidden state (for auxiliary heads)."""
        h = self.blocks(self.inp(x))
        return self.out(torch.nn.functional.silu(self.norm(h))), h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_hidden(x)[0]


class PpoActor(ResidualMlp):
    """Policy over the public 720-feature information state; returns logits."""

    def __init__(self, hidden: int = 512, blocks: int = 3, norm: str = "layer") -> None:
        super().__init__(FEATURES, ACTIONS, hidden, blocks, norm)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)


class OracleCritic(ResidualMlp):
    """Value of the full history: public features plus the opponent's hidden hand, and
    optionally the next stock draws. With value_bins it predicts a distribution over
    symlog-spaced returns (see value_support / two_hot); value() gives its mean."""

    def __init__(self, hidden: int = 512, blocks: int = 3, norm: str = "layer", stock: bool = False,
                 value_bins: int = 0, value_top: float = 640.0) -> None:
        inputs = FEATURES + ORACLE_FEATURES + (STOCK_FEATURES if stock else 0)
        super().__init__(inputs, value_bins or 1, hidden, blocks, norm)
        self.stock, self.value_bins = stock, value_bins
        if value_bins:
            self.register_buffer("support", value_support(value_bins, value_top), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return out if self.value_bins else out.squeeze(-1)

    def value(self, out: torch.Tensor) -> torch.Tensor:
        if not self.value_bins:
            return out.float()
        return (torch.softmax(out.float(), -1) * self.support).sum(-1)  # elementwise: stays fp32 under autocast


class BeliefHead(nn.Module):
    """Auxiliary actor head: which cards the opponent holds, from a decision's hidden
    state. Trained against the true hand so the shared trunk learns to track it."""

    def __init__(self, inputs: int, width: int | None = None) -> None:
        super().__init__()
        width = width or inputs
        self.net = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, width), nn.GELU(),
                                 nn.Linear(width, ORACLE_FEATURES))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def stock_planes(stock: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[..., 8 * 50] one-hot of the next 8 stock draws (packed by ext.stock_indexed)."""
    shifts = torch.arange(0, 8 * STOCK_DRAWS, 8, device=stock.device)
    cards = (stock.unsqueeze(-1) >> shifts) & 0xFF
    planes = cards.unsqueeze(-1) == torch.arange(50, device=stock.device)  # 0xFF matches no card
    return planes.flatten(-2).to(dtype)


def value_support(bins: int, top: float) -> torch.Tensor:
    """Bin centers spaced evenly in symlog space over [-top, top] (scaled return units)."""
    u = torch.linspace(-1.0, 1.0, bins, dtype=torch.float64) * torch.log1p(torch.tensor(top, dtype=torch.float64))
    return (torch.sign(u) * torch.expm1(u.abs())).float()


def two_hot(y: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Target distribution over the bins whose mean is exactly y (clamped to the support)."""
    y = torch.minimum(torch.maximum(y, support[0]), support[-1])  # tensor bounds: no host sync in graphs
    hi = torch.searchsorted(support, y.contiguous()).clamp(1, support.numel() - 1)
    lo = hi - 1
    w_hi = ((y - support[lo]) / (support[hi] - support[lo])).clamp(0.0, 1.0)
    target = y.new_zeros(*y.shape, support.numel())
    target.scatter_(-1, lo.unsqueeze(-1), (1.0 - w_hi).unsqueeze(-1))
    target.scatter_add_(-1, hi.unsqueeze(-1), w_hi.unsqueeze(-1))
    return target


_BITS: dict[torch.device, torch.Tensor] = {}


def card_plane(masks: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Unpack int64 card masks [N] into a [N, 50] 0/1 plane."""
    bits = _BITS.get(masks.device)
    if bits is None:
        bits = _BITS[masks.device] = torch.arange(ORACLE_FEATURES, device=masks.device)
    return ((masks.unsqueeze(1) >> bits) & 1).to(dtype)
