"""Event-sequence policies for Shin Matgo 50.

A decision is read as a token sequence: the opening as the observer saw it, one
token per committed action so far (cards played, flipped, captured and stolen,
from the observer's side of the table), and a query token describing the decision
at hand. Two query variants share everything else:

- "hybrid": the query token also carries the engine's full 720-feature summary of
  the current state.
- "events": the query token carries only the decision context (pending cards,
  choice metadata, decision kind) and the legal actions; the network must track
  the table from the events itself.

The engine's event records are observer-independent (see make_torch_event50); the
encoder hides cards drawn into the opponent's hand and keeps only their count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cugo_ppo_nets50 import ACTIONS, FEATURES, ORACLE_FEATURES, card_plane

EVENT_WORDS = 9
START_WORDS = 8
MAX_EVENTS = 48
KINDS = 12  # ActionKind50
EVENT_SLOTS = 8  # played, own drawn, flipped, my gain, opp gain, my loss, opp loss, shown
EVENT_IN = EVENT_SLOTS * 50 + KINDS + 2
START_IN = 5 * 50 + 12 + 8 + 4
# Decision context inside the 720 features: pending public cards, selected pi,
# decision one-hot, post-chance scalars (requested pi .. legal choice count) and
# the Gukjin prompt flag.
CONTEXT_INDEX = [*range(200, 250), *range(396, 453), *range(471, 476), 552]

_BITS: dict[torch.device, torch.Tensor] = {}


def _bits(x: torch.Tensor, width: int = 50) -> torch.Tensor:
    key = x.device
    if key not in _BITS:
        _BITS[key] = torch.arange(50, device=x.device)
    return (x.unsqueeze(-1) >> _BITS[key][:width]) & 1


class History:
    """Per-game event logs for one batch of environments, kept on the GPU."""

    def __init__(self, ext: Any, states: torch.Tensor) -> None:
        games = states.shape[0]
        self.start = ext.start(states)
        self.events = torch.zeros((games, MAX_EVENTS, EVENT_WORDS), dtype=torch.int64, device=states.device)
        self.count = torch.zeros(games, dtype=torch.int64, device=states.device)

    def record(self, env_ids: torch.Tensor, committed: torch.Tensor, events: torch.Tensor) -> None:
        rows = torch.nonzero(committed).squeeze(1)
        games = env_ids.index_select(0, rows)
        pos = self.count.index_select(0, games).clamp_max(MAX_EVENTS - 1)
        self.events[games, pos] = events.index_select(0, rows)
        self.count.index_add_(0, games, torch.ones_like(games))

    def check(self) -> None:
        if int(self.count.max()) > MAX_EVENTS:
            raise RuntimeError(f"event log overflow: {int(self.count.max())} > {MAX_EVENTS}")

    def batch(self, env_ids: torch.Tensor, seats: torch.Tensor, counts: torch.Tensor | None = None) -> "HistBatch":
        n = self.count.index_select(0, env_ids) if counts is None else counts
        return HistBatch(self.events.index_select(0, env_ids), n, self.start.index_select(0, env_ids), seats)


@dataclass
class HistBatch:
    events: torch.Tensor  # [B, MAX_EVENTS, EVENT_WORDS] int64
    count: torch.Tensor  # [B] events visible at this decision
    start: torch.Tensor  # [B, START_WORDS] int64
    seat: torch.Tensor  # [B] observer seat, int64

    def select(self, rows: torch.Tensor) -> "HistBatch":
        return HistBatch(self.events.index_select(0, rows), self.count.index_select(0, rows),
                         self.start.index_select(0, rows), self.seat.index_select(0, rows))


def encode_events(h: HistBatch, length: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Observer-relative event tokens [B, length, EVENT_IN] and their validity mask."""
    ev = h.events[:, :length]
    seat = h.seat.view(-1, 1)
    header = ev[..., 8]
    me = (header & 0xFF) == seat
    kind = (header >> 8) & 0xFF
    first = seat == 0
    zero = torch.zeros_like(ev[..., 1])
    slots = (ev[..., 0], torch.where(me, ev[..., 1], zero), ev[..., 2],
             torch.where(first, ev[..., 3], ev[..., 4]), torch.where(first, ev[..., 4], ev[..., 3]),
             torch.where(first, ev[..., 5], ev[..., 6]), torch.where(first, ev[..., 6], ev[..., 5]), ev[..., 7])
    # Unpack one card mask at a time: an int64 [B, L, 8, 50] intermediate costs >1 GB.
    planes = [_bits(m).to(dtype) for m in slots]
    opp_drawn = torch.where(me, 0, _bits(ev[..., 1]).sum(-1))
    tokens = torch.cat((*planes, F.one_hot(kind.clamp_max(KINDS - 1), KINDS).to(dtype),
                        me.unsqueeze(-1).to(dtype), opp_drawn.unsqueeze(-1).to(dtype)), -1)
    valid = torch.arange(length, device=ev.device).view(1, -1) < h.count.view(-1, 1)
    return tokens, valid


def encode_start(h: HistBatch) -> torch.Tensor:
    st, seat = h.start, h.seat
    first = seat == 0
    hand = torch.where(first, st[:, 0], st[:, 1])
    cap_me = torch.where(first, st[:, 3], st[:, 4])
    cap_opp = torch.where(first, st[:, 4], st[:, 3])
    planes = _bits(torch.stack((hand, st[:, 2], cap_me, cap_opp, st[:, 5]), -1)).flatten(-2)
    neutral = _bits(st[:, 6], 12)
    header = st[:, 7]
    mtype = header & 0xFF
    scalars = torch.stack((
        ((header >> 8) & 0xFF).float() / 4.0,
        (((header >> 16) & 1) == seat).float(),
        torch.where(first, (header >> 24) & 0xFF, (header >> 32) & 0xFF).float() / 8.0,
        torch.where(first, (header >> 32) & 0xFF, (header >> 24) & 0xFF).float() / 8.0,
    ), -1)
    return torch.cat((planes.float(), neutral.float(), F.one_hot(mtype.clamp_max(7), 8).float(), scalars), -1)


class SeqNet(nn.Module):
    """Transformer over [opening, events..., query]; reads the query position."""

    uses_history = True

    def __init__(self, mode: str, outputs: int, d_model: int = 256, layers: int = 4, heads: int = 8,
                 oracle: bool = False) -> None:
        super().__init__()
        if mode not in ("hybrid", "events"):
            raise ValueError("mode must be hybrid or events")
        self.mode, self.oracle = mode, oracle
        query_in = (FEATURES if mode == "hybrid" else len(CONTEXT_INDEX)) + ACTIONS
        query_in += ORACLE_FEATURES if oracle else 0
        self.register_buffer("context_index", torch.tensor(CONTEXT_INDEX), persistent=False)
        self.event_in = nn.Linear(EVENT_IN, d_model)
        self.start_in = nn.Linear(START_IN, d_model)
        self.query_in = nn.Sequential(nn.Linear(query_in, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.position = nn.Parameter(torch.zeros(MAX_EVENTS + 1, d_model))
        self.token_type = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.token_type, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, outputs))

    def _dtype(self) -> torch.dtype:
        return torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else self.event_in.weight.dtype

    def _query(self, features: torch.Tensor, legal: torch.Tensor, opp_hand: torch.Tensor | None,
               dtype: torch.dtype) -> torch.Tensor:
        query = features if self.mode == "hybrid" else features.index_select(-1, self.context_index)
        parts = [query.to(dtype), legal.to(dtype)]
        if self.oracle:
            parts.append(card_plane(opp_hand.reshape(-1), dtype).view(*opp_hand.shape, -1))
        return self.query_in(torch.cat(parts, -1)) + self.token_type[2]

    def _prefix(self, hist: HistBatch, length: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        ev_tokens, valid = encode_events(hist, length, dtype)
        x = torch.cat((
            (self.start_in(encode_start(hist).to(dtype)) + self.position[0] + self.token_type[0]).unsqueeze(1),
            self.event_in(ev_tokens) + self.position[1:length + 1] + self.token_type[1],
        ), 1)
        return x, valid

    def _out(self, h: torch.Tensor) -> torch.Tensor:
        out = self.head(self.norm(h))
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def forward(self, features: torch.Tensor, legal: torch.Tensor, hist: HistBatch,
                opp_hand: torch.Tensor | None = None) -> torch.Tensor:
        """One decision per row: [opening, events seen so far, query].

        Causal attention plus key padding: event e sees the opening and events <= e,
        the query sees every valid event. forward_games computes the same function.
        """
        dtype = self._dtype()
        length = max(1, int(hist.count.max()))
        prefix, valid = self._prefix(hist, length, dtype)
        x = torch.cat((prefix, self._query(features, legal, opp_hand, dtype).unsqueeze(1)), 1)
        ignore = torch.cat((torch.zeros_like(valid[:, :1]), ~valid, torch.zeros_like(valid[:, :1])), 1)
        causal = torch.ones(length + 2, length + 2, dtype=torch.bool, device=x.device).triu(1)
        h = self.encoder(x, mask=causal, src_key_padding_mask=ignore)
        return self._out(h[:, -1])

    def forward_games(self, hist: HistBatch, features: torch.Tensor, legal: torch.Tensor, seen: torch.Tensor,
                      opp_hand: torch.Tensor | None = None) -> torch.Tensor:
        """Every decision of one (game, seat) at once: [opening, all events, queries...].

        hist.count is the game's full event count; features/legal/seen/opp_hand are
        [S, Q, ...] per decision, seen[s, q] = events visible at that decision. Each
        query sees the opening, events < seen and itself, so row q equals forward()
        on that decision alone. Returns [S, Q, outputs].
        """
        dtype = self._dtype()
        length = max(1, int(hist.count.max()))
        prefix, _ = self._prefix(hist, length, dtype)
        q = seen.shape[1]
        x = torch.cat((prefix, self._query(features, legal, opp_hand, dtype)), 1)
        t = length + 1 + q
        dev = x.device
        allowed = torch.zeros(seen.shape[0], t, t, dtype=torch.bool, device=dev)
        allowed[:, :, 0] = True
        allowed[:, 1:length + 1, 1:length + 1] = torch.ones(length, length, dtype=torch.bool, device=dev).tril()
        allowed[:, length + 1:, 1:length + 1] = torch.arange(length, device=dev).view(1, 1, -1) < seen.unsqueeze(-1)
        diag = torch.arange(length + 1, t, device=dev)
        allowed[:, diag, diag] = True
        heads = self.encoder.layers[0].self_attn.num_heads
        h = self.encoder(x, mask=(~allowed).repeat_interleave(heads, 0))
        return self._out(h[:, length + 1:])


def seq_actor(mode: str, d_model: int, layers: int, heads: int) -> SeqNet:
    return SeqNet(mode, ACTIONS, d_model, layers, heads)


def seq_critic(mode: str, d_model: int, layers: int, heads: int) -> SeqNet:
    return SeqNet(mode, 1, d_model, layers, heads, oracle=True)


def policy_logits(model: nn.Module, features: torch.Tensor, legal: torch.Tensor,
                  hist: HistBatch | None, chunk: int = 8192) -> torch.Tensor:
    """Logits from either kind of policy; sequence models run in row chunks."""
    if not getattr(model, "uses_history", False):
        out = model(features)
        return (out[0] if isinstance(out, tuple) else out).float()
    if hist is None:
        raise ValueError("sequence policy needs a history batch")
    n = features.shape[0]
    if n <= chunk:
        return model(features, legal, hist).float()
    outs = []
    for i in range(0, n, chunk):
        rows = torch.arange(i, min(n, i + chunk), device=features.device)
        outs.append(model(features[i:i + chunk], legal[i:i + chunk], hist.select(rows)).float())
    return torch.cat(outs)
