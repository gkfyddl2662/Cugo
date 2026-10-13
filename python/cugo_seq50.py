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
from cugo_seq_attention50 import decode_attention

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
        # Events-mode callers may pass the 720 features or only the stored CONTEXT_INDEX columns.
        if self.mode == "events" and features.shape[-1] == FEATURES:
            features = features.index_select(-1, self.context_index)
        query = features
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

    def _encode(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """Pre-norm encoder stack over [B, T, d]; allowed [B, T, T] is True where a query may attend.

        Same parameters and math as the nn.TransformerEncoder that holds them, but the
        mask broadcasts over heads instead of being copied per head.
        """
        b, t, d = x.shape
        heads = self.encoder.layers[0].self_attn.num_heads
        mask = allowed.unsqueeze(1)
        for layer in self.encoder.layers:
            attn = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
            q, k, v = (z.reshape(b, t, heads, d // heads).transpose(1, 2) for z in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + attn.out_proj(a.transpose(1, 2).reshape(b, t, d))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return x

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
        key_ok = torch.cat((torch.ones_like(valid[:, :1]), valid, torch.ones_like(valid[:, :1])), 1)
        causal = torch.ones(length + 2, length + 2, dtype=torch.bool, device=x.device).tril()
        h = self._encode(x, causal.unsqueeze(0) & key_ok.unsqueeze(1))
        return self._out(h[:, -1])

    def forward_games(self, hist: HistBatch, features: torch.Tensor, legal: torch.Tensor, seen: torch.Tensor,
                      opp_hand: torch.Tensor | None = None, length: int | None = None) -> torch.Tensor:
        """Every decision of one (game, seat) at once: [opening, all events, queries...].

        hist.count is the game's full event count; features/legal/seen/opp_hand are
        [S, Q, ...] per decision, seen[s, q] = events visible at that decision. Each
        query sees the opening, events < seen and itself, so row q equals forward()
        on that decision alone. Returns [S, Q, outputs]. Passing length (>= the
        largest event count) avoids reading it back from the GPU.
        """
        dtype = self._dtype()
        length = max(1, int(hist.count.max())) if length is None else max(1, length)
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
        h = self._encode(x, allowed)
        return self._out(h[:, length + 1:])

    # Incremental inference. Opening and event tokens only attend backwards, so their
    # keys/values never change once computed: self-play caches them per (game, seat)
    # and runs one new token per event and one query token per decision.

    def start_token(self, start: torch.Tensor, seat: torch.Tensor) -> torch.Tensor:
        dtype = self._dtype()
        hb = HistBatch(start.new_zeros((start.shape[0], 1, EVENT_WORDS)), torch.zeros_like(seat), start, seat)
        return self.start_in(encode_start(hb).to(dtype)) + self.position[0] + self.token_type[0]

    def event_token(self, words: torch.Tensor, seat: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Token of one event per row; index = events before it (its position is 1 + index)."""
        dtype = self._dtype()
        hb = HistBatch(words.unsqueeze(1), torch.ones_like(seat), words.new_zeros((words.shape[0], START_WORDS)), seat)
        tokens, _ = encode_events(hb, 1, dtype)
        position = self.position.index_select(0, 1 + index.clamp_max(MAX_EVENTS - 1))
        return self.event_in(tokens[:, 0]) + position + self.token_type[1]

    def query_token(self, features: torch.Tensor, legal: torch.Tensor,
                    opp_hand: torch.Tensor | None = None) -> torch.Tensor:
        return self._query(features, legal, opp_hand, self._dtype())

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        return self._out(h)

    def cached_step(self, cache: "KVCache", x: torch.Tensor, slots: int | None = None) -> torch.Tensor:
        """Run one token per cache row (x [rows, d]); its key/value go to slot cache.length[row].

        Each token attends slots <= its own: everything cached plus itself. The caller
        advances cache.length for tokens that stay (events); a query does not, so the
        next event overwrites its slot. `slots` bounds the slots read (it must exceed
        every row's length). Returns the final hidden state [rows, d].

        Attention is a single-query decode kernel (cugo_seq_attention50); fused
        training kernels tile queries by 64 and waste most of their work here.
        """
        n, d = x.shape
        heads = cache.heads
        dh = d // heads
        t = cache.slots if slots is None else min(slots, cache.slots)
        pos = cache.length.clamp_max(t - 1)
        flat_rows = (cache.row_ids.view(n, 1) * heads + cache.head_ids.view(1, heads)).flatten()
        flat_pos = pos.repeat_interleave(heads)
        for i, layer in enumerate(self.encoder.layers):
            attn = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), attn.in_proj_weight, attn.in_proj_bias).chunk(3, -1)
            kc, vc = cache.k[i], cache.v[i]  # [rows, heads, slots, dh]
            kc.view(-1, cache.slots, dh).index_put_((flat_rows, flat_pos), k.reshape(n * heads, dh).to(kc.dtype))
            vc.view(-1, cache.slots, dh).index_put_((flat_rows, flat_pos), v.reshape(n * heads, dh).to(vc.dtype))
            a = decode_attention(q.reshape(n, heads, dh), kc, vc, pos, cache.use_triton)
            x = x + attn.out_proj(a.reshape(n, d))
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        return x


class KVCache:
    """Per-row keys/values of the opening and event tokens for every layer (bf16, head-major)."""

    def __init__(self, model: SeqNet, rows: int, device: torch.device) -> None:
        layers, d = len(model.encoder.layers), model.position.shape[1]
        self.heads = model.encoder.layers[0].self_attn.num_heads
        self.slots = MAX_EVENTS + 2  # opening + events + one scratch slot for the query
        self.k = torch.zeros((layers, rows, self.heads, self.slots, d // self.heads), dtype=torch.bfloat16,
                             device=device)
        self.v = torch.zeros_like(self.k)
        self.length = torch.zeros(rows, dtype=torch.int64, device=device)
        self.slot_ids = torch.arange(self.slots, device=device)
        self.row_ids = torch.arange(rows, device=device)
        self.head_ids = torch.arange(self.heads, device=device)
        self.use_triton = True


def stored_columns(mode: str) -> list[int] | None:
    """Feature columns a rollout must keep for this mode (None = all 720)."""
    return None if mode == "hybrid" else CONTEXT_INDEX


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
    if n == 0:  # e.g. every remaining game has the pool opponent to move
        return features.new_zeros((0, legal.shape[1]), dtype=torch.float32)
    if n <= chunk:
        return model(features, legal, hist).float()
    outs = []
    for i in range(0, n, chunk):
        rows = torch.arange(i, min(n, i + chunk), device=features.device)
        outs.append(model(features[i:i + chunk], legal[i:i + chunk], hist.select(rows)).float())
    return torch.cat(outs)
