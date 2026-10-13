"""Single-query attention over a KV cache (one decode step per row).

Each cache row holds `slots` keys/values per head ([rows, heads, slots, dh]); query i
reads cache row rows[i] (default: row i) and attends slots 0..pos[i]. The Triton
kernel runs one program per (query, head), reads only the slots it attends and does
the softmax in registers. Without Triton the same result comes from two batched
matrix products.
"""

from __future__ import annotations

import torch

try:  # Triton is optional (triton-windows on Windows)
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the environment
    triton = None


if triton is not None:
    @triton.jit
    def _decode_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, rows_ptr, out_ptr, heads, slots, scale,
                       DH: tl.constexpr, BLOCK_T: tl.constexpr):
        pid = tl.program_id(0)  # query * heads + head
        query = pid // heads
        pos = tl.load(pos_ptr + query)
        row = tl.load(rows_ptr + query)
        d = tl.arange(0, DH)
        t = tl.arange(0, BLOCK_T)
        keep = t <= pos
        base = (row * heads + pid % heads).to(tl.int64) * slots * DH
        q = tl.load(q_ptr + pid.to(tl.int64) * DH + d).to(tl.float32)
        k = tl.load(k_ptr + base + t[:, None] * DH + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        s = tl.where(keep, tl.sum(k * q[None, :], 1) * scale, float("-inf"))
        p = tl.exp(s - tl.max(s, 0))
        p = p / tl.sum(p, 0)
        v = tl.load(v_ptr + base + t[:, None] * DH + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        o = tl.sum(v * p[:, None], 0)
        tl.store(out_ptr + pid.to(tl.int64) * DH + d, o.to(out_ptr.dtype.element_ty))


def decode_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, pos: torch.Tensor,
                     use_triton: bool = True, rows: torch.Tensor | None = None) -> torch.Tensor:
    """q [n, heads, dh]; keys/values [rows, heads, slots, dh] (contiguous); pos [n] and rows [n]
    int64 (rows defaults to 0..n-1) -> [n, heads, dh]."""
    n, heads, dh = q.shape
    slots = keys.shape[2]
    scale = dh ** -0.5
    if rows is None:
        rows = torch.arange(n, device=q.device)
    if use_triton and triton is not None and q.is_cuda:
        q = q.contiguous()
        out = torch.empty_like(q)
        _decode_kernel[(n * heads,)](q, keys, values, pos, rows, out, heads, slots, scale,
                                     DH=dh, BLOCK_T=triton.next_power_of_2(slots))
        return out
    keys, values = keys.index_select(0, rows), values.index_select(0, rows)
    mask = torch.arange(slots, device=q.device).view(1, 1, 1, slots) <= pos.view(n, 1, 1, 1)
    scores = torch.matmul(q.unsqueeze(2), keys.transpose(-1, -2).to(q.dtype)).float() * scale
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1).to(q.dtype)
    return torch.matmul(probs, values.to(q.dtype)).squeeze(2)
