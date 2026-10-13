"""Single-query attention over a KV cache (one decode step per row).

Each cache row holds `slots` keys/values per head ([rows, heads, slots, dh]); row r
attends slots 0..pos[r]. The Triton kernel runs one program per (row, head), reads
only the slots it attends and does the softmax in registers. Without Triton the
same result comes from two batched matrix products.
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
    def _decode_kernel(q_ptr, k_ptr, v_ptr, pos_ptr, out_ptr, heads, slots, scale,
                       DH: tl.constexpr, BLOCK_T: tl.constexpr):
        pid = tl.program_id(0)  # row * heads + head
        pos = tl.load(pos_ptr + pid // heads)
        d = tl.arange(0, DH)
        t = tl.arange(0, BLOCK_T)
        keep = t <= pos
        base = pid.to(tl.int64) * slots * DH
        q = tl.load(q_ptr + pid.to(tl.int64) * DH + d).to(tl.float32)
        k = tl.load(k_ptr + base + t[:, None] * DH + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        s = tl.where(keep, tl.sum(k * q[None, :], 1) * scale, float("-inf"))
        p = tl.exp(s - tl.max(s, 0))
        p = p / tl.sum(p, 0)
        v = tl.load(v_ptr + base + t[:, None] * DH + d[None, :], mask=keep[:, None], other=0.0).to(tl.float32)
        o = tl.sum(v * p[:, None], 0)
        tl.store(out_ptr + pid.to(tl.int64) * DH + d, o.to(out_ptr.dtype.element_ty))


def decode_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, pos: torch.Tensor,
                     use_triton: bool = True) -> torch.Tensor:
    """q [n, heads, dh]; keys/values [n, heads, slots, dh] (contiguous); pos [n] int64 -> [n, heads, dh]."""
    n, heads, slots, dh = keys.shape
    scale = dh ** -0.5
    if use_triton and triton is not None and q.is_cuda:
        q = q.contiguous()
        out = torch.empty_like(q)
        _decode_kernel[(n * heads,)](q, keys, values, pos, out, heads, slots, scale,
                                     DH=dh, BLOCK_T=triton.next_power_of_2(slots))
        return out
    mask = torch.arange(slots, device=q.device).view(1, 1, 1, slots) <= pos.view(n, 1, 1, 1)
    scores = torch.matmul(q.unsqueeze(2), keys.transpose(-1, -2).to(q.dtype)).float() * scale
    probs = torch.softmax(scores.masked_fill(~mask, float("-inf")), -1).to(q.dtype)
    return torch.matmul(probs, values.to(q.dtype)).squeeze(2)
