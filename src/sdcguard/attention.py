"""Checks for fused attention (torch.nn.functional.scaled_dot_product_attention), which the matmul checksums
can't see: its two matmuls and the softmax run inside one kernel.

Backward (AEGIS, OSDI '26, section 5.1.1): with P the attention probabilities, dV = P^T dO, and every row of P
sums to 1, so for every head and feature the sum of dV over the keys equals the sum of dO over the queries:
1^T dV = (P 1)^T dO = 1^T dO. With grouped-query attention a key/value head's dV collects the query heads of
its group, so the right side sums over the group too. The backward kernel recomputes P from Q, K and the
forward's log-sum-exp, so a fault there (or in the forward's softmax statistics) breaks the identity as well.
It doesn't cover dQ and dK.

Forward: every output row is a convex combination of the value rows (the probabilities are >= 0 and sum to
1), so each output lies between the smallest and largest value of its head and feature. An output outside
that range is impossible, whatever the inputs: a cheap test for large errors (exponent bits).

Both read tensors the kernel already wrote; nothing in the model changes. Not checked: attention with
dropout (the identities hold only without it), with an explicit mask or bias tensor, and variable-length
(nested) batches; these are counted as skipped.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from .checksum import ABS_FLOOR, HALF_TINY_STEP, UNIT_ROUNDOFF, CheckStat

# Noise of the backward identity, in units of u x (|column of dV|_2 + |column of dO over the group|_2): the
# kernel rounds P (and dV) to bf16/fp16 before and after its products, an error of up to u per value that
# averages out like a random walk. Set from CPU runs; to be calibrated on GPUs (experiments/attention_calib.py).
ATT_C = 1.0

BACKWARD_OPS = {
    # op name: (index of grad_out, query, key, value, dropout_p; how to find a mask/bias or nested batch)
    "aten::_scaled_dot_product_flash_attention_backward": dict(dropout=10, nested=(6, 7)),
    "aten::_scaled_dot_product_cudnn_attention_backward": dict(dropout=13, bias=8, nested=(9, 10)),
    "aten::_scaled_dot_product_efficient_attention_backward": dict(dropout=9, bias=4),
    "aten::_scaled_dot_product_flash_attention_for_cpu_backward": dict(dropout=6, kw_mask="attn_mask"),
}
FORWARD_OPS = {
    "aten::_scaled_dot_product_flash_attention": dict(dropout=3),
    "aten::_scaled_dot_product_cudnn_attention": dict(dropout=5, bias=3),
    "aten::_scaled_dot_product_efficient_attention": dict(dropout=5, bias=3),
    "aten::_scaled_dot_product_flash_attention_for_cpu": dict(dropout=3, kw_mask="attn_mask"),
}


def _arg(args, kwargs, i, name=None, default=None):
    if name is not None and name in kwargs:
        return kwargs[name]
    return args[i] if i is not None and i < len(args) else default


def _defined(t) -> bool:
    return isinstance(t, torch.Tensor) and t.numel() > 0


def why_not(name: str, args, kwargs) -> Optional[str]:
    """Why this attention call can't be checked (None: it can)."""
    spec = BACKWARD_OPS.get(name) or FORWARD_OPS.get(name)
    if spec is None:
        return "not an attention op"
    p = _arg(args, kwargs, spec["dropout"], "dropout_p", 0.0)
    if p:
        return "dropout"
    if "bias" in spec and _defined(_arg(args, kwargs, spec["bias"])):
        return "mask or bias tensor"
    if "kw_mask" in spec and _defined(kwargs.get(spec["kw_mask"])):
        return "mask or bias tensor"
    if "nested" in spec and any(_defined(_arg(args, kwargs, i)) for i in spec["nested"]):
        return "variable-length batch"
    return None


def _heads(q_like: torch.Tensor, kv_like: torch.Tensor):
    if q_like.dim() != 4 or kv_like.dim() != 4:
        return None
    b, hq, _, d = q_like.shape
    hk = kv_like.shape[1]
    if hk == 0 or hq % hk or kv_like.shape[0] != b or kv_like.shape[-1] != d:
        return None
    return b, hq, hk, d


def _columns(x: torch.Tensor):
    """Over the sequence axis (-2), in fp32: the sum, the sum of |x|, the 2-norm and the largest |x|."""
    return (x.sum(-2, dtype=torch.float32), torch.linalg.vector_norm(x, 1, dim=-2, dtype=torch.float32),
            torch.linalg.vector_norm(x, 2, dim=-2, dtype=torch.float32), x.abs().amax(-2).float())


def backward_stat(dO: torch.Tensor, dV: torch.Tensor, z: float = 6.0,
                  abs_floor: float = ABS_FLOOR) -> Optional[CheckStat]:
    """The backward identity for every (batch, key/value head, feature). dO: (B, Hq, Tq, D), dV: (B, Hk, Tk, D)
    in bf16/fp16. Returns a queued check like checksum.matmul_zscores (rows: batch x head, blocks: features),
    or None when it can't be judged here (other dtypes or shapes)."""
    if not (isinstance(dO, torch.Tensor) and isinstance(dV, torch.Tensor)):     # (dV not asked for)
        return None
    if dV.dtype not in UNIT_ROUNDOFF or dO.dtype != dV.dtype:
        return None
    hd = _heads(dO, dV)
    if hd is None:
        return None
    b, hq, hk, d = hd
    g = hq // hk
    u = UNIT_ROUNDOFF[dV.dtype]
    tiny = HALF_TINY_STEP[dV.dtype]
    with torch.no_grad():
        # fp32 sums over the sequence straight from the bf16/fp16 tensors (no fp32 copy of them)
        rhs, o_l1, o_l2, _ = _columns(dO)
        rhs = rhs.view(b, hk, g, d).sum(2)                                     # (B, Hk, D)
        o_l1 = o_l1.view(b, hk, g, d).sum(2)
        o_l2 = o_l2.view(b, hk, g, d).pow(2).sum(2).sqrt()
        lhs, v_l1, v_l2, v_max = _columns(dV)
        tq, tk = dO.shape[-2] * g, dV.shape[-2]
        # our own fp32 sums: at most log2(T) roundings of 2^-24 each, on the absolute sums
        own = 2.0**-24 * (math.log2(tq + 1) * o_l1 + math.log2(tk + 1) * v_l1)
        # fp16 values below its normal range round on a fixed grid, half a step each (checksum.py explains)
        noise = ATT_C * u * (v_l2 + o_l2) + own + math.sqrt(tq + tk) * tiny + abs_floor
        diff = (lhs - rhs).abs()
        zz = diff / noise
        # The random-walk estimate fails when the roundings line up: with near-uniform attention every dV of a
        # column is nearly the same number and rounds the same way. As in the matmul checks, an alarm also has
        # to beat the worst case: up to u per rounded dV, and up to u per row of rounded P (its values sum to 1).
        worst = 1.25 * (u * (v_l1 + o_l1) + (tq + tk) * tiny) + own + abs_floor
        zz = torch.minimum(zz, z * diff / worst)
        # Inf/NaN: a dV the finite dO can't produce is corruption (|dV| <= the column's |dO| sum); a dO that
        # isn't finite can't be judged
        o_bad = ~torch.isfinite(o_l1)
        v_bad = ~torch.isfinite(v_l1)
        too_big = v_max > o_l1 * (1.0 + 2 * u) + tk * tiny + abs_floor     # larger than all of dO allows
        zz = torch.where(o_bad, torch.full_like(zz, -1.0), zz)
        zz = torch.where(~o_bad & (v_bad | too_big), torch.full_like(zz, float("inf")), zz)
        zz = torch.nan_to_num(zz, nan=-1.0)
        flat = zz.reshape(-1)
        mz, idx = flat.max(0)
        nz = noise.reshape(-1)[idx]
        # as in the matmul checks: a column that couldn't be judged makes the check incomplete unless another
        # column found corruption
        mz = torch.where((flat < 0).any() & (mz <= z), torch.full_like(mz, -1.0), mz)
    return CheckStat(max_z=mz, argmax=idx * 2, noise=nz, spans=((0, 1, d, 0),), m=b * hk, batched=False)


def forward_stat(O: torch.Tensor, V: torch.Tensor, abs_floor: float = ABS_FLOOR) -> Optional[CheckStat]:
    """Outputs outside the range of their head's values: impossible, so z is inf there (0 elsewhere).
    O: (B, Hq, Tq, D), V: (B, Hk, Tk, D)."""
    if not (isinstance(O, torch.Tensor) and isinstance(V, torch.Tensor)):
        return None
    if O.dtype not in UNIT_ROUNDOFF or V.dtype != O.dtype:
        return None
    hd = _heads(O, V)
    if hd is None or V.shape[-2] == 0:
        return None
    b, hq, hk, d = hd
    g = hq // hk
    u = UNIT_ROUNDOFF[O.dtype]
    with torch.no_grad():
        vmax = V.amax(-2).float().repeat_interleave(g, dim=1).unsqueeze(-2)     # (B, Hq, 1, D)
        vmin = V.amin(-2).float().repeat_interleave(g, dim=1).unsqueeze(-2)
        v_ok = torch.isfinite(vmax) & torch.isfinite(vmin)
        tol = 2 * u * torch.maximum(vmax.abs(), vmin.abs()) + abs_floor
        # compared in fp32 element by element, without an fp32 copy of the outputs
        bad = (O > vmax + tol) | (O < vmin - tol) | ~torch.isfinite(O)
        bad = (bad & v_ok).reshape(-1)
        idx = torch.argmax(bad.to(torch.uint8))
        hit = bad[idx]
        inf, zero, unjudged = (torch.tensor(x, device=O.device) for x in (float("inf"), 0.0, -1.0))
        mz = torch.where(hit, inf, torch.where(v_ok.any(), zero, unjudged))       # (no wait for the GPU)
        row_d = idx % d
        nz = tol.expand(b, hq, 1, d).reshape(b * hq, d)[(idx // (O.shape[-2] * d)), row_d]
    return CheckStat(max_z=mz, argmax=idx * 2, noise=nz, spans=((0, 1, d, 0),), m=b * hq * O.shape[-2],
                     batched=False)
