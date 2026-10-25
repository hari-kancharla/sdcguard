"""Checks of fused MoE experts in serving: a sampled exact recompute of the routed experts' output.

A fused MoE layer computes, for each token x and its top-k experts e_k with routing weights g_k,
y = sum_k g_k W2_e (act(W1_e x) * (W3_e x)), all experts at once in one kernel, so no single matmul can be checked
by a checksum from outside. Instead, at each model step one MoE layer (they take turns: the layer whose index is the
step number modulo the number of MoE layers) recomputes the output of SAMPLES rows of its batch from the layer's
own input, routing and weights, and compares it with the kernel's output. The rows are drawn from the step's real
rows when the plugin knows them (mark_step's rows, packed above the step number in the layer's step buffer), and a
row whose top-k experts are all -1 isn't judged: vLLM 0.32 (VLLM_MOE_SKIP_PADDING, on by default) gives the padding
rows of a CUDA graph's batch no experts, so no matmul computes them and their output is whatever moe_sum adds up
from the kernel's workspace.

The routing's validity: the active layer also reads the expert ids of every one of the step's real rows (when the
plugin knows them; padding rows come after them). vLLM's routers give every real token top_k experts, so a real row
with all its experts -1 (no expert computes it: its output is leftover workspace) or some of them -1 (some of its
experts are missing) is wrong whatever the numbers: a router fault, a corrupted id, or vLLM's padding mask marking a
real row. It raises an alert (finding real_rows_unrouted or real_rows_partly_routed, recheck output_wrong) naming
the layer, how many such rows the read saw, and the last of them.

The recompute follows the rounding of vLLM's Triton MoE kernel (TritonExperts, vLLM 0.31/0.32): the first matmul in
fp32 rounded to bf16/fp16, the activation rounded before the multiply and the product rounded (vLLM's act_and_mul
kernel), the second matmul in fp32 times the routing weight then rounded, and the experts summed in fp32 and
rounded once (moe_sum). The two sides then differ only where a different order of fp32 additions moves a value
across a rounding boundary (on the CPU against such an order, 99.97% of bf16 outputs came out identical). An
output's score is its distance from the recompute in units of
  u |r| + OUT_FLIPS 2u max_k |c_k| + KAPPA u sqrt(sum_k g_k^2 sum_i (W2_ni h_i)^2) + FLIPS max_k,i |g_k W2_ni| s_i
(u: the dtype's unit roundoff; c_k: each expert's rounded output, of which OUT_FLIPS may each be one rounding step
off; the third term is what intermediates rounded differently everywhere would move it by, the fourth what FLIPS of
them flipped at the worst places would, s_i the size of a flip of h_i or of its inputs). The alarm line is 6. A
wrong output is seen when it is off by more than about 6 times that: wrong experts, wrong routing weights, stale or
garbage rows, non-finite values and large bit errors, not small ones. Rows of other steps and other layers aren't
seen.

What it sees is the kernel's arithmetic and indexing: a wrong expert, another row's output, a wrong scaling,
non-finite values. Not a weight changed in memory: the kernel and the recompute read the same expert weights, so
they agree on a corrupted one (an alert raised for another reason says whether the weights changed since load, from
per-expert sums kept then). Not covered either: MoE layers on other kernels (FlashInfer, CUTLASS, DeepGEMM, Marlin,
the monolithic ones that route inside the kernel), quantized experts, experts with biases, activations other than
SiLU and GELU (gated), the router itself (the top-k choice is taken as given), outputs whose finalize is deferred,
and layers whose kernel combines across GPUs: expert parallelism's all-to-all, passthrough experts, and data
parallelism (vLLM's all-gather/reduce-scatter), whose output holds other GPUs' experts or weight shards, which the
recompute can't read. Each run's log lists what isn't checked.
"""
from __future__ import annotations

import math
import os
import struct
import threading
from typing import Optional

import torch

from .events import SDCEvent, device_identity
from .serve_norm import PLACE_KEPT, PLACE_MAXIMA, dtype_consts, pick

MOE_Z = 6.0
SAMPLES = 2              # rows recomputed per active call
KAPPA = 1.0              # weight of the "every intermediate rounded differently" term
FLIPS = 1.0              # worst-placed rounding flips of intermediates the bound allows (KAPPA covers many small)
OUT_FLIPS = 2.0          # how many experts' rounded outputs may each be one step off at one output
MOE_SCHEMA = 1
COLS = 11                # z|row, z|col, z|bits(y), z|bits(r), z|step, active calls, rows checked, unjudged, real
#                          rows with no experts and with some experts -1 (of all the step's known real rows), and
#                          (step << 32) | (the last such row + 1)
C_ROW, C_COL, C_OUT, C_REF, C_STEP, C_CALLS, C_ROWS, C_UNJ, C_UNROUTED, C_PARTIAL, C_BADROW = range(COLS)
ROUTE_BLOCK = 64         # real rows read at a time by the routing-validity scan
CAP_COLS = 4             # claimed, row, step, score bits
M32 = 0xFFFFFFFF
ROW_A, ROW_B = 7919, 104729       # row j at a layer's k-th turn (k = step // layers): (k ROW_A + j ROW_B) % rows
ROWS_SHIFT = 32                   # the step buffer: the step's real rows (0: unknown) << ROWS_SHIFT | step
ROWS_MAX = (1 << 31) - 1
ACT = {"silu": 0, "gelu": 1}

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}
FALLBACKS = [0]
OUT_DTYPES: dict = {}


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def _act(a: torch.Tensor, code: int) -> torch.Tensor:
    if code == ACT["silu"]:
        return a / (1.0 + torch.exp(-a))
    return 0.5 * a * (1.0 + torch.erf(a * 0.7071067811865476))


def _dact(a: torch.Tensor, code: int) -> torch.Tensor:
    if code == ACT["silu"]:
        s = torch.sigmoid(a)
        return s * (1.0 + a * (1.0 - s))
    return 0.5 * (1.0 + torch.erf(a * 0.7071067811865476)) + a * torch.exp(-0.5 * a * a) * 0.3989422804014327


def sample_rows(step: torch.Tensor, m, device, nlayers: int = 1) -> torch.Tensor:
    """The rows a layer's check takes at a step, of its first m rows (an int, or a tensor for the step's real rows):
    from the layer's own turn count, so a steady batch has every row taken in turn (from the step number itself,
    layers that take turns in a number of steps sharing a factor with the batch's rows would only ever see a few of
    them)."""
    j = torch.arange(SAMPLES, device=device, dtype=torch.int64)
    mm = m.clamp(min=1) if isinstance(m, torch.Tensor) else max(int(m), 1)
    return (torch.div(step, max(int(nlayers), 1), rounding_mode="floor") * ROW_A + j * ROW_B) % mm


def rows_of(raw: torch.Tensor, m: int) -> torch.Tensor:
    """The rows a step's samples are drawn from: its real rows packed in the step buffer when known (and at most the
    call's rows), else all m."""
    real = (raw >> ROWS_SHIFT) & ROWS_MAX
    return torch.where(real > 0, real.clamp(max=m), torch.full_like(real, m))


def pack_step(step: int, rows: Optional[int] = None) -> int:
    """The step buffer's value: the step number (low 32 bits) and, above it, the step's real rows (0: unknown)."""
    r = int(rows) if rows is not None and int(rows) > 0 else 0
    return (min(r, ROWS_MAX) << ROWS_SHIFT) | (int(step) & M32)


def recompute(x_rows: torch.Tensor, ids: torch.Tensor, g: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
              expert_map: Optional[torch.Tensor], act: int, routed_first: bool, dtype: torch.dtype,
              chunk: int = 512) -> tuple:
    """The routed experts' output for some rows, rounded as vLLM's Triton MoE kernel rounds, and the terms of the
    bound: (r, max_k |c_k|, sum_k g_k^2 sum_i (W2_ni h_i)^2, max_k,i |g_k W2_ni| s_i), each [rows, H] in fp32."""
    u = dtype_consts(dtype)[0]
    s, k = ids.shape
    e = ids.long()
    if expert_map is not None:
        e = torch.where(e >= 0, expert_map.long()[e.clamp(min=0)], e)
    valid = (e >= 0).reshape(-1)
    ec = e.clamp(min=0).reshape(-1)
    gk = g.float().reshape(-1)
    xr = x_rows.float().repeat_interleave(k, 0)
    a = torch.bmm(w13[ec].float(), xr.unsqueeze(-1)).squeeze(-1)           # [s*k, 2I] fp32
    if routed_first:
        a = a * gk[:, None]
    a = a.to(dtype).float()
    half = a.shape[1] // 2
    a1, a3 = a[:, :half], a[:, half:]
    s1 = _act(a1, act).to(dtype).float()
    h = (s1 * a3).to(dtype).float()
    sens = 2 * u * (h.abs() + (_dact(a1, act) * a1 * a3).abs() + (s1 * a3).abs())
    w2e = w2[ec]                                                             # [s*k, H, I]
    c = torch.zeros(s * k, w2.shape[1], device=x_rows.device)
    sq = torch.zeros_like(c)
    mx = torch.zeros_like(c)
    for i0 in range(0, half, chunk):                                         # [s*k, H, chunk] at a time
        wc = w2e[:, :, i0:i0 + chunk].float()
        p = wc * h[:, None, i0:i0 + chunk]
        c += p.sum(-1)
        sq += p.square().sum(-1)
        mx = torch.maximum(mx, (wc.abs() * sens[:, None, i0:i0 + chunk]).amax(-1))
    gw = torch.ones_like(gk) if routed_first else gk
    chat = (c * gw[:, None]).to(dtype).float()
    sq = sq * gw[:, None].square()
    mx = mx * gw[:, None].abs()
    keep = valid[:, None]
    zero = torch.zeros_like(chat)
    chat, sq, mx = torch.where(keep, chat, zero), torch.where(keep, sq, zero), torch.where(keep, mx, zero)
    shape = (s, k, -1)
    return (chat.view(shape).sum(1), chat.abs().view(shape).amax(1), sq.view(shape).sum(1),
            mx.view(shape).amax(1))


def scores(y: torch.Tensor, r: torch.Tensor, cmax: torch.Tensor, sq: torch.Tensor, mx: torch.Tensor,
           dtype: torch.dtype) -> tuple:
    """(z, judged): z per output; judged False where the recompute isn't finite or beyond the dtype's range."""
    u, sp, fm = dtype_consts(dtype)
    den = u * r.abs() + OUT_FLIPS * 2 * u * cmax + KAPPA * u * sq.sqrt() + FLIPS * mx + sp
    z = (y - r).abs() / den
    judged = (r.abs() <= fm) & torch.isfinite(den)
    z = torch.where(judged, torch.nan_to_num(z, nan=float("inf"), posinf=float("inf")), torch.zeros_like(z))
    return z, judged


def moe_check_torch(x: torch.Tensor, out: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor,
                    w13: torch.Tensor, w2: torch.Tensor, expert_map: Optional[torch.Tensor], act: int,
                    routed_first: bool, state: torch.Tensor, slot: int, stepbuf: torch.Tensor, nlayers: int,
                    index: int, cap_meta: torch.Tensor, cap_x: torch.Tensor, cap_y: torch.Tensor,
                    cap_ids: torch.Tensor, cap_g: torch.Tensor, line: float) -> None:
    """The check in PyTorch operations (no host sync: safe in CUDA graphs). It always computes; only the active
    layer's result counts. x, out: [rows, H]; topk_w, topk_ids: [rows, k]."""
    m, hdim = x.shape[0], x.shape[-1]
    if m == 0:
        return
    raw = stepbuf.reshape(-1)[0]
    step = raw & M32
    active = (step % max(nlayers, 1)) == index
    rows = sample_rows(step, rows_of(raw, m), x.device, nlayers)
    xr = x.reshape(m, hdim)[rows]
    y = out.reshape(m, hdim)[rows].float()
    ids, g = topk_ids.reshape(m, -1)[rows], topk_w.reshape(m, -1)[rows]
    pad = (ids == -1).all(-1)                                       # vLLM's padding rows: no experts, nothing computed
    r, cmax, sq, mx = recompute(xr, ids, g, w13, w2, expert_map, act, routed_first, out.dtype)
    z, judged = scores(y, r, cmax, sq, mx, out.dtype)
    judged = judged | pad.unsqueeze(-1)                             # (not judged, and not counted as unjudged)
    z = torch.where(active & ~pad.unsqueeze(-1), z, torch.zeros_like(z))
    known = ((raw >> ROWS_SHIFT) & ROWS_MAX) > 0                      # the step's real rows were given
    zmax, arg = z.reshape(-1).max(0)
    jrow, col = arg // hdim, arg % hdim
    row = pick(rows, jrow)
    yv, rv = pick(y.reshape(-1), arg), pick(r.reshape(-1), arg)
    bits = lambda v: v.reshape(1).view(torch.int32).to(torch.int64).reshape(()) & M32  # noqa: E731
    zb = bits(zmax) << 32
    keys = torch.stack([zb | row, zb | col, zb | bits(yv), zb | bits(rv), zb | step])
    keys = torch.where(active & (zmax > 0), keys, torch.zeros_like(keys))   # (nothing judged: the state as it was)
    srow = state[slot]
    srow[:5] = torch.maximum(srow[:5], keys)
    act64 = active.to(torch.int64)
    srow[C_CALLS] += act64
    srow[C_ROWS] += act64 * (~pad).sum()
    srow[C_UNJ] += act64 * (~judged).sum()
    # the routing's validity over every known real row: none may have experts -1
    allid = topk_ids.reshape(m, -1)
    real = torch.where(known, torch.clamp(((raw >> ROWS_SHIFT) & ROWS_MAX), max=m), torch.zeros_like(raw))
    isreal = torch.arange(m, device=x.device) < real
    nneg = (allid == -1).sum(-1)
    none_ = isreal & (nneg >= allid.shape[1])
    some = isreal & (nneg > 0) & (nneg < allid.shape[1])
    srow[C_UNROUTED] += act64 * none_.sum()
    srow[C_PARTIAL] += act64 * some.sum()
    last = torch.where(none_ | some, torch.arange(1, m + 1, device=x.device), torch.zeros_like(nneg)).max()
    srow[C_BADROW] = torch.maximum(srow[C_BADROW], torch.where(active & (last > 0), (step << 32) | last,
                                                                torch.zeros_like(raw)))
    take = (zmax > line) & (cap_meta[slot, 0] == 0)
    cap_x[slot, :hdim] = torch.where(take, pick(xr, jrow).float(), cap_x[slot, :hdim])
    cap_y[slot, :hdim] = torch.where(take, pick(y, jrow), cap_y[slot, :hdim])
    kk = ids.shape[1]
    cap_ids[slot, :kk] = torch.where(take, pick(ids, jrow).to(cap_ids.dtype), cap_ids[slot, :kk])
    cap_g[slot, :kk] = torch.where(take, pick(g, jrow).float(), cap_g[slot, :kk])
    rec = torch.stack([torch.ones_like(row), row, step, zb >> 32]).to(cap_meta.dtype)
    cap_meta[slot] = torch.where(take, rec, cap_meta[slot])


# ---------------------------------------------------------------------------------------------- Triton kernels
# Three launches per call, each one returning at once unless its layer is the active one this step: up (the first
# matmul for each sampled row and expert slot, rounded), down (activation, rounded, then the second matmul with the
# terms of the bound, times the routing weight, rounded) and compare (the experts summed, scored against the layer's
# output, the worst into the state, the first row over the line kept).
if HAVE_TRITON:
    @triton.jit
    def _rnd(v, OUT_BF16: tl.constexpr):
        """fp32 rounded to bf16 (to nearest, ties to even, by its bits: the same on every backend; Triton's CPU
        interpreter truncates in .to(bfloat16)) or to fp16, back as fp32."""
        if OUT_BF16:
            b = v.to(tl.int32, bitcast=True)
            b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536
            return tl.where(v == v, b.to(tl.float32, bitcast=True), v)
        else:
            return v.to(tl.float16).to(tl.float32)

    @triton.jit
    def _moe_live(stepbuf, nl, idx, M):
        """(step, live, the rows samples are drawn from: the step's real rows when packed above the step, else M)."""
        raw = tl.load(stepbuf)
        step = raw & 0xFFFFFFFF
        real = (raw >> 32) & 0x7FFFFFFF
        mr = tl.where(real > 0, tl.minimum(real, M), M)
        return step, (step % nl) == idx, mr

    @triton.jit
    def _moe_row(step, j, M, nl):
        return (((step // nl) * 7919 + j * 104729) % M).to(tl.int64)

    @triton.jit
    def _moe_expert(ids, emap, row, k, K, HAS_MAP: tl.constexpr):
        e = tl.load(ids + row * K + k).to(tl.int64)
        if HAS_MAP:
            e = tl.where(e >= 0, tl.load(emap + tl.maximum(e, 0)).to(tl.int64), e)
        return e

    @triton.jit
    def _moe_up(x, ids, gw, w13, emap, A, stepbuf, M, H, N2, K, sx, s13e, s13n, nl, idx,
                HAS_MAP: tl.constexpr, ROUTED_FIRST: tl.constexpr, OUT_BF16: tl.constexpr,
                BN: tl.constexpr, BH: tl.constexpr):
        jk = tl.program_id(0)
        step, live, mr = _moe_live(stepbuf, nl, idx, M)
        if live:
            j, k = jk // K, jk % K
            row = _moe_row(step, j, mr, nl)
            e = _moe_expert(ids, emap, row, k, K, HAS_MAP)
            n = tl.program_id(1) * BN + tl.arange(0, BN)
            nm = n < N2
            acc = tl.zeros([BN], dtype=tl.float32)
            if e >= 0:
                for h0 in range(0, H, BH):
                    hh = h0 + tl.arange(0, BH)
                    hm = hh < H
                    xv = tl.load(x + row * sx + hh, mask=hm, other=0.0).to(tl.float32)
                    wv = tl.load(w13 + e * s13e + n[:, None].to(tl.int64) * s13n + hh[None, :],
                                 mask=nm[:, None] & hm[None, :], other=0.0).to(tl.float32)
                    acc += tl.sum(wv * xv[None, :], 1)
                if ROUTED_FIRST:
                    acc = acc * tl.load(gw + row * K + k).to(tl.float32)
            tl.store(A + jk * N2 + n, _rnd(acc, OUT_BF16), mask=nm)

    @triton.jit
    def _moe_down(A, ids, gw, w2, emap, C, SQ, MX, stepbuf, M, H, I, K, s2e, s2n, nl, idx, u,
                  HAS_MAP: tl.constexpr, ROUTED_FIRST: tl.constexpr, OUT_BF16: tl.constexpr, ACT: tl.constexpr,
                  BO: tl.constexpr, BI: tl.constexpr):
        jk = tl.program_id(0)
        step, live, mr = _moe_live(stepbuf, nl, idx, M)
        if live:
            j, k = jk // K, jk % K
            row = _moe_row(step, j, mr, nl)
            e = _moe_expert(ids, emap, row, k, K, HAS_MAP)
            o = tl.program_id(1) * BO + tl.arange(0, BO)
            om = o < H
            acc = tl.zeros([BO], dtype=tl.float32)
            sq = tl.zeros([BO], dtype=tl.float32)
            mx = tl.zeros([BO], dtype=tl.float32)
            if e >= 0:
                for i0 in range(0, I, BI):
                    ii = i0 + tl.arange(0, BI)
                    im = ii < I
                    a1 = tl.load(A + jk * 2 * I + ii, mask=im, other=0.0)
                    a3 = tl.load(A + jk * 2 * I + I + ii, mask=im, other=0.0)
                    if ACT == 0:
                        sg = 1.0 / (1.0 + tl.exp(-a1))
                        s1 = a1 * sg
                        d1 = sg * (1.0 + a1 * (1.0 - sg))
                    else:
                        cdf = 0.5 * (1.0 + tl.math.erf(a1 * 0.7071067811865476))
                        s1 = a1 * cdf
                        d1 = cdf + a1 * tl.exp(-0.5 * a1 * a1) * 0.3989422804014327
                    s1 = _rnd(s1, OUT_BF16)
                    hv = _rnd(s1 * a3, OUT_BF16)
                    sens = 2.0 * u * (tl.abs(hv) + tl.abs(d1 * a1 * a3) + tl.abs(s1 * a3))
                    sens = tl.where(im, sens, 0.0)
                    hv = tl.where(im, hv, 0.0)
                    wv = tl.load(w2 + e * s2e + o[:, None].to(tl.int64) * s2n + ii[None, :],
                                 mask=om[:, None] & im[None, :], other=0.0).to(tl.float32)
                    p = wv * hv[None, :]
                    acc += tl.sum(p, 1)
                    sq += tl.sum(p * p, 1)
                    mx = tl.maximum(mx, tl.max(tl.abs(wv) * sens[None, :], 1))
                if ROUTED_FIRST:
                    gk = 1.0
                else:
                    gk = tl.load(gw + row * K + k).to(tl.float32)
                acc = _rnd(acc * gk, OUT_BF16)
                sq = sq * gk * gk
                mx = mx * tl.abs(gk)
            tl.store(C + jk * H + o, acc, mask=om)
            tl.store(SQ + jk * H + o, sq, mask=om)
            tl.store(MX + jk * H + o, mx, mask=om)

    @triton.jit
    def _moe_compare(out, x, ids, gw, C, SQ, MX, state, capm, capx, capy, capi, capg, stepbuf,
                     M, H, K, so, sx, capw, capk, slot, nl, idx, u, spacing, fmax, line,
                     NCOLS: tl.constexpr, SAMPLES_: tl.constexpr, OUT_FLIPS_: tl.constexpr, KAPPA_: tl.constexpr,
                     FLIPS_: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr):
        j = tl.program_id(0)
        step, live, mr = _moe_live(stepbuf, nl, idx, M)
        if live:
            row = _moe_row(step, j, mr, nl)
            kk = tl.arange(0, BK)
            km = kk < K
            # a padding row (all its experts -1: no matmul computes it) isn't judged or counted as checked
            nneg = tl.sum(tl.where(km & (tl.load(ids + row * K + kk, mask=km, other=0) == -1), 1, 0), 0)
            routed = nneg < K
            known = ((tl.load(stepbuf) >> 32) & 0x7FFFFFFF) > 0
            o = tl.program_id(1) * BO + tl.arange(0, BO)
            om = (o < H) & routed
            r = tl.zeros([BO], dtype=tl.float32)
            cmax = tl.zeros([BO], dtype=tl.float32)
            sq = tl.zeros([BO], dtype=tl.float32)
            mx = tl.zeros([BO], dtype=tl.float32)
            for k in range(0, K):
                base = (j * K + k) * H + o
                cv = tl.load(C + base, mask=om, other=0.0)
                r += cv
                cmax = tl.maximum(cmax, tl.abs(cv))
                sq += tl.load(SQ + base, mask=om, other=0.0)
                mx = tl.maximum(mx, tl.load(MX + base, mask=om, other=0.0))
            y = tl.load(out + row * so + o, mask=om, other=0.0).to(tl.float32)
            den = u * tl.abs(r) + OUT_FLIPS_ * 2.0 * u * cmax + KAPPA_ * u * tl.sqrt(sq) + FLIPS_ * mx + spacing
            z = tl.abs(y - r) / den
            z = tl.where(z == z, z, float("inf"))
            ok = om & (tl.abs(r) <= fmax) & (den < float("inf"))
            z = tl.where(ok, z, 0.0)
            nunj = tl.sum(tl.where(om, 1, 0) - tl.where(ok, 1, 0), 0)
            pz = tl.max(z, 0)
            pc = tl.argmax(z, 0)
            sel = tl.arange(0, BO) == pc
            py = tl.sum(tl.where(sel, y, 0.0), 0)
            pr = tl.sum(tl.where(sel, r, 0.0), 0)
            col = (tl.program_id(1) * BO + pc).to(tl.int64)
            sbase = state + slot * NCOLS
            zbits = pz.to(tl.int32, bitcast=True)
            zb = zbits.to(tl.int64) << 32
            if pz > 0.0:
                k0 = zb | row
                cur = tl.load(sbase)
                if zb >= (cur & -4294967296):      # (ties too: the same columns in any program order)
                    tl.atomic_max(sbase + 0, k0)
                    tl.atomic_max(sbase + 1, zb | col)
                    tl.atomic_max(sbase + 2, zb | (py.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                    tl.atomic_max(sbase + 3, zb | (pr.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                    tl.atomic_max(sbase + 4, zb | step)
            if (j == 0) & (tl.program_id(1) == 0):
                tl.atomic_add(sbase + 5, 1)
            if (tl.program_id(1) == 0) & routed:
                tl.atomic_add(sbase + 6, 1)
            if (j == 0) & (tl.program_id(1) == 0) & known:
                # the routing's validity: every known real row's expert ids, BR rows at a time
                nnone = tl.zeros([BR], dtype=tl.int32)
                nsome = tl.zeros([BR], dtype=tl.int32)
                last = tl.zeros([BR], dtype=tl.int64)
                for r0 in range(0, mr, BR):
                    rr = r0 + tl.arange(0, BR).to(tl.int64)
                    rm = rr < mr
                    blk = tl.load(ids + rr[:, None] * K + kk[None, :], mask=rm[:, None] & km[None, :], other=0)
                    nn = tl.sum(tl.where(rm[:, None] & km[None, :] & (blk == -1), 1, 0), 1)
                    bad_none = rm & (nn >= K)
                    bad_some = rm & (nn > 0) & (nn < K)
                    nnone += bad_none.to(tl.int32)
                    nsome += bad_some.to(tl.int32)
                    last = tl.maximum(last, tl.where(bad_none | bad_some, rr + 1, 0))
                tn = tl.sum(nnone, 0).to(tl.int64)
                ts = tl.sum(nsome, 0).to(tl.int64)
                if tn > 0:
                    tl.atomic_add(sbase + 8, tn)
                if ts > 0:
                    tl.atomic_add(sbase + 9, ts)
                lr = tl.max(last, 0)
                if lr > 0:
                    tl.atomic_max(sbase + 10, (step << 32) | lr)
            if nunj > 0:
                tl.atomic_add(sbase + 7, nunj.to(tl.int64))
            if pz > line:
                cb = capm + slot * 4
                old = tl.atomic_cas(cb, 0, 1)
                if old == 0:
                    tl.store(cb + 1, row.to(tl.int32))
                    tl.store(cb + 2, step.to(tl.int32))
                    tl.store(cb + 3, zbits)
                    for h0 in range(0, H, BO):
                        hh = h0 + tl.arange(0, BO)
                        hm = hh < H
                        tl.store(capx + slot * capw + hh, tl.load(x + row * sx + hh, mask=hm, other=0.0)
                                 .to(tl.float32), mask=hm)
                        tl.store(capy + slot * capw + hh, tl.load(out + row * so + hh, mask=hm, other=0.0)
                                 .to(tl.float32), mask=hm)
                    kk = tl.arange(0, BK)
                    km = kk < K
                    tl.store(capi + slot * capk + kk, tl.load(ids + row * K + kk, mask=km, other=0).to(tl.int64),
                             mask=km)
                    tl.store(capg + slot * capk + kk, tl.load(gw + row * K + kk, mask=km, other=0.0)
                             .to(tl.float32), mask=km)


def _p2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


SCRATCH: dict = {}        # (device, s*k, 2I, H) -> scratch buffers for the kernels (fixed for CUDA graphs)


def _scratch(dev, sk: int, n2: int, h: int) -> tuple:
    key = (str(dev), sk, n2, h)
    if key not in SCRATCH:
        with torch.inference_mode(False):
            SCRATCH[key] = (torch.zeros(sk, n2, device=dev), torch.zeros(sk, h, device=dev),
                            torch.zeros(sk, h, device=dev), torch.zeros(sk, h, device=dev))
    return SCRATCH[key]


def moe_check_triton(x, out, topk_w, topk_ids, w13, w2, expert_map, act, routed_first, state, slot, stepbuf,
                     nlayers, index, cap_meta, cap_x, cap_y, cap_ids, cap_g, line) -> None:
    m, hdim = x.shape[0], x.shape[-1]
    if m == 0:
        return
    u, sp, fm = dtype_consts(out.dtype)
    x2, y2 = x.reshape(m, hdim), out.reshape(m, hdim)
    x2 = x2 if x2.stride(-1) == 1 else x2.contiguous()
    y2 = y2 if y2.stride(-1) == 1 else y2.contiguous()
    ids = topk_ids.reshape(m, -1).contiguous()
    gw = topk_w.reshape(m, -1).contiguous()
    k = ids.shape[1]
    n2, ihalf = w13.shape[1], w2.shape[2]
    A, C, SQ, MX = _scratch(x.device, SAMPLES * k, n2, hdim)
    emap = expert_map if expert_map is not None else ids
    bf = out.dtype == torch.bfloat16
    BN, BH, BO, BI = 32, min(256, _p2(hdim)), 32, min(256, _p2(ihalf))
    _moe_up[(SAMPLES * k, -(-n2 // BN))](x2, ids, gw, w13, emap, A, stepbuf, m, hdim, n2, k, x2.stride(0),
                                         w13.stride(0), w13.stride(1), nlayers, index, HAS_MAP=expert_map is not None,
                                         ROUTED_FIRST=bool(routed_first), OUT_BF16=bf, BN=BN, BH=BH, num_warps=4)
    _moe_down[(SAMPLES * k, -(-hdim // BO))](A, ids, gw, w2, emap, C, SQ, MX, stepbuf, m, hdim, ihalf, k,
                                             w2.stride(0), w2.stride(1), nlayers, index, float(u),
                                             HAS_MAP=expert_map is not None, ROUTED_FIRST=bool(routed_first),
                                             OUT_BF16=bf, ACT=int(act), BO=BO, BI=BI, num_warps=4)
    BOC = min(1024, _p2(hdim))
    _moe_compare[(SAMPLES, -(-hdim // BOC))](y2, x2, ids, gw, C, SQ, MX, state, cap_meta, cap_x, cap_y, cap_ids, cap_g,
                                             stepbuf, m, hdim, k, y2.stride(0), x2.stride(0), cap_x.shape[1],
                                             cap_ids.shape[1], slot, nlayers, index, float(u), float(sp), float(fm),
                                             float(line), NCOLS=COLS, SAMPLES_=SAMPLES, OUT_FLIPS_=float(OUT_FLIPS),
                                             KAPPA_=float(KAPPA), FLIPS_=float(FLIPS), BO=BOC, BK=_p2(k),
                                             BR=ROUTE_BLOCK, num_warps=4)


def _dev_key(device) -> str:
    from ._group import _dev_key as kk
    return kk(device)


def kernel_usable(device, dtype) -> bool:
    if not HAVE_TRITON or os.environ.get("SDCGUARD_FAST", "1") == "0" or os.environ.get("SDCGUARD_MOE_FAST", "1") == "0":
        return False
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    return device.type == "cuda" or (device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1" and
                                     os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1")


def verify_kernel(device, dtype, act: int = 0) -> tuple:
    """The Triton kernels against the PyTorch check before their first use on a device, dtype and activation: a
    correct layer (both quiet, the same counts and nearly the same scores), one output made wrong (both alarm on its
    row and column and keep the same row), an expert map with a missing expert, an inactive step (nothing changes),
    a padding row (all experts -1, garbage output: both quiet, not counted), garbage rows past the step's real rows
    (both quiet), a real row with no experts and one with an expert missing (both counted, with the last such row,
    by the routing-validity scan). (ok, why)."""
    key = (_dev_key(device), dtype, int(act))
    if key in STATUS:
        return STATUS[key]
    if not kernel_usable(device, dtype):
        STATUS[key] = (False, "the Triton kernels aren't available here")
        return STATUS[key]
    from .guard import _exact_reductions_off, _exact_reductions_on
    hold = _exact_reductions_on()         # fp32 partial sums in cuBLAS, as served layers are checked
    try:
        gen = torch.Generator().manual_seed(0)
        E, H, I, K, M = 6, 192, 96, 3, 5
        w13 = (torch.randn(E, 2 * I, H, generator=gen) * 0.05).to(dtype).to(device)
        w2 = (torch.randn(E, H, I, generator=gen) * 0.05).to(dtype).to(device)
        x = torch.randn(M, H, generator=gen).to(dtype).to(device)
        g, ids = torch.topk(torch.softmax(torch.randn(M, E, generator=gen), -1), K, -1)
        g, ids = g.to(device), ids.to(torch.int32).to(device)
        emap = torch.tensor([0, 1, -1, 2, 3, 4], dtype=torch.int32, device=device)
        w13m, w2m = w13[:5].contiguous(), w2[:5].contiguous()
        ids_pad = ids.clone()
        ids_pad[M - 1] = -1                    # row M-1 is sampled at step 3 (rows 0 and M-1 of M = 5)
        ids_part = ids.clone()
        ids_part[1, 0] = -1                    # one expert of real row 1 missing (not sampled at step 3)
        for case in ("clean", "wrong", "map", "inactive", "padding", "real_rows", "unrouted_real", "partial_real"):
            res = []
            for fn in (moe_check_torch, moe_check_triton):
                st = torch.zeros(1, COLS, dtype=torch.int64, device=device)
                cm = torch.zeros(1, CAP_COLS, dtype=torch.int32, device=device)
                cx, cy = torch.zeros(1, H, device=device), torch.zeros(1, H, device=device)
                ci, cg = torch.zeros(1, 8, dtype=torch.int64, device=device), torch.zeros(1, 8, device=device)
                step = 3 if case != "inactive" else 4
                given = {"real_rows": 3, "unrouted_real": M, "partial_real": M}.get(case)
                sb = torch.full((1,), pack_step(step, given), dtype=torch.int64, device=device)
                use_map = case == "map"
                cids = ids_pad if case in ("padding", "unrouted_real") else ids_part if case == "partial_real" else ids
                y = _reference_layer(x, cids, g, w13m if use_map else w13, w2m if use_map else w2,
                                     emap if use_map else None, act, dtype)
                if case == "wrong":
                    rows = sample_rows(torch.tensor(step), M, "cpu", 4)
                    y[int(rows[0]), 17] = y[int(rows[0]), 17] * 4 + 1
                elif case in ("padding", "unrouted_real"):
                    y[M - 1] = 1000.0
                elif case == "real_rows":
                    y[3:] = 1000.0                 # past the 3 real rows: never sampled
                fn(x, y, g, cids, w13m if use_map else w13, w2m if use_map else w2, emap if use_map else None, act,
                   False, st, 0, sb, 4, 3, cm, cx, cy, ci, cg, MOE_Z)
                res.append((st.cpu(), cm.cpu(), cx.cpu(), cy.cpu(), ci.cpu(), cg.cpu()))
            (s0, c0, *k0), (s1, c1, *k1) = res
            z0, z1 = _f32(int(s0[0, C_ROW]) >> 32), _f32(int(s1[0, C_ROW]) >> 32)
            if s0[0, C_CALLS:].tolist() != s1[0, C_CALLS:].tolist():
                raise AssertionError(f"{case}: counts differ ({s0[0, C_CALLS:].tolist()} vs {s1[0, C_CALLS:].tolist()})")
            if case == "inactive":
                if s0.abs().sum() or s1.abs().sum():
                    raise AssertionError("an inactive step changed the state")
            elif case == "wrong":
                if not (z0 > MOE_Z and z1 > MOE_Z) or int(s0[0, C_COL]) & M32 != 17 or int(s1[0, C_COL]) & M32 != 17:
                    raise AssertionError(f"a wrong output wasn't located ({z0}, {z1})")
                if c0[0, :3].tolist() != c1[0, :3].tolist() or not torch.equal(k0[0], k1[0]) or \
                        not torch.equal(k0[1], k1[1]) or not torch.equal(k0[2][0, :K], k1[2][0, :K]):
                    raise AssertionError("the kept row differs")
            elif not (z0 <= 1.0 and z1 <= 1.0) or abs(z0 - z1) > 0.5:
                raise AssertionError(f"{case}: scores of a correct layer differ or are high ({z0} vs {z1})")
            elif int(s0[0, C_ROWS]) != (1 if case in ("padding", "unrouted_real") else SAMPLES):
                raise AssertionError(f"{case}: {int(s0[0, C_ROWS])} rows counted as checked")
            elif [int(s0[0, C_UNROUTED]), int(s0[0, C_PARTIAL]), int(s0[0, C_BADROW])] != \
                    {"unrouted_real": [1, 0, (3 << 32) | M], "partial_real": [0, 1, (3 << 32) | 2]}.get(case, [0, 0, 0]):
                raise AssertionError(f"{case}: real rows with experts -1 counted as {s0[0, C_UNROUTED:].tolist()}")
        STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    finally:
        _exact_reductions_off(hold)
    return STATUS[key]


def _reference_layer(x, ids, g, w13, w2, emap, act, dtype) -> torch.Tensor:
    """What vLLM's Triton MoE kernel computes for every row (for the first-use comparison and tests)."""
    r = recompute(x, ids, g, w13, w2, emap, act, False, dtype)[0]
    return r.to(dtype)


def moe_check_any(*args) -> None:
    """The check: the Triton kernels where they passed their first-use comparison, else the PyTorch one."""
    out, state, slot, act = args[1], args[9], args[10], args[7]
    OUT_DTYPES[(state.data_ptr(), slot)] = out.dtype
    with torch.no_grad(), torch.autocast(device_type=out.device.type, enabled=False):
        key = (_dev_key(out.device), out.dtype, int(act))
        if STATUS.get(key, (False,))[0]:
            try:
                moe_check_triton(*args)
                return
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
                FALLBACKS[0] += 1
        moe_check_torch(*args)


_lib_lock = threading.Lock()
_OP = [None]


def moe_op():
    """sdcguard::moe_check, registered once (vLLM runs its MoE layers inside an opaque op of its own, so this is
    called eagerly there; the op form is for a caller under torch.compile)."""
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::moe_check",
                                         mutates_args=("state", "cap_meta", "cap_x", "cap_y", "cap_ids", "cap_g"))
                def moe_check(x: torch.Tensor, out: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor,
                              w13: torch.Tensor, w2: torch.Tensor, expert_map: Optional[torch.Tensor], act: int,
                              routed_first: bool, state: torch.Tensor, slot: int, stepbuf: torch.Tensor,
                              nlayers: int, index: int, cap_meta: torch.Tensor, cap_x: torch.Tensor,
                              cap_y: torch.Tensor, cap_ids: torch.Tensor, cap_g: torch.Tensor, line: float) -> None:
                    moe_check_any(x, out, topk_w, topk_ids, w13, w2, expert_map, act, routed_first, state, slot,
                                  stepbuf, nlayers, index, cap_meta, cap_x, cap_y, cap_ids, cap_g, line)

                @moe_check.register_fake
                def _(*a):
                    return None

                _OP[0] = torch.ops.sdcguard.moe_check
            except (AttributeError, RuntimeError):
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "moe_check", None) or moe_check_any
        return _OP[0]


def check_call(args: tuple, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, out) -> None:
    """Check one call of a registered MoE layer from its input, routing and output (a finalized tensor; a deferred
    output isn't checked)."""
    if not isinstance(out, torch.Tensor) or not isinstance(x, torch.Tensor) or out.shape[-1] != x.shape[-1] or \
            out.shape[0] != x.shape[0]:
        return
    w13, w2, emap, act, rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg, line = args
    call = (x, out, topk_weights, topk_ids, w13, w2, emap, act, rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg, line)
    if torch.compiler.is_compiling():
        (_OP[0] if _OP[0] is not None else moe_op())(*call)
    else:
        moe_check_any(*call)


def recheck(x_row: torch.Tensor, y_row: torch.Tensor, ids: torch.Tensor, g: torch.Tensor, w13: torch.Tensor,
            w2: torch.Tensor, expert_map: Optional[torch.Tensor], act: int, routed_first: bool, dtype: torch.dtype,
            line: float = MOE_Z) -> dict:
    """The kept row recomputed on the CPU with the same rounding steps (fp32 arithmetic, as the GPU check), from the
    experts' weights as the layer holds them now."""
    e = ids.long().cpu()
    if expert_map is not None:
        em = expert_map.long().cpu()
        e = torch.where(e >= 0, em[e.clamp(min=0)], e)
    used = sorted({int(v) for v in e.tolist() if v >= 0})
    if not used:
        return {"verdict": "not_rechecked", "why": "the kept row's experts aren't on this GPU"}
    remap = {v: i for i, v in enumerate(used)}
    w13c = w13[torch.tensor(used, device=w13.device)].cpu()
    w2c = w2[torch.tensor(used, device=w2.device)].cpu()
    local = torch.tensor([[remap.get(int(v), -1) for v in e]])
    r, cmax, sq, mx = recompute(x_row.view(1, -1).to(dtype).cpu(), local, g.view(1, -1).float().cpu(), w13c, w2c,
                                None, act, routed_first, dtype)
    yf = y_row.view(1, -1).float().cpu()
    z, _ = scores(yf, r, cmax, sq, mx, dtype)
    worst = float(z.max()) if z.numel() else 0.0
    bad = int((z > line).sum())
    out = {"worst_score": worst, "outputs_over_line": bad, "outputs": int(z.numel()), "experts": used}
    if z.numel():                                  # the worst output's place and values (first of tied ones)
        c = int(z.reshape(-1).argmax())
        out.update(col=c, output=float(yf.reshape(-1)[c]), expected=float(r.reshape(-1)[c]))
    if bad:
        return {**out, "verdict": "output_wrong",
                "why": f"{bad} of {z.numel()} outputs of the kept row are farther from the recompute (the layer's "
                       "current weights, the kept input and routing) than its rounding allows"}
    return {**out, "verdict": "output_agrees",
            "why": "the CPU recompute of the kept row agrees with its output: the GPU's check disagreed, or the "
                   "weights changed back since"}


# ------------------------------------------------------------------------------------------------- the MoE layers
def describe(mod) -> tuple:
    """(what the check needs, or None; why not). A layer qualifies when vLLM runs it on its Triton MoE kernel with
    bf16/fp16 expert weights, no biases, a gated SiLU or GELU, and the routing weight applied on the output."""
    qm = getattr(mod, "quant_method", None)
    if type(qm).__name__ != "UnquantizedFusedMoEMethod":
        return None, f"{type(qm).__name__ if qm is not None else 'no quant method'}: quantized or another method"
    pc = getattr(getattr(mod, "moe_config", None), "moe_parallel_config", None)
    try:             # vLLM 0.32 (all2all_utils.maybe_make_prepare_finalize): the modular kernel itself gathers and
        #              combines across ranks with expert parallelism's all-to-all, with "passthrough" experts, and with
        #              data parallelism without it (its all-gather/reduce-scatter fallback, over expert-weight shards)
        across = bool(getattr(pc, "use_all2all_kernels", False) or getattr(pc, "use_passthrough_all2all", False) or
                      int(getattr(pc, "dp_size", 1) or 1) > 1 or int(getattr(pc, "pcp_size", 1) or 1) > 1)
    except Exception:  # noqa: BLE001  (properties that can't be read: treated as communicating, unchecked)
        across = True
    if across:
        return None, ("expert parallelism's all-to-all, data parallelism or prefill context parallelism: the layer's "
                      "rows or output combine other GPUs', which the recompute can't read")
    backend = getattr(getattr(qm, "unquantized_backend", None), "name", None)
    if backend != "TRITON":
        return None, f"the {backend or 'unknown'} MoE backend (only vLLM's Triton MoE kernel is checked)"
    w13, w2 = getattr(mod, "w13_weight", None), getattr(mod, "w2_weight", None)
    if not (isinstance(w13, torch.Tensor) and isinstance(w2, torch.Tensor) and w13.dim() == 3 and w2.dim() == 3):
        return None, "no expert weights the check knows"
    if w13.dtype not in (torch.bfloat16, torch.float16) or w2.dtype != w13.dtype:
        return None, f"expert weights in {w13.dtype}"
    if w13.shape[0] != w2.shape[0] or w13.shape[2] != w2.shape[1] or w13.shape[1] != 2 * w2.shape[2]:
        return None, f"expert weight shapes {tuple(w13.shape)} and {tuple(w2.shape)} aren't gate-up and down"
    if any(isinstance(getattr(mod, b, None), torch.Tensor) for b in ("w13_bias", "w2_bias")):
        return None, "experts with biases"
    act = getattr(getattr(mod, "activation", None), "value", getattr(mod, "activation", None))
    if str(act) not in ACT:
        return None, f"activation {act}"
    if not (w13.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
        return None, "not on a GPU"
    if getattr(mod, "_sdcguard_moe", None) is not None:
        return None, "already checked by another ServeGuard"
    return {"act": ACT[str(act)], "routed_first": bool(getattr(mod, "apply_router_weight_on_input", False)),
            "top_k": int(getattr(mod, "top_k", 0) or 0)}, None


class MoEChecks:
    """The fused MoE layers one ServeGuard also checks (ServeGuard.moe_checks()): add() each, finalize() once before
    the model is compiled or its CUDA graphs captured, mark_step() before each model step (it decides which layer is
    active) and end_step() after, poll() with the guard's reads."""

    def __init__(self, guard, line: float = MOE_Z):
        line = float(line)
        if not 5.0 <= line < float("inf"):
            raise ValueError(f"the MoE check's line must be at least 5, not {line!r}")
        self.guard, self.line = guard, line
        self.layers: list = []
        self.unsupported: dict = {}
        self._state, self._cap, self._step, self._host, self._read_from = {}, {}, {}, {}, {}
        self._marked: Optional[int] = None
        self._last_clean: dict = {}
        self.worst_seen: dict = {}
        self.checks_seen: dict = {}
        self.finalized = False
        self.stats = {"layers": 0, "active_calls": 0, "rows_checked": 0, "unjudged_outputs": 0,
                      "unrouted_real_rows": 0, "partly_routed_real_rows": 0, "alerts": 0}

    def add(self, mod, name: str) -> Optional[int]:
        if self.finalized:
            raise RuntimeError("MoE layers are added before finalize()")
        info, why = describe(mod)
        if why:
            self.unsupported[name] = why
            return None
        with torch.inference_mode(False):
            sums = (mod.w13_weight.detach().float().sum(1), mod.w2_weight.detach().float().sum(2))
        self.layers.append({"name": name, "module": mod, "device": mod.w13_weight.device, "sums": sums, **info,
                            "h": int(mod.w13_weight.shape[2])})
        self.stats["layers"] += 1
        return len(self.layers) - 1

    def finalize(self) -> int:
        if self.finalized:
            return len(self.layers)
        devs: dict = {}
        for lay in self.layers:
            devs.setdefault(lay["device"], []).append(lay)
        n = len(self.layers)
        for dev, lays in devs.items():
            hmax = max(lay["h"] for lay in lays)
            kmax = max(max(lay["top_k"], 1) for lay in lays)
            for i, lay in enumerate(lays):
                lay["index"], lay["nlayers"] = i, len(lays)
            with torch.inference_mode(False):
                self._state[dev] = torch.zeros(n, COLS, dtype=torch.int64, device=dev)
                self._cap[dev] = (torch.zeros(n, CAP_COLS, dtype=torch.int32, device=dev),
                                  torch.zeros(n, hmax, device=dev), torch.zeros(n, hmax, device=dev),
                                  torch.zeros(n, max(kmax, 64), dtype=torch.int64, device=dev),
                                  torch.zeros(n, max(kmax, 64), device=dev))
                self._step[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
                for lay in lays:                       # the kernels' scratch, before any CUDA graph is captured
                    w13 = lay["module"].w13_weight
                    _scratch(dev, SAMPLES * max(lay["top_k"], 1), int(w13.shape[1]), lay["h"])
                for dt, act in {(lay["module"].w13_weight.dtype, lay["act"]) for lay in lays}:
                    verify_kernel(dev, dt, act)
        moe_op()
        for slot, lay in enumerate(self.layers):
            lay["module"]._sdcguard_moe = self.args(slot)
        self.finalized = True
        return n

    def args(self, slot: int) -> tuple:
        lay = self.layers[slot]
        mod, dev = lay["module"], lay["device"]
        cm, cx, cy, ci, cg = self._cap[dev]
        return (mod.w13_weight, mod.w2_weight, getattr(mod, "expert_map", None), lay["act"], lay["routed_first"],
                self._state[dev], slot, self._step[dev], lay["nlayers"], lay["index"], cm, cx, cy, ci, cg, self.line)

    def refresh(self) -> int:
        """Take the experts' current weights as theirs (after an intended update), for weights_changed."""
        n = 0
        with torch.inference_mode(False):
            for lay in self.layers:
                mod = lay["module"]
                lay["sums"] = (mod.w13_weight.detach().float().sum(1), mod.w2_weight.detach().float().sum(2))
                n += 1
        return n

    def detach(self) -> None:
        try:
            if self.layers and self.finalized:
                self.poll(wait=True)
        finally:
            for lay in self.layers:
                lay["module"].__dict__.pop("_sdcguard_moe", None)
            self.layers.clear()
            for d in (self._state, self._cap, self._step, self._host, self._read_from):
                d.clear()
            self.finalized = False

    def mark_step(self, step: int, rows: Optional[int] = None) -> None:
        """Before a model step: its number and, when known, its real rows before padding (the rows samples are
        drawn from; unknown, all of a call's rows)."""
        step = int(step) & M32
        self._marked = step
        for dev, sb in self._step.items():
            sb.fill_(pack_step(step, rows))
            self._read_from.setdefault(dev, step)

    def end_step(self) -> None:
        """After a model step's work is queued: the runs before the next step (warm-up, CUDA graph capture) are
        checked over all their rows, under the last step's number."""
        if self._marked is None:
            return
        for sb in self._step.values():
            sb.fill_(pack_step(self._marked))

    def _window(self, dev):
        if self._marked is None:
            return None
        a = self._read_from.pop(dev, None)
        return [a if a is not None else self._marked, self._marked]

    def poll(self, wait: bool = False) -> list:
        events = []
        for dev, st in self._state.items():
            cm = self._cap[dev][0]
            if dev.type != "cuda" or wait:
                pend = self._host.pop(dev, None)
                if pend is not None:
                    pend[2].synchronize()
                    events += self._report(dev, pend[0], pend[1], pend[3])
                host, hcap = st.to("cpu", copy=True), cm.to("cpu", copy=True)
                st.zero_()
                events += self._report(dev, host, hcap, self._window(dev))
                continue
            pend = self._host.get(dev)
            if pend is not None and pend[2].query():
                events += self._report(dev, pend[0], pend[1], pend[3])
                pend = None
                self._host.pop(dev, None)
            if pend is None:
                hs = torch.empty(st.shape, dtype=st.dtype, pin_memory=True)
                hc = torch.empty(cm.shape, dtype=cm.dtype, pin_memory=True)
                hs.copy_(st, non_blocking=True)
                hc.copy_(cm, non_blocking=True)
                st.zero_()
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (hs, hc, ev, self._window(dev))
        return events

    def coverage(self) -> dict:
        return {"layers": len(self.layers), "active_calls": self.stats["active_calls"],
                "layers_checked": sum(self.checks_seen.get(lay["name"], 0) > 0 for lay in self.layers),
                "never_checked": sorted(lay["name"] for lay in self.layers if not self.checks_seen.get(lay["name"])),
                "not_attached": dict(self.unsupported), "samples_per_step": SAMPLES}

    def _report(self, dev, host, hcap, window) -> list:
        out = []
        worst = 0.0
        for slot, lay in enumerate(self.layers):
            if lay["device"] != dev or slot >= host.shape[0]:
                continue
            keys = [int(v) for v in host[slot].tolist()]
            self.stats["active_calls"] += keys[C_CALLS]
            self.stats["rows_checked"] += keys[C_ROWS]
            self.stats["unjudged_outputs"] += keys[C_UNJ]
            self.stats["unrouted_real_rows"] += keys[C_UNROUTED]
            self.stats["partly_routed_real_rows"] += keys[C_PARTIAL]
            self.checks_seen[lay["name"]] = self.checks_seen.get(lay["name"], 0) + keys[C_CALLS]
            routed_ok = not (keys[C_UNROUTED] or keys[C_PARTIAL])
            if not routed_ok:
                out.append(self._route_alert(dev, slot, keys, window))
            z = _f32(keys[C_ROW] >> 32)
            worst = max(worst, z)
            self.worst_seen[lay["name"]] = max(self.worst_seen.get(lay["name"], 0.0), z)
            if not (z > self.line):
                if keys[C_CALLS] and window is not None and routed_ok:     # (a routing alert isn't a clean read)
                    self._last_clean[lay["name"]] = list(window)
                continue
            out.append(self._alert(dev, slot, keys, [int(v) for v in hcap[slot].tolist()], window, z))
        g = self.guard
        if g is not None and getattr(g, "metrics", None) is not None and hasattr(g.metrics, "observe_moe"):
            try:
                g.metrics.observe_moe(self, out, worst)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
        return out

    def _route_alert(self, dev, slot, keys, window) -> SDCEvent:
        """Real rows of the step with experts -1 (the routing-validity scan): wrong whatever the numbers."""
        lay = self.layers[slot]
        none_, some = keys[C_UNROUTED], keys[C_PARTIAL]
        last = (keys[C_BADROW] & M32) - 1
        step = (keys[C_BADROW] >> 32) & M32
        captured = {"row": last, "step": step if self._marked is not None else None} if last >= 0 else None
        detail = {"where": lay["name"], "kind": "moe_routing",
                  "finding": "real_rows_unrouted" if none_ else "real_rows_partly_routed",
                  "real_rows_without_experts": none_, "real_rows_missing_some_experts": some,
                  "row": last if last >= 0 else None, "captured": captured,
                  "recheck": {"verdict": "output_wrong",
                              "why": "a real token's row had expert ids of -1 (vLLM's routers give every real token "
                                     "top_k experts; -1 is for CUDA-graph padding rows), so the layer's output for it "
                                     "isn't the sum of its experts"},
                  "read_window_steps": window, "last_clean_read_steps": self._last_clean.get(lay["name"]),
                  "reading": "a real row reached the experts with no expert (or some missing): a router fault, a "
                             "corrupted expert id, or a padding mask that marked a real row"}
        return self._emit(dev, detail)

    def _emit(self, dev, detail: dict) -> SDCEvent:
        g = self.guard
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        key = str(dev)
        if key not in ident:
            ident[key] = device_identity(dev)
        ev = SDCEvent(detector="serve_moe", step=None, detail=detail, identity=dict(ident[key]))
        self.stats["alerts"] += 1
        if g is not None:
            if getattr(g, "attribute", None) is not None:
                try:
                    g.attribute(ev)
                except Exception:  # noqa: BLE001
                    g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
            g.log.add(ev)
            if getattr(g, "on_alert", None):
                try:
                    g.on_alert(ev)
                except Exception:  # noqa: BLE001
                    pass
        return ev

    def _alert(self, dev, slot, keys, cap, window, z) -> SDCEvent:
        g = self.guard
        lay = self.layers[slot]
        mod = lay["module"]
        dtype = OUT_DTYPES.get((self._state[dev].data_ptr(), slot)) or mod.w13_weight.dtype
        captured, rc = None, None
        if cap[0]:
            cm, cx, cy, ci, cg = self._cap[dev]
            h, k = lay["h"], max(lay["top_k"], 1)
            x_row, y_row = cx[slot, :h].cpu(), cy[slot, :h].cpu()
            ids, gw = ci[slot, :k].cpu(), cg[slot, :k].cpu()
            cm[slot].zero_()
            captured = {"row": cap[1] & M32, "step": (cap[2] & M32) if self._marked is not None else None,
                        "score": _f32(cap[3]), "experts": ids.tolist(), "weights": gw.tolist()}
            try:
                rc = recheck(x_row, y_row, ids, gw, mod.w13_weight, mod.w2_weight, getattr(mod, "expert_map", None),
                             lay["act"], lay["routed_first"], dtype, self.line)
            except Exception as e:  # noqa: BLE001  (diagnosis is best effort)
                rc = {"verdict": "not_rechecked", "why": f"{type(e).__name__}: {e}"}
        changed = None
        try:
            now = (mod.w13_weight.detach().float().sum(1), mod.w2_weight.detach().float().sum(2))
            changed = not (torch.equal(now[0], lay["sums"][0]) and torch.equal(now[1], lay["sums"][1]))
        except Exception:  # noqa: BLE001
            pass
        if rc is not None and rc.get("verdict") == "output_agrees" and changed:
            rc = {**rc, "verdict": "weights_changed",
                  "why": "the kept row agrees with a recompute from the experts' current weights, which differ from "
                         "their weights at load: the weights changed in memory (refresh() after an intended update)"}
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        key = str(dev)
        if key not in ident:
            ident[key] = device_identity(dev)
        yv, rv = _f32(keys[C_OUT]), _f32(keys[C_REF])
        row, col, place = keys[C_ROW] & M32, keys[C_COL] & M32, PLACE_MAXIMA
        if captured is not None and rc is not None and rc.get("outputs_over_line") and rc.get("col") is not None:
            row, col, yv, rv, place = captured["row"], rc["col"], rc["output"], rc["expected"], PLACE_KEPT
        detail = {"where": lay["name"], "kind": "moe_experts", "max_z": z, "row": row, "col": col, "output": yv,
                  "expected": rv, "place": place,
                  "relative_error": (abs(yv - rv) / abs(rv)) if rv and rv == rv else None, "captured": captured,
                  "recheck": rc or {"verdict": "not_rechecked",
                                    "why": "no row was kept for this alert (the layer's kept row was taken by an "
                                           "earlier alert that hasn't been read yet)"},
                  "weights_changed_since_load": changed, "read_window_steps": window,
                  "last_clean_read_steps": self._last_clean.get(lay["name"]),
                  "check_config": {"line": self.line, "samples_per_step": SAMPLES, "kappa": KAPPA, "flips": FLIPS,
                                   "unit_roundoff": dtype_consts(dtype)[0],
                                   "reference": "sampled recompute with the Triton MoE kernel's rounding steps"},
                  "reading": "an output of the layer's routed experts differs from a recompute from the layer's own "
                             "input, routing and weights by more than the kernel's rounding allows"}
        ev = SDCEvent(detector="serve_moe", step=None, detail=detail, identity=dict(ident[key]))
        self.stats["alerts"] += 1
        if g is not None:
            if getattr(g, "attribute", None) is not None:
                try:
                    g.attribute(ev)
                except Exception:  # noqa: BLE001
                    g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
            g.log.add(ev)
            if getattr(g, "on_alert", None):
                try:
                    g.on_alert(ev)
                except Exception:  # noqa: BLE001
                    pass
        return ev
