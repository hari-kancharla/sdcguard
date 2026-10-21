"""Precise check fused into the matmul: block checksums of the fp32 accumulator, computed inside the GEMM.

The default check sums each layer's bf16/fp16 *output*. Rounding to bf16 buries small errors in noise; ByteDance's
AEGIS reports that in deployment, checksums from bf16 outputs located none of its incidents and every confirmed
checksum detection used fp32 accumulators (OSDI '26, section 8). sdcguard's precise mode checks the fp32 result,
but gets it from a second, fp32-output GEMM, which made a checked step 260% to 790% slower.

This module computes the linear layer itself in a Triton GEMM whose epilogue, before rounding, reduces every
128-column block of each output row to four float64 numbers: the plain sum, the signed sum (the same +-1 pattern
the other checks use), the sum of squares and the sum of absolute values. The layer's bf16/fp16 output comes from
the same accumulator, so the matmul runs once. The check then compares those sums with x . (block sums of W)
in float64 (that part is still separate work), under precise mode's fp32 noise model (checksum._acc_zscores), and reports z-scores in the same layout
as every other check (checksum.resolve decodes them).

Trade-offs, stated plainly:
  - on checked steps the layer's output comes from this Triton kernel, not cuBLAS: the result is the same fp32
    accumulation rounded the same way, but the summation order differs, so output bits can differ in the last
    place from a cuBLAS run, and a cuBLAS-specific defect (like known issue 6862950) isn't exercised on those
    calls; the default check is the one that covers cuBLAS;
  - nn.Linear layers under protect() only (their forward, dX and dW matmuls); other matmuls keep their checks;
  - experimental: sensitivity, false alarms and cost need a GPU run before any claim.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import torch

from . import checksum as cs

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False

BLOCK = 128                        # checksum block width = the kernel's tile width in N
_SIGNS: dict = {}


def available(x: torch.Tensor) -> bool:
    """Triton is importable and x is on a CUDA GPU (or the CPU under TRITON_INTERPRET=1, for tests)."""
    if not _HAVE_TRITON or x.dtype not in (torch.bfloat16, torch.float16):
        return False
    return x.is_cuda or (x.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1")


def _signs_for(n: int, device, epoch: int) -> torch.Tensor:
    """The signed-checksum pattern for every output column: the 128-wide pattern repeated over full blocks, then
    the remainder block's own pattern (as checksum._pair_weights uses for a narrower last block)."""
    key = (n, str(device), epoch)
    s = _SIGNS.get(key)
    if s is None:
        nfull, rem = divmod(n, BLOCK)
        parts = []
        if nfull:
            parts.append(cs._pair_weights(BLOCK, torch.float64, device, epoch)[:, 1].repeat(nfull))
        if rem:
            parts.append(cs._pair_weights(rem, torch.float64, device, epoch)[:, 1])
        s = torch.cat(parts).contiguous()
        if len(_SIGNS) > 64:
            _SIGNS.clear()
        _SIGNS[key] = s
    return s


if _HAVE_TRITON:
    @triton.jit
    def _fused_linear_kernel(X, W, BIAS, SIGNS, Y, STATS, M, N, K,
                             sxm, sxk, swn, swk, sym, syn, s_sm, s_sb,
                             inj_m, inj_n, inj_v,
                             HAS_BIAS: tl.constexpr, INJECT: tl.constexpr, EMULATE: tl.constexpr,
                             ROUND_BF16: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        mm = offs_m < M
        mn = offs_n < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            a = tl.load(X + offs_m[:, None] * sxm + kk[None, :] * sxk, mask=mm[:, None] & (kk[None, :] < K),
                        other=0.0)
            b = tl.load(W + offs_n[None, :] * swn + kk[:, None] * swk, mask=mn[None, :] & (kk[:, None] < K),
                        other=0.0)
            if EMULATE:          # tests only: Triton's CPU interpreter gets bf16 dots wrong
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_n, mask=mn, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]
        if INJECT:                                   # experiments: a fault in the accumulator, before rounding
            hit = (offs_m[:, None] == inj_m) & (offs_n[None, :] == inj_n)
            acc = tl.where(hit, acc + inj_v, acc)
        if ROUND_BF16:
            # tests only: Triton's CPU interpreter truncates fp32 to bf16; GPUs round to nearest even, as here
            bits = acc.to(tl.uint32, bitcast=True)
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            out = bits.to(tl.float32, bitcast=True).to(Y.dtype.element_ty)
        else:
            out = acc.to(Y.dtype.element_ty)
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn, out, mask=mm[:, None] & mn[None, :])
        c = tl.where(mn[None, :], acc, 0.0).to(tl.float64)
        s = tl.load(SIGNS + offs_n, mask=mn, other=0.0).to(tl.float64)
        base = STATS + offs_m * s_sm + pid_n * s_sb
        tl.store(base + 0, tl.sum(c, axis=1), mask=mm)
        tl.store(base + 1, tl.sum(c * s[None, :], axis=1), mask=mm)
        tl.store(base + 2, tl.sum(c * c, axis=1), mask=mm)
        tl.store(base + 3, tl.sum(tl.abs(c), axis=1), mask=mm)


def fused_mm(A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None, epoch: int = 0,
             inject: Optional[tuple] = None, bm: int = 64, bk: int = 32, num_warps: int = 8):
    """C = A @ B + bias computed by the fused kernel, in A's dtype, and (m, ceil(n/128), 4) float64 block
    statistics of the fp32 result before rounding: plain sum, signed sum, sum of squares, sum of absolute
    values. A: (m, k), B: (k, n), any strides (a linear layer's x @ w.t() and both backward matmuls are views,
    not copies). bias: (n,), added in fp32 before rounding, as cuBLAS does.
    inject=(row, col, value) adds value to one accumulator element before rounding (experiments)."""
    if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch: A{tuple(A.shape)} B{tuple(B.shape)}")
    M, K = A.shape
    N = B.shape[1]
    if bias is not None and tuple(bias.shape) != (N,):
        raise ValueError(f"bias must have shape ({N},), not {tuple(bias.shape)}")
    if not available(A):
        raise RuntimeError("the fused check needs Triton and a bf16/fp16 tensor on a CUDA GPU")
    if B.dtype != A.dtype:
        B = B.to(A.dtype)
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    nb = -(-N // BLOCK)
    stats = torch.empty((M, nb, 4), dtype=torch.float64, device=A.device)
    if M == 0 or N == 0:
        return C, stats
    signs = _signs_for(N, A.device, epoch)
    if bias is not None:
        if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            bias = bias.float()
        bias = bias.contiguous()
    im, inn, iv = inject if inject is not None else (0, 0, 0.0)
    from ._group import pipeline_stages
    grid = (triton.cdiv(M, bm), nb)
    kw = {"num_warps": num_warps}
    stages = pipeline_stages(A.device, None)
    if stages is not None:
        kw["num_stages"] = stages
    # W = B.t(): the kernel reads B as an (n, k) matrix through its strides
    _fused_linear_kernel[grid](A, B, bias if bias is not None else signs, signs, C, stats, M, N, K,
                               A.stride(0), A.stride(1), B.stride(1), B.stride(0), C.stride(0), C.stride(1),
                               stats.stride(0), stats.stride(1), int(im), int(inn), float(iv),
                               HAS_BIAS=bias is not None, INJECT=inject is not None,
                               EMULATE=A.device.type == "cpu",
                               ROUND_BF16=A.device.type == "cpu" and A.dtype == torch.bfloat16,
                               BM=bm, BN=BLOCK, BK=bk, **kw)
    return C, stats


def fused_linear(x2: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None, **kw):
    """y = x2 @ w.T + bias and its block statistics (fused_mm with B = w.t())."""
    return fused_mm(x2, w.t(), bias, **kw)


def zscores_from_stats(A: torch.Tensor, B: torch.Tensor, stats: torch.Tensor, bias: Optional[torch.Tensor] = None,
                       *, z: float = 6.0, nfma: Optional[int] = None, epoch: int = 0,
                       abs_floor: float = cs.ABS_FLOOR) -> cs.CheckStat:
    """Precise mode's check (checksum._acc_zscores) of C = A @ B + bias, with C's block statistics taken from
    the fused kernel instead of a stored fp32 C. A: (m, k) bf16/fp16; B: (k, n), for example w.T."""
    if B.dtype != A.dtype:
        B = B.to(A.dtype)                    # as fused_mm multiplies it
    with torch.autocast(device_type=A.device.type, enabled=False):
        return _zscores(A, B, stats, bias, z, nfma, epoch, abs_floor)


def _zscores(A, B, stats, bias, z, nfma, epoch, abs_floor) -> cs.CheckStat:
    A3, B3 = cs._as_batched(A), cs._as_batched(B)
    m, k = A3.shape[-2], A3.shape[-1]
    n = B3.shape[-1]
    u = cs.ACC_UNIT
    steps = k / nfma if nfma else 0.0
    spans, offset = [], 0
    for start, width, nblk in ((0, BLOCK, n // BLOCK), ((n // BLOCK) * BLOCK, n % BLOCK, 1)):
        if nblk and width:
            spans.append((start, width, nblk, offset))
            offset += m * nblk * 2
    A_norm = cs._norm_fp32(A3).view(A3.shape[0], m, 1)
    A_l1 = cs._norm_fp32(A3, ord=1).view(A3.shape[0], m, 1)
    maxes, args, noises, unknown = [], [], [], []
    for start, width, nblk, off in spans:
        end = start + width * nblk
        b0 = start // BLOCK
        st = stats[:, b0:b0 + nblk].unsqueeze(0)                              # (1, m, nblk, 4) float64
        Bblk = B3[..., start:end].reshape(B3.shape[0], k, nblk, width)
        B2 = cs._pair_sums64(Bblk, epoch)
        expected = cs._matmul64(A3, B2.view(B3.shape[0], k, 2 * nblk)).view(-1, m, nblk, 2)
        bound = A_l1 * Bblk.abs().amax(dim=(-3, -1)).float().unsqueeze(-2)
        if bias is not None:
            bb = bias[start:end].reshape(nblk, width)
            expected = expected + cs._pair_sums64(bb, epoch)
            bound = bound + bb.abs().amax(-1).float()
        actual = st[..., :2]
        cnorm = st[..., 2].clamp_min(0).sqrt().float()
        l1 = st[..., 3].float()
        b_frob = torch.linalg.vector_norm(Bblk.float(), dim=(-3, -1)).unsqueeze(-2)
        per_out = cs.ACC_C2 * math.sqrt(k) + cs.ACC_C3 * steps + math.sqrt(width)
        prod = (A_norm * b_frob).mul_(cs.ACC_C1 * math.sqrt(k) * u)
        prod = torch.where(torch.isfinite(prod), prod, bound * (u * 128))
        noise = (cnorm * (per_out * u) + prod).unsqueeze(-1)
        den = (noise + abs_floor).double()
        diff = (expected - actual).abs_()
        zs = diff / den
        worst = (l1 * (per_out * u) + prod).unsqueeze(-1).add_(abs_floor).double()
        zs[..., :1] = torch.minimum(zs[..., :1], diff[..., :1] / worst * z)
        zs = cs._judge_nonfinite(zs.float(), expected, actual, noise, bound.unsqueeze(-1), torch.float32, cnorm,
                                 width, 0.0, k)
        unknown.append((zs < 0).any())
        mz, arg = torch.max(zs.view(-1), 0)
        maxes.append(mz)
        noises.append(den.float().expand_as(zs).reshape(-1).gather(0, arg.view(1)).squeeze(0))
        args.append(arg if off == 0 else arg + off)
    if len(maxes) == 1:
        max_z, arg, nz = maxes[0], args[0], noises[0]
    else:
        mzs = torch.stack(maxes)
        best = mzs.argmax().view(1)
        max_z = mzs.gather(0, best).squeeze(0)
        arg = torch.stack(args).gather(0, best).squeeze(0)
        nz = torch.stack(noises).gather(0, best).squeeze(0)
    max_z = cs._complete_score(max_z, unknown, z)
    return cs.CheckStat(max_z=max_z.float(), argmax=arg, noise=nz, spans=tuple(spans), m=m, batched=False)


def check_linear_fused(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None, *, z: float = 6.0,
                       nfma: Optional[int] = None, epoch: int = 0, inject: Optional[tuple] = None):
    """(y, CheckResult): the linear layer's output from the fused kernel and the precise check of its fp32
    accumulator. x may have leading batch dimensions."""
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    y, stats = fused_linear(x2, w, bias, epoch=epoch, inject=inject)
    st = zscores_from_stats(x2, w.to(x2.dtype).t(), stats, bias, z=z, nfma=nfma, epoch=epoch)
    return y.reshape(*lead, w.shape[0]), cs.resolve([st], z)[0]


# ---------------------------------------------------------------------------------------------------------------
# The fp32 self-check: the same fused GEMM, scored inside the kernel, no float64 anywhere.
#
# The first version above leaves float64 block statistics for a separate float64 scoring step, which the
# October 20 GPU run found takes most of the time (about 1 ms per matmul on an H100, 3 to 23 ms on an L4) and made
# the check cost as much as precise mode. Following what AEGIS (OSDI '26) and FP-Sketch (arXiv 2609.19758) do,
# this version never computes a high-precision reference:
#   - the expected side x . (block sums of W) is accumulated in fp32 on the tensor cores by a second, narrow dot
#     in the same K loop: each block sum (computed once per call from W, in fp32) is split into three bf16 or
#     fp16 parts whose sum is exact, so the narrow dot sees the full fp32 value. In fp16 the block sums are first
#     divided by a power of two that puts the block's largest possible sum (128 x max|W|) between 2^14 and 2^15,
#     and the narrow dot's result is multiplied back (both exact): fp16 stops at 65504, and a block sum above
#     that made its parts infinite and left the check of an ordinary finite layer unjudged;
#   - precise mode's frozen noise model (checksum._acc_zscores: C1 sqrt(k) u |x| |W_block| for the accumulation,
#     |c| (C2 sqrt(k) + C3 steps + sqrt(width)) u per output) gets one more term for the expected side's own
#     fp32 accumulation, C1 sqrt(k) u |x| |block sums|_2 (the same model applied to the dot product x . sums,
#     where the column of block sums plays the part of a column of W), and the z-scores are computed in the
#     kernel's epilogue with the same rules for Inf/NaN and impossible values as the other checks;
#   - per call, besides the kernel: one small pass over W (block sums and norms; two in fp16, since the scale
#     needs the block's max|W| first), two row norms of x, and a max over the scores, all without a host sync.
# ---------------------------------------------------------------------------------------------------------------
NX = 16                                   # width of the narrow dot: 3 parts x 2 sums, padded to a tensor-core tile

if _HAVE_TRITON:
    @triton.jit
    def _pow2_scale(wmax):
        """The power of two that puts 128 x wmax (the largest block sum a block with max|W| = wmax can have) in
        [2^14, 2^15), so fp16 holds the three parts of every block sum; 1 when wmax isn't finite (that block's
        check is unjudged anyway). Exact bit arithmetic on the fp32 exponent, the same on every device."""
        e = ((wmax * 128.0).to(tl.int32, bitcast=True) >> 23) & 0xFF
        e = tl.where(e == 255, 141, tl.maximum(e, 15))
        return ((e - 14) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _bside_kernel(B, SIGNS, BX, AUX, K, N, sbk, sbn, s_xk, s_xb, s_xc,
                      BKP: tl.constexpr, BN: tl.constexpr, SPLIT_DT: tl.constexpr, STATS: tl.constexpr = True,
                      PARTS: tl.constexpr = True, SCALED: tl.constexpr = False):
        """A pass over W for block j (by default both parts below in one pass, as bf16 runs it). PARTS: per row k
        the plain and signed block sums, split into 3 parts each (BX[k, j, 0:6]), divided first by
        _pow2_scale(max|W| of the block) if SCALED (fp16: that pass runs after a STATS pass has finished AUX).
        STATS: per block the sums of squares of W, of its plain block sums and of its signed ones (unscaled),
        and max |W| (AUX[j, 0:4], atomics; NaN counts as Inf)."""
        pid_k = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_k = pid_k * BKP + tl.arange(0, BKP)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mk = offs_k < K
        mn = offs_n < N
        b = tl.load(B + offs_k[:, None] * sbk + offs_n[None, :] * sbn, mask=mk[:, None] & mn[None, :],
                    other=0.0).to(tl.float32)
        s = tl.load(SIGNS + offs_n, mask=mn, other=0.0).to(tl.float32)
        bp = tl.sum(b, axis=1)
        bs = tl.sum(b * s[None, :], axis=1)
        if PARTS:
            xp = bp
            xs = bs
            if SCALED:
                sc = _pow2_scale(tl.load(AUX + pid_n * 4 + 3))
                xp = bp / sc
                xs = bs / sc
            base = BX + offs_k * s_xk + pid_n * s_xb
            hi = xp.to(SPLIT_DT)
            r = xp - hi.to(tl.float32)
            mid = r.to(SPLIT_DT)
            lo = (r - mid.to(tl.float32)).to(SPLIT_DT)
            tl.store(base + 0 * s_xc, hi, mask=mk)
            tl.store(base + 1 * s_xc, mid, mask=mk)
            tl.store(base + 2 * s_xc, lo, mask=mk)
            hi = xs.to(SPLIT_DT)
            r = xs - hi.to(tl.float32)
            mid = r.to(SPLIT_DT)
            lo = (r - mid.to(tl.float32)).to(SPLIT_DT)
            tl.store(base + 3 * s_xc, hi, mask=mk)
            tl.store(base + 4 * s_xc, mid, mask=mk)
            tl.store(base + 5 * s_xc, lo, mask=mk)
        if STATS:
            tl.atomic_add(AUX + pid_n * 4 + 0, tl.sum(tl.sum(b * b, axis=1), axis=0))
            tl.atomic_add(AUX + pid_n * 4 + 1, tl.sum(bp * bp, axis=0))
            tl.atomic_add(AUX + pid_n * 4 + 2, tl.sum(bs * bs, axis=0))
            ab = tl.where(b == b, tl.abs(b), float("inf"))
            tl.atomic_max(AUX + pid_n * 4 + 3, tl.max(tl.max(ab, axis=1), axis=0))

    @triton.jit
    def _fused_check_kernel(X, W, BIAS, SIGNS, BX, AUX, ANORM, AL1, Y, Z, DEN, M, N, K,
                            sxm, sxk, swn, swk, sym, syn, s_xk, s_xb, s_xc, s_zm, s_zb, s_dm,
                            inj_m, inj_n, inj_v,
                            k_f, steps, u, c1, c2, c3, abs_floor, zthr, dt_max, big,
                            HAS_BIAS: tl.constexpr, INJECT: tl.constexpr, EMULATE: tl.constexpr,
                            ROUND_BF16: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                            NXC: tl.constexpr, SCALED: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        offs_x = tl.arange(0, NXC)
        mm = offs_m < M
        mn = offs_n < N
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        accx = tl.zeros((BM, NXC), dtype=tl.float32)
        for k0 in range(0, K, BK):
            kk = k0 + offs_k
            mk = kk < K
            a = tl.load(X + offs_m[:, None] * sxm + kk[None, :] * sxk, mask=mm[:, None] & mk[None, :], other=0.0)
            b = tl.load(W + offs_n[None, :] * swn + kk[:, None] * swk, mask=mn[None, :] & mk[:, None], other=0.0)
            bx = tl.load(BX + kk[:, None] * s_xk + pid_n * s_xb + offs_x[None, :] * s_xc,
                         mask=mk[:, None] & (offs_x[None, :] < 6), other=0.0)
            if EMULATE:          # tests only: Triton's CPU interpreter gets bf16 dots wrong
                acc = tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")
                accx = tl.dot(a.to(tl.float32), bx.to(tl.float32), accx, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
                accx = tl.dot(a, bx, accx)
        s = tl.load(SIGNS + offs_n, mask=mn, other=0.0).to(tl.float32)
        cols = tl.arange(0, NXC)
        e_p = tl.sum(tl.where(cols[None, :] < 3, accx, 0.0), axis=1)
        e_s = tl.sum(tl.where((cols[None, :] >= 3) & (cols[None, :] < 6), accx, 0.0), axis=1)
        bmax = tl.load(AUX + pid_n * 4 + 3)
        if SCALED:                                   # the parts were made from block sums / this power of two
            sc = _pow2_scale(bmax)
            e_p = e_p * sc
            e_s = e_s * sc
        bound_b = 0.0
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_n, mask=mn, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]
            e_p = e_p + tl.sum(bias, axis=0)
            e_s = e_s + tl.sum(bias * s, axis=0)
            bound_b = tl.max(tl.abs(bias), axis=0)
        if INJECT:                                   # experiments: a fault in the accumulator, before rounding
            hit = (offs_m[:, None] == inj_m) & (offs_n[None, :] == inj_n)
            acc = tl.where(hit, acc + inj_v, acc)
        if ROUND_BF16:
            bits = acc.to(tl.uint32, bitcast=True)
            bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
            out = bits.to(tl.float32, bitcast=True).to(Y.dtype.element_ty)
        else:
            out = acc.to(Y.dtype.element_ty)
        tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn, out, mask=mm[:, None] & mn[None, :])
        # ---- the score, in fp32
        c = tl.where(mn[None, :], acc, 0.0)
        a_p = tl.sum(c, axis=1)
        a_s = tl.sum(c * s[None, :], axis=1)
        cn = tl.sqrt(tl.sum(c * c, axis=1))
        cl1 = tl.sum(tl.abs(c), axis=1)
        width = tl.minimum(N - pid_n * BN, BN).to(tl.float32)
        an = tl.load(ANORM + offs_m, mask=mm, other=0.0)
        al1 = tl.load(AL1 + offs_m, mask=mm, other=0.0)
        bfro = tl.sqrt(tl.load(AUX + pid_n * 4 + 0))
        bsn_p = tl.sqrt(tl.load(AUX + pid_n * 4 + 1))       # |plain block sums|_2 over k
        bsn_s = tl.sqrt(tl.load(AUX + pid_n * 4 + 2))
        sk = tl.sqrt(k_f)
        per_out = c2 * sk + c3 * steps + tl.sqrt(width)
        bound = al1 * bmax + bound_b
        prod = an * bfro * (c1 * sk * u)
        prod = tl.where(prod == prod, prod, bound * (u * 128.0))
        prod = tl.where(tl.abs(prod) < float("inf"), prod, bound * (u * 128.0))
        noise = cn * (per_out * u) + prod
        # the expected side's own fp32 accumulation (the same model, for the dot product x . sums), with the
        # same stand-in as prod's when the squares don't fit in fp32
        ex_p = an * bsn_p * (c1 * sk * u)
        ex_s = an * bsn_s * (c1 * sk * u)
        ex_p = tl.where(tl.abs(ex_p) < float("inf"), ex_p, bound * (u * 128.0))
        ex_s = tl.where(tl.abs(ex_s) < float("inf"), ex_s, bound * (u * 128.0))
        noise_p = noise + ex_p
        noise_s = noise + ex_s
        den_p = noise_p + abs_floor
        den_s = noise_s + abs_floor
        d_p = tl.abs(e_p - a_p)
        d_s = tl.abs(e_s - a_s)
        z_p = d_p / den_p
        z_s = d_s / den_s
        worst = cl1 * (per_out * u) + prod + ex_p + abs_floor
        z_p = tl.minimum(z_p, d_p / worst * zthr)
        # Inf/NaN and impossible values, as checksum._judge_nonfinite
        lim = bound * 1.01
        e_bad = ~(tl.abs(e_p) < float("inf")) | ~(tl.abs(e_s) < float("inf"))
        a_bad = (~(tl.abs(a_p) < float("inf")) | ~(tl.abs(a_s) < float("inf"))) & ~e_bad
        n_bad = (~(noise_p < float("inf")) | ~(noise_s < float("inf"))) & ~e_bad & ~a_bad
        cap = bound * (1.02 + k_f * 2.0 ** -22)
        imposs = ~e_bad & ~a_bad & ~n_bad & (lim < big) & (cn > cap * tl.sqrt(width))
        for_a = tl.where(lim < tl.minimum(dt_max, big), float("inf"), -1.0)
        for_n = tl.where(lim < big, float("inf"), -1.0)
        z_p = tl.where(z_p == z_p, z_p, -1.0)
        z_s = tl.where(z_s == z_s, z_s, -1.0)
        z_p = tl.where(imposs, float("inf"), z_p)
        z_s = tl.where(imposs, float("inf"), z_s)
        z_p = tl.where(a_bad, for_a, tl.where(n_bad, for_n, z_p))
        z_s = tl.where(a_bad, for_a, tl.where(n_bad, for_n, z_s))
        z_p = tl.where(e_bad, -1.0, z_p)
        z_s = tl.where(e_bad, -1.0, z_s)
        zb = Z + offs_m * s_zm + pid_n * s_zb
        tl.store(zb + 0, z_p, mask=mm)
        tl.store(zb + 1, z_s, mask=mm)
        tl.store(DEN + offs_m * s_dm + pid_n, tl.where(z_p >= z_s, den_p, den_s), mask=mm)


def fused_mm_checked(A: torch.Tensor, B: torch.Tensor, bias: Optional[torch.Tensor] = None, *, z: float = 6.0,
                     nfma: Optional[int] = None, epoch: int = 0, abs_floor: float = cs.ABS_FLOOR,
                     inject: Optional[tuple] = None, bm: int = 64, bk: int = 32, num_warps: int = 8):
    """C = A @ B + bias from the fused kernel, and the check of its fp32 accumulator scored in the kernel
    (cs.CheckStat, decoded by checksum.resolve like every other check). No float64, no host sync.
    A: (m, k), B: (k, n), any strides; bias (n,) added in fp32 before rounding."""
    if A.dim() != 2 or B.dim() != 2 or A.shape[1] != B.shape[0]:
        raise ValueError(f"shape mismatch: A{tuple(A.shape)} B{tuple(B.shape)}")
    if not available(A):
        raise RuntimeError("the fused check needs Triton and a bf16/fp16 tensor on a CUDA GPU")
    M, K = A.shape
    N = B.shape[1]
    if bias is not None and tuple(bias.shape) != (N,):
        raise ValueError(f"bias must have shape ({N},), not {tuple(bias.shape)}")
    if B.dtype != A.dtype:
        B = B.to(A.dtype)
    dev = A.device
    C = torch.empty((M, N), dtype=A.dtype, device=dev)
    nb = -(-N // BLOCK)
    nfull, rem = divmod(N, BLOCK)
    spans, off = [], 0
    for start, width, nblk in ((0, BLOCK, nfull), (nfull * BLOCK, rem, 1)):
        if nblk and width:
            spans.append((start, width, nblk, off))
            off += M * nblk * 2
    if M == 0 or N == 0 or K == 0:
        zero = torch.zeros((), dtype=torch.float32, device=dev)
        if M and N:                          # k = 0: the output is the bias (or zero)
            C.copy_(bias.to(C.dtype).expand(M, N) if bias is not None else torch.zeros_like(C))
        return C, cs.CheckStat(max_z=zero, argmax=torch.zeros((), dtype=torch.int64, device=dev), noise=zero,
                               spans=tuple(spans), m=M, batched=False)
    signs = _signs_for(N, dev, epoch)
    from ._group import pipeline_stages
    kw = {"num_warps": num_warps}
    stages = pipeline_stages(dev, None)
    if stages is not None:
        kw["num_stages"] = stages
    # W side: block sums split in parts, and per-block norms
    BX = torch.zeros((K, nb, NX), dtype=A.dtype, device=dev)
    AUX = torch.zeros((nb, 4), dtype=torch.float32, device=dev)
    bkp = 32
    scaled = A.dtype == torch.float16
    bside = _bside_kernel[(triton.cdiv(K, bkp), nb)]
    args = (B, signs, BX, AUX, K, N, B.stride(0), B.stride(1), BX.stride(0), BX.stride(1), BX.stride(2))
    split_dt = tl.float16 if scaled else tl.bfloat16
    if scaled:          # fp16: the parts' scale needs the block's max|W|, so the statistics come first
        bside(*args, BKP=bkp, BN=BLOCK, SPLIT_DT=split_dt, STATS=True, PARTS=False, SCALED=False)
        bside(*args, BKP=bkp, BN=BLOCK, SPLIT_DT=split_dt, STATS=False, PARTS=True, SCALED=True)
    else:
        bside(*args, BKP=bkp, BN=BLOCK, SPLIT_DT=split_dt, STATS=True, PARTS=True, SCALED=False)
    with torch.autocast(device_type=dev.type, enabled=False):
        anorm = torch.linalg.vector_norm(A, dim=1, dtype=torch.float32)
        al1 = torch.linalg.vector_norm(A, ord=1, dim=1, dtype=torch.float32)
    Zt = torch.empty((M, nb, 2), dtype=torch.float32, device=dev)
    DEN = torch.empty((M, nb), dtype=torch.float32, device=dev)
    if bias is not None:
        if bias.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            bias = bias.float()
        bias = bias.contiguous()
    im, inn, iv = inject if inject is not None else (0, 0, 0.0)
    steps = K / nfma if nfma else 0.0
    _fused_check_kernel[(triton.cdiv(M, bm), nb)](
        A, B, bias if bias is not None else signs, signs, BX, AUX, anorm, al1, C, Zt, DEN, M, N, K,
        A.stride(0), A.stride(1), B.stride(1), B.stride(0), C.stride(0), C.stride(1),
        BX.stride(0), BX.stride(1), BX.stride(2), Zt.stride(0), Zt.stride(1), DEN.stride(0),
        int(im), int(inn), float(iv),
        float(K), float(steps), cs.ACC_UNIT, cs.ACC_C1, cs.ACC_C2, cs.ACC_C3, float(abs_floor), float(z),
        float(cs.DTYPE_MAX[torch.float32]), float(cs.BIG),
        HAS_BIAS=bias is not None, INJECT=inject is not None, EMULATE=dev.type == "cpu",
        ROUND_BF16=dev.type == "cpu" and A.dtype == torch.bfloat16, BM=bm, BN=BLOCK, BK=bk, NXC=NX, SCALED=scaled,
        **kw)
    # decode into checksum.resolve's layout: full blocks first ((m, nfull, 2)), then the remainder ((m, 1, 2))
    if rem and nfull:
        zf = torch.cat([Zt[:, :nfull].reshape(-1), Zt[:, nfull:].reshape(-1)])
        df = torch.cat([DEN[:, :nfull].unsqueeze(-1).expand(M, nfull, 2).reshape(-1),
                        DEN[:, nfull:].unsqueeze(-1).expand(M, 1, 2).reshape(-1)])
    else:
        zf = Zt.reshape(-1)
        df = DEN.unsqueeze(-1).expand(M, nb, 2).reshape(-1)
    unknown = (zf < 0).any()
    mz, arg = torch.max(zf, 0)
    noise = df.gather(0, arg.view(1)).squeeze(0)
    max_z = cs._complete_score(mz, [unknown], z)
    return C, cs.CheckStat(max_z=max_z.float(), argmax=arg, noise=noise, spans=tuple(spans), m=M, batched=False)
