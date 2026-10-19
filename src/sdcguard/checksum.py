"""Block checksums for matrix multiplies (algorithm-based fault tolerance).

For C = A @ B (+ bias), the sum of every block of `block` consecutive output
columns must equal A @ (column-block sums of B), plus the block sums of the bias.
We keep two such checksums per block: the plain sum and a sum with fixed random
+-1 signs, so changes that cancel out in one still show in the other. Checking
costs about 2/block of the matmul's FLOPs plus one extra read of A, B and C.

Low-precision outputs (bf16/fp16) are rounded, so the two sides never match
exactly. The residual is measured in units of the estimated rounding noise of
each output block, and anything above `z` (default 6) is treated as corruption.

That noise estimate assumes the outputs' rounding errors are independent. When many outputs in
a block are equal (B with identical columns, e.g. a matrix of ones), they round the same way and
their errors add up: up to sqrt(128) ~ 11 times the estimate in the plain sum. The signed sum's
random +-1 signs average such errors out, so it keeps the estimate. The plain sum additionally has
to exceed the worst rounding possible, u x sum of |outputs| (each output is off by at most u x its
size; 1.25x that for margin), before it counts: its z is min(z, threshold x z_worst).

Tiny fp16 outputs: below fp16's smallest normal number (6e-5) values sit on a fixed grid with steps
of 6e-8, so their rounding error doesn't shrink with their size. The noise estimate adds that
(sqrt(block width) x half a step); without it, small but correct fp16 gradients looked corrupted.

Tiny outputs: the noise estimate squares the outputs in fp32, and squares of values
below about 1e-19 lose precision or round to zero (fp32's smallest normal number is
about 1.2e-38). Then the estimated noise collapses and a correct block looks wildly
wrong: on an H100, GPT-2 small trained without learning-rate warmup produced attention
gradients around 1e-24 and z values up to ~90,000 on clean steps. So the noise never
goes below ABS_FLOOR (1e-20): differences smaller than that are not judged. They are far
too small to affect training, and a fault that makes a tiny value big is still caught.
Values above about 1e19 overflow the squares instead.

Infinite and NaN outputs are judged against what the inputs allow. For every output row and
block, |output| <= |alpha| * sum_k |a_k| * max |B| over the block (+ |beta| * max |bias|).
If that bound is below the largest value the output dtype holds (and below 1e18, where the fp32
checksum arithmetic itself starts to overflow), a finite computation can't produce Inf or NaN
there, so an Inf or NaN output is corruption (z = inf). Otherwise the block isn't judged: fp16
training with loss scaling (torch.amp.GradScaler) overflows on purpose now and then, and skips
those steps. Outputs too large to square in fp32 are judged the same way against 1e18, and a
block whose expected side isn't finite (Inf/NaN inputs, or sums beyond fp32) isn't judged.

Two entry points:
  matmul_zscores()  queues the check on the GPU and returns device tensors. It never
                    waits for the GPU, so it doesn't stall training. SDCGuard uses it and
                    reads all results once per guarded step (resolve()).
  check_matmul()    the same check, answered immediately (waits for the GPU). For tests
                    and one-off use.

matmul_zscores() must never make the CPU wait for the GPU: no .item(), no Python numbers
copied onto the GPU (torch.tensor(x, device=...) waits for the GPU on every call), and no
indexing with a GPU tensor (x[idx] with a 0-d tensor idx calls .item(); use gather). The GPU
self-test runs a checked step with PyTorch's sync debugger set to "error" to prove it, on a
model whose last layer has a remainder block (1000 outputs) so the multi-span path runs too.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from . import _group

# Use the Triton kernels on NVIDIA GPUs when available (see _group.py). Tests and the
# self-test switch this off to compare both paths.
USE_FAST = True

# Unit roundoff u = 2^-p (p = significand bits): rounding to nearest changes a value by at most u x
# its size (exactly that much, just above a power of two). The worst-case bound on the plain sum
# below relies on this; its 1.25 factor is the only margin.
UNIT_ROUNDOFF = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11}

# Differences below this are not judged (see "Tiny outputs" above).
ABS_FLOOR = 1e-20

# Below the smallest normal number, values sit on a fixed grid (the subnormal step) and rounding is
# off by up to half a step whatever their size. fp16's step (6e-8) matters for small gradients; bf16's
# (9e-41) is far below ABS_FLOOR, so it is left out.
HALF_TINY_STEP = {torch.bfloat16: 0.0, torch.float16: 2.0**-25}

# Largest finite value of each output dtype: Inf/NaN outputs are judged only when the inputs
# can't produce values this large (see above). BIG: the same for outputs too large to square.
DTYPE_MAX = {torch.bfloat16: float(torch.finfo(torch.bfloat16).max), torch.float16: 65504.0}
BIG = 1e18

# Above this many elements, the CPU/MPS fallback converts A to fp32 in row chunks.
_FP32_CHUNK_ELEMS = 1 << 26


@dataclass
class CheckResult:
    ok: bool                 # False means "looks corrupted"
    max_z: float             # worst residual, in units of rounding noise (inf/nan: non-finite output;
                             # -1: no block could be judged)
    nonfinite: bool          # NaN or Inf in the output (or values too large to square in fp32)
    batch: Optional[int] = None      # where the worst block is (for bmm)
    row: Optional[int] = None
    col_block: Optional[int] = None
    noise: Optional[float] = None    # estimated rounding noise of that block (its scale)

    @property
    def judged(self) -> bool:
        """False when nothing in the output could be judged (non-finite inputs, legitimate overflow):
        ok then means "no evidence either way", not "clean"."""
        return not (math.isfinite(self.max_z) and self.max_z < 0)


@dataclass
class CheckStat:
    """A queued check: device tensors plus the Python facts needed to decode them."""
    max_z: torch.Tensor          # 0-d float32 on the output's device
    argmax: torch.Tensor         # 0-d int64: flat index of the worst checksum (see resolve())
    noise: torch.Tensor          # 0-d float32: estimated noise at that checksum
    spans: tuple                 # ((start, width, nblk, offset), ...): offset = first flat index
    m: int
    batched: bool


def _as_batched(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() == 3 else t.unsqueeze(0)


def _sum_fp32(t: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim, accumulating in fp32 (some backends need an explicit upcast)."""
    try:
        return t.sum(-1, dtype=torch.float32)
    except (RuntimeError, TypeError):
        return t.float().sum(-1)


def _norm_fp32(t: torch.Tensor, dim: int = -1, ord: float = 2) -> torch.Tensor:
    """L2 (or L1) norm over `dim` in fp32."""
    try:
        return torch.linalg.vector_norm(t, ord=ord, dim=dim, dtype=torch.float32)
    except (RuntimeError, TypeError):
        return torch.linalg.vector_norm(t.float(), ord=ord, dim=dim)


_LOWP_GEMM_OK: dict = {}
_LOWP_FALLBACKS = [0]          # checks that fell back to the slow path because of a RuntimeError


def _split3(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """fp32 -> three low-precision parts whose sum is x to ~2^-27 relative error."""
    hi = x.to(dtype)
    r = x - hi                      # fp32 result (type promotion), one kernel
    mid = r.to(dtype)
    lo = (r - mid).to(dtype)
    return torch.cat([hi, mid, lo], dim=-1)


def _use_lowp_gemm(A3: torch.Tensor) -> bool:
    return A3.is_cuda and _LOWP_GEMM_OK.get(A3.dtype, True)


def _lowp_gemm(A3: torch.Tensor, Bs: torch.Tensor) -> torch.Tensor:
    """Low-precision inputs, fp32 output (cuBLAS accumulates in fp32). CUDA only."""
    if A3.shape[0] == 1 and Bs.shape[0] == 1:
        return torch.ops.aten.mm.dtype(A3[0], Bs[0], torch.float32).unsqueeze(0)
    Bs = Bs.expand(A3.shape[0], -1, -1) if Bs.shape[0] == 1 else Bs
    return torch.ops.aten.bmm.dtype(A3, Bs, torch.float32)


_SIGNS: dict = {}
# The noise model treats each output's rounding as independent of the signs. With one fixed pattern,
# outputs that happen to line up with it (B's columns proportional to the signs, say) round together
# and can cross the alarm line on a correct result. A fresh pattern in every process makes that
# alignment chance instead of a property of the data. SDCGUARD_SIGN_SEED pins it (tests, replays).
_SIGN_SEED = int(os.environ.get("SDCGUARD_SIGN_SEED") or int.from_bytes(os.urandom(4), "little"))


def _signs(width: int, device) -> torch.Tensor:
    """A pseudo-random +-1 pattern (fp32) for the second checksum, drawn once per process."""
    key = (width, str(device))
    if key not in _SIGNS:
        g = torch.Generator().manual_seed((_SIGN_SEED + width) % (2 ** 63))
        _SIGNS[key] = (torch.randint(0, 2, (width,), generator=g) * 2 - 1).float().to(device)
    return _SIGNS[key]


def _c_gemm(flat: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.mm.dtype(flat, W, torch.float32)


_PAIR_W: dict = {}


def _pair_weights(width: int, dtype: torch.dtype, device) -> torch.Tensor:
    """[1, s] as a (width, 2) matrix in `dtype` (exact: every entry is +-1). Cached."""
    key = (width, dtype, str(device))
    w = _PAIR_W.get(key)
    if w is None:
        s = _signs(width, device)
        w = torch.stack([torch.ones_like(s), s], dim=1).to(dtype)
        _PAIR_W[key] = w
    return w


def _pair_sums(Xb: torch.Tensor) -> torch.Tensor:
    """Plain and signed sums of every block, in fp32: (..., nblk, width) -> (..., nblk, 2).

    On NVIDIA GPUs both come from one pass (a skinny GEMM against [1, s] with fp32 output);
    elsewhere, two reductions.
    """
    width = Xb.shape[-1]
    if Xb.dtype in UNIT_ROUNDOFF and _use_lowp_gemm(Xb):
        try:
            Xc = Xb if Xb.is_contiguous() else Xb.contiguous()
            out = _c_gemm(Xc.view(-1, width), _pair_weights(width, Xb.dtype, Xb.device))
            return out.view(*Xb.shape[:-1], 2)
        except (RuntimeError, NotImplementedError, AttributeError):
            pass
    s = _pair_weights(width, Xb.dtype, Xb.device)[:, 1]
    return torch.stack([_sum_fp32(Xb), _sum_fp32(Xb * s)], dim=-1)


def _matmul_fp32(A3: torch.Tensor, Bb: torch.Tensor) -> torch.Tensor:
    """A3 (low precision, (bA, m, k)) @ Bb (fp32, (bB, k, nblk)) -> fp32 (b, m, nblk).

    On NVIDIA GPUs this multiplies A in its own precision against a 3-way split of Bb
    with an fp32 output, so A is never copied to fp32 and TF32 settings can't reduce the
    precision. Elsewhere (CPU, Apple GPUs) it converts A to fp32 (in row chunks if huge).
    """
    nblk = Bb.shape[-1]
    if _use_lowp_gemm(A3):
        try:
            scale = None
            if A3.dtype == torch.float16:
                # fp16 overflows at 65504: scale each column of Bb by a power of two (exact) into
                # range first. Per column, so one huge or non-finite column (left unscaled) doesn't
                # spoil the others.
                amax = Bb.abs().amax(dim=-2, keepdim=True)
                ok = torch.isfinite(amax) & (amax > 0)
                scale = torch.where(ok, torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30)))), 1.0)
                Bb = Bb / scale
            out = _lowp_gemm(A3, _split3(Bb, A3.dtype))                 # (b, m, 3*nblk)
            res = out.view(*out.shape[:-1], 3, nblk).sum(-2)
            return res * scale if scale is not None else res
        except (NotImplementedError, AttributeError, TypeError):
            _LOWP_GEMM_OK[A3.dtype] = False      # old PyTorch without fp32-output GEMMs: fall back for good
        except RuntimeError:
            _LOWP_FALLBACKS[0] += 1              # e.g. out of memory: fall back for this check only
    m = A3.shape[-2]
    if A3.numel() <= _FP32_CHUNK_ELEMS or m < 2:
        return torch.matmul(A3.float(), Bb)
    rows = max(1, _FP32_CHUNK_ELEMS // max(1, A3.shape[0] * A3.shape[-1]))
    return torch.cat([torch.matmul(A3[:, i:i + rows].float(), Bb) for i in range(0, m, rows)], dim=-2)


UNJUDGED = -1.0     # z of a block that can't be judged; a check whose every block is unjudged reports it


def _judge_nonfinite(z, expected, actual, noise, bound, dtype):
    """Judging Inf/NaN (see the module docstring). An expected side that isn't finite can't be
    judged (non-finite inputs, or values too large for fp32 sums): z = UNJUDGED. An Inf/NaN output, or
    one too large to square in fp32, is corruption (z = inf) only if the inputs can't produce it; if they
    can (fp16 loss-scaling overflow, say), it can't be judged either."""
    e_bad = ~torch.isfinite(expected)
    a_bad = ~torch.isfinite(actual) & ~e_bad
    big = ~torch.isfinite(noise) & ~e_bad & ~a_bad
    lim = bound * 1.01
    z = torch.where(z == z, z, UNJUDGED)                # inf/inf (both sides huge): nothing to judge
    z = torch.where(e_bad, UNJUDGED, z)
    z = torch.where(a_bad, torch.where(lim < min(DTYPE_MAX[dtype], BIG), float("inf"), UNJUDGED), z)
    return torch.where(big, torch.where(lim < BIG, float("inf"), UNJUDGED), z)


def matmul_zscores(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    block: int = 128,
    abs_floor: float = ABS_FLOOR,
    z: float = 6.0,
) -> Optional[CheckStat]:
    """Queue the check of C = beta*bias + alpha*(A @ B) without waiting for the GPU.

    Works for 2D (mm/addmm) and 3D (bmm). Returns None when the output dtype isn't
    supported (only bf16/fp16 for now; fp32 matmuls can run as TF32 on NVIDIA GPUs and
    need a different noise model).

    Layout: every block of `block` output columns gets two checksums (plain, signed), stored
    side by side as (..., nblk, 2), so the expected side, the actual side and the noise all
    line up without reshuffling.
    """
    u = UNIT_ROUNDOFF.get(C.dtype)
    if u is None:
        return None

    A3, B3, C3 = _as_batched(A), _as_batched(B), _as_batched(C)
    nb_batch, m, n = C3.shape
    k = A3.shape[-1]
    if A3.shape[-2] != m or B3.shape[-2] != k or B3.shape[-1] != n:
        raise ValueError(f"shape mismatch: A{tuple(A.shape)} B{tuple(B.shape)} C{tuple(C.shape)}")
    if A3.dtype != C3.dtype:
        A3 = A3.to(C3.dtype)
    if B3.dtype != C3.dtype:
        B3 = B3.to(C3.dtype)

    bs = max(1, min(block, n))
    spans, offset = [], 0
    for start, width, nblk in ((0, bs, n // bs), ((n // bs) * bs, n % bs, 1)):
        if nblk and width:
            spans.append((start, width, nblk, offset))
            offset += max(nb_batch, A3.shape[0], B3.shape[0]) * m * nblk * 2

    A_norm = _norm_fp32(A3).view(A3.shape[0], m, 1, 1)                      # (bA, m, 1, 1)
    A_l1 = _norm_fp32(A3, ord=1).view(A3.shape[0], m, 1)                    # (bA, m, 1)
    maxes, args, noises = [], [], []
    for start, width, nblk, off in spans:
        end = start + width * nblk
        Cb = C3[..., start:end].reshape(nb_batch, m, nblk, width)          # a view when possible
        Bblk = B3[..., start:end].reshape(B3.shape[0], k, nblk, width)
        # Two checksums per block: the plain sum, and a sum with fixed random +-1 signs. A fault
        # whose changes happen to cancel in one sum (e.g. zeroing a +3 and a -3) shows in the other.
        B2 = _pair_sums(Bblk)                                                # (bB, k, nblk, 2) fp32
        expected = _matmul_fp32(A3, B2.view(B3.shape[0], k, 2 * nblk))
        expected = expected.view(expected.shape[0], m, nblk, 2)
        # largest output these inputs allow, per row and block (for judging Inf/NaN outputs):
        # |a_row|_1 x max |B| over the block (NaN propagates, which counts like Inf)
        bound = A_l1 * Bblk.abs().amax(dim=(-3, -1)).float().unsqueeze(-2)  # (b, m, nblk) fp32
        if alpha != 1.0:
            expected = expected * alpha
            bound = bound * abs(alpha)
        if bias is not None and beta != 0:
            if bias.dim() == 1:                                              # (n,)
                bb = bias[start:end].reshape(nblk, width)
            else:                                                            # (m, n) or (1, n)
                bb = torch.broadcast_to(bias, (m, n))[:, start:end].reshape(m, nblk, width)
            expected = expected.add(_pair_sums(bb), alpha=beta)
            bound = bound + abs(beta) * bb.abs().amax(-1).float()             # (nblk,) or (m, nblk)

        actual = _pair_sums(Cb)                                              # (batch, m, nblk, 2)
        # Rounding noise of a block sum ~ unit roundoff x L2 norm of the outputs (same for +-1
        # signs), plus a floor for the fp32 rounding of `expected` itself (matters only when the
        # outputs nearly cancel): 1e-5 x |alpha| x |a_row| x |checksum column of B|.
        noise = _norm_fp32(Cb).mul_(u).unsqueeze(-1)                         # (batch, m, nblk, 1)
        b2n = _norm_fp32(B2, dim=-3).unsqueeze(-3)                           # (bB, 1, nblk, 2)
        bound = bound.unsqueeze(-1)                                          # (b, m, nblk, 1)
        floor = (A_norm * b2n).mul_(1e-5 * abs(alpha))
        # |a_row| or the checksum column too large to square in fp32: bound the same rounding
        # with the largest output instead
        floor = torch.where(torch.isfinite(floor), floor, bound * (1e-5 * 128))
        h = HALF_TINY_STEP[C3.dtype]
        den = (noise + floor).add_(abs_floor + math.sqrt(width) * h)
        diff = (expected - actual).abs_()
        zs = diff / den
        # The plain sum also has to beat the worst rounding possible, u x sum|outputs| (+ half a
        # subnormal step per output; see the module docstring): min(z, threshold x z_worst) alarms
        # only if both do. 1.25: margin.
        worst = (_norm_fp32(Cb, ord=1).mul_(1.25 * u).add_(1.25 * width * h).unsqueeze(-1)
                 + floor[..., :1]).add_(abs_floor)
        zs[..., :1] = torch.minimum(zs[..., :1], diff[..., :1] / worst * z)
        zs = _judge_nonfinite(zs, expected, actual, noise, bound, C3.dtype)
        mz, arg = torch.max(zs.view(-1), 0)
        maxes.append(mz)
        # gather, not den.view(-1)[arg]: indexing with a GPU tensor makes the CPU wait for the GPU
        noises.append(den.view(-1).gather(0, arg.view(1)).squeeze(0))
        args.append(arg if off == 0 else arg + off)

    if len(maxes) == 1:
        max_z, arg, nz = maxes[0], args[0], noises[0]
    else:
        mz = torch.stack(maxes)
        best = mz.argmax().view(1)
        max_z = mz.gather(0, best).squeeze(0)
        arg = torch.stack(args).gather(0, best).squeeze(0)
        nz = torch.stack(noises).gather(0, best).squeeze(0)
    return CheckStat(max_z=max_z.float(), argmax=arg, noise=nz, spans=tuple(spans), m=m,
                     batched=C.dim() == 3)


def resolve(stats: Sequence[CheckStat], z: float = 6.0) -> list[CheckResult]:
    """Read queued checks back from the GPU (one transfer per device for all of them)."""
    if not stats:
        return []
    zs, locs, nzs = [0.0] * len(stats), [0] * len(stats), [0.0] * len(stats)
    by_device: dict = {}
    for i, s in enumerate(stats):
        by_device.setdefault(s.max_z.device, []).append(i)
    for idx in by_device.values():
        n = len(idx)
        packed = torch.cat([torch.stack([stats[i].max_z for i in idx]).double(),
                            torch.stack([stats[i].argmax for i in idx]).double(),
                            torch.stack([stats[i].noise for i in idx]).double()]).cpu().tolist()
        for j, i in enumerate(idx):
            zs[i], locs[i], nzs[i] = packed[j], packed[n + j], packed[2 * n + j]
    out = []
    for s, mz, flat, nz in zip(stats, zs, locs, nzs):
        flat = int(flat)
        span = s.spans[0]
        for sp in s.spans:
            if flat >= sp[3]:
                span = sp
        start, width, nblk, off = span
        blk_flat = (flat - off) // 2                       # two checksums per block
        b_i, rem = divmod(blk_flat, s.m * nblk)
        r_i, c_i = divmod(rem, nblk)
        finite = math.isfinite(mz)
        out.append(CheckResult(ok=finite and mz <= z, max_z=mz, nonfinite=not finite,
                               batch=b_i if s.batched else None, row=r_i,
                               col_block=start // s.spans[0][1] + c_i, noise=nz))
    return out


def check_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    block: int = 128,
    z: float = 6.0,
    abs_floor: float = ABS_FLOOR,
) -> Optional[CheckResult]:
    """Check C = beta*bias + alpha*(A @ B) and answer immediately. None if dtype unsupported."""
    if USE_FAST and _group.available(C) and _group.supports(A, B, C, bias, beta, block):
        try:
            r = _group.check_now(A, B, C, bias, alpha=alpha, beta=beta, block=block, abs_floor=abs_floor, z=z)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001  (compile or launch problem: plain check from now on)
            _group.fell_back(C.device.type, C.dtype, f"check of {tuple(A.shape)} @ {tuple(B.shape)}", e)
            r = None
        if r is not None:
            mz, b, row, blk, den = r
            finite = math.isfinite(mz)
            return CheckResult(ok=finite and mz <= z, max_z=mz, nonfinite=not finite,
                               batch=int(b) if C.dim() == 3 else None, row=int(row), col_block=int(blk), noise=den)
    st = matmul_zscores(A, B, C, bias, alpha=alpha, beta=beta, block=block, abs_floor=abs_floor, z=z)
    return None if st is None else resolve([st], z)[0]


def check_linear(x, weight, bias, y, **kw) -> Optional[CheckResult]:
    """Check y = torch.nn.functional.linear(x, weight, bias) for any leading dims."""
    return check_matmul(
        x.reshape(-1, x.shape[-1]), weight.t(), y.reshape(-1, y.shape[-1]), bias, **kw
    )
