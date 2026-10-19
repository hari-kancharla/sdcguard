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

import contextlib
import math
import os
import threading
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
DTYPE_MAX = {torch.bfloat16: float(torch.finfo(torch.bfloat16).max), torch.float16: 65504.0,
             torch.float32: float(torch.finfo(torch.float32).max)}
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
        """False when any checksum could not be judged and no other checksum found corruption.
        Non-finite inputs or legitimate overflow then leave an incomplete check, not a clean one."""
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


# Training checks can also take a new pattern on every checked step (SDCGuard(fresh_signs=True), the default):
# epoch e > 0 is a different pattern. Two outputs of a block swapped where the pattern has the same sign cancel
# in both sums; with a new pattern each checked step, a fault that keeps doing that is seen on about half the
# checked steps (each pattern separates a given pair with probability 1/2) instead of never. It can still go
# unseen for several steps in a row. Serving computes its checksums once at load, so it stays on epoch 0.


_CACHE_LOCK = threading.RLock()     # the pattern caches are shared by every thread (one per GPU in autograd)


@contextlib.contextmanager
def sign_seed(seed: Optional[int]):
    """Use another process's sign patterns for a while (a replay judging a recorded output with the patterns
    that flagged it). It swaps a process-wide setting: for one-off tools such as `sdcguard replay`, not while
    other threads are checking matmuls. The Triton checks keep their own copy of the patterns, so use it with
    CPU tensors."""
    global _SIGN_SEED
    if seed is None:
        yield
        return
    with _CACHE_LOCK:
        old = _SIGN_SEED
        _SIGN_SEED = int(seed)
        _SIGNS.clear()
        _PAIR_W.clear()
    try:
        yield
    finally:
        with _CACHE_LOCK:
            _SIGN_SEED = old
            _SIGNS.clear()
            _PAIR_W.clear()


def _drop_old(cache: dict, epoch: int) -> None:
    """Keep epoch 0 and the current epoch only (the caches would grow by one pattern per step otherwise).
    Call with _CACHE_LOCK held."""
    if epoch:
        for k in [k for k in list(cache) if k[-1] not in (0, epoch)]:
            cache.pop(k, None)


def _signs(width: int, device, epoch: int = 0) -> torch.Tensor:
    """A pseudo-random +-1 pattern (fp32) for the second checksum: one per process and epoch.
    Epoch 0 draws each width on its own; a later epoch draws one pattern and narrower blocks use its start
    (one draw per checked step instead of one per block width)."""
    key = (width, str(device), epoch)
    out = _SIGNS.get(key)
    if out is not None:
        return out
    with _CACHE_LOCK:
        out = _SIGNS.get(key)
        if out is not None:
            return out
        _drop_old(_SIGNS, epoch)
        if epoch == 0:
            g = torch.Generator().manual_seed((_SIGN_SEED + width) % (2 ** 63))
            s = (torch.randint(0, 2, (width,), generator=g) * 2 - 1).float()
        else:
            n = max(128, width)
            base = _SIGNS.get(("base", n, epoch))
            if base is None:
                g = torch.Generator().manual_seed((_SIGN_SEED + n + 7919 * epoch) % (2 ** 63))
                base = _SIGNS[("base", n, epoch)] = (torch.randint(0, 2, (n,), generator=g) * 2 - 1).float()
            s = base[:width]
        out = _SIGNS[key] = s.to(device)
    return out


def _c_gemm(flat: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.mm.dtype(flat, W, torch.float32)


_PAIR_W: dict = {}


def _pair_weights(width: int, dtype: torch.dtype, device, epoch: int = 0) -> torch.Tensor:
    """[1, s] as a (width, 2) matrix in `dtype` (exact: every entry is +-1). Cached."""
    key = (width, dtype, str(device), epoch)
    w = _PAIR_W.get(key)
    if w is None:
        with _CACHE_LOCK:
            w = _PAIR_W.get(key)
            if w is None:
                _drop_old(_PAIR_W, epoch)
                s = _signs(width, device, epoch)
                w = torch.stack([torch.ones_like(s), s], dim=1).to(dtype)
                _PAIR_W[key] = w
    return w


def _pair_sums(Xb: torch.Tensor, epoch: int = 0) -> torch.Tensor:
    """Plain and signed sums of every block, in fp32: (..., nblk, width) -> (..., nblk, 2).

    On NVIDIA GPUs both come from one pass (a skinny GEMM against [1, s] with fp32 output);
    elsewhere, two reductions.
    """
    width = Xb.shape[-1]
    if Xb.dtype in UNIT_ROUNDOFF and _use_lowp_gemm(Xb):
        try:
            Xc = Xb if Xb.is_contiguous() else Xb.contiguous()
            out = _c_gemm(Xc.view(-1, width), _pair_weights(width, Xb.dtype, Xb.device, epoch))
            return out.view(*Xb.shape[:-1], 2)
        except (RuntimeError, NotImplementedError, AttributeError):
            pass
    s = _pair_weights(width, Xb.dtype, Xb.device, epoch)[:, 1]
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


UNJUDGED = -1.0     # incomplete check: at least one checksum unjudged, and no other checksum found corruption


def _complete_score(max_z: torch.Tensor, unknown: list[torch.Tensor], z: float) -> torch.Tensor:
    """Keep a detected failure; otherwise any unjudged checksum makes the whole check incomplete.

    Taking only a maximum drops negative UNJUDGED scores whenever another row is finite. Reducing the
    coverage flag separately preserves it across rows, blocks and remainder spans, without a host read.
    """
    incomplete = unknown[0] if len(unknown) == 1 else torch.stack(unknown).any()
    return torch.where(incomplete & (max_z <= z), UNJUDGED, max_z)


def _judge_nonfinite(z, expected, actual, noise, bound, dtype, cnorm=None, width: int = 1, step: float = 0.0,
                     k: int = 0):
    """Judging Inf/NaN (see the module docstring). An expected side that isn't finite can't be
    judged (non-finite inputs, or values too large for fp32 sums): z = UNJUDGED. An Inf/NaN output, or
    one too large to square in fp32, is corruption (z = inf) only if the inputs can't produce it; if they
    can (fp16 loss-scaling overflow, say), it can't be judged either.

    cnorm (the block's L2 norm), width and step (one subnormal step of the output dtype): finite outputs too
    large for the inputs are corruption too. Every output of the block is at most `bound` (plus rounding), so
    the block's norm is at most that times sqrt(width). Two huge outputs of opposite sign that cancel in both
    checksums (+1e18 and -1e18 where the pattern has the same sign) are caught this way (independent review,
    Oct 19, 2026)."""
    e_bad = ~torch.isfinite(expected)
    a_bad = ~torch.isfinite(actual) & ~e_bad
    big = ~torch.isfinite(noise) & ~e_bad & ~a_bad
    lim = bound * 1.01
    z = torch.where(z == z, z, UNJUDGED)                # inf/inf (both sides huge): nothing to judge
    z = torch.where(e_bad, UNJUDGED, z)
    if cnorm is not None:
        cn = cnorm.unsqueeze(-1) if cnorm.dim() == z.dim() - 1 else cnorm
        # (each output may also be off by the accumulation's worst-case rounding, k x 2^-22 of the bound)
        cap = bound * (1.02 + k * 2.0 ** -22) + step
        imposs = ~e_bad & ~a_bad & ~big & (lim < BIG) & (cn > cap * math.sqrt(width))
        z = torch.where(imposs, float("inf"), z)
    z = torch.where(a_bad, torch.where(lim < min(DTYPE_MAX[dtype], BIG), float("inf"), UNJUDGED), z)
    return torch.where(big, torch.where(lim < BIG, float("inf"), UNJUDGED), z)


def _pad_empty_k(A: torch.Tensor, B: torch.Tensor):
    """A @ B with k = 0 (an MoE expert that got no tokens, in a weight gradient) is all zeros: check it as a
    product with one zero column, so C must equal the bias (or zero)."""
    if A.shape[-1] == 0:
        A = A.new_zeros(*A.shape[:-1], 1)
        B = B.new_zeros(*B.shape[:-2], 1, B.shape[-1])
    return A, B


def is_empty(C: torch.Tensor) -> bool:
    """An output with no elements (m = 0 or n = 0): nothing to check."""
    return C.numel() == 0


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
    epoch: int = 0,
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
    if is_empty(C):
        raise ValueError("empty output: nothing to check (callers skip these, see is_empty)")
    A, B = _pad_empty_k(A, B)
    # backward passes often run under autocast, which would round this fp32 math to bf16/fp16 on CPUs
    # and Apple GPUs
    with torch.autocast(device_type=C.device.type, enabled=False):
        return _zscores(A, B, C, bias, alpha, beta, block, abs_floor, z, u, epoch)


def _zscores(A, B, C, bias, alpha, beta, block, abs_floor, z, u, epoch=0) -> CheckStat:
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
    maxes, args, noises, unknown = [], [], [], []
    for start, width, nblk, off in spans:
        end = start + width * nblk
        Cb = C3[..., start:end].reshape(nb_batch, m, nblk, width)          # a view when possible
        Bblk = B3[..., start:end].reshape(B3.shape[0], k, nblk, width)
        # Two checksums per block: the plain sum, and a sum with fixed random +-1 signs. A fault
        # whose changes happen to cancel in one sum (e.g. zeroing a +3 and a -3) shows in the other.
        B2 = _pair_sums(Bblk, epoch)                                         # (bB, k, nblk, 2) fp32
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
            elif bias.dim() <= 2:                                            # (m, n) or (1, n)
                bb = torch.broadcast_to(bias, (m, n))[:, start:end].reshape(m, nblk, width)
            else:                                                            # (batch, m, n) and friends: baddbmm
                bb = torch.broadcast_to(bias, (nb_batch, m, n))[..., start:end].reshape(nb_batch, m, nblk, width)
            expected = expected.add(_pair_sums(bb, epoch), alpha=beta)
            bound = bound + abs(beta) * bb.abs().amax(-1).float()             # (nblk,), (m, nblk) or (b, m, nblk)

        actual = _pair_sums(Cb, epoch)                                       # (batch, m, nblk, 2)
        # Rounding noise of a block sum ~ unit roundoff x L2 norm of the outputs (same for +-1
        # signs), plus a floor for the fp32 rounding of `expected` itself (matters only when the
        # outputs nearly cancel): 1e-5 x |alpha| x |a_row| x |checksum column of B|.
        cnorm = _norm_fp32(Cb)                                               # (batch, m, nblk)
        noise = (cnorm * u).unsqueeze(-1)                                    # (batch, m, nblk, 1)
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
        zs = _judge_nonfinite(zs, expected, actual, noise, bound, C3.dtype, cnorm, width, 2 * h, k)
        unknown.append((zs < 0).any())
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
    max_z = _complete_score(max_z, unknown, z)
    return CheckStat(max_z=max_z.float(), argmax=arg, noise=nz, spans=tuple(spans), m=m,
                     batched=C.dim() == 3)

# --- checks of fp32 accumulators (SDCGuard(precise=True)) -----------------------------------------------
#
# In precise mode a bf16/fp16 linear layer's matmul keeps its fp32 result (the tensor cores' accumulator)
# long enough to check it, and only then rounds it to bf16/fp16. That result has 16 more bits than the
# rounded one, so the check sees far smaller errors (AEGIS, OSDI '26, found its production incidents with
# checksums over fp32 accumulators, not over bf16 outputs).
#
# Both sides of the identity are added up in float64 (the inputs are bf16/fp16, so A @ (block sums of B)
# in float64 is all but exact), and the noise is the fp32 accumulation error of the result alone:
#     u * (C1 sqrt(k) |alpha| |a_row| |B_block| + (C2 sqrt(k) + C3 k / N + sqrt(width)) |C_block|),  u = 2^-24
# The first term is the random part of the accumulation's rounding: every one of the k products is aligned
# and truncated (tensor cores) or rounded (FMA) against the running sum, an error of up to one fp32 ulp of
# the running sum with a standard deviation of ulp / sqrt(12) ~ 0.58 u |running sum|, and no running sum can
# exceed |a_row| |b_column| (Cauchy-Schwarz). C1 = 0.6. The k / N term is the bias of truncating once per
# tensor-core step (N products, modes.n_fma); sqrt(width): rounding of the fp32 outputs themselves.
# Cauchy-Schwarz bounds the exact partial sums, but the sqrt(k) noise scaling additionally assumes
# sufficiently independent rounding errors. This is a statistical estimate requiring held-out validation,
# not a deterministic error bound or a universal false-alarm guarantee.
#
# Until Oct 19, 2026 the first term had no sqrt(k): it assumed the running sums stay near the final output.
# On real training (backward dW of the attention input and output layers, 4,096 tokens) they don't, and the
# check raised 72 to 84 false alarms in 2,970 clean checks on an A100, H100 and B200
# (results/precise_models). The term above was chosen from those measurements and the rounding argument,
# then frozen before the held-out run (experiments/precise_models.py --validate). Precise mode stays opt-in
# and experimental until that run passes. The plain sum also has to beat the same bound with |C_block|
# summed instead of squared (outputs whose errors line up), as in matmul_zscores. float64 is fast on
# data-center GPUs (A100, H100, B200) and slow on most others, another reason precise mode is opt-in.
ACC_UNIT = 2.0 ** -24
ACC_C1, ACC_C2, ACC_C3 = 0.6, 1.0, 1.0
_F64_CHUNK = 1 << 24          # elements converted to float64 at a time


def acc_supported(A: torch.Tensor, C: torch.Tensor) -> bool:
    return C.dtype == torch.float32 and A.dtype in UNIT_ROUNDOFF


def _pair_sums64(Xb: torch.Tensor, epoch: int = 0) -> torch.Tensor:
    """Plain and signed block sums in float64: (..., nblk, width) -> (..., nblk, 2), in chunks."""
    width = Xb.shape[-1]
    w = _pair_weights(width, torch.float64, Xb.device, epoch)
    flat = Xb.reshape(-1, width)
    out = torch.empty(flat.shape[0], 2, dtype=torch.float64, device=Xb.device)
    rows = max(1, _F64_CHUNK // max(1, width))
    for i in range(0, flat.shape[0], rows):
        out[i:i + rows] = flat[i:i + rows].double() @ w
    return out.view(*Xb.shape[:-1], 2)


def _matmul64(A3: torch.Tensor, Bb: torch.Tensor) -> torch.Tensor:
    """A3 ((bA, m, k), any dtype) @ Bb ((bB, k, c), float64) in float64, A converted in row chunks."""
    m, k = A3.shape[-2], A3.shape[-1]
    rows = max(1, _F64_CHUNK // max(1, A3.shape[0] * k))
    if rows >= m:
        return torch.matmul(A3.double(), Bb)
    return torch.cat([torch.matmul(A3[:, i:i + rows].double(), Bb) for i in range(0, m, rows)], dim=-2)


def matmul_zscores_acc(
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
    nfma: Optional[int] = None,
    epoch: int = 0,
) -> Optional[CheckStat]:
    """Queue the check of C = beta*bias + alpha*(A @ B) where A and B are bf16/fp16 and C is the fp32
    result (not rounded). Same layout and read-back as matmul_zscores (resolve() decodes both). None if
    the dtypes don't fit. nfma: tensor-core products per truncation step (None: no tensor cores)."""
    if not acc_supported(A, C):
        return None
    if is_empty(C):
        raise ValueError("empty output: nothing to check (callers skip these, see is_empty)")
    A, B = _pad_empty_k(A, B)
    with torch.autocast(device_type=C.device.type, enabled=False):     # its fp32/fp64 math stays as written
        return _acc_zscores(A, B, C, bias, alpha, beta, block, abs_floor, z, nfma, epoch)


def _acc_zscores(A, B, C, bias, alpha, beta, block, abs_floor, z, nfma, epoch=0) -> CheckStat:
    A3, B3, C3 = _as_batched(A), _as_batched(B), _as_batched(C)
    nb_batch, m, n = C3.shape
    k = A3.shape[-1]
    if A3.shape[-2] != m or B3.shape[-2] != k or B3.shape[-1] != n:
        raise ValueError(f"shape mismatch: A{tuple(A.shape)} B{tuple(B.shape)} C{tuple(C.shape)}")
    u = ACC_UNIT
    steps = k / nfma if nfma else 0.0

    bs = max(1, min(block, n))
    spans, offset = [], 0
    for start, width, nblk in ((0, bs, n // bs), ((n // bs) * bs, n % bs, 1)):
        if nblk and width:
            spans.append((start, width, nblk, offset))
            offset += max(nb_batch, A3.shape[0], B3.shape[0]) * m * nblk * 2

    A_norm = _norm_fp32(A3).view(A3.shape[0], m, 1)                          # (bA, m, 1)
    A_l1 = _norm_fp32(A3, ord=1).view(A3.shape[0], m, 1)
    maxes, args, noises, unknown = [], [], [], []
    for start, width, nblk, off in spans:
        end = start + width * nblk
        Cb = C3[..., start:end].reshape(nb_batch, m, nblk, width)
        Bblk = B3[..., start:end].reshape(B3.shape[0], k, nblk, width)
        B2 = _pair_sums64(Bblk, epoch)                                       # (bB, k, nblk, 2) float64
        expected = _matmul64(A3, B2.view(B3.shape[0], k, 2 * nblk)).view(-1, m, nblk, 2)
        bound = A_l1 * Bblk.abs().amax(dim=(-3, -1)).float().unsqueeze(-2)  # (b, m, nblk)
        if alpha != 1.0:
            expected = expected * alpha
            bound = bound * abs(alpha)
        if bias is not None and beta != 0:
            if bias.dim() == 1:
                bb = bias[start:end].reshape(nblk, width)
            elif bias.dim() <= 2:
                bb = torch.broadcast_to(bias, (m, n))[:, start:end].reshape(m, nblk, width)
            else:
                bb = torch.broadcast_to(bias, (nb_batch, m, n))[..., start:end].reshape(nb_batch, m, nblk, width)
            expected = expected + beta * _pair_sums64(bb, epoch)
            bound = bound + abs(beta) * bb.abs().amax(-1).float()

        actual = _pair_sums64(Cb, epoch)                                     # (b, m, nblk, 2) float64
        cnorm = _norm_fp32(Cb)                                               # (b, m, nblk)
        b_frob = torch.linalg.vector_norm(Bblk.float(), dim=(-3, -1)).unsqueeze(-2)   # (bB, 1, nblk)
        per_out = ACC_C2 * math.sqrt(k) + ACC_C3 * steps + math.sqrt(width)
        prod = (A_norm * b_frob).mul_(ACC_C1 * math.sqrt(k) * abs(alpha) * u)  # (b, m, nblk)
        prod = torch.where(torch.isfinite(prod), prod, bound * (u * 128))
        noise = (cnorm * (per_out * u) + prod).unsqueeze(-1)                 # (b, m, nblk, 1)
        den = (noise + abs_floor).double()
        diff = (expected - actual).abs_()
        zs = diff / den
        worst = (_norm_fp32(Cb, ord=1).mul_(per_out * u) + prod).unsqueeze(-1).add_(abs_floor).double()
        zs[..., :1] = torch.minimum(zs[..., :1], diff[..., :1] / worst * z)
        zs = _judge_nonfinite(zs.float(), expected, actual, noise, bound.unsqueeze(-1), torch.float32, cnorm, width,
                              0.0, k)
        unknown.append((zs < 0).any())
        mz, arg = torch.max(zs.view(-1), 0)
        maxes.append(mz)
        noises.append(den.float().expand_as(zs).reshape(-1).gather(0, arg.view(1)).squeeze(0))
        args.append(arg if off == 0 else arg + off)

    if len(maxes) == 1:
        max_z, arg, nz = maxes[0], args[0], noises[0]
    else:
        mz = torch.stack(maxes)
        best = mz.argmax().view(1)
        max_z = mz.gather(0, best).squeeze(0)
        arg = torch.stack(args).gather(0, best).squeeze(0)
        nz = torch.stack(noises).gather(0, best).squeeze(0)
    max_z = _complete_score(max_z, unknown, z)
    return CheckStat(max_z=max_z.float(), argmax=arg, noise=nz, spans=tuple(spans), m=m,
                     batched=C.dim() == 3)


def check_matmul_acc(A, B, C, bias=None, *, z: float = 6.0, nfma: Optional[int] = None, **kw) -> Optional[CheckResult]:
    """matmul_zscores_acc, answered immediately (tests and one-off use)."""
    if is_empty(C) and acc_supported(A, C):
        return CheckResult(ok=True, max_z=0.0, nonfinite=False)
    st = matmul_zscores_acc(A, B, C, bias, z=z, nfma=nfma, **kw)
    return None if st is None else resolve([st], z)[0]


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
    if is_empty(C) and C.dtype in UNIT_ROUNDOFF:
        return CheckResult(ok=True, max_z=0.0, nonfinite=False)       # nothing that could be wrong
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
