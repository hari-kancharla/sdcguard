"""Sampled higher-precision verification of linear layers in serving (ServeGuard.spot_checks(), SDCGUARD_SPOT=1).

The checksum check of a bf16/fp16 linear layer compares block sums of its rounded output with block sums of the
recomputed product, so its noise is the rounding of 128 outputs at once: it sees an output off by about a quarter of
its block's noise level, gross errors, not small ones (docs/METHOD.md: on training matmuls it flagged half of single
wrong values at about 29% of a typical output in their block, and none of 594 at 10%). This check
looks at far fewer outputs, but each one exactly: at each model step one of the registered layers (they take turns:
the layer whose index is the step number modulo the number of layers) recomputes SPOT_COLS output columns for up to
SPOT_ROWS of the step's real rows (all of them when there are fewer) from the layer's own input and weights, with
fp32 products (exact: bf16 and fp16 products fit in fp32) summed in fp32 with compensation (Kahan), and compares
each with the served output:

    z = |y - r| / (u |r| + b |r - bias| + A),   A = S u32 (C1 sqrt(K) |x_row| |w_col| + (sqrt(K) + K / NFMA + 1) |r|)

u is the output dtype's unit roundoff (2^-8 for bf16, 2^-11 for fp16): the served output is rounded once, so a
correct one is within u |r| of the exact result, give or take the served matmul's fp32 accumulation, whose error A
bounds (precise mode's accumulation model, docs/METHOD.md, with S = 6 times its margin: the random part of K fp32
roundings against running sums no larger than |x_row| |w_col|, and a truncation bias once per NFMA products); b is u
when the layer adds a bias (if the bias were added after rounding, a second rounding). The line is SPOT_Z = 2. So a
fault that moves the exact result by e before it is rounded is always over the line when
|e| > 3u |r| + 3b |r - bias| + 2.2 A (2u for the line, u for the rounding, b twice for the line and once for a
possible second rounding, and the accumulation's allowance twice plus the error it allows): for Gaussian-like data, a
median output is always caught above 1.3% (bf16) and 0.27% (fp16) at K = 1,536 and above 1.9% and 0.87% at
K = 8,960, about 1.2 points more in bf16 when the layer adds a small bias (2.5% at K = 1,536; min_caught below
computes it for any output); smaller faults are caught when the rounding falls their way.
These floors are the bound's arithmetic in the outputs sampled; the GPU run measures what it catches. An alert
names the layer, the row and the column.

Rows: the step's real rows when the plugin knows them (mark_step's rows, packed above the step number in the step
buffer), all of them when there are at most SPOT_ROWS, else SPOT_ROWS evenly spaced ones that move by one each turn,
so CUDA-graph padding rows aren't judged and every real row is covered in turn. Columns: SPOT_COLS evenly spaced ones
(N // SPOT_COLS apart) whose offset moves each turn by col_step(N), a step near the golden ratio of that spacing and
coprime with N and with small continued-fraction terms: a layer's N columns are all covered within N / SPOT_COLS
turns when SPOT_COLS divides N (within N turns otherwise), and a fault that touches a band of w adjacent columns is
sampled about once every N / (w x SPOT_COLS) turns, never more than about twice that apart for the layer widths
tested (moving the offset by one would sample it in one run of w turns every N / SPOT_COLS turns).

What it catches: a wrong output among the sampled ones, down to the floor above; a fault that touches many outputs
of a layer (a wrong kernel path, an SM computing its tiles wrong) is sampled within a few turns. What it doesn't:
an output it doesn't sample (each turn looks at SPOT_COLS x up to SPOT_ROWS outputs of one layer); a weight changed
in memory (the served matmul and the recompute read the same weights: the weight scan's job); layers it doesn't
register (quantized ones, and those the checksum check doesn't take), and layers it registers but doesn't check:
those on a device where the bound doesn't hold (bound_holds) or, on a GPU, no kernel is usable (their calls return
at once, so the traced graph is the same either way, though outside CUDA graphs each call still costs a few
microseconds). The first output over the line on a device in a read is kept (its layer, input row, weight row and
bias, copied on the GPU), read back with the read, and rechecked in float64 on the CPU.

One Triton kernel per registered layer per step, SPOT_COLS x (SPOT_ROWS / SPOT_RB) programs that return at once
unless their layer is the active one; outside CUDA-graph capture, a call whose layer isn't active is skipped on the
host (the host knows the step mark_step wrote), so only CUDA graphs carry the inactive layers' launches. The PyTorch
version (always computed, the result kept only for the active layer, no host sync) is the reference; on GPUs it runs
in place of the kernel only with SDCGUARD_SPOT_TORCH=1 (it adds about 40 operations to every layer's call, which
CUDA graphs would replay every step), or after a launch of the kernel failed.
"""
from __future__ import annotations

import functools
import math
import os
import threading
from typing import Optional

import torch

from .events import SDCEvent, device_identity
from .serve_norm import PLACE_MAXIMA, dtype_consts, pick

SPOT_Z = 2.0
SPOT_COLS = 8            # output columns recomputed per turn
SPOT_ROWS = 32           # rows recomputed per turn (all of the step's real rows when fewer)
SPOT_RB = 8              # rows per program
assert SPOT_ROWS % SPOT_RB == 0
BK = 128                 # inputs per step of a program's loop
ACC_UNIT = 2.0 ** -24
ACC_C1 = 0.6
SAFETY = 6.0
NFMA = 8                 # products per truncation of the tensor cores' accumulation (conservative)
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
M32 = 0xFFFFFFFF
ROWS_SHIFT = 32
ROWS_MAX = (1 << 31) - 1
COLS = 8                 # z|row, z|col, z|bits(y), z|bits(r), z|step, active calls, outputs judged, unjudged
C_ROW, C_COL, C_OUT, C_REF, C_STEP, C_CALLS, C_JUDGED, C_UNJ = range(COLS)
CAP_COLS = 7             # claimed, row, col, step, score bits, bias present, slot
SPOT_SCHEMA = 2          # the op's arguments (the vLLM plugin keys its compile cache on it); 2: one kept output a
                         # device, the column step and y's column stride

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}
_lib_lock = threading.Lock()
_OP: list = [None]
_ON: list = [None]       # _group._on, imported at first use
# the packed step value mark_step / end_step last wrote to each step buffer (by its address), and the layers whose
# kernel has run once (by step buffer address and slot): a call whose layer isn't active is skipped on the host
# outside CUDA-graph capture, once its kernel has run (so its Triton variant was compiled before any capture)
_HOST: dict = {}
_SEEN: set = set()
# (step buffer address, dtype) of layers registered but not spot-checked (the bound doesn't hold, or no usable kernel
# on a GPU): their calls return at once. They keep their arguments, so a server's traced graph is the same whatever
# finalize found, and vLLM's compile cache (keyed by settings, not by what finalize found) can't load a graph that
# disagrees with it.
_DISABLED: set = set()
PLACE_KEPT = ("the kept output, rechecked on the CPU (the first output over the line on its GPU in the read; max_z is "
              "the read's largest score, not necessarily this output's)")


def pack_step(step: int, rows: Optional[int] = None) -> int:
    """The step buffer's value: the step's real rows (0: unknown) above the step number."""
    r = int(rows) if rows is not None and int(rows) > 0 else 0
    return (min(r, ROWS_MAX) << ROWS_SHIFT) | (int(step) & M32)


def _f32(bits: int) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def acc_term(k: int, norm: torch.Tensor, ref_abs: torch.Tensor) -> torch.Tensor:
    """A: S times the model of an fp32 accumulation's error over K products (norm = |x_row| |w_col|)."""
    sk = math.sqrt(max(k, 1))
    return SAFETY * ACC_UNIT * (ACC_C1 * sk * norm + (sk + k / NFMA + 1.0) * ref_abs)


def min_caught(dtype: torch.dtype, k: int, norm: float, ref: float, bias: Optional[float] = None) -> float:
    """The smallest |e| (a change of the exact result before rounding) that is always over the line for an output
    whose exact value is ref (bias included) with |x_row| |w_col| = norm: |y - r| >= |e| (1 - u) - u |r| - |a| (1 + u)
    with the served accumulation's own error |a| at most A / S; with a bias, in case it was added after a first
    rounding, |y - r| >= |e| (1 - 2u) - u |r| - u |r - bias| - |a| (1 + 2u). Against SPOT_Z times the score's
    denominator."""
    u, sp, _ = dtype_consts(dtype)
    a = float(acc_term(k, torch.tensor(float(norm)), torch.tensor(abs(ref))))
    second = u * abs(ref - bias) if bias is not None else 0.0
    n_round = 2 if bias is not None else 1                 # (each rounding scales e and a by up to u)
    den = u * abs(ref) + second + a + sp
    return (SPOT_Z * den + u * abs(ref) + second + (1 + n_round * u) * a / SAFETY) / (1 - n_round * u)


def score_all(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    """Every output's score against a float64 recompute, as the check scores its samples (for tests and runs):
    x [M, K], w [N, K], b [N] or None, y [M, N] (the served output)."""
    xd, wd = x.double(), w.double()
    acc = xd @ wd.t()
    ref = acc + (b.double()[None, :] if b is not None else 0.0)
    norm = xd.norm(dim=1)[:, None] * wd.norm(dim=1)[None, :]
    u, sp, _ = dtype_consts(y.dtype)
    den = u * ref.abs() + (u * acc.abs() if b is not None else 0.0) + acc_term(x.shape[1], norm, ref.abs()) + sp
    return (y.double() - ref).abs() / den


def _max_cf_term(p: int, q: int) -> int:
    """The largest term of p / q's continued fraction after its integer part (0 < p < q): small terms mean the
    multiples of p modulo q spread evenly (the golden ratio's are all 1)."""
    best, num, den = 0, p, q
    while num:
        a = den // num
        best = max(best, a)
        den, num = num, den - a * num
    return best


@functools.lru_cache(maxsize=None)
def col_step(n: int) -> int:
    """How far the sampled columns' offset moves each turn: an integer near the golden ratio times their spacing
    s = n // SPOT_COLS, sharing no factor with n, whose ratio to s has continued-fraction terms of at most 2 (the
    nearest with the smallest largest term, if none within max(s / 4, 4) does; if that term is over 4, as at small
    widths with few steps below s that share no factor with n, the step below n whose remainder modulo s has the
    smallest largest term; for s under 5,
    the coprime integer nearest n times the golden ratio). Every offset in [0, n) comes once in n turns;
    modulo s (which decides the columns when SPOT_COLS divides n) the offsets are multiples of the step, which small
    terms spread evenly: the nearest coprime integer alone can sit next to a fraction like 8/13 (n = 4,096 gave 315
    of 512) and revisit nearly the same columns every 13 turns."""
    if n <= 2:
        return 1
    s = max(n // SPOT_COLS, 1)
    if s < 5:
        c = max(1, int(round(n * GOLDEN)))
        return next(p for d in range(n) for p in (c - d, c + d) if 1 <= p < n and math.gcd(p, n) == 1)
    c = int(round(s * GOLDEN))
    best = None
    for d in range(max(4, s // 4)):
        for p in (c - d, c + d):
            if 1 <= p < s and math.gcd(p, n) == 1:
                t = _max_cf_term(p, s)
                if t <= 2:
                    return p
                if best is None or t < best[0]:
                    best = (t, p)
    if best is None or best[0] > 4:                       # (small widths: a step above s acts as its remainder)
        for p in range(1, n):
            if p % s and math.gcd(p, n) == 1:
                t = _max_cf_term(p % s, s)
                if best is None or t < best[0]:
                    best = (t, p)
    return best[1] if best is not None else 1


def sampled(step: torch.Tensor, nl: int, m, n: int) -> tuple:
    """(rows [SPOT_ROWS], row mask, cols [SPOT_COLS]) for a step, as the kernel draws them; m: the rows samples come
    from (a 0-d tensor or an int)."""
    m = torch.as_tensor(m, device=step.device)
    t = step // max(nl, 1)
    i = torch.arange(SPOT_ROWS, device=step.device)
    mm = torch.clamp(m, min=1)
    rows = torch.where(m <= SPOT_ROWS, i, (t + i * torch.clamp(mm // SPOT_ROWS, min=1)) % mm)
    rmask = i < torch.clamp(m, max=SPOT_ROWS)
    j = torch.arange(SPOT_COLS, device=step.device)
    off = ((t % n) * col_step(n)) % n
    cols = (off + j * max(n // SPOT_COLS, 1)) % n
    return rows, rmask, cols


def bound_holds(device, dtype) -> Optional[str]:
    """Why the score's bound doesn't hold for this dtype's served matmuls on this device, or None. It assumes fp32
    accumulation with fp32 split-K reductions: PyTorch's reduced-precision reductions (SDCGUARD_EXACT=0 leaves them
    on) sum split-K partials in bf16 or fp16 (correct outputs then score up to 26 at K = 8,960), and fp16 accumulation
    sums in fp16."""
    if torch.device(device).type != "cuda":
        return None
    m = torch.backends.cuda.matmul

    def on(name) -> bool:
        v = getattr(m, name, False)
        return any(bool(x) for x in v) if isinstance(v, (tuple, list)) else bool(v)
    if dtype == torch.bfloat16 and on("allow_bf16_reduced_precision_reduction"):
        return "PyTorch's reduced-precision bf16 reductions are on (SDCGUARD_EXACT=0?): split-K partials summed in bf16"
    if dtype == torch.float16 and on("allow_fp16_reduced_precision_reduction"):
        return "PyTorch's reduced-precision fp16 reductions are on (SDCGUARD_EXACT=0?): split-K partials summed in fp16"
    if dtype == torch.float16 and on("allow_fp16_accumulation"):
        return "PyTorch's fp16 accumulation is on: fp16 matmuls sum in fp16"
    return None


def spot_check_torch(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
                     stepbuf: torch.Tensor, state: torch.Tensor, slot: int, nl: int, idx: int, cap_meta: torch.Tensor,
                     cap_x: torch.Tensor, cap_w: torch.Tensor, cap_v: torch.Tensor, line: float) -> None:
    """The check in PyTorch operations (no host sync: safe in CUDA graphs). It always computes; only the active
    layer's result counts. x [M, K], y [M, N], w [N, K]."""
    m, k = x.shape
    n = w.shape[0]
    if m == 0:
        return
    u, sp, fm = dtype_consts(y.dtype)
    raw = stepbuf.reshape(-1)[0]
    step = raw & M32
    active = (step % max(nl, 1)) == idx
    real = (raw >> ROWS_SHIFT) & ROWS_MAX
    mr = torch.where(real > 0, torch.clamp(real, max=m), torch.full_like(real, m))
    rows, rmask, cols = sampled(step, nl, mr, n)
    rows = torch.clamp(rows, max=m - 1)
    xr = x.index_select(0, rows).float()                              # [R, K]
    wc = w.index_select(0, cols).float()                              # [C, K]
    acc = (xr.double() @ wc.double().t()).float()                     # (the kernel's compensated fp32 sum, closer)
    ref = acc + (b.index_select(0, cols).float()[None, :] if b is not None else 0.0)
    yv = y.index_select(0, rows).index_select(1, cols).float()
    # the norms in float64: in float32 a row of large values squares to infinity (or a column of small ones to
    # zero), and an infinite bound would score a wrong output 0, as if right
    norm = (xr.double().square().sum(1).sqrt()[:, None] * wc.double().square().sum(1).sqrt()[None, :]).float()
    den = u * ref.abs() + (u * acc.abs() if b is not None else 0.0) + acc_term(k, norm, ref.abs()) + sp
    z = (yv - ref).abs() / den
    z = torch.where(z == z, z, torch.full_like(z, float("inf")))
    # judged only where the reference and its bound are finite: anything else is counted unjudged, never clean
    ok = rmask[:, None] & torch.isfinite(ref) & (ref.abs() <= fm) & torch.isfinite(den)
    z = torch.where(ok & active, z, torch.zeros_like(z))
    zmax, arg = z.reshape(-1).max(0)
    ri, ci = arg // SPOT_COLS, arg % SPOT_COLS
    row, col = pick(rows, ri), pick(cols, ci)
    bits = lambda v: v.reshape(1).view(torch.int32).to(torch.int64).reshape(()) & M32  # noqa: E731
    zb = bits(zmax) << 32
    keys = torch.stack([zb | row, zb | col, zb | bits(pick(yv.reshape(-1), arg)), zb | bits(pick(ref.reshape(-1), arg)),
                        zb | step])
    keys = torch.where(active & (zmax > 0), keys, torch.zeros_like(keys))
    srow = state[slot]
    srow[:5] = torch.maximum(srow[:5], keys)
    a64 = active.to(torch.int64)
    srow[C_CALLS] += a64
    srow[C_JUDGED] += a64 * ok.sum()
    srow[C_UNJ] += a64 * (rmask[:, None] & ~ok).sum()
    take = active & (zmax > line) & (cap_meta[0] == 0)
    cap_x[:k] = torch.where(take, pick(xr, ri), cap_x[:k])
    cap_w[:k] = torch.where(take, pick(wc, ci), cap_w[:k])
    bias_v = b.index_select(0, col.reshape(1)).float().reshape(()) if b is not None else torch.zeros((), device=x.device)
    cap_v[:2] = torch.where(take, torch.stack([pick(yv.reshape(-1), arg), bias_v]), cap_v[:2])
    rec = torch.stack([torch.ones_like(row), row, col, step, zb >> 32, torch.full_like(row, 1 if b is not None else 0),
                       torch.full_like(row, slot)]).to(cap_meta.dtype)
    cap_meta.copy_(torch.where(take, rec, cap_meta))


if HAVE_TRITON:
    # M (the step's rows) changes with every batch, and slot, idx and the column step with every layer: nothing may
    # depend on their values, so the variants compiled before CUDA graphs are captured cover every call
    @triton.jit(do_not_specialize=["M", "slot", "nl", "idx", "pstep"])
    def _spot_kernel(x, y, w, b, stepbuf, state, capm, capx, capw, capv, M, N, K, sx, sy, syc, sw, slot, nl, idx,
                     pstep, u, spacing, fmax, line, ka, kb, NCOLS: tl.constexpr, HAS_BIAS: tl.constexpr,
                     R_: tl.constexpr, RB: tl.constexpr, C_: tl.constexpr, BK_: tl.constexpr):
        cj = tl.program_id(0)
        rb = tl.program_id(1)
        raw = tl.load(stepbuf)
        step = raw & 0xFFFFFFFF
        if (step % nl) == idx:
            real = (raw >> 32) & 0x7FFFFFFF
            mr = tl.where(real > 0, tl.minimum(real, M), M).to(tl.int64)
            t = step // nl
            ii = rb * RB + tl.arange(0, RB).to(tl.int64)
            mm = tl.maximum(mr, 1)
            rows = tl.where(mr <= R_, ii, (t + ii * tl.maximum(mm // R_, 1)) % mm)
            rmask = ii < tl.minimum(mr, R_)
            rows = tl.minimum(rows, M - 1)
            off = ((t % N) * pstep) % N
            col = ((off + cj * tl.maximum(N // C_, 1)) % N).to(tl.int64)
            acc = tl.zeros([RB], dtype=tl.float32)
            comp = tl.zeros([RB], dtype=tl.float32)
            # the norms as scale x sqrt(sum of (value / scale)^2), the scale the largest magnitude so far (as BLAS's
            # nrm2 does): plain sums of squares overflow to infinity for large rows (and an infinite bound scores a
            # wrong output 0) or underflow to zero for small columns
            xsc = tl.zeros([RB], dtype=tl.float32)
            xsq = tl.zeros([RB], dtype=tl.float32)
            wsc = tl.zeros([1], dtype=tl.float32)
            wsq = tl.zeros([1], dtype=tl.float32)
            for k0 in range(0, K, BK_):
                kk = k0 + tl.arange(0, BK_)
                km = kk < K
                xv = tl.load(x + rows[:, None] * sx + kk[None, :], mask=rmask[:, None] & km[None, :],
                             other=0.0).to(tl.float32)
                wv = tl.load(w + col * sw + kk, mask=km, other=0.0).to(tl.float32)
                s = tl.sum(xv * wv[None, :], 1)
                yk = s - comp                               # Kahan: the chunks' sums added with compensation
                tk = acc + yk
                comp = (tk - acc) - yk
                acc = tk
                nxs = tl.maximum(xsc, tl.max(tl.abs(xv), 1))
                ix = 1.0 / tl.where(nxs > 0.0, nxs, 1.0)    # (one division a row a chunk, not one a value)
                rx = xsc * ix
                xq = xv * ix[:, None]
                xsq = xsq * rx * rx + tl.sum(xq * xq, 1)
                xsc = nxs
                nws = tl.maximum(wsc, tl.max(tl.abs(wv), 0))
                iw = 1.0 / tl.where(nws > 0.0, nws, 1.0)
                rw = wsc * iw
                wq = wv * iw
                wsq = wsq * rw * rw + tl.sum(wq * wq, 0)
                wsc = nws
            if HAS_BIAS:
                bv = tl.load(b + col).to(tl.float32)
                ref = acc + bv
                dbl = u * tl.abs(acc)
                bflag = 1
            else:
                bv = 0.0
                ref = acc
                dbl = 0.0
                bflag = 0
            yv = tl.load(y + rows * sy + col * syc, mask=rmask, other=0.0).to(tl.float32)
            norm = (xsc * tl.sqrt(xsq)) * (tl.sum(wsc, 0) * tl.sqrt(tl.sum(wsq, 0)))
            accb = ka * norm + kb * tl.abs(ref)             # acc_term: ka = S u32 C1 sqrt(K), kb = S u32 (sqrt(K) + ..)
            den = u * tl.abs(ref) + dbl + accb + spacing
            z = tl.abs(yv - ref) / den
            z = tl.where(z == z, z, float("inf"))
            # judged only where the reference and its bound are finite: anything else counts as unjudged, not clean
            ok = rmask & (tl.abs(ref) <= fmax) & (ref == ref) & (den == den) & (den < float("inf"))
            z = tl.where(ok, z, 0.0)
            pz = tl.max(z, 0)
            pi = tl.argmax(z, 0)
            sel = tl.arange(0, RB) == pi
            prow = tl.sum(tl.where(sel, rows, 0), 0)
            py = tl.sum(tl.where(sel, yv, 0.0), 0)
            pr = tl.sum(tl.where(sel, ref, 0.0), 0)
            sbase = state + slot * NCOLS
            zbits = pz.to(tl.int32, bitcast=True)
            zb = zbits.to(tl.int64) << 32
            if pz > 0.0:
                cur = tl.load(sbase)
                if zb >= (cur & -4294967296):
                    tl.atomic_max(sbase + 0, zb | prow)
                    tl.atomic_max(sbase + 1, zb | col)
                    tl.atomic_max(sbase + 2, zb | (py.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                    tl.atomic_max(sbase + 3, zb | (pr.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                    tl.atomic_max(sbase + 4, zb | step)
            if (cj == 0) & (rb == 0):
                tl.atomic_add(sbase + 5, 1)
            nok = tl.sum(ok.to(tl.int64), 0)
            if nok > 0:
                tl.atomic_add(sbase + 6, nok)
            nun = tl.sum((rmask & ~ok).to(tl.int64), 0)
            if nun > 0:
                tl.atomic_add(sbase + 7, nun)
            if pz > line:
                zero = tl.zeros([], dtype=tl.int64)         # (int64 literals: the compiler rejects int32 ones here)
                old = tl.atomic_cas(capm, zero, zero + 1)
                if old == 0:
                    tl.store(capm + 1, prow)
                    tl.store(capm + 2, col)
                    tl.store(capm + 3, step)
                    tl.store(capm + 4, zbits.to(tl.int64) & 0xFFFFFFFF)
                    tl.store(capm + 5, zero + bflag)
                    tl.store(capm + 6, zero + slot)
                    for k0 in range(0, K, BK_):
                        kk = k0 + tl.arange(0, BK_)
                        km = kk < K
                        tl.store(capx + kk, tl.load(x + prow * sx + kk, mask=km, other=0.0).to(tl.float32),
                                 mask=km)
                        tl.store(capw + kk, tl.load(w + col * sw + kk, mask=km, other=0.0).to(tl.float32),
                                 mask=km)
                    tl.store(capv, py)
                    tl.store(capv + 1, bv)


def _consts(dtype: torch.dtype, k: int) -> tuple:
    """(u, spacing, largest finite, ka, kb) of a layer's dtype and K, as the kernel takes them."""
    u, sp, fm = dtype_consts(dtype)
    sk = math.sqrt(max(k, 1))
    return float(u), float(sp), float(fm), SAFETY * ACC_UNIT * ACC_C1 * sk, SAFETY * ACC_UNIT * (sk + k / NFMA + 1.0)


def spot_check_triton(x, y, w, b, stepbuf, state, slot, nl, idx, cap_meta, cap_x, cap_w, cap_v, line) -> None:
    m, k = x.shape
    n = w.shape[0]
    if m == 0:
        return
    if _ON[0] is None:
        from ._group import _on
        _ON[0] = _on
    u, sp, fm, ka, kb = _consts(y.dtype, k)
    x = x if x.stride(-1) == 1 else x.contiguous()
    w = w if w.stride(-1) == 1 else w.contiguous()
    with _ON[0](x.device):
        _spot_kernel[(SPOT_COLS, SPOT_ROWS // SPOT_RB)](
            x, y, w, b if b is not None else w, stepbuf, state, cap_meta, cap_x, cap_w, cap_v, m, n, k, x.stride(0),
            y.stride(0), y.stride(1), w.stride(0), slot, max(nl, 1), idx, col_step(n), u, sp, fm, float(line), ka, kb,
            NCOLS=COLS, HAS_BIAS=b is not None, R_=SPOT_ROWS, RB=SPOT_RB, C_=SPOT_COLS, BK_=BK, num_warps=4)


_DEV_KEYS: dict = {}


def _dev_key(device) -> str:
    d = torch.device(device)
    key = _DEV_KEYS.get((d.type, d.index))
    if key is None:
        from ._group import _dev_key as k
        key = _DEV_KEYS[(d.type, d.index)] = k(d)
    return key


def kernel_usable(device, dtype) -> bool:
    from ._group import arch_ok
    if not HAVE_TRITON or os.environ.get("SDCGUARD_FAST", "1") == "0" or \
            os.environ.get("SDCGUARD_SPOT_FAST", "1") == "0" or dtype not in (torch.bfloat16, torch.float16):
        return False
    if device.type == "cuda":
        return arch_ok(device)
    return device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1" and \
        os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1"


def new_caps(device, kmax: int) -> tuple:
    """A device's kept output: (meta [CAP_COLS] int64, input row [kmax], weight row [kmax], (output, bias) [4])."""
    f32 = torch.float32
    return (torch.zeros(CAP_COLS, dtype=torch.int64, device=device), torch.zeros(kmax, dtype=f32, device=device),
            torch.zeros(kmax, dtype=f32, device=device), torch.zeros(4, dtype=f32, device=device))


def verify_kernel(device, dtype) -> tuple:
    """The Triton kernel against the PyTorch check before its first use on a device and dtype: a correct layer with
    and without a bias (both quiet, the same counts, nearly the same scores), one sampled output made wrong (both
    alarm on its row and column and keep the same row), an inactive step (nothing changes), garbage rows past the
    step's real rows (never sampled), an output with a column stride (y transposed), and in bf16 the wrong output
    again with inputs scaled by 2^100 and weights by 2^-100 (the same products; squares that overflow or underflow
    float32 must not hide it). (ok, why)."""
    key = (_dev_key(device), dtype)
    if key in STATUS:
        return STATUS[key]
    if not kernel_usable(device, dtype):
        STATUS[key] = (False, "the Triton kernel isn't available here")
        return STATUS[key]
    from .guard import _exact_reductions_off, _exact_reductions_on
    hold = _exact_reductions_on()
    try:
        gen = torch.Generator().manual_seed(0)
        for m, n, k in ((5, 24, 200), (40, 70, 333)):
            x = torch.randn(m, k, generator=gen).to(dtype).to(device)
            w = (torch.randn(n, k, generator=gen) * 0.05).to(dtype).to(device)
            bias = (torch.randn(n, generator=gen) * 0.1).to(dtype).to(device)
            cases = ("clean", "bias", "wrong", "inactive", "real_rows", "strided") + \
                (("scaled",) if dtype == torch.bfloat16 else ())
            for case in cases:
                bb = bias if case in ("bias", "wrong") else None
                xx, ww = x, w
                if case == "scaled":
                    xx, ww = (x.float() * 2.0 ** 100).to(dtype), (w.float() * 2.0 ** -100).to(dtype)
                y = torch.nn.functional.linear(xx, ww, bb)
                if case == "strided":
                    y = y.t().contiguous().t()
                step = 7 if case != "inactive" else 8
                given = 3 if case == "real_rows" else None
                if case == "real_rows":
                    y[3:] = 1000.0
                if case in ("wrong", "scaled"):
                    rows, rmask, cols = sampled(torch.tensor(step), 3, m, n)
                    r0, c0 = int(rows[0]), int(cols[1])
                    y[r0, c0] = (y[r0, c0].float() * (1 + 16 * dtype_consts(dtype)[0]) + 1e-3).to(dtype)
                res = []
                for fn in (spot_check_torch, spot_check_triton):
                    st = torch.zeros(2, COLS, dtype=torch.int64, device=device)
                    caps = new_caps(device, k)
                    sb = torch.full((1,), pack_step(step, given), dtype=torch.int64, device=device)
                    fn(xx, y, ww, bb, sb, st, 1, 3, 1, *caps, SPOT_Z)
                    res.append((st.cpu(), *[c.cpu() for c in caps]))
                (s0, c0m, c0x, c0w, c0v), (s1, c1m, c1x, c1w, c1v) = res
                z0, z1 = _f32(int(s0[1, C_ROW]) >> 32), _f32(int(s1[1, C_ROW]) >> 32)
                if s0[0].abs().sum() or s1[0].abs().sum():
                    raise AssertionError(f"{case}: another layer's state changed")
                if s0[1, C_CALLS:].tolist() != s1[1, C_CALLS:].tolist():
                    raise AssertionError(f"{case}: counts differ ({s0[1, C_CALLS:].tolist()} vs {s1[1, C_CALLS:].tolist()})")
                if case == "inactive":
                    if s0.abs().sum() or s1.abs().sum() or c0m.abs().sum() or c1m.abs().sum():
                        raise AssertionError("an inactive step changed the state")
                elif case in ("wrong", "scaled"):
                    if not (z0 > SPOT_Z and z1 > SPOT_Z) or int(s0[1, C_COL]) & M32 != c0 or \
                            int(s1[1, C_COL]) & M32 != c0 or int(s1[1, C_ROW]) & M32 != r0:
                        raise AssertionError(f"{case}: a wrong output wasn't located ({z0}, {z1})")
                    same = [0, 1, 2, 3, 5, 6]                   # (not the score's bits: the sums differ a little)
                    if c0m[same].tolist() != c1m[same].tolist() or int(c1m[6]) != 1 or not torch.equal(c0x, c1x) or \
                            not torch.equal(c0w, c1w) or not torch.equal(c0v[:2], c1v[:2]):
                        raise AssertionError(f"{case}: the kept output differs")
                elif not (z0 <= 1.0 and z1 <= 1.0) or abs(z0 - z1) > 0.25:
                    raise AssertionError(f"{case}: scores of a correct layer differ or are high ({z0} vs {z1})")
                elif int(s0[1, C_JUDGED]) != (3 if case == "real_rows" else min(m, SPOT_ROWS)) * SPOT_COLS:
                    raise AssertionError(f"{case}: {int(s0[1, C_JUDGED])} outputs judged")
        STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    finally:
        _exact_reductions_off(hold)
    return STATUS[key]


def _capturing(t: torch.Tensor) -> bool:
    return t.is_cuda and torch.cuda.is_current_stream_capturing()


def spot_check_any(*args) -> None:
    """The check: skipped on the host when its layer isn't active (outside CUDA-graph capture, once the layer's
    kernel has run); else the Triton kernel where it passed its first-use comparison, else the PyTorch one."""
    x, y, sb, slot, nl, idx = args[0], args[1], args[4], args[6], args[7], args[8]
    key = sb.data_ptr()
    if (key, y.dtype) in _DISABLED:
        return
    v = _HOST.get(key)
    if v is not None and (v & M32) % max(nl, 1) != idx and (key, slot) in _SEEN and not _capturing(y):
        return
    _SEEN.add((key, slot))
    skey = (_dev_key(y.device), y.dtype)
    if STATUS.get(skey, (False,))[0]:
        try:
            spot_check_triton(*args)
            return
        except Exception as e:  # noqa: BLE001  (never break serving: the PyTorch version from now on)
            STATUS[skey] = (False, f"a launch failed ({type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''})")
    with torch.no_grad(), torch.autocast(device_type=y.device.type, enabled=False):
        spot_check_torch(*args)


def spot_op():
    """sdcguard::spot_check, registered once (called from the linear layers' apply, so under torch.compile it is one
    opaque op)."""
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::spot_check", mutates_args=("state", "cap_meta", "cap_x", "cap_w",
                                                                               "cap_v"))
                def spot_check(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor],
                               stepbuf: torch.Tensor, state: torch.Tensor, slot: int, nl: int, idx: int,
                               cap_meta: torch.Tensor, cap_x: torch.Tensor, cap_w: torch.Tensor, cap_v: torch.Tensor,
                               line: float) -> None:
                    spot_check_any(x, y, w, b, stepbuf, state, slot, nl, idx, cap_meta, cap_x, cap_w, cap_v, line)

                @spot_check.register_fake
                def _(*a):
                    return None

                _OP[0] = torch.ops.sdcguard.spot_check
            except (AttributeError, RuntimeError):
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "spot_check", None) or spot_check_any
        return _OP[0]


def check_call(args: tuple, x: torch.Tensor, y: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
    """Check one call of a registered linear layer: x [.., K], y [.., N], the bias the matmul added (or None)."""
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)) or x.dim() < 1 or y.dim() < 1:
        return
    w, stepbuf, state, slot, nl, idx, cm, cx, cw, cv, line = args
    x2, y2 = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    if x2.shape[0] != y2.shape[0] or x2.shape[1] != w.shape[1] or y2.shape[1] != w.shape[0]:
        return
    b = bias if isinstance(bias, torch.Tensor) and bias.numel() == w.shape[0] else None
    call = (x2, y2, w, b, stepbuf, state, slot, nl, idx, cm, cx, cw, cv, line)
    if torch.compiler.is_compiling():
        (_OP[0] if _OP[0] is not None else spot_op())(*call)
    else:
        spot_check_any(*call)


def recheck(x_row: torch.Tensor, w_row: torch.Tensor, y: float, bias: Optional[float], dtype: torch.dtype,
            line: float = SPOT_Z) -> dict:
    """The kept output recomputed in float64 on the CPU from the kept input row and weight row."""
    u, sp, fm = dtype_consts(dtype)
    x, w = x_row.double().cpu(), w_row.double().cpu()
    acc = float((x * w).sum())
    ref = acc + (bias or 0.0)
    k = x.numel()
    norm = float(x.square().sum().sqrt() * w.square().sum().sqrt())
    den = u * abs(ref) + (u * abs(acc) if bias is not None else 0.0) + \
        float(acc_term(k, torch.tensor(norm), torch.tensor(abs(ref)))) + sp
    z = abs(y - ref) / den if math.isfinite(y) else float("inf")
    return {"verdict": "output_wrong" if z > line else "output_agrees", "z": z, "output": y, "expected": ref,
            "relative_error": abs(y - ref) / abs(ref) if ref else None}


def describe(layer) -> tuple:
    """(weight, why not): a layer qualifies with a 2-D bf16/fp16 weight on a GPU whose rows are contiguous."""
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return None, "no 2-D weight"
    if w.dtype not in (torch.bfloat16, torch.float16):
        return None, f"weights in {w.dtype}"
    if w.stride(-1) != 1:
        return None, "weight rows aren't contiguous"
    if not (w.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
        return None, "not on a GPU"
    return w, None


class SpotChecks:
    """The linear layers one ServeGuard also spot-checks (ServeGuard.spot_checks()): add() each, finalize() once
    before CUDA graphs are captured, mark_step() before each model step and end_step() after, poll() with the guard's
    reads."""

    def __init__(self, guard, line: float = SPOT_Z):
        line = float(line)
        if not 1.5 <= line < float("inf"):
            raise ValueError(f"the spot check's line must be at least 1.5, not {line!r}")
        self.guard, self.line = guard, line
        self.layers: list = []
        self.unsupported: dict = {}
        self._state, self._cap, self._step, self._host = {}, {}, {}, {}
        self._marked: Optional[int] = None
        self.worst_seen: dict = {}
        self.checks_seen: dict = {}
        self.finalized = False
        self.stats = {"layers": 0, "active_calls": 0, "outputs_judged": 0, "unjudged_outputs": 0, "alerts": 0,
                      "alerts_without_kept_output": 0}

    def add(self, layer, name: str) -> Optional[int]:
        if self.finalized:
            raise RuntimeError("layers are added before finalize()")
        w, why = describe(layer)
        if why:
            self.unsupported[name] = why
            return None
        if getattr(layer, "_sdcguard_spot", None) is not None:
            self.unsupported[name] = "already registered with another ServeGuard's spot checks"
            return None
        self.layers.append({"name": name, "module": layer, "device": w.device, "k": int(w.shape[1]),
                            "dtype": w.dtype})
        self.stats["layers"] += 1
        return len(self.layers) - 1

    def _why_not(self, dev, dtype) -> Optional[str]:
        """Why a device's layers of a dtype aren't spot-checked, or None: the bound doesn't hold there, or (on a GPU)
        the Triton kernel failed its first-use comparison and SDCGUARD_SPOT_TORCH=1 doesn't ask for the PyTorch
        version (about 40 operations more in every layer's call, replayed by CUDA graphs every step)."""
        why = bound_holds(dev, dtype)
        if why:
            return f"the bound doesn't hold: {why}"
        ok, kwhy = verify_kernel(dev, dtype)
        if not ok and torch.device(dev).type == "cuda" and os.environ.get("SDCGUARD_SPOT_TORCH", "0") != "1":
            return f"the Triton kernel isn't usable here ({kwhy}); SDCGUARD_SPOT_TORCH=1 uses the PyTorch version"
        return None

    def finalize(self) -> int:
        """Give every added layer its check arguments; where the bound doesn't hold or (on a GPU) no kernel is usable,
        those layers' calls return at once and they are listed as not spot-checked (they keep their arguments, so
        the traced graph doesn't depend on what finalize found). Returns the layers spot-checked."""
        if self.finalized:
            return len(self.layers)
        whys = {}
        for lay in self.layers:
            key = (lay["device"], lay["dtype"])
            if key not in whys:
                whys[key] = self._why_not(*key)
        every = list(self.layers)
        devs: dict = {}
        for lay in every:
            devs.setdefault(lay["device"], []).append(lay)
        for dev, lays in devs.items():
            with torch.inference_mode(False):
                self._state[dev] = torch.zeros(len(lays), COLS, dtype=torch.int64, device=dev)
                self._cap[dev] = new_caps(dev, max(lay["k"] for lay in lays))
                self._step[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
            key = self._step[dev].data_ptr()           # (a freed step buffer's address may come back)
            _HOST.pop(key, None)
            _SEEN.difference_update({s for s in _SEEN if s[0] == key})
            _DISABLED.difference_update({s for s in _DISABLED if s[0] == key})
            for i, lay in enumerate(lays):
                lay["slot"] = i
                lay["module"]._sdcguard_spot = self._args(lay, len(lays))
                if whys[(dev, lay["dtype"])]:
                    _DISABLED.add((key, lay["dtype"]))
        self._every = every
        self._left_out = {(_dev_key(d), t): w for (d, t), w in whys.items() if w}
        self.layers[:] = [lay for lay in every if not whys[(lay["device"], lay["dtype"])]]
        for lay in every:
            why = whys[(lay["device"], lay["dtype"])]
            if why:
                self.unsupported[lay["name"]] = why
        self.stats["layers"] = len(self.layers)
        spot_op()
        self.finalized = True
        return len(self.layers)

    def _args(self, lay, n: int) -> tuple:
        dev = lay["device"]
        return (lay["module"].weight, self._step[dev], self._state[dev], lay["slot"], n, lay["slot"],
                *self._cap[dev], self.line)

    def detach(self) -> None:
        """Stop checking: the layers' check arguments are removed (graphs captured with them must be rebuilt)."""
        for lay in getattr(self, "_every", self.layers):
            lay["module"].__dict__.pop("_sdcguard_spot", None)
        for sb in self._step.values():
            key = sb.data_ptr()
            _HOST.pop(key, None)
            _SEEN.difference_update({s for s in _SEEN if s[0] == key})
            _DISABLED.difference_update({s for s in _DISABLED if s[0] == key})
        self.layers.clear()
        self._every = []
        self._left_out = {}
        self._state.clear()
        self._cap.clear()
        self._step.clear()
        self._host.clear()

    def refresh(self) -> None:
        """After a weight reload: the layers' check arguments point at their weights now."""
        if not self.finalized:
            return
        every = getattr(self, "_every", self.layers)
        n = {dev: sum(1 for x in every if x["device"] == dev) for dev in self._state}
        for lay in every:
            if isinstance(getattr(lay["module"], "weight", None), torch.Tensor):
                lay["module"]._sdcguard_spot = self._args(lay, n[lay["device"]])

    def _write_step(self, value: int) -> None:
        for sb in self._step.values():
            sb.fill_(value)
            _HOST[sb.data_ptr()] = value

    def mark_step(self, step: int, rows: Optional[int] = None) -> None:
        self._marked = int(step) & M32
        self._write_step(pack_step(step, rows))

    def end_step(self) -> None:
        """After a model step's work is queued: the step's real rows are no longer known, so warm-up and capture runs
        in between are judged over all their rows (a linear layer computes every row of its input, padding rows too,
        so each output is still its row times the weights)."""
        if self._marked is None:
            return
        self._write_step(pack_step(self._marked))

    def poll(self, wait: bool = False) -> list:
        """Read each device's state and its kept output (in the same read, then both cleared on the GPU after the
        copy, in stream order), and report the previous read's alerts (on a GPU without wait: the copy of this read is
        reported at the next poll, when it has arrived)."""
        events = []
        for dev, st in self._state.items():
            cap = self._cap[dev]
            if dev.type != "cuda" or wait:
                pend = self._host.pop(dev, None)
                if pend is not None:
                    pend[2].synchronize()
                    events += self._report(dev, pend[0], pend[1])
                host, hcap = st.to("cpu", copy=True), tuple(c.to("cpu", copy=True) for c in cap)
                st.zero_()
                cap[0].zero_()
                events += self._report(dev, host, hcap)
                continue
            pend = self._host.get(dev)
            if pend is not None and pend[2].query():
                events += self._report(dev, pend[0], pend[1])
                pend = None
                self._host.pop(dev, None)
            if pend is None:
                hs = torch.empty(st.shape, dtype=st.dtype, pin_memory=True)
                hs.copy_(st, non_blocking=True)
                hc = []
                for c in cap:
                    h = torch.empty(c.shape, dtype=c.dtype, pin_memory=True)
                    h.copy_(c, non_blocking=True)
                    hc.append(h)
                st.zero_()                     # same stream: after the copies have read them
                cap[0].zero_()
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (hs, tuple(hc), ev)
        return events

    def coverage(self) -> dict:
        return {"layers": len(self.layers), "active_calls": self.stats["active_calls"],
                "outputs_judged": self.stats["outputs_judged"],
                "never_checked": sorted(x["name"] for x in self.layers if not self.checks_seen.get(x["name"])),
                "not_attached": dict(self.unsupported), "cols_per_turn": SPOT_COLS, "rows_per_turn": SPOT_ROWS,
                "kernel": {f"{d} {str(t).replace('torch.', '')}":
                           (f"left out: {getattr(self, '_left_out', {})[(d, t)]}"
                            if (d, t) in getattr(self, "_left_out", {}) else "Triton" if ok else f"PyTorch ({why})")
                           for (d, t), (ok, why) in sorted(STATUS.items(), key=str)}}

    def _report(self, dev, host, hcap) -> list:
        out = []
        meta = [int(v) for v in hcap[0].tolist()]
        for lay in self.layers:
            if lay["device"] != dev or lay["slot"] >= host.shape[0]:
                continue
            keys = [int(v) for v in host[lay["slot"]].tolist()]
            self.stats["active_calls"] += keys[C_CALLS]
            self.stats["outputs_judged"] += keys[C_JUDGED]
            self.stats["unjudged_outputs"] += keys[C_UNJ]
            self.checks_seen[lay["name"]] = self.checks_seen.get(lay["name"], 0) + keys[C_CALLS]
            z = _f32(keys[C_ROW] >> 32)
            self.worst_seen[lay["name"]] = max(self.worst_seen.get(lay["name"], 0.0), z)
            if z > self.line:
                mine = meta[0] and meta[6] == lay["slot"]
                out.append(self._alert(dev, lay, keys, meta if mine else None, hcap if mine else None, z,
                                       "another layer's output was kept in this read" if meta[0] else
                                       "no output was kept for this alert"))
        return out

    def _alert(self, dev, lay, keys, meta, hcap, z, why_none: str) -> SDCEvent:
        g = self.guard
        captured, rc = None, None
        if meta is not None:
            k = lay["k"]
            x_row, w_row, v = hcap[1][:k], hcap[2][:k], hcap[3].tolist()
            captured = {"row": meta[1], "col": meta[2], "step": meta[3] if self._marked is not None else None,
                        "score": _f32(meta[4])}
            try:
                rc = recheck(x_row, w_row, float(v[0]), float(v[1]) if meta[5] else None, lay["dtype"], self.line)
            except Exception as e:  # noqa: BLE001
                rc = {"verdict": "not_rechecked", "why": f"{type(e).__name__}: {e}"}
        else:
            self.stats["alerts_without_kept_output"] += 1
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        key = str(dev)
        if key not in ident:
            ident[key] = device_identity(dev)
        row, col, yv, rv, place = keys[C_ROW] & M32, keys[C_COL] & M32, _f32(keys[C_OUT]), _f32(keys[C_REF]), \
            PLACE_MAXIMA
        if captured is not None and rc is not None and rc.get("verdict") == "output_wrong":
            row, col, yv, rv, place = captured["row"], captured["col"], rc["output"], rc["expected"], PLACE_KEPT
        detail = {"where": lay["name"], "kind": "spot", "max_z": z, "alarm_line": self.line, "row": row, "col": col,
                  "output": yv, "expected": rv, "place": place,
                  "relative_error": (abs(yv - rv) / abs(rv)) if rv and rv == rv else None, "captured": captured,
                  "recheck": rc or {"verdict": "not_rechecked", "why": why_none},
                  "check_config": {"line": self.line, "cols_per_turn": SPOT_COLS, "rows_per_turn": SPOT_ROWS,
                                   "unit_roundoff": dtype_consts(lay["dtype"])[0], "safety": SAFETY, "nfma": NFMA,
                                   "reference": "fp32 products summed in fp32 with compensation"},
                  "reading": "a served output of the layer differs from its recompute from the layer's own input and "
                             "weights by more than twice its rounding and the matmul's accumulation allow"}
        ev = SDCEvent(detector="serve_spot", step=None, detail=detail, identity=dict(ident[key]))
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
