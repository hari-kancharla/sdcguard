"""Fast checks for NVIDIA GPUs (Triton): two kernel launches per check, one per step to judge them.

The plain check (checksum.matmul_zscores) launches about 25 small GPU operations per matmul.
On a GPT-2 sized model that is ~5,000 launches per checked step, and the CPU time to launch
them becomes most of the cost. Here:

  at the matmul    _stage     plain/signed sums and sum of squares of every 128-column block of
                              the output C, and plain/signed sums and the largest |value| of every
                              block of each row of B (and of the bias)
                   _compare   expected = alpha * A @ (block sums of B) + beta * (bias sums),
                              |a_row|, |checksum column of B|, z = |expected - actual| / noise,
                              the largest output the inputs allow (|A| @ block maxima of |B|, for
                              judging Inf/NaN outputs: see checksum.py), worst z of every tile
  at the flush     _worst     worst z of every check of the step, and where it is (one launch
                              per device and dtype, then one read-back per device)

A, B and C are read right after the matmul, in the same CUDA stream, like the plain check. So a
check sees exactly the values the matmul used, even if training code changes or frees them later
in the step (gradient accumulation, FSDP freeing gathered weights, reused communication buffers).

Same math and noise model as the plain check, including its sign patterns, so the two agree up
to the order of fp32 additions (see tests/test_fast.py and the GPU self-test).
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001  (missing or broken install: the plain check is used)
    HAVE_TRITON = False

BR = 32                     # rows of C, or of B, per program in _stage
BM = 64                     # rows of A per tile in _compare (depth per step: _cfg)
CHUNK = 1024                # tile results reduced per iteration in _worst
PART_FLOATS = 1 << 20       # size of each tile-result buffer (floats); more are added as needed

_BROKEN: dict = {}          # (device type, dtype) -> True after a compile/launch error
_SIGNS: dict = {}           # (device, epoch) -> (sign table on the device, pinned host copy)
_PREC: dict = {}
FALLBACKS = [0]             # compile/launch problems (the plain check took over)
WHY: list = []              # what went wrong, for each of those (the last 10)


def fell_back(device_type: str, dtype: torch.dtype, where: str, e: BaseException) -> None:
    """A Triton compile or launch problem: the plain check takes over for this device type and dtype."""
    FALLBACKS[0] += 1
    _BROKEN[(device_type, dtype)] = True
    msg = str(e).strip().splitlines()
    WHY.append(f"{where} ({device_type}, {str(dtype).replace('torch.', '')}): {type(e).__name__}: "
               f"{msg[0][:300] if msg else ''}")
    del WHY[:-10]
USED = [0]                  # checks done by these kernels


def enabled() -> bool:
    return HAVE_TRITON and os.environ.get("SDCGUARD_FAST", "1") != "0"


def arch_ok(device: torch.device) -> bool:
    """The Triton kernels run on GPU generations they were validated on (Turing through Hopper: T4, A100,
    L4, L40S, H100, H200 on Oct 18, 2026). On newer ones (compute capability 10 and up, Blackwell) the
    plain PyTorch checks run instead: a B200 hit a "misaligned address" fault in the first-use comparison,
    and a fault like that ends the process instead of falling back. SDCGUARD_TRITON_NEW_ARCH=1 opts in."""
    if device.type != "cuda":
        return True
    if os.environ.get("SDCGUARD_TRITON_NEW_ARCH") == "1":
        return True
    try:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_capability(idx)[0] < 10
    except Exception:  # noqa: BLE001
        return False


def available(C: torch.Tensor) -> bool:
    if not enabled() or C.dtype not in (torch.bfloat16, torch.float16) or C.dim() not in (2, 3):
        return False
    if C.is_cuda and not arch_ok(C.device):
        return False
    if _BROKEN.get((C.device.type, C.dtype)):
        return False
    if not C.is_cuda and not (C.device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1"
                              and os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1"):
        return False         # (the CPU case is for tests only: Triton's interpreter, slowly)
    return verified(C.device, C.dtype)


STATUS: dict = {}           # (device, dtype) -> (ok, why): the first-use comparison


def _dev_key(device: torch.device) -> str:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return str(device)


def status(device, dtype: torch.dtype):
    """(ok, why) of the first-use comparison on this device and dtype, or None if it hasn't run."""
    return STATUS.get((_dev_key(torch.device(device)), dtype))


def verified(device: torch.device, dtype: torch.dtype) -> bool:
    """Before its first use on a device, the Triton check must give the plain check's answers on
    clean and corrupted matmuls of every tile shape it uses (guards against a Triton version or
    GPU that compiles the kernels differently). Otherwise the plain check is used. Runs once per
    device and dtype, and waits for the GPU that one time."""
    key = (_dev_key(device), dtype)
    if key not in STATUS:
        from torch.utils._python_dispatch import _disable_current_modes
        used = USED[0]
        try:
            # outside every dispatch mode (the guard itself, a fault injector) and autocast
            with _disable_current_modes(), torch.no_grad(), torch.autocast(device.type, enabled=False):
                ok, why = _probe(device, dtype)
        except torch.cuda.OutOfMemoryError:
            USED[0] = used
            return False                    # no memory for the comparison right now: try again later
        except Exception as e:  # noqa: BLE001  (compile, launch or resource problem)
            ok, why = False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        USED[0] = used
        STATUS[key] = (ok, why)
        if not ok:
            FALLBACKS[0] += 1
            _BROKEN[(device.type, dtype)] = True
            WHY.append(f"first-use comparison ({key[0]}, {str(dtype).replace('torch.', '')}): {why}")
            del WHY[:-10]
    return STATUS[key][0]


def _probe(device: torch.device, dtype: torch.dtype):
    """Clean and corrupted matmuls covering every tile width (16/32/64/128 checksum columns, so
    every kernel configuration), narrow blocks, a bias, several column tiles, a batch and a split sum."""
    from . import checksum as cs
    g = torch.Generator().manual_seed(7)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    # the last shape is short and deep (like a weight gradient): its sum is split across programs
    shapes = [(96, 80, 300, True, None), (2, 40, 64, False, 136), (70, 48, 1500, False, None),
              (70, 48, 2600, True, None), (40, 50, 9000, True, None), (64, 96, 40, False, None),
              (48, 3000, 300, True, None)]
    corrupt = [(70, 140), (1, 33, 130), (5, 1300), (66, 2599), (33, 8500), (17, 39), (40, 250)]
    for (d0, d1, d2, with_bias, n3), spot in zip(shapes, corrupt):
        if n3 is not None:                                    # bmm: batch d0, (d1 x d2) @ (d2 x n3)
            a, b, bias = r(d0, d1, d2), r(d0, d2, n3), None
            c = torch.bmm(a, b)
        else:
            a, b = r(d0, d1), r(d1, d2)
            bias = r(d2) if with_bias else None
            c = torch.addmm(bias, a, b) if with_bias else a @ b
        bad = c.clone()
        bad[spot] = float("nan") if len(spot) == 3 else bad[spot] + 50.0      # always detectable
        for corrupted, cc in ((False, c), (True, bad)):
            grp = Group(device, dtype, 128, cs.ABS_FLOOR)
            grp.add(a, b, cc, bias, 1.0, 1.0)
            grp.launch()
            fz, fb, fr, fblk, _ = grp.row(0, grp.result[0].tolist())
            p = cs.resolve([cs.matmul_zscores(a, b, cc, bias)])[0]
            fast_bad = not (fz == fz and fz <= 6.0)
            if not corrupted:
                same = (not fast_bad) and p.ok and abs(fz - p.max_z) <= 0.05 + 0.02 * p.max_z
            else:
                same = fast_bad and not p.ok and (fr, fblk) == (p.row, p.col_block) and \
                    (cc.dim() == 2 or fb == p.batch)
            if not same:
                return False, (f"{tuple(cc.shape)} {'corrupted' if corrupted else 'clean'}: Triton z {fz} "
                               f"at {fb}/{fr}/{fblk}, plain z {p.max_z} at {p.batch}/{p.row}/{p.col_block}")
    return True, "agrees with the plain check"


def supports(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, bias, beta: float, block: int = 128) -> bool:
    """No bias, or a bias vector of length n; blocks up to 128 columns. Anything else (and empty
    matmuls): plain check."""
    if not 1 <= block <= 128 or C.numel() == 0 or A.shape[-1] == 0:
        return False
    if A.dim() not in (2, 3) or B.dim() not in (2, 3):
        return False
    return bias is None or beta == 0 or (bias.numel() == C.shape[-1] and bias.dim() == 1)


def _sign_offset(width: int) -> int:
    return (width - 1) * 128                 # widths 1..128, one row of 128 each (aligned loads)


def _sign_table(device: torch.device, epoch: int = 0) -> torch.Tensor:
    """The plain check's +-1 pattern for every block width 1..128 (checksum._signs), on the device.
    Built once per device and epoch, and copied without making the CPU wait for the GPU. Serving uses
    epoch 0; training checks move to a new epoch every checked step (see checksum._signs)."""
    key = (str(device), epoch)
    hit = _SIGNS.get(key)
    if hit is None:
        from .checksum import _signs
        if epoch:                                           # keep epoch 0 (serving) and this one
            for k in [k for k in _SIGNS if k[1] not in (0, epoch)]:
                _SIGNS.pop(k, None)
            # every width starts the same 128-wide pattern: one table in one operation
            host = torch.tril(_signs(128, "cpu", epoch).view(1, 128).expand(128, 128)).contiguous()
        else:
            host = torch.zeros(128, 128, dtype=torch.float32)   # (explicit: vLLM loads under bf16 default)
            for w in range(1, 129):
                host[w - 1, :w] = _signs(w, "cpu")
        host = host.reshape(-1)
        if device.type == "cuda":
            host = host.pin_memory()
            dev = host.to(device, non_blocking=True)
        else:
            dev = host.to(device)
        hit = _SIGNS[key] = (dev, host)
    return hit[0]


def _precision(device: torch.device) -> str:
    if device.type != "cuda":
        return "ieee"
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _PREC:
        # A is bf16/fp16, which TF32 holds exactly; "tf32x3" keeps the fp32 block sums of B to
        # ~fp32 accuracy on tensor cores. GPUs before Ampere have no TF32: plain fp32 math.
        _PREC[idx] = "tf32x3" if torch.cuda.get_device_capability(idx)[0] >= 8 else "ieee"
    return _PREC[idx]


def _on(device: torch.device):
    """Triton launches on the current CUDA device; make it the tensors' device if it isn't."""
    if device.type == "cuda" and device.index is not None and device.index != torch.cuda.current_device():
        return torch.cuda.device(device)
    return contextlib.nullcontext()


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _layout(n: int, block: int):
    """Same blocks as the plain check: full blocks of bs = min(block, n), then a narrower last one."""
    bs = max(1, min(block, n))
    rem = n % bs
    nblk = n // bs + (1 if rem else 0)
    return bs, rem, nblk


def _bj(J: int) -> int:
    """Checksum columns per comparison tile (2 per block): enough to read A few times."""
    return 16 if J <= 16 else 32 if J <= 32 else 64 if J <= 64 else 128


def _cfg(bj: int):
    """(depth per step, pipeline stages, warps) of a comparison tile. Wider tiles take smaller
    steps so the tile fits in the shared memory of every NVIDIA GPU from T4 (64 KB) on."""
    if bj <= 32:
        return 64, 3, 4
    if bj == 64:
        return 32, 3, 4
    return 32, 2, 8


_SMS: dict = {}


def _target_programs(device: torch.device) -> int:
    """Programs that keep the whole GPU busy: 4 per streaming multiprocessor."""
    if device.type != "cuda":
        return 64                     # Triton's CPU interpreter (tests): small, so splits get tested
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return 4 * _SMS[idx]


def _splits(tiles: int, k: int, bk: int, device: torch.device):
    """(parts, steps of bk rows of B per part) for the sum over k: one part unless the tiles alone
    would leave the GPU idle and the sum is long (at least 4 steps per part)."""
    steps = _cdiv(k, bk)
    target = _target_programs(device)
    if tiles >= target or steps < 8:
        return 1, steps
    per = _cdiv(steps, min(_cdiv(target, tiles), steps // 4))
    return _cdiv(steps, per), per


def _round16(x: int) -> int:
    return -(-x // 16) * 16


def _dims(t: torch.Tensor):
    """(batch, batch stride, rows, cols, row stride, col stride)."""
    if t.dim() == 3:
        return t.shape[0], t.stride(0), t.shape[1], t.shape[2], t.stride(1), t.stride(2)
    return 1, 0, t.shape[0], t.shape[1], t.stride(0), t.stride(1)


if HAVE_TRITON:

    # Memory layout of a check's work buffer W (fp32), rows of JS = J rounded up to 16 entries so
    # loads can be wide: R (outputs' block sums, nb*m rows of JS) | Q (sums of squares, nb*m*nblk) |
    # L1 (sums of |outputs|, nb*m*nblk) | B2 (B's block sums, bB*kb rows of JS, row k = bias) |
    # BQ (sums of squares of B2 over each tile of BR rows of B, bB*nkb rows of JS) | MX (largest |B|
    # per tile of BR rows and block, bB*(nkb+1)*nblk, row nkb = bias). On sm80+ in bf16, B2 also
    # goes to P16 as three bf16 parts (hi, mid, lo), each bB*kb rows of JS, `pst` apart.

    @triton.jit
    def _stage_c_part(pid, C_ptr, S_ptr, W_ptr, m, n, bs, nblk, rem, s_full, s_last, s_cb, s_cm, s_cn,
                      q_off, l1_off, JS, BR: tl.constexpr, BW: tl.constexpr):
        jb = pid % nblk
        t = pid // nblk
        ntm = (m + BR - 1) // BR
        pm = t % ntm
        pb = (t // ntm).to(tl.int64)
        rm = pm.to(tl.int64) * BR + tl.arange(0, BR)
        w = tl.arange(0, BW)
        cols = jb.to(tl.int64) * bs + w
        rmask = rm < m
        cmask = (w < bs) & (cols < n)
        c = tl.load(C_ptr + pb * s_cb + rm[:, None] * s_cm + cols[None, :] * s_cn,
                    mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        last = (jb == nblk - 1) & (rem > 0)
        s = tl.load(S_ptr + tl.where(last, s_last, s_full) + w, mask=cmask, other=0.0)
        row = (pb * m + rm) * JS + 2 * jb
        tl.store(W_ptr + row, tl.sum(c, axis=1), mask=rmask)
        tl.store(W_ptr + row + 1, tl.sum(c * s[None, :], axis=1), mask=rmask)
        tl.store(W_ptr + q_off + (pb * m + rm) * nblk + jb, tl.sum(c * c, axis=1), mask=rmask)
        tl.store(W_ptr + l1_off + (pb * m + rm) * nblk + jb, tl.sum(tl.abs(c), axis=1), mask=rmask)
        if jb == nblk - 1:
            _zero_pad(W_ptr + (pb * m + rm) * JS, 2 * nblk, JS, rmask)

    @triton.jit
    def _zero_pad(rows, J, JS, rmask):
        """Zeros in columns J..JS-1 of rows of JS: wide loads read them, so they must be numbers."""
        pc = J + tl.arange(0, 16)
        tl.store(rows[:, None] + pc[None, :], tl.zeros((rows.shape[0], 16), dtype=rows.dtype.element_ty),
                 mask=rmask[:, None] & (pc < JS)[None, :])

    @triton.jit
    def _bf16_parts(ptr, pst, v, mask):
        hi = v.to(tl.bfloat16)
        r1 = v - hi.to(tl.float32)
        mid = r1.to(tl.bfloat16)
        tl.store(ptr, hi, mask=mask)
        tl.store(ptr + pst, mid, mask=mask)
        tl.store(ptr + 2 * pst, (r1 - mid.to(tl.float32)).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _stage_b_part(pid, B_ptr, bias_ptr, S_ptr, W_ptr, P16_ptr, n, k, kb, bs, nblk, rem, s_full, s_last,
                      s_bb, s_bk, s_bn, s_bias, has_bias, b2_off, bq_off, mx_off, JS, pst,
                      BR: tl.constexpr, BW: tl.constexpr, NATIVE: tl.constexpr):
        jb = pid % nblk
        t = pid // nblk
        nkt = (kb + BR - 1) // BR
        tk = t % nkt
        bb = (t // nkt).to(tl.int64)
        rk = tk.to(tl.int64) * BR + tl.arange(0, BR)
        w = tl.arange(0, BW)
        cols = jb.to(tl.int64) * bs + w
        cmask = (w < bs) & (cols < n)
        kmask = rk < kb
        in_k = rk < k
        is_bias = (rk == k)[:, None] & (has_bias == 1)          # the bias is row k of B
        # (two loads, not one through a choice of pointers: that would hide from the compiler
        # that B's addresses run in order, and every value would be read on its own)
        x = tl.load(B_ptr + bb * s_bb + rk[:, None] * s_bk + cols[None, :] * s_bn,
                    mask=in_k[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        if has_bias == 1:
            if (tk * BR <= k) & (k < tk * BR + BR):
                x += tl.load(bias_ptr + rk[:, None] * 0 + cols[None, :] * s_bias,
                             mask=is_bias & cmask[None, :], other=0.0).to(tl.float32)
        last = (jb == nblk - 1) & (rem > 0)
        s = tl.load(S_ptr + tl.where(last, s_last, s_full) + w, mask=cmask, other=0.0)
        p_sum = tl.sum(x, axis=1)
        s_sum = tl.sum(x * s[None, :], axis=1)
        o = (bb * kb + rk) * JS + 2 * jb
        tl.store(W_ptr + b2_off + o, p_sum, mask=kmask)
        tl.store(W_ptr + b2_off + o + 1, s_sum, mask=kmask)
        if NATIVE:
            # three bf16 parts that add up to the fp32 sums (to ~2^-24), for the tensor cores
            _bf16_parts(P16_ptr + o, pst, p_sum, kmask)
            _bf16_parts(P16_ptr + o + 1, pst, s_sum, kmask)
        nkb = (k + BR - 1) // BR
        if tk * BR < k:
            # sums of squares of this tile's block sums (B's rows only), for |checksum column of B|
            q = W_ptr + bq_off + (bb * nkb + tk) * JS + 2 * jb
            tl.store(q, tl.sum(tl.where(in_k, p_sum * p_sum, 0.0), axis=0))
            tl.store(q + 1, tl.sum(tl.where(in_k, s_sum * s_sum, 0.0), axis=0))
        if jb == nblk - 1:
            rows = (bb * kb + rk) * JS
            _zero_pad(W_ptr + b2_off + rows, 2 * nblk, JS, kmask)
            if NATIVE:
                for i in tl.static_range(3):
                    _zero_pad(P16_ptr + i * pst + rows, 2 * nblk, JS, kmask)
            if tk * BR < k:
                qrow = W_ptr + bq_off + (bb * nkb + tk) * JS + tl.arange(0, 1)
                _zero_pad(qrow, 2 * nblk, JS, tl.arange(0, 1) == 0)
        # Largest |value| of this tile of B in this block (NaN counts as Inf), and of the bias:
        # rows 0..nkb-1 of the maxima are B's tiles of BR rows, row nkb is the bias.
        ax = tl.where(x == x, tl.abs(x), float("inf"))
        mrow = W_ptr + mx_off + bb * (nkb + 1) * nblk
        in_b = in_k[:, None] & cmask[None, :]
        if tk * BR < k:
            tl.store(mrow + tk * nblk + jb, tl.max(tl.max(tl.where(in_b, ax, 0.0), axis=1), axis=0))
        if has_bias == 1:
            if (tk * BR <= k) & (k < tk * BR + BR):
                tl.store(mrow + nkb * nblk + jb,
                         tl.max(tl.max(tl.where(is_bias & cmask[None, :], ax, 0.0), axis=1), axis=0))

    # Sizes and strides that decide whether loads can be wide (m, n, k, bs, strides, JS, offsets of
    # rows of JS) are left to Triton to specialize: a multiple of 16 lets it read 16 bytes at a time.
    @triton.jit(do_not_specialize=["kb", "nblk", "rem", "s_bias", "n_c", "q_off", "l1_off", "mx_off"])
    def _stage(C_ptr, B_ptr, bias_ptr, S_ptr, W_ptr, P16_ptr, m, n, k, kb, bs, nblk, rem, s_full, s_last,
               s_cb, s_cm, s_cn, s_bb, s_bk, s_bn, s_bias, has_bias, n_c, q_off, l1_off, b2_off, bq_off, mx_off,
               JS, pst, BR: tl.constexpr, BW: tl.constexpr, NATIVE: tl.constexpr):
        """Programs [0, n_c): sums of C's blocks. The rest: block sums of B (and the bias)."""
        pid = tl.program_id(0)
        if pid < n_c:
            _stage_c_part(pid, C_ptr, S_ptr, W_ptr, m, n, bs, nblk, rem, s_full, s_last, s_cb, s_cm, s_cn,
                          q_off, l1_off, JS, BR, BW)
        else:
            _stage_b_part(pid - n_c, B_ptr, bias_ptr, S_ptr, W_ptr, P16_ptr, n, k, kb, bs, nblk, rem, s_full,
                          s_last, s_bb, s_bk, s_bn, s_bias, has_bias, b2_off, bq_off, mx_off, JS, pst,
                          BR, BW, NATIVE)

    @triton.jit
    def _accum(A_ptr, W_ptr, P16_ptr, bi, rm, rj, mmask, jmask, k_lo, k_hi, k, nblk, s_ab, s_am, s_ak, b_step,
               bq_step, b2_off, bq_off, mx_off, mx_step, JS, pst, BM: tl.constexpr, BK: tl.constexpr,
               BJ: tl.constexpr, BR: tl.constexpr, PREC: tl.constexpr, NATIVE: tl.constexpr,
               EMULATE: tl.constexpr):
        """Rows k_lo..k_hi-1 (k_lo a multiple of BK) of the sum: A @ (block sums of B), |a_row|^2,
        |a_row|_1, |checksum column|^2, and the largest |B| per block."""
        nkb = (k + BR - 1) // BR
        a_base = A_ptr + bi * s_ab
        b_base = W_ptr + b2_off + bi * b_step
        p_base = P16_ptr + bi * b_step
        q_base = W_ptr + bq_off + bi * bq_step
        x_base = W_ptr + mx_off + bi * mx_step
        acc = tl.zeros((BM, BJ), dtype=tl.float32)
        asq = tl.zeros((BM,), dtype=tl.float32)
        asum = tl.zeros((BM,), dtype=tl.float32)
        bsq = tl.zeros((BJ,), dtype=tl.float32)
        bmax = tl.zeros((BJ,), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BK):
            k0 = tl.multiple_of(k0, BK)
            rk = k0 + tl.arange(0, BK).to(tl.int64)
            kmask = rk < k_hi
            a_in = tl.load(a_base + rm[:, None] * s_am + rk[None, :] * s_ak,
                           mask=mmask[:, None] & kmask[None, :], other=0.0)
            bmask = kmask[:, None] & (rj < JS)[None, :]       # (columns J..JS-1 are zeros)
            bo = rk[:, None] * JS + rj[None, :]
            if NATIVE:
                # bf16 A as it is, times the block sums of B as three bf16 parts (made in _stage):
                # three native tensor-core products, accumulated in fp32
                b_hi = tl.load(p_base + bo, mask=bmask, other=0.0)
                b_mid = tl.load(p_base + pst + bo, mask=bmask, other=0.0)
                b_lo = tl.load(p_base + 2 * pst + bo, mask=bmask, other=0.0)
                if EMULATE:          # tests only: Triton's CPU interpreter gets bf16 dots wrong
                    a32 = a_in.to(tl.float32)
                    acc = tl.dot(a32, b_hi.to(tl.float32), acc, input_precision="ieee")
                    acc = tl.dot(a32, b_mid.to(tl.float32), acc, input_precision="ieee")
                    acc = tl.dot(a32, b_lo.to(tl.float32), acc, input_precision="ieee")
                else:
                    acc = tl.dot(a_in, b_hi, acc)
                    acc = tl.dot(a_in, b_mid, acc)
                    acc = tl.dot(a_in, b_lo, acc)
            else:
                b = tl.load(b_base + bo, mask=bmask, other=0.0)
                acc = tl.dot(a_in.to(tl.float32), b, acc, input_precision=PREC)
            a = a_in.to(tl.float32)
            asq += tl.sum(a * a, axis=1)
            asum += tl.sum(tl.abs(a), axis=1)
            rt = k0 // BR + tl.arange(0, BK // BR)
            bsq += tl.sum(tl.load(q_base + rt[:, None] * JS + rj[None, :],
                                  mask=(rt < nkb)[:, None] & (rj < JS)[None, :], other=0.0), axis=0)
            xm = tl.load(x_base + rt[:, None] * nblk + (rj // 2)[None, :],
                         mask=(rt < nkb)[:, None] & jmask[None, :], other=0.0)
            bmax = tl.maximum(bmax, tl.max(xm, axis=0))
        return acc, asq, asum, bsq, bmax

    @triton.jit
    def _judge(acc, asq, asum, bsq, bmax, W_ptr, P_ptr, p_off, pid, bi, rm, rj, mmask, jmask, m, k, nblk, bs,
               rem, b_step, q_off, l1_off, bias_off, mx_off, mx_step, JS, has_bias, alpha, beta, u, abs_floor,
               half_tiny, out_max, zt, n2, BM: tl.constexpr, BJ: tl.constexpr, BR: tl.constexpr):
        """z of every checksum in the tile; stores the worst one, its noise and where it is."""
        nkb = (k + BR - 1) // BR
        expected = acc * alpha
        # largest output the inputs allow: |alpha| x |a_row|_1 x max|B| in the block (+ the bias)
        lim = tl.abs(alpha) * asum[:, None] * bmax[None, :]
        if has_bias == 1:
            # (bias_off: where the bias row of B2 starts, worked out on the host. Sizes like k can't
            # be used as tensors here: Triton turns an argument equal to 1 into a constant.)
            expected += beta * tl.load(W_ptr + bias_off + bi * b_step + rj, mask=jmask, other=0.0)[None, :]
            lim += tl.abs(beta) * tl.load(W_ptr + mx_off + bi * mx_step + nkb * nblk + rj // 2,
                                          mask=jmask, other=0.0)[None, :]
        both = mmask[:, None] & jmask[None, :]
        rrow = bi * m + rm
        actual = tl.load(W_ptr + rrow[:, None] * JS + rj[None, :], mask=mmask[:, None] & (rj < JS)[None, :],
                         other=0.0)
        sq = tl.load(W_ptr + q_off + rrow[:, None] * nblk + (rj // 2)[None, :], mask=both, other=0.0)
        # fp32 rounding of `expected` (matters when outputs nearly cancel). If |a_row| or the
        # checksum column is too large to square in fp32, bound it with the largest output instead.
        floor_c = 1e-5 * tl.abs(alpha) * tl.sqrt(asq)[:, None] * tl.sqrt(bsq)[None, :]
        floor_c = tl.where(floor_c < float("inf"), floor_c, 1e-5 * 128.0 * lim)
        # tiny fp16 outputs round on a fixed grid: half a subnormal step each (checksum.py explains)
        wblk = tl.where(((rj // 2) == nblk - 1) & (rem > 0), rem, bs).to(tl.float32)
        den = u * tl.sqrt(sq) + floor_c + abs_floor + (tl.sqrt(wblk) * half_tiny)[None, :]
        diff = tl.abs(expected - actual)
        z = diff / den
        # The plain sum also has to beat the worst rounding possible, u x sum|outputs|: outputs
        # that are all equal (e.g. B with identical columns) round the same way and add up,
        # which the random signs of the second sum average out (checksum.py explains).
        l1 = tl.load(W_ptr + l1_off + rrow[:, None] * nblk + (rj // 2)[None, :], mask=both, other=0.0)
        z_worst = diff / (1.25 * (u * l1 + (wblk * half_tiny)[None, :]) + floor_c + abs_floor)   # 1.25: margin
        z = tl.where((rj % 2 == 0)[None, :], tl.minimum(z, zt * z_worst), z)
        # Judging Inf/NaN (checksum.py explains): an expected side that isn't finite can't be judged
        # (non-finite inputs, or values too large for fp32 sums). An Inf/NaN output, or one too large
        # to square in fp32, is corruption only if the inputs can't produce it.
        lim = lim * 1.01
        e_bad = ~(tl.abs(expected) < float("inf"))
        a_bad = (~e_bad) & ~(tl.abs(actual) < float("inf"))
        big = (~e_bad) & (~a_bad) & ~(sq < float("inf"))
        z = tl.where(z == z, z, -1.0)             # inf/inf (both sides huge): nothing to judge (-1, as in checksum.py)
        z = tl.where(e_bad, -1.0, z)
        z = tl.where(a_bad, tl.where(lim < out_max, float("inf"), -1.0), z)
        z = tl.where(big, tl.where(lim < 1e18, float("inf"), -1.0), z)
        z = tl.where(both, z, -1.0)
        # Worst element of the tile, first in row-major order on ties. No tl.reshape here: Triton
        # lets a reshape that feeds a reduction reorder elements, which scrambles the index that
        # tl.max(..., return_indices=True) returns (the CPU interpreter doesn't show this).
        rmax, rcol = tl.max(z, axis=1, return_indices=True)
        mx, r = tl.max(rmax, axis=0, return_indices=True)
        c = tl.sum(tl.where(tl.arange(0, BM) == r, rcol, 0), axis=0)
        pick = (tl.arange(0, BM)[:, None] == r) & (tl.arange(0, BJ)[None, :] == c)
        d = tl.sum(tl.sum(tl.where(pick, den, 0.0), axis=1), axis=0)
        out = P_ptr + p_off
        tl.store(out + pid, mx)
        tl.store(out + n2 + pid, d)
        tl.store(out + 2 * n2 + pid, (r * BJ + c).to(tl.float32))

    @triton.jit
    def _tile(pid, m, nblk, ntm, ntj, BM: tl.constexpr, BJ: tl.constexpr):
        per_b = ntm * ntj
        bi = (pid // per_b).to(tl.int64)
        t2 = pid % per_b
        rm = (t2 // ntj).to(tl.int64) * BM + tl.arange(0, BM)
        rj = (t2 % ntj).to(tl.int64) * BJ + tl.arange(0, BJ)
        return bi, rm, rj, rm < m, rj < 2 * nblk

    @triton.jit(do_not_specialize=["p_off", "nblk", "bs", "rem", "q_off", "l1_off", "mx_off", "mx_step",
                                   "ntm", "ntj", "n2"])
    def _compare(A_ptr, W_ptr, P16_ptr, P_ptr, p_off, m, k, nblk, bs, rem, s_ab, s_am, s_ak, b_step, bq_step,
                 q_off, l1_off, b2_off, bias_off, bq_off, mx_off, mx_step, JS, pst, has_bias, alpha, beta, u, abs_floor,
                 half_tiny, out_max, zt, ntm, ntj, n2, BM: tl.constexpr, BK: tl.constexpr, BJ: tl.constexpr,
                 BR: tl.constexpr, PREC: tl.constexpr, NATIVE: tl.constexpr, EMULATE: tl.constexpr):
        """One program per tile of BM rows x BJ checksums: the whole sum, then the judgement."""
        pid = tl.program_id(0)
        bi, rm, rj, mmask, jmask = _tile(pid, m, nblk, ntm, ntj, BM, BJ)
        acc, asq, asum, bsq, bmax = _accum(A_ptr, W_ptr, P16_ptr, bi, rm, rj, mmask, jmask, 0, k, k, nblk, s_ab,
                                           s_am, s_ak, b_step, bq_step, b2_off, bq_off, mx_off, mx_step, JS, pst,
                                           BM, BK, BJ, BR, PREC, NATIVE, EMULATE)
        _judge(acc, asq, asum, bsq, bmax, W_ptr, P_ptr, p_off, pid, bi, rm, rj, mmask, jmask, m, k, nblk, bs, rem,
               b_step, q_off, l1_off, bias_off, mx_off, mx_step, JS, has_bias, alpha, beta, u, abs_floor, half_tiny,
               out_max, zt, n2, BM, BJ, BR)

    @triton.jit(do_not_specialize=["nblk", "mx_off", "mx_step", "ntm", "ntj", "steps", "splits"])
    def _compare_part(A_ptr, W_ptr, P16_ptr, X_ptr, m, k, nblk, s_ab, s_am, s_ak, b_step, bq_step, b2_off, bq_off,
                      mx_off, mx_step, JS, pst, ntm, ntj, steps, splits, BM: tl.constexpr, BK: tl.constexpr,
                      BJ: tl.constexpr, BR: tl.constexpr, PREC: tl.constexpr, NATIVE: tl.constexpr,
                      EMULATE: tl.constexpr):
        """Matmuls that are short and deep (weight gradients: few rows, a long sum) leave most of
        the GPU idle with one program per tile, so the sum is split into `splits` parts of `steps`
        steps of BK rows of B, one program each; _finish adds them up."""
        pid = tl.program_id(0)
        sp = pid % splits
        tile = pid // splits
        bi, rm, rj, mmask, jmask = _tile(tile, m, nblk, ntm, ntj, BM, BJ)
        k_lo = sp * steps * BK
        k_hi = tl.minimum(k_lo + steps * BK, k)
        acc, asq, asum, bsq, bmax = _accum(A_ptr, W_ptr, P16_ptr, bi, rm, rj, mmask, jmask, k_lo, k_hi, k, nblk,
                                           s_ab, s_am, s_ak, b_step, bq_step, b2_off, bq_off, mx_off, mx_step, JS,
                                           pst, BM, BK, BJ, BR, PREC, NATIVE, EMULATE)
        o = X_ptr + pid.to(tl.int64) * (BM * BJ + 2 * BM + 2 * BJ)
        i = tl.arange(0, BM)
        j = tl.arange(0, BJ)
        tl.store(o + i[:, None] * BJ + j[None, :], acc)
        tl.store(o + BM * BJ + i, asq)
        tl.store(o + BM * BJ + BM + i, asum)
        tl.store(o + BM * BJ + 2 * BM + j, bsq)
        tl.store(o + BM * BJ + 2 * BM + BJ + j, bmax)

    @triton.jit(do_not_specialize=["p_off", "m", "k", "nblk", "bs", "rem", "q_off", "l1_off", "mx_off", "mx_step",
                                   "ntm", "ntj", "n2", "splits"])
    def _finish(X_ptr, W_ptr, P_ptr, p_off, m, k, nblk, bs, rem, b_step, q_off, l1_off, bias_off, mx_off, mx_step, JS,
                has_bias, alpha, beta, u, abs_floor, half_tiny, out_max, zt, ntm, ntj, n2, splits,
                BM: tl.constexpr, BJ: tl.constexpr, BR: tl.constexpr):
        """Adds up the parts of every tile, then the judgement."""
        pid = tl.program_id(0)
        bi, rm, rj, mmask, jmask = _tile(pid, m, nblk, ntm, ntj, BM, BJ)
        i = tl.arange(0, BM)
        j = tl.arange(0, BJ)
        acc = tl.zeros((BM, BJ), dtype=tl.float32)
        asq = tl.zeros((BM,), dtype=tl.float32)
        asum = tl.zeros((BM,), dtype=tl.float32)
        bsq = tl.zeros((BJ,), dtype=tl.float32)
        bmax = tl.zeros((BJ,), dtype=tl.float32)
        for sp in range(0, splits):
            o = X_ptr + (pid.to(tl.int64) * splits + sp) * (BM * BJ + 2 * BM + 2 * BJ)
            acc += tl.load(o + i[:, None] * BJ + j[None, :])
            asq += tl.load(o + BM * BJ + i)
            asum += tl.load(o + BM * BJ + BM + i)
            bsq += tl.load(o + BM * BJ + 2 * BM + j)
            bmax = tl.maximum(bmax, tl.load(o + BM * BJ + 2 * BM + BJ + j))
        _judge(acc, asq, asum, bsq, bmax, W_ptr, P_ptr, p_off, pid, bi, rm, rj, mmask, jmask, m, k, nblk, bs, rem,
               b_step, q_off, l1_off, bias_off, mx_off, mx_step, JS, has_bias, alpha, beta, u, abs_floor, half_tiny,
               out_max, zt, n2, BM, BJ, BR)

    @triton.jit
    def _worst(T_ptr, OUT_ptr, CH: tl.constexpr):
        """Check p: worst z over its tiles, which tile, where in the tile, and the noise there."""
        p = tl.program_id(0)
        P = tl.load(T_ptr + 2 * p).to(tl.pointer_type(tl.float32), bitcast=True)
        n2 = tl.load(T_ptr + 2 * p + 1)
        best = n2 * 0.0 - 2.0
        best_i = n2 * 0
        for s0 in range(0, n2, CH):
            offs = s0 + tl.arange(0, CH)
            zz = tl.load(P + offs, mask=offs < n2, other=-3.0)
            mx, ix = tl.max(zz, axis=0, return_indices=True)
            take = mx > best
            best_i = tl.where(take, s0 + ix, best_i)
            best = tl.where(take, mx, best)
        o = OUT_ptr + p * 4
        tl.store(o, best.to(tl.float64))
        tl.store(o + 1, best_i.to(tl.float64))
        tl.store(o + 2, tl.load(P + 2 * n2 + best_i).to(tl.float64))
        tl.store(o + 3, tl.load(P + n2 + best_i).to(tl.float64))


class Group:
    """Checks of one step on one device in one dtype. add() does the work of a check right away;
    launch() queues the search for each check's worst z; row() decodes the read-back."""

    def __init__(self, device: torch.device, dtype: torch.dtype, block: int, abs_floor: float, z: float = 6.0,
                 epoch: int = 0):
        from .checksum import BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF
        self.device, self.dtype, self.block, self.abs_floor, self.z = device, dtype, block, abs_floor, z
        self.epoch = epoch            # which +-1 pattern the signed checksums use (see _sign_table)
        self.u = float(UNIT_ROUNDOFF[dtype])
        self.half_tiny = float(HALF_TINY_STEP[dtype])
        self.out_max = float(min(DTYPE_MAX[dtype], BIG))   # Inf/NaN outputs: see checksum.py
        self.meta: list = []          # per check: (tiles per batch row, tiles per row, BJ, buffer, offset, tiles)
        self.tab: list = []           # per check: address of its tile results, number of tiles
        self.result: Optional[torch.Tensor] = None      # (checks, 4) float64 after launch()
        self._bufs: list = []         # tile-result buffers, alive until the read-back
        self._buf: Optional[torch.Tensor] = None
        self._used = 0
        self._keep: list = []

    def __len__(self) -> int:
        return len(self.meta)

    def _signs(self) -> torch.Tensor:
        self._sign_ref = _sign_table(self.device, self.epoch)   # alive as long as this group, whatever the cache does
        return self._sign_ref

    def _space(self, n: int) -> int:
        if self._buf is None or self._used + n > self._buf.numel():
            self._buf = torch.empty(max(PART_FLOATS, n), device=self.device, dtype=torch.float32)
            self._bufs.append(self._buf)
            self._base = self._buf.data_ptr()
            self._used = 0
        off = self._used
        self._used += n
        return off

    def add(self, A, B, C, bias, alpha: float, beta: float) -> int:
        """Check C = beta*bias + alpha*(A @ B): queue the GPU work now. Returns the check's index."""
        if A.dtype != self.dtype:
            A = A.to(self.dtype)
        if B.dtype != self.dtype:
            B = B.to(self.dtype)
        hb = 1 if (bias is not None and beta != 0) else 0
        if hb and bias.dtype != self.dtype:
            bias = bias.to(self.dtype)
        a, b, c = _dims(A), _dims(B), _dims(C)
        nb, m, n, k = c[0], c[2], c[3], a[3]
        bs, rem, nblk = _layout(n, self.block)
        J = 2 * nblk
        JS = _round16(J)                                # row length of R, B2, BQ: wide loads
        kb = k + hb
        nkb = _cdiv(k, BR)
        bB = b[0]
        q_off = _round16(nb * m * JS)
        l1_off = _round16(q_off + nb * m * nblk)
        b2_off = _round16(l1_off + nb * m * nblk)
        bq_off = b2_off + bB * kb * JS
        bias_off = b2_off + k * JS                      # the bias is row k of B2
        mx_off = bq_off + bB * nkb * JS
        W = torch.empty(mx_off + bB * (nkb + 1) * nblk, device=self.device, dtype=torch.float32)
        prec = _precision(self.device)
        # bf16 tensor cores (sm80+). Tests can run the same math in Triton's CPU interpreter,
        # which gets bf16 dots wrong, with fp32 products instead (emulate).
        emulate = self.device.type == "cpu" and os.environ.get("SDCGUARD_TEST_NATIVE") == "1"
        native = self.dtype == torch.bfloat16 and (prec != "ieee" or emulate)
        pst = bB * kb * JS
        P16 = torch.empty(3 * pst, device=self.device, dtype=torch.bfloat16) if native else W
        bj = _bj(J)
        ntm, ntj = _cdiv(m, BM), _cdiv(J, bj)
        n2 = nb * ntm * ntj
        n_c = nb * _cdiv(m, BR) * nblk
        n_b = bB * _cdiv(kb, BR) * nblk
        b_step = kb * JS if bB > 1 else 0
        bq_step = nkb * JS if bB > 1 else 0
        mx_step = (nkb + 1) * nblk if bB > 1 else 0
        with _on(self.device):
            _stage[(n_c + n_b,)](
                C, B, bias if hb else B, self._signs(), W, P16, m, n, k, kb, bs, nblk, rem,
                _sign_offset(bs), _sign_offset(rem) if rem else 0, c[1], c[4], c[5], b[1], b[4], b[5],
                bias.stride(0) if hb else 0, hb, n_c, q_off, l1_off, b2_off, bq_off, mx_off, JS, pst, BR=BR,
                BW=128 if bs > 64 else 64, NATIVE=native)
            off = self._space(3 * n2)
            bk, stages, warps = _cfg(bj)
            s_ab = a[1] if a[0] > 1 else 0
            splits, steps = _splits(n2, k, bk, self.device)
            judge = (hb, float(alpha), float(beta), self.u, float(self.abs_floor), self.half_tiny, self.out_max,
                     float(self.z), ntm, ntj, n2)
            if splits == 1:
                _compare[(n2,)](
                    A, W, P16, self._buf, off, m, k, nblk, bs, rem, s_ab, a[4], a[5], b_step, bq_step, q_off,
                    l1_off, b2_off, bias_off, bq_off, mx_off, mx_step, JS, pst, *judge, BM=BM, BK=bk, BJ=bj, BR=BR,
                    PREC=prec, NATIVE=native, EMULATE=emulate, num_warps=warps, num_stages=stages)
            else:
                X = torch.empty(n2 * splits * (BM * bj + 2 * BM + 2 * bj), device=self.device, dtype=torch.float32)
                _compare_part[(n2 * splits,)](
                    A, W, P16, X, m, k, nblk, s_ab, a[4], a[5], b_step, bq_step, b2_off, bq_off, mx_off, mx_step,
                    JS, pst, ntm, ntj, steps, splits, BM=BM, BK=bk, BJ=bj, BR=BR, PREC=prec, NATIVE=native,
                    EMULATE=emulate, num_warps=warps, num_stages=stages)
                _finish[(n2,)](
                    X, W, self._buf, off, m, k, nblk, bs, rem, b_step, q_off, l1_off, bias_off, mx_off, mx_step, JS,
                    *judge, splits, BM=BM, BJ=bj, BR=BR, num_warps=warps)
        self.meta.append((ntm * ntj, ntj, bj, self._buf, off, n2))
        self.tab += (self._base + 4 * off, n2)
        USED[0] += 1
        return len(self.meta) - 1

    def launch(self) -> None:
        """Queue the search for every check's worst z. Never waits for the GPU."""
        P = len(self.meta)
        host = torch.tensor(self.tab, dtype=torch.int64)
        if self.device.type == "cuda":
            try:
                host = host.pin_memory()
                T = host.to(self.device, non_blocking=True)
            except RuntimeError:                 # no pinned memory here: a plain copy (waits)
                T = host.to(self.device)
        else:
            T = host
        self._keep = [host, T]                   # alive until the read-back
        out = torch.empty((P, 4), device=self.device, dtype=torch.float64)
        try:
            with _on(self.device):
                _worst[(P,)](T, out, CH=CHUNK)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001  (compile or launch problem: same answer with torch ops)
            fell_back(self.device.type, self.dtype, "worst-z search", e)
            out = self._worst_torch()
        self.result = out

    def _worst_torch(self) -> torch.Tensor:
        rows = []
        for (_pb, _ntj, _bj, buf, off, n2) in self.meta:
            z = buf[off:off + n2]
            i = torch.argmax(z)
            rows.append(torch.stack([z[i].double(), i.double(), buf[off + 2 * n2 + i].double(),
                                     buf[off + n2 + i].double()]))
        return torch.stack(rows)

    def row(self, idx: int, r) -> tuple:
        """Decode read-back row r of check idx: (max_z, batch, row, col_block, noise)."""
        per_b, ntj, bj = self.meta[idx][:3]
        z, t, li, den = r
        bi, t2 = divmod(int(t), per_b)
        tm, tj = divmod(t2, ntj)
        li = int(li)
        return z, bi, tm * BM + li // bj, (tj * bj + li % bj) // 2, den


def check_now(A, B, C, bias=None, alpha=1.0, beta=1.0, block=128, abs_floor=1e-20, z=6.0, epoch=0):
    """One check answered right away: (max_z, batch, row, col_block, noise)."""
    g = Group(C.device, C.dtype, block, abs_floor, z, epoch=epoch)
    g.add(A, B, C, bias, alpha, beta)
    g.launch()
    return g.row(0, g.result[0].tolist())
