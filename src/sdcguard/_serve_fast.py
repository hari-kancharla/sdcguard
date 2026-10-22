"""Fast serving check for NVIDIA GPUs (Triton).

The plain serving check (serve.check_static_impl) is about 40 small PyTorch operations per layer.
Under CUDA graphs each still costs a kernel, so on a 1.1B model it made decoding 2.6x slower
(H100, October 18). Three designs here do the same arithmetic per layer:

  v1  cuBLAS   E = x @ (block sums of W, as three bf16/fp16 parts)     (same call as the plain check)
      _check   for its rows and blocks: |x_row|^2 and |x_row|_1, the plain and signed sums, sum of
               squares and sum of |values| of each 128-column block of y, expected = parts of E summed
               x scale + bias sums, z of both checksums with the plain check's noise model and Inf/NaN
               rules, the worst z of the program, then two atomic maxima into the layer's state row
  v3  cuBLAS   E, the same call
      _sums    the four sums of each 128-wide block of y, and the row norms of x, in one launch
      _score   the same scoring as _check, one (row, block) pair per lane (see "v3" below)
  v5  _fused   one launch, no cuBLAS: E with Triton's dot, split over k, the row norms and the sums of y, and
               the last program of each tile scores it (see "v5" below)

check() runs v1 below V3_FROM_ROWS rows and v3 from there on: on an H100, v3 cost less than v1 summed over
Qwen2.5-7B's four linear layers from 256 rows on and more below (docs/METHOD.md, "The serving kernel"). v5 runs
below V5_BELOW_ROWS rows, 0 until a timing run says where it is cheaper. Each CUDA graph is captured for one batch
size, so the choice is fixed inside a graph. SDCGUARD_SERVE_KERNEL=v1, v3 or v5 forces one design (the timing run
and the tests use it).

The state row holds two int64 keys: (bits of z << 32) | position of the worst checksum, and
(bits of z << 32) | bits of its noise. z >= 0, and the bits of non-negative floats sort like the
floats, so an integer maximum keeps the worst z. No second pass, no host sync, safe in CUDA graphs.

Before its first use on a device and dtype each design in use must give the plain check's answers on clean
and corrupted layers of several shapes (verified()); otherwise the plain check is used.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import torch

from . import _group

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}          # (device, dtype) -> (ok, why)
FALLBACKS = [0]
WHY: list = []


def enabled() -> bool:
    return HAVE_TRITON and os.environ.get("SDCGUARD_FAST", "1") != "0" and \
        os.environ.get("SDCGUARD_SERVE_FAST", "1") != "0"


def _cpu_test() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1"


def usable(device: torch.device, dtype: torch.dtype) -> bool:
    """The fast check can run here (before the first-use comparison)."""
    if not enabled() or dtype not in (torch.bfloat16, torch.float16):
        return False
    return device.type == "cuda" or (device.type == "cpu" and _cpu_test())


def _pow2_floor(x: int) -> int:
    return 1 << (max(1, int(x)).bit_length() - 1)


def _pow2_ceil(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


TILE = 4096              # values of y (and of x) a program holds at once: at most ~160 registers, no spills


def config(m: int, k: int, nb: int, bs: int, device: torch.device, force_bm: Optional[int] = None):
    """(BM, BKX, CB, BW, blocks per program, programs over rows, programs over blocks, warps).
    Enough programs to fill the GPU (4 per SM): first by rows (each reads x once), then by
    splitting the blocks of a row across programs (they re-read the row of x: cheap for few rows).
    force_bm: rows per program (the first-use comparison tries every value)."""
    T = _group._target_programs(device)
    BW = max(16, _pow2_ceil(bs))
    BM = 16
    while BM > 1 and _group._cdiv(m, BM) < T:
        BM //= 2
    if force_bm:
        BM = force_bm
    rows_p = _group._cdiv(m, BM)
    CB = max(1, min(8, _pow2_floor(TILE // (BM * BW))))
    col_p = 1 if rows_p >= T else max(1, min(_group._cdiv(nb, CB), _group._cdiv(T, rows_p)))
    jbp = _group._cdiv(_group._cdiv(nb, col_p), CB) * CB
    col_p = _group._cdiv(nb, jbp)
    BKX = max(64, min(_pow2_ceil(k), _pow2_floor(TILE // BM)))
    return BM, BKX, CB, BW, jbp, rows_p, col_p, 4


if HAVE_TRITON:

    # m (rows: tokens in the batch) changes with every batch, so nothing may depend on its value: then the
    # variants compiled by warm() at load time cover every batch size (no compile while serving)
    @triton.jit(do_not_specialize=["slot", "jbp", "nb", "rem", "s_last", "m", "sc", "cst", "cw", "ctag", "cfo", "ncap"])
    def _check(X, Y, E, S, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, n, nb, bs, rem, s_full, s_last,
               sxm, sxk, sym, syn, es, two_nb, jbp, u, h, abs_floor, l1c, zt, lim_a, lim_big,
               CAPM, CAPD, cst, cw, ctag, cfo, ncap,
               BM: tl.constexpr, BKX: tl.constexpr, CB: tl.constexpr, BW: tl.constexpr, NPARTS: tl.constexpr,
               COUNTS: tl.constexpr, CAPTURE: tl.constexpr):
        pm = tl.program_id(0)
        pj = tl.program_id(1)
        rm = pm.to(tl.int64) * BM + tl.arange(0, BM)
        rmask = rm < m
        # |x_row|^2 and |x_row|_1 (fp32)
        asq = tl.zeros((BM,), dtype=tl.float32)
        asum = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, k, BKX):
            rk = k0 + tl.arange(0, BKX)
            xv = tl.load(X + rm[:, None] * sxm + rk[None, :] * sxk, mask=rmask[:, None] & (rk < k)[None, :],
                         other=0.0).to(tl.float32)
            asq += tl.sum(xv * xv, axis=1)
            asum += tl.sum(tl.abs(xv), axis=1)
        xn = tl.sqrt(asq)
        # per element of the chunk, over the whole loop: (bits of its worst z << 32) | its position, and
        # (bits of that z << 32) | bits of its noise; one reduction after the loop picks the program's worst
        run_k = tl.zeros((BM, CB), dtype=tl.int64)
        run_d = tl.zeros((BM, CB), dtype=tl.int64)
        unk_acc = tl.zeros((BM, CB), dtype=tl.int32)                      # checksums that couldn't be judged
        j_lo = pj * jbp
        j_hi = tl.minimum(j_lo + jbp, nb)
        w = tl.arange(0, BW)
        ci = tl.arange(0, CB)
        for j0 in range(j_lo, j_hi, CB):
            jb = j0 + ci
            bmask = jb < j_hi
            last = (jb == nb - 1) & (rem > 0)
            wid = tl.where(last, rem, bs)
            cmask = (w[None, :] < wid[:, None]) & bmask[:, None]                  # (CB, BW)
            cols = jb[:, None].to(tl.int64) * bs + w[None, :]
            sg = tl.load(S + tl.where(last, s_last, s_full)[:, None] + w[None, :], mask=cmask, other=0.0)
            yv = tl.load(Y + rm[:, None, None] * sym + cols[None, :, :] * syn,
                         mask=rmask[:, None, None] & cmask[None, :, :], other=0.0).to(tl.float32)
            plain = tl.sum(yv, axis=2)                                             # (BM, CB)
            signed = tl.sum(yv * sg[None, :, :], axis=2)
            sq = tl.sum(yv * yv, axis=2)
            l1 = tl.sum(tl.abs(yv), axis=2)
            both = rmask[:, None] & bmask[None, :]
            # expected = x @ B2: the parts of E added up in the plain check's order, x scale, + bias sums
            eo = rm[:, None] * es + 2 * jb[None, :]
            e0 = tl.zeros((BM, CB), dtype=tl.float32)
            e1 = tl.zeros((BM, CB), dtype=tl.float32)
            for i in tl.static_range(NPARTS):
                e0 += tl.load(E + eo + i * two_nb, mask=both, other=0.0)
                e1 += tl.load(E + eo + i * two_nb + 1, mask=both, other=0.0)
            if NPARTS == 3:      # (the parts were made from B2 / scale; plain fp32 B2 has no scale)
                e0 = e0 * tl.load(SCALE + 2 * jb, mask=bmask, other=1.0)[None, :]
                e1 = e1 * tl.load(SCALE + 2 * jb + 1, mask=bmask, other=1.0)[None, :]
            exp0 = e0 + tl.load(BSUM + 2 * jb, mask=bmask, other=0.0)[None, :]
            exp1 = e1 + tl.load(BSUM + 2 * jb + 1, mask=bmask, other=0.0)[None, :]
            # noise model of the plain check (serve.check_static_impl, checksum.py)
            bound = asum[:, None] * tl.load(BMAX + jb, mask=bmask, other=0.0)[None, :] + \
                tl.load(BMB + jb, mask=bmask, other=0.0)[None, :]
            noise = tl.sqrt(sq) * u
            # fp32 rounding of `expected`: the smaller of two bounds of it (checksum._zscores explains)
            fl0 = xn[:, None] * tl.load(B2N + 2 * jb, mask=bmask, other=0.0)[None, :] * 1e-5
            fl1 = xn[:, None] * tl.load(B2N + 2 * jb + 1, mask=bmask, other=0.0)[None, :] * 1e-5
            hl0 = asum[:, None] * tl.load(B2M + 2 * jb, mask=bmask, other=0.0)[None, :] * l1c
            hl1 = asum[:, None] * tl.load(B2M + 2 * jb + 1, mask=bmask, other=0.0)[None, :] * l1c
            fl0 = tl.minimum(tl.where(tl.abs(fl0) < float("inf"), fl0, float("inf")),
                             tl.where(tl.abs(hl0) < float("inf"), hl0, float("inf")))
            fl1 = tl.minimum(tl.where(tl.abs(fl1) < float("inf"), fl1, float("inf")),
                             tl.where(tl.abs(hl1) < float("inf"), hl1, float("inf")))
            fl0 = tl.where(fl0 < float("inf"), fl0, bound * (1e-5 * 128))
            fl1 = tl.where(fl1 < float("inf"), fl1, bound * (1e-5 * 128))
            wf = wid.to(tl.float32)
            tail = (abs_floor + tl.sqrt(wf) * h)[None, :]
            den0 = (noise + fl0) + tail
            den1 = (noise + fl1) + tail
            d0 = tl.abs(exp0 - plain)
            d1 = tl.abs(exp1 - signed)
            z0 = d0 / den0
            z1 = d1 / den1
            # the plain sum must also beat the worst rounding possible (equal outputs round alike)
            worst = (l1 * (1.25 * u) + (1.25 * h * wf)[None, :]) + fl0 + abs_floor
            zw = d0 / worst * zt
            z0 = tl.where((z0 == z0) & (zw == zw), tl.minimum(z0, zw), float("nan"))
            # Inf/NaN, judged against what the inputs allow
            lim = bound * 1.01
            nfin = ~(noise < float("inf"))
            e_bad0 = ~(tl.abs(exp0) < float("inf"))
            a_bad0 = (~e_bad0) & ~(tl.abs(plain) < float("inf"))
            big0 = nfin & (~e_bad0) & (~a_bad0)
            nan0 = ~(z0 == z0)
            z0 = tl.where(z0 == z0, z0, 0.0)
            z0 = tl.where(e_bad0, 0.0, z0)
            # finite outputs too large for the inputs (checksum._judge_nonfinite explains)
            cap = bound * (1.02 + k * 2.384185791015625e-07) + 2.0 * h
            imposs = (~nfin) & (lim < lim_big) & (tl.sqrt(sq) > cap * tl.sqrt(wf)[None, :])
            z0 = tl.where(imposs & (~e_bad0) & (~a_bad0), float("inf"), z0)
            z0 = tl.where(a_bad0, tl.where(lim < lim_a, float("inf"), 0.0), z0)
            z0 = tl.where(big0, tl.where(lim < lim_big, float("inf"), 0.0), z0)
            e_bad1 = ~(tl.abs(exp1) < float("inf"))
            a_bad1 = (~e_bad1) & ~(tl.abs(signed) < float("inf"))
            big1 = nfin & (~e_bad1) & (~a_bad1)
            nan1 = ~(z1 == z1)
            z1 = tl.where(z1 == z1, z1, 0.0)
            z1 = tl.where(e_bad1, 0.0, z1)
            z1 = tl.where(imposs & (~e_bad1) & (~a_bad1), float("inf"), z1)
            z1 = tl.where(a_bad1, tl.where(lim < lim_a, float("inf"), 0.0), z1)
            z1 = tl.where(big1, tl.where(lim < lim_big, float("inf"), 0.0), z1)
            # unjudged (they score 0 above), as in serve.check_static_impl and checksum._judge_nonfinite
            ok_a = lim < lim_a
            ok_big = lim < lim_big
            unk0 = (nan0 & (~imposs) & (~a_bad0) & (~big0)) | e_bad0 | (a_bad0 & (~ok_a)) | (big0 & (~ok_big))
            unk1 = (nan1 & (~imposs) & (~a_bad1) & (~big1)) | e_bad1 | (a_bad1 & (~ok_a)) | (big1 & (~ok_big))
            unk_acc += tl.where(both & unk0, 1, 0) + tl.where(both & unk1, 1, 0)
            z0 = tl.where(both, z0, -1.0)
            z1 = tl.where(both, z1, -1.0)
            # worst of this chunk (the plain checksum on ties), its position and noise
            sec = z1 > z0
            zc = tl.where(sec, z1, z0)
            # (NaN where the output's checksum was Inf/NaN, as in serve.check_static_impl)
            dc = tl.where(sec, tl.where(a_bad1, float("nan"), den1), tl.where(a_bad0, float("nan"), den0))
            zb_e = tl.maximum(zc, 0.0).to(tl.int32, bitcast=True).to(tl.int64)
            db_e = dc.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
            flat_e = (rm[:, None] * nb + jb[None, :]) * 2 + tl.where(sec, 1, 0).to(tl.int64)
            run_k = tl.maximum(run_k, tl.where(both, (zb_e << 32) | (flat_e & 0xFFFFFFFF), 0))
            run_d = tl.maximum(run_d, tl.where(both, (zb_e << 32) | db_e, 0))
        key = tl.max(tl.max(run_k, axis=1), axis=0)
        keyd = tl.max(tl.max(run_d, axis=1), axis=0)
        unk_n = tl.sum(tl.sum(unk_acc, axis=1), axis=0)
        zbits = key >> 32
        best_i = key & 0xFFFFFFFF
        best = zbits.to(tl.int32).to(tl.float32, bitcast=True)
        tl.atomic_max(STATE + slot * sc, key)
        tl.atomic_max(STATE + slot * sc + 1, keyd)
        if COUNTS:           # (serve.STATE_COLS: one check executed, from one program, and the unjudged count)
            first = (pm == 0) & (pj == 0)
            tl.atomic_add(STATE + slot * sc + 2, tl.where(first, 1, 0).to(tl.int64), mask=first)
            tl.atomic_add(STATE + slot * sc + 3, unk_n.to(tl.int64), mask=unk_n > 0)
        if CAPTURE:          # (serve.CAPTURE_SLOTS: only a program whose worst row is over the line gets here)
            if best > zt:
                taken = tl.atomic_xchg(CAPM + cfo + slot, 1)          # the layer's first row since the last read?
                if taken == 0:
                    c = tl.atomic_add(CAPM, 1)                        # a record of its own
                    if c < ncap:
                        bi = best_i // 2
                        crow = bi // nb
                        cblk = bi % nb
                        rec = CAPM + 2 + c * 6
                        tl.store(rec, ctag.to(tl.int64))
                        tl.store(rec + 1, crow.to(tl.int64))
                        tl.store(rec + 2, cblk.to(tl.int64))
                        tl.store(rec + 3, best_i.to(tl.int64))
                        tl.store(rec + 4, zbits)
                        tl.store(rec + 5, tl.load(CAPM + 1))
                        dst = CAPD + c.to(tl.int64) * cst
                        cwid = tl.where((cblk == nb - 1) & (rem > 0), rem, bs)
                        yv1 = tl.load(Y + crow * sym + (cblk * bs + w) * syn, mask=w < cwid, other=0.0)
                        tl.store(dst + w, yv1.to(tl.float32), mask=w < cwid)
                        for k0 in range(0, k, BKX):
                            rk = k0 + tl.arange(0, BKX)
                            xv1 = tl.load(X + crow * sxm + rk * sxk, mask=rk < k, other=0.0)
                            tl.store(dst + cw + rk, xv1.to(tl.float32), mask=rk < k)


def check_v1(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor, pair_rem: torch.Tensor,
          scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor,
          bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor, state: torch.Tensor, slot: int, z: float,
          abs_floor: float, _bm: Optional[int] = None, cap_meta: Optional[torch.Tensor] = None,
          cap_data: Optional[torch.Tensor] = None, cap_w: int = 0, cap_tag: int = 0) -> None:
    """Same arguments and result as serve.check_static_impl (the state row gets the worst z, and with
    serve.STATE_COLS columns the counts of checks executed and checksums that couldn't be judged; with cap_meta
    and cap_data, a row over the line is captured as serve.CAPTURE_SLOTS describes)."""
    from .checksum import BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF, l1_floor_coef
    from .serve import CAP_FLAGS, CAPTURE_SLOTS, STATE_COLS
    counts = state.dim() == 2 and state.shape[1] >= STATE_COLS
    m, n = y2.shape
    if m == 0 or n == 0:     # nothing to check (and not counted as a check, as in the plain check)
        return
    dt = y2.dtype
    if x2.dtype != dt:
        x2 = x2.to(dt)
    k = x2.shape[1]
    nb = b2n.shape[0]
    bs = pair_full.shape[0]
    if bs > 128:            # the sign table holds widths 1..128 (serve.check_static_any routes wider blocks away)
        raise ValueError(f"the Triton serving check takes blocks up to 128 wide, not {bs}")
    rem = n - (n // bs) * bs
    if parts.shape[1] == 6 * nb:                      # three low-precision parts (NVIDIA)
        if parts.dtype == dt and x2.is_cuda:
            E = torch.ops.aten.mm.dtype(x2, parts, torch.float32)
        else:
            E = x2.float() @ parts.float()
        nparts = 3
    else:                                             # plain fp32 block sums (CPU)
        E = x2.float() @ parts.float()
        nparts = 1
    BM, BKX, CB, BW, jbp, rows_p, col_p, warps = config(m, k, nb, bs, y2.device, _bm)
    st = _group.pipeline_stages(y2.device, None)          # 1 on Blackwell (_group.pipeline_stages explains)
    stages = {} if st is None else {"num_stages": st}
    capture = cap_meta is not None and cap_data is not None
    if capture and not (cap_meta.dtype == torch.int64 and cap_data.dtype == torch.float32 and cap_w >= bs
                        and cap_data.shape[1] >= cap_w + k and cap_meta.numel() > CAP_FLAGS + slot):
        raise ValueError("the capture buffer doesn't fit this layer")
    with _group._on(y2.device):
        _check[(rows_p, col_p)](
            x2, y2, E, _group._sign_table(y2.device), scale, b2n, b2max, bmax, bmax_bias, bsum, state, slot,
            state.stride(0), m, k, n, nb,
            bs, rem, _group._sign_offset(bs), _group._sign_offset(rem) if rem else 0, x2.stride(0), x2.stride(1),
            y2.stride(0), y2.stride(1), E.stride(0), 2 * nb, jbp, float(UNIT_ROUNDOFF[dt]), float(HALF_TINY_STEP[dt]),
            float(abs_floor), float(l1_floor_coef(k)), float(z), float(min(DTYPE_MAX[dt], BIG)), float(BIG),
            cap_meta if capture else state, cap_data if capture else E,
            cap_data.stride(0) if capture else 0, int(cap_w), int(cap_tag), CAP_FLAGS, CAPTURE_SLOTS,
            BM=BM, BKX=BKX, CB=CB, BW=BW, NPARTS=nparts, COUNTS=counts, CAPTURE=capture, num_warps=warps, **stages)


# ---------------------------------------------------------------- v3: reading y apart from scoring it
#
# v2 (two kernels, _emm and _pairs, October 22) agreed with the plain check everywhere but cost about twice as much
# as _check on an H100: its programs read their block of y and then scored it in the same code, and the scoring's
# many per-block values left too few programs resident to keep loads in flight (it got slower as each program took
# more blocks: 16 < 32 < 64). v3 keeps cuBLAS for E and splits the rest by what each part needs:
#
#   _sums   one launch, two jobs: programs below the split read P whole 128-wide blocks of y (a P x 128 tile, the
#           blocks of a row side by side) and write the four sums of each block (plain, signed, of squares and of
#           |values|, fp32); the programs after it read rows of x and write |x_row|_2^2 and |x_row|_1. Few values
#           live per program, so many programs share an SM and loads overlap.
#   _score  one (row, block) pair per lane: the four sums, E's parts, the row norms and the block's constants, the
#           same scoring as _check, one atomic maximum per program and only when it raises the layer's state.
#
# Deterministic: every sum is made in one program, in a fixed order.

if HAVE_TRITON:

    @triton.jit
    def _judge(plain, signed, sq, l1, xn, asum, e0, e1, jb, wid, ok, SCALE, B2N, B2M, BMAX, BMB, BSUM, k, u, h,
               abs_floor, l1c, zt, lim_a, lim_big, NPARTS: tl.constexpr):
        """The scoring of _check for a vector of (row, block) pairs (v3's _score and v5's _fused share it): from the
        four sums of the block of y, the row norms of x and E's parts summed (e0, e1), (z of the worse checksum, its
        noise, whether it is the signed one, and how many of the pair's two checksums couldn't be judged). ok masks
        the pairs that exist."""
        if NPARTS == 3:      # (the parts were made from B2 / scale; plain fp32 B2 has no scale)
            e0 = e0 * tl.load(SCALE + 2 * jb, mask=ok, other=1.0)
            e1 = e1 * tl.load(SCALE + 2 * jb + 1, mask=ok, other=1.0)
        exp0 = e0 + tl.load(BSUM + 2 * jb, mask=ok, other=0.0)
        exp1 = e1 + tl.load(BSUM + 2 * jb + 1, mask=ok, other=0.0)
        # noise model of the plain check (serve.check_static_impl, checksum.py), as in _check
        bound = asum * tl.load(BMAX + jb, mask=ok, other=0.0) + tl.load(BMB + jb, mask=ok, other=0.0)
        noise = tl.sqrt(sq) * u
        fl0 = xn * tl.load(B2N + 2 * jb, mask=ok, other=0.0) * 1e-5
        fl1 = xn * tl.load(B2N + 2 * jb + 1, mask=ok, other=0.0) * 1e-5
        hl0 = asum * tl.load(B2M + 2 * jb, mask=ok, other=0.0) * l1c
        hl1 = asum * tl.load(B2M + 2 * jb + 1, mask=ok, other=0.0) * l1c
        fl0 = tl.minimum(tl.where(tl.abs(fl0) < float("inf"), fl0, float("inf")),
                         tl.where(tl.abs(hl0) < float("inf"), hl0, float("inf")))
        fl1 = tl.minimum(tl.where(tl.abs(fl1) < float("inf"), fl1, float("inf")),
                         tl.where(tl.abs(hl1) < float("inf"), hl1, float("inf")))
        fl0 = tl.where(fl0 < float("inf"), fl0, bound * (1e-5 * 128))
        fl1 = tl.where(fl1 < float("inf"), fl1, bound * (1e-5 * 128))
        wf = wid.to(tl.float32)
        tail = abs_floor + tl.sqrt(wf) * h
        den0 = (noise + fl0) + tail
        den1 = (noise + fl1) + tail
        d0 = tl.abs(exp0 - plain)
        d1 = tl.abs(exp1 - signed)
        z0 = d0 / den0
        z1 = d1 / den1
        worst = (l1 * (1.25 * u) + (1.25 * h * wf)) + fl0 + abs_floor
        zw = d0 / worst * zt
        z0 = tl.where((z0 == z0) & (zw == zw), tl.minimum(z0, zw), float("nan"))
        lim = bound * 1.01
        nfin = ~(noise < float("inf"))
        e_bad0 = ~(tl.abs(exp0) < float("inf"))
        a_bad0 = (~e_bad0) & ~(tl.abs(plain) < float("inf"))
        big0 = nfin & (~e_bad0) & (~a_bad0)
        nan0 = ~(z0 == z0)
        z0 = tl.where(z0 == z0, z0, 0.0)
        z0 = tl.where(e_bad0, 0.0, z0)
        cap = bound * (1.02 + k * 2.384185791015625e-07) + 2.0 * h
        imposs = (~nfin) & (lim < lim_big) & (tl.sqrt(sq) > cap * tl.sqrt(wf))
        z0 = tl.where(imposs & (~e_bad0) & (~a_bad0), float("inf"), z0)
        z0 = tl.where(a_bad0, tl.where(lim < lim_a, float("inf"), 0.0), z0)
        z0 = tl.where(big0, tl.where(lim < lim_big, float("inf"), 0.0), z0)
        e_bad1 = ~(tl.abs(exp1) < float("inf"))
        a_bad1 = (~e_bad1) & ~(tl.abs(signed) < float("inf"))
        big1 = nfin & (~e_bad1) & (~a_bad1)
        nan1 = ~(z1 == z1)
        z1 = tl.where(z1 == z1, z1, 0.0)
        z1 = tl.where(e_bad1, 0.0, z1)
        z1 = tl.where(imposs & (~e_bad1) & (~a_bad1), float("inf"), z1)
        z1 = tl.where(a_bad1, tl.where(lim < lim_a, float("inf"), 0.0), z1)
        z1 = tl.where(big1, tl.where(lim < lim_big, float("inf"), 0.0), z1)
        ok_a = lim < lim_a
        ok_big = lim < lim_big
        unk0 = (nan0 & (~imposs) & (~a_bad0) & (~big0)) | e_bad0 | (a_bad0 & (~ok_a)) | (big0 & (~ok_big))
        unk1 = (nan1 & (~imposs) & (~a_bad1) & (~big1)) | e_bad1 | (a_bad1 & (~ok_a)) | (big1 & (~ok_big))
        unk = tl.where(ok & unk0, 1, 0) + tl.where(ok & unk1, 1, 0)
        z0 = tl.where(ok, z0, -1.0)
        z1 = tl.where(ok, z1, -1.0)
        sec = z1 > z0
        zc = tl.where(sec, z1, z0)
        dc = tl.where(sec, tl.where(a_bad1, float("nan"), den1), tl.where(a_bad0, float("nan"), den0))
        return zc, dc, sec, unk

    @triton.jit
    def _report(zc, dc, sec, unk, ok, pp, first, X, Y, STATE, slot, sc, nb, bs, rem, k, sxm, sxk, sym, syn, zt,
                CAPM, CAPD, cst, cw, ctag, cfo, ncap, BW: tl.constexpr, BKX: tl.constexpr, COUNTS: tl.constexpr,
                CAPTURE: tl.constexpr):
        """What _judge found for a vector of pairs (pp: their flat positions, row * nb + block) into the layer's
        state row: the worst score's two keys (an atomic maximum, only when it raises the state), the counts (first:
        this program counts the check as executed), and the capture of a row over the line (serve.CAPTURE_SLOTS)."""
        unk_n = tl.sum(unk, axis=0)
        zb_e = tl.maximum(zc, 0.0).to(tl.int32, bitcast=True).to(tl.int64)
        db_e = dc.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        flat_e = pp * 2 + tl.where(sec, 1, 0).to(tl.int64)
        key = tl.max(tl.where(ok, (zb_e << 32) | (flat_e & 0xFFFFFFFF), 0), axis=0)
        keyd = tl.max(tl.where(ok, (zb_e << 32) | db_e, 0), axis=0)
        # most programs can't raise the layer's worst score: read it first, and only those that can take the atomic
        if key > tl.load(STATE + slot * sc):
            tl.atomic_max(STATE + slot * sc, key)
        if keyd > tl.load(STATE + slot * sc + 1):
            tl.atomic_max(STATE + slot * sc + 1, keyd)
        if COUNTS:
            tl.atomic_add(STATE + slot * sc + 2, tl.where(first, 1, 0).to(tl.int64), mask=first)
            tl.atomic_add(STATE + slot * sc + 3, unk_n.to(tl.int64), mask=unk_n > 0)
        if CAPTURE:
            zbits = key >> 32
            best_i = key & 0xFFFFFFFF
            best = zbits.to(tl.int32).to(tl.float32, bitcast=True)
            if best > zt:
                taken = tl.atomic_xchg(CAPM + cfo + slot, 1)
                if taken == 0:
                    cnum = tl.atomic_add(CAPM, 1)
                    if cnum < ncap:
                        bi = best_i // 2
                        crow = bi // nb
                        cblk = bi % nb
                        rec = CAPM + 2 + cnum * 6
                        tl.store(rec, ctag.to(tl.int64))
                        tl.store(rec + 1, crow.to(tl.int64))
                        tl.store(rec + 2, cblk.to(tl.int64))
                        tl.store(rec + 3, best_i.to(tl.int64))
                        tl.store(rec + 4, zbits)
                        tl.store(rec + 5, tl.load(CAPM + 1))
                        dst = CAPD + cnum.to(tl.int64) * cst
                        w = tl.arange(0, BW)
                        cwid = tl.where((cblk == nb - 1) & (rem > 0), rem, bs)
                        yv1 = tl.load(Y + crow * sym + (cblk * bs + w) * syn, mask=w < cwid, other=0.0)
                        tl.store(dst + w, yv1.to(tl.float32), mask=w < cwid)
                        for k0 in range(0, k, BKX):
                            rk = k0 + tl.arange(0, BKX)
                            xv1 = tl.load(X + crow * sxm + rk * sxk, mask=rk < k, other=0.0)
                            tl.store(dst + cw + rk, xv1.to(tl.float32), mask=rk < k)

    @triton.jit(do_not_specialize=["m", "npairs", "nyt", "nb", "rem", "s_last"])
    def _sums(X, Y, S, OUT, XN, m, k, nb, bs, rem, s_full, s_last, sxm, sxk, sym, syn, npairs, nyt,
              P: tl.constexpr, BW: tl.constexpr, XR: tl.constexpr, BKX: tl.constexpr, FLAT: tl.constexpr):
        pid = tl.program_id(0)
        if pid < nyt:
            pp = pid.to(tl.int64) * P + tl.arange(0, P)                   # (row, block) pairs
            pmask = pp < npairs
            w = tl.arange(0, BW)
            if FLAT:
                # every block full width and y contiguous: pair p is the 128 values starting at p * bs
                yv = tl.load(Y + pp[:, None] * bs + w[None, :], mask=pmask[:, None] & (w < bs)[None, :],
                             other=0.0).to(tl.float32)
                sg = tl.load(S + s_full + w, mask=w < bs, other=0.0)[None, :]
            else:
                r = pp // nb
                jb = pp - r * nb
                last = (jb == nb - 1) & (rem > 0)
                wid = tl.where(last, rem, bs)
                cmask = (w[None, :] < wid[:, None]) & pmask[:, None]
                sg_full = tl.load(S + s_full + w, mask=w < bs, other=0.0)
                sg_last = tl.load(S + s_last + w, mask=w < rem, other=0.0)
                sg = tl.where(last[:, None], sg_last[None, :], sg_full[None, :])
                yv = tl.load(Y + r[:, None] * sym + (jb[:, None] * bs + w[None, :]) * syn, mask=cmask,
                             other=0.0).to(tl.float32)
            # one row of OUT per sum (4 x npairs): neighbouring pairs' sums are neighbours, for both kernels
            tl.store(OUT + pp, tl.sum(yv, axis=1), mask=pmask)
            tl.store(OUT + npairs + pp, tl.sum(yv * sg, axis=1), mask=pmask)
            tl.store(OUT + 2 * npairs + pp, tl.sum(yv * yv, axis=1), mask=pmask)
            tl.store(OUT + 3 * npairs + pp, tl.sum(tl.abs(yv), axis=1), mask=pmask)
        else:
            rm = (pid - nyt).to(tl.int64) * XR + tl.arange(0, XR)
            rmask = rm < m
            asq = tl.zeros((XR,), dtype=tl.float32)
            asum = tl.zeros((XR,), dtype=tl.float32)
            for k0 in range(0, k, BKX):
                rk = k0 + tl.arange(0, BKX)
                xv = tl.load(X + rm[:, None] * sxm + rk[None, :] * sxk, mask=rmask[:, None] & (rk < k)[None, :],
                             other=0.0).to(tl.float32)
                asq += tl.sum(xv * xv, axis=1)
                asum += tl.sum(tl.abs(xv), axis=1)
            tl.store(XN + rm, asq, mask=rmask)                            # (2 x m)
            tl.store(XN + m + rm, asum, mask=rmask)

    @triton.jit(do_not_specialize=["slot", "sc", "m", "nb", "rem", "npairs", "cst", "cw", "ctag", "cfo", "ncap"])
    def _score(X, Y, E, SUMS, XN, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, nb, bs, rem,
               sxm, sxk, sym, syn, es, two_nb, npairs, u, h, abs_floor, l1c, zt, lim_a, lim_big,
               CAPM, CAPD, cst, cw, ctag, cfo, ncap,
               Q: tl.constexpr, BW: tl.constexpr, BKX: tl.constexpr, NPARTS: tl.constexpr, COUNTS: tl.constexpr,
               CAPTURE: tl.constexpr):
        pid = tl.program_id(0)
        pp = pid.to(tl.int64) * Q + tl.arange(0, Q)
        pmask = pp < npairs
        r = pp // nb
        jb = pp - r * nb
        last = (jb == nb - 1) & (rem > 0)
        wid = tl.where(last, rem, bs)
        plain = tl.load(SUMS + pp, mask=pmask, other=0.0)
        signed = tl.load(SUMS + npairs + pp, mask=pmask, other=0.0)
        sq = tl.load(SUMS + 2 * npairs + pp, mask=pmask, other=0.0)
        l1 = tl.load(SUMS + 3 * npairs + pp, mask=pmask, other=0.0)
        xn = tl.sqrt(tl.load(XN + r, mask=pmask, other=0.0))
        asum = tl.load(XN + m + r, mask=pmask, other=0.0)
        # expected = x @ B2: the parts of E added up in the plain check's order (then x scale, + bias sums)
        eo = r * es + 2 * jb
        e0 = tl.zeros((Q,), dtype=tl.float32)
        e1 = tl.zeros((Q,), dtype=tl.float32)
        for i in tl.static_range(NPARTS):
            e0 += tl.load(E + eo + i * two_nb, mask=pmask, other=0.0)
            e1 += tl.load(E + eo + i * two_nb + 1, mask=pmask, other=0.0)
        zc, dc, sec, unk = _judge(plain, signed, sq, l1, xn, asum, e0, e1, jb, wid, pmask, SCALE, B2N, B2M, BMAX,
                                  BMB, BSUM, k, u, h, abs_floor, l1c, zt, lim_a, lim_big, NPARTS)
        _report(zc, dc, sec, unk, pmask, pp, pid == 0, X, Y, STATE, slot, sc, nb, bs, rem, k, sxm, sxk, sym, syn,
                zt, CAPM, CAPD, cst, cw, ctag, cfo, ncap, BW, BKX, COUNTS, CAPTURE)


# ---------------------------------------------------------------- v5: one kernel per layer for few rows
#
# With few rows v1 and v3 cost mostly fixed amounts per layer (the October 22 profile): cuBLAS spends 7 to 8 us on
# E at 1 row though its data loads in 0.4 to 4, and each further kernel adds at least 2 to 3 us and a gap. v5 does
# the whole check in one launch, with no cuBLAS call:
#
#   _fused  programs (split s of the sum over k, G blocks of y, BM rows). Each computes its part of E for its blocks
#           and rows with Triton's dot (fp32 accumulation, as cuBLAS) and of the row norms of x from the same loads,
#           and the four sums of some rows of its blocks of y (rows s, s + NS, ...), and writes them to a scratch
#           buffer. Then it adds 1 to its tile's counter; the program that finds NS - 1 there is the last of its tile:
#           it resets the counter, adds the NS parts in a fixed order (deterministic) and scores the tile's pairs
#           with _judge, as _score does.
#
# The counters are zero between launches (each last program resets its own). Each layer has its own counters, in
# V5_SETS sets, one per CUDA stream that runs v5: launches on one stream run one after another, and two streams never
# share a set (_counter_slice). Work captured into a CUDA graph keeps the set of the stream it was captured on, so one
# layer's graphs captured on one stream must not be replayed on two streams at the same time (vLLM captures on one
# stream and replays on one). A count left over by such misuse would make every later check of that tile wrong.
# Its sums run in another order than cuBLAS's, so a score can differ from v1's in its last bits; it is held to the
# plain check's answers, as v3 is.

if HAVE_TRITON:

    @triton.jit
    def _fused_tail(X, Y, EP, XP, YS, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, nb, bs, rem,
                    sxm, sxk, sym, syn, tile, rt, ct, u, h, abs_floor, l1c, zt, lim_a, lim_big,
                    CAPM, CAPD, cst, cw, ctag, cfo, ncap,
                    BM: tl.constexpr, G: tl.constexpr, BW: tl.constexpr, NS: tl.constexpr, SP: tl.constexpr,
                    NPARTS: tl.constexpr, BKX: tl.constexpr, COUNTS: tl.constexpr, CAPTURE: tl.constexpr):
        """The last program of a tile: the NS parts of E and of the row norms added up, the sums of y, scoring."""
        q = tl.arange(0, BM * G)                                             # the tile's (row, block) pairs
        rl = q // G
        gq = q - rl * G
        rg = rt.to(tl.int64) * BM + rl
        jb = ct * G + gq
        ok = (rg < m) & (jb < nb)
        last = (jb == nb - 1) & (rem > 0)
        wid = tl.where(last, rem, bs)
        sp = tl.arange(0, SP)
        lm = (sp < NS)[:, None] & ok[None, :]
        TS: tl.constexpr = BM * 2 * G
        # (written by the tile's other programs in this launch: read from L2, not from this SM's L1)
        eb = EP + (tile * NS).to(tl.int64) * (NPARTS * TS)
        eo = sp[:, None] * (NPARTS * TS) + (rl * (2 * G) + 2 * gq)[None, :]
        e0 = tl.zeros((BM * G,), dtype=tl.float32)
        e1 = tl.zeros((BM * G,), dtype=tl.float32)
        for i in tl.static_range(NPARTS):
            e0 += tl.sum(tl.load(eb + eo + i * TS, mask=lm, other=0.0, cache_modifier=".cg"), axis=0)
            e1 += tl.sum(tl.load(eb + eo + i * TS + 1, mask=lm, other=0.0, cache_modifier=".cg"), axis=0)
        xb = XP + (tile * NS).to(tl.int64) * (2 * BM)
        xo = sp[:, None] * (2 * BM) + rl[None, :]
        asq = tl.sum(tl.load(xb + xo, mask=lm, other=0.0, cache_modifier=".cg"), axis=0)
        asum = tl.sum(tl.load(xb + BM + xo, mask=lm, other=0.0, cache_modifier=".cg"), axis=0)
        xn = tl.sqrt(asq)
        yb = YS + tile.to(tl.int64) * (4 * BM * G)
        plain = tl.load(yb + q, mask=ok, other=0.0, cache_modifier=".cg")
        signed = tl.load(yb + BM * G + q, mask=ok, other=0.0, cache_modifier=".cg")
        sq = tl.load(yb + 2 * BM * G + q, mask=ok, other=0.0, cache_modifier=".cg")
        l1 = tl.load(yb + 3 * BM * G + q, mask=ok, other=0.0, cache_modifier=".cg")
        zc, dc, sec, unk = _judge(plain, signed, sq, l1, xn, asum, e0, e1, jb, wid, ok, SCALE, B2N, B2M, BMAX, BMB,
                                  BSUM, k, u, h, abs_floor, l1c, zt, lim_a, lim_big, NPARTS)
        _report(zc, dc, sec, unk, ok, rg * nb + jb, tile == 0, X, Y, STATE, slot, sc, nb, bs, rem, k, sxm, sxk, sym,
                syn, zt, CAPM, CAPD, cst, cw, ctag, cfo, ncap, BW, BKX, COUNTS, CAPTURE)

    @triton.jit
    def _fused_ysums(Y, S, YS, m, nb, bs, rem, s_full, s_last, sym, syn, tile, rt, ct, ps, j,
                     BM: tl.constexpr, G: tl.constexpr, BW: tl.constexpr, NS: tl.constexpr, FULL: tl.constexpr):
        """The four sums of the tile's blocks in one of its rows (local row ps + NS * j). FULL: every block is bs
        wide (bs = BW), so the masks are whole blocks and the loads vectorize."""
        g = tl.arange(0, G)
        jb = ct * G + g
        bmask = jb < nb
        w = tl.arange(0, BW)
        rl = ps + NS * j
        rg = rt.to(tl.int64) * BM + rl
        ok_r = (rl < BM) & (rg < m)
        if FULL:
            wmask = bmask[:, None] & (w < BW)[None, :]
            sg = tl.load(S + s_full + w)[None, :]
        else:
            last = (jb == nb - 1) & (rem > 0)
            wid = tl.where(last, rem, bs)
            wmask = (w[None, :] < wid[:, None]) & bmask[:, None]                  # (G, BW)
            sg = tl.load(S + tl.where(last, s_last, s_full)[:, None] + w[None, :], mask=wmask & ok_r, other=0.0)
        yv = tl.load(Y + rg * sym + (jb[:, None].to(tl.int64) * bs + w[None, :]) * syn, mask=wmask & ok_r,
                     other=0.0).to(tl.float32)
        o = rl * G + g
        yb = YS + tile.to(tl.int64) * (4 * BM * G)
        sm = bmask & ok_r
        tl.store(yb + o, tl.sum(yv, axis=1), mask=sm)
        tl.store(yb + BM * G + o, tl.sum(yv * sg, axis=1), mask=sm)
        tl.store(yb + 2 * BM * G + o, tl.sum(yv * yv, axis=1), mask=sm)
        tl.store(yb + 3 * BM * G + o, tl.sum(tl.abs(yv), axis=1), mask=sm)

    @triton.jit
    def _fused_e(X, P, rm, rmask, ct, k_lo, k_hi, k, sxm, sxk, spk, spc, two_nb, BM: tl.constexpr, G: tl.constexpr,
                 BK: tl.constexpr, NPARTS: tl.constexpr, IEEE: tl.constexpr, COLMASK: tl.constexpr):
        """This program's part of E (rows rm, the tile's 2 G columns of each part, k in [k_lo, k_hi)) and of the row
        norms of x from the same loads."""
        cc = tl.arange(0, 2 * G)
        col = ct * (2 * G) + cc
        a0 = tl.zeros((BM, 2 * G), dtype=tl.float32)
        a1 = tl.zeros((BM, 2 * G), dtype=tl.float32)
        a2 = tl.zeros((BM, 2 * G), dtype=tl.float32)
        asq = tl.zeros((BM,), dtype=tl.float32)
        asum = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BK):
            rk = k0 + tl.arange(0, BK)
            # (the split's length is a multiple of BK, so only the end of k cuts a step short: masking against k,
            # whose divisibility the compiler knows, keeps the loads vectorized)
            kmask = rk < k
            xv = tl.load(X + rm[:, None] * sxm + rk[None, :] * sxk, mask=rmask[:, None] & kmask[None, :], other=0.0)
            xf = xv.to(tl.float32)
            asq += tl.sum(xf * xf, axis=1)
            asum += tl.sum(tl.abs(xf), axis=1)
            pb = P + rk[:, None] * spk + col[None, :] * spc
            if COLMASK:
                pmk = kmask[:, None] & (col < two_nb)[None, :]
            else:
                pmk = kmask[:, None] & (cc < 2 * G)[None, :]
            p0 = tl.load(pb, mask=pmk, other=0.0)
            if IEEE:      # (Triton's CPU interpreter gets bf16 dots wrong; widening to fp32 is exact)
                a0 = tl.dot(xf, p0.to(tl.float32), a0, input_precision="ieee")
            else:
                a0 = tl.dot(xv, p0, a0)
            if NPARTS == 3:
                p1 = tl.load(pb + two_nb * spc, mask=pmk, other=0.0)
                p2 = tl.load(pb + 2 * two_nb * spc, mask=pmk, other=0.0)
                if IEEE:
                    a1 = tl.dot(xf, p1.to(tl.float32), a1, input_precision="ieee")
                    a2 = tl.dot(xf, p2.to(tl.float32), a2, input_precision="ieee")
                else:
                    a1 = tl.dot(xv, p1, a1)
                    a2 = tl.dot(xv, p2, a2)
        return a0, a1, a2, asq, asum

    # only what changes with the batch (m), or between layers without changing the work's shape, is left
    # unspecialized; the strides stay specialized, so contiguous x, y and parts get vectorized loads
    @triton.jit(do_not_specialize=["slot", "sc", "m", "nb", "rem", "s_last", "nct", "cst", "cw", "ctag", "cfo",
                                   "ncap"])
    def _fused(X, Y, P, S, EP, XP, YS, CNT, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, nb, bs, rem,
               s_full, s_last, sxm, sxk, sym, syn, spk, spc, two_nb, nct, kc, u, h, abs_floor, l1c, zt, lim_a,
               lim_big, CAPM, CAPD, cst, cw, ctag, cfo, ncap,
               BM: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, BW: tl.constexpr, NS: tl.constexpr,
               SP: tl.constexpr, RPS: tl.constexpr, NPARTS: tl.constexpr, IEEE: tl.constexpr, BKX: tl.constexpr,
               SPKU: tl.constexpr, NB2U: tl.constexpr, FULL: tl.constexpr, COUNTS: tl.constexpr,
               CAPTURE: tl.constexpr):
        ps = tl.program_id(0)                 # which part of the sum over k
        ct = tl.program_id(1)                 # which G blocks of y (2 G columns of each part of E)
        rt = tl.program_id(2)                 # which BM rows
        tile = rt * nct + ct
        r = tl.arange(0, BM)
        rm = rt.to(tl.int64) * BM + r
        rmask = rm < m
        cc = tl.arange(0, 2 * G)
        # spk and two_nb arrive divided by SPKU and NB2U (8 when they are multiples of 8): multiplied back here, the
        # compiler sees that a row of parts and each part's columns start on 16 bytes, so the loads vectorize
        spk = spk * SPKU
        two_nb = two_nb * NB2U
        k_lo = ps * kc
        k_hi = tl.minimum(k_lo + kc, k)
        # a tile whose blocks all exist needs no mask over the columns (a mask the compiler can't prove whole
        # 16-byte groups of would make every load a scalar one); only the last tile of a layer can be partial
        if (ct + 1) * G <= nb:
            a0, a1, a2, asq, asum = _fused_e(X, P, rm, rmask, ct, k_lo, k_hi, k, sxm, sxk, spk, spc, two_nb,
                                             BM, G, BK, NPARTS, IEEE, False)
        else:
            a0, a1, a2, asq, asum = _fused_e(X, P, rm, rmask, ct, k_lo, k_hi, k, sxm, sxk, spk, spc, two_nb,
                                             BM, G, BK, NPARTS, IEEE, True)
        TS: tl.constexpr = BM * 2 * G
        eb = EP + ((tile * NS + ps) * NPARTS).to(tl.int64) * TS
        eo = r[:, None] * (2 * G) + cc[None, :]
        tl.store(eb + eo, a0, mask=rmask[:, None])
        if NPARTS == 3:
            tl.store(eb + TS + eo, a1, mask=rmask[:, None])
            tl.store(eb + 2 * TS + eo, a2, mask=rmask[:, None])
        xb = XP + (tile * NS + ps).to(tl.int64) * (2 * BM)
        tl.store(xb + r, asq, mask=rmask)
        tl.store(xb + BM + r, asum, mask=rmask)
        if RPS <= 8:
            for j in tl.static_range(RPS):
                _fused_ysums(Y, S, YS, m, nb, bs, rem, s_full, s_last, sym, syn, tile, rt, ct, ps, j, BM, G, BW, NS,
                             FULL)
        else:
            for j in range(RPS):
                _fused_ysums(Y, S, YS, m, nb, bs, rem, s_full, s_last, sym, syn, tile, rt, ct, ps, j, BM, G, BW, NS,
                             FULL)
        if NS > 1:
            # every thread's stores before the count (the barrier), made visible to the GPU by the release
            tl.debug_barrier()
            prev = tl.atomic_add(CNT + tile, 1, sem="acq_rel")
            if prev == NS - 1:
                tl.store(CNT + tile, 0)                                 # (all NS have counted: ready for the next)
                tl.debug_barrier()
                _fused_tail(X, Y, EP, XP, YS, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, nb, bs, rem,
                            sxm, sxk, sym, syn, tile, rt, ct, u, h, abs_floor, l1c, zt, lim_a, lim_big,
                            CAPM, CAPD, cst, cw, ctag, cfo, ncap, BM, G, BW, NS, SP, NPARTS, BKX, COUNTS, CAPTURE)
        else:
            tl.debug_barrier()
            _fused_tail(X, Y, EP, XP, YS, SCALE, B2N, B2M, BMAX, BMB, BSUM, STATE, slot, sc, m, k, nb, bs, rem,
                        sxm, sxk, sym, syn, tile, rt, ct, u, h, abs_floor, l1c, zt, lim_a, lim_big,
                        CAPM, CAPD, cst, cw, ctag, cfo, ncap, BM, G, BW, NS, SP, NPARTS, BKX, COUNTS, CAPTURE)


# Set by the October 22 timing run (results/serve_cost/h100_20261022_140222.json, docs/METHOD.md): v3 was cheaper
# than v1 over the four 7B layers from 256 rows on, and 64 blocks per _sums program cheaper than 32 at most row
# counts, the rule written before the run (on the 1.5B model 64 was dearer from 1,024 rows on; reported there).
V3_FROM_ROWS = 256
SUMS_P = 64           # blocks of y per _sums program (a 64 x 128 tile)
SUMS_XR = 4           # rows of x per _sums program
SCORE_Q = 256         # (row, block) pairs per _score program, about two per lane


def check_v3(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor, pair_rem: torch.Tensor,
             scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor,
             bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor, state: torch.Tensor, slot: int, z: float,
             abs_floor: float, _bm: Optional[int] = None, cap_meta: Optional[torch.Tensor] = None,
             cap_data: Optional[torch.Tensor] = None, cap_w: int = 0, cap_tag: int = 0) -> None:
    """check(), with E from cuBLAS as in v1, then _sums and _score: same arguments, same result."""
    from .checksum import BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF, l1_floor_coef
    from .serve import CAP_FLAGS, CAPTURE_SLOTS, STATE_COLS
    counts = state.dim() == 2 and state.shape[1] >= STATE_COLS
    m, n = y2.shape
    if m == 0 or n == 0:
        return
    dt = y2.dtype
    if x2.dtype != dt:
        x2 = x2.to(dt)
    k = x2.shape[1]
    nb = b2n.shape[0]
    bs = pair_full.shape[0]
    if bs > 128:
        raise ValueError(f"the Triton serving check takes blocks up to 128 wide, not {bs}")
    rem = n - (n // bs) * bs
    E, nparts = _expected_parts(x2, parts, nb, dt)
    capture = cap_meta is not None and cap_data is not None
    if capture and not (cap_meta.dtype == torch.int64 and cap_data.dtype == torch.float32 and cap_w >= bs
                        and cap_data.shape[1] >= cap_w + k and cap_meta.numel() > CAP_FLAGS + slot):
        raise ValueError("the capture buffer doesn't fit this layer")
    st = _group.pipeline_stages(y2.device, None)
    stages = {} if st is None else {"num_stages": st}
    npairs = m * nb
    sums = torch.empty(4, npairs, dtype=torch.float32, device=y2.device)
    xn = torch.empty(2, m, dtype=torch.float32, device=y2.device)
    nyt = _group._cdiv(npairs, SUMS_P)
    bw = max(16, _pow2_ceil(bs))
    bkx = max(16, min(1024, _pow2_ceil(k)))
    with _group._on(y2.device):
        _sums[(nyt + _group._cdiv(m, SUMS_XR),)](
            x2, y2, _group._sign_table(y2.device), sums, xn, m, k, nb, bs, rem, _group._sign_offset(bs),
            _group._sign_offset(rem) if rem else 0, x2.stride(0), x2.stride(1), y2.stride(0), y2.stride(1),
            npairs, nyt, P=SUMS_P, BW=bw, XR=SUMS_XR, BKX=bkx,
            FLAT=bool(rem == 0 and y2.stride(1) == 1 and y2.stride(0) == n), num_warps=4, **stages)
        _score[(_group._cdiv(npairs, SCORE_Q),)](
            x2, y2, E, sums, xn, scale, b2n, b2max, bmax, bmax_bias, bsum, state, slot, state.stride(0), m, k, nb, bs,
            rem, x2.stride(0), x2.stride(1), y2.stride(0), y2.stride(1), E.stride(0), 2 * nb, npairs,
            float(UNIT_ROUNDOFF[dt]), float(HALF_TINY_STEP[dt]), float(abs_floor), float(l1_floor_coef(k)), float(z),
            float(min(DTYPE_MAX[dt], BIG)), float(BIG),
            cap_meta if capture else state, cap_data if capture else sums,
            cap_data.stride(0) if capture else 0, int(cap_w), int(cap_tag), CAP_FLAGS, CAPTURE_SLOTS,
            Q=SCORE_Q, BW=bw, BKX=max(16, min(4096, _pow2_ceil(k))), NPARTS=nparts, COUNTS=counts, CAPTURE=capture,
            num_warps=4, **stages)


V5_G = 8              # blocks of y per v5 tile (2 x 8 = 16 columns of each part of E, the smallest dot width)
V5_SETS = 16          # streams that can run v5 on one device (each has its own set of every layer's counters)
V5_ROWS_MAX = 255     # the most rows v5's counters are reserved for at load time (warm()), at least


V5_T64_FROM_ROWS = 17   # v5's tiles are 16 rows below this many rows and 64 from here on (until a timing run sets it)


def config_v5(m: int, k: int, nb: int, device: torch.device, force_bm: Optional[int] = None):
    """(BM, BK, splits of k, k per split) for _fused. Tiles of 16 rows below V5_T64_FROM_ROWS rows, else 64. The sum
    over k is split so that one tile of rows makes about two programs per SM (each part reading its stretch of the
    parts of E at once), at most 32 ways for 16-row tiles and 8 for 64-row ones: the last program of a tile reads back
    every split's partial E, so splitting further than the bandwidth needs would cost more than it saves. Nothing
    depends on m within a tile height, so warm() can compile every variant served."""
    bm = force_bm or (16 if m < V5_T64_FROM_ROWS else 64)
    bk = 128 if bm == 16 else 64
    nct = _group._cdiv(nb, V5_G)
    steps = _group._cdiv(k, bk)
    target = max(1, _group._target_programs(device) // 2)
    ns = max(1, min(32 if bm == 16 else 8, steps, _group._cdiv(target, nct)))
    per = _group._cdiv(steps, ns)
    ns = _group._cdiv(steps, per)                                   # (no empty split)
    return bm, bk, ns, per * bk


_CNT: dict = {}           # (device, layer) -> (counters, per set): V5_SETS sets, int32, zero between launches
_CNT_SET: dict = {}       # (device, stream) -> its set
_CNT_KEEP: list = []      # every counter buffer ever made: a captured CUDA graph may still use an older one
_CNT_LOCK = threading.Lock()
V5_FALLBACKS = [0]        # v5 calls that ran v1 or v3 instead (no free set of counters, or a capture needing more)


def _capturing(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


def _stream_key(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else 0


def _grow_counters(device: torch.device, layer: int, tiles: int) -> bool:
    """At least `tiles` counters per set for this layer (its parts' address) on this device. Never allocates while a
    CUDA graph is being captured (memory made then is not initialized until the graph runs): False then. A new
    buffer is zeroed and the device waited for before any stream can use it; sets are 32 counters apart (128 bytes,
    so every set's address is aligned alike: no recompile for an odd set) and grow at least twofold."""
    key = (_group._dev_key(device), layer)
    with _CNT_LOCK:
        ent = _CNT.get(key)
        if ent is not None and ent[1] >= tiles:
            return True
        if _capturing(device):
            return False
        per = -(-max(tiles, 2 * ent[1] if ent else 0, 32) // 32) * 32
        buf = torch.zeros(V5_SETS * per, dtype=torch.int32, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _CNT_KEEP.append(buf)
        _CNT[key] = (buf, per)
        return True


def _counter_slice(device: torch.device, layer: int, tiles: int) -> Optional[torch.Tensor]:
    """This layer's counters for one v5 launch on the current stream (zero, at least `tiles` of them), or None."""
    if not _grow_counters(device, layer, tiles):
        return None
    dk = _group._dev_key(device)
    stream = _stream_key(device)
    with _CNT_LOCK:
        idx = _CNT_SET.get((dk, stream))
        if idx is None:
            idx = sum(1 for d, _s in _CNT_SET if d == dk)
            if idx >= V5_SETS:
                return None
            _CNT_SET[(dk, stream)] = idx
        buf, per = _CNT[(dk, layer)]
    return buf[idx * per:(idx + 1) * per]


def _v5_tiles_max(nb: int) -> int:
    """The most tiles a v5 launch on a layer with nb blocks has at up to max(V5_ROWS_MAX, V5_BELOW_ROWS - 1) rows,
    with the tile heights config_v5 picks."""
    rows = max(V5_ROWS_MAX, V5_BELOW_ROWS - 1)
    t16 = _group._cdiv(min(rows, V5_T64_FROM_ROWS - 1), 16)
    t64 = _group._cdiv(rows, 64) if rows >= V5_T64_FROM_ROWS else 0
    return max(1, t16, t64) * _group._cdiv(nb, V5_G)


def _reserve_counters(device: torch.device, layer: int, nb: int) -> None:
    """At load time: enough counters for v5 on this layer for every batch size it serves."""
    _grow_counters(device, layer, _v5_tiles_max(nb))


def counters_clean(device: Optional[torch.device] = None) -> bool:
    """Every v5 counter is zero (call when no v5 launch is running: after waiting for the device)."""
    dk = None if device is None else _group._dev_key(device)
    with _CNT_LOCK:
        bufs = [b for (d, _l), (b, _p) in _CNT.items() if dk is None or d == dk]
    return all(int(b.abs().sum()) == 0 for b in bufs)


def check_v5(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor, pair_rem: torch.Tensor,
             scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor,
             bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor, state: torch.Tensor, slot: int, z: float,
             abs_floor: float, _bm: Optional[int] = None, cap_meta: Optional[torch.Tensor] = None,
             cap_data: Optional[torch.Tensor] = None, cap_w: int = 0, cap_tag: int = 0) -> None:
    """check(), with v5's one kernel (_fused): same arguments, same result. _bm: rows per tile (16 or 64)."""
    from .checksum import BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF, l1_floor_coef
    from .serve import CAP_FLAGS, CAPTURE_SLOTS, STATE_COLS
    counts = state.dim() == 2 and state.shape[1] >= STATE_COLS
    m, n = y2.shape
    if m == 0 or n == 0:
        return
    dt = y2.dtype
    if x2.dtype != dt:
        x2 = x2.to(dt)
    k = x2.shape[1]
    nb = b2n.shape[0]
    bs = pair_full.shape[0]
    if bs > 128:
        raise ValueError(f"the Triton serving check takes blocks up to 128 wide, not {bs}")
    rem = n - (n // bs) * bs
    nparts = 3 if parts.shape[1] == 6 * nb else 1
    # low-precision dots on the GPU's tensor cores (fp32 accumulation, as cuBLAS); in fp32 elsewhere
    ieee = parts.dtype != dt or not x2.is_cuda
    bm, bk, ns, kc = config_v5(m, k, nb, y2.device, _bm)
    nct = _group._cdiv(nb, V5_G)
    nrt = _group._cdiv(m, bm)
    tiles = nrt * nct
    cnt = _counter_slice(y2.device, parts.data_ptr(), tiles) if ns > 1 else state
    if cnt is None:
        V5_FALLBACKS[0] += 1
        return (check_v3 if m >= V3_FROM_ROWS else check_v1)(
            x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width, state, slot, z,
            abs_floor, cap_meta=cap_meta, cap_data=cap_data, cap_w=cap_w, cap_tag=cap_tag)
    capture = cap_meta is not None and cap_data is not None
    if capture and not (cap_meta.dtype == torch.int64 and cap_data.dtype == torch.float32 and cap_w >= bs
                        and cap_data.shape[1] >= cap_w + k and cap_meta.numel() > CAP_FLAGS + slot):
        raise ValueError("the capture buffer doesn't fit this layer")
    ne = tiles * ns * nparts * bm * 2 * V5_G
    nx = tiles * ns * 2 * bm
    buf = torch.empty(ne + nx + tiles * 4 * bm * V5_G, dtype=torch.float32, device=y2.device)
    st = _group.pipeline_stages(y2.device, None)
    stages = {} if st is None else {"num_stages": st}
    bw = max(16, _pow2_ceil(bs))
    spku = 8 if parts.stride(0) % 8 == 0 else 1
    nb2u = 8 if (2 * nb) % 8 == 0 else 1
    with _group._on(y2.device):
        _fused[(ns, nct, nrt)](
            x2, y2, parts, _group._sign_table(y2.device), buf[:ne], buf[ne:ne + nx], buf[ne + nx:], cnt, scale, b2n,
            b2max, bmax, bmax_bias, bsum, state, slot, state.stride(0), m, k, nb, bs, rem, _group._sign_offset(bs),
            _group._sign_offset(rem) if rem else 0, x2.stride(0), x2.stride(1), y2.stride(0), y2.stride(1),
            parts.stride(0) // spku, parts.stride(1), 2 * nb // nb2u, nct, kc,
            float(UNIT_ROUNDOFF[dt]), float(HALF_TINY_STEP[dt]), float(abs_floor), float(l1_floor_coef(k)), float(z),
            float(min(DTYPE_MAX[dt], BIG)), float(BIG),
            cap_meta if capture else state, cap_data if capture else buf,
            cap_data.stride(0) if capture else 0, int(cap_w), int(cap_tag), CAP_FLAGS, CAPTURE_SLOTS,
            BM=bm, G=V5_G, BK=bk, BW=bw, NS=ns, SP=_pow2_ceil(ns), RPS=_group._cdiv(bm, ns), NPARTS=nparts, IEEE=ieee,
            BKX=max(16, min(4096, _pow2_ceil(k))), SPKU=spku, NB2U=nb2u,
            FULL=bool(rem == 0 and bs == bw and y2.stride(1) == 1), COUNTS=counts, CAPTURE=capture, num_warps=4,
            **stages)


def _expected_parts(x2: torch.Tensor, parts: torch.Tensor, nb: int, dt: torch.dtype):
    """E = x @ parts as v1 makes it: cuBLAS with fp32 output for the three low-precision parts on a GPU, fp32
    elsewhere; and how many parts it has."""
    if parts.shape[1] == 6 * nb:
        if parts.dtype == dt and x2.is_cuda:
            return torch.ops.aten.mm.dtype(x2, parts, torch.float32), 3
        return x2.float() @ parts.float(), 3
    return x2.float() @ parts.float(), 1


# v5 runs below this many rows in auto mode; 0 until a timing run on GPUs says where it is cheaper (docs/METHOD.md),
# and only on the GPU generations it was timed and checked on (compute capability majors): a fault in a kernel that
# wasn't run on a generation ends the process instead of falling back
V5_BELOW_ROWS = 0
V5_MAJORS = (9,)
FORCED = ("v1", "v3", "v5")


def _v5_here(device: Optional[torch.device]) -> bool:
    if device is None or device.type != "cuda":
        return True                       # (Triton's CPU interpreter in the tests)
    try:
        return _group._major(device) in V5_MAJORS
    except Exception:  # noqa: BLE001
        return False


def mode() -> str:
    """"auto" (v5 below V5_BELOW_ROWS rows, v1 below V3_FROM_ROWS, v3 from there on), or one design forced by
    SDCGUARD_SERVE_KERNEL."""
    v = os.environ.get("SDCGUARD_SERVE_KERNEL", "auto")
    return v if v in FORCED else "auto"


def kernel_for(m: int, device: Optional[torch.device] = None) -> str:
    """The design check() runs for a batch of m rows (on this device)."""
    md = mode()
    if md != "auto":
        return md
    if m < V5_BELOW_ROWS and _v5_here(device):
        return "v5"
    return "v3" if m >= V3_FROM_ROWS else "v1"


def kernels(device: Optional[torch.device] = None) -> tuple:
    """The designs check() can run in this mode (on this device): the first-use comparison and warm() cover each."""
    md = mode()
    if md == "v5":
        return ("v5", "v1", "v3")        # (v5 runs v1 or v3 when it can't have counters)
    if md != "auto":
        return (md,)
    return (("v5",) if V5_BELOW_ROWS > 0 and _v5_here(device) else ()) + ("v1", "v3")


def tile_heights(kern: str = "v1") -> tuple:
    """The rows-per-program settings the first-use comparison and warm() try for a design (v3 has none: its
    programs take pairs, whatever the rows; v5 has two tile heights)."""
    return {"v3": (None,), "v5": (16, 64)}.get(kern, (1, 2, 4, 8, 16))


_CHECKS = {"v1": "check_v1", "v3": "check_v3", "v5": "check_v5"}


def check(*args, kern: Optional[str] = None, **kwargs) -> None:
    """Check one layer on the GPU (serve.check_static_impl's arguments and result) with the design for this batch
    size (kern forces one)."""
    kern = kern or kernel_for(args[1].shape[0], args[1].device)
    return globals()[_CHECKS[kern]](*args, **kwargs)


def verified(device: torch.device, dtype: torch.dtype) -> bool:
    """Before first use on a device and dtype: same answers as the plain check on clean and corrupted
    layers (shapes with a partial last block, a bias, one row, narrow layers, many rows). Waits for
    the GPU that one time. On any difference or error, the plain check is used."""
    key = (_group._dev_key(device), dtype)
    if key not in STATUS and not _group.arch_ok(device):
        STATUS[key] = (False, "not validated on this GPU generation yet (plain checks run instead)")
        return False
    if key not in STATUS:
        from torch.utils._python_dispatch import _disable_current_modes
        try:
            # outside every dispatch mode (a guard, a fault injector) and autocast
            with _disable_current_modes(), torch.no_grad(), torch.autocast(device.type, enabled=False):
                STATUS[key] = _probe(device, dtype)
        except torch.cuda.OutOfMemoryError:
            return False
        except Exception as e:  # noqa: BLE001
            STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
        if not STATUS[key][0]:
            FALLBACKS[0] += 1
            WHY.append(f"serving first-use comparison ({key[0]}, {str(dtype).replace('torch.', '')}): {STATUS[key][1]}")
            del WHY[:-10]
    return STATUS[key][0]


def decode(key_i: int, key_d: int, *counts: int):
    """(z, flat position, noise) from a state row (its count columns, if any, are ignored here)."""
    import struct
    zb = (int(key_i) >> 32) & 0xFFFFFFFF
    z = struct.unpack("<f", struct.pack("<I", zb))[0]
    nz = struct.unpack("<f", struct.pack("<I", int(key_d) & 0xFFFFFFFF))[0]
    return z, int(key_i) & 0xFFFFFFFF, nz


def _capture_buffers(device: torch.device, k: int, slots: int = 1):
    """A capture buffer (serve.CAPTURE_SLOTS) for `slots` layers with inputs up to k long."""
    from .serve import CAP_FLAGS, CAPTURE_SLOTS
    return (torch.zeros(CAP_FLAGS + slots, dtype=torch.int64, device=device),
            torch.zeros(CAPTURE_SLOTS + 1, 128 + k, dtype=torch.float32, device=device))


def _captured(cm: torch.Tensor, cd: torch.Tensor, k: int, width_of) -> list:
    """(record, row of y, row of x) of every row a capture buffer kept (on the CPU)."""
    from .serve import CAP_FLAGS, CAP_HEAD, CAP_REC, CAPTURE_SLOTS
    head = cm[:CAP_FLAGS].cpu()
    n = min(int(head[0]), CAPTURE_SLOTS)
    recs = head[CAP_HEAD:CAP_FLAGS].view(CAPTURE_SLOTS + 1, CAP_REC)[:n].tolist()
    data = cd[:n].cpu()
    return [(r, data[i, :width_of(r[2])], data[i, 128:128 + k]) for i, r in enumerate(recs)]


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal values, NaN where the other has NaN (a NaN's payload may change when it is widened to fp32)."""
    return a.shape == b.shape and bool(((a == b) | (a.isnan() & b.isnan())).all())


def _probe(device: torch.device, dtype: torch.dtype):
    """Clean and corrupted layers: shapes with a partial last block, a bias, one row, narrow layers, and
    every number of rows per program (1 to 16; the corrupted row lands inside a multi-row tile). The rows they
    capture (serve.CAPTURE_SLOTS) must be the same, bit for bit."""
    from .serve import STATE_COLS, LayerChecksums, check_static_impl
    g = torch.Generator().manual_seed(11)
    T = _group._target_programs(device)
    shapes = [(64, 256, 300, True, None), (1, 512, 1000, False, None), (200, 96, 40, True, None),
              (7, 2048, 5632, False, None), (33, 130, 129, True, None), (300, 64, 2000, False, None),
              (16 * T + 3, 64, 1100, True, None)]
    shapes += [(64, 192, 1100, i % 2 == 0, bm) for i, bm in enumerate(tile_heights("v1"))]
    # every design in use on every shape, whatever the rows (v1's shapes with fixed rows per program only for v1);
    # v5 with each of its tile heights on every shape
    runs = [(kern, sh) for kern in kernels(device) if kern != "v5" for sh in shapes if kern == "v1" or sh[4] is None]
    if "v5" in kernels(device):
        runs += [("v5", sh[:4] + (bm,)) for bm in tile_heights("v5") for sh in shapes if sh[4] is None]
    for kern, (m, k, n, with_bias, bm) in runs:
        W = (torch.randn(n, k, generator=g, dtype=torch.float32) / k ** 0.5).to(device=device, dtype=dtype)
        b = torch.randn(n, generator=g, dtype=torch.float32).to(device=device, dtype=dtype) if with_bias else None
        x = torch.randn(m, k, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        y = torch.nn.functional.linear(x, W, b)
        cs = LayerChecksums(W, b, 128)
        for corrupted in (False, True):
            yy = y.clone()
            if corrupted:
                r, c = (m * 2) // 3, (n * 3) // 5
                yy[r, c] = yy[r, c] + 50.0 if n % 2 else float("nan")
            ref = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=device)
            fast = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=device)
            rcap, fcap = _capture_buffers(device, k), _capture_buffers(device, k)
            tag = 3 if with_bias else 2
            check_static_impl(x, yy, *cs.tensors(True), ref, 0, 6.0, 1e-20, *rcap, 128, tag)
            check(x, yy, *cs.tensors(True), fast, 0, 6.0, 1e-20, kern=kern, _bm=bm, cap_meta=fcap[0],
                  cap_data=fcap[1], cap_w=128, cap_tag=tag)
            rz, rf, _ = decode(*ref[0, :2].tolist())
            fz, ff, _ = decode(*fast[0, :2].tolist())
            width_of = (lambda blk, cs=cs: cs.bs if blk < cs.nbf else cs.rem)
            rc, fc = _captured(*rcap, k, width_of), _captured(*fcap, k, width_of)
            same = len(rc) == len(fc) == (1 if corrupted else 0) and all(
                # (tag, row, block and epoch; the scores themselves may differ in their last bits, as above)
                a[0][:3] == b[0][:3] and a[0][5] == b[0][5] and _same(a[1], b[1]) and _same(a[2], b[2])
                for a, b in zip(rc, fc))
            if not same:
                return False, (f"{kern} ({m}, {k}, {n}, rows per program {bm}) "
                               f"{'corrupted' if corrupted else 'clean'}: "
                               f"captured rows differ (fast {[c[0] for c in fc]}, plain {[c[0] for c in rc]})")
            if ref[0, 2:].tolist() != fast[0, 2:].tolist():
                return False, (f"{kern} ({m}, {k}, {n}, rows per program {bm}) "
                               f"{'corrupted' if corrupted else 'clean'}: "
                               f"fast counts {fast[0, 2:].tolist()}, plain counts {ref[0, 2:].tolist()}")
            if not corrupted:
                ok = rz <= 6.0 and fz <= 6.0 and abs(fz - rz) <= 0.05 + 0.02 * rz
            else:
                ok = rz > 6.0 and fz > 6.0 and (rf // 2) == (ff // 2)
            if not ok:
                return False, (f"{kern} ({m}, {k}, {n}, rows per program {bm}) "
                               f"{'corrupted' if corrupted else 'clean'}: "
                               f"fast z {fz} at {ff}, plain z {rz} at {rf}")
    return True, "agrees with the plain serving check"


_WARM: set = set()


def warm(parts: torch.Tensor, pair_full: torch.Tensor, pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor,
         b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
         width: torch.Tensor, k: int, n: int, dtype: torch.dtype) -> None:
    """Compile the kernels for this layer shape at load time: each design in use, and for v1 every rows-per-program
    setting config() can choose. Only the number of rows changes between batches and nothing depends on its value,
    so no batch size needs a compile while serving (vLLM's JIT monitor flags any)."""
    dev = parts.device
    key = (_group._dev_key(dev), dtype, k, n, pair_full.shape[0])
    if key in _WARM or not ready(dev, dtype):
        return
    from torch.utils._python_dispatch import _disable_current_modes
    with _disable_current_modes(), torch.no_grad(), torch.autocast(dev.type, enabled=False):
        x = torch.zeros(16, k, device=dev, dtype=dtype)
        y = torch.zeros(16, n, device=dev, dtype=dtype)
        from .serve import STATE_COLS
        st = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)    # (the layout the guard's state has)
        cm, cd = _capture_buffers(dev, k)                                  # (the guard always passes one)
        if "v5" in kernels(dev):
            _reserve_counters(dev, parts.data_ptr(), b2n.shape[0])
        for kern in kernels(dev):
            for bm in tile_heights(kern):
                check(x, y, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width, st, 0,
                      6.0, 1e-20, kern=kern, _bm=bm, cap_meta=cm, cap_data=cd, cap_w=128, cap_tag=2)
    _WARM.add(key)


def ready(device: torch.device, dtype: torch.dtype) -> bool:
    """Passed its first-use comparison already (never runs it: safe inside CUDA graph capture)."""
    if not usable(device, dtype):
        return False
    st = STATUS.get((_group._dev_key(device), dtype))
    return bool(st and st[0])
