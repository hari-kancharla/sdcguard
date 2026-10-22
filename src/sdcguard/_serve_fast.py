"""Fast serving check for NVIDIA GPUs (Triton).

The plain serving check (serve.check_static_impl) is about 40 small PyTorch operations per layer.
Under CUDA graphs each still costs a kernel, so on a 1.1B model it made decoding 2.6x slower
(H100, October 18). Two designs here do the same arithmetic per layer; SDCGUARD_SERVE_KERNEL picks one
("v1", the default, or "v3"):

  v1  cuBLAS   E = x @ (block sums of W, as three bf16/fp16 parts)     (same call as the plain check)
      _check   for its rows and blocks: |x_row|^2 and |x_row|_1, the plain and signed sums, sum of
               squares and sum of |values| of each 128-column block of y, expected = parts of E summed
               x scale + bias sums, z of both checksums with the plain check's noise model and Inf/NaN
               rules, the worst z of the program, then two atomic maxima into the layer's state row
  v3  cuBLAS   E, the same call
      _sums    the four sums of each 128-wide block of y, and the row norms of x, in one launch
      _score   the same scoring as _check, one (row, block) pair per lane (see "v3" below)

The state row holds two int64 keys: (bits of z << 32) | position of the worst checksum, and
(bits of z << 32) | bits of its noise. z >= 0, and the bits of non-negative floats sort like the
floats, so an integer maximum keeps the worst z. No second pass, no host sync, safe in CUDA graphs.

Before its first use on a device and dtype it must give the plain check's answers on clean and
corrupted layers of several shapes (verified()); otherwise the plain check is used.
"""
from __future__ import annotations

import os
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
        # expected = x @ B2: the parts of E added up in the plain check's order, x scale, + bias sums
        eo = r * es + 2 * jb
        e0 = tl.zeros((Q,), dtype=tl.float32)
        e1 = tl.zeros((Q,), dtype=tl.float32)
        for i in tl.static_range(NPARTS):
            e0 += tl.load(E + eo + i * two_nb, mask=pmask, other=0.0)
            e1 += tl.load(E + eo + i * two_nb + 1, mask=pmask, other=0.0)
        if NPARTS == 3:
            e0 = e0 * tl.load(SCALE + 2 * jb, mask=pmask, other=1.0)
            e1 = e1 * tl.load(SCALE + 2 * jb + 1, mask=pmask, other=1.0)
        exp0 = e0 + tl.load(BSUM + 2 * jb, mask=pmask, other=0.0)
        exp1 = e1 + tl.load(BSUM + 2 * jb + 1, mask=pmask, other=0.0)
        # noise model of the plain check (serve.check_static_impl, checksum.py), as in _check
        bound = asum * tl.load(BMAX + jb, mask=pmask, other=0.0) + tl.load(BMB + jb, mask=pmask, other=0.0)
        noise = tl.sqrt(sq) * u
        fl0 = xn * tl.load(B2N + 2 * jb, mask=pmask, other=0.0) * 1e-5
        fl1 = xn * tl.load(B2N + 2 * jb + 1, mask=pmask, other=0.0) * 1e-5
        hl0 = asum * tl.load(B2M + 2 * jb, mask=pmask, other=0.0) * l1c
        hl1 = asum * tl.load(B2M + 2 * jb + 1, mask=pmask, other=0.0) * l1c
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
        unk_n = tl.sum(tl.where(pmask & unk0, 1, 0) + tl.where(pmask & unk1, 1, 0), axis=0)
        z0 = tl.where(pmask, z0, -1.0)
        z1 = tl.where(pmask, z1, -1.0)
        sec = z1 > z0
        zc = tl.where(sec, z1, z0)
        dc = tl.where(sec, tl.where(a_bad1, float("nan"), den1), tl.where(a_bad0, float("nan"), den0))
        zb_e = tl.maximum(zc, 0.0).to(tl.int32, bitcast=True).to(tl.int64)
        db_e = dc.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
        flat_e = pp * 2 + tl.where(sec, 1, 0).to(tl.int64)
        key = tl.max(tl.where(pmask, (zb_e << 32) | (flat_e & 0xFFFFFFFF), 0), axis=0)
        keyd = tl.max(tl.where(pmask, (zb_e << 32) | db_e, 0), axis=0)
        # most programs can't raise the layer's worst score: read it first, and only those that can take the atomic
        if key > tl.load(STATE + slot * sc):
            tl.atomic_max(STATE + slot * sc, key)
        if keyd > tl.load(STATE + slot * sc + 1):
            tl.atomic_max(STATE + slot * sc + 1, keyd)
        if COUNTS:
            first = pid == 0
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


SUMS_P = 32           # blocks of y per _sums program (a 32 x 128 tile)
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


def _expected_parts(x2: torch.Tensor, parts: torch.Tensor, nb: int, dt: torch.dtype):
    """E = x @ parts as v1 makes it: cuBLAS with fp32 output for the three low-precision parts on a GPU, fp32
    elsewhere; and how many parts it has."""
    if parts.shape[1] == 6 * nb:
        if parts.dtype == dt and x2.is_cuda:
            return torch.ops.aten.mm.dtype(x2, parts, torch.float32), 3
        return x2.float() @ parts.float(), 3
    return x2.float() @ parts.float(), 1


def kernel() -> str:
    """Which serving kernel check() runs: "v1" (cuBLAS for E, then _check; the default) or "v3" (cuBLAS for E, then
    _sums and _score), by SDCGUARD_SERVE_KERNEL."""
    return "v3" if os.environ.get("SDCGUARD_SERVE_KERNEL", "v1") == "v3" else "v1"


def tile_heights() -> tuple:
    """The rows-per-program settings the first-use comparison and warm() try for the selected kernel (v3 has none:
    its programs take pairs, whatever the rows)."""
    return (None,) if kernel() == "v3" else (1, 2, 4, 8, 16)


def check(*args, **kwargs) -> None:
    """Check one layer on the GPU (serve.check_static_impl's arguments and result) with the selected kernel."""
    return (check_v3 if kernel() == "v3" else check_v1)(*args, **kwargs)


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
    shapes += [(64, 192, 1100, i % 2 == 0, bm) for i, bm in enumerate(tile_heights())]
    for m, k, n, with_bias, bm in shapes:
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
            check(x, yy, *cs.tensors(True), fast, 0, 6.0, 1e-20, _bm=bm, cap_meta=fcap[0], cap_data=fcap[1],
                  cap_w=128, cap_tag=tag)
            rz, rf, _ = decode(*ref[0, :2].tolist())
            fz, ff, _ = decode(*fast[0, :2].tolist())
            width_of = (lambda blk, cs=cs: cs.bs if blk < cs.nbf else cs.rem)
            rc, fc = _captured(*rcap, k, width_of), _captured(*fcap, k, width_of)
            same = len(rc) == len(fc) == (1 if corrupted else 0) and all(
                # (tag, row, block and epoch; the scores themselves may differ in their last bits, as above)
                a[0][:3] == b[0][:3] and a[0][5] == b[0][5] and _same(a[1], b[1]) and _same(a[2], b[2])
                for a, b in zip(rc, fc))
            if not same:
                return False, (f"({m}, {k}, {n}, rows per program {bm}) {'corrupted' if corrupted else 'clean'}: "
                               f"captured rows differ (fast {[c[0] for c in fc]}, plain {[c[0] for c in rc]})")
            if ref[0, 2:].tolist() != fast[0, 2:].tolist():
                return False, (f"({m}, {k}, {n}, rows per program {bm}) {'corrupted' if corrupted else 'clean'}: "
                               f"fast counts {fast[0, 2:].tolist()}, plain counts {ref[0, 2:].tolist()}")
            if not corrupted:
                ok = rz <= 6.0 and fz <= 6.0 and abs(fz - rz) <= 0.05 + 0.02 * rz
            else:
                ok = rz > 6.0 and fz > 6.0 and (rf // 2) == (ff // 2)
            if not ok:
                return False, (f"({m}, {k}, {n}, rows per program {bm}) {'corrupted' if corrupted else 'clean'}: "
                               f"fast z {fz} at {ff}, plain z {rz} at {rf}")
    return True, "agrees with the plain serving check"


_WARM: set = set()


def warm(parts: torch.Tensor, pair_full: torch.Tensor, pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor,
         b2max: torch.Tensor, bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
         width: torch.Tensor, k: int, n: int, dtype: torch.dtype) -> None:
    """Compile the kernel for this layer shape at load time, for every rows-per-program setting config()
    can choose. Only the number of rows changes between batches and nothing depends on its value, so no
    batch size needs a compile while serving (vLLM's JIT monitor flags any)."""
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
        for bm in tile_heights():
            check(x, y, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width, st, 0, 6.0,
                  1e-20, _bm=bm, cap_meta=cm, cap_data=cd, cap_w=128, cap_tag=2)
    _WARM.add(key)


def ready(device: torch.device, dtype: torch.dtype) -> bool:
    """Passed its first-use comparison already (never runs it: safe inside CUDA graph capture)."""
    if not usable(device, dtype):
        return False
    st = STATUS.get((_group._dev_key(device), dtype))
    return bool(st and st[0])
