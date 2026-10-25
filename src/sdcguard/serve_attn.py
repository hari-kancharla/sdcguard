"""Checks of attention in serving: a sampled recompute of attention outputs from the KV cache, and a check that the
sampled tokens' own KV-cache writes went where attention reads them.

At each model step one attention layer (they take turns: the layer whose index is the step number modulo the number of
checked layers) recomputes SAMPLES of its outputs, each one query head of one token: even samples a token of the step,
odd ones the newest token of a request (a decode request's only token, so long-context decodes are sampled in batches
that are mostly prefill). One number picks the token (or request) and the head together, from the layer's own turn
count, so a steady batch has every (token, head) pair checked in turn (sample_python). For token i of request r at
position p (its request's seq_len - query_len + its offset in the query, as vLLM's backends place it), head h attends
the keys at positions 0 to p of its request's KV-head (h // (heads per KV head)), read from the paged KV cache through
the request's row of the block table: block block_table[r, t // block_size], offset t % block_size. The recompute is o =
sum_t p_t v_t with p_t = exp(s_t - max s) / sum exp(s - max s), s_t = scale (q . k_t), in fp32, from the same query row
and cache the backend read, using the same metadata tensors (query_start_loc, seq_lens, block_table), so padding and
CUDA-graph replays are seen as the backend sees them.

How far a correct backend can be from it. FlashAttention (2, 3) and vLLM's Triton attention both compute scores in
fp32 from 16-bit q and k, take exp in fp32, round P to the KV dtype before the P V product (FA: to the element type;
Triton: P.to(V.dtype)), accumulate in fp32, divide by the fp32 sum of unrounded P (FA computes it from P before
rounding as well), and round the output to its dtype once. Split-KV partials are fp32 (FA's out_accum, Triton's
segment buffers). So, with u the dtype's unit roundoff, A_d = sum_t p_t |v_td| and L the number of keys, a correct
output is within
  2u (A_d + |o_d|)                  rounding P (u A_d, plus u |o_d| if the sum were taken from rounded P), the output
                                    (u |o_d|), and partial outputs if a backend rounded them to the dtype (u A_d; none
                                    of vLLM 0.32's does, and it has no cascade attention: slack)
  + E_d                             score errors: |dp_t| <= p_t e_t moves o_d by at most sum_t p_t e_t (|v_td| + |o_d|),
                                    e_t = (D + 8) 2^-23 scale sum_i |q_i k_ti| (fp32 accumulation of q.k in any order,
                                    with truncation, on either side) + 2^-22 (|s_t| + |max s|) (scaling and the exp
                                    argument in fp32) + 2^-21 (fast exp on either side)
  + S_d                             P values under the dtype's smallest normal: absolute error up to half its subnormal
                                    spacing (2^-25 for fp16; 2^-126 for bf16, fp32's flush), times sum_t |v_td| / sum p
  + ((L / 16 + 32) 2^-22) (A_d + |o_d|)   fp32 accumulation of P V and of the sum (on either side; tensor cores add
                                    16 products at a time and truncate), the split-KV combine and the division
  + the dtype's subnormal spacing
of the recompute (first order; everything an fp32 rounding contributes is a few millionths of the bf16 terms). The
score of an output is its distance from the recompute divided by that bound: at most 1 for a correct output. The
line is 4 (SDCGUARD_ATTN_Z, at least 1.5). For bf16 that bound is about 1.6% of the size of the attended values for
ordinary logits (more where scale * sum |q k| is large: about 2.2% at 200), so an output off by more than about 6% of
them (9% at such logits) is caught for sure.

What a recompute catches: a wrong output from the attention kernel's arithmetic (a faulty SM, a bad split-KV or
softmax combine) and an output computed from other keys than the metadata names (wrong lengths or a stale schedule
inside the kernel, the wrong request's blocks). Not seen: metadata that is itself wrong (wrong seq_lens or block
table: the check reads the same), and a KV cache whose contents changed (the kernel and the recompute read the same
corrupted values). For the second, the sampled token's own write is checked too: the slot vLLM wrote its key and value
to (slot_mapping) must be the slot the block table gives its position, and that slot must hold exactly the key and
value the layer computed for it (bit for bit, every KV head). The rest of the cache (earlier tokens, prefix-cache
hits) isn't checked here.

Supported: vLLM's FlashAttention (versions 2 and 3) and Triton backends, causal decoder self-attention with a 16-bit KV
cache in the model's dtype. Listed as not checked, per layer or per call: FlashAttention 4 (Blackwell), ALiBi, attention
sinks, logit soft caps, sliding windows (the recompute follows FlashAttention's window, but no GPU run has checked it
yet), chunked local attention, multimodal prefix ranges, R-SWA, decode context parallelism, fused output quantization,
and MLA, FlashInfer and the other backends.

What a read reports, per layer: the worst score with its token row, head, dimension, output, expected value, position
and step (int64 keys with the score's bits on top, one atomic maximum each), the calls and samples checked, samples
with no token (padding) or not judged (a non-finite expected value), samples whose block table pointed outside the
cache, the writes checked and those wrong. The first sample over the line in a layer is kept (its query and output
rows and its place), and, once per read across layers, the keys and values it attended (exact, up to CAP_ROWS of
them): its alert rechecks it in float64 on the CPU.
"""
from __future__ import annotations

import os
import struct
from typing import Optional

import torch

from .events import SDCEvent, device_identity
from .serve_norm import PLACE_KEPT, PLACE_MAXIMA, S31, dtype_consts, pick, unwrap31

ATTN_Z = 4.0
MIN_LINE = 1.5           # a correct output scores at most 1 to first order
SAMPLES = 4              # (token, head) pairs recomputed per checked call
NSPLIT = 32              # programs per sample: the keys are split among them
BLOCK_N = 32             # keys per tile
CAP_ROWS = 65536         # at most this many kept keys and values (with their values: CAP_ROWS * 2 * D * 2 bytes)
LOG_R = 17               # binary-search steps over query_start_loc: up to 65,535 requests a step (a constant, so the
#                          kernels compile once, not once per batch size)
MAX_REQUESTS = (1 << (LOG_R - 1)) - 1
TORCH_MAX_KEYS = 8192    # the PyTorch check gathers a whole block-table row: on GPUs, only for rows of at most this
M32 = 0xFFFFFFFF
PICK_A, PICK_B, PICK_C = 1000003, 7919, 104729   # sample j at a layer's k-th turn: (k A + j B + index C) % (n heads)
TWO22, TWO21, TWO23 = 2.0 ** -22, 2.0 ** -21, 2.0 ** -23
COLS = 18
(C_ZT, C_ZH, C_ZD, C_ZO, C_ZR, C_ZS, C_ZP, C_CALLS, C_SAMPLES, C_SKIP, C_UNJ, C_BADBLK, C_KEYS, C_WN, C_WBAD,
 C_WSLOT, C_WLAST, C_WWHERE) = range(COLS)
CAPM = 12                # a kept sample: claimed, row, head, request, position, seq_len, first key, step, score bits,
CM_CLAIM, CM_ROW, CM_HEAD, CM_REQ, CM_POS, CM_SL, CM_LO, CM_STEP, CM_Z, CM_KEPT, CM_DIM, CM_HK = range(CAPM)
WS_HEAD = 4              # partials: max, sum p, sum p e, out-of-range blocks, then acc, A, acc_e, sum |v| (D each)
W_CONTENT, W_SLOT = 1, 2

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}        # (device key, dtype) -> (ok, why): the kernels' first-use comparison
FALLBACKS = [0]


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def p_spacing(dtype: torch.dtype) -> float:
    """Half the subnormal spacing of P's dtype, or fp32's smallest normal where that is larger (FlashAttention's
    exp2 flushes fp32 subnormals): the absolute error of a P value too small for the dtype's normal range."""
    return 2.0 ** -25 if dtype == torch.float16 else 2.0 ** -126


def bound_consts(dtype: torch.dtype, head_size: int) -> tuple:
    """(u, output spacing, P spacing, q.k accumulation coefficient) for outputs in dtype."""
    u, sp, _ = dtype_consts(dtype)
    return u, sp, p_spacing(dtype), (head_size + 8) * TWO23


def _log2_ceil(n: int) -> int:
    return max(1, int(n).bit_length() + 1)


def _p2(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


# --------------------------------------------------------------------------------------------- which samples
def sample_python(step: int, idx: int, nl: int, j: int, qsl: list, seq_lens: list, rows: int,
                  heads: int) -> Optional[dict]:
    """Sample j of the layer with index idx of nl at a step (the kernels and the PyTorch check take the same): its
    token row, request, position, seq_len and query head, or None for no token. One number picks the token (or the
    request) and the head together, from the layer's own turn count (step // nl): over consecutive turns of a steady
    batch it runs through every (token, head) pair (A is a prime, so prime to their number but for huge batches)."""
    nreq = len(seq_lens)
    ntok = min(int(qsl[nreq]), rows)
    h = (step // nl) * PICK_A + j * PICK_B + idx * PICK_C
    if ntok <= 0:
        return None
    if j % 2 == 0:
        pick_ = h % (ntok * heads)
        i = pick_ // heads
    else:
        rreal = next(k for k in range(nreq + 1) if qsl[k] >= ntok)
        if rreal <= 0:
            return None
        pick_ = h % (rreal * heads)
        i = int(qsl[pick_ // heads + 1]) - 1
    r = next(k for k in range(nreq + 1) if qsl[k] >= i + 1) - 1
    q0, q1, sl = int(qsl[r]), int(qsl[r + 1]), int(seq_lens[r])
    qlen = q1 - q0
    pos = sl - qlen + (i - q0)
    if not (0 <= i < rows and qlen > 0 and sl > 0 and 0 <= pos < sl):
        return None
    hq = pick_ % heads
    return {"row": i, "request": r, "position": pos, "seq_len": sl, "head": hq}


# ---------------------------------------------------------------------------------------------- PyTorch check
def _bits(v: torch.Tensor) -> torch.Tensor:
    return v.reshape(1).float().view(torch.int32).to(torch.int64).reshape(()) & M32


def attn_check_torch(q, o, kv, key, val, qsl, seq_lens, bt, slots, state, slot, ctrl, nl, idx, scale, left, line,
                     capm, capq, capo, has_write) -> None:
    """The check in PyTorch operations, without a host sync (static shapes: the keys of a whole block-table row are
    gathered and masked, so it is for small caches, the first-use comparison and CPU tests; the Triton kernels are
    what runs on GPUs). q: [T, Hq, D], o: [T, Hq, D], kv: [blocks, Hkv, block_size, 2D], key/val: [T, Hkv, D]."""
    dev = q.device
    T, Hq, D = q.shape
    NB, Hkv, N = kv.shape[0], kv.shape[1], kv.shape[2]
    R, W = seq_lens.shape[0], bt.shape[1]
    u, sig_o, hsig_p, g_qk = bound_consts(o.dtype, D)
    gq = Hq // Hkv
    step = ctrl[0] & M32
    live = (ctrl[1] != 0) & ((step % nl) == idx)
    srow = state[slot]
    srow[C_CALLS] += live.to(torch.int64)
    qsl64 = qsl.long()
    ntok = torch.clamp(qsl64[R], max=T)
    t = torch.arange(W * N, device=dev)
    for j in range(SAMPLES):
        h = torch.div(step, nl, rounding_mode="floor") * PICK_A + j * PICK_B + idx * PICK_C
        if j % 2 == 0:
            pk = h % (ntok * Hq).clamp(min=1)
            i = torch.div(pk, Hq, rounding_mode="floor")
            have = ntok > 0
        else:
            rreal = torch.searchsorted(qsl64, ntok.reshape(1)).reshape(())
            pk = h % (rreal * Hq).clamp(min=1)
            i = pick(qsl64, (torch.div(pk, Hq, rounding_mode="floor") + 1).clamp(max=R)) - 1
            have = (ntok > 0) & (rreal > 0)
        i = i.clamp(min=0, max=T - 1)
        r = (torch.searchsorted(qsl64, (i + 1).reshape(1)).reshape(()) - 1).clamp(min=0, max=R - 1)
        q0, q1 = pick(qsl64, r), pick(qsl64, r + 1)
        sl = pick(seq_lens, r).long()
        qlen = q1 - q0
        pos = sl - qlen + (i - q0)
        valid = live & have & (qlen > 0) & (sl > 0) & (pos >= 0) & (pos < sl) & (sl <= W * N)
        hq = pk % Hq
        hk = hq // gq
        lo = torch.clamp(pos - left, min=0) if left >= 0 else torch.zeros_like(pos)
        nkeys = pos - lo + 1
        keep = (t >= lo) & (t <= pos)
        blk = pick(bt, r).long()[torch.div(t, N, rounding_mode="floor").clamp(max=W - 1)]
        okb = (blk >= 0) & (blk < NB)
        bad = (keep & ~okb).any()
        blk = torch.where(okb, blk, torch.zeros_like(blk))
        rows = kv[blk, hk.expand_as(blk), t % N].float()                       # [W*N, 2D]
        rows = torch.where(keep[:, None], rows, torch.zeros_like(rows))        # (other rows may hold anything)
        K, V = rows[:, :D], rows[:, D:2 * D]
        qv = pick(pick(q, i), hq).float()
        s = (K @ qv) * scale
        s = torch.where(keep, s, torch.full_like(s, float("-inf")))
        m = s.max()
        p = torch.where(keep, torch.exp(s - m), torch.zeros_like(s))
        e = g_qk * scale * (K.abs() @ qv.abs()) + TWO22 * s.abs() + TWO21
        pe = torch.where(keep, p * e, torch.zeros_like(p))
        l = p.sum()
        Va = V.abs()
        ref = (p @ V) / l
        A = (p @ Va) / l
        E = (pe @ Va + ref.abs() * pe.sum()) / l
        Sm = hsig_p * torch.where(keep[:, None], Va, torch.zeros_like(Va)).sum(0) / l
        gam = (nkeys.float() / 16 + 32) * TWO22 + TWO22 * m.abs()
        bound = (2 * u + gam) * (A + ref.abs()) + E + Sm + sig_o
        ov = pick(pick(o, i), hq).float()
        z = (ov - ref).abs() / bound
        z = torch.nan_to_num(z, nan=float("inf"), posinf=float("inf"))
        finite = torch.isfinite(ref).all() & torch.isfinite(bound).all() & torch.isfinite(l) & (l > 0)
        judged = valid & ~bad & finite
        zmax, dz = z.max(0)
        zmax = torch.where(judged, zmax, torch.zeros_like(zmax))
        zb = _bits(zmax) << 32
        keys = torch.stack([zb | i, zb | hq, zb | dz, zb | _bits(pick(ov, dz)), zb | _bits(pick(ref, dz)), zb | step,
                            zb | pos])
        srow[:7] = torch.maximum(srow[:7], torch.where(judged, keys, torch.zeros_like(keys)))
        srow[C_SAMPLES] += judged.to(torch.int64)
        srow[C_SKIP] += (live & ~valid).to(torch.int64)
        srow[C_BADBLK] += (valid & bad).to(torch.int64)
        srow[C_UNJ] += (valid & ~bad & ~finite).to(torch.int64)
        srow[C_KEYS] += torch.where(judged, nkeys, torch.zeros_like(nkeys))
        take = judged & (zmax > line) & (capm[slot, CM_CLAIM] == 0)
        rec = torch.stack([torch.ones_like(i), i, hq, r, pos, sl, lo, step, zb >> 32, torch.zeros_like(i), dz,
                           hk]).to(capm.dtype)
        capm[slot] = torch.where(take, rec, capm[slot])
        capq[slot, :D] = torch.where(take, qv, capq[slot, :D])
        capo[slot, :D] = torch.where(take, ov, capo[slot, :D])
        if has_write:
            sn = slots.shape[0]
            ic = i.clamp(max=sn - 1)
            s_i = pick(slots, ic).long()
            want = pick(pick(bt, r).long(), torch.div(pos, N, rounding_mode="floor").clamp(min=0, max=W - 1)) * N + \
                pos % N
            okw = valid & (i < sn) & (s_i >= 0) & (s_i < NB * N)
            sc = torch.where(okw, s_i, torch.zeros_like(s_i))
            cache = pick(kv, torch.div(sc, N, rounding_mode="floor")).index_select(1, (sc % N).reshape(1)).squeeze(1)
            ck, cv = cache[:, :D], cache[:, D:2 * D]
            nk, nv = pick(key, ic), pick(val, ic)
            mk = ck.contiguous().view(torch.int16) != nk.contiguous().view(torch.int16)
            mv = cv.contiguous().view(torch.int16) != nv.contiguous().view(torch.int16)
            badc = okw & (mk.any() | mv.any())
            bads = okw & (s_i != want)
            flat = torch.arange(Hkv * D, device=dev).reshape(Hkv, D)
            big = torch.full_like(flat, Hkv * D)
            fk = torch.where(mk, flat, big).min()
            fv = torch.where(mv, flat, big).min()
            isv = fk >= Hkv * D
            first = torch.where(isv, fv, fk)
            where_ = (torch.div(first, D, rounding_mode="floor") << 12) | ((first % D) << 1) | isv.to(torch.int64)
            kind = torch.where(badc, W_CONTENT, 0) | torch.where(bads, W_SLOT, 0)
            badw = badc | bads
            srow[C_WN] += okw.to(torch.int64)
            srow[C_WBAD] += badc.to(torch.int64)
            srow[C_WSLOT] += bads.to(torch.int64)
            s31 = (step & S31) << 32                    # (31 bits: the key stays positive for the maximum)
            srow[C_WLAST] = torch.where(badw, torch.maximum(srow[C_WLAST], s31 | i), srow[C_WLAST])
            wv = s31 | (kind << 28) | torch.where(badc, where_, torch.zeros_like(where_))
            srow[C_WWHERE] = torch.where(badw, torch.maximum(srow[C_WWHERE], wv), srow[C_WWHERE])


# ----------------------------------------------------------------------------------------------- Triton check
if HAVE_TRITON:
    @triton.jit
    def _attn_live(ctrl, nl, idx):
        step = tl.load(ctrl) & 0xFFFFFFFF
        active = tl.load(ctrl + 1)
        return step, (active != 0) & ((step % nl) == idx)

    @triton.jit
    def _lower_bound(arr, n, x, LOG: tl.constexpr):
        """The smallest k in [0, n] with arr[k] >= x (arr non-decreasing, n + 1 entries; n when none is)."""
        lo = tl.full((), 0, tl.int64)
        hi = tl.full((), 0, tl.int64) + n
        for _ in tl.static_range(LOG):
            act = lo < hi
            mid = (lo + hi) // 2
            v = tl.load(arr + mid).to(tl.int64)
            big = v >= x
            hi = tl.where(act & big, mid, hi)
            lo = tl.where(act & (v < x), mid + 1, lo)
        return lo

    @triton.jit
    def _attn_sample(qsl, seqlens, step, nl, idx, j, T, R, W, N, Hq, LOG_R: tl.constexpr):
        ntok = tl.minimum(tl.load(qsl + R).to(tl.int64), T)
        h = (step // nl) * 1000003 + j * 7919 + idx * 104729
        p_tok = h % tl.maximum(ntok * Hq, 1)
        rreal = _lower_bound(qsl, R, ntok, LOG_R)
        p_req = h % tl.maximum(rreal * Hq, 1)
        rq = tl.minimum(p_req // Hq, R - 1)
        i_req = tl.load(qsl + rq + 1).to(tl.int64) - 1
        odd = (j % 2) == 1
        i = tl.where(odd, i_req, p_tok // Hq)
        pk = tl.where(odd, p_req, p_tok)
        have = (ntok > 0) & ((rreal > 0) | (j % 2 == 0))
        i = tl.minimum(tl.maximum(i, 0), T - 1)
        r = tl.minimum(tl.maximum(_lower_bound(qsl, R, i + 1, LOG_R) - 1, 0), R - 1)
        q0 = tl.load(qsl + r).to(tl.int64)
        q1 = tl.load(qsl + r + 1).to(tl.int64)
        sl = tl.load(seqlens + r).to(tl.int64)
        qlen = q1 - q0
        pos = sl - qlen + (i - q0)
        valid = have & (qlen > 0) & (sl > 0) & (pos >= 0) & (pos < sl) & (sl <= W * N)
        hq = pk % Hq
        return valid, i, r, pos, sl, hq

    # every integer and pointer argument unspecialized: the kernels compile once for a model (its dtypes and the
    # constexprs), not again for each batch size or buffer address, so finalize() can compile them before vLLM
    # captures any CUDA graph (its V1 runner captures without a warm-up run)
    @triton.jit(do_not_specialize=["T", "R", "W", "NB", "N", "Hq", "GQ", "D", "sqt", "sqh", "skb", "skh", "skn",
                                   "skd", "sbt", "nl", "idx", "left"],
                do_not_specialize_on_alignment=["q", "kv", "qsl", "seqlens", "bt", "ctrl", "ws"])
    def _attn_part(q, kv, qsl, seqlens, bt, ctrl, ws,
                   T, R, W, NB, N, Hq, GQ, D,
                   sqt, sqh, skb, skh, skn, skd, sbt,
                   nl, idx, scale, left, g_qk,
                   LOG_R: tl.constexpr, NSPLIT_: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                   WSW: tl.constexpr):
        j = tl.program_id(0)
        c = tl.program_id(1)
        step, live = _attn_live(ctrl, nl, idx)
        if live:
            valid, i, r, pos, sl, hq = _attn_sample(qsl, seqlens, step, nl, idx, j, T, R, W, N, Hq, LOG_R)
            if valid:
                hk = hq // GQ
                lo = tl.where(left >= 0, tl.maximum(pos - left, 0), 0)
                nkeys = pos - lo + 1
                chunk = (nkeys + NSPLIT_ - 1) // NSPLIT_
                t0 = (lo + c * chunk).to(tl.int32)
                t1 = tl.minimum(lo + (c + 1) * chunk, pos + 1).to(tl.int32)
                d = tl.arange(0, BD)
                dm = d < D
                qv = tl.load(q + i * sqt + hq * sqh + d, mask=dm, other=0.0).to(tl.float32)
                qa = tl.abs(qv)
                m = tl.full((), float("-inf"), tl.float32)
                bad = tl.full((), 0, tl.int32)
                for tb in range(t0, t1, BN):
                    tt = tb + tl.arange(0, BN)
                    tm = tt < t1
                    blk = tl.load(bt + r * sbt + tt // N, mask=tm, other=0).to(tl.int64)
                    okb = (blk >= 0) & (blk < NB)
                    bad += tl.sum(tl.where(tm & (blk < 0), 1, 0) + tl.where(tm & (blk >= NB), 1, 0), 0)
                    blk = tl.where(okb, blk, 0)
                    base = blk * skb + hk * skh + (tt % N).to(tl.int64) * skn
                    mk = tm[:, None] & dm[None, :]
                    k = tl.load(kv + base[:, None] + d[None, :] * skd, mask=mk, other=0.0).to(tl.float32)
                    s = tl.sum(k * qv[None, :], 1) * scale
                    s = tl.where(tm, s, float("-inf"))
                    m = tl.maximum(m, tl.max(s, 0))
                lsum = tl.full((), 0.0, tl.float32)
                le = tl.full((), 0.0, tl.float32)
                acc = tl.zeros([BD], dtype=tl.float32)
                aa = tl.zeros([BD], dtype=tl.float32)
                ae = tl.zeros([BD], dtype=tl.float32)
                v1 = tl.zeros([BD], dtype=tl.float32)
                for tb in range(t0, t1, BN):
                    tt = tb + tl.arange(0, BN)
                    tm = tt < t1
                    blk = tl.load(bt + r * sbt + tt // N, mask=tm, other=0).to(tl.int64)
                    blk = tl.where((blk >= 0) & (blk < NB), blk, 0)
                    base = blk * skb + hk * skh + (tt % N).to(tl.int64) * skn
                    mk = tm[:, None] & dm[None, :]
                    k = tl.load(kv + base[:, None] + d[None, :] * skd, mask=mk, other=0.0).to(tl.float32)
                    v = tl.load(kv + base[:, None] + (D + d[None, :]) * skd, mask=mk, other=0.0).to(tl.float32)
                    s = tl.sum(k * qv[None, :], 1) * scale
                    p = tl.where(tm, tl.exp(s - m), 0.0)
                    e = g_qk * scale * tl.sum(tl.abs(k) * qa[None, :], 1) + 2.384185791015625e-07 * tl.abs(s) + \
                        4.76837158203125e-07
                    pe = tl.where(tm, p * e, 0.0)
                    va = tl.abs(v)
                    lsum += tl.sum(p, 0)
                    le += tl.sum(pe, 0)
                    acc += tl.sum(p[:, None] * v, 0)
                    aa += tl.sum(p[:, None] * va, 0)
                    ae += tl.sum(pe[:, None] * va, 0)
                    v1 += tl.sum(tl.where(mk, va, 0.0), 0)
                wb = ws + (j * NSPLIT_ + c) * WSW
                tl.store(wb, m)
                tl.store(wb + 1, lsum)
                tl.store(wb + 2, le)
                tl.store(wb + 3, bad.to(tl.float32))
                tl.store(wb + 4 + d, acc, mask=dm)
                tl.store(wb + 4 + BD + d, aa, mask=dm)
                tl.store(wb + 4 + 2 * BD + d, ae, mask=dm)
                tl.store(wb + 4 + 3 * BD + d, v1, mask=dm)

    @triton.jit(do_not_specialize=["T", "R", "W", "NB", "N", "Hq", "GQ", "Hkv", "D", "SN", "sqt", "sqh", "sot", "soh",
                                   "skb", "skh", "skn", "skd", "sbt", "skt", "skh2", "svt", "svh", "slot", "nl", "idx",
                                   "left", "caprows", "capw"],
                do_not_specialize_on_alignment=["q", "o", "kv", "key", "val", "qsl", "seqlens", "bt", "slots", "ctrl",
                                                "ws", "state", "capm", "capq", "capo", "capkv", "capkvm"])
    def _attn_judge(q, o, kv, key, val, qsl, seqlens, bt, slots, ctrl, ws, state, capm, capq, capo, capkv, capkvm,
                    T, R, W, NB, N, Hq, GQ, Hkv, D, SN,
                    sqt, sqh, sot, soh, skb, skh, skn, skd, sbt, skt, skh2, svt, svh,
                    slot, nl, idx, left, u, sig_o, hsig_p, line, caprows, capw,
                    LOG_R: tl.constexpr, NSPLIT_: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
                    BH: tl.constexpr, WSW: tl.constexpr, NCOLS: tl.constexpr, NCAPM: tl.constexpr,
                    HAS_WRITE: tl.constexpr, KEEP_KV: tl.constexpr):
        j = tl.program_id(0)
        step, live = _attn_live(ctrl, nl, idx)
        if live:
            sb = state + slot * NCOLS
            if j == 0:
                tl.atomic_add(sb + 7, 1)
            valid, i, r, pos, sl, hq = _attn_sample(qsl, seqlens, step, nl, idx, j, T, R, W, N, Hq, LOG_R)
            if valid:
                hk = hq // GQ
                lo = tl.where(left >= 0, tl.maximum(pos - left, 0), 0)
                nkeys = pos - lo + 1
                d = tl.arange(0, BD)
                dm = d < D
                wbase = ws + j * NSPLIT_ * WSW
                mx = tl.full((), float("-inf"), tl.float32)
                for c in range(0, NSPLIT_):
                    mx = tl.maximum(mx, tl.load(wbase + c * WSW))
                lsum = tl.full((), 0.0, tl.float32)
                le = tl.full((), 0.0, tl.float32)
                bad = tl.full((), 0.0, tl.float32)
                acc = tl.zeros([BD], dtype=tl.float32)
                aa = tl.zeros([BD], dtype=tl.float32)
                ae = tl.zeros([BD], dtype=tl.float32)
                v1 = tl.zeros([BD], dtype=tl.float32)
                for c in range(0, NSPLIT_):
                    wb = wbase + c * WSW
                    mc = tl.load(wb)
                    w = tl.where(mc > float("-inf"), tl.exp(mc - mx), 0.0)
                    lsum += w * tl.load(wb + 1)
                    le += w * tl.load(wb + 2)
                    bad += tl.load(wb + 3)
                    acc += w * tl.load(wb + 4 + d, mask=dm, other=0.0)
                    aa += w * tl.load(wb + 4 + BD + d, mask=dm, other=0.0)
                    ae += w * tl.load(wb + 4 + 2 * BD + d, mask=dm, other=0.0)
                    v1 += tl.load(wb + 4 + 3 * BD + d, mask=dm, other=0.0)
                ref = acc / lsum
                ra = tl.abs(ref)
                gam = (nkeys.to(tl.float32) / 16.0 + 32.0) * 2.384185791015625e-07 + \
                    2.384185791015625e-07 * tl.abs(mx)
                bound = (2.0 * u + gam) * (aa / lsum + ra) + (ae + ra * le) / lsum + hsig_p * v1 / lsum + sig_o
                ov = tl.load(o + i * sot + hq * soh + d, mask=dm, other=0.0).to(tl.float32)
                z = tl.abs(ov - ref) / bound
                z = tl.where(z == z, z, float("inf"))
                z = tl.where(dm, z, 0.0)
                nonfin = tl.sum(tl.where(dm & (ra != ra), 1, 0) + tl.where(dm & (ra == float("inf")), 1, 0) +
                                tl.where(dm & (bound != bound), 1, 0) +
                                tl.where(dm & (bound == float("inf")), 1, 0), 0)
                notfin = (nonfin > 0) | (lsum != lsum) | (lsum <= 0.0) | (lsum == float("inf"))
                if bad > 0.0:
                    tl.atomic_add(sb + 11, 1)
                elif notfin:
                    tl.atomic_add(sb + 10, 1)
                else:
                    tl.atomic_add(sb + 8, 1)
                    tl.atomic_add(sb + 12, nkeys)
                    pz = tl.max(z, 0)
                    pd = tl.argmax(z, 0)
                    sel = d == pd
                    py = tl.sum(tl.where(sel, ov, 0.0), 0)
                    pr = tl.sum(tl.where(sel, ref, 0.0), 0)
                    zbits = pz.to(tl.int32, bitcast=True)
                    zb = zbits.to(tl.int64) << 32
                    if pz > 0.0:
                        k0 = zb | i
                        cur = tl.load(sb)
                        if zb >= (cur & -4294967296):      # (ties too: the same columns in any program order)
                            tl.atomic_max(sb + 0, k0)
                            tl.atomic_max(sb + 1, zb | hq)
                            tl.atomic_max(sb + 2, zb | pd.to(tl.int64))
                            tl.atomic_max(sb + 3, zb | (py.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                            tl.atomic_max(sb + 4, zb | (pr.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                            tl.atomic_max(sb + 5, zb | step)
                            tl.atomic_max(sb + 6, zb | pos)
                    if pz > line:
                        cb = capm + slot * NCAPM
                        zero = tl.full((), 0, tl.int32)
                        if tl.atomic_cas(cb, zero, zero + 1) == 0:
                            tl.store(cb + 1, i.to(tl.int32))
                            tl.store(cb + 2, hq.to(tl.int32))
                            tl.store(cb + 3, r.to(tl.int32))
                            tl.store(cb + 4, pos.to(tl.int32))
                            tl.store(cb + 5, sl.to(tl.int32))
                            tl.store(cb + 6, lo.to(tl.int32))
                            tl.store(cb + 7, step.to(tl.int32))
                            tl.store(cb + 8, zbits)
                            tl.store(cb + 10, pd.to(tl.int32))
                            tl.store(cb + 11, hk.to(tl.int32))
                            qv = tl.load(q + i * sqt + hq * sqh + d, mask=dm, other=0.0).to(tl.float32)
                            tl.store(capq + slot * capw + d, qv, mask=dm)
                            tl.store(capo + slot * capw + d, ov, mask=dm)
                            if KEEP_KV:
                                if nkeys <= caprows:
                                    if tl.atomic_cas(capkvm, zero, zero + 1) == 0:
                                        for tb in range(lo.to(tl.int32), (pos + 1).to(tl.int32), BN):
                                            tt = tb + tl.arange(0, BN)
                                            tm = tt <= pos
                                            blk = tl.load(bt + r * sbt + tt // N, mask=tm, other=0).to(tl.int64)
                                            blk = tl.where((blk >= 0) & (blk < NB), blk, 0)
                                            base = blk * skb + hk * skh + (tt % N).to(tl.int64) * skn
                                            mk = tm[:, None] & dm[None, :]
                                            kr = tl.load(kv + base[:, None] + d[None, :] * skd, mask=mk)
                                            vr = tl.load(kv + base[:, None] + (D + d[None, :]) * skd, mask=mk)
                                            dst = capkv + ((tt - lo).to(tl.int64)[:, None] * 2) * capw + d[None, :]
                                            tl.store(dst, kr, mask=mk)
                                            tl.store(dst + capw, vr, mask=mk)
                                        tl.store(capkvm + 1, zero + slot)
                                        tl.store(capkvm + 2, nkeys.to(tl.int32))
                                        tl.store(capkvm + 3, step.to(tl.int32))
                                        tl.store(cb + 9, zero + 1)
                if HAS_WRITE:
                    if i < SN:
                        s_i = tl.load(slots + i).to(tl.int64)
                        if (s_i >= 0) & (s_i < NB * N):
                            pb = tl.load(bt + r * sbt + pos // N).to(tl.int64)
                            want = pb * N + pos % N
                            cblk = s_i // N
                            coff = s_i % N
                            hh = tl.arange(0, BH)
                            hm = hh < Hkv
                            mk2 = hm[:, None] & dm[None, :]
                            cbase = cblk * skb + hh[:, None].to(tl.int64) * skh + coff * skn
                            ck = tl.load(kv + cbase + d[None, :] * skd, mask=mk2, other=0.0)
                            cvv = tl.load(kv + cbase + (D + d[None, :]) * skd, mask=mk2, other=0.0)
                            nk = tl.load(key + i * skt + hh[:, None] * skh2 + d[None, :], mask=mk2, other=0.0)
                            nv = tl.load(val + i * svt + hh[:, None] * svh + d[None, :], mask=mk2, other=0.0)
                            bk = mk2 & (ck.to(tl.int16, bitcast=True) != nk.to(tl.int16, bitcast=True))
                            bv = mk2 & (cvv.to(tl.int16, bitcast=True) != nv.to(tl.int16, bitcast=True))
                            nbk = tl.sum(tl.sum(tl.where(bk, 1, 0), 1), 0)
                            nbv = tl.sum(tl.sum(tl.where(bv, 1, 0), 1), 0)
                            tl.atomic_add(sb + 13, 1)
                            badc = (nbk + nbv) > 0
                            bads = s_i != want
                            if badc | bads:
                                flat = hh[:, None] * D + d[None, :]
                                big = Hkv * D
                                fk = tl.min(tl.min(tl.where(bk, flat, big), 1), 0)
                                fv = tl.min(tl.min(tl.where(bv, flat, big), 1), 0)
                                isv = fk >= big
                                first = tl.where(isv, fv, fk).to(tl.int64)
                                wh = ((first // D) << 12) | ((first % D) << 1) | isv.to(tl.int64)
                                kind = tl.where(badc, 1, 0) | tl.where(bads, 2, 0)
                                wh = tl.where(badc, wh, 0)
                                if badc:
                                    tl.atomic_add(sb + 14, 1)
                                if bads:
                                    tl.atomic_add(sb + 15, 1)
                                s31 = (step & 0x7FFFFFFF) << 32
                                tl.atomic_max(sb + 16, s31 | i)
                                tl.atomic_max(sb + 17, s31 | (kind.to(tl.int64) << 28) | wh)
            else:
                tl.atomic_add(sb + 9, 1)


def attn_check_triton(q, o, kv, key, val, qsl, seq_lens, bt, slots, state, slot, ctrl, nl, idx, scale, left, line,
                      capm, capq, capo, capkv, capkvm, ws, has_write, keep_kv) -> None:
    T, Hq, D = q.shape
    NB, Hkv, N = kv.shape[0], kv.shape[1], kv.shape[2]
    R, W = seq_lens.shape[0], bt.shape[1]
    u, sig_o, hsig_p, g_qk = bound_consts(o.dtype, D)
    BD = _p2(D)
    wsw = ws.shape[-1]
    if 4 + 4 * BD > wsw or BD > capq.shape[1]:
        raise ValueError(f"head size {D} is larger than the buffers allocated for this device")
    if R > MAX_REQUESTS:
        raise ValueError(f"{R} requests in a step: the kernels take at most {MAX_REQUESTS}")
    common = (T, R, W, NB, N, Hq, Hq // Hkv)
    _attn_part[(SAMPLES, NSPLIT)](q, kv, qsl, seq_lens, bt, ctrl, ws, *common, D,
                                  q.stride(0), q.stride(1), kv.stride(0), kv.stride(1), kv.stride(2), kv.stride(3),
                                  bt.stride(0), nl, idx, float(scale), int(left), float(g_qk),
                                  LOG_R=LOG_R, NSPLIT_=NSPLIT, BN=BLOCK_N, BD=BD, WSW=wsw, num_warps=4)
    kk, vv = (key, val) if has_write else (q, q)
    sk = (kk.stride(0), kk.stride(1), vv.stride(0), vv.stride(1))
    keep = bool(keep_kv) and capkv.dtype == kv.dtype and capkv.shape[-1] >= BD
    _attn_judge[(SAMPLES,)](q, o, kv, kk, vv, qsl, seq_lens, bt, slots if has_write else qsl, ctrl, ws, state, capm,
                            capq, capo, capkv, capkvm, *common, Hkv, D,
                            slots.shape[0] if has_write else 0,
                            q.stride(0), q.stride(1), o.stride(0), o.stride(1), kv.stride(0), kv.stride(1),
                            kv.stride(2), kv.stride(3), bt.stride(0), *sk,
                            slot, nl, idx, int(left), float(u), float(sig_o), float(hsig_p), float(line),
                            capkv.shape[0], capq.shape[1],
                            LOG_R=LOG_R, NSPLIT_=NSPLIT, BN=BLOCK_N, BD=BD, BH=_p2(Hkv), WSW=wsw, NCOLS=COLS,
                            NCAPM=CAPM, HAS_WRITE=bool(has_write), KEEP_KV=keep, num_warps=4)


# ------------------------------------------------------------------------------------------ kernels and choice
def _dev_key(device) -> str:
    from ._group import _dev_key as k
    return k(device)


def kernel_usable(device, dtype) -> bool:
    if not HAVE_TRITON or os.environ.get("SDCGUARD_FAST", "1") == "0" or \
            os.environ.get("SDCGUARD_ATTN_FAST", "1") == "0":
        return False
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    return device.type == "cuda" or (device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1" and
                                     os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1")


def make_case(device, dtype, *, D=64, Hq=4, Hkv=2, N=16, NB=12, lens=((5, 1), (37, 9), (0, 0)), seed=0,
              padded_rows=3):
    """A small paged-attention call: queries, a KV cache, the metadata vLLM passes (requests (seq_len, query_len);
    (0, 0) is a CUDA-graph padding request; padded_rows rows past the real tokens) and the outputs a correct
    backend gives (the float64 recompute rounded to the dtype). For the kernels' first-use comparison and tests."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = len(lens)
    qlens = [ql for _, ql in lens]
    T = sum(qlens) + padded_rows
    W = max(1, max(-(-sl // N) for sl, _ in lens))
    perm = torch.randperm(NB, generator=g)
    bt = torch.zeros(R, W, dtype=torch.int32)
    k = 0
    for r, (sl, _) in enumerate(lens):
        for b in range(-(-sl // N)):
            bt[r, b] = int(perm[k % NB])
            k += 1
    kv = (torch.randn(NB, Hkv, N, 2 * D, generator=g) * 0.8).to(dtype)
    q = (torch.randn(T, Hq, D, generator=g) * 0.5).to(dtype)
    key = torch.zeros(T, Hkv, D, dtype=dtype)
    val = torch.zeros(T, Hkv, D, dtype=dtype)
    qsl = torch.zeros(R + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(qlens).cumsum(0)
    sls = torch.tensor([sl for sl, _ in lens], dtype=torch.int32)
    slots = torch.full((T,), -1, dtype=torch.int64)
    o = torch.zeros(T, Hq, D, dtype=dtype)
    scale = D ** -0.5
    for r, (sl, ql) in enumerate(lens):
        for jj in range(ql):
            i = int(qsl[r]) + jj
            pos = sl - ql + jj
            blk, off = int(bt[r, pos // N]), pos % N
            slots[i] = blk * N + off
            key[i] = kv[blk, :, off, :D]
            val[i] = kv[blk, :, off, D:]
            for hq in range(Hq):
                hk = hq // (Hq // Hkv)
                ks = torch.stack([kv[int(bt[r, t // N]), hk, t % N] for t in range(pos + 1)]).double()
                s = (ks[:, :D] @ q[i, hq].double()) * scale
                p = torch.softmax(s, 0)
                o[i, hq] = (p @ ks[:, D:]).to(dtype)
    return {"q": q.to(device), "o": o.to(device), "kv": kv.to(device), "key": key.to(device), "val": val.to(device),
            "qsl": qsl.to(device), "seq_lens": sls.to(device), "bt": bt.to(device), "slots": slots.to(device),
            "scale": scale}


def buffers(device, slots: int, width: int, cap_rows: int, kv_dtype) -> dict:
    """The buffers the checks of `slots` layers write on one device (fixed addresses, for CUDA graphs)."""
    bd = _p2(width)
    return {"state": torch.zeros(slots, COLS, dtype=torch.int64, device=device),
            "capm": torch.zeros(slots, CAPM, dtype=torch.int32, device=device),
            "capq": torch.zeros(slots, bd, dtype=torch.float32, device=device),
            "capo": torch.zeros(slots, bd, dtype=torch.float32, device=device),
            "capkv": torch.zeros(max(1, cap_rows), 2, bd, dtype=kv_dtype, device=device),
            "capkvm": torch.zeros(4, dtype=torch.int32, device=device),
            "ws": torch.zeros(SAMPLES, NSPLIT, WS_HEAD + 4 * bd, dtype=torch.float32, device=device),
            "ctrl": torch.zeros(2, dtype=torch.int64, device=device)}


def run_case(fn, c: dict, b: dict, *, step=0, nl=1, idx=0, slot=0, line=ATTN_Z, left=-1, has_write=True,
             keep_kv=True, active=True):
    """One check of the call c (make_case) with the buffers b (buffers), at a step."""
    b["ctrl"][0] = step
    b["ctrl"][1] = 1 if active else 0
    args = (c["q"], c["o"], c["kv"], c["key"], c["val"], c["qsl"], c["seq_lens"], c["bt"], c["slots"], b["state"],
            slot, b["ctrl"], nl, idx, c["scale"], left, line, b["capm"], b["capq"], b["capo"])
    if fn is attn_check_torch:
        fn(*args, has_write)
    else:
        fn(*args, b["capkv"], b["capkvm"], b["ws"], has_write, keep_kv)


VERIFY_SHAPES = ((64, 4, 2), (128, 2, 1))     # head size, query heads, KV heads


def verify_kernel(device, dtype) -> tuple:
    """The Triton kernels against the PyTorch check before their first use on a device and dtype, on small calls
    (a decode with a long context, a chunked prefill, a prefill, a CUDA-graph padding request and padding rows) over
    several steps, so every kind of sample is drawn: correct outputs (both quiet, scores at most 1, the same counts),
    one output made wrong at every token (both alarm on the same token, head and dimension and keep it), a key that
    differs from what the cache holds (both find every write wrong), a block id outside the cache (both count it)
    and an inactive step (nothing counted). (ok, why)."""
    key = (_dev_key(device), dtype)
    if key in STATUS:
        return STATUS[key]
    if not kernel_usable(device, dtype):
        STATUS[key] = (False, "the Triton kernels aren't available here")
        return STATUS[key]
    counts = [C_CALLS, C_SAMPLES, C_SKIP, C_UNJ, C_BADBLK, C_KEYS, C_WN, C_WBAD, C_WSLOT]
    try:
        for D, Hq, Hkv in VERIFY_SHAPES:
            base = make_case(device, dtype, D=D, Hq=Hq, Hkv=Hkv, lens=((70, 1), (21, 6), (3, 3), (0, 0)), seed=D)
            wrong = dict(base, o=base["o"].clone())
            wrong["o"][:, 1, 5] = wrong["o"][:, 1, 5].float() * 3 + 1
            badw = dict(base, key=base["key"].clone())
            badw["key"][:, 0, 3] = badw["key"][:, 0, 3].float() + 1
            badblk = dict(base, bt=base["bt"].clone())
            badblk["bt"][0, 1] = 10 ** 6
            for case_i, c in enumerate((base, wrong, badw, badblk)):
                for step in range(6):
                    res = []
                    for fn in (attn_check_torch, attn_check_triton):
                        b = buffers(device, 1, D, 128, dtype)
                        run_case(fn, c, b, step=step)
                        res.append((b["state"].cpu(), b["capm"].cpu()))
                    (s0, m0), (s1, m1) = res
                    if s0[0, counts].tolist() != s1[0, counts].tolist():
                        raise AssertionError(f"counts differ: {s0[0, counts].tolist()} vs {s1[0, counts].tolist()}")
                    if int(s0[0, C_CALLS]) != 1 or int(s0[0, C_SAMPLES]) + int(s0[0, C_SKIP]) + \
                            int(s0[0, C_BADBLK]) + int(s0[0, C_UNJ]) != SAMPLES:
                        raise AssertionError(f"samples not all accounted for: {s0[0].tolist()}")
                    z0, z1 = _f32(int(s0[0, C_ZT]) >> 32), _f32(int(s1[0, C_ZT]) >> 32)
                    alarm = (z0 > ATTN_Z, z1 > ATTN_Z)
                    if alarm[0] != alarm[1]:
                        raise AssertionError(f"one check alarms and the other doesn't ({z0} vs {z1})")
                    if case_i == 1 and alarm[0]:
                        for col in (C_ZT, C_ZH, C_ZD, C_ZS, C_ZP):
                            if int(s0[0, col]) & M32 != int(s1[0, col]) & M32:
                                raise AssertionError("a wrong output wasn't located the same way")
                        # (when several samples alarm, the two may keep different ones: any one over the line)
                        if abs(z0 - z1) > 1e-3 * z0 or any(int(m[0, CM_CLAIM]) != 1 or _f32(int(m[0, CM_Z])) <= ATTN_Z
                                                           for m in (m0, m1)):
                            raise AssertionError(f"scores or kept samples differ ({z0} vs {z1})")
                    elif case_i != 1 and (z0 > 1.0 or z1 > 1.0 or abs(z0 - z1) > 0.05):
                        raise AssertionError(f"scores of correct outputs differ or exceed 1 ({z0} vs {z1})")
                    if (case_i == 2) != bool(int(s0[0, C_WBAD])) and int(s0[0, C_WN]):
                        raise AssertionError("a KV write was judged wrongly")
                    if int(s0[0, C_WSLOT]) or (case_i == 2 and int(s0[0, C_WBAD]) != int(s0[0, C_WN])):
                        raise AssertionError("KV writes were judged wrongly")
                    if case_i == 2 and int(s0[0, C_WN]) and (s0[0, C_WLAST].item() != s1[0, C_WLAST].item() or
                                                             s0[0, C_WWHERE].item() != s1[0, C_WWHERE].item()):
                        raise AssertionError("a wrong KV write wasn't located the same way")
            b = buffers(device, 1, D, 16, dtype)
            run_case(attn_check_triton, base, b, step=0, active=False)
            if int(b["state"].abs().sum()):
                raise AssertionError("an inactive step was counted")
        STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    return STATUS[key]


# ------------------------------------------------------------------------------------------------ the recheck
def recheck(q_row: torch.Tensor, o_row: torch.Tensor, k_rows: torch.Tensor, v_rows: torch.Tensor, scale: float,
            dtype: torch.dtype, line: float = ATTN_Z) -> dict:
    """The kept sample recomputed in float64 on the CPU from the keys and values it attended, kept as the check read
    them (the same bound)."""
    D = q_row.shape[-1]
    q = q_row.double().cpu()
    o = o_row.double().cpu()
    K, V = k_rows.double().cpu(), v_rows.double().cpu()
    u, sig_o, hsig_p, g_qk = bound_consts(dtype, D)
    s = (K @ q) * scale
    m = s.max()
    p = torch.exp(s - m)
    l = p.sum()
    ref = (p @ V) / l
    A = (p @ V.abs()) / l
    e = g_qk * scale * (K.abs() @ q.abs()) + TWO22 * s.abs() + TWO21
    E = ((p * e) @ V.abs() + ref.abs() * (p * e).sum()) / l
    Sm = hsig_p * V.abs().sum(0) / l
    gam = (K.shape[0] / 16 + 32) * TWO22 + TWO22 * abs(float(m))
    bound = (2 * u + gam) * (A + ref.abs()) + E + Sm + sig_o
    z = torch.nan_to_num((o - ref).abs() / bound, nan=float("inf"), posinf=float("inf"))
    worst = float(z.max()) if z.numel() else 0.0
    bad = int((z > line).sum())
    out = {"worst_score": worst, "outputs_over_line": bad, "outputs": int(o.numel()), "keys": int(K.shape[0])}
    if not torch.isfinite(ref).all():
        return {**out, "verdict": "not_judged", "why": "the kept keys or values aren't finite"}
    if z.numel():                                  # the worst output's dimension and values (first of tied ones)
        d = int(z.argmax())
        out.update(dim=d, output=float(o[d]), expected=float(ref[d]))
    if bad:
        return {**out, "verdict": "output_wrong",
                "why": f"{bad} of {o.numel()} values of the kept output are more than {line:g} times the rounding "
                       "bound from attention recomputed in float64 from its query and the keys and values it attended"}
    return {**out, "verdict": "output_agrees",
            "why": "the float64 recomputation of the kept sample agrees with its output: the GPU's check disagreed"}


# ------------------------------------------------------------------------------------------- the layers
BACKENDS = ("FlashAttentionImpl", "TritonAttentionImpl")


def why_not(layer, impl, model_dtype=None) -> Optional[str]:
    """Why a vLLM attention layer can't be checked, or None."""
    name = type(impl).__name__
    if type(layer).__name__ != "Attention":
        return f"{type(layer).__name__}: only plain decoder attention layers are checked"
    if name not in BACKENDS:
        return f"{name}: only vLLM's FlashAttention and Triton backends are checked"
    fa = getattr(impl, "vllm_flash_attn_version", None)
    if name == "FlashAttentionImpl" and fa not in (None, 2, 3):
        return (f"FlashAttention {fa} (the bound is written for FlashAttention 2 and 3; version 4, vLLM's choice on "
                "Blackwell, computes part of exp2 as a polynomial and hasn't been run against it)")
    at = str(getattr(impl, "attn_type", "decoder")).lower()
    if not at.endswith("decoder") or "encoder" in at:
        return f"attention type {at}"
    if getattr(impl, "alibi_slopes", None) is not None:
        return "ALiBi"
    if getattr(impl, "sinks", None) is not None:
        return "attention sinks"
    if getattr(impl, "logits_soft_cap", 0) not in (0, None, 0.0):
        return "a logit soft cap"
    sw = getattr(impl, "sliding_window", (-1, -1))
    if sw not in (None, (-1, -1), [-1, -1]):
        return "a sliding window (not yet run on a GPU)"
    if getattr(impl, "chunk_lookback", -1) not in (-1, None):
        return "chunked local attention"
    if getattr(impl, "use_alibi_sqrt", False):
        return "ALiBi"
    if (getattr(impl, "dcp_world_size", 1) or 1) > 1:
        return "decode context parallelism"
    kvd = str(getattr(impl, "kv_cache_dtype", "auto"))
    if kvd not in ("auto", "bfloat16", "float16"):
        return f"KV cache dtype {kvd}"
    if model_dtype is not None and kvd != "auto" and kvd != str(model_dtype).replace("torch.", ""):
        return f"KV cache dtype {kvd} differs from the model's"
    hs, nh, nkv = (getattr(impl, a, None) for a in ("head_size", "num_heads", "num_kv_heads"))
    if not all(isinstance(v, int) and v > 0 for v in (hs, nh, nkv)) or nh % nkv:
        return "head counts it can't read"
    if hs > 256:
        return f"head size {hs}"
    hv = getattr(layer, "head_size_v", hs)
    if hv != hs:
        return "values of another size than the keys"
    return None


def call_issue(impl, query, output, kv_cache, md, output_scale) -> Optional[str]:
    """Why one call can't be checked, or None (read on the host from shapes and the metadata's Python fields)."""
    if output_scale is not None:
        return "fused output quantization"
    if not all(isinstance(t, torch.Tensor) for t in (query, output, kv_cache)):
        return "inputs it can't read"
    if query.dim() != 3 or output.dim() != 3 or kv_cache.dim() != 4:
        return "tensor layouts it doesn't know"
    T, Hq, D = query.shape
    if output.shape != query.shape or output.dtype not in (torch.bfloat16, torch.float16) or \
            query.dtype != output.dtype or output.stride(-1) != 1 or query.stride(-1) != 1:
        return "outputs it can't compare"
    if kv_cache.dtype != query.dtype or kv_cache.shape[1] != impl.num_kv_heads or kv_cache.shape[3] != 2 * D:
        return "a KV cache layout it doesn't know"
    causal = getattr(md, "causal", True)
    if causal is not True:
        return "attention that isn't causal for every request"
    for f in ("mm_prefix_query_range_tensor", "mm_prefix_range_tensor", "rswa_prefix_lens"):
        if getattr(md, f, None) is not None:
            return "multimodal prefix ranges or R-SWA"
    for f in ("query_start_loc", "seq_lens", "block_table"):
        if not isinstance(getattr(md, f, None), torch.Tensor):
            return f"metadata without {f}"
    if md.block_table.dim() != 2 or md.query_start_loc.shape[0] != md.seq_lens.shape[0] + 1 or \
            md.block_table.shape[0] < md.seq_lens.shape[0]:
        return "metadata it can't read"
    if md.seq_lens.shape[0] > MAX_REQUESTS:
        return f"more than {MAX_REQUESTS:,} requests in a step"
    return None


def torch_path_issue(dev, md, kv_cache) -> Optional[str]:
    """Why the PyTorch check (used without the Triton kernels, outside CUDA graphs) doesn't take a call on this
    device, or None: it gathers each sample's whole block-table row of keys, so on a GPU it takes up to
    TORCH_MAX_KEYS of them (on the CPU, for tests, any number)."""
    if dev.type == "cuda" and md.block_table.shape[1] * kv_cache.shape[2] > TORCH_MAX_KEYS:
        return f"the PyTorch check gathers a whole block-table row: on GPUs only up to {TORCH_MAX_KEYS:,} keys"
    return None


class AttnChecks:
    """The attention layers one ServeGuard checks (ServeGuard.attn_checks()). add() each layer, then finalize() once,
    before CUDA graphs are captured: it allocates the buffers (fixed addresses) and gives each layer its arguments
    (layer._sdcguard_attn). check_call() runs after each call of a layer's backend."""

    def __init__(self, guard, line: float = ATTN_Z):
        line = float(line)
        if not MIN_LINE <= line < float("inf"):
            raise ValueError(f"the attention check's line must be at least {MIN_LINE} (a correct output scores at "
                             f"most 1), not {line!r}")
        self.guard, self.line = guard, line
        self.layers: list = []                 # slot -> {"name", "layer", "impl", "device", "scale", "left", "D"}
        self.unsupported: dict = {}            # name -> why it isn't checked
        self.bufs: dict = {}                   # device -> buffers()
        self.gate: dict = {}                   # device -> [active, step] (the host's view, for eager calls)
        self._host: dict = {}
        self._marked: Optional[int] = None
        self._read_from: dict = {}
        self._last_clean: dict = {}
        self.not_checked_calls: dict = {}      # why -> calls (seen on the host: eager calls, and each graph capture)
        self.worst_seen: dict = {}
        self.calls_seen: dict = {}
        self.finalized = False
        self.cap_rows = 0
        self.stats = {"layers": 0, "calls_checked": 0, "samples": 0, "skipped": 0, "unjudged": 0, "keys": 0,
                      "bad_blocks": 0, "writes_checked": 0, "writes_wrong": 0, "slots_wrong": 0, "alerts": 0,
                      "reads": 0, "torch_calls": 0, "graph_calls_skipped": 0}

    def add(self, layer, impl, name: str, device, model_dtype=None) -> Optional[int]:
        if self.finalized:
            raise RuntimeError("attention layers are added before finalize()")
        why = why_not(layer, impl, model_dtype)
        if why is None and getattr(layer, "_sdcguard_attn", None) is not None:
            why = "already checked by another ServeGuard"
        if why is None and not (torch.device(device).type == "cuda" or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
            why = "not on a GPU"
        if why:
            self.unsupported[name] = why
            return None
        sw = getattr(impl, "sliding_window", (-1, -1)) or (-1, -1)
        self.layers.append({"name": name, "layer": layer, "impl": impl, "device": torch.device(device),
                            "scale": float(impl.scale), "left": int(sw[0]), "D": int(impl.head_size),
                            "Hq": int(impl.num_heads), "Hkv": int(impl.num_kv_heads),
                            "write": getattr(impl, "kv_sharing_target_layer_name", None) is None,
                            "backend": type(impl).__name__})
        self.stats["layers"] += 1
        return len(self.layers) - 1

    def finalize(self, max_len: int = 32768, dtypes=(torch.bfloat16,)) -> int:
        """Allocate the buffers (keeping up to min(max_len, CAP_ROWS) keys of an alarming sample, in the first of
        dtypes, unless SDCGUARD_ATTN_KEEP_KV=0), compare the kernels with the PyTorch check on each device for each
        dtype, and give each layer its arguments."""
        if self.finalized:
            return len(self.layers)
        keep = os.environ.get("SDCGUARD_ATTN_KEEP_KV", "1") != "0"
        self.cap_rows = min(int(max_len), CAP_ROWS) if keep else 0
        devs: dict = {}
        for lay in self.layers:
            devs.setdefault(lay["device"], []).append(lay)
        for dev, lays in devs.items():
            with torch.inference_mode(False):
                self.bufs[dev] = buffers(dev, len(self.layers), max(lay["D"] for lay in lays), self.cap_rows,
                                         dtypes[0] if dtypes else torch.bfloat16)
            self.gate[dev] = [False, 0]
            for dt in dtypes:
                verify_kernel(dev, dt)
                if STATUS.get((_dev_key(dev), dt), (False,))[0]:
                    for shape in sorted({(lay["D"], lay["Hq"], lay["Hkv"]) for lay in lays}):
                        self._precompile(dev, dt, *shape)
        for dev, lays in devs.items():
            for n, lay in enumerate(lays):
                lay["index"], lay["count"] = n, len(lays)
        for slot, lay in enumerate(self.layers):
            lay["layer"]._sdcguard_attn = (self, slot)
        self.finalized = True
        return len(self.layers)

    def _precompile(self, dev, dtype, D: int, Hq: int, Hkv: int) -> None:
        """Launch the kernels once (inactive: they return at once) for each variant a layer of this shape will use,
        so none is compiled during a CUDA graph capture. A failure turns the kernels off for this device and dtype
        (the PyTorch check then runs outside graphs), before any graph is captured."""
        try:
            c = make_case(dev, dtype, D=D, Hq=Hq, Hkv=Hkv, lens=((5, 1), (0, 0)), NB=2, padded_rows=1)
            for keep in (True, False):
                b = buffers(dev, 1, D, 16 if keep else 0, dtype)
                for has_write in (True, False):
                    run_case(attn_check_triton, c, b, has_write=has_write, keep_kv=keep, active=False)
                if int(b["state"].abs().sum()):
                    raise AssertionError("an inactive launch counted")
        except Exception as e:  # noqa: BLE001
            STATUS[(_dev_key(dev), dtype)] = (False, f"compiling for head size {D}: {type(e).__name__}: "
                                                     f"{str(e).splitlines()[0][:160] if str(e) else ''}")

    def kernel_status(self) -> dict:
        devs = {_dev_key(lay["device"]) for lay in self.layers}
        return {f"{d} {str(dt).replace('torch.', '')}": v for (d, dt), v in STATUS.items() if d in devs}

    def detach(self) -> None:
        try:
            if self.layers and self.finalized:
                self.poll(wait=True)
        finally:
            for lay in self.layers:
                lay["layer"].__dict__.pop("_sdcguard_attn", None)
            self.layers.clear()
            for d in (self.bufs, self.gate, self._host, self._read_from):
                d.clear()
            self.finalized = False

    # ------------------------------------------------------------------------------------------------- steps
    def mark_step(self, step: int) -> None:
        """The model step the work queued next belongs to (outside any CUDA graph): checks judge only model steps."""
        step = int(step) & M32
        self._marked = step
        for dev, b in self.bufs.items():
            b["ctrl"][0:1].fill_(step)
            b["ctrl"][1:2].fill_(1)
            self.gate[dev][:] = [True, step]
            self._read_from.setdefault(dev, step)

    def end_step(self) -> None:
        for dev, b in self.bufs.items():
            b["ctrl"][1:2].fill_(0)
            self.gate[dev][0] = False

    def check(self, slot: int, impl, query, key, value, kv_cache, md, output, output_scale=None) -> None:
        """After one call of a registered layer's backend. Inside a CUDA graph capture the kernels are always
        launched (each step's layer is chosen on the GPU, from the step marked before the replay). In eager calls
        during a model step only the step's layer launches them; outside model steps (vLLM's warm-up runs before it
        captures graphs) every layer launches them, and they return at once: that compiles each kernel for the
        shapes it will meet before any capture (compiling inside one can fail)."""
        lay = self.layers[slot]
        dev = lay["device"]
        if torch.compiler.is_compiling():
            return self._skip("traced by torch.compile (vLLM runs attention in an opaque op, so this isn't expected)")
        capturing = dev.type == "cuda" and torch.cuda.is_current_stream_capturing()
        gate = self.gate.get(dev)
        in_step = gate is not None and bool(gate[0])
        if not capturing and in_step and gate[1] % lay["count"] != lay["index"]:
            return None
        why = call_issue(impl, query, output, kv_cache, md, output_scale)
        if why:
            return self._skip(why)
        has_write = bool(lay["write"]) and isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor) and \
            isinstance(getattr(md, "slot_mapping", None), torch.Tensor) and key.dim() == 3 and value.dim() == 3 and \
            key.dtype == kv_cache.dtype and value.dtype == kv_cache.dtype and key.stride(-1) == 1 and \
            value.stride(-1) == 1 and key.shape[0] == query.shape[0] and value.shape[0] == query.shape[0]
        b = self.bufs[dev]
        T = min(int(query.shape[0]), int(getattr(md, "num_actual_tokens", query.shape[0]) or query.shape[0]))
        if T <= 0:
            return None
        q, o = query[:T], output[:T]
        slots = md.slot_mapping if has_write else None
        lay["dtype"] = o.dtype
        args = (q, o, kv_cache, key if has_write else None, value if has_write else None, md.query_start_loc,
                md.seq_lens, md.block_table, slots, b["state"], slot, b["ctrl"], lay["count"], lay["index"],
                lay["scale"], lay["left"], self.line, b["capm"], b["capq"], b["capo"])
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=False):
            k = (_dev_key(dev), o.dtype)
            if STATUS.get(k, (False,))[0]:
                try:
                    attn_check_triton(*args, b["capkv"], b["capkvm"], b["ws"], has_write, self.cap_rows > 0)
                    return None
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as e:  # noqa: BLE001  (a compile or launch problem: the PyTorch check from now on)
                    msg = str(e).splitlines()[0][:200] if str(e) else ""
                    STATUS[k] = (False, f"{type(e).__name__}: {msg}")
                    FALLBACKS[0] += 1
            if capturing:
                self.stats["graph_calls_skipped"] += 1
                return self._skip("inside a CUDA graph without the Triton kernels (the PyTorch check is too slow "
                                  "there)")
            if not in_step:
                return None
            why = torch_path_issue(dev, md, kv_cache)
            if why:
                return self._skip(why)
            self.stats["torch_calls"] += 1
            attn_check_torch(*args, has_write)
        return None

    def _skip(self, why: str) -> None:
        self.not_checked_calls[why] = self.not_checked_calls.get(why, 0) + 1
        return None

    def _window(self, dev) -> Optional[list]:
        if self._marked is None:
            return None
        a = self._read_from.pop(dev, None)
        return [a if a is not None else self._marked, self._marked]

    # ------------------------------------------------------------------------------------------------- reads
    def poll(self, wait: bool = False) -> list:
        events = []
        for dev, b in self.bufs.items():
            st, cm = b["state"], b["capm"]
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

    def counts(self) -> dict:
        """Calls checked per layer: read back so far plus what hasn't been read (waits for the GPU)."""
        live: dict = {}
        for dev, b in self.bufs.items():
            rows = b["state"].to("cpu")
            pend = self._host.get(dev)
            if pend is not None:
                pend[2].synchronize()
                rows = rows + pend[0]
            for slot, lay in enumerate(self.layers):
                if lay["device"] == dev:
                    live[lay["name"]] = int(rows[slot, C_CALLS])
        return {lay["name"]: self.calls_seen.get(lay["name"], 0) + live.get(lay["name"], 0) for lay in self.layers}

    def coverage(self) -> dict:
        c = self.counts()
        return {"layers": len(self.layers), "layers_checked": sum(v > 0 for v in c.values()),
                "never_checked": sorted(n for n, v in c.items() if v == 0), "calls_checked": sum(c.values()),
                "samples_per_call": SAMPLES, "one_layer_a_step": True, "kernel": self.kernel_status(),
                "not_attached": dict(self.unsupported), "calls_not_checked": dict(self.not_checked_calls),
                "samples": self.stats["samples"], "keys_recomputed": self.stats["keys"],
                "samples_without_a_token": self.stats["skipped"], "samples_not_judged": self.stats["unjudged"],
                "writes_checked": self.stats["writes_checked"]}

    def _report(self, dev, host: torch.Tensor, hcap: torch.Tensor, window) -> list:
        out = []
        worst = 0.0
        self.stats["reads"] += 1
        for slot, lay in enumerate(self.layers):
            if lay["device"] != dev or slot >= host.shape[0]:
                continue
            k = [int(v) for v in host[slot].tolist()]
            for stat, col in (("calls_checked", C_CALLS), ("samples", C_SAMPLES), ("skipped", C_SKIP),
                              ("unjudged", C_UNJ), ("keys", C_KEYS), ("bad_blocks", C_BADBLK),
                              ("writes_checked", C_WN), ("writes_wrong", C_WBAD), ("slots_wrong", C_WSLOT)):
                self.stats[stat] += k[col]
            self.calls_seen[lay["name"]] = self.calls_seen.get(lay["name"], 0) + k[C_CALLS]
            z = _f32(k[C_ZT] >> 32)
            worst = max(worst, z)
            self.worst_seen[lay["name"]] = max(self.worst_seen.get(lay["name"], 0.0), z)
            alarmed = False
            if z > self.line:
                out.append(self._alert(dev, slot, k, [int(v) for v in hcap[slot].tolist()], window, z))
                alarmed = True
            if k[C_WBAD] or k[C_WSLOT]:
                out.append(self._write_alert(dev, slot, k, window))
                alarmed = True
            if k[C_BADBLK]:
                out.append(self._event(dev, slot, {
                    "where": lay["name"], "finding": "block_table_out_of_range", "samples": k[C_BADBLK],
                    "read_window_steps": window,
                    "reading": "the block table of a sampled request named KV-cache blocks outside the cache: the "
                               "attention kernel read memory that isn't part of the KV cache"}))
                alarmed = True
            if not alarmed and k[C_CALLS] and window is not None:
                self._last_clean[lay["name"]] = list(window)
        g = self.guard
        if g is not None and getattr(g, "metrics", None) is not None and hasattr(g.metrics, "observe_attn"):
            try:
                g.metrics.observe_attn(self, out, worst)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
        return out

    def _alert(self, dev, slot: int, k: list, cap: list, window, z: float) -> SDCEvent:
        lay = self.layers[slot]
        b = self.bufs[dev]
        dtype = lay.get("dtype") or b["capkv"].dtype
        yv, rv = _f32(k[C_ZO]), _f32(k[C_ZR])
        captured, rc = None, None
        if cap[CM_CLAIM]:
            D = lay["D"]
            q_row, o_row = b["capq"][slot, :D].to("cpu"), b["capo"][slot, :D].to("cpu")
            kvm = [int(v) for v in b["capkvm"].tolist()]
            kept = bool(cap[CM_KEPT]) and kvm[0] == 1 and kvm[1] == slot and kvm[3] == cap[CM_STEP]
            captured = {"row": cap[CM_ROW] & M32, "head": cap[CM_HEAD], "kv_head": cap[CM_HK],
                        "request_index": cap[CM_REQ], "position": cap[CM_POS], "seq_len": cap[CM_SL],
                        "first_key": cap[CM_LO], "step": (cap[CM_STEP] & M32) if self._marked is not None else None,
                        "score": _f32(cap[CM_Z]), "keys_kept": kept}
            if kept:
                L = kvm[2]
                kvr = b["capkv"][:L].to("cpu")
                rc = recheck(q_row, o_row, kvr[:, 0, :D], kvr[:, 1, :D], lay["scale"], dtype, self.line)
                b["capkvm"].zero_()
            else:
                rc = {"verdict": "not_rechecked",
                      "why": "the keys and values it attended weren't kept (another layer's alarm in this read took "
                             "the buffer, there were more than the buffer holds, the PyTorch check ran, or "
                             "SDCGUARD_ATTN_KEEP_KV=0)"}
            b["capm"][slot].zero_()
        place = {"row": k[C_ZT] & M32, "head": k[C_ZH] & M32, "dim": k[C_ZD] & M32, "position": k[C_ZP] & M32,
                 "step": (k[C_ZS] & M32) if self._marked is not None else None, "output": yv, "expected": rv,
                 "place": PLACE_MAXIMA}
        if captured is not None and rc is not None and rc.get("outputs_over_line") and rc.get("dim") is not None:
            place = {"row": captured["row"], "head": captured["head"], "dim": rc["dim"],
                     "position": captured["position"], "step": captured["step"], "output": rc["output"],
                     "expected": rc["expected"],
                     "place": PLACE_KEPT.replace("kept row's worst output", "kept sample's worst output").replace(
                         "the kept row is", "the kept sample is")}
        detail = {"where": lay["name"], "backend": lay["backend"], "finding": "output_wrong_or_misread", "max_z": z,
                  **place,
                  "captured": captured,
                  "recheck": rc or {"verdict": "not_rechecked",
                                    "why": "no sample was kept for this alert (the layer's kept sample was taken by an "
                                           "earlier alert that hasn't been read yet)"},
                  "read_window_steps": window, "last_clean_read_steps": self._last_clean.get(lay["name"]),
                  "check_config": {"line": self.line, "unit_roundoff": dtype_consts(dtype)[0], "samples": SAMPLES,
                                   "reference": "fp32 recompute of softmax(scale q.K) V from the KV cache, through the "
                                                "block table, for one query head of one token"},
                  "reading": "an attention output differs from attention recomputed from the request's own keys and "
                             "values in the KV cache by more than a correct backend's rounding allows: wrong "
                             "arithmetic in the attention kernel, or it read other keys than the metadata names"}
        return self._event(dev, slot, detail)

    def _write_alert(self, dev, slot: int, k: list, window) -> SDCEvent:
        lay = self.layers[slot]
        last, wh = k[C_WLAST], k[C_WWHERE]
        kind = (wh >> 28) & 0xF
        info = wh & 0x0FFFFFFF
        latest = {"step": unwrap31(last >> 32, self._marked) if self._marked is not None else None, "row": last & M32}
        # (their own key, with the step in its top bits: of the latest step's wrong writes, the largest packed one)
        wrong = {"content_wrong": bool(kind & W_CONTENT), "slot_wrong": bool(kind & W_SLOT)}
        if kind & W_CONTENT:
            wrong.update({"kv_head": info >> 12, "dim": (info >> 1) & 0x7FF, "part": "value" if info & 1 else "key"})
        detail = {"where": lay["name"], "backend": lay["backend"], "finding": "kv_write_wrong",
                  "writes_checked": k[C_WN], "writes_with_wrong_content": k[C_WBAD],
                  "writes_to_another_slot": k[C_WSLOT], "latest": latest, "a_wrong_write": wrong,
                  "a_wrong_write_is": "a wrong write of the latest step (the largest of that step's packed kinds and "
                                      "places): it can be another token than the latest row",
                  "captured": {"row": latest["row"], "step": latest["step"]},
                  "read_window_steps": window, "last_clean_read_steps": self._last_clean.get(lay["name"]),
                  "reading": "a sampled token's KV-cache slot doesn't hold the key and value the layer computed for "
                             "it, or vLLM wrote them to another slot than the one its position's block names: "
                             "attention would read wrong state for that position from now on"}
        return self._event(dev, slot, detail)

    def _event(self, dev, slot: int, detail: dict) -> SDCEvent:
        g = self.guard
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        key = str(dev)
        if key not in ident:
            ident[key] = device_identity(dev)
        ev = SDCEvent(detector="serve_attn", step=None, detail=detail, identity=dict(ident[key]))
        self.stats["alerts"] += 1
        if g is not None:
            if getattr(g, "attribute", None) is not None:
                try:
                    g.attribute(ev)
                except Exception:  # noqa: BLE001  (attribution only)
                    g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
            g.log.add(ev)
            if getattr(g, "on_alert", None):
                try:
                    g.on_alert(ev)
                except Exception:  # noqa: BLE001
                    pass
        return ev


def check_call(entry, impl, query, key, value, kv_cache, md, output, output_scale=None) -> None:
    """After a backend call of a registered layer (entry: layer._sdcguard_attn)."""
    checks, slot = entry
    checks.check(slot, impl, query, key, value, kv_cache, md, output, output_scale)
