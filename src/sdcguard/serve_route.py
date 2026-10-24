"""Checks of MoE routing in serving: did the router pick the experts and weights its policy gives for the logits?

A fused MoE layer's router turns each token's router logits into its top-k experts and their weights, by the
model's routing policy: scores by softmax or sigmoid; a correction bias added for choosing (not for the weights);
experts chosen within the best groups (DeepSeek's grouped top-k: groups ranked by their top-2 biased scores, or
their best score without a bias); weights renormalized to sum to 1 or not, then scaled by the routed scaling factor.
vLLM computes this in fused kernels of its own (topk_softmax, topk_sigmoid, grouped top-k), and a wrong expert is
an output from the wrong weights with nothing in the arithmetic of the experts wrong: the experts' recompute
(serve_moe.py) takes the routing as given and would agree.

The routers take turns, as the MoE layers do in serve_moe.py: at each model step one router (the one whose index is
the step modulo their number) recomputes its policy in fp32 for SAMPLES real rows of the step, from its own logits
and written independently of vLLM's kernels (one Triton launch per call that returns at once for the others; the
same check in PyTorch operations where Triton isn't usable). It judges:
  - the chosen experts: each one's choice score at least the k-th best within the groups chosen, and the k distinct.
    Scores within a tolerance of the k-th best may be picked either way (equal scores are, and vLLM's kernels round
    differently from this recompute): 1e-6 of the score's size for fp32 logits, plus 4 unit roundoffs of the logits'
    dtype for 16-bit logits (vLLM's grouped top-k kernel computes scores and adds the bias in the logits' dtype).
    Where groups near the cut score within twice that of each other, either may be chosen, and the experts are
    judged against the groups surely in;
  - the weights: the scores of the experts the router chose, renormalized over them and scaled, to 2e-5 plus 1e-4 of
    their size, plus 6 unit roundoffs of a 16-bit dtype.
A finding is a wrong expert set or a wrong weight; it is reported with the row's step and position, so the request is
named. A row whose experts are all -1 is padding vLLM marked and isn't judged. Policies other than these (custom
routing functions, hash routing, image-token biases, fused shared experts, routing simulators, expert load balancing)
are listed as not checked, as are monolithic MoE kernels that route inside the kernel (no router call to see).
"""
from __future__ import annotations

import os
import struct
import threading
from typing import Optional

import torch

from .events import SDCEvent, device_identity
from .serve_norm import pick

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

SAMPLES = 2
TIE = 1e-6
W_ABS, W_REL = 2e-5, 1e-4
U_CHOICE, U_WEIGHT = 4.0, 6.0
ROW_A, ROW_B = 7919, 104729
M32 = 0xFFFFFFFF
MAX_EXPERTS, MAX_K, MAX_GROUPS = 2048, 32, 64
COLS = 6
# calls judged, rows with a wrong expert set, rows with a wrong weight, last bad (step << 32 | kind << 24 | row),
# worst weight difference bits, rows judged
C_CALLS, C_SET, C_WEIGHT, C_LAST, C_WDIFF, C_ROWS = range(COLS)
KIND_SET, KIND_WEIGHT = 2, 1           # a wrong expert set outranks a wrong weight in the step's record
SCORING = {"softmax": 0, "sigmoid": 1}
STATUS: dict = {}
FALLBACKS = [0]


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def unit_roundoff(dtype) -> float:
    """Of 16-bit logits (half their dtype's epsilon); 0 for fp32 and wider, which the 1e-6 tie covers."""
    return float(torch.finfo(dtype).eps) / 2 if dtype in (torch.bfloat16, torch.float16) else 0.0


def _tol(v: torch.Tensor, u: float) -> torch.Tensor:
    a = torch.nan_to_num(v.abs(), posinf=0.0)
    return TIE * a.clamp(min=1.0) + U_CHOICE * u * a


CAP_COLS = 6
# captured: taken, row, step, kind, experts (logits width), logits dtype (0 fp32, 1 bf16, 2 fp16)
DTYPE_CODE = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}
CODE_DTYPE = {v: k for k, v in DTYPE_CODE.items()}


def judge(lf: torch.Tensor, kid: torch.Tensor, kw: torch.Tensor, k: int, scoring: int, renormalize: bool,
          bias: Optional[torch.Tensor], groups: int, topk_group: int, scale: float, u: float) -> tuple:
    """Rows of logits lf [r, E] (fp32 on the GPU; float64 for the recheck) against the router's ids and weights
    [r, k]: (wrong expert set [r], wrong weight [r], weight differences [r, k]), computed in lf's dtype."""
    e = lf.shape[-1]
    s = torch.softmax(lf, -1) if scoring == SCORING["softmax"] else torch.sigmoid(lf)
    choice = s + bias.to(lf.dtype).reshape(1, -1) if bias is not None else s
    neg = torch.full_like(choice, float("-inf"))
    thr = allow = torch.ones_like(choice, dtype=torch.bool)
    valid = (kid >= 0) & (kid < e)
    kc = kid.clamp(min=0, max=e - 1)
    kw = kw.to(lf.dtype)
    if groups:
        r, gsz = choice.shape[0], e // groups
        g = choice.view(r, groups, gsz)
        gs = g.topk(2, -1).values.sum(-1) if bias is not None else g.amax(-1)
        srt = gs.sort(-1, descending=True).values
        gth = srt[:, topk_group - 1]
        gnext = srt[:, topk_group] if topk_group < groups else torch.full_like(gth, float("-inf"))
        tolg = (2 * _tol(gth, u)).unsqueeze(-1)
        amb = (gnext.unsqueeze(-1) >= gth.unsqueeze(-1) - tolg)
        clear = gs >= gth.unsqueeze(-1)
        thr_g = torch.where(amb, gs > gth.unsqueeze(-1) + tolg, clear)
        allow_g = torch.where(amb, gs >= gth.unsqueeze(-1) - tolg, clear)
        thr = thr_g.unsqueeze(-1).expand(r, groups, gsz).reshape(r, e)
        allow = allow_g.unsqueeze(-1).expand(r, groups, gsz).reshape(r, e)
    # the bound a chosen expert's choice score must reach: the k-th best within the groups chosen (surely chosen,
    # where groups near the cut tie)
    bound = torch.where(thr, choice, neg).topk(k, -1).values[:, -1].unsqueeze(-1).expand(-1, k)
    if groups:
        # and, true whichever groups near the cut the router took: each chosen group holds k-th bests (within the
        # group) at least the topk_group-th smallest of those over the groups that may be chosen, and a chosen expert
        # is among the k best of its own group
        kwg = g.topk(k, -1).values[..., -1] if gsz >= k else torch.full_like(gs, float("-inf"))
        low = torch.where(allow_g, kwg, torch.full_like(kwg, float("inf"))).sort(-1).values[:, topk_group - 1]
        bound = torch.maximum(torch.maximum(bound, low.unsqueeze(-1)), kwg.gather(1, kc // gsz))
    ck = choice.gather(1, kc)
    ok = valid & allow.gather(1, kc) & (ck >= bound - _tol(bound, u))
    srt_id = kid.sort(-1).values
    dup = (srt_id[:, 1:] == srt_id[:, :-1]).any(-1) if k > 1 else torch.zeros_like(valid[:, 0])
    set_bad = ~ok.all(-1) | dup
    # weights: the scores of the experts the router chose, renormalized over them and scaled
    sk = torch.where(valid, s.gather(1, kc), torch.zeros_like(kw))
    ref = sk / (sk.sum(-1, keepdim=True) + 1e-20) if renormalize else sk
    ref = ref * scale
    diff = torch.where(valid, (kw - ref).abs(), torch.zeros_like(kw))
    w_bad = (~(diff <= W_ABS + (W_REL + U_WEIGHT * u) * ref.abs())).any(-1) & ~set_bad
    return set_bad, w_bad, diff


def route_check_torch(logits: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor, k: int, scoring: int,
                      renormalize: bool, bias: Optional[torch.Tensor], groups: int, topk_group: int, scale: float,
                      rows_buf: torch.Tensor, stepbuf: torch.Tensor, state: torch.Tensor, slot: int, nlayers: int,
                      index: int, cap_meta: torch.Tensor, cap_lg: torch.Tensor, cap_ids: torch.Tensor,
                      cap_w: torch.Tensor) -> None:
    """The check in PyTorch operations (no host sync: safe in CUDA graphs). It always computes; only the active
    router's result counts. logits [M, E]; topk_w, topk_ids [M, k] as the router returned them. The first bad row
    (logits, ids, weights) is kept in the cap_ tensors until it is read."""
    m, e = logits.shape[0], logits.shape[-1]
    if m == 0 or topk_ids.shape[0] < m:
        return
    u = unit_roundoff(logits.dtype)
    dev = logits.device
    real = rows_buf.reshape(-1)[0].clamp(min=0, max=m)
    step = stepbuf.reshape(-1)[0] & M32
    live = (real > 0) & ((step % max(nlayers, 1)) == index)
    j = torch.arange(SAMPLES, device=dev, dtype=torch.int64)
    rows = (step * ROW_A + j * ROW_B) % real.clamp(min=1)
    lf = logits[rows].float()
    kid = topk_ids[rows].long()
    kw = topk_w[rows].float()
    set_bad, w_bad, diff = judge(lf, kid, kw, k, scoring, renormalize, bias, groups, topk_group, scale, u)
    judged = live & ~(kid == -1).all(-1)
    srow = state[slot]
    srow[C_CALLS] += live.to(torch.int64)
    srow[C_ROWS] += judged.sum()
    sb, wb = set_bad & judged, w_bad & judged
    srow[C_SET] += sb.sum()
    srow[C_WEIGHT] += wb.sum()
    one = torch.ones((), dtype=torch.int64, device=dev)
    kind = torch.where(sb, KIND_SET * one, torch.where(wb, KIND_WEIGHT * one, 0 * one))
    rec = (step << 32) | (kind << 24) | (rows & 0xFFFFFF)
    rec = torch.where(kind > 0, rec, torch.zeros_like(rec)).amax()
    srow[C_LAST] = torch.maximum(srow[C_LAST], rec)
    worst = torch.where(judged.unsqueeze(-1), diff, torch.zeros_like(diff)).amax()
    worst = worst.float().reshape(1).view(torch.int32).to(torch.int64).reshape(()) & M32
    srow[C_WDIFF] = torch.maximum(srow[C_WDIFF], worst)
    bad = sb | wb
    jb = bad.to(torch.int64).argmax()
    take = bad.any() & (cap_meta[slot, 0] == 0)
    cap_lg[slot, :e] = torch.where(take, pick(lf, jb), cap_lg[slot, :e])
    cap_ids[slot, :k] = torch.where(take, pick(kid, jb), cap_ids[slot, :k])
    cap_w[slot, :k] = torch.where(take, pick(kw, jb), cap_w[slot, :k])
    meta = torch.stack([one, pick(rows, jb), step, pick(kind, jb), e * one, DTYPE_CODE.get(logits.dtype, 0) * one])
    cap_meta[slot] = torch.where(take, meta, cap_meta[slot])


def recheck(lg_row: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, policy: dict, dtype) -> dict:
    """The kept row judged again on the CPU in float64 (the logits as the router got them; the same policy and
    tolerances): output_wrong when it is wrong there too, output_agrees when it isn't (the GPU check erred)."""
    p = policy
    bias = p["bias"].detach().double().cpu() if p["bias"] is not None else None
    sb, wb, diff = judge(lg_row.double().reshape(1, -1).cpu(), ids.long().reshape(1, -1).cpu(),
                         w.double().reshape(1, -1).cpu(), p["k"], p["scoring"], p["renormalize"], bias, p["groups"],
                         p["topk_group"], p["scale"], unit_roundoff(dtype))
    out = {"experts": [int(v) for v in ids.tolist()], "weights": [float(v) for v in w.tolist()],
           "worst_weight_difference": float(diff.max()) if diff.numel() else 0.0}
    if bool(sb[0]) or bool(wb[0]):
        what = "experts" if bool(sb[0]) else "weights"
        return {**out, "verdict": "output_wrong",
                "why": f"recomputed in float64 on the CPU from the kept logits, the router's {what} are wrong for "
                       "its policy"}
    return {**out, "verdict": "output_agrees",
            "why": "recomputed in float64 on the CPU from the kept logits, the router's experts and weights fit its "
                   "policy: the GPU check's own result was wrong"}


if HAVE_TRITON:
    @triton.jit
    def _route_scores(lg, mx, den, SCORING_: tl.constexpr):
        if SCORING_ == 0:
            return tl.exp(lg - mx) / den
        else:
            return 1.0 / (1.0 + tl.exp(-lg))

    @triton.jit
    def _route_kernel(logits, tw, tid, bias, rows_buf, stepbuf, state, cap_meta, cap_lg, cap_ids, cap_w, E, K, sl,
                      stw, sti, groups, topk_group, scale, u, slot, nl, idx, dcode, CAPE, CAPK, SCORING_: tl.constexpr, RENORM: tl.constexpr, HAS_BIAS: tl.constexpr,
                      GROUPED: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr, BG: tl.constexpr):
        j = tl.program_id(0)
        step = tl.load(stepbuf) & 0xFFFFFFFF
        real = tl.load(rows_buf)
        if (real > 0) & ((step % nl) == idx):
            row = ((step * 7919 + j * 104729) % real).to(tl.int64)
            NEG = float("-inf")
            offs = tl.arange(0, BE)
            em = offs < E
            lg = tl.load(logits + row * sl + offs, mask=em, other=NEG).to(tl.float32)
            mx = tl.max(lg, 0)
            den = tl.sum(tl.where(em, tl.exp(lg - mx), 0.0), 0)
            s = tl.where(em, _route_scores(lg, mx, den, SCORING_), 0.0)
            ch = s
            if HAS_BIAS:
                ch = s + tl.load(bias + offs, mask=em, other=0.0).to(tl.float32)
            ch = tl.where(em, ch, NEG)
            thr = em
            go = tl.arange(0, BG)
            thr_g = go < 0
            allow_g = go < 0
            gsz = E // groups
            if GROUPED:
                gidx = offs // gsz
                gs = tl.full([BG], NEG, tl.float32)
                kwg = tl.full([BG], NEG, tl.float32)
                for g in range(0, groups):
                    v = tl.where(em & (gidx == g), ch, NEG)
                    m1 = tl.max(v, 0)
                    sc = m1
                    if HAS_BIAS:
                        a1 = tl.argmax(v, 0)
                        sc = m1 + tl.max(tl.where(offs == a1, NEG, v), 0)
                    gs = tl.where(go == g, sc, gs)
                    rv = v
                    kg = NEG
                    for t in range(0, K):
                        kg = tl.max(rv, 0)
                        rv = tl.where(offs == tl.argmax(rv, 0), NEG, rv)
                    kwg = tl.where(go == g, kg, kwg)
                rem = gs
                gth = NEG
                for t in range(0, topk_group):
                    gth = tl.max(rem, 0)
                    rem = tl.where(go == tl.argmax(rem, 0), NEG, rem)
                gnext = tl.max(rem, 0)
                ag = tl.abs(gth)
                ag = tl.where(ag == float("inf"), 0.0, ag)
                tolg = 2.0 * (1e-6 * tl.maximum(ag, 1.0) + 4.0 * u * ag)
                amb = gnext >= gth - tolg
                clear = (gs >= gth) & (go < groups)
                thr_g = tl.where(amb, (gs > gth + tolg) & (go < groups), clear)
                allow_g = tl.where(amb, (gs >= gth - tolg) & (go < groups), clear)
                thr = em & (offs < 0)
                for g in range(0, groups):
                    tg = tl.sum(tl.where((go == g) & thr_g, 1, 0), 0) > 0
                    thr = thr | (em & (gidx == g) & tg)
                rl = tl.where(allow_g, kwg, float("inf"))
                low = NEG
                for t in range(0, topk_group):
                    low = tl.min(rl, 0)
                    rl = tl.where(go == tl.argmin(rl, 0), float("inf"), rl)
            rem = tl.where(thr, ch, NEG)
            kth = NEG
            for t in range(0, K):
                kth = tl.max(rem, 0)
                rem = tl.where(offs == tl.argmax(rem, 0), NEG, rem)
            ko = tl.arange(0, BK)
            km = ko < K
            kid = tl.load(tid + row * sti + ko, mask=km, other=-1).to(tl.int32)
            kw = tl.load(tw + row * stw + ko, mask=km, other=0.0).to(tl.float32)
            pad = tl.sum(tl.where(km & (kid == -1), 1, 0), 0) == K
            valid = km & (kid >= 0) & (kid < E)
            kc = tl.minimum(tl.maximum(kid, 0), E - 1)
            lk = tl.load(logits + row * sl + kc, mask=km, other=0.0).to(tl.float32)
            sk = _route_scores(lk, mx, den, SCORING_)
            ck = sk
            if HAS_BIAS:
                ck = sk + tl.load(bias + kc, mask=km, other=0.0).to(tl.float32)
            alk = km
            bnd = tl.zeros([BK], tl.float32) + kth
            if GROUPED:
                gk = kc // gsz
                alk = km & (kc < 0)
                bnd = tl.maximum(bnd, low)
                for g in range(0, groups):
                    a_g = tl.sum(tl.where((go == g) & allow_g, 1, 0), 0) > 0
                    alk = alk | (km & (gk == g) & a_g)
                    own = tl.max(tl.where(go == g, kwg, NEG), 0)
                    bnd = tl.where(gk == g, tl.maximum(bnd, own), bnd)
            ak = tl.abs(bnd)
            ak = tl.where(ak == float("inf"), 0.0, ak)
            tol = 1e-6 * tl.maximum(ak, 1.0) + 4.0 * u * ak
            ok = valid & alk & (ck >= bnd - tol)
            nbad = tl.sum(tl.where(km & (ok == 0), 1, 0), 0)
            same = (kid[:, None] == kid[None, :]) & km[:, None] & km[None, :] & (ko[:, None] < ko[None, :])
            ndup = tl.sum(tl.sum(tl.where(same, 1, 0), 1), 0)
            set_bad = (nbad > 0) | (ndup > 0)
            skv = tl.where(valid, sk, 0.0)
            ref = skv
            if RENORM:
                ref = skv / (tl.sum(skv, 0) + 1e-20)
            ref = ref * scale
            diff = tl.where(valid, tl.abs(kw - ref), 0.0)
            wl = 2e-5 + (1e-4 + 6.0 * u) * tl.abs(ref)
            nw = tl.sum(tl.where(valid & ((diff <= wl) == 0), 1, 0), 0)
            w_bad = (nw > 0) & (set_bad == 0)
            judged = pad == 0
            base = state + slot * 6
            if j == 0:
                tl.atomic_add(base + 0, 1)
            if judged:
                tl.atomic_add(base + 5, 1)
                if set_bad:
                    tl.atomic_add(base + 1, 1)
                    tl.atomic_max(base + 3, (step << 32) | (2 << 24) | (row & 0xFFFFFF))
                elif w_bad:
                    tl.atomic_add(base + 2, 1)
                    tl.atomic_max(base + 3, (step << 32) | (1 << 24) | (row & 0xFFFFFF))
                worst = tl.max(diff, 0)
                tl.atomic_max(base + 4, worst.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF)
                if set_bad | w_bad:
                    mp = cap_meta + slot * 6
                    # the first bad row since the last read is kept (compare and swap in the flag's own int64:
                    # Triton's compiler rejects int32 literals on an int64 pointer, its interpreter doesn't)
                    zero = tl.full((), 0, tl.int64)
                    if tl.atomic_cas(mp, zero, zero + 1) == 0:
                        tl.store(cap_lg + slot * CAPE + offs, lg, mask=em)
                        tl.store(cap_ids + slot * CAPK + ko, kid.to(tl.int64), mask=km)
                        tl.store(cap_w + slot * CAPK + ko, kw, mask=km)
                        tl.store(mp + 1, row)
                        tl.store(mp + 2, step)
                        tl.store(mp + 3, tl.where(set_bad, 2, 1).to(tl.int64))
                        tl.store(mp + 4, E)
                        tl.store(mp + 5, dcode)


def _p2(n: int, lo: int = 2) -> int:
    return max(lo, 1 << (max(1, int(n)) - 1).bit_length())


def route_check_triton(logits, topk_w, topk_ids, k, scoring, renormalize, bias, groups, topk_group, scale, rows_buf,
                       stepbuf, state, slot, nlayers, index, cap_meta, cap_lg, cap_ids, cap_w) -> None:
    m, e = logits.shape[0], logits.shape[-1]
    if m == 0 or topk_ids.shape[0] < m:
        return
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    tw = topk_w if topk_w.stride(-1) == 1 else topk_w.contiguous()
    ti = topk_ids if topk_ids.stride(-1) == 1 else topk_ids.contiguous()
    _route_kernel[(SAMPLES,)](
        logits, tw, ti, bias if bias is not None else logits, rows_buf, stepbuf, state, cap_meta, cap_lg, cap_ids,
        cap_w, e, k, logits.stride(0), tw.stride(0), ti.stride(0), max(groups, 1), max(topk_group, 1), float(scale),
        unit_roundoff(logits.dtype), slot, max(nlayers, 1), index, DTYPE_CODE.get(logits.dtype, 0), cap_lg.stride(0),
        cap_ids.stride(0), SCORING_=scoring, RENORM=bool(renormalize), HAS_BIAS=bias is not None,
        GROUPED=bool(groups), BE=_p2(e, 16), BK=_p2(k), BG=_p2(groups), num_warps=4)


def new_caps(device, rows: int = 256, experts: int = MAX_EXPERTS) -> tuple:
    """The kept-row tensors: (meta [rows, CAP_COLS] int64, logits [rows, experts] fp32, ids [rows, MAX_K] int64,
    weights [rows, MAX_K] fp32)."""
    return (torch.zeros(rows, CAP_COLS, dtype=torch.int64, device=device),
            torch.zeros(rows, experts, dtype=torch.float32, device=device),
            torch.zeros(rows, MAX_K, dtype=torch.int64, device=device),
            torch.zeros(rows, MAX_K, dtype=torch.float32, device=device))


def _dev_key(device) -> str:
    return str(device)


def kernel_usable(device) -> bool:
    if not HAVE_TRITON or os.environ.get("SDCGUARD_FAST", "1") == "0" or \
            os.environ.get("SDCGUARD_ROUTE_FAST", "1") == "0":
        return False
    return device.type == "cuda" or (device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1" and
                                     os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1")


def verify_kernel(device, dtype) -> tuple:
    """The Triton kernel against the PyTorch check before its first use on a device and logits dtype: plain top-k
    (softmax, renormalized) and grouped top-k (sigmoid, a bias, scaled), each correct (both quiet, the same counts),
    with an expert moved and with a weight changed (both name the same finding and row), and an inactive step
    (nothing changes). (ok, why)."""
    key = (_dev_key(device), dtype)
    if key in STATUS:
        return STATUS[key]
    if not kernel_usable(device):
        STATUS[key] = (False, "the Triton kernel isn't available here")
        return STATUS[key]
    try:
        gen = torch.Generator().manual_seed(0)
        M, E = 6, 64
        lg = (torch.randn(M, E, generator=gen) * 2).to(dtype)
        bias = torch.linspace(-0.05, 0.05, E)
        pols = [(4, 0, True, None, 0, 0, 1.0), (6, 1, True, bias, 8, 3, 2.5)]
        for pol in pols:
            k = pol[0]
            w, ids = _reference_route(lg.float(), *pol)
            for case in ("clean", "expert", "weight", "inactive"):
                ww, ii = w.clone(), ids.clone()
                if case == "expert":
                    ii[:, 0] = lg.float().argmin(-1).to(ii.dtype)      # each row's lowest-scoring expert
                if case == "weight":
                    ww = ww * 1.25
                res = []
                for fn in (route_check_torch, route_check_triton):
                    st = torch.zeros(1, COLS, dtype=torch.int64, device=device)
                    sb = torch.full((1,), 3 if case != "inactive" else 4, dtype=torch.int64, device=device)
                    rb = torch.full((1,), M, dtype=torch.int64, device=device)
                    b = pol[3].to(device) if pol[3] is not None else None
                    caps = new_caps(device, 1, E)
                    fn(lg.to(device), ww.to(device), ii.to(device), k, pol[1], pol[2], b, pol[4], pol[5], pol[6], rb,
                       sb, st, 0, 4, 3, *caps)
                    res.append((st.cpu(), [c.cpu() for c in caps]))
                (s0, c0), (s1, c1) = res
                if not all(torch.equal(a, b) for a, b in zip(c0[:1] + c0[2:], c1[:1] + c1[2:])) or \
                        not torch.allclose(c0[1], c1[1]):
                    raise AssertionError(f"{case}: the kept rows differ")
                cnt = [C_CALLS, C_SET, C_WEIGHT, C_ROWS, C_LAST]
                if [int(s0[0, c]) for c in cnt] != [int(s1[0, c]) for c in cnt]:
                    raise AssertionError(f"{case}: counts differ ({s0[0].tolist()} vs {s1[0].tolist()})")
                if case == "inactive" and (s0.abs().sum() or s1.abs().sum()):
                    raise AssertionError("an inactive step changed the state")
                if case == "clean" and (s0[0, C_SET] or s0[0, C_WEIGHT] or s0[0, C_ROWS] != SAMPLES):
                    raise AssertionError(f"correct routing wasn't quiet ({s0[0].tolist()})")
                if case == "expert" and s0[0, C_SET] != SAMPLES:
                    raise AssertionError(f"a moved expert wasn't found ({s0[0].tolist()})")
                if case == "weight" and s0[0, C_WEIGHT] != SAMPLES:
                    raise AssertionError(f"a wrong weight wasn't found ({s0[0].tolist()})")
        STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    return STATUS[key]


def _reference_route(logits: torch.Tensor, k: int, scoring: int, renormalize: bool, bias, groups: int,
                     topk_group: int, scale: float) -> tuple:
    """Routing by the policy for every row (for the first-use comparison and tests): (weights fp32, ids int32)."""
    s = torch.softmax(logits.float(), -1) if scoring == 0 else torch.sigmoid(logits.float())
    ch = s + bias.float().reshape(1, -1) if bias is not None else s
    if groups:
        r, e = ch.shape
        g = ch.view(r, groups, e // groups)
        gs = g.topk(2, -1).values.sum(-1) if bias is not None else g.amax(-1)
        keep = torch.zeros_like(gs).scatter_(1, gs.topk(topk_group, -1).indices, 1.0)
        ch = ch.masked_fill(~(keep.unsqueeze(-1).expand(r, groups, e // groups).reshape(r, e) > 0), float("-inf"))
    ids = ch.topk(k, -1).indices
    w = s.gather(1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return (w * scale).float(), ids.to(torch.int32)


def route_check_any(*args) -> None:
    """The check: the Triton kernel where it passed its first-use comparison, else the PyTorch one."""
    logits = args[0]
    with torch.no_grad(), torch.autocast(device_type=logits.device.type, enabled=False):
        key = (_dev_key(logits.device), logits.dtype)
        if key not in STATUS and kernel_usable(logits.device) and not torch.cuda.is_current_stream_capturing():
            verify_kernel(logits.device, logits.dtype)
        if STATUS.get(key, (False,))[0]:
            try:
                route_check_triton(*args)
                return
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001
                STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
                FALLBACKS[0] += 1
        route_check_torch(*args)


_lib_lock = threading.Lock()
_OP = [None]


def route_op():
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::route_check",
                                         mutates_args=("state", "cap_meta", "cap_lg", "cap_ids", "cap_w"))
                def route_check(logits: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor, k: int,
                                scoring: int, renormalize: bool, bias: Optional[torch.Tensor], groups: int,
                                topk_group: int, scale: float, rows_buf: torch.Tensor, stepbuf: torch.Tensor,
                                state: torch.Tensor, slot: int, nlayers: int, index: int, cap_meta: torch.Tensor,
                                cap_lg: torch.Tensor, cap_ids: torch.Tensor, cap_w: torch.Tensor) -> None:
                    route_check_any(logits, topk_w, topk_ids, k, scoring, renormalize, bias, groups, topk_group,
                                    scale, rows_buf, stepbuf, state, slot, nlayers, index, cap_meta, cap_lg, cap_ids,
                                    cap_w)

                @route_check.register_fake
                def _(*a):
                    return None

                _OP[0] = torch.ops.sdcguard.route_check
            except (AttributeError, RuntimeError):
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "route_check", None) or route_check_any
        return _OP[0]


def check_call(args: tuple, router_logits, topk_weights, topk_ids) -> None:
    if not all(isinstance(t, torch.Tensor) for t in (router_logits, topk_weights, topk_ids)) or \
            router_logits.dim() != 2 or topk_ids.dim() != 2 or topk_ids.shape[-1] != args[0]:
        return
    call = (router_logits, topk_weights, topk_ids, *args)
    if torch.compiler.is_compiling():
        (_OP[0] if _OP[0] is not None else route_op())(*call)
    else:
        route_check_any(*call)


def policy_of(router) -> tuple:
    """(policy dict, why not): the routing policy of a vLLM router."""
    name = type(router).__name__
    if name not in ("FusedTopKRouter", "GroupedTopKRouter", "FusedTopKBiasRouter"):
        return None, f"the {name} router (only vLLM's top-k, grouped top-k and biased top-k routers are checked)"
    sc = str(getattr(router, "scoring_func", "softmax"))
    if sc not in SCORING:
        return None, f"scoring function {sc}"
    if getattr(router, "eplb_state", None) is not None:
        return None, "expert load balancing (the router returns physical expert ids)"
    if any(getattr(router, a, None) is not None for a in ("_hash_indices_table", "bias_vl")) or \
            int(getattr(router, "num_fused_shared_experts", 0) or 0) > 0:
        return None, "hash routing, image-token biases or fused shared experts"
    k = int(getattr(router, "top_k", 0) or 0)
    e = int(getattr(router, "global_num_experts", 0) or 0)
    if not (0 < k <= MAX_K) or (e and (e > MAX_EXPERTS or k > e)):
        return None, f"top-{k} of {e} experts (up to top-{MAX_K} of {MAX_EXPERTS} are checked)"
    bias = getattr(router, "e_score_correction_bias", None)
    bias = getattr(bias, "data", bias)
    bias = bias if isinstance(bias, torch.Tensor) else None
    scale = float(getattr(router, "routed_scaling_factor", 1.0) or 1.0) if name != "FusedTopKRouter" else 1.0
    groups = int(getattr(router, "num_expert_group", 0) or 0) if name == "GroupedTopKRouter" else 0
    topk_group = int(getattr(router, "topk_group", 0) or 0) if groups else 0
    if groups:
        if not (e > groups and e % groups == 0):
            # vLLM's GroupedTopKRouter without a valid grouping: plain top-k, and without a bias that is its fused
            # top-k with the default softmax and no scaling, whatever the router's own settings
            groups, topk_group = 0, 0
            if bias is None:
                sc, scale = "softmax", 1.0
        elif groups > MAX_GROUPS or not (0 < topk_group <= groups) or topk_group * (e // groups) < k:
            return None, f"{groups} expert groups choosing {topk_group}"
    return {"k": k, "scoring": SCORING[sc], "renormalize": bool(getattr(router, "renormalize", True)), "bias": bias,
            "groups": groups, "topk_group": topk_group, "scale": scale}, None


class RouteChecks:
    """The MoE routers one ServeGuard checks (ServeGuard.route_checks()): add() each at load and finalize() after the
    last, mark_step(step, rows) before each model step and end_step() after; poll() with the guard's reads."""

    def __init__(self, guard):
        self.guard = guard
        self.layers: list = []
        self.unsupported: dict = {}
        self._state, self._rows, self._step, self._host, self._caps = {}, {}, {}, {}, {}
        self.stats = {"layers": 0, "calls": 0, "rows": 0, "wrong_sets": 0, "wrong_weights": 0, "alerts": 0}
        self.seen: dict = {}

    def add(self, router, name: str, device) -> Optional[tuple]:
        pol, why = policy_of(router)
        if why:
            self.unsupported[name] = why
            return None
        if not (device.type == "cuda" or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
            self.unsupported[name] = "not on a GPU"
            return None
        if device not in self._state:
            with torch.inference_mode(False):
                self._state[device] = torch.zeros(256, COLS, dtype=torch.int64, device=device)
                self._rows[device] = torch.zeros(1, dtype=torch.int64, device=device)
                self._step[device] = torch.zeros(1, dtype=torch.int64, device=device)
                self._caps[device] = new_caps(device)
        slot = sum(1 for lay in self.layers if lay["device"] == device)
        if slot >= self._state[device].shape[0]:
            self.unsupported[name] = "more than 256 routers on one GPU"
            return None
        if pol["bias"] is not None and pol["bias"].device != device:
            pol["bias"] = pol["bias"].to(device)
        lay = {"name": name, "router": router, "device": device, "slot": slot, "policy": pol}
        self.layers.append(lay)
        self.stats["layers"] += 1
        route_op()
        return self._arm(lay)

    def _arm(self, lay) -> tuple:
        pol, dev = lay["policy"], lay["device"]
        n = sum(1 for x in self.layers if x["device"] == dev)
        a = (pol["k"], pol["scoring"], pol["renormalize"], pol["bias"], pol["groups"], pol["topk_group"],
             pol["scale"], self._rows[dev], self._step[dev], self._state[dev], lay["slot"], n, lay["slot"],
             *self._caps[dev])
        lay["router"]._sdcguard_route = a
        return a

    def finalize(self) -> None:
        """After the last add(): each router's turn among the routers on its GPU."""
        for lay in self.layers:
            self._arm(lay)

    def mark_step(self, step: int, rows: Optional[int] = None) -> None:
        for dev in self._state:
            self._step[dev].fill_(int(step) & M32)
            self._rows[dev].fill_(int(rows) if rows is not None else 0)

    def end_step(self) -> None:
        for dev in self._state:
            self._rows[dev].zero_()

    def detach(self) -> None:
        try:
            self.poll(wait=True)
        finally:
            for lay in self.layers:
                lay["router"].__dict__.pop("_sdcguard_route", None)
            self.layers.clear()
            for d in (self._state, self._rows, self._step, self._host, self._caps):
                d.clear()

    def coverage(self) -> dict:
        kern = {f"{d} {str(t).replace('torch.', '')}": ("Triton" if ok else f"PyTorch ({why})")
                for (d, t), (ok, why) in STATUS.items()}
        return {"routers": len(self.layers), "calls": self.stats["calls"], "rows": self.stats["rows"],
                "routers_checked": sum(self.seen.get(lay["name"], 0) > 0 for lay in self.layers),
                "not_attached": dict(self.unsupported), "samples_per_call": SAMPLES, "one_router_a_step": True,
                "kernel": kern}

    def poll(self, wait: bool = False) -> list:
        events = []
        for dev, st in list(self._state.items()):
            if dev.type != "cuda" or wait:
                pend = self._host.pop(dev, None)
                if pend is not None:
                    pend[1].synchronize()
                    events += self._report(dev, pend[0])
                host = st.to("cpu", copy=True)
                st.zero_()
                events += self._report(dev, host)
                continue
            pend = self._host.get(dev)
            if pend is not None and pend[1].query():
                events += self._report(dev, pend[0])
                self._host.pop(dev, None)
                pend = None
            if pend is None:
                hs = torch.empty(st.shape, dtype=st.dtype, pin_memory=True)
                hs.copy_(st, non_blocking=True)
                st.zero_()
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (hs, ev)
        return events

    def _report(self, dev, host) -> list:
        out = []
        for lay in self.layers:
            if lay["device"] != dev:
                continue
            k = [int(v) for v in host[lay["slot"]].tolist()]
            self.stats["calls"] += k[C_CALLS]
            self.stats["rows"] += k[C_ROWS]
            self.seen[lay["name"]] = self.seen.get(lay["name"], 0) + k[C_CALLS]
            self.stats["wrong_sets"] += k[C_SET]
            self.stats["wrong_weights"] += k[C_WEIGHT]
            if k[C_SET] or k[C_WEIGHT]:
                out.append(self._alert(dev, lay, k))
        return out

    def _recheck(self, dev, lay) -> dict:
        """The router's kept row (its first bad row since the last read), judged again in float64 on the CPU; the
        slot is then free for the next one."""
        caps = self._caps.get(dev)
        if caps is None:
            return {"verdict": "not_rechecked", "why": "no kept row"}
        slot = lay["slot"]
        meta = [int(v) for v in caps[0][slot].tolist()]
        if not meta[0]:
            return {"verdict": "not_rechecked", "why": "no row was kept (another check kept one first, or it was read "
                                                       "with an earlier alert)"}
        e, k = meta[4], lay["policy"]["k"]
        try:
            out = recheck(caps[1][slot, :e].cpu(), caps[2][slot, :k].cpu(), caps[3][slot, :k].cpu(), lay["policy"],
                          CODE_DTYPE.get(meta[5], torch.float32))
            out["captured"] = {"row": meta[1], "step": meta[2], "kind": "experts" if meta[3] == KIND_SET else "weights"}
        except Exception as ex:  # noqa: BLE001
            out = {"verdict": "not_rechecked", "why": f"{type(ex).__name__}: {ex}"}
        caps[0][slot].zero_()
        return out

    def _alert(self, dev, lay, k) -> SDCEvent:
        g = self.guard
        last = k[C_LAST]
        step, kind, row = (last >> 32) & M32, (last >> 24) & 0xFF, last & 0xFFFFFF
        finding = "wrong_experts" if kind == KIND_SET else "wrong_weights"
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        if str(dev) not in ident:
            ident[str(dev)] = device_identity(dev)
        pol = {kk: v for kk, v in lay["policy"].items() if kk != "bias"}
        pol["bias"] = lay["policy"]["bias"] is not None
        detail = {"where": lay["name"], "kind": "moe_routing", "finding": finding, "rows_judged": k[C_ROWS],
                  "rows_with_wrong_experts": k[C_SET], "rows_with_wrong_weights": k[C_WEIGHT],
                  "worst_weight_difference": _f32(k[C_WDIFF]) if k[C_WDIFF] else 0.0,
                  "captured": {"step": step, "row": row}, "policy": pol,
                  "reading": ("the router chose experts its policy doesn't give for these logits"
                              if finding == "wrong_experts" else
                              "the router's weights for its experts differ from its policy's")}
        detail["recheck"] = self._recheck(dev, lay)
        ev = SDCEvent(detector="serve_route", step=None, detail=detail, identity=dict(ident[str(dev)]))
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
