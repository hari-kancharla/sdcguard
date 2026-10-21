"""Detector for inference (serving): check every bf16/fp16 linear layer on every forward pass.

While a model serves, its weights don't change. So each layer's checksums (the block sums of its
weight matrix, the B side of the training check) are computed once, when the guard is attached,
and every forward pass only has to read the layer's input x and output y once more:

    expected = x @ (block sums of W)   (a skinny matmul: 2 columns per 128 outputs, as three bf16/fp16
                                        parts on NVIDIA GPUs, so 6/128 = about 4.7% of the layer's work)
    actual   = block sums of y         (one pass over y)

A mismatch beyond the estimated rounding noise reports a numerical disagreement. Causes to investigate
include incorrect computation, changed weights, and a checker or numerical-model problem. verify_weights()
compares weight fingerprints with their load-time values; it does not establish the cause by itself.

Same math and noise model as the training check (checksum.py): two checksums per 128-output
block (plain and fixed random +-1 signs), alarm at z > 6. One difference: a block the check can't judge
(inputs that aren't finite, or a legitimate overflow) scores 0 here, because the per-layer state only holds
non-negative scores. Separate execution and unjudged-checksum counts distinguish those cases from completed
clean checks and from layers that never ran.

Layers whose code might compute something other than x @ W^T + b are left alone and listed in
guard.unsupported: a subclass with its own forward, and a forward set on the instance. The one exception is
accelerate's device hook (dispatch_model, from_pretrained(device_map=...)), which keeps nn.Linear.forward as
_old_forward and calls it after moving the inputs to the weights' device: the check runs there, underneath the
hook. Layers whose weights are offloaded (on the meta device when attached) are listed too.

Built for serving engines:
  - The check's forward path does not explicitly wait for the GPU. Results go into a small state tensor
    (worst z per layer and execution/unjudged counts); poll(wait=False) reads it back asynchronously.
    Explicit snapshots, synchronous polls and detach() wait for queued work.
  - The check is a PyTorch custom op (sdcguard::check_static), so torch.compile treats it as one
    opaque call and CUDA graphs capture it like any other kernel.
  - On NVIDIA GPUs it costs one cuBLAS call and one Triton kernel per layer (_serve_fast.py), once
    that kernel has given this module's answers on a set of test layers; otherwise the PyTorch
    version here (about 40 small operations per layer) runs.

Usage (Hugging Face transformers or any nn.Module):
    guard = ServeGuard().attach(model)          # model already in bf16/fp16, on its device
    out = model.generate(...)
    for event in guard.poll():                  # call now and then, e.g. after each request
        print(event.to_json())
"""
from __future__ import annotations

import math
import threading
import weakref
from typing import Callable, Optional

import torch

from .checksum import ABS_FLOOR, BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF, _signs, _split3, l1_floor_coef
from .events import EventLog, SDCEvent, device_identity

# Per layer, two int64 keys: (bits of the worst z << 32) | flat index of that checksum, and
# (bits of the worst z << 32) | bits of its noise (NaN when that checksum of the output was Inf/NaN). z >= 0
# and the bits of non-negative floats sort like the floats, so the largest key holds the worst z (an integer
# maximum, also usable as an atomic).
# Then two counts since the last poll: checks executed (calls of the check for that layer, empty batches
# excluded) and checksums
# that couldn't be judged (Inf/NaN the inputs explain: score 0, but not evidence of a clean result). A worst
# score of 0 alone can't tell a clean layer from one never checked; the counts can. State tensors with only
# the two key columns (older callers, tests) still work: the counts are then skipped.
STATE_COLS = 4
COL_CHECKS, COL_UNJUDGED = 2, 3

# Original-event capture. The first time since the last read that a check of a layer scores above the alarm line,
# the check itself copies the row of x and the block of y it judged into a small buffer on the device, before the
# engine reuses that memory (the next replay of a CUDA graph overwrites its activations): the alert then carries
# the numbers that alarmed, not a recomputation that a one-time fault isn't in. Up to CAPTURE_SLOTS rows per device
# between reads, one per layer; rows that didn't fit are counted. No host sync and no data-dependent host logic, so
# it is safe in CUDA graphs; the Triton check copies only when a row is over the line (nothing on clean traffic).
#   cap_meta (int64): rows claimed since the last read, the read epoch, CAPTURE_SLOTS + 1 records of CAP_REC (the
#     last is where the PyTorch check writes when there is nothing to keep, so it needs no branch), then one flag
#     per layer. A record: tag ((layer slot + 1) x 2, + 1 when the layer's bias was in the matmul), row, column
#     block, flat position of the checksum, bits of its z, epoch.
#   cap_data (fp32, CAPTURE_SLOTS + 1 rows): the block of y (cap_w values), then the row of x (k values); bf16 and
#     fp16 values are exact in fp32.
CAPTURE_SLOTS = 4
CAP_REC = 6
CAP_HEAD = 2
CAP_FLAGS = CAP_HEAD + (CAPTURE_SLOTS + 1) * CAP_REC
OP_SCHEMA = 2           # the check op's arguments (vllm_plugin keys its compile cache on it); 2: capture buffers


_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def select_layer(i: int, share: float, offset: int = 0) -> bool:
    """Is linear layer number i (in model order) one of the checked share? Golden-ratio spacing: a
    share spreads over every kind of layer and every depth (every 4th layer would pick the same
    projection in every transformer block). Layer i sits at the point i x 0.618... (mod 1) and is taken
    when that point falls in a window of width `share` starting at offset x share: with share 1/R, offsets
    0 to R-1 split the layers between R replicas, every layer checked by exactly one. (Until Oct 19, 2026
    the offset shifted the layer number instead, and offsets 0 to 7 at share 0.125 left 10% of the layers
    unchecked everywhere; an independent review found it.)"""
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    return ((i * _PHI) % 1.0 - offset * share) % 1.0 < share


def _static(t: torch.Tensor) -> None:
    """Tell torch.compile this tensor keeps its memory address (like a KV cache): CUDA graphs then
    read and update it in place instead of copying it in on every replay."""
    try:
        torch._dynamo.mark_static_address(t)
    except Exception:  # noqa: BLE001  (older PyTorch: graphs still work, with a copy)
        pass


def _lowp_ok(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype in UNIT_ROUNDOFF


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b with an fp32 result. On NVIDIA GPUs, low-precision inputs with fp32 accumulation and
    output (one cuBLAS call, a never copied to fp32); elsewhere in fp32."""
    if _lowp_ok(a) and b.dtype == a.dtype:
        return torch.ops.aten.mm.dtype(a, b, torch.float32)
    return a.float() @ b.float()


class LayerChecksums:
    """Everything the check needs about one weight matrix W (n outputs x k inputs), computed once."""

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], block: int = 128):
        # A caller may attach or refresh under autocast. The cached reference must still be computed in
        # its declared precision; otherwise even healthy outputs can fail, particularly with bias cancellation.
        with torch.autocast(device_type=weight.device.type, enabled=False):
            self._initialize(weight, bias, block)

    def _initialize(self, weight: torch.Tensor, bias: Optional[torch.Tensor], block: int) -> None:
        # (every tensor here is created with an explicit dtype: vLLM processes weights under a bf16
        # default dtype, set_default_torch_dtype in its model loader)
        W = weight.detach()
        n, k = W.shape
        dt = W.dtype
        dev = W.device
        bs = max(1, min(block, n))
        nbf, rem = n // bs, n % bs
        nb = nbf + (1 if rem else 0)
        self.n, self.k, self.nb, self.bs, self.nbf, self.rem, self.dtype = n, k, nb, bs, nbf, rem, dt
        self.has_bias = bias is not None
        lowp = dev.type == "cuda"
        # [1, s] per block width (s = the fixed +-1 pattern the training check uses for that width).
        # Exact in bf16/fp16, so on NVIDIA GPUs the block sums of y come from skinny matmuls.
        self.pair_full = self._pair(bs, dt if lowp else torch.float32, dev)
        self.pair_rem = self._pair(rem, dt if lowp else torch.float32, dev) if rem else self.pair_full
        Wf = W.float()
        # B2 (k x 2nb): column 2b = sum of block b's weight rows, column 2b+1 = their signed sum
        parts = []
        if nbf:
            Wb = Wf[:nbf * bs].view(nbf, bs, k)
            parts.append(torch.einsum("bjk,jc->kbc", Wb, self._pair(bs, torch.float32, dev)).reshape(k, 2 * nbf))
        if rem:
            parts.append(torch.einsum("jk,jc->kc", Wf[nbf * bs:], self._pair(rem, torch.float32, dev)))
        B2 = torch.cat(parts, dim=1).contiguous()                               # (k, 2nb) fp32
        self.b2n = torch.linalg.vector_norm(B2, dim=0).view(nb, 2)              # |checksum column|
        self.b2max = torch.linalg.vector_norm(B2, ord=float("inf"), dim=0).view(nb, 2)    # its largest |entry|
        # fp16 tops out at 65504: scale each checksum column by a power of two (exact) first
        if dt == torch.float16:
            amax = B2.abs().amax(dim=0)
            ok = torch.isfinite(amax) & (amax > 0)
            scale = torch.where(ok, torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30)))), 1.0)
        else:
            scale = torch.ones(2 * nb, device=dev, dtype=torch.float32)
        self.scale = scale.view(nb, 2)
        # tensor cores read B2 as three low-precision parts that add up to it (to about 2^-27)
        self.parts = _split3(B2 / scale, dt).contiguous() if lowp else B2
        rows_max = Wf.abs().amax(dim=1)                                         # (n,)
        self.bmax = self._block_reduce(rows_max, nbf, bs, rem, "amax")
        if bias is not None:
            bf = bias.detach().float()
            bp = []
            if nbf:
                bp.append(bf[:nbf * bs].view(nbf, bs) @ self._pair(bs, torch.float32, dev))
            if rem:
                bp.append((bf[nbf * bs:] @ self._pair(rem, torch.float32, dev)).view(1, 2))
            self.bsum = torch.cat(bp, dim=0)                                    # (nb, 2)
            self.bmax_bias = self._block_reduce(bf.abs(), nbf, bs, rem, "amax")
        else:
            self.bsum = torch.zeros(nb, 2, device=dev, dtype=torch.float32)
            self.bmax_bias = torch.zeros(nb, device=dev, dtype=torch.float32)
        self._z1 = torch.zeros(nb, device=dev, dtype=torch.float32)
        self._z2 = torch.zeros(nb, 2, device=dev, dtype=torch.float32)
        width = torch.full((nb,), float(bs), device=dev, dtype=torch.float32)
        if rem:
            width[-1] = float(rem)
        self.width = width
        self.wsq = width.sqrt()
        self.fingerprint = self._fingerprint(W)
        for t in self.tensors(True) + self.tensors(False):
            _static(t)

    @staticmethod
    def _pair(width: int, dtype: torch.dtype, dev) -> torch.Tensor:
        s = _signs(width, dev)
        return torch.stack([torch.ones_like(s), s], dim=1).to(dtype)

    @staticmethod
    def _block_reduce(v: torch.Tensor, nbf: int, bs: int, rem: int, how: str) -> torch.Tensor:
        out = []
        if nbf:
            out.append(v[:nbf * bs].view(nbf, bs).amax(dim=1))
        if rem:
            out.append(v[nbf * bs:].amax().view(1))
        return torch.cat(out)

    @staticmethod
    def _fingerprint(W: torch.Tensor) -> torch.Tensor:
        """Two sums of the weights' raw bits (plain, and weighted by position): any bit flip changes
        at least one of them. Computed in chunks, so it needs little extra memory."""
        bits = W.contiguous().view(-1).view(torch.int16 if W.element_size() == 2 else torch.int32)
        tot = torch.zeros(2, dtype=torch.int64, device=W.device)
        chunk = 1 << 24
        for i in range(0, bits.numel(), chunk):
            b = bits[i:i + chunk].to(torch.int64)
            pos = torch.arange(i, i + b.numel(), device=W.device, dtype=torch.int64) % 65521 + 1
            tot += torch.stack([b.sum(), (b * pos).sum()])
        return tot

    def tensors(self, with_bias: bool = True):
        """Arguments for the check op, after x2 and y2. with_bias=False: the layer ran without adding
        its bias this time (vLLM does that on some tensor-parallel ranks)."""
        bmb, bs = (self.bmax_bias, self.bsum) if with_bias else (self._z1, self._z2)
        return (self.parts, self.pair_full, self.pair_rem, self.scale, self.b2n, self.b2max, self.bmax, bmb, bs,
                self.wsq, self.width)


def _block_stats(y2: torch.Tensor, pair: torch.Tensor, start: int, width: int, nblk: int):
    """Plain and signed sums (fp32), sum of squares and sum of |values| of every block of one span."""
    m = y2.shape[0]
    yb = y2[:, start:start + width * nblk]
    yb = (yb if yb.is_contiguous() else yb.contiguous()).view(m * nblk, width)
    sums = _mm_fp32(yb, pair).view(m, nblk, 2)
    yf = yb.float()
    return sums, yf.square().sum(-1).view(m, nblk), yf.abs().sum(-1).view(m, nblk)


def check_static_impl(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                      pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor,
                      bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
                      width: torch.Tensor,
                      state: torch.Tensor, slot: int, z: float, abs_floor: float,
                      cap_meta: Optional[torch.Tensor] = None, cap_data: Optional[torch.Tensor] = None,
                      cap_w: int = 0, cap_tag: int = 0) -> None:
    """Check y2 = x2 @ W^T (+ bias) for one layer and fold the worst z into state[slot]; with cap_meta and
    cap_data, keep the row that alarmed (see CAPTURE_SLOTS).

    No host sync and no data-dependent Python control flow, so it is safe inside CUDA graphs."""
    with torch.autocast(device_type=x2.device.type, enabled=False):
        _check_static_impl(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum,
                           wsq, width, state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag)


def _capture(x2: torch.Tensor, y2: torch.Tensor, mz: torch.Tensor, arg: torch.Tensor, zbits: torch.Tensor,
             nb: int, bs: int, n: int, cap_meta: torch.Tensor, cap_data: torch.Tensor, cap_w: int, cap_tag: int,
             slot: int, z: float) -> None:
    """The PyTorch check's original-event capture (see CAPTURE_SLOTS), without a branch: when this call's worst
    score is over the line and the layer hasn't kept a row since the last read, the row and block it came from go
    into the next free record; otherwise the same writes go to the spare last record."""
    m, k = x2.shape
    S = CAPTURE_SLOTS
    fo = CAP_FLAGS + slot
    hit = mz > z
    old = cap_meta[fo].clone()
    first = hit & (old == 0)
    cap_meta[fo] = torch.where(hit, torch.ones_like(old), old)
    c = cap_meta[0].clone()
    cap_meta[0] = c + first.to(torch.int64)
    idx = torch.where(first & (c < S), c, torch.full_like(c, S)).view(1)
    bi = torch.div(arg, 2, rounding_mode="floor")
    blk = torch.remainder(bi, nb)
    row = torch.div(bi, nb, rounding_mode="floor").clamp(0, m - 1)
    cols = blk * bs + torch.arange(bs, device=y2.device)
    inside = (cols < n).view(1, bs)
    yrow = y2.index_select(0, row.view(1)).gather(1, cols.clamp(max=n - 1).view(1, bs)).float()
    cap_data[:, :bs].index_copy_(0, idx, torch.where(inside, yrow, torch.zeros_like(yrow)))
    cap_data[:, cap_w:cap_w + k].index_copy_(0, idx, x2.index_select(0, row.view(1)).float())
    rec = torch.stack([torch.full_like(c, cap_tag), row, blk, arg.to(torch.int64), zbits, cap_meta[1].clone()])
    cap_meta[CAP_HEAD:CAP_FLAGS].view(S + 1, CAP_REC).index_copy_(0, idx, rec.view(1, CAP_REC))


def _check_static_impl(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                       pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor,
                       bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
                       width: torch.Tensor,
                       state: torch.Tensor, slot: int, z: float, abs_floor: float,
                       cap_meta: Optional[torch.Tensor] = None, cap_data: Optional[torch.Tensor] = None,
                       cap_w: int = 0, cap_tag: int = 0) -> None:
    dt = y2.dtype
    u = UNIT_ROUNDOFF[dt]
    h = HALF_TINY_STEP[dt]
    m, n = y2.shape
    nb = b2n.shape[0]
    if x2.dtype != dt:
        x2 = x2.to(dt)
    bs = pair_full.shape[0]
    nbf = n // bs
    rem = n - nbf * bs
    # expected = x @ B2 (+ bias sums)
    split = parts.shape[1] == 6 * nb                    # three low-precision parts (NVIDIA) or plain fp32
    if split and parts.dtype == dt and _lowp_ok(x2):
        e = torch.ops.aten.mm.dtype(x2, parts, torch.float32)                  # (m, 3 * 2nb)
        expected = e.view(m, 3, 2 * nb).sum(1).view(m, nb, 2) * scale
    elif split:
        expected = (x2.float() @ parts.float()).view(m, 3, 2 * nb).sum(1).view(m, nb, 2) * scale
    else:
        expected = (x2.float() @ parts.float()).view(m, nb, 2)
    expected = expected + bsum
    stats = []
    if nbf:
        stats.append(_block_stats(y2, pair_full, 0, bs, nbf))
    if rem:
        stats.append(_block_stats(y2, pair_rem, nbf * bs, rem, 1))
    actual = torch.cat([s[0] for s in stats], dim=1)                            # (m, nb, 2)
    sq = torch.cat([s[1] for s in stats], dim=1)                                # (m, nb)
    l1 = torch.cat([s[2] for s in stats], dim=1)
    xn = torch.linalg.vector_norm(x2, dim=1, dtype=torch.float32).view(m, 1, 1)
    xl1 = torch.linalg.vector_norm(x2, ord=1, dim=1, dtype=torch.float32).view(m, 1)
    bound = (xl1 * bmax + bmax_bias).unsqueeze(-1)                              # (m, nb, 1)
    noise = (sq.sqrt() * u).unsqueeze(-1)                                       # (m, nb, 1)
    # fp32 rounding of `expected`: the smaller of two bounds of it (checksum._zscores explains)
    f_l2 = xn * b2n * 1e-5                                                      # (m, nb, 2)
    f_l1 = xl1.unsqueeze(-1) * b2max * l1_floor_coef(x2.shape[1])
    inf = float("inf")
    floor = torch.minimum(torch.where(torch.isfinite(f_l2), f_l2, inf), torch.where(torch.isfinite(f_l1), f_l1, inf))
    floor = torch.where(torch.isfinite(floor), floor, bound * (1e-5 * 128))
    den = (noise + floor) + (abs_floor + wsq.view(1, nb, 1) * h)
    diff = (expected - actual).abs()
    zs = diff / den
    worst = (l1 * (1.25 * u) + 1.25 * h * width).unsqueeze(-1) + floor[..., :1] + abs_floor
    zs = torch.cat([torch.minimum(zs[..., :1], diff[..., :1] / worst * z), zs[..., 1:]], dim=-1)
    # Inf/NaN, judged against what the inputs allow (see checksum.py). Blocks that can't be judged score 0
    # here, not -1 as in training: the per-layer state only holds non-negative scores.
    e_bad = ~torch.isfinite(expected)
    a_bad = ~torch.isfinite(actual) & ~e_bad
    big = ~torch.isfinite(noise) & ~e_bad & ~a_bad
    lim = bound * 1.01
    inf = torch.full_like(zs, float("inf"))
    zero = torch.zeros_like(zs)
    nan = zs != zs                              # inf/inf (both sides huge): nothing to judge
    zs = torch.where(zs == zs, zs, zero)
    zs = torch.where(e_bad, zero, zs)
    # finite outputs too large for the inputs (checksum._judge_nonfinite explains)
    cap = bound * (1.02 + x2.shape[1] * 2.0 ** -22) + 2 * h
    imposs = ~e_bad & ~a_bad & ~big & (lim < BIG) & (noise / u > cap * wsq.view(1, nb, 1))
    zs = torch.where(imposs, inf, zs)
    lim_a = min(DTYPE_MAX[dt], BIG)
    zs = torch.where(a_bad, torch.where(lim < lim_a, inf, zero), zs)
    zs = torch.where(big, torch.where(lim < BIG, inf, zero), zs)
    # checksums that couldn't be judged (they score 0 above; training scores them UNJUDGED, see
    # checksum._judge_nonfinite): nothing to compare, the expected side isn't finite, or a non-finite or
    # too-large output the inputs can explain
    unjudged = ((nan & ~imposs & ~a_bad & ~big) | e_bad | (a_bad & ~(lim < lim_a)) | (big & ~(lim < BIG))).sum()
    # one extra 0 at the end: an empty batch (m = 0) then scores 0 instead of failing the reduction, with no
    # data-dependent branch (this also runs traced into vLLM's compiled graph)
    flat = torch.cat([zs.reshape(-1), zs.new_zeros(1)])
    mz, arg = torch.max(flat, 0)
    # the noise kept with the score is NaN where the output's checksum itself was Inf/NaN, so the report can tell
    # an Inf/NaN output from finite values too large for the inputs (both score inf) without another read
    den = torch.where(a_bad, torch.full_like(zs, float("nan")), den.expand_as(zs))
    nz = torch.cat([den.reshape(-1), den.new_ones(1)]).gather(0, arg.view(1)).squeeze(0)
    zraw = mz.float().clamp_min(0).view(torch.int32).to(torch.int64)
    zb = zraw << 32
    keys = torch.stack([zb | arg.to(torch.int64), zb | (nz.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF)])
    state[slot, :2] = torch.maximum(state[slot, :2], keys)
    if state.shape[1] >= STATE_COLS:            # (a shape, not a value: no host sync, safe in CUDA graphs)
        # one check executed, unless the batch was empty (counted without a branch on the batch size)
        executed = torch.ones_like(y2[:, 0], dtype=torch.int64).sum().clamp(max=1)
        state[slot, COL_CHECKS:COL_UNJUDGED + 1] += torch.stack([executed, unjudged.to(torch.int64)])
    if cap_meta is not None and cap_data is not None and m > 0:
        _capture(x2, y2, mz, arg, zraw, nb, bs, n, cap_meta, cap_data, cap_w, cap_tag, slot, z)


def decode_state(rows: torch.Tensor):
    """(z, flat index, noise) per row of a state tensor (on the CPU)."""
    zs = (rows[:, 0] >> 32).to(torch.int32).view(torch.float32).tolist()
    flats = (rows[:, 0] & 0xFFFFFFFF).tolist()
    nzs = (rows[:, 1] & 0xFFFFFFFF).to(torch.int32).view(torch.float32).tolist()
    return list(zip(zs, flats, nzs))


def decode_counts(rows: torch.Tensor):
    """(checks executed, unjudged checksums) per row of a state tensor (on the CPU); (0, 0) for a state without
    the count columns."""
    if rows.dim() != 2 or rows.shape[1] < STATE_COLS:
        return [(0, 0)] * rows.shape[0]
    return list(zip(rows[:, COL_CHECKS].tolist(), rows[:, COL_UNJUDGED].tolist()))


def check_static_any(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                     pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor,
                     bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
                     width: torch.Tensor,
                     state: torch.Tensor, slot: int, z: float, abs_floor: float,
                     cap_meta: Optional[torch.Tensor] = None, cap_data: Optional[torch.Tensor] = None,
                     cap_w: int = 0, cap_tag: int = 0) -> None:
    """The Triton check where it has passed its first-use comparison, else the PyTorch one."""
    with torch.autocast(device_type=x2.device.type, enabled=False):
        _check_static_any(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum,
                          wsq, width, state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag)


def _check_static_any(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                      pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor,
                      bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
                      width: torch.Tensor,
                      state: torch.Tensor, slot: int, z: float, abs_floor: float,
                      cap_meta: Optional[torch.Tensor] = None, cap_data: Optional[torch.Tensor] = None,
                      cap_w: int = 0, cap_tag: int = 0) -> None:
    if x2.shape[0] == 0:                       # empty batch: nothing to check
        return
    from . import _group, _serve_fast
    # (its sign table holds 128; the kernel is handed addresses: ordinary tensors only, _group.own_memory)
    if pair_full.shape[0] <= 128 and _serve_fast.ready(y2.device, y2.dtype) and _group.own_memory(x2, y2):
        try:
            _serve_fast.check(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq,
                              width, state, slot, z, abs_floor, cap_meta=cap_meta, cap_data=cap_data,
                              cap_w=cap_w, cap_tag=cap_tag)
            return
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001  (compile/launch problem: the PyTorch check from now on)
            key = (_dev_key(y2.device), y2.dtype)
            _serve_fast.STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
            _serve_fast.FALLBACKS[0] += 1
            _serve_fast.WHY.append(f"serving check ({key[0]}): {_serve_fast.STATUS[key][1]}")
    check_static_impl(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width,
                      state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag)


def _dev_key(device: torch.device) -> str:
    from ._group import _dev_key as k
    return k(device)


_lib_lock = threading.Lock()
_OP = [None]


def _op():
    """sdcguard::check_static, registered once (torch.compile sees one opaque call; CUDA graphs
    capture its kernels)."""
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::check_static", mutates_args=("state", "cap_meta", "cap_data"))
                def check_static(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                                 pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, b2max: torch.Tensor,
                                 bmax: torch.Tensor, bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor,
                                 width: torch.Tensor,
                                 state: torch.Tensor, slot: int, z: float, abs_floor: float,
                                 cap_meta: torch.Tensor, cap_data: torch.Tensor, cap_w: int, cap_tag: int) -> None:
                    check_static_any(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum,
                                     wsq, width, state, slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag)

                @check_static.register_fake
                def _(x2, y2, parts, pair_full, pair_rem, scale, b2n, b2max, bmax, bmax_bias, bsum, wsq, width, state,
                      slot, z, abs_floor, cap_meta, cap_data, cap_w, cap_tag):
                    return None

                _OP[0] = torch.ops.sdcguard.check_static
            except (AttributeError, RuntimeError):      # very old PyTorch, or already registered
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "check_static", None) or check_static_any
        return _OP[0]


def _is_linear_forward(f, mod: torch.nn.Module) -> bool:
    """f is mod's own nn.Linear.forward, bound to mod (what accelerate leaves on the instance after removing a hook)."""
    return getattr(f, "__func__", None) is torch.nn.Linear.forward and getattr(f, "__self__", None) is mod


def _accelerate_hooked(mod: torch.nn.Module) -> bool:
    """Is mod's instance forward accelerate's hook wrapper (add_hook_to_module), with nn.Linear.forward kept as
    mod._old_forward? Only then does the hook call the plain linear underneath."""
    return getattr(mod, "_hf_hook", None) is not None and _is_linear_forward(mod.__dict__.get("_old_forward"), mod)


def _location(w) -> Optional[tuple]:
    """(device, address) of a weight tensor's memory, or None for no tensor or an empty one."""
    if not isinstance(w, torch.Tensor) or w.numel() == 0:
        return None
    return str(w.device), w.data_ptr()


_NO_TENSOR = -1


def _writes(mod) -> tuple:
    """How many in-place writes the layer's weight and bias have had (PyTorch's version counters): _NO_TENSOR for
    one the layer doesn't have, None for one there's no count for (an inference tensor keeps none). A write through
    .data or by another device isn't counted."""
    out = []
    for t in (getattr(mod, "weight", None), getattr(mod, "bias", None)):
        if not isinstance(t, torch.Tensor):
            out.append(_NO_TENSOR)
            continue
        try:
            out.append(int(t._version))
        except Exception:  # noqa: BLE001  (inference tensors raise; diagnosis is best effort)
            out.append(None)
    return tuple(out)


def _rewritten(before, now) -> Optional[bool]:
    """Whether the weight or bias had an in-place write between two _writes() readings: True, False, or None when a
    count is missing on either side (and nothing counted says True)."""
    unknown = False
    for b, c in zip(before or (None, None), now):
        if b is None or c is None:
            unknown = True
        elif b != c:
            return True
    return None if unknown else False


# Why a layer's weights can't be read back as its own (ServeGuard._unreadable_weights)
_SHARED = "they share memory with another checked layer's weights, as vLLM's prefetch CPU offloading does"
_MOVED = "they moved to other memory after the layer was registered, as vLLM's CPU offloading does"


class ServeGuard:
    """Always-on checks for inference. See the module docstring.

    z: alert threshold in units of the estimated rounding noise (default 6, as in training).
    on_alert(event): optional callback, called from poll().
    evidence_dir: where to save the captured row of each alert as an evidence file (`sdcguard replay` and
        `sdcguard casefile` read it), at most max_evidence of them; without it the alert still carries the
        float64 recheck of the captured row (see CAPTURE_SLOTS and _recheck).
    """

    def __init__(self, z: float = 6.0, block: int = 128, abs_floor: float = ABS_FLOOR,
                 on_alert: Optional[Callable[[SDCEvent], None]] = None, log_path: Optional[str] = None,
                 inline: bool = True, exact_reductions: bool = True, webhook: Optional[str] = None,
                 webhook_format: str = "slack", metrics_port: Optional[int] = None, metrics=None,
                 evidence_dir: Optional[str] = None, max_evidence: int = 20):
        self.z, self.block, self.abs_floor = z, block, abs_floor
        self.evidence_dir, self.max_evidence = evidence_dir, max_evidence
        self.evidence: list = []                   # evidence files saved from captured rows
        self.evidence_meta: dict = {}              # more fields for their .json (sdcguard screen: its run's tags)
        self._cap: dict = {}                       # device -> (cap_meta, cap_data), see CAPTURE_SLOTS
        self._cap_w = max(128, int(block))         # values of y a captured block can hold
        self._epoch: dict = {}                     # device -> reads of its state so far (captures carry theirs)
        self._captures: list = []                  # captured rows read back before the alert they belong to
        self._dropped = 0                          # rows the capture buffer had no room for, at the last read
        # exact_reductions: like the training guard, keep bf16/fp16 split-K partial sums in fp32 while
        # attached (the noise model assumes one rounding per output; PyTorch's default lets cuBLAS add
        # partial sums in bf16). A global PyTorch setting; restored by detach().
        self.exact_reductions = exact_reductions
        self._restore = None
        # inline: under torch.compile, where the Triton check isn't available, trace the PyTorch check
        # into the graph so Inductor fuses its operations; otherwise it runs as the sdcguard::check_static
        # custom op (the Triton check on NVIDIA GPUs).
        self.inline = inline
        # webhook: post each alert (Slack/Discord/JSON, see sdcguard.alerts); metrics_port: serve Prometheus
        # metrics on this port (needs prometheus-client); metrics: an existing sdcguard.alerts.Metrics
        from .alerts import Metrics, Webhook, chain
        self.webhook = Webhook(webhook, fmt=webhook_format) if webhook else None
        self.on_alert = chain(on_alert, self.webhook)
        self.metrics = metrics if metrics is not None else (Metrics(port=metrics_port) if metrics_port else None)
        self.log = EventLog(log_path)
        self.enabled = True
        self.layers: list[dict] = []               # slot -> {"name", "module", "cs", "device"}
        self._state: dict = {}                     # device -> (slots, STATE_COLS) int64 keys
        self._host: dict = {}                      # device -> (pinned copy, CUDA event) of a pending read
        self._identity: dict = {}
        self.worst_seen: dict = {}                 # layer name -> worst score ever read back (clean or not)
        self.checks_seen: dict = {}                # layer name -> checks executed, read back so far
        self.unjudged_seen: dict = {}              # layer name -> checksums that couldn't be judged, so far
        self.stats = {"layers": 0, "skipped_layers": 0, "polls": 0, "alerts": 0, "checks_executed": 0,
                      "unjudged_checksums": 0, "captured": 0, "captures_dropped": 0}
        self.unsupported: dict = {}                # layer name -> why it isn't checked

    # ---------------------------------------------------------------- setup
    def attach(self, model: torch.nn.Module, share: float = 1.0, offset: int = 0) -> "ServeGuard":
        """Check every nn.Linear whose weight is bf16/fp16. Call after the model is on its device,
        in its serving dtype, and before torch.compile / CUDA graph capture.

        share < 1 checks only that share of the linear layers (spread over layer kinds and depths, see
        select_layer), for a lower cost and less coverage. Detection then depends on whether a faulty
        computation reaches one of the selected layers. With share 1/R, replicas with offsets 0 to R-1
        select every layer between them, each exactly once; each replica/request still checks its own share."""
        from . import _swap
        took_flags = False
        if self.exact_reductions and self._restore is None:
            from .guard import _exact_reductions_off, _exact_reductions_on
            self._restore = _exact_reductions_on() or ()
            took_flags = True
            if self._restore:
                # a guard dropped without detach() lets go of the settings when it's collected, so they don't
                # stay changed for the rest of the process (letting go twice is a no-op)
                weakref.finalize(self, _exact_reductions_off, self._restore)
        first = len(self.layers)
        try:
            self._attach(model, share, offset)
        except BaseException:
            # nothing half-done: the layers this call took over go back, and so do the settings
            for lay in self.layers[first:]:
                self._release(lay)
            del self.layers[first:]
            self.stats["layers"] = len(self.layers)
            if took_flags:
                from .guard import _exact_reductions_off
                _exact_reductions_off(self._restore)
                self._restore = None
            raise
        return self

    def _attach(self, model: torch.nn.Module, share: float, offset: int) -> None:
        from . import _swap
        bypassed = {**_swap.multihead_out_projs(model), **_swap.fused_encoder_linears(model)}
        mine = {id(lay["module"]) for lay in self.layers}
        i = 0
        for name, mod in model.named_modules():
            if not isinstance(mod, torch.nn.Linear):
                continue
            label = name or type(mod).__name__
            if id(mod) in mine:
                continue
            if getattr(mod, "_sdcguard_slot", None) or _swap.is_guarded(mod):
                # (a second ServeGuard, or a training guard's protect(), has it: listed, never doubled)
                self.unsupported[label] = ("already checked by another ServeGuard" if getattr(mod, "_sdcguard_slot", None)
                                           else "already checked by a training guard (protect)")
                self.stats["skipped_layers"] += 1
                continue
            if mod.weight.dtype not in UNIT_ROUNDOFF:
                # listed, so coverage_report() shows it (an fp32 router kept in a bf16 model, say)
                self.unsupported[label] = (f"{str(mod.weight.dtype).replace('torch.', '')} weights: the serving "
                                           "check takes bf16 and fp16; not checked")
                self.stats["skipped_layers"] += 1
                continue
            if mod.weight.numel() == 0:
                self.unsupported[label] = "no weights (a layer with 0 inputs or outputs): nothing to check"
                self.stats["skipped_layers"] += 1
                continue
            if type(mod).forward is not torch.nn.Linear.forward:
                # its output isn't x @ W^T + b, so the check would misfire: leave it alone, say so
                self.unsupported[label] = f"{type(mod).__name__} has its own forward"
                self.stats["skipped_layers"] += 1
                continue
            if id(mod) in bypassed:
                self.unsupported[label] = bypassed[id(mod)]
                self.stats["skipped_layers"] += 1
                continue
            if mod.weight.device.type == "meta":
                self.unsupported[label] = "weights not on a device when attached (offloaded); not checked"
                self.stats["skipped_layers"] += 1
                continue
            # A forward set on the instance runs instead of the class's. accelerate's device hooks (dispatch_model,
            # device_map=...) keep the real forward as _old_forward and call it after moving the inputs to the
            # weights' device: the check goes there, underneath the hook. Any other instance forward could compute
            # something else, so it's left alone and listed (a check of one sample can't prove it's a plain linear
            # layer for every input).
            hooked = False
            if "forward" in mod.__dict__ and not _is_linear_forward(mod.__dict__["forward"], mod):
                if not _accelerate_hooked(mod):
                    self.unsupported[label] = "forward replaced on the instance (not an accelerate device hook); not checked"
                    self.stats["skipped_layers"] += 1
                    continue
                hooked = True
            take = select_layer(i, share, offset)
            i += 1
            if not take:
                self.stats["skipped_layers"] += 1
                continue
            slot = self.add_layer(mod, label)
            _swap.take_over(mod, self._make_forward(mod))
            if hooked:          # accelerate's hook calls _old_forward: the subclass's forward, bound to this layer
                mod.__dict__["_old_forward"] = type(mod).forward.__get__(mod)
            self.layers[slot]["took_over"] = True

    def _release(self, lay: dict) -> None:
        """Give one layer its own forward back (whatever hooks were added or removed while it was attached)."""
        from . import _swap
        mod = lay["module"]
        if lay.get("took_over"):
            _swap.give_back(mod)
            for attr in ("forward", "_old_forward"):
                f = mod.__dict__.get(attr)
                if getattr(f, "__func__", None) in _swap.forwards():
                    if attr == "forward":            # (a hook removed meanwhile left the bound forward behind)
                        del mod.__dict__[attr]
                    else:
                        mod.__dict__[attr] = type(mod).forward.__get__(mod)
        mod.__dict__.pop("_sdcguard_slot", None)
        # Engine plugins retain op arguments directly rather than taking over the module's forward.
        # Drop those references too, or an engine can keep checking an orphaned state after detach().
        mod.__dict__.pop("_sdcguard_args", None)
        mod.__dict__.pop("_sdcguard_trace", None)

    def add_layer(self, mod: torch.nn.Module, name: str, weight: Optional[torch.Tensor] = None,
                  bias: Optional[torch.Tensor] = None) -> int:
        """Register one layer (also used by engine plugins). Returns its slot."""
        w = mod.weight if weight is None else weight
        b = getattr(mod, "bias", None) if bias is None else bias
        # attached inside torch.inference_mode() (a server's request loop), the checksums and the state would
        # be inference tensors, and poll()'s in-place reset of the state outside that mode would fail after
        # zeroing it: everything the guard keeps is made as ordinary tensors
        with torch.inference_mode(False), torch.no_grad():
            cs = LayerChecksums(w, b, self.block)
            self._prepare(cs, w)
            slot = len(self.layers)
            # where the layer's weight is now: weights that later live elsewhere may not be its own any more
            # (_unreadable_weights)
            self.layers.append({"name": name, "module": mod, "cs": cs, "device": w.device,
                                "weight_at": _location(getattr(mod, "weight", None)), "writes": _writes(mod)})
            mod._sdcguard_slot = slot + 1              # 0 means "not registered"
            self._grow(w.device)
        self.stats["layers"] += 1
        return slot

    def _prepare(self, cs: "LayerChecksums", w: torch.Tensor) -> None:
        """The Triton check's first-use comparison and compile, before serving (blocks up to 128 wide only:
        its sign table holds 128; wider blocks use the PyTorch check)."""
        from . import _serve_fast
        if cs.bs <= 128 and _serve_fast.usable(w.device, w.dtype):
            _serve_fast.verified(w.device, w.dtype)        # first-use comparison, once per device and dtype
            try:
                _serve_fast.warm(*cs.tensors(True), cs.k, cs.n, cs.dtype)    # compile now, not while serving
            except Exception:  # noqa: BLE001  (a compile problem shows up, and falls back, at the first check)
                pass

    def _grow(self, device: torch.device) -> None:
        """The state tensor of a device has room for 4,096 layers, and its capture buffer for rows of x as long
        as its widest layer's input; past that they are reallocated (and layers attached earlier get their forward
        rebuilt, since they hold references to them). Layers are registered before serving starts, so this never
        happens inside a captured CUDA graph."""
        n = len(self.layers)
        kmax = max([lay["cs"].k for lay in self.layers if lay["device"] == device] or [1])
        old, cap = self._state.get(device), self._cap.get(device)
        rebuild = False
        st = old
        if old is None or old.shape[0] < n:
            st = torch.zeros(max(4096, 2 * n), STATE_COLS, device=device, dtype=torch.int64)
            if old is not None:
                st[:old.shape[0]] = old
                rebuild = True
            _static(st)
            self._state[device] = st
        cm, cd = cap if cap is not None else (None, None)
        if cm is None or cm.numel() < CAP_FLAGS + st.shape[0]:
            new = torch.zeros(CAP_FLAGS + st.shape[0], device=device, dtype=torch.int64)
            if cm is not None:
                new[:cm.numel()] = cm
                rebuild = True
            _static(new)
            cm = new
        if cd is None or cd.shape[1] < self._cap_w + kmax:
            new = torch.zeros(CAPTURE_SLOTS + 1, self._cap_w + max(8192, -(-kmax // 1024) * 1024), device=device,
                              dtype=torch.float32)
            if cd is not None:
                new[:, :cd.shape[1]] = cd
                rebuild = True
            _static(new)
            cd = new
        self._cap[device] = (cm, cd)
        if not rebuild:
            return
        for slot, lay in enumerate(self.layers):
            mod = lay["module"]
            if lay["device"] != device:
                continue
            if lay.get("took_over"):
                mod.__dict__["_sdcguard_forward"] = self._make_forward(mod)
            if getattr(mod, "_sdcguard_args", None) is not None:      # registered by an engine plugin (vLLM)
                mod._sdcguard_args = (self.op_args(slot, False), self.op_args(slot, True))

    def state(self, device) -> torch.Tensor:
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return self._state[device]

    def read_slot(self, slot: int):
        """(z, flat index, noise) of one layer since the last poll, without resetting (waits for the GPU)."""
        st = self._state[self.layers[slot]["device"]]
        return decode_state(st[slot:slot + 1].detach().to("cpu"))[0]

    def uses_fast(self, slot: int) -> bool:
        """This layer's checks run as the Triton kernel (else the PyTorch check)."""
        from . import _serve_fast
        lay = self.layers[slot]
        return lay["cs"].bs <= 128 and _serve_fast.ready(lay["device"], lay["cs"].dtype)

    def check(self, slot: int, x: torch.Tensor, y: torch.Tensor, with_bias: bool = True) -> None:
        """Queue the check of layer `slot` for input x and output y (any leading dims)."""
        if not self.enabled:
            return
        lay = self.layers[slot]
        cs = lay["cs"]
        if y.dtype not in UNIT_ROUNDOFF or x.shape[-1] != cs.k or y.shape[-1] != cs.n:
            return
        check_static_any(x.reshape(-1, cs.k), y.reshape(-1, cs.n), *self.op_args(slot, with_bias))

    def op_args(self, slot: int, with_bias: bool = True) -> tuple:
        """Everything the check op takes after x2 and y2, for layer `slot`."""
        lay = self.layers[slot]
        return (*lay["cs"].tensors(with_bias), self._state[lay["device"]], slot, float(self.z),
                float(self.abs_floor), *self._cap_args(slot, with_bias))

    def _cap_args(self, slot: int, with_bias: bool) -> tuple:
        """The capture buffer's arguments of the check op for layer `slot` (see CAPTURE_SLOTS)."""
        lay = self.layers[slot]
        cm, cd = self._cap[lay["device"]]
        return cm, cd, self._cap_w, (slot + 1) * 2 + (1 if with_bias and lay["cs"].has_bias else 0)

    def _make_forward(self, mod: torch.nn.Linear):
        """The forward of an attached layer (sdcguard._swap calls it with the layer it runs on): the plain
        linear, then one call of the check op. Kept free of Python that torch.compile can't trace (no locks,
        no data-dependent branches)."""
        guard, slot = self, mod._sdcguard_slot - 1
        lay = self.layers[slot]
        cs, state, op = lay["cs"], self._state[lay["device"]], _op()
        k, n, z, floor = cs.k, cs.n, float(self.z), float(self.abs_floor)
        dtype, device = cs.dtype, lay["device"]
        trace = guard.inline and not guard.uses_fast(slot)
        with_bias = (*cs.tensors(True), state, slot, z, floor, *guard._cap_args(slot, True))
        without = (*cs.tensors(False), state, slot, z, floor, *guard._cap_args(slot, False))

        def forward(m, x):
            b = m.bias
            # at::linear adds the bias inside the matmul (one rounding) for 2-D inputs and contiguous ones with
            # a plain bias; otherwise it rounds the matmul, then adds the bias (rounded again). The check
            # assumes one rounding, so in the second case it checks the matmul alone and adds the bias after,
            # the way at::linear does (same values).
            fused = b is None or x.dim() <= 2 or (x.is_contiguous() and b.dim() == 1 and b.is_contiguous())
            y = torch.nn.functional.linear(x, m.weight, b if fused else None)
            # (a layer cast to another dtype, or a replica on another device, isn't what the checksums
            # describe: it runs unchecked until refresh(); verify_weights() lists it)
            if guard.enabled and x.numel() > 0 and y.dtype == dtype and y.device == device:
                if torch.compiler.is_compiling():
                    call = check_static_impl if trace else op
                else:
                    call = check_static_any
                call(x.reshape(-1, k), y.reshape(-1, n), *(with_bias if fused and b is not None else without))
            return y if fused else y + b

        return forward

    def detach(self) -> None:
        """Collect outstanding results, then release the attached layers (waits for queued GPU work).

        Aggregate stats and the event log remain available. Per-layer snapshots belong to this attachment,
        so a later attach() cannot inherit execution evidence from a different model with the same names.
        Captured/compiled model graphs must be discarded or rebuilt after changing their instrumentation.
        """
        try:
            if self.layers:
                self.poll(wait=True)
        finally:
            if self._restore:
                from .guard import _exact_reductions_off
                _exact_reductions_off(self._restore)
            self._restore = None
            for lay in self.layers:
                self._release(lay)
            self.layers.clear()
            self._state.clear()
            self._host.clear()
            self._cap.clear()
            self._epoch.clear()
            self._captures.clear()
            self.worst_seen.clear()
            self.checks_seen.clear()
            self.unjudged_seen.clear()
            self.unsupported.clear()
            self.stats["layers"] = 0

    # ---------------------------------------------------------------- results
    def counts(self) -> dict:
        """Per layer: checks executed and checksums that couldn't be judged, everything read back so far plus
        what hasn't been polled yet, without resetting (waits for the GPU). This, not worst_z(), says whether a
        layer was checked."""
        live: dict = {}
        for dev, st in self._state.items():
            rows = decode_counts(st[:len(self.layers)].detach().to("cpu"))
            pending = self._host.get(dev)
            if pending is not None:                  # a non-blocking read still on its way: count it too
                pending[1].synchronize()
                prev = decode_counts(pending[0])
                rows = [(a + c, b + d) for (a, b), (c, d) in zip(rows, prev + [(0, 0)] * (len(rows) - len(prev)))]
            for slot, (ex, un) in enumerate(rows):
                if self.layers[slot]["device"] == dev:
                    live[self.layers[slot]["name"]] = (ex, un)
        out = {}
        for lay in self.layers:
            name = lay["name"]
            ex, un = live.get(name, (0, 0))
            out[name] = {"checks": self.checks_seen.get(name, 0) + ex, "unjudged": self.unjudged_seen.get(name, 0) + un}
        return out

    def coverage_report(self) -> dict:
        """What actually ran: layers attached, layers checked at least once, layers never checked, checks
        executed, checksums that couldn't be judged, and layers left unchecked at attach (with why)."""
        c = self.counts()
        return {"layers": len(self.layers), "layers_checked": sum(v["checks"] > 0 for v in c.values()),
                "never_checked": sorted(n for n, v in c.items() if v["checks"] == 0),
                "checks_executed": sum(v["checks"] for v in c.values()),
                "unjudged_checksums": sum(v["unjudged"] for v in c.values()),
                "layers_with_unjudged": sorted(n for n, v in c.items() if v["unjudged"]),
                "not_attached": dict(self.unsupported)}

    def worst_z(self) -> dict:
        """Worst score of each layer since the last poll, without resetting (waits for the GPU). A
        positive score proves a check contributed a residual. Zero is ambiguous: no execution, an exactly
        matching result, and an unjudged result can all produce zero; counts() tells them apart."""
        out = {}
        for dev, st in self._state.items():
            rows = [r[0] for r in decode_state(st[:len(self.layers)].detach().to("cpu"))]
            pending = self._host.get(dev)
            if pending is not None:                  # a non-blocking read still on its way: count it too
                pending[1].synchronize()
                prev = [r[0] for r in decode_state(pending[0])]
                rows = [max(a, b) if a == a and b == b else float("nan") for a, b in
                        zip(rows, prev + [0.0] * (len(rows) - len(prev)))]
            for slot, z in enumerate(rows):
                if self.layers[slot]["device"] == dev:
                    out[self.layers[slot]["name"]] = z
        return out

    def poll(self, wait: bool = False) -> list[SDCEvent]:
        """Collect alerts. Without wait, never stalls the GPU: it starts a copy of the state and
        reports the previous copy once it has arrived (so an alert shows up one poll later).
        With wait=True, reads the current state right away (waits for queued GPU work)."""
        self.stats["polls"] += 1
        events = []
        for dev, st in self._state.items():
            if dev.type != "cuda" or wait:
                pending = self._host.pop(dev, None)
                if pending is not None:              # report a non-blocking read that is still pending first
                    pending[1].synchronize()
                    events += self._report(dev, pending[0], pending[2] if len(pending) > 2 else None)
                n = len(self.layers)
                host = st[:n].detach().to("cpu", copy=True)
                st[:n].zero_()
                events += self._report(dev, host, self._next_epoch(dev, n))
                continue
            pending = self._host.get(dev)
            if pending is not None and pending[1].query():
                events += self._report(dev, pending[0], pending[2] if len(pending) > 2 else None)
                pending = None
                self._host.pop(dev, None)
            if pending is None:
                n = len(self.layers)
                buf = torch.empty((n, STATE_COLS), dtype=st.dtype, pin_memory=True)
                buf.copy_(st[:n], non_blocking=True)
                st[:n].zero_()                 # same stream: runs after the copy has read st
                epoch = self._next_epoch(dev, n)
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (buf, ev, epoch)
        return events

    def _next_epoch(self, dev: torch.device, n: int) -> int:
        """After a read of a device's state: the epoch that read covers, which the rows captured during it carry.
        On the device, queued after the read like the state's reset: the epoch moves on, and every layer may
        keep a row again."""
        e = self._epoch.get(dev, 0)
        cap = self._cap.get(dev)
        if cap is not None:
            cap[0][1:2].add_(1)
            cap[0][CAP_FLAGS:CAP_FLAGS + n].zero_()
        self._epoch[dev] = e + 1
        return e

    def _take_captures(self, dev: torch.device) -> list:
        """The rows captured on a device so far (waits for the GPU: called only to report an alert), and the
        buffer freed for the next ones. Rows read earlier and not yet matched to their alert come first."""
        mine = [r for r in self._captures if r["device"] == dev]
        self._captures = [r for r in self._captures if r["device"] != dev]
        cap = self._cap.get(dev)
        if cap is None:
            return mine
        cm, cd = cap
        head = cm[:CAP_FLAGS].to("cpu")
        claimed = int(head[0])
        kept = min(claimed, CAPTURE_SLOTS)
        if not claimed:
            return mine
        data = cd[:kept].to("cpu") if kept else None
        for i, (tag, row, blk, flat, zbits, epoch) in enumerate(
                head[CAP_HEAD:CAP_FLAGS].view(CAPTURE_SLOTS + 1, CAP_REC)[:kept].tolist()):
            slot = tag // 2 - 1
            if not 0 <= slot < len(self.layers) or self.layers[slot]["device"] != dev:
                continue
            cs = self.layers[slot]["cs"]
            if not 0 <= blk < cs.nb:
                continue
            import struct
            mine.append({"device": dev, "slot": slot, "with_bias": bool(tag % 2), "row": row, "col_block": blk,
                         "flat": flat, "z": struct.unpack("<f", struct.pack("<I", zbits & 0xFFFFFFFF))[0],
                         "epoch": epoch, "y": data[i, :self._cap_w].clone(),
                         "x": data[i, self._cap_w:self._cap_w + cs.k].clone()})
        self.stats["captures_dropped"] += claimed - kept
        self._dropped = claimed - kept
        # free for the next rows (queued after the reads above; the layers' flags reset with the next read)
        cm[:1].zero_()
        cm[CAP_HEAD:CAP_FLAGS].zero_()
        return mine

    def _block_of(self, slot: int, blk: int) -> tuple:
        cs = self.layers[slot]["cs"]
        return blk * cs.bs, (cs.bs if blk < cs.nbf else cs.rem)

    def _slice(self, slot: int, cap: dict) -> tuple:
        """The captured problem as one row and one block: A = the row of x (1, k), B = the block's weights as they
        are now (k, width), C = the captured block of y (1, width), and that block of the bias when the matmul
        added it."""
        lay = self.layers[slot]
        cs, dt = lay["cs"], lay["cs"].dtype
        start, width = self._block_of(slot, cap["col_block"])
        A = cap["x"].to(dt).view(1, cs.k)
        C = cap["y"][:width].to(dt).view(1, width)
        B = lay["module"].weight.detach()[start:start + width].to("cpu").t().contiguous()
        b = getattr(lay["module"], "bias", None)
        bias = b.detach()[start:start + width].to("cpu") if cap["with_bias"] and isinstance(b, torch.Tensor) else None
        return A, B, C, bias, start, width

    def _recheck(self, slot: int, cap: dict, changed, unreadable: Optional[str], rewritten: bool = False) -> dict:
        """The captured row and block judged again on the CPU from the numbers that alarmed and the layer's weights
        as they are when the alert is read (the capture keeps x and y, not the weights), by screen.cpu_recheck (the
        same test evidence files get: exact_block's rounding envelope, then the check itself with the same checksums
        and noise model, its expected side computed independently):
          output_wrong   some outputs are outside the rounding envelope: wrong for these weights
          output_off     every output is inside the envelope, but the CPU check flags the block too: errors that
                         add up (a statistical mismatch, not a confirmed fault)
          output_agrees  the CPU doesn't reproduce the mismatch with these weights: the check's own side (its GPU
                         arithmetic, or its checksums) disagreed, unless the matmul ran with other numbers than the
                         recheck has; nothing here says this output was wrong
          weights_changed  the weights don't match their load-time fingerprint
          unknown        the weights were written in place since load (rewritten) though their values match now,
                         so the matmul may have used others; or they can't be read back as the layer's own; or the
                         block couldn't be judged
        These weights are the ones the matmul used when they match load time and had no in-place write since, as
        far as PyTorch counts writes (one through .data, or by another device, isn't counted)."""
        lay = self.layers[slot]
        cs = lay["cs"]
        out = {"by": "screen.cpu_recheck on the CPU: the captured row of x and block of y, with the layer's weights "
                     "as they are at the alert"}
        if unreadable:
            return {**out, "verdict": "unknown",
                    "why": f"the layer's weights can't be read back as its own: {unreadable}"}
        w = getattr(lay["module"], "weight", None)
        if not (isinstance(w, torch.Tensor) and tuple(w.shape) == (cs.n, cs.k)):
            return {**out, "verdict": "unknown", "why": "the layer's weight isn't the matrix its checksums describe"}
        from .screen import cpu_recheck
        A, B, C, bias, start, width = self._slice(slot, cap)
        r = cpu_recheck(A, B, C, bias, 1.0, 1.0, 0, 0, block=cs.bs, z=float(self.z), abs_floor=float(self.abs_floor))
        out.update(columns=[start, start + width], cpu=r)
        if changed:
            out.update(verdict="weights_changed", why="the layer's weights changed since load, so a recheck with "
                       "today's weights can't say whether this output was right when it was computed")
        elif rewritten:
            out.update(verdict="unknown", why="the layer's weights were written in place since load"
                       + (" (their values match load time now)" if changed is False else "")
                       + ", so the matmul may have run with other weights than the recheck has: it can't say whether "
                       "this output was right")
        elif not r.get("done") or not r.get("judged"):
            out.update(verdict="unknown", why=r.get("reading") or r.get("why") or "the block couldn't be judged")
        elif r.get("output_mismatch_confirmed"):
            out.update(verdict="output_wrong", why=f"{r['outputs_outside_rounding_envelope']} of the {width} captured "
                       "outputs are outside the rounding envelope of x @ W^T recomputed in float64 with the layer's "
                       "weights as they are now: wrong for these weights")
        elif r.get("output_wrong"):
            out.update(verdict="output_off", why="every captured output is inside the rounding envelope, but the CPU "
                       f"check flags the block too (score {r['max_z']:.1f}): errors that add up, a statistical "
                       "mismatch rather than a confirmed fault")
        else:
            out.update(verdict="output_agrees", why=f"the CPU doesn't reproduce the mismatch with the layer's weights "
                       f"as they are now (score {r['max_z']:.2f}): the check's own side disagreed, unless the matmul "
                       "ran with other numbers than the recheck has; nothing here says this output was wrong")
        if changed is None and out["verdict"] != "weights_changed":
            out["note"] = "whether the weights changed since load couldn't be read"
        return out

    def _save_capture(self, dev: torch.device, slot: int, cap: dict, recheck: dict, changed,
                      rewritten: bool = False) -> Optional[str]:
        """The captured row as an evidence file (screen._save_evidence's format: replay and casefile read it):
        A = the row of x, B = the block's weights (as they are at the alert), C = the captured block of y."""
        import os
        from types import SimpleNamespace
        from .screen import _save_evidence
        lay = self.layers[slot]
        cs = lay["cs"]
        A, B, C, bias, start, width = self._slice(slot, cap)
        os.makedirs(self.evidence_dir, exist_ok=True)
        res = SimpleNamespace(max_z=cap["z"], nonfinite=not bool(torch.isfinite(C).all()), row=0, col_block=0,
                              batch=None)
        meta = {"phase": "serving",
                "captured_at_check": "the check copied this row of the layer's input and this block of its output "
                                     "when it scored them, before the memory was reused: the numbers that alarmed",
                "layer_row": cap["row"], "layer_col_block": cap["col_block"], "layer_columns": [start, start + width],
                "weights": "the layer's weights as they were when the alert was read",
                "weights_changed_since_load": changed, "weights_rewritten_since_load": rewritten, "recheck": recheck,
                "check_config": {"block": cs.bs, "z": float(self.z), "abs_floor": float(self.abs_floor)},
                "record_device": dev}
        meta = {**{k: v for k, v in self.evidence_meta.items() if k not in meta}, **meta}
        return _save_evidence(self.evidence_dir, len(self.evidence), res,
                              (A, B, C, bias, 1.0, 1.0, "aten::addmm" if bias is not None else "aten::mm",
                               f"{lay['name']} (serving)"), self._identity.get(str(dev), {}), meta)

    def _report(self, dev: torch.device, host: torch.Tensor, epoch: Optional[int] = None) -> list[SDCEvent]:
        out = []
        rows = decode_state(host)
        for slot, (ex, un) in enumerate(decode_counts(host)[:len(self.layers)]):
            lay = self.layers[slot]
            if lay["device"] != dev or not (ex or un):
                continue
            self.checks_seen[lay["name"]] = self.checks_seen.get(lay["name"], 0) + ex
            self.unjudged_seen[lay["name"]] = self.unjudged_seen.get(lay["name"], 0) + un
            self.stats["checks_executed"] += ex
            self.stats["unjudged_checksums"] += un
        try:
            return self._report_rows(dev, rows, out, epoch)
        finally:
            if self.metrics is not None:
                try:
                    self.metrics.observe(self, [(self.layers[s]["name"], r[0]) for s, r in
                                                enumerate(rows[:len(self.layers)]) if self.layers[s]["device"] == dev], out)
                except Exception:  # noqa: BLE001  (metrics never break serving)
                    pass

    def _report_rows(self, dev: torch.device, rows: list, out: list, epoch: Optional[int] = None) -> list[SDCEvent]:
        unreadable = None                         # slots whose weights can't be read back (for the first alert)
        caps = None                               # rows the checks captured (read at the first alert)
        for slot, (zv, flat, noise) in enumerate(rows[:len(self.layers)]):
            lay = self.layers[slot]
            if lay["device"] == dev:
                prev = self.worst_seen.get(lay["name"], 0.0)
                self.worst_seen[lay["name"]] = zv if (zv != zv or zv > prev) else prev
            if not (zv > self.z):                 # also true for inf
                continue
            cs = lay["cs"]
            blk_flat = int(flat) // 2
            row, block = divmod(blk_flat, cs.nb)
            key = str(dev)
            if key not in self._identity:
                self._identity[key] = device_identity(dev)
            # Did the weights change since their checksums were computed? Yes: memory corruption, or an
            # intended update nobody told the guard about (call refresh() after updating weights).
            # No: the weight fingerprint is unchanged. A checksum alert still does not establish a cause.
            # Unknown when what the layer's weight holds now may not be its own (_unreadable_weights).
            changed = None
            if unreadable is None:
                unreadable = self._unreadable_weights()
            w = getattr(lay["module"], "weight", None)
            if isinstance(w, torch.Tensor) and tuple(w.shape) == (cs.n, cs.k) and slot not in unreadable:
                try:
                    changed = not torch.equal(LayerChecksums._fingerprint(w.detach()), cs.fingerprint)
                except Exception:  # noqa: BLE001  (diagnosis is best effort)
                    changed = None
            # Written in place since load, even if the values are back now? Then the weights the matmul used may
            # not be the ones the recheck reads (an outside review's case, October 21, 2026: a weight changed,
            # the forward run, the weight put back before the read; the right output was called wrong).
            rewritten = _rewritten(lay.get("writes"), _writes(lay["module"]))
            # An infinite score means an output the inputs can't produce: Inf/NaN in it (the check then keeps NaN as
            # that checksum's noise), or finite values too large for the inputs.
            nonfinite = math.isinf(zv) and math.isnan(noise)
            # the row the check captured when it alarmed (CAPTURE_SLOTS), from this read's epoch
            if caps is None:
                caps = []
                self._dropped = 0
                try:
                    caps = self._take_captures(dev)
                except Exception:  # noqa: BLE001  (the alert itself must never be lost over this)
                    caps = []
            cap = next((c for c in caps if c["slot"] == slot and (epoch is None or c["epoch"] == epoch)), None)
            recheck, evidence, cap_info = None, None, None
            if cap is not None:
                caps.remove(cap)
                self.stats["captured"] += 1
                cap_info = {"row": cap["row"], "col_block": cap["col_block"], "with_bias": cap["with_bias"],
                            "z": cap["z"]}
                try:
                    recheck = self._recheck(slot, cap, changed, unreadable.get(slot), rewritten)
                except Exception as e:  # noqa: BLE001
                    recheck = {"verdict": "unknown", "why": f"the recheck failed: {type(e).__name__}: {e}"}
                if self.evidence_dir and len(self.evidence) < self.max_evidence:
                    try:
                        evidence = self._save_capture(dev, slot, cap, recheck, changed, rewritten)
                        self.evidence.append(evidence)
                    except Exception as e:  # noqa: BLE001  (never break serving over a file)
                        evidence = f"not saved: {type(e).__name__}: {e}"
            verdict = (recheck or {}).get("verdict")
            if changed:
                reading = ("the layer's weights changed since load (they no longer match their load-time "
                           "checksums): in GPU memory, or updated without calling refresh()")
            elif rewritten:
                reading = recheck["why"] if verdict == "unknown" else (
                    "the output disagrees with its checksums; the layer's weights were written in place since load"
                    + (" (their values match load time now)" if changed is False else "")
                    + ": weights updated and put back without calling refresh(), a wrong matmul result, or a fault "
                    "in the check's own GPU arithmetic")
            elif verdict in ("output_wrong", "output_off", "output_agrees"):
                reading = recheck["why"] + (
                    "" if changed is not False else
                    " (they match their load-time fingerprint, and PyTorch counted no in-place write to them since)"
                    if rewritten is False else " (they match their load-time fingerprint)")
            elif changed is False:
                reading = ("the output disagrees with its checksums while the weights are unchanged: a wrong matmul "
                           "result, or a fault in the check's own GPU arithmetic; rerun the request on another GPU "
                           "to tell which")
            elif slot in unreadable:
                reading = ("the output disagrees with its checksums; whether the weights changed can't be read "
                           f"back: {unreadable[slot]}")
            else:
                reading = None
            ev = SDCEvent(detector="serve_checksum", step=None,
                          detail={"where": lay["name"], "max_z": zv, "nonfinite": nonfinite,
                                  "row": row, "col_block": block, "noise": noise,
                                  "checksum": "plain" if int(flat) % 2 == 0 else "signed",
                                  "weights_changed_since_load": changed,
                                  "weights_rewritten_since_load": rewritten,
                                  "captured": cap_info,
                                  "recheck": recheck,
                                  "evidence": evidence,
                                  "capture_missing": (None if cap is not None else
                                                      f"the capture buffer was full ({self._dropped} rows not kept)"
                                                      if self._dropped else "no row was captured for this alert"),
                                  "reading": reading},
                          identity=dict(self._identity[key]))
            self.stats["alerts"] += 1
            self.log.add(ev)
            out.append(ev)
        if caps:
            # rows captured after this read, whose alerts come with the next one
            self._captures += [c for c in caps if epoch is not None and c["epoch"] > epoch]
        # the callback runs after every layer's result is recorded: one that raises can't lose the others (the
        # state they came from was already reset)
        err = None
        for ev in out:
            if self.on_alert:
                try:
                    self.on_alert(ev)
                except Exception as e:  # noqa: BLE001
                    err = err or e
        if err is not None:
            raise err
        return out

    def verify_weights(self) -> list[str]:
        """Names of layers whose weights changed since attach() (memory corruption, or someone
        updated the weights; call refresh() after an intended update). Waits for the GPU. Layers whose weights
        can't be read back as their own (_unreadable_weights: vLLM's CPU offloading) aren't compared."""
        bad = []
        unreadable = self._unreadable_weights()
        for slot, lay in enumerate(self.layers):
            w = lay["module"].weight if hasattr(lay["module"], "weight") else None
            if w is None or slot in unreadable:
                continue
            fp = LayerChecksums._fingerprint(w.detach())
            if not torch.equal(fp, lay["cs"].fingerprint):
                bad.append(lay["name"])
        return bad

    def refresh(self) -> int:
        """Recompute every layer's checksums from its current weights. Call after updating weights on
        purpose (e.g. an RL weight sync), or every update will look like corruption. Returns how many layers
        were refreshed (see refresh_layer for those that can't be)."""
        unreadable = self._unreadable_weights()
        with torch.inference_mode(False):             # ordinary tensors, as in add_layer
            return sum(self._refresh_layer(slot, None, None, unreadable) for slot in range(len(self.layers)))

    def refresh_layer(self, slot: int, weight: Optional[torch.Tensor] = None,
                      bias: Optional[torch.Tensor] = None) -> bool:
        """Recompute one layer's checksums, writing them into the same tensors (CUDA graphs captured
        earlier keep reading the right memory). False when they can't be (the layer then runs unchecked, listed in
        unsupported): weights in a dtype the check doesn't take, or weights it can't read back as the layer's own
        (_unreadable_weights). A weight passed in (vLLM's reload passes the one it just loaded) is taken as the
        layer's own unless another checked layer's weight uses its memory."""
        with torch.inference_mode(False):             # ordinary tensors, as in add_layer
            return self._refresh_layer(slot, weight, bias, None if weight is not None else self._unreadable_weights())

    def _unreadable_weights(self) -> dict:
        """Slots whose weights the guard can't read back as theirs, with why. Alerts then don't say whether the
        weights changed, verify_weights() doesn't compare them, and refresh() doesn't take them. Either another
        registered layer's weight (a different tensor: one weight two layers hold, tied, is fine) uses the same
        memory, or the weight now lives in other memory than when the layer was registered, with the same shape
        and dtype. vLLM's CPU offloaders do both after loading: the prefetch offloader points layers N, N + its
        capacity, ... at one static GPU buffer and copies each layer's weights in before it runs, so what is there
        later belongs to whichever of them ran last (also a layer the guard doesn't check)."""
        seen: dict = {}
        out: dict = {}
        for slot, lay in enumerate(self.layers):
            w = getattr(lay["module"], "weight", None)
            at = _location(w)
            if at is None:
                continue
            cs = lay["cs"]
            if lay.get("weight_at") not in (None, at) and w.device == lay["device"] and w.dtype == cs.dtype and \
                    tuple(w.shape) == (cs.n, cs.k):
                out[slot] = _MOVED
            for other, ow in seen.get(at, ()):
                if ow is not w:
                    out.setdefault(slot, _SHARED)
                    out.setdefault(other, _SHARED)
            seen.setdefault(at, []).append((slot, w))
        return out

    def _shares_memory(self, slot: int, w: torch.Tensor) -> bool:
        """Does another registered layer's weight use the memory of w (unless it's this layer's weight too, tied)?
        (_unreadable_weights)"""
        at = _location(w)
        own = getattr(self.layers[slot]["module"], "weight", None)
        return at is not None and any(
            i != slot and o is not own and _location(o) == at
            for i, o in ((i, getattr(lay["module"], "weight", None)) for i, lay in enumerate(self.layers)))

    def _refresh_layer(self, slot: int, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                       unreadable: Optional[dict]) -> bool:
        """unreadable: _unreadable_weights() when the weights are read from the layer (None: they were passed)."""
        lay = self.layers[slot]
        mod = lay["module"]
        w = mod.weight if weight is None else weight
        b = getattr(mod, "bias", None) if bias is None else bias
        why = None
        if w.dtype not in UNIT_ROUNDOFF:
            # (cast to fp32, say: the checks need bf16 or fp16; the layer runs unchecked until it's back, and its
            # weights are then taken wherever they are, as at registration)
            why = (f"weights are {str(w.dtype).replace('torch.', '')} now (the check needs bf16 or fp16): not "
                   "checked")
            lay["weight_at"] = None
        elif unreadable is not None and slot in unreadable or unreadable is None and self._shares_memory(slot, w):
            # its checksums could come from another layer's weights; the old ones don't describe the update
            reason = unreadable[slot] if unreadable else _SHARED
            why = (f"its weights can't be read back as its own ({reason}): refresh() can't recompute its checksums, "
                   "so it isn't checked after the update" + ("; refresh_layer(slot, weight=...) takes weights known "
                                                             "to be its own" if reason is _MOVED else ""))
        if why is not None:
            if lay.get("took_over"):
                mod.__dict__["_sdcguard_forward"] = None
            if hasattr(mod, "_sdcguard_args"):
                mod._sdcguard_args = None
            self.unsupported[lay["name"]] = why
            return False
        self.unsupported.pop(lay["name"], None)
        new = LayerChecksums(w, b if isinstance(b, torch.Tensor) else None, self.block)
        old = lay["cs"]
        if (new.n, new.k) != (old.n, old.k) or new.parts.shape != old.parts.shape or new.dtype != old.dtype \
                or w.device != lay["device"]:
            # shape, dtype or device changed: nothing captured can stay valid anyway; the layer's forward is
            # rebuilt around the new checksums
            # Read existing rows while their device and checksum layout still describe the original checks.
            # Otherwise a moved layer loses its counters, or old flat indices get decoded with the new shape.
            self.poll(wait=True)
            lay["cs"], lay["device"] = new, w.device
            lay["weight_at"] = _location(getattr(mod, "weight", None))
            lay["writes"] = _writes(mod)
            self._prepare(new, w)
            self._grow(w.device)
            if lay.get("took_over"):
                mod.__dict__["_sdcguard_forward"] = self._make_forward(mod)
            if hasattr(mod, "_sdcguard_args"):
                mod._sdcguard_args = (self.op_args(slot, False), self.op_args(slot, True))
                mod._sdcguard_trace = self.inline and not self.uses_fast(slot)
            return True
        with torch.no_grad():
            for a, c in zip(old.tensors(True) + old.tensors(False), new.tensors(True) + new.tensors(False)):
                if a.data_ptr() != c.data_ptr():
                    a.copy_(c)
            old.fingerprint.copy_(new.fingerprint)
        lay["writes"] = _writes(mod)
        if lay.get("weight_at") is None:             # (back from a dtype the check doesn't take)
            lay["weight_at"] = _location(getattr(mod, "weight", None))
        # A supported dtype may have returned after an unsupported one without changing shape. The earlier
        # refresh disabled this forward; copying the tensors alone would leave it unchecked forever.
        if lay.get("took_over") and mod.__dict__.get("_sdcguard_forward") is None:
            mod.__dict__["_sdcguard_forward"] = self._make_forward(mod)
        if hasattr(mod, "_sdcguard_args") and mod._sdcguard_args is None:
            mod._sdcguard_args = (self.op_args(slot, False), self.op_args(slot, True))
            mod._sdcguard_trace = self.inline and not self.uses_fast(slot)
        return True
