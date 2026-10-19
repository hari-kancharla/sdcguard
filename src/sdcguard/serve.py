"""Detector for inference (serving): check every bf16/fp16 linear layer on every forward pass.

While a model serves, its weights don't change. So each layer's checksums (the block sums of its
weight matrix, the B side of the training check) are computed once, when the guard is attached,
and every forward pass only has to read the layer's input x and output y once more:

    expected = x @ (block sums of W)   (a skinny matmul: 2 columns per 128 outputs, as three bf16/fp16
                                        parts on NVIDIA GPUs, so 6/128 = about 4.7% of the layer's work)
    actual   = block sums of y         (one pass over y)

A mismatch beyond the rounding noise means the matmul came out wrong, or the weights changed in
GPU memory since they were loaded (a bit flip in HBM or GDDR that ECC missed, or a GPU without
ECC). verify_weights() tells the two apart.

Same math and noise model as the training check (checksum.py): two checksums per 128-output
block (plain and fixed random +-1 signs), alarm at z > 6. One difference: a block the check can't judge
(inputs that aren't finite, or a legitimate overflow) scores 0 here, because the per-layer state only holds
non-negative scores; serving doesn't count such blocks separately (sdcguard screen and check count passes
whose output isn't finite instead).

Layers whose code might compute something other than x @ W^T + b are left alone and listed in
guard.unsupported: a subclass with its own forward, and a forward set on the instance. The one exception is
accelerate's device hook (dispatch_model, from_pretrained(device_map=...)), which keeps nn.Linear.forward as
_old_forward and calls it after moving the inputs to the weights' device: the check runs there, underneath the
hook. Layers whose weights are offloaded (on the meta device when attached) are listed too.

Built for serving engines:
  - Nothing ever waits for the GPU. Results go into a small state tensor on the GPU (worst z per
    layer since the last poll, and where it was); poll() reads it back without stalling.
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
from typing import Callable, Optional

import torch

from .checksum import ABS_FLOOR, BIG, DTYPE_MAX, HALF_TINY_STEP, UNIT_ROUNDOFF, _signs, _split3
from .events import EventLog, SDCEvent, device_identity

# Per layer, two int64 keys: (bits of the worst z << 32) | flat index of that checksum, and
# (bits of the worst z << 32) | bits of its noise. z >= 0 and the bits of non-negative floats sort like
# the floats, so the largest key holds the worst z (an integer maximum, also usable as an atomic).
STATE_COLS = 2


_PHI = (math.sqrt(5.0) - 1.0) / 2.0


def select_layer(i: int, share: float, offset: int = 0) -> bool:
    """Is linear layer number i (in model order) one of the checked share? Golden-ratio spacing: a
    share spreads over every kind of layer and every depth (every 4th layer would pick the same
    projection in every transformer block). Different offsets pick different layers."""
    if share >= 1.0:
        return True
    if share <= 0.0:
        return False
    return ((i + offset) * _PHI) % 1.0 < share


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
        return (self.parts, self.pair_full, self.pair_rem, self.scale, self.b2n, self.bmax, bmb, bs, self.wsq,
                self.width)


def _block_stats(y2: torch.Tensor, pair: torch.Tensor, start: int, width: int, nblk: int):
    """Plain and signed sums (fp32), sum of squares and sum of |values| of every block of one span."""
    m = y2.shape[0]
    yb = y2[:, start:start + width * nblk]
    yb = (yb if yb.is_contiguous() else yb.contiguous()).view(m * nblk, width)
    sums = _mm_fp32(yb, pair).view(m, nblk, 2)
    yf = yb.float()
    return sums, yf.square().sum(-1).view(m, nblk), yf.abs().sum(-1).view(m, nblk)


def check_static_impl(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                      pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, bmax: torch.Tensor,
                      bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor,
                      state: torch.Tensor, slot: int, z: float, abs_floor: float) -> None:
    """Check y2 = x2 @ W^T (+ bias) for one layer and fold the worst z into state[slot].

    No host sync and no data-dependent Python control flow, so it is safe inside CUDA graphs."""
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
    floor = xn * b2n * 1e-5                                                     # (m, nb, 2)
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
    zs = torch.where(zs == zs, zs, zero)
    zs = torch.where(e_bad, zero, zs)
    zs = torch.where(a_bad, torch.where(lim < min(DTYPE_MAX[dt], BIG), inf, zero), zs)
    zs = torch.where(big, torch.where(lim < BIG, inf, zero), zs)
    # one extra 0 at the end: an empty batch (m = 0) then scores 0 instead of failing the reduction, with no
    # data-dependent branch (this also runs traced into vLLM's compiled graph)
    flat = torch.cat([zs.reshape(-1), zs.new_zeros(1)])
    mz, arg = torch.max(flat, 0)
    nz = torch.cat([den.expand_as(zs).reshape(-1), den.new_ones(1)]).gather(0, arg.view(1)).squeeze(0)
    zb = mz.float().clamp_min(0).view(torch.int32).to(torch.int64) << 32
    keys = torch.stack([zb | arg.to(torch.int64), zb | (nz.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF)])
    state[slot] = torch.maximum(state[slot], keys)


def decode_state(rows: torch.Tensor):
    """(z, flat index, noise) per row of a state tensor (on the CPU)."""
    zs = (rows[:, 0] >> 32).to(torch.int32).view(torch.float32).tolist()
    flats = (rows[:, 0] & 0xFFFFFFFF).tolist()
    nzs = (rows[:, 1] & 0xFFFFFFFF).to(torch.int32).view(torch.float32).tolist()
    return list(zip(zs, flats, nzs))


def check_static_any(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                     pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, bmax: torch.Tensor,
                     bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor,
                     state: torch.Tensor, slot: int, z: float, abs_floor: float) -> None:
    """The Triton check where it has passed its first-use comparison, else the PyTorch one."""
    if x2.shape[0] == 0:                       # empty batch: nothing to check
        return
    from . import _serve_fast
    if _serve_fast.ready(y2.device, y2.dtype):
        try:
            _serve_fast.check(x2, y2, parts, pair_full, pair_rem, scale, b2n, bmax, bmax_bias, bsum, wsq, width,
                              state, slot, z, abs_floor)
            return
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as e:  # noqa: BLE001  (compile/launch problem: the PyTorch check from now on)
            key = (_dev_key(y2.device), y2.dtype)
            _serve_fast.STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
            _serve_fast.FALLBACKS[0] += 1
            _serve_fast.WHY.append(f"serving check ({key[0]}): {_serve_fast.STATUS[key][1]}")
    check_static_impl(x2, y2, parts, pair_full, pair_rem, scale, b2n, bmax, bmax_bias, bsum, wsq, width, state,
                      slot, z, abs_floor)


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
                @torch.library.custom_op("sdcguard::check_static", mutates_args=("state",))
                def check_static(x2: torch.Tensor, y2: torch.Tensor, parts: torch.Tensor, pair_full: torch.Tensor,
                                 pair_rem: torch.Tensor, scale: torch.Tensor, b2n: torch.Tensor, bmax: torch.Tensor,
                                 bmax_bias: torch.Tensor, bsum: torch.Tensor, wsq: torch.Tensor, width: torch.Tensor,
                                 state: torch.Tensor, slot: int, z: float, abs_floor: float) -> None:
                    check_static_any(x2, y2, parts, pair_full, pair_rem, scale, b2n, bmax, bmax_bias, bsum, wsq,
                                     width, state, slot, z, abs_floor)

                @check_static.register_fake
                def _(x2, y2, parts, pair_full, pair_rem, scale, b2n, bmax, bmax_bias, bsum, wsq, width, state, slot,
                      z, abs_floor):
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


class ServeGuard:
    """Always-on checks for inference. See the module docstring.

    z: alert threshold in units of the estimated rounding noise (default 6, as in training).
    on_alert(event): optional callback, called from poll().
    """

    def __init__(self, z: float = 6.0, block: int = 128, abs_floor: float = ABS_FLOOR,
                 on_alert: Optional[Callable[[SDCEvent], None]] = None, log_path: Optional[str] = None,
                 inline: bool = True, exact_reductions: bool = True, webhook: Optional[str] = None,
                 webhook_format: str = "slack", metrics_port: Optional[int] = None, metrics=None):
        self.z, self.block, self.abs_floor = z, block, abs_floor
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
        self.stats = {"layers": 0, "skipped_layers": 0, "polls": 0, "alerts": 0}
        self.unsupported: dict = {}                # layer name -> why it isn't checked

    # ---------------------------------------------------------------- setup
    def attach(self, model: torch.nn.Module, share: float = 1.0, offset: int = 0) -> "ServeGuard":
        """Check every nn.Linear whose weight is bf16/fp16. Call after the model is on its device,
        in its serving dtype, and before torch.compile / CUDA graph capture.

        share < 1 checks only that share of the linear layers (spread over layer kinds and depths, see
        select_layer), for a lower cost. A defective GPU unit corrupts every matmul that runs on it
        (round 3, NVBit: all layers flagged at once), so a share still catches it in the same pass;
        give replicas different offsets to cover every layer across a fleet."""
        if self.exact_reductions and self._restore is None:
            from .guard import _exact_reductions_on
            self._restore = _exact_reductions_on() or ()
        i = 0
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and not getattr(mod, "_sdcguard_slot", None):
                if mod.weight.dtype not in UNIT_ROUNDOFF:
                    self.stats["skipped_layers"] += 1
                    continue
                if type(mod).forward is not torch.nn.Linear.forward:
                    # its output isn't x @ W^T + b, so the check would misfire: leave it alone, say so
                    self.unsupported[name or type(mod).__name__] = f"{type(mod).__name__} has its own forward"
                    self.stats["skipped_layers"] += 1
                    continue
                label = name or type(mod).__name__
                if mod.weight.device.type == "meta":
                    self.unsupported[label] = "weights not on a device when attached (offloaded); not checked"
                    self.stats["skipped_layers"] += 1
                    continue
                # A forward set on the instance runs instead of nn.Linear.forward. accelerate's device hooks
                # (dispatch_model, device_map=...) keep the real forward as _old_forward and call it after moving
                # the inputs to the weights' device: the check goes there, underneath the hook. Any other
                # instance forward could compute something else, so it's left alone and listed (a check of one
                # sample can't prove it's a plain linear layer for every input).
                target = "forward"
                if "forward" in mod.__dict__ and not _is_linear_forward(mod.__dict__["forward"], mod):
                    if not _accelerate_hooked(mod):
                        self.unsupported[label] = "forward replaced on the instance (not an accelerate device hook); not checked"
                        self.stats["skipped_layers"] += 1
                        continue
                    target = "_old_forward"
                take = select_layer(i, share, offset)
                i += 1
                if not take:
                    self.stats["skipped_layers"] += 1
                    continue
                slot = self.add_layer(mod, label)
                self.layers[slot]["wraps"] = (target, mod.__dict__.get(target))
                setattr(mod, target, self._make_forward(mod))
        return self

    def add_layer(self, mod: torch.nn.Module, name: str, weight: Optional[torch.Tensor] = None,
                  bias: Optional[torch.Tensor] = None) -> int:
        """Register one layer (also used by engine plugins). Returns its slot."""
        w = mod.weight if weight is None else weight
        b = getattr(mod, "bias", None) if bias is None else bias
        cs = LayerChecksums(w, b, self.block)
        from . import _serve_fast
        if _serve_fast.usable(w.device, w.dtype):
            _serve_fast.verified(w.device, w.dtype)        # first-use comparison, once per device and dtype
            try:
                _serve_fast.warm(*cs.tensors(True), cs.k, cs.n, cs.dtype)    # compile now, not while serving
            except Exception:  # noqa: BLE001  (a compile problem shows up, and falls back, at the first check)
                pass
        slot = len(self.layers)
        self.layers.append({"name": name, "module": mod, "cs": cs, "device": w.device})
        mod._sdcguard_slot = slot + 1              # 0 means "not registered"
        self._grow(w.device)
        self.stats["layers"] += 1
        return slot

    def _grow(self, device: torch.device) -> None:
        """The state tensor of a device has room for 4,096 layers; past that it is reallocated (and
        layers attached earlier get their forward rebuilt, since they hold a reference to it)."""
        old = self._state.get(device)
        n = len(self.layers)
        if old is not None and old.shape[0] >= n:
            return
        new = torch.zeros(max(4096, 2 * n), STATE_COLS, device=device, dtype=torch.int64)
        _static(new)
        if old is not None:
            new[:old.shape[0]] = old
            self._state[device] = new
            for slot, lay in enumerate(self.layers):
                mod = lay["module"]
                if lay["device"] != device:
                    continue
                if "wraps" in lay:
                    setattr(mod, lay["wraps"][0], self._make_forward(mod))
                if hasattr(mod, "_sdcguard_args"):         # registered by an engine plugin (vLLM)
                    mod._sdcguard_args = (self.op_args(slot, False), self.op_args(slot, True))
        self._state[device] = new

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
        return _serve_fast.ready(lay["device"], lay["cs"].dtype)

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
                float(self.abs_floor))

    def _make_forward(self, mod: torch.nn.Linear):
        """Forward = the layer's own forward, then one call of the check op. Kept free of Python
        that torch.compile can't trace (no locks, no data-dependent branches)."""
        guard, slot = self, mod._sdcguard_slot - 1
        lay = self.layers[slot]
        plain = type(mod).forward.__get__(mod)              # nn.Linear.forward (attach() makes sure)
        cs, state, op = lay["cs"], self._state[lay["device"]], _op()
        k, n, z, floor = cs.k, cs.n, float(self.z), float(self.abs_floor)

        trace = guard.inline and not guard.uses_fast(slot)

        def forward(x):
            y = plain(x)
            if guard.enabled and x.numel() > 0:
                # torch.compile: the custom op (or the PyTorch check traced into the graph); eager: a plain call,
                # without the custom-op dispatch (that cost ~100 us per layer in eager decoding on the H100)
                if torch.compiler.is_compiling():
                    call = check_static_impl if trace else op
                else:
                    call = check_static_any
                call(x.reshape(-1, k), y.reshape(-1, n), *lay["cs"].tensors(), state, slot, z, floor)
            return y

        return forward

    def detach(self) -> None:
        if self._restore:
            from .guard import _exact_reductions_off
            _exact_reductions_off(self._restore)
        self._restore = None
        for lay in self.layers:
            mod = lay["module"]
            if "wraps" in lay:
                attr, saved = lay["wraps"]
                if saved is None:
                    mod.__dict__.pop(attr, None)
                else:
                    setattr(mod, attr, saved)
            mod._sdcguard_slot = 0
        self.layers.clear()
        self._state.clear()

    # ---------------------------------------------------------------- results
    def worst_z(self) -> dict:
        """Worst score of each layer since the last poll, without resetting (waits for the GPU). A
        layer at exactly 0 has not been checked since then: a quick way to see the checks really run."""
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
                    events += self._report(dev, pending[0])
                n = len(self.layers)
                host = st[:n].detach().to("cpu", copy=True)
                st[:n].zero_()
                events += self._report(dev, host)
                continue
            pending = self._host.get(dev)
            if pending is not None and pending[1].query():
                events += self._report(dev, pending[0])
                pending = None
                self._host.pop(dev, None)
            if pending is None:
                n = len(self.layers)
                buf = torch.empty((n, STATE_COLS), dtype=st.dtype, pin_memory=True)
                buf.copy_(st[:n], non_blocking=True)
                st[:n].zero_()                 # same stream: runs after the copy has read st
                ev = torch.cuda.Event()
                ev.record()
                self._host[dev] = (buf, ev)
        return events

    def _report(self, dev: torch.device, host: torch.Tensor) -> list[SDCEvent]:
        out = []
        rows = decode_state(host)
        try:
            return self._report_rows(dev, rows, out)
        finally:
            if self.metrics is not None:
                try:
                    self.metrics.observe(self, [(self.layers[s]["name"], r[0]) for s, r in
                                                enumerate(rows[:len(self.layers)]) if self.layers[s]["device"] == dev], out)
                except Exception:  # noqa: BLE001  (metrics never break serving)
                    pass

    def _report_rows(self, dev: torch.device, rows: list, out: list) -> list[SDCEvent]:
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
            # No: the matmul itself came out wrong.
            changed = None
            w = getattr(lay["module"], "weight", None)
            if isinstance(w, torch.Tensor) and tuple(w.shape) == (cs.n, cs.k):
                try:
                    changed = not torch.equal(LayerChecksums._fingerprint(w.detach()), cs.fingerprint)
                except Exception:  # noqa: BLE001  (diagnosis is best effort)
                    changed = None
            ev = SDCEvent(detector="serve_checksum", step=None,
                          detail={"where": lay["name"], "max_z": zv, "nonfinite": not math.isfinite(zv),
                                  "row": row, "col_block": block, "noise": noise,
                                  "checksum": "plain" if int(flat) % 2 == 0 else "signed",
                                  "weights_changed_since_load": changed,
                                  "reading": ("the layer's weights changed in GPU memory (or were updated "
                                              "without calling refresh())" if changed else
                                              "the matmul came out wrong (weights unchanged)" if changed is False
                                              else None)},
                          identity=dict(self._identity[key]))
            self.stats["alerts"] += 1
            self.log.add(ev)
            if self.on_alert:
                self.on_alert(ev)
            out.append(ev)
        return out

    def verify_weights(self) -> list[str]:
        """Names of layers whose weights changed since attach() (memory corruption, or someone
        updated the weights; call refresh() after an intended update). Waits for the GPU."""
        bad = []
        for lay in self.layers:
            w = lay["module"].weight if hasattr(lay["module"], "weight") else None
            if w is None:
                continue
            fp = LayerChecksums._fingerprint(w.detach())
            if not torch.equal(fp, lay["cs"].fingerprint):
                bad.append(lay["name"])
        return bad

    def refresh(self) -> None:
        """Recompute every layer's checksums from its current weights. Call after updating weights on
        purpose (e.g. an RL weight sync), or every update will look like corruption."""
        for slot in range(len(self.layers)):
            self.refresh_layer(slot)

    def refresh_layer(self, slot: int, weight: Optional[torch.Tensor] = None,
                      bias: Optional[torch.Tensor] = None) -> None:
        """Recompute one layer's checksums, writing them into the same tensors (CUDA graphs captured
        earlier keep reading the right memory)."""
        lay = self.layers[slot]
        mod = lay["module"]
        w = mod.weight if weight is None else weight
        b = getattr(mod, "bias", None) if bias is None else bias
        new = LayerChecksums(w, b if isinstance(b, torch.Tensor) else None, self.block)
        old = lay["cs"]
        if (new.n, new.k) != (old.n, old.k) or new.parts.shape != old.parts.shape:
            lay["cs"] = new                      # shape changed: nothing captured can stay valid anyway
            return
        with torch.no_grad():
            for a, c in zip(old.tensors(True) + old.tensors(False), new.tensors(True) + new.tensors(False)):
                if a.data_ptr() != c.data_ptr():
                    a.copy_(c)
            old.fingerprint.copy_(new.fingerprint)
