"""Detector 1: checksum every (sampled) matrix multiply, forward and backward.

Usage:
    guard = SDCGuard(sample_rate=1.0).watch(model)   # watch() is optional: names layers in alerts
    for step, batch in enumerate(loader):
        if step % 50 == 0:                     # check every 50th step: +2.2% on GPT-2 small (H100)
            with guard.at_step(step):
                loss = model(batch)
                loss.backward()
        else:
            loss = model(batch)
            loss.backward()
        optimizer.step()
    print(guard.stats, guard.log.events)

It hooks PyTorch at the operator level (TorchDispatchMode), so it needs no model
changes and also sees the backward-pass matmuls autograd launches. Checks are queued
on the GPU as they happen and read back once, when the `with` block exits, so the
detector never makes the GPU wait in the middle of a step. With torch.compile it still works,
but PyTorch runs checked steps in eager mode while the guard is active.

Exact reductions on checked steps (NVIDIA): by default PyTorch lets cuBLAS add up partial
results of bf16/fp16 matmuls in bf16/fp16 ("reduced precision reduction", used for speed on
big inner dimensions). That adds rounding error the checksum's noise model doesn't cover, and
on GPT-2 small it caused false alarms. While the guard is active it switches that off, so
partial sums stay in fp32 (split-K itself stays allowed), and restores the setting on exit. The
same goes for fp16 accumulation (torch.backends.cuda.matmul.allow_fp16_accumulation, PyTorch 2.7+),
whose fp16 sums the noise model doesn't cover either. Unchecked steps are untouched. Turn this off
with SDCGuard(exact_reductions=False).

Precise mode (SDCGuard(precise=True), with protect()): on checked steps each linear layer's matmuls keep
their fp32 result (the accumulator) long enough to check it, then round it to bf16/fp16. The check then
sees errors far smaller than bf16 rounding hides (checksum.matmul_zscores_acc). The model gets the same
rounding of the same fp32 result, but cuBLAS may pick a different kernel for an fp32 output, so values
on checked steps can differ from an unchecked step in the last bit; unchecked steps don't change.
"""
from __future__ import annotations

import collections
import contextlib
import functools
import math
import os
import random
import sys
import threading
import warnings
import weakref
from typing import Callable, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import _group
from . import attention as _att
from . import checksum as _cs
from .checksum import ABS_FLOOR, UNIT_ROUNDOFF, CheckResult, matmul_zscores, matmul_zscores_acc, resolve
from .events import EventLog, SDCEvent, device_identity

_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

aten = torch.ops.aten


class SDCError(RuntimeError):
    """Raised when raise_on_alert=True and a check fails."""


def op_name(func) -> str:
    try:
        return func._schema.name          # e.g. "aten::mm"
    except AttributeError:
        return str(func)


# Ops whose output is a plain matrix product we can verify. Apple GPUs (MPS) run nn.Linear as
# "_mps_linear" or "linear" and its backward as "linear_backward" instead of mm/addmm.
FORWARD_MATMUL_OPS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::_mps_linear", "aten::linear",
                      "aten::baddbmm", "aten::addbmm", "aten::mv", "aten::addmv"}
MATMUL_OPS = FORWARD_MATMUL_OPS | {"aten::linear_backward", "aten::mps_linear_backward", "aten::_grouped_mm"}
# Matrix products the guard sees but can't check yet: FP8 (scaled), integer and packed-weight GEMMs, and addmm
# fused with its activation. They are counted (stats["matmuls_unchecked"], unchecked_ops, coverage()) so a
# model that runs its work through them doesn't look covered.
UNCHECKED_MATMUL_OPS = {"aten::_scaled_mm", "aten::_scaled_mm_v2", "aten::_scaled_grouped_mm",
                        "aten::_scaled_grouped_mm_v2", "aten::_addmm_activation", "aten::_int_mm",
                        "aten::_weight_int8pack_mm", "aten::_weight_int4pack_mm",
                        "aten::_weight_int4pack_mm_for_cpu", "aten::_weight_int4pack_mm_with_scales_and_zeros"}


def unchecked_matmul_flops(name: str, args, out) -> int:
    """Multiply-adds of one unchecked matrix product (for coverage)."""
    try:
        a = args[1] if name == "aten::_addmm_activation" else args[0]
        o = out[0] if isinstance(out, (tuple, list)) else out
        if name.startswith("aten::_scaled_grouped_mm") and a.dim() == 2 and args[1].dim() == 2:
            return int(a.shape[0] * a.shape[1] * args[1].shape[1])
        return int(o.numel() * a.shape[-1])
    except Exception:  # noqa: BLE001
        return 0


def _grouped_checks(args, kwargs, out):
    """torch._grouped_mm (MoE experts): one check per group, split on `offs` (cumulative group ends). Reading
    offs is one small host copy per grouped matmul on checked steps. Rows (2-D x 3-D) or columns (3-D x 2-D) past
    offs[-1] belong to no group: torch._grouped_mm leaves them unwritten, so they are never checked."""
    a, b = args[0], args[1]
    offs = kwargs.get("offs", args[2] if len(args) > 2 else None)
    bias = kwargs.get("bias", args[3] if len(args) > 3 else None)
    # PyTorch 2.14 documents the bias as "not jagged and broadcastable to the result shape of each group": one
    # bias shared by every group, never one per group (and 2.14.1 still rejects any bias, "Bias not supported
    # yet"). A group whose output the bias doesn't broadcast to isn't checked rather than checked with a guess.
    def shared(shape) -> Optional[bool]:
        if bias is None:
            return None
        try:
            return torch.broadcast_shapes(tuple(bias.shape), tuple(shape)) == tuple(shape)
        except RuntimeError:
            return False

    if a.dim() == 3 and b.dim() == 3:
        if bias is None or shared(out.shape[1:]):
            return [(a, b, out, bias, 1.0, 1.0)]
        return []
    if offs is None:
        return []
    ends = [int(e) for e in offs.tolist()]
    checks, start = [], 0
    for i, end in enumerate(ends):
        # An empty group has nothing to check, except in a weight gradient (2-D x 2-D, grouped along k): an expert
        # that got no tokens still has an (m, n) output, and it must be all zeros (checked as a k = 0 product)
        if end > start or (a.dim() == 2 and b.dim() == 2):
            if a.dim() == 2 and b.dim() == 3:                 # tokens grouped by expert: rows of a and out
                c = (a[start:end], b[i], out[start:end])
            elif a.dim() == 3 and b.dim() == 2:               # grouped along n: columns of b and out
                c = (a[i], b[:, start:end], out[:, start:end])
            elif a.dim() == 2 and b.dim() == 2:               # grouped along k (weight gradients): one output each
                c = (a[:, start:end], b[start:end, :], out[i])
            else:
                c = None
            if c is not None and (bias is None or shared(c[2].shape)):
                checks.append((*c, bias, 1.0, 1.0))
        start = end
    return checks


def _reduced_precision_flags():
    m = torch.backends.cuda.matmul
    return m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction


# The reduced-precision settings are process-wide, and guards (training steps, ServeGuards, the screen) can
# overlap in any order: the first holder saves and changes them, the last one to let go puts them back.
_FLAGS_LOCK = threading.RLock()      # (re-entrant: a collected ServeGuard can let go from inside a garbage collection)
_FLAGS = {"holds": 0, "changed": None}


class _FlagHold:
    """One holder of the exact-reduction settings (truthy; let go once with _exact_reductions_off)."""
    __slots__ = ("live",)

    def __init__(self):
        self.live = True

    def __bool__(self):
        return True


def _exact_reductions_on() -> Optional[_FlagHold]:
    """Turn off reduced-precision reductions in bf16/fp16 cuBLAS matmuls until every holder has let go.
    Returns the hold (None if nothing could be changed). Settings already off (including the tuple form) are
    left alone."""
    if not torch.cuda.is_available():
        return None
    with _FLAGS_LOCK:
        try:
            if _FLAGS["holds"] == 0:
                bf16, fp16 = _reduced_precision_flags()
                m = torch.backends.cuda.matmul
                try:                                  # PyTorch 2.7+: fp16 GEMMs that accumulate in fp16
                    acc16 = m.allow_fp16_accumulation
                except AttributeError:
                    acc16 = None
                if bf16 is True:
                    m.allow_bf16_reduced_precision_reduction = False
                if fp16 is True:
                    m.allow_fp16_reduced_precision_reduction = False
                if acc16 is True:
                    m.allow_fp16_accumulation = False
                _FLAGS["changed"] = (bf16 is True, fp16 is True, acc16 is True)
        except (AttributeError, RuntimeError, TypeError):
            return None
        _FLAGS["holds"] += 1
        return _FlagHold()


def _exact_reductions_off(hold) -> None:
    """Let go of a hold from _exact_reductions_on (twice, or None, is a no-op)."""
    if not isinstance(hold, _FlagHold) or not hold.live:
        return
    hold.live = False
    with _FLAGS_LOCK:
        _FLAGS["holds"] = max(0, _FLAGS["holds"] - 1)
        if _FLAGS["holds"]:
            return
        changed, _FLAGS["changed"] = _FLAGS["changed"], None
        if not changed:
            return
        m = torch.backends.cuda.matmul
        if changed[0]:
            m.allow_bf16_reduced_precision_reduction = True
        if changed[1]:
            m.allow_fp16_reduced_precision_reduction = True
        if len(changed) > 2 and changed[2]:
            m.allow_fp16_accumulation = True


def _flat2d(t: torch.Tensor) -> torch.Tensor:
    # With zero input/output channels, reshape(-1, 0) cannot infer the leading dimension.
    return t.reshape(math.prod(t.shape[:-1]), t.shape[-1])


def matmul_checks(name: str, args, kwargs, out):
    """List of (A, B, C, bias, alpha, beta) products to verify for one op call."""
    if name == "aten::_grouped_mm":
        return _grouped_checks(args, kwargs, out)
    if name in ("aten::mm", "aten::bmm"):
        return [(args[0], args[1], out, None, 1.0, 1.0)]
    if name == "aten::addmm":
        beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        return [(args[1], args[2], out, args[0], float(alpha), float(beta))]
    if name == "aten::baddbmm":                                   # beta * input + alpha * (b1 @ b2), batched
        beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        return [(args[1], args[2], out, args[0], float(alpha), float(beta))]
    if name == "aten::addbmm":                                    # beta * input + alpha * sum_b (b1[b] @ b2[b])
        b1, b2 = args[1], args[2]
        beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        bb, m, k = b1.shape
        A = b1.permute(1, 0, 2).reshape(m, bb * k)                   # one matmul over the stacked k
        return [(A, b2.reshape(bb * k, b2.shape[-1]), out, args[0], float(alpha), float(beta))]
    if name == "aten::mv":                                        # matrix @ vector: a one-column matmul
        return [(args[0], args[1].unsqueeze(-1), out.unsqueeze(-1), None, 1.0, 1.0)]
    if name == "aten::addmv":
        beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        bias = args[0]
        bias = bias.unsqueeze(-1) if bias.dim() == 1 else bias       # (m,) -> one value per row
        return [(args[1], args[2].unsqueeze(-1), out.unsqueeze(-1), bias, float(alpha), float(beta))]
    if name in ("aten::_mps_linear", "aten::linear"):            # y = x W^T + b
        x, w = args[0], args[1]
        b = args[2] if len(args) > 2 else kwargs.get("bias")
        return [(_flat2d(x), w.t(), _flat2d(out), b, 1.0, 1.0)]
    if name in ("aten::linear_backward", "aten::mps_linear_backward"):   # (input, grad_out, weight, mask)
        x, gy, w = args[0], args[1], args[2]
        grad_x, grad_w = out[0], out[1]
        checks = []
        if isinstance(grad_x, torch.Tensor) and grad_x.numel():
            checks.append((_flat2d(gy), w, _flat2d(grad_x), None, 1.0, 1.0))        # dX = dY W
        if isinstance(grad_w, torch.Tensor) and grad_w.numel():
            checks.append((_flat2d(gy).t(), _flat2d(x), grad_w, None, 1.0, 1.0))    # dW = dY^T X
        return checks
    return []


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype F.linear would compute in here: the autocast dtype if autocast is on for x's device."""
    dev = x.device.type
    try:
        if torch.is_autocast_enabled(dev):
            return torch.get_autocast_dtype(dev)
    except (TypeError, RuntimeError, AttributeError):        # older PyTorch: no device argument
        if dev == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
        if dev == "cpu" and torch.is_autocast_cpu_enabled():
            return torch.get_autocast_cpu_dtype()
    return x.dtype


def _acc_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b for bf16/fp16 a and b, with the fp32 result (the accumulator) instead of a rounded one.
    Autocast is off inside: backward passes often run under it, and it would round the result."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        return _acc_mm_raw(a, b)


def _acc_mm_raw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_cuda:
        try:
            return torch.ops.aten.mm.dtype(a, b, torch.float32)
        except (RuntimeError, NotImplementedError, AttributeError, TypeError):
            pass                                 # older PyTorch: the same in fp32, with TF32 off
        mm = torch.backends.cuda.matmul
        saved = mm.allow_tf32
        mm.allow_tf32 = False
        try:
            return a.float() @ b.float()
        finally:
            mm.allow_tf32 = saved
    return a.float() @ b.float()


_DTENSOR: dict = {}


def _is_dtensor(t) -> bool:
    """A DTensor (tensor parallel, FSDP2). An isinstance check, which torch.compile can trace (attribute probes
    on a DTensor being traced don't see its attributes, so they said no and the DTensor reached a custom op)."""
    if t is None:
        return False
    cls = _dtensor_class()
    if cls is not None:
        return isinstance(t, cls)
    return hasattr(t, "to_local") and hasattr(t, "placements") and hasattr(t, "device_mesh")


def _dtensor_class():
    """torch.distributed.tensor.DTensor (None without it), looked up once. protect() looks it up before any
    compiled forward runs: inside a compiled activation checkpoint, writing this cache is a side effect
    torch.compile(fullgraph=True) refuses."""
    if "cls" not in _DTENSOR:
        try:
            from torch.distributed.tensor import DTensor
        except Exception:  # noqa: BLE001  (a PyTorch build without distributed)
            DTensor = None
        _DTENSOR["cls"] = DTensor
    return _DTENSOR["cls"]


def _placement_role(p, dims):
    """One mesh dimension's placement of a tensor, as R, P or (S, the matmul dimension it shards)."""
    if p.is_replicate():
        return "R"
    if p.is_partial():
        return "P" if str(getattr(p, "reduce_op", "sum")) in ("sum", "avg") else None
    if p.is_shard() and type(p).__name__ == "Shard":          # not _StridedShard (FSDP2 + TP interleaving)
        d = p.dim % len(dims)
        return ("S", dims[d])
    return None


def _dtensor_local(A, B, C, bias):
    """Tensor parallel: when C, A, B (and the bias) are DTensors, this rank's piece of the check. Returns the
    local tensors when this rank's output shard is exactly the matmul of its local operands (plus its local
    bias), which is how DTensor runs mm, addmm and bmm under the usual shardings: row or column parallel,
    a sharded contraction (the output a partial sum, checked before the all-reduce), batch sharding. Returns a
    reason string when it isn't, and the check is skipped."""
    if not any(_is_dtensor(t) for t in (A, B, C, bias)):
        return None
    if not all(_is_dtensor(t) for t in (A, B, C)) or (bias is not None and not _is_dtensor(bias)):
        return "DTensor and plain tensors mixed in one matmul"
    mesh = C.device_mesh
    if any(t.device_mesh != mesh for t in (A, B, C) + ((bias,) if bias is not None else ())):
        return "operands on different device meshes"
    batch = C.dim() == 3
    da = ("b", "m", "k") if A.dim() == 3 else ("m", "k")
    db = ("b", "k", "n") if B.dim() == 3 else ("k", "n")
    dc = ("b", "m", "n") if batch else ("m", "n")
    dbias = None if bias is None else (("n",) if bias.dim() == 1 else ("m", "n") if bias.dim() == 2 else
                                       ("b", "m", "n"))
    for i in range(mesh.ndim):
        ra, rb, rc = (_placement_role(t.placements[i], d) for t, d in ((A, da), (B, db), (C, dc)))
        if None in (ra, rb, rc):
            return f"placements {A.placements}, {B.placements} -> {C.placements}"
        ok = ((ra, rb, rc) == ("R", "R", "R")
              or (rc == ("S", "m") and ra == ("S", "m") and rb == "R")
              or (rc == ("S", "n") and ra == "R" and rb == ("S", "n"))
              or (rc == ("S", "b") and ra == ("S", "b") and rb in (("S", "b"), "R") and (rb == "R") == (B.dim() == 2))
              or (rc == "P" and ((ra == ("S", "k") and rb == ("S", "k")) or (ra, rb) in (("P", "R"), ("R", "P")))))
        if not ok:
            return f"placements {A.placements}, {B.placements} -> {C.placements}"
    if bias is not None:
        # the bias as this rank added it: DTensor turns a replicated bias into the output's placement locally
        # (a partial output gets bias / ranks, a column-sharded one its slice), with no communication
        from torch.distributed.tensor import Partial, Replicate, Shard
        target = []
        for i in range(mesh.ndim):
            rc = _placement_role(C.placements[i], dc)
            pb = bias.placements[i]
            rbias = _placement_role(pb, dbias)
            if rc == "P":
                want, want_role = Partial(C.placements[i].reduce_op), "P"
            elif rc == ("S", "n"):
                want, want_role = Shard(bias.dim() - 1), ("S", "n")
            elif rc != "R" and bias.dim() > 1 and rbias == rc:
                want, want_role = pb, rc
            else:
                want, want_role = Replicate(), "R"
            if rbias == want_role:
                target.append(pb)
            elif pb.is_replicate():
                target.append(want)
            else:
                return f"placements {A.placements}, {B.placements} -> {C.placements} with bias {bias.placements}"
        to_partial = [i for i, t in enumerate(target) if t.is_partial() and not bias.placements[i].is_partial()]
        sharded = [Replicate() if i in to_partial else t for i, t in enumerate(target)]
        with torch.no_grad():
            if tuple(sharded) != tuple(bias.placements):
                bias = bias.redistribute(mesh, sharded)          # replicate -> shard: a local slice
            bl = bias.to_local()
            for i in to_partial:                                 # replicate -> partial: what DTensor's addmm does
                part = getattr(target[i], "_partition_value", None)
                if part is None:
                    return "a replicated bias on a partial output (this PyTorch's DTensor doesn't say how it splits)"
                bl = part(bl, mesh, i)
    a, b, c = (t.to_local() for t in (A, B, C))
    if bias is None:
        bl = None
    if a.shape[-1] != b.shape[-2] or tuple(c.shape[-2:]) != (a.shape[-2], b.shape[-1]):
        return f"local shapes {tuple(a.shape)} @ {tuple(b.shape)} -> {tuple(c.shape)} don't line up"
    if bl is not None:
        try:
            ok = torch.broadcast_shapes(tuple(bl.shape), tuple(c.shape)) == tuple(c.shape)
        except RuntimeError:
            ok = False
        if not ok:
            return f"local bias {tuple(bl.shape)} doesn't broadcast to output {tuple(c.shape)}"
    return a, b, c, bl


def _stamp(t):
    """What would show that a tensor changed since: its version counter, storage size and address."""
    if t is None:
        return None
    try:
        return (t._version, t.untyped_storage().nbytes(), t.data_ptr())
    except Exception:  # noqa: BLE001  (a tensor subclass without plain storage)
        return ("unreadable",)


def _content(t):
    """Three integer sums of a tensor's bits (recompute.fingerprint), queued on its stream: a change the version
    counter doesn't see (an update through .data, a write by faulty hardware or another kernel) changes them. A
    transposed 2-D view is read through its contiguous transpose, so no copy is made."""
    if t is None:
        return None
    from .recompute import fingerprint
    try:
        x = t.detach()
        if not x.is_contiguous() and x.dim() == 2 and x.t().is_contiguous():
            x = x.t()
        return fingerprint(x)
    except Exception:  # noqa: BLE001
        return None


class _CheckedLinear(torch.autograd.Function):
    """y = x W^T + b with its forward and both backward matmuls checked (fast mode, see protect()).

    Inputs arrive already in the compute dtype and autocast is off inside, so the matmuls here are
    exactly the ones autocast would run (same dtype, same kernels); the casts happen outside, as
    ordinary differentiable ops, so gradients reach the fp32 weights the usual way.
    """

    @staticmethod
    def forward(ctx, x, w, b, guard, name):
        ctx.save_for_backward(x, w)
        ctx.has_bias = b is not None
        ctx.guard, ctx.name = guard, name
        return _checked_forward(guard, name, x, w, b)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        need = ctx.needs_input_grad
        gx, gw, gb = _checked_backward(ctx.guard, ctx.name, x, w, gy, need[0], need[1], ctx.has_bias and need[2])
        return gx, gw, gb, None, None


def _checked_forward(guard, name, x, w, b):
    """y = x W^T + b with the matmul checked (fast mode; the eager autograd function and the compiled custom
    operator both run this). Inputs are in the compute dtype; autocast is off."""
    if guard.precise:
        # the fp32 result, checked, then rounded once (bias added in fp32 first, as cuBLAS does)
        x2 = _flat2d(x)
        y = guard._fused_mm(name, "forward", x2, w.t(), b)
        if y is not None:
            return y.view(*x.shape[:-1], w.shape[0])
        y32 = _acc_mm(x2, w.t())
        if b is not None:
            y32 = y32.add_(b.float())
        guard._accumulator(name, "forward", x2, w.t(), y32, b)
        return y32.to(x.dtype).view(*x.shape[:-1], w.shape[0])
    if b is None or x.dim() == 2 or (x.is_contiguous() and b.dim() == 1 and b.is_contiguous()):
        # what at::linear does here (PyTorch 2.8 to 2.14): one matmul with the bias fused in (one rounding)
        y = torch.nn.functional.linear(x, w, b)
        guard._linear_check(name, "forward", _flat2d(x), w.t(), _flat2d(y), b)
        return y
    # otherwise at::linear runs the matmul, rounds, then adds the bias (rounded again): check the
    # matmul by itself, then add the bias the same way
    y = torch.nn.functional.linear(x, w)
    guard._linear_check(name, "forward", _flat2d(x), w.t(), _flat2d(y), None)
    return y + b                     # not in place: the queued check (and any evidence) holds y as checked


def _checked_backward(guard, name, x, w, gy, need_x: bool, need_w: bool, need_b: bool):
    """dX, dW and db of a checked linear layer, with both matmuls checked while the step is checked. A backward
    that runs after its step ended (backward outside at_step) is computed the same way but not checked: its
    checks would be judged under a later step's number and settings. Counted in stats["backward_outside_step"]."""
    gy2 = _flat2d(gy)
    gx = gw = gb = None
    if not guard._active:
        guard._backward_outside_step()
        if need_x:
            gx = gy2.mm(w).view(x.shape)
        if need_w:
            gw = gy2.t().mm(_flat2d(x)).to(w.dtype)
        if need_b:
            gb = gy2.sum(0)
        return gx, gw, gb
    if guard.precise:
        if need_x:
            gx = guard._fused_mm(name, "backward dX", gy2, w, None)
        if need_x and gx is not None:
            gx = gx.view(x.shape)
        elif need_x:
            gx32 = _acc_mm(gy2, w)
            guard._accumulator(name, "backward dX", gy2, w, gx32, None)
            gx = gx32.to(gy2.dtype).view(x.shape)
        if need_w:
            x2 = _flat2d(x)
            gw = guard._fused_mm(name, "backward dW", gy2.t(), x2, None)
        if need_w and gw is None:
            gw32 = _acc_mm(gy2.t(), x2)
            guard._accumulator(name, "backward dW", gy2.t(), x2, gw32, None)
            gw = gw32.to(w.dtype)
        elif need_w and gw.dtype != w.dtype:
            gw = gw.to(w.dtype)
        if need_b:
            gb = gy2.sum(0)
        return gx, gw, gb
    if need_x:
        gx2 = gy2.mm(w)                                          # dX = dY W
        guard._linear_check(name, "backward dX", gy2, w, gx2, None)
        gx = gx2.view(x.shape)
    if need_w:
        x2 = _flat2d(x)
        gw = gy2.t().mm(x2)                                      # dW = dY^T X
        guard._linear_check(name, "backward dW", gy2.t(), x2, gw, None)
    if need_b:
        gb = gy2.sum(0)
    return gx, gw, gb


def _autograd_node() -> Optional[str]:
    """Name of the autograd node running right now (None in the forward pass, and in the forward that
    activation checkpointing recomputes during backward, reentrant or not: that is forward work)."""
    try:
        node = torch._C._current_autograd_node()
    except AttributeError:
        return None
    if node is None:
        return None
    name = node.name()
    if "CheckpointFunction" in name:
        return None
    ck = _checkpoint_frame()
    return None if ck is not None and ck[0] == "recompute" else name


def _checkpoint_frame():
    """("forward", frame) while a non-reentrant torch.utils.checkpoint call runs its forward, ("recompute", frame)
    while backward recomputes it; the frame is that call's own record (_CheckpointFrame). None otherwise, and when
    this PyTorch's checkpoint isn't built that way. Found through the saved-tensor hooks the checkpoint installs:
    its forward's hook holds the frame, its recompute's hook a weak reference to it."""
    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:  # noqa: BLE001  (a PyTorch without it)
        return None
    if not hooks:
        return None
    pack = getattr(hooks[0], "__wrapped__", hooks[0])        # (the recompute's is wrapped by torch._dynamo.disable)
    owner = getattr(pack, "__qualname__", "").split(".", 1)[0]
    if owner not in ("_checkpoint_hook", "_recomputation_hook"):
        return None
    try:
        cells = dict(zip(pack.__code__.co_freevars, (c.cell_contents for c in pack.__closure__ or ())))
    except (AttributeError, ValueError):
        return None
    if owner == "_checkpoint_hook":
        frame = cells.get("frame")
        return None if frame is None else ("forward", frame)
    ref = cells.get("target_frame_ref")
    frame = ref() if callable(ref) else None
    return None if frame is None else ("recompute", frame)


def _reentrant_checkpoint_call():
    """The autograd node of the reentrant torch.utils.checkpoint call whose first forward is running (it runs
    inside CheckpointFunction.forward, under no_grad), or None. A walk up the Python stack, made only on checked
    steps when gradients are off."""
    mod = sys.modules.get("torch.utils.checkpoint")
    code = getattr(getattr(getattr(mod, "CheckpointFunction", None), "forward", None), "__code__", None)
    if code is None:
        return None
    f = sys._getframe(1)
    for _ in range(256):
        if f is None:
            return None
        if f.f_code is code:
            return f.f_locals.get("ctx")
        f = f.f_back
    return None


def _checked_path(guard) -> bool:
    """Whether a protected layer's forward runs checked (fast mode). While its step is checked, except in an
    activation-checkpoint recompute: that makes the choice the forward it repeats made, remembered per
    checkpointed call (SDCGuard._remember). So a non-reentrant recompute saves the same tensors as its forward
    (anything else is a CheckpointError), and a backward after the step ended is counted in
    stats["backward_outside_step"] whichever kind of checkpoint ran it. A reentrant recompute inside a checked
    step is checked even when its forward wasn't: it builds its own graph, so nothing has to match."""
    ck = _checkpoint_frame()
    if ck is not None:
        kind, frame = ck
        if kind == "forward":
            if guard._active:
                guard._remember(frame)
            return guard._active
        return guard._remembered(frame)
    try:
        node = torch._C._current_autograd_node()
    except AttributeError:
        node = None
    if node is not None and "CheckpointFunction" in node.name():        # a reentrant recompute
        return guard._active or guard._remembered(node)
    if guard._active and not torch.is_grad_enabled():
        call = _reentrant_checkpoint_call()
        if call is not None:
            guard._remember(call)
    return guard._active


def _forget(table: dict, key: int, ref) -> None:
    """Weak-reference callback: a remembered checkpoint call is gone (its graph was freed)."""
    if table.get(key) is ref:
        del table[key]


def _transposed_linear_class():
    """Hugging Face's Conv1D (GPT-2, GPT, DistilGPT2): a linear layer with its weight stored (in, out). Only
    looked up when transformers is already imported, so sdcguard never imports it."""
    mod = sys.modules.get("transformers.pytorch_utils")
    return getattr(mod, "Conv1D", None) if mod is not None else None


def _is_transposed_linear(mod: torch.nn.Module) -> bool:
    cls = _transposed_linear_class()
    if cls is None or not isinstance(mod, cls):
        return False
    w = getattr(mod, "weight", None)
    return isinstance(w, torch.Tensor) and w.dim() == 2 and getattr(mod, "nf", None) == w.shape[1]


def _linear_like(mod: torch.nn.Module) -> bool:
    return isinstance(mod, torch.nn.Linear) or _is_transposed_linear(mod)


def _plain_forward(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The layer's own forward: nn.Linear's, or Conv1D's for a transposed linear layer."""
    if mod.__dict__.get("_sdcguard_transposed"):
        return type(mod)._sdcguard_base.forward(mod, x)
    return torch.nn.Linear.forward(mod, x)


def _unsupported_linear(mod: torch.nn.Module) -> Optional[str]:
    """Why fast mode can't take over this nn.Linear or Conv1D (None if it can). The checked path computes the
    plain linear itself, so a layer whose forward does anything else must keep its own forward."""
    from . import _swap
    if _swap.is_guarded(mod):
        return "already checked by another guard"
    if _is_transposed_linear(mod):
        if type(mod).forward is not _transposed_linear_class().forward:
            return f"{type(mod).__name__} has its own forward"
    elif type(mod).forward is not torch.nn.Linear.forward:
        return f"{type(mod).__name__} has its own forward"
    if "forward" in mod.__dict__:
        return "forward replaced on the instance (a hook or wrapper)"
    return None


def _autocast_cache_on() -> bool:
    try:
        return bool(torch.is_autocast_cache_enabled())
    except AttributeError:                        # older PyTorch
        return True


# autocast drops its cast cache when the outermost autocast region ends (torch.clear_autocast_cache); the
# fast mode's cast cache follows it, so a step with several autocast regions casts as often as autocast does
_CLEAR = {"orig": None}
_IN_STEP: "weakref.WeakSet" = weakref.WeakSet()


def _follow_autocast_cache() -> None:
    if _CLEAR["orig"] is not None:
        return
    orig = torch.clear_autocast_cache

    def clear_autocast_cache():
        orig()
        for g in list(_IN_STEP):
            g._casts.clear()

    clear_autocast_cache.__doc__ = orig.__doc__
    _CLEAR["orig"] = orig
    torch.clear_autocast_cache = clear_autocast_cache


def _fast_forward(mod: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """A protected layer's forward (see SDCGuard.protect): its plain forward, except on checked steps. A
    method of the layer's class (sdcguard._swap), so it always runs with the weights of the layer it's called
    on, copies and replicas included."""
    guard = mod._sdcguard_guard
    if not guard._active:
        # (while a checkpointed call whose forward ran checked is alive, its recompute may be this call: see
        # _checked_path; the compiled graph never reads that table, so it can't make the graph recompile)
        if torch.compiler.is_compiling() or not guard._checked_calls:
            return _plain_forward(mod, x)
        return _fast_checked(guard, mod, x)
    if torch.compiler.is_compiling():
        return _compiled_checked(guard, mod, x)
    return _fast_checked(guard, mod, x)


def _compiled_checked(guard, mod, x):
    """A checked step inside torch.compile (fullgraph=True included): the same checked forward as eager, as one
    custom operator the compiler keeps as an opaque call (see _compiled.py). Unchecked steps compile to the
    layer's plain forward, so they keep every compiler optimization."""
    inputs = (x, mod.weight) + ((mod.bias,) if mod.bias is not None else ())
    eligible = (torch.float16, torch.bfloat16, torch.float32)
    if any(t.dtype not in eligible for t in inputs):
        return _plain_forward(mod, x)
    tp = any(_is_dtensor(t) for t in inputs)
    if tp:
        from . import _compiled
        if not (_compiled.TENSOR_PARALLEL and all(_is_dtensor(t) for t in inputs)):
            # DTensor can't run the operator here (no sharding rules in this PyTorch, or DTensor and plain tensors
            # mixed): recorded (Dynamo replays this write on every run of the graph) so the checked step says so
            # instead of passing as checked
            guard._compiled_tp_skipped[mod._sdcguard_name] = True
            return _plain_forward(mod, x)
    if not torch.is_autocast_enabled(x.device.type) and any(t.dtype != x.dtype for t in inputs):
        return _plain_forward(mod, x)
    dt = _compute_dtype(x)
    if dt not in UNIT_ROUNDOFF:
        return _plain_forward(mod, x)
    from . import _compiled
    w = mod.weight.to(dt)
    if mod.__dict__.get("_sdcguard_transposed"):
        w = w.t()
    b = mod.bias.to(dt) if mod.bias is not None else None
    # with DTensors, DTensor runs the operator on this rank's local shards (its sharding rules, _compiled.py)
    return _compiled.checked_linear(x.to(dt), w, b, guard._op_id, mod._sdcguard_name, tp)


@torch.compiler.disable
def _fast_checked(guard, mod, x):
    # Outside a checked step this runs only for a checkpoint recompute of a forward that ran checked: the same
    # computation, with no check queued (see _linear_check)
    if not _checked_path(guard):
        return _plain_forward(mod, x)
    # Autocast only converts eligible floating inputs. In particular a float64 layer must stay float64,
    # and an invalid mixed-dtype call must keep PyTorch's error instead of being silently repaired by our
    # casts. Delegate those cases to the original operator, which owns the complete dtype contract.
    inputs = (x, mod.weight) + ((mod.bias,) if mod.bias is not None else ())
    eligible = (torch.float16, torch.bfloat16, torch.float32)
    if any(t.dtype not in eligible for t in inputs):
        return _plain_forward(mod, x)
    if not torch.is_autocast_enabled(x.device.type) and any(t.dtype != x.dtype for t in inputs):
        return _plain_forward(mod, x)
    dt = _compute_dtype(x)
    if dt not in UNIT_ROUNDOFF:                  # fp32/TF32: not checked
        return _plain_forward(mod, x)
    if guard.precise and any(_is_dtensor(t) for t in inputs):
        if guard._active:                        # (counted by the forward, not again by a recompute after the step)
            guard._skip_dtensor("precise=True on tensor-parallel weights")  # its fp32-output matmul has no DTensor rule
        return _plain_forward(mod, x)
    w = guard._cast(mod.weight, dt)
    if mod.__dict__.get("_sdcguard_transposed"):
        w = w.t()                                # Conv1D stores (in, out); the check works on (out, in)
    b = guard._cast(mod.bias, dt) if mod.bias is not None else None
    with torch.autocast(device_type=x.device.type, enabled=False):
        return _CheckedLinear.apply(x.to(dt), w, b, guard, mod._sdcguard_name)


class SDCGuard(TorchDispatchMode):
    """Checksums sampled matmuls while active.

    sample_rate: share of matmul ops checked while the guard is on (1.0 = all).
    z: alert threshold, in units of the estimated rounding noise.
    immediate: read every check back right away (slower on GPUs; alerts raise inside the op).
    exact_reductions: on NVIDIA, keep bf16/fp16 matmul partial sums in fp32 while the guard is
        active, fp16 accumulation included (see the module docstring). Default True.
    on_check_start(op_name, C) -> token and on_check(op_name, token, result): optional
        callbacks for experiments (e.g. matching injected faults to checks).
    on_result(detail, result): optional, called for every check with where/pass/shapes
        (for diagnostics).
    abs_floor: differences below this are not judged (default checksum.ABS_FLOOR = 1e-20).
    every: check schedule for on_step(): every N-th step (default 10).
    precise: with protect(), check each linear layer's fp32 accumulator before it's rounded (much smaller
        errors show; see the module docstring). on_accumulator(layer, pass, C32): optional, called with each
        fp32 result before its check (fault-injection experiments corrupt it there). precise="fused": the same
        check with the block sums taken inside the matmul itself (a Triton GEMM, fused.py), instead of a
        second, fp32-output matmul; falls back to precise=True where Triton or a GPU isn't available.
        Experimental, not yet measured on GPUs.
    fresh_signs: give every checked step a new +-1 pattern for the signed checksums (default True). Two
        wrong outputs in a block can cancel in both sums for one pattern (two values swapped where the
        pattern has the same sign); a fault that repeats that way on every step then still shows on a
        later step, instead of hiding for the whole run.
    attention: also check fused attention (scaled_dot_product_attention) in the default mode (not with
        protect()): in the backward pass, the sum of dV over the keys must equal the sum of dO over the
        queries; in the forward pass, every output must lie within the range of its head's values (see
        attention.py). Experimental and off by default until calibrated on GPUs. Calls with dropout, a mask
        or bias tensor, or variable-length batches are counted in stats["attention_skipped"].
    evidence_dir: save the inputs and output of the first `max_evidence` matmuls that fail their check (with the
        GPU's identity, the software stack and a CPU recheck of the flagged block) as evidence_<n>.pt/.json in
        this folder, for `sdcguard replay` and `sdcguard casefile`. The event's detail names the file. Costs memory
        on checked steps: every checked matmul's operands stay referenced until the step's results are read
        back. Matmuls checked by precise="fused" aren't saved (their check is of the accumulator, not the output).
        An operand that changed between the check and the read-back (an in-place op such as a residual add_ or
        relu_, an update through .data, an optimizer step inside at_step, or a weight FSDP2 freed after use) isn't
        saved as evidence: each kept operand's version, storage and a fingerprint of its contents (one extra read
        of it, on checked steps with evidence_dir only) are compared at the read-back. The event says so,
        stats["evidence_errors"] counts it, and from then on the guard keeps a copy of every checked matmul's
        operands at check time, so a fault that comes back is saved as it was checked.
    evidence_copy: keep that copy from the start (host memory, copied without waiting for the GPU; costs host
        memory and PCIe bandwidth on checked steps). Needed for evidence under FSDP2 and in-place models.
    confirm: with evidence_dir, recompute a failed matmul this many times right away, on the same GPU, with the
        settings its step ran with (TF32, reductions, BLAS library) and the same check, before training goes on.
        The alert says whether the error came back (detail["confirmed_in_job"]), and the evidence file keeps
        those recomputations, which `sdcguard casefile` counts as same-GPU replays run the incident's way. Costs
        `confirm` matmuls and checks per saved failure, nothing otherwise. Default 3; 0 turns it off.
    recompute: on checked steps, also compare every activation-checkpointed block's forward outputs with the
        recompute training already does in the backward pass, through fingerprints of their bits (recompute.py).
        Covers every operation inside those blocks (attention, normalization, activations, MoE routing), not
        just matmuls; needs gradient checkpointing and deterministic kernels (RecomputeCheck.calibrate() says
        whether clean recomputes match). Experimental and off by default.
    replicas: at the start of checked steps, compare an exact checksum of the weights of the model given to
        protect() or watch() across data-parallel ranks (replica.py): after an optimizer step every copy must
        match bit for bit, so a rank whose update or weight memory went wrong (optimizer math, a bad write) is
        named. A collective, so it runs only on the checked steps every rank shares: the steps that are a
        multiple of `every`, and the steps of an after_restart() window (call it on every rank). A window
        observe() opened doesn't run it: each rank sees its own loss, and a rank that spiked alone would wait
        in the comparison while the others don't. at_step() on any other step skips it with a warning, counted
        in stats["replica_skipped"] (set `every` to the schedule you check on). For DDP, or for HSDP with
        `replica_group` set to the group whose ranks hold identical shards; fully sharded weights (DTensor)
        without a replica group, and a guard given no model, are skipped with a warning and counted in
        stats["replica_skipped"].
    """

    def __init__(
        self,
        sample_rate: float = 1.0,
        block: int = 128,
        z: float = 6.0,
        on_alert: Optional[Callable[[SDCEvent], None]] = None,
        raise_on_alert: bool = False,
        on_check: Optional[Callable] = None,
        on_check_start: Optional[Callable] = None,
        log_path: Optional[str] = None,
        immediate: bool = False,
        max_pending: int = 4096,
        seed: int = 0,
        exact_reductions: bool = True,
        on_result: Optional[Callable] = None,
        abs_floor: Optional[float] = None,
        every: int = 10,
        precise: bool = False,
        on_accumulator: Optional[Callable] = None,
        fresh_signs: bool = True,
        attention: bool = False,
        recompute: bool = False,
        evidence_dir: Optional[str] = None,
        max_evidence: int = 3,
        evidence_copy: bool = False,
        confirm: int = 3,
        replicas: bool = False,
        replica_group=None,
    ):
        super().__init__()
        self.attention = attention
        self.every = every
        self.fresh_signs = fresh_signs        # a new signed-checksum pattern every checked step
        self._epoch = 0
        self.precise = precise                # check fp32 accumulators (protect() only; see the module docstring)
        self.on_accumulator = on_accumulator  # experiments: on_accumulator(layer, pass, C32) before each check
        self._nfma: dict = {}
        self._dense: Optional[tuple] = None          # (first, last + 1) steps of a restart window (every rank's)
        self._spike: Optional[tuple] = None          # (first, last + 1) steps of an observe() window (this rank's)
        self.abs_floor = ABS_FLOOR if abs_floor is None else abs_floor
        self.exact_reductions = exact_reductions
        self.on_result = on_result
        self._restore_flags = None
        self._depth = 0
        self._is_matmul: dict = {}
        self.sample_rate = sample_rate
        self.block = block
        self.z = z
        self.on_alert = on_alert
        self.raise_on_alert = raise_on_alert
        self.on_check = on_check
        self.on_check_start = on_check_start
        self.immediate = immediate
        self.max_pending = max_pending
        self.log = EventLog(log_path)
        self.step: Optional[int] = None
        # skipped_oom: checks skipped because the GPU ran out of memory for them (training goes on)
        self.stats = {"matmuls_seen": 0, "checked": 0, "checked_fp32": 0, "unjudged": 0, "skipped_dtype": 0,
                      "skipped_oom": 0, "alerts": 0, "attention_checked": 0, "attention_skipped": 0}
        if precise == "fused":
            self.stats.update(checked_fused=0, fused_fallbacks=0)
        self.evidence_dir = evidence_dir      # failed checks' operands go here (see the class docstring)
        self.max_evidence = max_evidence
        self.evidence: list = []              # paths written so far
        self._kept: list = []                 # (queue entry, operands, stamps) of this step's checks, when evidence_dir
        self.evidence_copy = evidence_copy    # copy operands at check time (see the class docstring)
        self.confirm = max(0, int(confirm))   # recomputations of a saved failure, right away (see the docstring)
        self._fused_off = False               # precise="fused": the kernel failed once, precise=True from then on
        self.fused_why: Optional[str] = None
        # experiments: fused_inject(layer, pass, (m, n)) -> (row, col, value) or None, a fault in the accumulator
        self.fused_inject: Optional[Callable] = None
        self.attention_skipped: dict[str, int] = {}     # why attention calls weren't checked
        self._lock = threading.RLock()        # autograd runs one thread per GPU when a process has several
        self._streams: dict = {}              # CUDA streams the checks ran on (the flush waits for them)
        # Fast checks on NVIDIA GPUs (see _group.py), per (device, dtype): each check's GPU work is
        # queued at its matmul; their results are searched and read back together at the flush.
        self._groups: dict = {}
        self.ops_seen: dict[str, int] = {}    # which matmul ops this device actually runs
        self.z_values: list[float] = []      # sample of clean max-z values, for calibration
        self.max_z_seen = 0.0                # largest finite z ever seen (calibration margin)
        self._z_cap = 20000
        self._rng = random.Random(seed)
        self._pending: list = []
        self._flushed: list = []                    # (see _flush)
        self._module: Optional[str] = None
        self._hooks: list = []
        self._identity: dict = {}
        self.unchecked_ops: dict = {}         # op -> calls seen that the guard can't check (UNCHECKED_MATMUL_OPS)
        self._compiled_tp_skipped: dict = {}  # tensor-parallel layers that ran unchecked inside torch.compile
        self._watch: dict = {}                # observe(): recent loss and gradient-norm values
        self.anomalies: list = []             # observe(): spikes that started a checked window
        self._protected: dict = {}            # module -> name, for modules wrapped by protect()
        from . import _compiled
        self._op_id = _compiled.register(self)  # how a compiled graph's custom operator finds this guard
        self.skipped: dict = {}               # layer name -> why protect() left it alone
        self._active = False                  # fast mode: True while a checked step runs
        self._casts: dict = {}                # fast mode: low-precision copies of weights, this step
        # fast mode: id -> weak reference of each activation-checkpointed call whose forward ran checked (its
        # recompute makes the same choice, see _checked_path); an entry goes when its call's graph is freed
        self._checked_calls: dict = {}
        self.stats["recompute_compared"] = 0
        self.stats["recompute_mismatches"] = 0
        self.stats["recompute_steps_without_blocks"] = 0
        self.stats["replica_checks"] = 0
        self.stats["replica_mismatches"] = 0
        self.stats["replica_skipped"] = 0
        self.replicas = replicas
        self.replica_group = replica_group
        self._replica_model: Optional[torch.nn.Module] = None
        self._recompute = None                # recompute comparison by fingerprints (recompute=True)
        if recompute:
            from .recompute import RecomputeCheck
            # (through a lambda: an on_alert set after construction reaches recompute alerts too)
            self._recompute = RecomputeCheck(on_alert=lambda ev: self.on_alert(ev) if self.on_alert else None,
                                             log=self.log)
            self._recompute.enabled = False   # only on checked steps

    def _every_rank_checks(self, step: int) -> bool:
        """A checked step every rank shares: a multiple of `every`, or in an after_restart() window. Not a step
        checked only because observe() saw this rank's own loss or gradient norm spike."""
        if self._dense is not None and self._dense[0] <= step < self._dense[1]:
            return True
        return bool(self.every) and step % self.every == 0

    def _in_spike_window(self, step: int) -> bool:
        """In the window of checked steps observe() opened (this rank's own)."""
        return self._spike is not None and self._spike[0] <= step < self._spike[1]

    def _replica_check(self, step: int) -> None:
        """replicas=True: compare the weights' checksum across data-parallel ranks. A collective, so only on the
        steps every rank checks (_every_rank_checks): a rank that enters it alone hangs the job."""
        if not self.replicas:
            return
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size(self.replica_group) < 2:
            return
        if not self._every_rank_checks(step):
            # expected on a step of this rank's own observe() window; any other step is at_step() called off the
            # `every` schedule, whose comparison would otherwise be lost without a word
            if not self._in_spike_window(step):
                self.stats["replica_skipped"] += 1
                if not self.__dict__.get("_warned_off_schedule"):
                    self._warned_off_schedule = True
                    warnings.warn(f"sdcguard: replicas=True compares the weights only on the checked steps every rank "
                                  f"shares (multiples of every={self.every}, after_restart() windows); step {step} "
                                  "isn't one, so it was skipped (set `every` to your schedule). Counted in "
                                  "stats['replica_skipped']", RuntimeWarning, stacklevel=4)
            return
        if self._replica_model is None:          # the default mode sees operations, not the model's weights
            self.stats["replica_skipped"] += 1
            if not self.__dict__.get("_warned_no_replica_model"):
                self._warned_no_replica_model = True
                warnings.warn("sdcguard: replicas=True, but the guard wasn't given the model (pass it to watch() or "
                              "protect()); skipping the replica check. Counted in stats['replica_skipped']",
                              RuntimeWarning, stacklevel=4)
            return
        from . import replica
        params = [p for p in self._replica_model.parameters()]
        if self.replica_group is None and any(replica.is_sharded(p) for p in params):
            self.stats["replica_skipped"] += 1
            if not self.__dict__.get("_warned_sharded_replicas"):
                self._warned_sharded_replicas = True
                warnings.warn("sdcguard: replicas=True, but the weights are sharded (DTensor) and no replica_group "
                              "was given, so ranks hold different shards; skipping the replica check",
                              RuntimeWarning, stacklevel=4)
            return
        suspects, event = replica.check_replicas(params, group=self.replica_group, step=step)
        self.stats["replica_checks"] += 1
        if event is None:
            return
        self.stats["replica_mismatches"] += 1
        self.stats["alerts"] += 1
        self.log.add(event)                    # every rank keeps it in its own log
        if self.on_alert and event.detail.get("reporter", True):
            self.on_alert(event)               # one rank delivers it (one webhook post, not one per rank)
        if self.raise_on_alert:
            raise SDCError(event.to_json())

    def _rc_begin(self) -> None:
        rc = self._recompute
        if rc is not None:
            rc.install()
            rc.step = self.step
            rc.enabled = True

    def _rc_end(self, raise_alerts: bool) -> None:
        """Compare the checked step's recomputes (one small read-back) and report mismatches."""
        rc = self._recompute
        if rc is None or not rc.enabled:
            return
        rc.enabled = False
        before = rc.stats["recomputes_compared"]
        try:
            events = rc.flush()
        except Exception:  # noqa: BLE001  (an on_alert callback raised: alerts are already recorded)
            if raise_alerts:
                raise
            return
        finally:
            compared = rc.stats["recomputes_compared"] - before
            self.stats["recompute_compared"] += compared
            if compared == 0:
                self.stats["recompute_steps_without_blocks"] += 1
                if self.stats["recompute_steps_without_blocks"] == 1:
                    import warnings
                    warnings.warn("sdcguard: recompute=True, but no checkpointed block was compared on this "
                                  "checked step. Is gradient checkpointing on, did the step call backward(), and "
                                  "was the model passed to protect() or watch()?", RuntimeWarning, stacklevel=3)
        self.stats["recompute_mismatches"] += len(events)
        self.stats["alerts"] += len(events)
        if raise_alerts and self.raise_on_alert and events:
            raise SDCError(events[0].to_json())

    # ------------------------------------------------------------------ setup
    def watch(self, model: torch.nn.Module) -> "SDCGuard":
        """Name the layer (e.g. 'blocks.3.attn.q') in forward-pass alerts. Optional, except with replicas=True in
        the default mode: the replica check compares the weights of the model given here (or to protect()). With
        recompute=True in the default mode, also lets the recompute check reach the checkpoint references the
        model already holds."""
        if self._recompute is not None:
            self._recompute.attach(model)
        if self.replicas:
            self._replica_model = model
        def pre(name):
            def hook(_m, _inp):
                self._module = name
            return hook

        def post(_m, _inp, _out):
            self._module = None

        for name, mod in model.named_modules():
            if _linear_like(mod):
                self._hooks.append(mod.register_forward_pre_hook(pre(name)))
                # (also when the forward raises: a name left behind would label later backward matmuls)
                self._hooks.append(mod.register_forward_hook(post, always_call=True))
        return self

    def unwatch(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def protect(self, model: torch.nn.Module) -> "SDCGuard":
        """Fast mode: check the matmuls of every nn.Linear in `model` (forward, dX and dW).

        Instead of routing every PyTorch operation through Python (the default operator-level
        hook), only the linear layers' own matmuls do, and only on checked steps. On other steps
        each layer runs its normal forward, plus two Python `if`s. Matmuls outside nn.Linear (for
        example attention written with torch.bmm, or einsum) are not checked in this mode; use the
        default mode, or coverage() to see what share of the matmul work the linear layers are.
        Activation checkpointing works on either side of a step's boundary: a recompute runs the way
        its forward ran (see _checked_path). Under torch.compile that holds for a checkpoint inside the
        compiled region; an eager checkpoint around a module compiled on its own recomputes it with the
        graph of the step its backward runs in (checked or not), so keep that forward and its backward
        in one step.

        Works with autocast (bf16/fp16), DDP, FSDP2 and tensor parallel (DTensor: each rank checks its own
        shard). Under torch.compile, fullgraph=True included, unchecked steps run the plain compiled graph;
        on checked steps each linear layer is one custom operator (sdcguard::checked_linear) that runs and
        checks its matmuls with cuBLAS, as in eager mode (see _compiled.py); tensor-parallel layers included:
        DTensor runs the operator on each rank's local shards, which it checks. Where DTensor can't (a PyTorch
        without DTensor sharding rules for custom operators, DTensor and plain tensors mixed in one layer), the
        layer runs unchecked, stats["skipped_dtensor_compiled"] counts it and a warning names it. The default
        mode (no protect()) doesn't run inside torch.compile(fullgraph=True): PyTorch can't compile under an
        operator-level hook.
        """
        from . import _swap
        _dtensor_class()                      # (looked up now, outside any compiled forward)
        if self._recompute is not None:
            self._recompute.attach(model)     # its checkpoint references (Hugging Face, checkpoint_wrapper)
        if self.replicas:
            self._replica_model = model
        direct = _swap.multihead_out_projs(model)
        for name, mod in model.named_modules():
            if _linear_like(mod) and mod not in self._protected:
                why = direct.get(id(mod)) or _unsupported_linear(mod)
                if why:                       # its own forward does more than the plain linear: leave it alone
                    self.skipped[name or type(mod).__name__] = why
                    continue
                self._protected[mod] = name or type(mod).__name__
                if _is_transposed_linear(mod):
                    mod.__dict__["_sdcguard_transposed"] = True
                mod.__dict__["_sdcguard_guard"] = self
                mod.__dict__["_sdcguard_name"] = self._protected[mod]
                _swap.take_over(mod, _fast_forward)
        return self

    def unprotect(self) -> None:
        """Undo protect(): the layers run their own forward again."""
        from . import _swap
        if self._recompute is not None:
            self._recompute.uninstall()
        for mod in self._protected:
            if mod.__dict__.get("_sdcguard_guard") is self:
                _swap.give_back(mod)
        self._protected.clear()

    def _cast(self, p: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
        """p in dtype dt, cast once and reused for as long as autocast reuses its own casts: a layer used
        twice in one autocast region then sums its weight gradient the way autocast would. Like autocast's
        cache, this one holds only leaf tensors that need gradients, is off when autocast's is
        (cache_enabled=False), and is dropped when the outermost autocast region ends, so weights changed
        in place between regions (p.data.add_(), as weight perturbation and manual updates do, which no
        version counter sees) are cast again."""
        if p.dtype == dt:
            return p
        # (a checkpoint recompute after its step: not cached, since only a step's end clears the cache)
        if not self._active or not (p.requires_grad and p.is_leaf) or not _autocast_cache_on():
            return p.to(dt)
        # Grad mode is part of the key: a cast made under no_grad (an eval forward in the same step, or
        # the first pass of reentrant activation checkpointing) has no autograd history and must not be
        # reused by a forward that needs gradients. The data pointer catches `p.data = new_tensor`.
        try:
            ptr = p.data_ptr()
        except RuntimeError:                  # no storage (torch.func transforms wrap tensors): no cache
            return p.to(dt)
        key = (id(p), dt, p._version, torch.is_grad_enabled(), ptr)
        hit = self._casts.get(key)
        if hit is None or hit[0] is not p:
            hit = self._casts[key] = (p, p.to(dt))
        return hit[1]

    def _remember(self, call) -> None:
        """An activation-checkpointed call (its frame, or a reentrant call's autograd node) whose forward ran
        checked: its recompute makes the same choice (_checked_path). Held by a weak reference, so the table
        never outlives the graphs it describes."""
        key = id(call)
        ref = self._checked_calls.get(key)
        if ref is not None and ref() is call:
            return
        try:
            self._checked_calls[key] = weakref.ref(call, functools.partial(_forget, self._checked_calls, key))
        except TypeError:                      # (not weakly referenceable; checkpoint frames and nodes always are)
            pass

    def _remembered(self, call) -> bool:
        ref = self._checked_calls.get(id(call))
        return ref is not None and ref() is call

    def _linear_check(self, name, pass_, A, B, C, bias) -> None:
        """Queue the check of one linear-layer matmul (fast mode). Outside a checked step (a checkpoint recompute
        of a forward that ran checked, see _checked_path), nothing: it would be judged as part of a later step."""
        if not self._active:
            return
        with self._lock:
            self.stats["matmuls_seen"] += 1
            self.ops_seen["linear " + pass_] = self.ops_seen.get("linear " + pass_, 0) + 1
            if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
                return
            op = "aten::addmm" if bias is not None else "aten::mm"
            self._queue(op, name if pass_ == "forward" else f"{name} ({pass_})",
                        None if pass_ == "forward" else pass_, A, B, C, bias, 1.0, 1.0)
            if self.immediate or len(self._pending) >= self.max_pending:
                self.flush()

    def _fused_mm(self, name, pass_, A, B, bias) -> Optional[torch.Tensor]:
        """precise="fused": compute A @ B (+ bias) with the fused kernel and queue the check of its fp32
        accumulator. None: not available here, or outside a checked step (the caller uses the fp32-output matmul
        instead)."""
        if self.precise != "fused" or self._fused_off or not self._active:
            return None
        from . import fused as _fused
        A, B, bias = (_group.ready(t) for t in (A, B, bias))
        if not _fused.available(A) or A.shape[1] == 0 or A.shape[0] == 0 or B.shape[1] == 0:
            return None
        if not _group.own_memory(A, B, bias):     # a DTensor or another wrapper: the kernel can't be handed it
            return None
        with self._lock:
            self.stats["matmuls_seen"] += 1
            self.ops_seen["linear " + pass_] = self.ops_seen.get("linear " + pass_, 0) + 1
            if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
                with torch.autocast(device_type=A.device.type, enabled=False):
                    C = A @ B.to(A.dtype)                     # not sampled: the ordinary matmul, unchecked
                return C if bias is None else C.add_(bias.to(C.dtype))
            inj = self.fused_inject(name, pass_, (A.shape[0], B.shape[1])) if self.fused_inject else None
            fp64 = os.environ.get("SDCGUARD_FUSED_FP64") == "1"     # the first version, for comparisons
            try:
                with torch.autocast(device_type=A.device.type, enabled=False):
                    if fp64:
                        C, stats = _fused.fused_mm(A, B, bias, epoch=self._epoch, inject=inj)
                    else:
                        C, st = _fused.fused_mm_checked(A, B, bias, z=self.z, abs_floor=self.abs_floor,
                                                        nfma=self._tensor_core_steps(A, A.dtype),
                                                        epoch=self._epoch, inject=inj)
            except _OOM:
                raise
            except Exception as e:  # noqa: BLE001  (compile or launch problem: precise=True from now on)
                self._fused_off = True
                self.stats["fused_fallbacks"] += 1
                msg = str(e).strip().splitlines()
                self.fused_why = f"{name} ({pass_}): {type(e).__name__}: {msg[0][:300] if msg else ''}"
                import warnings
                warnings.warn(f"sdcguard: the fused precise check failed, precise=True takes over: {self.fused_why}",
                              RuntimeWarning, stacklevel=2)
                return None
            where = name if pass_ == "forward" else f"{name} ({pass_})"
            token = self.on_check_start(where, C) if self.on_check_start is not None else None
            detail = {"op": "aten::addmm" if bias is not None else "aten::mm", "where": where,
                      "pass": "forward" if pass_ == "forward" else "backward",
                      "shape_A": tuple(A.shape), "shape_B": tuple(B.shape),
                      "dtype": f"float32 accumulator of {A.dtype} (fused)", "device": C.device}
            if fp64:
                try:
                    st = _fused.zscores_from_stats(A, B, stats, bias, z=self.z, abs_floor=self.abs_floor,
                                                   nfma=self._tensor_core_steps(C, A.dtype), epoch=self._epoch)
                except _OOM:
                    self.stats["skipped_oom"] += 1
                    return C
            if C.is_cuda:
                s = torch.cuda.current_stream(C.device)
                self._streams[(C.device.index, s.cuda_stream)] = s
            self.stats["checked"] += 1
            self.stats["checked_fp32"] += 1
            self.stats["checked_fused"] += 1
            self._pending.append(("s", st, None, token, detail))
            if self.immediate or len(self._pending) >= self.max_pending:
                self.flush()
            return C

    def _accumulator(self, name, pass_, A, B, C32, bias) -> None:
        """Precise mode: an fp32 matmul result before it's rounded. Experiments may change it first (only on
        checked steps, where it is checked)."""
        if self.on_accumulator is not None and self._active:
            self.on_accumulator(name, pass_, C32)
        self._linear_check(name, pass_, A, B, C32, bias)

    def _tensor_core_steps(self, C: torch.Tensor, dtype: torch.dtype) -> Optional[int]:
        if not C.is_cuda:
            return None
        key = (C.device.index, dtype)
        if key not in self._nfma:
            from .modes import n_fma
            self._nfma[key] = n_fma("bf16" if dtype == torch.bfloat16 else "fp16",
                                    torch.cuda.get_device_capability(C.device))
        return self._nfma[key]

    def coverage(self, forward_fn: Callable[[], None]) -> dict:
        """Run forward_fn() once (a forward pass, no backward needed) and report what share of its
        matmul work (multiply-adds) runs in bf16/fp16 inside protected linear layers: what fast mode
        can check on checked steps, before any sample_rate sampling. Run it in the same dtype/autocast
        region as the real workload; merely attaching to a float32 linear does not make it checked.
        Matmuls elsewhere, including the ones inside fused attention (scaled_dot_product_attention),
        count as unchecked. This is a potential-coverage estimate of the observed forward operations,
        not proof that checks executed: probe hooks can change which fused path PyTorch selects, and
        backward work depends on which inputs/parameters require gradients. `guard_stats` is a separate
        snapshot of cumulative actual checks/skips since construction. `share_checked_in_fast_mode` is
        retained as a compatibility alias for `share_eligible_in_fast_mode`."""
        total, inside, eligible, depth = [0], [0], [0], [0]
        unchecked: dict = {}

        class _Count(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                name = op_name(func)
                if name in MATMUL_OPS:
                    for A, B, _C, *_ in matmul_checks(name, args, kwargs, out):
                        flops = A.numel() * B.shape[-1]
                        total[0] += flops
                        if depth[0] > 0:
                            inside[0] += flops
                            if _C.dtype in UNIT_ROUNDOFF:
                                eligible[0] += flops
                        elif name == "aten::_grouped_mm":          # fast mode only checks nn.Linear layers
                            unchecked[name] = unchecked.get(name, 0) + flops
                elif name.startswith("aten::_scaled_dot_product") or name == "aten::scaled_dot_product_attention":
                    q, k = args[0], args[1]            # fused attention: Q K^T and P V, never checked here
                    total[0] += 2 * q.numel() * k.shape[-2]
                    unchecked["attention"] = unchecked.get("attention", 0) + 2 * q.numel() * k.shape[-2]
                elif name in UNCHECKED_MATMUL_OPS:
                    flops = unchecked_matmul_flops(name, args, out)
                    total[0] += flops
                    unchecked[name] = unchecked.get(name, 0) + flops
                return out

        def enter(*_):
            depth[0] += 1

        def leave(*_):
            depth[0] -= 1

        hooks = []
        for mod in self._protected:
            # the window is the layer's own forward: matmuls in the user's forward pre-hooks and hooks (an adapter,
            # a probe) run outside it in fast mode too, and aren't checked there
            hooks.append(mod.register_forward_pre_hook(enter))
            hooks.append(mod.register_forward_hook(leave, prepend=True))
        try:
            with torch.no_grad(), _Count():
                forward_fn()
        finally:
            for h in hooks:
                h.remove()
        share = eligible[0] / total[0] if total[0] else 0.0
        return {"matmul_multiply_adds": total[0], "in_protected_linear_layers": inside[0],
                "eligible_multiply_adds_in_protected_linear_layers": eligible[0],
                "coverage_kind": "potential", "share_eligible_in_fast_mode": share,
                "share_checked_in_fast_mode": share, "skipped_layers": dict(self.skipped),
                "unchecked_multiply_adds_by_op": unchecked,
                "guard_stats": dict(self.stats)}

    # ------------------------------------------------------------------ schedule
    def should_check(self, step: int) -> bool:
        """True on every `every`-th step, and on every step of a window set by after_restart() or observe()."""
        return self._in_spike_window(step) or self._every_rank_checks(step)

    def after_restart(self, step: int, steps: int = 50) -> "SDCGuard":
        """Call when training resumes from a checkpoint, e.g. after a crash. Every step from `step`
        to `step + steps` is checked, so a GPU that is still bad gets named before it costs another
        run, even if it crashed the last run between two scheduled checks. With replicas=True, call it on
        every rank: the replica check runs on these steps too."""
        self._dense = (step, step + steps)
        return self

    def observe(self, step: int, loss=None, grad_norm=None, steps: int = 50, threshold: float = 8.0) -> bool:
        """Watch the training loss and gradient norm (Python numbers; call after each step, or whenever they are
        logged). When one isn't finite, or jumps more than `threshold` robust deviations above its recent median,
        every step from the next one for `steps` steps is checked (should_check() is True there), whatever the
        `every` schedule says. A loss or gradient spike is how silent corruption usually first shows in training
        (ByteDance's AEGIS and Huawei's torch_npu both escalate on it); this turns the spike into a checked
        window instead of waiting for the next scheduled check. Returns True when it started (or extended) one.
        The window is this rank's own (in data parallel each rank may see a different loss), so the replica check
        (replicas=True), a collective, doesn't run on its steps."""
        import statistics
        why = []
        for name, v in (("loss", loss), ("grad_norm", grad_norm)):
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            hist = self._watch.setdefault(name, collections.deque(maxlen=200))
            if not math.isfinite(v):
                why.append(f"{name} is {v}")
                continue
            if len(hist) >= 20:
                med = statistics.median(hist)
                mad = statistics.median(abs(x - med) for x in hist) * 1.4826
                spread = max(mad, 1e-3 * abs(med), 1e-12)
                if (v - med) / spread > threshold:
                    why.append(f"{name} {v:.4g}, recent median {med:.4g}")
            hist.append(v)
        if not why:
            return False
        start = step + 1
        end = max(self._spike[1] if self._spike is not None and self._spike[1] > start else 0, start + steps)
        self._spike = (start, end)
        self.stats["anomaly_windows"] = self.stats.get("anomaly_windows", 0) + 1
        self.anomalies.append({"step": step, "why": why, "checked_through": end - 1})
        import logging
        logging.getLogger("sdcguard").warning("sdcguard: %s at step %s; checking every step through step %s",
                                              "; ".join(why), step, end - 1)
        return True

    @contextlib.contextmanager
    def on_step(self, step: int):
        """Wrap one training step (forward and backward). Checks it if should_check(step); otherwise
        does nothing. Yields whether this step is checked."""
        if self.should_check(step):
            with self.at_step(step):
                yield True
        else:
            yield False

    @contextlib.contextmanager
    def at_step(self, step: int):
        """Enter the guard and tag alerts with a training step. After protect(), only the protected
        linear layers are checked (fast mode); otherwise every matmul PyTorch runs is. Nested inside a
        checked step it does nothing: the outer step is the one being checked (and its number stays)."""
        if self._active or self._depth > 0:
            yield self
            return
        self.step = step
        self._replica_check(step)
        self._option_warnings()
        if self._protected:
            with self._fast_step():
                yield self
        else:
            with self:
                yield self

    @contextlib.contextmanager
    def _fast_step(self):
        if self._active:                     # nested: the outer step does the bookkeeping
            yield
            return
        self._new_epoch()
        restore = _exact_reductions_on() if self.exact_reductions else None
        self._snapshot_gemm_settings()
        _follow_autocast_cache()
        seen_before = self.stats["matmuls_seen"]
        self._active = True
        self._casts.clear()
        _IN_STEP.add(self)
        self._rc_begin()
        try:
            yield
        except Exception as e:
            self._end_fast_step(restore)
            self._flush_after_error(e)
            self._rc_end(raise_alerts=False)
            raise
        except BaseException:
            self._end_fast_step(restore)
            self._drop_pending()
            if self._recompute is not None:
                self._recompute.enabled = False
            raise
        self._end_fast_step(restore)
        rc_before = self._recompute.stats["recomputes_compared"] if self._recompute is not None else 0
        try:
            self.flush()
        finally:
            self._rc_end(raise_alerts=True)
        rc_after = self._recompute.stats["recomputes_compared"] if self._recompute is not None else 0
        tp_skipped = self._compiled_tp_report()
        if rc_after == rc_before and not tp_skipped:  # (that step checked something, or said why it didn't)
            self._no_checks_warning(seen_before)

    @contextlib.contextmanager
    def _local_shards(self, tp: bool):
        """Inside the compiled operator, a tensor-parallel layer's operands are this rank's local shards: its
        checks count as tensor-parallel checks (stats["checked_dtensor"]), as in eager mode."""
        if not tp:
            yield
            return
        prev = self.__dict__.get("_tp_local", False)
        self._tp_local = True
        try:
            yield
        finally:
            self._tp_local = prev

    def _compiled_tp_report(self) -> int:
        """Tensor-parallel layers that ran inside torch.compile on this checked step weren't checked: counted in
        stats["skipped_dtensor_compiled"] (layer runs), named in the warning (once). Returns how many."""
        if not self._compiled_tp_skipped:
            return 0
        names = dict(self._compiled_tp_skipped)
        self._compiled_tp_skipped.clear()         # in place: the compiled graph writes into this same dict
        self.stats["skipped_dtensor_compiled"] = self.stats.get("skipped_dtensor_compiled", 0) + len(names)
        if not self.__dict__.get("_warned_tp_compiled"):
            self._warned_tp_compiled = True
            warnings.warn(f"sdcguard: {len(names)} tensor-parallel layers ran inside torch.compile and weren't "
                          f"checked ({', '.join(sorted(names)[:3])}{' ...' if len(names) > 3 else ''}); tensor "
                          "parallel is checked in eager mode. Counted in stats['skipped_dtensor_compiled']",
                          stacklevel=4)
        return len(names)

    def _option_warnings(self) -> None:
        """Options that do nothing in the mode the guard runs in, said once instead of passing silently; and, once
        per process, NVIDIA's listed cuBLAS wrong-result issues that apply to this machine (known_issues.py)."""
        if self.__dict__.get("_warned_options"):
            return
        self._warned_options = True
        try:
            if torch.cuda.is_available():
                from . import known_issues
                known_issues.log_once()
        except Exception:  # noqa: BLE001  (a report, never in the way of training)
            pass
        if self.precise and not self._protected:
            warnings.warn("sdcguard: precise=True checks the fp32 accumulators of protect()ed linear layers; without "
                          "protect() the guard checks rounded outputs as usual", stacklevel=4)
        if self.attention and self._protected:
            warnings.warn("sdcguard: attention=True works in the default mode only (no protect()); fused attention "
                          "isn't checked here", stacklevel=4)

    def _backward_outside_step(self) -> None:
        with self._lock:
            self.stats["backward_outside_step"] = self.stats.get("backward_outside_step", 0) + 1
            if not self.__dict__.get("_warned_outside"):
                self._warned_outside = True
                warnings.warn("sdcguard: a protected layer's backward ran after its checked step ended; it isn't "
                              "checked (put loss.backward() inside at_step / on_step). Counted in "
                              "stats['backward_outside_step']", stacklevel=4)

    def _no_checks_warning(self, seen_before: int) -> None:
        """A checked step in which no protected layer ran a checked matmul: usually a sign the checks aren't
        reaching the model, so it's said once instead of passing as a clean step (and every such step is counted
        in stats["checked_steps_without_checks"])."""
        if self.stats["matmuls_seen"] != seen_before:
            return
        self.stats["checked_steps_without_checks"] = self.stats.get("checked_steps_without_checks", 0) + 1
        if self.__dict__.get("_warned_empty"):
            return
        self._warned_empty = True
        warnings.warn(f"sdcguard: checked step {self.step} ran no checks. The protected layers didn't run a checked "
                      "matmul: they ran in float32 (only bf16/fp16 matmuls are checked: use autocast), the forward "
                      "ran outside at_step / on_step, or the protected layers weren't used this step",
                      stacklevel=4)

    def _end_fast_step(self, restore) -> None:
        self._active = False
        _IN_STEP.discard(self)
        self._casts.clear()
        _exact_reductions_off(restore)

    def _uncount(self, entry) -> None:
        """A queued check that was never judged: taken back out of the count it was added to."""
        detail = entry[4] if len(entry) > 4 and isinstance(entry[4], dict) else {}
        kind = str(detail.get("kind", ""))
        key = "attention_checked" if kind.startswith("attention") else "checked"
        self.stats[key] = max(0, self.stats[key] - 1)
        if detail.get("tensor_parallel_shard"):
            self.stats["checked_dtensor"] = max(0, self.stats.get("checked_dtensor", 0) - 1)

    def _drop_pending(self) -> None:
        with self._lock:
            for entry in self._pending:               # queued, never judged: not counted as checked
                self._uncount(entry)
            self._pending.clear()
            self._groups.clear()
            self._streams.clear()
            self._kept.clear()                        # (their evidence operands are let go too)

    def _flush_after_error(self, error: BaseException) -> None:
        """The step raised. Its queued checks are still read back (the step most likely to raise is the one a
        faulty GPU broke, through NaN gradients for example): alerts are recorded and passed to on_alert, but
        the step's own exception is the one that propagates, with a note when a check failed."""
        try:
            with self._lock:
                results = self._flush(raise_alerts=False)
        except Exception:  # noqa: BLE001  (e.g. the GPU is gone: don't hide the step's own error)
            self._drop_pending()
            return
        bad = sum(not r.ok for r in results)
        if bad:
            note = (f"sdcguard: {bad} matmul result(s) of this step failed their checksum before this error (see "
                    "the guard's log)")
            if hasattr(error, "add_note"):
                error.add_note(note)
            else:
                # Python 3.10 has no add_note (PEP 678, 3.11): the same __notes__ list, which code can read and 3.11+
                # prints with the traceback
                try:
                    notes = getattr(error, "__notes__", None)
                    if not isinstance(notes, list):
                        notes = []
                        error.__notes__ = notes
                    notes.append(note)
                except Exception:  # noqa: BLE001  (an exception type that takes no attributes: the log has it)
                    pass

    def _new_epoch(self) -> None:
        """Start a checked step: with fresh_signs, its signed checksums use a pattern no earlier step used,
        so a fault that keeps changing the same outputs in a way one pattern can't see shows on a later step."""
        if self.fresh_signs:
            with self._lock:
                self._epoch += 1

    def __enter__(self):
        if self._depth == 0:
            self._new_epoch()
            self._rc_begin()
        if self.exact_reductions and self._depth == 0:
            self._restore_flags = _exact_reductions_on()
        if self._depth == 0:
            self._snapshot_gemm_settings()
        self._depth += 1
        try:
            return super().__enter__()
        except BaseException:
            self._leave()
            raise

    def _leave(self):
        self._depth -= 1
        if self._depth == 0:
            _exact_reductions_off(self._restore_flags)
            self._restore_flags = None

    def __exit__(self, exc_type, exc, tb):
        try:
            result = super().__exit__(exc_type, exc, tb)
        finally:
            self._leave()
        outer = self._depth == 0
        if exc_type is None:
            try:
                self.flush()
            finally:
                if outer:
                    self._rc_end(raise_alerts=True)
        elif isinstance(exc, Exception):
            self._flush_after_error(exc)
            if outer:
                self._rc_end(raise_alerts=False)
        else:
            self._drop_pending()
            if outer and self._recompute is not None:
                self._recompute.enabled = False
        return result

    # ------------------------------------------------------------------ checking
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)      # this mode is paused here, so no recursion
        kind = self._is_matmul.get(func)
        if kind is None:
            n = op_name(func)
            kind = self._is_matmul[func] = ("mm" if n in MATMUL_OPS else "attn" if n in _att.BACKWARD_OPS or
                                            n in _att.FORWARD_OPS else "unchecked" if n in UNCHECKED_MATMUL_OPS
                                            else "")
        if not kind:
            return out
        name = op_name(func)
        if kind == "unchecked":
            with self._lock:
                self.stats["matmuls_unchecked"] = self.stats.get("matmuls_unchecked", 0) + 1
                self.unchecked_ops[name] = self.unchecked_ops.get(name, 0) + 1
            return out
        if kind == "attn":
            if self.attention:
                self._attention_op(name, args, kwargs, out)
            return out
        # inside a watched layer's forward (watch()) it's that layer's forward work, also when activation
        # checkpointing recomputes it from inside a backward node
        node = _autograd_node() if self._module is None else None
        where = node if node is not None else (self._module or "forward")
        with self._lock:
            self.stats["matmuls_seen"] += 1
            self.ops_seen[name] = self.ops_seen.get(name, 0) + 1
            if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
                return out
            try:
                checks = matmul_checks(name, args, kwargs, out)
            except Exception as e:  # noqa: BLE001  (a tensor subclass the views don't work on: skip, never crash)
                if not any(_is_dtensor(t) for t in args if isinstance(t, torch.Tensor)):
                    raise
                self._skip_dtensor(f"{name}: {type(e).__name__}")
                checks = []
            for A, B, C, bias, alpha, beta in checks:
                self._queue(name, where, node, A, B, C, bias, alpha, beta)
            if self.immediate or len(self._pending) >= self.max_pending:
                self.flush()
        return out

    def _attention_op(self, name, args, kwargs, out) -> None:
        """Queue the attention checks of one fused attention call (attention.py)."""
        node = _autograd_node()
        where = node if node is not None else (self._module or "forward")
        with self._lock:
            if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
                return
            why = _att.why_not(name, args, kwargs)
            backward = name in _att.BACKWARD_OPS
            try:
                if why is None:
                    # Q and K too: queries or keys that aren't finite leave their heads unjudged, not blamed
                    scale = _att._arg(args, kwargs, None, "scale")
                    if backward:
                        dO, dV = args[0], out[2]
                        st = _att.backward_stat(dO, dV, z=self.z, abs_floor=self.abs_floor, q=args[1], k=args[2],
                                                scale=scale)
                        shapes = (tuple(dO.shape), tuple(dV.shape))
                    else:
                        O, V = out[0], args[2]
                        st = _att.forward_stat(O, V, abs_floor=self.abs_floor, q=args[0], k=args[1], scale=scale)
                        shapes = (tuple(O.shape), tuple(V.shape))
                    why = None if st is not None else "dtype or layout"
            except _OOM:
                self.stats["skipped_oom"] += 1
                return
            if why is not None:
                self.stats["attention_skipped"] += 1
                self.attention_skipped[why] = self.attention_skipped.get(why, 0) + 1
                return
            dev = st.max_z.device
            detail = {"op": name, "where": where, "pass": "backward" if backward else "forward",
                      "kind": "attention_backward" if backward else "attention_forward",
                      "check": "attention: sum of dV over keys vs sum of dO over queries" if backward else
                      "attention: outputs within the range of the values",
                      "shape_A": shapes[0], "shape_B": shapes[1], "dtype": str((dV if backward else O).dtype),
                      "device": dev}
            if dev.type == "cuda":
                s = torch.cuda.current_stream(dev)
                self._streams[(dev.index, s.cuda_stream)] = s
            self.stats["attention_checked"] += 1
            self._pending.append(("s", st, None, None, detail))
            if self.immediate or len(self._pending) >= self.max_pending:
                self.flush()

    def _queue(self, name, where, node, A, B, C, bias, alpha, beta) -> None:
        """Queue one check (its GPU work starts now; results are read at the flush)."""
        local = _dtensor_local(A, B, C, bias)
        if isinstance(local, str):
            self._skip_dtensor(local)
            return
        if local is not None:
            A, B, C, bias = local
        # a local shard whose collective may still be running (from to_local() or DTensor's sharding rules): waited
        # for, so the check reads its numbers, not a wrapper with no memory of its own (_group.ready)
        A, B, C, bias = (_group.ready(t) for t in (A, B, C, bias))
        if _cs.is_empty(C):                       # m = 0 or n = 0 (an MoE expert with no tokens): nothing to check
            self.stats["empty"] = self.stats.get("empty", 0) + 1
            return
        acc = self.precise and _cs.acc_supported(A, C)
        if C.dtype not in UNIT_ROUNDOFF and not acc:
            self.stats["skipped_dtype"] += 1
            return
        token = self.on_check_start(name, C) if self.on_check_start is not None else None
        detail = {"op": name, "where": where, "pass": "backward" if node else "forward",
                  "shape_A": tuple(A.shape), "shape_B": tuple(B.shape),
                  "dtype": f"float32 accumulator of {A.dtype}" if acc else str(C.dtype), "device": C.device}
        shard = local is not None or bool(self.__dict__.get("_tp_local"))   # (local shards from _compiled.py too)
        if shard:
            detail["tensor_parallel_shard"] = True
        entry = None
        try:
            if acc:
                st = matmul_zscores_acc(A, B, C, bias, alpha=alpha, beta=beta, block=self.block,
                                        abs_floor=self.abs_floor, z=self.z,
                                        nfma=self._tensor_core_steps(C, A.dtype), epoch=self._epoch)
                entry = ("s", st, None, token, detail)
            elif _cs.USE_FAST and _group.available(C) and _group.supports(A, B, C, bias, beta, self.block):
                key = (C.device, C.dtype, self._epoch)
                try:
                    g = self._groups.get(key)
                    if g is None:
                        g = self._groups[key] = _group.Group(C.device, C.dtype, self.block, self.abs_floor,
                                                             self.z, epoch=self._epoch)
                    entry = ("g", g, g.add(A, B, C, bias, alpha, beta), token, detail)
                except _OOM:
                    raise
                except Exception as e:  # noqa: BLE001  (compile or launch problem: plain check from now on)
                    _group.fell_back(C.device.type, C.dtype, f"check of {name} {tuple(A.shape)} @ {tuple(B.shape)}",
                                     e)
                    entry = None
            if entry is None:
                st = matmul_zscores(A, B, C, bias, alpha=alpha, beta=beta, block=self.block,
                                    abs_floor=self.abs_floor, z=self.z, epoch=self._epoch)
                if st is None:
                    self.stats["skipped_dtype"] += 1
                    return
                entry = ("s", st, None, token, detail)
        except _OOM:
            self.stats["skipped_oom"] += 1          # this check doesn't fit in memory right now
            return
        if C.is_cuda:
            s = torch.cuda.current_stream(C.device)
            self._streams[(C.device.index, s.cuda_stream)] = s
        self.stats["checked"] += 1
        if shard:                                 # this rank's shard of a tensor-parallel matmul
            self.stats["checked_dtensor"] = self.stats.get("checked_dtensor", 0) + 1
        if acc:
            self.stats["checked_fp32"] += 1
        self._pending.append(entry)
        if self.evidence_dir is not None and len(self.evidence) < self.max_evidence:
            self._keep(entry, A, B, C, bias, alpha, beta, name, where)

    def _snapshot_gemm_settings(self) -> None:
        """The GEMM settings this checked step runs with (TF32, reductions, BLAS library), read once at its start
        when evidence is kept: the read-back, and the evidence it saves, come after the step put them back."""
        self._step_gemm = None
        if self.evidence_dir is not None:
            try:
                from .check import gemm_settings
                self._step_gemm = gemm_settings()
            except Exception:  # noqa: BLE001  (then the evidence records them as they are when it's saved)
                pass

    def _keep(self, entry, A, B, C, bias, alpha, beta, name, where) -> None:
        """Hold this check's operands for evidence: references with a stamp of each tensor's version and storage
        (so a change before the read-back is noticed), or copies taken now (evidence_copy)."""
        ts = (A, B, C, bias)
        if self.evidence_copy:
            try:
                ts = tuple(None if t is None else
                           (t.detach().to("cpu", non_blocking=True) if t.is_cuda else t.detach().clone())
                           for t in ts)
            except Exception:  # noqa: BLE001  (out of host memory, a tensor subclass: no evidence for this one)
                return
            stamps = None
        else:
            # the version, storage and address of each operand, and its contents' fingerprint
            stamps = tuple((_stamp(t), _content(t)) for t in ts)
        self._kept.append((entry, (*ts, alpha, beta, name, where, self._epoch), stamps,
                           self.__dict__.get("_step_gemm")))

    def _skip_dtensor(self, why: str) -> None:
        """A tensor-parallel matmul whose local pieces aren't a plain matmul: skipped and counted, never raised."""
        self.stats["skipped_dtensor"] = self.stats.get("skipped_dtensor", 0) + 1
        seen = self.__dict__.setdefault("_dtensor_warned", set())
        if why not in seen and len(seen) < 8:
            seen.add(why)
            warnings.warn(f"sdcguard: a tensor-parallel matmul isn't checked ({why}); counted in "
                          "stats['skipped_dtensor']", stacklevel=3)

    def flush(self) -> list[CheckResult]:
        """Read all queued checks back from the GPU, record them, and raise alerts."""
        with self._lock:
            return self._flush()

    def _read_back(self, pending, groups, streams):
        """The queued checks' results, read back: (plain results, or None when out of memory; grouped results by
        id(group), with groups that ran out of memory left out)."""
        for s in streams.values():                  # checks queued on other CUDA streams finish first
            cur = torch.cuda.current_stream(s.device)
            if s != cur:
                cur.wait_stream(s)
        plain_entries = [p for p in pending if p[0] == "s"]
        try:
            plain = resolve([p[1] for p in plain_entries], self.z)
        except _OOM:
            plain = None
        by_dev: dict = {}
        for g in groups.values():
            if len(g):
                try:
                    g.launch()
                except _OOM:
                    continue
                by_dev.setdefault(g.device, []).append(g)
        rows = {}
        for gs in by_dev.values():
            try:
                back = torch.cat([g.result for g in gs]).cpu().tolist()      # one read-back per device
            except _OOM:
                continue
            i = 0
            for g in gs:
                rows[id(g)] = back[i:i + len(g)]
                i += len(g)
        return plain, rows

    def _flush(self, raise_alerts: bool = True) -> list[CheckResult]:
        if not self._pending:
            return []
        pending, self._pending = self._pending, []
        groups, self._groups = self._groups, {}
        streams, self._streams = self._streams, {}
        try:
            plain, rows = self._read_back(pending, groups, streams)
        except BaseException:
            # nothing was judged (a CUDA error after a GPU fault, say): none of these checks counts as checked,
            # and their evidence operands are let go
            for p in pending:
                self._uncount(p)
            self._kept = []
            raise
        results, kept = [], []
        it = iter(plain or ())
        for p in pending:
            if p[0] == "s":
                if plain is None:
                    self._uncount(p)
                    self.stats["skipped_oom"] += 1
                    continue
                results.append(next(it))
                kept.append(p)
                continue
            g, idx = p[1], p[2]
            if id(g) not in rows:                   # out of memory while reading these back
                self._uncount(p)
                self.stats["skipped_oom"] += 1
                continue
            mz, b, r, blk, den = g.row(idx, rows[id(g)][idx])
            finite = math.isfinite(mz)
            results.append(CheckResult(ok=finite and mz <= self.z, max_z=mz, nonfinite=not finite,
                                       batch=b if len(p[4]["shape_A"]) == 3 else None,
                                       row=r, col_block=blk, noise=den))
            kept.append(p)
        self._flushed = kept                        # the queue entries `results` belong to, in order
        ops_of = {id(e): (ops, stamps, gemm) for e, ops, stamps, gemm in self._kept}
        self._kept = []
        first_error = callback_error = None
        for entry, res in zip(kept, results):
            _kind, _st, _idx, token, detail = entry
            if res.ok and not res.judged:
                self.stats["unjudged"] += 1          # e.g. NaN inputs: no evidence either way
            elif res.ok and detail.get("kind") != "attention_forward":     # (that one is 0 or inf: no scale)
                self.max_z_seen = max(self.max_z_seen, res.max_z)
                if len(self.z_values) < self._z_cap:
                    self.z_values.append(res.max_z)
                elif self._rng.random() < 0.01:
                    self.z_values[self._rng.randrange(self._z_cap)] = res.max_z
            # a callback that raises mustn't cost the alerts after it: every result is recorded first, then
            # the first callback error is raised
            try:
                if self.on_check is not None:
                    self.on_check(detail["op"], token, res)
                if self.on_result is not None:
                    self.on_result(detail, res)
            except Exception as e:  # noqa: BLE001
                callback_error = callback_error or e
            if not res.ok:
                kept_ops = ops_of.get(id(entry))
                if kept_ops is not None and len(self.evidence) < self.max_evidence:
                    ops, stamps, gemm = kept_ops
                    changed = [] if stamps is None else \
                        [n for n, t, (st, fp) in zip(("A", "B", "C", "bias"), ops[:4], stamps)
                         if _stamp(t) != st or (fp is not None and not torch.equal(fp, _content(t)))]
                    if changed:
                        self.stats["evidence_errors"] = self.stats.get("evidence_errors", 0) + 1
                        self.evidence_copy = True         # a fault that comes back is saved as it was checked
                        detail["evidence"] = (f"not saved: {', '.join(changed)} changed between the check and the "
                                              "read-back (an in-place op, an optimizer step inside at_step, or a "
                                              "weight freed after use, as FSDP2 does); later checked steps keep a "
                                              "copy of their operands")
                    else:
                        replays = self._confirm(res, ops, detail, gemm)
                        if replays is not None:
                            detail["confirmed_in_job"] = {k: replays[k] for k in (
                                "recomputations", "recomputations_that_failed_the_check",
                                "recomputations_that_couldnt_be_judged", "reading")}
                        detail["evidence"] = self._save_evidence(res, ops, detail, gemm, replays)
                event, err = self._alert(detail, res)
                callback_error = callback_error or err
                if self.raise_on_alert and first_error is None:
                    first_error = SDCError(event.to_json())
        if raise_alerts and first_error is not None:
            raise first_error
        if raise_alerts and callback_error is not None:
            raise callback_error
        return results

    def _confirm(self, res: CheckResult, ops: tuple, detail: dict, gemm: Optional[dict]) -> Optional[dict]:
        """Recompute a failed matmul `confirm` times on its GPU, with its step's GEMM settings and the same check
        (block, alarm line, floor, sign patterns, precise mode's fp32 accumulator): does the error come back right
        away? Best effort: None when it couldn't run."""
        if not self.confirm:
            return None
        dev = detail.get("device")
        if not isinstance(dev, torch.device):
            return None
        try:
            from .check import gemm_settings_applied
            from .screen import _replay
            A, B, C, bias, alpha, beta = ops[:6]
            epoch = ops[8] if len(ops) > 8 else 0
            A, B, C = (t.detach().to(dev) for t in (A, B, C))
            bias = bias.detach().to(dev) if bias is not None else None
            cfg = {"block": self.block, "z": self.z, "abs_floor": self.abs_floor}
            hold = _exact_reductions_on() if self.exact_reductions else None
            try:
                with gemm_settings_applied(gemm), torch.no_grad():
                    out = _replay(None, self.confirm, dev, A, B, C, bias, alpha, beta, epoch, cfg)
            finally:
                _exact_reductions_off(hold)
            bad, n = out["recomputations_that_failed_the_check"], out["recomputations"]
            out["reading"] = (f"the error came back in {bad} of {n} recomputations right away on this GPU, with "
                              "the step's settings" if bad else
                              f"none of {n} recomputations right away on this GPU, with the step's settings, was "
                              "wrong: a fault that doesn't repeat at once, or one these few recomputations missed")
            return out
        except Exception as e:  # noqa: BLE001  (a confirmation is extra evidence; the alert stands either way)
            self.stats["confirm_errors"] = self.stats.get("confirm_errors", 0) + 1
            return {"recomputations": 0, "recomputations_that_failed_the_check": 0,
                    "recomputations_that_couldnt_be_judged": 0, "reading": f"couldn't recompute it here: "
                                                                          f"{type(e).__name__}: {str(e)[:160]}"}

    def _save_evidence(self, res: CheckResult, ops: tuple, detail: dict, gemm: Optional[dict] = None,
                       replays: Optional[dict] = None) -> Optional[str]:
        """Write one failed check's operands (evidence_<n>.pt/.json in evidence_dir). Never raises."""
        try:
            import os
            from .screen import _nvidia_smi, _save_evidence
            os.makedirs(self.evidence_dir, exist_ok=True)
            dev = detail.get("device")
            ident = dict(device_identity(dev if isinstance(dev, torch.device) else None))
            if isinstance(dev, torch.device) and dev.type == "cuda":
                ident.update(_nvidia_smi(ident.get("gpu_uuid") or ident.get("gpu_index")))
            n = len(self.evidence)
            while os.path.exists(os.path.join(self.evidence_dir, f"evidence_{n}.pt")):
                n += 1                            # never replace an earlier run's evidence
            A, C = ops[0], ops[2]
            nfma = None
            if isinstance(dev, torch.device) and dev.type == "cuda" and _cs.acc_supported(A, C):
                from .modes import n_fma
                nfma = n_fma("bf16" if A.dtype == torch.bfloat16 else "fp16", torch.cuda.get_device_capability(dev))
            path = _save_evidence(self.evidence_dir, n, res, ops, ident,
                                  {"source": "SDCGuard", "step": self.step, "pass": detail.get("pass"),
                                   "check": detail.get("dtype"), "gemm_settings": gemm,
                                   "check_config": {"block": self.block, "z": self.z, "abs_floor": self.abs_floor,
                                                    "nfma": nfma},
                                   "record_device": dev if isinstance(dev, torch.device) else None,
                                   "in_job_replay": replays if replays and replays.get("recomputations") else None})
            self.evidence.append(path + ".pt")
            return path + ".pt"
        except Exception as e:  # noqa: BLE001  (evidence is best effort: the alert itself must still happen)
            self.stats["evidence_errors"] = self.stats.get("evidence_errors", 0) + 1
            return f"not saved: {type(e).__name__}: {str(e)[:160]}"

    def _alert(self, detail: dict, res: CheckResult) -> SDCEvent:
        self.stats["alerts"] += 1
        dev = detail.pop("device")
        key = str(dev)
        if key not in self._identity:
            self._identity[key] = device_identity(dev)
        detail.update(max_z=res.max_z, nonfinite=res.nonfinite, batch=res.batch, row=res.row,
                      col_block=res.col_block, noise=res.noise, alarm_line=self.z)
        event = SDCEvent(detector="checksum", step=self.step, detail=detail,
                         identity=dict(self._identity[key]))
        self.log.add(event)
        err = None
        if self.on_alert:
            try:
                self.on_alert(event)
            except Exception as e:  # noqa: BLE001  (raised by _flush once every result is recorded)
                err = e
        return event, err
