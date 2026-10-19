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
partial sums stay in fp32 (split-K itself stays allowed), and restores the setting on exit.
Unchecked steps are untouched. Turn this off with SDCGuard(exact_reductions=False).

Precise mode (SDCGuard(precise=True), with protect()): on checked steps each linear layer's matmuls keep
their fp32 result (the accumulator) long enough to check it, then round it to bf16/fp16. The check then
sees errors far smaller than bf16 rounding hides (checksum.matmul_zscores_acc). The model gets the same
rounding of the same fp32 result, but cuBLAS may pick a different kernel for an fp32 output, so values
on checked steps can differ from an unchecked step in the last bit; unchecked steps don't change.
"""
from __future__ import annotations

import contextlib
import math
import random
import threading
from typing import Callable, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from . import _group
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
FORWARD_MATMUL_OPS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::_mps_linear", "aten::linear"}
MATMUL_OPS = FORWARD_MATMUL_OPS | {"aten::linear_backward", "aten::mps_linear_backward"}


def _reduced_precision_flags():
    m = torch.backends.cuda.matmul
    return m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction


def _exact_reductions_on() -> Optional[tuple]:
    """Turn off reduced-precision reductions in bf16/fp16 cuBLAS matmuls. Returns what to restore
    (None if nothing was changed). Settings already off (including the tuple form) are left alone."""
    if not torch.cuda.is_available():
        return None
    try:
        bf16, fp16 = _reduced_precision_flags()
        m = torch.backends.cuda.matmul
        if bf16 is True:
            m.allow_bf16_reduced_precision_reduction = False
        if fp16 is True:
            m.allow_fp16_reduced_precision_reduction = False
        return (bf16 is True, fp16 is True)
    except (AttributeError, RuntimeError, TypeError):
        return None


def _exact_reductions_off(changed: Optional[tuple]) -> None:
    if not changed:
        return
    m = torch.backends.cuda.matmul
    if changed[0]:
        m.allow_bf16_reduced_precision_reduction = True
    if changed[1]:
        m.allow_fp16_reduced_precision_reduction = True


def _flat2d(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(-1, t.shape[-1])


def matmul_checks(name: str, args, kwargs, out):
    """List of (A, B, C, bias, alpha, beta) products to verify for one op call."""
    if name in ("aten::mm", "aten::bmm"):
        return [(args[0], args[1], out, None, 1.0, 1.0)]
    if name == "aten::addmm":
        beta = kwargs.get("beta", args[3] if len(args) > 3 else 1)
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        return [(args[1], args[2], out, args[0], float(alpha), float(beta))]
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
        if guard.precise:
            # the fp32 result, checked, then rounded once (bias added in fp32 first, as cuBLAS does)
            x2 = _flat2d(x)
            y32 = _acc_mm(x2, w.t())
            if b is not None:
                y32 = y32.add_(b.float())
            guard._accumulator(name, "forward", x2, w.t(), y32, b)
            return y32.to(x.dtype).view(*x.shape[:-1], w.shape[0])
        if b is None or x.dim() == 2 or (x.is_contiguous() and b.dim() == 1 and b.is_contiguous()):
            # what at::linear does here (PyTorch 2.5 to 2.14): one matmul with the bias fused in (one rounding)
            y = torch.nn.functional.linear(x, w, b)
            guard._linear_check(name, "forward", _flat2d(x), w.t(), _flat2d(y), b)
            return y
        # otherwise at::linear runs the matmul, rounds, then adds the bias (rounded again): check the
        # matmul by itself, then add the bias the same way
        y = torch.nn.functional.linear(x, w)
        guard._linear_check(name, "forward", _flat2d(x), w.t(), _flat2d(y), None)
        return y.add_(b)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        guard, name = ctx.guard, ctx.name
        gy2 = _flat2d(gy)
        gx = gw = gb = None
        if guard.precise:
            if ctx.needs_input_grad[0]:
                gx32 = _acc_mm(gy2, w)
                guard._accumulator(name, "backward dX", gy2, w, gx32, None)
                gx = gx32.to(gy2.dtype).view(x.shape)
            if ctx.needs_input_grad[1]:
                x2 = _flat2d(x)
                gw32 = _acc_mm(gy2.t(), x2)
                guard._accumulator(name, "backward dW", gy2.t(), x2, gw32, None)
                gw = gw32.to(w.dtype)
            if ctx.has_bias and ctx.needs_input_grad[2]:
                gb = gy2.sum(0)
            return gx, gw, gb, None, None
        if ctx.needs_input_grad[0]:
            gx2 = gy2.mm(w)                                          # dX = dY W
            guard._linear_check(name, "backward dX", gy2, w, gx2, None)
            gx = gx2.view(x.shape)
        if ctx.needs_input_grad[1]:
            x2 = _flat2d(x)
            gw = gy2.t().mm(x2)                                      # dW = dY^T X
            guard._linear_check(name, "backward dW", gy2.t(), x2, gw, None)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = gy2.sum(0)
        return gx, gw, gb, None, None


def _autograd_node() -> Optional[str]:
    """Name of the autograd node running right now (None in the forward pass)."""
    try:
        node = torch._C._current_autograd_node()
    except AttributeError:
        return None
    return node.name() if node is not None else None


def _unsupported_linear(mod: torch.nn.Module) -> Optional[str]:
    """Why fast mode can't take over this nn.Linear (None if it can). The checked path computes the
    plain linear itself, so a layer whose forward does anything else must keep its own forward."""
    if type(mod).forward is not torch.nn.Linear.forward:
        return f"{type(mod).__name__} has its own forward"
    if "forward" in mod.__dict__:
        return "forward replaced on the instance (a hook or wrapper)"
    return None


class SDCGuard(TorchDispatchMode):
    """Checksums sampled matmuls while active.

    sample_rate: share of matmul ops checked while the guard is on (1.0 = all).
    z: alert threshold, in units of the estimated rounding noise.
    immediate: read every check back right away (slower on GPUs; alerts raise inside the op).
    exact_reductions: on NVIDIA, keep bf16/fp16 matmul partial sums in fp32 while the guard is
        active (see the module docstring). Default True.
    on_check_start(op_name, C) -> token and on_check(op_name, token, result): optional
        callbacks for experiments (e.g. matching injected faults to checks).
    on_result(detail, result): optional, called for every check with where/pass/shapes
        (for diagnostics).
    abs_floor: differences below this are not judged (default checksum.ABS_FLOOR = 1e-20).
    every: check schedule for on_step(): every N-th step (default 10).
    precise: with protect(), check each linear layer's fp32 accumulator before it's rounded (much smaller
        errors show; see the module docstring). on_accumulator(layer, pass, C32): optional, called with each
        fp32 result before its check (fault-injection experiments corrupt it there).
    fresh_signs: give every checked step a new +-1 pattern for the signed checksums (default True). Two
        wrong outputs in a block can cancel in both sums for one pattern (two values swapped where the
        pattern has the same sign); a fault that repeats that way on every step then still shows on a
        later step, instead of hiding for the whole run.
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
    ):
        super().__init__()
        self.every = every
        self.fresh_signs = fresh_signs        # a new signed-checksum pattern every checked step
        self._epoch = 0
        self.precise = precise                # check fp32 accumulators (protect() only; see the module docstring)
        self.on_accumulator = on_accumulator  # experiments: on_accumulator(layer, pass, C32) before each check
        self._nfma: dict = {}
        self._dense: Optional[tuple] = None          # (first, last + 1) steps of a restart window
        self.abs_floor = ABS_FLOOR if abs_floor is None else abs_floor
        self.exact_reductions = exact_reductions
        self.on_result = on_result
        self._restore_flags: Optional[tuple] = None
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
                      "skipped_oom": 0, "alerts": 0}
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
        self._module: Optional[str] = None
        self._hooks: list = []
        self._identity: dict = {}
        self._protected: dict = {}            # module -> name, for modules wrapped by protect()
        self.skipped: dict = {}               # layer name -> why protect() left it alone
        self._active = False                  # fast mode: True while a checked step runs
        self._casts: dict = {}                # fast mode: low-precision copies of weights, this step

    # ------------------------------------------------------------------ setup
    def watch(self, model: torch.nn.Module) -> "SDCGuard":
        """Name the layer (e.g. 'blocks.3.attn.q') in forward-pass alerts. Optional."""
        def pre(name):
            def hook(_m, _inp):
                self._module = name
            return hook

        def post(_m, _inp, _out):
            self._module = None

        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self._hooks.append(mod.register_forward_pre_hook(pre(name)))
                self._hooks.append(mod.register_forward_hook(post))
        return self

    def unwatch(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def protect(self, model: torch.nn.Module) -> "SDCGuard":
        """Fast mode: check the matmuls of every nn.Linear in `model` (forward, dX and dW).

        Instead of routing every PyTorch operation through Python (the default operator-level
        hook), only the linear layers' own matmuls do, and only on checked steps. On other steps
        each layer runs its normal forward, plus one Python `if`. Matmuls outside nn.Linear (for
        example attention written with torch.bmm, or einsum) are not checked in this mode; use the
        default mode, or coverage() to see what share of the matmul work the linear layers are.

        Works with autocast (bf16/fp16), DDP and FSDP2. Under torch.compile, unchecked steps run
        compiled; checked steps run the linear layers eagerly.
        """
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and mod not in self._protected:
                why = _unsupported_linear(mod)
                if why:                       # its own forward does more than the plain linear: leave it alone
                    self.skipped[name or type(mod).__name__] = why
                    continue
                self._protected[mod] = name or type(mod).__name__
                mod.forward = self._make_forward(mod, self._protected[mod])
        return self

    def unprotect(self) -> None:
        """Undo protect(): the layers run their own forward again."""
        for mod in self._protected:
            if "forward" in mod.__dict__:
                del mod.forward
        self._protected.clear()

    def _make_forward(self, mod: torch.nn.Linear, name: str):
        guard = self
        plain = type(mod).forward.__get__(mod)

        @torch.compiler.disable
        def checked(x):
            dt = _compute_dtype(x)
            if dt not in UNIT_ROUNDOFF:                  # fp32/TF32: not checked
                return plain(x)
            w = guard._cast(mod.weight, dt)
            b = guard._cast(mod.bias, dt) if mod.bias is not None else None
            with torch.autocast(device_type=x.device.type, enabled=False):
                return _CheckedLinear.apply(x.to(dt), w, b, guard, name)

        def forward(x):
            if not guard._active:
                return plain(x)
            return checked(x)

        return forward

    def _cast(self, p: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
        """p in dtype dt, cast once per checked step and reused, as autocast caches its casts: a layer
        used twice in one step then sums its weight gradient the way autocast would."""
        if p.dtype == dt:
            return p
        # Grad mode is part of the key: a cast made under no_grad (an eval forward in the same step, or
        # the first pass of reentrant activation checkpointing) has no autograd history and must not be
        # reused by a forward that needs gradients.
        key = (id(p), dt, p._version, torch.is_grad_enabled())
        hit = self._casts.get(key)
        if hit is None or hit[0] is not p:
            hit = self._casts[key] = (p, p.to(dt))
        return hit[1]

    def _linear_check(self, name, pass_, A, B, C, bias) -> None:
        """Queue the check of one linear-layer matmul (fast mode)."""
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

    def _accumulator(self, name, pass_, A, B, C32, bias) -> None:
        """Precise mode: an fp32 matmul result before it's rounded. Experiments may change it first."""
        if self.on_accumulator is not None:
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
        matmul work (multiply-adds) runs inside protected linear layers: what fast mode checks.
        Matmuls elsewhere, including the ones inside fused attention (scaled_dot_product_attention),
        count as unchecked. Each forward matmul has two backward matmuls of the same size, so the
        share holds for the whole step."""
        total, inside, depth = [0], [0], [0]

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
                elif name.startswith("aten::_scaled_dot_product") or name == "aten::scaled_dot_product_attention":
                    q, k = args[0], args[1]            # fused attention: Q K^T and P V, never checked here
                    total[0] += 2 * q.numel() * k.shape[-2]
                return out

        def enter(*_):
            depth[0] += 1

        def leave(*_):
            depth[0] -= 1

        hooks = []
        for mod in self._protected:
            hooks.append(mod.register_forward_pre_hook(enter))
            hooks.append(mod.register_forward_hook(leave))
        try:
            with torch.no_grad(), _Count():
                forward_fn()
        finally:
            for h in hooks:
                h.remove()
        share = inside[0] / total[0] if total[0] else 0.0
        return {"matmul_multiply_adds": total[0], "in_protected_linear_layers": inside[0],
                "share_checked_in_fast_mode": share, "skipped_layers": dict(self.skipped)}

    # ------------------------------------------------------------------ schedule
    def should_check(self, step: int) -> bool:
        """True on every `every`-th step, and on every step of a window set by after_restart()."""
        if self._dense is not None and self._dense[0] <= step < self._dense[1]:
            return True
        return bool(self.every) and step % self.every == 0

    def after_restart(self, step: int, steps: int = 50) -> "SDCGuard":
        """Call when training resumes from a checkpoint, e.g. after a crash. Every step from `step`
        to `step + steps` is checked, so a GPU that is still bad gets named before it costs another
        run, even if it crashed the last run between two scheduled checks."""
        self._dense = (step, step + steps)
        return self

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
        linear layers are checked (fast mode); otherwise every matmul PyTorch runs is."""
        self.step = step
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
        self._active = True
        self._casts.clear()
        ok = False
        try:
            yield
            ok = True
        finally:
            self._active = False
            self._casts.clear()
            _exact_reductions_off(restore)
            if not ok:
                with self._lock:
                    self._pending.clear()
                    self._groups.clear()
                    self._streams.clear()
        self.flush()

    def _new_epoch(self) -> None:
        """Start a checked step: with fresh_signs, its signed checksums use a pattern no earlier step used,
        so a fault that keeps changing the same outputs in a way one pattern can't see shows on a later step."""
        if self.fresh_signs:
            with self._lock:
                self._epoch += 1

    def __enter__(self):
        if self._depth == 0:
            self._new_epoch()
        if self.exact_reductions and self._depth == 0:
            self._restore_flags = _exact_reductions_on()
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
        if exc_type is None:
            self.flush()
        else:
            with self._lock:
                self._pending.clear()
                self._groups.clear()
                self._streams.clear()
        return result

    # ------------------------------------------------------------------ checking
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)      # this mode is paused here, so no recursion
        is_mm = self._is_matmul.get(func)
        if is_mm is None:
            is_mm = self._is_matmul[func] = op_name(func) in MATMUL_OPS
        if not is_mm:
            return out
        name = op_name(func)
        node = _autograd_node()
        where = node if node is not None else (self._module or "forward")
        with self._lock:
            self.stats["matmuls_seen"] += 1
            self.ops_seen[name] = self.ops_seen.get(name, 0) + 1
            if self.sample_rate < 1.0 and self._rng.random() >= self.sample_rate:
                return out
            for A, B, C, bias, alpha, beta in matmul_checks(name, args, kwargs, out):
                self._queue(name, where, node, A, B, C, bias, alpha, beta)
            if self.immediate or len(self._pending) >= self.max_pending:
                self.flush()
        return out

    def _queue(self, name, where, node, A, B, C, bias, alpha, beta) -> None:
        """Queue one check (its GPU work starts now; results are read at the flush)."""
        acc = self.precise and _cs.acc_supported(A, C)
        if C.dtype not in UNIT_ROUNDOFF and not acc:
            self.stats["skipped_dtype"] += 1
            return
        token = self.on_check_start(name, C) if self.on_check_start is not None else None
        detail = {"op": name, "where": where, "pass": "backward" if node else "forward",
                  "shape_A": tuple(A.shape), "shape_B": tuple(B.shape),
                  "dtype": f"float32 accumulator of {A.dtype}" if acc else str(C.dtype), "device": C.device}
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
        if acc:
            self.stats["checked_fp32"] += 1
        self._pending.append(entry)

    def flush(self) -> list[CheckResult]:
        """Read all queued checks back from the GPU, record them, and raise alerts."""
        with self._lock:
            return self._flush()

    def _flush(self) -> list[CheckResult]:
        if not self._pending:
            return []
        pending, self._pending = self._pending, []
        groups, self._groups = self._groups, {}
        streams, self._streams = self._streams, {}
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
        results, kept = [], []
        it = iter(plain or ())
        for p in pending:
            if p[0] == "s":
                if plain is None:
                    self.stats["checked"] -= 1
                    self.stats["skipped_oom"] += 1
                    continue
                results.append(next(it))
                kept.append(p)
                continue
            g, idx = p[1], p[2]
            if id(g) not in rows:                   # out of memory while reading these back
                self.stats["checked"] -= 1
                self.stats["skipped_oom"] += 1
                continue
            mz, b, r, blk, den = g.row(idx, rows[id(g)][idx])
            finite = math.isfinite(mz)
            results.append(CheckResult(ok=finite and mz <= self.z, max_z=mz, nonfinite=not finite,
                                       batch=b if len(p[4]["shape_A"]) == 3 else None,
                                       row=r, col_block=blk, noise=den))
            kept.append(p)
        first_error = None
        for (_kind, _st, _idx, token, detail), res in zip(kept, results):
            if res.ok and not res.judged:
                self.stats["unjudged"] += 1          # e.g. NaN inputs: no evidence either way
            elif res.ok:
                self.max_z_seen = max(self.max_z_seen, res.max_z)
                if len(self.z_values) < self._z_cap:
                    self.z_values.append(res.max_z)
                elif self._rng.random() < 0.01:
                    self.z_values[self._rng.randrange(self._z_cap)] = res.max_z
            if self.on_check is not None:
                self.on_check(detail["op"], token, res)
            if self.on_result is not None:
                self.on_result(detail, res)
            if not res.ok:
                event = self._alert(detail, res)
                if self.raise_on_alert and first_error is None:
                    first_error = SDCError(event.to_json())
        if first_error is not None:
            raise first_error
        return results

    def _alert(self, detail: dict, res: CheckResult) -> SDCEvent:
        self.stats["alerts"] += 1
        dev = detail.pop("device")
        key = str(dev)
        if key not in self._identity:
            self._identity[key] = device_identity(dev)
        detail.update(max_z=res.max_z, nonfinite=res.nonfinite, batch=res.batch, row=res.row,
                      col_block=res.col_block, noise=res.noise)
        event = SDCEvent(detector="checksum", step=self.step, detail=detail,
                         identity=dict(self._identity[key]))
        self.log.add(event)
        if self.on_alert:
            self.on_alert(event)
        return event
