"""Exact recompute check: catches wrong results from any operation inside activation-checkpointed blocks.

Activation checkpointing already runs each checkpointed block twice: once in the forward pass, and again in the
backward pass to rebuild the activations it didn't keep. With deterministic kernels on a healthy GPU, the two runs
produce bit-identical outputs. This module fingerprints every output of the first run, fingerprints the same
outputs of the recompute, and compares them. Any difference means something inside that block (a matmul,
attention, a normalization, an activation, MoE routing) computed differently the second time.

This is the method behind 12 of the 18 production incidents in ByteDance's AEGIS (OSDI '26, section 5.1.2,
"self-equivalence-based deterministic detection", and section 7.2). It needs no noise model and no threshold, and
it covers operations that matmul checksums can't see. It adds no recomputation of its own (training already
recomputes); the cost is three integer reductions per block output per run.

What it can't see:
  - a fault that gives exactly the same wrong bits both times (a GPU that is consistently wrong in that block):
    the matmul checksums and `sdcguard check` (which compares with a float64 reference) cover that case;
  - blocks that aren't checkpointed, or steps run without autograd (no backward, no recompute); coverage()
    reports which share of the model's parameters ran inside compared blocks;
  - blocks checkpointed with preserve_rng_state=False: dropout masks differ between the runs by design, so those
    blocks are skipped and counted (stats["skipped_rng_not_preserved"]), never compared;
  - with selective activation checkpointing (a context_fn), the operations it saves are not recomputed, so a
    fault in one of them shows the same wrong value both times; those blocks are compared and counted
    (stats["selective_blocks"]);
  - a nondeterministic kernel inside a block makes clean runs differ, which reports as a mismatch. Turn on
    deterministic algorithms where available and run calibrate() on a healthy GPU first; it says whether clean
    recomputes are bit-identical for your model.

A mismatch says the two runs of a block computed different bits. It doesn't say why: a hardware fault, a
software bug that reads uninitialized or out-of-bounds memory (AEGIS caught one of those, section 8), or a
nondeterministic kernel. Run calibrate() and `sdcguard casefile` before blaming a GPU.

Use it on its own:

    from sdcguard.recompute import RecomputeCheck
    rc = RecomputeCheck(on_alert=print).install(model)   # pass the model: references to checkpoint that it
                                                         # already holds (Hugging Face, checkpoint_wrapper) are
                                                         # routed too, so the order doesn't matter
    ...train as usual (torch.utils.checkpoint, Hugging Face gradient_checkpointing_enable(), ...)
    rc.flush()                                        # once per step, after backward: reads results back

or through SDCGuard(recompute=True), which installs it with protect() and flushes it with each checked step.
"""
from __future__ import annotations

import contextlib
import functools
import sys
import threading
from typing import Callable, Optional

import torch

from .events import EventLog, SDCEvent, device_identity

# The fingerprint's position weights: splitmix64's output mixing (Steele, Lea and Flood, OOPSLA 2014) of
# position + _FP_SEED. Written as signed int64 (these are 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9 and
# 0x94D049BB133111EB), with wrapping products and masked (logical) shifts, so the PyTorch version and the Triton
# kernel (_fingerprint.py, which repeats them) compute the same integers.
_FP_SEED = -7046029254386353131
_FP_M1 = -4658895280553007687
_FP_M2 = -7723592293110705685
_FP_CHUNK = 1 << 20               # words per step of the PyTorch version (its int64 temporaries: 8 MB each)
_ORIGINAL: Optional[Callable] = None
_TRUE_ORIGINAL: Optional[Callable] = None   # torch's checkpoint, kept for good (a reference captured while
                                            # installed must still work after uninstall)
_PARTIALS: list = []              # (module, attribute, the partial it held) for partials rerouted to the check
_INSTALLED: list = []             # RecomputeCheck objects currently installed (the newest one records)
_PATCHED: list = []               # (module, attribute) names that held torch's checkpoint and now hold ours
_LOCK = threading.RLock()


def fingerprint(t: torch.Tensor) -> torch.Tensor:
    """A fingerprint of a tensor's raw bits: three int64 sums over its 16-bit words (8-bit ones for 1-byte
    dtypes), mod 2^64: the words, and the words times two pseudo-random odd 64-bit weights of their position
    (splitmix64's mixing of the position). Tensors that differ in one word always get different fingerprints.
    Tensors that differ in several words (neighbouring changes that add up to zero, swaps, a 64-bit element whose
    words change by +1, -2, +1) get the same one only if the difference cancels in both weighted sums: for
    weights that behave like random numbers, a chance of at most about 2^-48 for each sum. A fingerprint, so a
    comparison of two is not a bit-for-bit comparison of the tensors, but misses only by such a coincidence.
    Computed on the tensor's device with no host sync and no full-size int64 copy."""
    from . import _fingerprint, _group
    x = _group.ready(t.detach())            # a tensor-parallel output whose collective may still be running
    if x.numel() == 0:
        return torch.zeros(3, dtype=torch.int64, device=x.device)
    x = x.contiguous().view(-1)
    bits = x.view(torch.int16) if x.element_size() % 2 == 0 else x.view(torch.int8)
    # the kernel is handed the tensor's address: only an ordinary tensor with memory of its own (_group.own_memory)
    if _group.own_memory(bits) and _fingerprint.usable(bits.device, _fingerprint_torch):
        return _fingerprint.fingerprint_bits(bits)     # one pass, the same three sums exactly
    return _fingerprint_torch(bits)


def _lsr(z: torch.Tensor, s: int) -> torch.Tensor:
    """Logical right shift of int64 values (>> is arithmetic on signed integers)."""
    return (z >> s) & ((1 << (64 - s)) - 1)


def _fp_weights(pos: torch.Tensor):
    """The two odd 64-bit weights of each position (int64, wrapping), as _fingerprint's kernel computes them."""
    z = pos + _FP_SEED
    z = (z ^ _lsr(z, 30)) * _FP_M1
    z = (z ^ _lsr(z, 27)) * _FP_M2
    z = z ^ _lsr(z, 31)
    return z | 1, (z ^ _lsr(z, 32)) | 1


def _fingerprint_torch(bits: torch.Tensor) -> torch.Tensor:
    """The three sums of a flat int16 or int8 tensor with PyTorch operations (the reference for the kernel), in
    steps of _FP_CHUNK words. int64 sums and products wrap around in two's complement, on every device, so the
    order of the additions doesn't matter."""
    out = torch.zeros(3, dtype=torch.int64, device=bits.device)
    for i in range(0, bits.numel(), _FP_CHUNK):
        v = bits[i:i + _FP_CHUNK].to(torch.int64)
        w1, w2 = _fp_weights(torch.arange(i, i + v.numel(), device=bits.device, dtype=torch.int64))
        out += torch.stack([v.sum(), (v * w1).sum(), (v * w2).sum()])
    return out


def _outputs(out) -> list:
    """The tensors a block returned (nested tuples, lists and dicts), in a fixed order."""
    found = []

    def walk(o):
        if isinstance(o, torch.Tensor):
            found.append(o)
        elif isinstance(o, (tuple, list)):
            for v in o:
                walk(v)
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k])
    walk(out)
    return found


def _name(function) -> str:
    f = function
    while isinstance(f, functools.partial):
        f = f.func
    owner = getattr(f, "__self__", None)
    if isinstance(owner, torch.nn.Module):
        return type(owner).__name__ + "." + getattr(f, "__name__", "forward")
    if isinstance(f, torch.nn.Module):
        return type(f).__name__
    return getattr(f, "__qualname__", None) or type(f).__name__


class RecomputeCheck:
    """Compares each checkpointed block's forward outputs with its recompute, through their fingerprints."""

    def __init__(self, on_alert: Optional[Callable[[SDCEvent], None]] = None, log: Optional[EventLog] = None,
                 log_path: Optional[str] = None, max_pending: int = 8192):
        self.on_alert = on_alert
        self.log = log if log is not None else EventLog(log_path)
        self.max_pending = max_pending
        self.step: Optional[int] = None
        self.enabled = True
        self.stats = {"blocks_run": 0, "recomputes_compared": 0, "mismatches": 0, "never_recomputed": 0,
                      "dropped_queue_full": 0, "skipped_rng_not_preserved": 0, "selective_blocks": 0}
        self._models: list = []       # weak references to models whose checkpoint references install() reroutes
        self._pending: list = []      # (name, call index, first fingerprints, recompute fingerprints, device)
        self._calls = 0
        self._open: dict = {}         # call index -> [name, first fingerprints, recomputed?]
        self._installed = False
        self._identity: dict = {}

    # -------------------------------------------------------------- install / uninstall
    def install(self, model: Optional[torch.nn.Module] = None) -> "RecomputeCheck":
        """Route torch.utils.checkpoint.checkpoint through this check (also checkpoint_sequential, which calls
        it, and names other modules imported it under). With `model`, also the references its modules already
        hold: Hugging Face's gradient_checkpointing_enable() and torch's checkpoint_wrapper store a
        functools.partial of checkpoint on the module, so without the model those blocks would be skipped when
        checkpointing was turned on first. Uninstall with uninstall()."""
        global _ORIGINAL, _TRUE_ORIGINAL
        import weakref
        import torch.utils.checkpoint as tuc
        with _LOCK:
            if model is not None and not any(r() is model for r in self._models):
                self._models.append(weakref.ref(model))
            if self._installed:
                self._reroute()
                return self
            if _ORIGINAL is None:
                _ORIGINAL = tuc.checkpoint
                if _TRUE_ORIGINAL is None:
                    _TRUE_ORIGINAL = _ORIGINAL
                tuc.checkpoint = _dispatch
                # code that imported checkpoint by name already holds the original function (Hugging Face:
                # `from torch.utils.checkpoint import checkpoint` in modeling_utils; torch's checkpoint_wrapper
                # imports it under another name): point those names at the check too, and put them back later
                for mod in list(sys.modules.values()):
                    d = getattr(mod, "__dict__", None)
                    if not isinstance(d, dict) or mod is tuc or d.get("_dispatch") is _dispatch:
                        continue                  # (this module keeps the original in _ORIGINAL)
                    for attr, val in list(d.items()):
                        if val is _ORIGINAL:
                            setattr(mod, attr, _dispatch)
                            _PATCHED.append((mod, attr))
            _INSTALLED.append(self)
            self._installed = True
            self._reroute()
        return self

    def attach(self, model: torch.nn.Module) -> "RecomputeCheck":
        """Remember `model`: install() (now or later) routes the checkpoint references its modules hold."""
        import weakref
        with _LOCK:
            if not any(r() is model for r in self._models):
                self._models.append(weakref.ref(model))
            if self._installed:
                self._reroute()
        return self

    def _reroute(self) -> None:
        """Point partials of torch's checkpoint stored on the attached models' modules at the check."""
        for ref in list(self._models):
            m = ref()
            if m is None:
                self._models.remove(ref)
                continue
            for mod in m.modules():
                for attr, val in list(vars(mod).items()):
                    if isinstance(val, functools.partial) and val.func in (_ORIGINAL, _TRUE_ORIGINAL):
                        new = functools.partial(_dispatch, *val.args, **(val.keywords or {}))
                        mod.__dict__[attr] = new
                        _PARTIALS.append((mod, attr, val, new))

    def coverage(self, model: torch.nn.Module, step_fn: Callable[[], None]) -> dict:
        """Run step_fn() (one forward and backward) once and report what the check covers: how many blocks were
        compared, and which share of the model's parameters belong to modules that ran inside a compared
        block (the rest, such as embeddings and the output layer when only decoder layers are checkpointed,
        get no recompute check)."""
        inside: set = set()
        depth = [0]

        def hook(mod, inp, out):
            if depth[0] > 0:
                inside.add(id(mod))
        h = torch.nn.modules.module.register_module_forward_hook(hook)
        saved_cp = self.checkpoint

        def counting(function, *a, **k):
            def f(*x, **y):
                depth[0] += 1
                try:
                    return function(*x, **y)
                finally:
                    depth[0] -= 1
            return saved_cp(f, *a, **k)
        before = dict(self.stats)
        was_installed, was_enabled = self._installed, self.enabled
        self.checkpoint = counting
        self.enabled = True
        try:
            self.install(model)
            step_fn()
            self.flush()
        finally:
            self.checkpoint = saved_cp
            self.enabled = was_enabled
            h.remove()
            if not was_installed:
                self.uninstall()
        total = sum(p.numel() for p in model.parameters())
        covered = sum(p.numel() for mod in model.modules() if id(mod) in inside
                      for p in mod.parameters(recurse=False))
        outside = sorted({type(mod).__name__ for mod in model.modules()
                          if any(True for _ in mod.parameters(recurse=False)) and id(mod) not in inside})
        return {"blocks_compared": self.stats["recomputes_compared"] - before["recomputes_compared"],
                "blocks_skipped_rng": self.stats["skipped_rng_not_preserved"] - before["skipped_rng_not_preserved"],
                "parameter_share_inside_compared_blocks": round(covered / total, 4) if total else 0.0,
                "module_types_with_parameters_outside": outside[:20]}

    def uninstall(self) -> None:
        global _ORIGINAL
        import torch.utils.checkpoint as tuc
        with _LOCK:
            if not self._installed:
                return
            _INSTALLED.remove(self)
            self._installed = False
            if not _INSTALLED and _ORIGINAL is not None:
                tuc.checkpoint = _ORIGINAL
                for mod, attr in _PATCHED:
                    if getattr(mod, attr, None) is _dispatch:
                        setattr(mod, attr, _ORIGINAL)
                _PATCHED.clear()
                for mod, attr, old, new in _PARTIALS:
                    if mod.__dict__.get(attr) is new:
                        mod.__dict__[attr] = old
                _PARTIALS.clear()
                _ORIGINAL = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # -------------------------------------------------------------- the wrapped checkpoint
    def checkpoint(self, function, *args, **kwargs):
        """Drop-in for torch.utils.checkpoint.checkpoint that also compares the recompute."""
        original = _ORIGINAL or _TRUE_ORIGINAL or __import__("torch.utils.checkpoint",
                                                             fromlist=["checkpoint"]).checkpoint
        if function is None or not self.enabled:
            return original(function, *args, **kwargs)
        if kwargs.get("preserve_rng_state", True) is False:
            # the recompute draws new random numbers (dropout): the two runs differ by design
            with _LOCK:
                self.stats["skipped_rng_not_preserved"] += 1
            return original(function, *args, **kwargs)
        if kwargs.get("context_fn") not in (None, getattr(torch.utils.checkpoint, "noop_context_fn", None)):
            with _LOCK:
                self.stats["selective_blocks"] += 1
        with _LOCK:
            self._calls += 1
            idx = self._calls
            self.stats["blocks_run"] += 1
        name = _name(function)
        state = {"runs": 0}

        def wrapped(*a, **k):
            out = function(*a, **k)
            try:
                with torch.no_grad():
                    fps = [fingerprint(t) for t in _outputs(out)]
            except Exception:  # noqa: BLE001  (an exotic output type: never break training over a check)
                return out
            state["runs"] += 1
            if state["runs"] == 1:
                with _LOCK:
                    if len(self._open) >= self.max_pending:          # never flushed (e.g. eval loops): drop oldest
                        self._open.pop(next(iter(self._open)))
                        self.stats["never_recomputed"] += 1
                    self._open[idx] = [name, fps]
            else:
                with _LOCK:
                    first = self._open.pop(idx, None)
                    if first is not None and len(first[1]) == len(fps):
                        if len(self._pending) >= self.max_pending:
                            self.stats["dropped_queue_full"] += 1
                        else:
                            dev = fps[0].device if fps else None
                            self._pending.append((first[0], idx, first[1], fps, dev))
            return out

        # recompute must run the whole block, so its outputs exist to compare (non-reentrant checkpointing
        # otherwise stops as soon as the saved tensors it needs are rebuilt); this only has to be set at forward
        early = getattr(torch.utils.checkpoint, "set_checkpoint_early_stop", None)
        ctx = early(False) if early is not None else contextlib.nullcontext()
        with ctx:
            return original(wrapped, *args, **kwargs)

    # -------------------------------------------------------------- results
    def flush(self) -> list[SDCEvent]:
        """Compare every recompute queued since the last flush (one device-to-host read per device) and report
        mismatches. Blocks whose forward ran but were never recomputed (no backward) are counted and dropped."""
        with _LOCK:
            pending, self._pending = self._pending, []
            never, self._open = len(self._open), {}
            self.stats["never_recomputed"] += never
        if not pending:
            return []
        # every fingerprint of a device in one comparison and one read-back: the step's blocks together, not a
        # stack, compare and reduction per block (about 70 small GPU operations a step for 8 blocks, which cost
        # more than the fingerprints did)
        by_dev: dict = {}
        for i, p in enumerate(pending):
            if p[2]:
                by_dev.setdefault(str(p[2][0].device), []).append(i)
        verdicts: dict = {}
        for idxs in by_dev.values():
            d0 = pending[idxs[0]][2][0].device
            first = [x.to(d0) for i in idxs for x in pending[i][2]]
            again = [x.to(d0) for i in idxs for x in pending[i][3]]
            differ = (torch.stack(first) != torch.stack(again)).any(dim=-1).cpu().tolist()
            pos = 0
            for i in idxs:
                k = len(pending[i][2])
                verdicts[i] = [j for j, bad in enumerate(differ[pos:pos + k]) if bad]
                pos += k
        events = []
        err = None
        for i, (name, idx, _a, _b, dev) in enumerate(pending):
            self.stats["recomputes_compared"] += 1
            bad = verdicts.get(i) or []
            if not bad:
                continue
            self.stats["mismatches"] += 1
            key = str(dev)
            if key not in self._identity:
                try:
                    self._identity[key] = device_identity(dev if isinstance(dev, torch.device) else None)
                except Exception:  # noqa: BLE001
                    self._identity[key] = {}
            ev = SDCEvent(detector="recompute", step=self.step,
                          detail={"where": name, "block_call": idx, "outputs_that_differ": bad,
                                  "reading": "this checkpointed block gave different bits when training recomputed "
                                             "it: an operation inside it (matmul, attention, normalization, "
                                             "activation or routing) computed a different result the second time. "
                                             "On a healthy GPU with deterministic kernels the two runs match "
                                             "exactly; rerun calibrate() if nondeterministic kernels are possible"},
                          identity=dict(self._identity[key]))
            self.log.add(ev)
            events.append(ev)
        for ev in events:
            if self.on_alert:
                try:
                    self.on_alert(ev)
                except Exception as e:  # noqa: BLE001
                    err = err or e
        if err is not None:
            raise err
        return events

    def calibrate(self, step_fn: Callable[[], None], steps: int = 3) -> dict:
        """Run `step_fn` (one forward and backward with checkpointing) a few times on a healthy GPU and report
        whether every recompute was bit-identical. Mismatches here mean a nondeterministic kernel inside some
        block, not a fault: fix determinism (or leave recompute checking off) before relying on alerts."""
        before = dict(self.stats)
        alerts = []
        saved, self.on_alert = self.on_alert, (lambda ev: alerts.append(ev))
        log_saved = self.log
        self.log = EventLog(None)
        try:
            with self:
                for _ in range(steps):
                    step_fn()
                    self.flush()
        finally:
            self.on_alert, self.log = saved, log_saved
        compared = self.stats["recomputes_compared"] - before["recomputes_compared"]
        return {"recomputes_compared": compared, "mismatches": len(alerts),
                "blocks_that_differed": sorted({ev.detail["where"] for ev in alerts}),
                "deterministic": compared > 0 and not alerts,
                "note": None if compared else "no recompute ran: is gradient checkpointing on, and did step_fn call "
                                              "backward()?"}


def _dispatch(function=None, *args, **kwargs):
    """The function installed as torch.utils.checkpoint.checkpoint."""
    with _LOCK:
        rc = _INSTALLED[-1] if _INSTALLED else None
    if rc is None:          # a reference captured while installed, called after uninstall: plain checkpoint
        orig = _ORIGINAL or _TRUE_ORIGINAL or __import__("torch.utils.checkpoint", fromlist=["checkpoint"]).checkpoint
        if orig is _dispatch:
            raise RuntimeError("torch.utils.checkpoint.checkpoint was replaced by something else that calls back here")
        return orig(function, *args, **kwargs)
    return rc.checkpoint(function, *args, **kwargs)
