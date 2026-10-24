"""Detector 2: the canary batch.

Every N steps, all GPUs run the same small batch through the same weights and
compare a fingerprint of every layer's output. A rank that disagrees with the
majority is reported as suspect, together with the first disagreeing layer call.
The discrepancy can come from hardware, software or different execution state;
the fingerprint alone does not identify its cause. This can expose differences
in any covered op (norms, attention, MoE routing), not just matmuls. Same idea as phase 1 of
ByteDance's SDCHunter (OSDI '26).

With `tol=0` (the default) each layer call's fingerprint is of its output's raw bits: the three
position-weighted sums of `recompute.fingerprint`, with the output's dtype and shape. Two outputs that differ in
one value always disagree, and so do swapped values and changes that add up to zero (but for a coincidence of
about 2^-48 per sum). Some GPU kernels aren't bit-for-bit deterministic; for them `tol` > 0 compares the sum and
the sum of |values| of each output within that relative tolerance instead, which can't see swapped values or
changes that cancel in both sums.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

from .events import suspect_event

# dtypes in a fixed order, for the fingerprint's dtype column (a dtype not listed gets -1, still compared)
_DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int64, torch.int32, torch.int16,
           torch.int8, torch.uint8, torch.bool, torch.complex64, torch.complex128)
_PAD = torch.iinfo(torch.int64).min      # a missing layer call in an exact fingerprint (no real row has it)
_MASK = (1 << 64) - 1


def _shape_code(shape) -> int:
    """The shape folded into one signed 64-bit integer (splitmix64's mixing after each dimension)."""
    z = 0x9E3779B97F4A7C15 * (len(shape) + 1) & _MASK
    for d in shape:
        z = (z ^ (int(d) + 0x632BE59BD9B4E019)) & _MASK
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
        z ^= z >> 31
    return z - (1 << 64) if z >= 1 << 63 else z


def exact_row(t: torch.Tensor) -> torch.Tensor:
    """(6,) int64 on the CPU: the three raw-bit sums of recompute.fingerprint, the dtype's place in _DTYPES (-1 if
    not there), the number of elements and the shape's code."""
    from .recompute import fingerprint
    meta = [_DTYPES.index(t.dtype) if t.dtype in _DTYPES else -1, t.numel(), _shape_code(t.shape)]
    return torch.cat([fingerprint(t).cpu(), torch.tensor(meta, dtype=torch.int64)])


class Canary:
    def __init__(self, model: torch.nn.Module, tol: float = 0.0,
                 forward_fn: Optional[Callable] = None,
                 include: Optional[Callable[[str, torch.nn.Module], bool]] = None):
        self.model = model
        self.tol = tol
        self.forward_fn = forward_fn or (lambda m, batch: m(batch))
        include = include or (lambda name, mod: len(list(mod.children())) == 0)  # leaf modules
        self.modules = [(n or type(m).__name__, m) for n, m in model.named_modules() if include(n, m)]
        self.last_names: list[str] = []

    @torch.no_grad()
    def fingerprints(self, batch) -> torch.Tensor:
        """One row per layer call, in execution order. tol 0: (calls, 6) int64, exact_row of each layer's output.
        tol > 0: (calls, 2) float64, [sum, sum of |x|]."""
        rows, names = [], []
        hooks = []
        exact = self.tol == 0
        for name, mod in self.modules:
            def hook(_m, _inp, out, name=name):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
                    if exact:
                        rows.append(exact_row(t.detach()))
                    else:
                        acc = torch.float32 if t.device.type == "mps" else torch.float64  # no fp64 on Apple GPUs
                        tt = t.detach().to(acc)
                        rows.append(torch.stack([tt.sum(), tt.abs().sum()]).double().cpu())
                    names.append(name)
            hooks.append(mod.register_forward_hook(hook))
        # A model can deliberately mix training and eval modules (frozen BatchNorm, teacher branches).
        # Calling model.train(previous_root_flag) afterwards would overwrite every child's own mode.
        training_states = [(mod, mod.training) for mod in self.model.modules()]
        try:
            self.model.eval()   # no dropout: every rank must compute the same thing
            self.forward_fn(self.model, batch)
        finally:
            for mod, was_training in training_states:
                mod.training = was_training
            for h in hooks:
                h.remove()
        self.last_names = names
        if rows:
            return torch.stack(rows)
        return torch.empty(0, 6, dtype=torch.int64) if exact else torch.empty(0, 2, dtype=torch.float64)

    def _agree(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        if not a.is_floating_point() or not b.is_floating_point():   # exact fingerprints: every word equal
            return a.dtype == b.dtype and a.shape == b.shape and bool((a == b).all())
        both_nan = torch.isnan(a) & torch.isnan(b)
        if bool(((a == b) | both_nan).all()):
            return True
        if self.tol == 0:
            return False
        # an infinity is close only to the same infinity (inf - inf is NaN, inf <= inf is True: a relative
        # tolerance would call a finite value close to +inf)
        finite = torch.isfinite(a) & torch.isfinite(b)
        close = finite & ((a - b).abs() <= self.tol * torch.maximum(a.abs(), b.abs()) + 1e-300)
        return bool((close | (a == b) | both_nan).all())

    @torch.no_grad()
    def check(self, batch, group=None, step: Optional[int] = None):
        """All ranks call this together. Returns (suspect_ranks, first_bad_layer, event): the job's global ranks,
        also with a group."""
        fp = self.fingerprints(batch)
        dev = next(self.model.parameters()).device
        # all_gather needs equal shapes on the backend's device (NCCL: GPU, gloo: CPU)
        comm_dev = dev if dev.type == "cuda" else torch.device("cpu")
        n = torch.tensor([fp.shape[0]], device=comm_dev)
        dist.all_reduce(n, op=dist.ReduceOp.MAX, group=group)
        n = int(n)
        if fp.shape[0] < n:   # a rank whose model took a different path: pad, the rows will disagree
            pad = _PAD if not fp.is_floating_point() else float("inf")
            fp = torch.cat([fp, torch.full((n - fp.shape[0], fp.shape[1]), pad, dtype=fp.dtype)])
        fp = fp.to(comm_dev)
        world = dist.get_world_size(group)
        gathered = [torch.empty_like(fp) for _ in range(world)]
        dist.all_gather(gathered, fp, group=group)
        gathered = [g.cpu() for g in gathered]
        for li in range(n):
            vals = [g[li] for g in gathered]
            # group ranks that agree with each other at this layer call
            groups: list[list[int]] = []
            for r, v in enumerate(vals):
                for grp in groups:
                    if self._agree(vals[grp[0]], v):
                        grp.append(r)
                        break
                else:
                    groups.append([r])
            if len(groups) > 1:
                groups.sort(key=len, reverse=True)
                if len(groups[0]) > len(groups[1]):
                    suspects = [r for grp in groups[1:] for r in grp]
                else:
                    suspects = list(range(world))   # no clear majority
                name = self.last_names[li] if li < len(self.last_names) else f"call {li}"
                event = suspect_event("canary", step, suspects, world, group, dev,
                                      {"first_bad_layer": name, "where": name,
                                       "check": "the same canary batch must give the same outputs on every rank"})
                return event.detail["suspect_ranks"], name, event
        return [], None, None
