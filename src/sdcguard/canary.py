"""Detector 2: the canary batch.

Every N steps, all GPUs run the same small batch through the same weights and
compare a fingerprint of every layer's output. A rank that disagrees with the
majority is reported as suspect, together with the first disagreeing layer call.
The discrepancy can come from hardware, software or different execution state;
the fingerprint alone does not identify its cause. This can expose differences
in any covered op (norms, attention, MoE routing), not just matmuls. Same idea as phase 1 of
ByteDance's SDCHunter (OSDI '26).

Some GPU kernels aren't bit-for-bit deterministic, so comparisons use a relative
tolerance (`tol`, 0 = exact).
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

from .events import SDCEvent, device_identity


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
        """(layer calls, 2) float64: [sum, sum of |x|] of each layer's output, in execution order."""
        rows, names = [], []
        hooks = []
        for name, mod in self.modules:
            def hook(_m, _inp, out, name=name):
                t = out[0] if isinstance(out, (tuple, list)) else out
                if isinstance(t, torch.Tensor):
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
        return torch.stack(rows) if rows else torch.empty(0, 2, dtype=torch.float64)

    def _agree(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        both_nan = torch.isnan(a) & torch.isnan(b)
        if bool(((a == b) | both_nan).all()):
            return True
        if self.tol == 0:
            return False
        # an infinity is close only to the same infinity (inf - inf is NaN, inf <= inf is True: a relative
        # tolerance would call a finite value close to +inf; found by an independent review, Oct 19, 2026)
        finite = torch.isfinite(a) & torch.isfinite(b)
        close = finite & ((a - b).abs() <= self.tol * torch.maximum(a.abs(), b.abs()) + 1e-300)
        return bool((close | (a == b) | both_nan).all())

    @torch.no_grad()
    def check(self, batch, group=None, step: Optional[int] = None):
        """All ranks call this together. Returns (suspect_ranks, first_bad_layer, event)."""
        fp = self.fingerprints(batch)
        dev = next(self.model.parameters()).device
        # all_gather needs equal shapes on the backend's device (NCCL: GPU, gloo: CPU)
        comm_dev = dev if dev.type == "cuda" else torch.device("cpu")
        n = torch.tensor([fp.shape[0]], device=comm_dev)
        dist.all_reduce(n, op=dist.ReduceOp.MAX, group=group)
        n = int(n)
        if fp.shape[0] < n:   # a rank whose model took a different path: pad, the rows will disagree
            fp = torch.cat([fp, torch.full((n - fp.shape[0], 2), float("inf"), dtype=fp.dtype)])
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
                event = SDCEvent(detector="canary", step=step,
                                 detail={"suspect_ranks": suspects, "first_bad_layer": name,
                                         "world": world},
                                 identity=device_identity(dev))
                return suspects, name, event
        return [], None, None
