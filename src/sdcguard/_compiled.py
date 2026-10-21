"""protect() inside torch.compile, fullgraph=True included, tensor parallel included.

On a checked step, a protected layer's forward under compilation is one custom operator,
sdcguard::checked_linear, which the compiler keeps as an opaque call: at run time it runs the same checked
forward as eager mode (guard._checked_forward), and its backward is a second operator that runs the checked
backward. The compiler sees the guard's state (`_active`) while tracing, so a checked and an unchecked version
of the graph are compiled once each; unchecked steps run the plain, fully optimized graph. The matmuls a checked
step runs are cuBLAS calls made by the operator, as in eager mode, not Inductor's own GEMM kernels.

Tensor parallel: both operators have DTensor sharding rules (torch.distributed.tensor.experimental.
register_sharding), so DTensor runs them on each rank's local shards, and each rank checks its own piece, as in
eager mode: replicated, batch or sequence sharded, column parallel (weight and bias sharded by output), and row
parallel (the output a partial sum, checked before DTensor's all-reduce; a replicated bias is split the way
DTensor's own addmm splits it). These are the layouts torch's ColwiseParallel and RowwiseParallel produce, so
DTensor adds no communication; a layer with another layout is redistributed by DTensor to one of them first.

Both operators are tagged cudagraph_unsafe: under mode="reduce-overhead" Inductor runs them outside the CUDA
graph, so their checks run on every step (inside a captured graph a custom operator's Python never runs on
replay, and its checks would silently stop).
"""
from __future__ import annotations

import itertools
import weakref
from typing import Optional

import torch

_GUARDS: "weakref.WeakValueDictionary" = weakref.WeakValueDictionary()
_IDS = itertools.count(1)
_TAGS = tuple(t for t in (getattr(torch.Tag, "cudagraph_unsafe", None),) if t is not None)


def register(guard) -> int:
    i = next(_IDS)
    _GUARDS[i] = guard
    return i


def _empty(like: torch.Tensor) -> torch.Tensor:
    return like.new_empty(0)


@torch.library.custom_op("sdcguard::checked_linear", mutates_args=(), tags=_TAGS)
def checked_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], guard_id: int,
                   name: str, tp: bool = False) -> torch.Tensor:
    from .guard import _checked_forward
    g = _GUARDS.get(guard_id)
    with torch.autocast(device_type=x.device.type, enabled=False):
        if g is None or not g._active:
            return torch.nn.functional.linear(x, w, b)
        with g._local_shards(tp):
            y = _checked_forward(g, name, x, w, b)
    return y if y._base is None else y.clone()       # a fresh tensor (a custom operator's output can't be a view)


@checked_linear.register_fake
def _(x, w, b, guard_id, name, tp=False):
    return x.new_empty((*x.shape[:-1], w.shape[0]))


@torch.library.custom_op("sdcguard::checked_linear_backward", mutates_args=(), tags=_TAGS)
def checked_linear_backward(gy: torch.Tensor, x: torch.Tensor, w: torch.Tensor, guard_id: int, name: str,
                            need_x: bool, need_w: bool, need_b: bool,
                            tp: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .guard import _checked_backward
    g = _GUARDS.get(guard_id)
    with torch.autocast(device_type=gy.device.type, enabled=False):
        if g is None:
            gy2 = gy.reshape(-1, gy.shape[-1])
            gx = gy2.mm(w).view(x.shape) if need_x else None
            gw = gy2.t().mm(x.reshape(-1, x.shape[-1])) if need_w else None
            gb = gy2.sum(0) if need_b else None
        else:
            with g._local_shards(tp):
                gx, gw, gb = _checked_backward(g, name, x, w, gy, need_x, need_w, need_b)
    out = []
    for t in (gx, gw, gb):
        if t is None:
            out.append(_empty(gy))
        else:
            out.append(t if t._base is None else t.clone())
    return tuple(out)


@checked_linear_backward.register_fake
def _(gy, x, w, guard_id, name, need_x, need_w, need_b, tp=False):
    return (gy.new_empty(x.shape) if need_x else gy.new_empty(0),
            gy.new_empty(w.shape) if need_w else gy.new_empty(0),
            gy.new_empty(w.shape[0]) if need_b else gy.new_empty(0))


def _setup(ctx, inputs, output):
    x, w, b, guard_id, name, tp = inputs
    ctx.save_for_backward(x, w)
    ctx.has_bias = b is not None
    ctx.guard_id, ctx.name, ctx.tp = guard_id, name, tp


def _backward(ctx, gy):
    x, w = ctx.saved_tensors
    nx, nw, nb = ctx.needs_input_grad[:3]
    nb = bool(nb and ctx.has_bias)
    gx, gw, gb = checked_linear_backward(gy, x, w, ctx.guard_id, ctx.name, bool(nx), bool(nw), nb, ctx.tp)
    return gx if nx else None, gw if nw else None, gb if nb else None, None, None, None


checked_linear.register_autograd(_backward, setup_context=_setup)


def _register_sharding() -> bool:
    """DTensor sharding rules for both operators, one mesh dimension at a time (DTensor combines them across a
    multi-dimensional mesh). Returns whether DTensors can go through the operators."""
    try:
        from torch.distributed.tensor import Partial, Replicate, Shard
        from torch.distributed.tensor.experimental import register_sharding
    except Exception:  # noqa: BLE001  (a PyTorch build without distributed, or without the API)
        return False
    R = Replicate()

    def forward_rules(x, w, b, guard_id, name, tp=False):
        last = x.ndim - 1

        def bias(p):
            return None if b is None else p
        rest = [None, None, None]
        rules = [([R], [R, R, bias(R)] + rest)]
        rules += [([Shard(i)], [Shard(i), R, bias(R)] + rest) for i in range(last)]      # batch, sequence
        rules.append(([Shard(last)], [R, Shard(0), bias(Shard(0))] + rest))              # column parallel
        rules.append(([Partial()], [Shard(last), Shard(1), bias(Partial())] + rest))     # row parallel
        rules.append(([Partial()], [Partial(), R, bias(Partial())] + rest))              # a partial input
        return rules

    def backward_rules(gy, x, w, guard_id, name, need_x, need_w, need_b, tp=False):
        last = gy.ndim - 1

        def out(gx, gw, gb):                  # gradients that aren't asked for come back empty: replicated
            return [gx if need_x else R, gw if need_w else R, gb if need_b else R]
        rest = [None] * 6
        rules = [(out(R, R, R), [R, R, R] + rest)]
        rules += [(out(Shard(i), Partial(), Partial()), [Shard(i), Shard(i), R] + rest) for i in range(last)]
        rules.append((out(Partial(), Shard(0), Shard(0)), [Shard(last), R, Shard(0)] + rest))  # column parallel
        rules.append((out(Shard(last), Shard(1), R), [R, Shard(last), Shard(1)] + rest))       # row parallel
        rules.append((out(R, Partial(), R), [R, Partial(), R] + rest))                         # a partial input
        return rules

    try:
        register_sharding(torch.ops.sdcguard.checked_linear.default)(forward_rules)
        register_sharding(torch.ops.sdcguard.checked_linear_backward.default)(backward_rules)
    except Exception:  # noqa: BLE001
        return False
    return True


TENSOR_PARALLEL = _register_sharding()
