"""Detector 3: data-parallel replicas must hold bit-identical weights.

With DDP, every GPU applies the same averaged gradients, so after each optimizer
step all copies of the model must match bit for bit. A GPU that corrupts its own
update (optimizer math, a bad memory write) drifts away silently. We compare an
exact integer checksum of the raw weight bits across ranks.

Only for full replicas (DDP, or the replica groups of HSDP). Fully sharded FSDP
ranks hold different shards, so their checksums differ by design.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from .events import suspect_event

_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _local(p: torch.Tensor) -> torch.Tensor:
    """The tensor this rank holds: a DTensor's local shard, or the tensor itself."""
    to_local = getattr(p, "to_local", None)
    return to_local() if callable(to_local) else p


def is_sharded(p: torch.Tensor) -> bool:
    """A DTensor (FSDP2, tensor parallel): ranks hold different shards, so whole-group checksums differ."""
    return callable(getattr(p, "to_local", None)) and hasattr(p, "placements")


@torch.no_grad()
def weight_fingerprint(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Three int64 numbers summarizing the exact bits of all parameters (a DTensor's local shard).

    Each parameter gets recompute.fingerprint's three sums of its raw bits (a single flipped bit always changes
    them, swapped values usually do), weighted by the parameter's position so two swapped parameters change the
    result too. One pass over the weights with no full-size integer copy (on NVIDIA GPUs a single Triton kernel
    per parameter, otherwise PyTorch reductions; both give the same integers).
    """
    from .recompute import fingerprint
    total = None
    for i, p in enumerate(params):
        fp = fingerprint(_local(p)) * (i + 1)
        total = fp if total is None else total + fp
    if total is None:
        raise ValueError("no parameters")
    return total


@torch.no_grad()
def check_replicas(params: Iterable[torch.Tensor], group=None, step: Optional[int] = None):
    """All ranks call this together. Returns (suspect_ranks, event or None): the job's global ranks, also with a
    group.

    Majority vote: ranks that disagree with the most common fingerprint are suspects.
    With only 2 ranks a disagreement can't say which one is wrong, so both are listed.
    """
    fp = weight_fingerprint(params)
    world = dist.get_world_size(group)
    gathered = [torch.empty_like(fp) for _ in range(world)]
    dist.all_gather(gathered, fp, group=group)
    keys = [tuple(g.tolist()) for g in gathered]
    majority, votes = Counter(keys).most_common(1)[0]
    suspects = [r for r, key in enumerate(keys) if key != majority]
    if suspects and votes == len(keys) - len(suspects) and votes <= len(suspects):
        suspects = list(range(world))   # no clear majority
    event = None
    if suspects:
        event = suspect_event("replica", step, suspects, world, group, fp.device,
                              {"where": "data-parallel weights", "check": "replica weights must match bit for bit"})
        suspects = event.detail["suspect_ranks"]
    return suspects, event
