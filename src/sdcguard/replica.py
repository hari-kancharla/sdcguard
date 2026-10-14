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

from .events import SDCEvent, device_identity

_INT_VIEW = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@torch.no_grad()
def weight_fingerprint(params: Iterable[torch.Tensor]) -> torch.Tensor:
    """Two int64 numbers summarizing the exact bits of all parameters.

    [plain sum of bit patterns, position-weighted sum] so a single flipped bit
    always changes the result, and swapped values usually do too.
    """
    total = None
    for p in params:
        bits = p.detach().reshape(-1).view(_INT_VIEW[p.element_size()]).to(torch.int64)
        w = (torch.arange(bits.numel(), device=bits.device, dtype=torch.int64) % 65521) + 1
        fp = torch.stack([bits.sum(), (bits * w).sum()])
        total = fp if total is None else total + fp
    if total is None:
        raise ValueError("no parameters")
    return total


@torch.no_grad()
def check_replicas(params: Iterable[torch.Tensor], group=None, step: Optional[int] = None):
    """All ranks call this together. Returns (suspect_ranks, event or None).

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
        event = SDCEvent(detector="replica", step=step,
                         detail={"suspect_ranks": suspects, "world": world},
                         identity=device_identity(fp.device))
    return suspects, event
