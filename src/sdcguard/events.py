"""Alert events: what every detector reports when it finds something."""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch


def device_identity(device: Optional[torch.device] = None) -> dict:
    """Host, rank and GPU identity for an alert: enough to find and pull the physical chip.

    For NVIDIA GPUs: name, UUID (as nvidia-smi prints it), PCI bus id, and the physical
    index (CUDA_VISIBLE_DEVICES is taken into account).
    """
    info = {"host": socket.gethostname(), "rank": None, "device": None, "gpu_uuid": None}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        info["rank"] = torch.distributed.get_rank()
    if device is not None:
        info["device"] = str(device)
        if device.type == "cuda":
            idx = device.index if device.index is not None else torch.cuda.current_device()
            try:
                props = torch.cuda.get_device_properties(idx)
                info["gpu_name"] = props.name
                uuid = str(getattr(props, "uuid", "") or "")
                info["gpu_uuid"] = (uuid if uuid.startswith("GPU-") else f"GPU-{uuid}") if uuid else None
                dom, bus, dev = (getattr(props, a, None) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
                if bus is not None:
                    info["pci_bus_id"] = f"{dom or 0:08X}:{bus:02X}:{dev or 0:02X}.0"
            except Exception:  # noqa: BLE001  (identity is best effort; never break training)
                pass
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            ids = [v.strip() for v in visible.split(",")] if visible else None
            info["gpu_index"] = ids[idx] if ids and idx < len(ids) else str(idx)
    return info


@dataclass
class SDCEvent:
    detector: str            # "checksum" | "canary" | "replica"
    step: Optional[int]
    detail: dict = field(default_factory=dict)
    identity: dict = field(default_factory=dict)
    time: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


class EventLog:
    """Keeps events in memory and optionally appends them to a JSONL file."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.events: list[SDCEvent] = []
        self.write_errors = 0                 # appends to the file that failed (the events are still kept here)
        if path:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            except OSError:
                pass                          # reported at the first write that fails

    def add(self, event: SDCEvent) -> None:
        """Keep the event, then append it to the file. A write that fails (a full disk, a path that is a folder)
        is counted and logged, never raised: an alert must not crash training or serving, nor be lost on the
        way to on_alert."""
        self.events.append(event)
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(event.to_json() + "\n")
            except OSError as e:
                self.write_errors += 1
                if self.write_errors == 1:
                    import logging
                    logging.getLogger("sdcguard").warning("sdcguard: couldn't write alerts to %s (%s); they are "
                                                          "still in the guard's log and go to on_alert",
                                                          self.path, e)

    def first_step(self) -> Optional[int]:
        steps = [e.step for e in self.events if e.step is not None]
        return min(steps) if steps else None


def rollback_checkpoint(first_alert_step: int, checkpoint_steps: list[int], check_every: int = 1,
                        windows: int = 1) -> Optional[int]:
    """Newest checkpoint saved no later than the last check that passed.

    A checkpoint saved at step s holds the weights after step s. With checks every
    `check_every` steps, the last check that passed ran at first_alert_step - check_every.
    Assumption: a GPU that had already gone bad would have failed that check too. Faults that
    come and go can slip past one check (a step with no fault, or one too small to flag), so
    windows=2 steps back one more check interval for extra safety, at the cost of redoing
    more steps. Returns None if no checkpoint is old enough.
    """
    cutoff = first_alert_step - check_every * max(1, windows)
    safe = [s for s in checkpoint_steps if s <= cutoff]
    return max(safe) if safe else None


def _global_rank(group, r: int) -> int:
    """Rank r of `group` as a rank of the whole job (r itself without a group)."""
    if group is None:
        return r
    import torch.distributed as dist
    get = getattr(dist, "get_global_rank", None) or getattr(dist.distributed_c10d, "_get_global_rank", None)
    try:
        return int(get(group, r))
    except Exception:  # noqa: BLE001  (a group this PyTorch can't map: its own rank is still listed)
        return r


def suspect_event(detector: str, step, suspects: list, world: int, group, device, detail: dict) -> SDCEvent:
    """An alert from a cross-rank comparison (replica weights, canary outputs). All ranks call it together, after
    finding the same suspects (ranks of `group`, `world` its size). The event names the suspect GPU (when there is
    exactly one, with a clear majority) instead of the reporting rank, lists every suspect's identity, and marks
    one rank as the reporter, the one that delivers it to on_alert (one webhook post per mismatch, not one per
    rank). Ranks in the event are the job's global ranks, the ones event.identity and the launcher use; with a
    group, suspect_group_ranks keeps them as ranks of the group."""
    import torch.distributed as dist
    me = dict(device_identity(device))
    ids: list = [None] * world
    try:
        dist.all_gather_object(ids, me, group=group)
    except Exception:  # noqa: BLE001  (identities are best effort; the ranks are still named)
        ids = [None] * world
    rank = dist.get_rank(group)
    clear = len(suspects) < world
    reporter = min(r for r in range(world) if r not in suspects) if clear else 0
    who = [{"rank": _global_rank(group, r), **{k: (ids[r] or {}).get(k) for k in ("host", "gpu_uuid", "gpu_name",
                                                                                   "pci_bus_id")}}
           for r in suspects]
    full = {"suspect_ranks": [_global_rank(group, r) for r in suspects], "world": world, "suspects": who,
            "reported_by_rank": _global_rank(group, rank), "reporter": rank == reporter, "clear_majority": clear}
    if group is not None:
        full["suspect_group_ranks"] = list(suspects)
    full.update(detail)
    identity = dict(ids[suspects[0]] or me) if clear and len(suspects) == 1 else me
    return SDCEvent(detector=detector, step=step, detail=full, identity=identity)
