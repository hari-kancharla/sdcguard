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
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, event: SDCEvent) -> None:
        self.events.append(event)
        if self.path:
            with open(self.path, "a") as f:
                f.write(event.to_json() + "\n")

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
