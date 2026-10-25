"""Scheduled weight integrity for serving (ServeGuard.weight_scan(), SDCGUARD_WEIGHT_SCAN=1 in the vLLM plugin).

The arithmetic checks see a changed weight only once it moves an output block past their line, so most single bit
flips in a served model's weights go unseen (docs/METHOD.md, "Bit flips in a served model's weights": the flips of a
weight's top exponent bit were caught, nearly all others weren't). This scan doesn't wait for an output to go wrong:
at each read of the results, the next layers in turn have their tensors fingerprinted on the GPU and compared with
their reference fingerprint, so every registered tensor is compared once per pass over the model. A single changed
bit (and nearly any other change) that persists until its layer's turn is named, however small its effect on the
outputs.

What is scanned: each layer the guard checks (its weight and bias; for a quantized layer, the tensors its kernel
reads: packed weights, scales, zero points, bias), and layers registered for the scan alone (the vLLM plugin adds the
bf16 and fp16 linear layers SDCGUARD_SHARE leaves out of the arithmetic checks). What isn't: a fault in the
arithmetic itself (the weights stay right), a change undone before its layer's turn, tensors that share their memory
with another layer's (vLLM's CPU offloading points several layers at one buffer) or that moved after their first
comparison, and anything not registered (embeddings, the LM head, the KV cache; in the plugin also quantized layers
SDCGUARD_SHARE leaves out and layers the arithmetic check can't take, which it lists as not scanned).

The reference: an entry's fingerprint when it was registered (at load) or refreshed. Until its first comparison the
reference is provisional: if the layer's tensors are then somewhere else (vLLM processes a reloaded layer's weights
in temporary tensors and copies them into the originals afterwards), the reference is taken again from the tensors
as they are at that turn. After its first comparison, tensors that moved aren't compared (and are listed), so a move
is never called a change.

The fingerprint of a tensor's raw bits (viewed as 16-bit, 32-bit or 8-bit integers by its element size; 8-byte
elements as pairs of 32-bit ones) is two int64 sums: of every value, and of each 4,096-value block's sum weighted by
the block's position. A single bit flip changes the first (by 2^k, never a multiple of 2^64). Changes that cancel in
the first are seen only when they don't also cancel in the second: within one 4,096-value block they do (two opposite
flips of one bit in one block aren't seen). It is computed on the tensor's own device in chunks of 4M values (about
32 MB of temporary memory, plus a contiguous copy of a tensor that isn't contiguous), and on a CUDA device its result is read back without waiting for the GPU: the copy is
started at one read and taken at a later one (or at once when the guard is asked to wait).

Cost: each read compares layers until it has compared about SDCGUARD_WEIGHT_SCAN_MB (64 by default) of their tensors
(the last layer of a read can take it past). A pass over a model takes about (its bytes / that budget) reads.

An intended weight update (an RL weight sync, a reload) looks the same as corruption: ServeGuard.refresh() and
refresh_layer() make the references provisional again, and so does refresh() here.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Sequence

import torch

from .events import SDCEvent, device_identity

log = logging.getLogger("sdcguard")

BLOCK = 4096
CHUNK = BLOCK * 1024                    # values per fingerprint step (an int64 copy of one chunk is 32 MB)
POSITION_MOD = 65521
DEFAULT_BUDGET_MB = 64.0


def _int_view(t: torch.Tensor) -> torch.Tensor:
    """A tensor's raw bits as a flat integer tensor (no copy for a contiguous tensor)."""
    flat = t.detach().contiguous().view(-1)
    es = flat.element_size()
    if es == 2:
        return flat.view(torch.int16)
    if es == 4:
        return flat.view(torch.int32)
    if es == 8:
        return flat.view(torch.int64).view(torch.int32)
    return flat.view(torch.uint8)


def fingerprint(t: Optional[torch.Tensor]) -> torch.Tensor:
    """[2] int64 on the tensor's device: the sum of its raw bit values, and the sum of each BLOCK-value block's sum
    times (block index mod 65521) + 1. A missing or empty tensor gives zeros (on the CPU)."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return torch.zeros(2, dtype=torch.int64)
    v = _int_view(t)
    n = v.numel()
    full = n - n % BLOCK
    out = torch.zeros(2, dtype=torch.int64, device=v.device)
    for i in range(0, full, CHUNK):
        j = min(i + CHUNK, full)
        sums = v[i:j].view(-1, BLOCK).sum(1, dtype=torch.int64)
        w = torch.arange(i // BLOCK, j // BLOCK, device=v.device, dtype=torch.int64) % POSITION_MOD + 1
        out[0] += sums.sum()
        out[1] += (sums * w).sum()
    if full < n:
        tail = v[full:].sum(dtype=torch.int64)
        out[0] += tail
        out[1] += tail * ((full // BLOCK) % POSITION_MOD + 1)
    return out


def _fingerprints(tensors: Sequence[Optional[torch.Tensor]], device: torch.device) -> torch.Tensor:
    """[k, 2] int64 on device. A missing tensor's zeros are made there (a copy from the CPU's pageable memory would
    make the host wait for the GPU's queued work at every read)."""
    if not tensors:
        return torch.zeros(0, 2, dtype=torch.int64, device=device)
    out = []
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.numel():
            f = fingerprint(t)
            out.append(f if f.device == device else f.to(device))
        else:
            out.append(torch.zeros(2, dtype=torch.int64, device=device))
    return torch.stack(out)


def _location(t) -> Optional[tuple]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return str(t.device), t.data_ptr(), tuple(t.shape), t.dtype


class WeightScan:
    """The scan of one ServeGuard (guard.weight_scan()). poll() is called by the guard's poll."""

    def __init__(self, guard, budget_mb: float = DEFAULT_BUDGET_MB):
        self.guard = guard
        self.budget = max(1, int(float(budget_mb) * (1 << 20)))
        self.entries: list = []
        self._slots: set = set()                   # guard layer slots already taken in
        self.cursor = 0
        self.paused = False                        # (no new comparisons while True: for measuring the cost)
        self._pending: dict = {}                   # device -> [(flags, event, [(entry index, step)]), ...]
        self._pass_started = time.monotonic()
        self._warned: set = set()
        self.not_scanned: dict = {}                # name -> why
        self.stats = {"entries": 0, "bytes": 0, "reads": 0, "entries_compared": 0, "bytes_compared": 0,
                      "passes": 0, "last_pass_seconds": None, "changed": 0, "alerts": 0, "references_retaken": 0}

    # --------------------------------------------------------------------------------------------- entries
    def add(self, name: str, get: Callable[[], Sequence[Optional[torch.Tensor]]], kind: str,
            module: Optional[torch.nn.Module] = None) -> Optional[int]:
        """Register tensors to scan. get() returns them as they are now (read again at each turn); their
        fingerprint now is the reference, provisional until the first comparison. kind: "checked", "quantized" or
        "unchecked" (in the scan only). Returns the entry's index, or None when there's nothing to compare."""
        tensors = list(get())
        live = [t for t in tensors if isinstance(t, torch.Tensor) and t.numel()]
        if not live:
            self.not_scanned[name] = "no weight tensors to compare"
            return None
        e = {"name": name, "get": get, "kind": kind, "module": module, "device": live[0].device,
             "bytes": sum(t.numel() * t.element_size() for t in live), "alerted": False, "excluded": None,
             "last_clean": None, "last_clean_step": None, "compared": 0}
        with torch.inference_mode(False), torch.no_grad():
            self._take_reference(e, tensors)
        self.entries.append(e)
        self.stats["entries"] = len(self.entries)
        self.stats["bytes"] += e["bytes"]
        self.not_scanned.pop(name, None)
        return len(self.entries) - 1

    def sync(self) -> None:
        """Take in the guard's checked layers registered since the last call."""
        g = self.guard
        if all(slot in self._slots for slot in range(len(g.layers))):
            return
        unreadable = g._unreadable_weights() if hasattr(g, "_unreadable_weights") else {}
        for slot, lay in enumerate(g.layers):
            if slot in self._slots:
                continue
            self._slots.add(slot)
            if slot in unreadable:
                self.not_scanned[lay["name"]] = f"its weights can't be read back as its own: {unreadable[slot]}"
                continue
            mod, src = lay["module"], lay.get("source")
            if src is not None:          # (read through the layer's current source: a reload replaces it)
                self.add(lay["name"], (lambda m=mod, k=slot: g.layers[k]["source"].live(m)), "quantized", mod)
            else:
                self.add(lay["name"], (lambda m=mod: (getattr(m, "weight", None), _bias(m))), "checked", mod)

    def refresh(self, module: Optional[torch.nn.Module] = None) -> int:
        """After an intended update: every entry (or one module's) takes a new provisional reference from its tensors
        now, and is compared again (an alerted or excluded one too). Returns how many entries were refreshed.
        Comparisons started before are dropped."""
        n = 0
        self._pending.clear()
        with torch.inference_mode(False), torch.no_grad():
            for e in self.entries:
                if module is not None and e["module"] is not module:
                    continue
                self._take_reference(e, list(e["get"]()))
                e["alerted"], e["excluded"], e["last_clean"], e["last_clean_step"] = False, None, None, None
                self.not_scanned.pop(e["name"], None)
                n += 1
        return n

    def _take_reference(self, e: dict, tensors: list) -> None:
        e["ref"] = _fingerprints(tensors, e["device"])
        e["at"] = [_location(t) for t in tensors]
        e["ids"] = [id(t) for t in tensors]
        e["provisional"] = True

    def _shared(self, e: dict, tensors: list) -> tuple:
        """(why, the other entry) when another entry's tensor (a different tensor: one tensor two layers hold is fine)
        is in the same memory, else (None, None): vLLM's prefetch offloader points layers N, N + its capacity, ... at
        one GPU buffer and copies each layer's weights in before it runs, so what is there belongs to whichever ran
        last. Both entries are then left out."""
        mine = {}
        for t in tensors:
            loc = _location(t)
            if loc:
                mine[(loc[0], loc[1])] = id(t)
        if not mine:
            return None, None
        for other in self.entries:
            if other is e:
                continue
            for loc, oid in zip(other.get("at") or (), other.get("ids") or ()):
                if loc and mine.get((loc[0], loc[1]), oid) != oid:
                    return f"shares its memory with {other['name']}'s tensors (vLLM's CPU offloading)", other
        return None, None

    # ------------------------------------------------------------------------------------------------ reads
    def poll(self, wait: bool = False) -> list:
        """Report the comparisons that have come back, and start the next ones (up to the byte budget)."""
        self.sync()
        self.stats["reads"] += 1
        events = self._collect(wait)
        if self.entries and not self.paused:
            events += self._start(wait)
        g = self.guard
        if getattr(g, "metrics", None) is not None:
            try:
                g.metrics.observe_weights(self, events)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
        return events

    def _start(self, wait: bool) -> list:
        events: list = []
        started: dict = {}
        spent, tried = 0, 0
        n = len(self.entries)
        step = getattr(self.guard, "_marked_step", None)
        with torch.inference_mode(False), torch.no_grad():
            while tried < n and (spent < self.budget or not started):
                i = self.cursor
                self.cursor = (self.cursor + 1) % n
                tried += 1
                if self.cursor == 0:
                    self._pass_done()
                e = self.entries[i]
                if e["alerted"] or e["excluded"]:
                    continue
                tensors = list(e["get"]())
                why, other = self._shared(e, tensors)
                if why:
                    self._exclude(e, why)
                    if other is not None and not other["excluded"]:     # both: whichever ran last owns the memory
                        self._exclude(other, f"shares its memory with {e['name']}'s tensors (vLLM's CPU offloading)")
                    continue
                if [_location(t) for t in tensors] != e["at"]:
                    if e["provisional"]:
                        # never compared: the reference is taken again here (a reload processed the weights in
                        # temporary tensors and copied them into the originals)
                        self._take_reference(e, tensors)
                        self.stats["references_retaken"] += 1
                        continue
                    self._exclude(e, "its tensors moved or changed shape after they were compared (call refresh() "
                                     "after an intended reload)")
                    continue
                now = _fingerprints(tensors, e["device"])
                started.setdefault(e["device"], []).append(((i, step), torch.ne(now, e["ref"]).any()))
                spent += e["bytes"]
        for dev, items in started.items():
            flags = torch.stack([f for _, f in items]).to(torch.int32)
            idx = [k for k, _ in items]
            if self._async(dev) and not wait:
                buf, ev = self._copy(flags, dev)
                self._pending.setdefault(dev, []).append((buf, ev, idx))
            else:
                events += self._judge(flags.cpu(), idx)
            self.stats["entries_compared"] += len(idx)
            self.stats["bytes_compared"] += sum(self.entries[i]["bytes"] for i, _ in idx)
        return events

    @staticmethod
    def _async(dev: torch.device) -> bool:
        return dev.type == "cuda"

    @staticmethod
    def _copy(flags: torch.Tensor, dev: torch.device) -> tuple:
        """A pinned host copy of the flags, started now, and the event that says it has arrived."""
        buf = torch.empty(flags.shape, dtype=torch.int32, pin_memory=True)
        buf.copy_(flags, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(dev))
        return buf, ev

    def _exclude(self, e: dict, why: str) -> None:
        e["excluded"] = why
        self.not_scanned[e["name"]] = why
        if e["name"] not in self._warned:
            self._warned.add(e["name"])
            log.warning("sdcguard: the weight scan no longer compares %s: %s", e["name"], why)

    def _pass_done(self) -> None:
        now = time.monotonic()
        self.stats["passes"] += 1
        self.stats["last_pass_seconds"] = round(now - self._pass_started, 3)
        self._pass_started = now

    def _collect(self, wait: bool) -> list:
        """The comparisons started at earlier reads whose results have arrived (all of them, waiting, with wait)."""
        events = []
        for dev in list(self._pending):
            left = []
            for buf, ev, idx in self._pending[dev]:
                if wait:
                    ev.synchronize()
                if ev.query():
                    events += self._judge(buf, idx)
                else:
                    left.append((buf, ev, idx))
            if left:
                self._pending[dev] = left
            else:
                del self._pending[dev]
        return events

    def _judge(self, flags: torch.Tensor, idx: list) -> list:
        events = []
        now = time.monotonic()
        for f, (i, step) in zip(flags.tolist(), idx):
            e = self.entries[i]
            if e["excluded"]:                    # left out after this comparison started (its memory is shared)
                continue
            e["compared"] += 1
            e["provisional"] = False
            if not f:
                e["last_clean"], e["last_clean_step"] = now, step
                continue
            if e["alerted"]:
                continue
            e["alerted"] = True
            self.stats["changed"] += 1
            events.append(self._alert(e, now, step))
        return events

    def _alert(self, e: dict, now: float, step: Optional[int]) -> SDCEvent:
        g = self.guard
        ident = getattr(g, "_identity", None)
        key = str(e["device"])
        if isinstance(ident, dict):
            if key not in ident:
                ident[key] = device_identity(e["device"])
            identity = dict(ident[key])
        else:
            identity = device_identity(e["device"])
        clean = e["last_clean_step"]
        window = None
        if step is not None:
            window = [clean + 1 if clean is not None and clean < step else step, step]
        detail = {"where": e["name"], "finding": "weights_changed", "kind": e["kind"],
                  "weights_changed_since_load": True,
                  "recheck": {"verdict": "weights_changed",
                              "why": "the layer's tensors no longer match their reference fingerprint (taken at "
                                     "load, or at the last refresh)"},
                  "read_window_steps": window,
                  "last_clean_read_steps": [clean, clean] if clean is not None else None,
                  "seconds_since_last_clean_comparison": (round(now - e["last_clean"], 3)
                                                          if e["last_clean"] is not None else None),
                  "last_pass_seconds": self.stats["last_pass_seconds"],
                  "scan": {"budget_bytes": self.budget, "entries": len(self.entries)},
                  "reading": "a weight bit changed after load: memory corruption, or an update the guard wasn't "
                             "told about (call refresh() after an intended one). Every request this layer served "
                             "since its last clean comparison may have used the changed weights."}
        ev = SDCEvent(detector="serve_weights", step=None, detail=detail, identity=identity)
        self.stats["alerts"] += 1
        if getattr(g, "attribute", None) is not None:
            try:
                g.attribute(ev)
            except Exception:  # noqa: BLE001  (naming requests is extra; the alert itself goes out)
                g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
        g.log.add(ev)
        if getattr(g, "on_alert", None):
            try:
                g.on_alert(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

    def coverage(self) -> dict:
        by_kind: dict = {}
        for e in self.entries:
            by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        return {"entries": len(self.entries), "by_kind": by_kind, "bytes": self.stats["bytes"],
                "budget_bytes_per_read": self.budget, "passes": self.stats["passes"],
                "last_pass_seconds": self.stats["last_pass_seconds"],
                "never_compared": sorted(e["name"] for e in self.entries if not e["compared"] and not e["excluded"]),
                "changed": sorted(e["name"] for e in self.entries if e["alerted"]),
                "not_scanned": dict(self.not_scanned)}


def _bias(m) -> Optional[torch.Tensor]:
    b = getattr(m, "bias", None)
    return b if isinstance(b, torch.Tensor) else None
