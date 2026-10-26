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

The fingerprint of a tensor's raw bits (viewed as 16-bit, 32-bit or 8-bit integers by its element size, signed
for 16 and 32 bits; 8-byte elements as pairs of 32-bit ones) is two int64 sums, wrapping modulo 2^64: of every value
x_i, and of every value times a key for its position, x_i * k(i), where k(i) is a fixed 64-bit mix of the value's
index (MurmurHash3's 64-bit finalizer of i XOR a constant: a bijection, so no two positions share a key). What each
sees:
- one changed value, whatever the change: always (the first sum moves by the change, less than 2^32 and not 0);
- two values swapped: the second sum moves by (b - a) * (k(i) - k(j)), which is 0 modulo 2^64 only when
  k(i) - k(j) is a multiple of 2^49 for 16-bit values (2^33 for 32-bit, 2^57 for 8-bit), since b - a has at most 15
  (31, 7) factors of 2: about one pair in 2^49 (2^33, 2^57) for keys that behave like random numbers;
- any other pattern of changes: unseen only when both sums cancel; for changes that don't depend on the keys, the
  second sum behaves like a random 64-bit number, so heuristically about one pattern in 2^49 (16-bit values).
The keys are fixed and public: this is for faults (bit flips, stale or misplaced copies), not for someone who
changes weights on purpose and knows the keys. On a CPU or without Triton, it is computed in chunks of 1M values
(about 40 MB of temporary memory, plus a contiguous copy of a tensor that isn't contiguous), and on a CUDA device its
result is read back without waiting for the GPU: the copy is started at one read and taken at a later one (or at
once when the guard is asked to wait).

Cost: each read compares layers until it has compared about SDCGUARD_WEIGHT_SCAN_MB (64 by default) of their tensors
(the last layer of a read can take it past). A pass over a model takes about (its bytes / that budget) reads. On a GPU
each tensor's fingerprint is one Triton kernel launch (the same two sums, compared with the PyTorch version on each
device and element size before its first use; where they differ, or without Triton, the PyTorch version is used).

An intended weight update (an RL weight sync, a reload) looks the same as corruption: ServeGuard.refresh() and
refresh_layer() make the references provisional again, and so does refresh() here.

model_digest() uses the same fingerprint once, on every parameter of a loaded model, for its identity (sdcguard.stack's
weights_digest, which `sdcguard ready --qualified` holds a server to), whether or not the scan is on.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from typing import Callable, Optional, Sequence

import torch

from .events import SDCEvent, device_identity

log = logging.getLogger("sdcguard")

BLOCK = 4096                            # values per program of the kernel
CHUNK = 1 << 20                         # values per step of the PyTorch version (5 int64 temporaries: 40 MB)
KEY_SEED = 0x2545F4914F6CDD1D           # the position keys: k(i) = mix(i XOR KEY_SEED)
MIX1 = 0xFF51AFD7ED558CCD               # MurmurHash3's 64-bit finalizer (fmix64)
MIX2 = 0xC4CEB9FE1A85EC53
LOW31 = (1 << 31) - 1                   # (x >> 33) & LOW31: a logical shift of an int64


def _s64(c: int) -> int:
    """A 64-bit constant as the int64 with the same bits."""
    return c - (1 << 64) if c >= (1 << 63) else c


SEED64, M1, M2 = _s64(KEY_SEED), _s64(MIX1), _s64(MIX2)
DEFAULT_BUDGET_MB = 64.0
DEFAULT_MAX_MB = 1024.0                 # the most a read compares to meet a verification-age target

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}                       # (device, element size) -> (the kernel is used, why not)
_LOCK = threading.Lock()


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


def keys(start: int, n: int, device) -> torch.Tensor:
    """The position keys k(start) .. k(start + n - 1) as int64 (wrapping arithmetic: the same bits as uint64)."""
    h = torch.arange(start, start + n, dtype=torch.int64, device=device) ^ SEED64
    h = (h ^ ((h >> 33) & LOW31)) * M1
    h = (h ^ ((h >> 33) & LOW31)) * M2
    return h ^ ((h >> 33) & LOW31)


def fingerprint(t: Optional[torch.Tensor]) -> torch.Tensor:
    """[2] int64 on the tensor's device: the sum of its raw bit values x_i, and the sum of x_i * k(i) (both modulo
    2^64; see the module's docstring). A missing or empty tensor gives zeros (on the CPU)."""
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return torch.zeros(2, dtype=torch.int64)
    v = _int_view(t)
    n = v.numel()
    out = torch.zeros(2, dtype=torch.int64, device=v.device)
    for i in range(0, n, CHUNK):
        j = min(i + CHUNK, n)
        x = v[i:j].to(torch.int64)
        out[0] += x.sum()
        out[1] += (x * keys(i, j - i, v.device)).sum()
    return out


# ------------------------------------------------------------------------------------- the fingerprint kernel
# One launch per tensor: a program per BLOCK values reads their raw bits as integers, makes each one's position key,
# and adds the sum of the values and the sum of each value times its key into the tensor's row with int64 atomics.
# Integer addition wraps the same way in any order, so the two sums equal fingerprint()'s exactly; the PyTorch version
# takes several launches per CHUNK and stays as the reference and the fallback.
if HAVE_TRITON:
    @triton.jit
    def _fingerprint_kernel(v, out, n, BLOCK_: tl.constexpr, SEED: tl.constexpr, MUL1: tl.constexpr,
                            MUL2: tl.constexpr, LOW: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK_ + tl.arange(0, BLOCK_).to(tl.int64)
        x = tl.load(v + offs, mask=offs < n, other=0).to(tl.int64)
        h = offs ^ SEED
        h = (h ^ ((h >> 33) & LOW)) * MUL1
        h = (h ^ ((h >> 33) & LOW)) * MUL2
        h = h ^ ((h >> 33) & LOW)
        tl.atomic_add(out, tl.sum(x, 0))
        tl.atomic_add(out + 1, tl.sum(x * h, 0))


def _launch(t: torch.Tensor, row: torch.Tensor) -> None:
    """Add t's fingerprint into row ([2] int64, zeros, on t's device) with the kernel, on t's device and its current
    stream (Triton launches on the current device; the zeros before it and the reads after it are on t's)."""
    from ._group import _on
    v = _int_view(t)
    n = v.numel()
    with _on(v.device):
        _fingerprint_kernel[(triton.cdiv(n, BLOCK),)](v, row, n, BLOCK_=BLOCK, SEED=SEED64, MUL1=M1, MUL2=M2,
                                                      LOW=LOW31, num_warps=4)


def _key(device: torch.device, element_size: int) -> tuple:
    from ._group import _dev_key
    return _dev_key(device), int(element_size)          # "cuda" and "cuda:0" are one device


def _why_not(device: torch.device) -> Optional[str]:
    """Why the kernel can't be used on this device (None: it can be tried)."""
    import os
    from ._group import arch_ok
    if not HAVE_TRITON:
        return "Triton isn't installed"
    if os.environ.get("SDCGUARD_FAST", "1") == "0" or os.environ.get("SDCGUARD_WEIGHT_SCAN_FAST", "1") == "0":
        return "turned off (SDCGUARD_FAST=0 or SDCGUARD_WEIGHT_SCAN_FAST=0)"
    if device.type == "cuda":
        return None if arch_ok(device) else "a GPU generation the Triton kernels aren't validated on " \
                                             "(SDCGUARD_TRITON_NEW_ARCH=1 opts in)"
    if device.type == "cpu" and os.environ.get("TRITON_INTERPRET") == "1" and \
            os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1":
        return None
    return f"no Triton kernel on {device.type}"


def kernel_usable(device: torch.device) -> bool:
    return _why_not(device) is None


def _disable(key: tuple, e: Exception, when: str) -> None:
    with _LOCK:
        STATUS[key] = (False, f"{when}: {type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    log.warning("sdcguard: the weight scan's fingerprint kernel isn't used on %s (element size %s) from now on: %s",
                key[0], key[1], STATUS[key][1])


def verify_kernel(device: torch.device, element_size: int) -> tuple:
    """The kernel against fingerprint() before its first use on a device and element size: tensors with and
    without a partial last block, a single value, a block-aligned length, extreme bit patterns, and a view that
    isn't contiguous. (ok, why)."""
    key = _key(device, element_size)
    with _LOCK:
        if key in STATUS:
            return STATUS[key]
    why = _why_not(device)
    dt = {1: torch.uint8, 2: torch.bfloat16, 4: torch.float32, 8: torch.float64}.get(int(element_size))
    if why is None and dt is None:
        why = f"element size {element_size}"
    if why is not None:
        with _LOCK:
            STATUS[key] = (False, why)
        return STATUS[key]
    try:
        gen = torch.Generator().manual_seed(0)
        cases = []
        for n in (1, 7, BLOCK, BLOCK + 1, 3 * BLOCK - 5, 70001):
            if dt == torch.uint8:
                cases.append(torch.randint(0, 256, (n,), generator=gen, dtype=torch.uint8))
            else:
                cases.append((torch.randn(n, generator=gen) * 3).to(dt))
        bits = torch.full((BLOCK + 3,), -1, dtype={1: torch.uint8, 2: torch.int16, 4: torch.int32,
                                                   8: torch.int64}[int(element_size)])
        cases.append(bits.view(dt))                                    # every bit set (-1, or 255 unsigned)
        cases.append(torch.randn(96, 130, generator=gen).to(dt).t())   # not contiguous
        for c in cases:
            c = c.to(device)
            row = torch.zeros(2, dtype=torch.int64, device=device)
            _launch(c, row)
            want = fingerprint(c)
            if not torch.equal(row.cpu(), want.cpu()):
                raise AssertionError(f"{tuple(c.shape)}: kernel {row.tolist()} != reference {want.tolist()}")
        with _LOCK:
            STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        _disable(key, e, "its first-use comparison failed")
    return STATUS[key]


def _fingerprints(tensors: Sequence[Optional[torch.Tensor]], device: torch.device) -> torch.Tensor:
    """[k, 2] int64 on device: one launch of the kernel per tensor where it passed its first-use comparison, else
    fingerprint(). A missing tensor's row stays zeros, made on the device (a copy from the CPU's pageable memory would
    make the host wait for the GPU's queued work at every read). A launch that raises (a variant the first-use
    comparison didn't compile: a tensor of 2^31 values or more, an unaligned one) turns the kernel off on that device
    and element size, and that tensor's row is the PyTorch version's."""
    out = torch.zeros(len(tensors), 2, dtype=torch.int64, device=device)
    for i, t in enumerate(tensors):
        if not (isinstance(t, torch.Tensor) and t.numel()):
            continue
        if t.device == device and verify_kernel(device, t.element_size())[0]:
            try:
                _launch(t, out[i])
                continue
            except Exception as e:  # noqa: BLE001  (never break serving: the PyTorch version from now on)
                _disable(_key(device, t.element_size()), e, "a launch failed")
                out[i].zero_()
        f = fingerprint(t)
        out[i] = f if f.device == device else f.to(device)
    return out


def param_tensors(name: str, p):
    """A parameter's tensors, as vLLM's own weight digests take them: a shared-weight parameter's partitions, a
    tensor subclass's inner tensors (TorchAO), else the parameter itself."""
    parts = getattr(p, "partitions", None)
    if isinstance(parts, dict) and parts:
        for k, v in parts.items():
            yield from param_tensors(f"{name}.{k}", v)
        return
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass
        wrapped = is_traceable_wrapper_subclass(p)
    except Exception:  # noqa: BLE001
        wrapped = False
    if wrapped:
        for a in p.__tensor_flatten__()[0]:
            yield from param_tensors(f"{name}.{a}", getattr(p, a))
        return
    yield name, p


def model_digest(model) -> Optional[str]:
    """One identity of a loaded model's weights: every parameter (a tensor two modules hold, once) fingerprinted as
    the scan does (the Triton kernel on its device where it passed its first-use comparison, else the PyTorch
    version: the same two sums either way), then the names, shapes, dtypes and fingerprints hashed together. One
    pass over the weights, and one wait for each device's results: for model load, never per request. One changed
    value is always another digest (see the module's docstring for swaps and other patterns); like the scan, this is
    for faults and mix-ups, not for someone who changes weights on purpose and knows the keys. None when the model
    has no parameter; raises when one can't be read (a caller leaves the field out)."""
    named, seen = [], set()
    for pname, p in model.named_parameters():
        for tname, t in param_tensors(pname, p):
            if isinstance(t, torch.Tensor) and id(t) not in seen:
                seen.add(id(t))
                named.append((tname, t))
    if not named:
        return None
    named.sort(key=lambda x: x[0])
    by_device: dict = {}
    for i, (_, t) in enumerate(named):
        by_device.setdefault(t.device, []).append(i)
    fps: list = [None] * len(named)
    with torch.no_grad():
        for dev, idx in by_device.items():
            rows = _fingerprints([named[i][1] for i in idx], dev).cpu().tolist()
            for i, f in zip(idx, rows):
                fps[i] = f
    h = hashlib.sha256()
    for (n, t), f in zip(named, fps):
        h.update(json.dumps([n, list(t.shape), str(t.dtype).replace("torch.", ""), f]).encode())
    return h.hexdigest()[:16]


def _location(t) -> Optional[tuple]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return str(t.device), t.data_ptr(), tuple(t.shape), t.dtype


class WeightScan:
    """The scan of one ServeGuard (guard.weight_scan()). poll() is called by the guard's poll."""

    def __init__(self, guard, budget_mb: float = DEFAULT_BUDGET_MB, target_age: Optional[float] = None,
                 max_mb: float = DEFAULT_MAX_MB, max_gap: float = 4.0):
        self.guard = guard
        self.budget = max(1, int(float(budget_mb) * (1 << 20)))
        # a verification-age target (seconds): each read compares enough bytes that a pass over the model takes at
        # most 2/3 of it at the measured time between reads, between the budget and max_mb (see target())
        self.target_age = float(target_age) if target_age else None
        self.max_budget = max(self.budget, int(float(max_mb) * (1 << 20)))
        self._last_read: Optional[float] = None
        self._interval: Optional[float] = None     # seconds between reads (moving average)
        self.max_gap = float(max_gap)              # a longer gap between reads (an idle server) counts as this
        self._age_warned = False
        self.entries: list = []
        self._slots: set = set()                   # guard layer slots already taken in
        self.cursor = 0
        self.paused = False                        # (no new comparisons while True: for measuring the cost)
        self._pending: dict = {}                   # device -> [(flags, event, [(entry index, step)]), ...]
        self._pass_started = time.monotonic()
        self._warned: set = set()
        self.not_scanned: dict = {}                # name -> why
        self.stats = {"entries": 0, "bytes": 0, "reads": 0, "entries_compared": 0, "bytes_compared": 0,
                      "passes": 0, "last_pass_seconds": None, "changed": 0, "alerts": 0, "references_retaken": 0,
                      "budget_bytes_now": self.budget, "target_met": None}

    # --------------------------------------------------------------------------------------------- entries
    def add(self, name: str, get: Callable[[], Sequence[Optional[torch.Tensor]]], kind: str,
            module: Optional[torch.nn.Module] = None) -> Optional[int]:
        """Register tensors to scan. get() returns them as they are now (read again at each turn); their
        fingerprint now is the reference, provisional until the first comparison. kind: "checked", "quantized",
        "unchecked" (a linear layer in the scan only) or "other" (the model's other parameters: embeddings, the LM
        head, norm weights, MoE experts' weights). Returns the entry's index, or None when there's nothing to compare."""
        tensors = list(get())
        live = [t for t in tensors if isinstance(t, torch.Tensor) and t.numel()]
        if not live:
            self.not_scanned[name] = "no weight tensors to compare"
            return None
        e = {"name": name, "get": get, "kind": kind, "module": module, "device": live[0].device,
             "bytes": sum(t.numel() * t.element_size() for t in live), "alerted": False, "excluded": None,
             "last_clean": None, "last_clean_step": None, "compared": 0, "since": time.monotonic()}
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

    def refresh(self, module: Optional[torch.nn.Module] = None, kind: Optional[str] = None,
                name: Optional[str] = None) -> int:
        """After an intended update: every entry (or one module's, one kind's, or the one with that name) takes a new
        provisional reference from its tensors now, and is compared again (an alerted or excluded one too). Returns
        how many entries were refreshed. Comparisons started before are dropped."""
        n = 0
        self._pending.clear()
        now = time.monotonic()
        with torch.inference_mode(False), torch.no_grad():
            for e in self.entries:
                if (module is not None and e["module"] is not module) or (kind is not None and e["kind"] != kind) or \
                        (name is not None and e["name"] != name):
                    continue
                self._take_reference(e, list(e["get"]()))
                e["alerted"], e["excluded"], e["last_clean"], e["last_clean_step"] = False, None, None, None
                e["since"] = now
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
        now = time.monotonic()
        if self._last_read is not None:
            dt = min(now - self._last_read, self.max_gap)
            self._interval = dt if self._interval is None else 0.8 * self._interval + 0.2 * dt
        self._last_read = now
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

    def target(self) -> int:
        """The bytes this read compares: the budget, or with a verification-age target, enough that a pass over the
        entries still compared takes at most 2/3 of the target at the measured time between reads (gaps longer than
        max_gap, an idle server, count as max_gap), up to max_mb; stats["target_met"] is False when that isn't
        enough, and a warning is logged the first time)."""
        if not self.target_age or self._interval is None or self._interval <= 0:
            return self.budget
        live = sum(e["bytes"] for e in self.entries if not e["alerted"] and not e["excluded"])
        reads = max(1.0, (self.target_age * 2 / 3) / self._interval)
        want = int(live / reads) + 1
        met = want <= self.max_budget
        if not met and not self._age_warned:
            self._age_warned = True
            log.warning("sdcguard: the weight scan can't compare every weight within %gs at %.2fs between reads: "
                        "it would need %.0f MB a read, above SDCGUARD_WEIGHT_SCAN_MAX_MB (%.0f); a pass takes about "
                        "%.0fs", self.target_age, self._interval, want / (1 << 20), self.max_budget / (1 << 20),
                        live / self.max_budget * self._interval)
        self.stats["target_met"] = met
        return max(self.budget, min(want, self.max_budget))

    def oldest_age(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds since the least recently verified entry (still compared) was last found clean, or registered if it
        hasn't been yet; None with nothing to compare."""
        now = time.monotonic() if now is None else now
        ages = [now - (e["last_clean"] if e["last_clean"] is not None else e["since"])
                for e in self.entries if not e["alerted"] and not e["excluded"]]
        return max(ages) if ages else None

    def _start(self, wait: bool) -> list:
        events: list = []
        started: dict = {}
        spent, tried = 0, 0
        n = len(self.entries)
        budget = self.target()
        self.stats["budget_bytes_now"] = budget
        step = getattr(self.guard, "_marked_step", None)
        with torch.inference_mode(False), torch.no_grad():
            while tried < n and (spent < budget or not started):
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
                  "scan": {"budget_bytes": self.stats["budget_bytes_now"], "entries": len(self.entries),
                           "target_age_seconds": self.target_age},
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
        from ._group import _dev_key
        devs = {_dev_key(e["device"]) for e in self.entries}
        age = self.oldest_age()
        return {"entries": len(self.entries), "by_kind": by_kind, "bytes": self.stats["bytes"],
                "budget_bytes_per_read": self.budget, "budget_bytes_now": self.stats["budget_bytes_now"],
                # target_met: the schedule (each read's bytes) can meet the target at the measured time between
                # reads, a prediction; oldest_within_target: the oldest verification age now is within it, measured
                "target_age_seconds": self.target_age, "target_met": self.stats["target_met"],
                "oldest_verification_age_seconds": round(age, 3) if age is not None else None,
                "oldest_within_target": (age <= self.target_age) if self.target_age and age is not None else None,
                "passes": self.stats["passes"],
                "last_pass_seconds": self.stats["last_pass_seconds"],
                "never_compared": sorted(e["name"] for e in self.entries if not e["compared"] and not e["excluded"]),
                "changed": sorted(e["name"] for e in self.entries if e["alerted"]),
                "not_scanned": dict(self.not_scanned),
                # the fingerprint's implementation per device and element size: the Triton kernel, or PyTorch and why
                "kernel": {f"{d} {es}-byte": ("Triton" if ok else f"PyTorch ({why})")
                           for (d, es), (ok, why) in sorted(STATUS.items()) if d in devs}}


def _bias(m) -> Optional[torch.Tensor]:
    b = getattr(m, "bias", None)
    return b if isinstance(b, torch.Tensor) else None
