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

Repair (repair=True, SDCGUARD_WEIGHT_REPAIR=1 in the plugin): a host copy of each entry's tensors is kept with its
reference, and an entry found changed in at most REPAIR_MAX_VALUES values is written back in place at the end of the
read that found it, then verified against its reference (_repair_pending). It can't tell an update made without
refresh() that changes that few values from a fault, and reverts it too; larger changes are left alone and reported.

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
# Repair writes an entry back only when at most this many of its values differ from the copy kept with its
# reference: a fault in memory changes a few values, an update the guard wasn't told about (an RL weight sync, a
# reload without refresh()) changes nearly all of them, and writing the old weights over one would be a silent error
# of sdcguard's own making.
REPAIR_MAX_VALUES = 64
HOST_SHARE = 0.5                        # repair's copies take at most this share of the free host memory per local GPU


def _host_room() -> Optional[int]:
    """Bytes of host memory still free for this process: MemAvailable, or the cgroup's limit less its use when that
    is lower (a container sees the host's MemAvailable); None when neither can be read."""
    room = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    room = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError, IndexError):
        pass
    for lim_path, use_path in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                               ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                                "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            with open(lim_path) as f:
                lim = f.read().strip()
            if lim == "max":
                continue
            with open(use_path) as f:
                left = int(lim) - int(f.read().strip())
            room = left if room is None else min(room, left)
            break
        except (OSError, ValueError):
            continue
    return room


def _local_gpus() -> int:
    """The GPUs on this host that may each run a worker taking its own copies (LOCAL_WORLD_SIZE when set)."""
    import os
    try:
        n = int(os.environ.get("LOCAL_WORLD_SIZE") or 0)
    except ValueError:
        n = 0
    if n <= 0:
        try:
            n = torch.cuda.device_count() if torch.cuda.is_available() else 1
        except Exception:  # noqa: BLE001
            n = 1
    return max(1, n)

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


def before_repair(restored: Optional[dict], step) -> bool:
    """Whether an output computed at `step` (None: not known) may predate a weight repair recorded as `restored`
    ({"step": the last step queued before the write-back}): such an output may have used the changed weights."""
    if restored is None:
        return False
    at = restored.get("step")
    return step is None or at is None or step <= at


def repair_why(restored: dict, step) -> str:
    """Why a recheck with the restored weights can't judge an output from before the repair."""
    return (f"sdcguard's weight repair wrote this layer's load-time weights back after step {restored.get('step')}, "
            f"and this output (step {step}) may have been computed before that, with the changed weights: a recheck "
            "with the weights as they are now can't say whether it was right")


def _location(t) -> Optional[tuple]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return str(t.device), t.data_ptr(), tuple(t.shape), t.dtype


class WeightScan:
    """The scan of one ServeGuard (guard.weight_scan()). poll() is called by the guard's poll."""

    def __init__(self, guard, budget_mb: float = DEFAULT_BUDGET_MB, target_age: Optional[float] = None,
                 max_mb: float = DEFAULT_MAX_MB, max_gap: float = 4.0, repair: bool = False,
                 repair_max_values: int = REPAIR_MAX_VALUES, host_allowance: Optional[int] = None):
        self.guard = guard
        # repair: keep a host-RAM copy of every entry's tensors as they were when its reference was taken, and when an
        # entry is found changed by at most repair_max_values values, copy it back in place at the end of that read
        # and compare again (_repair_pending). The copies take at most host_allowance bytes (default: HOST_SHARE of
        # the host memory free at the first copy, divided among the local GPUs); repair turns itself off past it.
        self.repair = bool(repair)
        self.repair_max_values = max(0, int(repair_max_values))
        self.repair_off: Optional[str] = None      # why repair turned itself off
        self._host_allowance = host_allowance
        self._to_repair: list = []                 # [(entry index, its generation, alert window, last clean)]
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
                      "budget_bytes_now": self.budget, "target_met": None, "repaired": 0, "repair_failed": 0,
                      "host_copy_bytes": 0}

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
             "last_clean": None, "last_clean_step": None, "compared": 0, "since": time.monotonic(),
             "gen": -1, "repairs": 0}
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
        how many entries were refreshed. Comparisons started before, and repairs not yet made, are dropped."""
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
        # (an intended update: the copy kept now is of the new weights, and the old change is no longer one)
        self._to_repair = [r for r in self._to_repair if r[1] == self.entries[r[0]]["gen"]]
        return n

    def _take_reference(self, e: dict, tensors: list) -> None:
        e["ref"] = _fingerprints(tensors, e["device"])
        e["at"] = [_location(t) for t in tensors]
        e["ids"] = [id(t) for t in tensors]
        e["provisional"] = True
        e["ref_step"] = getattr(self.guard, "_marked_step", None)    # (the steps after it may see a change)
        e["gen"] += 1                            # comparisons and repairs started before are about other tensors
        if self.repair:                          # the clean copy: the same values the reference fingerprints
            self._keep_copy(e, tensors)

    @staticmethod
    def _nbytes(ts) -> int:
        return sum(c.numel() * c.element_size() for c in ts or () if c is not None)

    def _keep_copy(self, e: dict, tensors: list) -> None:
        """e["clean"]: a host copy of each tensor (a copy already kept of the same shape and dtype is written over,
        not allocated again). Turns repair off for every entry when the copies would pass the host allowance."""
        old = e.get("clean") or []
        reuse = [k < len(old) and old[k] is not None and isinstance(t, torch.Tensor) and t.numel() > 0 and
                 old[k].shape == t.shape and old[k].dtype == t.dtype for k, t in enumerate(tensors)]
        need = sum(t.numel() * t.element_size() for t, r in zip(tensors, reuse)
                   if isinstance(t, torch.Tensor) and t.numel() and not r)
        freed = self._nbytes(c for k, c in enumerate(old) if not (k < len(reuse) and reuse[k]))
        if self._host_allowance is None:
            room = _host_room()
            if room is None:
                self._host_allowance = 0
                self._repair_disable("the free host memory couldn't be read (/proc/meminfo, the cgroup's limit), so "
                                     "the host copies can't be sized safely")
                return
            self._host_allowance = int(HOST_SHARE * max(room, 0) / _local_gpus())
        after = self.stats["host_copy_bytes"] - freed + need
        if after > self._host_allowance:
            self._repair_disable(f"the host copies would take {after / (1 << 30):.2f} GiB, more than this worker's "
                                 f"allowance of {self._host_allowance / (1 << 30):.2f} GiB ({HOST_SHARE:g} of the free "
                                 f"host memory, divided among {_local_gpus()} local GPUs)")
            return
        clean = []
        for k, (t, r) in enumerate(zip(tensors, reuse)):
            if not (isinstance(t, torch.Tensor) and t.numel()):
                clean.append(None)
            elif r:
                old[k].copy_(t.detach())
                clean.append(old[k])
            else:
                clean.append(t.detach().to("cpu", copy=True))
        e["clean"] = clean
        self.stats["host_copy_bytes"] = after

    def _repair_disable(self, why: str) -> None:
        self.repair, self.repair_off = False, why
        for e in self.entries:
            e["clean"] = None
        self.stats["host_copy_bytes"] = 0
        self._to_repair = []
        log.warning("sdcguard: weight repair is off: %s; the scan still compares and alerts", why)

    def _drop_copy(self, e: dict) -> None:
        if e.get("clean"):
            self.stats["host_copy_bytes"] -= self._nbytes(e["clean"])
            e["clean"] = None

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
        if self._to_repair:                # (after both: a change found by this read is repaired by this read)
            self._repair_pending()
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
                started.setdefault(e["device"], []).append(((i, step, e["gen"]), torch.ne(now, e["ref"]).any()))
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
            self.stats["bytes_compared"] += sum(self.entries[i]["bytes"] for i, _, _ in idx)
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
        self._drop_copy(e)                   # (never written back: its memory isn't its own, or moved)
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
        for f, (i, step, gen) in zip(flags.tolist(), idx):
            e = self.entries[i]
            if e["excluded"]:                    # left out after this comparison started (its memory is shared)
                continue
            if gen != e["gen"]:                  # its reference was taken again (a refresh, a repair) since it started
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
            ev = self._alert(e, now, step)
            events.append(ev)
            if self.repair and e.get("clean"):
                self._to_repair.append((i, e["gen"], ev.detail.get("read_window_steps"),
                                        ev.detail.get("last_clean_read_steps")))
        return events

    # ----------------------------------------------------------------------------------------------- repair
    def _repair_pending(self) -> None:
        """Each entry found changed by this read or an earlier one (with repair on): when at most repair_max_values of
        its values differ from the copy kept with its reference, the copy is written back in place (the same memory,
        so CUDA graphs that read it stay valid; after the GPU has finished the work queued so far), then fingerprinted
        and compared with its reference at once. Restored: it is compared again from then on (a change that comes back
        alerts again). Not restored (more values changed than a fault would change, its memory isn't its own, it
        moved, or it still doesn't match): it stays changed, and isn't tried again until a refresh. Either way a
        record goes to the alert log and on_alert, naming the requests served between the layer's last clean
        comparison and the repair; it isn't an alert, and the alert that found the change stands."""
        todo, self._to_repair = self._to_repair, []
        g = self.guard
        for i, gen, window, clean in todo:
            e = self.entries[i]
            if gen != e["gen"] or not e["alerted"]:      # refreshed since, or already restored
                continue
            at = getattr(g, "_marked_step", None)       # the last step queued: it, and the ones before, ran first
            counts = None
            try:
                with torch.inference_mode(False), torch.no_grad():
                    ok, why, counts = self._restore(e, at)
            except Exception as ex:  # noqa: BLE001  (a failed repair leaves the alert as it was)
                ok, why = False, f"the restore failed: {type(ex).__name__}: {ex}"[:300]
            if ok:
                e["alerted"] = False
                e["repairs"] += 1
                e["gen"] += 1                            # (comparisons in flight saw the changed weights)
                e["last_clean"], e["last_clean_step"] = time.monotonic(), at
                e["compared_at_repair"] = e["compared"]
                self.stats["repaired"] += 1
            else:
                self.stats["repair_failed"] += 1
            self._repair_record(e, ok, why, at, window, clean, counts)

    def _written(self, e: dict, at) -> None:
        """Tell the guard the entry's tensors were written in place after step `at` (whether or not the write then
        verifies), so its checks judge outputs from before it as computed with other weights."""
        g = self.guard
        if e["module"] is not None and hasattr(g, "_weights_restored"):
            try:
                g._weights_restored(e["module"], at)
            except Exception:  # noqa: BLE001  (diagnosis is best effort)
                pass

    def _count_changed(self, pairs: list) -> tuple:
        """(values that differ from the copy, stopped early): counted chunk by chunk on the host (CHUNK values at a
        time, so a large entry needs no full host copy), stopping once past repair_max_values. A value is an element
        of the tensor's raw bits (_int_view: a 32-bit word, for 8-byte elements)."""
        changed = 0
        for t, c in pairs:
            tv, cv = _int_view(t), _int_view(c)
            for a in range(0, tv.numel(), CHUNK):
                changed += int(torch.ne(tv[a:a + CHUNK].to("cpu"), cv[a:a + CHUNK]).sum())
                if changed > self.repair_max_values:
                    return changed, True
        return changed, False

    def _restore(self, e: dict, at=None) -> tuple:
        """(restored, why not, {"values_changed", "values", "at_least"}) for one changed entry."""
        if e["excluded"]:
            return False, f"the scan no longer compares it: {e['excluded']}", None
        tensors = list(e["get"]())
        clean = e.get("clean") or []
        if [_location(t) for t in tensors] != e["at"] or len(clean) != len(tensors):
            return False, "its tensors moved or changed since the clean copy was taken", None
        why, _ = self._shared(e, tensors)
        if why:
            return False, f"it {why}", None
        pairs = []
        for t, c in zip(tensors, clean):
            if c is None or not isinstance(t, torch.Tensor):
                continue
            if t.shape != c.shape or t.dtype != c.dtype:
                return False, "a tensor's shape or dtype differs from its clean copy", None
            pairs.append((t, c))
        dev = e["device"]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)          # no queued kernel still reads the weights while they're compared
        changed, stopped = self._count_changed(pairs)
        counts = {"values_changed": changed, "values": sum(_int_view(c).numel() for _, c in pairs),
                  "at_least": stopped}
        if changed > self.repair_max_values:
            return False, (f"{'at least ' if stopped else ''}{changed} of {counts['values']} values differ from the "
                           f"copy kept with the reference, more than a fault in memory would change (repair writes "
                           f"back at most {self.repair_max_values}): an update sdcguard wasn't told about (call "
                           f"refresh() after one) or a stray write; the layer keeps its weights"), counts
        try:
            for t, c in pairs:
                if t.is_inference():
                    with torch.inference_mode():
                        t.copy_(c)
                else:
                    with torch.inference_mode(False), torch.no_grad():     # (no_grad last: inference_mode(False)
                        t.copy_(c)                                           # turns gradients back on)
        finally:
            self._written(e, at)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        now = _fingerprints(tensors, dev)
        if bool(torch.equal(now.cpu(), e["ref"].cpu())):
            return True, None, counts
        return False, ("the restored weights still don't match their reference fingerprint (the copy, or the GPU "
                       "memory, is bad)"), counts

    def _repair_record(self, e: dict, ok: bool, why: Optional[str], at: Optional[int], window, clean,
                       counts: Optional[dict]) -> SDCEvent:
        g = self.guard
        ident = getattr(g, "_identity", None)
        identity = dict(ident.get(str(e["device"]), {})) if isinstance(ident, dict) else {}
        n = (counts or {}).get("values_changed")
        start = window[0] if window else at
        detail = {"where": e["name"], "kind": e["kind"],
                  "finding": "weights_restored" if ok else "weights_restore_failed",
                  "recheck": {"verdict": "weights_restored" if ok else "restore_failed",
                              "why": "the copy kept with the reference was written back and the tensors match their "
                                     "reference fingerprint again" if ok else why},
                  "repairs": e["repairs"], "values_changed": n, "values": (counts or {}).get("values"),
                  "values_changed_at_least": bool((counts or {}).get("at_least")),
                  "repaired_at_step": at,
                  # the steps that may have used the changed weights: from the alert's window to the repair
                  "read_window_steps": [start, at] if start is not None and at is not None and start <= at else None,
                  "last_clean_read_steps": clean,
                  "reading": (f"restored in place ({n} changed value{'s' if n != 1 else ''} written back from the "
                              "copy kept at load) and verified; the requests this layer served from its last clean "
                              "comparison to the repair (potentially_exposed) may have had answers computed with the "
                              "changed weights, and so may prefix-cache blocks written then; a change that comes back "
                              "is alerted again" if ok else
                              f"not restored ({why}): the layer keeps its changed weights; take the server out of "
                              "service")}
        ev = SDCEvent(detector="serve_weights", step=None, detail=detail, identity=identity)
        (log.warning if ok else log.error)("sdcguard: weight repair of %s: %s", e["name"],
                                           f"restored ({n} values) and verified" if ok else why)
        if getattr(g, "attribute", None) is not None:
            try:
                g.attribute(ev)
            except Exception:  # noqa: BLE001  (naming requests is extra; the record itself goes out)
                g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
        g.log.add(ev)
        if getattr(g, "on_alert", None):
            try:
                g.on_alert(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

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
        window, since = None, None
        if step is not None:
            if clean is not None:                # since the last clean comparison
                start, since = (clean + 1 if clean < step else step), "the last clean comparison"
            else:                                # never found clean: since the reference was taken (at load: step 0)
                ref = e.get("ref_step")
                start = min(ref + 1, step) if ref is not None else 0
                since = "the reference (at load or the last refresh)" if ref is not None else "load"
            window = [start, step]
        detail = {"where": e["name"], "finding": "weights_changed", "kind": e["kind"],
                  "weights_changed_since_load": True,
                  "recheck": {"verdict": "weights_changed",
                              "why": "the layer's tensors no longer match their reference fingerprint (taken at "
                                     "load, or at the last refresh)"},
                  "read_window_steps": window, "window_since": since,
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
                "repair": {"on": self.repair, "off_because": self.repair_off,
                           "max_values": self.repair_max_values, "host_copy_bytes": self.stats["host_copy_bytes"],
                           "host_allowance_bytes": self._host_allowance,
                           "repaired": self.stats["repaired"], "failed": self.stats["repair_failed"],
                           "entries_repaired": sorted(e["name"] for e in self.entries if e.get("repairs")),
                           # restored, but no scheduled comparison has come back since the repair (its own check
                           # right after the write-back aside)
                           "not_compared_since_repair": sorted(
                               e["name"] for e in self.entries if e.get("repairs") and not e["excluded"] and
                               e["compared"] <= e.get("compared_at_repair", -1))},
                "changed": sorted(e["name"] for e in self.entries if e["alerted"]),
                "not_scanned": dict(self.not_scanned),
                # the fingerprint's implementation per device and element size: the Triton kernel, or PyTorch and why
                "kernel": {f"{d} {es}-byte": ("Triton" if ok else f"PyTorch ({why})")
                           for (d, es), (ok, why) in sorted(STATUS.items()) if d in devs}}


def _bias(m) -> Optional[torch.Tensor]:
    b = getattr(m, "bias", None)
    return b if isinstance(b, torch.Tensor) else None
