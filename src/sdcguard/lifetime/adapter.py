"""The adapter interface: record lifetime events from a real server's hooks, and hold one event until a condition
holds, so an ordering the explorer found in a model can be driven in the real implementation.

A server process runs a Probe. Its hook wrappers (vllm_nixl.py for vLLM's NIXL path) call
  probe.alloc(region, owner) / probe.free(region, by)            the KV block manager gave out or took back a block
  probe.publish(desc, regions, by)                               blocks published to another worker (descriptor)
  probe.submit(op, kind, by, src=(desc, i) or region, dst=...)   an asynchronous transfer started
  probe.cancel(op, by) / probe.stopped(op)                       cancel requested / the transport confirmed it stopped
  probe.complete(op, marks_ready=region) / probe.apply_completion(op, region)
  probe.consume(request, regions)                                the model runs request on these blocks
  probe.evidence(what, **facts)                                  a fact a safety rule needs that the ledger does not
                                                                 model (a request attempt and its lease, a transfer
                                                                 reported done, a lease check's result, the engine's
                                                                 decision to admit or recompute); see obligations.py
and, at the points where an ordering can be changed, probe.checkpoint(point, key) before doing the work. A Hold says:
"at checkpoint (point, key), wait until this condition is seen in the event stream" (for example, hold the decode
side's read submission for request X until the prefill side has freed X's blocks). Holding a thread in one process
while the other processes run is an ordering the transport allows (a slow consumer); a hold never fabricates an
event.

Events go to a sink. LocalSink feeds a Ledger in this process (one-process tests, a single-node server).
JsonlSink appends one JSON line per event to a file per process; merge() replays the files in timestamp order into
one Ledger, so a two-process prefill/decode run is checked as one history. Holds across processes use a holds file
the controller writes and the probes poll (HoldFile), and the event files themselves (a hold's condition is checked
against every process's events)."""

from __future__ import annotations

import json
import math
import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass

from .ledger import Ledger


def _region(r):
    return tuple(r) if isinstance(r, (list, tuple)) else (str(r), 0)


class LocalSink:
    """Apply events to a Ledger in this process."""

    def __init__(self, ledger: Ledger | None = None):
        self.ledger = ledger or Ledger()
        self.events: list = []
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def emit(self, ev: dict):
        with self._cv:
            self.events.append(ev)
            apply(self.ledger, ev, len(self.events))
            self._cv.notify_all()

    def seen(self, pred) -> bool:
        with self._lock:
            return any(pred(e) for e in self.events)

    def history(self) -> list:
        with self._lock:
            return list(self.events)

    def wait(self, pred, timeout: float) -> bool:
        end = time.monotonic() + timeout
        with self._cv:
            while not any(pred(e) for e in self.events):
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
            return True


class JsonlSink:
    """Append events to path (one JSON object per line); other processes' files are read for hold conditions."""

    def __init__(self, path: str, peers: tuple = (), asynchronous: bool = False):
        self.path = path
        self.peers = tuple(peers)
        self._lock = threading.Lock()
        self._read: dict = {}           # path -> (bytes read so far, events parsed so far)
        self.errors = 0
        self._queue = queue.Queue(maxsize=16384) if asynchronous else None
        self._writer = None
        if self._queue is not None:
            self._writer = threading.Thread(target=self._write_loop, daemon=True, name="lifetime-writer")
            self._writer.start()

    def emit(self, ev: dict):
        if self._queue is not None:
            try:
                self._queue.put_nowait(ev)
            except queue.Full:
                self.errors += 1
            return
        self._write(ev)

    def _write(self, ev):
        line = json.dumps(ev, sort_keys=True)
        with self._lock, open(self.path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def _write_loop(self):
        while True:
            ev = self._queue.get()
            try:
                if ev is None:
                    return
                self._write(ev)
            except Exception:  # recorder failures must not escape into serving
                self.errors += 1
            finally:
                self._queue.task_done()

    def flush(self, timeout=1.0):
        end = time.monotonic() + timeout
        while self._queue is not None and self._queue.unfinished_tasks and time.monotonic() < end:
            time.sleep(0.001)
        return self._queue is None or not self._queue.unfinished_tasks

    def close(self):
        self.flush()
        if self._queue is not None:
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                self.errors += 1
            if self._writer is not None:
                self._writer.join(timeout=1.0)

    def _events_of(self, p) -> list:
        """The events in file p, reading only what was appended since the last call (whole lines only)."""
        off, evs = self._read.get(p, (0, []))
        try:
            with open(p, "rb") as f:
                f.seek(off)
                chunk = f.read()
        except OSError:
            return evs
        end = chunk.rfind(b"\n")
        if end >= 0:
            evs = evs + [json.loads(x) for x in chunk[:end].split(b"\n") if x.strip()]
            off += end + 1
        self._read[p] = (off, evs)
        return evs

    def _all(self):
        out = []
        for p in (self.path, *self.peers):
            if os.path.exists(p):
                out.extend(self._events_of(p))
        return out

    def history(self) -> list:
        return sorted(self._all(), key=lambda e: (e.get("t", 0), e.get("pid", 0), e.get("seq", 0)))

    def seen(self, pred) -> bool:
        return any(pred(e) for e in self._all())

    def wait(self, pred, timeout: float, poll: float = 0.005) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.seen(pred):
                return True
            time.sleep(poll)
        return False


class JsonlDirSink(JsonlSink):
    """One event file per process in a shared directory (dir/<pid>.jsonl); hold conditions see every file there."""

    def __init__(self, directory: str, pid: int | None = None, asynchronous: bool = False):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        super().__init__(os.path.join(directory, f"{os.getpid() if pid is None else pid}.jsonl"),
                         asynchronous=asynchronous)

    def _all(self):
        self.peers = tuple(os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir))
                           if f.endswith(".jsonl") and os.path.join(self.dir, f) != self.path)
        return super()._all()


def read_events(paths) -> list:
    """Read one consistent snapshot of the recorded files.

    A writer's unfinished trailing line is not an event yet. A malformed *complete* line,
    unreadable file, or invalid event envelope is evidence loss, retained as an explicit
    recording_error instead of silently turning a damaged trace into a clean one.
    """
    if isinstance(paths, (str, os.PathLike)) and os.path.isdir(paths):
        paths = [os.path.join(paths, f) for f in sorted(os.listdir(paths)) if f.endswith(".jsonl")]
    elif isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    evs = []
    for p in paths:
        try:
            with open(p, "rb") as f:
                data = f.read()
        except OSError as exc:
            evs.append(_read_error(p, None, type(exc).__name__))
            continue
        for number, line in enumerate(data.splitlines(keepends=True), 1):
            if not line.endswith(b"\n") or not line.strip():
                continue
            try:
                ev = json.loads(line)
                if not isinstance(ev, dict) or not isinstance(ev.get("kind"), str):
                    raise ValueError("invalid event")
                if not isinstance(ev.get("args", {}), dict):
                    raise ValueError("invalid args")
                if type(ev.get("t")) not in (int, float) or not math.isfinite(ev["t"]):
                    raise ValueError("invalid time")
                if ev.get("mono") is not None and (type(ev["mono"]) not in (int, float)
                                                   or not math.isfinite(ev["mono"])):
                    raise ValueError("invalid monotonic time")
                if ev.get("clock_id") is not None and not isinstance(ev["clock_id"], str):
                    raise ValueError("invalid clock identity")
                if "seq" in ev and type(ev["seq"]) is not int:
                    raise ValueError("invalid sequence")
                evs.append(ev)
            except (ValueError, UnicodeError):
                evs.append(_read_error(p, number, "malformed event"))
    clocks = {e.get("clock_id") for e in evs}
    monotonic = len(clocks) == 1 and None not in clocks and all(
        isinstance(e.get("mono"), (int, float)) for e in evs)
    # Monotonic clocks are comparable only within the same host boot. The evaluator
    # independently checks that boundary; fallback wall ordering is legacy ledger data.
    evs.sort(key=lambda e: (e.get("mono" if monotonic else "t", 0),
                            str(e.get("pid", "")), e.get("seq", 0)))
    return evs


def _read_error(path, line, reason):
    return {"kind": "evidence", "args": {"what": "recording_error", "path": str(path),
                                           "line": line, "reason": reason}, "t": 0}


def merge_events(evs, track_content: bool = True) -> Ledger:
    """Replay the already-read event snapshot; do not race a second file read."""
    led = Ledger(track_content=track_content)
    descs: dict = {}
    for i, e in enumerate(evs):
        apply(led, e, i, descs)
    return led


def merge(paths, track_content: bool = True) -> Ledger:
    """Replay several event files (or a directory) into one legacy lifetime ledger."""
    return merge_events(read_events(paths), track_content=track_content)


def apply(led: Ledger, ev: dict, step: int, descs: dict | None = None):
    """Apply one recorded event to the ledger. Descriptor handles are resolved to the generation current when the
    descriptor was published, so a later submit that names (descriptor, index) refers to that allocation."""
    if descs is None:
        descs = led.__dict__.setdefault("_adapter_descs", {})
    k = ev["kind"]
    a = ev.get("args", {})
    if k == "alloc":
        led.alloc(_region(a["region"]), a["owner"], step)
    elif k == "free":
        led.free(_region(a["region"]), a["by"], step)
    elif k == "share":
        led.share(_region(a["region"]), a["request"], step)
    elif k == "write_local":
        led.write_local(_region(a["region"]), a.get("label", "local write"), step)
    elif k == "publish":
        hs = [led.current(_region(r)) for r in a["regions"]]
        descs[a["desc"]] = hs
        led.publish(a["desc"], hs, a["by"], step)
    elif k == "accept":
        led.accept(a["desc"], a["by"], step)
    elif k == "submit":
        def res(x):
            if x is None:
                return None
            if isinstance(x, dict) and "desc" in x:
                return descs[x["desc"]][x.get("i", 0)]
            return led.current(_region(x))
        payload = tuple(a["payload"]) if a.get("payload") is not None else None
        led.submit(a["op"], a["kind"], a["by"], src=res(a.get("src")), dst=res(a.get("dst")), payload=payload,
                   step=step)
    elif k == "cancel":
        led.cancel(a["op"], a["by"], step)
    elif k == "stopped":
        led.stopped(a["op"], step)
    elif k == "complete":
        mr = a.get("marks_ready")
        led.complete(a["op"], step, marks_ready=_region(mr) if mr is not None else None)
    elif k == "apply_completion":
        led.apply_completion(a["op"], _region(a["region"]), step)
    elif k == "consume":
        led.consume(a["request"], [led.current(_region(r)) for r in a["regions"]], step)
    elif k in ("checkpoint", "held", "released", "note", "evidence"):
        led._log(step, k, ev.get("text", k))
    else:
        raise ValueError(f"unknown lifetime event {k!r}")


def holds_from_file(path: str) -> list:
    """Holds written by a controller: [{"point", "key", "until": {"kind", "args"}, "timeout"}]."""
    with open(path) as f:
        return holds_from_spec(json.load(f))


def holds_from_spec(spec: list) -> list:
    """The same holds from an already-parsed list (the search's findings carry one). A condition that names blocks
    symbolically (see resolve) is resolved when the hold starts, against the events recorded by then."""
    out = []
    for h in spec:
        u = h["until"]
        pred = None if _symbolic(u) else until_event(u["kind"], **u.get("args", {}))
        out.append(Hold(h["point"], h.get("key"), pred, float(h.get("timeout", 10.0)), bool(h.get("once", True)),
                        spec=u))
    return out


_SYMBOLS = ("desc_block", "key_block")


def _symbolic(x) -> bool:
    if isinstance(x, dict):
        if len(x) == 1 and next(iter(x)) in _SYMBOLS:
            return True
        return any(_symbolic(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return any(_symbolic(v) for v in x)
    return False


def resolve(spec, key, ctx: dict, history: list):
    """Replace symbolic block names in a hold condition with this run's blocks, so a hold found in one run can be
    applied to a later run of a server that numbers its blocks and requests differently each time:
      {"desc_block": i}   the i-th block of the descriptor this checkpoint is about (its ctx "desc")
      {"key_block": i}    the i-th block allocated to this checkpoint's key (the request being held)
    Returns None when a name cannot be resolved from what was recorded so far (the hold then just waits out its
    timeout, which is still a legal delay)."""
    descs, owned = {}, []
    for e in history:
        a = e.get("args", {})
        if e.get("kind") == "publish":
            descs[a.get("desc")] = a.get("regions") or []
        elif e.get("kind") == "alloc" and key is not None and a.get("owner") == key and a.get("region") not in owned:
            owned.append(a.get("region"))

    class _Missing(Exception):
        pass

    def sub(x):
        if isinstance(x, dict):
            if len(x) == 1 and "desc_block" in x:
                rs = descs.get((ctx or {}).get("desc"), [])
                if x["desc_block"] >= len(rs):
                    raise _Missing
                return list(rs[x["desc_block"]])
            if len(x) == 1 and "key_block" in x:
                if x["key_block"] >= len(owned):
                    raise _Missing
                return list(owned[x["key_block"]])
            return {k: sub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [sub(v) for v in x]
        return x
    try:
        return sub(spec)
    except _Missing:
        return None


@dataclass
class Hold:
    """At checkpoint (point, key), wait until an event matching until(event) has been recorded, or timeout seconds
    pass (then proceed and record that the hold expired: the ordering was not reached)."""
    point: str
    key: object
    until: object
    timeout: float = 10.0
    once: bool = True
    spec: dict | None = None        # the condition as JSON, when it came from a holds spec


class Probe:
    """The per-process recorder and hold point. pid tells processes apart in merged histories."""

    def __init__(self, sink, pid: int | None = None, holds: list | None = None, record_checkpoints: bool = False,
                 holds_path: str | None = None, pauses: tuple | None = None, health_interval: float = 0.0):
        self.sink = sink
        self.pid = os.getpid() if pid is None else pid
        self.holds = list(holds or [])
        self.holds_path = holds_path      # a controller may rewrite this file between requests; reread on change
        self._holds_mtime = None
        self.record_checkpoints = record_checkpoints   # a baseline run for the search records every checkpoint
        self._seq = 0
        self._lock = threading.Lock()
        self.session = uuid.uuid4().hex
        self.host = socket.gethostname()
        try:
            with open("/proc/sys/kernel/random/boot_id") as f:
                self.clock_id = self.host + ":" + f.read().strip()
        except OSError:
            # A hostname alone does not establish a shared clock across reboots.
            self.clock_id = None
        self._dropped = 0
        self._health_stop = threading.Event()
        self._health_thread = None
        self.held: list = []           # (point, key, reached: bool, waited seconds)
        self.pauses = None             # (probability, max seconds, random.Random): random short delays at checkpoints
        if pauses is not None:
            import random
            prob, max_s, seed = pauses
            self.pauses = (prob, max_s, random.Random(seed))
        if health_interval > 0:
            def health():
                while not self._health_stop.wait(health_interval):
                    self.evidence("recording_health", healthy=self._dropped + getattr(self.sink, "errors", 0) == 0)
            self._health_thread = threading.Thread(target=health, daemon=True, name="lifetime-health")
            self._health_thread.start()

    def close(self):
        self._health_stop.set()
        if self._health_thread is not None:
            self._health_thread.join(timeout=1.0)
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()

    def recording_error(self, reason="recording failed"):
        """Permanently taint this recording; a missing event cannot become clean evidence."""
        with self._lock:
            self._dropped += 1

    def _emit(self, ev_kind, _text="", **args):
        with self._lock:
            self._seq += 1
            seq = self._seq
            try:
                self.sink.emit({"t": time.time(), "mono": time.monotonic(), "host": self.host,
                                "clock_id": self.clock_id, "session": self.session, "pid": self.pid,
                                "tid": threading.get_ident(), "seq": seq,
                                "dropped": self._dropped + getattr(self.sink, "errors", 0),
                                "kind": ev_kind, "args": args, "text": _text})
            except Exception:
                self._dropped += 1

    # recording
    def alloc(self, region, owner): self._emit("alloc", region=list(_region(region)), owner=owner)
    def free(self, region, by): self._emit("free", region=list(_region(region)), by=by)
    def share(self, region, request): self._emit("share", region=list(_region(region)), request=request)
    def write_local(self, region, label=""): self._emit("write_local", region=list(_region(region)), label=label)

    def publish(self, desc, regions, by):
        self._emit("publish", desc=desc, regions=[list(_region(r)) for r in regions], by=by)

    def accept(self, desc, by): self._emit("accept", desc=desc, by=by)

    def submit(self, op, kind, by, src=None, dst=None, payload=None):
        def enc(x):
            if x is None:
                return None
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and x[0].startswith("desc:"):
                return {"desc": x[0][5:], "i": x[1]}
            return list(_region(x))
        self._emit("submit", op=op, kind=kind, by=by, src=enc(src), dst=enc(dst),
                   payload=list(payload) if payload is not None else None)

    def cancel(self, op, by): self._emit("cancel", op=op, by=by)
    def stopped(self, op): self._emit("stopped", op=op)

    def complete(self, op, marks_ready=None):
        self._emit("complete", op=op, marks_ready=list(_region(marks_ready)) if marks_ready is not None else None)

    def apply_completion(self, op, region): self._emit("apply_completion", op=op, region=list(_region(region)))

    def consume(self, request, regions):
        self._emit("consume", request=request, regions=[list(_region(r)) for r in regions])

    def evidence(self, what: str, **facts):
        """Record a fact for the safety rules (obligations.py) that the ledger's rules do not use. Only facts that
        were observed are passed; a fact not passed stays unknown, never assumed."""
        self._emit("evidence", f"evidence {what} " + " ".join(f"{k}={facts[k]}" for k in sorted(facts)),
                   what=what, **facts)

    # control
    def _reload_holds(self):
        if not self.holds_path:
            return
        try:
            m = os.path.getmtime(self.holds_path)
        except OSError:
            return
        if m != self._holds_mtime:
            self._holds_mtime = m
            try:
                self.holds = holds_from_file(self.holds_path)
            except (OSError, ValueError, KeyError):
                self.holds = []

    def checkpoint(self, point: str, key=None, **ctx):
        """Called by a hook before work that the explorer may reorder. Waits if a hold is armed here, until an event
        matching its condition is recorded after the hold began (an event from before cannot release it). ctx names
        what the work is about (for example desc=the descriptor a read will use), for symbolic hold conditions."""
        self._reload_holds()
        if self.record_checkpoints:
            self._emit("checkpoint", f"checkpoint {point} {key}", point=point, key=key, **ctx)
        if self.pauses is not None:
            prob, max_s, rng = self.pauses
            with self._lock:
                d = rng.uniform(0, max_s) if rng.random() < prob else 0.0
            if d:
                self._emit("note", f"paused {point} {key} for {d:.3f}s")
                time.sleep(d)
        for h in list(self.holds):
            if h.point == point and (h.key is None or h.key == key):
                if h.once:
                    self.holds.remove(h)
                until = h.until
                if until is None and h.spec is not None:
                    # A hold already deliberately pauses this thread. Flush only here,
                    # so its symbolic lookup sees our publication/checkpoint without
                    # adding synchronous disk I/O to ordinary serving hooks.
                    flush = getattr(self.sink, "flush", None)
                    if flush is not None and not flush(timeout=min(1.0, h.timeout)):
                        self.recording_error("recording did not flush before symbolic hold")
                    hist = self.sink.history() if hasattr(self.sink, "history") else []
                    r = resolve(h.spec, key, ctx, hist)
                    until = until_event(r["kind"], **r.get("args", {})) if r is not None else (lambda e: False)
                    self._emit("note", f"hold condition for {point} {key}: {json.dumps(r) if r else 'unresolved'}")
                self._emit("held", f"held {point} {key}")
                t0 = time.monotonic()
                since = time.time()
                reached = self.sink.wait(lambda e, u=until: e.get("t", 0) >= since and u(e), h.timeout)
                waited = time.monotonic() - t0
                self.held.append((point, key, reached, waited))
                self._emit("released", f"released {point} {key} "
                           f"({'condition reached' if reached else 'hold expired'}) after {waited:.3f}s")
                return reached
        return None


def desc_ref(desc: str, i: int = 0) -> tuple:
    """Name the i-th block of a published descriptor as a transfer's source or destination."""
    return (f"desc:{desc}", i)


def until_event(ev_kind: str, /, **match):
    """A hold condition: an event of this kind whose args include these values. "*" matches any value (the key must
    be present); inside a region, ["P", "*"] matches any block of P. {"not": v} matches anything but v (a missing
    value included); {"has": v} matches a list with an element matching v. All of it is plain JSON, so a hold found
    by the search can be written to a holds file and replayed later (holds_from_file)."""
    def same(want, got):
        if want == "*":
            return got is not None
        if isinstance(want, dict) and len(want) == 1 and "not" in want:
            return not same(want["not"], got)
        if isinstance(want, dict) and len(want) == 1 and "has" in want:
            return isinstance(got, list) and any(same(want["has"], g) for g in got)
        if isinstance(want, (list, tuple)):
            return isinstance(got, list) and len(got) == len(want) and all(same(w, g) for w, g in zip(want, got))
        return want == got

    def pred(e):
        if e.get("kind") != ev_kind:
            return False
        a = e.get("args", {})
        return all(same(v, a.get(k)) for k, v in match.items())
    return pred
