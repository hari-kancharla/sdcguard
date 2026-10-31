"""Optional callback transport for one existing forward observer.

Owner: Hari Krishna Kancharla.
This bridge emits no allocator or initialization evidence and starts no engine.
The supplied observer must establish its own native associations and accounting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
from types import MappingProxyType
import uuid
import weakref

from . import forward_inputs as contract

MAX_CALLBACKS = 32768
MAX_IN_FLIGHT = 64
MAX_RUNNERS = 256
_FAILED = object()
_METHODS = ("runner_initialized", "invocation_enter", "prepared", "begin_copy",
            "end_copy", "abandon_copy", "invocation_leave", "retire_runner",
            "runner_closed", "close")


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if type(value) is list:
        return tuple(_freeze(child) for child in value)
    return value


def _token(value):
    return (type(value) is str and 0 < len(value) <= 128
            and all(33 <= ord(char) < 127 for char in value))


def _same(left, right):
    return (getattr(left, "__func__", left) is getattr(right, "__func__", right)
            and getattr(left, "__self__", None) is getattr(right, "__self__", None))


class AssignmentHealth:
    """Sticky callback accounting. It is not an isolation verdict or writer."""

    def __init__(self):
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._errors = 0
        self._reasons = []

    def error(self, reason):
        if os.getpid() != self.pid:
            return
        with self._lock:
            self._errors += 1
            if len(self._reasons) < contract.MAX_REASONS:
                self._reasons.append(str(reason)[:contract.MAX_TEXT])

    def snapshot(self):
        if os.getpid() != self.pid:
            return {"healthy": False, "errors": None, "reasons": ("process changed",)}
        with self._lock:
            return {"healthy": not self._errors, "errors": self._errors,
                    "reasons": tuple(self._reasons)}


@dataclass(frozen=True)
class Registration:
    runner_class: object
    engine: object
    _installation: object = field(default=None, repr=False, compare=False)
    _model_getter: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self._installation is None:
            return
        from .vllm_forward import Installation
        if type(self._installation) is not weakref.ReferenceType:
            raise ValueError("model binding installation reference differs")
        owner = self._installation()
        if type(owner) is not Installation:
            raise ValueError("model binding installation class differs")
        getter = owner.model_binding_for
        if (getattr(getter, "__self__", None) is not owner
                or getattr(getter, "__func__", None) is not Installation.model_binding_for):
            raise ValueError("model binding installation method differs")
        object.__setattr__(self, "_model_getter", weakref.WeakMethod(getter))

    def model_binding_installation(self):
        """Resolve one weak, process-local capability without retaining its owner."""
        if self._installation is None:
            return None
        from .vllm_forward import Installation
        if type(self._installation) is not weakref.ReferenceType:
            raise ValueError("model binding installation reference differs")
        owner = self._installation()
        if (type(owner) is not Installation or owner.pid != os.getpid()
                or owner.runner_class is not self.runner_class):
            raise ValueError("model binding installation identity differs")
        contract._bounded(owner.engine)
        if _freeze(owner.engine) != self.engine:
            raise ValueError("model binding engine registration changed")
        getter = owner.model_binding_for
        baseline = self._model_getter() if type(self._model_getter) is weakref.WeakMethod else None
        if (baseline is None or getattr(baseline, "__self__", None) is not owner
                or not _same(getter, baseline)
                or getattr(baseline, "__func__", None) is not Installation.model_binding_for):
            raise ValueError("model binding installation method changed")
        return owner

    def model_binding_for(self, runner):
        owner = self.model_binding_installation()
        if owner is None or type(runner) is not self.runner_class:
            raise ValueError("model binding runner capability is unavailable")
        return owner.model_binding_for(runner)


@dataclass(frozen=True)
class RunnerInitialization:
    runner: object
    arguments: tuple
    runner_config: object
    block_tables: object
    cache_label: str


@dataclass(frozen=True)
class RunnerInvocation:
    runner: object
    scheduler_output: object
    forward_id: int | None


@dataclass(frozen=True)
class PreparedRows:
    requests: tuple
    num_tokens: int


@dataclass(frozen=True)
class CopyBoundary:
    runner: object
    forward_id: int
    cache_label: str
    prepared: PreparedRows


@dataclass(frozen=True)
class FrozenModelEntryCopy:
    record: object


@dataclass(frozen=True)
class InvocationOutcome:
    returned: bool
    exception_type: str | None
    eager_copy: bool
    poisoned: bool


@dataclass
class _Invocation:
    key: str
    token: str | None
    runner: object
    pid: int
    frame: str | None = None
    copied: bool = False
    begun: bool = False


class ForwardBridge:
    """One process-local optional observer, without another model hook."""

    def __init__(self, installation, factory):
        self.pid = os.getpid()
        self.factory = factory
        self.health = AssignmentHealth()
        self.closed = False
        self._lock = threading.Lock()
        self._calls = 0
        self._bytes = self._nodes = self._cells = 0
        self._inflight = {}
        self._used_tokens = set()
        self._frame_owners = {}
        self._runners = {}
        self._installation = weakref.ref(installation)
        contract._bounded(installation.engine)
        registration = Registration(installation.runner_class, _freeze(installation.engine),
                                    self._installation)
        self._observer = factory(registration, self.health)
        self._registered = self._observer
        try:
            self._callbacks = {name: getattr(self._observer, name) for name in _METHODS}
            if not all(callable(method) for method in self._callbacks.values()):
                raise ValueError("assignment callback protocol is incomplete")
        except BaseException:
            self.health.error("assignment callback protocol is incomplete")
            try:
                self._observer.close(timeout=1.0)
            except BaseException:
                pass
            raise

    def _current(self):
        # Never acquire inherited locks or call a parent's observer after fork.
        return self.pid == os.getpid()

    def _call(self, name, *args, cleanup=False, **kwargs):
        if not self._current() or (self.closed and not cleanup):
            return _FAILED
        try:
            unchanged = self._observer is self._registered and all(
                _same(getattr(self._observer, key), fn) for key, fn in self._callbacks.items())
            if not unchanged:
                self.health.error("assignment callback registration changed")
                if not cleanup:
                    return _FAILED
            if not cleanup:
                with self._lock:
                    self._calls += 1
                    if self._calls > MAX_CALLBACKS:
                        raise ValueError("assignment callback bound exceeded")
            return self._callbacks[name](*args, **kwargs)
        except BaseException as exc:
            self.health.error(name + ": " + type(exc).__name__)
            if not cleanup and not isinstance(exc, Exception):
                raise
            return _FAILED

    def _runner(self, runner):
        key = id(runner)
        with self._lock:
            previous = self._runners.get(key)
            if previous is not None and previous[0]() is runner:
                return previous
            if len(self._runners) >= MAX_RUNNERS:
                raise ValueError("assignment runner bound exceeded")
            owner = weakref.ref(self)
            def collected(reference):
                bridge = owner()
                if bridge is not None and bridge._current():
                    with bridge._lock:
                        if bridge._runners.get(key, (None,))[0] is reference:
                            bridge._runners.pop(key, None)
            entry = [weakref.ref(runner, collected), "new"]
            self._runners[key] = entry
            return entry

    def _active(self, invocation, runner):
        if invocation is None or not self._current() or self.closed:
            return False
        with self._lock:
            valid = (type(invocation) is _Invocation and invocation.pid == self.pid
                     and self._inflight.get(invocation.key) is invocation
                     and invocation.runner() is runner)
        if not valid:
            self.health.error("assignment invocation lifetime or runner differs")
        return valid

    def _claim(self, token):
        with self._lock:
            if (not _token(token) or token in self._used_tokens
                    or len(self._used_tokens) >= MAX_CALLBACKS):
                self.health.error("assignment token is invalid, reused or exceeds bounds")
                return False
            self._used_tokens.add(token)
            return True

    def retire(self, runner, reason, *, cleanup=False):
        if not self._current() or self.closed:
            return
        try:
            entry = self._runner(runner)
            if reason == "native shutdown" and entry[1] == "closed":
                return
            entry[1] = "retired"
            self._call("retire_runner", runner, reason, cleanup=cleanup)
        except BaseException as exc:
            self.health.error("retirement: " + type(exc).__name__)
            if not cleanup and not isinstance(exc, Exception):
                raise

    def initialized(self, runner, bound, state):
        if not self._current() or self.closed:
            return
        try:
            entry = self._runner(runner)
            entry[1] = "ready"
            parameters = tuple(bound.arguments.items())[1:]
            self._call("runner_initialized", RunnerInitialization(
                runner, parameters, runner.kv_cache_config, runner.block_tables, state.pool))
        except Exception as exc:
            self.health.error("initialization: " + type(exc).__name__)

    def enter(self, runner, scheduler_output, forward_id):
        if not self._current() or self.closed:
            return None
        with self._lock:
            if len(self._inflight) >= MAX_IN_FLIGHT:
                self.health.error("assignment invocation bound exceeded")
                return None
            invocation = _Invocation(uuid.uuid4().hex, None, weakref.ref(runner), self.pid)
            self._inflight[invocation.key] = invocation
        try:
            value = self._call("invocation_enter", RunnerInvocation(runner, scheduler_output, forward_id))
        except BaseException:
            with self._lock:
                self._inflight.pop(invocation.key, None)
            raise
        if value is None:
            self.health.error("assignment positive-work invocation was not observed")
        elif self._claim(value):
            invocation.token = value
        return invocation

    def prepared(self, invocation, runner, requests, count):
        if self._active(invocation, runner) and invocation.token is not None:
            installation = self._installation()
            active = getattr(installation.local, "active", None) if installation is not None else None
            if (active is None or active.get("assignment") is not invocation
                    or active.get("runner") is not runner or type(requests) is not list
                    or not 0 < len(requests) <= contract.MAX_REQUESTS
                    or not all(contract._text(value) for value in requests)
                    or not contract._integer(count, 1, contract.MAX_TOKENS)):
                self.health.error("assignment preparation lifetime or bounds differ")
                return
            self._call("prepared", invocation.token, PreparedRows(tuple(requests), count))

    def _coverage(self, state):
        installation = self._installation()
        if installation is None or not installation.bridge_coverage(state):
            self.health.error("assignment forward-hook coverage changed")
            return False
        return True

    def begin(self, invocation, state, active):
        if not self._active(invocation, state.runner()) or invocation.token is None:
            return
        if (active.get("assignment") is not invocation or active.get("runner") is not state.runner()
                or active.get("state") is not state):
            self.health.error("assignment copy is outside its native invocation")
            return
        if invocation.begun:
            self.health.error("assignment copy interval was already started")
            return
        invocation.begun = True
        if not self._coverage(state):
            return
        pending = active.get("pending")
        if pending is None:
            self.health.error("assignment prepared rows are missing")
            return
        value = self._call("begin_copy", invocation.token, CopyBoundary(
            state.runner(), active["forward_id"], state.pool, PreparedRows(tuple(pending[0]), pending[1])))
        if self._claim(value):
            invocation.frame = value
            self._frame_owners[value] = invocation

    def copied(self, invocation, state, snapshot):
        if not self._active(invocation, state.runner()) or invocation.frame is None:
            return
        installation = self._installation()
        active = getattr(installation.local, "active", None) if installation is not None else None
        if (active is None or active.get("assignment") is not invocation
                or active.get("state") is not state):
            self.health.error("assignment copy is outside its native invocation")
            return
        if (invocation.copied or not invocation.begun
                or self._frame_owners.get(invocation.frame) is not invocation):
            self.health.error("assignment copy interval lifetime differs")
            return
        try:
            raw, nodes = contract._bounded(snapshot)
            cells = sum(len(layer["seq_lens"]) + len(layer["query_start_loc"])
                        + len(layer["write_slots"]) + sum(map(len, layer["block_table"]))
                        for layer in snapshot["layers"])
            with self._lock:
                if (self._bytes + len(raw) > contract.MAX_TOTAL_BYTES
                        or self._nodes + nodes > contract.MAX_TOTAL_NODES
                        or self._cells + cells > contract.MAX_TOTAL_CELLS):
                    raise ValueError("assignment immutable-copy bound exceeded")
                self._bytes += len(raw)
                self._nodes += nodes
                self._cells += cells
            frozen = FrozenModelEntryCopy(_freeze(snapshot))
            if self._coverage(state):
                result = self._call("end_copy", invocation.frame, frozen)
                if result is not _FAILED and self._coverage(state):
                    invocation.copied = True
        except Exception as exc:
            self.health.error("immutable copy: " + type(exc).__name__)

    def abandon(self, invocation, runner, reason):
        if self._active(invocation, runner):
            self.health.error(reason)
            if invocation.token is not None and not invocation.copied:
                self._call("abandon_copy", invocation.token, invocation.frame, reason, cleanup=True)

    def leave(self, invocation, *, runner, returned, exception_type, poisoned):
        if not self._active(invocation, runner):
            return
        try:
            if invocation.token is not None:
                if not invocation.copied:
                    self.abandon(invocation, runner, "assignment invocation had no complete copy")
                self._call("invocation_leave", invocation.token, InvocationOutcome(
                    returned, exception_type, invocation.copied, poisoned), cleanup=True)
        finally:
            with self._lock:
                self._inflight.pop(invocation.key, None)
                if self._frame_owners.get(invocation.frame) is invocation:
                    self._frame_owners.pop(invocation.frame, None)

    def runner_closed(self, runner, exception_type):
        if not self._current() or self.closed:
            return
        try:
            entry = self._runner(runner)
            if entry[1] == "closed":
                return
            entry[1] = "closed"
            if self._call("runner_closed", runner, exception_type, cleanup=True) is not True:
                self.health.error("assignment runner closure is incomplete")
        except BaseException as exc:
            self.health.error("runner closure: " + type(exc).__name__)

    def close(self):
        if not self._current() or self.closed:
            return
        if self._inflight:
            self.health.error("assignment detached during invocation")
        self.closed = True
        if self._call("close", timeout=1.0, cleanup=True) is not True:
            self.health.error("assignment process closure is incomplete")
