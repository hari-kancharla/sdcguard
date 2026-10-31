"""Internal passive assembly of recorded allocator-to-model mapping facts.

Owner: Hari Krishna Kancharla.
Nothing in this module activates an engine, a plugin or a device operation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import functools
import inspect
import os
import threading
from types import MappingProxyType
import weakref

from . import assignment_init as initialization
from . import assignment_obligations as contract

MAX_BUFFER_FACTS = 512
MAX_BUFFER_BYTES = 2 * 1024 * 1024
MAX_DISPATCHES = 1024
MAX_REQUEST_OBSERVATIONS = 8192
MAX_OPEN_MUTATIONS = 64
MAX_OUTBOX_RECORDS = 64
MAX_OUTBOX_BYTES = 8 * 1024 * 1024
_MISSING = object()


class MappingGap(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise MappingGap(reason)


def _plain(value):
    if isinstance(value, MappingProxyType):
        return {key: _plain(child) for key, child in value.items()}
    if type(value) is tuple:
        return [_plain(child) for child in value]
    if type(value) is dict:
        return {key: _plain(child) for key, child in value.items()}
    if type(value) is list:
        return [_plain(child) for child in value]
    return value


def _same_method(value, function, receiver):
    return getattr(value, "__func__", None) is function and getattr(value, "__self__", None) is receiver


def _callback(function):
    @functools.wraps(function)
    def observed(self, *args, **kwargs):
        if not self._admit():
            return None
        try:
            return function(self, *args, **kwargs)
        except BaseException:
            self._error("mapping callback " + function.__name__ + " is incomplete")
            raise
        finally:
            self._release()
    return observed


@dataclass
class _Dispatch:
    output: object
    runner: object
    requests: dict
    facts: list
    entered: bool = False
    left: bool = False
    frame: str | None = None
    copied: bool = False
    invocation: str | None = None
    prepared: tuple | None = None


class MappingObserver:
    """One explicitly assembled process interval using the existing bridge."""

    def __init__(self, registration, health, *, bindings, source_root, sink_factory,
                 initialization_observer=None, closure_callback=None, recipient_factory=None):
        from . import mapping_runtime
        self.pid = os.getpid()
        self.registration, self.health = registration, health
        self.bindings = bindings
        self.adapter = mapping_runtime.NATIVE_ADAPTER
        self.description = contract.adapter_contract(self.adapter)
        self.source_hashes = mapping_runtime.validate_sources(source_root, self.adapter)
        _require(registration.runner_class is bindings.initialization.runner, "bridge runner class differs")
        _require(callable(sink_factory) and callable(getattr(health, "error", None)), "mapping sink or health is missing")
        self.sink_factory = sink_factory
        _require(closure_callback is None or callable(closure_callback), "closure callback is unsupported")
        self._closure_callback = closure_callback
        self._closure_receipt = None
        self.recipient = None
        _require(recipient_factory is None or callable(recipient_factory), "recipient factory is unsupported")
        self.source_root = source_root
        self._header_reader = mapping_runtime.observed_header
        self._lock = threading.RLock()
        self._local = threading.local()
        self._closed = self._closing = self._sealed = False
        self._errors = self._events = self._buffer_bytes = self._epoch = 0
        self._buffer = []
        self._outbox = deque()
        self._outbox_bytes = 0
        self._draining = None
        self._writer = None
        self._hooks = []
        self._initial_token = self._association = None
        self._refs = None
        self._bridge_runner = self._bridge_config = self._bridge_tables = None
        self._bridge_label = None
        self._allocations = {}
        self._mutations = {}
        self._dispatches = {}
        self._invocations = {}
        self._frames = {}
        self._pending = {}
        self._steps = {}
        self._schedules_open = 0
        self._producers = 0
        self._names = {}
        self._request_observations = self._allocation_observations = self._cells = 0
        self._profile = None
        self._dependencies = None
        self._building_header = False
        self._sink_callback = self._initialization_fact
        self.initialization = initialization_observer or initialization.InitializationObserver(
            self._sink_callback, health, maximum_identities=32768)
        _require(type(self.initialization) is initialization.InitializationObserver
                 and self.initialization.pid == self.pid and self.initialization._facts == 0
                 and not self.initialization._inits and not self.initialization._closed,
                 "initialization observer must be fresh")
        self.initialization.sink, self.initialization.health = self._sink_callback, health
        from .forward_bridge import Registration
        if (type(registration) is Registration
                and registration.model_binding_installation() is not None):
            self.initialization.bind_model_references(registration)
        self.registry = self.initialization.registry
        self.process = self.registry.new_token()
        self.methods = {**self.description["mutation_methods"], **self.description["guard_methods"]}
        self.coverage_names = sorted(self.methods)
        self._getter_owner = next(owner for owner in bindings.initialization.coordinator.__mro__
                                  if "get_blocks" in vars(owner))
        self._getter = vars(self._getter_owner)["get_blocks"]
        try:
            if self.initialization._bindings is None:
                self.initialization.attach(bindings.initialization, source_root, registration.engine["version"])
            else:
                _require(self.initialization._bindings is bindings.initialization and self.initialization.coverage(),
                         "preinstalled initialization bindings differ")
            self._attach()
            if recipient_factory is not None:
                from .allocation_recipient_observer import AllocationRecipientObserver
                selected_recipient = recipient_factory(self)
                _require(type(selected_recipient) is AllocationRecipientObserver,
                         "recipient factory did not return the source-reviewed passive observer")
                self.recipient = selected_recipient
                self.recipient.attach()
        except BaseException:
            self._cleanup_hooks()
            self.initialization.detach()
            raise

    def _current(self):
        return self.pid == os.getpid() and not self._closed and not self._closing

    def _mark_error(self, reason):
        if os.getpid() != self.pid:
            return
        self._errors += 1
        try:
            self.health.error(str(reason)[:512])
        except BaseException:
            pass

    def _error(self, reason):
        self._mark_error(reason)
        if self._writer is not None and not self._closed:
            try:
                self._writer.gap(str(reason)[:512])
            except BaseException:
                pass

    def _stage(self, kind, **facts):
        if not self._current():
            return False
        try:
            plain = _plain(facts)
            raw, _ = contract._bounded(plain)
            with self._lock:
                _require(self._events < contract.MAX_RECORDS - 2, "mapping event bound exceeded")
                self._events += 1
                if self._writer is None:
                    _require(len(self._buffer) < MAX_BUFFER_FACTS
                             and self._buffer_bytes + len(raw) <= MAX_BUFFER_BYTES,
                             "mapping pre-header buffer bound exceeded")
                    self._buffer.append((kind, plain))
                    self._buffer_bytes += len(raw)
                    return True
                _require(len(self._outbox) < MAX_OUTBOX_RECORDS
                         and self._outbox_bytes + len(raw) <= MAX_OUTBOX_BYTES,
                         "mapping journal admission buffer exceeds bounds")
                self._outbox.append((kind, plain, len(raw)))
                self._outbox_bytes += len(raw)
                return True
        except BaseException as exc:
            self._mark_error("mapping fact admission failed")
            if not isinstance(exc, Exception):
                raise
            return False

    def _flush(self):
        with self._lock:
            if self._writer is None or self._draining:
                return True
            owner = object()
            self._draining = owner
        accepted = True
        try:
            while True:
                with self._lock:
                    if not self._outbox:
                        self._draining = None
                        return accepted
                    kind, facts, size = self._outbox.popleft()
                try:
                    admitted = self._writer.emit(kind, **facts) is True
                    if not admitted:
                        self._mark_error("mapping journal rejected a fact")
                    accepted = admitted and accepted
                except BaseException as exc:
                    self._mark_error("mapping journal admission raised")
                    accepted = False
                    if not isinstance(exc, Exception):
                        raise
                finally:
                    with self._lock:
                        self._outbox_bytes -= size
        finally:
            with self._lock:
                if self._draining is owner:
                    self._draining = None
        return accepted

    def _event(self, kind, **facts):
        accepted = self._stage(kind, **facts)
        return self._flush() and accepted

    @_callback
    def _initialization_fact(self, fact):
        value = fact.record()
        kind = value.pop("kind")
        if kind == "init_begin":
            if self._initial_token is not None and value["init"] != self._initial_token:
                self._error("more than one initialization is outside this process interval")
            else:
                self._initial_token = value["init"]
        return self._event(kind, **value)

    def _binding(self, *, scheduler=None, runner=None):
        _require(self._current() and self.coverage(), "mapping observation coverage is unavailable")
        found = self.initialization.lookup(scheduler=scheduler, runner=runner)
        _require(found is not None, "mapping initialization association is unavailable")
        fact, refs = found
        recorded = fact.record()
        token = recorded["init"]
        _require(token == self._initial_token and (self._association is None or token == self._association),
                 "mapping cache association changed")
        if self._association is None:
            with self._lock:
                _require(self._current() and self._association is None and not self._building_header,
                         "mapping profile initialization overlaps another observation")
                self._building_header = True
            writer = None
            try:
                profile, dependencies = self._header_reader(self.registration, self.bindings, refs)
                header = dict(adapter=self.adapter, source_sha256=dict(self.source_hashes), profile=profile,
                              coverage=list(self.coverage_names), dependencies=dependencies,
                              process_incarnation=self.process, health=self.health)
                writer = self.sink_factory(**header)
                with self._lock:
                    active = self._current() and self._association is None
                    self._writer = writer
                    if active:
                        self._association, self._refs = token, refs
                        self._profile, self._dependencies = _plain(profile), _plain(dependencies)
                if not active:
                    self._mark_error("mapping profile completed after interval retirement")
                    try:
                        writer.close(timeout=0.0)
                    except BaseException:
                        self._mark_error("retired mapping journal cleanup failed")
                    raise MappingGap("mapping interval retired while its profile was recorded")
            finally:
                with self._lock:
                    self._building_header = False
            with self._lock:
                _require(len(self._buffer) <= MAX_OUTBOX_RECORDS
                         and self._buffer_bytes <= MAX_OUTBOX_BYTES,
                         "buffered initialization exceeds journal admission bounds")
                for kind, fields in self._buffer:
                    size = len(contract._bounded(fields)[0])
                    self._outbox.append((kind, fields, size))
                    self._outbox_bytes += size
                self._buffer.clear()
                self._buffer_bytes = 0
            _require(self._flush(), "mapping journal rejected buffered initialization")
        else:
            _require(self._refs["runner"]() is refs["runner"]()
                     and self._refs["pool"]() is refs["pool"](), "mapping participant lifetime changed")
            profile, dependencies = self._header_reader(self.registration, self.bindings, refs)
            _require(profile == self._profile and dependencies == self._dependencies,
                     "mapping effective profile changed")
        return recorded, refs

    def coverage(self):
        if not self._current():
            return False
        return self._coverage_slots()

    def _coverage_slots(self):
        if self.pid != os.getpid() or self._closed:
            return False
        try:
            return (self.initialization.sink is self._sink_callback
                    and self.initialization.health is self.health
                    and self.initialization.registry is self.registry
                    and self.initialization.coverage()
                    and bool(self._hooks)
                    and all(getattr(owner, name, _MISSING) is installed
                            for owner, name, original, installed, own in self._hooks)
                    and getattr(self._getter_owner, "get_blocks", _MISSING) is self._getter)
        except Exception:
            return False

    def _attach(self):
        b = self.bindings.initialization
        owners = {"BlockPool": b.pool, "SingleTypeKVCacheManager": self.bindings.manager_base,
                  "FullAttentionManager": b.group_manager}
        for label in self.coverage_names:
            owner_name, name = label.split(".")
            owner = owners[owner_name]
            _require(name in vars(owner) and callable(vars(owner)[name]), "mutation defining owner differs")
            self._install(owner, name, lambda fn, label=label: self._mutation_wrapper(label, fn))
        self._install(b.scheduler, "schedule", self._schedule_wrapper)

    def _install(self, owner, name, make):
        own = name in vars(owner)
        original = getattr(owner, name)
        _require(callable(original) and not hasattr(original, "_mapping_observer"), "mapping hook already installed")
        supplied = make(original)
        @functools.wraps(original)
        def installed(*args, **kwargs):
            if not self._admit():
                return original(*args, **kwargs)
            try:
                return supplied(*args, **kwargs)
            finally:
                self._release()
        installed._mapping_observer = weakref.ref(self)
        setattr(owner, name, installed)
        self._hooks.append((owner, name, original, installed, own))

    def _admit(self):
        if self.pid != os.getpid():
            return False
        exceeded = False
        with self._lock:
            if not self._current():
                return False
            if self._producers >= MAX_OPEN_MUTATIONS + 1:
                exceeded = True
            else:
                self._producers += 1
        if exceeded:
            self._error("mapping producer nesting bound exceeded")
        return not exceeded

    def _release(self):
        if self.pid == os.getpid():
            with self._lock:
                self._producers -= 1

    def _cleanup_hooks(self):
        for owner, name, original, installed, own in reversed(self._hooks):
            try:
                if getattr(owner, name, _MISSING) is installed:
                    if own:
                        setattr(owner, name, original)
                    else:
                        delattr(owner, name)
            except BaseException:
                self._mark_error("mapping hook cleanup failed")
        self._hooks.clear()
        if self.recipient is not None:
            self.recipient.safe(self.recipient.detach)

    def _receiver(self, method, receiver):
        if self._refs is None:
            _require(self._bridge_runner is not None and self._bridge_runner() is not None,
                     "allocation observation precedes a real runner initialization")
            self._binding(runner=self._bridge_runner())
        role = self.methods[method]
        if self._refs[role]() is not receiver:
            return False
        expected = self.bindings.initialization.pool if role == "pool" else self.bindings.initialization.group_manager
        _require(type(receiver) is expected, "mutation receiver has an unsupported class")
        return True

    def _stack(self):
        if not hasattr(self._local, "mutations"):
            self._local.mutations = []
        return self._local.mutations

    def _mutation_begin(self, label, receiver):
        if not self._receiver(label, receiver):
            return None
        _require(self.coverage(), "mapping mutation coverage changed")
        with self._lock:
            _require(len(self._mutations) < MAX_OPEN_MUTATIONS, "mapping mutation nesting exceeds bounds")
            token = self.registry.new_token()
            parent = self._stack()[-1] if self._stack() else None
            self._mutations[token] = label
            self._stack().append(token)
            self._epoch += 1
            self._stage("mutation_begin", association=self._association, mutation=token, method=label,
                        receiver=self.registry.identify(receiver, self.methods[label]), parent=parent)
        self._flush()
        return token

    def _mutation_end(self, token, success):
        with self._lock:
            _require(token in self._mutations and self._stack() and self._stack()[-1] == token,
                     "mapping mutation closure differs")
            self._mutations.pop(token)
            self._stack().pop()
            self._epoch += 1
            self._stage("mutation_end", association=self._association, mutation=token, success=success)
        self._flush()

    def _row_identity(self, receiver, name):
        mapping = receiver.req_to_blocks
        _require(type(mapping) in (dict, defaultdict) and contract._text(name), "unsupported native assignment map")
        row = mapping.get(name)
        _require(row is None or type(row) is list and len(row) <= contract.MAX_BLOCKS,
                 "unsupported native assignment row")
        return None if row is None else tuple(id(block) for block in row)

    def _guard_before(self, label, receiver, arguments):
        if label.endswith("._remove_blocks_in_range"):
            return self._row_identity(receiver, arguments["request_id"])
        if label.endswith(".cache_blocks"):
            request = arguments["request"]
            _require(type(request) is self.bindings.request, "cache request class differs")
            mapping = receiver.num_cached_block
            _require(type(mapping) is dict and contract._text(request.request_id), "cache counter is unavailable")
            present, count = request.request_id in mapping, mapping.get(request.request_id)
            _require(not present or contract._integer(count), "native cache counter is not an integer")
            return request.request_id, present, count
        if label.endswith("._cache_partial_tail_block"):
            _require(type(receiver.block_size) is int and receiver.block_size > 0
                     and type(receiver.block_pool.hash_block_size) is int
                     and receiver.block_size == receiver.block_pool.hash_block_size,
                     "partial-tail granularity is unsupported")
        return None

    def _guard_after(self, label, receiver, arguments, before, result):
        if label.endswith("._apply_cow"):
            return "the native CoW body completed"
        if label.endswith("._remove_blocks_in_range"):
            if self._row_identity(receiver, arguments["request_id"]) != before:
                return "the native skipped-range body replaced assignment entries"
        elif label.endswith(".cache_blocks"):
            name, present, count = before
            mapping = receiver.num_cached_block
            _require(type(mapping) is dict, "cache counter changed type")
            _require(name not in mapping or contract._integer(mapping.get(name)), "native cache counter changed type")
            if (name in mapping, mapping.get(name)) != (present, count):
                return "the native cache body registered cached blocks"
        elif label == "BlockPool.reset_prefix_cache":
            _require(type(result) is bool, "native reset outcome is not boolean")
            if result:
                return "the native prefix reset body completed"
        return None

    def _acquire(self, token, pool, result):
        _require(type(result) is list and len(result) <= contract.MAX_BLOCKS,
                 "native acquisition return is not a bounded concrete list")
        _require(type(pool.blocks) is list and len(pool.blocks) == pool.num_gpu_blocks,
                 "native pool table changed")
        indexes, seen = [], set()
        for block in result:
            index = block.block_id
            _require(contract._integer(index, 1, pool.num_gpu_blocks - 1) and index not in seen
                     and block.pool is pool and block.is_null is False and pool.blocks[index] is block,
                     "native acquisition block identity differs")
            indexes.append(index)
            seen.add(index)
        with self._lock:
            _require(self._allocation_observations + len(indexes) <= contract.MAX_BLOCKS,
                     "allocation observation bound exceeded")
            facts = []
            for index in indexes:
                ordinal = self._allocations.get(index, 0) + 1
                self._allocations[index] = ordinal
                facts.append(dict(block=index, incarnation=ordinal))
            self._allocation_observations += len(indexes)
            self._stage("acquire", association=self._association, mutation=token, blocks=facts)
        self._flush()
        return facts

    def _mutation_wrapper(self, label, original):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            if not self._current():
                return original(*args, **kwargs)
            token = recipient_call = None
            before = arguments = None
            receiver = args[0] if args else kwargs.get("self")
            try:
                token = self._mutation_begin(label, receiver)
                if token is not None:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = bound.arguments
                    before = self._guard_before(label, receiver, arguments)
                    if self.recipient is not None and label in (
                            "SingleTypeKVCacheManager.add_local_computed_blocks",
                            "SingleTypeKVCacheManager.allocate_external_computed_blocks",
                            "SingleTypeKVCacheManager.allocate_new_blocks"):
                        recipient_call = self.recipient.safe(self.recipient.phase_begin,
                                                            label, receiver, arguments, token)
            except Exception:
                self._error("native mutation entry observation is incomplete")
            try:
                result = original(*args, **kwargs)
            except BaseException as native_exception:
                if self.recipient is not None:
                    try:
                        self.recipient.safe(self.recipient.phase_end, recipient_call, None, native_exception)
                    except BaseException:
                        self.recipient.error("recipient exceptional lower closure was interrupted")
                if token is not None:
                    try:
                        self._mutation_end(token, False)
                    except BaseException:
                        self._error("failed native mutation closure is incomplete")
                raise
            if token is not None:
                try:
                    if label == "BlockPool.get_new_blocks":
                        acquired = self._acquire(token, receiver, result)
                        if self.recipient is not None:
                            if arguments is None:
                                self.recipient.error("recipient pool arguments were not recorded")
                            else:
                                self.recipient.safe(self.recipient.pool_return, token, receiver,
                                                    arguments["num_blocks"], result, acquired)
                    if arguments is not None:
                        reason = self._guard_after(label, receiver, arguments, before, result)
                        if reason is not None:
                            self._event("unsupported_branch", association=self._association,
                                        mutation=token, method=label, reason=reason)
                except BaseException as exc:
                    self._error("native mutation return observation is incomplete")
                    if not isinstance(exc, Exception):
                        try:
                            self._mutation_end(token, True)
                        except BaseException:
                            self._error("interrupted mutation closure is incomplete")
                        raise
                if self.recipient is not None:
                    self.recipient.safe(self.recipient.phase_end, recipient_call, result, None)
                try:
                    self._mutation_end(token, True)
                except Exception:
                    self._error("native mutation closure is incomplete")
            return result
        return wrapper

    def _request_facts(self, scheduler, output):
        values = output.num_scheduled_tokens
        _require(type(output) is self.bindings.scheduler_output and type(values) is dict
                 and len(values) <= contract.MAX_REQUESTS and type(scheduler.requests) is dict,
                 "native schedule or request inventory is unsupported")
        _require(contract._integer(output.total_num_scheduled_tokens)
                 and sum(values.values()) == output.total_num_scheduled_tokens,
                 "native scheduled total differs")
        requests, facts = {}, []
        for name, count in values.items():
            _require(contract._text(name) and contract._integer(count, 1, contract.MAX_CONTEXT),
                     "native scheduled request dimensions differ")
            request = scheduler.requests.get(name)
            _require(type(request) is self.bindings.request and request.request_id == name
                     and request.resumable is False and request.streaming_queue is None,
                     "native request lifetime is unavailable or streaming")
            token = self.registry.identify(request, "request")
            _require(self._names.get(name, token) == token, "another request lifetime reuses a recorded name")
            self._names[name] = token
            requests[name] = token, weakref.ref(request), count
            facts.append(dict(request=name, incarnation=token, query_tokens=count))
        return requests, facts

    def _schedule_wrapper(self, original):
        @functools.wraps(original)
        def wrapper(scheduler, *args, **kwargs):
            if not self._current():
                return original(scheduler, *args, **kwargs)
            with self._lock:
                overlap = bool(self._schedules_open)
                self._schedules_open += 1
            if overlap:
                self._error("native scheduler invocations overlap")
            try:
                ready = False
                try:
                    self._binding(scheduler=scheduler)
                    ready = True
                except Exception:
                    self._error("schedule initialization binding is incomplete")
                try:
                    result = original(scheduler, *args, **kwargs)
                except BaseException:
                    self._error("native schedule did not return a complete schedule object")
                    raise
                if ready:
                    try:
                        recorded, refs = self._binding(scheduler=scheduler)
                        requests, facts = self._request_facts(scheduler, result)
                        token = self.registry.identify(result, "schedule_output")
                        step = scheduler.current_step
                        _require(token not in self._dispatches and len(self._dispatches) < MAX_DISPATCHES
                                 and contract._integer(step, 1)
                                 and step > self._steps.get(recorded["objects"]["scheduler"], 0),
                                 "native schedule output lifetime or step differs")
                        _require(self._request_observations + len(requests) <= MAX_REQUEST_OBSERVATIONS,
                                 "request observation bound exceeded")
                        self._request_observations += len(requests)
                        self._steps[recorded["objects"]["scheduler"]] = step
                        self._dispatches[token] = _Dispatch(weakref.ref(result), refs["runner"], requests, facts)
                        self._event("schedule", association=self._association, core=recorded["objects"]["core"],
                                    core_scheduler=recorded["objects"]["scheduler"], scheduler=recorded["objects"]["scheduler"],
                                    dispatch=token, step=step, requests=facts)
                    except Exception:
                        self._error("schedule return observation is incomplete")
                return result
            finally:
                if self.pid == os.getpid():
                    with self._lock:
                        self._schedules_open -= 1
        return wrapper

    @_callback
    def runner_initialized(self, boundary):
        self.initialization.runner_initialized(boundary)
        _require(self._current() and type(boundary.runner) is self.bindings.initialization.runner,
                 "bridge runner initialization differs")
        _require(contract._text(boundary.cache_label, 128), "bridge cache label is unavailable")
        self._bridge_runner = weakref.ref(boundary.runner)
        self._bridge_config = weakref.ref(boundary.runner_config)
        self._bridge_tables = weakref.ref(boundary.block_tables)
        self._bridge_label = boundary.cache_label

    @_callback
    def invocation_enter(self, boundary):
        recorded, refs = self._binding(runner=boundary.runner)
        token = self.registry.identify(boundary.scheduler_output, "schedule_output")
        call = self._dispatches.get(token)
        _require(call is not None and call.output() is boundary.scheduler_output
                 and call.runner() is boundary.runner and call.requests and not call.entered,
                 "runner invocation has no unique positive native schedule")
        requests, facts = self._request_facts(refs["scheduler"](), boundary.scheduler_output)
        _require(facts == call.facts and all(requests[name][1]() is value[1]() for name, value in call.requests.items()),
                 "native request identity changed before runner entry")
        runner_token = recorded["objects"]["runner"]
        _require(runner_token not in self._pending, "runner has an unfinished direct invocation")
        invocation = self.registry.new_token()
        call.entered, call.invocation = True, invocation
        self._invocations[invocation] = token
        self._pending[runner_token] = invocation
        self._event("worker_entry", association=self._association, dispatch=token,
                    runner=runner_token, requests=facts)
        return invocation

    @_callback
    def prepared(self, token, rows):
        call = self._call(token)
        _require(call.prepared is None and type(rows.requests) is tuple and rows.requests
                 and len(set(rows.requests)) == len(rows.requests) and set(rows.requests) == set(call.requests)
                 and contract._integer(rows.num_tokens, 1, contract.MAX_CONTEXT)
                 and rows.num_tokens == sum(value[2] for value in call.requests.values()),
                 "prepared rows differ from the native schedule")
        call.prepared = rows.requests, rows.num_tokens

    def _requests_current(self, call, refs):
        output = call.output()
        _require(output is not None, "native schedule output was collected during its invocation")
        requests, facts = self._request_facts(refs["scheduler"](), output)
        _require(facts == call.facts
                 and all(requests[name][1]() is value[1]() for name, value in call.requests.items()),
                 "native request lifetime changed at the copy boundary")

    def _call(self, token):
        _require(self._current() and token in self._invocations, "mapping invocation is unavailable")
        call = self._dispatches[self._invocations[token]]
        _require(not call.left, "mapping invocation is already complete")
        return call

    def _allocator_rows(self, call, refs):
        coordinator, pool = refs["coordinator"](), refs["pool"]()
        getter = coordinator.get_blocks
        _require(_same_method(getter, self._getter, coordinator), "ordered allocator getter changed")
        rows = []
        for name in call.prepared[0]:
            groups = getter(name)
            _require(type(groups) is tuple and len(groups) == 1 and type(groups[0]) is list
                     and len(groups[0]) <= contract.MAX_BLOCKS, "ordered allocator getter shape differs")
            self._cells += len(groups[0])
            _require(self._cells <= contract.MAX_CELLS, "assignment snapshot cell bound exceeded")
            blocks = []
            unavailable = False
            for block in groups[0]:
                index = block.block_id
                _require(contract._integer(index, 0, pool.num_gpu_blocks - 1)
                         and block.pool is pool and pool.blocks[index] is block,
                         "ordered allocator block belongs to another pool")
                ordinal = self._allocations.get(index)
                if ordinal is None:
                    unavailable = True
                else:
                    blocks.append(dict(block=index, incarnation=ordinal))
            rows.append(dict(request=name, incarnation=call.requests[name][0],
                             blocks=None if unavailable else blocks))
        return rows

    @_callback
    def begin_copy(self, token, boundary):
        call = self._call(token)
        recorded, refs = self._binding(runner=boundary.runner)
        self._requests_current(call, refs)
        runner_token = recorded["objects"]["runner"]
        _require(self._pending.get(runner_token) == token and call.runner() is boundary.runner
                 and call.frame is None and call.prepared == (boundary.prepared.requests, boundary.prepared.num_tokens)
                 and self._bridge_runner is not None and self._bridge_runner() is boundary.runner
                 and self._bridge_config() is boundary.runner.kv_cache_config
                 and self._bridge_tables() is boundary.runner.block_tables and boundary.cache_label == self._bridge_label
                 and contract._integer(boundary.forward_id, 1), "copy boundary is not the current native invocation")
        _require(len(self._frames) < contract.MAX_FRAMES, "mapping frame bound exceeded")
        frame = self.registry.new_token()
        call.frame = frame
        self._frames[frame] = dict(invocation=token, forward_id=boundary.forward_id, ended=False)
        self._event("frame_begin", association=self._association, dispatch=self._invocations[token],
                    runner=runner_token, frame=frame, epoch=self._epoch, coverage=self.coverage_names)
        rows = self._allocator_rows(call, refs)
        self._event("assignments", frame=frame, manager=recorded["objects"]["manager"],
                    pool=recorded["objects"]["pool"], rows=rows)
        return frame

    def _model_entry(self, call, frame, snapshot):
        value = _plain(snapshot.record)
        contract._bounded(value)
        _require(value["observation_point"] == "model_entry" and value["snapshot_copy_complete"] is True
                 and value["forward_device_complete"] is None
                 and contract._integer(value["forward_id"], 1) and value["forward_id"] == frame["forward_id"]
                 and value["pool_incarnation"] == self._bridge_label
                 and tuple(value["requests"]) == call.prepared[0]
                 and contract._integer(value["num_tokens"], 1) and value["num_tokens"] == call.prepared[1],
                 "immutable model-entry copy identity differs")
        expected_names = self._profile["layer_names"]
        _require(type(value["layers"]) is list and len(value["layers"]) == len(expected_names),
                 "model-entry layer coverage differs")
        layers, lengths, queries, seen = [], None, None, set()
        for layer in value["layers"]:
            name = layer["name"]
            _require(type(name) is str and name in expected_names and name not in seen,
                     "model-entry layer identity differs")
            seen.add(name)
            _require(type(layer["block_size"]) is int and layer["block_size"] == self._binding_dimensions()["block_size"]
                     and type(layer["pool_num_blocks"]) is int and layer["pool_num_blocks"] == self._binding_dimensions()["capacity"]
                     and layer["causal"] is True and layer["window"] is None
                     and layer["layout"] == self._profile["layout"] and layer["backend"] == self._profile["backend"]
                     and layer["backend_version"] == self._profile["backend_version"],
                     "model-entry attention interpretation differs")
            sequence, starts, tables = layer["seq_lens"], layer["query_start_loc"], layer["block_table"]
            n = len(call.prepared[0])
            _require(type(sequence) is list and len(sequence) == n and all(contract._integer(v, 1, contract.MAX_CONTEXT) for v in sequence)
                     and type(starts) is list and len(starts) == n + 1 and starts[0] == 0
                     and all(contract._integer(v) for v in starts) and starts[-1] == call.prepared[1]
                     and type(tables) is list and len(tables) == n, "model-entry row dimensions differ")
            current_queries = [b - a for a, b in zip(starts, starts[1:])]
            _require(all(0 < q <= length and q == call.requests[name][2]
                         for name, q, length in zip(call.prepared[0], current_queries, sequence)),
                     "model-entry query identity differs")
            if lengths is None:
                lengths, queries = list(sequence), current_queries
            _require(sequence == lengths and current_queries == queries, "layer context or query lengths differ")
            width = self._binding_dimensions()["block_size"]
            copied = []
            for row, length in zip(tables, sequence):
                _require(type(row) is list and len(row) == (length + width - 1) // width
                         and all(contract._integer(b, 0, self._binding_dimensions()["capacity"] - 1) for b in row),
                         "model-entry context table is not clipped or bounded")
                self._cells += len(row)
                _require(self._cells <= contract.MAX_CELLS, "model-entry cell bound exceeded")
                copied.append(list(row))
            layers.append(dict(name=name, rows=copied))
        return dict(requests=list(call.prepared[0]), seq_lens=lengths, query_lengths=queries, layers=layers,
                    semantics=dict(observation_point="model_entry", causal=True, window=None,
                                   layout=self._profile["layout"], backend=self._profile["backend"]), copy_complete=True)

    def _binding_dimensions(self):
        return self._writer_profile_dimensions

    @_callback
    def end_copy(self, frame_token, snapshot):
        frame = self._frames[frame_token]
        call = self._call(frame["invocation"])
        recorded, refs = self._binding(runner=call.runner())
        self._requests_current(call, refs)
        self._writer_profile_dimensions = recorded["dimensions"]
        _require(not frame["ended"] and self._pending.get(recorded["objects"]["runner"]) == frame["invocation"],
                 "mapping frame is no longer current")
        model = self._model_entry(call, frame, snapshot)
        accepted = self._event("model_entry", frame=frame_token, runner=recorded["objects"]["runner"],
                               worker_pool=recorded["objects"]["worker_cache"], **model)
        accepted = self._event("frame_end", frame=frame_token, epoch=self._epoch,
                               coverage=self.coverage_names) and accepted
        frame["ended"], call.copied = True, accepted
        return accepted

    @_callback
    def abandon_copy(self, token, frame_token, reason):
        self._error("positive invocation has an incomplete model-entry copy")
        if frame_token in self._frames and not self._frames[frame_token]["ended"]:
            self._event("frame_end", frame=frame_token, epoch=self._epoch, coverage=self.coverage_names)
            self._frames[frame_token]["ended"] = True

    @_callback
    def invocation_leave(self, token, outcome):
        call = self._call(token)
        _require(type(outcome.returned) is bool and type(outcome.eager_copy) is bool and type(outcome.poisoned) is bool,
                 "native invocation outcome is incomplete")
        if not outcome.returned or outcome.poisoned or not outcome.eager_copy or not call.copied:
            self._error("positive native invocation did not complete its mapping copy")
        call.left = True
        runner = call.runner()
        if runner is not None:
            self._pending.pop(self.registry.identify(runner, "runner"), None)

    @_callback
    def retire_runner(self, runner, reason):
        if not self._current():
            return
        self.initialization.retire_runner(runner, reason)
        if self._association is not None and self._refs["runner"]() is runner:
            self._event("retire", association=self._association,
                        reason="shutdown" if reason == "native shutdown" else "cache_replacement")

    def runner_closed(self, runner, exception_type):
        if os.getpid() != self.pid:
            return False
        if self._closed:
            return self._sealed
        if not self._admit():
            return False
        try:
            if self._refs is None or self._refs["runner"]() is not runner:
                self._error("closed runner has no observed mapping association")
            if exception_type is not None:
                self._error("native shutdown raised before recording closure")
        finally:
            self._release()
        return self.close(timeout=1.0)

    def close(self, *, timeout=1.0):
        if os.getpid() != self.pid:
            return False
        with self._lock:
            if self._closed:
                return self._sealed
            if self._closing:
                return False
            self._closing = True
            incomplete = (self._producers or self._draining or self._building_header or self._outbox
                          or self._schedules_open or self._mutations
                          or any(not frame["ended"] for frame in self._frames.values())
                          or any(call.requests and not (call.entered and call.left and call.copied)
                                 for call in self._dispatches.values()))
        if not self._coverage_slots():
            self._error("mapping coverage was lost before interval closure")
        if incomplete:
            self._error("mapping interval closed with unfinished positive work or mutations")
        if self.recipient is not None:
            self.recipient.safe(self.recipient.close_interval)
        self._cleanup_hooks()
        try:
            self.initialization.detach()
        except BaseException:
            self._error("mapping initialization cleanup failed")
        result = False
        if self._writer is None:
            self._mark_error("no supported actual profile established a mapping journal")
        else:
            try:
                result = self._writer.close(timeout=timeout) is True
                status = self._writer.status
                result = (result and status.get("published") is True and status.get("healthy") is True
                          and status.get("closed") is True and status.get("sealed") is True
                          and status.get("footer_written") is True and not self._errors)
                result = result and self.health.snapshot()["healthy"] is True
            except BaseException:
                self._mark_error("mapping journal closure failed")
        self._sealed, self._closed = result, True
        if self._closure_callback is not None:
            try:
                self._closure_receipt = self._closure_callback(dict(
                    close_succeeded=result, identity=dict(adapter=self.adapter,
                        source_sha256=dict(self.source_hashes), profile=_plain(self._profile),
                        dependencies=_plain(self._dependencies), process=self.process),
                    observer=self.status)) is True
            except BaseException:
                # Storage acknowledgement is separate from the completed
                # interval. Failed receipt publication is never retried.
                self._closure_receipt = False
        return result

    @property
    def status(self):
        if os.getpid() != self.pid:
            return dict(healthy=False, process="changed", runtime_isolation="not established")
        journal = None if self._writer is None else self._writer.status
        healthy = (not self._errors and self.health.snapshot()["healthy"] is True
                   and (journal is None or journal.get("healthy") is True)
                   and (self._sealed if self._closed else self.coverage()))
        artifacts = None if self._writer is None else {
            key: str(value) for key in ("path", "incomplete_path")
            if (value := getattr(self._writer, key, None)) is not None}
        return dict(healthy=healthy, errors=self._errors,
                    closed=self._closed, sealed=self._sealed, events=self._events,
                    positive_dispatches=sum(bool(call.requests) for call in self._dispatches.values()),
                    journal=journal, artifacts=artifacts, closure_receipt=self._closure_receipt,
                    export_ready=self._closure_receipt is True and self._sealed and healthy,
                    ownership="unknown", physical_reads="unknown", content_provenance="unknown",
                    runtime_isolation="not established")


def mapping_factory(directory, source_root, *, sink_factory=None, closure_callback=None, recipient_factory=None):
    """Return an explicit bridge factory. Merely creating it imports no engine."""
    def factory(registration, health):
        from .mapping_runtime import load_bindings
        bindings = load_bindings(source_root, registration.engine["version"], registration)
        selected = sink_factory
        if selected is None:
            from .assignment_recording import AssignmentRecorder
            selected = lambda **fields: AssignmentRecorder(directory, **fields)
        return MappingObserver(registration, health, bindings=bindings, source_root=source_root,
                               sink_factory=selected, closure_callback=closure_callback,
                               recipient_factory=recipient_factory)
    return factory
