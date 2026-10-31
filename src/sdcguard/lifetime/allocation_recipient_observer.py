"""Passive positive allocation-recipient facts for one explicit mapping interval.

Owner: Hari Krishna Kancharla.
No engine, traffic, device copy or permission check is activated by this module.
A separate settled mapping capture is mandatory before these facts can be judged.
"""
from __future__ import annotations

import __future__
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
from types import CodeType
import weakref

from . import allocation_recipient_obligations as contract
from . import vllm_nixl


OUTER = "KVCacheManager.allocate_slots"
COMPUTED = "KVCacheCoordinator.allocate_new_computed_blocks"
COORDINATED = "KVCacheCoordinator.allocate_new_blocks"
LOCAL = "SingleTypeKVCacheManager.add_local_computed_blocks"
EXTERNAL = "SingleTypeKVCacheManager.allocate_external_computed_blocks"
ORDINARY = "SingleTypeKVCacheManager.allocate_new_blocks"
METHODS = (OUTER, COMPUTED, COORDINATED, LOCAL, EXTERNAL, ORDINARY)
_PATHS = {"KVCacheManager": "vllm/v1/core/kv_cache_manager.py",
          "KVCacheCoordinator": "vllm/v1/core/kv_cache_coordinator.py",
          "SingleTypeKVCacheManager": "vllm/v1/core/single_type_kv_cache_manager.py",
          "FullAttentionManager": "vllm/v1/core/single_type_kv_cache_manager.py",
          "BlockPool": "vllm/v1/core/block_pool.py",
          "Scheduler": "vllm/v1/core/sched/scheduler.py"}
MAX_OUTER_CALLS = 1024
MAX_PHASE_CALLS = 4096
MAX_NESTING = 8
MAX_RETURN_BLOCKS = 512
MAX_BLOCK_FACTS = 8192
_MISSING = object()


class RecipientGap(ValueError):
    pass


def _require(value, reason):
    if not value:
        raise RecipientGap(reason)


def _integer(value, low=0, high=2**31 - 1):
    return type(value) is int and low <= value <= high


def _code_value(value):
    if type(value) is CodeType:
        return dict(code=value.co_code.hex(), consts=[_code_value(v) for v in value.co_consts],
                    names=list(value.co_names), varnames=list(value.co_varnames),
                    freevars=list(value.co_freevars), cellvars=list(value.co_cellvars),
                    argcount=value.co_argcount, posonly=value.co_posonlyargcount,
                    kwonly=value.co_kwonlyargcount, flags=value.co_flags & ~__future__.annotations.compiler_flag,
                    exceptiontable=getattr(value, "co_exceptiontable", b"").hex())
    if type(value) is tuple:
        return {"tuple": [_code_value(v) for v in value]}
    if type(value) is bytes:
        return {"bytes": value.hex()}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if value is Ellipsis:
        return {"ellipsis": True}
    raise RecipientGap("native code constant is unsupported")


def _fingerprint(code):
    return hashlib.sha256(json.dumps(_code_value(code), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _expected_code(source_root, label):
    owner, name = label.split(".")
    path = Path(source_root) / _PATHS[owner]
    # Compilation creates code objects only. Engine modules are never executed.
    from .. import config_inputs
    raw = config_inputs._read_regular(Path(source_root).absolute(), Path(_PATHS[owner]).parts, 2 * 1024 * 1024)
    _require(hashlib.sha256(raw).hexdigest() == contract.source_pins()[_PATHS[owner]],
             "recipient native source pin differs")
    module = compile(raw, str(path), "exec", dont_inherit=True)
    classes = [item for item in module.co_consts if type(item) is CodeType and item.co_name == owner]
    _require(len(classes) == 1, "native source defining class is ambiguous")
    functions = [item for item in classes[0].co_consts if type(item) is CodeType and item.co_name == name]
    _require(len(functions) == 1, "native source method is ambiguous")
    return _fingerprint(functions[0])


def _function_state(function):
    """Freeze the actual code and closed delegation, without calling it."""
    _require(inspect.isfunction(function), "recipient chain contains a non-Python callable")
    cells = function.__closure__ or ()
    _require(len(cells) == len(function.__code__.co_freevars), "recipient closure shape differs")
    return (function, function.__code__, tuple((cell, cell.cell_contents) for cell in cells),
            getattr(function, "__wrapped__", _MISSING))


def _function_matches(state):
    function, code, captured, wrapped = state
    cells = function.__closure__ or ()
    return (function.__code__ is code and len(cells) == len(captured)
            and all(cell is before and cell.cell_contents is value
                    for cell, (before, value) in zip(cells, captured))
            and getattr(function, "__wrapped__", _MISSING) is wrapped)


class AllocationRecipientObserver:
    """An off-by-default extension of the already bound mapping observer.

    Native objects are held only through existing weak identities or during a
    currently executing call. Stored facts contain no native objects or content.
    """

    def __init__(self, mapping):
        self.mapping = weakref.ref(mapping)
        self.pid = os.getpid()
        self.session = mapping.registry.new_token()
        self.records = []
        self._lock = threading.RLock()
        self._local = threading.local()
        self._open = {}
        self._names = {}
        self._codes = {}
        self._slots = {}
        self._compositions = []
        self._chains = None
        self._native_sources = None
        self._header = None
        self._closed = False
        self._source_stable = self._coverage_stable = True
        self._errors = self._dropped = 0
        self._outer_calls = self._phase_calls = self._pool_returns = self._block_facts = 0
        self._empty_returns = 0
        self._exported = False
        self._export_attempted = False
        self._export_in_progress = False
        self._body_bytes = self._body_nodes = 0

    def _mapping(self):
        value = self.mapping()
        _require(value is not None and self.pid == os.getpid(), "recipient mapping lifetime or process changed")
        return value

    def _stack(self):
        if not hasattr(self._local, "calls"):
            self._local.calls = []
        return self._local.calls

    def error(self, reason):
        if self.pid != os.getpid():
            return
        self._errors += 1
        if self._header is not None and not self._closed:
            try:
                self._emit("gap", call=self._stack()[-1] if self._stack() else None, reason=str(reason)[:256])
            except Exception:
                pass

    def safe(self, function, *args):
        try:
            return function(*args)
        except Exception:
            self.error("recipient " + function.__name__ + " observation is incomplete")
            return None

    def _emit(self, kind, **facts):
        try:
            return self._admit_record(kind, **facts)
        except Exception:
            with self._lock:
                self._dropped += 1
            raise

    def _admit_record(self, kind, **facts):
        with self._lock:
            _require(not self._closed and self.pid == os.getpid(), "recipient interval is closed or belongs to another process")
            record = dict(schema=1, session=self.session, process=self._mapping().process,
                          seq=len(self.records), kind=kind, **facts)
            raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            copied = json.loads(raw)
            _require(len(raw) + 1 <= contract.MAX_RECORD_BYTES and len(self.records) < contract.MAX_RECORDS - 1,
                     "recipient record admission bound exceeded")
            measured = contract.measure([copied])
            _require(measured["body_nodes"] <= contract.MAX_RECORD_NODES, "recipient record node bound exceeded")
            _require(self._body_bytes + measured["body_bytes"] + len(self.records) + 2 <= contract.MAX_TOTAL_BYTES - 8192
                     and self._body_nodes + measured["body_nodes"] <= contract.MAX_TOTAL_NODES - 256,
                     "recipient aggregate admission bound exceeded")
            self.records.append(copied)
            self._body_bytes += measured["body_bytes"]
            self._body_nodes += measured["body_nodes"]

    def attach(self):
        mapping = self._mapping()
        try:
            self._native_sources = self._check_native_sources(mapping)
            self._attach_checked()
        except BaseException:
            self._source_stable = self._coverage_stable = False
            self.error("recipient native source or hook setup is incomplete")
            try:
                mapping._cleanup_hooks()
            except BaseException:
                self.error("recipient failed-attachment cleanup is incomplete")
            raise

    def _check_native_sources(self, mapping):
        """Check all defining primary functions before adopting loaded code."""
        b = mapping.bindings.initialization
        candidates = {}
        def add(owner, name, function):
            label = owner.__name__ + "." + name
            _require(label not in candidates, "recipient source original is repeated")
            expected = _expected_code(mapping.source_root, label)
            _require(inspect.isfunction(function) and not hasattr(function, "__wrapped__")
                     and _fingerprint(function.__code__) == expected,
                     "recipient native original differs from its pinned defining function: " + label)
            names = function.__code__.co_freevars
            cells = function.__closure__ or ()
            _require((not names and not cells) or (
                names == ("__class__",) and len(cells) == 1 and cells[0].cell_contents is owner),
                "recipient native defining-class closure differs: " + label)
            candidates[label] = (owner, name, function, expected)
        for owner, name, original, installed, own in mapping._hooks:
            defining = owner if own else next(cls for cls in owner.__mro__[1:]
                if vars(cls).get(name) is original)
            add(defining, name, original)
        for owner, name in ((b.manager, "allocate_slots"), (b.coordinator, "allocate_new_computed_blocks"),
                            (b.coordinator, "allocate_new_blocks"), (b.manager, "create_kv_cache_blocks"),
                            (mapping.bindings.manager_base, "get_num_skipped_tokens")):
            defining = next(cls for cls in owner.__mro__ if name in vars(cls))
            function = vars(defining)[name]
            if name == "allocate_slots":
                tracked = vllm_nixl._INSTALLED_HOOKS.get(defining, {}).get(name)
                if tracked is not None:
                    cells = vllm_nixl._closure(function)
                    maker = vllm_nixl._nested_code(vllm_nixl.install_manager, "mk_alloc")
                    codes = [code for code in maker.co_consts if type(code) is CodeType and code.co_name == "alloc"]
                    _require(tracked[0] is function and len(codes) == 1 and function.__code__ is codes[0]
                             and cells is not None and inspect.isfunction(cells.get("fn"))
                             and getattr(function, "__wrapped__", None) is cells["fn"],
                             "recipient initial lifetime chain is unsupported")
                    function = cells["fn"]
            add(defining, name, function)
        add(mapping._getter_owner, "get_blocks", mapping._getter)
        required = set(mapping.methods) | set(METHODS) | {"Scheduler.schedule",
            "KVCacheManager.create_kv_cache_blocks", "SingleTypeKVCacheManager.get_num_skipped_tokens",
            "KVCacheCoordinator.get_blocks"}
        _require(set(candidates) == required and len(candidates) == 22,
                 "recipient complete pinned native-original inventory differs")
        return candidates

    def _attach_checked(self):
        mapping = self._mapping()
        b = mapping.bindings.initialization
        for label in (OUTER, COMPUTED, COORDINATED):
            owner = b.manager if label == OUTER else next(
                cls for cls in b.coordinator.__mro__ if label.split(".")[1] in vars(cls))
            mapping._install(owner, label.split(".")[1],
                             lambda function, label=label: self.wrap(label, function))
        for label in METHODS + ("BlockPool.get_new_blocks",):
            name = label.split(".")[1]
            expected = _expected_code(mapping.source_root, label)
            matches = [hook for hook in mapping._hooks if hook[1] == name and (
                hook[0] is b.manager if label == OUTER else
                hook[0] is b.pool if label.startswith("BlockPool.") else
                hook[0] is mapping.bindings.manager_base if label.startswith("SingleType") else
                hook[0].__name__ == "KVCacheCoordinator")]
            _require(len(matches) == 1, "recipient installed callable binding is ambiguous")
            owner, name, original, installed, _ = matches[0]
            leaf = original
            if label == OUTER:
                tracked = vllm_nixl._INSTALLED_HOOKS.get(owner, {}).get(name)
                if tracked is not None:
                    _require(tracked[0] is original, "recipient lifetime wrapper is not the exact tracked callable")
                    leaf = getattr(original, "__wrapped__", None)
                    _require(inspect.isfunction(leaf), "recipient lifetime native leaf is unavailable")
                    vllm_nixl.register_composition(owner, name, original, installed)
                    self._compositions.append((owner, name, original, installed))
            _require(inspect.isfunction(leaf) and _fingerprint(leaf.__code__) == expected,
                     "recipient native method code differs from the pinned complete function")
            self._codes[label] = (leaf, expected)
            self._slots[label] = (owner, name, installed)
        for label, owner in (("KVCacheManager.create_kv_cache_blocks", b.manager),
                             ("SingleTypeKVCacheManager.get_num_skipped_tokens", mapping.bindings.manager_base)):
            function = vars(owner).get(label.split(".")[1])
            expected = _expected_code(mapping.source_root, label)
            _require(inspect.isfunction(function) and _fingerprint(function.__code__) == expected,
                     "recipient native source helper differs")
            self._codes[label] = (function, expected)
            self._slots[label] = (owner, label.split(".")[1], function)
        for label, (owner, name, function, expected) in self._native_sources.items():
            if label not in self._codes:
                hooks = [hook for hook in mapping._hooks if hook[0] is owner and hook[1] == name]
                installed = hooks[0][3] if len(hooks) == 1 else function
                self._codes[label] = (function, expected)
                self._slots[label] = (owner, name, installed)
        self._capture_chains()
        _require(self.coverage(), "recipient source-checked hook inventory is not covered")

    def _capture_chains(self):
        """Admit only exact reviewed templates once, then retain their identities.

        The first coverage call also supports detached helper controls that do
        not initialize an engine. Real attachment freezes every chain here at
        setup. No lost chain can be recaptured after its first observation.
        """
        from .mapping_observer import MappingObserver
        _require(self._coverage_stable and self._chains is None and bool(self._slots)
                 and set(self._slots) == set(self._codes),
                 "recipient chain capture is missing, repeated or incomplete")
        mapping = self._mapping()
        hooks = tuple(mapping._hooks)
        installed_code = vllm_nixl._nested_code(MappingObserver._install, "installed")
        recipient_code = vllm_nixl._nested_code(AllocationRecipientObserver.wrap, "observed")
        mutation_code = vllm_nixl._nested_code(MappingObserver._mutation_wrapper, "wrapper")
        schedule_code = vllm_nixl._nested_code(MappingObserver._schedule_wrapper, "wrapper")
        by_slot, states, dynamic_recipient = {}, [], False
        for hook in hooks:
            _require(type(hook) is tuple and len(hook) == 5, "recipient mapping hook shape differs")
            owner, name, original, installed, own = hook
            key = (owner, name)
            _require(key not in by_slot and type(own) is bool and inspect.isfunction(original)
                     and inspect.isfunction(installed) and vars(owner).get(name, _MISSING) is installed,
                     "recipient exact defining hook slot is missing or repeated")
            _require(installed.__code__ is installed_code, "recipient installed wrapper code is unsupported")
            cells = vllm_nixl._closure(installed)
            _require(set(cells) == {"original", "self", "supplied"}
                     and cells["original"] is original and cells["self"] is mapping
                     and getattr(installed, "__wrapped__", _MISSING) is original,
                     "recipient installed wrapper delegation differs")
            supplied = cells["supplied"]
            _require(inspect.isfunction(supplied), "recipient supplied wrapper is unavailable")
            supplied_cells = vllm_nixl._closure(supplied)
            labels = [label for label, slot in self._slots.items()
                      if slot[0] is owner and slot[1] == name and slot[2] is installed]
            if labels and labels[0] in (OUTER, COMPUTED, COORDINATED):
                _require(len(labels) == 1 and supplied.__code__ is recipient_code
                         and set(supplied_cells) == {"label", "original", "self", "signature"}
                         and supplied_cells["label"] == labels[0] and supplied_cells["self"] is self,
                         "recipient outer or coordinator wrapper is unsupported")
            elif supplied.__code__ is mutation_code:
                label = supplied_cells.get("label")
                _require(set(supplied_cells) == {"label", "original", "self", "signature"}
                         and type(label) is str and label in mapping.methods
                         and label.split(".")[1] == name and supplied_cells["self"] is mapping
                         and mapping.recipient is self,
                         "recipient lower mutation wrapper is unsupported")
                dynamic_recipient = True
            else:
                _require(name == "schedule" and supplied.__code__ is schedule_code
                         and set(supplied_cells) == {"original", "self"}
                         and supplied_cells["self"] is mapping,
                         "recipient scheduler wrapper is unsupported")
            _require(supplied_cells.get("original") is original
                     and getattr(supplied, "__wrapped__", _MISSING) is original,
                     "recipient supplied wrapper native delegation differs")
            if "signature" in supplied_cells:
                _require(type(supplied_cells["signature"]) is inspect.Signature
                         and supplied_cells["signature"] == inspect.signature(original),
                         "recipient supplied wrapper signature differs from its native delegate")
            known = [entry for entry in self._compositions if entry[0] is owner and entry[1] == name]
            if known:
                _require(len(known) == 1 and known[0][2] is original and known[0][3] is installed
                         and vllm_nixl._composition_matches(*known[0]),
                         "recipient known lifetime delegation differs")
                native = vllm_nixl._closure(original)["fn"]
                _require(self._codes[OUTER][0] is native, "recipient lifetime native leaf differs")
                states.append(_function_state(native))
            else:
                _require(not hasattr(original, "__wrapped__"), "recipient native leaf has an unknown wrapper")
                if labels:
                    _require(len(labels) == 1 and self._codes[labels[0]][0] is original,
                             "recipient direct native leaf differs")
            states.extend((_function_state(installed), _function_state(supplied), _function_state(original)))
            by_slot[key] = hook
        for label, (owner, name, function) in self._slots.items():
            _require(vars(owner).get(name, _MISSING) is function, "recipient defining callable slot differs")
            if (owner, name) not in by_slot:
                _require(self._codes[label][0] is function and not hasattr(function, "__wrapped__"),
                         "recipient direct helper delegation differs")
                states.append(_function_state(function))
        getter = None
        if hasattr(mapping, "_getter_owner") or hasattr(mapping, "_getter"):
            owner, function = mapping._getter_owner, mapping._getter
            _require(vars(owner).get("get_blocks", _MISSING) is function
                     and not hasattr(function, "__wrapped__"), "recipient block getter delegation differs")
            getter = (owner, function, _function_state(function))
        self._chains = dict(hooks=hooks, states=tuple(states), slots=dict(self._slots), codes=dict(self._codes),
                            getter=getter, dynamic_recipient=dynamic_recipient,
                            compositions=tuple(self._compositions))

    def _chains_match(self):
        if self._chains is None:
            self._capture_chains()
        mapping, captured = self._mapping(), self._chains
        current = mapping._hooks
        return (len(current) == len(captured["hooks"])
            and all(type(now) is tuple and len(now) == 5
                    and all(a is b for a, b in zip(now, before))
                    and vars(now[0]).get(now[1], _MISSING) is now[3]
                    for now, before in zip(current, captured["hooks"]))
            and set(self._slots) == set(captured["slots"]) and set(self._codes) == set(captured["codes"])
            and all(self._slots[label][0] is value[0] and self._slots[label][1] == value[1]
                    and self._slots[label][2] is value[2]
                    and vars(value[0]).get(value[1], _MISSING) is value[2]
                    for label, value in captured["slots"].items())
            and all(self._codes[label][0] is value[0] and self._codes[label][1] == value[1]
                    for label, value in captured["codes"].items())
            and all(_function_matches(state) for state in captured["states"])
            and (not captured["dynamic_recipient"] or mapping.recipient is self)
            and len(self._compositions) == len(captured["compositions"])
            and all(type(now) is tuple and len(now) == len(before)
                    and all(a is b for a, b in zip(now, before))
                    for now, before in zip(self._compositions, captured["compositions"]))
            and all(vllm_nixl._composition_matches(*entry) for entry in captured["compositions"])
            and ((captured["getter"] is None and not hasattr(mapping, "_getter_owner") and not hasattr(mapping, "_getter")) or (
                captured["getter"] is not None and mapping._getter_owner is captured["getter"][0]
                and mapping._getter is captured["getter"][1]
                and vars(captured["getter"][0]).get("get_blocks", _MISSING) is captured["getter"][1]
                and _function_matches(captured["getter"][2]))))

    def coverage(self):
        if self.pid != os.getpid() or self._closed or not self._coverage_stable:
            return False
        try:
            valid = (bool(self._slots)
                and all(getattr(owner, name, None) is installed for owner, name, installed in self._slots.values())
                and all(_fingerprint(function.__code__) == expected for function, expected in self._codes.values())
                and self._chains_match()
                and all(vllm_nixl._composition_matches(*entry) for entry in self._compositions))
        except Exception:
            valid = False
        if not valid and self._coverage_stable:
            self._coverage_stable = False
            self.error("recipient callable coverage was lost during the interval")
        return valid and self._coverage_stable

    def detach(self):
        for entry in reversed(self._compositions):
            if not vllm_nixl.unregister_composition(*entry):
                self.error("recipient composition unregister was incomplete")
        self._compositions.clear()
        self._slots.clear()

    def _ensure(self):
        mapping = self._mapping()
        _require(self._native_sources is not None and len(self._native_sources) == 22,
                 "recipient native originals were not source-verified at attachment")
        _require(mapping._refs is not None and mapping._association is not None and mapping._writer is not None,
                 "recipient initialization association is unavailable")
        refs = mapping._refs
        scheduler, group, pool = (refs[role]() for role in ("scheduler", "group_manager", "pool"))
        config = scheduler.vllm_config
        _require(type(scheduler) is mapping.bindings.initialization.scheduler
                 and config.scheduler_config.async_scheduling is False
                 and config.scheduler_config.scheduler_cls is None
                 and scheduler.defer_block_free is False
                 and type(config.parallel_config.pipeline_parallel_size) is int
                 and config.parallel_config.pipeline_parallel_size == 1,
                 "recipient synchronous policy is not observed")
        _require(type(group._partial_hit_reqs) is dict and not group._partial_hit_reqs
                 and type(pool._reuse_watchers) is dict and not pool._reuse_watchers,
                 "recipient sharing, CoW or unpin state is unsupported")
        _require(self.coverage() and mapping.coverage(), "recipient source or callable coverage changed")
        policy = dict(scheduler_class="Scheduler", async_scheduling=False, scheduler_cls=None,
                      defer_block_free=False, max_concurrent_batches=1,
                      concurrency_basis="source_derived_from_async_false_pp1")
        if self._header is None:
            _require(not self.records, "recipient facts precede their native header")
            _require(type(mapping._writer.session) is str, "recipient mapping session is unavailable")
            objects = {role: mapping.registry.identify(refs[role](), role)
                       for role in ("scheduler", "manager", "coordinator", "group_manager", "pool")}
            fields = dict(adapter=contract.DEFAULT_ADAPTER, source_sha256=dict(mapping.source_hashes),
                profile=mapping._profile, dependencies=mapping._dependencies,
                mapping_session=mapping._writer.session, association=mapping._association,
                objects=objects, policy=policy, coverage=sorted(METHODS),
                callables={label: mapping.registry.identify(self._slots[label][2], "recipient_callable")
                           for label in METHODS})
            self._emit("header", **fields)
            self._header = self.records[0]
        else:
            _require(self._header["mapping_session"] == mapping._writer.session
                     and self._header["association"] == mapping._association
                     and self._header["policy"] == policy,
                     "recipient native association changed")
        return mapping, refs

    def _blocks(self, values):
        mapping, refs = self._ensure()
        _require(type(values) in (list, tuple) and len(values) <= MAX_RETURN_BLOCKS,
                 "recipient block references are not a bounded concrete container")
        pool = refs["pool"]()
        facts, seen = [], set()
        for block in values:
            index = block.block_id
            _require(_integer(index, 1, pool.num_gpu_blocks - 1) and index not in seen
                     and block.pool is pool and block.is_null is False and pool.blocks[index] is block,
                     "recipient returned block object or pool differs")
            ordinal = mapping._allocations.get(index)
            _require(_integer(ordinal, 1), "recipient allocation ordinal was not observed")
            facts.append(dict(block=index, incarnation=ordinal))
            seen.add(index)
        return facts

    def _request(self, request, refs):
        mapping = self._mapping()
        scheduler = refs["scheduler"]()
        _require(type(request) is mapping.bindings.request and type(scheduler.requests) is dict
                 and type(request.request_id) is str and 0 < len(request.request_id) <= 128
                 and request.request_id.isascii() and all(ord(c) >= 32 for c in request.request_id)
                 and scheduler.requests.get(request.request_id) is request,
                 "recipient actual Request is not the bound inventory object")
        params = request.sampling_params
        _require(params is not None and _integer(params.n, 1, 1), "recipient native Request n is unsupported")
        policy = dict(n=params.n, resumable=request.resumable, streaming=request.streaming_queue is not None,
                      preemptions=request.num_preemptions, replay_start=request.replay_start,
                      shared_prefix_boundary=request.shared_prefix_boundary, in_flight_tokens=request.num_in_flight_tokens,
                      pooling_params_absent=request.pooling_params is None, lora_request_absent=request.lora_request is None)
        expected = dict(n=1, resumable=False, streaming=False, preemptions=0, replay_start=0,
                        shared_prefix_boundary=0, in_flight_tokens=0, pooling_params_absent=True, lora_request_absent=True)
        _require(all(type(policy[k]) is type(v) and policy[k] == v for k, v in expected.items()),
                 "recipient actual Request policy is unsupported")
        token = mapping.registry.identify(request, "request")
        _require(self._names.get(request.request_id, token) == token, "recipient Request name changed lifetime")
        self._names[request.request_id] = token
        return token, policy

    def _outer_begin(self, receiver, arguments):
        mapping = self._mapping()
        if mapping._refs is None:
            _require(mapping._bridge_runner is not None and mapping._bridge_runner() is not None,
                     "recipient allocation precedes runner initialization")
            mapping._binding(runner=mapping._bridge_runner())
        if mapping._bridge_runner is not None and mapping._bridge_runner() is not None:
            mapping._binding(runner=mapping._bridge_runner())
        mapping, refs = self._ensure()
        if refs["manager"]() is not receiver:
            return None
        _require(not self._open and not self._stack() and not any(not frame["ended"] for frame in mapping._frames.values()),
                 "recipient allocation overlaps an unresolved call or frame")
        request = arguments["request"]
        token, policy = self._request(request, refs)
        state = dict(num_computed_tokens=request.num_computed_tokens, num_tokens=request.num_tokens,
                     num_prompt_tokens=request.num_prompt_tokens, max_model_len=receiver.max_model_len)
        _require(all(_integer(value) for value in state.values()), "recipient native Request dimensions are unavailable")
        selected = {key: arguments[key] for key in ("num_new_tokens", "num_new_computed_tokens", "num_lookahead_tokens",
            "num_external_computed_tokens", "delay_cache_blocks", "num_encoder_tokens", "full_sequence_must_fit",
            "reserved_blocks", "has_scheduled_reqs")}
        skip = arguments["skip_zeroing_group_ids"]
        _require(type(skip) is tuple and skip in ((), (0,)) and type(receiver.retained_hit_group_ids) in (tuple, list)
                 and not receiver.retained_hit_group_ids, "recipient zeroing or retained-group selectors are unsupported")
        selected["skip_zeroing_group_ids"] = list(skip)
        selected["retained_hit_group_ids"] = []
        computed = arguments["new_computed_blocks"]
        canonical = mapping.bindings.initialization.manager_module.KVCacheBlocks
        _require(computed is None or type(computed) is canonical, "recipient computed result type is not canonical")
        groups = receiver.empty_kv_cache_blocks.blocks if computed is None else computed.blocks
        _require(type(groups) is tuple and len(groups) == 1 and type(groups[0]) in (tuple, list) and not groups[0],
                 "recipient local computed sharing is unsupported")
        selected["new_computed_blocks"] = None if computed is None else [[]]
        selected["computed_blocks_native_empty"] = groups is receiver.empty_kv_cache_blocks.blocks
        call = mapping.registry.new_token()
        _require(self._outer_calls < MAX_OUTER_CALLS, "recipient outer-call bound exceeded")
        self._emit("allocation_begin", call=call, request=request.request_id, request_incarnation=token,
                   request_policy=policy, request_state=state, arguments=selected)
        self._open[call] = dict(kind="outer", request=weakref.ref(request), name=request.request_id,
                                policy=policy, state=state, arguments=selected)
        self._stack().append(call)
        self._outer_calls += 1
        return call

    def phase_begin(self, label, receiver, arguments, mutation):
        mapping, refs = self._ensure()
        _require(self._stack() and len(self._stack()) < MAX_NESTING, "recipient phase has no bounded outer call")
        outer = self._stack()[0]
        root = self._open[outer]
        request = root["request"]()
        self._request(request, refs)
        _require(arguments["request_id"] == root["name"], "recipient lower request differs from actual outer Request")
        role = "coordinator" if label.startswith("KVCacheCoordinator") else "group_manager"
        _require(refs[role]() is receiver, "recipient lower receiver differs from the actual bound graph")
        if label == COMPUTED:
            computed = arguments["new_computed_blocks"]
            _require(type(computed) is tuple and len(computed) == 1 and type(computed[0]) in (list, tuple) and not computed[0],
                     "recipient computed coordinator has local shared blocks")
            group = refs["group_manager"]()
            _require(type(group.num_cached_block) is dict, "recipient tracked-control state is unavailable")
            fields = dict(new_computed_blocks=[[]], num_local_computed_tokens=arguments["num_local_computed_tokens"],
                num_external_computed_tokens=arguments["num_external_computed_tokens"],
                skip_zeroing_group_ids=list(arguments["skip_zeroing_group_ids"]),
                already_tracked=root["name"] in group.num_cached_block)
        elif label == COORDINATED:
            fields = {key: arguments[key] for key in ("num_tokens", "num_tokens_main_model", "num_encoder_tokens")}
        elif label == LOCAL:
            computed = arguments["new_computed_blocks"]
            _require(type(computed) in (list, tuple) and not computed, "recipient local sharing is unsupported")
            fields = dict(new_computed_blocks=[], num_local_computed_tokens=arguments["num_local_computed_tokens"],
                          num_external_computed_tokens=arguments["num_external_computed_tokens"])
        elif label == EXTERNAL:
            fields = {key: arguments[key] for key in ("num_local_computed_tokens", "num_external_computed_tokens", "record_for_zeroing")}
        else:
            _require(label == ORDINARY, "recipient phase method is unsupported")
            fields = {key: arguments[key] for key in ("num_tokens", "num_tokens_main_model")}
        _require(self._phase_calls < MAX_PHASE_CALLS, "recipient phase-call bound exceeded")
        call = mapping.registry.new_token()
        self._emit("phase_begin", call=call, outer=outer, parent=self._stack()[-1], method=label,
                   receiver=self._header["objects"][role], request=root["name"], arguments=fields, mapping_mutation=mutation)
        self._open[call] = dict(kind="phase", method=label, outer=outer)
        self._stack().append(call)
        self._phase_calls += 1
        return call

    def pool_return(self, mutation, pool, requested, result, facts):
        mapping, refs = self._ensure()
        _require(refs["pool"]() is pool and type(result) is list and _integer(requested, 0, MAX_RETURN_BLOCKS)
                 and len(result) == requested and self._blocks(result) == facts,
                 "recipient concrete pool return differs from observed acquisition")
        _require(self._block_facts + len(facts) <= MAX_BLOCK_FACTS, "recipient block-fact bound exceeded")
        stack = self._stack()
        outer = stack[0] if stack else None
        phase = stack[-1] if len(stack) > 1 else None
        self._emit("pool_return", outer=outer, phase=phase, mutation=mutation,
                   pool=self._header["objects"]["pool"], requested_blocks=requested, blocks=facts)
        self._pool_returns += 1
        self._block_facts += len(facts)
        self._empty_returns += not facts
        if outer is None or phase is None:
            self.error("recipient pool return has no actual Request phase context")

    def phase_end(self, call, result, exception):
        if call is None:
            return
        _require(self._stack() and self._stack()[-1] == call and call in self._open,
                 "recipient phase closure is unbalanced")
        item = self._open[call]
        method = item["method"]
        returned, kind = None, "NoneType"
        if exception is None:
            if method == ORDINARY:
                _require(type(result) is list, "recipient ordinary allocation return is not a concrete list")
                returned, kind = self._blocks(result), "list"
            elif method == COORDINATED:
                _require(type(result) is tuple and len(result) == 1 and type(result[0]) is list,
                         "recipient coordinator return is not a concrete one-group tuple")
                returned, kind = [self._blocks(result[0])], "tuple"
            else:
                _require(result is None, "recipient None-returning phase changed type")
        self._emit("phase_end", call=call, outer=item["outer"], outcome="returned" if exception is None else "raised",
                   return_type=kind, returned=returned,
                   exception_type=None if exception is None else type(exception).__name__)
        self._stack().pop();self._open.pop(call)

    def _outer_end(self, call, result, exception):
        if call is None:
            return
        mapping = self._mapping()
        _require(mapping._bridge_runner is not None and mapping._bridge_runner() is not None,
                 "recipient runner identity is unavailable at outer closure")
        mapping._binding(runner=mapping._bridge_runner())
        mapping, refs = self._ensure()
        _require(self._stack() == [call] and call in self._open, "recipient outer closure is unbalanced")
        item = self._open[call]
        request = item["request"]()
        token, policy = self._request(request, refs)
        state = dict(num_computed_tokens=request.num_computed_tokens, num_tokens=request.num_tokens,
                     num_prompt_tokens=request.num_prompt_tokens, max_model_len=refs["manager"]().max_model_len)
        _require(policy == item["policy"] and state == item["state"], "recipient Request metadata changed during allocation")
        returned, kind, outcome = None, "NoneType", "raised" if exception is not None else "not_allocated"
        if exception is None and result is not None:
            _require(type(result) is mapping.bindings.initialization.manager_module.KVCacheBlocks
                     and type(result.blocks) is tuple and len(result.blocks) == 1
                     and type(result.blocks[0]) in (list, tuple), "recipient outer return type is not canonical")
            returned, kind, outcome = [self._blocks(result.blocks[0])], "KVCacheBlocks", "returned"
        self._emit("allocation_end", call=call, outcome=outcome, return_type=kind, returned=returned,
                   exception_type=None if exception is None else type(exception).__name__,
                   request_revalidated=True, graph_revalidated=True)
        self._stack().pop();self._open.pop(call)

    def wrap(self, label, original):
        signature = inspect.signature(original)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            if self.pid != os.getpid() or self._closed:
                if self.pid == os.getpid():
                    self.error("recipient call arrived after finite interval closure")
                return original(*args, **kwargs)
            call = None
            try:
                bound = signature.bind(*args, **kwargs);bound.apply_defaults()
                receiver = bound.arguments["self"]
                call = self.safe(self._outer_begin, receiver, bound.arguments) if label == OUTER else self.safe(
                    self.phase_begin, label, receiver, bound.arguments, None)
            except Exception:
                self.error("recipient outer/coordinator entry is incomplete")
            try:
                result = original(*args, **kwargs)
            except BaseException as exception:
                try:
                    self.safe(self._outer_end if label == OUTER else self.phase_end, call, None, exception)
                except BaseException:
                    # Never replace the already pending native exception.
                    self.error("recipient exceptional closure was interrupted")
                raise
            self.safe(self._outer_end if label == OUTER else self.phase_end, call, result, None)
            return result
        return observed

    def close_interval(self):
        self._source_stable = all(_fingerprint(fn.__code__) == expected for fn, expected in self._codes.values())
        self._coverage_stable = self.coverage() and self._coverage_stable
        if self._open or not self._source_stable or not self._coverage_stable:
            self.error("recipient interval has incomplete calls or changed coverage")
        self._closed = True

    @property
    def status(self):
        return dict(healthy=not self._errors and not self._dropped, closed=self._closed, errors=self._errors, dropped=self._dropped,
                    records=len(self.records), open_calls=len(self._open), exported=self._exported,
                    export_attempted=self._export_attempted, export_in_progress=self._export_in_progress,
                    read_rights="unknown", lease_join="unknown", physical_reads="unknown")
