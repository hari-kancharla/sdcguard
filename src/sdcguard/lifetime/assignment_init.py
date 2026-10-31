"""Passive initialization facts for the pinned single-process mapping adapter.

Owner: Hari Krishna Kancharla.
This component is inactive until an assembler explicitly supplies native bindings,
the existing runner bridge and a bounded sink. It starts no engine and supplies no
allocation, frame, ownership, device-read or runtime-isolation verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
from types import MappingProxyType
import uuid
import weakref

from . import assignment_obligations as contract
from .. import config_inputs


_MISSING = object()


class InitializationGap(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise InitializationGap(reason)


def _integer(value, minimum=1, maximum=2**31 - 1):
    return type(value) is int and minimum <= value <= maximum


def _freeze(value, budget=None, depth=0):
    budget = [0] if budget is None else budget
    budget[0] += 1
    _require(budget[0] <= 8192 and depth <= 16, "initialization fact bound exceeded")
    if type(value) is dict:
        _require(len(value) <= 128 and all(type(k) is str for k in value), "invalid fact fields")
        return MappingProxyType({k: _freeze(v, budget, depth + 1) for k, v in value.items()})
    if type(value) in (list, tuple):
        _require(len(value) <= 256, "initialization sequence bound exceeded")
        return tuple(_freeze(v, budget, depth + 1) for v in value)
    if type(value) is str:
        _require(len(value) <= 512 and not any(ord(c) < 32 for c in value), "invalid fact text")
        return value
    _require(value is None or type(value) is bool or type(value) is int and value.bit_length() <= 63,
             "native object cannot be serialized as an initialization fact")
    return value


def _thaw(value):
    if isinstance(value, MappingProxyType):
        return {k: _thaw(v) for k, v in value.items()}
    if type(value) is tuple:
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class InitializationFact:
    kind: str
    fields: object

    def record(self):
        """Return an ordinary independent host copy for a supplied writer."""
        return {"kind": self.kind, **_thaw(self.fields)}


class IdentityRegistry:
    """Bounded identity, without native equality or strong object retention."""

    def __init__(self, maximum=4096):
        _require(_integer(maximum, maximum=65536), "invalid identity bound")
        self.pid = os.getpid()
        self.maximum = maximum
        self._lock = threading.RLock()
        self._entries = {}
        self._issued = set()

    def _current(self):
        _require(os.getpid() == self.pid, "process changed")

    def new_token(self):
        self._current()
        with self._lock:
            _require(len(self._issued) < self.maximum, "identity lifetime bound exceeded")
            token = uuid.uuid4().hex
            _require(token not in self._issued, "identity token collision")
            self._issued.add(token)
            return token

    def identify(self, obj, role):
        self._current()
        _require(type(role) is str and role, "invalid identity role")
        key = id(obj)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0]() is obj:
                _require(entry[2] == role, "native identity changed role")
                return entry[1]
            token = self.new_token()
            owner = weakref.ref(self)

            def collected(reference):
                registry = owner()
                if registry is not None and os.getpid() == registry.pid:
                    with registry._lock:
                        if registry._entries.get(key, (None,))[0] is reference:
                            registry._entries.pop(key, None)

            try:
                reference = weakref.ref(obj, collected)
            except TypeError:
                raise InitializationGap("native identity is not weak-referenceable") from None
            self._entries[key] = (reference, token, role)
            return token

    def reference(self, obj, role):
        token = self.identify(obj, role)
        return token, weakref.ref(obj)


@dataclass(frozen=True)
class NativeBindings:
    """Already imported native objects supplied by the future trusted assembler."""

    core_module: object
    manager_module: object
    coordinator_module: object
    core: type
    core_proc: type
    executor: type
    uniproc: type
    wrapper: type
    worker: type
    runner: type
    scheduler: type
    async_scheduler: type
    manager: type
    coordinator: type
    group_manager: type
    pool: type


@dataclass
class _Init:
    token: str
    objects: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    phase: str = "begin"
    poisoned: bool = False
    dimensions: dict | None = None
    worker_list: object = None
    resolved: tuple | None = None
    started: bool = False
    ready: bool = False
    consumed: bool = False
    constructors: set = field(default_factory=set)
    factories: set = field(default_factory=set)


class InitializationObserver:
    """Emit partial init facts; allocation and frame assembly are absent."""

    def __init__(self, sink, health, *, maximum_identities=4096, maximum_initializations=64,
                 maximum_facts=512):
        _require(callable(sink) and callable(getattr(health, "error", None)), "sink or health is missing")
        _require(_integer(maximum_initializations, maximum=256)
                 and _integer(maximum_facts, maximum=16384), "invalid initialization bound")
        self.pid = os.getpid()
        self.registry = IdentityRegistry(maximum_identities)
        self.sink, self.health = sink, health
        self.maximum_initializations, self.maximum_facts = maximum_initializations, maximum_facts
        self._lock = threading.RLock()
        self._local = threading.local()
        self._inits, self._epochs = {}, {}
        self._hooks = []
        self._facts = self._errors = 0
        self._closed = False
        self._bindings = None
        self._scheduler_signature = None
        self._model_binding_getter = None
        self._model_registration = None

    def bind_model_references(self, registration):
        """Retain only a weak association with the actual forward installation."""
        from .forward_bridge import Registration
        from .vllm_forward import Installation
        _require(self._current() and self._facts == 0 and not self._inits
                 and self._model_binding_getter is None and type(registration) is Registration,
                 "model reference registration is not a fresh local installation")
        owner = registration.model_binding_installation()
        _require(owner is not None, "model reference installation capability is unavailable")
        getter = owner.model_binding_for
        _require(getter.__self__ is owner
                 and getter.__func__ is Installation.model_binding_for,
                 "model reference getter is not the observed installation method")
        self._model_binding_getter = weakref.WeakMethod(getter)
        self._model_registration = registration

    def _current(self):
        return self.pid == os.getpid() and not self._closed

    def _error(self, reason, item=None):
        if not self._current():
            return
        if item is not None:
            item.poisoned = True
            item.ready = False
        self._errors += 1
        try:
            self.health.error(reason)
        except BaseException:
            pass

    def _emit(self, kind, fields):
        _require(self._current(), "observer is closed or process changed")
        _require(self._facts < self.maximum_facts, "initialization fact count exceeded")
        fact = InitializationFact(kind, _freeze(fields))
        _require(len(json.dumps(fact.record()).encode()) <= 65536, "initialization fact bytes exceeded")
        self._facts += 1
        _require(self.sink(fact) is True, "initialization sink rejected a fact")

    def _stack(self):
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    def _active(self, name):
        for entry in reversed(self._stack()):
            if entry[0] == name:
                return entry
        return None

    def _call(self, name, original, args, kwargs):
        """Observation failures cannot replace a native call, return or exception."""
        if not self._current():
            return original(*args, **kwargs)
        if len(self._stack()) >= 64:
            self._error("initialization nesting bound exceeded", self._parent())
            return original(*args, **kwargs)
        entry = None
        try:
            signature = inspect.signature(original)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if name == "scheduler" and "kv_cache_config" not in bound.arguments:
                # The pinned AsyncScheduler forwards its variadic args unchanged.
                signature = self._scheduler_signature
                bound = self._scheduler_signature.bind(*args, **kwargs)
                bound.apply_defaults()
            observed = dict(bound.arguments)
            for parameter in signature.parameters.values():
                if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                    extra = observed.pop(parameter.name)
                    _require(not extra.keys() & observed.keys(), "ambiguous native bound arguments")
                    observed.update(extra)
            entry = (name, observed, None)
            item = self._enter(name, observed)
            entry = (name, observed, item)
        except Exception:
            self._error("initialization entry observation failed", self._parent())
        self._stack().append(entry or (name, {}, None))
        try:
            result = original(*args, **kwargs)
        except BaseException:
            try:
                self._error("native initialization call raised", entry[2] if entry else self._parent())
            except BaseException:
                pass
            raise
        else:
            try:
                if entry is not None and entry[2] is not None:
                    self._leave(name, entry[1], result, entry[2])
            except Exception:
                self._error("initialization return observation failed", entry[2] if entry else self._parent())
            return result
        finally:
            try:
                self._stack().pop()
                if name == "core" and entry is not None and entry[2] is not None:
                    entry[2].worker_list = None
            except BaseException:
                self._error("initialization cleanup incomplete")

    def _parent(self):
        entry = self._active("core")
        return entry[2] if entry is not None else None

    def _bind(self, item, role, obj):
        token, reference = self.registry.reference(obj, role)
        previous = item.objects.get(role)
        _require(previous is None or previous == token, "native participant changed")
        item.objects[role], item.refs[role] = token, reference
        return token

    def _obj(self, item, role):
        value = item.refs[role]()
        _require(value is not None, "native participant was collected")
        return value

    def _valid(self, item):
        _require(self.coverage() and not item.poisoned
                 and all(self._epochs.get(k) == v for k, v in item.generation.items()),
                 "initialization resource generation changed")

    def _reserve(self, item, roles):
        with self._lock:
            tokens = {item.objects[r] for r in roles}
            for previous in self._inits.values():
                if previous is not item and tokens.intersection(previous.generation):
                    previous.poisoned, previous.ready = True, False
            for token in tokens:
                self._epochs[token] = self._epochs.get(token, 0) + 1
                item.generation[token] = self._epochs[token]

    def _retire_objects(self, objects):
        with self._lock:
            for item in self._inits.values():
                if any(reference() is obj for reference in item.refs.values() for obj in objects):
                    item.poisoned, item.ready = True, False

    def _classes(self, item, roles):
        return {r: type(self._obj(item, r)).__module__ + "." + type(self._obj(item, r)).__name__ for r in roles}

    def _participants(self, item, roles):
        return {"objects": {r: item.objects[r] for r in roles}, "classes": self._classes(item, roles)}

    def _dimensions(self, config, scheduler_size):
        groups = config.kv_cache_groups
        _require(type(groups) is list and len(groups) == 1, "unsupported cache groups")
        group, names = groups[0], groups[0].layer_names
        _require(type(names) is list and 0 < len(names) <= 128
                 and all(type(n) is str and 0 < len(n) <= 512 for n in names)
                 and len(set(names)) == len(names), "unsupported layer inventory")
        size = group.kv_cache_spec.block_size
        _require(_integer(config.num_blocks, 2) and _integer(size) and type(scheduler_size) is int
                 and size == scheduler_size == config.hash_block_size, "unsupported block granularity")
        _require(type(config.kv_cache_layout) is str and config.kv_cache_layout == "LBHNC",
                 "unsupported cache layout")
        return dict(capacity=config.num_blocks, block_size=size, scheduler_block_size=scheduler_size,
                    hash_block_size=config.hash_block_size, groups=1, layout=config.kv_cache_layout,
                    layer_names=list(names))

    def _enter(self, name, a):
        b = self._bindings
        if name == "core":
            core, config = a["self"], a["vllm_config"]
            self._retire_objects((core,))
            _require(len(self._inits) < self.maximum_initializations, "initialization count exceeded")
            _require(type(core) in (b.core, b.core_proc) and core.vllm_config is config
                     and type(core.model_executor) is b.uniproc
                     and core.model_executor.vllm_config is config, "unsupported parent initialization")
            executor, wrapper = core.model_executor, core.model_executor.driver_worker
            worker, runner = wrapper.worker, wrapper.worker.model_runner
            _require(type(wrapper) is b.wrapper and type(worker) is b.worker and type(runner) is b.runner,
                     "unsupported worker chain")
            item = _Init(self.registry.new_token())
            for role, obj in (("core", core), ("executor", executor), ("wrapper", wrapper),
                              ("worker", worker), ("runner", runner), ("vllm_config", config)):
                self._bind(item, role, obj)
            self._reserve(item, ("core", "executor", "wrapper", "worker", "runner"))
            self._inits[item.token] = item
            self._emit("init_begin", {"init": item.token, **self._participants(item, ("core", "executor"))})
            return item
        item = self._parent()
        if name == "scheduler":
            scheduler = a["self"]
            if type(scheduler) not in (b.scheduler, b.async_scheduler):
                raise InitializationGap("unsupported scheduler class")
            nested = self._active("scheduler")
            if nested is not None:
                _require(nested[1]["self"] is scheduler, "overlapping scheduler constructor")
                return nested[2]
            candidates = [v for v in self._inits.values() if v.phase == "returned" and not v.poisoned
                          and v.refs["scheduler_config"]() is a["kv_cache_config"]]
            _require(len(candidates) == 1, "returned scheduler config association is missing or ambiguous")
            item = candidates[0]
            self._valid(item)
            _require(not item.consumed and a["vllm_config"] is self._obj(item, "vllm_config"),
                     "scheduler configuration parent differs")
            item.consumed = True
            self._bind(item, "scheduler", scheduler)
            return item
        if item is None:
            scheduler = self._active("scheduler")
            item = scheduler[2] if scheduler is not None else None
        if item is None:
            # The same resolver is also called after the parent init returns.
            # An unrelated helper call supplies no initialization evidence.
            if "self" in a:
                self._retire_objects((a["self"],))
            return None
        self._valid(item)
        if name in ("derive", "resolve"):
            return item
        if name == "executor":
            _require(item.phase == "derived" and a["self"] is self._obj(item, "executor")
                     and a["kv_cache_configs"] is item.worker_list, "worker configuration call edge differs")
            worker_config, scheduler_config = self._obj(item, "worker_config"), self._obj(item, "scheduler_config")
            _require(item.resolved is not None, "resolved granularity was not observed")
            size, hash_size = item.resolved
            self._profile(item)
            dimensions = self._dimensions(worker_config, size)
            _require(self._dimensions(scheduler_config, size) == dimensions and hash_size == size
                     and worker_config.cache_hit_alignment_tokens == scheduler_config.cache_hit_alignment_tokens
                     and _integer(worker_config.cache_hit_alignment_tokens), "finalized config dimensions differ")
            item.dimensions = dimensions
            self._emit("configs_derived", dict(init=item.token, worker_config=item.objects["worker_config"],
                       scheduler_config=item.objects["scheduler_config"], dimensions=dimensions))
            item.phase = "executor"
        elif name == "wrapper":
            _require(self._active("executor") is not None and a["self"] is self._obj(item, "wrapper")
                     and a["kv_cache_configs"] is item.worker_list and a["self"].global_rank == 0
                     and type(a["self"].global_rank) is int and type(a["self"].rpc_rank) is int
                     and a["self"].rpc_rank == 0,
                     "selected worker list or rank differs")
        elif name == "worker":
            _require(item.phase == "executor" and self._active("wrapper") is not None
                     and a["self"] is self._obj(item, "worker")
                     and a["kv_cache_config"] is self._obj(item, "worker_config")
                     and a["self"].model_runner is self._obj(item, "runner"), "selected worker input differs")
            self._emit("worker_init", dict(init=item.token, executor=item.objects["executor"],
                       selected_config=item.objects["worker_config"], rank=0,
                       **self._participants(item, ("wrapper", "worker", "runner"))))
            item.phase = "worker"
        elif name in ("manager", "coordinator", "pool", "group_manager", "coordinator_factory", "group_factory"):
            _require(self._active("scheduler") is not None, "allocator construction has no scheduler parent")
            parent = {"manager": "scheduler", "coordinator_factory": "manager",
                      "coordinator": "coordinator_factory", "pool": "coordinator",
                      "group_factory": "coordinator", "group_manager": "group_factory"}[name]
            _require(self._active(parent) is not None, "allocator constructor call edge is missing")
            expected = {"manager": b.manager, "coordinator": b.coordinator,
                        "pool": b.pool, "group_manager": b.group_manager}
            if name in expected:
                _require(type(a["self"]) is expected[name], "unsupported allocator constructor class")
                self._bind(item, name, a["self"])
            if name in ("manager", "coordinator", "coordinator_factory"):
                _require(a["kv_cache_config"] is self._obj(item, "scheduler_config"),
                         "allocator constructor config edge differs")
            elif name in ("group_manager", "group_factory"):
                _require(a["kv_cache_spec"] is self._obj(item, "scheduler_config").kv_cache_groups[0].kv_cache_spec
                         and a["block_pool"] is self._obj(item, "pool"), "group constructor pool edge differs")
            elif name == "pool":
                _require(a["num_gpu_blocks"] == item.dimensions["capacity"]
                         and a["enable_caching"] is False
                         and a["hash_block_size"] == item.dimensions["hash_block_size"],
                         "pool constructor dimensions differ")
        return item

    def _leave(self, name, a, result, item):
        self._valid(item)
        if name == "derive":
            configs = a["kv_cache_configs"]
            _require(item.phase == "begin" and type(configs) is list and len(configs) == 1
                     and result is not configs[0], "scheduler derivation copy edge differs")
            item.worker_list = configs
            self._bind(item, "worker_config", configs[0])
            self._bind(item, "scheduler_config", result)
            item.phase = "derived"
        elif name == "resolve":
            _require(a["kv_cache_config"] is self._obj(item, "scheduler_config")
                     and a["vllm_config"] is self._obj(item, "vllm_config")
                     and type(result) is tuple and len(result) == 2 and all(_integer(v) for v in result),
                     "granularity return edge differs")
            item.resolved = result
        elif name == "worker":
            _require(item.phase == "runner", "real runner initialization was not observed")
        elif name == "core":
            _require(item.phase == "runner" and result is self._obj(item, "scheduler_config"),
                     "outer initialization return differs or is incomplete")
            self._check_runner(item)
            self._emit("init_return", dict(init=item.token, scheduler_config=item.objects["scheduler_config"]))
            item.phase = "returned"
        elif name in ("manager", "coordinator", "pool", "group_manager"):
            _require(name not in item.constructors, "allocator constructor repeated")
            item.constructors.add(name)
        elif name in ("coordinator_factory", "group_factory"):
            role = "coordinator" if name == "coordinator_factory" else "group_manager"
            _require(result is self._obj(item, role), "allocator factory returned another object")
            _require(name not in item.factories, "allocator factory repeated")
            item.factories.add(name)
        elif name == "scheduler":
            # The AsyncScheduler base call is not completion of its outer constructor.
            outer = [x for x in self._stack() if x[0] == "scheduler"]
            if len(outer) > 1:
                return
            _require(item.constructors == {"manager", "coordinator", "pool", "group_manager"}
                     and item.factories == {"coordinator_factory", "group_factory"},
                     "native allocator constructor edges are incomplete")
            self._check_allocator(item)
            self._emit("scheduler_ready", dict(init=item.token, passed_config=item.objects["scheduler_config"],
                       **self._participants(item, ("scheduler", "coordinator", "manager", "group_manager", "pool")),
                       dimensions=item.dimensions, null_block=0))
            item.phase, item.ready = "ready", True

    def _check_runner(self, item):
        runner = self._obj(item, "runner")
        self._profile(item)
        _require(runner.kv_cache_config is self._obj(item, "runner_config")
                 and runner.block_tables is self._obj(item, "block_tables"), "runner cache incarnation changed")
        _require(self._dimensions(runner.kv_cache_config, item.dimensions["block_size"]) == item.dimensions,
                 "copied runner configuration differs")
        _require(runner.kernel_block_sizes == [item.dimensions["block_size"]]
                 and type(runner.kernel_block_sizes) is list, "unsupported selected kernel granularity")
        groups = runner.attn_groups
        _require(type(groups) is list and len(groups) == 1 and type(groups[0]) is list and groups[0],
                 "attention group inventory is missing")
        names = []
        for group in groups[0]:
            _require(group.kv_cache_group_id == 0 and type(group.kv_cache_group_id) is int
                     and group.kernel_block_size is None and type(group.kernel_block_stride) is int
                     and group.kernel_block_stride == 0
                     and group.kernel_block_table is None and group.kernel_block_offsets is None,
                     "per-attention-group block remapping is unsupported")
            self._dense_spec(group.kv_cache_spec)
            _require(group.kv_cache_spec.block_size == item.dimensions["block_size"],
                     "attention group specification differs")
            _require(type(group.layer_names) is list, "attention layer inventory is not concrete")
            names.extend(group.layer_names)
        _require(len(names) == len(set(names)) and set(names) == set(item.dimensions["layer_names"]),
                 "attention layer inventory differs")
        tables = runner.block_tables
        _require(type(tables.num_kv_cache_groups) is int and tables.num_kv_cache_groups == 1
                 and type(tables.block_sizes) is list and tables.block_sizes == [item.dimensions["block_size"]]
                 and type(tables.cp_size) is int and tables.cp_size == 1
                 and type(tables.cp_rank) is int and tables.cp_rank == 0
                 and tables.redirect_writes_to_null_block is False, "unsupported actual block table mapping")

    def _profile(self, item):
        config = self._obj(item, "vllm_config")
        parallel, model, cache = config.parallel_config, config.model_config, config.cache_config
        if self._model_binding_getter is not None:
            from .. import model_reference
            from .vllm_forward import Installation
            runner = self._obj(item, "runner")
            getter = self._model_binding_getter() if self._model_binding_getter is not None else None
            _require(getter is not None, "initialization model installation was collected")
            _require(self._model_registration.model_binding_installation() is getter.__self__,
                     "initialization model installation capability changed")
            _require(type(getter.__self__) is Installation and getter.__self__.pid == self.pid,
                     "initialization model installation class or process differs")
            _require(getter.__func__ is Installation.model_binding_for
                     and getattr(type(getter.__self__), "model_binding_for", None) is getter.__func__
                     and getter.__self__.runner_class is type(runner),
                     "initialization model getter or runner class differs")
            _require(runner.model_config is model, "initialization and runner model configs differ")
            binding = getter(runner)
            model_reference.observe(model, binding)
            _require(config.model_config is model and runner.model_config is model,
                     "initialization model config changed during observation")
        else:
            _require(type(model.model) is str and model.model == "Qwen/Qwen3-0.6B",
                     "local initialization model has no matching installation binding")
        _require(model.enforce_eager is True
                 and model.architectures == ["Qwen3ForCausalLM"]
                 and cache.enable_prefix_caching is False and cache.cache_dtype == "auto"
                 and config.speculative_config is None and config.lora_config is None
                 and parallel.enable_dbo is False and type(parallel.worker_extension_cls) is str
                 and parallel.worker_extension_cls == "",
                 "unsupported effective initialization profile")
        for name in ("tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size",
                     "prefill_context_parallel_size", "data_parallel_size"):
            _require(type(getattr(parallel, name)) is int and getattr(parallel, name) == 1,
                     "unsupported initialization topology")
        _require(type(parallel.ubatch_size) is int and parallel.ubatch_size in (0, 1),
                 "unsupported initialization microbatching")
        _require(config.compilation_config.cudagraph_mode.name == "NONE"
                 and config.compilation_config.mode.name == "NONE"
                 and self._obj(item, "worker").use_v2_model_runner is True,
                 "unsupported compiled or alternate runner")
        worker_config = self._obj(item, "runner_config" if "runner_config" in item.refs else "worker_config")
        _require(type(worker_config.kv_cache_groups) is list and len(worker_config.kv_cache_groups) == 1,
                 "unsupported initialization cache groups")
        group = worker_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        self._dense_spec(spec)
        _require(group.is_eagle_group is False and group.host_resident is False,
                 "unsupported dense full-attention group")

    def _dense_spec(self, spec):
        _require(type(spec).__module__ + "." + type(spec).__name__ == "vllm.v1.kv_cache_interface.FullAttentionSpec"
                 and spec.non_causal is False and type(spec.tokens_per_state) is int and spec.tokens_per_state == 1
                 and spec.kv_quant_mode.name == "NONE"
                 and all(getattr(spec, n) is None for n in ("sliding_window", "attention_chunk_size",
                     "page_size_padded", "num_head_slots", "state_content_bytes")),
                 "unsupported dense full-attention initialization")

    def _check_allocator(self, item):
        scheduler, manager = self._obj(item, "scheduler"), self._obj(item, "manager")
        coordinator, group, pool = (self._obj(item, r) for r in ("coordinator", "group_manager", "pool"))
        config = self._obj(item, "scheduler_config")
        _require(scheduler.kv_cache_config is config and scheduler.kv_cache_manager is manager
                 and manager.coordinator is coordinator and manager.block_pool is pool
                 and coordinator.block_pool is pool and len(coordinator.single_type_managers) == 1
                 and coordinator.single_type_managers[0] is group and group.block_pool is pool
                 and group.kv_cache_spec is config.kv_cache_groups[0].kv_cache_spec,
                 "actual allocator pointer chain differs")
        _require(manager.kv_cache_config is config and coordinator.kv_cache_config is config
                 and self._dimensions(config, scheduler.block_size) == item.dimensions
                 and scheduler.hash_block_size == item.dimensions["hash_block_size"]
                 and pool.num_gpu_blocks == item.dimensions["capacity"]
                 and group.block_size == item.dimensions["block_size"]
                 and manager.enable_caching is False and pool.enable_caching is False,
                 "actual allocator dimensions differ")
        null = pool.null_block
        _require(type(pool.blocks) is list and len(pool.blocks) == pool.num_gpu_blocks
                 and pool.blocks[0] is null and type(null.block_id) is int and null.block_id == 0
                 and null.is_null is True and null.pool is pool, "reserved null block identity differs")

    def runner_initialized(self, boundary):
        """Called only by the existing bridge after its sole native runner wrapper."""
        try:
            item = self._parent()
            _require(item is not None and item.phase == "worker" and item.started
                     and self._active("worker") is not None, "runner callback has no selected worker parent")
            self._valid(item)
            _require(type(boundary.arguments) is tuple
                     and all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                             for pair in boundary.arguments), "runner bound arguments are not concrete")
            arguments = dict(boundary.arguments)
            _require(len(arguments) == len(boundary.arguments)
                     and arguments.keys() == {"kv_cache_config", "is_profiling", "kv_cache_allocation_context"},
                     "runner bound arguments are incomplete or duplicated")
            _require(boundary.runner is self._obj(item, "runner")
                     and arguments["kv_cache_config"] is self._obj(item, "worker_config")
                     and arguments["is_profiling"] is False
                     and boundary.runner_config is boundary.runner.kv_cache_config
                     and boundary.block_tables is boundary.runner.block_tables
                     and boundary.runner_config is not self._obj(item, "worker_config")
                     and boundary.runner_config is not self._obj(item, "scheduler_config"),
                     "real runner input or copied configuration edge differs")
            self._bind(item, "runner_config", boundary.runner_config)
            self._bind(item, "block_tables", boundary.block_tables)
            self._check_runner(item)
            cache = self.registry.new_token()
            item.objects["worker_cache"] = cache
            self._emit("runner_ready", dict(init=item.token, runner=item.objects["runner"],
                       input_config=item.objects["worker_config"], runner_config=item.objects["runner_config"],
                       worker_pool=cache, dimensions=item.dimensions,
                       kernel_block_size=boundary.runner.kernel_block_sizes[0], ratio=1))
            item.phase = "runner"
        except Exception:
            self._error("runner initialization callback is incomplete", self._parent())

    def retire_runner(self, runner, reason):
        """The existing bridge calls this before native replacement or teardown."""
        if not self._current():
            return
        try:
            current = self._parent()
            for item in self._inits.values():
                if item.refs["runner"]() is runner:
                    expected = (item is current and item.phase == "worker" and not item.started
                                and reason == "cache initialization began" and self._active("worker") is not None)
                    if expected:
                        item.started = True
                    else:
                        item.poisoned, item.ready = True, False
        except Exception:
            self._error("runner retirement is incomplete", self._parent())

    def binding_for(self, core, scheduler, runner):
        """Recheck actual attachment for a future schedule assembler; emit no frame."""
        try:
            _require(self._current() and not self._errors and self.coverage(), "initialization hook coverage changed")
            found = [v for v in self._inits.values() if v.ready and not v.poisoned
                     and v.refs["core"]() is core and v.refs["scheduler"]() is scheduler
                     and v.refs["runner"]() is runner]
            _require(len(found) == 1, "complete initialization association is unavailable")
            item = found[0]
            self._valid(item)
            _require(core.scheduler is scheduler and core.model_executor is self._obj(item, "executor")
                     and core.model_executor.driver_worker is self._obj(item, "wrapper")
                     and self._obj(item, "wrapper").worker is self._obj(item, "worker")
                     and self._obj(item, "worker").model_runner is runner, "actual core attachment differs")
            self._check_runner(item)
            self._check_allocator(item)
            return InitializationFact("initialization_binding", _freeze(dict(
                init=item.token, objects=dict(item.objects), dimensions=item.dimensions,
                scope="initialization only", ownership="unknown", physical_reads="unknown")))
        except Exception:
            self._error("initialization binding is unavailable")
            return None

    def lookup(self, *, scheduler=None, runner=None):
        """Return a current fact and weak participants for one actual receiver.

        This integration accessor supplies no mutable initialization state. It
        rechecks the complete binding without holding the bookkeeping lock over
        native attribute reads.
        """
        try:
            _require((scheduler is None) != (runner is None), "lookup requires exactly one native receiver")
            _require(self._current() and self.coverage(), "initialization lookup coverage is unavailable")
            role, receiver = ("scheduler", scheduler) if scheduler is not None else ("runner", runner)
            with self._lock:
                found = [item for item in self._inits.values() if item.ready and not item.poisoned
                         and item.refs.get(role, lambda: None)() is receiver]
            _require(len(found) == 1, "initialization lookup is missing or ambiguous")
            item = found[0]
            core = self._obj(item, "core")
            actual_scheduler = self._obj(item, "scheduler")
            actual_runner = self._obj(item, "runner")
            fact = self.binding_for(core, actual_scheduler, actual_runner)
            _require(fact is not None, "initialization lookup binding is unavailable")
            self._valid(item)
            _require(item.ready and not item.poisoned
                     and item.refs[role]() is receiver, "initialization changed during lookup")
            return fact, MappingProxyType(dict(item.refs))
        except Exception:
            self._error("initialization lookup is unavailable")
            return None

    def attach(self, bindings, source_root, version):
        """Explicit internal assembly only. No plugin or CLI activates this component."""
        _require(self._current() and self._bindings is None, "initialization observer is already attached")
        expected = contract.adapter_contract()
        _require(version == expected["version"], "unsupported engine version")
        root = Path(source_root).resolve()
        for path, digest in contract.source_pins().items():
            raw = config_inputs._read_regular(root, Path(path).parts, 2 * 1024 * 1024)
            _require(hashlib.sha256(raw).hexdigest() == digest, "engine source pin mismatch")
        roles = {"core": bindings.core, "executor": bindings.uniproc, "wrapper": bindings.wrapper,
                 "worker": bindings.worker, "runner": bindings.runner, "scheduler": bindings.scheduler,
                 "manager": bindings.manager, "coordinator": bindings.coordinator,
                 "group_manager": bindings.group_manager, "pool": bindings.pool}
        for role, cls in roles.items():
            _require(cls.__module__ + "." + cls.__name__ in expected["classes"][role], "unsupported native class")
        _require(bindings.core_proc.__module__ + "." + bindings.core_proc.__name__ in expected["classes"]["core"]
                 and bindings.async_scheduler.__module__ + "." + bindings.async_scheduler.__name__
                 in expected["classes"]["scheduler"], "unsupported native subclass")
        _require(bindings.executor.__module__ + "." + bindings.executor.__name__ == "vllm.v1.executor.abstract.Executor"
                 and bindings.core_module.__name__ == "vllm.v1.engine.core"
                 and bindings.manager_module.__name__ == "vllm.v1.core.kv_cache_manager"
                 and bindings.coordinator_module.__name__ == "vllm.v1.core.kv_cache_coordinator",
                 "unsupported native helper owner")
        self._bindings = bindings
        self._scheduler_signature = inspect.signature(bindings.scheduler.__init__)
        hooks = [(bindings.core, "_initialize_kv_caches", "core"),
                 (bindings.core_module, "generate_scheduler_kv_cache_config", "derive"),
                 (bindings.core_module, "resolve_kv_cache_block_sizes", "resolve"),
                 (bindings.executor, "initialize_from_config", "executor"),
                 (bindings.wrapper, "initialize_from_config", "wrapper"),
                 (bindings.worker, "initialize_from_config", "worker"),
                 (bindings.scheduler, "__init__", "scheduler"),
                 (bindings.async_scheduler, "__init__", "scheduler"),
                 (bindings.manager, "__init__", "manager"),
                 (bindings.coordinator, "__init__", "coordinator"),
                 (bindings.pool, "__init__", "pool"),
                 (bindings.group_manager, "__init__", "group_manager"),
                 (bindings.manager_module, "get_kv_cache_coordinator", "coordinator_factory"),
                 (bindings.coordinator_module, "get_manager_for_kv_cache_spec", "group_factory")]
        try:
            for owner, attribute, name in hooks:
                own = attribute in vars(owner)
                original = getattr(owner, attribute)
                _require(callable(original) and not hasattr(original, "_initialization_observer"),
                         "native method is already instrumented")
                def make(fn, label):
                    @functools.wraps(fn)
                    def wrapper(*args, **kwargs):
                        return self._call(label, fn, args, kwargs)
                    wrapper._initialization_observer = weakref.ref(self)
                    return wrapper
                installed = make(original, name)
                setattr(owner, attribute, installed)
                self._hooks.append((owner, attribute, original, installed, own))
        except BaseException:
            try:
                self.detach()
            except BaseException:
                pass
            raise
        return self

    def coverage(self):
        return self._current() and bool(self._hooks) and all(
            getattr(owner, attribute, _MISSING) is installed for owner, attribute, _, installed, _ in self._hooks)

    def status(self):
        return dict(scope="native initialization component only", facts=self._facts, errors=self._errors,
                    ready_associations=sum(v.ready and not v.poisoned for v in self._inits.values())
                        if self._current() and not self._errors and self.coverage() else 0,
                    allocation_observer="absent", frame_assembler="absent", runtime_isolation="not established")

    def detach(self):
        if self.pid != os.getpid():
            return
        try:
            for item in self._inits.values():
                item.poisoned, item.ready = True, False
            for owner, attribute, original, installed, own in reversed(self._hooks):
                try:
                    if getattr(owner, attribute, _MISSING) is installed:
                        if own:
                            setattr(owner, attribute, original)
                        else:
                            delattr(owner, attribute)
                except BaseException:
                    self._error("initialization hook cleanup incomplete")
        finally:
            self._hooks.clear()
            self._model_binding_getter = None
            self._model_registration = None
            self._closed = True
