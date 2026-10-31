"""Explicit source and observed-profile inputs for passive mapping assembly.

Owner: Hari Krishna Kancharla.
Importing this module loads no engine or device library. Source hashes and
canonical class references establish local consistency, not runtime attestation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import functools
import hashlib
import importlib
from pathlib import Path
import platform
import re
from types import MappingProxyType
import weakref


NATIVE_ADAPTER = "vllm.0.32.uniproc.eager.native.v1"
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_LIMIT = 2 * 1024 * 1024


class MappingRuntimeGap(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise MappingRuntimeGap(reason)


def _text(value, maximum=256):
    return (type(value) is str and 0 < len(value) <= maximum
            and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value))


def _observed_errors(function):
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except MappingRuntimeGap:
            raise
        except Exception as exc:
            raise MappingRuntimeGap(function.__name__ + " unavailable: " + type(exc).__name__) from None
    return guarded


@dataclass(frozen=True)
class MappingBindings:
    initialization: object
    manager_base: type
    request: type
    scheduler_output: type
    profile_types: Mapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        _require(isinstance(self.profile_types, Mapping), "profile type bindings are missing")
        object.__setattr__(self, "profile_types", MappingProxyType(dict(self.profile_types)))


@_observed_errors
def validate_sources(source_root, adapter=NATIVE_ADAPTER):
    """Read exact pinned regular files without imports or symlink traversal."""
    from . import assignment_obligations
    from .. import config_inputs

    expected = assignment_obligations.source_pins(adapter)
    root = Path(source_root).absolute()
    actual = {}
    for name, digest in expected.items():
        raw = config_inputs._read_regular(root, Path(name).parts, _SOURCE_LIMIT)
        actual[name] = hashlib.sha256(raw).hexdigest()
        _require(actual[name] == digest, "mapping source pin differs: " + name)
    return actual


def _module(root, name, path, *, sources):
    module = importlib.import_module(name)
    expected = (root / path).absolute()
    _require(getattr(module, "__name__", None) == name
             and _text(getattr(module, "__file__", None), 4096)
             and Path(module.__file__).absolute() == expected,
             "canonical native module path differs: " + name)
    spec = getattr(module, "__spec__", None)
    _require(spec is not None and getattr(spec, "origin", None) == str(expected),
             "canonical native module origin differs: " + name)
    _require(path in sources, "native module is outside the reviewed mapping pins: " + name)
    return module


def _class(module, name):
    cls = vars(module).get(name)
    _require(isinstance(cls, type) and cls.__name__ == name and cls.__module__ == module.__name__,
             "canonical native class differs: " + module.__name__ + "." + name)
    return cls


def _registration(registration, version):
    engine = registration.engine
    _require(isinstance(engine, Mapping) and engine.get("name") == "vllm"
             and type(engine.get("version")) is str and engine["version"] == version,
             "forward registration engine identity differs")
    return engine


@_observed_errors
def load_bindings(source_root, version, registration):
    """Resolve explicit existing native interfaces. Do not wrap or run them."""
    from .assignment_init import NativeBindings
    from . import assignment_obligations

    expected = assignment_obligations.adapter_contract(NATIVE_ADAPTER)
    _require(type(version) is str and version == expected["version"], "unsupported mapping engine version")
    actual = validate_sources(source_root)
    engine = _registration(registration, version)
    supplied = engine.get("source_sha256")
    _require(isinstance(supplied, Mapping) and bool(supplied)
             and all(type(name) is str and name in actual and digest == actual[name]
                     for name, digest in supplied.items()), "forward registration source identity differs")
    root = Path(source_root).absolute()
    package = importlib.import_module("vllm")
    _require(package.__name__ == "vllm" and package.__version__ == version
             and _text(getattr(package, "__file__", None), 4096)
             and Path(package.__file__).absolute() == root / "vllm/__init__.py",
             "loaded engine version or package path differs")
    modules = {}
    names = (
        "vllm.v1.engine.core", "vllm.v1.executor.abstract", "vllm.v1.executor.uniproc_executor",
        "vllm.v1.worker.worker_base", "vllm.v1.worker.gpu_worker", "vllm.v1.worker.gpu.model_runner",
        "vllm.v1.core.sched.scheduler", "vllm.v1.core.sched.async_scheduler",
        "vllm.v1.core.kv_cache_manager", "vllm.v1.core.kv_cache_coordinator",
        "vllm.v1.core.single_type_kv_cache_manager", "vllm.v1.core.block_pool", "vllm.v1.request",
        "vllm.v1.core.sched.output", "vllm.v1.kv_cache_interface", "vllm.v1.worker.utils",
        "vllm.model_executor.models.qwen3", "vllm.model_executor.layers.attention.attention",
        "vllm.v1.worker.gpu.model_states.default", "vllm.v1.attention.backend",
        "vllm.v1.attention.backends.flash_attn",
    )
    for name in names:
        path = name.replace(".", "/") + ".py"
        modules[name] = _module(root, name, path, sources=actual)
    c = lambda mod, name: _class(modules[mod], name)
    core = modules["vllm.v1.engine.core"]
    manager = modules["vllm.v1.core.kv_cache_manager"]
    coordinator = modules["vllm.v1.core.kv_cache_coordinator"]
    initialization = NativeBindings(
        core_module=core, manager_module=manager, coordinator_module=coordinator,
        core=c("vllm.v1.engine.core", "EngineCore"), core_proc=c("vllm.v1.engine.core", "EngineCoreProc"),
        executor=c("vllm.v1.executor.abstract", "Executor"),
        uniproc=c("vllm.v1.executor.uniproc_executor", "UniProcExecutor"),
        wrapper=c("vllm.v1.worker.worker_base", "WorkerWrapperBase"),
        worker=c("vllm.v1.worker.gpu_worker", "Worker"), runner=c("vllm.v1.worker.gpu.model_runner", "GPUModelRunner"),
        scheduler=c("vllm.v1.core.sched.scheduler", "Scheduler"),
        async_scheduler=c("vllm.v1.core.sched.async_scheduler", "AsyncScheduler"),
        manager=c("vllm.v1.core.kv_cache_manager", "KVCacheManager"),
        coordinator=c("vllm.v1.core.kv_cache_coordinator", "KVCacheCoordinatorNoPrefixCache"),
        group_manager=c("vllm.v1.core.single_type_kv_cache_manager", "FullAttentionManager"),
        pool=c("vllm.v1.core.block_pool", "BlockPool"))
    _require(registration.runner_class is initialization.runner, "registered runner is not the canonical mapping runner")
    runner_module = modules["vllm.v1.worker.gpu.model_runner"]
    specs, attention = modules["vllm.v1.kv_cache_interface"], modules["vllm.v1.attention.backend"]
    # The pinned runner imports these enum references. Their underlying numeric
    # values are never interpreted or presented as a separate source pin.
    compilation_mode, graph_mode = runner_module.CompilationMode, runner_module.CUDAGraphMode
    _require(compilation_mode.__module__ == graph_mode.__module__ == "vllm.config.compilation",
             "native compilation mode references differ")
    torch = importlib.import_module("torch")
    _require(getattr(torch, "__name__", None) == "torch" and isinstance(getattr(torch, "__version__", None), str)
             and _text(str(torch.__version__)), "observed torch version is missing")
    profile_types = dict(
        torch=torch, model=c("vllm.model_executor.models.qwen3", "Qwen3ForCausalLM"),
        model_state=c("vllm.v1.worker.gpu.model_states.default", "DefaultModelState"),
        attention=c("vllm.model_executor.layers.attention.attention", "Attention"),
        impl=c("vllm.v1.attention.backends.flash_attn", "FlashAttentionImpl"),
        metadata=c("vllm.v1.attention.backends.flash_attn", "FlashAttentionMetadata"),
        backend=c("vllm.v1.attention.backends.flash_attn", "FlashAttentionBackend"),
        attention_group=c("vllm.v1.worker.utils", "AttentionGroup"),
        spec=c("vllm.v1.kv_cache_interface", "FullAttentionSpec"),
        quant_none=specs.KVQuantMode.NONE, decoder=attention.AttentionType.DECODER,
        compile_none=compilation_mode.NONE, graph_none=graph_mode.NONE)
    return MappingBindings(initialization,
                           c("vllm.v1.core.single_type_kv_cache_manager", "SingleTypeKVCacheManager"),
                           c("vllm.v1.request", "Request"), c("vllm.v1.core.sched.output", "SchedulerOutput"),
                           profile_types)


def _participants(bindings, participant_refs):
    _require(type(participant_refs) is MappingProxyType, "initialization participants are not immutable weak references")
    native = bindings.initialization
    classes = dict(core=(native.core, native.core_proc), executor=(native.uniproc,), wrapper=(native.wrapper,),
                   worker=(native.worker,), runner=(native.runner,), scheduler=(native.scheduler, native.async_scheduler),
                   manager=(native.manager,), coordinator=(native.coordinator,), group_manager=(native.group_manager,),
                   pool=(native.pool,))
    found = {}
    for role, allowed in classes.items():
        ref = participant_refs.get(role)
        _require(isinstance(ref, weakref.ReferenceType), "native participant weak reference is missing: " + role)
        obj = ref()
        _require(obj is not None and type(obj) in allowed, "native participant class or lifetime differs: " + role)
        found[role] = obj
    return found


@_observed_errors
def observed_header(registration, bindings, participant_refs, *, model_binding=None):
    """Re-derive observed header fields. Per-request streaming checks are separate."""
    from . import assignment_obligations
    from . import vllm_forward

    expected = assignment_obligations.adapter_contract(NATIVE_ADAPTER)
    _registration(registration, expected["version"])
    participants = _participants(bindings, participant_refs)
    runner, worker = participants["runner"], participants["worker"]
    _require(registration.runner_class is type(runner) is bindings.initialization.runner,
             "registered or actual runner class differs")
    types = bindings.profile_types
    # This gate reads native metadata only. Its constant-filled return value is
    # intentionally ignored; emitted fields below come from observed state.
    model = runner.model_config
    reference = None
    from .forward_bridge import Registration
    if (type(registration) is Registration
            and registration.model_binding_installation() is not None):
        model_binding = registration.model_binding_for(runner)
    if model.model != "Qwen/Qwen3-0.6B" or model_binding is not None:
        from .. import model_reference
        if model_binding is None:
            getter = getattr(registration, "model_binding_for", None)
            _require(callable(getter), "local model reference binding is unavailable")
            model_binding = getter(runner)
        reference = model_reference.observe(model, model_binding)
        vllm_forward._profile(runner, types, model_binding=model_binding)
    else:
        vllm_forward._profile(runner, types)
    model, parallel, cache = runner.model_config, runner.parallel_config, runner.kv_cache_config
    config = runner.vllm_config
    _require(worker.model_runner is runner and worker.use_v2_model_runner is True
             and config.use_v2_model_runner is True, "actual worker runner mode differs")
    _require(runner.compilation_config.cudagraph_mode is types["graph_none"], "graph mode is unsupported")
    _require(type(parallel.ubatch_size) is int and parallel.ubatch_size in (0, 1)
             and parallel.worker_extension_cls == "" and type(parallel.worker_extension_cls) is str,
             "microbatching or worker extension is unsupported")
    _require(config.speculative_config is runner.speculative_config is None
             and config.lora_config is runner.lora_config is None, "effective optional runner features differ")
    revision = model.revision
    _require(type(revision) is str and _REVISION.fullmatch(revision) is not None,
             "model revision is not an immutable full commit identity")
    group = cache.kv_cache_groups[0]
    names = list(group.layer_names)
    layers = runner.compilation_config.static_forward_context
    groups = runner.attn_groups
    _require(type(groups) is list and len(groups) == 1 and type(groups[0]) is list
             and 0 < len(groups[0]) <= 128, "actual attention group inventory is incomplete")
    collected = []
    for attention_group in groups[0]:
        _require(type(attention_group) is types["attention_group"]
                 and attention_group.backend is types["backend"]
                 and type(attention_group.kv_cache_group_id) is int and attention_group.kv_cache_group_id == 0
                 and attention_group.kernel_block_size is None
                 and type(attention_group.kernel_block_stride) is int and attention_group.kernel_block_stride == 0
                 and attention_group.kernel_block_table is None and attention_group.kernel_block_offsets is None,
                 "actual attention group mapping differs")
        spec = attention_group.kv_cache_spec
        _require(type(spec) is types["spec"] and spec.block_size == group.kv_cache_spec.block_size
                 and spec.non_causal is False and type(spec.tokens_per_state) is int and spec.tokens_per_state == 1
                 and spec.kv_quant_mode is types["quant_none"]
                 and all(getattr(spec, key) is None for key in ("sliding_window", "attention_chunk_size",
                         "page_size_padded", "num_head_slots", "state_content_bytes")),
                 "actual attention group specification differs")
        row = attention_group.layer_names
        _require(type(row) is list and 0 < len(row) <= 128 and all(_text(name, 512) for name in row),
                 "actual attention group layers differ")
        collected.extend(row)
    _require(len(collected) == len(set(collected)) and set(collected) == set(names),
             "actual attention group layer coverage differs")
    selectors, backends = [], []
    sharing = False
    for name in names:
        layer = layers[name]
        _require(layer.attn_backend is types["backend"], "actual layer backend differs")
        selectors.append(layer.impl.vllm_flash_attn_version)
        backends.append(layer.attn_backend.get_name())
        sharing = sharing or (layer.kv_sharing_target_layer_name is not None
                              or layer.impl.kv_sharing_target_layer_name is not None)
    _require(selectors and all(type(value) is int and value == selectors[0] for value in selectors)
             and all(type(value) is str and value == backends[0] for value in backends),
             "actual layer backend observations disagree")
    profile = dict(model=model.model, architecture=model.architectures[0],
                   runner="V2" if config.use_v2_model_runner and worker.use_v2_model_runner else "unsupported",
                   model_revision=revision, layer_names=names, eager=model.enforce_eager,
                   tp=parallel.tensor_parallel_size, pp=parallel.pipeline_parallel_size,
                   dcp=parallel.decode_context_parallel_size, pcp=parallel.prefill_context_parallel_size,
                   dp=parallel.data_parallel_size, prefix_caching=runner.cache_config.enable_prefix_caching,
                   speculative=runner.speculative_config is not None, lora=runner.lora_config is not None,
                   microbatching=(parallel.enable_dbo or parallel.ubatch_size > 1 or runner.ubatch_runner is not None),
                   attention="dense_causal_full" if not group.kv_cache_spec.non_causal
                       and group.kv_cache_spec.sliding_window is None else "unsupported",
                   cache_groups=len(cache.kv_cache_groups), layout=cache.kv_cache_layout,
                   backend=backends[0], backend_version=str(selectors[0]),
                   kv_dtype=runner.cache_config.cache_dtype, kv_sharing=sharing,
                   jit_warmup=config.kernel_config.enable_jit_warmup)
    if reference is not None:
        profile["model_reference"] = reference
    for key, value in expected["profile"].items():
        if reference is not None and key == "model":
            continue
        _require(type(profile[key]) is type(value) and profile[key] == value,
                 "actual observed profile is unsupported: " + key)
    _require(runner.model_config is model and runner.parallel_config is parallel and runner.kv_cache_config is cache
             and model.revision == revision and cache.kv_cache_groups[0] is group and group.layer_names == names
             and config.kernel_config.enable_jit_warmup is profile["jit_warmup"]
             and all(participant_refs[role]() is obj for role, obj in participants.items()),
             "native header inputs changed during observation")
    _require(isinstance(types["torch"].__version__, str), "actual torch version is missing")
    dependencies = dict(python=platform.python_version(), torch=str(types["torch"].__version__), flash_attn=str(selectors[0]))
    _require(all(_text(value) for value in dependencies.values()), "actual dependency observation is missing")
    return profile, dependencies
