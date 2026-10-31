"""CPU source-fragment and host lifecycle tests. Owner: Hari Krishna Kancharla."""
import ast
from concurrent.futures import Future
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import weakref
import zipfile

import pytest

from sdcguard.lifetime import assignment_init as native
from sdcguard.lifetime import assignment_obligations as contract


FIXTURE = Path(__file__).parent / "fixtures/assignment_init"


class Fields:
    def __init__(self, **values):
        self.__dict__.update(values)


class Health:
    def __init__(self):
        self.errors = []

    def error(self, reason):
        self.errors.append(reason)


def named(name, module, base=object, **methods):
    return type(name, (base,), {"__module__": module, **methods})


def fragment(source, name, namespace, owner=None):
    """Compile only a selected pinned function, with ordinary host dependencies."""
    tree = ast.parse(source)
    nodes = tree.body if owner is None else next(n for n in tree.body if isinstance(n, ast.ClassDef)
                                                 and n.name == owner).body
    node = copy.deepcopy(next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned CPU method fragment>", "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="session")
def primary(tmp_path_factory):
    root = tmp_path_factory.mktemp("assignment-primary")
    metadata = json.loads((FIXTURE / "SOURCE.json").read_text())
    raw = (FIXTURE / "primary.zip").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == metadata["archive_sha256"]
    sources = {}
    with zipfile.ZipFile(FIXTURE / "primary.zip") as archive:
        assert set(archive.namelist()) == set(contract.source_pins())
        for entry in metadata["files"]:
            value = archive.read(entry["path"])
            assert hashlib.sha256(value).hexdigest() == entry["sha256"] == contract.source_pins()[entry["path"]]
            assert hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest() == entry["git_blob_sha"]
            target = root / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            sources[entry["path"]] = value.decode()
    return root, sources


def runtime(primary, *, sink=None, manager_base=object, pool_setup=None, group_setup=None,
            runner_callback=None, runner_retire=None, spec_setup=None, group_class=None,
            layer_names=None, capacity=8, coordinator_base=object, manager_setup=None, **limits):
    root, sources = primary
    facts, health = [], Health()
    observer = native.InitializationObserver(sink or (lambda value: facts.append(value) or True), health, **limits)
    Spec = named("FullAttentionSpec", "vllm.v1.kv_cache_interface", Fields)

    def spec():
        value = Spec(block_size=16, non_causal=False, prefix_cacheable=True, dcp_sharded=True,
                    tokens_per_state=1, sliding_window=None, attention_chunk_size=None,
                    page_size_padded=None, num_head_slots=None, state_content_bytes=None,
                    kv_quant_mode=Fields(name="NONE"))
        if spec_setup is not None:
            spec_setup(value)
        return value

    def cache_config():
        return Fields(num_blocks=capacity, kv_cache_layout="LBHNC", hash_block_size=None,
                      cache_hit_alignment_tokens=None, hisparse_host_num_blocks=None,
                      needs_kv_cache_zeroing=False,
                      kv_cache_groups=[Fields(layer_names=list(layer_names or ["model.layers.0.self_attn.attn"]), kv_cache_spec=spec(),
                                             is_eagle_group=False, host_resident=False)])

    config = Fields(model_config=Fields(model="Qwen/Qwen3-0.6B", max_model_len=64, enforce_eager=True,
                                       architectures=["Qwen3ForCausalLM"]),
                    cache_config=Fields(enable_prefix_caching=False, cache_dtype="auto", kv_cache_layout="LBHNC",
                                        block_size=16, prefix_match_unit=None),
                    scheduler_config=Fields(enable_chunked_prefill=False),
                    parallel_config=Fields(tensor_parallel_size=1, pipeline_parallel_size=1,
                        decode_context_parallel_size=1, prefill_context_parallel_size=1,
                        data_parallel_size=1, ubatch_size=0, enable_dbo=False, worker_extension_cls=""),
                    compilation_config=Fields(compilation_time=0, encoder_compilation_time=0,
                        cudagraph_mode=Fields(name="NONE"), mode=Fields(name="NONE")),
                    speculative_config=None, lora_config=None, kv_transfer_config=None,
                    validate_block_size=lambda: None)

    core_module = ModuleType("vllm.v1.engine.core")
    manager_module = ModuleType("vllm.v1.core.kv_cache_manager")
    coordinator_module = ModuleType("vllm.v1.core.kv_cache_coordinator")
    utils_ns = {"copy": copy, "UniformTypeKVCacheSpecs": type("Uniform", (), {})}
    core_module.generate_scheduler_kv_cache_config = fragment(sources["vllm/v1/core/kv_cache_utils.py"],
            "generate_scheduler_kv_cache_config", utils_ns)
    core_module.resolve_kv_cache_block_sizes = fragment(sources["vllm/v1/core/kv_cache_utils.py"],
            "resolve_kv_cache_block_sizes", {})

    runner_copy_tree = ast.parse(sources["vllm/v1/worker/gpu/model_runner.py"])
    runner_cls = next(n for n in runner_copy_tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
    initialize = next(n for n in runner_cls.body if isinstance(n, ast.FunctionDef) and n.name == "initialize_kv_cache")
    statements = [copy.deepcopy(n) for n in initialize.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "kv_cache_config" or
                          isinstance(t, ast.Attribute) and t.attr == "kv_cache_config" for t in n.targets)]
    assert len(statements) == 2
    copy_code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])),
                        "<pinned native config copy statements>", "exec")

    def runner_initialize(self, kv_cache_config, is_profiling=False, kv_cache_allocation_context=None):
        (runner_retire or observer.retire_runner)(self, "cache initialization began")
        local = {"self": self, "kv_cache_config": kv_cache_config, "deepcopy": copy.deepcopy}
        exec(copy_code, local)
        self.block_tables = Fields(num_kv_cache_groups=1, block_sizes=[16], cp_size=1, cp_rank=0,
                                   redirect_writes_to_null_block=False)
        self.kernel_block_sizes = [16]
        self.attn_groups = [[Fields(layer_names=list(self.kv_cache_config.kv_cache_groups[0].layer_names),
            kv_cache_spec=self.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
            kv_cache_group_id=0, kernel_block_size=None, kernel_block_stride=0,
            kernel_block_table=None, kernel_block_offsets=None)]]
        if self.alter is not None:
            self.alter(self)
        if not is_profiling:
            (runner_callback or observer.runner_initialized)(Fields(runner=self, arguments=(
                ("kv_cache_config", kv_cache_config), ("is_profiling", is_profiling),
                ("kv_cache_allocation_context", kv_cache_allocation_context)),
                runner_config=self.kv_cache_config, block_tables=self.block_tables, cache_label="label-only"))

    Runner = named("GPUModelRunner", "vllm.v1.worker.gpu.model_runner", initialize_kv_cache=runner_initialize,
                   prepare_attn=lambda self: None, execute_model=lambda self, *a: None, shutdown=lambda self: None)
    worker_ns = dict(ensure_kv_transfer_initialized=lambda *a: None, has_kv_transfer_group=lambda: False,
                    nullcontext=nullcontext, current_platform=Fields(is_xpu=lambda: False),
                    record_kv_cache_layout=lambda cache, layout: setattr(cache, "kv_cache_layout", layout))
    worker_initialize = fragment(sources["vllm/v1/worker/gpu_worker.py"], "initialize_from_config", worker_ns, "Worker")
    Worker = named("Worker", "vllm.v1.worker.gpu_worker", initialize_from_config=worker_initialize,
                   _maybe_get_memory_pool_context=lambda self, tag: nullcontext())
    wrapper_initialize = fragment(sources["vllm/v1/worker/worker_base.py"], "initialize_from_config",
                                 {"set_current_vllm_config": lambda c: nullcontext()}, "WorkerWrapperBase")
    Wrapper = named("WorkerWrapperBase", "vllm.v1.worker.worker_base", initialize_from_config=wrapper_initialize)
    executor_initialize = fragment(sources["vllm/v1/executor/abstract.py"], "initialize_from_config", {}, "Executor")
    Executor = named("Executor", "vllm.v1.executor.abstract", initialize_from_config=executor_initialize)
    run_method = fragment(sources["vllm/v1/serial_utils.py"], "run_method", {})
    collective = fragment(sources["vllm/v1/executor/uniproc_executor.py"], "collective_rpc",
                         dict(run_method=run_method, AsyncModelRunnerOutput=type("Async", (), {}), Future=Future), "UniProcExecutor")
    UniProc = named("UniProcExecutor", "vllm.v1.executor.uniproc_executor", Executor, collective_rpc=collective,
                   get_kv_cache_specs=lambda self: [{"layer": spec()}],
                   get_supported_kv_cache_layouts=lambda self: [["LBHNC"]],
                   set_kv_cache_layout=lambda self, name: None,
                   determine_available_memory=lambda self: [1000],
                   compile_or_warm_up_model=lambda self: None)
    worker, wrapper, runner, executor = Worker.__new__(Worker), Wrapper(), Runner(), UniProc()
    runner.alter = None
    worker.vllm_config, worker.cache_config, worker.model_runner, worker.use_v2_model_runner = config, config.cache_config, runner, True
    wrapper.worker, wrapper.global_rank, wrapper.rpc_rank, wrapper.vllm_config = worker, 0, 0, config
    executor.driver_worker, executor.vllm_config = wrapper, config

    def pool_init(self, num_gpu_blocks, enable_caching, hash_block_size):
        self.num_gpu_blocks, self.enable_caching, self.hash_block_size = num_gpu_blocks, enable_caching, hash_block_size
        self.blocks = [Fields(block_id=i, is_null=i == 0, pool=self) for i in range(num_gpu_blocks)]
        self.null_block = self.blocks[0]
        if pool_setup is not None:
            pool_setup(self)

    Pool = named("BlockPool", "vllm.v1.core.block_pool", __init__=pool_init)
    def group_init(self, kv_cache_spec, block_pool, **kwargs):
        self.kv_cache_spec, self.block_pool, self.block_size = kv_cache_spec, block_pool, kv_cache_spec.block_size
        if group_setup is not None:
            group_setup(self)
    Group = group_class or named("FullAttentionManager", "vllm.v1.core.single_type_kv_cache_manager", manager_base, __init__=group_init)
    if group_class is not None:
        Group.__init__ = group_init
    coordinator_module.get_manager_for_kv_cache_spec = fragment(
        sources["vllm/v1/core/single_type_kv_cache_manager.py"], "get_manager_for_kv_cache_spec",
        dict(KVCacheSpecRegistry=Fields(get_manager_class=lambda spec, role: Group),
             SlidingWindowSpec=type("Sliding", (), {}), ChunkedLocalAttentionSpec=type("Chunked", (), {}),
             HiSparseResidentSpec=type("Sparse", (), {})))
    def coordinator_init(self, kv_cache_config):
        self.kv_cache_config = kv_cache_config
        self.block_pool = Pool(kv_cache_config.num_blocks, False, kv_cache_config.hash_block_size)
        self.single_type_managers = (coordinator_module.get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_config.kv_cache_groups[0].kv_cache_spec, block_pool=self.block_pool,
            max_in_flight_tokens=64, max_model_len=64),)
    Coordinator = named("KVCacheCoordinatorNoPrefixCache", "vllm.v1.core.kv_cache_coordinator", coordinator_base, __init__=coordinator_init)
    manager_module.get_kv_cache_coordinator = lambda kv_cache_config: Coordinator(kv_cache_config)
    def manager_init(self, kv_cache_config):
        self.kv_cache_config, self.enable_caching = kv_cache_config, False
        self.coordinator = manager_module.get_kv_cache_coordinator(kv_cache_config=kv_cache_config)
        self.block_pool = self.coordinator.block_pool
        if manager_setup is not None:
            manager_setup(self)
    Manager = named("KVCacheManager", "vllm.v1.core.kv_cache_manager", __init__=manager_init)
    def scheduler_init(self, vllm_config, kv_cache_config, block_size, hash_block_size=None):
        self.vllm_config, self.kv_cache_config = vllm_config, kv_cache_config
        self.block_size, self.hash_block_size = block_size, hash_block_size
        self.kv_cache_manager = Manager(kv_cache_config)
        if getattr(self, "alter_allocator", None) is not None:
            self.alter_allocator()
    Scheduler = named("Scheduler", "vllm.v1.core.sched.scheduler", __init__=scheduler_init)
    def async_init(self, *args, **kwargs):
        Scheduler.__init__(self, *args, **kwargs)
        if getattr(self, "raise_after_base", None) is not None:
            raise self.raise_after_base
        self.async_initialized = True
    Async = named("AsyncScheduler", "vllm.v1.core.sched.async_scheduler", Scheduler, __init__=async_init)
    core_module.__dict__.update(time=__import__("time"), envs=Fields(VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=False),
        register_all_kvcache_specs=lambda c: None, resolve_kv_cache_layout=lambda *a: Fields(name="LBHNC"),
        get_kv_cache_configs=lambda *a: [cache_config()], update_kv_cache_capacity=lambda *a: None,
        resolve_cache_hit_alignment_tokens=lambda *a: 16,
        logger=Fields(info_once=lambda *a, **k: None, info=lambda *a, **k: None))
    core_initialize = fragment(sources["vllm/v1/engine/core.py"], "_initialize_kv_caches", core_module.__dict__, "EngineCore")
    Core = named("EngineCore", "vllm.v1.engine.core", _initialize_kv_caches=core_initialize)
    CoreProc = named("EngineCoreProc", "vllm.v1.engine.core", Core)
    core = Core()
    core.vllm_config, core.model_executor, core.scheduler = config, executor, None
    bindings = native.NativeBindings(core_module, manager_module, coordinator_module, Core, CoreProc,
        Executor, UniProc, Wrapper, Worker, Runner, Scheduler, Async, Manager, Coordinator, Group, Pool)
    observer.attach(bindings, root, "0.32.0")
    return Fields(observer=observer, facts=facts, health=health, config=config, core=core, runner=runner,
                  worker=worker, wrapper=wrapper, executor=executor, bindings=bindings,
                  source_root=root, make_cache=cache_config, spec=spec)


@pytest.fixture
def system(primary):
    value = runtime(primary)
    yield value
    value.observer.detach()


def initialize(s, asynchronous=False):
    config = s.core._initialize_kv_caches(s.config)
    # Real core resolves once more after its init method returns.
    size, hashed = s.bindings.core_module.resolve_kv_cache_block_sizes(config, s.config)
    cls = s.bindings.async_scheduler if asynchronous else s.bindings.scheduler
    scheduler = cls(vllm_config=s.config, kv_cache_config=config, block_size=size, hash_block_size=hashed)
    s.core.scheduler = scheduler
    return scheduler


def test_lookup_rechecks_one_actual_receiver_and_returns_only_weak_participants(system):
    scheduler = initialize(system)
    by_scheduler = system.observer.lookup(scheduler=scheduler)
    by_runner = system.observer.lookup(runner=system.runner)
    assert by_scheduler is not None and by_runner is not None
    fact, refs = by_scheduler
    assert fact.record() == by_runner[0].record()
    assert refs["scheduler"]() is scheduler and refs["runner"]() is system.runner
    assert all(isinstance(ref, weakref.ReferenceType) for ref in refs.values())
    with pytest.raises(TypeError):
        refs["scheduler"] = weakref.ref(scheduler)
    with pytest.raises(TypeError):
        fact.fields["init"] = "changed"


@pytest.mark.parametrize("case", ["no-receiver", "both", "foreign", "replaced", "retired", "ambiguous"])
def test_lookup_rejects_missing_ambiguous_or_stale_native_association(system, case):
    scheduler = initialize(system)
    if case == "no-receiver":
        result = system.observer.lookup()
    elif case == "both":
        result = system.observer.lookup(scheduler=scheduler, runner=system.runner)
    elif case == "foreign":
        result = system.observer.lookup(runner=Fields())
    else:
        if case == "replaced":
            system.core.scheduler = Fields()
        elif case == "retired":
            system.observer.retire_runner(system.runner, "native shutdown")
        else:
            item = next(iter(system.observer._inits.values()))
            duplicate = copy.copy(item)
            duplicate.token = system.observer.registry.new_token()
            system.observer._inits[duplicate.token] = duplicate
        result = system.observer.lookup(scheduler=scheduler)
    assert result is None and system.observer.status()["ready_associations"] == 0


def test_lookup_does_not_keep_a_replaced_native_runner_alive(system):
    scheduler = initialize(system)
    found = system.observer.lookup(scheduler=scheduler)
    reference = weakref.ref(system.runner)
    system.worker.model_runner = None
    system.runner = None
    gc.collect()
    assert reference() is None and found[1]["runner"]() is None
    assert system.observer.lookup(scheduler=scheduler) is None


def test_lookup_cannot_revive_an_earlier_cache_association(system):
    earlier = initialize(system)
    later = initialize(system)
    assert later is not earlier
    assert system.observer.lookup(scheduler=earlier) is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_pinned_core_dispatch_copy_and_constructor_chain(system, asynchronous):
    s = system
    scheduler = initialize(s, asynchronous)
    assert not s.health.errors
    assert [f.kind for f in s.facts] == ["init_begin", "configs_derived", "worker_init", "runner_ready", "init_return", "scheduler_ready"]
    records = [f.record() for f in s.facts]
    assert "kernel_block_size" not in records[1]["dimensions"]
    assert "ratio" not in records[1]["dimensions"]
    assert records[1]["dimensions"]["hash_block_size"] == 16
    assert records[3]["kernel_block_size"] == 16 and records[3]["ratio"] == 1
    assert records[1]["worker_config"] != records[1]["scheduler_config"] != records[3]["runner_config"]
    binding = s.observer.binding_for(s.core, scheduler, s.runner)
    assert binding is not None and binding.record()["ownership"] == "unknown"
    assert s.observer.status()["frame_assembler"] == "absent"
    assert contract.evaluate(records)["verdict"] == "unknown"


def test_no_runner_methods_or_model_hooks_are_wrapped(system):
    cls = system.bindings.runner
    assert not any(getattr(getattr(cls, name), "_initialization_observer", None) for name in
                   ("initialize_kv_cache", "prepare_attn", "execute_model", "shutdown"))
    assert not hasattr(system.runner, "_forward_pre_hooks")


def test_source_native_no_microbatch_defaults_and_explicit_one_both_work(system):
    assert system.config.parallel_config.ubatch_size == 0
    assert system.config.parallel_config.worker_extension_cls == ""
    system.config.parallel_config.ubatch_size = 1
    scheduler = initialize(system)
    assert system.observer.binding_for(system.core, scheduler, system.runner) is not None


def test_sink_receives_immutable_host_only_independent_facts(system):
    initialize(system)
    fact = system.facts[1]
    with pytest.raises(TypeError):
        fact.fields["dimensions"]["capacity"] = 99
    with pytest.raises(TypeError):
        fact.fields["dimensions"]["layer_names"][0] = "changed"
    value = fact.record()
    value["dimensions"]["layer_names"][0] = "independent"
    assert fact.record()["dimensions"]["layer_names"][0] != "independent"


@pytest.mark.parametrize("change", [
    lambda s: setattr(s.config.cache_config, "enable_prefix_caching", True),
    lambda s: setattr(s.config.model_config, "enforce_eager", False),
    lambda s: setattr(s.config.parallel_config, "tensor_parallel_size", 2),
    lambda s: setattr(s.config.parallel_config, "ubatch_size", 2),
    lambda s: setattr(s.config.parallel_config, "worker_extension_cls", "extension"),
    lambda s: setattr(s.config.compilation_config.cudagraph_mode, "name", "FULL"),
    lambda s: setattr(s.wrapper, "global_rank", False),
])
def test_unsupported_profile_never_becomes_ready(system, change):
    change(system)
    system.core._initialize_kv_caches(system.config)
    assert system.health.errors
    assert system.observer.status()["ready_associations"] == 0


@pytest.mark.parametrize("change", [
    lambda r: setattr(r, "kernel_block_sizes", [8]),
    lambda r: setattr(r.attn_groups[0][0], "kernel_block_size", 8),
    lambda r: setattr(r.attn_groups[0][0], "kernel_block_stride", 2),
    lambda r: setattr(r.attn_groups[0][0], "kernel_block_table", object()),
    lambda r: setattr(r.attn_groups[0][0], "layer_names", ["extra"]),
    lambda r: setattr(r.block_tables, "redirect_writes_to_null_block", True),
    lambda r: setattr(r.kv_cache_config, "num_blocks", 10),
])
def test_actual_runner_results_are_checked(system, change):
    system.runner.alter = change
    system.core._initialize_kv_caches(system.config)
    assert system.health.errors
    assert "runner_ready" not in [f.kind for f in system.facts]


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_original_exception_identity_and_args_survive(system, exception_type):
    exception = exception_type("host exception")
    seen = []
    def raises():
        seen.append(True)
        raise exception
    system.executor.compile_or_warm_up_model = raises
    with pytest.raises(exception_type) as caught:
        system.core._initialize_kv_caches(system.config)
    assert caught.value is exception and seen == [True]
    assert "init_return" not in [f.kind for f in system.facts]
    assert system.observer.status()["ready_associations"] == 0


def test_profile_initialization_does_not_emit_ready(system):
    before = len(system.facts)
    system.runner.initialize_kv_cache(system.make_cache(), is_profiling=True)
    assert len(system.facts) == before
    scheduler = initialize(system)
    assert system.observer.binding_for(system.core, scheduler, system.runner) is not None


def test_missing_bridge_callback_cannot_complete_outer_init(system):
    original = system.observer.runner_initialized
    system.observer.runner_initialized = lambda boundary: None
    system.core._initialize_kv_caches(system.config)
    system.observer.runner_initialized = original
    assert "init_return" not in [f.kind for f in system.facts]


def test_outer_subclass_constructor_failure_is_not_ready(system):
    exception = RuntimeError("ordinary host constructor failure")
    system.bindings.async_scheduler.raise_after_base = exception
    with pytest.raises(RuntimeError) as caught:
        initialize(system, asynchronous=True)
    assert caught.value is exception
    assert "scheduler_ready" not in [f.kind for f in system.facts]


def test_actual_reserved_null_identity_is_checked(system):
    def alter(self):
        self.kv_cache_manager.block_pool.null_block = Fields(block_id=0, is_null=True)
    system.bindings.scheduler.alter_allocator = alter
    initialize(system)
    assert "scheduler_ready" not in [f.kind for f in system.facts]


def test_missing_copy_edge_is_unknown(system):
    def alter(runner):
        item = system.observer._parent()
        runner.kv_cache_config = item.refs["worker_config"]()
    system.runner.alter = alter
    system.core._initialize_kv_caches(system.config)
    assert "runner_ready" not in [f.kind for f in system.facts]


def test_valid_historical_worker_config_collection_does_not_break_binding(system):
    scheduler = initialize(system)
    item = next(iter(system.observer._inits.values()))
    gc.collect()
    assert item.refs["worker_config"]() is None
    assert system.observer.binding_for(system.core, scheduler, system.runner) is not None


def test_actual_core_attachment_is_required(system):
    scheduler = initialize(system)
    system.core.scheduler = object()
    assert system.observer.binding_for(system.core, scheduler, system.runner) is None


def test_reinitialization_retires_old_binding_without_label_reuse(system):
    old = initialize(system)
    old_cache = system.facts[3].record()["worker_pool"]
    new = initialize(system)
    new_cache = [f.record()["worker_pool"] for f in system.facts if f.kind == "runner_ready"][-1]
    assert old_cache != new_cache
    assert system.observer.binding_for(system.core, new, system.runner) is not None
    assert not next(iter(system.observer._inits.values())).ready
    assert old is not new


def test_unsupported_replacement_cannot_leave_old_binding_ready(system):
    initialize(system)
    system.core.model_executor = object()
    with pytest.raises(AttributeError):
        system.core._initialize_kv_caches(system.config)
    assert system.observer.status()["ready_associations"] == 0


def test_later_overlapping_init_cannot_revive_earlier_parent(system):
    s = system
    original_warmup = s.executor.compile_or_warm_up_model
    entered = []
    def nested_host_lifecycle():
        if not entered:
            entered.append(True)
            s.core._initialize_kv_caches(s.config)
        return original_warmup()
    s.executor.compile_or_warm_up_model = nested_host_lifecycle
    s.core._initialize_kv_caches(s.config)
    assert len(s.observer._inits) == 2
    first = next(iter(s.observer._inits.values()))
    assert first.poisoned and first.phase != "returned"
    assert s.observer.status()["ready_associations"] == 0


def test_cache_replacement_under_another_core_invalidates_first(system):
    initialize(system)
    other = system.bindings.core()
    other.vllm_config, other.model_executor = system.config, system.executor
    other._initialize_kv_caches(system.config)
    first = next(iter(system.observer._inits.values()))
    assert first.poisoned and not first.ready


def test_changed_wrapper_is_detected_and_not_removed_on_detach(system):
    initialize(system)
    replacement = lambda *a, **k: None
    system.bindings.core_module.resolve_kv_cache_block_sizes = replacement
    assert not system.observer.coverage()
    system.observer.detach()
    assert system.bindings.core_module.resolve_kv_cache_block_sizes is replacement


@pytest.mark.parametrize("change", ["delete", "replace"])
def test_status_rechecks_coverage_before_any_binding_request(system, change):
    initialize(system)
    assert system.observer.status()["ready_associations"] == 1
    owner = system.bindings.coordinator_module
    name = "get_manager_for_kv_cache_spec"
    if change == "delete":
        delattr(owner, name)
    else:
        setattr(owner, name, lambda *args, **kwargs: None)
    assert system.observer.coverage() is False
    assert system.observer.status()["ready_associations"] == 0
    assert not system.health.errors


def test_deleted_last_alias_does_not_stop_remaining_owned_cleanup(system):
    initialize(system)
    hooks = list(system.observer._hooks)
    owner, name, _, _, _ = hooks[-1]
    delattr(owner, name)
    system.observer.detach()
    assert name not in vars(owner)
    assert system.observer._closed and not system.observer._hooks
    assert system.observer.status()["ready_associations"] == 0
    for current_owner, current_name, original, _, own in hooks[:-1]:
        if own:
            assert getattr(current_owner, current_name) is original
        else:
            assert current_name not in vars(current_owner)
    system.observer.detach()


def test_cleanup_failure_does_not_stop_other_restorations_or_close(system):
    initialize(system)
    hooks = list(system.observer._hooks)
    blocked_owner, blocked_name, original, installed, _ = hooks[-2]

    class BlockRestore(ModuleType):
        def __setattr__(self, name, value):
            if name == blocked_name and value is original:
                raise RuntimeError("ordinary observer cleanup failure")
            return super().__setattr__(name, value)

    blocked_owner.__class__ = BlockRestore
    try:
        system.observer.detach()
        assert system.observer._closed and not system.observer._hooks
        assert system.health.errors
        assert system.observer.status()["ready_associations"] == 0
        assert getattr(blocked_owner, blocked_name) is installed
        for owner, name, before, _, own in hooks:
            if owner is blocked_owner and name == blocked_name:
                continue
            if own:
                assert getattr(owner, name) is before
            else:
                assert name not in vars(owner)
    finally:
        blocked_owner.__class__ = ModuleType
        setattr(blocked_owner, blocked_name, original)


def test_original_attach_exception_survives_deleted_alias_during_rollback(system):
    hooks = list(system.observer._hooks)
    system.observer.detach()
    owner = system.bindings.coordinator_module
    deleted_owner = system.bindings.manager_module
    deleted_name = "get_kv_cache_coordinator"
    exception = RuntimeError("ordinary observer installation failure")

    class RejectInstall(ModuleType):
        def __setattr__(self, name, value):
            if name == "get_manager_for_kv_cache_spec" and hasattr(value, "_initialization_observer"):
                delattr(deleted_owner, deleted_name)
                raise exception
            return super().__setattr__(name, value)

    owner.__class__ = RejectInstall
    other = native.InitializationObserver(lambda fact: True, Health())
    try:
        with pytest.raises(RuntimeError) as caught:
            other.attach(system.bindings, system.source_root, "0.32.0")
        assert caught.value is exception
        assert other._closed and not other._hooks
        assert deleted_name not in vars(deleted_owner)
        for current_owner, name, original, _, own in hooks:
            if current_owner is deleted_owner and name == deleted_name:
                continue
            if own:
                assert getattr(current_owner, name) is original
            else:
                assert name not in vars(current_owner)
    finally:
        owner.__class__ = ModuleType
        other.detach()


def test_detach_restores_owned_hooks_only(system):
    hooks = list(system.observer._hooks)
    system.observer.detach()
    for owner, name, original, installed, own in hooks:
        if own:
            assert getattr(owner, name) is original
        else:
            assert name not in vars(owner)
    system.observer.detach()


def test_live_native_objects_are_not_held_by_completed_observer(primary):
    s = runtime(primary)
    initialize(s)
    observer = s.observer
    references = [weakref.ref(s.core), weakref.ref(s.runner), weakref.ref(s.worker)]
    del s
    gc.collect()
    assert all(reference() is None for reference in references)
    observer.detach()


@pytest.mark.parametrize("limit,value", [("maximum_facts", 2), ("maximum_identities", 3)])
def test_observer_bounds_fail_closed_without_changing_native_return(primary, limit, value):
    s = runtime(primary, **{limit: value})
    try:
        result = s.core._initialize_kv_caches(s.config)
        assert result.num_blocks == 8
        assert s.health.errors and s.observer.status()["ready_associations"] == 0
    finally:
        s.observer.detach()


def test_rejected_sink_marks_health_without_replacing_native_result(primary):
    s = runtime(primary, sink=lambda fact: False)
    try:
        assert s.core._initialize_kv_caches(s.config).num_blocks == 8
        assert s.health.errors
    finally:
        s.observer.detach()


def test_raising_sink_marks_health_without_replacing_native_result(primary):
    def raises(fact):
        raise RuntimeError("host recorder failure")
    s = runtime(primary, sink=raises)
    try:
        assert s.core._initialize_kv_caches(s.config).num_blocks == 8
        assert s.health.errors
    finally:
        s.observer.detach()


def test_source_pin_change_is_rejected_before_any_wrapper(system, tmp_path):
    metadata = json.loads((FIXTURE / "SOURCE.json").read_text())
    for entry in metadata["files"]:
        path = tmp_path / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((system.source_root / entry["path"]).read_bytes())
    path = tmp_path / metadata["files"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    other = native.InitializationObserver(lambda fact: True, Health())
    with pytest.raises(native.InitializationGap, match="source pin"):
        other.attach(system.bindings, tmp_path, "0.32.0")
    assert not other._hooks


def test_no_late_attachment_to_an_existing_cache_is_invented(primary):
    s = runtime(primary)
    try:
        scheduler = initialize(s)
        s.observer.detach()
        other = native.InitializationObserver(lambda fact: True, Health())
        other.attach(s.bindings, s.source_root, "0.32.0")
        assert other.binding_for(s.core, scheduler, s.runner) is None
        other.detach()
    finally:
        s.observer.detach()


def test_partial_hook_installation_rolls_back_owned_methods(system):
    system.observer.detach()
    core_original = system.bindings.core._initialize_kv_caches
    system.bindings.coordinator_module.get_manager_for_kv_cache_spec = None
    other = native.InitializationObserver(lambda fact: True, Health())
    with pytest.raises(native.InitializationGap):
        other.attach(system.bindings, system.source_root, "0.32.0")
    assert not other._hooks
    assert system.bindings.core._initialize_kv_caches is core_original


def test_initialization_bound_retires_previous_live_association(primary):
    s = runtime(primary, maximum_initializations=1)
    try:
        initialize(s)
        s.core._initialize_kv_caches(s.config)
        assert s.observer.status()["ready_associations"] == 0
    finally:
        s.observer.detach()


def test_identity_registry_ignores_equality_and_hashing():
    class Equal:
        __hash__ = None
        def __eq__(self, other):
            raise AssertionError("native equality must not run")
    registry = native.IdentityRegistry(8)
    left, right = Equal(), Equal()
    first = registry.identify(left, "native")
    assert registry.identify(left, "native") == first
    assert registry.identify(right, "native") != first
    reference = weakref.ref(left)
    del left
    gc.collect()
    assert reference() is None
    assert registry.identify(Equal(), "native") != first


def test_identity_registry_does_not_retain_objects_or_reuse_roles():
    registry = native.IdentityRegistry(8)
    obj = Fields()
    registry.identify(obj, "one")
    with pytest.raises(native.InitializationGap, match="role"):
        registry.identify(obj, "two")
    with pytest.raises(native.InitializationGap, match="weak-referenceable"):
        registry.identify([], "list")


def test_forked_registry_never_uses_inherited_lock(monkeypatch):
    registry = native.IdentityRegistry(8)
    monkeypatch.setattr(native.os, "getpid", lambda: registry.pid + 1)
    with pytest.raises(native.InitializationGap, match="process changed"):
        registry.identify(Fields(), "native")


def test_no_offline_import_of_engine_or_device_modules():
    import subprocess
    import sys
    code = ('import sys; from sdcguard.lifetime import assignment_init; '
            'assert not any(n == "torch" or n == "vllm" or n.startswith("vllm.") for n in sys.modules)')
    assert subprocess.run([sys.executable, "-B", "-c", code], capture_output=True).returncode == 0
