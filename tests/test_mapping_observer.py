"""Host assembly and retained primary fragments, never engine or GPU runs.

Owner: Hari Krishna Kancharla.
"""
import ast
from collections import defaultdict, deque
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
from enum import Enum
import gc
import json
from pathlib import Path
import threading
from types import MappingProxyType, SimpleNamespace
import weakref

import pytest

from sdcguard.lifetime import assignment_obligations as C
from sdcguard.lifetime import assignment_recording as J
from sdcguard.lifetime import forward_bridge as B
from sdcguard.lifetime import mapping_observer as M
from sdcguard.lifetime import mapping_runtime as R
from sdcguard.lifetime import vllm_forward as V
from test_assignment_init import Fields, fragment, initialize, named, primary, runtime
from test_mapping_runtime import header_environment
from test_vllm_forward import Tensor


class Mode(Enum):
    NONE = 0


class MemorySink:
    """The same envelope/accounting protocol, confined to immutable host data."""
    def __init__(self, **fields):
        self.health = fields.pop("health")
        process = fields.pop("process_incarnation")
        self.records = [dict(schema=1, session="f" * 32, process=process, seq=0, kind="header",
                             pid=__import__("os").getpid(), **deepcopy(fields))]
        self.closed = self.published = False
        self.errors = 0
        self.reject = None
        self.session, self.process = self.records[0]["session"], process

    def emit(self, kind, **facts):
        if self.closed or kind == self.reject:
            self.errors += 1
            return False
        self.records.append(dict(schema=1, session=self.session, process=self.process,
                                 seq=len(self.records), kind=kind, **deepcopy(facts)))
        return True

    def gap(self, reason):
        self.errors += 1
        self.emit("gap", reason=reason)

    def close(self, timeout=1.0):
        if not self.closed:
            self.closed = True
            errors = self.errors + self.health.snapshot()["errors"]
            self.records.append(dict(schema=1, session=self.session, process=self.process,
                seq=len(self.records), kind="footer", records=len(self.records) + 1,
                samples=sum(r["kind"] == "frame_end" for r in self.records),
                acquisitions=sum(r["kind"] == "acquire" for r in self.records),
                dropped=0, errors=errors, healthy=not errors))
            self.published = not errors
        return self.published

    @property
    def status(self):
        return dict(published=self.published, healthy=not self.errors, closed=self.closed,
                    sealed=self.closed, footer_written=self.closed)


def source_class(source, name, methods, namespace, *, base=object):
    tree = ast.parse(source)
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    nodes = [deepcopy(n) for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in nodes} == set(methods)
    for node in nodes:
        node.decorator_list = []
    node = ast.ClassDef(name=name, bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=nodes, decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    namespace.update(Base=base, __name__="vllm.v1.core.single_type_kv_cache_manager")
    exec(compile(ast.fix_missing_locations(module), "<retained primary class methods>", "exec"), namespace)
    return namespace[name]


def observer_environment(primary, tmp_path, *, journal=False, two_layers=False, installation_builder=None,
                         request_count=1, capacity=8, recipient_factory=None, runtime_options=None):
    source_root, sources = primary
    fixture = Path(__file__).parent / "fixtures/assignment_native/request.txt"
    target = source_root / "vllm/v1/request.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fixture.read_bytes())
    request_tree = ast.parse(fixture.read_bytes())
    request_class = deepcopy(next(n for n in request_tree.body if isinstance(n, ast.ClassDef) and n.name == "Request"))
    request_module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), request_class], type_ignores=[])
    request_namespace = {"__name__": "vllm.v1.request"}
    exec(compile(ast.fix_missing_locations(request_module), "<retained Request definition, initializer unused>", "exec"), request_namespace)
    Request = request_namespace["Request"]
    Output = named("SchedulerOutput", "vllm.v1.core.sched.output")
    base_names = {label.split(".")[1] for label in C.adapter_contract()["mutation_methods"]
                  if label.startswith("SingleTypeKVCacheManager.")}
    base_names.update(label.split(".")[1] for label in C.adapter_contract()["guard_methods"]
                      if label.startswith("SingleTypeKVCacheManager."))
    ns = dict(cdiv=lambda a, b: (a + b - 1) // b)
    source = sources["vllm/v1/core/single_type_kv_cache_manager.py"]
    Base = source_class(source, "SingleTypeKVCacheManager", base_names, ns)
    Base.get_num_skipped_tokens = lambda self, tokens: 0
    Base._has_partial_local_hit = lambda *a: False
    Base.reachable_block_mask = lambda *a, **k: None
    Group = source_class(source, "FullAttentionManager", {"cache_blocks", "_cache_partial_tail_block"}, dict(ns), base=Base)

    class Queue:
        def __init__(self, blocks): self.blocks = deque(blocks)
        def popleft_n(self, n): return [self.blocks.popleft() for _ in range(n)]
        def append_n(self, values): self.blocks.extend(values)
        def prepend_n(self, values): self.blocks.extendleft(values)
        def remove(self, block): self.blocks.remove(block)

    def pool_setup(pool):
        pool._reuse_watchers = {}
        pool.metrics_collector = None
        pool.enable_kv_cache_events = False
        pool.cached_block_hash_to_block = {}
        pool.cached_block_hashes_by_block = {}
        for block in pool.blocks:
            block.ref_cnt, block.block_hash, block.reset_hash = 0, None, lambda: None
        pool.free_block_queue = Queue(pool.blocks[1:])

    def group_setup(group):
        group.req_to_blocks = defaultdict(list)
        group.num_cached_block, group._partial_hit_reqs = {}, {}
        group.new_block_ids, group._pending_cow_copies = [], []
        group._record_new_block_ids = group.enable_caching = group.use_eagle = False
        group._null_block = group.block_pool.null_block
        group.kv_cache_group_id, group.dcp_world_size = 0, 1
        group.cache_hit_alignment_tokens = group.scheduler_block_size = 16

    def spec_setup(spec):
        spec.kv_quant_mode = Mode.NONE
        spec.num_kv_heads, spec.head_size, spec.head_size_v, spec.dtype = 2, 4, 4, "bfloat16"

    layer_names = ["model.layers.0.self_attn.attn"] + (["model.layers.1.self_attn.attn"] if two_layers else [])
    value = runtime(primary, manager_base=Base, group_class=Group, pool_setup=pool_setup,
                    group_setup=group_setup, spec_setup=spec_setup,
                    runner_callback=lambda boundary: None, runner_retire=lambda *a: None,
                    layer_names=layer_names, capacity=capacity, **(runtime_options or {}))
    template_dir = tmp_path / "header-template"; template_dir.mkdir()
    template = header_environment(template_dir)
    runner, config = value.runner, value.config
    config.use_v2_model_runner = True
    config.kernel_config = Fields(enable_jit_warmup=False)
    config.model_config.revision = "c" * 40
    config.model_config.rswa_window = None
    config.model_config.hf_text_config = Fields(num_hidden_layers=len(layer_names))
    config.compilation_config.mode = config.compilation_config.cudagraph_mode = Mode.NONE
    layer_name = "model.layers.0.self_attn.attn"
    config.compilation_config.static_forward_context = {name: template.runner.compilation_config.static_forward_context[name] for name in layer_names}
    for layer in config.compilation_config.static_forward_context.values():
        layer.kv_cache = Tensor(shape=(capacity, 2, 16, 8), dtype="bfloat16", device=template.runner.device)
    request_names = ["request-a"] if request_count == 1 else ["request-" + str(i) for i in range(request_count)]
    runner.__dict__.update(model=template.runner.model, model_state=template.runner.model_state,
        vllm_config=config, model_config=config.model_config, parallel_config=config.parallel_config,
        cache_config=config.cache_config, compilation_config=config.compilation_config,
        speculative_config=None, lora_config=None, ubatch_runner=None, pcp_manager=None,
        is_encoder_decoder=False, supports_mm_inputs=False, device=template.runner.device,
        batch=Fields(num_reqs=request_count, num_tokens=request_count, num_draft_tokens=0, req_ids=request_names),
        execute_calls=0, shutdown_result=object())
    profile_types = dict(template.bindings.profile_types)
    profile_types.update(spec=type(value.spec()), compile_none=Mode.NONE, graph_none=Mode.NONE, quant_none=Mode.NONE)
    profile_types["global_hook_registry"] = Fields(_global_forward_pre_hooks={})
    def alter(actual):
        actual.attn_groups = [[profile_types["attention_group"](
            backend=profile_types["backend"], kv_cache_group_id=0, kernel_block_size=None, kernel_block_stride=0,
            kernel_block_table=None, kernel_block_offsets=None, kv_cache_spec=actual.kv_cache_config.kv_cache_groups[0].kv_cache_spec,
            layer_names=list(layer_names))]]
    runner.alter = alter
    for label in C.adapter_contract()["mutation_methods"] | C.adapter_contract()["guard_methods"]:
        if label.startswith("BlockPool."):
            name = label.split(".")[1]
            function = fragment(sources["vllm/v1/core/block_pool.py"], name,
                dict(BlockHashToBlockMap=dict, logger=Fields(warning=lambda *a: None, info=lambda *a: None)), "BlockPool")
            setattr(value.bindings.pool, name, function)
    value.bindings.pool.get_num_free_blocks = lambda self: len(self.free_block_queue.blocks)
    value.bindings.pool._notify_reuse = fragment(sources["vllm/v1/core/block_pool.py"], "_notify_reuse", {}, "BlockPool")
    value.bindings.pool.cache_full_blocks = lambda *a, **k: None
    value.bindings.coordinator.get_blocks = fragment(sources["vllm/v1/core/kv_cache_coordinator.py"], "get_blocks", {}, "KVCacheCoordinator")
    def schedule(scheduler):
        scheduler.current_step += 1
        if scheduler.schedule_error is not None:
            raise scheduler.schedule_error
        for request in scheduler.requests.values():
            scheduler.kv_cache_manager.coordinator.single_type_managers[0].allocate_new_blocks(request.request_id, 18, 18)
        return Output(num_scheduled_tokens={name: 1 for name in scheduler.requests},
                      total_num_scheduled_tokens=len(scheduler.requests))
    # The stand-in output initializer stores only host scalar fields.
    Output.__init__ = lambda self, **fields: self.__dict__.update(fields)
    value.bindings.scheduler.schedule = schedule
    value.bindings.wrapper._apply_mm_cache = fragment(sources["vllm/v1/worker/worker_base.py"], "_apply_mm_cache", {}, "WorkerWrapperBase")
    value.bindings.wrapper.execute_model = fragment(sources["vllm/v1/worker/worker_base.py"], "execute_model", {}, "WorkerWrapperBase")
    value.wrapper.mm_receiver_cache = None
    value.worker._pp_send_work = []
    value.worker.annotate_profile = lambda output: nullcontext()
    value.bindings.worker.execute_model = fragment(sources["vllm/v1/worker/gpu_worker.py"], "execute_model",
        dict(get_pp_group=lambda: Fields(is_first_rank=True, is_last_rank=True),
             ModelRunnerOutput=type(runner.model.result), AsyncModelRunnerOutput=type("Async", (), {}),
             NoneType=type(None)), "Worker")
    value.bindings.uniproc.execute_model = fragment(sources["vllm/v1/executor/uniproc_executor.py"], "execute_model", {}, "UniProcExecutor")
    def prepare(actual, batch): return batch
    def execute(actual, scheduler_output, intermediate_tensors=None, dummy_run=False, is_profile=False):
        actual.execute_calls += 1
        if scheduler_output.total_num_scheduled_tokens:
            actual.prepare_attn(actual.batch)
            return actual.model()
        return actual.model.result
    def shutdown(actual):
        for name in ("model", "model_state", "kv_cache_config"):
            delattr(actual, name)
        return actual.shutdown_result
    type(runner).prepare_attn, type(runner).execute_model, type(runner).shutdown = prepare, execute, shutdown
    runner.is_pooling_model = False
    context = template.env.context
    context.attn_metadata = {name: context.attn_metadata[name] for name in layer_names}
    context.slot_mapping = {name: Tensor([(2 * i + 2) * 16 + 1 for i in range(request_count)], device=runner.device)
                            for name in layer_names}
    context.no_compile_layers = config.compilation_config.static_forward_context
    context.is_padding = Tensor([False] * request_count, dtype="bool", device=runner.device)
    context.cudagraph_runtime_mode = Mode.NONE
    for metadata in context.attn_metadata.values():
        metadata.num_actual_tokens, metadata.max_query_len, metadata.max_seq_len = request_count, 1, 18
        metadata.block_table = Tensor([[2 * i + 1, 2 * i + 2] for i in range(request_count)], device=runner.device)
        metadata.seq_lens = Tensor([18] * request_count, device=runner.device)
        metadata.query_start_loc = Tensor(list(range(request_count + 1)), device=runner.device)
    sinks = []
    directory = tmp_path / "assignment-journal"; directory.mkdir()
    def sink_factory(**fields):
        sink = J.AssignmentRecorder(directory, **fields) if journal else MemorySink(**fields)
        sinks.append(sink)
        return sink
    bindings = R.MappingBindings(value.bindings, Base, Request, Output, profile_types)
    engine = dict(name="vllm", version="0.32.0", source_sha256=V.contract.source_pins(),
                  dependencies=dict(python="actual fields rederived", torch="actual fields rederived", flash_attn="3"))
    def factory(registration, health, *, closure_callback=None):
        value.mapping = M.MappingObserver(registration, health, bindings=bindings, source_root=source_root,
            sink_factory=sink_factory, initialization_observer=value.observer, closure_callback=closure_callback,
            recipient_factory=recipient_factory)
        return value.mapping
    forward_dir = tmp_path / "forward"; forward_dir.mkdir()
    if installation_builder is None:
        value.installation = V.Installation(type(runner), forward_dir, engine, profile_types, lambda: context,
                                           assignment_factory=factory).attach()
    else:
        value.installation = installation_builder(value, type(runner), forward_dir, engine, profile_types,
                                                 lambda: context, factory, directory, source_root)
    value.scheduler = initialize(value)
    value.scheduler.current_step, value.scheduler.requests, value.scheduler.schedule_error = 0, {}, None
    for name in request_names:
        request = Request.__new__(Request)
        request.request_id, request.resumable, request.streaming_queue = name, False, None
        value.scheduler.requests[request.request_id] = request
    value.sinks, value.context, value.bindings_mapping, value.journal_directory = sinks, context, bindings, directory
    return value


@pytest.fixture
def assembled(primary, tmp_path):
    value = observer_environment(primary, tmp_path)
    try:
        yield value
    finally:
        value.installation.detach()


def run(value):
    output = value.scheduler.schedule()
    result = value.executor.execute_model(output)
    assert result is value.runner.model.result
    return output


def test_same_native_dispatch_through_primary_uniproc_and_one_existing_hook(assembled):
    value = assembled
    output = run(value)
    assert value.runner.execute_calls == 1
    assert len(value.runner.model._forward_pre_hooks) == 1 and len(value.installation.wrappers) == 4
    assert value.mapping.registry is value.observer.registry
    assert value.runner.shutdown() is value.runner.shutdown_result
    assert value.sinks[0].closed and value.mapping.status["sealed"]
    result = C.evaluate(value.sinks[0].records)
    assert result["verdict"] == "satisfied" and result["summary"]["compared_layer_positions"] == 2
    assert result["ownership_verdict"] == result["physical_read_verdict"] == result["content_provenance"] == "unknown"
    assert output.num_scheduled_tokens == {"request-a": 1}


def test_outer_status_includes_writer_failure_and_lost_hook_coverage(assembled):
    run(assembled)
    assert assembled.mapping.status["healthy"]
    assembled.sinks[0].errors += 1
    assert assembled.mapping.status["journal"]["healthy"] is False
    assert assembled.mapping.status["healthy"] is False
    assembled.runner.shutdown()
    assert assembled.mapping.status["healthy"] is False and not assembled.mapping.status["sealed"]


def test_real_journal_roundtrip_uses_native_profile_without_new_observations(primary, tmp_path):
    value = observer_environment(primary, tmp_path, journal=True)
    try:
        run(value)
        value.runner.shutdown()
        sink, = value.sinks
        assert sink.status["published"] and not sink.incomplete_path.exists()
        records = [json.loads(line) for line in sink.path.read_text().splitlines()]
        assert records[0]["adapter"] == R.NATIVE_ADAPTER and len(records[0]["source_sha256"]) == 36
        assert C.evaluate(records)["verdict"] == "satisfied"
    finally:
        value.installation.detach()


def test_raw_contradictory_table_is_retained_before_forward_semantic_rejection(assembled):
    metadata = next(iter(assembled.context.attn_metadata.values()))
    metadata.block_table.data[0][0] = 0
    run(assembled)
    assembled.runner.shutdown()
    report = C.evaluate(assembled.sinks[0].records)
    assert report["verdict"] == "violated"
    assert report["findings"][0]["contradictions"][0]["model_entry_block"] == 0
    assert report["ownership_verdict"] == report["physical_read_verdict"] == "unknown"


def test_successful_empty_pool_acquisition_is_recorded_without_allocating_an_ordinal(assembled):
    run(assembled)
    pool = assembled.scheduler.kv_cache_manager.block_pool
    result = pool.get_new_blocks(0)
    assert result == []
    before = dict(assembled.mapping._allocations)
    assembled.runner.shutdown()
    records = assembled.sinks[0].records
    assert [r["blocks"] for r in records if r["kind"] == "acquire"] == [
        [{"block": 1, "incarnation": 1}, {"block": 2, "incarnation": 1}], []]
    assert before == {1: 1, 2: 1} and C.evaluate(records)["verdict"] == "satisfied"


def test_missing_allocator_row_stays_unknown_and_getter_does_not_insert(assembled):
    output = assembled.scheduler.schedule()
    group = assembled.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    group.req_to_blocks.pop("request-a")
    assembled.executor.execute_model(output)
    assert "request-a" not in group.req_to_blocks
    assembled.runner.shutdown()
    assert C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


@pytest.mark.parametrize("field,value", [("resumable", True), ("streaming_queue", deque())])
def test_streaming_is_unavailable_without_touching_request_contents(assembled, field, value):
    request = assembled.scheduler.requests["request-a"]
    setattr(request, field, value)
    result = assembled.scheduler.schedule()
    assert result.total_num_scheduled_tokens == 1
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]
    assert C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


def test_native_profile_does_not_change_the_accepted_default_or_old_raw_reinspection():
    fixture = Path(__file__).parent / "fixtures/assignment_native"
    default = C.adapter_contract()
    assert default == json.loads((fixture / "default-profile.json").read_text())
    assert len(default["files"]) == 35
    native = C.adapter_contract(R.NATIVE_ADAPTER)
    assert len(native["files"]) == 36 and native["files"][:-1] == default["files"]
    records = [json.loads(line) for line in (fixture / "old-profile.jsonl").read_text().splitlines()]
    assert C.evaluate(records)["verdict"] == "satisfied"
    from sdcguard import assignment_report
    assert assignment_report.inspect_payload({"old.jsonl": (fixture / "old-profile.jsonl").read_bytes()})["verdict"] == "satisfied"


def test_layers_are_converted_independently_from_the_existing_immutable_copy(primary, tmp_path):
    value = observer_environment(primary, tmp_path, two_layers=True)
    try:
        second = value.context.attn_metadata["model.layers.1.self_attn.attn"]
        second.block_table.data[0][0] = 0
        run(value)
        value.runner.shutdown()
        records = value.sinks[0].records
        metadata = next(r for r in records if r["kind"] == "model_entry")
        assert [r["rows"] for r in metadata["layers"]] == [[[1, 2]], [[0, 2]]]
        report = C.evaluate(records)
        assert report["verdict"] == "violated" and report["summary"]["contradictions"] == 1
        assert report["findings"][0]["contradictions"][0]["layer"] == "model.layers.1.self_attn.attn"
    finally:
        value.installation.detach()


def test_equal_scheduled_values_do_not_replace_the_actual_schedule_object(assembled):
    output = assembled.scheduler.schedule()
    copied = deepcopy(output)
    result = assembled.executor.execute_model(copied)
    assert result is assembled.runner.model.result and assembled.runner.execute_calls == 1
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"] and C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


@pytest.mark.parametrize("error", [RuntimeError("native"), KeyboardInterrupt(), SystemExit(7)])
def test_original_schedule_control_and_exception_identity_are_preserved(assembled, error):
    assembled.scheduler.schedule_error = error
    with pytest.raises(type(error)) as caught:
        assembled.scheduler.schedule()
    assert caught.value is error and assembled.mapping._schedules_open == 0
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]


@pytest.mark.parametrize("error", [RuntimeError("native model"), KeyboardInterrupt(), SystemExit(8)])
def test_original_model_exception_survives_observer_and_journal_cleanup(assembled, error):
    output = assembled.scheduler.schedule()
    assembled.runner.model.error = error
    with pytest.raises(type(error)) as caught:
        assembled.executor.execute_model(output)
    assert caught.value is error and assembled.runner.execute_calls == 1
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]


@pytest.mark.parametrize("kind", ["worker_entry", "assignments", "model_entry", "frame_end"])
def test_rejected_mapping_fact_preserves_native_result_and_prevents_healthy_closure(assembled, kind):
    output = assembled.scheduler.schedule()
    assembled.sinks[0].reject = kind
    result = assembled.executor.execute_model(output)
    assert result is assembled.runner.model.result and assembled.runner.execute_calls == 1
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]
    assert C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


def test_a_positive_schedule_without_runner_entry_is_incomplete(assembled):
    assembled.scheduler.schedule()
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]
    assert C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


def test_empty_native_schedule_does_not_invent_a_runner_frame(assembled):
    assembled.scheduler.requests.clear()
    output = assembled.scheduler.schedule()
    assert output.total_num_scheduled_tokens == 0
    assembled.runner.shutdown()
    records = assembled.sinks[0].records
    assert len([r for r in records if r["kind"] == "schedule"]) == 1
    assert not any(r["kind"] in ("worker_entry", "frame_begin", "frame_end") for r in records)
    assert C.evaluate(records)["verdict"] == "unknown"


@pytest.mark.parametrize("kind", ["mutation", "getter", "initialization"])
def test_deleted_or_foreign_coverage_cannot_seal_even_after_a_complete_frame(assembled, kind):
    run(assembled)
    if kind == "mutation":
        assembled.bindings.pool.touch = lambda *a: None
    elif kind == "getter":
        assembled.bindings.coordinator.get_blocks = lambda *a: ([],)
    else:
        del assembled.bindings.coordinator_module.get_manager_for_kv_cache_spec
    assembled.runner.shutdown()
    assert not assembled.mapping.status["sealed"]
    assert C.evaluate(assembled.sinks[0].records)["verdict"] == "unknown"


def test_noop_cache_and_range_calls_remain_supported_and_do_not_consume_replay_input(assembled):
    run(assembled)
    group = assembled.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    request = assembled.scheduler.requests["request-a"]
    class Untouched:
        def __iter__(self): raise AssertionError("replay input was consumed")
    group.cache_blocks(request, 0, replay_boundaries=Untouched())
    group._remove_blocks_in_range("absent", 0, 2)
    group._remove_blocks_in_range("request-a", 1, 1)
    assert "absent" not in group.req_to_blocks
    assembled.runner.shutdown()
    records = assembled.sinks[0].records
    assert not any(r["kind"] == "unsupported_branch" for r in records)
    assert C.evaluate(records)["verdict"] == "satisfied"
    methods = [r["method"] for r in records if r["kind"] == "mutation_begin"]
    assert "FullAttentionManager.cache_blocks" in methods and "SingleTypeKVCacheManager.cache_blocks" in methods


def test_successful_range_replacement_is_a_recorded_branch_and_preserves_history(assembled):
    run(assembled)
    group = assembled.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    group._remove_blocks_in_range("request-a", 0, 1)
    assembled.runner.shutdown()
    records = assembled.sinks[0].records
    branches = [r for r in records if r["kind"] == "unsupported_branch"]
    assert len(branches) == 1 and branches[0]["method"] == "SingleTypeKVCacheManager._remove_blocks_in_range"
    assert C.evaluate(records)["verdict"] == "satisfied"


def test_false_reset_is_noop_and_true_reset_is_recorded_without_read_right_claims(assembled):
    run(assembled)
    group = assembled.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    pool = group.block_pool
    assert pool.reset_prefix_cache() is False
    group.free("request-a")
    assert pool.reset_prefix_cache() is True
    assembled.runner.shutdown()
    records = assembled.sinks[0].records
    branches = [r for r in records if r["kind"] == "unsupported_branch"]
    assert len(branches) == 1 and branches[0]["method"] == "BlockPool.reset_prefix_cache"
    report = C.evaluate(records)
    assert report["verdict"] == "satisfied" and report["ownership_verdict"] == "unknown"


def test_unpin_iterable_is_consumed_only_by_native_code_and_does_not_create_a_generation(assembled):
    run(assembled)
    group = assembled.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    pool, block = group.block_pool, group.req_to_blocks["request-a"][0]
    observations = []
    def blocks():
        observations.append("native iteration")
        yield block
    callbacks = []
    before = dict(assembled.mapping._allocations)
    pool.unpin_blocks(blocks(), lambda value: callbacks.append(value))
    assert observations == ["native iteration"] and callbacks == [] and block.ref_cnt == 0
    assert assembled.mapping._allocations == before
    assembled.runner.shutdown()
    report = C.evaluate(assembled.sinks[0].records)
    assert report["verdict"] == "satisfied" and report["ownership_verdict"] == "unknown"


def test_no_strong_native_graph_is_retained_after_finite_interval_closure(primary, tmp_path):
    value = observer_environment(primary, tmp_path)
    run(value)
    runner_ref = weakref.ref(value.runner)
    request_ref = weakref.ref(value.scheduler.requests["request-a"])
    mapping = value.mapping
    value.runner.shutdown()
    value.installation.detach()
    del value
    gc.collect()
    assert runner_ref() is None and request_ref() is None
    assert mapping.status["sealed"]


def close_while_blocked(value, operation, entered, release):
    """Ordinary host callbacks only, with bounded test cleanup on every path."""
    returned, errors, closed = [], [], []
    def producer():
        try:
            returned.append(operation())
        except BaseException as error:
            errors.append(error)
    def closer():
        closed.append(value.mapping.close(timeout=0.05))
    worker = threading.Thread(target=producer)
    worker.start()
    closer_thread = None
    try:
        assert entered.wait(2), "host callback did not reach its controlled boundary"
        closer_thread = threading.Thread(target=closer)
        closer_thread.start()
        closer_thread.join(1)
        assert not closer_thread.is_alive(), "close waited for an admitted observer callback"
        assert closed == [False]
    finally:
        release.set()
        worker.join(2)
        if closer_thread is not None:
            closer_thread.join(2)
    assert not worker.is_alive() and not errors
    assert returned == [value.runner.model.result]
    assert not value.mapping.status["sealed"]
    assert C.evaluate(value.sinks[0].records)["verdict"] == "unknown"


def test_close_accounts_for_callback_after_native_invocation_bookkeeping(assembled):
    output = assembled.scheduler.schedule()
    entered, release = threading.Event(), threading.Event()
    identify = assembled.mapping.registry.identify
    def blocked(value, role):
        if role == "runner":
            entered.set()
            assert release.wait(3)
        return identify(value, role)
    assembled.mapping.registry.identify = blocked
    close_while_blocked(assembled, lambda: assembled.executor.execute_model(output), entered, release)


def test_journal_emission_does_not_hold_observer_state_lock_over_close(assembled):
    output = assembled.scheduler.schedule()
    sink = assembled.sinks[0]
    entered, release = threading.Event(), threading.Event()
    emit = sink.emit
    def blocked(kind, **facts):
        if kind == "model_entry":
            entered.set()
            assert release.wait(3)
        return emit(kind, **facts)
    sink.emit = blocked
    close_while_blocked(assembled, lambda: assembled.executor.execute_model(output), entered, release)


@pytest.mark.parametrize("lookup_number", [1, 2, 3])
def test_binding_getter_does_not_hold_observer_state_lock_over_close(assembled, lookup_number):
    output = assembled.scheduler.schedule()
    entered, release = threading.Event(), threading.Event()
    lookup = assembled.mapping.initialization.lookup
    seen = []
    def blocked(*, scheduler=None, runner=None):
        if runner is not None:
            seen.append(None)
            if len(seen) == lookup_number:
                entered.set()
                assert release.wait(3)
        return lookup(scheduler=scheduler, runner=runner)
    assembled.mapping.initialization.lookup = blocked
    close_while_blocked(assembled, lambda: assembled.executor.execute_model(output), entered, release)


def test_profile_creation_after_close_is_unhealthy_and_its_writer_is_closed(assembled):
    entered, release = threading.Event(), threading.Event()
    create = assembled.mapping.sink_factory
    created = []
    def blocked(**fields):
        entered.set()
        assert release.wait(3)
        writer = create(**fields)
        created.append(writer)
        return writer
    assembled.mapping.sink_factory = blocked
    result, errors = [], []
    def producer():
        try:
            result.append(assembled.scheduler.schedule())
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=producer)
    worker.start()
    try:
        assert entered.wait(2)
        assert assembled.mapping.close(timeout=0.05) is False
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive() and not errors and len(result) == 1
    assert len(created) == 1 and created[0].closed and not created[0].published
    assert not assembled.mapping.status["sealed"]
