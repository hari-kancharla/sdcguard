"""CPU structural stand-ins for passive metadata recording, not GPU evidence."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS

import pytest

from sdcguard.lifetime import forward_inputs as C
from sdcguard.lifetime import vllm_forward as V


DEVICE = "test-device"


class Device:
    type = "cuda"

    def __eq__(self, other):
        return self is other


class Tensor:
    def __init__(self, data=None, *, shape=None, dtype="int64", device=None):
        self.data = copy.deepcopy(data)
        self.dtype = dtype
        self.device = device
        if shape is not None:
            self.shape = tuple(shape)
        elif data and isinstance(data[0], list):
            self.shape = (len(data), len(data[0]))
        else:
            self.shape = (len(data),)
        self.ndim = len(self.shape)
        self.copies = 0

    def __getitem__(self, item):
        if isinstance(item, tuple):
            rows, cols = item
            data = [row[cols] for row in self.data[rows]]
        else:
            data = self.data[item]
        return Tensor(data, dtype=self.dtype, device=self.device)

    def detach(self):
        return self

    def to(self, *, device, copy, non_blocking):
        assert device == "cpu" and copy is True and non_blocking is False
        self.copies += 1
        return Tensor(self.data, dtype=self.dtype, device="cpu")

    def tolist(self):
        return copy.deepcopy(self.data)

    def data_ptr(self):
        return id(self)

    def stride(self):
        out, product = [], 1
        for dim in reversed(self.shape):
            out.append(product)
            product *= dim
        return tuple(reversed(out))

    def is_contiguous(self):
        return True


class Model:
    def __init__(self):
        self._forward_pre_hooks = {}
        self.result = object()
        self.error = None
        self.after_hook = None

    def register_forward_pre_hook(self, callback):
        key = len(self._forward_pre_hooks)
        self._forward_pre_hooks[key] = callback
        return NS(remove=lambda: self._forward_pre_hooks.pop(key, None))

    def __call__(self):
        for callback in list(self._forward_pre_hooks.values()):
            assert callback(self, ()) is None
        if self.after_hook is not None:
            self.after_hook()
        if self.error is not None:
            raise self.error
        return self.result


class Spec:
    pass


class Attention:
    pass


class Impl:
    pass


class Metadata:
    pass


class ModelState:
    pass


def environment(tmp_path):
    device = Device()
    stream = object()
    graph_none = object()
    compile_none = object()
    quant_none = object()
    torch = NS(Tensor=Tensor, int32="int32", int64="int64", bool="bool",
               float16="float16", bfloat16="bfloat16", cuda=NS(current_stream=lambda d: stream))
    types = dict(torch=torch, model=Model, attention=Attention, impl=Impl, metadata=Metadata,
                 model_state=ModelState, compile_none=compile_none, spec=Spec,
                 quant_none=quant_none, decoder="decoder", graph_none=graph_none)
    names = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    spec = Spec()
    spec.__dict__.update(non_causal=False, tokens_per_state=1, sliding_window=None,
                         attention_chunk_size=None, page_size_padded=None, num_head_slots=None,
                         state_content_bytes=None, kv_quant_mode=quant_none, block_size=16,
                         num_kv_heads=2, head_size=4, head_size_v=4, dtype="bfloat16")
    layers = {}
    metadata = {}
    slots = {}
    for name in names:
        impl = Impl()
        impl.__dict__.update(vllm_flash_attn_version=3, sliding_window=(-1, -1),
                             kv_sharing_target_layer_name=None, alibi_slopes=None, sinks=None,
                             attn_type="decoder", dcp_world_size=1)
        layer = Attention()
        layer.__dict__.update(impl=impl, attn_backend=NS(get_name=lambda: "FLASH_ATTN"),
                              attn_type="decoder", mm_prefix_clamp_sliding_window=False,
                              kv_cache_dtype="auto", kv_sharing_target_layer_name=None,
                              sliding_window=None, kv_cache=Tensor(shape=(8, 2, 16, 8),
                                                                  dtype="bfloat16", device=device))
        layers[name] = layer
        meta = Metadata()
        meta.__dict__.update(causal=True, mm_prefix_query_range_tensor=None, rswa_prefix_lens=None,
                             rswa_window=None, rswa_window_tensor=None, dcp_context_kv_lens=None,
                             # Ordinary metadata builder, flash_attn.py:865,963.
                             max_dcp_context_kv_len=0, num_actual_tokens=3,
                             max_query_len=2, max_seq_len=18,
                             block_table=Tensor([[1, 2, 7], [3, 7, 7]], device=device),
                             seq_lens=Tensor([18, 1], device=device),
                             query_start_loc=Tensor([0, 2, 3], device=device))
        metadata[name] = meta
        slots[name] = Tensor([32, 33, 48], device=device)
    context = NS(cudagraph_runtime_mode=graph_none, ubatch_slices=None, dp_metadata=None,
                 attn_metadata=metadata, slot_mapping=slots, no_compile_layers=layers,
                 is_padding=Tensor([False, False, False], dtype="bool", device=device))

    class Runner:
        def initialize_kv_cache(self, config=None, is_profiling=False):
            self.init_calls += 1
            if self.init_error:
                raise self.init_error
            if config is not None:
                self.kv_cache_config = config
            self.block_tables = object()
            return self.init_result

        def prepare_attn(self, input_batch):
            self.prepare_calls += 1
            if self.prepare_error:
                raise self.prepare_error
            return self.prepare_result

        def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
            self.execute_calls += 1
            if scheduler_output.total_num_scheduled_tokens == 0:
                return self.model.result
            if not dummy_run and not self.skip_prepare:
                self.prepare_attn(self.batch)
            if not self.skip_model:
                return self.model()
            return self.model.result

        def shutdown(self):
            self.shutdown_calls += 1
            if self.shutdown_error is not None:
                raise self.shutdown_error
            # Native v0.32.0 gpu/model_runner.py:2337-2371 deletes these
            # objects. The recorder must retain its own cleanup handles.
            for name in ("model", "model_state", "kv_cache_config"):
                if hasattr(self, name):
                    delattr(self, name)
            return self.shutdown_result

    runner = Runner()
    runner.__dict__.update(model=Model(), model_state=ModelState(), vllm_config=NS(use_v2_model_runner=True),
                           model_config=NS(model="Qwen/Qwen3-0.6B", enforce_eager=True,
                                           architectures=["Qwen3ForCausalLM"], rswa_window=None,
                                           hf_text_config=NS(num_hidden_layers=2), revision="c" * 40),
                           cache_config=NS(enable_prefix_caching=False, cache_dtype="auto"),
                           parallel_config=NS(enable_dbo=False, tensor_parallel_size=1,
                                              pipeline_parallel_size=1, decode_context_parallel_size=1,
                                              prefill_context_parallel_size=1, data_parallel_size=1),
                           is_encoder_decoder=False, supports_mm_inputs=False,
                           speculative_config=None, lora_config=None, ubatch_runner=None,
                           pcp_manager=None, kernel_block_sizes=[16], device=device,
                           compilation_config=NS(static_forward_context=layers, mode=compile_none),
                           kv_cache_config=NS(num_blocks=8, kv_cache_layout="LBHNC", kv_cache_groups=[
                               NS(kv_cache_spec=spec, layer_names=names, is_eagle_group=False, host_resident=False)]),
                           block_tables=object(), init_calls=0, prepare_calls=0, execute_calls=0,
                           shutdown_calls=0, shutdown_error=None, shutdown_result=object(),
                           init_error=None, prepare_error=None, init_result=object(), prepare_result=object(),
                           skip_prepare=False, skip_model=False,
                           batch=NS(num_reqs=2, num_tokens=3, num_draft_tokens=0, req_ids=["request-b", "request-a"]))
    engine = dict(name="vllm", version="0.32.0", source_sha256=C.source_pins(),
                  dependencies=dict(python="3.13.0", torch="CPU structural stand-in", flash_attn="3"))
    installation = V.Installation(Runner, tmp_path, engine, types, lambda: context).attach()
    return NS(installation=installation, runner=runner, context=context, types=types, engine=engine,
              directory=tmp_path, scheduler=NS(total_num_scheduled_tokens=3))


@pytest.fixture
def env(tmp_path):
    value = environment(tmp_path)
    try:
        yield value
    finally:
        value.installation.detach()


def records(env):
    env.installation.detach()
    return [[json.loads(line) for line in p.read_text().splitlines()]
            for p in sorted(env.directory.glob("*.jsonl"))]


def test_valid_boundary_snapshot_is_immutable_and_not_physical_evidence(env):
    runner = env.runner
    assert runner.initialize_kv_cache() is runner.init_result
    old_rows = [m.block_table.data for m in env.context.attn_metadata.values()]
    runner.model.after_hook = lambda: [row.__setitem__(0, [6, 6, 6]) for row in old_rows]
    assert runner.execute_model(env.scheduler) is runner.model.result
    session, = records(env)
    snapshot = session[1]
    assert snapshot["requests"] == ["request-b", "request-a"]
    assert snapshot["layers"][0]["block_table"] == [[1, 2], [3]]
    assert snapshot["layers"][0]["write_slots"] == [32, 33, 48]
    assert snapshot["observation_point"] == "model_entry"
    report = C.audit_records(session)
    assert report["observation_status"] == "observed"
    assert report["physical_read_verdict"] == "unknown"
    assert report["ownership_verdict"] == "unknown"
    assert report["evidence_depth"] == "approximate"
    assert runner.init_calls == runner.prepare_calls == runner.execute_calls == 1


@pytest.mark.parametrize("change", [
    lambda e: setattr(e.context, "cudagraph_runtime_mode", object()),
    lambda e: setattr(e.context, "ubatch_slices", []),
    lambda e: setattr(e.context, "attn_metadata", [e.context.attn_metadata]),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "causal", False),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "rswa_window", 16),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "max_dcp_context_kv_len", None),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "num_actual_tokens", 2),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "max_query_len", 1),
    lambda e: setattr(next(iter(e.context.attn_metadata.values())), "max_seq_len", 17),
    lambda e: setattr(next(iter(e.context.no_compile_layers.values())).impl, "sliding_window", (15, 0)),
    lambda e: setattr(next(iter(e.context.no_compile_layers.values())).impl, "attn_type", "encoder"),
    lambda e: setattr(next(iter(e.context.no_compile_layers.values())).impl, "dcp_world_size", 2),
    lambda e: setattr(e.runner.compilation_config, "mode", object()),
    lambda e: setattr(e.runner, "model_state", object()),
    lambda e: e.context.is_padding.data.__setitem__(0, True),
    lambda e: setattr(e.runner, "skip_prepare", True),
    lambda e: setattr(e.runner, "skip_model", True),
    lambda e: setattr(e.runner, "block_tables", object()),
    lambda e: setattr(e.runner.batch, "req_ids", ["same", "same"]),
    lambda e: setattr(e.runner.batch, "num_draft_tokens", 1),
])
def test_unsupported_or_missing_facts_become_unknown_without_changing_engine_result(env, change):
    env.runner.initialize_kv_cache()
    change(env)
    assert env.runner.execute_model(env.scheduler) is env.runner.model.result
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "unknown"
    assert any(row["kind"] == "gap" for row in session)


def test_dummy_invocation_does_not_reuse_prior_rows(env):
    env.runner.initialize_kv_cache()
    env.runner.execute_model(env.scheduler)
    env.runner.execute_model(env.scheduler, dummy_run=True)
    env.runner.execute_model(NS(total_num_scheduled_tokens=0))
    session, = records(env)
    assert len([r for r in session if r["kind"] == "snapshot"]) == 1
    assert C.audit_records(session)["observation_status"] == "observed"


def test_startup_profiling_before_cache_has_no_recording_side_effect(env):
    cache = env.runner.kv_cache_config
    del env.runner.kv_cache_config
    assert env.runner.execute_model(env.scheduler, dummy_run=True, is_profile=True) is env.runner.model.result
    assert env.runner.execute_model(NS(total_num_scheduled_tokens=0)) is env.runner.model.result
    assert not list(env.directory.glob("*.jsonl"))
    env.runner.initialize_kv_cache(cache, is_profiling=True)
    assert not list(env.directory.glob("*.jsonl"))
    env.runner.initialize_kv_cache(cache)
    env.runner.execute_model(env.scheduler)
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "observed"


def test_pool_initialization_changes_both_session_and_pool_identity(env):
    for _ in range(2):
        env.runner.initialize_kv_cache()
        env.runner.execute_model(env.scheduler)
    sessions = records(env)
    assert len(sessions) == 2
    assert len({s[0]["session"] for s in sessions}) == 2
    assert len({s[1]["pool_incarnation"] for s in sessions}) == 2


@pytest.mark.parametrize("where", ["init", "prepare", "model"])
def test_original_exceptions_propagate_by_identity(env, where):
    error = RuntimeError("original engine error")
    if where == "init":
        env.runner.init_error = error
        call = env.runner.initialize_kv_cache
    else:
        env.runner.initialize_kv_cache()
        if where == "prepare":
            env.runner.prepare_error = error
        else:
            env.runner.model.error = error
        call = lambda: env.runner.execute_model(env.scheduler)
    with pytest.raises(RuntimeError) as caught:
        call()
    assert caught.value is error


def test_observer_pid_error_does_not_escape_original_call(env, monkeypatch):
    env.runner.initialize_kv_cache()
    monkeypatch.setattr(env.installation, "_pid", lambda: (_ for _ in ()).throw(ValueError("recorder")))
    assert env.runner.execute_model(env.scheduler) is env.runner.model.result
    monkeypatch.undo()
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "unknown"


def test_unknown_profile_is_not_retried_per_request(env):
    env.runner.model_config.enforce_eager = False
    env.runner.initialize_kv_cache()
    for _ in range(4):
        assert env.runner.execute_model(env.scheduler) is env.runner.model.result
    assert len(records(env)) == 1


def test_disabled_install_does_not_import_engine_or_device(monkeypatch):
    monkeypatch.delenv("SDCGUARD_FORWARD_DIR", raising=False)
    monkeypatch.setattr(V.importlib, "import_module", lambda name: pytest.fail("unexpected import: " + name))
    assert V.install() == {"status": "disabled", "hooks": []}


def test_source_mismatch_installs_no_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(tmp_path / "output"))
    monkeypatch.setattr(V.importlib, "import_module", lambda name: NS(__version__="0.32.0", __file__=str(tmp_path / "vllm/__init__.py")))
    result = V.install()
    assert result["status"] == "unknown" and result["hooks"] == []
    assert V._INSTALLATION is None


def test_missing_later_method_leaves_class_unchanged(tmp_path):
    class Incomplete:
        def initialize_kv_cache(self):
            pass
        def prepare_attn(self):
            pass
    before = dict(vars(Incomplete))
    installation = V.Installation(Incomplete, tmp_path, {}, {}, lambda: None)
    with pytest.raises(AttributeError):
        installation.attach()
    assert dict(vars(Incomplete)) == before


def test_repeated_attach_rejected_and_detach_restores_only_owned_wrappers(env):
    installation = env.installation
    wrappers = dict(installation.wrappers)
    with pytest.raises(ValueError):
        installation.attach()
    original, own_wrapper = wrappers["execute_model"]
    replacement = lambda *a, **k: None
    installation.runner_class.execute_model = replacement
    installation.detach()
    assert installation.runner_class.execute_model is replacement
    for name in ("initialize_kv_cache", "prepare_attn", "shutdown"):
        assert getattr(installation.runner_class, name) is wrappers[name][0]


def test_queue_drop_is_explicitly_unhealthy(env, tmp_path):
    profile = dict(C.SUPPORTED_PROFILE, model_revision="c" * 40, layer_names=["layer"])
    writer = V.JsonlRecorder(tmp_path / "bounded", env.engine, profile, queue_bytes=1)
    assert writer.emit("gap", reason="example", forward_id=None) is False
    assert writer.close()
    data = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert data[-1]["dropped"] == 1 and data[-1]["healthy"] is False
    assert C.audit_records(data)["observation_status"] == "unknown"


def test_partial_state_failure_closes_writer(env, monkeypatch):
    env.runner.model.register_forward_pre_hook = lambda f: (_ for _ in ()).throw(RuntimeError("observer setup"))
    env.runner.initialize_kv_cache()
    assert not env.installation.states
    sessions = records(env)
    assert len(sessions) == 2
    assert all(s[-1]["kind"] == "footer" and s[-1]["healthy"] is False for s in sessions)


def test_fork_reset_does_not_touch_inherited_writer_lock(env):
    env.runner.initialize_kv_cache()
    state = env.installation.states[env.runner]
    env.installation.pid = -1
    state.pid = -1
    state.writer._lock.acquire()
    try:
        env.installation._pid()
        assert not env.installation.states
        assert not env.runner.model._forward_pre_hooks
    finally:
        state.writer._lock.release()
        state.writer.close()
    env.runner.execute_model(env.scheduler)
    sessions = records(env)
    assert len(sessions) == 2
    assert len({s[0]["session"] for s in sessions}) == 2


def test_writer_close_has_a_bounded_wait(env, tmp_path):
    profile = dict(C.SUPPORTED_PROFILE, model_revision="c" * 40, layer_names=["layer"])
    writer = V.JsonlRecorder(tmp_path / "slow", env.engine, profile)
    original = writer._file
    entered, release = threading.Event(), threading.Event()
    class SlowFile:
        def write(self, value):
            entered.set()
            assert release.wait(5)
            return original.write(value)
        def flush(self):
            return original.flush()
        def close(self):
            return original.close()
    writer._file = SlowFile()
    writer.emit("gap", reason="test", forward_id=None)
    assert entered.wait(1)
    assert writer.close(timeout=0) is False
    release.set()
    assert writer.close(timeout=1) is True


def test_recording_directory_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        V._directory(alias / "child")
    assert not (target / "child").exists()


def test_unknown_request_iterator_is_not_consumed(env):
    env.runner.initialize_kv_cache()
    touched = []
    def requests():
        touched.append(True)
        yield "request-b"
        yield "request-a"
    env.runner.batch.req_ids = requests()
    env.runner.execute_model(env.scheduler)
    assert not touched
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "unknown"


def test_metadata_tensor_subclass_is_rejected_before_slicing(env):
    class SpecialTensor(Tensor):
        def __getitem__(self, key):
            pytest.fail("unsupported tensor subclass was sliced")
    env.runner.initialize_kv_cache()
    next(iter(env.context.attn_metadata.values())).seq_lens = SpecialTensor([18, 1], device=env.runner.device)
    env.runner.execute_model(env.scheduler)
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "unknown"


def test_shared_metadata_objects_are_copied_once_per_slice(env, monkeypatch):
    env.runner.initialize_kv_cache()
    names = list(env.context.attn_metadata)
    env.context.attn_metadata[names[1]] = env.context.attn_metadata[names[0]]
    env.context.slot_mapping[names[1]] = env.context.slot_mapping[names[0]]
    calls = []
    original = V._copy_integer
    def count(*args):
        calls.append(args[1])
        return original(*args)
    monkeypatch.setattr(V, "_copy_integer", count)
    env.runner.execute_model(env.scheduler)
    assert len(calls) == 4
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "observed"


def test_equal_but_distinct_metadata_objects_are_copied_separately(env, monkeypatch):
    env.runner.initialize_kv_cache()
    calls = []
    original = V._copy_integer
    def count(*args):
        calls.append(args[1])
        return original(*args)
    monkeypatch.setattr(V, "_copy_integer", count)
    env.runner.execute_model(env.scheduler)
    assert len(calls) == 8


def test_concurrent_runner_observation_marks_gap_without_waiting(env):
    env.runner.initialize_kv_cache()
    state = env.installation.states[env.runner]
    state.busy.acquire()
    try:
        assert env.runner.execute_model(env.scheduler) is env.runner.model.result
    finally:
        state.busy.release()
    session, = records(env)
    assert C.audit_records(session)["observation_status"] == "unknown"
    assert any(r.get("reason") == "concurrent runner invocation" for r in session)


def test_session_cell_limit_is_visible_as_a_drop(env, monkeypatch):
    env.runner.initialize_kv_cache()
    monkeypatch.setattr(C, "MAX_TOTAL_CELLS", 1)
    env.runner.execute_model(env.scheduler)
    session, = records(env)
    assert session[-1]["dropped"] == 1 and session[-1]["healthy"] is False


def test_unexpected_exception_messages_do_not_copy_engine_payloads(env, monkeypatch):
    env.runner.initialize_kv_cache()
    monkeypatch.setattr(env.installation, "get_context", lambda: (_ for _ in ()).throw(RuntimeError("private prompt material")))
    env.runner.execute_model(env.scheduler)
    session, = records(env)
    assert "private prompt material" not in json.dumps(session)
    assert any(r.get("reason") == "RuntimeError" for r in session)


def test_recording_open_stays_with_anchored_directory_after_parent_replacement(env, tmp_path, monkeypatch):
    output = tmp_path / "output"
    moved = tmp_path / "preserved"
    elsewhere = tmp_path / "elsewhere"
    output.mkdir()
    elsewhere.mkdir()
    original = os.open
    replaced = []
    def changing_open(path, flags, *args, **kwargs):
        if str(path).endswith(".jsonl") and not replaced:
            assert kwargs.get("dir_fd") is not None
            output.rename(moved)
            output.symlink_to(elsewhere, target_is_directory=True)
            replaced.append(True)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(V.os, "open", changing_open)
    profile = dict(C.SUPPORTED_PROFILE, model_revision="c" * 40, layer_names=["layer"])
    writer = V.JsonlRecorder(output, env.engine, profile)
    assert writer.close()
    assert replaced and len(list(moved.glob("*.jsonl"))) == 1
    assert not list(elsewhere.iterdir())


def test_pinned_source_read_rejects_symlink_and_special_file(tmp_path, monkeypatch):
    import hashlib
    engine = tmp_path / "engine"
    engine.mkdir()
    real = tmp_path / "real.py"
    real.write_bytes(b"source\n")
    path = engine / "checked.py"
    monkeypatch.setattr(C, "source_pins", lambda: {"checked.py": hashlib.sha256(real.read_bytes()).hexdigest()})
    path.symlink_to(real)
    with pytest.raises((OSError, ValueError)):
        V.inspect_source(engine, "0.32.0")
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        V.inspect_source(engine, "0.32.0")


def test_pinned_source_regular_bytes_are_checked_without_engine_import(tmp_path, monkeypatch):
    import hashlib
    (tmp_path / "checked.py").write_bytes(b"checked source\n")
    pins = {"checked.py": hashlib.sha256(b"checked source\n").hexdigest()}
    monkeypatch.setattr(C, "source_pins", lambda: pins)
    assert V.inspect_source(tmp_path, "0.32.0") == pins
    (tmp_path / "checked.py").write_bytes(b"different source\n")
    with pytest.raises(ValueError, match="pin mismatch"):
        V.inspect_source(tmp_path, "0.32.0")


def test_duplicate_preparation_poison_cannot_be_repaired_by_a_third_preparation(env):
    env.installation.detach()
    model_calls = []
    env.runner.model.after_hook = lambda: model_calls.append(True)

    def execute(runner, scheduler_output, dummy_run=False, is_profile=False):
        runner.execute_calls += 1
        runner.prepare_attn(runner.batch)
        # Both later row sets have compatible dimensions and write maps. Only
        # the invocation provenance exposes the ambiguous request association.
        runner.batch.req_ids = ["second-b", "second-a"]
        runner.prepare_attn(runner.batch)
        runner.batch.req_ids = ["third-b", "third-a"]
        runner.prepare_attn(runner.batch)
        return runner.model()

    env.installation.runner_class.execute_model = execute
    env.installation.attach()
    env.runner.initialize_kv_cache()
    assert env.runner.execute_model(env.scheduler) is env.runner.model.result
    session, = records(env)
    assert not any(record["kind"] == "snapshot" for record in session)
    assert any(record.get("reason") == "multiple batch preparations" for record in session)
    assert env.runner.prepare_calls == 3
    assert env.runner.execute_calls == 1
    assert model_calls == [True]
    assert C.audit_records(session)["observation_status"] == "unknown"


def shutdown_records(state):
    """Read after native shutdown, without detach or extra writer cleanup."""
    assert state.writer._closed
    assert not state.writer._thread.is_alive()
    return [json.loads(line) for line in state.writer.path.read_text().splitlines()]


def test_native_shutdown_finishes_recording_before_manual_detach(env):
    runner = env.runner
    runner.initialize_kv_cache()
    runner.execute_model(env.scheduler)
    state = env.installation.states[runner]
    model = runner.model
    assert runner.shutdown() is runner.shutdown_result
    assert runner.shutdown_calls == 1
    assert not hasattr(runner, "model") and not hasattr(runner, "kv_cache_config")
    assert runner not in env.installation.states
    assert not model._forward_pre_hooks
    session = shutdown_records(state)
    assert session[-1]["kind"] == "footer" and session[-1]["healthy"] is True
    assert C.audit_records(session)["observation_status"] == "observed"
    assert session[1]["forward_device_complete"] is None
    assert C.audit_records(session)["physical_read_verdict"] == "unknown"


@pytest.mark.parametrize("error", [RuntimeError("engine shutdown"), KeyboardInterrupt()])
def test_native_shutdown_exception_identity_is_preserved_with_unhealthy_footer(env, error):
    env.runner.initialize_kv_cache()
    env.runner.execute_model(env.scheduler)
    state = env.installation.states[env.runner]
    env.runner.shutdown_error = error
    with pytest.raises(type(error)) as caught:
        env.runner.shutdown()
    assert caught.value is error
    assert env.runner.shutdown_calls == 1
    assert env.runner not in env.installation.states
    assert not env.runner.model._forward_pre_hooks
    session = shutdown_records(state)
    assert any(row.get("reason") == "native runner shutdown raised" for row in session)
    assert C.audit_records(session)["observation_status"] == "unknown"


def test_native_shutdown_before_initialization_and_repeated_shutdown_are_safe(env):
    assert env.runner.shutdown() is env.runner.shutdown_result
    assert env.runner.shutdown() is env.runner.shutdown_result
    assert env.runner.shutdown_calls == 2
    assert not env.installation.states
    assert not list(env.directory.glob("*.jsonl"))


def test_native_shutdown_closes_only_the_selected_runner(env):
    first = env.runner
    second = copy.copy(first)
    second.model = Model()
    for runner in (first, second):
        runner.initialize_kv_cache()
        runner.execute_model(env.scheduler)
    first_state = env.installation.states[first]
    second_state = env.installation.states[second]
    assert first.shutdown() is first.shutdown_result
    assert C.audit_records(shutdown_records(first_state))["observation_status"] == "observed"
    assert env.installation.states[second] is second_state
    assert not second_state.writer._closed
    assert second.model._forward_pre_hooks
    second.execute_model(env.scheduler)
    assert second.shutdown() is second.shutdown_result
    second_session = shutdown_records(second_state)
    assert len([row for row in second_session if row["kind"] == "snapshot"]) == 2
    assert C.audit_records(second_session)["observation_status"] == "observed"


def test_shutdown_then_explicit_reinitialization_creates_a_fresh_session(env):
    runner = env.runner
    saved = {name: getattr(runner, name) for name in ("model", "model_state", "kv_cache_config")}
    runner.initialize_kv_cache()
    runner.execute_model(env.scheduler)
    first = env.installation.states[runner]
    runner.shutdown()
    before = first.writer.path.read_bytes()
    runner.shutdown()
    assert first.writer.path.read_bytes() == before
    runner.__dict__.update(saved)
    runner.initialize_kv_cache()
    runner.execute_model(env.scheduler)
    second = env.installation.states[runner]
    runner.shutdown()
    assert first.pool != second.pool and first.writer.session != second.writer.session
    assert C.audit_records(shutdown_records(first))["observation_status"] == "observed"
    assert C.audit_records(shutdown_records(second))["observation_status"] == "observed"


def test_native_shutdown_attempts_writer_close_even_when_hook_removal_fails(env, monkeypatch):
    env.runner.initialize_kv_cache()
    env.runner.execute_model(env.scheduler)
    state = env.installation.states[env.runner]
    model = env.runner.model
    original_remove = state.handle.remove
    monkeypatch.setattr(state.handle, "remove", lambda: (_ for _ in ()).throw(RuntimeError("cleanup")))
    assert env.runner.shutdown() is env.runner.shutdown_result
    session = shutdown_records(state)
    assert any(row.get("reason") == "model pre-hook removal failed" for row in session)
    assert C.audit_records(session)["observation_status"] == "unknown"
    before = state.writer.path.read_bytes()
    model()
    assert state.writer.path.read_bytes() == before
    original_remove()


@pytest.mark.parametrize("engine_error", [None, RuntimeError("original shutdown")])
def test_native_shutdown_preserves_result_when_writer_close_raises(env, monkeypatch, engine_error):
    env.runner.initialize_kv_cache()
    env.runner.execute_model(env.scheduler)
    state = env.installation.states[env.runner]
    original_close = state.writer.close
    attempts = []
    def broken_close():
        attempts.append(True)
        raise RuntimeError("writer cleanup")
    monkeypatch.setattr(state.writer, "close", broken_close)
    env.runner.shutdown_error = engine_error
    try:
        if engine_error is None:
            assert env.runner.shutdown() is env.runner.shutdown_result
        else:
            with pytest.raises(RuntimeError) as caught:
                env.runner.shutdown()
            assert caught.value is engine_error
        assert attempts == [True]
        assert env.runner.shutdown_calls == 1
        assert env.runner not in env.installation.states
        assert not state.writer._closed
    finally:
        original_close()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "unknown"


def test_native_shutdown_during_active_capture_is_an_explicit_gap(env):
    env.runner.initialize_kv_cache()
    state = env.installation.states[env.runner]
    model = env.runner.model
    model.after_hook = env.runner.shutdown
    assert env.runner.execute_model(env.scheduler) is model.result
    assert env.runner.shutdown_calls == 1
    session = shutdown_records(state)
    assert any(row.get("reason") == "recorder closed during runner invocation" for row in session)
    assert C.audit_records(session)["observation_status"] == "unknown"
    assert session[1]["forward_device_complete"] is None
