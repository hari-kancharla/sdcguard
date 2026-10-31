"""CPU callback controls only. They are not native assignment evidence."""
from __future__ import annotations

import copy
from enum import Enum
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace as NS
import weakref

import pytest

from sdcguard.lifetime import forward_bridge as B
from sdcguard.lifetime import forward_inputs as C
from sdcguard.lifetime import vllm_forward as V
from test_vllm_forward import environment, shutdown_records


class Observer:
    def __init__(self):
        self.events = []
        self.copies = []
        self.fail = None
        self.closed_check = None
        self.closed_error = None
        self.invocation_token = None
        self.frame_token = None
        self.control_name = None
        self.control_error = None

    def event(self, name, *values):
        self.events.append((name, *values))
        if self.fail == name:
            raise RuntimeError("private callback payload")
        if self.control_name == name:
            raise self.control_error

    def runner_initialized(self, value):
        self.event("ready", value)

    def invocation_enter(self, value):
        self.event("enter", value)
        return self.invocation_token or "invocation-" + str(len(self.events))

    def prepared(self, invocation, value):
        self.event("prepared", invocation, value)

    def begin_copy(self, invocation, value):
        self.event("begin", invocation, value)
        return self.frame_token or "frame-" + str(len(self.events))

    def end_copy(self, frame, value):
        self.copies.append(value)
        self.event("copied", frame, value)

    def abandon_copy(self, invocation, frame, reason):
        self.event("abandon", invocation, frame, reason)

    def invocation_leave(self, invocation, value):
        self.event("leave", invocation, value)

    def retire_runner(self, runner, reason):
        self.event("retire", runner, reason)

    def runner_closed(self, runner, exception_type):
        if self.closed_check is not None:
            self.closed_check(runner, exception_type)
        if self.closed_error is not None:
            raise self.closed_error
        self.event("runner_closed", runner, exception_type)
        return True

    def close(self, *, timeout):
        self.event("close", timeout)
        return True


@pytest.fixture
def bridged(tmp_path):
    value = environment(tmp_path)
    value.installation.detach()
    value.observer = Observer()
    value.types["global_hook_registry"] = NS(_global_forward_pre_hooks={})
    value.factory = lambda registration, health: value.observer
    value.installation = V.Installation(type(value.runner), tmp_path, value.engine, value.types,
                                       lambda: value.context, assignment_factory=value.factory).attach()
    try:
        yield value
    finally:
        value.installation.detach()


def names(value):
    return [event[0] for event in value.observer.events]


def test_legacy_registration_has_no_model_capability(bridged):
    registration = B.Registration(type(bridged.runner), B._freeze(bridged.engine))
    assert registration.model_binding_installation() is None
    with pytest.raises(ValueError):
        registration.model_binding_for(bridged.runner)


@pytest.mark.parametrize("changed", ["pid", "runner", "engine", "method", "class_method"])
def test_model_registration_rejects_changed_installation(bridged, monkeypatch, changed):
    owner = bridged.installation
    registration = B.Registration(owner.runner_class, B._freeze(owner.engine), weakref.ref(owner))
    assert registration.model_binding_installation() is owner
    if changed == "pid":
        monkeypatch.setattr(owner, "pid", -1)
    elif changed == "runner":
        monkeypatch.setattr(owner, "runner_class", type("AnotherRunner", (), {}))
    elif changed == "engine":
        monkeypatch.setattr(owner, "engine", {**owner.engine, "version": "different"})
    elif changed == "method":
        monkeypatch.setattr(owner, "model_binding_for", lambda runner: None)
    else:
        monkeypatch.setattr(type(owner), "model_binding_for", lambda self, runner: None)
    with pytest.raises(ValueError):
        registration.model_binding_installation()


def test_model_registration_does_not_retain_installation(tmp_path):
    owner = V.Installation(type("HostRunner", (), {}), tmp_path, {}, {}, lambda: None)
    registration = B.Registration(owner.runner_class, B._freeze(owner.engine), weakref.ref(owner))
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    with pytest.raises(ValueError):
        registration.model_binding_installation()


def test_model_registration_rejects_foreign_runner(bridged):
    owner = bridged.installation
    registration = B.Registration(owner.runner_class, B._freeze(owner.engine), weakref.ref(owner))
    with pytest.raises(ValueError):
        registration.model_binding_for(NS(model_config=bridged.runner.model_config))


def run(value):
    value.runner.initialize_kv_cache(value.runner.kv_cache_config)
    result = value.runner.execute_model(value.scheduler)
    assert result is value.runner.model.result
    return value.installation.states[value.runner]


def test_one_hook_same_copy_and_immutable_callback_view(bridged, monkeypatch):
    calls = []
    original = V._copy_integer
    monkeypatch.setattr(V, "_copy_integer", lambda *a: (calls.append(a[1]), original(*a))[1])
    state = run(bridged)
    assert len(bridged.runner.model._forward_pre_hooks) == 1
    assert len(bridged.installation.wrappers) == 4
    assert len(calls) == 8
    value, = bridged.observer.copies
    assert value.record["requests"] == ("request-b", "request-a")
    assert value.record["layers"][0]["block_table"] == ((1, 2), (3,))
    with pytest.raises(TypeError):
        value.record["layers"][0]["block_table"][0][0] = 7
    with pytest.raises(TypeError):
        value.record["requests"] = ("other",)
    next(iter(bridged.context.attn_metadata.values())).block_table.data[0][0] = 7
    assert value.record["layers"][0]["block_table"][0][0] == 1
    assert names(bridged) == ["retire", "ready", "enter", "prepared", "begin", "copied", "leave"]
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


@pytest.mark.parametrize("change", ["layer-table", "write-slot", "repeated-block"])
def test_raw_contradiction_reaches_callback_before_forward_semantic_validation(bridged, change):
    metadata = next(iter(bridged.context.attn_metadata.values()))
    if change == "layer-table":
        metadata.block_table.data[0][0] = 6
    elif change == "write-slot":
        next(iter(bridged.context.slot_mapping.values())).data[0] = 1
    else:
        metadata.block_table.data[0][1] = 1
    state = run(bridged)
    assert len(bridged.observer.copies) == 1
    assert bridged.installation.bridge.health.snapshot()["healthy"] is True
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "unknown"


@pytest.mark.parametrize("where", ["ready", "enter", "prepared", "begin", "copied", "leave"])
def test_callback_exception_does_not_discard_valid_forward_or_native_result(bridged, where):
    bridged.observer.fail = where
    state = run(bridged)
    health = bridged.installation.bridge.health.snapshot()
    assert health["healthy"] is False
    assert "private callback payload" not in json.dumps(health)
    bridged.observer.fail = None
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


@pytest.mark.parametrize("error", [RuntimeError("native"), KeyboardInterrupt(), SystemExit(3)])
def test_native_exception_identity_survives_failed_observer_cleanup(bridged, error):
    bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    bridged.runner.model.error = error
    bridged.observer.fail = "leave"
    with pytest.raises(type(error)) as caught:
        bridged.runner.execute_model(bridged.scheduler)
    assert caught.value is error
    assert bridged.runner.execute_calls == 1


def test_normal_native_shutdown_closes_selected_observer_without_manual_detach(bridged):
    state = run(bridged)
    old_model = bridged.runner.model
    def closed(runner, exception_type):
        assert not hasattr(runner, "kv_cache_config")
        assert state.writer._closed
    bridged.observer.closed_check = closed
    before = names(bridged)
    assert bridged.runner.shutdown() is bridged.runner.shutdown_result
    assert names(bridged)[len(before):] == ["retire", "runner_closed"]
    assert not old_model._forward_pre_hooks
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"
    bridged.runner.shutdown()
    assert names(bridged).count("runner_closed") == 1
    assert not bridged.installation.bridge.closed
    assert "close" not in names(bridged)


def test_shutdown_cleanup_base_exception_cannot_mask_original(bridged):
    run(bridged)
    original = RuntimeError("native shutdown")
    bridged.runner.shutdown_error = original
    bridged.observer.closed_error = KeyboardInterrupt()
    with pytest.raises(RuntimeError) as caught:
        bridged.runner.shutdown()
    assert caught.value is original
    assert bridged.installation.bridge.health.snapshot()["healthy"] is False


@pytest.mark.parametrize("kind", ["global", "wrapper", "single-foreign-hook"])
def test_boundary_coverage_loss_cannot_complete_callback_copy(bridged, kind):
    bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    state = bridged.installation.states[bridged.runner]
    if kind == "global":
        bridged.types["global_hook_registry"]._global_forward_pre_hooks[1] = lambda: None
    elif kind == "wrapper":
        own = type(bridged.runner).shutdown
        type(bridged.runner).shutdown = lambda runner: own(runner)
    else:
        key = next(iter(bridged.runner.model._forward_pre_hooks))
        bridged.runner.model._forward_pre_hooks[key] = lambda *args: None
    bridged.runner.execute_model(bridged.scheduler)
    assert not bridged.observer.copies
    assert bridged.installation.bridge.health.snapshot()["healthy"] is False


def test_callback_replacement_and_copy_bound_are_independent_of_forward(bridged, monkeypatch):
    bridged.observer.prepared = lambda *args: None
    state = run(bridged)
    assert not bridged.observer.copies
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


@pytest.mark.parametrize("limit", ["MAX_CALLBACKS", "MAX_IN_FLIGHT"])
def test_callback_caps_are_explicit_and_native_work_is_not_retried(bridged, monkeypatch, limit):
    monkeypatch.setattr(B, limit, 0)
    state = run(bridged)
    assert bridged.runner.execute_calls == 1
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


def test_fork_reset_never_calls_factory_or_inherited_assignment_lock(bridged):
    parent_state = run(bridged)
    bridge = bridged.installation.bridge
    bridge._lock.acquire()
    bridge.health._lock.acquire()
    bridged.installation.pid = bridge.pid = -1
    try:
        bridged.installation._pid()
        assert bridged.installation.bridge is None
        assert "not observed" in bridged.installation.bridge_error
    finally:
        bridge._lock.release()
        bridge.health._lock.release()
        parent_state.writer.close()
    bridged.runner.execute_model(bridged.scheduler)
    assert len(bridged.observer.copies) == 1


@pytest.mark.parametrize("boundary", ["changed-pid", "closed"])
def test_direct_initialized_boundary_never_touches_inherited_or_closed_state(bridged, monkeypatch, boundary):
    bridge = bridged.installation.bridge
    touched = []
    class Trap:
        def __enter__(self):
            touched.append("lock")
            raise AssertionError("unavailable bridge state was accessed")
        def __exit__(self, *args):
            return False
    before = list(bridged.observer.events)
    with monkeypatch.context() as patch:
        if boundary == "changed-pid":
            patch.setattr(bridge, "pid", -1)
        else:
            patch.setattr(bridge, "closed", True)
        patch.setattr(bridge, "_lock", Trap())
        patch.setattr(bridge.health, "_lock", Trap())
        # No native state is supplied. An unavailable boundary must return
        # before inspecting it, entering either mutex or calling the observer.
        assert bridge.initialized(None, None, None) is None
    assert touched == []
    assert bridged.observer.events == before


def test_shutdown_does_not_close_another_runner_or_reopen_any_session(bridged):
    from test_vllm_forward import Model
    first = bridged.runner
    second = copy.copy(first)
    second.model = Model()
    for runner in (first, second):
        runner.initialize_kv_cache(runner.kv_cache_config)
        runner.execute_model(bridged.scheduler)
    second_state = bridged.installation.states[second]
    first.shutdown()
    assert not second_state.writer._closed
    assert not bridged.installation.bridge.closed
    assert [row[1] for row in bridged.observer.events if row[0] == "runner_closed"] == [first]
    second.execute_model(bridged.scheduler)
    assert second.shutdown() is second.shutdown_result
    assert len([r for r in shutdown_records(second_state) if r["kind"] == "snapshot"]) == 2
    assert "close" not in names(bridged)


def test_copy_budget_failure_keeps_forward_valid_and_callback_unfinished(bridged):
    bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    state = bridged.installation.states[bridged.runner]
    bridged.installation.bridge._bytes = C.MAX_TOTAL_BYTES
    bridged.runner.execute_model(bridged.scheduler)
    assert not bridged.observer.copies
    assert "abandon" in names(bridged)
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


def test_concurrent_observation_marks_assignment_incomplete_without_wait(bridged):
    bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    state = bridged.installation.states[bridged.runner]
    state.busy.acquire()
    try:
        bridged.runner.execute_model(bridged.scheduler)
    finally:
        state.busy.release()
    assert "enter" in names(bridged) and "abandon" in names(bridged)
    assert not bridged.observer.copies
    assert not bridged.installation.bridge.health.snapshot()["healthy"]


def test_late_factory_request_does_not_replace_active_installation(bridged, monkeypatch):
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(bridged.directory))
    monkeypatch.setattr(V, "_INSTALLATION", bridged.installation)
    different = lambda *args: pytest.fail("late factory called")
    assert V.install(assignment_factory=different)["status"] == "unknown"
    assert V.install(assignment_factory=bridged.factory)["status"] == "installed"
    assert V._INSTALLATION is bridged.installation


def test_none_path_does_not_import_bridge_or_call_factory(tmp_path):
    code = """import importlib.abc,os,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch','vllm') or fullname.endswith('forward_bridge'):
   raise AssertionError('off-path import '+fullname)
sys.meta_path.insert(0,Deny())
from sdcguard.lifetime import vllm_forward as V
os.environ.pop('SDCGUARD_FORWARD_DIR',None)
assert V.install(assignment_factory=lambda *a: (_ for _ in ()).throw(AssertionError())) == {'status':'disabled','hooks':[]}
assert 'sdcguard.lifetime.forward_bridge' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_pinned_native_initial_copy_reaches_shared_initializer_callback(bridged):
    class Quant(Enum):
        NONE = 0
    # Native KVQuantMode is an enum, whose identity survives deepcopy.
    bridged.types["quant_none"] = Quant.NONE
    bridged.runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec.kv_quant_mode = Quant.NONE
    fixtures = Path(__file__).with_name("fixtures") / "forward_bridge"
    descriptor = json.loads((fixtures / "SOURCE.json").read_text())
    fragment = (fixtures / "runner_init.txt").read_bytes()
    assert descriptor["source_sha256"] == C.source_pins()[descriptor["path"]]
    assert hashlib.sha256(fragment).hexdigest() == descriptor["fragment_sha256"]
    scope = {"deepcopy": copy.deepcopy}
    exec(compile("from __future__ import annotations\n" + textwrap.dedent(fragment.decode()),
                 str(fixtures / "runner_init.txt"), "exec"), scope)
    bridged.installation.detach()
    type(bridged.runner).initialize_kv_cache = scope["initialize_kv_cache"]
    bridged.runner.parallel_config.cp_kv_cache_interleave_size = 1
    bridged.installation = V.Installation(type(bridged.runner), bridged.directory, bridged.engine,
        bridged.types, lambda: bridged.context, assignment_factory=bridged.factory).attach()
    input_config = bridged.runner.kv_cache_config
    assert bridged.runner.initialize_kv_cache(input_config) is None
    boundary = [event[1] for event in bridged.observer.events if event[0] == "ready"][-1]
    assert dict(boundary.arguments)["kv_cache_config"] is input_config
    assert boundary.runner_config is bridged.runner.kv_cache_config
    assert boundary.runner_config is not input_config
    assert boundary.block_tables is bridged.runner.block_tables
    assert bridged.runner.cp_interleave == 1


def test_factory_failure_leaves_separately_requested_forward_observer_usable(bridged):
    bridged.installation.detach()
    def fail(registration, health):
        raise ValueError("private factory message")
    bridged.installation = V.Installation(type(bridged.runner), bridged.directory, bridged.engine,
        bridged.types, lambda: bridged.context, assignment_factory=fail).attach()
    assert bridged.installation.bridge is None
    assert bridged.installation.bridge_error == "assignment setup: ValueError"
    state = run(bridged)
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


@pytest.mark.parametrize("child", [False, True])
def test_repeated_explicit_install_keeps_bridge_unknown_status_visible(bridged, monkeypatch, child):
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(bridged.directory))
    monkeypatch.setattr(V, "_INSTALLATION", bridged.installation)
    if child:
        monkeypatch.setattr(bridged.installation, "pid", -1)
    else:
        bridged.installation.bridge_error = "assignment setup: ValueError"
    result = V.install(assignment_factory=bridged.factory)
    assert result["status"] == "installed"
    assert result["assignment_bridge"]["status"] == "unknown"
    assert result["assignment_bridge"]["reason"]


def test_profiling_retires_without_ready_and_reinit_uses_same_process_observer(bridged):
    state = run(bridged)
    factory_bridge = bridged.installation.bridge
    before = len(bridged.observer.events)
    config = bridged.runner.kv_cache_config
    bridged.runner.initialize_kv_cache(config, is_profiling=True)
    assert names(bridged)[before:] == ["retire"]
    assert state.writer._closed
    assert not bridged.installation.states
    bridged.runner.initialize_kv_cache(config)
    assert bridged.installation.bridge is factory_bridge
    assert names(bridged)[before:] == ["retire", "retire", "ready"]


@pytest.mark.parametrize("kind", ["invocation", "frame", "cross-role"])
def test_observer_tokens_cannot_be_reused_even_after_native_invocation_finished(bridged, kind):
    if kind == "invocation":
        bridged.observer.invocation_token = "repeated-token"
    elif kind == "frame":
        bridged.observer.frame_token = "repeated-token"
    else:
        bridged.observer.invocation_token = bridged.observer.frame_token = "same-role-token"
    state = run(bridged)
    bridged.runner.execute_model(bridged.scheduler)
    assert len(bridged.observer.copies) == (0 if kind == "cross-role" else 1)
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    bridged.runner.shutdown()
    assert C.audit_records(shutdown_records(state))["observation_status"] == "observed"


def test_stale_internal_invocation_cannot_deliver_or_finish_another_copy(bridged):
    retained = []
    bridged.runner.model.after_hook = lambda: retained.append(bridged.installation.local.active["assignment"])
    state = run(bridged)
    bridge = bridged.installation.bridge
    stale, = retained
    before = len(bridged.observer.events)
    bridge.prepared(stale, bridged.runner, ["request-b", "request-a"], 3)
    bridge.begin(stale, state, {"pending": (["request-b", "request-a"], 3), "forward_id": 2})
    bridge.copied(stale, state, {})
    bridge.leave(stale, runner=bridged.runner, returned=True, exception_type=None, poisoned=False)
    assert len(bridged.observer.events) == before
    assert not bridge.health.snapshot()["healthy"]
    assert not bridge._inflight


def test_cross_runner_and_forged_context_cannot_use_a_live_invocation(bridged):
    from test_vllm_forward import Model
    second = copy.copy(bridged.runner)
    second.model = Model()
    for runner in (bridged.runner, second):
        runner.initialize_kv_cache(runner.kv_cache_config)
    bridge = bridged.installation.bridge
    second_state = bridged.installation.states[second]
    checked = []
    def misuse():
        invocation = bridged.installation.local.active["assignment"]
        count = len(bridged.observer.events)
        bridge.prepared(invocation, second, ["request-b", "request-a"], 3)
        bridge.begin(invocation, second_state, {"pending": (["request-b", "request-a"], 3), "forward_id": 1})
        bridge.copied(invocation, second_state, {})
        bridge.leave(invocation, runner=second, returned=True, exception_type=None, poisoned=False)
        forged = B._Invocation(invocation.key, invocation.token, invocation.runner, invocation.pid)
        bridge.prepared(forged, bridged.runner, ["request-b", "request-a"], 3)
        assert len(bridged.observer.events) == count
        assert bridge._inflight[invocation.key] is invocation
        checked.append(True)
    bridged.runner.model.after_hook = misuse
    bridged.runner.execute_model(bridged.scheduler)
    assert checked == [True] and not bridge._inflight
    assert not bridge.health.snapshot()["healthy"]
    assert len(bridged.observer.copies) == 1


@pytest.mark.parametrize("where", ["retire", "ready", "enter", "prepared", "begin", "copied"])
@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(4)])
def test_ordinary_callback_process_control_is_not_swallowed(bridged, where, error):
    if where not in ("retire", "ready"):
        bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    bridged.observer.control_name = where
    bridged.observer.control_error = error
    call = (lambda: bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)) if where in (
        "retire", "ready") else lambda: bridged.runner.execute_model(bridged.scheduler)
    with pytest.raises(type(error)) as caught:
        call()
    assert caught.value is error
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    if where == "retire":
        assert bridged.runner.init_calls == 0
    if where == "enter":
        assert bridged.runner.execute_calls == 0
        assert not bridged.installation.bridge._inflight
    bridged.observer.control_name = None


@pytest.mark.parametrize("where", ["leave", "runner_closed"])
@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit(9)])
@pytest.mark.parametrize("native_error", [None, RuntimeError("original native")])
def test_cleanup_control_cannot_replace_native_return_or_exception(bridged, where, control, native_error):
    bridged.runner.initialize_kv_cache(bridged.runner.kv_cache_config)
    bridged.observer.control_name = where
    bridged.observer.control_error = control
    if where == "leave":
        bridged.runner.model.error = native_error
        result = bridged.runner.model.result
        call = lambda: bridged.runner.execute_model(bridged.scheduler)
    else:
        bridged.runner.shutdown_error = native_error
        result = bridged.runner.shutdown_result
        call = bridged.runner.shutdown
    if native_error is None:
        assert call() is result
    else:
        with pytest.raises(RuntimeError) as caught:
            call()
        assert caught.value is native_error
    assert not bridged.installation.bridge.health.snapshot()["healthy"]
    bridged.observer.control_name = None
