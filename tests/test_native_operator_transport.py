"""Inert transport policy controls. Owner: Hari Krishna Kancharla.

These controls use explicit native stand-ins and do not qualify UCX,
the cloud runtime, CUDA, a model, or serving traffic.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(BUNDLE))
from sdcguard.native_operator import transport


def synthetic_events():
    requested = dict(num_threads="4")
    supplied = dict(requested, ucx_error_handling_mode="none")
    return [dict(kind="backend_attempt", call_id=1, backend="UCX", requested_parameters=requested),
        dict(kind="backend_return", call_id=1, backend="UCX", supplied_parameters=supplied,
            accepted_parameters=dict(supplied), native_returned=True, api_sha256=transport.API_SHA256)]


def synthetic_transport(run_id, packet_sha256):
    """Return plainly synthetic observations for offline gate controls only."""
    identity = dict(path="/synthetic/nixl_cu13.libs/" + transport.UCP_NAME,
                    sha256=transport.UCP_SHA256, version=[1, 23, 0])
    common = dict(schema=transport.CASE_SCHEMA, source_verified=True, api_sha256=transport.API_SHA256,
        api_path="/synthetic/nixl_cu13/_api.py", explicit_gpu_allocation=False,
        explicit_cuda_workload=False, returncode=0, child_exit_observed=True, stdout="", stderr="",
        python_executable="/synthetic/python")
    return dict(schema=transport.SCHEMA, owner="Hari Krishna Kancharla", run_id=run_id,
        packet_sha256=packet_sha256, status="PASS", failure=None, scope=transport.SCOPE,
        profile=transport.profile(), gpu_submitted=False, parent_process_id=10000,
        cases=dict(selected=dict(common, mode="selected", status="PASS", failure=None, stage="complete", process_id=10002,
                events=synthetic_events(), accepted_parameters=dict(num_threads="4", ucx_error_handling_mode="none"),
                ucx_identity=identity, agent_released=True)))


def fake_api(monkeypatch):
    calls = []
    class Agent:
        def create_backend(self, backend, initParams={}):
            calls.append((self, backend, dict(initParams)))
            if getattr(self, "native_error", None) is not None:
                raise self.native_error
            self.accepted = dict(initParams)
            if getattr(self, "return_other_mode", False):
                self.accepted["ucx_error_handling_mode"] = "peer"
            return getattr(self, "native_return", None)

        def get_backend_params(self, backend):
            assert backend == "UCX"
            return dict(self.accepted)

    monkeypatch.setattr(transport, "_guard_api", lambda api: Agent.__dict__["create_backend"])
    return SimpleNamespace(nixl_agent=Agent), calls


def test_policy_preserves_other_options_return_and_caller_map(monkeypatch):
    api, calls = fake_api(monkeypatch)
    events = []
    policy = transport.install_policy(events.append, api=api)
    agent = api.nixl_agent()
    sentinel = agent.native_return = object()
    requested = dict(num_threads="4", engine_config="MAX_RMA_RAILS=2")
    before = dict(requested)
    assert agent.create_backend("UCX", requested) is sentinel
    assert requested == before
    assert calls[0][1:] == ("UCX", dict(before, ucx_error_handling_mode="none"))
    assert policy == transport.profile() and transport.validate_events(events)
    assert events[1]["accepted_parameters"] == calls[0][2]


def test_non_ucx_backend_passes_through_without_transport_receipt(monkeypatch):
    api, calls = fake_api(monkeypatch)
    events = []
    transport.install_policy(events.append, api=api)
    agent = api.nixl_agent()
    agent.create_backend("OTHER", dict(num_threads="2"))
    assert calls[0][1:] == ("OTHER", dict(num_threads="2")) and events == []


def test_explicit_conflict_is_recorded_and_never_reaches_native_constructor(monkeypatch):
    api, calls = fake_api(monkeypatch)
    events = []
    transport.install_policy(events.append, api=api)
    with pytest.raises(ValueError, match="conflicting"):
        api.nixl_agent().create_backend("UCX", dict(ucx_error_handling_mode="peer"))
    assert calls == [] and [x["kind"] for x in events] == ["backend_attempt", "backend_failure"]
    assert not transport.validate_events(events)


def test_unsupported_parameter_shape_still_records_the_attempt_and_hold_reason(monkeypatch):
    api, calls = fake_api(monkeypatch)
    events = []
    transport.install_policy(events.append, api=api)
    with pytest.raises(ValueError, match="parameter map"):
        api.nixl_agent().create_backend("UCX", None)
    assert calls == [] and [event["kind"] for event in events] == ["backend_attempt", "backend_failure"]
    assert events[0]["requested_parameters"] is None and not transport.validate_events(events)


def test_native_error_is_not_swallowed_or_replaced_by_recorder_error(monkeypatch):
    api, _ = fake_api(monkeypatch)
    events = []
    def record(value):
        events.append(value)
        if value["kind"] == "backend_failure":
            raise OSError("inert receipt failure")
    transport.install_policy(record, api=api)
    agent = api.nixl_agent()
    original = agent.native_error = RuntimeError("inert native error")
    with pytest.raises(RuntimeError) as caught:
        agent.create_backend("UCX", {})
    assert caught.value is original
    assert events[-1]["kind"] == "backend_failure"


def test_installer_refuses_a_second_recorder_or_replaced_wrapper(monkeypatch):
    api, _ = fake_api(monkeypatch)
    events = []
    callback = events.append
    assert transport.install_policy(callback, api=api) == transport.install_policy(callback, api=api)
    with pytest.raises(ValueError, match="another recorder"):
        transport.install_policy(lambda event: None, api=api)
    monkeypatch.setattr(api.nixl_agent, "create_backend", lambda *args: None)
    with pytest.raises(ValueError, match="differs"):
        transport.install_policy(callback, api=api)


def test_native_accepted_parameters_must_match_the_supplied_map(monkeypatch):
    api, _ = fake_api(monkeypatch)
    events = []
    transport.install_policy(events.append, api=api)
    agent = api.nixl_agent()
    agent.return_other_mode = True
    with pytest.raises(ValueError, match="did not retain"):
        agent.create_backend("UCX", {})
    assert events[-1]["kind"] == "backend_failure" and not transport.validate_events(events)


def guarded_fixture(tmp_path, monkeypatch):
    source = ("class nixl_agent:\n"
              "    def create_backend(self, backend: str, initParams: dict[str, str] = {}):\n"
              "        return self.agent.createBackend(backend, initParams)\n")
    path = tmp_path / "_api.py"
    path.write_text(source)
    module = ModuleType("nixl_cu13._api")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), vars(module))
    monkeypatch.setattr(transport, "API_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return module, path


def test_source_guard_accepts_matching_inert_source_and_loaded_method(tmp_path, monkeypatch):
    api, _ = guarded_fixture(tmp_path, monkeypatch)
    assert transport._guard_api(api) is api.nixl_agent.create_backend


@pytest.mark.parametrize("change", ["source", "method", "defaults", "namespace", "filename", "globals"])
def test_source_guard_rejects_changed_source_or_loaded_code(tmp_path, monkeypatch, change):
    api, path = guarded_fixture(tmp_path, monkeypatch)
    original = api.nixl_agent.create_backend
    if change == "source":
        path.write_text(path.read_text() + "# inert changed bytes\n")
    elif change == "method":
        api.nixl_agent.create_backend = lambda self, backend, initParams={}: None
    elif change == "defaults":
        original.__defaults__ = ({"ucx_error_handling_mode": "peer"},)
    elif change == "namespace":
        api.__name__ = "nixl_cu12._api"
    elif change == "filename":
        original.__code__ = original.__code__.replace(co_filename=str(tmp_path / "elsewhere.py"))
    else:
        from types import FunctionType
        api.nixl_agent.create_backend = FunctionType(original.__code__, {}, original.__name__, original.__defaults__)
    with pytest.raises(ValueError):
        transport._guard_api(api)


@pytest.mark.parametrize("change", ["missing-return", "extra-call", "accepted-peer", "drop-other-param",
    "different-call", "failure", "wrong-source", "wrong-backend", "boolean-id"])
def test_event_verification_does_not_accept_installation_intent_as_native_acceptance(change):
    events = synthetic_events()
    if change == "missing-return":
        events.pop()
    elif change == "extra-call":
        events += deepcopy(events)
    elif change == "accepted-peer":
        events[1]["accepted_parameters"]["ucx_error_handling_mode"] = "peer"
    elif change == "drop-other-param":
        del events[1]["accepted_parameters"]["num_threads"]
    elif change == "different-call":
        events[1]["call_id"] = 2
    elif change == "failure":
        events[1]["kind"] = "backend_failure"
    elif change == "wrong-source":
        events[1]["api_sha256"] = "0" * 64
    elif change == "wrong-backend":
        events[1]["backend"] = "OTHER"
    else:
        events[0]["call_id"] = True
    assert not transport.validate_events(events)


def test_receipt_is_bound_to_each_actual_worker_session_and_role():
    value = dict(schema=transport.RECEIPT_SCHEMA, status="PASS", role="decode", profile=transport.profile(),
        context=dict(run_id="1" * 32, packet_sha256="2" * 64, arm_id="observer-on",
            engine_session_id="3" * 32, process_id=1000), events=synthetic_events(),
        process_identity=dict(pid=1000, start=15, group=900, session=900))
    args = ("1" * 32, "2" * 64, "observer-on", "3" * 32, "decode")
    assert transport.validate_receipt(value, *args)
    for key, changed in (("run_id", "9" * 32), ("packet_sha256", "9" * 64),
                         ("engine_session_id", "9" * 32), ("arm_id", "observer-off"), ("process_id", True)):
        other = deepcopy(value)
        other["context"][key] = changed
        assert not transport.validate_receipt(other, *args)
    value["role"] = "prefill"
    assert not transport.validate_receipt(value, *args)


@pytest.mark.parametrize("change", ["selected-failed", "native-import-unavailable", "not-reaped",
    "wrong-library", "wrong-version", "wrong-run", "missing-selected", "accepted-peer", "not-bound"])
def test_readiness_holds_when_native_selection_or_identity_is_unverified(change):
    value = synthetic_transport("1" * 32, "2" * 64)
    assert transport.validate(value, "1" * 32, "2" * 64)
    selected = value["cases"]["selected"]
    if change == "selected-failed":
        selected["status"] = "FAIL"
    elif change == "native-import-unavailable":
        selected["source_verified"] = False
        selected["status"] = "UNKNOWN"
    elif change == "not-reaped":
        selected["child_exit_observed"] = False
    elif change == "wrong-library":
        selected["ucx_identity"]["sha256"] = "9" * 64
    elif change == "wrong-version":
        selected["ucx_identity"]["version"] = [1, 22, 0]
    elif change == "wrong-run":
        value["run_id"] = "9" * 32
    elif change == "missing-selected":
        del value["cases"]["selected"]
    elif change == "accepted-peer":
        selected["accepted_parameters"]["ucx_error_handling_mode"] = "peer"
    else:
        value["packet_sha256"] = "9" * 64
    assert not transport.validate(value, "1" * 32, "2" * 64)


def test_probe_uses_only_selected_native_child_and_isolated_clean_environment(monkeypatch):
    calls = []
    expected = synthetic_transport("1" * 32, "2" * 64)
    expected["parent_process_id"] = os.getpid()
    expected["cases"]["selected"]["process_id"] = os.getpid() + 2
    monkeypatch.setenv("UCX_TLS", "inert-inherited-override")
    monkeypatch.setenv("NIXL_PLUGIN_DIR", "/inert/override")
    def run(command, **kwargs):
        calls.append((command, kwargs))
        mode = command[-2]
        case = deepcopy(expected["cases"][mode])
        for key in ("stdout", "stderr", "returncode", "child_exit_observed"):
            del case[key]
        Path(command[-1]).write_text(json.dumps(case))
        assert kwargs["env"]["UCX_RCACHE_MAX_UNRELEASED"] == "1024"
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "UCX_TLS" not in kwargs["env"] and "NIXL_PLUGIN_DIR" not in kwargs["env"]
        assert command[:3] == [sys.executable, "-I", "-B"] and command[-3] == "--native-case"
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(transport.subprocess, "run", run)
    value = transport.probe("1" * 32, "2" * 64)
    assert [command[-2] for command, _ in calls] == ["selected"]
    assert transport.validate(value, "1" * 32, "2" * 64)
    assert value["gpu_submitted"] is False


def test_native_import_failure_is_unknown_without_starting_other_work(monkeypatch):
    def unavailable(name):
        assert name == "nixl_cu13._api"
        raise ImportError("inert unavailable CUDA 13 native library")
    monkeypatch.setattr(transport.importlib, "import_module", unavailable)
    result = transport._native_case("selected")
    assert result["status"] == "UNKNOWN" and result["events"] == []
    assert result["failure"]["kind"] == "ImportError"
