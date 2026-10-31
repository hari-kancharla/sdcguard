"""Passive plan preparation and shell activation; no engine or GPU is executed."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from sdcguard import assessment
from sdcguard import recording_plan as plan


@pytest.fixture
def supported(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    source_pin = deepcopy(assessment.pins())
    expected = {}
    for name in source_pin["variants"]["fixed"]:
        path = engine / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# inert source identity fixture: " + name + "\n")
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source_pin["variants"]["fixed"] = expected
    source_pin["variants"]["stock"] = dict(expected, **{next(iter(expected)): None})
    monkeypatch.setattr(assessment, "pins", lambda: deepcopy(source_pin))
    config = {"schema": 1, "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
              "engine_root": "engine", "model": "Qwen/Qwen3-0.6B", "model_revision": "a" * 40,
              "topology": deepcopy(assessment.TOPOLOGY)}
    return config, engine


def test_plan_preserves_declared_deployment_and_starts_with_empty_shared_recording(tmp_path, supported):
    config, engine = supported
    config["ready"] = {"metrics_urls": ["http://127.0.0.1:8000/metrics"]}
    config["events"] = "old-recording"
    before = deepcopy(config)
    out = tmp_path / "new-plan"
    result = plan.prepare_recording_plan(config, out, base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "prepared"
    assert config == before
    deployment = json.loads((out / "deployment.json").read_text())
    assert deployment == dict(before, events=str(out / "events"), engine_root=str(engine))
    assert list((out / "events").iterdir()) == []
    manifest = json.loads((out / "PLAN_MANIFEST.json").read_text())
    assert manifest["runtime_evidence"] == "not_collected"
    assert manifest["engine"]["fix"] == "present"
    assert "not attested" in manifest["declared_model"]["basis"]
    assert "unverified" in manifest["plugin_loading"]
    for name, sha in manifest["files_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    assert "not a readiness or correctness verdict" in (out / "README.txt").read_text()


@pytest.mark.parametrize("change", [
    {"schema": True}, {"engine": "other"}, {"engine_version": "0.33.0"},
    {"nixl_version": "1.4.2"}, {"model": "other/model"}, {"model_revision": "main"},
    {"model_revision": "a" * 39}, {"model_revision": "A" * 40}, {"engine_root": None},
    {"events": []}, {"unknown": True}, {"ready": {"metrics_urls": []}},
])
def test_unsupported_or_missing_fact_is_inconclusive_without_output(tmp_path, supported, change):
    config, _ = supported
    config.update(change)
    out = tmp_path / "never-created"
    result = plan.prepare_recording_plan(config, out, base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive" and result["missing"]
    assert not out.exists()


@pytest.mark.parametrize("config", [None, [], {}, {"schema": 1}])
def test_invalid_config_shapes_never_prepare(tmp_path, config):
    result = plan.prepare_recording_plan(config, tmp_path / "out", base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", [{"tp": 2}, {"same_host": 1}, {"eager": False}, {"extra": 1}])
def test_topology_cannot_be_coerced_or_expanded(tmp_path, supported, change):
    config, _ = supported
    config["topology"].update(change)
    result = plan.validate_deployment(config, base=tmp_path, environ={})
    assert result["status"] == "inconclusive"


def test_changed_source_does_not_fit_based_on_version_string(tmp_path, supported):
    config, engine = supported
    next(path for path in engine.rglob("*.py") if path.is_file()).write_text("# different source\n")
    result = plan.prepare_recording_plan(config, tmp_path / "out", base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive"
    assert result["engine"]["fix"] == "unknown"
    assert not (tmp_path / "out").exists()


def test_supported_stock_source_can_be_observed_without_claiming_fix_presence(tmp_path, supported):
    config, engine = supported
    absent = next(iter(assessment.pins()["variants"]["stock"]))
    (engine / absent).unlink()
    result = plan.prepare_recording_plan(config, tmp_path / "out", base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "prepared"
    assert result["engine"]["adapter_fits"] is True
    assert result["engine"]["fix"] == "absent"
    assert "not prove" in result["limits"][0]


@pytest.mark.parametrize("change", [{"engine_root": "bad\0path"}, {"engine_version": float("nan")}])
def test_malformed_source_declaration_is_structured_inconclusive(tmp_path, supported, change):
    config, _ = supported
    config.update(change)
    report = plan.validate_deployment(config, base=tmp_path, environ={})
    assert report["status"] == "inconclusive"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("destination", [None, "bad\0path"])
def test_invalid_output_path_never_raises_or_prepares(supported, destination):
    config, engine = supported
    report = plan.prepare_recording_plan(config, destination, base=engine.parent, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert report["status"] == "inconclusive"


@pytest.mark.parametrize("env", [
    {"SDCGUARD_LIFETIME_HOLDS": "{}"}, {"SDCGUARD_LIFETIME_HOLDS": " "},
    {"SDCGUARD_NIXL_UCX_PARAMS": "{}"}, {"SDCGUARD": "1"}, {"SDCGUARD_REQUIRE": "1"},
    {"VLLM_PLUGINS": "other"}, {"VLLM_PLUGINS": ""}, {"VLLM_PLUGINS": " sdcguard"},
])
def test_conflicting_inherited_settings_refuse_preparation(tmp_path, supported, env):
    config, _ = supported
    result = plan.prepare_recording_plan(config, tmp_path / "out", base=tmp_path, environ=env, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive"
    assert not (tmp_path / "out").exists()


def source_script(path, env=None):
    setting = {"PATH": os.defpath}
    setting.update(env or {})
    command = '. "$1"; _record_plan_test_status=$?; printf "%s\\n" "${SDCGUARD_LIFETIME_DIR-unset}" "${SDCGUARD_LIFETIME_ENGINE-unset}" "${SDCGUARD_LIFETIME_RECORD-unset}" "${SDCGUARD-unset}" "${VLLM_PLUGINS-unset}"; exit "$_record_plan_test_status"'
    return subprocess.run(["sh", "-c", command, "record-plan-test", str(path)], env=setting,
                          cwd=path.parent.parent, capture_output=True, text=True)


def test_snippets_quote_paths_and_only_enable_observation(tmp_path, supported):
    config, _ = supported
    out = tmp_path / "plan's $(touch injected) `touch other`"
    result = plan.prepare_recording_plan(config, out, base=tmp_path,
                                         environ={"VLLM_PLUGINS": "other,sdcguard"}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "prepared"
    for role in ("producer", "consumer"):
        sourced = source_script(out / (role + ".env.sh"), {"VLLM_PLUGINS": "other,sdcguard"})
        assert sourced.returncode == 0, sourced.stderr
        assert sourced.stdout.splitlines() == [str(out / "events"), {"producer": "P", "consumer": "D"}[role], "1", "unset", "other,sdcguard"]
    assert not (tmp_path / "injected").exists()
    assert not (tmp_path / "other").exists()
    assert list((out / "events").iterdir()) == []


@pytest.mark.parametrize("env", [
    {"SDCGUARD_LIFETIME_HOLDS": "{}"}, {"SDCGUARD_NIXL_UCX_PARAMS": "{}"},
    {"SDCGUARD_REQUIRE": "1"}, {"SDCGUARD": "1"}, {"VLLM_PLUGINS": "other"},
    {"VLLM_PLUGINS": ""},
])
def test_snippet_rechecks_settings_and_preserves_environment_on_refusal(tmp_path, supported, env):
    config, _ = supported
    out = tmp_path / "out"
    assert plan.prepare_recording_plan(config, out, base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")["status"] == "prepared"
    inherited = dict(env, SDCGUARD_LIFETIME_DIR="earlier-dir", SDCGUARD_LIFETIME_ENGINE="earlier-engine",
                     SDCGUARD_LIFETIME_RECORD="0")
    sourced = source_script(out / "producer.env.sh", inherited)
    assert sourced.returncode == 2
    assert "refused" in sourced.stderr
    assert sourced.stdout.splitlines()[:3] == ["earlier-dir", "earlier-engine", "0"]


@pytest.mark.parametrize("kind", ["directory", "file", "dangling_symlink"])
def test_existing_destination_is_never_overwritten(tmp_path, supported, kind):
    config, _ = supported
    out = tmp_path / "existing"
    if kind == "directory":
        out.mkdir()
        (out / "keep").write_text("owner data")
    elif kind == "file":
        out.write_text("owner data")
    else:
        out.symlink_to(tmp_path / "absent", target_is_directory=True)
    result = plan.prepare_recording_plan(config, out, base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive"
    if kind == "directory":
        assert (out / "keep").read_text() == "owner data" and sorted(x.name for x in out.iterdir()) == ["keep"]
    elif kind == "file":
        assert out.read_text() == "owner data"
    else:
        assert out.is_symlink() and not (tmp_path / "absent").exists()


def test_failure_preserves_partial_files_and_has_no_completion_manifest(tmp_path, supported, monkeypatch):
    config, _ = supported
    write = plan._write_new
    order = []

    def interrupted(path, content):
        order.append(path.name)
        if path.name == "consumer.env.sh":
            raise OSError("fixture write failure")
        return write(path, content)

    monkeypatch.setattr(plan, "_write_new", interrupted)
    out = tmp_path / "partial"
    result = plan.prepare_recording_plan(config, out, base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "inconclusive"
    assert result["partial_output"] == str(out)
    assert (out / "deployment.json").is_file() and (out / "producer.env.sh").is_file()
    assert not (out / "PLAN_MANIFEST.json").exists()
    assert order == ["deployment.json", "producer.env.sh", "consumer.env.sh"]


def test_completion_manifest_is_last_and_no_process_is_started(tmp_path, supported, monkeypatch):
    config, _ = supported
    order = []
    write = plan._write_new

    def observed(path, content):
        order.append(path.name)
        return write(path, content)

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not start a process")

    monkeypatch.setattr(plan, "_write_new", observed)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = plan.prepare_recording_plan(config, tmp_path / "out", base=tmp_path, environ={}, producer_engine_id="P", consumer_engine_id="D")
    assert result["status"] == "prepared"
    assert order[-1] == "PLAN_MANIFEST.json"


def test_main_reports_invalid_json_without_output(tmp_path, capsys):
    config = tmp_path / "invalid.json"
    config.write_text("not JSON")
    assert plan.main([str(config), "--out", str(tmp_path / "out"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "inconclusive"
    assert not (tmp_path / "out").exists()


def test_main_uses_input_config_directory_for_relative_engine_root(tmp_path, supported, monkeypatch, capsys):
    config, _ = supported
    path = tmp_path / "deployment.json"
    path.write_text(json.dumps(config))
    for name in (*plan._BLOCKED_NONEMPTY, "SDCGUARD", "VLLM_PLUGINS"):
        monkeypatch.delenv(name, raising=False)
    assert plan.main([str(path), "--out", str(tmp_path / "out"), "--json", "--producer-engine-id", "P", "--consumer-engine-id", "D"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


@pytest.mark.parametrize('producer,consumer', [(None,None), ('P',None), ('P','P'), ('P\nwrong','D'), ('','D')])
def test_plan_requires_distinct_real_engine_ids_before_writing(tmp_path, supported, producer, consumer):
    config, _ = supported
    report = plan.prepare_recording_plan(config, tmp_path / 'missing-ids', base=tmp_path, environ={},
        producer_engine_id=producer, consumer_engine_id=consumer)
    assert report['status'] == 'inconclusive' and not (tmp_path / 'missing-ids').exists()


def test_generated_fallback_and_connector_ids_attribute_the_same_regions(tmp_path, supported, monkeypatch):
    from types import SimpleNamespace
    from test_lifetime_vllm_nixl import Manager, ConnScheduler
    from sdcguard.lifetime import vllm_nixl as observer
    from sdcguard.lifetime.adapter import Probe, JsonlDirSink
    config, _ = supported
    out = tmp_path / 'composed'
    assert plan.prepare_recording_plan(config, out, base=tmp_path, environ={},
        producer_engine_id='P', consumer_engine_id='D')['status'] == 'prepared'
    env = source_script(out / 'producer.env.sh').stdout.splitlines()
    monkeypatch.setenv('SDCGUARD_LIFETIME_ENGINE', env[1])
    observer.reset()
    sink = JsonlDirSink(str(out / 'events'))
    observer._PROBE = Probe(sink)
    manager_cls = type('ManagerWithoutEngineId', (Manager,), {})
    connector_cls = type('ActualConnector', (ConnScheduler,), {'engine_id':'P'})
    observer.install_manager(manager_cls)
    observer.install_connector(SchedulerCls=connector_cls)
    manager = manager_cls('unused')
    del manager.engine_id  # Real KVCacheManager has no connector engine_id field.
    request = SimpleNamespace(request_id='ordinary-single-request')
    manager.allocate_slots(request, 1)
    connector_cls().request_finished(request, manager.get_block_ids(request.request_id))
    sink.flush()
    events = sink.history()
    allocated = next(e['args']['region'] for e in events if e['kind']=='alloc')
    published = next(e['args']['regions'][0] for e in events if e['kind']=='publish')
    assert allocated == published and allocated[0] == 'P'
    observer.reset()
