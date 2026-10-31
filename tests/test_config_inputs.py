"""Static input and saved evidence boundaries, using local synthetic files only."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from sdcguard import config_inputs as inputs


def document():
    return {"schema": inputs.SCHEMA, "deployment_id": "fixture", "advertised_context_tokens": 1024,
            "model": {"id": "example/model", "revision": "a" * 40, "architecture": "ExampleArchitecture"},
            "workers": [{"id": "P", "role": "prefill", "engine": "vllm", "version": "0.32.0",
                         "build_commit": "b" * 40, "parameters": {"max_model_len": 1024}}]}


def write_config(base, value=None):
    path = base / "deployment.json"
    path.write_text(json.dumps(document() if value is None else value, indent=2), encoding="utf-8")
    return path


def attach_log(base, worker, text="synthetic saved startup\n", name="startup.log", deployment=None):
    raw = text.encode("utf-8")
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    worker["startup_log"] = {"path": name, "sha256": hashlib.sha256(raw).hexdigest(),
                             "config_sha256": inputs.binding_digest(worker, document() if deployment is None
                                                                    else deployment),
                             "worker_id": worker["id"]}
    return path


def test_exact_original_bytes_and_unknown_parameters_are_preserved(tmp_path):
    value = document()
    value["workers"][0]["parameters"]["future_option"] = {"nested": [True, 3, "value"]}
    path = write_config(tmp_path, value)
    before = path.read_bytes()
    loaded, provenance = inputs.load_deployment(path)
    assert loaded == value
    assert provenance["raw_config_sha256"] == hashlib.sha256(before).hexdigest()
    assert provenance["source_path"] == str(path)
    assert path.read_bytes() == before
    assert "error" in provenance["logs"]["P"]
    assert "no authenticated attestation" in provenance["binding_scope"]


def test_missing_identity_and_one_role_remain_missing(tmp_path):
    value = {"schema": inputs.SCHEMA, "workers": [{"id": "P"}]}
    loaded, _ = inputs.load_deployment(write_config(tmp_path, value))
    assert loaded == value and "model" not in loaded
    assert "version" not in loaded["workers"][0] and "role" not in loaded["workers"][0]


def test_unsupported_declared_identity_is_not_parser_support_claim(tmp_path):
    value = document()
    value["model"] = {"architecture": "UnreviewedArchitecture"}
    value["workers"][0].update(engine="unreviewed-engine", version="custom-release")
    loaded, _ = inputs.load_deployment(write_config(tmp_path, value))
    assert loaded == value


@pytest.mark.parametrize("raw", [
    '{"schema":"sdcguard.config-safety.v1","schema":"sdcguard.config-safety.v1","workers":[]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P","parameters":{"x":1,"x":2}}]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P","parameters":{"x":NaN}}]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P","parameters":{"x":Infinity}}]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P","parameters":{"x":-Infinity}}]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P","parameters":{"x":1e9999}}]}',
    '{"schema":"sdcguard.config-safety.v1","workers":[{"id":"P\\ud800"}]}',
    '[]', '{',
])
def test_ambiguous_or_malformed_json_is_rejected(tmp_path, raw):
    path = tmp_path / "deployment.json"
    path.write_text(raw)
    with pytest.raises(ValueError):
        inputs.load_deployment(path)


@pytest.mark.parametrize("change", [
    lambda d: d.update(extra=True),
    lambda d: d.pop("schema"),
    lambda d: d.update(schema=1),
    lambda d: d.update(advertised_context_tokens=True),
    lambda d: d.update(advertised_context_tokens=0),
    lambda d: d.update(advertised_context_tokens="1024"),
    lambda d: d.update(model=[]),
    lambda d: d["model"].update(extra="unknown"),
    lambda d: d["model"].update(revision="main"),
    lambda d: d["model"].update(architecture=["ExampleArchitecture"]),
    lambda d: d.update(workers=[]),
    lambda d: d.update(workers=[{"id": str(i)} for i in range(33)]),
    lambda d: d["workers"].append(deepcopy(d["workers"][0])),
    lambda d: d["workers"][0].pop("id"),
    lambda d: d["workers"][0].update(extra="unknown"),
    lambda d: d["workers"][0].update(role="both"),
    lambda d: d["workers"][0].update(engine=12),
    lambda d: d["workers"][0].update(version=None),
    lambda d: d["workers"][0].update(build_commit="b" * 39),
    lambda d: d["workers"][0].update(components=[]),
    lambda d: d["workers"][0].update(connector="NixlConnector"),
    lambda d: d["workers"][0].update(parameters=[]),
    lambda d: d["workers"][0].update(environment=None),
    lambda d: d["workers"][0].update(layout=False),
])
def test_invalid_schema_fields_are_rejected(tmp_path, change):
    value = document()
    change(value)
    with pytest.raises(ValueError):
        inputs.load_deployment(write_config(tmp_path, value))


def test_worker_limit_accepts_32_distinct_ids(tmp_path):
    value = {"schema": inputs.SCHEMA, "workers": [{"id": str(i)} for i in range(32)]}
    assert len(inputs.load_deployment(write_config(tmp_path, value))[0]["workers"]) == 32


def test_config_size_utf8_nesting_and_regular_file_bounds(tmp_path):
    path = tmp_path / "deployment.json"
    for raw in [b" " * (inputs.MAX_CONFIG_BYTES + 1), b"\xff", b"[" * 2000 + b"]" * 2000]:
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            inputs.load_deployment(path)
    with pytest.raises(ValueError):
        inputs.load_deployment(tmp_path)
    with pytest.raises(ValueError):
        inputs.load_deployment(tmp_path / "missing")


def test_binding_is_order_independent_and_excludes_only_startup_metadata():
    worker = document()["workers"][0]
    deployment = document()
    payload = {"worker": worker, "model": deployment["model"], "advertised_context_tokens": 1024}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    assert inputs.binding_digest(dict(reversed(list(worker.items()))), deployment) == expected
    worker["startup_log"] = {"arbitrary": "metadata excluded"}
    assert inputs.binding_digest(worker, deployment) == expected
    worker["parameters"]["future_option"] = True
    assert inputs.binding_digest(worker, deployment) != expected
    with pytest.raises(ValueError):
        inputs.binding_digest({"bad": float("nan")}, deployment)


def test_bound_relative_log_loaded_without_altering_files(tmp_path):
    value = document()
    log = attach_log(tmp_path, value["workers"][0], name="logs/startup.log")
    path = write_config(tmp_path, value)
    before = {p: p.read_bytes() for p in [log, path]}
    loaded, provenance = inputs.load_deployment(path)
    evidence = provenance["logs"]["P"]
    assert loaded == value
    assert evidence["text"] == "synthetic saved startup\n"
    assert evidence["path"] == str(log)
    assert evidence["sha256"] == hashlib.sha256(before[log]).hexdigest()
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.parametrize("change,reason", [
    (lambda w: w["startup_log"].update(sha256="0" * 64), "sha256"),
    (lambda w: w["startup_log"].update(config_sha256="0" * 64), "config_sha256"),
    (lambda w: w["startup_log"].update(worker_id="D"), "worker_id"),
    (lambda w: w["parameters"].update(max_model_len=2048), "config_sha256"),
])
def test_bad_evidence_binding_stays_an_explicit_log_error(tmp_path, change, reason):
    value = document()
    attach_log(tmp_path, value["workers"][0])
    change(value["workers"][0])
    loaded, provenance = inputs.load_deployment(write_config(tmp_path, value))
    assert loaded == value
    assert reason in provenance["logs"]["P"]["error"]
    assert "text" not in provenance["logs"]["P"]


@pytest.mark.parametrize("change", [
    lambda d: d["model"].update(revision="c" * 40),
    lambda d: d["model"].update(id="different/model"),
    lambda d: d["model"].update(architecture="DifferentArchitecture"),
    lambda d: d.update(advertised_context_tokens=2048),
])
def test_root_model_or_context_changes_invalidate_saved_startup(tmp_path, change):
    value = document()
    attach_log(tmp_path, value["workers"][0], deployment=value)
    change(value)
    _, provenance = inputs.load_deployment(write_config(tmp_path, value))
    assert "config_sha256" in provenance["logs"]["P"]["error"]
    assert "text" not in provenance["logs"]["P"]


def test_binding_requires_deployment_and_cannot_use_worker_only_mode():
    worker = document()["workers"][0]
    with pytest.raises(TypeError):
        inputs.binding_digest(worker)
    with pytest.raises(ValueError):
        inputs.binding_digest(worker, None)


@pytest.mark.parametrize("metadata", [
    {}, {"path": "startup.log"},
    {"path": "startup.log", "sha256": "0" * 64, "config_sha256": "0" * 64, "extra": True},
    {"path": "startup.log", "sha256": "no", "config_sha256": "0" * 64},
    {"path": "startup.log", "sha256": "0" * 64, "config_sha256": "0" * 64, "worker_id": 3},
])
def test_malformed_log_metadata_is_structural_error(tmp_path, metadata):
    value = document()
    value["workers"][0]["startup_log"] = metadata
    with pytest.raises(ValueError):
        inputs.load_deployment(write_config(tmp_path, value))


@pytest.mark.parametrize("name", ["../outside.log", "/outside.log", "logs/../startup.log", "./startup.log",
                                  "logs//startup.log", "logs\\startup.log", "https://example.invalid/log"])
def test_unsafe_paths_are_never_followed(tmp_path, name):
    worker = document()["workers"][0]
    attach_log(tmp_path, worker)
    worker["startup_log"]["path"] = name
    assert "error" in inputs.read_startup(worker, tmp_path, document())


def test_config_and_log_symlinks_and_parent_symlinks_are_rejected(tmp_path):
    value = document()
    real = attach_log(tmp_path, value["workers"][0])
    alias = tmp_path / "alias.log"
    alias.symlink_to(real)
    value["workers"][0]["startup_log"]["path"] = alias.name
    assert "error" in inputs.read_startup(value["workers"][0], tmp_path, value)
    directory = tmp_path / "aliases"
    directory.symlink_to(tmp_path, target_is_directory=True)
    value["workers"][0]["startup_log"]["path"] = "aliases/startup.log"
    assert "error" in inputs.read_startup(value["workers"][0], tmp_path, value)
    path = write_config(tmp_path)
    config_alias = tmp_path / "alias.json"
    config_alias.symlink_to(path)
    with pytest.raises(ValueError):
        inputs.load_deployment(config_alias)
    with pytest.raises(ValueError):
        inputs.load_deployment(directory / path.name)


def test_bad_or_missing_log_bytes_do_not_crash_loader(tmp_path):
    worker = document()["workers"][0]
    path = attach_log(tmp_path, worker)
    for raw in [b"\xff", b"x" * (inputs.MAX_LOG_BYTES + 1)]:
        path.write_bytes(raw)
        worker["startup_log"]["sha256"] = hashlib.sha256(raw).hexdigest()
        assert "error" in inputs.read_startup(worker, tmp_path, document())
    path.unlink()
    assert "error" in inputs.read_startup(worker, tmp_path, document())


def test_empty_log_is_captured_for_evaluator_to_diagnose(tmp_path):
    worker = document()["workers"][0]
    attach_log(tmp_path, worker, text="")
    assert inputs.read_startup(worker, tmp_path, document())["text"] == ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires local POSIX named pipe")
def test_named_pipe_is_rejected_without_waiting_for_a_writer(tmp_path):
    worker = document()["workers"][0]
    path = attach_log(tmp_path, worker)
    path.unlink()
    os.mkfifo(path)
    assert "regular file" in inputs.read_startup(worker, tmp_path, document())["error"]


def test_file_mutation_during_capture_never_returns_frozen_evidence(tmp_path, monkeypatch):
    worker = document()["workers"][0]
    path = attach_log(tmp_path, worker)
    original = os.read
    changed = False

    def mutate(fd, count):
        nonlocal changed
        raw = original(fd, count)
        if raw and not changed:
            changed = True
            path.write_text("replacement bytes\n")
        return raw

    monkeypatch.setattr(inputs.os, "read", mutate)
    result = inputs.read_startup(worker, tmp_path, document())
    assert "error" in result and "text" not in result


def test_bad_binding_is_rejected_before_any_log_file_read(tmp_path, monkeypatch):
    worker = document()["workers"][0]
    attach_log(tmp_path, worker)
    worker["startup_log"]["config_sha256"] = "0" * 64

    def forbidden(*args, **kwargs):
        pytest.fail("bad binding must be rejected before opening a log")

    monkeypatch.setattr(inputs, "_read_regular", forbidden)
    assert "config_sha256" in inputs.read_startup(worker, tmp_path, document())["error"]
