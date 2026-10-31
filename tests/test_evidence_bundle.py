"""Portable contract packets, with inert configs and existing CPU components."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from sdcguard import evidence_bundle as bundle
from sdcguard import evidence_components
from sdcguard import evidence_io as io
from sdcguard import evidence_report as reports
from sdcguard.config_inputs import binding_digest
from sdcguard.config_protections import VLLM_PATCH
from test_config_capacity import worker as capacity_worker, startup
from test_assessment_bundle import supported


def static_config(tmp_path, *, engine="vllm", short=False, logs=True):
    root = tmp_path.resolve()
    value = {"schema": "sdcguard.config-safety.v1", "deployment_id": "synthetic-packet-control",
             "advertised_context_tokens": 512,
             "model": {"id": "example/placeholder", "revision": "a" * 40,
                       "architecture": "Qwen3ForCausalLM"}, "workers": []}
    for role in ("prefill", "decode"):
        w = capacity_worker(engine)
        w.update(id=role, role=role)
        w["layout"].update(parameters_complete=True, environment_complete=True,
                            kv_pool_class="MHATokenToKVPool")
        if engine == "vllm":
            w["components"]["runtime_patch_sha256"] = VLLM_PATCH
            w["connector"].update(transfer_mode="pull", bidirectional=False,
                                   extra_config={"bidirectional_kv_xfer": False})
            if short:
                w["parameters"]["num_gpu_blocks_override"] = 32
        else:
            w["parameters"].update(disaggregation_mode=role, kv_canary="log", kv_canary_real_data="all",
                                     max_total_tokens=1024 if not short else 496)
        if logs:
            raw = startup(w).encode()
            if engine == "sglang" and short:
                raw = raw.replace(b"max_total_num_tokens=512", b"max_total_num_tokens=496")
            name = role + ".log"
            (root / name).write_bytes(raw)
            w["startup_log"] = dict(path=name, sha256=io.digest(raw),
                                    config_sha256=binding_digest(w, value), worker_id=role)
        value["workers"].append(w)
    path = root / "deployment.json"
    path.write_bytes(io.json_bytes(value))
    return path


def rehash_packet(root):
    path = root / "manifest.json"
    value = io.parse_json(path.read_bytes())
    value["files"] = io.file_receipts({name: raw for name, raw in io.snapshot(root).items()
                                        if name != "manifest.json"})
    value["directories"] = io.names(root, directories=True)
    path.write_bytes(io.json_bytes(value))


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_static_packet_rejudges_and_moves_without_original_inputs(tmp_path, engine):
    source = tmp_path / "inputs"
    source.mkdir()
    config = static_config(source, engine=engine)
    before = io.snapshot(source)
    packet = tmp_path / "packet"
    result = bundle.create_bundle(packet, config=config)
    assert result["report"]["status"] == "fine"
    assert io.snapshot(source) == before
    moved = tmp_path / "moved"
    shutil.copytree(packet, moved)
    shutil.rmtree(source)
    shutil.rmtree(packet)
    result = bundle.verify_bundle(moved)
    assert result["integrity"] == "verified", result
    assert result["scope_verdict"] == "fine"
    assert result["runtime_isolation"] == "not established"
    assert result["core_facts_recomputed"] is True
    assert str(tmp_path) not in json.dumps(result["report"]["native_sections"]["static"]["provenance"])


def test_existing_components_form_one_two_engine_contract_profile(tmp_path):
    packet = tmp_path / "packet"
    result = bundle.create_bundle(packet, components=True)
    report = result["report"]
    assert report["status"] == "fine"
    assert len(report["findings"]) == 2
    assert {row["rule_id"] for row in report["findings"]} == {"destination_incarnation_retired_after_writers"}
    assert {row["resource_kind"] for row in report["findings"]} == {
        "destination_kv_allocation", "protocol_held_auxiliary_metadata"}
    result = bundle.verify_bundle(packet)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "fine"
    assert result["component_execution"]["status"] == "satisfied"
    assert all(row["facts"]["physical_write_safety"] == "unknown" for row in report["findings"])


def test_known_violation_survives_missing_current_component_refresh(tmp_path, monkeypatch):
    config = static_config(tmp_path, short=True)
    packet = tmp_path / "packet"
    assert bundle.create_bundle(packet, config=config, components=True)["report"]["status"] == "not fine"
    result = bundle.verify_bundle(packet, rerun_components=False)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "not fine"
    monkeypatch.setattr(evidence_components, "collect", lambda *args: (_ for _ in ()).throw(ValueError("CPU unavailable")))
    result = bundle.verify_bundle(packet)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "not fine"
    assert result["component_execution"]["status"] == "unknown"


def test_missing_current_component_refresh_is_not_a_pass(tmp_path):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, components=True)
    result = bundle.verify_bundle(packet, rerun_components=False)
    assert result["integrity"] == "verified"
    assert result["scope_verdict"] == "inconclusive"


def test_internal_operating_system_temp_alias_is_canonicalized(tmp_path, monkeypatch):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, config=static_config(tmp_path))
    real = tmp_path / "real-private-temp"
    real.mkdir()
    alias = tmp_path / "temp-alias"
    alias.symlink_to(real, target_is_directory=True)

    @contextmanager
    def aliased_temporary(**kwargs):
        yield str(alias)

    monkeypatch.setattr(bundle, "TemporaryDirectory", aliased_temporary)
    result = bundle.verify_bundle(packet)
    assert result["integrity"] == "verified", result
    assert result["scope_verdict"] == "fine"


def test_unavailable_startup_preserves_unknown_after_moving(tmp_path):
    config = static_config(tmp_path, logs=False)
    packet = tmp_path / "packet"
    report = bundle.create_bundle(packet, config=config)["report"]
    assert report["status"] == "inconclusive"
    capture = io.parse_json((packet / "inputs/static/capture.json").read_bytes())
    assert all(not row["captured"] and row["reason"] for row in capture["workers"])
    moved = tmp_path / "moved"
    shutil.copytree(packet, moved)
    result = bundle.verify_bundle(moved)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "inconclusive"


@pytest.mark.parametrize("member", ["report.json", "metadata/catalog.json", "metadata/evaluators.json", "index.html"])
def test_editing_and_rehashing_derived_claims_does_not_make_them_trusted(tmp_path, member):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, config=static_config(tmp_path, short=True))
    path = packet / member
    if member == "index.html":
        path.write_bytes(path.read_bytes().replace(b"not fine", b"fine"))
    else:
        data = io.parse_json(path.read_bytes())
        if member == "report.json":
            data["status"] = "fine"
            for row in data["findings"]:
                row["verdict"] = "satisfied"
                row["required"] = False
        else:
            data["untrusted_override"] = True
        path.write_bytes(io.json_bytes(data))
    rehash_packet(packet)
    result = bundle.verify_bundle(packet)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_captured_log_must_match_original_declared_hash_even_after_manifest_rehash(tmp_path):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, config=static_config(tmp_path))
    log = packet / "inputs/static/logs/000.txt"
    log.write_bytes(log.read_bytes().replace(b"override=33", b"override=65"))
    rehash_packet(packet)
    assert bundle.verify_bundle(packet)["integrity"] == "failed"


def test_dropping_an_engine_cannot_shrink_the_selected_component_profile(tmp_path):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, components=True)
    path = packet / "inputs/components.json"
    raw = io.parse_json(path.read_bytes())
    del raw["engines"]["sglang"]
    path.write_bytes(io.json_bytes(raw))
    rehash_packet(packet)
    assert bundle.verify_bundle(packet)["integrity"] == "failed"


def test_recorded_packet_retains_original_verdict_and_global_unknown(supported, tmp_path):
    from sdcguard import assessment_bundle
    legacy = tmp_path / "legacy"
    assessment_bundle.create_bundle(supported["path"], legacy, run_ready=False)
    packet = tmp_path / "unified"
    report = bundle.create_bundle(packet, recorded_packet=legacy, components=True)["report"]
    assert report["status"] == "fine"
    recorded, = [row for row in report["findings"] if row["evidence_kind"] == "recorded_engine_events"]
    assert recorded["verdict"] == "satisfied"
    assert recorded["facts"]["native_verdict"] == "satisfied"
    assert bundle.verify_bundle(packet)["scope_verdict"] == "fine"


def test_empty_recording_cannot_be_hidden_by_positive_cpu_components(supported, tmp_path):
    from sdcguard import assessment_bundle
    supported["events"].write_text("")
    legacy = tmp_path / "legacy"
    assessment_bundle.create_bundle(supported["path"], legacy, run_ready=False)
    packet = tmp_path / "unified"
    report = bundle.create_bundle(packet, recorded_packet=legacy, components=True)["report"]
    assert report["status"] == "inconclusive"
    findings = [row for row in report["findings"] if row["evidence_kind"] == "recorded_engine_events"]
    assert len(findings) == 1 and findings[0]["verdict"] == "unknown"
    assert bundle.verify_bundle(packet)["scope_verdict"] == "inconclusive"


def test_existing_destination_is_preserved(tmp_path):
    output = tmp_path / "packet"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(OSError):
        bundle.create_bundle(output, config=static_config(tmp_path))
    assert (output / "keep.txt").read_text() == "preserve"
    assert io.names(output) == ["keep.txt"]


def test_incomplete_capture_is_marked_and_never_verifies_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "render", lambda value: (_ for _ in ()).throw(ValueError("render failure")))
    packet = tmp_path / "packet"
    with pytest.raises(io.EvidenceError, match="incomplete"):
        bundle.create_bundle(packet, config=static_config(tmp_path))
    manifest = io.parse_json((packet / "manifest.json").read_bytes())
    assert manifest["complete"] is False
    assert bundle.verify_bundle(packet)["integrity"] == "incomplete"


def test_render_escapes_subject_and_points_to_local_artifacts(tmp_path):
    packet = tmp_path / "packet"
    report = bundle.create_bundle(packet, config=static_config(tmp_path))["report"]
    report["findings"][0]["subject"] = "<img src=x onerror=alert(1)>"
    rendered = reports.render(report)
    assert b"<img src=x" not in rendered and b"&lt;img src=x" in rendered
    assert b"href='report.json'" in rendered and b"href='manifest.json'" in rendered


def test_top_level_command_catalog_and_capture_are_dispatched(tmp_path):
    from sdcguard.cli import main
    assert main(["evidence", "catalog"]) == 0
    assert main(["evidence", "collect", "--config", str(static_config(tmp_path)),
                 "--out", str(tmp_path / "packet")]) == 0
    assert main(["evidence", "verify", str(tmp_path / "packet")]) == 0


def test_component_execution_error_is_inconclusive_not_a_reported_violation(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_components, "collect", lambda *args: (_ for _ in ()).throw(AssertionError("mock unavailable")))
    assert bundle.main(["collect", "--components", "--out", str(tmp_path / "packet")]) == 2
    assert not (tmp_path / "packet").exists()


def test_manifest_count_requires_an_integer_not_a_boolean(tmp_path):
    packet = tmp_path / "packet"
    bundle.create_bundle(packet, config=static_config(tmp_path))
    path = packet / "manifest.json"
    value = io.parse_json(path.read_bytes())
    value["files"]["report.json"]["bytes"] = True
    path.write_bytes(io.json_bytes(value))
    result = bundle.verify_bundle(packet)
    assert result["integrity"] == "failed"
    assert any("count types" in error for error in result["errors"])


@pytest.mark.parametrize("value", [1, "false", None])
def test_component_selection_is_explicitly_typed_before_execution(tmp_path, monkeypatch, value):
    monkeypatch.setattr(evidence_components, "collect", lambda *args: pytest.fail("must not execute"))
    with pytest.raises(io.EvidenceError, match="Boolean"):
        bundle.create_bundle(tmp_path / "packet", components=value)
