"""Portable assessment evidence; synthetic existing-case observations only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from sdcguard import assessment
from sdcguard import assessment_bundle as bundle
from test_lifetime_obligations import recording


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def supported(tmp_path, monkeypatch):
    """Hashes identify fixture source bytes, never a real engine or GPU run."""
    root = tmp_path / "inputs"
    root.mkdir()
    engine = root / "engine"
    pin = deepcopy(assessment.pins())
    for variant in pin["variants"]:
        for name, value in pin["variants"][variant].items():
            if value is None:
                continue
            raw = (variant + ":" + name + "\n").encode()
            pin["variants"][variant][name] = hashlib.sha256(raw).hexdigest()
            if variant == "fixed":
                path = engine / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    monkeypatch.setattr(assessment, "pins", lambda: deepcopy(pin))
    identity = assessment.inspect_engine(engine, "0.32.0")

    def events(*, rejected=False, stale=False, no_check=False):
        r = recording(rejected=rejected, stale=stale,
                      check=None if no_check else "lease_mismatch" if rejected else "matched")
        r.events[0]["args"].update(engine_identity=deepcopy(identity), nixl_version="1.4.1")
        for pid in ("D", "DS"):
            r.event("note", 0.1, pid, engine_identity=deepcopy(identity), nixl_version="1.4.1")
        r.events.sort(key=lambda row: row["mono"])
        counts = {}
        for row in r.events:
            counts[row["pid"]] = counts.get(row["pid"], 0) + 1
            row["seq"] = counts[row["pid"]]
        return r.health()

    event_dir = root / "events"
    event_dir.mkdir()
    event_file = event_dir / "worker.jsonl"
    event_file.write_text("".join(json.dumps(row) + "\n" for row in events()))
    config = {"schema": 1, "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
              "engine_root": "engine", "model": "Qwen/Qwen3-0.6B", "model_revision": "a" * 40,
              "topology": assessment.TOPOLOGY, "events": "events"}
    config_path = root / "deployment.json"
    _dump(config_path, config)
    return {"root": root, "config": config, "path": config_path, "events": event_file,
            "engine": engine, "identity": identity, "make_events": events}


def test_fine_packet_keeps_source_presence_execution_and_missing_outcomes_separate(supported, tmp_path):
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "fine"
    sections = report["sections"]
    assert sections["applicability"]["status"] == "supported"
    assert sections["applicability"]["live_model_weights_verified"] is False
    assert sections["fix_presence"]["status"] == "present"
    assert sections["check_execution"]["status"] == "observed"
    assert sections["client_outcomes"]["status"] == "unknown"
    assert sections["regression"]["status"] == "not supplied"
    assert sections["overhead"]["status"] == "not measured"
    assert report["capture"]["complete"] is True
    assert report["configuration"]["model_revision"] == "a" * 40
    assert (out / "evidence/config-original.json").read_bytes() == supported["path"].read_bytes()
    assert json.loads((out / "config.json").read_text())["events"] == "evidence/events"
    manifest = json.loads((out / "manifest.json").read_text())
    assert sum(e["role"] == "engine_source" for e in manifest["files"].values()) == 9
    verified = bundle.verify_bundle(out)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "fine"
    assert verified["authenticity"] == "not established" and verified["remote_attestation"] is False


def test_a_known_rejection_does_not_invent_client_recovery(supported, tmp_path):
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in supported["make_events"](rejected=True)))
    report = bundle.create_bundle(supported["path"], tmp_path / "packet")
    assert report["status"] == "fine"
    assert report["findings"][0]["verdict"] == "correctly_rejected"
    assert report["sections"]["client_outcomes"]["requests"][0]["outcome"] == "unknown"


def test_fix_source_present_is_distinct_from_unchecked_violated_recording(supported, tmp_path):
    rows = supported["make_events"](stale=True, no_check=True)
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "not fine"
    assert report["sections"]["fix_presence"]["status"] == "present"
    assert report["sections"]["check_execution"]["status"] == "unknown"
    verified = bundle.verify_bundle(out)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "not fine"


def test_unknown_evidence_packet_integrity_does_not_turn_into_fine(supported, tmp_path):
    rows = supported["make_events"]()
    rows = [row for row in rows if row.get("args", {}).get("what") != "admit"]
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "inconclusive"
    assert report["findings"][0]["verdict"] == "unknown"
    verified = bundle.verify_bundle(out)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "inconclusive"


def test_mutable_model_revision_stays_unknown_in_a_complete_local_packet(supported, tmp_path):
    _dump(supported["path"], dict(supported["config"], model_revision="main"))
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "inconclusive" and report["capture"]["complete"] is True
    assert "not recorded" in report["model_binding"]
    assert bundle.verify_bundle(out)["integrity"] == "verified"


def test_unbound_recording_does_not_report_verified_check_execution(supported, tmp_path):
    rows = supported["make_events"]()
    next(row for row in rows if row["pid"] == "D" and row["kind"] == "note")["args"].pop("engine_identity")
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "inconclusive"
    assert report["sections"]["check_execution"]["status"] == "unknown"
    assert bundle.verify_bundle(out)["integrity"] == "verified"


def test_moved_packet_verifies_after_original_inputs_are_gone(supported, tmp_path):
    source = tmp_path / "packet"
    bundle.create_bundle(supported["path"], source)
    moved = tmp_path / "another-machine-copy"
    shutil.copytree(source, moved)
    shutil.rmtree(source)
    shutil.rmtree(supported["root"])
    result = bundle.verify_bundle(moved)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "fine"


@pytest.mark.parametrize("change", ["alter", "delete", "extra", "symlink"])
def test_tampered_or_missing_evidence_is_rejected(supported, tmp_path, change):
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    path = out / "evidence/events/000000.jsonl"
    if change == "alter":
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "delete":
        path.unlink()
    elif change == "extra":
        (out / "unlisted.txt").write_text("unlisted")
    else:
        path.unlink()
        path.symlink_to(supported["events"])
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def _rehash(out, name):
    path = out / name
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    _dump(manifest_path, manifest)


def test_rehashed_false_client_claim_is_not_a_valid_derived_report(supported, tmp_path):
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["sections"]["client_outcomes"]["status"] = "recovered"
    _dump(path, report)
    _rehash(out, "report.json")
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def test_rehashed_authenticity_claim_is_not_local_integrity(supported, tmp_path):
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["capture"].update(remote_attestation=True, authenticity="signed")
    _dump(path, report)
    _rehash(out, "report.json")
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def test_rehashed_fine_verdict_cannot_replace_unknown_recording(supported, tmp_path):
    supported["events"].write_text("")
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    path = out / "report.json"
    report = json.loads(path.read_text())
    report["status"] = "fine"
    report = assessment.report_sections(report, supported["config"])
    _dump(path, report)
    _rehash(out, "report.json")
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def test_manifest_traversal_is_rejected_without_reading_outside_packet(supported, tmp_path):
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../outside.txt"] = {"sha256": "a" * 64, "bytes": 0, "role": "recording"}
    _dump(path, manifest)
    result = bundle.verify_bundle(out)
    assert result["integrity"] == "failed" and "unsafe relative" in result["errors"][0]


@pytest.mark.parametrize("kind", ["events", "engine", "membership", "config"])
def test_changing_sources_after_capture_can_never_package_fine(supported, tmp_path, monkeypatch, kind):
    ordinary = assessment.assess

    def changed(*args, **kwargs):
        report = ordinary(*args, **kwargs)
        assert report["status"] == "fine"
        if kind == "events":
            supported["events"].write_text(supported["events"].read_text() + "\n")
        elif kind == "engine":
            path = next(supported["engine"].rglob("*.py"))
            path.write_bytes(path.read_bytes() + b"# changed\n")
        elif kind == "membership":
            (supported["events"].parent / "new-worker.jsonl").write_text("")
        else:
            supported["path"].write_text(supported["path"].read_text() + "\n")
        return report

    monkeypatch.setattr(assessment, "assess", changed)
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "inconclusive" and report["capture"]["complete"] is False
    assert any("changed" in value for value in report["capture"]["errors"])
    assert bundle.verify_bundle(out)["integrity"] == "incomplete"


def test_packet_mutation_during_verification_is_rejected(supported, tmp_path, monkeypatch):
    out = tmp_path / "packet"
    bundle.create_bundle(supported["path"], out)
    ordinary = assessment.assess

    def changed(*args, **kwargs):
        report = ordinary(*args, **kwargs)
        path = out / "evidence/events/000000.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        return report

    monkeypatch.setattr(assessment, "assess", changed)
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def test_retargeting_configured_engine_root_during_capture_is_inconclusive(supported, tmp_path, monkeypatch):
    alias = supported["root"] / "engine-alias"
    alias.symlink_to(supported["engine"], target_is_directory=True)
    _dump(supported["path"], dict(supported["config"], engine_root="engine-alias"))
    other = tmp_path / "other-engine"
    shutil.copytree(supported["engine"], other)
    ordinary = assessment.assess

    def changed(*args, **kwargs):
        report = ordinary(*args, **kwargs)
        assert report["status"] == "fine"
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        return report

    monkeypatch.setattr(assessment, "assess", changed)
    report = bundle.create_bundle(supported["path"], tmp_path / "packet")
    assert report["status"] == "inconclusive"
    assert "a configured source path changed during assessment capture" in report["missing"]


def test_offline_verification_never_refreshes_readiness(supported, tmp_path, monkeypatch):
    config = dict(supported["config"], ready={"metrics_urls": ["http://127.0.0.1:8000/metrics"]})
    _dump(supported["path"], config)
    calls = []

    def ready(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout='{"status":"ready"}', stderr="")

    monkeypatch.setattr(assessment.subprocess, "run", ready)
    out = tmp_path / "packet"
    assert bundle.create_bundle(supported["path"], out)["status"] == "fine"
    assert len(calls) == 1
    calls.clear()
    assert bundle.verify_bundle(out)["integrity"] == "verified"
    assert calls == []


def test_existing_destination_is_never_overwritten(supported, tmp_path):
    out = tmp_path / "owned"
    out.mkdir()
    (out / "owner.txt").write_text("keep exactly")
    with pytest.raises(FileExistsError):
        bundle.create_bundle(supported["path"], out)
    assert list(out.iterdir()) == [out / "owner.txt"]
    assert (out / "owner.txt").read_text() == "keep exactly"


def test_source_symlink_does_not_copy_an_external_file(supported, tmp_path):
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("PRIVATE_EXTERNAL_SENTINEL")
    path = next(supported["engine"].rglob("*.py"))
    path.unlink()
    path.symlink_to(outside)
    out = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], out)
    assert report["status"] == "inconclusive"
    assert all(b"PRIVATE_EXTERNAL_SENTINEL" not in f.read_bytes() for f in out.rglob("*") if f.is_file())


def test_injected_packet_directory_link_cannot_create_files_outside(supported, tmp_path, monkeypatch):
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    ordinary = bundle._write_packet

    def injected(root, name, *args):
        if name == "evidence/config-original.json":
            (root / "evidence").symlink_to(outside, target_is_directory=True)
        return ordinary(root, name, *args)

    monkeypatch.setattr(bundle, "_write_packet", injected)
    report = bundle.create_bundle(supported["path"], tmp_path / "packet")
    assert report["status"] == "inconclusive" and list(outside.iterdir()) == []


@pytest.mark.parametrize("extra", [{"api_key": "PRIVATE_CONFIG_SENTINEL"},
                                   {"ready": {"metrics_urls": ["https://example.com/metrics?token=PRIVATE_CONFIG_SENTINEL"]}}])
def test_unrelated_secret_fields_are_not_copied(supported, tmp_path, extra):
    _dump(supported["path"], dict(supported["config"], **extra))
    out = tmp_path / "packet"
    assert bundle.create_bundle(supported["path"], out)["status"] == "inconclusive"
    assert not (out / "evidence/config-original.json").exists()
    assert all(b"PRIVATE_CONFIG_SENTINEL" not in f.read_bytes() for f in out.rglob("*") if f.is_file())


def test_write_failure_preserves_captured_evidence_without_completion_marker(supported, tmp_path, monkeypatch):
    ordinary = bundle._write_packet

    def failed(root, name, *args):
        assert not (root / "manifest.json").exists()
        if name == "report.html":
            raise OSError("simulated output failure")
        return ordinary(root, name, *args)

    monkeypatch.setattr(bundle, "_write_packet", failed)
    out = tmp_path / "packet"
    with pytest.raises(OSError):
        bundle.create_bundle(supported["path"], out)
    assert (out / "report.json").is_file()
    assert (out / "evidence/events/000000.jsonl").is_file()
    assert not (out / "manifest.json").exists()
    assert bundle.verify_bundle(out)["integrity"] == "failed"


def test_html_is_static_and_escapes_recorded_content(supported, tmp_path):
    report = bundle.create_bundle(supported["path"], tmp_path / "packet")
    report["missing"] = ["<script>alert('x')</script>"]
    text = bundle.render_html(report).decode()
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert "href='report.json'" in text and "src=" not in text


def test_cli_out_and_module_verification_use_only_saved_inputs(tmp_path):
    """Unknown real-source hashes are valid inconclusive evidence, not fixture support."""
    engine = tmp_path / "engine"
    for name in assessment.pins()["variants"]["fixed"]:
        path = engine / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# unknown local source\n")
    events = tmp_path / "events.jsonl"
    events.write_text("")
    config = {"schema": 1, "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
              "engine_root": "engine", "model": "Qwen/Qwen3-0.6B", "model_revision": "a" * 40,
              "topology": assessment.TOPOLOGY, "events": "events.jsonl"}
    path = tmp_path / "deployment.json"
    _dump(path, config)
    out = tmp_path / "packet"
    assert assessment.main([str(path), "--json", "--out", str(out)]) == 2
    done = subprocess.run([sys.executable, "-B", "-m", "sdcguard.assessment_bundle", "verify", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout)
    assert result["integrity"] == "verified" and result["scope_verdict"] == "inconclusive"
