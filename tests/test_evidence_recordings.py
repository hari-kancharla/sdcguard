"""Saved packet normalization only; no engine or serving request is used."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

import pytest

from sdcguard import assessment_bundle as bundle
from sdcguard import evidence_recordings as bridge
from sdcguard.lifetime.obligations import Finding, VERDICTS
from test_assessment_bundle import supported


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


def packet(supported, tmp_path):
    root = tmp_path / "packet"
    report = bundle.create_bundle(supported["path"], root, run_ready=False)
    assert report["capture"]["complete"] is True
    return root, report


def rehash(root, name):
    path = root / name
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  bytes=path.stat().st_size)
    manifest_path.write_text(json.dumps(manifest) + "\n")


def test_normalize_verified_recording_keeps_exact_scope_and_does_not_change_input(supported, tmp_path):
    root, native = packet(supported, tmp_path)
    before = files(root)
    result = bridge.normalize(root)
    assert files(root) == before
    assert result["legacy_verification"]["integrity"] == "verified"
    assert result["legacy_scope_verdict"] == "fine"
    finding, = result["findings"]
    assert finding["verdict"] == finding["facts"]["native_verdict"] == "satisfied"
    assert finding["facts"]["native_finding"] == json.loads((root / "report.json").read_text())["findings"][0]
    assert finding["subject"] == "request:" + native["findings"][0]["request"]
    assert finding["scenario"] == "recorded-source-lease"
    assert finding["scope"] == "source_lifetime_and_scheduler_admission"
    assert finding["resource_kind"] == "source_kv_allocation"
    assert finding["evidence_kind"] == "recorded_engine_events"
    assert finding["facts"]["runtime_model_use"] == "not observed"
    assert finding["facts"]["answer_correctness"] == "not judged"
    assert finding["facts"]["authenticity"] == "not established"
    assert result["identity"]["model"]["live_weights_verified"] is False
    assert result["identity"]["model"]["revision"] == "a" * 40
    assert result["identity"]["kernel"] == "not observed"


def test_known_rejection_is_not_renamed_recovery(supported, tmp_path):
    supported["events"].write_text("".join(json.dumps(row) + "\n"
                                         for row in supported["make_events"](rejected=True)))
    root, _ = packet(supported, tmp_path)
    finding, = bridge.normalize(root)["findings"]
    assert finding["verdict"] == "correctly_rejected"
    assert finding["facts"]["native_verdict"] == "correctly_rejected"


def test_global_model_gap_cannot_be_hidden_by_locally_satisfied_history(supported, tmp_path):
    config = dict(supported["config"], model_revision="main")
    supported["path"].write_text(json.dumps(config))
    root, native = packet(supported, tmp_path)
    assert native["status"] == "inconclusive"
    assert native["findings"][0]["verdict"] == "satisfied"
    finding, = bridge.normalize(root)["findings"]
    assert finding["verdict"] == "unknown"
    assert finding["facts"]["native_verdict"] == "satisfied"
    assert "resolved Qwen3-0.6B model commit; main is not a pin" in finding["missing"]


def test_empty_recording_emits_required_unknown_instead_of_no_rules(supported, tmp_path):
    supported["events"].write_text("")
    root, native = packet(supported, tmp_path)
    assert not native["findings"]
    finding, = bridge.normalize(root)["findings"]
    assert finding["verdict"] == "unknown"
    assert finding["subject"] == "recording:unattributed"
    assert finding["facts"]["native_finding"] == {}
    assert "recorded request attempts" in finding["missing"]


@pytest.mark.parametrize("verdict", VERDICTS)
def test_projection_preserves_all_original_outcome_names(verdict):
    value = Finding("lease_valid_at_use", "opaque/request:17", verdict=verdict).as_dict()
    value["not_observed"] = list(value["not_observed"])
    report = {"status": "not fine" if verdict == "violated" else "fine", "missing": [],
              "findings": [value], "engine": {}, "model_binding": "declared revision only"}
    finding, = bridge._canonical_findings(report, {})
    assert finding["verdict"] == verdict
    assert finding["facts"]["native_verdict"] == verdict
    assert finding["subject"] == "request:opaque/request:17"
    assert finding["evidence"] == value["evidence"]


def test_duplicate_native_request_findings_are_rejected():
    value = Finding("lease_valid_at_use", "same").as_dict()
    value["not_observed"] = list(value["not_observed"])
    with pytest.raises(bridge.RecordedEvidenceError, match="duplicate native request"):
        bridge._canonical_findings({"status": "fine", "missing": [],
                                    "findings": [value, deepcopy(value)]}, {})


@pytest.mark.parametrize("member", ["manifest.json", "report.json", "config.json",
                                    "metadata/engine-pins.json", "evidence/config-original.json"])
def test_duplicate_keys_are_rejected_even_if_legacy_parser_would_choose_last(supported, tmp_path, member):
    root, _ = packet(supported, tmp_path)
    path = root / member
    value = json.loads(path.read_text())
    key = next(iter(value))
    duplicate = json.dumps(key) + ":" + json.dumps(value[key]) + ","
    path.write_text("{" + duplicate + path.read_text().lstrip()[1:])
    if member != "manifest.json":
        rehash(root, member)
    with pytest.raises(bridge.RecordedEvidenceError, match="invalid captured JSON"):
        bridge.normalize(root)


def test_duplicate_event_keys_are_not_silently_accepted(supported, tmp_path):
    root, _ = packet(supported, tmp_path)
    name = "evidence/events/000000.jsonl"
    path = root / name
    lines = path.read_text().splitlines(keepends=True)
    first = json.loads(lines[0])
    lines[0] = '{"kind":' + json.dumps(first["kind"]) + "," + lines[0].lstrip()[1:]
    path.write_text("".join(lines))
    rehash(root, name)
    with pytest.raises(bridge.RecordedEvidenceError, match="invalid captured JSON"):
        bridge.normalize(root)


def test_unfinished_saved_event_line_is_not_ignored(supported, tmp_path):
    root, _ = packet(supported, tmp_path)
    name = "evidence/events/000000.jsonl"
    path = root / name
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    rehash(root, name)
    with pytest.raises(bridge.RecordedEvidenceError, match="unfinished line"):
        bridge.normalize(root)


@pytest.mark.parametrize("kind", ["alter", "extra", "delete", "symlink", "fifo"])
def test_bad_packet_members_cannot_become_normalized_findings(supported, tmp_path, kind):
    root, _ = packet(supported, tmp_path)
    path = root / "report.json"
    if kind == "alter":
        path.write_bytes(path.read_bytes() + b"\n")
    elif kind == "extra":
        (root / "extra.txt").write_text("extra")
    elif kind == "delete":
        path.unlink()
    elif kind == "symlink":
        path.unlink()
        path.symlink_to(supported["path"])
    else:
        import os
        os.mkfifo(root / "not-a-file")
    with pytest.raises(bridge.RecordedEvidenceError):
        bridge.normalize(root)


def test_linked_ancestor_is_not_followed(supported, tmp_path):
    root, _ = packet(supported, tmp_path)
    link = tmp_path / "linked"
    link.symlink_to(root, target_is_directory=True)
    with pytest.raises(bridge.RecordedEvidenceError):
        bridge.normalize(link)


def test_original_packet_change_during_verification_is_rejected(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)
    ordinary = bundle.verify_bundle
    calls = []

    def changing(directory):
        assert Path(directory) != root
        result = ordinary(directory)
        calls.append(directory)
        if len(calls) == 1:
            (root / "new.txt").write_text("changed")
        return result

    monkeypatch.setattr(bundle, "verify_bundle", changing)
    with pytest.raises(bridge.RecordedEvidenceError, match="saved packet changed"):
        bridge.normalize(root)
    assert len(calls) == 2


def test_private_copy_change_is_rejected(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)
    ordinary = bundle.verify_bundle
    calls = []

    def changing(directory):
        result = ordinary(directory)
        calls.append(directory)
        if len(calls) == 2:
            (Path(directory) / "new.txt").write_text("changed")
        return result

    monkeypatch.setattr(bundle, "verify_bundle", changing)
    with pytest.raises(bridge.RecordedEvidenceError, match="private assessment copy changed"):
        bridge.normalize(root)


def test_existing_offline_verifier_cannot_be_skipped(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)
    monkeypatch.setattr(bundle, "verify_bundle", lambda _: {"integrity": "failed", "errors": ["probe refusal"]})
    with pytest.raises(bridge.RecordedEvidenceError, match="probe refusal"):
        bridge.normalize(root)


def test_offline_bridge_never_starts_process_or_network(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)

    def refused(*args, **kwargs):
        raise AssertionError("network or process operation attempted")

    monkeypatch.setattr(socket, "socket", refused)
    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    result = bridge.normalize(root)
    assert result["legacy_scope_verdict"] == "fine"


def test_moved_packet_survives_without_original_input_tree(supported, tmp_path):
    root, _ = packet(supported, tmp_path)
    before = bridge.normalize(root)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    shutil.rmtree(root)
    shutil.rmtree(supported["root"])
    after = bridge.normalize(moved)
    assert before == after


def test_member_byte_bound_is_enforced_before_legacy_verifier(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)
    monkeypatch.setattr(bridge, "MAX_FILE_BYTES", 1)
    with pytest.raises(bridge.RecordedEvidenceError, match="byte limit"):
        bridge.normalize(root)


def test_file_count_bound_is_enforced(supported, tmp_path, monkeypatch):
    root, _ = packet(supported, tmp_path)
    monkeypatch.setattr(bridge, "MAX_FILES", 1)
    with pytest.raises(bridge.RecordedEvidenceError, match="file count"):
        bridge.normalize(root)
