"""Offline declared-workload coverage. Owner: Hari Krishna Kancharla.

These controls remove or change metadata only. They execute no engine.
"""
from copy import deepcopy
import hashlib
import json
import shutil

import pytest

from sdcguard import assessment, assessment_bundle, client_inventory
from sdcguard import evidence_bundle, evidence_combined, evidence_recordings, upgrade
from test_assessment_bundle import supported


def declare(supported, *, missing=False):
    transfer = {"remote_request_id": "R", "remote_engine_id": "producer",
                "remote_lease_slot": 0, "remote_lease_nonce": 7, "remote_block_ids": [[4]]}
    value = {"schema": client_inventory.SCHEMA, "workload": "ordinary-declared-batch",
             "complete": True, "requests": [{"client_request_id": "client-X", "transfer": transfer}]}
    if missing:
        other = deepcopy(value["requests"][0])
        other["client_request_id"] = "entirely-unrecorded-client"
        other["transfer"]["remote_request_id"] = "unrecorded-source-request"
        value["requests"].append(other)
    path = supported["root"] / "client-inventory.json"
    path.write_text(json.dumps(value))
    config = dict(supported["config"], schema=2, client_inventory=path.name)
    supported["path"].write_text(json.dumps(config))
    return config, path


def rehash(packet, name):
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    raw = (packet / name).read_bytes()
    manifest["files"][name].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest_path.write_text(json.dumps(manifest))


def test_complete_declared_workload_preserves_narrow_findings(supported, tmp_path):
    config, inventory = declare(supported)
    out = tmp_path / "assessment"
    result = assessment_bundle.create_bundle(supported["path"], out, run_ready=False)
    assert result["schema"] == 2 and result["scope"] == assessment.WORKLOAD_SCOPE
    assert result["status"] == "fine", result
    coverage = result["request_coverage"]
    assert coverage["status"] == "complete"
    assert coverage["declared_requests"] == coverage["matched_requests"] == 1
    assert coverage["requests"][0]["engine_request"] == "X"
    assert result["sections"]["check_execution"]["status"] == "observed"
    assert result["sections"]["client_outcomes"]["status"] == "unknown"
    assert result["integrity_scope"].startswith("local evidence only")
    assert (out / "evidence/client-inventory.json").read_bytes() == inventory.read_bytes()
    checked = assessment_bundle.verify_bundle(out)
    assert checked["integrity"] == "verified" and checked["scope_verdict"] == "fine", checked


def test_entirely_missing_request_stays_in_denominator(supported, tmp_path):
    declare(supported, missing=True)
    result = assessment_bundle.create_bundle(supported["path"], tmp_path / "assessment", run_ready=False)
    assert result["status"] == "inconclusive"
    assert result["findings"][0]["verdict"] == "satisfied"
    coverage = result["request_coverage"]
    assert coverage["declared_requests"] == 2 and coverage["matched_requests"] == 1
    assert coverage["requests"][1]["client_request_id"] == "entirely-unrecorded-client"
    assert coverage["requests"][1]["status"] == "unknown"
    assert result["sections"]["check_execution"]["status"] == "unknown"
    assert "request_coverage" in result["sections"]["missing_evidence"]
    checked = assessment_bundle.verify_bundle(tmp_path / "assessment")
    assert checked["integrity"] == "verified" and checked["scope_verdict"] == "inconclusive", checked


def test_schema2_requires_inventory_even_with_known_lifecycle(supported):
    result = assessment.assess(dict(supported["config"], schema=2), base=supported["root"], run_ready=False)
    assert result["status"] == "inconclusive"
    assert result["request_coverage"]["status"] == "unknown"
    assert result["findings"][0]["verdict"] == "satisfied"


def test_schema1_cannot_silently_ignore_inventory(supported):
    config, _ = declare(supported)
    config["schema"] = 1
    result = assessment.assess(config, base=supported["root"], run_ready=False)
    assert result["status"] == "inconclusive"
    assert "client inventory requires workload config schema 2" in result["missing"]


def test_coverage_does_not_replace_missing_lifecycle_evidence(supported, tmp_path):
    declare(supported)
    rows = supported["make_events"]()
    rows = [row for row in rows if row.get("args", {}).get("what") != "admit"]
    supported["events"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = assessment_bundle.create_bundle(supported["path"], tmp_path / "assessment", run_ready=False)
    assert result["request_coverage"]["status"] == "complete"
    assert result["status"] == "inconclusive" and result["findings"][0]["verdict"] == "unknown"


@pytest.mark.parametrize("member", ["evidence/client-inventory.json", "report.json"])
def test_rehashed_edits_cannot_forge_complete_coverage(supported, tmp_path, member):
    declare(supported, missing=True)
    out = tmp_path / "assessment"
    assessment_bundle.create_bundle(supported["path"], out, run_ready=False)
    value = json.loads((out / member).read_text())
    if member == "report.json":
        value["request_coverage"]["status"] = "complete"
    else:
        value["requests"].pop()
    (out / member).write_text(json.dumps(value))
    rehash(out, member)
    assert assessment_bundle.verify_bundle(out)["integrity"] == "failed"


def test_inventory_change_during_capture_cannot_be_accepted(supported, tmp_path, monkeypatch):
    _, inventory = declare(supported)
    actual = assessment.assess
    def assess(*args, **kwargs):
        result = actual(*args, **kwargs)
        inventory.write_text(inventory.read_text() + " ")
        return result
    monkeypatch.setattr(assessment, "assess", assess)
    result = assessment_bundle.create_bundle(supported["path"], tmp_path / "assessment", run_ready=False)
    assert result["capture"]["complete"] is False and result["status"] == "inconclusive"


def test_portable_assessment_replays_without_original_inventory(supported, tmp_path):
    declare(supported)
    out = tmp_path / "assessment"
    assessment_bundle.create_bundle(supported["path"], out, run_ready=False)
    shutil.rmtree(supported["root"])
    checked = assessment_bundle.verify_bundle(out)
    assert checked["integrity"] == "verified" and checked["scope_verdict"] == "fine", checked


def test_coverage_deficit_survives_unified_reports_and_upgrade(supported, tmp_path):
    declare(supported, missing=True)
    original = tmp_path / "assessment"
    assessment_bundle.create_bundle(supported["path"], original, run_ready=False)
    normalized = evidence_recordings.normalize(original)
    assert normalized["request_coverage"]["declared_requests"] == 2
    assert normalized["findings"][0]["facts"]["native_verdict"] == "satisfied"
    assert normalized["findings"][0]["verdict"] == "unknown"
    unified = tmp_path / "unified"
    result = evidence_bundle.create_bundle(unified, recorded_packet=original)
    assert result["report"]["status"] == "inconclusive"
    assert b"Declared request coverage" in (unified / "index.html").read_bytes()
    assert evidence_bundle.verify_bundle(unified)["integrity"] == "verified"
    result = evidence_combined.create_bundle(tmp_path / "combined", recorded_packet=original,
                                            require_native=["decode"])
    assert result["report"]["status"] == "inconclusive"
    assert result["report"]["legacy"]["scope_verdict"] == "inconclusive"
    compared = upgrade.compare_packets(unified, unified, tmp_path / "comparison")
    assert compared["scope_verdict"] == "inconclusive"
    assert compared["decision"] == "blocked"
    assert compared["runtime_upgrade_witness"] == "not established"
