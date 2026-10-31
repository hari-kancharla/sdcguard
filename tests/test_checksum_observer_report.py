"""Offline reports retain scope gaps. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import json

import pytest

from sdcguard import checksum_observer_report as reports, cli, evidence_io as io
from test_kv_checksum_observer import native, closed_cpu_snapshot, partial_span_cpu_snapshot, empty_replicas_cpu_snapshot


def _verification(value):
    return next(record for record in value["records"] if record["kind"] == "verification")


def _budget(value):
    records = value["records"]
    value["budgets"]["retained_records"] = len(records)
    value["budgets"]["retained_bytes"] = sum(len(json.dumps(r, sort_keys=True, allow_nan=False).encode()) for r in records)
    value["budgets"]["reserved_record_bytes"] = max(value["budgets"]["reserved_record_bytes"], value["budgets"]["retained_bytes"])
    value["budgets"]["retained_units"] = sum(sum(len(r.get(k) or []) for k in
        ("expected_units", "compared_units", "missing_units")) for r in records)


def test_original_cpu_callbacks_have_complete_conditional_scope_without_release_claim(closed_cpu_snapshot):
    report = reports.evaluate(io.json_bytes(closed_cpu_snapshot))
    assert report["qualification"] == "cpu_source_functions"
    assert report["summary"]["verification_calls"] == 1
    assert report["summary"]["native_results"] == {"passed": 1}
    assert report["summary"]["chunk_scope"] == {"complete": 1}
    assert report["release_decision"] == "HOLD"
    assert report["runtime_qualified"] is False
    assert report["client_inventory"] == "unknown"
    assert report["summary"]["unverified_observations"] == 0
    assert "runtime_qualification_pending" in report["hold_reasons"]


@pytest.mark.parametrize("fixture,reason", [
    ("partial_span_cpu_snapshot", "partial_checksum_span"),
    ("empty_replicas_cpu_snapshot", "empty_replica_inputs"),
])
def test_engine_pass_not_covered_reasons_are_named_in_report(request, fixture, reason):
    snapshot = request.getfixturevalue(fixture)
    report = reports.evaluate(io.json_bytes(snapshot))
    row = report["verification_calls"][0]
    assert row["native_result"] == "passed"
    assert row["coverage"] != "complete"
    assert row["engine_pass_not_covered_reason"] == reason
    assert row["conditional_scope"]["compared_units"] is None
    assert row["conditional_scope"]["missing_units"] is None
    assert "Engine pass, not covered: " + reason in reports.render_text(report)
    assert report["release_decision"] == "HOLD"


@pytest.mark.parametrize("section,key,value", [
    (None, "closed", False), (None, "runtime_qualified", True),
    (None, "qualification", "runtime_approved"),
    ("health", "errors", 1), ("health", "active_calls", True),
    ("health", "budget_exhausted", True),
    ("health", "inflight_position_units", 1), ("health", "inflight_position_bytes", 1),
    ("source", "supported", False), ("source", "revision", "unknown"),
    ("budgets", "retained_records", 0), ("budgets", "retained_units", 0),
    ("budgets", "reserved_state_bytes", 1),
])
def test_native_pass_survives_but_gap_prevents_complete_scope(closed_cpu_snapshot, section, key, value):
    snapshot = deepcopy(closed_cpu_snapshot)
    target = snapshot if section is None else snapshot[section]
    target[key] = value
    report = reports.evaluate(io.json_bytes(snapshot))
    assert report["summary"]["native_results"] == {"passed": 1}
    assert report["summary"]["chunk_scope"] == {"unknown": 1}
    assert report["release_decision"] == "HOLD"


@pytest.mark.parametrize("change", ["missing_plan", "plan_request", "reordered", "unknown_kind"])
def test_missing_or_unsupported_event_cannot_supply_causal_scope(closed_cpu_snapshot, change):
    value = deepcopy(closed_cpu_snapshot)
    plan = next(r for r in value["records"] if r["kind"] == "recv_plan")
    if change == "missing_plan":
        value["records"].remove(plan)
    elif change == "plan_request":
        plan["request_id"] = "synthetic-unjoined-plan"
    elif change == "reordered":
        value["records"].reverse()
    else:
        value["records"].append({"kind": "unsupported_observation", "sequence": value["records"][-1]["sequence"] + 1})
    _budget(value)
    report = reports.evaluate(io.json_bytes(value))
    assert report["summary"]["verification_calls"] == 1
    assert report["summary"]["chunk_scope"] == {"unknown": 1}


def test_scheduled_call_without_verification_remains_visible(closed_cpu_snapshot):
    value = deepcopy(closed_cpu_snapshot)
    value["records"] = [record for record in value["records"] if record["kind"] != "verification"]
    _budget(value)
    report = reports.evaluate(io.json_bytes(value))
    assert report["summary"]["verification_calls"] == 0
    assert report["summary"]["unverified_observations"] == 1
    assert report["unverified_observations"][0]["coverage"] == "unknown"
    assert report["unverified_observations"][0]["reason"] == "receive_plan_without_verification"
    assert report["unverified_observations"][0]["native_plan_reason"] == "native_checksum_receive_scheduled"
    assert report["release_decision"] == "HOLD"


@pytest.mark.parametrize("change", ["token_count", "block_count"])
def test_verification_cannot_replace_the_recorded_load_plan(closed_cpu_snapshot, change):
    value = deepcopy(closed_cpu_snapshot)
    plan = next(record for record in value["records"] if record["kind"] == "recv_plan")
    if change == "token_count":
        plan["load_plan"]["num_tokens"] += 16
    else:
        plan["load_plan"]["groups"][0]["block_count"] += 1
    _budget(value)
    report = reports.evaluate(io.json_bytes(value))
    row = report["verification_calls"][0]
    assert row["native_result"] == "passed"
    assert row["coverage"] == "unknown"
    assert "planned_chunk_scope_disagrees_with_verification" in row["recording_issues"]


@pytest.mark.parametrize("key,value", [("max_units", True), ("max_records", 0), ("max_bytes", 100)])
def test_malformed_budgets_are_refused(closed_cpu_snapshot, key, value):
    snapshot = deepcopy(closed_cpu_snapshot)
    snapshot["budgets"][key] = value
    with pytest.raises(ValueError, match="budgets"):
        reports.evaluate(io.json_bytes(snapshot))


def test_cli_capture_and_portable_verify_recompute_original_cpu_observations(closed_cpu_snapshot, tmp_path, capsys):
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_bytes(io.json_bytes(closed_cpu_snapshot))
    packet = tmp_path / "observations"
    assert cli.main(["checksum-observer", "capture", "--snapshot", str(snapshot),
                     "--out", str(packet), "--json"]) == 2
    captured = json.loads(capsys.readouterr().out)
    assert captured["summary"]["chunk_scope"] == {"complete": 1}
    snapshot.unlink()
    assert cli.main(["checksum-observer", "verify", str(packet), "--json"]) == 2
    verified = json.loads(capsys.readouterr().out)
    assert verified == captured
    assert (packet / "report.html").is_file()
    assert "Independent client inventory: unknown" in (packet / "report.txt").read_text()


def test_rehashed_report_cannot_promote_candidate_to_release(closed_cpu_snapshot, tmp_path):
    source = tmp_path / "snapshot.json"
    source.write_bytes(io.json_bytes(closed_cpu_snapshot))
    packet = tmp_path / "packet"
    reports.capture(source, packet)
    path = packet / "report.json"
    report = json.loads(path.read_bytes())
    report["release_decision"] = "PASS"
    path.write_bytes(io.json_bytes(report))
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"]["report.json"] = {"sha256": io.digest(path.read_bytes()), "bytes": path.stat().st_size}
    manifest_path.write_bytes(io.json_bytes(manifest))
    with pytest.raises(ValueError, match="recomputed observations"):
        reports.verify(packet)
