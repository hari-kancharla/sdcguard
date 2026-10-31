"""Recorded native results are not interchangeable with coverage or completion."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from sdcguard import cli, evidence_io as io
from sdcguard.lifetime.obligations import Facts
from sdcguard import native_checks as checks

CASE = Path(__file__).parent / "data/native_check_owner_r3"


@pytest.fixture(scope="module")
def original():
    source = json.loads((CASE / "SOURCE.json").read_bytes())
    for name, receipt in source["files"].items():
        raw = (CASE / name).read_bytes()
        assert {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)} == receipt
    return (CASE / "RESULT.json").read_bytes(), (CASE / "ordinary-evidence.zip").read_bytes()


def normalized(**kwargs):
    values = dict(native_result="passed", expected_units=["group:0", "group:1"],
                  compared_units=["group:0", "group:1"], completed=True, evidence_complete=True)
    values.update(kwargs)
    return checks.assess_check("checksum", **values)


@pytest.mark.parametrize("changes,coverage", [
    ({"compared_units": ["group:0"]}, "partial"),
    ({"compared_units": []}, "not_covered"),
    ({"expected_units": [], "compared_units": []}, "not_covered"),
    ({"expected_units": None}, "unknown"),
    ({"compared_units": None}, "unknown"),
    ({"evidence_complete": False}, "unknown"),
    ({"applicable": False}, "not_covered"),
])
def test_native_pass_does_not_erase_coverage_gap(changes, coverage):
    result = normalized(**changes)
    assert result["native_result"] == "passed"
    assert result["check_status"] == "unverified"
    assert result["coverage"] == coverage


@pytest.mark.parametrize("result", ["admitted", "activity", "scheduled", "unverified"])
def test_native_noncomparison_observations_never_become_pass(result):
    assert normalized(native_result=result)["check_status"] == "unverified"


def test_complete_comparison_and_native_mismatch_are_distinct():
    assert normalized()["check_status"] == "passed"
    result = normalized(native_result="mismatch", compared_units=["group:0"])
    assert result["check_status"] == "mismatch"
    assert result["coverage"] == "partial"
    assert result["missing_units"] == ["group:1"]


@pytest.mark.parametrize("field,value", [
    ("completed", 1), ("completed", 0), ("evidence_complete", 1), ("applicable", 0),
    ("expected_units", ["group:0", "group:0"]), ("compared_units", ["group:0", "group:0"]),
    ("expected_units", [""]), ("compared_units", ["outside"]),
    ("reason", ""), ("reason", False),
])
def test_contradictory_or_untyped_units_are_refused(field, value):
    with pytest.raises(ValueError):
        normalized(**{field: value})


def test_saved_real_19_explicit_lease_checks_are_complete_but_run_is_not(original):
    result, archive = original
    report = checks.evaluate_owner(result, archive)
    assert report["summary"]["declared_requests"] == 19
    assert report["summary"]["matched_requests"] == 19
    assert report["client_inventory"]["status"] == "complete"
    assert report["summary"]["checks"]["lease"] == {"passed": 19}
    assert report["summary"]["checks"]["checksum"] == {"unverified": 19}
    assert report["summary"]["both_lease_and_checksum_passed"] == 0
    assert report["release_decision"] == "HOLD"
    assert report["run_complete"] is False
    assert "client_inventory_incomplete" not in report["hold_reasons"]
    assert all(r["checks"]["lease"]["native_result"] == "passed" for r in report["requests"])
    assert all(r["checks"]["lease"]["expected_units"] == ["remote-rank:prefill:0"] for r in report["requests"])
    assert all(r["router_forwarded_lease_fields"] == "unknown" for r in report["requests"])
    assert all(r["recorded_model_cache_groups"] == 1 for r in report["requests"])
    assert "physical reads" in " ".join(report["limits"])
    lease_only = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert lease_only["hold_reasons"] == ["owner_run_completion_unverified"]


def test_real_19_cli_capture_then_verify_retains_original_inputs(original, tmp_path, capsys):
    owner, packet = tmp_path / "owner", tmp_path / "packet"
    owner.mkdir()
    (owner / "RESULT.json").write_bytes(original[0])
    (owner / "ordinary-evidence.zip").write_bytes(original[1])
    assert cli.main(["native-checks", "capture", "--owner-result", str(owner), "--out", str(packet),
                     "--require", "lease"]) == 2
    assert checks.TITLE in capsys.readouterr().out
    assert cli.main(["native-checks", "verify", str(packet), "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["checks"]["lease"] == {"passed": 19}
    assert report["summary"]["matched_requests"] == 19
    assert report["hold_reasons"] == ["owner_run_completion_unverified"]
    assert (packet / "inputs/RESULT.json").read_bytes() == original[0]
    assert (packet / "inputs/ordinary-evidence.zip").read_bytes() == original[1]
    assert (packet / "report.html").is_file()


def rewritten(original, edit):
    """Detached CPU controls; the saved run on disk is never edited."""
    result = json.loads(original[0])
    with zipfile.ZipFile(BytesIO(original[1])) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    edit(result, files)
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    raw = stream.getvalue()
    result["archive_sha256"] = io.digest(raw)
    return io.json_bytes(result), raw


def edit_batch(files, change):
    capture = json.loads(files[checks.CAPTURE + "CAPTURE.json"])
    change(capture["client_batch"])
    files[checks.CAPTURE + "CLIENT_BATCH.json"] = io.json_bytes(capture["client_batch"])
    files[checks.CAPTURE + "CAPTURE.json"] = io.json_bytes(capture)
    summary = json.loads(files["ordinary/SUMMARY.json"])
    summary["ordinary_capture"] = capture
    files["ordinary/SUMMARY.json"] = io.json_bytes(summary)


@pytest.mark.parametrize("change", [
    lambda b: b["references"].pop(),
    lambda b: b["references"][0].update(complete=False),
    lambda b: b["rounds"][0]["requests"][1].update(decode_id=b["references"][0]["decode_id"]),
])
def test_missing_incomplete_or_duplicate_requests_never_shrink_denominator(original, change):
    result, archive = rewritten(original, lambda _, files: edit_batch(files, change))
    report = checks.evaluate_owner(result, archive)
    assert report["summary"]["declared_requests"] == 19
    assert len(report["requests"]) == 19
    assert report["release_decision"] == "HOLD"
    assert report["client_inventory"]["status"] == "unknown"


def test_typed_full_handoff_equality_cannot_alias_true_with_one(original):
    def change(batch):
        row = batch["references"][0]
        row["stages"][0]["response"]["kv_transfer_params"]["extra_field"] = 1
        row["stages"][1]["body"]["kv_transfer_params"]["extra_field"] = True
    result, archive = rewritten(original, lambda _, files: edit_batch(files, change))
    report = checks.evaluate_owner(result, archive)
    assert report["summary"]["declared_requests"] == 19
    assert report["summary"]["matched_requests"] == 0
    assert "typed producer descriptor differs" in " ".join(report["hold_reasons"])


def test_absent_native_check_cannot_borrow_producer_holds_logical_satisfaction():
    from test_lifetime_obligations import recording, lease
    events = recording().health()
    assert lease(events).verdict == "satisfied"
    report = checks._lease(Facts(events), "X", True, [], True)
    assert report["native_result"] == "unverified"
    assert report["check_status"] == "unverified"
    assert "no_recorded_lease_check" in report["missing_evidence"]


def test_explicit_native_pass_does_not_require_new_producer_ownership_proof(original):
    files = checks._archive(original[1])
    events, issues = checks._raw_events(files)
    facts = Facts(events)
    rid = facts.attempts[0]
    consumer = checks._stream(facts.all("attempt", rid)[0])
    only_consumer = [event for event in events if checks._stream(event) == consumer]
    batch = json.loads(files[checks.CAPTURE + "CLIENT_BATCH.json"])
    rows, _ = checks._declared_rows(batch)
    inventory, _ = checks._inventory(batch, rows, only_consumer)
    assert inventory["status"] == "complete"
    transfer = next(row["transfer"] for row in rows if row["join"]["engine_request"] == rid)
    report = checks._lease(Facts(only_consumer), rid, True, issues, True, transfer)
    assert report["native_result"] == "passed"
    assert report["check_status"] == "passed"


@pytest.mark.parametrize("what,field,value", [
    ("lease_check", "details_complete", False),
    ("lease_check", "expected_ranks", [["prefill", 0], ["prefill", 0]]),
    ("lease_check", "observed_ranks", []),
    ("lease_check", "op_ids", ["unjoined-op"]),
    ("read_submission_result", "remote_rank", True),
])
def test_native_pass_with_partial_unjoined_or_untyped_scope_is_unverified(original, what, field, value):
    files = checks._archive(original[1])
    events, _ = checks._raw_events(files)
    facts = Facts(events)
    rid = facts.attempts[0]
    event = facts.all(what, rid)[0]
    event["args"][field] = value
    report = checks._lease(Facts(events), rid, True, [], True)
    assert report["native_result"] == "passed"
    assert report["check_status"] == "unverified"


@pytest.mark.parametrize("mutation", ["drop", "missing_health", "source_version", "partial_line"])
def test_actual_native_positives_keep_coverage_unknown_on_recording_gaps(original, mutation):
    def edit(_, files):
        name = checks.CAPTURE + "lifetime/decode/186.jsonl"
        events = [json.loads(line) for line in files[name].splitlines()]
        if mutation == "drop":
            events[0]["dropped"] = 1
        elif mutation == "missing_health":
            events = [e for e in events if e["args"].get("what") != "recording_health"]
        elif mutation == "source_version":
            events[0]["args"]["engine_identity"]["version"] = "0.32.1"
        files[name] = b"".join(io.json_bytes(event).replace(b"\n", b"") + b"\n" for event in events)
        if mutation == "partial_line":
            files[name] = files[name][:-1]
    result, archive = rewritten(original, edit)
    report = checks.evaluate_owner(result, archive)
    assert report["summary"]["declared_requests"] == 19
    assert report["summary"]["checks"]["lease"] == {"unverified": 19}
    assert all(r["checks"]["lease"]["native_result"] == "passed" for r in report["requests"])


def complete_objects():
    """Exact success labels from the owner producers, not a promoted saved run."""
    return [
        {"complete": True, "exit_code": 0, "failure": None, "cleanup": {"complete": True}},
        {"status": "ordinary_capture_qualified", "cleanup": {"complete": True}, "errors": [],
         "client_batch": {"status": "measured"}, "startup_interval_health": {"status": "observed_settled"},
         "capture": {"status": "captured"}, "qualification": {"status": "fine"}},
        {"status": "descriptive_client_measurement"},
        {"schema": "sdcguard.owner-job-status.v1", "complete": True, "stage": "complete"},
    ]


def synthetic_complete(original):
    """A detached assembly control. These invented terminal facts are not run evidence."""
    def edit(result, files):
        capture = json.loads(files[checks.CAPTURE + "CAPTURE.json"])
        capture.update(deepcopy(complete_objects()[1]))
        # Keep the independently recorded batch and every request; only the CPU
        # control's completion fields are synthetic, never the historical file.
        capture["client_batch"] = json.loads(files[checks.CAPTURE + "CLIENT_BATCH.json"])
        capture["client_batch"]["status"] = "measured"
        files[checks.CAPTURE + "CLIENT_BATCH.json"] = io.json_bytes(capture["client_batch"])
        files[checks.CAPTURE + "CAPTURE.json"] = io.json_bytes(capture)
        summary = json.loads(files["ordinary/SUMMARY.json"])
        summary.update(complete_objects()[2], ordinary_capture=capture)
        summary.pop("stopped_after_incomplete_capture", None)
        summary.pop("stopped_after_incomplete_segment", None)
        files["ordinary/SUMMARY.json"] = io.json_bytes(summary)
        job = json.loads(files["JOB_STATUS.json"])
        job.update(complete_objects()[3], cause="synthetic CPU completion control")
        files["JOB_STATUS.json"] = io.json_bytes(job)
        finished = {**complete_objects()[0], "run_id": result["run_id"]}
        files["FINISHED.json"] = io.json_bytes(finished)
        result["report"] = finished
    return rewritten(original, edit)


def test_synthetic_complete_shape_reaches_full_policy_assembly(original):
    result, archive = synthetic_complete(original)
    lease_only = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert lease_only["release_decision"] == "PASS"
    assert lease_only["run_complete"] is True
    assert lease_only["hold_reasons"] == []
    default = checks.evaluate_owner(result, archive)
    assert default["release_decision"] == "HOLD"
    assert default["summary"]["checks"]["lease"] == {"passed": 19}
    assert default["summary"]["checks"]["checksum"] == {"unverified": 19}
    assert "owner_run_completion_unverified" not in default["hold_reasons"]
    assert json.loads(original[0])["report"]["complete"] is False


@pytest.mark.parametrize("mutation,reason", [
    ("wrong_expected_nonce", "lease_detail_nonce_client_binding_unverified"),
    ("contradictory_observed_nonce", "lease_detail_matched_nonce_disagrees"),
])
def test_synthetic_complete_rejects_detail_nonce_contradictions(original, mutation, reason):
    def edit(_, files):
        for name in list(files):
            if "/lifetime/" not in name or not name.endswith(".jsonl"):
                continue
            events = [json.loads(line) for line in files[name].splitlines()]
            for event in events:
                args = event.get("args", {})
                if args.get("what") == "lease_check_detail":
                    if mutation == "wrong_expected_nonce":
                        args.update(expected_nonce=1, observed_nonce=1)
                    else:
                        args["observed_nonce"] = args["expected_nonce"] + 1
            files[name] = b"".join(io.json_bytes(event).replace(b"\n", b"") + b"\n" for event in events)
    result, archive = rewritten(synthetic_complete(original), edit)
    report = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert report["summary"]["declared_requests"] == report["summary"]["matched_requests"] == 19
    assert report["release_decision"] == "HOLD"
    assert report["summary"]["checks"]["lease"] == {"unverified": 19}
    assert all(row["checks"]["lease"]["native_result"] == "passed" for row in report["requests"])
    assert all(reason in row["checks"]["lease"]["missing_evidence"] for row in report["requests"])


def test_synthetic_complete_holds_when_request_outcome_is_not_satisfied(original):
    def edit(_, files):
        for name in list(files):
            if "/lifetime/" not in name or not name.endswith(".jsonl"):
                continue
            events = [json.loads(line) for line in files[name].splitlines()]
            for event in events:
                if event.get("args", {}).get("what") == "admit":
                    event["args"]["transferred"] = False
            files[name] = b"".join(io.json_bytes(event).replace(b"\n", b"") + b"\n" for event in events)
    result, archive = rewritten(synthetic_complete(original), edit)
    report = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert report["summary"]["declared_requests"] == report["summary"]["matched_requests"] == 19
    assert report["summary"]["checks"]["lease"] == {"passed": 19}
    assert all(row["request_outcome"] == "unknown" for row in report["requests"])
    assert report["release_decision"] == "HOLD"
    assert any("request_outcome_not_satisfied:unknown" in reason for reason in report["hold_reasons"])


@pytest.mark.parametrize("change", ["legacy_schema", "missing_outcome_requirement"])
def test_old_or_incomplete_promotion_policy_is_not_reinterpreted(change):
    value = checks.policy(["lease"])
    assert value["required_request_outcome"] == "satisfied"
    if change == "legacy_schema":
        value["schema"] = "sdcguard.native-check-policy.v1"
    else:
        value.pop("required_request_outcome")
    with pytest.raises(ValueError, match="policy"):
        checks._policy(value)


@pytest.mark.parametrize("field,value,reason", [
    ("lease_slot", 1, "lease_detail_slot_binding_unverified"),
    ("remote_rank", 1, "lease_detail_rank_coverage_unverified"),
    ("op_ids", ["unjoined-detail-operation"], "lease_detail_operation_binding_unverified"),
    ("done", False, "lease_detail_completion_unverified"),
    ("expected_nonce", True, "lease_detail_nonce_client_binding_unverified"),
    ("cause", "lease_mismatch", "lease_detail_cause_disagrees"),
    ("what", "unavailable_detail", "lease_detail_rank_coverage_unverified"),
])
def test_synthetic_complete_requires_bound_detail_for_each_expected_rank(original, field, value, reason):
    def edit(_, files):
        for name in list(files):
            if "/lifetime/" not in name or not name.endswith(".jsonl"):
                continue
            events = [json.loads(line) for line in files[name].splitlines()]
            for event in events:
                args = event.get("args", {})
                if args.get("what") == "lease_check_detail":
                    args[field] = value
            files[name] = b"".join(io.json_bytes(event).replace(b"\n", b"") + b"\n" for event in events)
    result, archive = rewritten(synthetic_complete(original), edit)
    report = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert report["summary"]["matched_requests"] == 19
    assert report["release_decision"] == "HOLD"
    assert report["summary"]["checks"]["lease"] == {"unverified": 19}
    assert all(reason in row["checks"]["lease"]["missing_evidence"] for row in report["requests"])


@pytest.mark.parametrize("name,field,value", [
    ("FINISHED.json", "exit_code", False),
    ("FINISHED.json", "failure", "synthetic missing proof"),
    ("FINISHED.json", "cleanup", {"complete": False}),
    ("JOB_STATUS.json", "complete", False),
    ("JOB_STATUS.json", "stage", "ordinary_capture"),
])
def test_synthetic_contradictory_completion_holds_full_policy(original, name, field, value):
    def edit(result, files):
        obj = json.loads(files[name])
        obj[field] = value
        files[name] = io.json_bytes(obj)
        if name == "FINISHED.json":
            result["report"] = obj
    result, archive = rewritten(synthetic_complete(original), edit)
    report = checks.evaluate_owner(result, archive, checks.policy(["lease"]))
    assert report["release_decision"] == "HOLD"
    assert report["run_complete"] is False


@pytest.mark.parametrize("name,value", [
    ("JOB_STATUS.json", None), ("JOB_STATUS.json", []),
    ("MODEL_REFERENCE.json", []), ("STARTED.json", None),
])
def test_malformed_owner_identity_metadata_does_not_crash(original, name, value):
    result, archive = rewritten(original, lambda _, files: files.update({name: io.json_bytes(value)}))
    report = checks.evaluate_owner(result, archive)
    assert report["release_decision"] == "HOLD"
    assert report["summary"]["declared_requests"] == 19
    assert any("identity" in reason for reason in report["hold_reasons"])


@pytest.mark.parametrize("field,value", [("run_id", "not-an-owner-uuid"), ("packet_sha256", "G" * 64)])
def test_unsupported_owner_identifier_formats_are_held(original, field, value):
    result, archive = rewritten(original, lambda result, _: result.update({field: value}))
    report = checks.evaluate_owner(result, archive)
    assert report["release_decision"] == "HOLD"
    assert "owner_run_identity_missing" in report["hold_reasons"]


def test_exact_owner_completion_labels_are_accepted_without_relabeling():
    assert checks._run_complete(*complete_objects()) is True


@pytest.mark.parametrize("part,field,value", [
    (0, "exit_code", False), (0, "exit_code", 0.0), (0, "failure", "still running"),
    (0, "cleanup", {"complete": False}), (0, "complete", 1),
    (1, "status", "complete"), (1, "errors", None), (1, "cleanup", {"complete": 1}),
    (1, "capture", None), (1, "qualification", {"status": "unknown"}),
    (2, "status", "inconclusive"), (3, "complete", False), (3, "stage", "ordinary_capture"),
])
def test_contradictory_owner_completion_shapes_remain_incomplete(part, field, value):
    objects = complete_objects()
    objects[part][field] = value
    assert checks._run_complete(*objects) is False


@pytest.mark.parametrize("field,value", [("startup", []), ("profile", None), ("topology", []), ("transfer", 1)])
def test_malformed_startup_stays_unknown_without_traceback(original, field, value):
    def edit(_, files):
        capture = json.loads(files[checks.CAPTURE + "CAPTURE.json"])
        if field == "startup":
            capture[field] = value
        else:
            capture["startup"]["decode"][field] = value
        files[checks.CAPTURE + "CAPTURE.json"] = io.json_bytes(capture)
        summary = json.loads(files["ordinary/SUMMARY.json"])
        summary["ordinary_capture"] = capture
        files["ordinary/SUMMARY.json"] = io.json_bytes(summary)
    result, archive = rewritten(original, edit)
    report = checks.evaluate_owner(result, archive)
    assert report["release_decision"] == "HOLD"
    assert report["context"]["adapter_supported"] is False
    assert report["summary"]["declared_requests"] == 19


def test_rehashed_derived_report_cannot_replace_raw_result(original, tmp_path):
    owner, packet = tmp_path / "owner", tmp_path / "packet"
    owner.mkdir()
    (owner / "RESULT.json").write_bytes(original[0])
    (owner / "ordinary-evidence.zip").write_bytes(original[1])
    checks.capture(owner, packet)
    report = json.loads((packet / "report.json").read_bytes())
    report["release_decision"] = "PASS"
    (packet / "report.json").write_bytes(io.json_bytes(report))
    manifest = json.loads((packet / "manifest.json").read_bytes())
    raw = (packet / "report.json").read_bytes()
    manifest["files"]["report.json"] = {"sha256": io.digest(raw), "bytes": len(raw)}
    (packet / "manifest.json").write_bytes(io.json_bytes(manifest))
    with pytest.raises(ValueError, match="recomputed raw evidence"):
        checks.verify(packet)


@pytest.mark.parametrize("name", ["../outside.json", "/absolute.json", "a\\b.json", "a/../b.json"])
def test_unsafe_zip_names_are_refused_without_extracting(name):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(name, b"{}")
    with pytest.raises(ValueError):
        checks._archive(stream.getvalue())


def test_html_escapes_client_controlled_text(original):
    result, archive = rewritten(original, lambda _, files: edit_batch(files,
        lambda batch: batch["references"][0].update(decode_id="<script>alert(1)</script>")))
    report = checks.evaluate_owner(result, archive)
    output = checks.render_html(report)
    assert b"<script>" not in output
    assert b"&lt;script&gt;" in output


def test_invalid_or_empty_policy_is_refused():
    for value in ([], ["lease", "lease"], ["unknown"], [["lease"]]):
        with pytest.raises(ValueError):
            checks.policy(value)
