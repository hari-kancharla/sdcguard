"""Required-check policies over saved data only. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from sdcguard import cli, evidence_io as io, required_gate as gate

DATA = Path(__file__).parent / "data"
CHECKSUM = DATA / "checksum_owner_870c1aac"
LEASE = DATA / "native_check_owner_r3"


@pytest.fixture(scope="module")
def checksum_run():
    source = json.loads((CHECKSUM / "SOURCE.json").read_bytes())
    for name, receipt in source["files"].items():
        raw = (CHECKSUM / name).read_bytes()
        assert {"bytes": len(raw), "sha256": io.digest(raw)} == receipt
    return tuple((CHECKSUM / name).read_bytes() for name in ("RESULT.json", "ordinary-evidence.zip", "CPU_PREFLIGHT.json"))


def edit_run(original, change):
    result, archive, preflight = original
    result, preflight = json.loads(result), json.loads(preflight)
    files = gate.read_archive(archive)
    change(result, files, preflight)
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as packed:
        for name, raw in files.items():
            packed.writestr(name, raw)
    archive = stream.getvalue()
    result["archive_sha256"] = io.digest(archive)
    return io.json_bytes(result), archive, io.json_bytes(preflight)


def test_real_saved_checksum_requests_pass_only_in_explicit_observed_scope(checksum_run):
    result = gate.checksum_gate(*checksum_run, ["checksum"], "observer-on")
    assert result["decision"] == "PASS"
    assert result["selected_requests"] == result["passed_requests"] == 90
    assert result["total_planned_requests"] == 180 and result["excluded_requests"] == 90
    assert result["hold_reasons"] == [] and result["runtime_qualified"] is False
    assert all(row["decision"] == "PASS" and not row["reasons"] for row in result["requests"])


def test_default_denominator_includes_the_unobserved_baseline(checksum_run):
    result = gate.checksum_gate(*checksum_run, ["checksum"])
    assert result["decision"] == "HOLD"
    assert result["selected_requests"] == 180 and result["excluded_requests"] == 0
    assert result["passed_requests"] == 90
    held = [row for row in result["requests"] if row["decision"] == "HOLD"]
    assert len(held) == 90 and all("observer_disabled" in row["reasons"] for row in held)


@pytest.mark.parametrize("required", [["lease", "checksum"], ["checksum", "lease"]])
def test_combined_mode_is_explicitly_unsupported(checksum_run, required):
    result = gate.checksum_gate(*checksum_run, required, "observer-on")
    assert result["decision"] == "HOLD"
    assert "combined_lease_checksum_not_supported" in result["hold_reasons"]
    assert all("combined_lease_checksum_not_supported" in row["reasons"] for row in result["requests"])


def test_checksum_run_cannot_borrow_historical_lease_passes(checksum_run):
    result = gate.checksum_gate(*checksum_run, ["lease"], "observer-on")
    assert result["decision"] == "HOLD"
    assert all("lease:not_recorded" in row["reasons"] for row in result["requests"])


def test_unknown_arm_never_passes_vacuously(checksum_run):
    result = gate.checksum_gate(*checksum_run, ["checksum"], "missing")
    assert result["decision"] == "HOLD" and result["selected_requests"] == 0
    assert "selected_arm_not_in_plan" in result["hold_reasons"]


@pytest.mark.parametrize("field,value", [("version", "0.34.0"), ("source", "0" * 40),
    ("plan", "0" * 64), ("failure", "recorded failure"), ("schema", "unknown")])
def test_unknown_version_or_contradictory_build_holds(checksum_run, field, value):
    def edit(_, files, preflight):
        if field == "version":
            preflight["installed_engine"]["versions"]["vllm"] = value
        elif field == "source":
            preflight["source_revision"] = value
        elif field == "plan":
            preflight["plan_sha256"] = value
        else:
            preflight[field] = value
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD"
    assert all(row["decision"] == "HOLD" and row["reasons"] for row in result["requests"])
    assert any(reason.startswith(("unsupported_recorded_engine_version", "recorded_engine_build_unverified"))
               for reason in result["hold_reasons"])


def test_hash_consistent_unknown_engine_version_is_still_unsupported(checksum_run):
    def edit(result, files, preflight):
        preflight["installed_engine"]["versions"]["vllm"] = "0.99.0"
        witness = json.loads(files["BUILD_WITNESS.json"])
        witness["witness"]["installed_identity_sha256"] = io.digest(io.json_bytes(preflight["installed_engine"]))
        witness["sha256"] = io.digest(io.json_bytes(witness["witness"]))
        files["BUILD_WITNESS.json"] = io.json_bytes(witness)
        preflight["build_witness"] = deepcopy(witness["witness"])
        preflight["build_witness_sha256"] = result["build_witness_sha256"] = witness["sha256"]
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD"
    assert result["hold_reasons"] == ["unsupported_recorded_engine_version"]


def test_consistently_copied_cpu_evidence_never_qualifies_runtime_coverage(checksum_run):
    def edit(_, files, preflight):
        captures = json.loads(files["CAPTURES.json"])
        for kind, key in (("checksum", "checksum_snapshots"), ("bindings", "binding_snapshots")):
            name = next(n for n in files if n.startswith("observer-on/decode/" + kind + "-"))
            value = json.loads(files[name])
            value["snapshot"]["qualification"] = "cpu_source_functions"
            files[name] = io.json_bytes(value)
            captures["captures"][0][key] = [deepcopy(value["snapshot"])]
        files["CAPTURES.json"] = io.json_bytes(captures)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD"
    assert "runtime_capture_source_unverified" in result["hold_reasons"]
    assert not any("assembly_disagrees" in reason for reason in result["hold_reasons"])


def test_unknown_lease_version_holds_even_with_complete_cpu_control():
    from test_native_checks import synthetic_complete, rewritten, checks
    original = tuple((LEASE / n).read_bytes() for n in ("RESULT.json", "ordinary-evidence.zip"))
    def edit(_, files):
        capture = json.loads(files[checks.CAPTURE + "CAPTURE.json"])
        for role in ("prefill", "decode"):
            capture["startup"][role]["engine_version"] = "0.99.0"
            name = checks.CAPTURE + "startup-" + role + ".json"
            startup = json.loads(files[name]);startup["facts"] = deepcopy(capture["startup"][role])
            files[name] = io.json_bytes(startup)
        files[checks.CAPTURE + "CAPTURE.json"] = io.json_bytes(capture)
        summary = json.loads(files["ordinary/SUMMARY.json"]);summary["ordinary_capture"] = capture
        files["ordinary/SUMMARY.json"] = io.json_bytes(summary)
    result = gate.lease_gate(*rewritten(synthetic_complete(original), edit), ["lease"])
    assert result["decision"] == "HOLD"
    assert "unsupported_recorded_engine_config" in result["hold_reasons"]
    assert "owner_run_completion_unverified" not in result["hold_reasons"]


@pytest.mark.parametrize("kind", ["close", "snapshot", "startup", "cleanup", "membership", "supervisor", "cpu"])
def test_raw_receipt_gaps_cannot_borrow_assembled_passes(checksum_run, kind):
    def edit(_, files, preflight):
        if kind in ("cleanup", "membership", "supervisor"):
            name = "observer-on/CLEANUP.json"
            value = json.loads(files[name])
            if kind == "cleanup":
                value["terminal_handles"].pop()
            elif kind == "membership":
                value["process_membership"][0]["start"] += 1
            else:
                value["supervisor_receipts"][0]["engine_returncode"] = 1
        elif kind == "startup":
            name = next(n for n in files if n.startswith("observer-on/decode/startup-"))
            value = json.loads(files[name]);value["chunked_prefill"] = False
        else:
            name = next(n for n in files if n.startswith("observer-on/decode/checksum-"))
            value = json.loads(files[name])
            if kind == "close":
                value["original_shutdown_completed"] = False
            elif kind == "snapshot":
                value["snapshot"]["capture_context"]["process_id"] += 1
            else:
                value["snapshot"]["qualification"] = "cpu_source_functions"
        files[name] = io.json_bytes(value)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD" and result["hold_reasons"]
    assert all(row["decision"] == "HOLD" for row in result["requests"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "finish", "started", "sequence"])
def test_client_journal_is_independently_reconciled(checksum_run, mutation):
    def edit(_, files, preflight):
        name = "client/CLIENT_RECEIPTS.jsonl"
        rows = [json.loads(line) for line in files[name].splitlines()]
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows.append(deepcopy(rows[-1]))
        elif mutation == "finish":
            rows[1]["receipt"]["http_status"] = 503
        elif mutation == "started":
            rows[0]["receipt"]["request_body"]["request_id"] = "different"
        else:
            rows[0]["sequence"] = True
        files[name] = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD"
    assert any(reason.startswith("client_journal_") for reason in result["hold_reasons"])


def test_saved_oct29_lease_success_does_not_erase_incomplete_run():
    result = gate.inspect(LEASE, ["lease"])
    assert result["decision"] == "HOLD" and result["selected_requests"] == 19
    assert result["hold_reasons"] == ["owner_run_completion_unverified"]
    assert all(row["checks"]["lease"]["check_status"] == "passed" for row in result["requests"])


@pytest.mark.parametrize("kind", ["root_exit", "reaped_exit", "final_stage", "outer_witness"])
def test_independent_review_contradictions_hold(checksum_run, kind):
    def edit(result, files, preflight):
        if kind == "outer_witness":
            result["build_witness_sha256"] = "0" * 64
        elif kind == "final_stage":
            value = json.loads(files["FINISHED.json"])
            value["stage"] = "incomplete"
            files["FINISHED.json"] = io.json_bytes(value)
            result["report"] = value
        else:
            name = "observer-on/CLEANUP.json"
            value = json.loads(files[name])
            root = value["groups"][0]
            rows = value["terminal_handles"] if kind == "root_exit" else value["reaped_leaders"]
            next(row for row in rows if row["pid"] == root)["returncode"] = 137
            files[name] = io.json_bytes(value)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD" and result["hold_reasons"]
    assert all(row["decision"] == "HOLD" and row["reasons"] for row in result["requests"])


@pytest.mark.parametrize("name", ["stock_RESULT.json", "fixed_RESULT.json"])
def test_saved_oct27_summary_cannot_supply_unrecorded_per_request_checks(name):
    path = Path(__file__).parents[1] / "src/sdcguard/case_data/61229" / name
    result = gate.inspect(path, ["lease"])
    assert result["decision"] == "HOLD" and result["total_planned_requests"] is None
    assert "legacy_summary_has_no_complete_per_request_check_evidence" in result["hold_reasons"]


def test_cli_returns_zero_only_for_the_complete_supported_single_check(capsys):
    assert cli.main(["gate", str(CHECKSUM), "--require", "checksum", "--arm", "observer-on", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "PASS" and result["passed_requests"] == 90
    assert cli.main(["gate", str(CHECKSUM), "--require", "checksum", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "HOLD"


def test_required_flag_cannot_dispatch_to_legacy_execution(monkeypatch, capsys):
    from sdcguard import gate as legacy
    def forbidden(*_):
        raise AssertionError("legacy execution path must not be entered")
    monkeypatch.setattr(legacy, "main", forbidden)
    assert cli.main(["gate", "run", "--require", "checksum", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["decision"] == "HOLD"


def test_no_requests_or_no_units_cannot_pass():
    assert gate.decide([], ["checksum"])["decision"] == "HOLD"
    result = gate.decide([dict(client_request_id="cpu-control", checks={"checksum": dict(
        native_result="passed", expected_units=[], compared_units=[], completed=True,
        evidence_complete=True, applicable=True)})], ["checksum"])
    assert result["decision"] == "HOLD"



def test_prefix_cache_startup_matches_only_the_declared_profile(checksum_run):
    files = gate.read_archive(checksum_run[1])
    plan = json.loads(files["client/REQUEST_PLAN.json"])
    arm = plan["arms"][1]
    name = next(n for n in files if n.startswith("observer-on/decode/startup-"))
    startup = json.loads(files[name])
    assert gate._startup(startup, plan, arm, "decode")
    plan["prefix_caching"] = True
    assert not gate._startup(startup, plan, arm, "decode")
    startup["prefix_caching"] = True
    assert gate._startup(startup, plan, arm, "decode")


def test_prefix_enabled_run_without_new_cache_observations_holds(checksum_run):
    def edit(_, files, preflight):
        plan = json.loads(files["client/REQUEST_PLAN.json"])
        plan["prefix_caching"] = True
        files["client/REQUEST_PLAN.json"] = io.json_bytes(plan)
        for name in ("client/CLIENT_INVENTORY.json", "CAPTURES.json"):
            value = json.loads(files[name])
            value["plan_sha256"] = io.digest(files["client/REQUEST_PLAN.json"])
            files[name] = io.json_bytes(value)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    assert result["decision"] == "HOLD" and result["selected_requests"] == 90
    assert result["passed_requests"] == 0
    assert all("native_cache_snapshot_missing_or_ambiguous:prefill" in r["reasons"] for r in result["requests"])


def raw_prefix_envelopes(checksum_run):
    """Synthetic envelope controls, not native cache or GPU evidence."""
    files = gate.read_archive(checksum_run[1])
    plan = json.loads(files["client/REQUEST_PLAN.json"])
    plan["prefix_caching"] = True
    captures = json.loads(files["CAPTURES.json"])
    capture = captures["captures"][0]
    for name in list(files):
        if "/startup-" in name:
            value = json.loads(files[name])
            value["prefix_caching"] = True
            files[name] = io.json_bytes(value)
    for role in ("prefill", "decode"):
        prefix = "observer-on/" + role + "/"
        name = next(n for n in files if n.startswith(prefix+"startup-"))
        startup = json.loads(files[name])
        context = {k: startup[k] for k in (*gate.CONTEXT, "process_id")}
        snapshot = dict(context, role=role, qualification="source_bound_candidate", synthetic_envelope_control=True)
        receipt = dict(schema="sdcguard.owner-checksum-close.v1", context=context, role=role,
            original_shutdown_completed=True, original_shutdown_error=None, recorder_errors=[], snapshot=snapshot)
        files[prefix+"cache-"+str(context["process_id"])+".json"] = io.json_bytes(receipt)
        if role == "decode":
            capture["startup"] = startup
            envelope = capture
        else:
            binding_name = next(n for n in files if n.startswith("observer-on/decode/bindings-"))
            binding = json.loads(files[binding_name])
            binding.update(context=context, role=role)
            binding["snapshot"].update(context)
            files[prefix+"bindings-"+str(context["process_id"])+".json"] = io.json_bytes(binding)
            envelope = dict(startup=startup, capture_complete=True, capture_errors=[],
                            binding_snapshots=[binding["snapshot"]])
            capture["prefill_capture"] = envelope
        envelope["cache_snapshots"] = [deepcopy(snapshot)]
    assert gate._raw_checksum_evidence(files, plan, captures) == []
    return files, plan, captures


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("change,reason", [
    ("missing", "native_snapshot_assembly_disagrees"),
    ("role", "native_cache_role_disagrees"),
    ("pid", "native_snapshot_context_disagrees"),
    ("assembly", "native_snapshot_assembly_disagrees"),
    ("cpu", "non_runtime_native_evidence"),
])
def test_raw_prefix_receipts_are_independently_required(checksum_run, role, change, reason):
    files, plan, captures = raw_prefix_envelopes(checksum_run)
    name = next(n for n in files if n.startswith("observer-on/"+role+"/cache-"))
    value = json.loads(files[name])
    if change == "missing": del files[name]
    elif change == "assembly": value["snapshot"]["extra"] = "changed"
    elif change == "pid": value["snapshot"]["process_id"] += 1
    elif change == "role": value["snapshot"]["role"] = "decode" if role == "prefill" else "prefill"
    else:
        value["snapshot"]["qualification"] = "cpu_source_functions"
        envelope = captures["captures"][0]
        if role == "prefill": envelope = envelope["prefill_capture"]
        envelope["cache_snapshots"] = [deepcopy(value["snapshot"])]
    if change != "missing": files[name] = io.json_bytes(value)
    errors = gate._raw_checksum_evidence(files, plan, captures)
    assert any(e.startswith(reason+":") for e in errors)


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_checksum_must_be_enabled_on_both_servers(checksum_run, role, value):
    def edit(_, files, preflight):
        name = next(n for n in files if n.startswith("observer-on/"+role+"/startup-"))
        startup = json.loads(files[name])
        if value is None: startup.pop("checksum_enabled")
        else: startup["checksum_enabled"] = value
        files[name] = io.json_bytes(startup)
        if role == "decode":
            captures = json.loads(files["CAPTURES.json"])
            captures["captures"][0]["startup"] = deepcopy(startup)
            captures["captures"][0]["startup_sha256"] = io.digest(io.json_bytes(startup))
            files["CAPTURES.json"] = io.json_bytes(captures)
    result = gate.checksum_gate(*edit_run(checksum_run, edit), ["checksum"], "observer-on")
    reason = "checksum_check_not_enabled:observer-on:" + role
    assert result["decision"] == "HOLD" and result["passed_requests"] == 0
    assert all(reason in row["reasons"] for row in result["requests"])


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_lease_must_be_enabled_on_both_servers(role, value):
    from test_native_checks import synthetic_complete, rewritten, checks
    original = tuple((LEASE / n).read_bytes() for n in ("RESULT.json", "ordinary-evidence.zip"))
    def edit(_, files):
        capture = json.loads(files[checks.CAPTURE + "CAPTURE.json"])
        transfer = capture["startup"][role]["transfer"]
        if value is None: transfer.pop("lease_verification")
        else: transfer["lease_verification"] = value
        name = checks.CAPTURE + "startup-" + role + ".json"
        startup = json.loads(files[name]);startup["facts"] = deepcopy(capture["startup"][role])
        files[name] = io.json_bytes(startup)
        files[checks.CAPTURE + "CAPTURE.json"] = io.json_bytes(capture)
        summary = json.loads(files["ordinary/SUMMARY.json"]);summary["ordinary_capture"] = capture
        files["ordinary/SUMMARY.json"] = io.json_bytes(summary)
    result = gate.lease_gate(*rewritten(synthetic_complete(original), edit), ["lease"])
    reason = "lease_check_not_enabled:" + role
    assert result["decision"] == "HOLD" and result["passed_requests"] == 0
    assert all(reason in row["reasons"] for row in result["requests"])
