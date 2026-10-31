"""Offline native-check execution coverage. Owner: Hari Krishna Kancharla.

A recorded native outcome and a complete comparison scope are different facts.
This module starts no SDK, engine, serving request or device work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from importlib.resources import files as resources
import html
from io import BytesIO
from pathlib import Path
import re
import stat
import zipfile
import zlib

from . import client_inventory, evidence_envelope as envelope, evidence_io as io
from . import evidence_native_io as bounded
from .lifetime.adapter import read_events
from .lifetime.obligations import Facts, Finding, evaluate

SCHEMA = "sdcguard.native-check-coverage.v1"
POLICY_SCHEMA = "sdcguard.native-check-policy.v2"
PROFILE = "vllm032-fixed-nixl-pull-single-rank.v1"
TITLE = "Engine pass vs actually covered"
CHECKS = ("lease", "checksum", "bootstrap", "canary")
NATIVE_RESULTS = ("passed", "mismatch", "unverified", "admitted", "activity", "scheduled")
CAPTURE = "ordinary/ordinary-capture/"
MAX_UNITS = 65536
LIMITS = [
    "Coverage concerns only the declared ordinary capture client batch, not all server traffic.",
    "Lease units are remote request leases by rank, not KV-content checksums or model layers.",
    "Recorded source fingerprints cover the named adapter files, not an immutable full engine build.",
    "Local hashes establish consistency only. Authenticity and loaded weights are not established.",
    "Router forwarding is unknown without recorded ingress and egress identity relations.",
    "Ownership, read rights, physical reads, answer correctness and isolation remain unknown.",
    "A release hold is a policy result, not a claim of engine corruption.",
    "CPU controls and source-only demonstrations cannot qualify an actual release.",
]


def _units(value, name):
    if value is None:
        return None
    if (type(value) is not list or len(value) > MAX_UNITS
            or any(type(x) is not str or not x or len(x) > 1024 for x in value)
            or len(set(value)) != len(value)):
        raise ValueError(name + " must be unique bounded nonempty unit strings or null")
    return sorted(value)


def assess_check(check_id, *, native_result, expected_units, compared_units, completed,
                 evidence_complete, applicable=True, reason=None):
    """Classify normalized evidence; this API does not authenticate runtime inputs."""
    if check_id not in CHECKS or native_result not in NATIVE_RESULTS:
        raise ValueError("unsupported check or native result")
    if (type(completed) is not bool or type(evidence_complete) is not bool
            or (applicable is not None and type(applicable) is not bool)
            or (reason is not None and (type(reason) is not str or not reason or len(reason) > 1024))):
        raise ValueError("explicit boolean completion and bounded reason are required")
    expected, compared = _units(expected_units, "expected_units"), _units(compared_units, "compared_units")
    if expected is not None and compared is not None and not set(compared) <= set(expected):
        raise ValueError("compared units include units outside the eligible scope")
    missing = sorted(set(expected or ()) - set(compared or ())) if expected is not None else None
    coverage = ("unknown" if expected is None or compared is None or not evidence_complete
                else "not_covered" if not expected or not compared or applicable is False
                else "partial" if missing else "complete")
    status = "unverified"
    derived_reason = reason
    if applicable is not True:
        derived_reason = derived_reason or ("not_applicable" if applicable is False else "applicability_unknown")
    elif native_result == "mismatch" and evidence_complete and completed and compared:
        status = "mismatch"
        derived_reason = derived_reason or "recorded_native_mismatch"
    elif (native_result == "passed" and completed and evidence_complete
          and coverage == "complete"):
        status, derived_reason = "passed", reason or "complete_recorded_comparison"
    else:
        derived_reason = derived_reason or (
            "recording_incomplete" if not evidence_complete else
            "expected_scope_unknown" if expected is None else
            "no_eligible_comparison_units" if not expected else
            "comparison_scope_partial" if coverage == "partial" else
            "no_recorded_comparison" if not compared else
            "comparison_not_completed" if not completed else "no_recorded_passing_result")
    return {"check": check_id, "native_result": native_result, "check_status": status,
            "coverage": coverage, "coverage_complete": coverage == "complete",
            "expected_units": expected, "compared_units": compared,
            "missing_units": missing, "completed": completed,
            "evidence_complete": evidence_complete, "applicable": applicable,
            "reason": derived_reason}


def _same(a, b):
    """Typed JSON equality: true, 1 and 1.0 cannot alias identity fields."""
    return io.json_bytes(a) == io.json_bytes(b)


def policy(required=None):
    required = ["lease", "checksum"] if required is None else required
    if (type(required) is not list or not required
            or any(type(x) is not str or x not in CHECKS for x in required)
            or len(set(required)) != len(required)):
        raise io.EvidenceError("required checks must be a nonempty unique list of supported check names")
    return {"schema": POLICY_SCHEMA, "required_checks": sorted(required),
            "completion_required": True, "required_request_outcome": "satisfied",
            "scope": "ordinary capture client batch only"}


def _policy(value):
    if type(value) is not dict or not _same(value, policy(value.get("required_checks"))):
        raise io.EvidenceError("unsupported native-check release policy")
    return value


def evaluator_hashes():
    names = ("native_checks.py", "client_inventory.py", "lifetime/adapter.py", "lifetime/obligations.py",
             "evidence_io.py", "evidence_envelope.py", "evidence_native_io.py", "evidence_json_budget.py",
             "config_inputs.py", "case_data/engine-pins.json")
    root = resources("sdcguard")
    return {n: io.digest(root.joinpath(*n.split("/")).read_bytes()) for n in names}


def _archive(raw):
    """Retain bounded regular ZIP members without extracting archive paths."""
    files = {}
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) > io.MAX_FILES:
                raise io.EvidenceError("archive member count exceeds its bound")
            total = 0
            for info in entries:
                io.relative(info.filename)
                mode = info.external_attr >> 16
                if (info.filename in files or info.is_dir() or info.flag_bits & 1
                        or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                        or info.file_size > io.MAX_FILE_BYTES):
                    raise io.EvidenceError("archive member is duplicate, non-regular or unsupported")
                total += info.file_size
                if total > io.MAX_TOTAL_BYTES:
                    raise io.EvidenceError("archive expanded bytes exceed their bound")
                data = archive.read(info)
                if len(data) != info.file_size:
                    raise io.EvidenceError("archive member size changed")
                files[info.filename] = data
    except (zipfile.BadZipFile, RuntimeError, OSError, NotImplementedError,
            EOFError, zlib.error) as exc:
        raise io.EvidenceError("unsupported evidence archive: " + str(exc)) from exc
    envelope.validate_snapshot(files, envelope.directories_for(files))
    return files


def _json(files, name):
    if name not in files:
        raise io.EvidenceError("required raw evidence is missing: " + name)
    return io.parse_json(files[name])


def _args(event):
    value = event.get("args", {})
    return value if type(value) is dict else {}


def _stream(event):
    return (event.get("session"), event.get("pid"))


def _ref(event):
    return {k: event.get(k) for k in ("session", "pid", "seq", "clock_id")}


def _raw_events(files):
    selected = {n: raw for n, raw in files.items()
                if n.startswith(CAPTURE + "lifetime/") and n.endswith(".jsonl")}
    issues = []
    if not selected:
        return [], ["no_lifetime_recording"]
    for name, raw in selected.items():
        if not raw or not raw.endswith(b"\n"):
            issues.append("unterminated_lifetime_recording:" + name)
        for line in raw.splitlines():
            try:
                value = io.parse_json(line)
                if type(value) is not dict:
                    raise io.EvidenceError("event is not an object")
            except io.EvidenceError:
                issues.append("malformed_lifetime_recording:" + name)
                break
    events = envelope.evaluate_private(selected, envelope.directories_for(selected),
        lambda root: read_events([root / name for name in sorted(selected)]))
    return events, issues


def _declared_rows(batch):
    """Preserve planned slots, missing rows and surplus rows before normalization."""
    if type(batch) is not dict:
        return [], ["client_batch_missing"]
    load = batch.get("load", {})
    valid_load = (type(load) is dict and all(type(load.get(k)) is int and 1 <= load[k] <= 10000
                  for k in ("references", "rounds", "requests_per_round")))
    references = batch.get("references", [])
    rounds = batch.get("rounds", [])
    references = references if type(references) is list else []
    rounds = rounds if type(rounds) is list else []
    planned = (load["references"] + load["rounds"] * load["requests_per_round"]) if valid_load else None
    if planned is not None and planned > client_inventory.MAX_REQUESTS:
        raise io.EvidenceError("declared client denominator exceeds its bound")
    rows, issues = [], ([] if valid_load else ["declared_client_load_unknown"])
    def append(slot, raw):
        rows.append({"slot": slot, "raw": raw, "client_request_id":
                     raw.get("decode_id") if type(raw) is dict and type(raw.get("decode_id")) is str else None})
    for i in range(max(load["references"] if valid_load else 0, len(references))):
        append("reference[" + str(i) + "]", references[i] if i < len(references) else None)
    for r in range(max(load["rounds"] if valid_load else 0, len(rounds))):
        rr = rounds[r] if r < len(rounds) and type(rounds[r]) is dict else {}
        requests = rr.get("requests", [])
        requests = requests if type(requests) is list else []
        for i in range(max(load["requests_per_round"] if valid_load else 0, len(requests))):
            append("round[" + str(r) + "].request[" + str(i) + "]", requests[i] if i < len(requests) else None)
    if len(rows) > client_inventory.MAX_REQUESTS:
        raise io.EvidenceError("retained client denominator exceeds its bound")
    if not rows:
        issues.append("no_declared_client_requests")
    return rows, issues


def _inventory(batch, rows, events):
    issues = []
    try:
        for row in rows:
            raw = row["raw"]
            if type(raw) is not dict:
                raise ValueError("a planned client row was not recorded")
            stages = raw.get("stages")
            if type(stages) is not list or len(stages) != 2:
                raise ValueError("two complete ordered client stages are required")
            produced = stages[0].get("response", {}).get("kv_transfer_params")
            sent = stages[1].get("body", {}).get("kv_transfer_params")
            if not _same(produced, sent):
                raise ValueError("typed producer descriptor differs from client decode body")
        inventory = client_inventory.from_performance(batch, workload="ordinary capture client batch")
        if len(inventory["requests"]) != len(rows):
            raise ValueError("normalized inventory changed the declared denominator")
        result = client_inventory.reconcile(inventory, events)
        for row, normalized, joined in zip(rows, inventory["requests"], result["requests"]):
            row.update({"client_request_id": normalized["client_request_id"],
                        "transfer": normalized["transfer"], "join": joined})
        issues.extend(result.get("issues", []))
        return result, issues
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        issues.append("client_inventory_unverified:" + str(exc))
        return {"status": "unknown", "declared_requests": len(rows), "matched_requests": 0,
                "requests": [], "unexpected_attempts": []}, issues


def _source_note(facts, attempt):
    if type(attempt.get("seq")) is not int:
        return None
    pins = io.parse_json(resources("sdcguard").joinpath("case_data/engine-pins.json").read_bytes())
    expected = {"adapter_fits": True, "file_sha256": pins["variants"]["fixed"], "fix": "present",
                "inspected_source_only": True, "variant": "fixed", "version": pins["version"]}
    required = {"LeaseVerifier._resolve", "LeaseVerifier.add_check", "LeaseVerifier.step"}
    prior = [e for e in facts.streams.get(_stream(attempt), ()) if e.get("kind") == "note"
             and type(e.get("seq")) is int and e["seq"] < attempt["seq"]]
    if any(_args(e).get("engine_identity") is not None
           and not _same(_args(e)["engine_identity"], expected) for e in prior):
        return None
    notes = [e for e in prior
             if type(_args(e).get("hooks")) is list
             and all(type(x) is str for x in _args(e)["hooks"])
             and required <= set(_args(e)["hooks"])
             and _same(_args(e).get("engine_identity"), expected)
             and _args(e).get("nixl_version") == "1.4.1"
             and type(e.get("seq")) is int and e["seq"] < attempt["seq"]]
    return notes[0] if notes else None


def _ranks(value):
    if (type(value) is not list or not value or len(value) > MAX_UNITS
            or any(type(x) is not list or len(x) != 2 or type(x[0]) is not str or not x[0]
                   or type(x[1]) is not int or x[1] < 0 for x in value)):
        return None
    units = ["remote-rank:" + x[0] + ":" + str(x[1]) for x in value]
    return None if len(set(units)) != len(units) else sorted(units)


def _lease_details(facts, request, attempt, check, client_transfer):
    """Bind each native comparison detail to the client's exact request lease."""
    details = facts.all("lease_check_detail", request)
    missing = []
    a, c = _args(attempt), _args(check)
    remote = a.get("remote_engine")
    if len(details) != 1:
        return [], ["lease_detail_rank_coverage_unverified"]
    detail = details[0]
    value = _args(detail)
    if (type(value.get("remote_engine")) is not str or value["remote_engine"] != remote
            or type(value.get("remote_rank")) is not int or value["remote_rank"] != 0):
        missing.append("lease_detail_rank_coverage_unverified")
    try:
        client_inventory._transfer(client_transfer)
    except (ValueError, KeyError, TypeError):
        client_transfer = {}
        missing.append("client_lease_descriptor_unverified")
    if (client_transfer.get("remote_engine_id") != remote
            or client_transfer.get("remote_request_id") != a.get("remote_request")):
        missing.append("lease_detail_client_request_binding_unverified")
    nonces = (value.get("expected_nonce"), a.get("lease_nonce"), client_transfer.get("remote_lease_nonce"))
    if (any(type(n) is not int or not 0 < n < 1 << 53 for n in nonces)
            or nonces[0] != nonces[1] or nonces[0] != nonces[2]):
        missing.append("lease_detail_nonce_client_binding_unverified")
    slots = (value.get("lease_slot"), a.get("lease_slot"), client_transfer.get("remote_lease_slot"))
    if (any(type(n) is not int or not 0 <= n < 1 << 64 for n in slots)
            or slots[0] != slots[1] or slots[0] != slots[2]):
        missing.append("lease_detail_slot_binding_unverified")
    observed = value.get("observed_nonce")
    if value.get("done") is not True:
        missing.append("lease_detail_completion_unverified")
    if type(observed) is not int or not -(1 << 63) <= observed < 1 << 63:
        missing.append("lease_detail_observed_nonce_unverified")
    if c.get("cause") == "matched":
        if value.get("cause") != "matched":
            missing.append("lease_detail_cause_disagrees")
        if type(observed) is not int or type(nonces[0]) is not int or observed != nonces[0]:
            missing.append("lease_detail_matched_nonce_disagrees")
    elif c.get("cause") == "lease_mismatch":
        if (value.get("cause") != "lease_mismatch" or type(observed) is not int
                or type(nonces[0]) is not int or observed == nonces[0]):
            missing.append("lease_detail_mismatch_unverified")
    else:
        missing.append("lease_detail_cause_unverified")
    if not _same(value.get("op_ids"), a.get("op_ids")):
        missing.append("lease_detail_operation_binding_unverified")
    return details, missing


def _lease(facts, request, eligible, raw_issues, source_supported, client_transfer=None):
    missing, witness = [], []
    attempt = facts.all("attempt", request)
    checks = facts.all("lease_check", request)
    native = ("passed" if len(checks) == 1 and _args(checks[0]).get("result") == "passed" else
              "mismatch" if len(checks) == 1 and _args(checks[0]).get("result") == "failed"
              and _args(checks[0]).get("cause") == "lease_mismatch" else "unverified")
    expected, compared = None, None
    if len(attempt) == 1:
        a = _args(attempt[0])
        if type(a.get("remote_engine")) is str and type(a.get("reader_rank")) is int:
            expected = ["remote-rank:" + a["remote_engine"] + ":0"]
    if len(checks) == 1:
        expected_recorded, compared = _ranks(_args(checks[0]).get("expected_ranks")), _ranks(_args(checks[0]).get("observed_ranks"))
        if expected is None or not _same(expected_recorded, expected):
            missing.append("eligible_remote_rank_scope_disagrees")
        if expected is not None and compared is not None and not set(compared) <= set(expected):
            compared = None
            missing.append("unexpected_or_ambiguous_compared_rank")
    if len(attempt) != 1:
        missing.append("unambiguous_attempt_missing")
    if len(checks) != 1:
        missing.append("no_recorded_lease_check" if not checks else "ambiguous_lease_checks")
    elif len(attempt) == 1:
        attempt, check = attempt[0], checks[0]
        a, c = _args(attempt), _args(check)
        details, detail_issues = _lease_details(facts, request, attempt, check, client_transfer)
        missing.extend(detail_issues)
        note = _source_note(facts, attempt)
        if not note or not source_supported:
            missing.append("unsupported_recorded_source_or_config")
        sent, done = facts.all("read_submission_result", request), facts.all("transfer_done", request)
        ids = a.get("op_ids")
        try:
            ids = _units(ids, "op_ids")
        except ValueError:
            ids = None
        if not ids:
            missing.append("operation_identity_unknown")
        if (a.get("mapping_complete") is not True or a.get("lease_fields") is not True
                or type(a.get("reader_rank")) is not int or a["reader_rank"] != 0):
            missing.append("attempt_scope_unverified")
        if c.get("details_complete") is not True or c.get("cause") not in ("matched", "lease_mismatch"):
            missing.append("native_comparison_details_unverified")
        if ((native == "passed" and c.get("cause") != "matched")
                or not _same(c.get("op_ids"), a.get("op_ids"))):
            missing.append("native_result_operation_binding_unverified")
        if len(sent) != 1 or len(done) != 1:
            missing.append("unambiguous_submission_or_completion_missing")
        else:
            sent, done = sent[0], done[0]
            s, d = _args(sent), _args(done)
            required = [e for e in (note, attempt, sent, done, *details, check) if e]
            witness = [_ref(e) for e in required]
            if (s.get("submitted") is not True or type(s.get("remote_rank")) is not int
                    or s["remote_rank"] != 0 or not _same(s.get("op_ids"), a.get("op_ids"))
                    or not _same(d.get("op_ids"), a.get("op_ids"))):
                missing.append("actual_submission_or_completion_binding_unverified")
            if any(_stream(e) != _stream(attempt) for e in required):
                missing.append("consumer_stream_binding_unverified")
            sequences = [e.get("seq") for e in required]
            if (any(type(n) is not int for n in sequences)
                    or any(x >= y for x, y in zip(sequences, sequences[1:]))):
                missing.append("consumer_causal_order_unverified")
            finding = Finding("native_lease_check_coverage", request)
            facts.coverage(required, check, finding)
            missing.extend(finding.missing)
    if not eligible:
        missing.append("client_descriptor_join_unverified")
    missing.extend(raw_issues)
    result = assess_check("lease", native_result=native, expected_units=expected,
        compared_units=compared, completed=len(checks) == 1 and _args(checks[0]).get("details_complete") is True,
        evidence_complete=not missing, reason=missing[0] if missing else None)
    result.update({"missing_evidence": list(dict.fromkeys(missing)), "witness": witness,
                   "scope": "one remote request lease on prefill rank 0"})
    return result


def _context(capture, summary, files):
    context, supported = {}, True
    startups = capture.get("startup")
    startups = startups if type(startups) is dict else {}
    for role in ("prefill", "decode"):
        facts = startups.get(role, {})
        recorded = _json(files, CAPTURE + "startup-" + role + ".json")
        if type(facts) is not dict or type(recorded) is not dict or not _same(recorded.get("facts"), facts):
            facts, supported = {}, False
        profile, topology, transfer = facts.get("profile", {}), facts.get("topology", {}), facts.get("transfer", {})
        if any(type(value) is not dict for value in (profile, topology, transfer)):
            profile, topology, transfer, supported = {}, {}, {}, False
        context[role] = {"engine_version": facts.get("engine_version"), "profile": profile,
                         "topology": topology, "transfer": transfer}
        if (facts.get("engine_version") != "0.32.0" or type(profile.get("cache_groups")) is not int
                or profile.get("cache_groups") != 1 or any(type(profile.get(k)) is not int or profile[k] != 1
                   for k in ("tp", "pp", "dcp", "dp"))
                or any(type(topology.get(k)) is not int or topology[k] != 1 for k in ("tp", "pp", "dcp", "dp", "pcp"))
                or transfer.get("configured_connector") != "NixlConnector"
                or transfer.get("scheduler_mode") != "pull" or transfer.get("worker_mode") != "pull"
                or transfer.get("lease_verification") is not True):
            supported = False
    return {"engine": "vllm", "version": "0.32.0" if supported else "unknown",
            "model": summary.get("model"), "model_revision": summary.get("model_revision"),
            "recorded_startup": context, "adapter_supported": supported,
            "full_engine_build": "unknown", "checksum_observation": "not recorded"}, supported


def _run_complete(finished, capture, summary, job_status):
    """Read the owner producers' exact completion contracts, separately from checks."""
    if any(type(value) is not dict for value in (finished, capture, summary, job_status)):
        return False
    def status(value, name, expected):
        return type(value) is dict and value.get(name) == expected
    return (finished.get("complete") is True and type(finished.get("exit_code")) is int
            and finished["exit_code"] == 0 and finished.get("failure") is None
            and type(finished.get("cleanup")) is dict and finished["cleanup"].get("complete") is True
            and type(capture.get("cleanup")) is dict and capture["cleanup"].get("complete") is True
            and capture.get("status") == "ordinary_capture_qualified"
            and type(capture.get("errors")) is list and capture["errors"] == []
            and status(capture.get("client_batch"), "status", "measured")
            and status(capture.get("startup_interval_health"), "status", "observed_settled")
            and status(capture.get("capture"), "status", "captured")
            and status(capture.get("qualification"), "status", "fine")
            and summary.get("status") == "descriptive_client_measurement"
            and summary.get("stopped_after_incomplete_capture", False) is False
            and summary.get("stopped_after_incomplete_segment", False) is False
            and job_status.get("schema") == "sdcguard.owner-job-status.v1"
            and job_status.get("complete") is True and job_status.get("stage") == "complete")


def evaluate_owner(result_raw, archive_raw, selected_policy=None):
    """Recompute scoped native coverage from retained owner-result bytes only."""
    selected_policy = _policy(policy() if selected_policy is None else selected_policy)
    result, files = io.parse_json(result_raw), _archive(archive_raw)
    if type(result) is not dict:
        raise io.EvidenceError("owner result must be an object")
    capture = _json(files, CAPTURE + "CAPTURE.json")
    summary, finished = _json(files, "ordinary/SUMMARY.json"), _json(files, "FINISHED.json")
    if any(type(v) is not dict for v in (capture, summary, finished)):
        raise io.EvidenceError("raw owner capture objects are required")
    issues = []
    if result.get("archive_sha256") != io.digest(archive_raw):
        issues.append("archive_sha256_disagrees")
    if not _same(result.get("report"), finished):
        issues.append("owner_finished_report_disagrees")
    run_id, packet = result.get("run_id"), result.get("packet_sha256")
    if (type(run_id) is not str or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
            or type(packet) is not str or re.fullmatch(r"[0-9a-f]{64}", packet) is None):
        issues.append("owner_run_identity_missing")
    identity_objects = {}
    for name in ("JOB_STATUS.json", "MODEL_REFERENCE.json", "STARTED.json"):
        value = _json(files, name)
        if type(value) is not dict:
            issues.append("owner_identity_object_invalid:" + name)
            value = {}
        identity_objects[name] = value
        if value.get("run_id") != run_id:
            issues.append("run_identity_disagrees:" + name)
        if name != "STARTED.json" and value.get("packet_sha256") != packet:
            issues.append("owner_packet_identity_disagrees:" + name)
    if summary.get("run_id") != run_id or finished.get("run_id") != run_id:
        issues.append("capture_run_identity_disagrees")
    if ("job_status" in finished
            and not _same(finished["job_status"], identity_objects["JOB_STATUS.json"])):
        issues.append("duplicated_job_status_disagrees")
    batch = capture.get("client_batch")
    if (not _same(batch, _json(files, CAPTURE + "CLIENT_BATCH.json"))
            or not _same(capture, summary.get("ordinary_capture"))):
        issues.append("duplicated_capture_or_client_batch_disagrees")
    rows, row_issues = _declared_rows(batch)
    events, raw_issues = _raw_events(files)
    inventory, inventory_issues = _inventory(batch, rows, events)
    issues.extend(row_issues + inventory_issues)
    context, supported = _context(capture, summary, files)
    facts = Facts(events)
    logical = {f.request: f.verdict for f in evaluate(events, [])}
    requests = []
    for row in rows:
        join = row.get("join", {})
        request = join.get("engine_request") if join.get("status") == "matched" else None
        eligible = request is not None and not issues
        lease = (_lease(facts, request, eligible, raw_issues, supported, row.get("transfer")) if request else
                 assess_check("lease", native_result="unverified", expected_units=None, compared_units=None,
                              completed=False, evidence_complete=False, reason="client_descriptor_join_unverified"))
        checks = {"lease": lease}
        for name in CHECKS[1:]:
            checks[name] = assess_check(name, native_result="unverified", expected_units=None,
                compared_units=None, completed=False, evidence_complete=False,
                reason="no_recorded_checksum_observation" if name == "checksum" else "unsupported_engine_check")
        requests.append({"slot": row["slot"], "client_request_id": row["client_request_id"],
            "engine_request_id": request, "client_join": join.get("status", "unknown"),
            "request_outcome": logical.get(request, "unknown"), "checks": checks,
            "client_sent_lease_fields": "recorded" if row.get("transfer") else "unknown",
            "client_prefill_descriptor_handoff": "exact typed match" if row.get("transfer") else "unknown",
            "router_forwarded_lease_fields": "unknown", "recorded_model_cache_groups":
             context["recorded_startup"]["decode"]["profile"].get("cache_groups")})
    run_complete = _run_complete(finished, capture, summary, identity_objects["JOB_STATUS.json"])
    holds = list(issues + raw_issues)
    if not run_complete:
        holds.append("owner_run_completion_unverified")
    if not supported:
        holds.append("unsupported_recorded_engine_config")
    if inventory.get("status") != "complete":
        holds.append("client_inventory_incomplete")
    for request in requests:
        if request["request_outcome"] != "satisfied":
            holds.append(request["slot"] + ":request_outcome_not_satisfied:" + request["request_outcome"])
        for check in selected_policy["required_checks"]:
            if request["checks"][check]["check_status"] != "passed":
                holds.append(request["slot"] + ":" + check + ":" + request["checks"][check]["reason"])
    if not requests:
        holds.append("no_declared_requests")
    counts = {name: dict(Counter(r["checks"][name]["check_status"] for r in requests)) for name in CHECKS}
    native_counts = {name: dict(Counter(r["checks"][name]["native_result"] for r in requests)) for name in CHECKS}
    both = sum(all(r["checks"][name]["check_status"] == "passed" for name in ("lease", "checksum")) for r in requests)
    return {"schema": SCHEMA, "title": TITLE, "owner": "Hari Krishna Kancharla", "profile": PROFILE,
        "scope": "declared ordinary/ordinary-capture client batch only", "run_id": run_id,
        "owner_packet_sha256": packet, "raw_receipts": {"RESULT.json": io.digest(result_raw),
          "ordinary-evidence.zip": io.digest(archive_raw)}, "policy": selected_policy,
        "policy_sha256": io.digest(io.json_bytes(selected_policy)), "context": context,
        "run_complete": run_complete, "run_failure": finished.get("failure"),
        "release_decision": "HOLD" if holds else "PASS", "hold_reasons": list(dict.fromkeys(holds)),
        "summary": {"declared_requests": len(requests), "matched_requests": inventory.get("matched_requests", 0),
          "both_lease_and_checksum_passed": both, "checks": counts, "native_results": native_counts},
        "client_inventory": inventory, "requests": requests, "limits": LIMITS}


def render_text(report):
    summary = report["summary"]
    lines = [TITLE, "Release: " + report["release_decision"], report["scope"],
             str(summary["both_lease_and_checksum_passed"]) + " of " + str(summary["declared_requests"])
             + " requests verified by both lease and checksum checks.",
             "Owner run complete: " + str(report["run_complete"]).lower(),
             "Required checks: " + ", ".join(report["policy"]["required_checks"]),
             "Required request outcome: " + report["policy"]["required_request_outcome"]]
    for name in report["policy"]["required_checks"]:
        lines.append(name + ": " + str(summary["native_results"][name].get("passed", 0))
            + " native passes recorded; " + str(summary["checks"][name].get("passed", 0)) + " of "
            + str(summary["declared_requests"]) + " requests have complete passing check coverage.")
    for row in report["requests"]:
        lines.append(row["slot"] + " | " + (row["client_request_id"] or "ID unknown") + " | " +
            "; ".join(name + ": native=" + row["checks"][name]["native_result"] +
              ", coverage=" + row["checks"][name]["coverage"] + ", result=" + row["checks"][name]["check_status"]
              + ", reason=" + row["checks"][name]["reason"] for name in report["policy"]["required_checks"]))
    lines.extend(["", "Hold reasons:", *["* " + reason for reason in report["hold_reasons"]],
                  "", "Limits:", *["* " + limit for limit in LIMITS]])
    return "\n".join(lines) + "\n"


def render_html(report):
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>" + TITLE + "</title><style>body{font:16px system-ui;margin:2rem;max-width:100rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}</style><h1>" + TITLE +
        "</h1><pre>" + html.escape(render_text(report)) + "</pre></html>\n").encode()


def _derived(result, archive, selected):
    report = evaluate_owner(result, archive, selected)
    return report, {"report.json": io.json_bytes(report), "report.txt": render_text(report).encode(),
                    "report.html": render_html(report)}


def capture(owner_result, output, *, required=None):
    envelope.reject_overlap(output, [owner_result])
    result, archive = io.read_file(owner_result, "RESULT.json"), io.read_file(owner_result, "ordinary-evidence.zip")
    selected = policy(required)
    report, derived = _derived(result, archive, selected)
    if (result != io.read_file(owner_result, "RESULT.json")
            or archive != io.read_file(owner_result, "ordinary-evidence.zip")):
        raise io.EvidenceError("owner inputs changed during capture")
    saved = {"inputs/RESULT.json": result, "inputs/ordinary-evidence.zip": archive,
             "policy.json": io.json_bytes(selected), "evaluators.json": io.json_bytes(evaluator_hashes()), **derived}
    envelope.complete(output, saved, envelope.directories_for(saved), schema=SCHEMA, profile=PROFILE)
    return report


def verify(packet):
    files, directories = bounded.capture_snapshot(packet)
    if envelope.check_manifest(files, directories, schema=SCHEMA) != PROFILE:
        raise io.EvidenceError("unsupported native-check packet profile")
    expected = {"inputs/RESULT.json", "inputs/ordinary-evidence.zip", "policy.json", "evaluators.json",
                "report.json", "report.txt", "report.html", "manifest.json"}
    if set(files) != expected or not _same(_json(files, "evaluators.json"), evaluator_hashes()):
        raise io.EvidenceError("native-check packet inventory or evaluator identity differs")
    selected = _policy(_json(files, "policy.json"))
    report, derived = _derived(files["inputs/RESULT.json"], files["inputs/ordinary-evidence.zip"], selected)
    if any(files[name] != raw for name, raw in derived.items()):
        raise io.EvidenceError("retained native-check reports disagree with recomputed raw evidence")
    bounded.recheck_snapshot(packet, files, directories)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard native-checks", description=TITLE + "; offline saved records only")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("capture", help="retain and recompute a saved owner result")
    p.add_argument("--owner-result", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--require", choices=CHECKS, action="append", default=None)
    p.add_argument("--json", action="store_true")
    p = commands.add_parser("verify", help="recompute the retained packet")
    p.add_argument("packet")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = (capture(args.owner_result, args.out, required=args.require)
                  if args.command == "capture" else verify(args.packet))
        print(io.json_bytes(report).decode() if args.json else render_text(report), end="")
        return 0 if report["release_decision"] == "PASS" else 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Native-check evidence is inconclusive: " + str(exc))
        return 2
