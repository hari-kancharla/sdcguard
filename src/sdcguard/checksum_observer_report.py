"""Offline inspection of candidate checksum observations.

Owner: Hari Krishna Kancharla.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
from importlib.resources import files as resources
import json
from pathlib import Path

from . import evidence_envelope as envelope, evidence_io as io
from . import evidence_native_io as bounded
from . import kv_checksum_observer as observer

SCHEMA = "sdcguard.checksum-observer.packet.v1"
REPORT_SCHEMA = "sdcguard.checksum-observer-report.v1"
TITLE = "Native checksum calls and observed chunk scope"
MAX_RECORDS = 4096
MAX_UNITS = 65536
LIMITS = [
    "Counts describe recorded verification calls, not an independent client request inventory.",
    "Native success and complete chunk comparison scope are separate observations.",
    "Unknown compared and missing unit sets are null. A partial span may include native comparisons.",
    "This candidate profile has no accepted GPU validation or runtime overhead result.",
    "The checksum source profile is separate from the historical lease build.",
    "Client completeness, router forwarding, model-layer coverage, ownership and physical reads remain unknown.",
    "Hashes establish local consistency only. They do not authenticate a recorder or attest execution.",
    "This command cannot qualify a release, including when all observed calls have complete chunk scope.",
]


def _same(a, b):
    return io.json_bytes(a) == io.json_bytes(b)


def evaluator_hashes():
    root = resources("sdcguard")
    names = ("checksum_observer_report.py", "kv_checksum_observer.py", "evidence_io.py",
             "evidence_envelope.py", "evidence_native_io.py", "evidence_json_budget.py",
             "config_inputs.py", "case_data/checksum-observer-pins.json", "checksum_profiles.py",
             "case_data/checksum-builds.json")
    return {name: io.digest(root.joinpath(*name.split("/")).read_bytes()) for name in names}


def _count(value):
    return type(value) is int and value >= 0


def _planned_scope_matches(plan, proof):
    captured = plan.get("load_plan")
    scalar_keys = ("parallel", "speculative_disabled", "native_tp_size", "num_workers",
                   "num_local_computed_tokens", "plan_sequence", "num_tokens",
                   "planned_group_count", "configured_group_count", "configured_group_ids")
    if (type(captured) is not dict or captured.get("source_supported") is not True
            or any(key not in captured or key not in proof or not _same(captured[key], proof[key])
                   for key in scalar_keys)):
        return False
    group_keys = ("group_id", "spec_type", "spec_source_supported", "block_size", "sliding_window",
                  "attention_chunk_size", "non_causal", "spec_fields_recorded", "start", "block_count")
    original, observed = captured.get("groups"), proof.get("groups")
    if type(original) is not list or type(observed) is not list or len(original) != len(observed):
        return False
    return all(type(a) is dict and type(b) is dict and all(key in a and key in b
        and _same(a[key], b[key]) for key in group_keys) for a, b in zip(original, observed))


def evaluate(raw, *, revision=None):
    """Recompute conditional chunk scope, retaining all native outcomes."""
    value = io.parse_json(raw)
    if type(value) is not dict or value.get("schema") != observer.SCHEMA:
        raise io.EvidenceError("unsupported checksum observer snapshot")
    records = value.get("records")
    if type(records) is not list or len(records) > MAX_RECORDS:
        raise io.EvidenceError("checksum snapshot records exceed the supported bound")
    issues = []
    if value.get("profile") != observer.PROFILE:
        issues.append("unsupported_source_profile")
    qualification = value.get("qualification")
    if qualification not in ("cpu_source_functions", "source_bound_candidate"):
        issues.append("source_qualification_unknown")
    if value.get("runtime_qualified") is not False or value.get("release_qualifying") is not False:
        issues.append("unsupported_runtime_or_release_claim")
    source = value.get("source")
    observed_revision = source.get("revision") if type(source) is dict else None
    selected = revision if revision is not None else observed_revision
    if not observer.profiles.supported(selected):
        issues.append("unsupported_source_revision")
        selected = observer.REVISION
    profile = observer.source_profile(selected)
    if (type(source) is not dict or source.get("supported") is not True
            or source.get("revision") != profile["revision"]
            or not _same(source.get("files"), profile["files"])):
        issues.append("declared_source_identity_unverified")
    if value.get("closed") is not True:
        issues.append("observer_not_closed")
    health = value.get("health")
    if (type(health) is not dict or health.get("complete") is not True
            or health.get("budget_exhausted") is not False
            or any(not _count(health.get(key)) or health[key] != 0
                   for key in ("errors", "dropped_records", "late_calls", "active_calls",
                               "inflight_position_units", "inflight_position_bytes"))):
        issues.append("recording_health_unverified")
    budgets = value.get("budgets")
    if (type(budgets) is not dict or any(not _count(budgets.get(key)) for key in
            ("max_records", "max_units", "max_bytes", "retained_records", "retained_units", "retained_bytes",
             "fixed_metadata_bytes", "reserved_record_bytes", "reserved_state_bytes"))
            or not 1 <= budgets["max_records"] <= MAX_RECORDS
            or not 1 <= budgets["max_units"] <= MAX_UNITS
            or not 65536 <= budgets["max_bytes"] <= 8 * 1024 * 1024):
        raise io.EvidenceError("unsupported checksum snapshot budgets")
    unit_limit = budgets["max_units"]
    if (budgets["retained_records"] != len(records) or len(records) > budgets["max_records"]
            or budgets["retained_units"] > unit_limit
            or budgets["fixed_metadata_bytes"] != observer.FIXED_METADATA_BYTES
            or budgets["reserved_record_bytes"] < budgets["retained_bytes"]
            or budgets["fixed_metadata_bytes"] + budgets["reserved_record_bytes"] +
                budgets["reserved_state_bytes"] > budgets["max_bytes"]):
        issues.append("snapshot_budget_counters_disagree")
    if value.get("closed") is True and (type(budgets.get("reserved_state_bytes")) is not int
            or budgets["reserved_state_bytes"] != 0):
        issues.append("observer_state_not_settled")
    if type(health) is dict and health.get("complete") is True:
        encoded_bytes = sum(len(json.dumps(r, sort_keys=True, allow_nan=False).encode()) for r in records)
        if encoded_bytes != budgets["retained_bytes"]:
            issues.append("snapshot_record_byte_count_disagrees")
    sequences = []
    plans = {}
    for record in records:
        if (type(record) is not dict or type(record.get("kind")) is not str
                or not _count(record.get("sequence")) or record["sequence"] == 0):
            issues.append("record_shape_unverified")
        else:
            sequences.append(record["sequence"])
            if record["kind"] not in ("recv_plan", "verification", "not_verified"):
                issues.append("unsupported_record_kind")
            if record["kind"] == "recv_plan" and "plan_sequence" in record:
                plan = record["plan_sequence"]
                if not _count(plan) or plan == 0 or plan >= record["sequence"]:
                    issues.append("receive_plan_sequence_unverified")
                elif plan in plans:
                    issues.append("ambiguous_receive_plan")
                else:
                    plans[plan] = record
    if len(set(sequences)) != len(sequences):
        issues.append("ambiguous_record_sequence")
    if sequences != sorted(sequences):
        issues.append("record_order_unverified")
    issues = list(dict.fromkeys(issues))
    rows = []
    total_units = 0
    joined_plans = set()
    for index, record in enumerate(records):
        if type(record) is not dict or record.get("kind") != "verification":
            continue
        try:
            derived = observer.classify_verification(record, max_units=unit_limit)
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise io.EvidenceError("unsupported verification metadata: " + str(exc)) from exc
        reasons = list(issues)
        request = record.get("request_id")
        if type(request) is not str or not request or len(request) > 1024:
            reasons.append("request_identity_unverified")
            request = None
        proof = record.get("proof")
        proof = proof if type(proof) is dict else {}
        plan_id = proof.get("plan_sequence")
        plan = plans.get(plan_id) if _count(plan_id) else None
        update = record.get("origin_update_sequence")
        sequence = record.get("sequence")
        if (plan is None or plan.get("request_id") != request
                or not _count(update) or not _count(sequence)
                or not plan["sequence"] < update < sequence):
            reasons.append("receive_plan_to_verification_join_unverified")
        elif plan_id in joined_plans:
            reasons.append("ambiguous_receive_plan_verification")
        else:
            joined_plans.add(plan_id)
            if not _planned_scope_matches(plan, proof):
                reasons.append("planned_chunk_scope_disagrees_with_verification")
        for name, expected in derived.items():
            if name in record and not _same(record[name], expected):
                reasons.append("derived_observation_disagrees:" + name)
        total_units += len(derived.get("expected_units") or []) + len(derived.get("compared_units") or [])
        if total_units > unit_limit:
            raise io.EvidenceError("aggregate checksum chunk scope exceeds its bound")
        row = {"record_index": index, "sequence": record.get("sequence"),
               "request_id": request, "native_result": derived["native_result"],
               "native_return": derived["native_return"],
               "conditional_scope": derived,
               "coverage": "unknown" if reasons else derived["coverage"],
               "reason": reasons[0] if reasons else derived["reason"],
               "recording_issues": list(dict.fromkeys(reasons))}
        row["engine_pass_not_covered_reason"] = next((name for name in
            (observer.CACHED_PREFIX_REASON, "partial_checksum_span", "empty_replica_inputs") if derived["native_result"] == "passed"
            and name in derived["reasons"]), None)
        rows.append(row)
    retained_units = 0
    unit_shapes_valid = True
    for record in records:
        if type(record) is not dict:
            continue
        for key in ("expected_units", "compared_units", "missing_units"):
            units = record.get(key)
            if units is None:
                continue
            if type(units) is not list or any(type(unit) is not str for unit in units):
                unit_shapes_valid = False
            else:
                retained_units += len(units)
    if not unit_shapes_valid or retained_units != budgets["retained_units"]:
        issues.append("snapshot_retained_units_disagree")
        for row in rows:
            row.update(coverage="unknown", reason="snapshot_retained_units_disagree")
            row["recording_issues"].append("snapshot_retained_units_disagree")
    unverified = []
    for index, record in enumerate(records):
        if type(record) is not dict:
            continue
        plan_id = record.get("plan_sequence")
        if (record.get("kind") == "not_verified" or
                record.get("kind") == "recv_plan" and (not _count(plan_id) or plan_id not in joined_plans)):
            scheduled = record.get("kind") == "recv_plan" and _count(plan_id)
            unverified.append({"record_index": index, "sequence": record.get("sequence"),
                "request_id": record.get("request_id"),
                "reason": "receive_plan_without_verification" if scheduled else record.get("reason", "native_verification_not_recorded"),
                "native_plan_reason": record.get("reason") if scheduled else None,
                "coverage": "unknown"})
    if not rows:
        issues.append("no_recorded_native_verification_calls")
    return {"schema": REPORT_SCHEMA, "title": TITLE, "owner": "Hari Krishna Kancharla",
            "input_sha256": io.digest(raw), "profile": value.get("profile"),
            "qualification": qualification, "source": source,
            "release_decision": "HOLD", "runtime_qualified": False,
            "client_inventory": "unknown", "health": health,
            "summary": {"recorded_events": len(records), "verification_calls": len(rows),
                        "native_results": dict(Counter(row["native_result"] for row in rows)),
                        "chunk_scope": dict(Counter(row["coverage"] for row in rows)),
                        "unverified_observations": len(unverified)},
            "hold_reasons": list(dict.fromkeys(issues + ["runtime_qualification_pending",
                    "independent_client_inventory_not_joined", "joint_release_profile_not_qualified"])),
            "verification_calls": rows, "unverified_observations": unverified, "limits": LIMITS}


def render_text(report):
    summary = report["summary"]
    lines = [TITLE, "Release: HOLD", "Qualification: " + str(report["qualification"]),
             "Recorded native verification calls: " + str(summary["verification_calls"]),
             "Independent client inventory: unknown",
             "Native outcomes: " + str(summary["native_results"]),
             "Conditional chunk scope: " + str(summary["chunk_scope"])]
    for row in report["verification_calls"]:
        lines.append(str(row["sequence"]) + " | " + (row["request_id"] or "request unknown") +
                     " | native=" + row["native_result"] + " | chunk scope=" + row["coverage"] +
                     " | " + row["reason"])
        if row["engine_pass_not_covered_reason"] is not None:
            lines.append("Engine pass, not covered: " + row["engine_pass_not_covered_reason"])
    for row in report["unverified_observations"]:
        request = row["request_id"] if type(row["request_id"]) is str else "request unknown"
        lines.append(str(row["sequence"]) + " | " + request + " | unverified | " + str(row["reason"]))
    lines.extend(["", "Hold reasons:", *["* " + r for r in report["hold_reasons"]],
                  "", "Limits:", *["* " + r for r in LIMITS]])
    return "\n".join(lines) + "\n"


def _derived(raw):
    report = evaluate(raw)
    text = render_text(report)
    markup = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>" + TITLE + "</title><style>body{font:16px system-ui;margin:2rem;max-width:90rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}</style><h1>" + TITLE +
        "</h1><pre>" + html.escape(text) + "</pre></html>\n")
    return report, {"report.json": io.json_bytes(report), "report.txt": text.encode(),
                    "report.html": markup.encode()}


def capture(snapshot, output):
    path = Path(snapshot).absolute()
    envelope.reject_overlap(output, [path])
    raw = io.read_file(path.parent, path.name)
    report, derived = _derived(raw)
    if raw != io.read_file(path.parent, path.name):
        raise io.EvidenceError("checksum snapshot changed during capture")
    selected = report["source"].get("revision") if type(report["source"]) is dict else None
    if not observer.profiles.supported(selected):
        selected = observer.REVISION
    saved = {"inputs/snapshot.json": raw, "source-profile.json": io.json_bytes(observer.source_profile(selected)),
             "evaluators.json": io.json_bytes(evaluator_hashes()), **derived}
    envelope.complete(output, saved, envelope.directories_for(saved), schema=SCHEMA, profile=observer.PROFILE)
    return report


def verify(packet):
    files, directories = bounded.capture_snapshot(packet)
    if envelope.check_manifest(files, directories, schema=SCHEMA) != observer.PROFILE:
        raise io.EvidenceError("unsupported checksum observation packet")
    expected = {"inputs/snapshot.json", "source-profile.json", "evaluators.json",
                "report.json", "report.txt", "report.html", "manifest.json"}
    selected = io.parse_json(files["source-profile.json"]).get("revision")
    if (set(files) != expected or not _same(io.parse_json(files["evaluators.json"]), evaluator_hashes())
            or not _same(io.parse_json(files["source-profile.json"]), observer.source_profile(selected))):
        raise io.EvidenceError("checksum packet inventory or evaluator identity differs")
    report, derived = _derived(files["inputs/snapshot.json"])
    recorded = report["source"].get("revision") if type(report["source"]) is dict else None
    expected_revision = recorded if observer.profiles.supported(recorded) else observer.REVISION
    if expected_revision != selected:
        raise io.EvidenceError("checksum packet mixes source revisions")
    if any(files[name] != raw for name, raw in derived.items()):
        raise io.EvidenceError("retained checksum reports disagree with recomputed observations")
    bounded.recheck_snapshot(packet, files, directories)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard checksum-observer",
                                     description="Offline inspection of candidate checksum snapshots")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inspect")
    p.add_argument("snapshot")
    p.add_argument("--json", action="store_true")
    p = commands.add_parser("capture")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", action="store_true")
    p = commands.add_parser("verify")
    p.add_argument("packet")
    p.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            report = capture(args.snapshot, args.out)
        elif args.command == "verify":
            report = verify(args.packet)
        else:
            path = Path(args.snapshot).absolute()
            report = evaluate(io.read_file(path.parent, path.name))
        print(io.json_bytes(report).decode() if args.json else render_text(report), end="")
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Checksum observation is inconclusive: " + str(exc))
        return 2
