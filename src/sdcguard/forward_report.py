"""Portable, offline inspection of recorded forward inputs.

Owner: Hari Krishna Kancharla.
No engine, request, device work or ownership inference is performed here.
"""
from __future__ import annotations

import argparse
import html
from importlib.resources import files
import json
from pathlib import Path

from . import evidence_io as io
from .lifetime import forward_inputs

FORMAT = "sdcguard.forward-inputs.v1"
REPORT = "sdcguard.forward-inputs-report.v1"
MAX_INPUT_FILES = 64
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_RECORDS = 8192
LIMITS = [
    "Recorded forward metadata describes an attention input plan, not observed physical reads.",
    "Request ownership and allocation/lease joins are not evaluated by this format.",
    "A valid decode block mapping does not prove the provenance of transferred source bytes.",
    "Hashes check local consistency, not remote authenticity or live deployment identity.",
    "Snapshot-copy completion is separate from forward-device completion.",
    "This command starts no engine, serving request, device work or network operation.",
]


def evaluator_hashes():
    paths = {"forward_report.py": Path(__file__),
             "lifetime/forward_inputs.py": Path(forward_inputs.__file__),
             "model_reference.py": Path(__file__).parent / "model_reference.py",
             "evidence_io.py": Path(io.__file__),
             "config_inputs.py": Path(io.config_inputs.__file__)}
    result = {name: io.digest(path.read_bytes()) for name, path in paths.items()}
    result["case_data/forward-pins.json"] = io.digest(
        files("sdcguard").joinpath("case_data/forward-pins.json").read_bytes())
    result["case_data/61229/MODEL_PIN.json"] = io.digest(
        files("sdcguard").joinpath("case_data/61229/MODEL_PIN.json").read_bytes())
    return result


def parse_records(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_INPUT_BYTES:
        raise io.EvidenceError("forward input exceeds byte limit")
    if not raw or not raw.endswith(b"\n"):
        raise io.EvidenceError("empty or unfinished JSONL stream")
    lines = raw.splitlines()
    if len(lines) > MAX_RECORDS:
        raise io.EvidenceError("forward record count exceeds limit")
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip() or len(line) > MAX_RECORD_BYTES:
            raise io.EvidenceError("empty or oversized forward record at line " + str(number))
        records.append(io.parse_json(line, max_nodes=forward_inputs.MAX_RECORD_NODES))
    return records


def _unknown(reason):
    return {"observation_status": "unknown", "evidence_basis": "unavailable",
            "evidence_depth": "approximate", "physical_read_verdict": "unknown",
            "ownership_verdict": "unknown", "reasons": [reason]}


def _check_inputs(payload):
    if not payload or len(payload) > MAX_INPUT_FILES:
        raise io.EvidenceError("forward input file count is empty or exceeds limit")
    if sum(len(raw) for raw in payload.values()) > MAX_INPUT_BYTES:
        raise io.EvidenceError("forward input total exceeds byte limit")
    for name, raw in payload.items():
        io.relative(name)
        if not name.endswith(".jsonl") or len(raw) > MAX_INPUT_BYTES:
            raise io.EvidenceError("forward input must contain only bounded JSONL files")


def capture_inputs(root):
    """Enforce the forward budget before retaining an entire generic packet."""
    names = io.names(root)
    if not names or len(names) > MAX_INPUT_FILES or any(not name.endswith(".jsonl") for name in names):
        raise io.EvidenceError("forward input requires a bounded set of JSONL files")
    payload = {}
    remaining = MAX_INPUT_BYTES
    for name in names:
        raw = io.config_inputs._read_regular(Path(root), io.relative(name), remaining)
        payload[name] = raw
        remaining -= len(raw)
    if io.names(root) != names:
        raise io.EvidenceError("forward input inventory changed during capture")
    for name, raw in payload.items():
        if io.config_inputs._read_regular(Path(root), io.relative(name), len(raw)) != raw:
            raise io.EvidenceError("forward input changed during capture")
    return payload


def inspect_payload(payload):
    """Recompute observations from complete retained bytes, never supplied verdicts."""
    _check_inputs(payload)
    sessions = []
    seen = set()
    duplicates = set()
    for name, raw in sorted(payload.items()):
        try:
            records = parse_records(raw)
            result = forward_inputs.audit_records(records)
            session = records[0].get("session") if isinstance(records[0], dict) else None
            if isinstance(session, str):
                if session in seen:
                    duplicates.add(session)
                seen.add(session)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            session = None
            result = _unknown(type(exc).__name__ + ": " + str(exc))
        sessions.append({"file": name, "sha256": io.digest(raw), "session": session,
                         "assessment": result})
    if duplicates:
        for entry in sessions:
            if entry["session"] in duplicates:
                entry["assessment"] = _unknown("session identity appears in more than one file")
    observed = bool(sessions) and all(
        entry["assessment"].get("observation_status") == "observed" for entry in sessions)
    return {"schema": REPORT, "observation_status": "observed" if observed else "unknown",
            "scope_verdict": "inconclusive", "evidence_depth": "approximate",
            "physical_read_verdict": "unknown", "ownership_verdict": "unknown",
            "authenticity": "not established", "sessions": sessions,
            "input_files": io.file_receipts(payload), "limits": list(LIMITS)}


def render_report(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for entry in report["sessions"]:
        result = entry["assessment"]
        rows.append("<tr><td>" + esc(entry["file"]) + "</td><td>" +
                    esc(result.get("observation_status", "unknown")) + "</td><td>" +
                    esc(result.get("evidence_basis", "unavailable")) + "</td></tr>")
    details = esc(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Forward-input observations</title><style>body{max-width:64rem;margin:3rem auto;"
            "padding:0 1rem;font:16px/1.5 system-ui}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #bbb;padding:.5rem;text-align:left}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere}h1{line-height:1.2}</style><h1>Forward-input observations</h1>"
            "<p>Observation status: <strong>" + esc(report["observation_status"]) + "</strong>.</p>"
            "<p>Physical reads: unknown. Ownership: unknown. Isolation verdict: inconclusive.</p>"
            "<table><thead><tr><th>Recording</th><th>Observation</th><th>Basis</th></tr></thead><tbody>" +
            "".join(rows) + "</tbody></table><ul>" +
            "".join("<li>" + esc(limit) + "</li>" for limit in LIMITS) +
            "</ul><details><summary>Recomputed evidence</summary><pre>" + details +
            "</pre></details></html>\n").encode("utf-8")


def create_packet(input_directory, output):
    source = Path(input_directory).absolute()
    target = Path(output).absolute()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise io.EvidenceError("output and input trees must be separate")
    payload = capture_inputs(source)
    report = inspect_payload(payload)
    prepared = {"inputs/" + name: raw for name, raw in payload.items()}
    prepared.update({"report.json": io.json_bytes(report), "index.html": render_report(report),
                     "metadata/evaluators.json": io.json_bytes(evaluator_hashes())})
    manifest = {"schema": FORMAT, "files": io.file_receipts(prepared)}
    prepared["manifest.json"] = io.json_bytes(manifest)
    if (len(prepared) > io.MAX_FILES or sum(map(len, prepared.values())) > io.MAX_TOTAL_BYTES
            or any(len(raw) > io.MAX_FILE_BYTES for raw in prepared.values())):
        raise io.EvidenceError("forward output exceeds portable packet bounds")
    # Read again before any output is created; a moving recording is not a snapshot.
    if capture_inputs(source) != payload:
        raise io.EvidenceError("input changed during forward capture")
    root = io.create_directory(target)
    io.write_snapshot(root, prepared)
    return {"directory": str(root), "observation_status": report["observation_status"],
            "scope_verdict": "inconclusive", "physical_read_verdict": "unknown",
            "ownership_verdict": "unknown"}


def verify_packet(path):
    try:
        captured = io.snapshot(Path(path).absolute())
        manifest = io.parse_json(captured["manifest.json"])
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "files"}
                or manifest["schema"] != FORMAT):
            raise io.EvidenceError("unsupported forward packet manifest")
        members = {name: raw for name, raw in captured.items() if name != "manifest.json"}
        if io.json_bytes(manifest["files"]) != io.json_bytes(io.file_receipts(members)):
            raise io.EvidenceError("forward packet inventory or bytes differ from manifest")
        payload = {name[len("inputs/"):]: raw for name, raw in members.items() if name.startswith("inputs/")}
        expected_names = {"report.json", "index.html", "metadata/evaluators.json"} | {
            "inputs/" + name for name in payload}
        if set(members) != expected_names:
            raise io.EvidenceError("unexpected forward packet members")
        if io.parse_json(members["metadata/evaluators.json"]) != evaluator_hashes():
            raise io.EvidenceError("packet evaluator differs from the installed implementation")
        report = inspect_payload(payload)
        if (io.json_bytes(io.parse_json(members["report.json"])) != io.json_bytes(report)
                or members["index.html"] != render_report(report)):
            raise io.EvidenceError("forward report does not match its retained inputs")
        return {"integrity": "verified", "observation_status": report["observation_status"],
                "scope_verdict": "inconclusive", "physical_read_verdict": "unknown",
                "ownership_verdict": "unknown", "authenticity": "not established", "report": report}
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return {"integrity": "invalid", "observation_status": "unknown", "scope_verdict": "inconclusive",
                "physical_read_verdict": "unknown", "ownership_verdict": "unknown",
                "authenticity": "not established", "reason": type(exc).__name__ + ": " + str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard forward-inputs")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="capture and inspect completed local JSONL observations")
    inspect.add_argument("input_directory")
    inspect.add_argument("--out", required=True)
    verify = sub.add_parser("verify", help="recompute a portable forward-input report offline")
    verify.add_argument("packet")
    args = parser.parse_args(argv)
    try:
        result = (create_packet(args.input_directory, args.out) if args.command == "inspect"
                  else verify_packet(args.packet))
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result = {"integrity": "invalid", "observation_status": "unknown", "scope_verdict": "inconclusive",
                  "reason": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result.get("observation_status") == "observed" and result.get("integrity") != "invalid" else 2
