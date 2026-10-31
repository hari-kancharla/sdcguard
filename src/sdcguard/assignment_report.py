"""Portable inspection of the recorded allocator-to-model mapping relation.

Owner: Hari Krishna Kancharla.
This command reads completed local records. It starts no engine or request.
"""
from __future__ import annotations

import argparse
import html
from importlib.resources import files
import json
from pathlib import Path
import re

from . import evidence_io as io
from .lifetime import assignment_obligations as contract
from .lifetime import mapping_closure as closure

FORMAT = "sdcguard.assignment-packet.v1"
REPORT = "sdcguard.assignment-packet-report.v1"
NATIVE_FORMAT = "sdcguard.assignment-native-packet.v1"
NATIVE_REPORT = "sdcguard.assignment-native-packet-report.v1"
MAX_INPUT_FILES = 64
MAX_INPUT_BYTES = contract.MAX_TOTAL_BYTES
MAX_RECORD_BYTES = contract.MAX_RECORD_BYTES
MAX_RECORDS = contract.MAX_RECORDS
_SESSION = re.compile(r"[0-9a-f]{32}\Z")
LIMITS = [
    "The verdict covers only allocator-to-model-entry mapping consistency in the recorded interval.",
    "Matching maps can share an erroneous assignment. They do not prove ownership or content provenance.",
    "Missing assignments and zero references do not prove a block was unowned.",
    "Model-entry metadata does not observe physical GPU loads or forward-device completion.",
    "Recorded call edges and local hashes do not authenticate a runtime or a remote deployment.",
    "Raw-format inspection does not check native recorder closure. GPU validation remains separate.",
    "This command starts no engine, request, device work or network operation.",
]


def evaluator_hashes(native=False):
    paths = {"assignment_report.py": Path(__file__),
             "lifetime/assignment_obligations.py": Path(contract.__file__),
             "model_reference.py": Path(__file__).parent / "model_reference.py",
             "evidence_io.py": Path(io.__file__),
             "config_inputs.py": Path(io.config_inputs.__file__)}
    if native:
        paths["lifetime/mapping_closure.py"] = Path(closure.__file__)
    result = {name: io.digest(path.read_bytes()) for name, path in paths.items()}
    result["case_data/assignment-pins.json"] = io.digest(
        files("sdcguard").joinpath("case_data/assignment-pins.json").read_bytes())
    result["case_data/61229/MODEL_PIN.json"] = io.digest(
        files("sdcguard").joinpath("case_data/61229/MODEL_PIN.json").read_bytes())
    return result


def parse_records(raw):
    if type(raw) is not bytes or len(raw) > MAX_INPUT_BYTES:
        raise io.EvidenceError("assignment input exceeds byte limit")
    if not raw or not raw.endswith(b"\n"):
        raise io.EvidenceError("empty or unfinished JSONL stream")
    lines = raw.splitlines()
    if len(lines) > MAX_RECORDS:
        raise io.EvidenceError("assignment record count exceeds limit")
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip() or len(line) > MAX_RECORD_BYTES:
            raise io.EvidenceError("empty or oversized assignment record at line " + str(number))
        records.append(io.parse_json(line, max_nodes=contract.MAX_RECORD_NODES))
    return records


def _unknown(reason):
    result = contract._base()
    result["missing"] = [reason]
    return result


def _check_inputs(payload):
    if type(payload) is not dict or not payload or len(payload) > MAX_INPUT_FILES:
        raise io.EvidenceError("assignment input file count is empty or exceeds limit")
    if any(type(raw) is not bytes for raw in payload.values()):
        raise io.EvidenceError("assignment inputs must be captured bytes")
    if sum(map(len, payload.values())) > MAX_INPUT_BYTES:
        raise io.EvidenceError("assignment input total exceeds byte limit")
    for name in payload:
        io.relative(name)
        if not name.endswith(".jsonl"):
            raise io.EvidenceError("assignment input must contain only JSONL files")


def capture_inputs(root):
    names = io.names(root)
    if not names or len(names) > MAX_INPUT_FILES or any(not n.endswith(".jsonl") for n in names):
        raise io.EvidenceError("assignment input requires a bounded set of JSONL files")
    payload = {}
    remaining = MAX_INPUT_BYTES
    for name in names:
        raw = io.config_inputs._read_regular(Path(root), io.relative(name), remaining)
        payload[name] = raw
        remaining -= len(raw)
    if io.names(root) != names:
        raise io.EvidenceError("assignment input inventory changed during capture")
    for name, raw in payload.items():
        if io.config_inputs._read_regular(Path(root), io.relative(name), len(raw)) != raw:
            raise io.EvidenceError("assignment input changed during capture")
    return payload


def inspect_payload(payload):
    """Evaluate sessions separately. Never join pools or identities across files."""
    _check_inputs(payload)
    sessions = []
    seen = set()
    duplicates = set()
    for name, raw in sorted(payload.items()):
        session = None
        identity = None
        # A broken tail cannot hide a duplicate, valid first header. Keep
        # identity extraction separate from full-session acceptance.
        try:
            first = raw.split(b"\n", 1)[0]
            header = (io.parse_json(first, max_nodes=contract.MAX_RECORD_NODES)
                      if 0 < len(first) <= MAX_RECORD_BYTES else None)
            if (type(header) is dict and header.get("kind") == "header"
                    and type(header.get("session")) is str
                    and _SESSION.fullmatch(header["session"]) is not None):
                session = header["session"]
                if session in seen:
                    duplicates.add(session)
                seen.add(session)
                # These are recorded fields, not authenticated stack identity.
                identity = {key: header.get(key) for key in
                            ("adapter", "source_sha256", "profile", "dependencies")}
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            pass
        try:
            assessment = contract.evaluate(parse_records(raw))
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            assessment = _unknown(type(exc).__name__ + ": " + str(exc))
        sessions.append({"file": name, "sha256": io.digest(raw), "session": session,
                         "recorded_identity": identity, "assessment": assessment})
    for entry in sessions:
        if entry["session"] in duplicates:
            entry["assessment"] = _unknown("session identity appears in more than one file")
    verdicts = [entry["assessment"]["verdict"] for entry in sessions]
    verdict = ("violated" if "violated" in verdicts else
               "satisfied" if verdicts and all(v == "satisfied" for v in verdicts) else "unknown")
    return {"schema": REPORT, "property": contract.PROPERTY, "verdict": verdict,
            "scope_verdict": {"satisfied": "fine", "violated": "not fine", "unknown": "inconclusive"}[verdict],
            "ownership_verdict": "unknown", "physical_read_verdict": "unknown",
            "content_provenance": "unknown", "runtime_isolation": "not established",
            "authenticity": "not established", "sessions": sessions,
            "recording_bookkeeping": "not checked", "client_attribution": "unknown",
            "input_files": io.file_receipts(payload), "limits": list(LIMITS)}


def inspect_native_payload(payload, receipts):
    """Validate native close accounting before judging the recorded relation."""
    captured = closure.validate_captured(payload, receipts)
    report = inspect_payload(captured["payload"])
    report.update(schema=NATIVE_REPORT, recording_bookkeeping=captured["bookkeeping"],
                  closure_receipts=io.file_receipts(captured["receipts"]))
    report["limits"] = [limit for limit in LIMITS if not limit.startswith("Raw-format inspection")]
    report["limits"].append(
        "Native journal and settled-close receipts are locally consistent. This is not runtime authentication or GPU validation.")
    return report


def render_report(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for entry in report["sessions"]:
        result = entry["assessment"]
        rows.append("<tr><td>" + esc(entry["file"]) + "</td><td>" + esc(result["verdict"]) +
                    "</td><td>" + esc(result["summary"]["compared_layer_positions"]) + "</td></tr>")
    details = esc(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            "<title>Allocator-to-model mapping</title><style>body{max-width:64rem;margin:3rem auto;"
            "padding:0 1rem;font:16px/1.5 system-ui}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #bbb;padding:.5rem;text-align:left}pre{white-space:pre-wrap;"
            "overflow-wrap:anywhere}h1{line-height:1.2}</style><h1>Allocator-to-model mapping</h1>"
            "<p>Named mapping rule: <strong>" + esc(report["scope_verdict"]) + "</strong>.</p>"
            "<p>Ownership, content provenance and physical reads: unknown. Runtime isolation: not established.</p>"
            "<table><thead><tr><th>Recording</th><th>Mapping verdict</th><th>Compared layer positions</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table><ul>" +
            "".join("<li>" + esc(limit) + "</li>" for limit in report["limits"]) +
            "</ul><details><summary>Recorded identity and recomputed evidence</summary><pre>" + details +
            "</pre></details></html>\n").encode("utf-8")


def _status(report):
    result = {key: report[key] for key in
              ("property", "verdict", "scope_verdict", "ownership_verdict", "physical_read_verdict",
               "content_provenance", "runtime_isolation", "authenticity")}
    result.update(recording_bookkeeping=report.get("recording_bookkeeping", "not established"),
                  client_attribution="unknown")
    return result


def create_packet(input_directory, output, *, native=False):
    source = Path(input_directory).absolute()
    target = Path(output).absolute()
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise io.EvidenceError("output and input trees must be separate")
    captured = closure.capture_closed(source) if native else None
    payload = captured["payload"] if native else capture_inputs(source)
    receipts = captured["receipts"] if native else {}
    report = inspect_native_payload(payload, receipts) if native else inspect_payload(payload)
    prepared = {"inputs/" + name: raw for name, raw in payload.items()}
    prepared.update({"closures/" + name: raw for name, raw in receipts.items()})
    prepared.update({"report.json": io.json_bytes(report), "index.html": render_report(report),
                     "metadata/evaluators.json": io.json_bytes(evaluator_hashes(native))})
    prepared["manifest.json"] = io.json_bytes({"schema": NATIVE_FORMAT if native else FORMAT,
                                              "files": io.file_receipts(prepared)})
    if (len(prepared) > io.MAX_FILES or sum(map(len, prepared.values())) > io.MAX_TOTAL_BYTES
            or any(len(raw) > io.MAX_FILE_BYTES for raw in prepared.values())):
        raise io.EvidenceError("assignment output exceeds portable packet bounds")
    if ((native and closure.capture_closed(source) != captured)
            or (not native and capture_inputs(source) != payload)):
        raise io.EvidenceError("input changed during assignment capture")
    root = io.create_directory(target)
    io.write_snapshot(root, prepared)
    return {"directory": str(root), **_status(report)}


def verify_packet(path):
    try:
        captured = io.snapshot(Path(path).absolute())
        manifest = io.parse_json(captured["manifest.json"])
        if (type(manifest) is not dict or set(manifest) != {"schema", "files"}
                or manifest["schema"] not in (FORMAT, NATIVE_FORMAT)):
            raise io.EvidenceError("unsupported assignment packet manifest")
        members = {name: raw for name, raw in captured.items() if name != "manifest.json"}
        if io.json_bytes(manifest["files"]) != io.json_bytes(io.file_receipts(members)):
            raise io.EvidenceError("assignment inventory or bytes differ from manifest")
        payload = {name[len("inputs/"):]: raw for name, raw in members.items() if name.startswith("inputs/")}
        native = manifest["schema"] == NATIVE_FORMAT
        receipts = {name[len("closures/"):]: raw for name, raw in members.items() if name.startswith("closures/")}
        if set(members) != {"report.json", "index.html", "metadata/evaluators.json"} | {
                "inputs/" + name for name in payload} | ({"closures/" + name for name in receipts} if native else set()):
            raise io.EvidenceError("unexpected assignment packet members")
        if io.json_bytes(io.parse_json(members["metadata/evaluators.json"])) != io.json_bytes(evaluator_hashes(native)):
            raise io.EvidenceError("packet evaluator differs from the installed implementation")
        report = inspect_native_payload(payload, receipts) if native else inspect_payload(payload)
        if (io.json_bytes(io.parse_json(members["report.json"])) != io.json_bytes(report)
                or members["index.html"] != render_report(report)):
            raise io.EvidenceError("assignment report does not match its retained inputs")
        return {"integrity": "verified", **_status(report), "report": report}
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return {"integrity": "invalid", **_status(contract._base()),
                "reason": type(exc).__name__ + ": " + str(exc)}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard assignment")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="capture and evaluate completed local mapping records")
    inspect.add_argument("input_directory")
    inspect.add_argument("--out", required=True)
    capture = sub.add_parser("capture", help="capture native journals and require settled-close receipts")
    capture.add_argument("input_directory")
    capture.add_argument("--out", required=True)
    verify = sub.add_parser("verify", help="recompute a retained mapping packet offline")
    verify.add_argument("packet")
    args = parser.parse_args(argv)
    try:
        result = (create_packet(args.input_directory, args.out, native=args.command == "capture")
                  if args.command in ("inspect", "capture")
                  else verify_packet(args.packet))
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result = {"integrity": "invalid", **_status(contract._base()),
                  "reason": type(exc).__name__ + ": " + str(exc)}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 2 if result.get("integrity") == "invalid" else {
        "satisfied": 0, "violated": 1, "unknown": 2}[result["verdict"]]
