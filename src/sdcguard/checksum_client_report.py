"""Offline client-to-checksum evidence packets.

Owner: Hari Krishna Kancharla.
"""
from __future__ import annotations

import argparse
import html
from importlib.resources import files as resources
import json
from pathlib import Path

from . import checksum_client as client
from . import evidence_envelope as envelope, evidence_io as io
from . import evidence_native_io as bounded

SCHEMA = "sdcguard.checksum-client.packet.v1"
PROFILE = "vllm-62211-client-checksum-join.v1"
TITLE = "Client requests and native checksum coverage"
INPUTS = ("REQUEST_PLAN.json", "CLIENT_INVENTORY.json", "CAPTURES.json")


def _same(a, b):
    return io.json_bytes(a) == io.json_bytes(b)


def evaluator_hashes():
    return {**client.evaluator_hashes(),
            "checksum_client_report.py": io.digest(resources("sdcguard").joinpath("checksum_client_report.py").read_bytes())}


def render_text(report):
    lines = [TITLE, "Release: HOLD", "Runtime qualified: false",
             "Counts cover planned client requests, including warmups and missing receipts.",
             "Summary:"]
    for name, count in report["summary"].items():
        lines.append("  " + str(name) + ": " + str(count))
    inventory = report.get("client_inventory", {})
    lines.append("Client inventory: " + str(inventory.get("status", "unknown")))
    lines.extend(["", "Per request:"])
    for row in report["requests"]:
        lines.append(str(row["arm_id"]) + "/" + str(row["slot"]) + " | " +
                     str(row.get("phase_id")) + " | " + str(row.get("kind")) +
                     " | client=" + str(row.get("client_request_id")) +
                     " | native=" + str(row.get("native_result")) +
                     " | coverage=" + str(row.get("coverage")) +
                     " | check=" + str(row.get("check_status")))
        lines.append("  client receipt=" + str(row.get("client_status")) +
                     "; ID binding=" + str(row.get("binding_status")) +
                     "; internal ID=" + str(row.get("internal_request_id")))
        for reason in row.get("reasons", []):
            lines.append("  " + str(reason))
            if reason in ("partial_checksum_span", "empty_replica_inputs") and row.get("native_result") == "passed":
                lines.append("  Engine pass, not covered: " + reason)
    for row in report.get("unexpected_checksum_calls", []):
        lines.append("Unexpected native call: " + json.dumps(row, sort_keys=True, allow_nan=False))
    lines.extend(["", "Hold reasons:"])
    lines.extend("* " + str(reason) for reason in report["hold_reasons"])
    lines.extend(["", "Limits:"])
    lines.extend("* " + str(reason) for reason in report.get("limits", []))
    lines.append("* Local hashes establish consistency, not trusted execution or recorder authenticity.")
    lines.append("* Historical lease observations cannot qualify this checksum source revision.")
    return "\n".join(lines) + "\n"


def _derived(raws):
    report = client.evaluate(*raws)
    if report.get("release_decision") != "HOLD" or report.get("runtime_qualified") is not False:
        raise io.EvidenceError("unsupported client checksum release or runtime claim")
    text = render_text(report)
    markup = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>" + TITLE + "</title><style>body{font:16px system-ui;margin:2rem;max-width:95rem}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5}</style><h1>" + TITLE +
        "</h1><pre>" + html.escape(text) + "</pre></html>\n")
    return report, {"report.json": io.json_bytes(report), "report.txt": text.encode(), "report.html": markup.encode()}


def _read_inputs(paths):
    return tuple(io.read_file(Path(p).absolute().parent, Path(p).name) for p in paths)


def inspect(plan, inventory, captures):
    paths = (plan, inventory, captures)
    raws = _read_inputs(paths)
    report, _ = _derived(raws)
    if raws != _read_inputs(paths):
        raise io.EvidenceError("client checksum inputs changed during inspection")
    return report


def capture(plan, inventory, captures, output):
    paths = tuple(Path(p).absolute() for p in (plan, inventory, captures))
    envelope.reject_overlap(output, paths)
    raws = _read_inputs(paths)
    report, derived = _derived(raws)
    if raws != _read_inputs(paths):
        raise io.EvidenceError("client checksum inputs changed during capture")
    saved = {"inputs/" + name: raw for name, raw in zip(INPUTS, raws)}
    saved.update({"source-profile.json": io.json_bytes(client.source_profile()),
                  "evaluators.json": io.json_bytes(evaluator_hashes()), **derived})
    envelope.complete(output, saved, envelope.directories_for(saved), schema=SCHEMA, profile=PROFILE)
    return report


def verify(packet):
    files, directories = bounded.capture_snapshot(packet)
    if envelope.check_manifest(files, directories, schema=SCHEMA) != PROFILE:
        raise io.EvidenceError("unsupported client checksum packet")
    expected = {*("inputs/" + n for n in INPUTS), "source-profile.json", "evaluators.json",
                "report.json", "report.txt", "report.html", "manifest.json"}
    if (set(files) != expected or not _same(io.parse_json(files["source-profile.json"]), client.source_profile())
            or not _same(io.parse_json(files["evaluators.json"]), evaluator_hashes())):
        raise io.EvidenceError("client checksum packet inventory or evaluator identity differs")
    report, derived = _derived(tuple(files["inputs/" + name] for name in INPUTS))
    if any(files[name] != raw for name, raw in derived.items()):
        raise io.EvidenceError("retained client checksum reports disagree with recomputed raw evidence")
    bounded.recheck_snapshot(packet, files, directories)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard checksum-client",
        description="Reconcile planned ordinary requests with captured native checksum calls offline")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "capture"):
        child = commands.add_parser(name)
        child.add_argument("--plan", required=True)
        child.add_argument("--inventory", required=True)
        child.add_argument("--captures", required=True)
        if name == "capture":
            child.add_argument("--out", required=True)
        child.add_argument("--json", action="store_true")
    child = commands.add_parser("verify")
    child.add_argument("packet")
    child.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = verify(args.packet)
        elif args.command == "capture":
            report = capture(args.plan, args.inventory, args.captures, args.out)
        else:
            report = inspect(args.plan, args.inventory, args.captures)
        print(io.json_bytes(report).decode() if args.json else render_text(report), end="")
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print("Client checksum evidence is inconclusive: " + str(exc))
        return 2
