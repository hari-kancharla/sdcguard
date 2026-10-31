"""Compare captured evidence under one installed preservation policy.

Owner: Hari Krishna Kancharla.
This command reads local packets and runs the existing ordinary CPU checks.
It never starts an inference server or submits serving traffic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from . import evidence_bundle as evidence
from . import evidence_io as io
from . import upgrade_policy as policy

SCHEMA = "sdcguard.upgrade-report.v1"
FORMAT = "sdcguard.upgrade-packet.v1"
_SIDES = ("baseline", "candidate")
_COLLECTIONS = ("matched_outcomes", "input_deltas", "native_candidate_violations",
                "policy_violations", "gaps", "newly_established", "remediations", "scope_notes")
_LIMITS = [
    "Only the named, selected evidence contracts are compared.",
    "A preserved contract does not establish runtime request isolation or an upgrade execution witness.",
    "Static facts describe supplied config and startup evidence; loaded model weights are not attested.",
    "Fresh CPU checks concern their own pinned source fragments, not the candidate deployment build.",
    "A copied recording is unchanged evidence, not proof that a candidate executed it.",
    "Unknown kernel identities remain outside these narrow scopes.",
    "Local hashes establish consistency, not authenticity.",
]


def evaluator_hashes():
    root = Path(__file__).parent
    return {name: io.digest((root / name).read_bytes())
            for name in ("upgrade.py", "upgrade_policy.py")}


def _policy_identity():
    descriptor = policy.policy_descriptor()
    raw = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return descriptor, io.digest(raw)


def _capture(root):
    root = Path(root).absolute()
    directories = io.names(root, directories=True)
    files = io.snapshot(root)
    if io.names(root, directories=True) != directories:
        raise io.EvidenceError("packet directory inventory changed during capture")
    return root, files, directories


def _snapshot_identity(files, directories):
    value = {"files": io.file_receipts(files), "directories": directories}
    return io.digest(io.json_bytes(value))


def _replace_private_paths(value, root):
    if isinstance(value, str):
        return value.replace(str(root), "captured-inputs")
    if isinstance(value, list):
        return [_replace_private_paths(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _replace_private_paths(item, root) for key, item in value.items()}
    return value


def _empty_comparison():
    descriptor, digest = _policy_identity()
    return dict(policy=descriptor, policy_sha256=digest,
                runtime_upgrade_witness="not established", **{key: [] for key in _COLLECTIONS})


def _candidate_violations(report):
    return [{"finding": {key: deepcopy(row[key]) for key in
                ("rule_id", "semantic_revision", "evidence_kind", "scope", "resource_kind", "subject", "scenario")},
             "baseline_verdict": None, "candidate_verdict": "violated",
             "regression_attribution": "not established"}
            for row in report["findings"] if row["required"] and row["verdict"] == "violated"]


def _captured_verification(target):
    value = evidence.verify_bundle(target, rerun_components=False)
    if value["integrity"] == "verified":
        # A portable comparison must reproduce without treating a later CPU
        # availability change as changed historical bytes. Fresh execution is
        # mandatory in _current_gate, outside this saved-fact comparison.
        value["errors"] = [error for error in value["errors"]
            if error != "current trusted CPU component execution was not refreshed"]
        value["scope_verdict"] = value["report"]["status"]
    value["verification_scope"] = "captured facts only; current CPU refresh is separate"
    return value


def _evaluate(captures):
    """Deterministically compare saved facts, without a current gate approval."""
    verification = {}
    with TemporaryDirectory(prefix="sdcguard-upgrade-") as temporary:
        # Only this owned temporary directory may resolve an OS temp alias.
        work = Path(temporary).resolve()
        for side in _SIDES:
            files, directories = captures[side]
            target = io.create_directory(work / side)
            io.write_snapshot(target, files)
            for name in directories:
                io.make_directory(target, name)
            verification[side] = _replace_private_paths(_captured_verification(target), work)

    valid = {side: verification[side]["integrity"] == "verified" for side in _SIDES}
    if all(valid.values()):
        try:
            comparison = policy.compare(verification["baseline"]["report"], verification["candidate"]["report"])
        except Exception as exc:
            comparison = _empty_comparison()
            comparison["native_candidate_violations"] = _candidate_violations(verification["candidate"]["report"])
            comparison["gaps"].append({"code": "installed_policy_unavailable", "section": "comparison",
                "message": "The installed preservation policy could not complete; attribution is unknown.",
                "reason": type(exc).__name__ + ": " + str(exc)})
    else:
        comparison = _empty_comparison()
        if valid["candidate"]:
            comparison["native_candidate_violations"] = _candidate_violations(verification["candidate"]["report"])
    for side in _SIDES:
        verified = verification[side]
        if not valid[side]:
            comparison["gaps"].append({"code": "input_verification_failed", "section": side,
                "message": "The captured packet did not verify with the installed evaluator.",
                "errors": deepcopy(verified["errors"])})
        elif verified["errors"]:
            comparison["gaps"].append({"code": "current_verification_gap", "section": side,
                "message": "Captured-fact verification retained a proof gap.",
                "errors": deepcopy(verified["errors"])})
    candidate = verification["candidate"]
    known_failure = (bool(comparison["native_candidate_violations"] or comparison["policy_violations"])
                     or valid["candidate"] and candidate["scope_verdict"] == "not fine")
    if valid["candidate"] and candidate["scope_verdict"] == "inconclusive":
        comparison["gaps"].append({"code": "candidate_scope_inconclusive", "section": "candidate",
                                   "message": "Required candidate facts remain unknown."})
    verdict = "not fine" if known_failure else "inconclusive" if comparison["gaps"] else "fine"
    inputs = {}
    for side in _SIDES:
        files, directories = captures[side]
        inputs[side] = {"packet_sha256": _snapshot_identity(files, directories),
            "file_count": len(files), "verification": verification[side]}
    return {"schema": SCHEMA, "captured_scope_verdict": verdict,
        "current_gate": "run upgrade verify; this saved report does not grant approval",
        "comparison": "preserved" if verdict == "fine" else "violated" if verdict == "not fine" else "inconclusive",
        "runtime_upgrade_witness": "not established", "runtime_isolation": "not established",
        "authenticity": "not established", "inputs": inputs, "comparison_details": comparison,
        "limits": list(_LIMITS)}


def _current_gate(report):
    """Refresh every selected trusted CPU profile; never trust an old receipt."""
    execution, gaps = {}, []
    for side in _SIDES:
        verified = report["inputs"][side]["verification"]
        if verified["integrity"] != "verified":
            execution[side] = {"status": "not selected from an unverified packet"}
            continue
        profile = verified["report"]["profile"]
        if profile["components"] != "all":
            execution[side] = {"status": "not selected"}
            continue
        try:
            from .evidence_components import normalize
            current = evidence._collect_components()
            normalize(current)
            execution[side] = {"scope": "source_fragment_cpu", "status": current["status"],
                "basis": "fresh trusted ordinary component execution with CPU mocks",
                "historical_authenticity": "not established", "physical_write_safety": "unknown"}
            if current["status"] != "satisfied":
                gaps.append({"side": side, "reason": "Current trusted CPU component proof was incomplete."})
        except Exception as exc:
            execution[side] = {"scope": "source_fragment_cpu", "status": "unknown",
                "reason": type(exc).__name__ + ": " + str(exc), "physical_write_safety": "unknown"}
            gaps.append({"side": side, "reason": "Current trusted CPU component execution was unavailable."})
    captured = report["captured_scope_verdict"]
    verdict = "not fine" if captured == "not fine" else "inconclusive" if gaps or captured == "inconclusive" else "fine"
    return {"scope_verdict": verdict,
        "decision": "allowed within named scopes" if verdict == "fine" else "blocked",
        "current_execution": execution, "current_execution_gaps": gaps,
        "runtime_upgrade_witness": "not established", "authenticity": "not established"}


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    details = report["comparison_details"]
    rows = []
    for category in ("native_candidate_violations", "policy_violations", "gaps", "input_deltas",
                     "matched_outcomes", "newly_established", "remediations"):
        for item in details[category]:
            rows.append("<tr><td>" + esc(category) + "</td><td><pre>"
                        + esc(json.dumps(item, sort_keys=True, indent=2)) + "</pre></td></tr>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Scoped upgrade comparison</title><style>body{font:15px/1.4 system-ui;max-width:1100px;"
        "margin:24px auto;padding:0 16px;color:#15232c}table{border-collapse:collapse;width:100%}"
        "td,th{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #ccd4da}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Captured named contracts: "
        + esc(report["captured_scope_verdict"]) + "</h1><p>This saved report grants no current gate approval. "
        "Run <code>sdcguard upgrade verify COMPARISON</code> to refresh selected CPU checks "
        "and obtain the current bounded gate verdict. A runtime upgrade witness is not established.</p><p>Policy: "
        + esc(details["policy"]["name"]) + "</p><p>"
        "<a href='report.json'>Full comparison and verified inputs</a> | "
        "<a href='manifest.json'>File manifest</a></p><table><thead><tr><th>Category</th>"
        "<th>Finding</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table><h2>Limits</h2><ul>"
        + "".join("<li>" + esc(note) + "</li>" for note in _LIMITS) + "</ul></html>\n").encode()


def _input_captures(files, directories):
    captures = {}
    allowed = {"report.json", "index.html", "manifest.json", "metadata/policy.json",
               "metadata/upgrade-evaluators.json"}
    for side in _SIDES:
        prefix = "inputs/" + side + "/"
        members = {name[len(prefix):]: raw for name, raw in files.items() if name.startswith(prefix)}
        if "manifest.json" not in members:
            raise io.EvidenceError("comparison is missing a captured input packet")
        dirs = [name[len(prefix):] for name in directories if name.startswith(prefix)]
        captures[side] = (members, dirs)
        allowed.update(prefix + name for name in members)
    if set(files) != allowed:
        raise io.EvidenceError("comparison contains unexpected or missing files")
    expected_dirs = {"inputs", "metadata", "inputs/baseline", "inputs/candidate"}
    for side, (_, dirs) in captures.items():
        expected_dirs.update("inputs/" + side + "/" + name for name in dirs)
    if set(directories) != expected_dirs:
        raise io.EvidenceError("comparison contains unexpected or missing directories")
    return captures


def compare_packets(baseline, candidate, output):
    """Capture both inputs and retain a new portable comparison packet."""
    original = {"baseline": _capture(baseline), "candidate": _capture(candidate)}
    output = Path(output).absolute()
    for root, _, _ in original.values():
        if output == root or root in output.parents:
            raise io.EvidenceError("comparison output must be outside both input packets")
    captures = {side: (files, dirs) for side, (_, files, dirs) in original.items()}
    report = _evaluate(captures)
    current = _current_gate(report)
    saved = {"metadata/policy.json": io.json_bytes(policy.policy_descriptor()),
             "metadata/upgrade-evaluators.json": io.json_bytes(evaluator_hashes()),
             "report.json": io.json_bytes(report), "index.html": render(report)}
    directories = {"inputs", "metadata", "inputs/baseline", "inputs/candidate"}
    for side, (files, dirs) in captures.items():
        saved.update({"inputs/" + side + "/" + name: raw for name, raw in files.items()})
        directories.update("inputs/" + side + "/" + name for name in dirs)
    # The combined artifact uses the same aggregate bounds as one evidence packet.
    if len(saved) + 1 > io.MAX_FILES or sum(map(len, saved.values())) > io.MAX_TOTAL_BYTES:
        raise io.EvidenceError("combined comparison exceeds supported packet bounds")
    root = io.create_directory(output)
    try:
        io.write_snapshot(root, saved)
        for name in sorted(directories):
            io.make_directory(root, name)
        for source, files, dirs in original.values():
            if io.snapshot(source) != files or io.names(source, directories=True) != dirs:
                raise io.EvidenceError("an original packet changed during comparison")
        if io.snapshot(root) != saved:
            raise io.EvidenceError("comparison output changed before completion")
        manifest = {"schema": FORMAT, "complete": True, "errors": [],
                    "files": io.file_receipts(saved), "directories": io.names(root, directories=True)}
        if sum(map(len, saved.values())) + len(io.json_bytes(manifest)) > io.MAX_TOTAL_BYTES:
            raise io.EvidenceError("comparison including its manifest exceeds supported bounds")
        io.write_file(root, "manifest.json", io.json_bytes(manifest))
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        try:
            partial = io.snapshot(root)
            io.write_file(root, "manifest.json", io.json_bytes({"schema": FORMAT, "complete": False,
                "errors": [str(exc)], "files": io.file_receipts(partial),
                "directories": io.names(root, directories=True)}))
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        raise io.EvidenceError("comparison capture is incomplete: " + str(exc)) from exc
    return {"directory": str(root), "capture_complete": True, "report": report, **current}


def verify_comparison(directory):
    """Recompute a captured comparison and refresh both trusted CPU profiles."""
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "runtime_upgrade_witness": "not established", "authenticity": "not established"}
    try:
        root, files, directories = _capture(directory)
        manifest = io.parse_json(files["manifest.json"])
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "complete", "errors", "files", "directories"}
                or manifest["schema"] != FORMAT or type(manifest["complete"]) is not bool
                or not isinstance(manifest["errors"], list)
                or any(not isinstance(error, str) for error in manifest["errors"])):
            raise io.EvidenceError("unsupported comparison manifest")
        if not isinstance(manifest["files"], dict):
            raise io.EvidenceError("comparison file receipts must be an object")
        for name, receipt in manifest["files"].items():
            io.relative(name)
            if (not isinstance(receipt, dict) or set(receipt) != {"sha256", "bytes"}
                    or not isinstance(receipt["sha256"], str) or len(receipt["sha256"]) != 64
                    or type(receipt["bytes"]) is not int or receipt["bytes"] < 0):
                raise io.EvidenceError("comparison file receipt has invalid hash or count types")
        if manifest["files"] != io.file_receipts({name: raw for name, raw in files.items() if name != "manifest.json"}):
            raise io.EvidenceError("comparison membership, hashes or lengths disagree")
        if manifest["directories"] != directories:
            raise io.EvidenceError("comparison directory inventory disagrees")
        if not manifest["complete"]:
            result.update(integrity="incomplete", errors=manifest["errors"])
            return result
        if manifest["errors"]:
            raise io.EvidenceError("completed comparison retains capture errors")
        if files["metadata/policy.json"] != io.json_bytes(policy.policy_descriptor()):
            raise io.EvidenceError("comparison policy differs from the installed reviewed policy")
        if files["metadata/upgrade-evaluators.json"] != io.json_bytes(evaluator_hashes()):
            raise io.EvidenceError("comparison evaluator is unreviewed")
        report = _evaluate(_input_captures(files, directories))
        if files["report.json"] != io.json_bytes(report) or files["index.html"] != render(report):
            raise io.EvidenceError("comparison does not reproduce from captured inputs and current CPU checks")
        if io.snapshot(root) != files or io.names(root, directories=True) != directories:
            raise io.EvidenceError("comparison packet changed during verification")
        current = _current_gate(report)
        if io.snapshot(root) != files or io.names(root, directories=True) != directories:
            raise io.EvidenceError("comparison packet changed during current CPU refresh")
        result.update(integrity="verified", report=report, core_facts_recomputed=True, **current)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result["errors"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard upgrade", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare", help="verify two evidence packets and compare named contracts")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify", help="recompute a saved comparison offline")
    verify.add_argument("directory", type=Path)
    commands.add_parser("policy", help="show the fixed installed preservation policy")
    args = parser.parse_args(argv)
    try:
        if args.command == "policy":
            descriptor, digest = _policy_identity()
            print(json.dumps({"policy": descriptor, "policy_sha256": digest}, indent=2, sort_keys=True))
            return 0
        result = (compare_packets(args.baseline, args.candidate, args.out) if args.command == "compare"
                  else verify_comparison(args.directory))
        print(json.dumps(result, indent=2, sort_keys=True))
        if result.get("integrity", "verified") != "verified":
            return 2
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[result["scope_verdict"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [str(exc)]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
