"""Preserve retained native facts without approving a runtime upgrade.

Owner: Hari Krishna Kancharla.
Inputs are independently reverified under their installed packet contracts.
No engine, request or candidate execution is started by this module.
"""
from copy import deepcopy
import html
import json
from pathlib import Path

from . import evidence_bundle as legacy
from . import evidence_combined as combined
from . import evidence_envelope as envelope
from . import evidence_io as io
from . import evidence_json_budget as json_budget
from . import evidence_native_io as bounded
from . import upgrade as old_upgrade
from . import upgrade_policy as legacy_policy

FORMAT = "sdcguard.upgrade-packet.v2"
SCHEMA = "sdcguard.upgrade-report.v2"
SIDES = ("baseline", "candidate")
METADATA = {"metadata/native-policy.json", "metadata/legacy-policy.json",
            "metadata/upgrade-evaluators.json"}
BASE_FILES = METADATA | {"report.json", "index.html", "manifest.json"}
LIMITS = (
    "Captured-fact preservation is separate from the current upgrade gate.",
    "A native journal is retained evidence, never proof of candidate execution.",
    "Changed native captures have no immutable reusable workload or reviewed transition binding.",
    "A native candidate violation can block without attributing it to an upgrade.",
    "Existing policy comparisons concern only independently verified nested legacy reports.",
    "Current trusted CPU fragments do not execute or attest the candidate deployment.",
    "Ownership, read rights, content, physical loads and runtime isolation are not established.",
    "Local hashes establish consistency only; they provide no authenticity or remote attestation.",
)


def native_policy_descriptor():
    return {"name": "preserve-retained-native-facts.v1", "semantic_revision": 1,
        "evidence_contract": combined.FORMAT,
        "native_contract": combined.contract()["native"],
        "required_slots": "exact named sets; removal or downgrade is a gap",
        "same_raw_journals": "retained evidence only, not a candidate execution",
        "changed_raw_journals": "incomparable without workload and source-transition binding",
        "candidate_violation": "blocks independently of regression attribution",
        "current_required_native_gate": "inconclusive without an independent runtime witness",
        "outcome_order": "no ordering of recovered, refused or correctly_rejected",
        "limits": list(LIMITS)}


def evaluator_hashes():
    root = Path(__file__).parent
    names = set(combined.evaluator_hashes()) | set(old_upgrade.evaluator_hashes()) | {
        "upgrade_combined.py", "upgrade_entry.py"}
    return {name: io.digest((root / name).read_bytes()) for name in sorted(names)}


def _metadata():
    return {"metadata/native-policy.json": io.json_bytes(native_policy_descriptor()),
        "metadata/legacy-policy.json": io.json_bytes(legacy_policy.policy_descriptor()),
        "metadata/upgrade-evaluators.json": io.json_bytes(evaluator_hashes())}


def _schema(files):
    manifest = io.parse_json(files.get("manifest.json", b""))
    if type(manifest) is not dict:
        raise io.EvidenceError("input manifest must be an object")
    value = manifest.get("schema")
    return value if type(value) is str else None


def _portable(value, root):
    if type(value) is str:
        return value.replace(str(root), "captured-inputs")
    if type(value) is list:
        return [_portable(item, root) for item in value]
    if type(value) is dict:
        return {key: _portable(item, root) for key, item in value.items()}
    return value


def _verify_input(files, directories):
    """Only verified top-level reports supply facts; stored flags never do."""
    schema = None
    try:
        schema = _schema(files)
        if schema == combined.FORMAT:
            checked = combined.verify_snapshot(files, directories)
        elif schema == legacy.FORMAT:
            def evaluate(root):
                return _portable(old_upgrade._captured_verification(root), root)
            checked = envelope.evaluate_private(files, directories, evaluate)
        else:
            raise io.EvidenceError("unsupported input evidence packet contract")
        checked["verification_scope"] = "captured facts only; current CPU refresh is separate"
        return schema, checked
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return schema, {"integrity": "failed", "scope_verdict": "inconclusive",
            "errors": [type(exc).__name__ + ": " + str(exc)],
            "verification_scope": "no verified input report",
            "authenticity": "not established", "runtime_isolation": "not established"}


def _legacy_part(schema, checked):
    if checked["integrity"] != "verified":
        return None
    return checked["report"]["legacy"] if schema == combined.FORMAT else checked


def _requirements(schema, checked):
    if checked["integrity"] != "verified":
        return None
    if schema == combined.FORMAT:
        return deepcopy(checked["report"]["requirements"])
    return {"legacy": True, "native_mapping": []}


def _gap(gaps, code, section, message, **facts):
    gaps.append({"code": code, "section": section, "message": message, **deepcopy(facts)})


def _native_sections(schema, checked):
    if schema != combined.FORMAT or checked["integrity"] != "verified":
        return {}
    return {row["slot"]: row for row in checked["report"]["native_mapping"]}


def _native_violations(sections):
    return [{"slot": slot, "finding": deepcopy(row), "candidate_verdict": "violated",
             "regression_attribution": "not established"}
            for slot, section in sorted(sections.items())
            if section["integrity"] == "verified"
            for row in section["findings"] if row["required"] and row["verdict"] == "violated"]


def _streams(section):
    return sorted({row["journal_sha256"] for row in section["streams"]})


def _evaluate(captures, capture_errors=None):
    all_inputs = {side + "/" + name: raw for side in SIDES for name, raw in captures[side][0].items()}
    json_budget.count_members(all_inputs)
    verification, formats, requirements, old, natives = {}, {}, {}, {}, {}
    gaps, preservation = [], []
    for side in SIDES:
        notice = (capture_errors or {}).get(side)
        if notice is None:
            formats[side], verification[side] = _verify_input(*captures[side])
        else:
            formats[side] = None
            verification[side] = {"integrity": "unavailable", "scope_verdict": "inconclusive",
                "errors": [notice], "verification_scope": "No input bytes were retained; no facts are established.",
                "authenticity": "not established", "runtime_isolation": "not established"}
        checked = verification[side]
        requirements[side] = _requirements(formats[side], checked)
        old[side] = _legacy_part(formats[side], checked)
        natives[side] = _native_sections(formats[side], checked)
        if checked["integrity"] != "verified":
            _gap(gaps, "input_verification_failed", side,
                 "No approved report exists for this input envelope.", errors=checked["errors"])
        elif checked["scope_verdict"] == "inconclusive":
            _gap(gaps, "input_scope_inconclusive", side, "Required captured facts remain unknown.")

    if all(requirements[side] is not None for side in SIDES):
        left, right = (requirements[side] for side in SIDES)
        if left["legacy"] != right["legacy"]:
            _gap(gaps, "legacy_selection_changed", "legacy",
                 "Selected legacy scope changed; removal cannot satisfy the baseline contract.",
                 baseline=left["legacy"], candidate=right["legacy"])
        if left["native_mapping"] != right["native_mapping"]:
            _gap(gaps, "required_native_slots_changed", "native_mapping",
                 "Required native slots were added or removed; requirements are not interchangeable.",
                 removed=sorted(set(left["native_mapping"]) - set(right["native_mapping"])),
                 added=sorted(set(right["native_mapping"]) - set(left["native_mapping"])))

    legacy_comparison = None
    if all(old[side] is not None and old[side]["integrity"] == "verified" for side in SIDES):
        try:
            legacy_comparison = legacy_policy.compare(old["baseline"]["report"], old["candidate"]["report"])
            for item in legacy_comparison["gaps"]:
                _gap(gaps, "legacy_preservation_gap", "legacy", "Existing preservation policy retained a gap.",
                     detail=item)
        except Exception as exc:
            _gap(gaps, "legacy_policy_unavailable", "legacy", "Existing policy could not complete.",
                 reason=type(exc).__name__ + ": " + str(exc))
    else:
        for side in SIDES:
            if requirements[side] is not None and requirements[side]["legacy"]:
                if old[side] is None or old[side]["integrity"] != "verified":
                    _gap(gaps, "legacy_input_unverified", side, "Selected legacy input did not verify.")

    selected = sorted(set(natives["baseline"]) | set(natives["candidate"]))
    for slot in selected:
        before, after = natives["baseline"].get(slot), natives["candidate"].get(slot)
        record = {"slot": slot, "status": "inconclusive",
                  "runtime_upgrade_witness": "not established", "regression_attribution": "not established"}
        if before is None or after is None:
            record["reason"] = "A required native slot is not present in both verified envelopes."
            _gap(gaps, "native_slot_missing", slot, record["reason"])
        elif any(row["integrity"] != "verified" or row["scope_verdict"] == "inconclusive"
                 for row in (before, after)):
            record["reason"] = "Required native evidence is unavailable, invalid, conflicted or unknown."
            _gap(gaps, "native_slot_unverified", slot, record["reason"])
        else:
            left, right = _streams(before), _streams(after)
            record.update(baseline_raw_journals=left, candidate_raw_journals=right)
            if left and left == right:
                record.update(status="retained evidence", reason="The same raw journals were retained.",
                    candidate_execution="not established", baseline_scope_verdict=before["scope_verdict"],
                    candidate_scope_verdict=after["scope_verdict"])
            else:
                record["reason"] = "Changed captures lack a reusable workload and reviewed source-transition binding."
                _gap(gaps, "changed_native_capture_incomparable", slot, record["reason"])
        preservation.append(record)

    native_failures = _native_violations(natives["candidate"])
    candidate = verification["candidate"]
    candidate_negative = candidate["integrity"] == "verified" and candidate["scope_verdict"] == "not fine"
    policy_negative = bool(legacy_comparison and legacy_comparison["policy_violations"])
    verdict = "not fine" if candidate_negative or native_failures or policy_negative else "inconclusive" if gaps else "fine"
    inputs = {side: {"packet_format": formats[side],
                       "packet_sha256": envelope.identity(*captures[side]) if not (capture_errors or {}).get(side) else None,
                       "file_count": len(captures[side][0]), "verification": verification[side]}
              for side in SIDES}
    descriptor = native_policy_descriptor()
    return {"schema": SCHEMA, "captured_scope_verdict": verdict,
        "comparison": "preserved" if verdict == "fine" else "violated" if verdict == "not fine" else "inconclusive",
        "current_gate": "run upgrade verify; a saved report grants no current approval",
        "requirements": requirements, "inputs": inputs,
        "comparison_details": {"policy": descriptor, "policy_sha256": io.digest(io.json_bytes(descriptor)),
            "legacy_policy": legacy_policy.policy_descriptor(), "legacy_comparison": legacy_comparison,
            "native_preservation": preservation, "native_candidate_violations": native_failures, "gaps": gaps},
        "runtime_upgrade_witness": "not established", "runtime_isolation": "not established",
        "ownership_verdict": "unknown", "physical_read_verdict": "unknown",
        "authenticity": "not established", "limits": list(LIMITS)}


def _current_gate(report):
    execution, gaps = {}, []
    for side in SIDES:
        value = report["inputs"][side]
        old = _legacy_part(value["packet_format"], value["verification"])
        if old is None or old["integrity"] != "verified":
            execution[side] = {"status": "not selected from an unverified or absent legacy input"}
        elif old["report"]["profile"]["components"] != "all":
            execution[side] = {"status": "not selected"}
        else:
            wrapper = {"report": {"legacy": deepcopy(old)}, "errors": [], "scope_verdict": old["scope_verdict"]}
            try:
                combined.refresh_components(wrapper)
            except Exception as exc:
                wrapper["errors"].append("Current trusted CPU refresh failed: " + type(exc).__name__ + ": " + str(exc))
            execution[side] = wrapper.get("component_execution", {"status": "unknown", "scope": "source_fragment_cpu"})
            for reason in wrapper["errors"]:
                gaps.append({"side": side, "reason": reason})
    required = sorted({slot for value in report["requirements"].values() if value is not None
                       for slot in value["native_mapping"]})
    if required:
        gaps.append({"section": "native_mapping", "required_slots": required,
            "reason": "No immutable workload, reviewed source transition and independent candidate execution witness exists."})
    captured = report["captured_scope_verdict"]
    verdict = "not fine" if captured == "not fine" else "inconclusive" if gaps or captured == "inconclusive" else "fine"
    return {"scope_verdict": verdict, "decision": "blocked" if verdict != "fine" else "allowed within named scopes",
        "current_execution": execution, "current_execution_gaps": gaps,
        "runtime_upgrade_witness": "not established", "authenticity": "not established"}


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    details = report["comparison_details"]
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Retained evidence preservation</title><style>body{font:16px/1.5 system-ui;max-width:1000px;"
        "margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Captured named facts: " + esc(report["captured_scope_verdict"]) + "</h1>"
        "<p>This saved report grants no current approval. A required native upgrade execution gate remains "
        "inconclusive without an independent runtime witness, unless a verified candidate violation blocks it. "
        "Retained journals do not demonstrate candidate execution.</p><p><a href='report.json'>Full recomputed report</a> | "
        "<a href='manifest.json'>Retained inputs and hashes</a></p><h2>Native preservation</h2><pre>"
        + esc(json.dumps(details["native_preservation"], sort_keys=True, indent=2))
        + "</pre><h2>Independent candidate violations</h2><pre>"
        + esc(json.dumps(details["native_candidate_violations"], sort_keys=True, indent=2))
        + "</pre><h2>Missing evidence</h2><pre>" + esc(json.dumps(details["gaps"], sort_keys=True, indent=2))
        + "</pre><h2>Limits</h2><ul>" + "".join("<li>" + esc(note) + "</li>" for note in LIMITS)
        + "</ul></html>\n").encode()


def _capture_inputs(baseline, candidate, metadata):
    originals, captures, retained = {}, {}, {}
    errors = {}
    directories = ["inputs", "inputs/baseline", "inputs/candidate", "metadata"]
    for side, source in zip(SIDES, (baseline, candidate)):
        byte_allowance = io.MAX_TOTAL_BYTES - 3 * io.MAX_FILE_BYTES - sum(map(len, metadata.values())) - sum(map(len, retained.values()))
        file_allowance = io.MAX_FILES - len(BASE_FILES) - len(retained)
        directory_allowance = io.MAX_FILES - 1 - len(directories)
        try:
            if min(byte_allowance, file_allowance, directory_allowance) < 0:
                raise io.EvidenceError("upgrade input aggregate allowance is exhausted")
            files, dirs = bounded.capture_snapshot(source, max_files=file_allowance, max_bytes=byte_allowance,
                max_directories=directory_allowance, max_depth=bounded.MAX_DIRECTORY_DEPTH - 2)
            used_nodes = json_budget.count_members(retained) + json_budget.count_members(metadata)
            json_budget.count_members(files, max_nodes=io.MAX_JSON_NODES - used_nodes)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            captures[side] = ({}, [])
            errors[side] = (type(exc).__name__ + ": " + str(exc))[:4096]
            continue
        prefix = "inputs/" + side + "/"
        retained.update({prefix + name: raw for name, raw in files.items()})
        directories = envelope.directories_for(retained, directories + [prefix + name for name in dirs])
        envelope.validate_snapshot(retained, directories)
        captures[side] = (files, dirs)
        errors[side] = None
        originals[side] = Path(source).absolute()
    return originals, captures, retained, directories, errors


def _input_captures(files, directories, profile):
    if type(profile) is not dict or set(profile) != {"input_formats", "capture_errors"}:
        raise io.EvidenceError("unsupported upgrade selection profile")
    formats = profile["input_formats"]
    if type(formats) is not dict or set(formats) != set(SIDES):
        raise io.EvidenceError("upgrade input formats are missing")
    errors = profile["capture_errors"]
    if type(errors) is not dict or set(errors) != set(SIDES):
        raise io.EvidenceError("upgrade capture notices are missing")
    captures, expected = {}, set(BASE_FILES)
    expected_dirs = {"inputs", "inputs/baseline", "inputs/candidate", "metadata"}
    for side in SIDES:
        prefix = "inputs/" + side + "/"
        members, dirs = envelope.subset(files, directories, prefix)
        notice = errors[side]
        if notice is not None:
            if type(notice) is not str or not 0 < len(notice) <= 4096 or members or dirs or formats[side] is not None:
                raise io.EvidenceError("unavailable upgrade input has conflicting members or metadata")
            captures[side] = ({}, [])
            continue
        # An invalid input may be retained, but its format cannot be replaced by a stored flag.
        try:
            observed = _schema(members)
        except (ValueError, TypeError, KeyError, RecursionError):
            observed = None
        if formats[side] != observed:
            raise io.EvidenceError("upgrade input format differs from captured bytes")
        captures[side] = (members, dirs)
        expected.update(prefix + name for name in members)
        expected_dirs.update(prefix + name for name in dirs)
    if set(files) != expected or set(directories) != expected_dirs:
        raise io.EvidenceError("upgrade contains unexpected or missing members")
    return captures


def compare_packets(baseline, candidate, output):
    envelope.reject_overlap(output, [baseline, candidate])
    metadata = _metadata()
    originals, captures, retained, directories, errors = _capture_inputs(baseline, candidate, metadata)
    report = _evaluate(captures, errors)
    current = _current_gate(report)
    saved = {**retained, **metadata, "report.json": io.json_bytes(report), "index.html": render(report)}
    profile = {"input_formats": {side: report["inputs"][side]["packet_format"] for side in SIDES},
               "capture_errors": errors}
    for side in originals:
        bounded.recheck_snapshot(originals[side], *captures[side])
    root = envelope.complete(output, saved, directories, schema=FORMAT, profile=profile)
    for side in originals:
        bounded.recheck_snapshot(originals[side], *captures[side])
    return {"directory": root, "capture_complete": True, "report": report, **current}


def verify_comparison(directory):
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "runtime_upgrade_witness": "not established", "authenticity": "not established"}
    try:
        files, directories = bounded.capture_snapshot(directory)
        profile = envelope.check_manifest(files, directories, schema=FORMAT)
        captures = _input_captures(files, directories, profile)
        for name, raw in _metadata().items():
            if files[name] != raw:
                raise io.EvidenceError("upgrade policy or evaluator identity is unreviewed")
        report = _evaluate(captures, profile["capture_errors"])
        if files["report.json"] != io.json_bytes(report) or files["index.html"] != render(report):
            raise io.EvidenceError("upgrade report does not reproduce from retained input contracts")
        current = _current_gate(report)
        bounded.recheck_snapshot(directory, files, directories)
        result.update(integrity="verified", report=report, core_facts_recomputed=True, **current)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result["errors"].append(type(exc).__name__ + ": " + str(exc))
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(prog="sdcguard upgrade", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    commands.add_parser("policy")
    args = parser.parse_args(argv)
    try:
        if args.command == "policy":
            descriptor = native_policy_descriptor()
            print(json.dumps({"policy": descriptor, "policy_sha256": io.digest(io.json_bytes(descriptor))}, sort_keys=True, indent=2))
            return 0
        result = (compare_packets(args.baseline, args.candidate, args.out) if args.command == "compare"
                  else verify_comparison(args.directory))
        print(json.dumps(result, sort_keys=True, indent=2))
        if result.get("integrity", "verified") != "verified":
            return 2
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[result["scope_verdict"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [str(exc)]}))
        return 2
