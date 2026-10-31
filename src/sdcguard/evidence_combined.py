"""Reverify named native mapping slots alongside existing evidence packets.

Owner: Hari Krishna Kancharla.
Each section retains its own contract. No engine or serving traffic is started.
"""
from copy import deepcopy
import html
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from . import assignment_report
from . import evidence_bundle as legacy
from . import evidence_envelope as envelope
from . import evidence_io as io
from . import evidence_json_budget as json_budget
from . import evidence_native_io as bounded
from . import evidence_native_mapping as mapping

FORMAT = "sdcguard.evidence-envelope.v1"
SCHEMA = "sdcguard.evidence-report.v2"
SLOT = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
BASE_FILES = {"metadata/catalog.json", "metadata/evaluators.json", "report.json", "index.html"}
LIMITS = (
    "Only the selected named contracts are judged; unselected checks did not run.",
    "Native mapping agreement does not establish ownership, content or physical reads.",
    "A required missing or unsupported input is unknown, not an isolation finding.",
    "Repeated journals are retained evidence aliases, not independent executions.",
    "Recorded identities and local hashes do not authenticate a deployed engine.",
    "No candidate runtime execution or upgrade workload binding is established.",
    "All nested inputs share one artifact count, byte, directory and depth bound.",
)


def slot_name(name):
    if type(name) is not str or SLOT.fullmatch(name) is None:
        raise io.EvidenceError("native slot names need a lowercase letter followed by letters, digits, _ or -")
    return name


def contract():
    return {"schema": "sdcguard.combined-catalog.v1", "legacy": legacy.catalog.catalog(),
            "native": {"rule_id": assignment_report.contract.PROPERTY, "semantic_revision": 1,
                "evidence_kind": mapping.EVIDENCE_KIND, "scope": mapping.SCOPE,
                "resource_kind": mapping.RESOURCE, "allowed_verdicts": list(mapping.VERDICTS),
                "required_selection": "explicit named slots; supplied slots are always required",
                "identity_basis": "recorded locators, not stable upgrade subjects",
                "missing_or_invalid": "unknown", "limits": list(mapping.LIMITS)}}


def evaluator_hashes():
    root = Path(__file__).parent
    names = set(legacy.MODULES) | set(assignment_report.evaluator_hashes(native=True)) | {
        "evidence_combined.py", "evidence_envelope.py", "evidence_native_mapping.py",
        "evidence_native_io.py", "evidence_entry.py", "evidence_json_budget.py", "cli.py"}
    return {name: io.digest((root / name).read_bytes()) for name in sorted(names)}


def _selection(profile):
    if type(profile) is not dict or set(profile) != {"requirements", "inputs"}:
        raise io.EvidenceError("unsupported combined selection")
    required, inputs = profile["requirements"], profile["inputs"]
    if (type(required) is not dict or set(required) != {"legacy", "native_mapping"}
            or type(required["legacy"]) is not bool or type(required["native_mapping"]) is not list
            or not 1 <= len(required["native_mapping"]) <= 16):
        raise io.EvidenceError("combined evidence needs one to sixteen explicit native slots")
    names = required["native_mapping"]
    for name in names:
        slot_name(name)
    if names != sorted(set(names)):
        raise io.EvidenceError("required native slots are not sorted and unique")
    if (type(inputs) is not dict or set(inputs) != {"legacy", "native_mapping"}
            or type(inputs["native_mapping"]) is not dict or set(inputs["native_mapping"]) != set(names)):
        raise io.EvidenceError("required native slots do not match the input inventory")
    records = [(inputs["native_mapping"][name], "inputs/native/" + name) for name in names]
    if required["legacy"]:
        records.append((inputs["legacy"], "inputs/legacy"))
    elif inputs["legacy"] is not None:
        raise io.EvidenceError("unselected legacy evidence has an input")
    for record, expected in records:
        if type(record) is not dict or set(record) != {"path", "capture_error"}:
            raise io.EvidenceError("unsupported combined input record")
        if record["path"] is None:
            if type(record["capture_error"]) is not str or not 0 < len(record["capture_error"]) <= 4096:
                raise io.EvidenceError("unavailable evidence needs a bounded capture notice")
        elif record["path"] != expected or record["capture_error"] is not None:
            raise io.EvidenceError("combined input is not at its fixed captured location")
    return required, inputs


def _check_members(profile, files, directories):
    required, inputs = _selection(profile)
    selected = [inputs["native_mapping"][name] for name in required["native_mapping"]]
    if required["legacy"]:
        selected.append(inputs["legacy"])
    prefixes = [record["path"] + "/" for record in selected if record["path"] is not None]
    expected = BASE_FILES | {"manifest.json"}
    extra_directories = set()
    for prefix in prefixes:
        expected.update(name for name in files if name.startswith(prefix))
        extra_directories.add(prefix[:-1])
        extra_directories.update(name for name in directories if name.startswith(prefix))
    if set(files) != expected or directories != envelope.directories_for(files, extra_directories):
        raise io.EvidenceError("combined evidence contains unexpected or missing members")


def _portable(value, root):
    if type(value) is str:
        return value.replace(str(root), "captured-inputs")
    if type(value) is list:
        return [_portable(item, root) for item in value]
    if type(value) is dict:
        return {key: _portable(item, root) for key, item in value.items()}
    return value


def _legacy_section(record, files, directories):
    if record["path"] is None:
        return {"integrity": "unavailable", "scope_verdict": "inconclusive",
                "errors": [record["capture_error"]]}
    members, dirs = envelope.subset(files, directories, record["path"] + "/")
    def evaluate(root):
        checked = _portable(legacy.verify_bundle(root, rerun_components=False), root)
        if checked["integrity"] == "verified":
            checked["scope_verdict"] = checked["report"]["status"]
            checked["errors"] = [error for error in checked["errors"]
                if error != "current trusted CPU component execution was not refreshed"]
        return checked
    return envelope.evaluate_private(members, dirs, evaluate)


def _native_section(name, record, files, directories, index):
    result = {"slot": name, "required": True, "integrity": "unavailable",
              "scope_verdict": "inconclusive", "findings": [], "streams": [],
              "recorded_identities": [], "errors": []}
    if record["path"] is None:
        result["errors"].append(record["capture_error"])
        return result
    members, dirs = envelope.subset(files, directories, record["path"] + "/")
    try:
        normalized = mapping.normalize_snapshot(members, dirs, packet_index=index)
        result.update(integrity="verified", scope_verdict=normalized["scope_verdict"],
                      findings=deepcopy(normalized["findings"]), normalized=normalized)
        for row in result["findings"]:
            row["subject"]["slot"] = name
        result["streams"] = [{"session": row["session"], "journal_sha256": row["sha256"],
                              "file": row["file"]} for row in normalized["native_report"]["sessions"]]
        for row in normalized["native_report"]["sessions"]:
            identity = row["recorded_identity"]
            profile = identity.get("profile")
            sources = identity.get("source_sha256")
            result["recorded_identities"].append({"session": row["session"],
                "adapter": identity.get("adapter"),
                "model": profile.get("model") if type(profile) is dict else None,
                "model_revision": profile.get("model_revision") if type(profile) is dict else None,
                "source_files_sha256": io.digest(io.json_bytes(sources)) if type(sources) is dict else None,
                "configuration_sha256": io.digest(io.json_bytes(profile)) if type(profile) is dict else None,
                "basis": "recorded fields; deployment identity and loaded weights are not authenticated"})
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result.update(integrity="invalid", errors=[type(exc).__name__ + ": " + str(exc)])
    return result


def _stream_groups(sections):
    sessions, journals = {}, {}
    for section in sections:
        for stream in section["streams"]:
            pointer = {"slot": section["slot"], "session": stream["session"], "file": stream["file"]}
            sessions.setdefault(stream["session"], set()).add(stream["journal_sha256"])
            journals.setdefault(stream["journal_sha256"], []).append(pointer)
    conflicts = sorted(session for session, hashes in sessions.items() if len(hashes) > 1)
    for section in sections:
        for row in section["findings"]:
            if row["subject"]["session"] in conflicts:
                row["facts"]["recomputed_native_verdict"] = row["verdict"]
                row["verdict"] = "unknown"
                row["missing"].append("The recorded session appears with conflicting raw journals.")
        if section["integrity"] == "verified":
            verdicts = {row["verdict"] for row in section["findings"]}
            section["scope_verdict"] = ("not fine" if "violated" in verdicts else
                "inconclusive" if "unknown" in verdicts else "fine")
    return {"unique_raw_streams": len(journals), "independent_executions": "not established",
            "aliases": [{"journal_sha256": digest, "locations": rows} for digest, rows in sorted(journals.items())
                        if len(rows) > 1], "conflicting_sessions": conflicts}


def _missing(sections, old):
    missing = [{"section": "native_mapping", "slot": section["slot"], "errors": section["errors"],
                "semantic_missing": [{"subject": deepcopy(row["subject"]), "missing": list(row["missing"])}
                    for row in section["findings"] if row["verdict"] == "unknown"]}
               for section in sections if section["scope_verdict"] == "inconclusive"]
    if old is not None and old["scope_verdict"] == "inconclusive":
        missing.append({"section": "legacy", "errors": old["errors"],
                        "semantic_missing": deepcopy(old.get("report", {}).get("missing", []))})
    return missing


def recompute(profile, files, directories):
    envelope.validate_snapshot(files, directories)
    required, inputs = _selection(profile)
    old = _legacy_section(inputs["legacy"], files, directories) if required["legacy"] else None
    sections = [_native_section(name, inputs["native_mapping"][name], files, directories, index)
                for index, name in enumerate(required["native_mapping"])]
    groups = _stream_groups(sections)
    missing = _missing(sections, old)
    statuses = [section["scope_verdict"] for section in sections] + ([old["scope_verdict"]] if old else [])
    status = "not fine" if "not fine" in statuses else "inconclusive" if "inconclusive" in statuses else "fine"
    result = {"schema": SCHEMA, "requirements": deepcopy(required), "status": status,
              "scope": "Selected recorded mapping and existing named contracts only",
              "legacy": old, "native_mapping": sections, "stream_inventory": groups,
              "missing": missing, "runtime_isolation": "not established", "authenticity": "not established",
              "runtime_upgrade_witness": "not established", "ownership_verdict": "unknown",
              "physical_read_verdict": "unknown", "limits": list(LIMITS)}
    io.config_inputs._json_values(result, max_nodes=io.MAX_JSON_NODES)
    return result


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for section in report["native_mapping"]:
        identities = "\n".join(json.dumps(identity, sort_keys=True) for identity in section["recorded_identities"])
        rows.append("<tr><td>" + esc(section["slot"]) + "</td><td>" + esc(section["scope_verdict"])
                    + "</td><td>" + esc(section["integrity"]) + "</td><td><pre>" + esc(identities) + "</pre></td></tr>")
    legacy_state = "not selected" if report["legacy"] is None else report["legacy"]["scope_verdict"]
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Scoped inference evidence</title><style>body{font:16px/1.5 system-ui;max-width:960px;"
        "margin:24px auto;padding:0 16px}table{width:100%;border-collapse:collapse}td,th{padding:8px;"
        "border-bottom:1px solid #bbb;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Within the selected contracts: " + esc(report["status"]) + "</h1>"
        "<p>Native slots judge recorded allocator-to-model-entry mapping. Ownership and physical reads remain unknown. "
        "Runtime isolation and an upgrade execution witness are not established.</p><p>Existing evidence section: "
        + esc(legacy_state) + ".</p><p><a href='report.json'>Full recomputed sections</a> | "
        "<a href='manifest.json'>Retained inputs and file hashes</a></p><table><tr><th>Required native slot</th>"
        "<th>Scoped result</th><th>Input verification</th><th>Recorded identity</th></tr>"
        + "".join(rows) + "</table><h2>Missing facts</h2><pre>"
        + esc(json.dumps(report["missing"], indent=2, sort_keys=True)) + "</pre><h2>Limits</h2><ul>"
        + "".join("<li>" + esc(note) + "</li>" for note in LIMITS) + "</ul></html>\n").encode()


def _attach(source, prefix, files, directories, metadata_bytes, metadata_nodes=0):
    parents = envelope.directories_for({}, [prefix])
    added = set(parents) - set(directories)
    # Reserve the largest permitted JSON report, HTML report and manifest, plus known metadata.
    byte_allowance = io.MAX_TOTAL_BYTES - 3 * io.MAX_FILE_BYTES - metadata_bytes - sum(map(len, files.values()))
    file_allowance = io.MAX_FILES - len(BASE_FILES) - 1 - len(files)
    directory_allowance = io.MAX_FILES - 2 - len(directories) - len(added)
    if min(byte_allowance, file_allowance, directory_allowance) < 0:
        raise io.EvidenceError("combined input allowance is exhausted")
    snapshot, dirs = bounded.capture_snapshot(source, max_files=file_allowance, max_bytes=byte_allowance,
        max_directories=directory_allowance, max_depth=bounded.MAX_DIRECTORY_DEPTH - len(io.relative(prefix)))
    candidate = {**files, **{prefix + "/" + name: raw for name, raw in snapshot.items()}}
    combined_dirs = envelope.directories_for(candidate, list(directories) + parents
                                             + [prefix + "/" + name for name in dirs])
    json_budget.count_members(candidate, max_nodes=io.MAX_JSON_NODES - metadata_nodes)
    envelope.validate_snapshot(candidate, combined_dirs)
    files.update({prefix + "/" + name: raw for name, raw in snapshot.items()})
    directories[:] = combined_dirs
    return Path(source).absolute(), snapshot, dirs


def create_bundle(output, *, native_packets=(), require_native=(), legacy_packet=None,
                  config=None, components=False, recorded_packet=None):
    supplied = {}
    for name, path in native_packets:
        slot_name(name)
        if name in supplied:
            raise io.EvidenceError("a native slot was assigned more than once")
        supplied[name] = path
    required = sorted({slot_name(name) for name in require_native} | set(supplied))
    if not 1 <= len(required) <= 16:
        raise io.EvidenceError("select one to sixteen required native slots")
    if type(components) is not bool:
        raise io.EvidenceError("component selection must be Boolean")
    has_legacy = legacy_packet is not None or config is not None or components or recorded_packet is not None
    if legacy_packet is not None and (config is not None or components or recorded_packet is not None):
        raise io.EvidenceError("a retained legacy packet cannot be mixed with new legacy input selection")
    envelope.reject_overlap(output, list(supplied.values())
                            + [path for path in (legacy_packet, recorded_packet) if path is not None])
    profile = {"requirements": {"legacy": has_legacy, "native_mapping": required},
               "inputs": {"legacy": None, "native_mapping": {}}}
    metadata = {"metadata/catalog.json": io.json_bytes(contract()),
                "metadata/evaluators.json": io.json_bytes(evaluator_hashes())}
    metadata_nodes = json_budget.count_members(metadata)
    files, directories, originals = {}, [], []
    def capture(path, prefix):
        try:
            original = _attach(path, prefix, files, directories, sum(map(len, metadata.values())), metadata_nodes)
            originals.append(original)
            return {"path": prefix, "capture_error": None}
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            return {"path": None, "capture_error": (type(exc).__name__ + ": " + str(exc))[:4096]}
    if has_legacy:
        if legacy_packet is not None:
            profile["inputs"]["legacy"] = capture(legacy_packet, "inputs/legacy")
        else:
            prior_names = set(files)
            prior_directories = list(directories)
            prior_original_count = len(originals)
            try:
                with TemporaryDirectory(prefix="sdcguard-envelope-legacy-") as temporary:
                    work = Path(temporary).resolve()
                    recorded = None
                    retained = None
                    if recorded_packet is not None:
                        # Reserve the root and six outer/legacy parents; static logs need two more.
                        retained = bounded.capture_snapshot(recorded_packet,
                            max_files=io.MAX_FILES - 2 * (len(BASE_FILES) + 1),
                            max_bytes=io.MAX_TOTAL_BYTES - 3 * io.MAX_FILE_BYTES - sum(map(len, metadata.values())),
                            max_directories=max(0, io.MAX_FILES - 7 - (2 if config is not None else 0)),
                            max_depth=bounded.MAX_DIRECTORY_DEPTH - 4)
                        json_budget.count_members(retained[0], max_nodes=io.MAX_JSON_NODES - metadata_nodes)
                        recorded = io.create_directory(work / "recorded")
                        envelope.materialize(recorded, *retained)
                    legacy.create_bundle(work / "legacy", config=config, components=components, recorded_packet=recorded)
                    profile["inputs"]["legacy"] = capture(work / "legacy", "inputs/legacy")
                    if retained is not None:
                        bounded.recheck_snapshot(recorded_packet, *retained)
                        if profile["inputs"]["legacy"]["path"] is not None:
                            generated = next(item[1] for item in originals[prior_original_count:]
                                             if item[0] == work / "legacy")
                            shared = {name: generated["inputs/recorded/" + name] for name in retained[0]}
                            if shared != retained[0]:
                                raise io.EvidenceError("generated legacy packet did not preserve the assessment bytes")
                            originals.append((Path(recorded_packet).absolute(), shared, retained[1]))
                    retained = None
                    originals[:] = [item for item in originals if item[0] != work / "legacy"]
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
                for name in set(files) - prior_names:
                    files.pop(name)
                directories[:] = prior_directories
                del originals[prior_original_count:]
                retained = None
                profile["inputs"]["legacy"] = {"path": None,
                    "capture_error": (type(exc).__name__ + ": " + str(exc))[:4096]}
    for name in required:
        profile["inputs"]["native_mapping"][name] = (capture(supplied[name], "inputs/native/" + name)
            if name in supplied else {"path": None, "capture_error": "No packet supplied for this required slot."})
    report = recompute(profile, files, directories)
    saved = {**files, **metadata, "report.json": io.json_bytes(report), "index.html": render(report)}
    for root, snapshot, dirs in originals:
        bounded.recheck_snapshot(root, snapshot, dirs)
    root = envelope.complete(output, saved, directories, schema=FORMAT, profile=profile)
    for original, snapshot, dirs in originals:
        bounded.recheck_snapshot(original, snapshot, dirs)
    return {"directory": root, "capture_complete": True, "report": report}


def verify_snapshot(files, directories):
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "authenticity": "not established", "runtime_isolation": "not established"}
    try:
        profile = envelope.check_manifest(files, directories, schema=FORMAT)
        _check_members(profile, files, directories)
        if files["metadata/catalog.json"] != io.json_bytes(contract()):
            raise io.EvidenceError("combined contract differs from the installed contract")
        if files["metadata/evaluators.json"] != io.json_bytes(evaluator_hashes()):
            raise io.EvidenceError("combined evaluator identity is unreviewed")
        report = recompute(profile, files, directories)
        if files["report.json"] != io.json_bytes(report) or files["index.html"] != render(report):
            raise io.EvidenceError("combined report does not reproduce from retained evidence")
        result.update(integrity="verified", scope_verdict=report["status"], report=report,
                      core_facts_recomputed=True,
                      verification_scope="captured facts only; selected current CPU refresh is separate")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result["errors"].append(str(exc))
    return result


def refresh_components(checked):
    gaps = []
    checked["component_execution"] = "not selected"
    old = checked["report"]["legacy"]
    if old is not None and old["integrity"] == "verified" and old["report"]["profile"]["components"] == "all":
        try:
            from .evidence_components import normalize
            current = legacy._collect_components()
            normalize(current)
            checked["component_execution"] = {"scope": "source_fragment_cpu", "status": current["status"]}
            if current["status"] != "satisfied":
                gaps.append("Current trusted CPU component proof was incomplete.")
        except (OSError, ValueError, TypeError, KeyError, RecursionError, AssertionError) as exc:
            checked["component_execution"] = {"scope": "source_fragment_cpu", "status": "unknown",
                                              "reason": type(exc).__name__ + ": " + str(exc)}
            gaps.append("Current trusted CPU component refresh failed: " + str(exc))
        if gaps and checked["scope_verdict"] != "not fine":
            checked["scope_verdict"] = "inconclusive"
    checked["errors"].extend(gaps)
    return checked


def verify_bundle(directory, *, rerun_components=True):
    try:
        if type(rerun_components) is not bool:
            raise io.EvidenceError("component refresh selection must be Boolean")
        files, directories = bounded.capture_snapshot(directory)
        checked = verify_snapshot(files, directories)
        if checked["integrity"] == "verified" and rerun_components:
            refresh_components(checked)
        elif checked["integrity"] == "verified":
            old = checked["report"]["legacy"]
            if old is not None and old["integrity"] == "verified" and old["report"]["profile"]["components"] == "all":
                checked["component_execution"] = "not refreshed"
                checked["errors"].append("Current trusted CPU component execution was not refreshed.")
                if checked["scope_verdict"] != "not fine":
                    checked["scope_verdict"] = "inconclusive"
        bounded.recheck_snapshot(directory, files, directories)
        return checked
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [str(exc)],
                "authenticity": "not established", "runtime_isolation": "not established"}
