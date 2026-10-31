"""Installed setup files and bounded offline preflight of supplied local inputs.

Owner: Hari Krishna Kancharla.
No serving process, request, readiness fetch or device work is started.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
from importlib.resources import files as package_files
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from . import config_inputs, evidence_bundle, evidence_io as io

EXPORT_FORMAT = "sdcguard.operator-setup.v1"
PREFLIGHT_FORMAT = "sdcguard.operator-preflight.v1"
SUMMARY_SCHEMA = "sdcguard.operator-preflight-report.v1"
_SECTIONS = ("static", "recording")
_RESOURCE_NAMES = (
    "README.md", "static-deployment.example.json", "recording-deployment.example.json", "SUPPORT.json",
    "profiles/vllm-capacity-shortfall.json", "profiles/sglang-unobserved-capacity.json",
    "profiles/ascend-declared-capacity.json", "profiles/lmcache-unverified-layout.json",
    "guides/CONFIG_SAFETY.md", "guides/EVIDENCE_REPORT.md", "guides/UPGRADE_GATE.md",
    "guides/SUPPORT_61229.md", "guides/LIFETIME_RULES.md", "guides/OWNER_ACTIONS.md", "guides/OVERLAP.md",
    "guides/H100_OBSERVER_RESULT.md", "guides/H100_OBSERVER_RESULT.json",
    "guides/FORWARD_INPUTS.md", "guides/ASSIGNMENT.md", "guides/ASSIGNMENT_NATIVE.md",
    "guides/NATIVE_EVIDENCE.md",
    "guides/ALLOCATION_RECIPIENT.md",
    "guides/NATIVE_CHECK_COVERAGE.md",
    "guides/CHECKSUM_OBSERVER.md",
    "guides/CHECKSUM_CLIENT.md",
)
_LIMITS = [
    "Only selected named static and saved-recording contracts are judged.",
    "This command starts no engine, serving request, readiness fetch or device work.",
    "Missing identities, original log associations and recording facts are not supplied by this wrapper.",
    "Local hashes check consistency, not authenticity or remote deployment state.",
    "Runtime isolation, actual model reads, loaded weights and physical write safety are not established.",
    "The combined output is bounded to 512 regular files and 512 MiB, with 64 MiB per file.",
    "An unretained selected section stays unknown; its standalone command remains separate.",
]


def evaluator_hashes():
    return {"operator_setup.py": io.digest(Path(__file__).read_bytes()),
            "evidence_evaluators": evidence_bundle.evaluator_hashes()}


def _resources():
    root = package_files("sdcguard").joinpath("operator_data")
    manifest = io.parse_json(root.joinpath("manifest.json").read_bytes())
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "owner", "files", "source_inputs", "limits"}
            or manifest["schema"] != EXPORT_FORMAT or manifest["owner"] != "Hari Krishna Kancharla"
            or not isinstance(manifest["files"], dict) or set(manifest["files"]) != set(_RESOURCE_NAMES)):
        raise io.EvidenceError("installed operator resource inventory differs from the fixed setup profile")
    payload = {}
    for name in _RESOURCE_NAMES:
        io.relative(name)
        raw = root.joinpath(*name.split("/")).read_bytes()
        receipt = manifest["files"][name]
        if (not isinstance(receipt, dict) or type(receipt.get("bytes")) is not int
                or len(raw) > io.MAX_FILE_BYTES or receipt != {"sha256": io.digest(raw), "bytes": len(raw)}):
            raise io.EvidenceError("installed operator resource differs from its pin: " + name)
        payload[name] = raw
    expected_sources = {name: ("docs/" + name.split("/", 1)[1] if name.startswith("guides/")
                        else "examples/config-safety/" + name.split("/", 1)[1])
                        for name in _RESOURCE_NAMES if name.startswith(("guides/", "profiles/"))}
    expected_sources["recording-deployment.example.json"] = "examples/deployment-61229.json"
    if (not isinstance(manifest["source_inputs"], dict) or set(manifest["source_inputs"]) != set(expected_sources)
            or any(manifest["source_inputs"][name] != {"source": source, "sha256": io.digest(payload[name])}
                   for name, source in expected_sources.items())
            or not isinstance(manifest["limits"], list) or any(not isinstance(item, str) for item in manifest["limits"])):
        raise io.EvidenceError("installed setup source receipts do not bind the fixed local resource inputs")
    if sum(map(len, payload.values())) > io.MAX_TOTAL_BYTES:
        raise io.EvidenceError("installed operator resources exceed supported bounds")
    return payload, manifest


def _seal(root, schema, *, complete=True, errors=()):
    captured = io.snapshot(root)
    manifest = {"schema": schema, "complete": complete, "errors": list(errors),
                "files": io.file_receipts(captured), "directories": io.names(root, directories=True)}
    raw = io.json_bytes(manifest)
    if len(captured) + 1 > io.MAX_FILES or sum(map(len, captured.values())) + len(raw) > io.MAX_TOTAL_BYTES:
        raise io.EvidenceError("completed operator output exceeds its aggregate bounds")
    io.write_file(root, "manifest.json", raw)
    return manifest


def export_setup(output):
    payload, packaged = _resources()
    root = io.create_directory(output)
    try:
        io.write_snapshot(root, payload)
        io.write_file(root, "metadata/resources.json", io.json_bytes(packaged))
        io.write_file(root, "metadata/evaluators.json", io.json_bytes(evaluator_hashes()))
        _seal(root, EXPORT_FORMAT)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        try:
            _seal(root, EXPORT_FORMAT, complete=False, errors=[_error(exc)])
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            pass
        raise io.EvidenceError("setup export is incomplete: " + _error(exc)) from exc
    return {"directory": str(root), "capture_complete": True,
            "scope_verdict": "inconclusive", "setup": "exported; deployment facts not assessed",
            "runtime_isolation": "not established"}


def _path(value):
    result = Path(value).absolute()
    if any(part in (".", "..") for part in result.parts):
        raise io.EvidenceError("input traversal is unsupported")
    return result


def _read_config(path):
    raw = config_inputs._read_regular(path.parent, [path.name], config_inputs.MAX_CONFIG_BYTES)
    value = io.parse_json(raw, max_nodes=config_inputs.MAX_JSON_NODES)
    if not isinstance(value, dict):
        raise io.EvidenceError("configuration must be a JSON object")
    return raw, value


def _recording_paths(config, base):
    """Bound a dedicated event directory; inspect only named engine files later."""
    from . import assessment_bundle
    assessment_bundle._safe_config(config)
    for key in ("engine_root", "events"):
        value = config.get(key)
        if not isinstance(value, str):
            continue
        path = _path(base / value)
        if not path.exists() and not path.is_symlink():
            continue
        if key == "engine_root" or path.is_dir():
            directory = config_inputs._directory(path)
            import os
            os.close(directory)
            if key == "events":
                io.names(path)
        else:
            io.read_file(path.parent, path.name)


def _snapshot(root):
    directories = io.names(root, directories=True)
    payload = io.snapshot(root)
    if directories != io.names(root, directories=True):
        raise io.EvidenceError("section directory inventory changed during capture")
    return payload, directories


def _error(exc):
    return type(exc).__name__ + ": " + str(exc)


def _capture(section, path, workspace):
    receipt = {"schema": "sdcguard.operator-input.v1", "section": section,
               "input_path": str(path), "input_sha256": None, "retained_kind": None,
               "capture_notes": []}
    target = workspace / section
    protected = [path]
    kind = None
    try:
        raw, config = _read_config(path)
        receipt["input_sha256"] = io.digest(raw)
        if section == "recording":
            protected += [path.parent / config[key] for key in ("engine_root", "events")
                          if isinstance(config.get(key), str)]
        elif isinstance(config.get("workers"), list):
            protected += [path.parent / worker["startup_log"]["path"] for worker in config["workers"]
                          if isinstance(worker, dict) and isinstance(worker.get("startup_log"), dict)
                          and isinstance(worker["startup_log"].get("path"), str)]
        if section == "static":
            evidence_bundle.create_bundle(target, config=path, components=False)
            kind = "evidence"
        else:
            from . import assessment_bundle
            _recording_paths(config, path.parent)
            legacy = workspace / "legacy"
            assessment_bundle.create_bundle(path, legacy, run_ready=False)
            verified = assessment_bundle.verify_bundle(legacy)
            if verified["integrity"] == "verified":
                try:
                    evidence_bundle.create_bundle(target, recorded_packet=legacy, components=False)
                    kind = "evidence"
                except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
                    receipt["capture_notes"].append("Common recording capture unavailable: " + _error(exc))
            if kind is None:
                target = legacy
                kind = "legacy_capture"
                receipt["capture_notes"].append("Recorded capture is incomplete or could not supply the common rule contract.")
        if io.digest(_read_config(path)[0]) != receipt["input_sha256"]:
            receipt["capture_notes"].append("Input configuration changed during preflight capture.")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        receipt["capture_notes"].append(_error(exc))
        if target.exists():
            kind = "evidence" if section == "static" else None
    payload, directories = {}, []
    if kind is not None and target.exists():
        try:
            payload, directories = _snapshot(target)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            receipt["capture_notes"].append("Section could not be retained within regular-file bounds: " + _error(exc))
            kind = None
    if kind is not None and not payload:
        receipt["capture_notes"].append("Section capture produced no retained regular files.")
        kind = None
    receipt["retained_kind"] = kind
    return {"receipt": receipt, "files": payload if kind is not None else {},
            "directories": directories if kind is not None else [], "protected_inputs": protected}


def _receipt(value, section):
    if (not isinstance(value, dict) or set(value) != {"schema", "section", "input_path", "input_sha256", "retained_kind", "capture_notes"}
            or value["schema"] != "sdcguard.operator-input.v1" or value["section"] != section
            or not isinstance(value["input_path"], str)
            or value["input_sha256"] is not None and (not isinstance(value["input_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", value["input_sha256"]) is None)
            or value["retained_kind"] not in (None, "evidence", "legacy_capture")
            or section == "static" and value["retained_kind"] == "legacy_capture"
            or not isinstance(value["capture_notes"], list)
            or any(not isinstance(item, str) for item in value["capture_notes"])):
        raise io.EvidenceError("unsupported selected input receipt")
    return value


def _section(section, receipt, root):
    receipt = _receipt(receipt, section)
    result = {"scope_verdict": "inconclusive", "integrity": "not captured",
              "missing": list(receipt["capture_notes"]), "findings": [], "identities": {},
              "receipt": deepcopy(receipt)}
    if section == "recording":
        result["proposed_fix_source"] = {"status": "unknown", "runtime_execution": "not established"}
    if receipt["retained_kind"] is None:
        result["missing"].append("Selected input has no retained, independently recomputable section packet.")
        return result
    if receipt["retained_kind"] == "legacy_capture":
        from . import assessment_bundle
        verified = assessment_bundle.verify_bundle(root)
        result.update(integrity=verified["integrity"], missing=result["missing"] + [
            "Retained legacy capture does not establish the common recorded-event contract."])
        return result
    verified = evidence_bundle.verify_bundle(root, rerun_components=False)
    result["integrity"] = verified["integrity"]
    if verified["integrity"] != "verified":
        result["missing"].extend(error.replace(str(root), "sections/" + section) for error in verified["errors"])
        return result
    report = verified["report"]
    expected = {"static": section == "static", "recorded": section == "recording", "components": "none"}
    if report["profile"] != expected:
        result["integrity"] = "failed"
        result["missing"].append("Section profile differs from its fixed selected input kind.")
        return result
    source = "inputs/static/original.json" if section == "static" else "inputs/recorded/evidence/config-original.json"
    if receipt["input_sha256"] is None or io.digest(io.read_file(root, source)) != receipt["input_sha256"]:
        result["missing"].append("Captured original configuration does not match the input receipt.")
        return result
    status = verified["scope_verdict"]
    result["scope_verdict"] = "not fine" if status == "not fine" else "inconclusive" if result["missing"] else status
    result["missing"].extend(report["missing"])
    result["identities"] = deepcopy(report["identities"])
    if section == "recording":
        # The common verifier has already recomputed this retained legacy
        # report. Its verification receipt intentionally contains no report.
        try:
            native = io.parse_json(io.read_file(root, "inputs/recorded/report.json"))
            result["proposed_fix_source"] = deepcopy(native["sections"]["fix_presence"])
            result["proposed_fix_source"]["scope"] = "inspected source presence, separate from recorded check execution"
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            result["proposed_fix_source"]["reason"] = "Source-presence summary unavailable: " + _error(exc)
    for row in report["findings"]:
        result["findings"].append({key: deepcopy(row[key]) for key in
            ("rule_id", "semantic_revision", "evidence_kind", "scope", "resource_kind", "subject", "scenario", "verdict", "required", "missing")})
        if row["required"] and row["verdict"] in ("unknown", "not_exercised"):
            result["missing"].extend(row["missing"])
    result["missing"] = list(dict.fromkeys(result["missing"]))
    return result


def _safe_section(section, receipt, root):
    _receipt(receipt, section)
    try:
        return _section(section, receipt, root)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        reason = _error(exc).replace(str(root), "sections/" + section)
        result = {"scope_verdict": "inconclusive", "integrity": "not established", "missing": [reason],
                  "findings": [], "identities": {}, "receipt": deepcopy(receipt)}
        if section == "recording":
            result["proposed_fix_source"] = {"status": "unknown", "runtime_execution": "not established"}
        return result


def _selection(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "static", "recording"}
            or value["schema"] != PREFLIGHT_FORMAT
            or any(type(value[key]) is not bool for key in _SECTIONS)
            or not any(value[key] for key in _SECTIONS)):
        raise io.EvidenceError("unsupported or empty selected preflight inventory")
    return value


def _compose(selection, sections):
    selection = _selection(selection)
    statuses = {value["scope_verdict"] for value in sections.values()}
    status = "not fine" if "not fine" in statuses else "fine" if statuses == {"fine"} else "inconclusive"
    return {"schema": SUMMARY_SCHEMA, "scope_verdict": status, "selection": deepcopy(selection),
        "scope": "Selected named static configuration and saved recorded-event contracts only",
        "sections": sections, "readiness_network": "disabled", "components": "not selected",
        "runtime_isolation": "not established", "authenticity": "not established", "limits": list(_LIMITS)}


def _report(root, selection, receipts):
    return _compose(selection, {side: _safe_section(side, receipts[side], root / "sections" / side)
        for side in _SECTIONS if selection[side]})


def render(report):
    esc = lambda value: html.escape(str(value), quote=True)
    rows = []
    for side, section in report["sections"].items():
        link = ("sections/" + side + "/index.html" if section["receipt"]["retained_kind"] == "evidence"
                else "sections/" + side + "/report.html")
        if section["scope_verdict"] == "not fine":
            failed = sorted({row["rule_id"] for row in section["findings"]
                             if row["required"] and row["verdict"] == "violated"})
            note = "Retained facts violate named contracts: " + ", ".join(failed) + "."
            if section["missing"]:
                note += " Separate gaps: " + "; ".join(section["missing"])
        elif section["scope_verdict"] == "fine":
            note = "The retained facts satisfy the selected named contracts."
        else:
            note = "; ".join(section["missing"]) or "Required facts remain unestablished."
        detail = esc(note)
        if side == "recording":
            detail += " Proposed backport source: " + esc(section["proposed_fix_source"]["status"]) + "."
        if section["receipt"]["retained_kind"] is not None:
            detail += " See <a href='" + link + "'>retained section</a>."
        rows.append("<tr><td>" + esc(side) + "</td><td>" + esc(section["scope_verdict"])
                    + "</td><td>" + detail + "</td></tr>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>Offline operator preflight</title>"
        "<style>body{font:15px/1.4 system-ui;max-width:1000px;margin:24px auto;padding:0 16px}"
        "table{border-collapse:collapse;width:100%}td,th{text-align:left;vertical-align:top;padding:8px;"
        "border-bottom:1px solid #ccd4da}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>"
        "<h1>Within selected named scopes: " + esc(report["scope_verdict"]) + "</h1><p>"
        + esc(report["scope"]) + ". Runtime isolation is not established.</p><p>"
        "<a href='report.json'>Full summary and identities</a> | <a href='manifest.json'>File manifest</a>"
        "</p><table><thead><tr><th>Selected input</th><th>Verdict</th><th>Facts and gaps</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table><h2>Limits</h2><ul>"
        + "".join("<li>" + esc(item) + "</li>" for item in _LIMITS) + "</ul></html>\n").encode()


def _projection(selection, receipts, retained, judgments, captures):
    sections = {side: (deepcopy(judgments[side]) if side in retained else _safe_section(side, receipts[side], None))
                for side in receipts}
    for side in sections:
        sections[side]["receipt"] = deepcopy(receipts[side])
    report = _compose(selection, sections)
    payload = {"metadata/selection.json": io.json_bytes(selection),
               "metadata/evaluators.json": io.json_bytes(evaluator_hashes()),
               "report.json": io.json_bytes(report), "index.html": render(report)}
    directories = set()
    for side, receipt in receipts.items():
        payload["receipts/" + side + ".json"] = io.json_bytes(receipt)
        if side not in retained:
            continue
        prefix = "sections/" + side + "/"
        payload.update({prefix + name: raw for name, raw in captures[side]["files"].items()})
        directories.update(prefix + name for name in captures[side]["directories"])
    for name in payload:
        parts = io.relative(name)
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    manifest = {"schema": PREFLIGHT_FORMAT, "complete": True, "errors": [],
                "files": io.file_receipts(payload), "directories": sorted(directories)}
    total = sum(map(len, payload.values())) + len(io.json_bytes(manifest))
    fits = (len(payload) + 1 <= io.MAX_FILES and len(directories) + 1 <= io.MAX_FILES
            and all(name.count("/") < 32 for name in directories)
            and total <= io.MAX_TOTAL_BYTES and all(len(raw) <= io.MAX_FILE_BYTES for raw in payload.values()))
    return fits, payload, sorted(directories), report


def _protect_destination(destination, captures):
    # Resolved aliases are used only to refuse writes. Input reads always use
    # the original no-follow paths, never these resolved forms.
    try:
        destinations = (destination, destination.resolve())
        for capture in captures.values():
            for supplied in capture["protected_inputs"]:
                for source in (supplied.absolute(), supplied.resolve()):
                    if any(target == source or source in target.parents or target in source.parents
                           for target in destinations):
                        raise io.EvidenceError("preflight output intersects a configured input file or source tree")
    except (OSError, RuntimeError) as exc:
        raise io.EvidenceError("output/input separation could not be established: " + _error(exc)) from exc


def preflight(output, *, static=None, recording=None):
    selected = {"schema": PREFLIGHT_FORMAT, "static": static is not None, "recording": recording is not None}
    _selection(selected)
    inputs = {side: _path(value) for side, value in (("static", static), ("recording", recording)) if value is not None}
    destination = _path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("preflight output already exists")
    for path in inputs.values():
        if destination == path or destination in path.parents:
            raise io.EvidenceError("preflight output may not contain or replace an input config")
    with TemporaryDirectory(prefix="sdcguard-operator-") as temporary:
        work = Path(temporary).resolve()
        captures, judgments = {}, {}
        for side, path in inputs.items():
            side_root = io.create_directory(work / side)
            captures[side] = _capture(side, path, side_root)
            retained = io.create_directory(work / (side + "-retained"))
            io.write_snapshot(retained, captures[side]["files"])
            for name in captures[side]["directories"]:
                io.make_directory(retained, name)
            judgments[side] = _safe_section(side, captures[side]["receipt"], retained)
        _protect_destination(destination, captures)
        order = sorted(captures, key=lambda side: (judgments[side]["scope_verdict"] != "not fine",
                       sum(map(len, captures[side]["files"].values())), side))
        receipts = {side: deepcopy(value["receipt"]) for side, value in captures.items()}
        retained = set()
        for receipt in receipts.values():
            receipt["retained_kind"] = None
        for side in order:
            capture = captures[side]
            if capture["receipt"]["retained_kind"] is None:
                continue
            proposed = deepcopy(receipts)
            proposed[side] = deepcopy(capture["receipt"])
            fits, _, _, _ = _projection(selected, proposed, retained | {side}, judgments, captures)
            if not fits:
                receipts[side]["capture_notes"].append(
                    "Selected section could not fit the aggregate preflight bounds; use its standalone command separately.")
            else:
                receipts, retained = proposed, retained | {side}
        fits, payload, directories, report = _projection(selected, receipts, retained, judgments, captures)
        if not fits:
            raise io.EvidenceError("selected preflight metadata exceeds aggregate output bounds")
        root = io.create_directory(destination)
        try:
            io.write_snapshot(root, payload)
            for name in directories:
                io.make_directory(root, name)
            _seal(root, PREFLIGHT_FORMAT)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            try:
                _seal(root, PREFLIGHT_FORMAT, complete=False, errors=[_error(exc)])
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                pass
            raise io.EvidenceError("preflight capture is incomplete: " + _error(exc)) from exc
    return {"directory": str(root), "capture_complete": True, "scope_verdict": report["scope_verdict"], "report": report}


def _manifest(root):
    directories = io.names(root, directories=True)
    captured = io.snapshot(root)
    value = io.parse_json(captured["manifest.json"])
    if (not isinstance(value, dict) or set(value) != {"schema", "complete", "errors", "files", "directories"}
            or value["schema"] not in (EXPORT_FORMAT, PREFLIGHT_FORMAT) or type(value["complete"]) is not bool
            or not isinstance(value["errors"], list) or any(not isinstance(item, str) for item in value["errors"])):
        raise io.EvidenceError("unsupported operator manifest")
    if not isinstance(value["files"], dict):
        raise io.EvidenceError("operator file receipts must be an object")
    for name, receipt in value["files"].items():
        io.relative(name)
        if (not isinstance(receipt, dict) or set(receipt) != {"sha256", "bytes"}
                or not isinstance(receipt["sha256"], str) or len(receipt["sha256"]) != 64
                or type(receipt["bytes"]) is not int or receipt["bytes"] < 0):
            raise io.EvidenceError("operator file receipt has invalid hash or count types")
    if value["files"] != io.file_receipts({name: raw for name, raw in captured.items() if name != "manifest.json"}):
        raise io.EvidenceError("operator membership, file hashes or lengths disagree")
    if value["directories"] != directories or io.names(root, directories=True) != directories:
        raise io.EvidenceError("operator directory inventory disagrees")
    if value["complete"] is not True or value["errors"]:
        raise io.EvidenceError("operator capture is incomplete")
    return value, captured, directories


def _verify_export(captured, directories):
    payload, packaged = _resources()
    expected = dict(payload, **{"metadata/resources.json": io.json_bytes(packaged),
                              "metadata/evaluators.json": io.json_bytes(evaluator_hashes())})
    if set(captured) != set(expected) | {"manifest.json"} or any(captured[name] != raw for name, raw in expected.items()):
        raise io.EvidenceError("exported setup differs from this installed resource/evaluator profile")
    expected_dirs = {"/".join(io.relative(name)[:index]) for name in expected
                     for index in range(1, len(io.relative(name)))}
    if set(directories) != expected_dirs:
        raise io.EvidenceError("exported setup directory inventory differs from the fixed profile")
    return {"setup": "verified local setup files; deployment facts not assessed", "scope_verdict": "inconclusive"}


def _verify_preflight(root, captured, directories):
    if captured["metadata/evaluators.json"] != io.json_bytes(evaluator_hashes()):
        raise io.EvidenceError("preflight evaluator differs from the installed profile")
    selection = _selection(io.parse_json(captured["metadata/selection.json"]))
    expected = {"manifest.json", "report.json", "index.html", "metadata/selection.json", "metadata/evaluators.json"}
    receipts = {}
    for side in _SECTIONS:
        if not selection[side]:
            continue
        name = "receipts/" + side + ".json"
        expected.add(name)
        receipts[side] = _receipt(io.parse_json(captured[name]), side)
        prefix = "sections/" + side + "/"
        members = {name for name in captured if name.startswith(prefix)}
        if receipts[side]["retained_kind"] is None and members:
            raise io.EvidenceError("unretained section unexpectedly contains packet files")
        if receipts[side]["retained_kind"] is not None and not members:
            raise io.EvidenceError("selected retained section packet is missing")
        expected |= members
    if set(captured) != expected:
        raise io.EvidenceError("operator output does not preserve the fixed selected inventory")
    expected_dirs = {"metadata", "receipts"}
    for side, receipt in receipts.items():
        if receipt["retained_kind"] is not None:
            prefix = "sections/" + side
            expected_dirs.update({"sections", prefix})
            expected_dirs.update(name for name in directories if name.startswith(prefix + "/"))
    if set(directories) != expected_dirs:
        raise io.EvidenceError("operator output contains unselected or unexpected directories")
    report = _report(root, selection, receipts)
    if captured["report.json"] != io.json_bytes(report) or captured["index.html"] != render(report):
        raise io.EvidenceError("operator summary does not reproduce from retained selected facts")
    return {"scope_verdict": report["scope_verdict"], "report": report}


def verify_output(directory):
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "runtime_isolation": "not established", "authenticity": "not established"}
    try:
        root = _path(directory)
        manifest, captured, directories = _manifest(root)
        with TemporaryDirectory(prefix="sdcguard-operator-verify-") as temporary:
            work = Path(temporary).resolve()
            io.write_snapshot(work, captured)
            for name in directories:
                io.make_directory(work, name)
            verified = (_verify_export(captured, directories) if manifest["schema"] == EXPORT_FORMAT
                        else _verify_preflight(work, captured, directories))
        if io.snapshot(root) != captured or io.names(root, directories=True) != directories:
            raise io.EvidenceError("operator output changed during verification")
        result.update(integrity="verified", **verified)
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result["errors"].append(_error(exc))
    return result


def main(argv=None):
    if argv and argv[0] == "run":
        from .native_operator.command import main as run
        return run(argv[1:])
    parser = argparse.ArgumentParser(prog="sdcguard operator", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", help="start the pinned native-check profile, capture ordinary requests, and verify", add_help=False)
    export = commands.add_parser("export", help="export setup guides and distinct input templates without a checkout")
    export.add_argument("output", type=Path)
    pre = commands.add_parser("preflight", help="capture selected local config/recording inputs offline")
    pre.add_argument("--static", type=Path)
    pre.add_argument("--recording", type=Path)
    pre.add_argument("--out", required=True, type=Path)
    verify = commands.add_parser("verify", help="recompute retained section results or verify exported setup bytes")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (export_setup(args.output) if args.command == "export" else
                  preflight(args.out, static=args.static, recording=args.recording) if args.command == "preflight" else
                  verify_output(args.directory))
        print(json.dumps(result, sort_keys=True, indent=2))
        if result.get("integrity", "verified") != "verified":
            return 2
        if args.command == "export" or "setup" in result:
            return 0
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[result["scope_verdict"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [_error(exc)]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
