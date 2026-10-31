"""Portable captured inputs with offline re-evaluation of the named contracts.

Owner: Hari Krishna Kancharla.
No serving engine, readiness request, owner replay or caller-supplied source is
executed. The optional component profile uses the existing trusted CPU checks.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from . import config_inputs, config_safety
from . import evidence_catalog as catalog
from . import evidence_io as io
from . import evidence_report as report_api

FORMAT = "sdcguard.evidence-packet.v1"
MANIFEST = "manifest.json"
MODULES = (
    "evidence_bundle.py", "evidence_io.py", "evidence_report.py", "evidence_catalog.py",
    "evidence_components.py", "evidence_recordings.py", "config_inputs.py", "config_safety.py",
    "config_capacity.py", "config_protections.py", "assessment.py", "assessment_bundle.py", "client_inventory.py",
    "lifetime/adapter.py", "lifetime/ledger.py", "lifetime/obligations.py", "lifetime/vllm_nixl.py",
    "lifetime/component_conformance.py", "lifetime/generation_obligations.py",
    "lifetime/generation_observations.py", "case_data/engine-pins.json",
)


def evaluator_hashes():
    root = Path(__file__).parent
    return {name: io.digest((root / name).read_bytes()) for name in MODULES}


def _profile(config, components, recorded):
    if type(components) is not bool:
        raise io.EvidenceError("component selection must be Boolean")
    value = {"static": config is not None, "recorded": recorded is not None,
             "components": "all" if components else "none"}
    if not any((value["static"], value["recorded"], components)):
        raise io.EvidenceError("select config, saved recording packet or the existing CPU components")
    return value


def _collect_components():
    from .evidence_components import collect, _json_copy
    try:
        return _json_copy(collect("all"))
    except Exception as exc:
        raise io.EvidenceError("trusted CPU component execution could not complete: "
                               + type(exc).__name__ + ": " + str(exc)) from exc


def _capture_static(path):
    document, provenance = config_inputs.load_deployment(path)
    path = Path(path).absolute()
    raw = config_inputs._read_regular(path.parent, [path.name], config_inputs.MAX_CONFIG_BYTES)
    if io.digest(raw) != provenance["raw_config_sha256"]:
        raise io.EvidenceError("source configuration changed during capture")
    files = {"inputs/static/original.json": raw}
    receipts = []
    for index, worker in enumerate(document["workers"]):
        log = provenance["logs"][worker["id"]]
        item = {"index": index, "worker_id": worker["id"], "captured": "error" not in log}
        if item["captured"]:
            name = f"logs/{index:03d}.txt"
            data = log["text"].encode("utf-8")
            if io.digest(data) != log["sha256"]:
                raise io.EvidenceError("captured startup bytes do not reproduce their hash")
            files["inputs/static/" + name] = data
            item["path"] = name
        else:
            item["reason"] = log["error"]
        receipts.append(item)
    capture = {"schema": "sdcguard.static-capture.v1", "workers": receipts}
    portable = _portable_static(document, capture, files)
    files["inputs/static/capture.json"] = io.json_bytes(capture)
    files["inputs/static/deployment.json"] = io.json_bytes(portable)
    return files


def _portable_static(original, capture, files):
    config_inputs._validate(original)
    if (not isinstance(capture, dict) or set(capture) != {"schema", "workers"}
            or capture["schema"] != "sdcguard.static-capture.v1"
            or not isinstance(capture["workers"], list)
            or len(capture["workers"]) != len(original["workers"])):
        raise io.EvidenceError("unsupported static capture inventory")
    portable = deepcopy(original)
    for index, (worker, item) in enumerate(zip(portable["workers"], capture["workers"])):
        if (not isinstance(item, dict) or type(item.get("index")) is not int or item["index"] != index
                or item.get("worker_id") != worker["id"] or type(item.get("captured")) is not bool):
            raise io.EvidenceError("static capture does not bind every worker exactly once")
        if item["captured"]:
            name = f"logs/{index:03d}.txt"
            if set(item) != {"index", "worker_id", "captured", "path"} or item["path"] != name:
                raise io.EvidenceError("unexpected captured startup path")
            metadata = worker.get("startup_log")
            config_inputs._log_metadata(metadata)
            io.relative(metadata["path"])
            if (metadata.get("worker_id", worker["id"]) != worker["id"]
                    or metadata["config_sha256"].lower() != config_inputs.binding_digest(worker, original)):
                raise io.EvidenceError("original startup metadata does not bind this worker/model/context")
            raw = files.get("inputs/static/" + name)
            if not isinstance(raw, bytes) or io.digest(raw) != metadata["sha256"].lower():
                raise io.EvidenceError("captured startup bytes disagree with the original configuration")
            raw.decode("utf-8")
            metadata["path"] = name
        else:
            if (set(item) != {"index", "worker_id", "captured", "reason"}
                    or not isinstance(item["reason"], str) or not item["reason"]):
                raise io.EvidenceError("unavailable startup evidence needs an explicit receipt")
            # Unavailable bytes cannot be made successful by relabelling a path.
            worker.pop("startup_log", None)
    return portable


def _static_from_packet(root, files):
    original = io.parse_json(files["inputs/static/original.json"], max_nodes=config_inputs.MAX_JSON_NODES)
    capture = io.parse_json(files["inputs/static/capture.json"])
    expected = _portable_static(original, capture, files)
    portable = io.parse_json(files["inputs/static/deployment.json"], max_nodes=config_inputs.MAX_JSON_NODES)
    if portable != expected:
        raise io.EvidenceError("portable static configuration is not derived from captured inputs")
    path = Path(root) / "inputs/static/deployment.json"
    document, provenance = config_inputs.load_deployment(path)
    native = config_safety.assess(document, provenance, path.parent)
    native["provenance"]["source_path"] = "inputs/static/deployment.json"
    native["provenance"]["original_config_sha256"] = io.digest(files["inputs/static/original.json"])
    native["provenance"]["capture_notices"] = [deepcopy(item) for item in capture["workers"] if not item["captured"]]
    for index, worker in enumerate(native["workers"]):
        log = native["provenance"]["logs"][worker["id"]]
        if "error" not in log:
            name = f"inputs/static/logs/{index:03d}.txt"
            log["path"] = name
            worker["startup"]["path"] = name
    native.pop("report_sha256")
    native["report_sha256"] = config_safety._sha(native)
    return native


def _expected_files(profile, files):
    expected = {MANIFEST, "report.json", "index.html", "metadata/catalog.json", "metadata/evaluators.json"}
    if profile["static"]:
        expected |= {"inputs/static/original.json", "inputs/static/deployment.json", "inputs/static/capture.json"}
        capture = io.parse_json(files["inputs/static/capture.json"])
        original = io.parse_json(files["inputs/static/original.json"], max_nodes=config_inputs.MAX_JSON_NODES)
        _portable_static(original, capture, files)
        expected |= {"inputs/static/" + item["path"] for item in capture["workers"] if item["captured"]}
    if profile["components"] == "all":
        expected.add("inputs/components.json")
    if profile["recorded"]:
        prefix = "inputs/recorded/"
        nested = io.parse_json(files[prefix + MANIFEST])
        if not isinstance(nested, dict) or not isinstance(nested.get("files"), dict):
            raise io.EvidenceError("saved recording packet inventory is missing")
        expected.add(prefix + MANIFEST)
        for name in nested["files"]:
            io.relative(name)
            expected.add(prefix + name)
    return expected


def _recompute(root, profile, files):
    static = _static_from_packet(root, files) if profile["static"] else None
    components = io.parse_json(files["inputs/components.json"]) if profile["components"] == "all" else None
    recorded = None
    if profile["recorded"]:
        from .evidence_recordings import normalize
        recorded = normalize(Path(root) / "inputs/recorded")
    return report_api.build(profile=profile, static=static, components=components, recorded=recorded)


def create_bundle(output, *, config=None, components=False, recorded_packet=None):
    """Capture provided bytes; write a completed manifest only after re-evaluation."""
    profile = _profile(config, components, recorded_packet)
    captured, directories = {}, []
    if config is not None:
        captured.update(_capture_static(config))
    if recorded_packet is not None:
        from .evidence_recordings import normalize
        normalize(recorded_packet)
        before = io.names(recorded_packet, directories=True)
        saved = io.snapshot(recorded_packet)
        if io.names(recorded_packet, directories=True) != before:
            raise io.EvidenceError("recording packet directories changed during capture")
        captured.update({"inputs/recorded/" + name: raw for name, raw in saved.items()})
        directories.extend("inputs/recorded/" + name for name in before)
    if components:
        captured["inputs/components.json"] = io.json_bytes(_collect_components())
    captured["metadata/catalog.json"] = io.json_bytes(catalog.catalog())
    captured["metadata/evaluators.json"] = io.json_bytes(evaluator_hashes())
    root = io.create_directory(output)
    try:
        io.write_snapshot(root, captured)
        for directory in directories:
            io.make_directory(root, directory)
        report = _recompute(root, profile, captured)
        io.write_file(root, "report.json", io.json_bytes(report))
        io.write_file(root, "index.html", report_api.render(report))
        files = io.snapshot(root)
        if set(files) | {MANIFEST} != _expected_files(profile, files):
            raise io.EvidenceError("created packet contains an unexpected or missing file")
        manifest = {"schema": FORMAT, "profile": profile, "complete": True, "errors": [],
                    "integrity": "local SHA256 consistency only; no authenticity claim",
                    "files": io.file_receipts(files), "directories": io.names(root, directories=True)}
        raw_manifest = io.json_bytes(manifest)
        if (len(files) + 1 > io.MAX_FILES
                or sum(len(raw) for raw in files.values()) + len(raw_manifest) > io.MAX_TOTAL_BYTES):
            raise io.EvidenceError("completed packet including its manifest exceeds supported bounds")
        io.write_file(root, MANIFEST, raw_manifest)
        return {"directory": str(root), "report": report, "capture_complete": True}
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        # Preserve partial capture as explicitly incomplete; never overwrite a file.
        try:
            files = io.snapshot(root)
            manifest = {"schema": FORMAT, "profile": profile, "complete": False,
                        "errors": [str(exc)], "integrity": "local SHA256 consistency only; no authenticity claim",
                        "files": io.file_receipts(files), "directories": io.names(root, directories=True)}
            io.write_file(root, MANIFEST, io.json_bytes(manifest))
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        raise io.EvidenceError("evidence capture is incomplete: " + str(exc)) from exc


def verify_bundle(directory, *, rerun_components=True):
    """Verify hashes and re-judge captured facts using the installed rule set."""
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "authenticity": "not established", "runtime_isolation": "not established",
              "component_execution": "not selected"}
    try:
        if type(rerun_components) is not bool:
            raise io.EvidenceError("component refresh selection must be Boolean")
        root = Path(directory).absolute()
        directories = io.names(root, directories=True)
        files = io.snapshot(root)
        manifest = io.parse_json(files[MANIFEST])
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "profile", "complete", "errors", "integrity", "files", "directories"}
                or manifest["schema"] != FORMAT or type(manifest["complete"]) is not bool
                or not isinstance(manifest["errors"], list)
                or any(not isinstance(error, str) for error in manifest["errors"])
                or manifest["integrity"] != "local SHA256 consistency only; no authenticity claim"):
            raise io.EvidenceError("unsupported evidence manifest")
        if not isinstance(manifest["files"], dict):
            raise io.EvidenceError("manifest file receipts must be an object")
        for name, receipt in manifest["files"].items():
            io.relative(name)
            if (not isinstance(receipt, dict) or set(receipt) != {"sha256", "bytes"}
                    or not isinstance(receipt["sha256"], str) or len(receipt["sha256"]) != 64
                    or type(receipt["bytes"]) is not int or receipt["bytes"] < 0):
                raise io.EvidenceError("manifest file receipt has invalid hash or count types")
        if manifest["files"] != io.file_receipts({name: raw for name, raw in files.items() if name != MANIFEST}):
            raise io.EvidenceError("file membership, hashes or lengths disagree with the manifest")
        if manifest["directories"] != directories or io.names(root, directories=True) != directories:
            raise io.EvidenceError("packet directory inventory changed or disagrees")
        if manifest["complete"] is not True:
            result.update(integrity="incomplete", errors=manifest["errors"])
            return result
        if manifest["errors"]:
            raise io.EvidenceError("completed packet retains capture errors")
        profile = manifest["profile"]
        if (not isinstance(profile, dict) or set(profile) != {"static", "recorded", "components"}
                or type(profile["static"]) is not bool or type(profile["recorded"]) is not bool
                or profile["components"] not in ("none", "all")):
            raise io.EvidenceError("unsupported evidence profile")
        if set(files) != _expected_files(profile, files):
            raise io.EvidenceError("packet evidence does not match the trusted profile inventory")
        if files["metadata/catalog.json"] != io.json_bytes(catalog.catalog()):
            raise io.EvidenceError("packet catalog differs from the installed trusted catalog")
        if files["metadata/evaluators.json"] != io.json_bytes(evaluator_hashes()):
            raise io.EvidenceError("evaluator identity is unreviewed; re-capture with the installed rule set")
        with TemporaryDirectory(prefix="sdcguard-evidence-verify-") as temporary:
            # This directory was created here. Resolve the operating system's
            # temp alias before the no-follow reader checks its private path.
            work = Path(temporary).resolve()
            io.write_snapshot(work, files)
            for name in directories:
                io.make_directory(work, name)
            report = _recompute(work, profile, files)
        if files["report.json"] != io.json_bytes(report):
            raise io.EvidenceError("reported findings do not reproduce from captured facts")
        if files["index.html"] != report_api.render(report):
            raise io.EvidenceError("readable report differs from recomputed findings")
        result.update(integrity="verified", scope_verdict=report["status"], report=report,
                      files_verified=len(files) - 1, core_facts_recomputed=True)
        if profile["components"] == "all":
            if not rerun_components:
                if result["scope_verdict"] != "not fine":
                    result["scope_verdict"] = "inconclusive"
                result["component_execution"] = "not refreshed"
                result["errors"].append("current trusted CPU component execution was not refreshed")
            else:
                try:
                    from .evidence_components import normalize
                    current = _collect_components()
                    normalize(current)
                    result["component_execution"] = {"scope": "source_fragment_cpu", "status": current["status"],
                        "basis": "fresh execution of installed trusted ordinary functions with CPU mocks",
                        "historical_receipt_authenticity": "not established", "physical_write_safety": "unknown"}
                    refreshed = current["status"] == "satisfied"
                except (OSError, ValueError, TypeError, KeyError, RecursionError, AssertionError) as exc:
                    refreshed = False
                    result["component_execution"] = {"scope": "source_fragment_cpu", "status": "unknown",
                        "reason": str(exc), "physical_write_safety": "unknown"}
                if not refreshed:
                    if result["scope_verdict"] != "not fine":
                        result["scope_verdict"] = "inconclusive"
                    result["errors"].append("current trusted CPU component proof was incomplete")
        if io.snapshot(root) != files or io.names(root, directories=True) != directories:
            raise io.EvidenceError("packet changed during offline verification")
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result.update(integrity="failed", scope_verdict="inconclusive")
        result.pop("report", None)
        result["errors"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard evidence", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="capture config, saved packets and existing CPU components")
    collect.add_argument("--config", type=Path)
    collect.add_argument("--assessment-packet", type=Path)
    collect.add_argument("--components", action="store_true", help="run both existing trusted CPU component checks")
    collect.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify", help="recompute a portable packet offline; no serving traffic")
    verify.add_argument("directory", type=Path)
    commands.add_parser("catalog", help="show named rule contracts; shape validation is not verdict proof")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            print(json.dumps(catalog.catalog(), indent=2, sort_keys=True))
            return 0
        if args.command == "collect":
            packet = create_bundle(args.out, config=args.config, components=args.components,
                                   recorded_packet=args.assessment_packet)
            result = {"directory": packet["directory"], "capture_complete": True,
                      "scope_verdict": packet["report"]["status"], "runtime_isolation": "not established"}
        else:
            result = verify_bundle(args.directory)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result.get("integrity", "verified") != "verified":
            return 2
        return {"fine": 0, "not fine": 1, "inconclusive": 2}[result["scope_verdict"]]
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"scope_verdict": "inconclusive", "errors": [str(exc)]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
