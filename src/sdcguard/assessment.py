"""Scoped, read-only assessment of the supported #61229 deployment recording.

No search, holds, request mutation or regression replay is executed here. An
optional configured readiness check uses the existing read-only metrics check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path
import re
import subprocess
import sys

from .lifetime.adapter import read_events
from .lifetime.obligations import CONSUMER_CHECKS, evaluate

SCOPE = "Recorded #61229 source lifetime and scheduler admission; no whole-server correctness guarantee"
WORKLOAD_SCOPE = "Declared workload coverage, source lifetime and scheduler admission; no whole-server correctness guarantee"
TOPOLOGY = {"same_host": True, "tp": 1, "pp": 1, "dcp": 1, "pcp": 1,
            "cache_groups": 1, "prefix_caching": False, "eager": True, "connector": "NixlConnector"}


def pins():
    return json.loads(files("sdcguard").joinpath("case_data/engine-pins.json").read_text())


def inspect_engine(root: str | Path, version: str) -> dict:
    """Match the full relevant source files; version strings/symbol names alone do not prove a fix."""
    pin = pins()
    result = {"version": version, "adapter_fits": False, "fix": "unknown", "variant": None,
              "inspected_source_only": True, "file_sha256": {}}
    if version != pin["version"]:
        return result
    root = Path(root).resolve()
    try:
        for name in pin["variants"]["fixed"]:
            path = root / name
            if not path.resolve().is_relative_to(root):
                return result
            result["file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    except OSError:
        return result
    for variant, expected in pin["variants"].items():
        if result["file_sha256"] == expected:
            result.update(adapter_fits=True, variant=variant, fix="present" if variant == "fixed" else "absent")
    return result


def installed_engine_identity() -> dict:
    """Called once at recorder setup, without importing an inference engine."""
    mod = sys.modules.get("vllm")
    if mod is None or not getattr(mod, "__file__", None):
        return {"adapter_fits": False, "fix": "unknown"}
    return inspect_engine(Path(mod.__file__).resolve().parent.parent, getattr(mod, "__version__", "unknown"))


def report_sections(report: dict, config: dict) -> dict:
    """Keep the narrow rule result distinct from unprovided product evidence."""
    config = config if isinstance(config, dict) else {}
    identity = report.get("engine", {})
    topology = config.get("topology")
    revision = config.get("model_revision")
    applicable = (type(config.get("schema")) is int and config.get("schema") in (1, 2)
                  and config.get("engine") == "vllm" and isinstance(topology, dict)
                  and topology.keys() == TOPOLOGY.keys()
                  and all(type(topology[k]) is type(v) and topology[k] == v for k, v in TOPOLOGY.items())
                  and config.get("nixl_version") == "1.4.1"
                  and config.get("model") == "Qwen/Qwen3-0.6B"
                  and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None
                  and identity.get("adapter_fits") is True)
    findings = report.get("findings", [])
    source_bound = (identity.get("adapter_fits") is True
                    and "recording bound to the inspected engine source" not in report.get("missing", []))
    checks = [{"request": f["request"],
               "status": "observed" if f.get("protection") == CONSUMER_CHECKS
                         and f.get("verdict") != "unknown" and source_bound else "unknown",
               "basis": f.get("relationship", "not established")} for f in findings]
    report["sections"] = {
        "applicability": {"status": "supported" if applicable else "inconclusive",
                          "basis": "declared model/config and exact inspected local engine source",
                          "live_model_weights_verified": False},
        "fix_presence": {"status": identity.get("fix", "unknown"),
                         "basis": "exact relevant local source files; source presence does not prove execution"},
        "check_execution": {"status": "observed" if any(x["status"] == "observed" for x in checks) else "unknown",
                            "requests": checks},
        "lifecycle": {"status": report["status"], "findings": findings},
        "client_outcomes": {"status": "unknown", "requests": [
            {"request": f["request"], "outcome": "unknown", "basis": "client terminal evidence not supplied"}
            for f in findings]},
        "regression": {"status": "not supplied", "basis": "this assessment does not execute or certify a regression replay"},
        "overhead": {"status": "not measured", "basis": "no matched performance measurements supplied to this assessment"},
        "missing_evidence": {"lifecycle": list(report.get("missing", [])),
                             "client_outcomes": ["attributed terminal observations and clean references"],
                             "regression": ["a separately judged pinned replay result"],
                             "overhead": ["matched checks-off/on performance records"]},
    }
    if config.get("schema") == 2:
        coverage = report.get("request_coverage", {"status": "unknown", "issues": ["client inventory unavailable"]})
        report["sections"]["request_coverage"] = coverage
        report["sections"]["missing_evidence"]["request_coverage"] = list(coverage.get("issues", []))
        # Observing a check for one request does not qualify the declared batch.
        report["sections"]["check_execution"]["status"] = (
            "observed" if coverage.get("status") == "complete" and checks
            and all(x["status"] == "observed" for x in checks) else "unknown")
    report["configuration"] = {key: config.get(key) for key in
                               ("engine", "engine_version", "nixl_version", "model", "model_revision", "topology")}
    report["integrity_scope"] = "local evidence only; no authenticity signature or remote deployment attestation"
    return report


def assess(config: dict, *, base: Path = Path("."), run_ready=True) -> dict:
    report = {"schema": 1, "scope": SCOPE, "status": "inconclusive", "missing": [], "findings": [],
              "excluded": ["retry/reused-id cases", "new failure orderings", "actual attention consumption",
                           "whole-response correctness", "naturally faulty GPU diagnosis"]}
    if not isinstance(config, dict) or type(config.get("schema")) is not int or config["schema"] not in (1, 2):
        report["missing"].append("supported deployment config schema 1 or 2")
        return report_sections(report, config)
    if config["schema"] == 2:
        report.update(schema=2, scope=WORKLOAD_SCOPE)
    elif "client_inventory" in config:
        report["missing"].append("client inventory requires workload config schema 2")
    topology = config.get("topology")
    if (config.get("engine") != "vllm" or not isinstance(topology, dict) or topology.keys() != TOPOLOGY.keys()
            or any(type(topology[k]) is not type(v) or topology[k] != v for k, v in TOPOLOGY.items())):
        report["missing"].append("supported vLLM single-host/single-rank eager NIXL pull topology")
    if config.get("nixl_version") != "1.4.1":
        report["missing"].append("supported NIXL 1.4.1")
    revision = config.get("model_revision")
    if (config.get("model") != "Qwen/Qwen3-0.6B" or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        report["missing"].append("resolved Qwen3-0.6B model commit; main is not a pin")
    root = config.get("engine_root")
    identity = inspect_engine(base / root, config.get("engine_version")) if isinstance(root, str) else {
        "adapter_fits": False, "fix": "unknown"}
    report["engine"] = identity
    report["model_binding"] = ("config declares a resolved revision; live weights are outside this check"
                               if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
                               else "resolved model revision not recorded; live weights are outside this check")
    if not identity["adapter_fits"]:
        report["missing"].append("exact supported engine source files and version")
    events_path = config.get("events")
    events = read_events(str(base / events_path)) if isinstance(events_path, str) else []
    found = evaluate(events, [], rules=["lease_valid_at_use"])
    report["findings"] = [f.as_dict() for f in found]
    if config["schema"] == 2:
        from .client_inventory import read_inventory, reconcile
        inventory_path = config.get("client_inventory")
        try:
            inventory = read_inventory(base / inventory_path) if isinstance(inventory_path, str) else None
            report["request_coverage"] = reconcile(inventory, events)
        except (OSError, ValueError, TypeError) as exc:
            report["request_coverage"] = {"status": "unknown", "issues": ["client inventory unavailable or unsupported: " + str(exc)]}
        if report["request_coverage"]["status"] != "complete":
            report["missing"].append("complete one-to-one coverage of the client-declared workload")
    if not found:
        report["missing"].append("recorded request attempts")
    if any(f.verdict == "unknown" for f in found):
        report["missing"].append("complete unambiguous rule evidence")
    # Bind every relevant recorder stream to the inspected source and model.
    streams = {(e.get("session"), e.get("pid")) for f in found for e in f.evidence
               if isinstance(e.get("session"), str) and type(e.get("pid")) in (int, str)}
    for stream in streams:
        records = [e for e in events if (e.get("session"), e.get("pid")) == stream
                 and e.get("kind") == "note" and isinstance(e.get("args"), dict)
                 and "engine_identity" in e["args"]]
        notes = [e["args"] for e in records]
        first_seq = min((e["seq"] for f in found for e in f.evidence
                         if (e.get("session"), e.get("pid")) == stream and type(e.get("seq")) is int), default=-1)
        if (not records or not any(type(e.get("seq")) is int and e["seq"] <= first_seq for e in records)
                or any(n["engine_identity"] != identity or n.get("nixl_version") != "1.4.1" for n in notes)):
            report["missing"].append("recording bound to the inspected engine source")
    ready = config.get("ready")
    if ready is not None:
        if (not isinstance(ready, dict) or set(ready) != {"metrics_urls"}
                or not isinstance(ready["metrics_urls"], list) or not ready["metrics_urls"]):
            report["missing"].append("supported readiness configuration")
        elif run_ready:
            argv = [sys.executable, "-m", "sdcguard.ready", "--json"]
            for url in ready["metrics_urls"]:
                if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                    report["missing"].append("valid readiness metrics URL")
                    break
                argv += ["--url", url]
            else:
                try:
                    completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
                    report["readiness"] = {"exit_code": completed.returncode,
                                           "result": json.loads(completed.stdout)}
                    if completed.returncode != 0:
                        report["missing"].append("readiness checks completed successfully")
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    report["missing"].append("readiness result available")
        else:
            report["missing"].append("configured readiness checks not run")
    report["missing"] = list(dict.fromkeys(report["missing"]))
    if report["missing"]:
        return report_sections(report, config)
    report["status"] = "not fine" if any(f.verdict == "violated" for f in found) else "fine"
    return report_sections(report, config)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sdcguard assess", description=__doc__)
    p.add_argument("config")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", metavar="NEWDIR", help="save a portable local evidence bundle; never overwrite")
    a = p.parse_args(argv)
    path = Path(a.config)
    try:
        if a.out:
            from .assessment_bundle import create_bundle
            report = create_bundle(path, Path(a.out))
        else:
            report = assess(json.loads(path.read_text()), base=path.resolve().parent)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {"status": "inconclusive", "scope": SCOPE, "missing": [str(exc)]}
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Within scope: {report['status'].upper()}")
        print(f"Scope: {report['scope']}")
        print(f"Proposed PR #62065 backport in inspected source: {report.get('engine', {}).get('fix', 'unknown')}")
        for why in report.get("missing", []):
            print(f"  Inconclusive: {why}")
        for f in report.get("findings", []):
            print(f"  {f['verdict']}: {f['relationship']}")
    return {"fine": 0, "not fine": 1, "inconclusive": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
