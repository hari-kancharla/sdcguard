"""Prepare passive observation instructions without starting an inference engine."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
from typing import Mapping

from .assessment import TOPOLOGY, inspect_engine


SCOPE = "Observation instructions for the pinned #61229 topology; no runtime evidence collected"
_REQUIRED = {"schema", "engine", "engine_version", "nixl_version", "engine_root", "model",
             "model_revision", "topology"}
_OPTIONAL = {"events", "ready"}
_BLOCKED_NONEMPTY = ("SDCGUARD_LIFETIME_HOLDS", "SDCGUARD_NIXL_UCX_PARAMS", "SDCGUARD_REQUIRE")
_LIMITS = [
    "A prepared plan does not prove that the plugin loaded or that any hook ran.",
    "The model revision is declared; live model weights are not attested.",
    "Assess actual recordings for runtime hook coverage and scoped verdicts.",
    "No server, inference traffic, GPU allocation, replay, holds or transport override is started here.",
]


def _environment_problems(environ: Mapping[str, str]) -> list[str]:
    missing = []
    for name in _BLOCKED_NONEMPTY:
        if environ.get(name):
            missing.append(f"{name} is nonempty; observation-only activation refused")
    if environ.get("SDCGUARD") == "1":
        missing.append("SDCGUARD=1 already enables numerical features; observation-only activation refused")
    if "VLLM_PLUGINS" in environ:
        selection = environ["VLLM_PLUGINS"]
        if not isinstance(selection, str) or "sdcguard" not in selection.split(","):
            missing.append("VLLM_PLUGINS does not select the sdcguard entry point; loading is unverified")
    return missing


def validate_deployment(config: dict, *, base: Path = Path("."),
                        environ: Mapping[str, str] | None = None) -> dict:
    """Inspect local source and declarations, not a running server or model."""
    report = {"schema": "sdcguard.recording-plan/1", "status": "inconclusive", "scope": SCOPE,
              "missing": [], "limits": list(_LIMITS)}
    missing = report["missing"]
    if not isinstance(config, dict):
        missing.append("deployment must be a JSON object")
        return report
    if not _REQUIRED <= config.keys() or config.keys() - (_REQUIRED | _OPTIONAL):
        missing.append("supported deployment fields only; all required fields must be present")
    if type(config.get("schema")) is not int or config.get("schema") != 1:
        missing.append("supported deployment config schema 1")
    topology = config.get("topology")
    if (config.get("engine") != "vllm" or not isinstance(topology, dict)
            or topology.keys() != TOPOLOGY.keys()
            or any(type(topology[k]) is not type(value) or topology[k] != value
                   for k, value in TOPOLOGY.items())):
        missing.append("exact supported single-host/single-rank eager NIXL pull topology")
    if config.get("nixl_version") != "1.4.1":
        missing.append("declared NIXL 1.4.1")
    revision = config.get("model_revision")
    if (config.get("model") != "Qwen/Qwen3-0.6B" or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        missing.append("declared Qwen/Qwen3-0.6B with a resolved 40-hex model revision")
    if "events" in config and (not isinstance(config["events"], str) or not config["events"]):
        missing.append("events must be a nonempty path string when supplied")
    ready = config.get("ready")
    if "ready" in config and (not isinstance(ready, dict) or set(ready) != {"metrics_urls"}
            or not isinstance(ready.get("metrics_urls"), list) or not ready["metrics_urls"]
            or any(not isinstance(url, str) or not url.startswith(("http://", "https://"))
                   for url in ready["metrics_urls"])):
        missing.append("supported optional readiness metrics URLs")
    root, version = config.get("engine_root"), config.get("engine_version")
    identity = {"adapter_fits": False, "fix": "unknown"}
    if isinstance(root, str) and root and isinstance(version, str):
        try:
            engine_root = (Path(base) / root).resolve()
            identity = inspect_engine(engine_root, version)
            report["engine_root"] = str(engine_root)
        except (OSError, ValueError):
            missing.append("readable local engine source path")
    report["engine"] = identity
    if identity.get("adapter_fits") is not True:
        missing.append("exact supported engine version and complete relevant source hashes")
    missing.extend(_environment_problems(os.environ if environ is None else environ))
    report["plugin_loading"] = "unverified; generated environment requests recording only"
    if not missing:
        report["status"] = "supported"
    return report


def environment_snippet(events: Path, role: str, engine_id: str) -> str:
    """POSIX shell fragment: validate inherited settings before changing any variable."""
    if role not in ("producer", "consumer"):
        raise ValueError("role must be producer or consumer")
    if not isinstance(engine_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", engine_id) is None:
        raise ValueError("an explicit actual engine ID is required")
    guards = []
    for name in _BLOCKED_NONEMPTY:
        guards.append(f'''    if [ -n "${{{name}-}}" ]; then
        printf '%s\\n' '{name} is nonempty; observation-only activation refused' >&2
        return 2
    fi''')
    return f'''# Source this file in the {role} shell; continue only if sourcing succeeds.
# This requests observation. It does not prove plugin loading or hook execution.
_sdcguard_record_plan_{role}() {{
{chr(10).join(guards)}
    if [ "${{SDCGUARD-}}" = 1 ]; then
        printf '%s\\n' 'SDCGUARD=1 already enables numerical features; observation-only activation refused' >&2
        return 2
    fi
    if [ "${{VLLM_PLUGINS+x}}" = x ]; then
        case ",${{VLLM_PLUGINS}}," in
            *,sdcguard,*) ;;
            *) printf '%s\\n' 'VLLM_PLUGINS does not select sdcguard; activation refused' >&2; return 2 ;;
        esac
    fi
    export SDCGUARD_LIFETIME_DIR={shlex.quote(str(events))}
    export SDCGUARD_LIFETIME_ENGINE={shlex.quote(engine_id)}
    export SDCGUARD_LIFETIME_RECORD=1
}}
_sdcguard_record_plan_{role}
'''


def _write_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def prepare_recording_plan(config: dict, destination: Path, *, base: Path = Path("."),
                           environ: Mapping[str, str] | None = None,
                           producer_engine_id: str | None = None, consumer_engine_id: str | None = None) -> dict:
    """Write a new instruction directory; never overwrite or erase partial output."""
    partial_output = None
    try:
        destination = Path(destination)
        if destination.exists() or destination.is_symlink():
            return {"schema": "sdcguard.recording-plan/1", "status": "inconclusive", "scope": SCOPE,
                    "missing": ["output destination already exists; no files were overwritten"],
                    "limits": list(_LIMITS)}
        report = validate_deployment(config, base=base, environ=environ)
        if report["status"] != "supported":
            return report
        ids = {"producer": producer_engine_id, "consumer": consumer_engine_id}
        if (any(not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}", value) is None
                for value in ids.values()) or producer_engine_id == consumer_engine_id):
            report.update(status="inconclusive")
            report["missing"].append("explicit, distinct actual producer/consumer engine IDs from the existing deployment")
            return report
        report["declared_engine_ids"] = ids
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=False)
        partial_output = str(destination)
        events = destination / "events"
        events.mkdir()
        deployment = deepcopy(config)
        deployment.update(engine_root=report["engine_root"], events=str(events))
        contents = {
            "deployment.json": json.dumps(deployment, indent=2) + "\n",
            "producer.env.sh": environment_snippet(events, "producer", producer_engine_id),
            "consumer.env.sh": environment_snippet(events, "consumer", consumer_engine_id),
            "README.txt": (
                "Passive recording plan for the existing supported deployment\n\n"
                "Source producer.env.sh in the producer shell and consumer.env.sh in the consumer shell. "
                "Continue only if sourcing succeeds; use the shell's && operator when chaining commands. "
                "The supplied engine IDs must equal the actual engine_id values in those existing server commands. "
                "Role labels are not engine IDs. Start the existing supported server commands yourself, unchanged. "
                "No server command is generated.\n\n"
                "Both roles write to the same initially empty events directory. Never combine this plan's "
                "recordings with an earlier run. The plan changes only the recorder environment. "
                "It refuses nonempty SDCGUARD_LIFETIME_HOLDS, SDCGUARD_NIXL_UCX_PARAMS and SDCGUARD_REQUIRE, "
                "inherited SDCGUARD=1, or an explicit plugin selection excluding sdcguard. "
                "No variable is silently unset; resolve conflicting settings explicitly.\n\n"
                + "\n".join(_LIMITS) + "\n\n"
                "The declared model revision is not proof of loaded weights. Source matching is local file "
                "inspection, not remote attestation. A prepared plan is not a readiness or correctness verdict. "
                "Actual recordings must contain the evaluator's required identity and health receipts.\n\n"
                "After the owner-run observation is complete and recorders have flushed, assess the recording:\n"
                "sdcguard assess " + shlex.quote(str(destination / "deployment.json"))
                + " --out " + shlex.quote(str(destination / "assessment")) + " --json\n"
                + "sdcguard report-verify " + shlex.quote(str(destination / "assessment")) + "\n"
                + "sdcguard evidence collect --assessment-packet " + shlex.quote(str(destination / "assessment"))
                + " --out " + shlex.quote(str(destination / "evidence")) + "\n"
                + "sdcguard evidence verify " + shlex.quote(str(destination / "evidence")) + "\n"
            ),
        }
        for name, value in contents.items():
            _write_new(destination / name, value)
        manifest = {"schema": "sdcguard.recording-plan/1", "status": "prepared", "scope": SCOPE,
                    "engine": report["engine"], "declared_model": {
                        "model": config["model"], "revision": config["model_revision"],
                        "basis": "deployment declaration only; live weights not attested"},
                    "events_dir": str(events), "declared_engine_ids": ids, "runtime_evidence": "not_collected",
                    "plugin_loading": report["plugin_loading"], "limits": list(_LIMITS),
                    "files_sha256": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
                                     for name in contents}}
        # The manifest is written last; absent manifest means preparation did not finish.
        _write_new(destination / "PLAN_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
        report.update(status="prepared", destination=str(destination), events_dir=str(events),
                      manifest=str(destination / "PLAN_MANIFEST.json"))
        return report
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"schema": "sdcguard.recording-plan/1", "status": "inconclusive", "scope": SCOPE,
                "missing": [f"plan preparation incomplete: {exc}"], "limits": list(_LIMITS),
                "partial_output": partial_output}


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sdcguard record-plan", description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--producer-engine-id", help="actual existing producer engine_id; required to prepare")
    parser.add_argument("--consumer-engine-id", help="actual existing consumer engine_id; required to prepare")
    args = parser.parse_args(argv)
    try:
        report = prepare_recording_plan(json.loads(args.config.read_text()), args.out,
                                        base=args.config.resolve().parent,
                                        producer_engine_id=args.producer_engine_id, consumer_engine_id=args.consumer_engine_id)
    except (OSError, ValueError, TypeError) as exc:
        report = {"status": "inconclusive", "scope": SCOPE, "missing": [str(exc)], "limits": list(_LIMITS)}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Recording plan: {report['status'].upper()}")
        print(f"Scope: {SCOPE}")
        if report.get("destination"):
            print(f"Instructions: {report['destination']}")
        for why in report.get("missing", []):
            print(f"  Inconclusive: {why}")
        for limit in report["limits"]:
            print(f"  {limit}")
    return 0 if report["status"] == "prepared" else 2


if __name__ == "__main__":
    raise SystemExit(main())
