"""Bridge saved recording packets through the existing offline assessment.

Owner: Hari Krishna Kancharla.
Only local captured bytes are read. A private temporary copy isolates the
existing verifier from changes to the supplied packet. Engine source members
are inspected as bytes, never imported or executed. No readiness endpoint is
refreshed, and no recording, replay, request or engine is started.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
from tempfile import TemporaryDirectory

from . import assessment_bundle
from . import config_inputs
from . import evidence_catalog
from . import evidence_io
from .lifetime.obligations import VERDICTS


EVIDENCE_KIND = "recorded_engine_events"
RULE_ID = "lease_valid_at_use"
SCENARIO = "recorded-source-lease"
MAX_FILES = 4096
MAX_DIRECTORIES = 4096
MAX_DEPTH = 64
MAX_FILE_BYTES = assessment_bundle.MAX_FILE_BYTES
MAX_TOTAL_BYTES = assessment_bundle.MAX_TOTAL_BYTES
_NATIVE_KEYS = {"rule", "request", "verdict", "protection", "assumption", "relationship",
                "evidence", "missing", "detail", "ordering", "not_observed"}


class RecordedEvidenceError(ValueError):
    """The saved packet cannot support a normalized evidence result."""


def _strict_json(raw, name):
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=config_inputs._no_duplicates,
                           parse_constant=config_inputs._invalid_constant)
        budget = (config_inputs.MAX_JSON_NODES if name in ("config.json", "evidence/config-original.json")
                  else evidence_io.MAX_JSON_NODES)
        config_inputs._json_values(value, max_nodes=budget)
        return value
    except (UnicodeError, ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise RecordedEvidenceError("invalid captured JSON: " + name) from exc


def _relative(parts):
    name = "/".join(parts)
    assessment_bundle._relative(name)
    return name


def _snapshot(directory):
    """Capture bounded stable regular files without following any path links."""
    root = Path(directory).absolute()
    descriptor = config_inputs._directory(root)
    before = os.fstat(descriptor)
    files, total, directories = {}, 0, 0
    directory_names = []

    def walk(fd, prefix):
        nonlocal total, directories
        if len(prefix) > MAX_DEPTH:
            raise RecordedEvidenceError("packet directory depth exceeds supported bound")
        signature = config_inputs._signature(os.fstat(fd))
        for name in sorted(os.listdir(fd)):
            parts = prefix + (name,)
            relative = _relative(parts)
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                directories += 1
                directory_names.append(relative)
                if directories > MAX_DIRECTORIES:
                    raise RecordedEvidenceError("packet directory count exceeds supported bound")
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    if config_inputs._signature(os.fstat(child)) != config_inputs._signature(info):
                        raise RecordedEvidenceError("packet directory changed while being captured")
                    walk(child, parts)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                if len(files) >= MAX_FILES:
                    raise RecordedEvidenceError("packet file count exceeds supported bound")
                raw = config_inputs._read_regular(root, parts, MAX_FILE_BYTES)
                total += len(raw)
                if total > MAX_TOTAL_BYTES:
                    raise RecordedEvidenceError("packet byte count exceeds supported bound")
                files[relative] = raw
            else:
                raise RecordedEvidenceError("packet member is a link or non-regular file")
        if config_inputs._signature(os.fstat(fd)) != signature:
            raise RecordedEvidenceError("packet membership changed while being captured")

    try:
        walk(descriptor, ())
        named = os.stat(root, follow_symlinks=False)
        if (not stat.S_ISDIR(named.st_mode)
                or config_inputs._signature(before) != config_inputs._signature(os.fstat(descriptor))
                or config_inputs._signature(before) != config_inputs._signature(named)):
            raise RecordedEvidenceError("packet root changed while being captured")
    finally:
        os.close(descriptor)
    if "manifest.json" not in files:
        raise RecordedEvidenceError("saved assessment manifest is missing")
    return files, tuple(sorted(directory_names))


def _validate_json_members(snapshot):
    parsed = {}
    for name, raw in snapshot.items():
        if name.endswith(".json"):
            parsed[name] = _strict_json(raw, name)
        elif name.endswith(".jsonl"):
            for number, line in enumerate(raw.splitlines(keepends=True), 1):
                if line.strip():
                    if not line.endswith(b"\n"):
                        raise RecordedEvidenceError("captured recording has an unfinished line")
                    row = _strict_json(line, name + ":" + str(number))
                    if not isinstance(row, dict):
                        raise RecordedEvidenceError("captured event must be a JSON object")
    return parsed


def _digest_snapshot(snapshot, directories):
    inventory = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                 for name, raw in sorted(snapshot.items())}
    return hashlib.sha256(json.dumps({"files": inventory, "directories": directories}, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _strings(value, name):
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise RecordedEvidenceError("invalid " + name)
    return value


def _native_finding(value):
    if not isinstance(value, dict) or set(value) != _NATIVE_KEYS:
        raise RecordedEvidenceError("saved native finding has unsupported fields")
    if (value["rule"] != RULE_ID or value["verdict"] not in VERDICTS
            or not isinstance(value["request"], str) or not value["request"]):
        raise RecordedEvidenceError("saved native finding identity or verdict is unsupported")
    _strings(value["missing"], "native missing facts")
    _strings(value["not_observed"], "native observation limits")
    if (not isinstance(value["evidence"], list)
            or any(not isinstance(item, dict) for item in value["evidence"])):
        raise RecordedEvidenceError("saved native evidence references are malformed")
    for key in ("assumption", "relationship", "detail", "ordering"):
        if not isinstance(value[key], str):
            raise RecordedEvidenceError("invalid native finding " + key)
    if value["protection"] is not None and not isinstance(value["protection"], str):
        raise RecordedEvidenceError("invalid native protection")
    return value


def _canonical_findings(report, configuration):
    """Preserve native outcomes while exposing the whole packet's scope gaps."""
    definition = evidence_catalog.describe(RULE_ID, EVIDENCE_KIND)
    global_missing = _strings(report.get("missing"), "legacy report missing facts")
    if report.get("status") not in ("fine", "not fine", "inconclusive"):
        raise RecordedEvidenceError("legacy scope verdict is unsupported")
    if report["status"] == "inconclusive" and not global_missing:
        global_missing = ["legacy assessment remains inconclusive"]
    native = report.get("findings")
    if not isinstance(native, list):
        raise RecordedEvidenceError("legacy finding inventory is missing")
    checked = [_native_finding(value) for value in native]
    requests = [value["request"] for value in checked]
    if len(set(requests)) != len(requests):
        raise RecordedEvidenceError("duplicate native request finding")
    if not checked:
        checked = [None]
    findings = []
    for value in checked:
        missing = list(dict.fromkeys(global_missing + (value["missing"] if value else
                                                       ["recorded request attempts"])))
        original = value["verdict"] if value else "unknown"
        # A complete per-request history cannot replace a missing source/model
        # binding required by the enclosing supported-deployment assessment.
        verdict = "unknown" if global_missing or value is None else original
        facts = {"native_finding": deepcopy(value) if value else {},
                 "native_verdict": original, "legacy_scope_verdict": report["status"],
                 "global_missing": list(global_missing),
                 "engine_identity": deepcopy(report.get("engine", {})),
                 "configuration": deepcopy(configuration),
                 "model_binding": report.get("model_binding"),
                 "runtime_model_use": "not observed",
                 "answer_correctness": "not judged",
                 "authenticity": "not established"}
        finding = {"rule_id": RULE_ID, "semantic_revision": definition["semantic_revision"],
                   "subject": "request:" + value["request"] if value else "recording:unattributed",
                   "scenario": SCENARIO, "evidence_kind": EVIDENCE_KIND,
                   "scope": definition["scope"], "resource_kind": definition["resource_kind"],
                   "verdict": verdict, "facts": facts, "missing": missing,
                   "evidence": deepcopy(value["evidence"]) if value else []}
        errors = evidence_catalog.validate_finding(finding, EVIDENCE_KIND)
        if errors:
            raise RecordedEvidenceError("recorded finding violates catalog: " + "; ".join(errors))
        findings.append(finding)
    return findings


def normalize(directory):
    """Verify a saved packet and return its scoped canonical recorded findings.

    The input is unchanged. Verification and interpretation use a private copy
    of captured bytes, followed by a complete original-packet stability check.
    Plain hashes establish consistency only, not producer authenticity.
    """
    try:
        snapshot, directories = _snapshot(directory)
        parsed = _validate_json_members(snapshot)
        report = parsed.get("report.json")
        configuration = parsed.get("config.json")
        if not isinstance(report, dict) or not isinstance(configuration, dict):
            raise RecordedEvidenceError("legacy report and portable configuration are required")
        with TemporaryDirectory(prefix="sdcguard-recorded-") as temporary:
            local = Path(temporary).resolve()
            for name in directories:
                local.joinpath(*name.split("/")).mkdir(parents=True, exist_ok=True, mode=0o700)
            for name, raw in snapshot.items():
                destination = local.joinpath(*name.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with destination.open("xb") as stream:
                    stream.write(raw)
            verified = assessment_bundle.verify_bundle(local)
            if verified.get("integrity") != "verified" or verified.get("core_facts_recomputed") is not True:
                raise RecordedEvidenceError("legacy packet verification failed: "
                                            + "; ".join(verified.get("errors", [])))
            if verified.get("scope_verdict") != report.get("status"):
                raise RecordedEvidenceError("verified scope verdict differs from captured report")
            findings = _canonical_findings(report, report.get("configuration", configuration))
            repeated = assessment_bundle.verify_bundle(local)
            if repeated != verified or _snapshot(local) != (snapshot, directories):
                raise RecordedEvidenceError("private assessment copy changed during interpretation")
        if _snapshot(directory) != (snapshot, directories):
            raise RecordedEvidenceError("saved packet changed during interpretation")
        result = {"findings": findings,
                "identity": {"engine": configuration.get("engine"),
                             "engine_version": configuration.get("engine_version"),
                             "nixl_version": configuration.get("nixl_version"),
                             "model": {"id": configuration.get("model"),
                                       "revision": configuration.get("model_revision"),
                                       "live_weights_verified": False},
                             "topology": deepcopy(configuration.get("topology")),
                             "source_identity": deepcopy(report.get("engine", {})),
                             "configuration": deepcopy(report.get("configuration", {})),
                             "packet_snapshot_sha256": _digest_snapshot(snapshot, directories),
                             "identity_basis": "captured local source and recorded declarations; no remote attestation",
                             "kernel": "not observed"},
                "legacy_scope_verdict": report["status"],
                "legacy_verification": deepcopy(verified)}
        if report.get("schema") == 2:
            result["request_coverage"] = deepcopy(report["request_coverage"])
        return result
    except RecordedEvidenceError:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError) as exc:
        raise RecordedEvidenceError("saved recording packet is unsupported or unstable: " + str(exc)) from exc
