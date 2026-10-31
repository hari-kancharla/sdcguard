"""Recompute closed native mapping packets for a portable evidence section.

Owner: Hari Krishna Kancharla.
Captured files are data. The installed mapping evaluator supplies the verdict.
This module does not activate an engine or establish read permission.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from . import assignment_report as native
from . import evidence_io as io
from . import evidence_json_budget as json_budget
from . import evidence_native_io as bounded_io

SCHEMA = "sdcguard.native-mapping-section.v1"
EVIDENCE_KIND = "recorded_model_entry_metadata"
SCOPE = "recorded_allocator_to_model_entry_mapping"
RESOURCE = "cache_assignment_incarnation"
VERDICTS = ("satisfied", "violated", "unknown")
LIMITS = (
    "Only the existing allocator-to-model-entry mapping relation is judged.",
    "Recorded identities and local close receipts are not runtime authentication.",
    "An allocation ordinal is not a native content generation or lease nonce.",
    "Mapping agreement does not establish ownership, content or physical reads.",
    "Separate sessions are not joined by request text, block number or time.",
    "A copied packet is unchanged evidence, not a new execution witness.",
)


class NativeMappingEvidenceError(io.EvidenceError):
    pass


def _require(condition, message):
    if not condition:
        raise NativeMappingEvidenceError(message)


def _validate_snapshot(snapshot, directories):
    _require(type(snapshot) is dict and 0 < len(snapshot) <= io.MAX_FILES,
             "native packet file inventory is missing or exceeds its bound")
    _require(type(directories) is list and len(directories) < io.MAX_FILES
             and all(type(name) is str for name in directories)
             and directories == sorted(set(directories)),
             "native packet directory inventory is malformed")
    required_directories = set()
    for name in directories:
        parts = io.relative(name)
        _require(len(parts) <= bounded_io.MAX_DIRECTORY_DEPTH,
                 "native packet directory depth exceeds its bound")
        required_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
        _require(name not in snapshot, "native file and directory identities conflict")
    total = 0
    for name, raw in snapshot.items():
        parts = io.relative(name)
        _require(len(parts) <= bounded_io.MAX_DIRECTORY_DEPTH + 1,
                 "native packet file depth exceeds its bound")
        _require(type(raw) is bytes and len(raw) <= io.MAX_FILE_BYTES,
                 "native packet contains an unsupported member")
        total += len(raw)
        _require(total <= io.MAX_TOTAL_BYTES, "native packet exceeds its byte bound")
        required_directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    _require(required_directories <= set(directories),
             "native packet parent directories are missing")
    json_budget.count_members(snapshot)
    manifest = io.parse_json(snapshot.get("manifest.json", b""))
    _require(type(manifest) is dict and manifest.get("schema") == native.NATIVE_FORMAT,
             "native evidence requires a completed native packet with close receipts")


def snapshot_identity(snapshot, directories):
    _validate_snapshot(snapshot, directories)
    return io.digest(io.json_bytes({"files": io.file_receipts(snapshot),
                                    "directories": directories}))


def _project(report, packet_identity, packet_index):
    _require(report.get("schema") == native.NATIVE_REPORT
             and report.get("property") == native.contract.PROPERTY
             and report.get("recording_bookkeeping") == "locally consistent",
             "native report does not have the required mapping and close scope")
    _require(all(report.get(key) == value for key, value in (
        ("ownership_verdict", "unknown"), ("physical_read_verdict", "unknown"),
        ("content_provenance", "unknown"), ("client_attribution", "unknown"),
        ("runtime_isolation", "not established"), ("authenticity", "not established"))),
        "native evidence scope differs from this adapter")
    findings = []
    for index, entry in enumerate(report["sessions"]):
        assessment = entry["assessment"]
        verdict = assessment["verdict"]
        _require(verdict in VERDICTS and assessment["property"] == native.contract.PROPERTY,
                 "native evaluator returned an unsupported rule or verdict")
        facts = {key: deepcopy(assessment[key]) for key in (
            "summary", "allocation_incarnations", "closed_frames", "complete_frames",
            "findings_omitted", "findings_complete")}
        facts.update(witnesses_omitted=assessment.get("witnesses_omitted"),
                     witnesses_complete=assessment.get("witnesses_complete"),
                     counter_basis="recomputed evaluator counters; not missing runtime observations")
        facts.update(native_verdict=verdict, recording_bookkeeping="locally consistent",
                     recorded_identity_sha256=io.digest(io.json_bytes(entry["recorded_identity"])),
                     identity_basis="digest of recorded fields; not authenticated deployment identity",
                     ownership_verdict="unknown", physical_read_verdict="unknown",
                     content_provenance="unknown", client_attribution="unknown")
        missing = list(assessment["missing"])
        if verdict == "unknown" and not missing:
            missing.append("The native evaluator did not establish this named mapping contract.")
        findings.append({
            "rule_id": native.contract.PROPERTY, "semantic_revision": 1,
            "evidence_kind": EVIDENCE_KIND, "scope": SCOPE, "resource_kind": RESOURCE,
            "subject": {"packet": packet_index, "session": entry["session"], "file": entry["file"]},
            "scenario": "recorded-model-entry-mapping", "verdict": verdict,
            "required": True, "facts": facts, "missing": missing,
            "evidence": {"packet_snapshot_sha256": packet_identity,
                         "session_index": index, "input_file_sha256": entry["sha256"],
                         "full_report": "retained native mapping report"},
        })
    _require(bool(findings), "native packet contains no mapping subject")
    return findings


def normalize_snapshot(snapshot, directories, *, packet_index=0):
    """Re-run the installed native verifier from retained bytes, never a flag."""
    _require(type(packet_index) is int and 0 <= packet_index < 16,
             "native packet index is outside its supported bound")
    identity = snapshot_identity(snapshot, directories)
    with TemporaryDirectory(prefix="sdcguard-native-evidence-") as temporary:
        root = Path(temporary).resolve()
        for name in directories:
            (root / name).mkdir(parents=True, exist_ok=True)
        io.write_snapshot(root, snapshot)
        _require(io.names(root, directories=True) == directories,
                 "native packet directory inventory does not reproduce")
        checked = native.verify_packet(root)
    _require(checked.get("integrity") == "verified",
             "native packet cannot be reverified: "
             + str(checked.get("reason", "unknown reason")).replace(str(root), "captured-native"))
    report = checked["report"]
    findings = _project(report, identity, packet_index)
    return {"schema": SCHEMA, "packet_snapshot_sha256": identity,
            "findings": findings, "scope_verdict": report["scope_verdict"],
            "native_report": deepcopy(report), "limits": list(LIMITS),
            "ownership_verdict": "unknown", "physical_read_verdict": "unknown",
            "content_provenance": "unknown", "client_attribution": "unknown",
            "runtime_isolation": "not established", "authenticity": "not established",
            "runtime_upgrade_witness": "not established"}


def capture(directory, *, packet_index=0):
    """Capture a stable, regular local tree and retain its exact original bytes."""
    root = Path(directory).absolute()
    snapshot, directories = bounded_io.capture_snapshot(root)
    normalized = normalize_snapshot(snapshot, directories, packet_index=packet_index)
    bounded_io.recheck_snapshot(root, snapshot, directories)
    return {"files": snapshot, "directories": directories, "normalized": normalized}
