"""Canonical findings from the existing pinned, ordinary CPU components.

Collection invokes only the trusted packaged component runner. Normalization
does not execute fragments or caller-provided code: it validates the saved
receipt, translates raw observations and recomputes the existing generic rule.
This portable profile requires both reviewed engines. A single-engine result
can still be collected for standalone inspection but is not this full profile.

Saved observations do not authenticate execution. Process/session/pool IDs are
opaque and differ on a fresh collection; fresh execution is a separate witness,
not a byte-for-byte replay or a physical-write/isolation attestation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

from .lifetime import component_conformance as component
from .lifetime.generation_obligations import SCOPE, evaluate_generation_obligations
from .lifetime.generation_observations import (
    SOURCE_PINS, sglang_component_observations, vllm_probe_observations,
)

RULE_ID = "destination_incarnation_retired_after_writers"
EVIDENCE_KIND = "source_fragment_cpu"
_ENGINES = frozenset({"vllm", "sglang"})
_ROOT_KEYS = {"schema", "scope", "physical_write_safety", "evidence_origin", "status", "engines"}
_ITEM_KEYS = {"source", "assessment", "observation_count", "receipt"}
_ENVELOPE_KEYS = {"t", "mono", "host", "clock_id", "session", "pid", "tid", "seq",
                  "dropped", "kind", "args", "text"}
_BASE_METADATA = {"scope", "source", "complete_from_start", "closed"}
_PROFILE = {
    "vllm": dict(resource_kind="destination_kv_allocation", generation_basis="observed_allocation",
                 scenario="ordinary_destination_read_done_release",
                 proof_domain="CPU allocation and destination transport receipts",
                 metadata={"manager_engine", "worker_engine", "pool", "single_cache_group"},
                 kinds={"component_start", "component_end", "alloc", "free", "note", "evidence",
                        "submit", "complete", "publish", "accept", "checkpoint"},
                 writers=["transport"]),
    "sglang": dict(resource_kind="protocol_held_auxiliary_metadata", generation_basis="protocol_hold",
                   scenario="ordinary_two_peer_matching_generation_auxiliary_release",
                   proof_domain="protocol manager pending-writer bookkeeping",
                   metadata={"metadata_pool"},
                   kinds={"component_start", "component_end", "hold", "drain_ack", "metadata_free",
                          "release_outcome", "observation"}, writers=["0", "1"]),
}


def collect(engine="all"):
    """Run the existing verified ordinary CPU path, accepting no source path."""
    if not isinstance(engine, str) or engine not in ("all", "vllm", "sglang"):
        raise ValueError("unsupported component selection")
    return component.run_component_conformance(engine)


def _json_copy(value):
    """Keep existing tuple/list observations portable, refusing non-JSON facts."""
    def check(item):
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("non-finite component fact")
            return
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("component fact keys must be strings")
            for child in item.values():
                check(child)
            return
        if type(item) in (list, tuple):
            for child in item:
                check(child)
            return
        raise ValueError("component facts must be JSON values")
    try:
        check(value)
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("component receipt is not finite JSON") from exc


def _keys(value, expected, what):
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("malformed " + what + " fields")


def _receipt(engine, item):
    _keys(item, _ITEM_KEYS, "component result")
    if item["source"] != SOURCE_PINS[engine]:
        raise ValueError("unsupported component source pin or source-file hashes")
    receipt = item["receipt"]
    _keys(receipt, {"metadata", "observations"}, "component receipt")
    metadata, observations = receipt["metadata"], receipt["observations"]
    profile = _PROFILE[engine]
    if (not isinstance(metadata, dict) or set(metadata) - (_BASE_METADATA | profile["metadata"])
            or metadata.get("scope") != SCOPE or metadata.get("source") != SOURCE_PINS[engine]):
        raise ValueError("unsupported component metadata or source identity")
    for key in ("complete_from_start", "closed", "single_cache_group"):
        if key in metadata and metadata[key] is not None and type(metadata[key]) is not bool:
            raise ValueError("malformed component coverage declaration")
    for key in ("manager_engine", "worker_engine", "pool", "metadata_pool"):
        if key in metadata and metadata[key] is not None and not isinstance(metadata[key], str):
            raise ValueError("malformed component resource namespace")
    if not isinstance(observations, list):
        raise ValueError("component observations must be a list")
    if type(item["observation_count"]) is not int or item["observation_count"] != len(observations):
        raise ValueError("component observation count does not match raw receipts")
    for event in observations:
        if (not isinstance(event, dict) or set(event) - _ENVELOPE_KEYS
                or not isinstance(event.get("kind"), str) or event["kind"] not in profile["kinds"]):
            raise ValueError("unsupported component observation envelope or kind")
        if "args" in event and not isinstance(event["args"], dict):
            raise ValueError("component observation args must be an object")
        for key in ("t", "mono"):
            if key in event and event[key] is not None and type(event[key]) not in (int, float):
                raise ValueError("malformed component clock value")
        for key in ("pid", "tid", "seq", "dropped"):
            if key in event and event[key] is not None and type(event[key]) is not int:
                raise ValueError("malformed component integer envelope")
        for key in ("host", "clock_id", "session", "text"):
            if key in event and event[key] is not None and not isinstance(event[key], str):
                raise ValueError("malformed component text envelope")
    return receipt


def normalize(result):
    """Re-evaluate the full existing two-engine profile; refuse integrity errors.

    Missing proof facts retain the trusted evaluator's unknown verdict. Bad
    schemas, pins, unsupported inventory and contradictory derived claims raise
    ValueError. A caller's run flag or supplied source code is never accepted.
    """
    result = _json_copy(result)
    _keys(result, _ROOT_KEYS, "component bundle")
    if (result["schema"] != "sdcguard.component-conformance.v1" or result["scope"] != SCOPE
            or result["physical_write_safety"] != "unknown"
            or result["evidence_origin"] != "pinned_original_functions_with_cpu_mocks"
            or not isinstance(result["engines"], dict) or set(result["engines"]) != _ENGINES):
        raise ValueError("unsupported component schema, scope or complete engine inventory")
    # Verify the installed trusted inventory, never a source supplied in result.
    component._verified_bundle()
    from .evidence_catalog import describe, validate_finding

    catalog = describe(RULE_ID, EVIDENCE_KIND)
    if (type(catalog.get("semantic_revision")) is not int or catalog["semantic_revision"] != 1
            or catalog.get("scope") != SCOPE):
        raise ValueError("component rule catalog mapping changed")
    findings, statuses = [], []
    for engine in ("vllm", "sglang"):
        item = result["engines"][engine]
        receipt = _receipt(engine, item)
        translate = vllm_probe_observations if engine == "vllm" else sglang_component_observations
        translated = translate(receipt["observations"], receipt["metadata"])
        assessment = evaluate_generation_obligations(translated).as_dict()
        if item["assessment"] != assessment:
            raise ValueError("claimed component assessment disagrees with raw re-evaluation")
        profile = _PROFILE[engine]
        if profile["resource_kind"] not in catalog.get("allowed_resource_kinds", []):
            raise ValueError("component resource semantics are not catalogued")
        if assessment["status"] == "satisfied":
            obligations = assessment["obligations"]
            if (len(obligations) != 1 or obligations[0]["writers"] != profile["writers"]
                    or obligations[0]["generation_basis"] != profile["generation_basis"]):
                raise ValueError("receipt is outside the reviewed ordinary component profile")
        statuses.append(assessment["status"])
        facts = dict(engine=engine, engine_version="unknown", source=deepcopy(SOURCE_PINS[engine]),
                     source_manifest_sha256=component.MANIFEST_SHA256,
                     obligations=assessment["obligations"], physical_write_safety="unknown",
                     generation_basis=profile["generation_basis"], observation_count=item["observation_count"],
                     coverage=translated["coverage"], proof_domain=profile["proof_domain"],
                     receipt_authentication="not established from saved observations")
        finding = dict(rule_id=RULE_ID, semantic_revision=1, subject={"engine": engine},
            scenario=profile["scenario"], evidence_kind=EVIDENCE_KIND, scope=SCOPE,
            resource_kind=profile["resource_kind"], verdict=assessment["status"], facts=facts,
            missing=list(assessment["reasons"]), evidence=dict(receipt=receipt,
                translated_observations=translated, source_manifest_sha256=component.MANIFEST_SHA256))
        errors = validate_finding(finding, EVIDENCE_KIND)
        if errors:
            raise ValueError("canonical component finding does not fit its catalog: " + "; ".join(errors))
        findings.append(finding)
    expected_status = "satisfied" if all(status == "satisfied" for status in statuses) else "unknown"
    if result["status"] != expected_status:
        raise ValueError("claimed bundle status disagrees with raw re-evaluation")
    return findings
