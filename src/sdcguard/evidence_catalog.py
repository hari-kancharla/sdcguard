"""Immutable metadata for existing evidence rules, with shape validation only.

The original evaluators remain the sole source of security verdicts. A valid
finding here has the catalog's shape and scope; this module never establishes
that its supplied facts prove a verdict or that a deployment was authenticated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType


SCHEMA = "sdcguard.evidence-catalog.v1"
SEMANTIC_REVISION = 1
_STATIC_SCOPE = "static_configuration_and_saved_startup"
_RECORDED_SCOPE = "source_lifetime_and_scheduler_admission"
_STATIC_VERDICTS = ("satisfied", "violated", "unknown")
_STATIC_LIMITS = (
    "Facts are supplied through the normalized deployment wrapper, not arbitrary native launch arguments.",
    "Declared completeness permits mapped source defaults; it does not establish actual process settings.",
    "A satisfied static contract does not prove request isolation, current runtime state or authenticated identity.",
)


@dataclass(frozen=True)
class _Rule:
    rule_id: str
    evidence_kind: str
    scope: str
    resource_kind: str
    allowed_verdicts: tuple[str, ...]
    kind: str
    required_fact_keys: tuple[str, ...]
    assumptions: tuple[str, ...]
    missing_facts: tuple[str, ...]
    state_fields: tuple[str, ...] = ()
    allowed_resource_kinds: tuple[str, ...] = ()
    state_values: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fact_constants: tuple[tuple[str, object], ...] = ()
    resource_generation_basis: tuple[tuple[str, str], ...] = ()
    expected_when: str = "when emitted by the original evaluator"


def _static(rule_id, resource_kind, kind, required, missing, *, verdicts=_STATIC_VERDICTS,
            states=(), values=(), constants=(), expected_when="when emitted for the applicable worker profile"):
    return _Rule(rule_id, "static_config", _STATIC_SCOPE, resource_kind, verdicts, kind,
                 required, _STATIC_LIMITS, missing, states, (), values,
                 (("runtime_execution", "not established"),) + constants, (), expected_when)


_RULES = (
    _Rule(
        "lease_valid_at_use", "recorded_engine_events", _RECORDED_SCOPE, "source_kv_allocation",
        ("satisfied", "violated", "correctly_rejected", "recovered", "refused", "unknown", "not_exercised"),
        "obligation", ("native_finding", "native_verdict", "engine_identity", "configuration", "model_binding"),
        ("Recorded identities, ownership and transport observations describe the exact attempt.",
         "Positive findings require complete same-host recordings, comparable clocks and health receipts.",
         "Scheduler admission is the closest observed event; actual model reads and answer correctness are outside scope.",
         "Protocol rejection, independently recorded recovery and client refusal remain distinct outcomes."),
        ("Exact attempt and publication attribution", "Complete healthy source and consumer streams",
         "Source ownership or an attributed lease check", "Scheduler admission or observed closure",
         "Independent completion evidence before reporting recovery"),
        ("native_verdict", "native_finding.protection"),
        expected_when="for each attempt selected by the existing recorded-event evaluator",
    ),
    _Rule(
        "destination_incarnation_retired_after_writers", "source_fragment_cpu", "source_fragment_cpu",
        "destination_incarnation", ("satisfied", "unknown"), "obligation",
        ("obligations", "physical_write_safety", "generation_basis"),
        ("Only reviewed source-pinned component functions and normalized CPU observations are covered.",
         "Complete closed observations bind the destination incarnation, writer set, terminal fences and release.",
         "A protocol-held auxiliary metadata slot is not a physical KV allocation generation.",
         "A timeout is not an observed drain fence; physical write safety remains unknown."),
        ("Pinned source identity and fresh component boundaries", "Complete lossless comparable observation streams",
         "Attributed destination incarnation and complete writer set", "Terminal fence for each writer before release"),
        ("generation_basis",), ("destination_kv_allocation", "protocol_held_auxiliary_metadata"),
        (("generation_basis", ("observed_allocation", "protocol_hold")),),
        (("physical_write_safety", "unknown"),),
        (("destination_kv_allocation", "observed_allocation"),
         ("protocol_held_auxiliary_metadata", "protocol_hold")),
        "for each reviewed engine component in the CPU conformance result",
    ),
    _static(
        "capacity.planned", "configured_kv_pool", "obligation",
        ("advertised_context_tokens", "configured_context_tokens", "block_or_page_size_tokens"),
        ("Reviewed full-attention capacity mapping", "Exact context and page/block units",
         "Configured pool target or a sound capacity bound", "Reserved blocks and cache-group geometry"),
        states=("state", "advertised_context_tokens", "configured_context_tokens", "usable_capacity_tokens",
                "configured_pool_upper_bound_tokens", "configured_total_blocks", "reserved_null_blocks"),
        expected_when="for every supplied worker, including unknown unsupported profiles",
    ),
    _static(
        "capacity.startup", "saved_startup_kv_pool", "obligation",
        ("advertised_context_tokens", "configured_context_tokens", "block_or_page_size_tokens"),
        ("Bound exact saved startup bytes", "Unambiguous context and capacity receipt",
         "The exact release's planning or finalized-capacity log interpretation"),
        states=("advertised_context_tokens", "startup_pool_tokens", "startup_context_tokens",
                "usable_capacity_tokens", "logged_equivalent_capacity_tokens"),
        expected_when="for every supplied worker; planning receipts are not relabelled completed allocations",
    ),
    _static(
        "protection.retirement", "retirement_source_capability", "capability",
        ("state", "runtime_execution", "identity_basis"),
        ("Exact engine/component and overlay identities", "Applicable connector direction and source capability",
         "Known required verification setting when the source contract has such a setting"),
        states=("state", "fix_status", "source_commit", "runtime_patch_sha256", "plugin_commit",
                "vllm_commit", "scope", "peer_protocol"),
    ),
    _static(
        "protection.deferred_release", "configured_destination_retention", "obligation",
        ("state", "backend", "deferred_timeout_seconds", "timeout_is_drain_proof"),
        ("Reviewed backend and deferred-release setting", "Finite positive declared timeout",
         "Explicit environment value or source default from declared complete input"),
        states=("state", "backend", "deferred_timeout_seconds"),
        values=(("state", ("on", "off", "unknown")),), constants=(("timeout_is_drain_proof", False),),
    ),
    _static(
        "protection.failed_send_drain", "failed_send_source_hardening", "capability",
        ("state", "backend"),
        ("Exact backend applicability", "Named failed-send source hardening presence or absence"),
        verdicts=("satisfied", "violated", "unknown", "not_applicable"),
        states=("state", "backend", "missing_commit"),
        expected_when="for SGLang profiles; backend-specific non-applicability remains explicit",
    ),
    _static(
        "protection.checksum_configured", "handoff_checksum_configuration", "inventory",
        ("state", "runtime_execution", "identity_basis"),
        ("Known explicit checksum setting or a mapped source default from complete input",),
        verdicts=("satisfied", "unknown"), states=("state", "architecture_coverage", "both_roles_required"),
        values=(("state", ("on", "off")),),
    ),
    _static(
        "protection.canary_configured", "canary_configuration", "inventory",
        ("state", "mode", "real_kv_mode", "runtime_execution", "identity_basis"),
        ("Known canary action and real-KV mode", "Explicit settings or mapped defaults from complete input"),
        verdicts=("satisfied", "unknown"), states=("state", "mode", "real_kv_mode", "real_kv_basis"),
        values=(("state", ("on", "off")), ("mode", ("none", "log", "raise")),
                ("real_kv_mode", ("none", "partial", "all"))),
    ),
    _static(
        "detection.architecture_coverage", "source_detector_architecture_coverage", "capability",
        ("architecture", "declared_pool", "mode", "real_kv_mode", "coverage"),
        ("Exact architecture and declared cache-pool combination", "Reviewed source attacher and known detector mode"),
        verdicts=("satisfied", "unknown"),
        states=("architecture", "declared_pool", "mode", "real_kv_mode", "coverage", "scope"),
    ),
    _static(
        "compatibility.pairing", "connector_source_layout_pair", "obligation",
        ("vllm_commit", "lmcache_commit", "connector", "kv_layout", "state"),
        ("Exact engine and cache-component source pair", "Native/external connector selector",
         "Applicable packed/separate layout, paging and disk-tier facts"),
        states=("state", "connector", "use_native", "kv_layout", "has_mamba_layers", "kv_spec_kind",
                "slot_compression", "tensor_rank", "kernel_block_tokens", "scheduler_block_tokens"),
    ),
    _static(
        "deployment.peer_profile", "declared_peer_source_profile", "capability", (),
        ("Known reviewed source and connector identities for every peer",),
        verdicts=("satisfied", "unknown"), expected_when="once per static deployment report",
    ),
    _static(
        "deployment.disaggregation_enable_kv_checksum_agreement", "peer_checksum_configuration", "obligation",
        ("values",), ("Known checksum setting for each applicable prefill/decode worker",),
        states=("values",), expected_when="when the existing SGLang peer-setting mapping applies",
    ),
    _static(
        "deployment.disaggregation_decode_allocation_policy_agreement", "peer_allocation_configuration", "obligation",
        ("values",), ("Known allocation policy for each applicable prefill/decode worker",),
        states=("values",), expected_when="when the existing SGLang peer-setting mapping applies",
    ),
    _static(
        "deployment.roles", "deployment_role_coverage", "obligation", (),
        ("Both prefill and decode workers",), verdicts=("unknown",),
        expected_when="only when the original report emits missing role coverage",
    ),
    _static(
        "deployment.context", "advertised_context_declaration", "obligation", (),
        ("Advertised endpoint context",), verdicts=("unknown",),
        expected_when="only when the original report emits missing advertised context",
    ),
)


def _check_definition(rule):
    """Refuse contradictory or duplicated catalog definitions at construction."""
    scopes = {"static_config": _STATIC_SCOPE, "recorded_engine_events": _RECORDED_SCOPE,
              "source_fragment_cpu": "source_fragment_cpu"}
    if rule.evidence_kind not in scopes or rule.scope != scopes[rule.evidence_kind]:
        raise ValueError("catalog scope does not match its evidence domain")
    if rule.kind not in ("obligation", "inventory", "capability"):
        raise ValueError("unsupported catalog rule kind")
    for name in ("allowed_verdicts", "required_fact_keys", "state_fields", "allowed_resource_kinds"):
        values = getattr(rule, name)
        if any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
            raise ValueError(f"duplicated or malformed catalog {name}")
    for name in ("state_values", "fact_constants", "resource_generation_basis"):
        keys = [key for key, _ in getattr(rule, name)]
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicated catalog {name}")
    for _, values in rule.state_values:
        if len(values) != len(set(values)):
            raise ValueError("duplicated catalog state values")
    if not rule.allowed_verdicts:
        raise ValueError("catalog rule has no allowed verdicts")
    if any(type(value) not in (str, bool, int, type(None)) for _, value in rule.fact_constants):
        raise ValueError("catalog fact constants must be immutable JSON scalars")


for _definition in _RULES:
    _check_definition(_definition)
if len({rule.rule_id for rule in _RULES}) != len(_RULES):
    raise ValueError("duplicated catalog rule identity")
_INDEX = MappingProxyType({rule.rule_id: rule for rule in _RULES})


def _metadata(rule):
    resources = rule.allowed_resource_kinds or (rule.resource_kind,)
    return {
        "id": rule.rule_id, "semantic_revision": SEMANTIC_REVISION,
        "evidence_kind": rule.evidence_kind, "scope": rule.scope,
        "resource_kind": rule.resource_kind, "allowed_resource_kinds": list(resources),
        "allowed_verdicts": list(rule.allowed_verdicts), "verdict_ordering": "unordered",
        "kind": rule.kind, "required_fact_keys": list(rule.required_fact_keys),
        "required_identity_fields": ["subject", "scenario"],
        "assumptions": list(rule.assumptions), "missing_facts": list(rule.missing_facts),
        "state_fields": list(rule.state_fields),
        "state_values": {key: list(values) for key, values in rule.state_values},
        "fact_constants": dict(rule.fact_constants),
        "resource_generation_basis": dict(rule.resource_generation_basis),
        "state_preservation": "explicit policy required; no automatic outcome or state ranking",
        "expected_when": rule.expected_when,
        "validation_scope": "metadata and finding shape only; original evaluator must establish the verdict",
    }


def catalog():
    """Return an independent JSON-safe copy of the immutable versioned catalog."""
    return {"schema": SCHEMA, "semantic_revision": SEMANTIC_REVISION,
            "validation_scope": "metadata and finding shape only; no evidence reevaluation or authentication",
            "rules": {rule.rule_id: _metadata(rule) for rule in _RULES}}


def describe(rule_id, evidence_kind):
    """Return metadata for this exact rule/evidence domain or raise ValueError."""
    if not isinstance(rule_id, str) or not isinstance(evidence_kind, str):
        raise ValueError("rule and evidence kind must be strings")
    rule = _INDEX.get(rule_id)
    if rule is None or rule.evidence_kind != evidence_kind:
        raise ValueError("unknown rule or unsupported evidence kind")
    return _metadata(rule)


def _json_errors(value, depth=0, seen=None, count=None, *, max_nodes=100_000):
    if seen is None:
        seen = set()
    if count is None:
        if type(max_nodes) is not int or max_nodes <= 0:
            return ["JSON node budget must be a positive integer"]
        count = [0]
    count[0] += 1
    if count[0] > max_nodes:
        return ["finding exceeds supported JSON size"]
    if depth > 64:
        return ["finding exceeds supported JSON nesting"]
    if value is None or type(value) in (bool, int, str):
        return []
    if type(value) is float:
        return [] if math.isfinite(value) else ["finding contains a non-finite JSON number"]
    if isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return ["finding contains circular JSON data"]
        seen.add(id(value))
        errors = []
        try:
            if isinstance(value, dict):
                if any(not isinstance(key, str) for key in value):
                    errors.append("finding JSON object keys must be strings")
                children = value.values()
            else:
                children = value
            for child in children:
                errors.extend(_json_errors(child, depth + 1, seen, count, max_nodes=max_nodes))
                if count[0] > max_nodes:
                    break
        finally:
            seen.remove(id(value))
        return errors
    return ["finding contains a value outside JSON"]


def _identity(value):
    return (isinstance(value, str) and bool(value.strip())) or (isinstance(value, dict) and bool(value))


def validate_finding(finding, evidence_kind):
    """List contract-shape errors; an empty list never proves the verdict.

    Unknown/not-exercised/non-applicable findings may lack conclusive facts.
    Security conclusions must still be recomputed by the original evaluator.
    """
    if not isinstance(finding, dict):
        return ["finding must be a JSON object"]
    errors = _json_errors(finding)
    try:
        metadata = describe(finding.get("rule_id"), evidence_kind)
    except ValueError as exc:
        return list(dict.fromkeys(errors + [str(exc)]))
    if type(finding.get("semantic_revision")) is not int or finding["semantic_revision"] != SEMANTIC_REVISION:
        errors.append("finding semantic revision does not match the catalog")
    if finding.get("scope") != metadata["scope"]:
        errors.append("finding scope does not match the catalog evidence domain")
    if finding.get("resource_kind") not in metadata["allowed_resource_kinds"]:
        errors.append("finding resource kind does not match the catalog")
    if finding.get("verdict") not in metadata["allowed_verdicts"]:
        errors.append("finding verdict is not in the rule's existing domain")
    if "evidence_kind" in finding and finding["evidence_kind"] != evidence_kind:
        errors.append("finding evidence kind contradicts its supplied evidence domain")
    for key in metadata["required_identity_fields"]:
        if not _identity(finding.get(key)):
            errors.append(f"finding requires a nonempty {key} identity")
    missing = finding.get("missing")
    if not isinstance(missing, list) or any(not isinstance(x, str) or not x for x in missing):
        errors.append("finding missing facts must be a list of nonempty strings")
    facts = finding.get("facts")
    if not isinstance(facts, dict):
        errors.append("finding facts must be a JSON object")
    else:
        if finding.get("verdict") not in ("unknown", "not_exercised", "not_applicable"):
            for key in metadata["required_fact_keys"]:
                if key not in facts or facts[key] is None:
                    errors.append(f"conclusive finding requires fact: {key}")
        for key, expected in metadata["fact_constants"].items():
            if key in facts and (type(facts[key]) is not type(expected) or facts[key] != expected):
                errors.append(f"fact {key} contradicts the rule's scope")
        for key, values in metadata["state_values"].items():
            if key in facts and facts[key] not in values:
                errors.append(f"fact {key} is outside the catalog state vocabulary")
        resource = finding.get("resource_kind")
        expected_basis = metadata["resource_generation_basis"].get(resource) if isinstance(resource, str) else None
        if expected_basis is not None and "generation_basis" in facts and facts["generation_basis"] != expected_basis:
            errors.append("generation basis does not match the destination resource kind")
    # Callers cannot weaken the immutable catalog by supplying replacement lists.
    for key in ("kind", "required_fact_keys", "state_fields", "allowed_verdicts", "allowed_resource_kinds"):
        if key in finding and finding[key] != metadata[key]:
            errors.append(f"finding cannot redefine catalog {key}")
    return list(dict.fromkeys(errors))
