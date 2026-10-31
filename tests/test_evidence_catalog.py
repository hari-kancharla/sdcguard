"""Catalog contracts preserve existing scopes and make no security judgments."""
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from sdcguard import evidence_catalog as rules


STATIC_IDS = {
    "capacity.planned", "capacity.startup", "protection.retirement", "protection.deferred_release",
    "protection.failed_send_drain", "protection.checksum_configured", "protection.canary_configured",
    "detection.architecture_coverage", "compatibility.pairing", "deployment.peer_profile",
    "deployment.disaggregation_enable_kv_checksum_agreement",
    "deployment.disaggregation_decode_allocation_policy_agreement", "deployment.roles", "deployment.context",
}


def finding(rule_id="capacity.planned", evidence_kind="static_config", verdict="unknown"):
    meta = rules.describe(rule_id, evidence_kind)
    return {"rule_id": rule_id, "semantic_revision": 1, "evidence_kind": evidence_kind,
            "scope": meta["scope"], "resource_kind": meta["allowed_resource_kinds"][0],
            "subject": "worker:P", "scenario": "saved-contract", "verdict": verdict,
            "facts": {}, "missing": ["evidence not supplied"]}


def component(kind="destination_kv_allocation"):
    value = finding("destination_incarnation_retired_after_writers", "source_fragment_cpu", "satisfied")
    value["resource_kind"] = kind
    value["facts"] = {"obligations": [], "physical_write_safety": "unknown",
                      "generation_basis": "observed_allocation" if kind == "destination_kv_allocation" else "protocol_hold"}
    value["missing"] = []
    return value


def test_catalog_is_json_safe_versioned_and_covers_existing_named_rules():
    catalog = rules.catalog()
    assert json.loads(json.dumps(catalog, allow_nan=False)) == catalog
    assert catalog["schema"] == "sdcguard.evidence-catalog.v1"
    assert catalog["semantic_revision"] == 1
    assert set(catalog["rules"]) == STATIC_IDS | {
        "lease_valid_at_use", "destination_incarnation_retired_after_writers"}
    for key, meta in catalog["rules"].items():
        assert meta["id"] == key
        assert len(meta["required_fact_keys"]) == len(set(meta["required_fact_keys"]))
        assert meta["required_identity_fields"] == ["subject", "scenario"]
        assert meta["verdict_ordering"] == "unordered"
        assert meta["assumptions"] and meta["missing_facts"]
        assert "shape only" in meta["validation_scope"]


def test_returned_catalog_and_description_cannot_mutate_definitions():
    original = rules.catalog()
    changed = rules.catalog()
    changed["rules"]["capacity.planned"]["allowed_verdicts"].append("safe")
    changed["rules"]["capacity.planned"]["required_fact_keys"].clear()
    changed["rules"]["capacity.planned"]["fact_constants"]["runtime_execution"] = "observed"
    description = rules.describe("protection.canary_configured", "static_config")
    description["state_values"]["mode"].append("disabled_but_safe")
    assert rules.catalog() == original
    with pytest.raises(TypeError):
        rules._INDEX["extra"] = rules._RULES[0]
    with pytest.raises(FrozenInstanceError):
        rules._RULES[0].kind = "inventory"


def test_verdicts_keep_recovery_refusal_and_rejection_distinct():
    lease = rules.describe("lease_valid_at_use", "recorded_engine_events")
    assert lease["allowed_verdicts"] == ["satisfied", "violated", "correctly_rejected", "recovered",
                                          "refused", "unknown", "not_exercised"]
    generation = rules.describe("destination_incarnation_retired_after_writers", "source_fragment_cpu")
    assert generation["allowed_verdicts"] == ["satisfied", "unknown"]
    assert "violated" not in generation["allowed_verdicts"]


@pytest.mark.parametrize("rule_id,evidence_kind", [
    ("no_such_rule", "static_config"), ("lease_valid_at_use", "static_config"),
    ("capacity.planned", "recorded_engine_events"), ("capacity.planned", "runtime_gpu"),
    ([], "static_config"), ("capacity.planned", None),
])
def test_unknown_rules_or_evidence_domains_are_refused(rule_id, evidence_kind):
    with pytest.raises(ValueError):
        rules.describe(rule_id, evidence_kind)


def test_component_resources_cannot_be_relabelled_as_one_another():
    for resource in ["destination_kv_allocation", "protocol_held_auxiliary_metadata"]:
        assert rules.validate_finding(component(resource), "source_fragment_cpu") == []
    mixed = component("protocol_held_auxiliary_metadata")
    mixed["facts"]["generation_basis"] = "observed_allocation"
    assert any("generation basis" in x for x in rules.validate_finding(mixed, "source_fragment_cpu"))
    mixed["resource_kind"] = "physical_kv_pages"
    assert any("resource kind" in x for x in rules.validate_finding(mixed, "source_fragment_cpu"))


def test_cpu_component_never_claims_physical_write_safety():
    value = component()
    value["facts"]["physical_write_safety"] = "satisfied"
    assert any("scope" in x for x in rules.validate_finding(value, "source_fragment_cpu"))
    value = component()
    value["scope"] = "deployed_engine"
    assert any("scope" in x for x in rules.validate_finding(value, "source_fragment_cpu"))
    value = component()
    value["verdict"] = "violated"
    assert any("verdict" in x for x in rules.validate_finding(value, "source_fragment_cpu"))


def test_known_off_inventory_is_known_without_inventing_mandatory_policy():
    checksum = finding("protection.checksum_configured", verdict="satisfied")
    checksum["facts"] = {"state": "off", "runtime_execution": "not established", "identity_basis": "operator supplied"}
    checksum["missing"] = []
    assert rules.validate_finding(checksum, "static_config") == []
    meta = rules.describe(checksum["rule_id"], "static_config")
    assert meta["kind"] == "inventory" and "state" in meta["state_fields"]
    assert "explicit policy" in meta["state_preservation"]
    canary = finding("protection.canary_configured", verdict="satisfied")
    canary["facts"] = {"state": "on", "mode": "log", "real_kv_mode": "partial",
                        "runtime_execution": "not established", "identity_basis": "operator supplied"}
    assert rules.validate_finding(canary, "static_config") == []
    assert {"state", "mode", "real_kv_mode"} <= set(rules.describe(canary["rule_id"], "static_config")["state_fields"])
    canary["verdict"] = "violated"
    assert any("verdict" in x for x in rules.validate_finding(canary, "static_config"))


def test_unknown_findings_preserve_missing_evidence_without_positive_fact_requirements():
    for rule_id in STATIC_IDS:
        assert rules.validate_finding(finding(rule_id), "static_config") == []
    assert rules.validate_finding(finding("lease_valid_at_use", "recorded_engine_events"),
                                  "recorded_engine_events") == []


def test_conclusive_finding_requires_catalog_facts_but_is_not_reevaluated():
    value = finding(verdict="satisfied")
    errors = rules.validate_finding(value, "static_config")
    assert any("advertised_context_tokens" in x for x in errors)
    # Contradictory capacity arithmetic still belongs to the original evaluator.
    # This deliberately demonstrates that shape validation is not a safety pass.
    value["facts"] = {"advertised_context_tokens": 1000, "configured_context_tokens": 1,
                      "block_or_page_size_tokens": 1}
    assert rules.validate_finding(value, "static_config") == []


@pytest.mark.parametrize("change", [
    lambda f: f.update(rule_id="unrecognized"),
    lambda f: f.update(semantic_revision=True),
    lambda f: f.update(semantic_revision=2),
    lambda f: f.update(scope="whole_deployment"),
    lambda f: f.update(resource_kind="physical_isolation"),
    lambda f: f.update(resource_kind=["physical_isolation"]),
    lambda f: f.update(verdict="fine"),
    lambda f: f.update(evidence_kind="recorded_engine_events"),
    lambda f: f.update(subject=""),
    lambda f: f.update(scenario=3),
    lambda f: f.update(missing={"fact": "missing"}),
    lambda f: f.update(missing=[None]),
    lambda f: f.update(facts=[]),
    lambda f: f.update(required_fact_keys=[]),
    lambda f: f.update(required_fact_keys=["x", "x"]),
    lambda f: f.update(allowed_verdicts=["unknown", "safe"]),
    lambda f: f.update(kind="inventory"),
    lambda f: f["facts"].update(value=float("nan")),
    lambda f: f["facts"].update(value={1: "silently coerced key"}),
    lambda f: f["facts"].update(value=object()),
])
def test_malformed_or_redefined_contracts_have_errors(change):
    value = finding()
    change(value)
    assert rules.validate_finding(value, "static_config")


def test_direct_validator_handles_nonobjects_cycles_and_nested_data():
    assert rules.validate_finding([], "static_config")
    value = finding()
    value["facts"]["cycle"] = value
    assert any("circular" in x for x in rules.validate_finding(value, "static_config"))
    value = finding()
    nested = {}
    value["facts"]["deep"] = nested
    for _ in range(70):
        child = {}
        nested["x"] = child
        nested = child
    assert any("nesting" in x for x in rules.validate_finding(value, "static_config"))
    value = finding()
    value["facts"]["wide"] = [None] * 100_001
    assert any("JSON size" in x for x in rules.validate_finding(value, "static_config"))


@pytest.mark.parametrize("field,value", [
    ("required_fact_keys", ("request", "request")),
    ("allowed_verdicts", ("unknown", "unknown")),
    ("state_fields", ("state", "state")),
    ("state_values", (("state", ("on",)), ("state", ("off",)))),
])
def test_catalog_definition_cannot_contain_duplicate_contract_fields(field, value):
    with pytest.raises(ValueError):
        rules._check_definition(replace(rules._RULES[0], **{field: value}))


def test_backend_specific_nonapplicability_is_preserved_without_inventing_proof():
    value = finding("protection.failed_send_drain", verdict="not_applicable")
    assert rules.validate_finding(value, "static_config") == []
    value = finding("deployment.roles", verdict="satisfied")
    assert any("verdict" in x for x in rules.validate_finding(value, "static_config"))


def test_native_verdict_vocabulary_still_matches_the_existing_event_evaluator():
    from sdcguard.lifetime.obligations import VERDICTS
    assert rules.describe("lease_valid_at_use", "recorded_engine_events")["allowed_verdicts"] == list(VERDICTS)


def test_static_rule_catalog_covers_actual_unknown_report_inventory(tmp_path):
    from sdcguard import config_safety
    doc = {"schema": "sdcguard.config-safety.v1", "workers": [{"id": "P"}]}
    result = config_safety.assess(doc, {}, tmp_path)
    native = [rule for worker in result["workers"] for rule in worker["rules"]] + result["deployment_rules"]
    assert {item["id"] for item in native} <= STATIC_IDS
    for item in native:
        assert item["verdict"] in rules.describe(item["id"], "static_config")["allowed_verdicts"]


def test_startup_state_catalog_names_the_existing_native_capacity_fact():
    from sdcguard import config_capacity
    from test_config_capacity import worker, startup
    item = worker("vllm")
    native = config_capacity.evaluate(item, 512, startup(item))[1]
    assert native["verdict"] == "satisfied"
    entry = rules.describe("capacity.startup", "static_config")
    key = "logged_equivalent_capacity_tokens"
    assert key in native["facts"] and key in entry["state_fields"]
    assert "startup_group_equivalent_tokens" not in entry["state_fields"]
