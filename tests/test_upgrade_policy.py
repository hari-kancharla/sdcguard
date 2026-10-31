"""Pure comparison tests using existing static evaluators and canonical fixtures.

Owner: Hari Krishna Kancharla.
Synthetic fixtures are policy inputs, not runtime execution evidence.
"""
from copy import deepcopy
import ast
import hashlib
import inspect
import json

import pytest

from sdcguard import config_safety, evidence_report
from sdcguard import upgrade_policy as policy
from sdcguard.config_protections import VLLM_PATCH
from test_config_capacity import worker as capacity_worker, startup


def static_report(tmp_path, *, engine="vllm", blocks=33, pool=512, logs=True, change=None):
    document = {
        "schema": "sdcguard.config-safety.v1", "deployment_id": "synthetic-policy-input",
        "advertised_context_tokens": 512,
        "model": {"id": "example/placeholder", "revision": "a" * 40, "architecture": "Qwen3ForCausalLM"},
        "workers": [],
    }
    for role in ("prefill", "decode"):
        worker = capacity_worker(engine)
        worker.update(id=role, role=role)
        worker["layout"].update(parameters_complete=True, environment_complete=True,
                                kv_pool_class="MHATokenToKVPool")
        if engine == "vllm":
            worker["components"]["runtime_patch_sha256"] = VLLM_PATCH
            worker["connector"].update(transfer_mode="pull", bidirectional=False,
                                       extra_config={"bidirectional_kv_xfer": False})
            worker["parameters"]["num_gpu_blocks_override"] = blocks
        else:
            worker["parameters"].update(disaggregation_mode=role, kv_canary="log",
                                         kv_canary_real_data="all", max_total_tokens=2048)
        document["workers"].append(worker)
    if change:
        change(document)
    log_map = {}
    for worker in document["workers"]:
        if logs:
            text = startup(worker)
            if engine == "sglang":
                text = text.replace("max_total_num_tokens=512", "max_total_num_tokens=" + str(pool))
            log_map[worker["id"]] = {"text": text, "path": worker["id"] + ".log",
                                     "sha256": hashlib.sha256(text.encode()).hexdigest()}
        else:
            log_map[worker["id"]] = {"error": "startup bytes not supplied", "path": "not supplied"}
    native = config_safety.assess(document, {"logs": log_map}, tmp_path)
    return evidence_report.build(profile={"static": True, "recorded": False, "components": "none"}, static=native)


def both(document, container, key, value):
    for worker in document["workers"]:
        worker.setdefault(container, {})[key] = value


def rows(report, rule):
    return [row for row in report["findings"] if row["rule_id"] == rule]


def gap_codes(result):
    return {row["code"] for row in result["gaps"]}


def no_blockers(result):
    assert result["gaps"] == []
    assert result["native_candidate_violations"] == []
    assert result["policy_violations"] == []


def recorded_report(verdict="satisfied", *, protection="recorded ownership", snapshot="b" * 64):
    from sdcguard.evidence_catalog import describe
    metadata = describe("lease_valid_at_use", "recorded_engine_events")
    identity = {
        "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
        "model": {"id": "example/placeholder", "revision": "a" * 40, "live_weights_verified": False},
        "topology": {"tp": 1, "pp": 1}, "source_identity": {"commit": "c" * 40},
        "configuration": {"connector": "NixlConnector"}, "packet_snapshot_sha256": snapshot,
        "kernel": "not observed",
    }
    finding = {"rule_id": "lease_valid_at_use", "semantic_revision": 1,
        "evidence_kind": "recorded_engine_events", "scope": metadata["scope"],
        "resource_kind": metadata["resource_kind"], "subject": "request:policy-fixture",
        "scenario": "recorded-source-lease", "verdict": verdict, "required": True,
        "facts": {"native_finding": {"protection": protection}, "native_verdict": verdict,
                  "engine_identity": identity["source_identity"], "configuration": identity["configuration"],
                  "model_binding": "fixture only", "runtime_model_use": "not observed"}, "missing": []}
    return evidence_report.build(profile={"static": False, "recorded": True, "components": "none"},
        recorded={"identity": identity, "findings": [finding], "legacy_scope_verdict": "fine"})


def test_descriptor_is_fixed_json_safe_and_independent(tmp_path):
    value = policy.policy_descriptor()
    assert value["name"] == "preserve-named-contracts.v1"
    value["clauses"].clear()
    assert policy.policy_descriptor()["clauses"]
    report = static_report(tmp_path)
    result = policy.compare(report, report)
    encoded = json.dumps(result["policy"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert result["policy_sha256"] == hashlib.sha256(encoded).hexdigest()
    json.loads(json.dumps(result, allow_nan=False))


def test_same_static_contract_preserved_without_runtime_witness(tmp_path):
    report = static_report(tmp_path)
    assert report["status"] == "fine"
    original = deepcopy(report)
    result = policy.compare(report, report)
    no_blockers(result)
    assert all(row["outcome"] == "preserved" for row in result["matched_outcomes"])
    assert result["runtime_upgrade_witness"] == "not established"
    assert report == original


@pytest.mark.parametrize("left,right", [(33, 65), (65, 33)])
def test_pool_target_changes_preserve_same_bound_property(tmp_path, left, right):
    base, candidate = (static_report(tmp_path, blocks=value) for value in (left, right))
    result = policy.compare(base, candidate)
    no_blockers(result)
    deltas = [row for row in result["input_deltas"] if row["path"][-1] == "num_gpu_blocks_override"]
    assert len(deltas) == 2 and all(row["classification"] == "preserved_capacity_target" for row in deltas)
    assert all(row["outcome"] == "preserved" for row in result["matched_outcomes"])


def test_pool_boundary_is_regression_when_same_contract(tmp_path):
    result = policy.compare(static_report(tmp_path), static_report(tmp_path, blocks=32))
    assert result["native_candidate_violations"]
    assert all(row["regression_attribution"] == "regressed" for row in result["native_candidate_violations"])
    assert not result["gaps"]


def test_bad_unchanged_candidate_still_violated_not_regressed(tmp_path):
    report = static_report(tmp_path, blocks=32)
    result = policy.compare(report, report)
    assert result["native_candidate_violations"]
    assert all(row["regression_attribution"] == "still_violated" for row in result["native_candidate_violations"])
    assert not any(row["outcome"] == "regressed" for row in result["matched_outcomes"])


def test_repaired_negative_baseline_not_silently_approved(tmp_path):
    result = policy.compare(static_report(tmp_path, blocks=32), static_report(tmp_path))
    assert result["native_candidate_violations"] == []
    assert result["remediations"]
    assert "baseline_not_approved" in gap_codes(result)


def test_sglang_finalized_pool_changes_can_preserve_capacity(tmp_path):
    base = static_report(tmp_path, engine="sglang", pool=512)
    candidate = static_report(tmp_path, engine="sglang", pool=1024,
                              change=lambda d: both(d, "parameters", "max_total_tokens", 4096))
    assert base["status"] == candidate["status"] == "fine"
    result = policy.compare(base, candidate)
    no_blockers(result)
    assert any(row["classification"] == "preserved_capacity_target" for row in result["input_deltas"])
    assert all(row["outcome"] in ("preserved", "resolved_by_startup") for row in result["matched_outcomes"])


def test_missing_startup_blocks_target_exception(tmp_path):
    result = policy.compare(static_report(tmp_path), static_report(tmp_path, blocks=65, logs=False))
    assert "unclassified_input_delta" in gap_codes(result)
    assert "required_candidate_unknown" in gap_codes(result)
    assert not result["native_candidate_violations"]


def test_new_bound_startup_can_establish_unknown_with_same_identity(tmp_path):
    result = policy.compare(static_report(tmp_path, logs=False), static_report(tmp_path))
    no_blockers(result)
    assert {row["finding"]["rule_id"] for row in result["newly_established"]} == {"capacity.startup"}


def test_trusted_sglang_resolution_is_not_a_required_downgrade(tmp_path):
    result = policy.compare(static_report(tmp_path, engine="sglang", logs=False), static_report(tmp_path, engine="sglang"))
    no_blockers(result)
    assert any(row["outcome"] == "resolved_by_startup" for row in result["matched_outcomes"])


@pytest.mark.parametrize("change", [
    lambda d: d["model"].update(revision="b" * 40),
    lambda d: d.update(advertised_context_tokens=256),
    lambda d: both(d, "parameters", "max_num_seqs", 100),
    lambda d: both(d, "environment", "UNMAPPED_INPUT", "enabled"),
    lambda d: both(d, "components", "unmapped_component", "new"),
])
def test_unclassified_identity_and_flag_changes_block_even_if_native_fine(tmp_path, change):
    candidate = static_report(tmp_path, change=change)
    assert candidate["status"] == "fine"
    result = policy.compare(static_report(tmp_path), candidate)
    assert result["gaps"]
    assert not result["native_candidate_violations"]


def test_worker_order_and_display_labels_are_representation(tmp_path):
    def change(document):
        document["deployment_id"] = "another display label"
        document["workers"].reverse()
    result = policy.compare(static_report(tmp_path), static_report(tmp_path, change=change))
    no_blockers(result)
    assert all(row["classification"] == "representation" for row in result["input_deltas"])


@pytest.mark.parametrize("rule,key,value", [
    ("protection.checksum_configured", "disaggregation_enable_kv_checksum", False),
    ("protection.canary_configured", "kv_canary", "log"),
    ("protection.canary_configured", "kv_canary_real_data", "all"),
])
def test_same_mapped_default_can_be_made_explicit(tmp_path, rule, key, value):
    def remove(document):
        for worker in document["workers"]:
            worker["parameters"].pop(key, None)
    base = static_report(tmp_path, engine="sglang", change=remove)
    # Only defaults whose native resolved setting matches are representational.
    candidate = static_report(tmp_path, engine="sglang", change=lambda d: both(d, "parameters", key, value))
    result = policy.compare(base, candidate)
    matching = [row for row in result["input_deltas"] if row["path"][-1] == key]
    if key == "disaggregation_enable_kv_checksum":
        no_blockers(result)
        assert all(row["classification"] == "same_resolved_default" for row in matching)
    else:
        # Source defaults are none, so these are actual changes, not aliases.
        assert not any(row["classification"] == "same_resolved_default" for row in matching)


def test_checksum_removal_is_explicit_policy_violation_with_inventory_satisfied(tmp_path):
    base = static_report(tmp_path, engine="sglang", change=lambda d: both(d, "parameters", "disaggregation_enable_kv_checksum", True))
    candidate = static_report(tmp_path, engine="sglang", change=lambda d: both(d, "parameters", "disaggregation_enable_kv_checksum", False))
    assert all(row["verdict"] == "satisfied" for row in rows(candidate, "protection.checksum_configured"))
    result = policy.compare(base, candidate)
    assert len(result["policy_violations"]) == 2
    assert {row["clause"] for row in result["policy_violations"]} == {"enabled_checksum_removed"}
    assert all(row["regression_attribution"] == "regressed" for row in result["policy_violations"])
    assert not result["native_candidate_violations"]


def test_checksum_can_be_enabled_with_source_coverage_and_peer_agreement(tmp_path):
    result = policy.compare(static_report(tmp_path, engine="sglang"), static_report(tmp_path, engine="sglang",
        change=lambda d: both(d, "parameters", "disaggregation_enable_kv_checksum", True)))
    no_blockers(result)


@pytest.mark.parametrize("left,right,weakening", [("raise", "log", True), ("raise", "none", True),
    ("log", "none", True), ("log", "raise", False), ("none", "log", False)])
def test_canary_action_preservation_is_explicit_not_native_verdict_order(tmp_path, left, right, weakening):
    base = static_report(tmp_path, engine="sglang", change=lambda d: both(d, "parameters", "kv_canary", left))
    candidate = static_report(tmp_path, engine="sglang", change=lambda d: both(d, "parameters", "kv_canary", right))
    result = policy.compare(base, candidate)
    assert bool(result["policy_violations"]) is weakening
    if not weakening:
        no_blockers(result)
    assert all(row["verdict"] == "satisfied" for row in rows(candidate, "protection.canary_configured"))


@pytest.mark.parametrize("container,key,value", [
    ("environment", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT", "60"),
    ("parameters", "kv_canary_real_data", "partial"),
])
def test_timeouts_and_real_kv_changes_are_not_ranked(tmp_path, container, key, value):
    result = policy.compare(static_report(tmp_path, engine="sglang"), static_report(tmp_path, engine="sglang",
        change=lambda d: both(d, container, key, value)))
    assert "unclassified_input_delta" in gap_codes(result)
    assert not result["policy_violations"]
    assert not result["native_candidate_violations"]


def test_native_retention_disable_is_regressed_under_same_identity(tmp_path):
    result = policy.compare(static_report(tmp_path, engine="sglang"), static_report(tmp_path, engine="sglang",
        change=lambda d: both(d, "environment", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE", "0")))
    assert {row["regression_attribution"] for row in result["native_candidate_violations"]} == {"regressed"}


def test_unsupported_source_delta_does_not_become_known_upgrade(tmp_path):
    result = policy.compare(static_report(tmp_path), static_report(tmp_path,
        change=lambda d: both(d, "components", "runtime_patch_sha256", "d" * 64)))
    assert "unclassified_input_delta" in gap_codes(result)
    assert "required_candidate_unknown" in gap_codes(result)


def test_exact_saved_overlay_removal_is_reviewed_but_native_policy_fails(tmp_path):
    result = policy.compare(static_report(tmp_path), static_report(tmp_path,
        change=lambda d: both(d, "components", "runtime_patch_sha256", None)))
    assert any(row["classification"] == "reviewed_source_transition" for row in result["input_deltas"])
    assert {row["regression_attribution"] for row in result["native_candidate_violations"]} == {"regressed"}


@pytest.mark.parametrize("mutation", [
    lambda r: r["findings"].pop(),
    lambda r: r["findings"][0].update(semantic_revision=2),
    lambda r: r["findings"][0].update(scope="runtime isolation"),
    lambda r: r["findings"][0].update(subject={"worker": "renamed"}),
    lambda r: r["findings"][0].update(required=False),
])
def test_contract_identity_and_required_downgrades_block(tmp_path, mutation):
    base = static_report(tmp_path)
    candidate = deepcopy(base)
    mutation(candidate)
    assert policy.compare(base, candidate)["gaps"]


def test_empty_required_set_cannot_pass_and_status_flags_are_not_authority(tmp_path):
    base = static_report(tmp_path)
    candidate = deepcopy(base)
    candidate.update(status="fine", trusted=True, verified=True)
    candidate["findings"].clear()
    assert "no_required_findings" in gap_codes(policy.compare(base, candidate))


def test_known_candidate_violation_survives_unrelated_comparison_gap(tmp_path):
    base = static_report(tmp_path)
    candidate = static_report(tmp_path, blocks=32, change=lambda d: d["model"].update(revision="b" * 40))
    result = policy.compare(base, candidate)
    assert result["native_candidate_violations"] and result["gaps"]
    assert {row["regression_attribution"] for row in result["native_candidate_violations"]} == {"not established"}


@pytest.mark.parametrize("left,right", [("recovered", "refused"), ("refused", "correctly_rejected"),
    ("correctly_rejected", "recovered"), ("satisfied", "refused")])
def test_terminal_outcomes_remain_unordered(left, right):
    result = policy.compare(recorded_report(left), recorded_report(right))
    assert "outcome_changed_without_ordering" in gap_codes(result)
    assert not result["native_candidate_violations"]
    assert not result["policy_violations"]


def test_recorded_same_packet_is_scoped_evidence_not_runtime_witness():
    report = recorded_report("recovered")
    result = policy.compare(report, report)
    no_blockers(result)
    assert result["matched_outcomes"][0]["outcome"] == "preserved"
    assert result["runtime_upgrade_witness"] == "not established"


def test_changed_recording_has_no_reusable_stimulus_identity():
    result = policy.compare(recorded_report(), recorded_report(snapshot="e" * 64))
    assert "section_identity_changed" in gap_codes(result)
    assert "finding_incomparable" in gap_codes(result)


def test_changed_recorded_protection_is_not_automatically_ranked():
    result = policy.compare(recorded_report(), recorded_report(protection="lease verifier"))
    assert "recorded_protection_changed" in gap_codes(result)


def test_component_resources_keep_distinct_source_scope():
    from sdcguard.evidence_components import collect
    report = evidence_report.build(profile={"static": False, "recorded": False, "components": "all"}, components=collect())
    result = policy.compare(report, report)
    no_blockers(result)
    assert {row["finding"]["resource_kind"] for row in result["matched_outcomes"]} == {
        "destination_kv_allocation", "protocol_held_auxiliary_metadata"}
    assert result["runtime_upgrade_witness"] == "not established"


@pytest.mark.parametrize("bad", [None, [], {"schema": "other"}])
def test_malformed_noncanonical_inputs_rejected(bad):
    with pytest.raises(ValueError):
        policy.compare(bad, bad)


def test_module_has_no_io_or_engine_imports():
    tree = ast.parse(inspect.getsource(policy))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names}
    assert imported <= {"__future__", "copy", "hashlib", "json", "re", None}
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"open", "exec", "eval", "compile", "__import__"}
