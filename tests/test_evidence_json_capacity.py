"""Evidence representation bounds remain separate from deployment input bounds.

Owner: Hari Krishna Kancharla.
Synthetic reports below test capacity and shape only, not runtime behavior.
"""
from copy import deepcopy
import json

import pytest

from sdcguard import config_inputs, evidence_catalog as catalog
from sdcguard import evidence_io as io, evidence_recordings, evidence_report
from sdcguard import operator_setup, upgrade_policy
from test_upgrade_policy import recorded_report


def report_with_many_small_findings():
    seed = recorded_report()
    findings = []
    for index in range(387):
        row = deepcopy(seed["findings"][0])
        row["subject"] = "request:synthetic-capacity-" + str(index)
        row["facts"]["native_finding"]["observations"] = list(range(150))
        findings.append(row)
    return evidence_report.build(
        profile=seed["profile"], recorded={"identity": seed["identities"]["recorded"],
        "findings": findings, "legacy_scope_verdict": "fine"})


def test_deployment_node_limit_is_unchanged_at_exact_boundary():
    config_inputs._json_values([None] * 99_999)
    with pytest.raises(ValueError, match="supported bounds"):
        config_inputs._json_values([None] * 100_000)
    assert config_inputs.MAX_JSON_NODES == 100_000


def test_explicit_budget_is_global_and_propagates_through_children():
    value = {"one": [1, 2], "two": [3, 4]}
    config_inputs._json_values(value, max_nodes=9)
    with pytest.raises(ValueError, match="supported bounds"):
        config_inputs._json_values(value, max_nodes=8)
    assert not catalog._json_errors(value, max_nodes=7)
    assert catalog._json_errors(value, max_nodes=6)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, None])
def test_json_budget_must_be_a_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        config_inputs._json_values(None, max_nodes=budget)
    with pytest.raises(io.EvidenceError, match="positive integer"):
        io.parse_json(b"null", max_nodes=budget)
    assert catalog._json_errors(None, max_nodes=budget)


def test_evidence_reader_and_writer_have_an_exact_finite_node_bound():
    value = [None] * 999_999
    raw = io.json_bytes(value)
    assert io.parse_json(raw) == value
    value.append(None)
    with pytest.raises(ValueError, match="supported bounds"):
        io.json_bytes(value)
    with pytest.raises(io.EvidenceError, match="supported bounds"):
        io.parse_json(json.dumps(value).encode())
    assert io.MAX_JSON_NODES == upgrade_policy.MAX_REPORT_JSON_NODES == 1_000_000


def test_large_evidence_report_passes_without_widening_config_inputs(tmp_path):
    value = {"schema": config_inputs.SCHEMA, "workers": [
        {"id": "prefill", "role": "prefill", "parameters": {"observations": [None] * 100_000}}]}
    raw = json.dumps(value).encode()
    assert len(raw) < config_inputs.MAX_CONFIG_BYTES
    assert io.parse_json(raw) == value
    assert evidence_recordings._strict_json(raw, "report.json") == value
    for name in ("config.json", "evidence/config-original.json"):
        with pytest.raises(evidence_recordings.RecordedEvidenceError):
            evidence_recordings._strict_json(raw, name)
    path = tmp_path / "deployment.json"
    path.write_bytes(raw)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="supported bounds"):
        config_inputs.load_deployment(path)
    with pytest.raises(io.EvidenceError, match="supported bounds"):
        operator_setup._read_config(path)
    assert path.read_bytes() == before


def test_complete_many_finding_report_survives_policy_shape_validation():
    report = report_with_many_small_findings()
    with pytest.raises(ValueError, match="supported bounds"):
        config_inputs._json_values(report)
    assert all(not catalog.validate_finding(row, row["evidence_kind"])
               for row in report["findings"])
    assert io.parse_json(io.json_bytes(report)) == report
    result = upgrade_policy.compare(report, report)
    assert result["gaps"] == result["native_candidate_violations"] == result["policy_violations"] == []
    assert len(result["matched_outcomes"]) == 387
    assert result["runtime_upgrade_witness"] == "not established"


def test_whole_report_extension_does_not_widen_per_finding_or_semantic_contract():
    report = recorded_report()
    row = report["findings"][0]
    row["facts"]["native_finding"]["oversized"] = [None] * 100_000
    assert catalog.validate_finding(row, row["evidence_kind"])
    result = upgrade_policy.compare(report, report)
    assert any(gap["code"] == "finding_contract_changed" for gap in result["gaps"])
    report = report_with_many_small_findings()
    report["findings"][0]["required"] = "true"
    with pytest.raises(ValueError, match="malformed canonical finding"):
        upgrade_policy.compare(report, report)


def test_whole_report_above_one_million_nodes_is_rejected():
    report = recorded_report()
    report["native_sections"]["oversized"] = [None] * 1_000_000
    with pytest.raises(ValueError, match="finite JSON canonical reports"):
        upgrade_policy.compare(report, report)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"n":NaN}', b'{"n":1e999}',
                                 b'{"s":"\\ud800"}', b'[' * 66 + b'0' + b']' * 66])
def test_larger_evidence_budget_keeps_strict_json_contract(raw):
    with pytest.raises(io.EvidenceError):
        io.parse_json(raw)
    with pytest.raises(evidence_recordings.RecordedEvidenceError):
        evidence_recordings._strict_json(raw, "report.json")
