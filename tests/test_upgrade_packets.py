"""Compare real local captures; edited claims have no authority."""
from copy import deepcopy
import json
import shutil

import pytest

from sdcguard import evidence_bundle, evidence_components, evidence_io as io, upgrade
from sdcguard.config_inputs import binding_digest
from test_config_capacity import startup
from test_evidence_bundle import static_config, rehash_packet


def packet(root, label, *, engine="vllm", mutate=None, logs=True, components=False):
    inputs = root / (label + "-inputs")
    inputs.mkdir()
    config = static_config(inputs, engine=engine, logs=logs)
    if mutate:
        value = io.parse_json(config.read_bytes())
        mutate(value)
        for worker in value["workers"]:
            if "startup_log" in worker:
                raw = startup(worker).encode()
                (inputs / worker["startup_log"]["path"]).write_bytes(raw)
                worker["startup_log"].update(sha256=io.digest(raw), config_sha256=binding_digest(worker, value))
        config.write_bytes(io.json_bytes(value))
    target = root / label
    result = evidence_bundle.create_bundle(target, config=config, components=components)
    return target, result["report"]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_same_packet_preserves_scoped_contract_without_runtime_witness(tmp_path, engine):
    source, native = packet(tmp_path, "base", engine=engine)
    assert native["status"] == "fine"
    before = io.snapshot(source)
    out = tmp_path / "gate"
    result = upgrade.compare_packets(source, source, out)
    assert result["scope_verdict"] == "fine", result
    assert result["report"]["comparison"] == "preserved"
    assert result["report"]["runtime_upgrade_witness"] == "not established"
    assert io.snapshot(source) == before
    verified = upgrade.verify_comparison(out)
    assert verified["integrity"] == "verified" and verified["scope_verdict"] == "fine", verified
    moved = tmp_path / "moved"
    shutil.copytree(out, moved)
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "base-inputs")
    shutil.rmtree(out)
    assert upgrade.verify_comparison(moved)["integrity"] == "verified"


@pytest.mark.parametrize("baseline,candidate,expected", [(33, 65, "fine"), (65, 33, "fine"), (33, 32, "not fine")])
def test_pool_change_recomputes_both_bound_capacity_contracts(tmp_path, baseline, candidate, expected):
    def size(value):
        return lambda config: [worker["parameters"].update(num_gpu_blocks_override=value)
                               for worker in config["workers"]]
    base, _ = packet(tmp_path, "base", mutate=size(baseline))
    cand, _ = packet(tmp_path, "cand", mutate=size(candidate))
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == expected, result
    assert result["report"]["comparison_details"]["input_deltas"]
    if expected == "not fine":
        violations = result["report"]["comparison_details"]["native_candidate_violations"]
        assert violations and all(row["regression_attribution"] == "regressed" for row in violations)


def test_different_known_model_revision_does_not_pass_even_with_clean_reports(tmp_path):
    base, _ = packet(tmp_path, "base")
    cand, standalone = packet(tmp_path, "cand", mutate=lambda value: value["model"].update(revision="b" * 40))
    assert standalone["status"] == "fine"
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "inconclusive", result
    assert result["report"]["comparison_details"]["gaps"]


def test_unmapped_config_change_is_not_allowed_by_clean_native_verdicts(tmp_path):
    base, _ = packet(tmp_path, "base")
    cand, standalone = packet(tmp_path, "cand", mutate=lambda value: value["workers"][0]["parameters"].update(unreviewed_setting=True))
    assert standalone["status"] == "fine"
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "inconclusive"


def test_preexisting_violation_blocks_without_claiming_upgrade_regression(tmp_path):
    base, _ = packet(tmp_path, "base", mutate=lambda value: [worker["parameters"].update(num_gpu_blocks_override=32)
                                                           for worker in value["workers"]])
    result = upgrade.compare_packets(base, base, tmp_path / "gate")
    assert result["scope_verdict"] == "not fine"
    assert all(row["regression_attribution"] == "still_violated"
               for row in result["report"]["comparison_details"]["native_candidate_violations"])


def test_known_candidate_violation_survives_invalid_baseline(tmp_path):
    base, _ = packet(tmp_path, "base")
    (base / "index.html").write_bytes(b"edited")
    cand, _ = packet(tmp_path, "cand", mutate=lambda value: [worker["parameters"].update(num_gpu_blocks_override=32)
                                                           for worker in value["workers"]])
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "not fine"
    details = result["report"]["comparison_details"]
    assert details["native_candidate_violations"] and details["gaps"]
    assert all(row["regression_attribution"] == "not established" for row in details["native_candidate_violations"])


def test_invalid_candidate_never_contributes_an_authoritative_violation(tmp_path):
    base, _ = packet(tmp_path, "base")
    cand, _ = packet(tmp_path, "cand")
    (cand / "report.json").write_bytes(b'{"status":"not fine"}\n')
    rehash_packet(cand)
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "inconclusive"
    assert not result["report"]["comparison_details"]["native_candidate_violations"]


@pytest.mark.parametrize("extra", [None, [], "unmapped"])
def test_unmapped_connector_shape_cannot_erase_other_workers_failure(tmp_path, extra):
    base, _ = packet(tmp_path, "base")

    def change(value):
        value["workers"][0]["connector"]["extra_config"] = extra
        value["workers"][1]["parameters"]["num_gpu_blocks_override"] = 32

    cand, standalone = packet(tmp_path, "cand", mutate=change)
    assert standalone["status"] == "not fine"
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "not fine", result
    details = result["report"]["comparison_details"]
    assert details["gaps"] and details["native_candidate_violations"]
    assert upgrade.verify_comparison(tmp_path / "gate")["scope_verdict"] == "not fine"


def test_unavailable_policy_preserves_independently_verified_candidate_failure(tmp_path, monkeypatch):
    base, _ = packet(tmp_path, "base")
    cand, _ = packet(tmp_path, "cand", mutate=lambda value: [worker["parameters"].update(num_gpu_blocks_override=32)
                                                           for worker in value["workers"]])
    monkeypatch.setattr(upgrade.policy, "compare", lambda *args: (_ for _ in ()).throw(AssertionError("policy unavailable")))
    result = upgrade.compare_packets(base, cand, tmp_path / "gate")
    assert result["scope_verdict"] == "not fine"
    details = result["report"]["comparison_details"]
    assert any(row["code"] == "installed_policy_unavailable" for row in details["gaps"])
    assert all(row["regression_attribution"] == "not established" for row in details["native_candidate_violations"])


@pytest.mark.parametrize("member", ["report.json", "index.html", "metadata/policy.json", "metadata/upgrade-evaluators.json"])
def test_rehashed_edited_comparison_is_not_a_trusted_report(tmp_path, member):
    base, _ = packet(tmp_path, "base")
    out = tmp_path / "gate"
    upgrade.compare_packets(base, base, out)
    (out / member).write_bytes(b"{}\n")
    rehash_packet(out)
    result = upgrade.verify_comparison(out)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_nested_evidence_edit_is_rechecked_after_both_manifests_are_rehashed(tmp_path):
    base, _ = packet(tmp_path, "base")
    out = tmp_path / "gate"
    upgrade.compare_packets(base, base, out)
    nested = out / "inputs/candidate"
    (nested / "report.json").write_bytes(b'{"status":"fine"}\n')
    rehash_packet(nested)
    rehash_packet(out)
    result = upgrade.verify_comparison(out)
    assert result["integrity"] == "failed" and result["scope_verdict"] == "inconclusive"


def test_current_component_execution_gap_cannot_be_saved_as_a_pass(tmp_path, monkeypatch):
    base, _ = packet(tmp_path, "base", components=True)
    monkeypatch.setattr(evidence_components, "collect", lambda *args: (_ for _ in ()).throw(ValueError("CPU unavailable")))
    result = upgrade.compare_packets(base, base, tmp_path / "gate")
    assert result["scope_verdict"] == "inconclusive"
    assert result["report"]["captured_scope_verdict"] == "fine"
    assert result["current_execution_gaps"]


@pytest.mark.parametrize("short,expected", [(False, "inconclusive"), (True, "not fine")])
def test_later_refresh_gap_preserves_integrity_and_known_failure(tmp_path, monkeypatch, short, expected):
    base, _ = packet(tmp_path, "base", components=True)
    change = (lambda value: [worker["parameters"].update(num_gpu_blocks_override=32)
                            for worker in value["workers"]]) if short else None
    cand, _ = packet(tmp_path, "cand", components=True, mutate=change)
    out = tmp_path / "gate"
    original = upgrade.compare_packets(base, cand, out)
    before = io.snapshot(out)
    monkeypatch.setattr(evidence_components, "collect", lambda *args: (_ for _ in ()).throw(ValueError("CPU unavailable")))
    result = upgrade.verify_comparison(out)
    assert result["integrity"] == "verified", result
    assert result["scope_verdict"] == expected
    assert result["report"] == original["report"]
    assert result["current_execution_gaps"]
    assert io.snapshot(out) == before


def test_output_inside_input_is_refused_before_mutating_it(tmp_path):
    base, _ = packet(tmp_path, "base")
    before = io.snapshot(base)
    with pytest.raises(io.EvidenceError, match="outside"):
        upgrade.compare_packets(base, base, base / "gate")
    assert io.snapshot(base) == before


def test_existing_output_is_not_overwritten(tmp_path):
    base, _ = packet(tmp_path, "base")
    out = tmp_path / "keep"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(OSError):
        upgrade.compare_packets(base, base, out)
    assert io.snapshot(out) == {"keep.txt": b"keep"}


def test_input_mutation_before_output_completion_is_marked_incomplete(tmp_path, monkeypatch):
    base, _ = packet(tmp_path, "base")
    original = upgrade.render

    def mutate(report):
        (base / "changed.txt").write_text("changed")
        return original(report)

    monkeypatch.setattr(upgrade, "render", mutate)
    out = tmp_path / "gate"
    with pytest.raises(io.EvidenceError, match="incomplete"):
        upgrade.compare_packets(base, base, out)
    assert io.parse_json((out / "manifest.json").read_bytes())["complete"] is False
    assert upgrade.verify_comparison(out)["integrity"] == "incomplete"


def test_manifest_receipt_count_does_not_accept_boolean(tmp_path):
    base, _ = packet(tmp_path, "base")
    out = tmp_path / "gate"
    upgrade.compare_packets(base, base, out)
    manifest = io.parse_json((out / "manifest.json").read_bytes())
    manifest["files"]["report.json"]["bytes"] = True
    (out / "manifest.json").write_bytes(io.json_bytes(manifest))
    assert upgrade.verify_comparison(out)["integrity"] == "failed"


def test_unexpected_empty_directory_or_symlink_is_not_ignored(tmp_path):
    base, _ = packet(tmp_path, "base")
    out = tmp_path / "gate"
    upgrade.compare_packets(base, base, out)
    (out / "extra").mkdir()
    rehash_packet(out)
    assert upgrade.verify_comparison(out)["integrity"] == "failed"
    (out / "link").symlink_to(base)
    assert upgrade.verify_comparison(out)["integrity"] == "failed"


def test_readable_report_escapes_untrusted_text(tmp_path):
    base, _ = packet(tmp_path, "base")
    report = upgrade.compare_packets(base, base, tmp_path / "gate")["report"]
    report = deepcopy(report)
    report["comparison_details"]["gaps"].append({"message": "<img src=x onerror=alert(1)>"})
    rendered = upgrade.render(report)
    assert b"<img src=x" not in rendered and b"&lt;img src=x" in rendered
    assert b"href='report.json'" in rendered


def test_cli_dispatch_and_exit_codes(tmp_path, capsys):
    from sdcguard.cli import main
    base, _ = packet(tmp_path, "base")
    assert main(["upgrade", "policy"]) == 0
    out = tmp_path / "gate"
    assert main(["upgrade", "compare", str(base), str(base), "--out", str(out)]) == 0
    assert main(["upgrade", "verify", str(out)]) == 0
    assert main(["upgrade", "verify", str(tmp_path / "missing")]) == 2
    assert "runtime_upgrade_witness" in capsys.readouterr().out
