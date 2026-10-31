"""Offline retained-record comparisons. Owner: Hari Krishna Kancharla.

Edited copies are synthetic evidence controls, never independent engine runs.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sdcguard import cli, evidence_io as io, required_gate as gate
from sdcguard import transfer_coverage as transfer, transfer_upgrade as upgrade
from sdcguard import kv_checksum_observer as observer
from test_transfer_coverage import repack

SAVED = Path(__file__).parent / "data/checksum_owner_3a2ad847"
NAMES = ("RESULT.json", "ordinary-evidence.zip", "CPU_PREFLIGHT.json")


@pytest.fixture(scope="module")
def saved():
    pin = json.loads((SAVED / "SOURCE.json").read_bytes())
    for name, identity in pin["files"].items():
        raw = (SAVED / name).read_bytes()
        assert {"bytes": len(raw), "sha256": io.digest(raw)} == identity
    return tuple((SAVED / n).read_bytes() for n in NAMES)


@pytest.fixture(scope="module")
def compared(saved):
    return upgrade.evaluate(saved, saved)


@pytest.fixture(scope="module")
def baseline(saved):
    return transfer.evaluate(*saved)


def test_real_saved_self_comparison_keeps_both_full_prompt_holds(compared, baseline):
    assert compared["upgrade_decision"] == "HOLD"
    assert compared["runtime_upgrade_witness"] == "not established"
    assert compared["recording_relation"] == "same_evidence"
    assert compared["build_relation"] == "same_recorded_build"
    assert compared["summary"] == dict(compared_slots=153, preserved=153, regressed=0,
                                      unverifiable=0, lost_passing_evidence=0)
    for side in ("baseline", "candidate"):
        assert compared[side]["full_prompt_gate"] == baseline["full_prompt_gate"]
        assert compared[side]["transfer_decision"] == "PASS"
        assert compared[side]["summary"]["compared_transfer_tokens"] == 9822
        assert compared[side]["summary"]["decode_cached_tokens_outside_transfer"] == 99968
    assert "independent_recordings_not_established" in compared["hold_reasons"]
    assert "changed_build_transition_not_qualified" in compared["hold_reasons"]


def test_one_lost_record_is_unknown_and_never_leaves_denominator(saved):
    def remove(captures, files):
        records = captures["captures"][0]["checksum_snapshots"][0]["records"]
        records.remove([r for r in records if r["kind"] == "verification"][1])
    result = upgrade.evaluate(saved, repack(saved, remove))
    assert result["summary"] == dict(compared_slots=153, preserved=152, regressed=0,
                                    unverifiable=1, lost_passing_evidence=1)
    row = result["requests"][1]
    assert row["coverage_change"] == "UNVERIFIABLE"
    assert row["candidate"]["tokens"]["expected_transfer"] == 22
    assert row["candidate"]["tokens"]["compared_transfer"] is None
    assert "transfer_plan_or_verification_missing_or_ambiguous" in row["candidate"]["reasons"]
    assert result["recording_relation"] == "overlapping_recording_identity"
    assert result["upgrade_decision"] == "HOLD"


def test_synthetic_recorded_mismatch_differs_from_missing_evidence(saved):
    def change(captures, files):
        records = captures["captures"][0]["checksum_snapshots"][0]["records"]
        row = [r for r in records if r["kind"] == "verification"][1]
        row["native_return"] = False
        row["proof"]["groups"][0]["native_group_result"] = [9]
        row["groups"][0].update(native_result="mismatch", returned_positions=[9])
        row.update(observer.classify_verification(row))
    result = upgrade.evaluate(saved, repack(saved, change))
    assert result["summary"]["regressed"] == 1
    assert result["requests"][1]["candidate"]["status"] == "MISMATCH"
    assert result["requests"][1]["reasons"] == ["candidate_recorded_native_mismatch"]
    assert result["upgrade_decision"] == "HOLD"


@pytest.mark.parametrize("change", ["unsupported_proof", "scope_change", "malformed_replica"])
def test_mismatch_with_uncertain_or_changed_scope_is_unverifiable(baseline, change):
    left = deepcopy(baseline["requests"][1]); right = deepcopy(left)
    right.update(status="MISMATCH", decision="HOLD", reasons=["native_checksum_mismatch"])
    right["comparison"].update(status="MISMATCH", native_result="mismatch", reasons=["native_checksum_mismatch"])
    if change == "unsupported_proof":
        right["reasons"].append("source_profile_unverified")
    elif change == "scope_change":
        right["tokens"]["expected_transfer"] += 16
        right["tokens"]["decode_cache_outside_transfer"] -= 16
    else:
        right["comparison"]["compared_units"] = None
    assert upgrade._change(left, right, True, True)[0] == "UNVERIFIABLE"


@pytest.mark.parametrize("change", ["version", "binary", "outer_run_id", "bad_json", "missing_preflight"])
def test_unqualified_inputs_cannot_preserve_coverage_or_claim_new_build(saved, change):
    candidate = list(saved)
    if change in ("version", "binary"):
        value = json.loads(candidate[2])
        if change == "version":
            value["installed_engine"]["versions"]["vllm"] = "synthetic-unknown-version"
        else:
            member = next(iter(value["installed_engine"]["compiled_members"].values()))
            member["sha256"] = "a" * 64
        candidate[2] = io.json_bytes(value)
    elif change == "outer_run_id":
        value = json.loads(candidate[0]); value["run_id"] = "synthetic-other-run"
        candidate[0] = io.json_bytes(value)
    elif change == "bad_json":
        candidate[0] = b'{"run_id":1,"run_id":2}'
    else:
        candidate[2] = None
    result = upgrade.evaluate(saved, tuple(candidate))
    assert result["upgrade_decision"] == "HOLD"
    assert result["comparison_status"] == "UNVERIFIABLE"
    assert result["summary"]["compared_slots"] == 153
    assert result["summary"]["preserved"] == result["summary"]["regressed"] == 0
    assert result["build_relation"] == "unknown"


def test_repacking_or_changing_paths_is_not_an_independent_run(saved, tmp_path):
    copied = tmp_path / "copy"; copied.mkdir()
    candidate = repack(saved, lambda captures, files: None)
    for name, raw in zip(NAMES, candidate):
        (copied / name).write_bytes(raw)
    value = upgrade.inspect(SAVED, copied)
    assert value["recording_relation"] == "overlapping_recording_identity"
    assert value["build_relation"] == "same_recorded_build"
    assert value["summary"]["preserved"] == 153
    assert value["upgrade_decision"] == "HOLD"


def test_engine_identity_ignores_run_paths_observer_witness_and_installer_version(saved):
    pre = json.loads(saved[2]); changed = deepcopy(pre)
    changed.update(run_id="different", packet_sha256="f" * 64, build_witness={"edited": True}, product={"edited": True})
    changed["installed_engine"]["direct_url"] = {"url": "file:///different"}
    changed["installed_engine"]["versions"]["uv"] = "different"
    changed["installed_engine"]["versions"]["triton"] = "different"
    changed["installed_engine"]["python_identity"]["version"] = [3, 12, 4]
    changed["installed_engine"]["python_identity"]["prefix"] = "/different"
    assert upgrade._engine_identity(pre) == upgrade._engine_identity(changed)


@pytest.mark.parametrize("field", ["source_revision", "source_files", "compiled_members", "payload"])
def test_engine_identity_uses_recorded_engine_bytes(saved, field):
    pre = json.loads(saved[2]); changed = deepcopy(pre); engine = changed["installed_engine"]
    if field == "source_revision":
        engine[field] = "a" * 40
    elif field in ("source_files", "compiled_members"):
        next(iter(engine[field].values()))["sha256"] = "b" * 64
    elif field == "payload":
        next(iter(engine["precompiled_payload"]["extracted_members"].values()))["sha256"] = "c" * 64
    assert upgrade._engine_identity(pre) != upgrade._engine_identity(changed)


@pytest.mark.parametrize("change", ["prompt", "order", "concurrency", "kind", "group", "max_tokens",
    "temperature", "omitted_temperature", "special_tokens", "headers", "path", "extra_sampling"])
def test_workload_contract_checks_actual_requests_and_order(saved, change):
    files = gate.read_archive(saved[1]); pre = json.loads(saved[2])
    original = upgrade._contracts(files, pre)
    plan = json.loads(files["client/REQUEST_PLAN.json"])
    inventory = json.loads(files["client/CLIENT_INVENTORY.json"])
    row = plan["arms"][0]["requests"][0]; stage = inventory["requests"][0]["decode"]
    if change == "prompt": stage["request_body"]["prompt"][0] += 1
    elif change == "order": plan["arms"][0]["requests"][:2] = plan["arms"][0]["requests"][1::-1]
    elif change == "concurrency": row["concurrency"] = 2
    elif change == "kind": row["kind"] = "measured"
    elif change == "group": row["cache_group"] = "different"
    elif change == "max_tokens": stage["request_body"]["max_tokens"] += 1
    elif change == "temperature": stage["request_body"]["temperature"] = 0.1
    elif change == "omitted_temperature": del stage["request_body"]["temperature"]
    elif change == "special_tokens": stage["request_body"]["add_special_tokens"] = True
    elif change == "headers": stage["request_headers"]["Example"] = "different"
    elif change == "path": stage["path"] = "/v1/chat/completions"
    else: stage["request_body"]["top_k"] = 4
    files["client/REQUEST_PLAN.json"] = io.json_bytes(plan)
    files["client/CLIENT_INVENTORY.json"] = io.json_bytes(inventory)
    assert original["workload_sha256"] != upgrade._contracts(files, pre)["workload_sha256"]


@pytest.mark.parametrize("change", ["model", "tokenizer", "generation", "kv_dtype", "block_size", "parallel", "prefix", "transport", "runtime_version", "python_patch", "missing_config"])
def test_model_and_configuration_drift_is_not_like_for_like(saved, change):
    files = gate.read_archive(saved[1]); pre = json.loads(saved[2]); original = upgrade._contracts(files, pre)
    captures = json.loads(files["CAPTURES.json"]); startup = captures["captures"][0]["startup"]
    if change in ("model", "tokenizer", "generation"):
        name = {"model": "model.safetensors", "tokenizer": "tokenizer.json", "generation": "generation_config.json"}[change]
        pre["model_cache"]["files"][name]["sha256"] = "d" * 64
    elif change == "kv_dtype": startup["kv_dtype"] = "fp8"
    elif change == "block_size": startup["block_size"] = 32
    elif change == "parallel": startup["parallel"]["tp"] = 2
    elif change == "prefix": startup["prefix_caching"] = False
    elif change == "transport": startup["transport_profile"]["parameters"]["ucx_error_handling_mode"] = "peer"
    elif change == "runtime_version": pre["installed_engine"]["versions"]["triton"] = "3.9.0"
    elif change == "python_patch": pre["installed_engine"]["python_identity"]["version"] = [3, 12, 4]
    else: del startup["max_num_seqs"]
    files["CAPTURES.json"] = io.json_bytes(captures)
    if change == "missing_config":
        with pytest.raises(io.EvidenceError, match="configuration is incomplete"):
            upgrade._contracts(files, pre)
    else:
        assert original["config_sha256"] != upgrade._contracts(files, pre)["config_sha256"]


@pytest.mark.parametrize("change", ["missing_triton", "empty_triton", "unknown_triton", "null_abi", "boolean_patch"])
def test_matching_unknown_runtime_facts_are_not_a_contract(saved, change):
    pre = json.loads(saved[2]); engine = pre["installed_engine"]
    if change == "missing_triton": del engine["versions"]["triton"]
    elif change == "empty_triton": engine["versions"]["triton"] = ""
    elif change == "unknown_triton": engine["versions"]["triton"] = "unknown"
    elif change == "null_abi": engine["python_identity"]["soabi"] = None
    else: engine["python_identity"]["version"][-1] = True
    for side in (pre, deepcopy(pre)):
        with pytest.raises(io.EvidenceError, match="runtime identity"):
            upgrade._runtime_identity(side)


def test_both_inputs_with_self_consistent_unknown_runtime_stay_unverifiable(saved):
    from sdcguard.native_operator import installed
    pre = json.loads(saved[2])
    pre["installed_engine"]["versions"]["triton"] = "unknown"
    witness = installed.witness(pre)
    digest = installed.digest(installed.encoded(witness))
    pre.update(build_witness=witness, build_witness_sha256=digest)
    def mutate(captures, files):
        files["BUILD_WITNESS.json"] = io.json_bytes(dict(witness=witness, sha256=digest))
    raw = list(repack(saved, mutate))
    result = json.loads(raw[0]); result["build_witness_sha256"] = digest
    raw[0], raw[2] = io.json_bytes(result), io.json_bytes(pre)
    # The existing transfer policy deliberately has no dependency-version
    # comparison obligation. The new comparator must add it without altering it.
    assert transfer.evaluate(*raw)["transfer_decision"] == "PASS"
    result = upgrade.evaluate(tuple(raw), tuple(raw))
    assert result["comparison_status"] == "UNVERIFIABLE"
    assert result["summary"]["preserved"] == 0
    assert result["summary"]["unverifiable"] == 153
    assert result["upgrade_decision"] == "HOLD"


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_slot_inventory_never_uses_intersection_only(saved, change):
    files = gate.read_archive(saved[1]); pre = json.loads(saved[2])
    inventory = json.loads(files["client/CLIENT_INVENTORY.json"])
    if change == "missing": inventory["requests"].pop()
    else:
        value = deepcopy(inventory["requests"][0])
        if change == "extra": value["slot"] = 10000
        inventory["requests"].append(value)
    files["client/CLIENT_INVENTORY.json"] = io.json_bytes(inventory)
    with pytest.raises(io.EvidenceError): upgrade._contracts(files, pre)


@pytest.mark.parametrize("change", ["cache_span", "equal_count_different_positions", "prefill_cache", "zero", "baseline_unknown", "missing_side"])
def test_scope_changes_and_unknown_baseline_never_become_preservation(baseline, change):
    left = deepcopy(baseline["requests"][1]); right = deepcopy(left)
    if change == "cache_span":
        right["tokens"]["expected_transfer"] -= 16; right["tokens"]["decode_cache_outside_transfer"] += 16
    elif change == "equal_count_different_positions": right["comparison"]["expected_units"] = ["group:0:chunk:8", "group:0:chunk:9"]
    elif change == "prefill_cache": right["tokens"]["prefill_cache_reused"] += 16
    elif change == "zero": right.update(decision="NOT_EXERCISED", status="NOT_EXERCISED")
    elif change == "baseline_unknown": left.update(decision="HOLD", status="UNOBSERVABLE")
    else: right = None
    assert upgrade._change(left, right, True, True)[0] == "UNVERIFIABLE"


def test_cli_saved_result_is_offline_hold_with_readable_and_json_modes(compared, monkeypatch, capsys):
    monkeypatch.setattr(upgrade, "inspect", lambda *a: compared)
    assert cli.main(["transfer-compare", str(SAVED), str(SAVED), "--json"]) == 2
    assert json.loads(capsys.readouterr().out) == compared
    assert cli.main(["transfer-compare", str(SAVED), str(SAVED)]) == 2
    text = capsys.readouterr().out
    assert "153 preserved, 0 regressed, 0 unverifiable" in text
    assert "same_evidence" in text and "full-prompt HOLD" in text


def test_cli_does_not_import_engine_sdk_device_or_network_modules():
    import subprocess, sys
    code = '''import sys
class Guard:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'torch','vllm','modal','nixl','requests','httpx'}:
   raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
from sdcguard.cli import main
raise SystemExit(main(['transfer-compare',sys.argv[1],sys.argv[1],'--json']))
'''
    result = subprocess.run([sys.executable, "-B", "-c", code, str(SAVED)], capture_output=True, text=True)
    assert result.returncode == 2 and not result.stderr
    assert json.loads(result.stdout)["summary"]["preserved"] == 153


def test_rechecks_both_inputs_after_comparing(saved, tmp_path, monkeypatch):
    paths = [tmp_path / "before", tmp_path / "after"]
    for path in paths:
        path.mkdir()
        for name, raw in zip(NAMES, saved): (path / name).write_bytes(raw)
    def mutate(*args):
        (paths[0] / "RESULT.json").write_bytes(b'{}')
        return {}
    monkeypatch.setattr(upgrade, "evaluate", mutate)
    with pytest.raises(io.EvidenceError, match="changed during inspection"):
        upgrade.inspect(*paths)


def test_optional_preflight_appearing_during_comparison_is_refused(saved, tmp_path, monkeypatch):
    for name, raw in zip(NAMES[:2], saved[:2]): (tmp_path / name).write_bytes(raw)
    def mutate(*args):
        (tmp_path / "CPU_PREFLIGHT.json").write_bytes(saved[2]); return {}
    monkeypatch.setattr(upgrade, "evaluate", mutate)
    with pytest.raises(io.EvidenceError, match="changed during inspection"):
        upgrade.inspect(tmp_path, tmp_path)


def test_nonregular_input_and_missing_directory_are_held(tmp_path, capsys):
    (tmp_path / "RESULT.json").symlink_to(SAVED / "RESULT.json")
    assert upgrade.main([str(tmp_path), str(SAVED), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["upgrade_decision"] == "HOLD"
    assert upgrade.main([str(tmp_path / "absent"), str(SAVED)]) == 2
    assert "evidence_unverifiable" in capsys.readouterr().out


def test_original_fixture_hashes_remain_pinned(saved):
    assert tuple((SAVED / n).read_bytes() for n in NAMES) == saved
