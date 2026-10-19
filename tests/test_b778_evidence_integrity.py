"""End-to-end regressions discovered against the b778d27 source archive.

These exercise public file ingestion and saved incident artifacts rather than
only the helper functions that manufacture them.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sdcguard import board, certify, check, checksum, cli, fleet, screen


def _card():
    r = {"schema": board.SCHEMA, "result": "PASS", "date": "2026-10-19", "minutes": 5,
         "gpu": {"name": "NVIDIA H100", "compute_cap": "9.0", "id_hash": "0123456789ab"},
         "software": {}, "provider": {}, "checked": {"results": 100, "unjudged": 0},
         "wrong": {"workload_training": 0, "workload_serving": 0, "stress": 0, "memory_words": 0},
         "headroom": {}, "repeatable": True, "fingerprint": {"key": "abc", "output": "def"},
         "speed": {}, "health": {}, "notes": [], "board": None, "simulated_faults": False}
    r["digest"] = board.digest(r)
    return r


def test_board_accepts_real_file_ingestion_with_source_metadata(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "result.json").write_text(json.dumps(_card()))
    rows = board.from_dirs([str(inp)])
    assert rows[0]["_source"]["label"] == "result.json"
    report = board.build(rows, str(tmp_path / "board"))
    assert report["accepted"] == 1, report["rejected"]
    assert json.loads((tmp_path / "board" / "results.json").read_text())[0]["_source"] == rows[0]["_source"]


def test_board_standalone_script_needs_no_torch_or_package_install(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    card = _card()
    card["gpu"]["compute_cap"] = "7.5"
    card["math_modes"] = {"ran": ["fp16", "fp32", "fp64", "int8"], "matmuls": 4, "wrong": 0,
                          "not_checked": {"bf16": "needs Ampere", "tf32": "needs Ampere", "fp8": "needs Ada"},
                          "errors": {}, "typical_error_caught": {}}
    card["wrong"]["math_modes"] = 0
    card["digest"] = board.digest(card)
    (inp / "result.json").write_text(json.dumps(card))
    # -S and an empty PYTHONPATH leave only the script directory and the standard library.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-S", board.__file__, "--dir", str(inp),
                        "--out", str(tmp_path / "out")], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "board: 1 results" in p.stdout, p.stdout


def _signed_bundle(tmp_path):
    key, pub = certify.keygen(str(tmp_path / "issuer"))
    d = tmp_path / "bundle"
    d.mkdir()
    summary = {"gpu": {"gpu_uuid": "GPU-test"}, "result": "PASS", "matmuls_checked": 10,
               "evidence": [str(d / "evidence_0")]}
    for name in ("summary.json", "check.json", "result.json"):
        (d / name).write_text(json.dumps(summary))
    (d / "evidence_0.json").write_text('{"record": "test"}')
    (d / "evidence_0.pt").write_bytes(b"test evidence bytes")
    full = {"out_dir": str(d), "evidence": summary["evidence"]}
    certificate = certify.write(summary, str(d), key_path=key, full=full)
    return d, certificate, pub


@pytest.mark.parametrize("missing", ["check.json", "result.json", "evidence_0.pt", "evidence_0.json"])
def test_certificate_does_not_call_a_missing_manifest_file_a_match(tmp_path, missing):
    d, certificate, pub = _signed_bundle(tmp_path)
    assert certify.verify(certificate, pub)["ok"]
    (d / missing).unlink()
    verified = certify.verify(certificate, pub)
    assert verified["signature_valid"]  # signature and complete evidence are separate facts
    assert not verified["ok"] and verified["evidence_files_match"] is False
    assert any(missing in p and "missing" in p for p in verified["problems"])
    row = fleet.collect([str(d)])[0]
    assert row["result"] == "INCOMPLETE" and any(missing in n for n in row["notes"])


def test_a_copied_bundle_uses_its_local_evidence_not_the_old_absolute_path(tmp_path):
    d, _, _ = _signed_bundle(tmp_path)
    copied = tmp_path / "copy"
    shutil.copytree(d, copied)
    # Both locations still exist, as when reviewing a downloaded incident archive.
    rows = fleet.collect([str(copied)])
    assert rows[0]["evidence"] == [str(copied / "evidence_0")]


def test_batched_replay_preserves_bias_and_scaling(tmp_path):
    a = torch.ones(2, 4, 8, dtype=torch.bfloat16)
    b = torch.ones(2, 8, 128, dtype=torch.bfloat16)
    bias = torch.full((2, 4, 128), 64.0, dtype=torch.bfloat16)
    c = torch.baddbmm(bias, a, b, alpha=0.5, beta=2.0)
    c[1, 2, 3] += 256
    evidence = tmp_path / "batched.pt"
    torch.save({"A": a, "B": b, "C": c, "bias": bias, "alpha": 0.5, "beta": 2.0,
                "op": "aten::baddbmm", "where": "attention", "row": 2, "col_block": 0, "batch": 1,
                "epoch": 0}, evidence)
    replayed = screen.replay(str(evidence), times=2, device="cpu")
    assert replayed["recomputations_that_failed_the_check"] == 0, replayed
    rc = replayed["recorded_block_on_cpu"]
    assert rc["done"] and rc["judged"] and rc["output_wrong"], rc


def test_stress_and_workload_evidence_cannot_overwrite_each_other(tmp_path):
    a = torch.ones(2, 8, dtype=torch.bfloat16)
    b = torch.ones(8, 128, dtype=torch.bfloat16)
    c = a @ b
    c[0, 0] += 100
    res = checksum.check_matmul(a, b, c)
    stress = screen._save_evidence(str(tmp_path), 0, res,
                                  (a, b, c, None, 1.0, 1.0, "aten::mm", "stress check"),
                                  {}, {"phase": "stress"})
    training = screen._save_evidence(str(tmp_path), 0, res,
                                    (a, b, c, None, 1.0, 1.0, "aten::mm", "blocks.0"),
                                    {}, {"phase": "training"})
    assert stress != training
    assert json.loads(Path(stress + ".json").read_text())["phase"] == "stress"
    assert json.loads(Path(training + ".json").read_text())["phase"] == "training"


def test_child_that_crashes_cannot_reuse_previous_pass(tmp_path, monkeypatch):
    old = tmp_path / "out" / "GPU-test"
    old.mkdir(parents=True)
    (old / "check.json").write_text(json.dumps({"result": "PASS", "run_id": "previous"}))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class Crashed:
        def __init__(self, *a, **kw):
            (old.parent / ".child_0.json").write_text(json.dumps({"out_dir": str(old), "run_id": "current"}))

        def wait(self, **kw):
            return 2

    monkeypatch.setattr(check.subprocess, "Popen", Crashed)
    args = SimpleNamespace(out=str(old.parent), minutes=0.01, seed=0, simulate_faults=0.0, size="tiny", provider=None)
    found, problems, failed = check._children(args)
    assert found == [] and len(problems) == 1 and failed == 0


def test_stress_fault_before_a_later_error_keeps_counts_and_evidence(monkeypatch):
    original = checksum.check_matmul
    calls = 0

    def wrong_then_error(a, b, c):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("device stopped after the first corrupted result")
        c[1, 2] += 100
        return original(a, b, c)

    monkeypatch.setattr(check, "_mat_size", lambda dev: 16)
    monkeypatch.setattr(checksum, "check_matmul", wrong_then_error)
    result = check.stress_check(torch.device("cpu"), torch.bfloat16, 0.01)
    assert result["wrong"] == 1 and result["matmuls_checked"] == 1
    assert result["failures"] and "device stopped" in result["error"]


def test_simulated_nhc_failure_cannot_overwrite_a_real_health_status(tmp_path, monkeypatch, capsys):
    cli._nhc_status(str(tmp_path), "OK: a real screen passed", 0)
    previous = (tmp_path / "status.json").read_bytes()
    monkeypatch.setattr(cli, "_run_screen", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(cli, "_fresh_summaries", lambda *a: [(str(tmp_path), {
        "result": "FAIL", "SIMULATED_FAULTS": "yes", "gpu": {"gpu_uuid": "GPU-demo"}})])
    code = cli.main(["nhc", "--out", str(tmp_path), "--simulate-faults", "0.5"])
    assert code == 2 and "SIMULATED" in capsys.readouterr().out
    assert (tmp_path / "status.json").read_bytes() == previous


def test_simulated_fleet_failure_is_visible_but_not_a_quarantine_candidate(tmp_path):
    inp = tmp_path / "input"
    inp.mkdir()
    (inp / "summary.json").write_text(json.dumps({
        "result": "FAIL", "SIMULATED_FAULTS": "yes", "corrupted_results_training": 1,
        "gpu": {"gpu_uuid": "GPU-demo"}, "matmuls_checked": 1,
    }))
    report = fleet.write(fleet.collect([str(inp)]), str(tmp_path / "report"))
    assert report["stats"]["simulated_runs"] == 1 and report["gpus"][0]["result"] == "FAIL"
    assert (tmp_path / "report" / "quarantine.txt").read_text() == ""


def test_memory_fault_before_a_later_error_keeps_the_failed_word_count(monkeypatch):
    original = torch.ne
    calls = 0

    def wrong_then_error(actual, expected):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("memory test stopped after a bad word was read")
        actual[0] ^= 1
        return original(actual, expected)

    monkeypatch.setattr(torch, "ne", wrong_then_error)
    result = check.memory_test(torch.device("cpu"), 0.01)
    assert result["wrong_words"] == 1 and result["words_checked"] > 0
    assert "memory test stopped" in result["error"]


def test_partial_fingerprint_run_card_reports_only_completed_runs(tmp_path, monkeypatch):
    original = check.Fingerprinter.run
    calls = 0

    def third_run_stops(self):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("third fingerprint did not complete")
        return original(self)

    monkeypatch.setattr(check.Fingerprinter, "run", third_run_stops)
    full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    assert full["result"] == "INCOMPLETE" and len(full["fingerprints"]) == 2
    card = json.loads((Path(full["out_dir"]) / "result.json").read_text())
    repeatability = dict(check.lines(card))["repeatable"]
    assert "2 recorded runs bit-identical" in repeatability
    assert "3 of 3" not in repeatability
    # Older cards did not record this count; never invent it when reading one.
    card["fingerprint"].pop("runs", None)
    assert dict(check.lines(card))["repeatable"].startswith("recorded runs bit-identical")

