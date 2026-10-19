"""The cases an independent review reproduced (Oct 19, 2026), each turned into a test: a check that can't
run or can't judge never becomes a PASS, failures keep replayable evidence, serving leaves altered layers
alone, empty batches work on every path, and the math-mode check sees errors one checksum pattern can't."""
import html
import json
import os
import sys
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sdcguard import board, certify, check, modes, screen
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


def test_a_math_mode_check_that_errors_makes_the_result_incomplete(tmp_path):
    def fail(*a, **k):
        raise RuntimeError("simulated launch failure")

    with patch.object(modes, "check_once", side_effect=fail):
        full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    assert full["result"] == "INCOMPLETE"
    pub = check.public(full)
    rows = dict(check.lines(pub))
    assert rows["math modes"].startswith("none ran; ERROR in bf16 fp16 tf32 fp32 fp64 int8")
    assert any("couldn't run" in t for _lvl, t in pub["notes"])
    assert "INCOMPLETE" in check.svg(pub) and html.escape("couldn't run or couldn't judge") in check.svg(pub)
    card = dict(pub, gpu=dict(pub["gpu"], name="NVIDIA L4"), minutes=5)
    card.pop("digest")
    card["digest"] = board.digest(card)
    assert "incomplete" in board.validate(card)


def test_unsupported_modes_are_not_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(modes, "available", lambda m, dev: "needs compute capability 8.0 or newer"
                        if m in ("bf16", "tf32") else None)
    full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    assert full["result"] == "PASS" and full["math_modes"]["errors"] == {}
    assert "unsupported: bf16 tf32" in dict(check.lines(check.public(full)))["math modes"]


def _nan_init(real_init):
    def init(self, *a, **k):
        real_init(self, *a, **k)
        with torch.no_grad():
            for p in self.parameters():
                p.fill_(float("nan"))
    return init


def test_a_screen_that_cant_judge_its_workload_is_incomplete_and_so_is_its_certificate(tmp_path):
    with patch.object(screen.ScreenModel, "__init__", _nan_init(screen.ScreenModel.__init__)):
        s = screen.screen_one(torch.device("cpu"), minutes=0, size="tiny", out_root=str(tmp_path), max_steps=1)
    assert s["result"] == "INCOMPLETE" and s["unjudged_training_checks"] == s["matmuls_checked"] > 0
    assert s["serving_passes_with_nonfinite_output"] == 1
    cert = certify.make(s)
    assert cert["result"] == "INCOMPLETE" and cert["workload"]["unjudged_training_checks"] == s["matmuls_checked"]
    assert "doesn't vouch" in cert["meaning"]


def test_replay_doesnt_call_an_unjudgeable_result_correct(tmp_path):
    a = torch.full((4, 8), float("nan"), dtype=torch.bfloat16)
    b = torch.ones((8, 128), dtype=torch.bfloat16)
    path = tmp_path / "nan_evidence.pt"
    torch.save({"A": a, "B": b, "C": a @ b, "bias": None, "alpha": 1.0, "beta": 0.0, "op": "mm", "where": "test"}, path)
    r = screen.replay(str(path), times=2, device="cpu")
    assert r["recomputations_that_couldnt_be_judged"] == 2 and r["recorded_output_check"]["judged"] is False
    assert "no evidence either way" in r["reading"] and "correct" not in r["reading"].split(";")[0]


def test_a_math_mode_failure_saves_evidence_that_replays(tmp_path):
    real = modes.gemm

    def corrupt_int8(mode, a, b):
        c = real(mode, a, b)
        if mode == "int8":
            c[0, 0] += 1
        return c

    with patch.object(modes, "gemm", side_effect=corrupt_int8):
        full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    assert full["result"] == "FAIL" and full["wrong"]["math_modes"] > 0
    mode_ev = [e for e in full["evidence"] if os.path.basename(e).startswith("math_mode_evidence_")]
    assert mode_ev and all(os.path.exists(e + ".pt") and os.path.exists(e + ".json") for e in mode_ev)
    info = json.load(open(mode_ev[0] + ".json"))
    assert info["mode"] == "int8" and info["replay"].startswith("sdcguard replay ")
    with patch.object(modes, "gemm", side_effect=corrupt_int8):
        again = screen.replay(mode_ev[0] + ".pt", times=2, device="cpu")
    assert again["recomputations_that_failed_the_check"] == 2 and "wrong again" in again["reading"]
    clean = screen.replay(mode_ev[0] + ".pt", times=2, device="cpu")
    assert clean["recomputations_that_failed_the_check"] == 0 and clean["recorded_wrong_blocks"] >= 1


def test_a_stress_failure_saves_evidence(tmp_path, monkeypatch):
    from sdcguard import checksum as cs
    orig = cs.check_matmul
    calls = {"n": 0}

    def flaky(A, B, C, *a, **k):
        calls["n"] += 1
        if calls["n"] == 3 and A.shape[0] == check._mat_size(torch.device("cpu")):
            C = C.clone()
            C[0, 0] = 1000.0
        return orig(A, B, C, *a, **k)

    monkeypatch.setattr(cs, "check_matmul", flaky)
    full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path))
    assert full["wrong"]["stress"] >= 1 and full["result"] == "FAIL"
    assert any(os.path.exists(e + ".pt") and json.load(open(e + ".json")).get("phase") == "stress"
               for e in full["evidence"] if not os.path.basename(e).startswith("math_mode"))


def test_serving_leaves_a_math_changing_instance_forward_alone():
    lin = nn.Linear(128, 128, bias=False).bfloat16()
    with torch.no_grad():
        lin.weight.copy_(torch.eye(128).bfloat16())
    original = lin.forward
    lin.forward = lambda x: original(x) * 2
    x = torch.ones(1, 128).bfloat16()
    before = lin(x)
    g = ServeGuard().attach(lin)
    with torch.no_grad():
        after = lin(x)
        lin(x)
    assert torch.equal(before, after) and g.poll(wait=True) == []
    assert "forward replaced on the instance" in list(g.unsupported.values())[0]
    g.detach()
    assert lin.forward(x).equal(before)


def test_vllm_empty_batch_on_the_compiled_path(monkeypatch, tmp_path):
    sys.path.insert(0, HERE)
    from test_serve import _fake_vllm, _plugin_env, _plugin_model
    M, Runner = _fake_vllm(monkeypatch)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path)
    vllm_plugin.register()
    model = _plugin_model(M, n_layers=1)
    empty = torch.empty(0, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        assert model(empty).shape == (0, 256)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        assert compiled(empty).shape == (0, 256)
    vllm_plugin._STATE["guard"] = None


def test_vllm_reads_results_after_a_while_not_only_every_16_steps(monkeypatch, tmp_path):
    sys.path.insert(0, HERE)
    from test_serve import _fake_vllm, _plugin_env, _plugin_model
    M, Runner = _fake_vllm(monkeypatch)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path, SDCGUARD_POLL_EVERY="16", SDCGUARD_POLL_SECONDS="0")
    monkeypatch.setitem(vllm_plugin._STATE, "steps", 0)
    vllm_plugin.register()
    model = _plugin_model(M, n_layers=1)
    runner = Runner(model)
    g = vllm_plugin.guard()
    with torch.no_grad():
        model[0].weight.view(torch.int16)[5, 9] ^= (1 << 14)
        runner.execute_model(torch.randn(1, 256).bfloat16())
    assert g.stats["alerts"] == 1            # one request, then idle: delivered without waiting for 15 more
    vllm_plugin._STATE["guard"] = None


def test_vllm_time_based_reads_wait_for_the_gpu_and_count_based_ones_dont(monkeypatch, tmp_path):
    """A non-blocking read reports the previous one, so on a GPU an alert read on the time condition would
    otherwise wait for another step (outside review, Oct 19, 2026). There's no idle timer: documented."""
    sys.path.insert(0, HERE)
    from test_serve import _fake_vllm, _plugin_env, _plugin_model
    M, Runner = _fake_vllm(monkeypatch)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path, SDCGUARD_POLL_EVERY="3", SDCGUARD_POLL_SECONDS="2")
    monkeypatch.setitem(vllm_plugin._STATE, "steps", 0)
    clock = [1000.0]
    monkeypatch.setattr(vllm_plugin.time, "monotonic", lambda: clock[0])
    vllm_plugin.register()
    model = _plugin_model(M, n_layers=1)
    runner = Runner(model)
    g = vllm_plugin.guard()
    reads = []
    real_poll = g.poll
    monkeypatch.setattr(g, "poll", lambda wait=False: reads.append(wait) or real_poll(wait=wait))
    x = torch.randn(1, 256).bfloat16()
    with torch.no_grad():
        for t in (1000.1, 1000.2, 1000.3):        # steps 1-3 within 2 s: only step 3 reads, without waiting
            clock[0] = t
            runner.execute_model(x)
        assert reads == [False]
        clock[0] = 1003.0                          # 2.7 s after the last read: read right away, waiting
        runner.execute_model(x)
        assert reads == [False, True]
    vllm_plugin._STATE["guard"] = None


def test_a_persistent_column_swap_is_caught_by_the_math_modes():
    """Two outputs of a block swapped after every GEMM: invisible to the plain sum, and to any one signed
    sum where both columns share a sign. Fresh patterns every GEMM catch it."""
    real = modes.gemm

    def swapped(mode, a, b):
        c = real(mode, a, b).clone()
        c[:, [0, 1]] = c[:, [1, 0]]
        return c

    with patch.object(modes, "gemm", side_effect=swapped):
        r = modes.screen(torch.device("cpu"), 0, shape=(128, 256, 256))
    assert r["matmuls"] == len(modes.MODES) * len(modes.VARIANTS) and r["wrong"] >= 0.8 * r["matmuls"]


def test_board_rejects_cards_whose_math_mode_counts_disagree():
    seed = os.path.join(HERE, "..", "docs", "board", "seed")
    with open(os.path.join(seed, sorted(os.listdir(seed))[0])) as f:
        card = json.load(f)
    card["wrong"]["math_modes"] = 0
    card["math_modes"] = {"ran": ["fp32"], "matmuls": 6, "wrong": 1, "not_checked": {}, "errors": {},
                          "typical_error_caught": {"fp32": 0.001}}
    card.pop("digest", None)
    card["digest"] = board.digest(card)
    assert "disagree" in board.validate(card)


def test_math_mode_evidence_reaches_the_certificate_and_the_fleet_archive(tmp_path):
    """Found by an outside review (Oct 19, 2026): evidence from the math modes and the stress check was saved
    but left out of the certificate's hashes, the fleet report's counts and the RMA archive."""
    pytest.importorskip("cryptography")
    import zipfile
    from sdcguard import fleet
    real = modes.gemm

    def corrupt_int8(mode, a, b):
        c = real(mode, a, b)
        if mode == "int8":
            c[0, 0] += 1
        return c

    key, pub = certify.keygen(str(tmp_path / "issuer"))
    with patch.object(modes, "gemm", side_effect=corrupt_int8):
        full = check.check_one("cpu", minutes=0.01, out_root=str(tmp_path / "out"), key_path=key)
    assert full["result"] == "FAIL"
    mode_ev = [e for e in full["evidence"] if os.path.basename(e).startswith("math_mode_evidence_")]
    cert_path = os.path.join(full["out_dir"], "certificate.json")
    cert = json.load(open(cert_path))
    assert cert["result"] == "FAIL" and cert["corrupted_results"]["math_modes"] == full["wrong"]["math_modes"] > 0
    assert {os.path.basename(e) + ".pt" for e in mode_ev} <= set(cert["evidence_sha256"])
    assert certify.verify(cert_path, pub)["ok"]

    rows = fleet.collect([str(tmp_path / "out")])
    assert rows[0]["result"] == "FAIL" and rows[0]["corrupted"] == full["wrong"]["math_modes"]
    report = fleet.write(rows, str(tmp_path / "fleet"))
    with zipfile.ZipFile(next(iter(report["rma"].values()))) as z:
        names = set(z.namelist())
        readme = z.read("README.txt").decode()
    assert {"check.json", "certificate.json", "summary.json"} <= names
    assert {os.path.basename(e) + ext for e in mode_ev for ext in (".pt", ".json")} <= names
    assert "in the math-mode check" in readme and f"sdcguard replay {os.path.basename(mode_ev[0])}.pt" in readme

    with open(mode_ev[0] + ".pt", "ab") as f:            # tamper with one evidence file
        f.write(b"changed")
    v = certify.verify(cert_path, pub)
    assert not v["ok"] and not v["evidence_files_match"]


def test_any_unjudged_check_makes_the_run_incomplete():
    """The README says a run that couldn't judge part of its work isn't a PASS; until Oct 19, 2026 up to 1% of
    unjudged checks were let through with a warning (outside review)."""
    from sdcguard import verdict
    assert verdict.from_counts(0, 1000, 0) == "PASS"
    assert verdict.from_counts(0, 1000, 1) == "INCOMPLETE" == verdict.from_counts(0, 0, 0)
    assert verdict.from_counts(1, 1000, 5) == "FAIL"
    assert verdict.unjudged_note(1, 1000, "x")[0][0] == "bad"


def test_both_replay_commands_exit_the_same_way(monkeypatch):
    from types import SimpleNamespace
    from sdcguard import cli
    for r, code in (({"recomputations": 2, "recomputations_that_failed_the_check": 1}, 1),
                    ({"recomputations": 2, "recomputations_that_failed_the_check": 0,
                      "recomputations_that_couldnt_be_judged": 2}, 2),
                    ({"recomputations": 2, "recomputations_that_failed_the_check": 0,
                      "recomputations_that_couldnt_be_judged": 1}, 0)):
        monkeypatch.setattr(screen, "replay", lambda *a, _r=r, **k: _r)
        assert cli.cmd_replay(SimpleNamespace(evidence="x", times=2, device="cpu")) == code
        assert screen.main(["--replay", "x", "--times", "2", "--device", "cpu"]) == code
