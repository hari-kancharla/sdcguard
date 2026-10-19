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
    assert "changes the output" in list(g.unsupported.values())[0]
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
