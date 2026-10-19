"""Release checks must distinguish unsupported acceleration from failed verification and failed behavior."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sdcguard import _group, selftest


def _triton_environment(monkeypatch, capability=(12, 0)):
    # (12: a generation the kernels haven't been validated on. Blackwell, 10, runs them since Oct 19.)
    monkeypatch.setattr(_group, "enabled", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    monkeypatch.setattr(_group, "FALLBACKS", [0])
    monkeypatch.setattr(_group, "WHY", [])
    monkeypatch.delenv("SDCGUARD_TRITON_NEW_ARCH", raising=False)
    monkeypatch.setattr(_group, "available", lambda *_: pytest.fail("policy-disabled kernel was probed"))


def test_selftest_distinguishes_an_intentional_fallback_on_a_new_generation(monkeypatch):
    _triton_environment(monkeypatch)
    checks, note = selftest._training_triton_checks(torch.device("cuda:0"), torch.bfloat16)
    assert all(passed for _, passed in checks)
    assert "intentionally disabled" in note and "plain check" in note
    assert not any("passed its first-use comparison" in label for label, _ in checks)


def test_selftest_blackwell_opted_out_is_an_intentional_fallback(monkeypatch):
    _triton_environment(monkeypatch, (10, 0))
    monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "0")
    checks, note = selftest._training_triton_checks(torch.device("cuda:0"), torch.bfloat16)
    assert all(passed for _, passed in checks) and "intentionally disabled" in note


def test_selftest_does_not_suppress_a_real_fallback_error(monkeypatch):
    _triton_environment(monkeypatch)
    _group.FALLBACKS[0] = 1
    _group.WHY.append("earlier kernel launch failed")
    checks, _ = selftest._training_triton_checks(torch.device("cuda:0"), torch.bfloat16)
    assert any(not passed and "earlier kernel launch failed" in label for label, passed in checks)


def test_selftest_unknown_capability_is_not_an_intentional_skip(monkeypatch):
    _triton_environment(monkeypatch)

    def failed(*_):
        raise RuntimeError("driver query failed")

    monkeypatch.setattr(torch.cuda, "get_device_capability", failed)
    checks, _ = selftest._training_triton_checks(torch.device("cuda:0"), torch.bfloat16)
    assert any(not passed and "driver query failed" in label for label, passed in checks)


@pytest.mark.parametrize("capability,opt_in", [((9, 0), False), ((10, 0), False), ((12, 0), True)])
def test_selftest_required_kernel_verification_still_fails(capability, opt_in, monkeypatch):
    _triton_environment(monkeypatch, capability)
    if opt_in:
        monkeypatch.setenv("SDCGUARD_TRITON_NEW_ARCH", "1")
    monkeypatch.setattr(torch, "empty", lambda *a, **k: object())
    monkeypatch.setattr(_group, "available", lambda *_: False)
    monkeypatch.setattr(_group, "status", lambda *_: (False, "clean/corrupt comparison disagreed"))
    checks, _ = selftest._training_triton_checks(torch.device("cuda:0"), torch.bfloat16)
    assert any(not passed and "comparison disagreed" in label for label, passed in checks)


@pytest.fixture
def vllm_experiment():
    path = Path(__file__).resolve().parents[1] / "experiments" / "vllm_check.py"
    spec = importlib.util.spec_from_file_location("review_vllm_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reports():
    baseline = {"model": "test-model", "vllm": "0.31.0", "enforce_eager": False,
                "output_tokens_per_s": 1000.0, "first_outputs": ["hello"]}
    guarded = {**baseline, "share": "1", "clean": [{"layers": 1, "layers_checked_since_last_read": 1,
               "alerts": [], "changed_weights": [], "worst_clean_z": 0.2, "triton_layers": 0}],
               "flipped": ["layer"], "after_flip": [{"alerts": [{"where": "layer"}],
                                                      "changed_weights": ["layer"]}]}
    return baseline, guarded


@pytest.mark.parametrize("fault", [None, "outputs", "clean_alert", "no_checks", "missed_fault",
                                   "wrong_layer", "missing_worker", "metadata"])
def test_vllm_experiment_exit_code_tracks_behavior_even_when_processes_succeed(
        fault, monkeypatch, tmp_path, vllm_experiment):
    baseline, guarded = _reports()
    if fault == "outputs":
        guarded["first_outputs"] = ["different"]
    elif fault == "clean_alert":
        guarded["clean"][0]["alerts"] = [{"where": "layer"}]
    elif fault == "no_checks":
        guarded["clean"][0]["layers_checked_since_last_read"] = 0
    elif fault == "missed_fault":
        guarded["after_flip"][0]["alerts"] = []
    elif fault == "wrong_layer":
        guarded["after_flip"][0]["alerts"] = [{"where": "other-layer"}]
    elif fault == "missing_worker":
        guarded["after_flip"] = []
    elif fault == "metadata":
        guarded["enforce_eager"] = True

    def completed(cmd, **_):
        role = cmd[cmd.index("--role") + 1]
        (tmp_path / f"{role}.json").write_text(json.dumps(guarded if role == "guarded" else baseline))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(vllm_experiment.subprocess, "run", completed)
    monkeypatch.setattr(vllm_experiment.sys, "argv", ["vllm_check.py", "--out", str(tmp_path)])
    assert vllm_experiment.main() == (0 if fault is None else 1)


def test_vllm_failed_processes_do_not_accept_stale_results(monkeypatch, tmp_path, vllm_experiment):
    baseline, guarded = _reports()
    for name, value in (("baseline", baseline), ("guarded", guarded), ("vanilla", baseline)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    monkeypatch.setattr(vllm_experiment.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(vllm_experiment.sys, "argv", ["vllm_check.py", "--out", str(tmp_path)])
    assert vllm_experiment.main() == 1


def test_vllm_failed_vanilla_process_is_not_reported_as_success(monkeypatch, tmp_path, vllm_experiment):
    baseline, guarded = _reports()

    def completed(cmd, **_):
        role = cmd[cmd.index("--role") + 1]
        (tmp_path / f"{role}.json").write_text(json.dumps(guarded if role == "guarded" else baseline))
        return SimpleNamespace(returncode=1 if role == "vanilla" else 0)

    monkeypatch.setattr(vllm_experiment.subprocess, "run", completed)
    monkeypatch.setattr(vllm_experiment.sys, "argv", ["vllm_check.py", "--out", str(tmp_path)])
    assert vllm_experiment.main() == 1
