"""Modal vLLM orchestration preserves validation failures and diagnostics; no Modal/GPU calls occur."""
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch, tmp_path):
    class Builder:
        def __getattr__(self, _):
            return lambda *a, **k: self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **_):
            return lambda f: f

        def local_entrypoint(self):
            return lambda f: f

    fake = SimpleNamespace(Image=SimpleNamespace(debian_slim=lambda **_: Builder()), App=App,
                           Volume=SimpleNamespace(from_name=lambda *a, **k: Builder()))
    monkeypatch.setitem(sys.modules, "modal", fake)
    path = Path(__file__).resolve().parents[1] / "scripts/modal_vllm.py"
    spec = importlib.util.spec_from_file_location("review_modal_vllm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    monkeypatch.setattr(module, "RUNS", str(tmp_path / "remote-results"))
    monkeypatch.setattr(module, "_locked", lambda *a: None)
    commits = []
    monkeypatch.setattr(module, "vol", SimpleNamespace(commit=lambda: commits.append(True)))
    module.review_commits = commits
    return module


def _fake_tee(module, monkeypatch, codes):
    calls = []

    def tee(cmd, log_path, **_):
        name = Path(log_path).stem
        calls.append(name)
        value = codes.get(name, 0)
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(f"fixture log for {name}: {value}\n")
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(module, "_tee", tee)
    return calls


@pytest.mark.parametrize("failed", ["entry_points", "qwen1.5b", "qwen7b"])
def test_failed_vllm_step_fails_remote_task_and_keeps_diagnostics(runner, monkeypatch, failed):
    calls = _fake_tee(runner, monkeypatch, {failed: 9})
    with pytest.raises(RuntimeError, match="validation failed"):
        runner.vllm()
    out = Path(runner.RUNS) / "vllm"
    status = json.loads((out / "STATUS.json").read_text())
    assert status["passed"] is False and status["exit_codes"][failed] == 9
    assert (out / f"{failed}.txt").exists() and (out / "TIME.txt").exists()
    assert runner.review_commits
    if failed == "entry_points":
        assert calls == ["entry_points"], "Do not rent more model work after a broken plugin setup."
    else:
        assert calls == ["entry_points", "qwen1.5b", "qwen7b"]


def test_complete_vllm_success_has_all_three_results(runner, monkeypatch):
    _fake_tee(runner, monkeypatch, {})
    assert "finished" in runner.vllm()
    status = json.loads((Path(runner.RUNS) / "vllm/STATUS.json").read_text())
    assert status == {"exit_codes": {"entry_points": 0, "qwen1.5b": 0, "qwen7b": 0}, "passed": True}


@pytest.mark.parametrize("failure", [7, RuntimeError("demo launch failed")])
def test_demo_failures_are_committed_and_propagated(runner, monkeypatch, failure):
    _fake_tee(runner, monkeypatch, {"demo": failure})
    with pytest.raises(RuntimeError):
        runner.demo()
    out = Path(runner.RUNS) / "demo"
    assert (out / "demo.txt").exists()
    assert json.loads((out / "STATUS.json").read_text())["passed"] is False
    assert runner.review_commits


def test_demo_success_is_explicit(runner, monkeypatch):
    _fake_tee(runner, monkeypatch, {})
    assert "exit code 0" in runner.demo()
    assert json.loads((Path(runner.RUNS) / "demo/STATUS.json").read_text())["passed"] is True


@pytest.mark.parametrize("phase", ["vllm", "demo"])
def test_locked_phase_is_not_a_fresh_validation(runner, monkeypatch, phase):
    calls = _fake_tee(runner, monkeypatch, {})
    monkeypatch.setattr(runner, "_locked", lambda *a: "NOT STARTED: an earlier run may be active")
    with pytest.raises(RuntimeError, match="NOT STARTED"):
        getattr(runner, phase)()
    assert calls == []


def _archive():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        payload = b"saved diagnostics\n"
        info = tarfile.TarInfo("gpu_r3/log.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return data.getvalue()


def _remote(value):
    def call(*a, **k):
        if isinstance(value, Exception):
            raise value
        return value
    return SimpleNamespace(remote=call)


@pytest.mark.parametrize("phase", ["vllm", "demo", "fetch"])
def test_download_failure_is_not_success(runner, monkeypatch, phase, capsys):
    monkeypatch.setattr(runner, "vllm", _remote("vllm finished"))
    monkeypatch.setattr(runner, "demo", _remote("demo finished"))
    monkeypatch.setattr(runner, "bundle", _remote(RuntimeError("download interrupted")))
    with pytest.raises(SystemExit, match="download failed"):
        runner.main(phase)
    assert "Done." not in capsys.readouterr().out


@pytest.mark.parametrize("outcome", [RuntimeError("kernel failed"), "NOT STARTED: lock exists"])
def test_failed_or_locked_remote_still_downloads_then_fails(runner, monkeypatch, outcome, capsys):
    monkeypatch.setattr(runner, "vllm", _remote(outcome))
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    with pytest.raises(SystemExit):
        runner.main("vllm")
    assert (runner.LOCAL / "runs/gpu_r3/log.txt").read_bytes() == b"saved diagnostics\n"
    assert "Done." not in capsys.readouterr().out


@pytest.mark.parametrize("phase", ["vllm", "demo", "fetch"])
def test_successful_remote_and_download_complete(runner, monkeypatch, phase, capsys):
    monkeypatch.setattr(runner, "vllm", _remote("vllm finished"))
    monkeypatch.setattr(runner, "demo", _remote("demo finished"))
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    runner.main(phase)
    assert (runner.LOCAL / "runs/gpu_r3/log.txt").exists()
    assert "Done." in capsys.readouterr().out


def test_remote_and_download_failures_are_both_preserved(runner, monkeypatch):
    monkeypatch.setattr(runner, "vllm", _remote(RuntimeError("kernel failed")))
    monkeypatch.setattr(runner, "bundle", _remote(RuntimeError("download interrupted")))
    with pytest.raises(SystemExit) as exc:
        runner.main("vllm")
    assert "kernel failed" in str(exc.value) and "download interrupted" in str(exc.value)


def test_experiment_gate_accepts_real_current_serving_report_fields(monkeypatch):
    import torch
    from sdcguard import vllm_plugin
    from sdcguard.serve import ServeGuard

    path = Path(__file__).resolve().parents[1] / "experiments/vllm_check.py"
    spec = importlib.util.spec_from_file_location("review_vllm_report_compatibility", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    torch.manual_seed(927)
    layer = torch.nn.Linear(16, 32).bfloat16()
    guard = ServeGuard().attach(torch.nn.ModuleDict({"layer": layer}))
    monkeypatch.setitem(vllm_plugin._STATE, "guard", guard)
    x = torch.randn(4, 16).bfloat16()
    try:
        with torch.no_grad():
            layer(x)
        clean = experiment._guard_report(None)
        with torch.no_grad():
            layer.weight[0, 0].add_(100)
            layer(x)
        after = experiment._guard_report(None)
        assert experiment._guarded_failures({"clean": [clean], "after_flip": [after], "flipped": ["layer"]}) == []
    finally:
        guard.detach()
