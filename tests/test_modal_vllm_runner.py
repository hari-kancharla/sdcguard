"""Modal vLLM orchestration preserves validation failures and diagnostics; no Modal/GPU calls occur."""
import importlib.util
import io
import json
import os
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
    spec = importlib.util.spec_from_file_location("modal_vllm_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    monkeypatch.setattr(module, "RUNS", str(tmp_path / "remote-results"))
    monkeypatch.setattr(module, "_locked", lambda *a: None)
    commits = []
    monkeypatch.setattr(module, "vol", SimpleNamespace(commit=lambda: commits.append(True)))
    module.volume_commits = commits
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
    assert runner.volume_commits
    if failed == "entry_points":
        assert calls == ["entry_points"], "Do not rent more model work after a broken plugin setup."
    else:
        assert calls == ["entry_points", "qwen1.5b", "qwen7b"]


def test_complete_vllm_success_has_all_three_results(runner, monkeypatch):
    _fake_tee(runner, monkeypatch, {})
    assert "finished" in runner.vllm()
    status = json.loads((Path(runner.RUNS) / "vllm/STATUS.json").read_text())
    assert status["exit_codes"] == {"entry_points": 0, "qwen1.5b": 0, "qwen7b": 0} and status["passed"] is True
    assert {"source", "minutes", "estimated_cost_usd", "cost_note"} <= set(status)


def test_a_new_run_keeps_the_earlier_runs_evidence(runner, monkeypatch):
    _fake_tee(runner, monkeypatch, {})
    earlier = Path(runner.RUNS) / "vllm"
    earlier.mkdir(parents=True)
    (earlier / "qwen7b.txt").write_text("the earlier run's log\n")
    runner.vllm(source={"sha256": "abc", "files": 1})
    kept = [p for p in Path(runner.RUNS).iterdir() if p.name.startswith("vllm_before_")]
    assert len(kept) == 1 and (kept[0] / "qwen7b.txt").read_text() == "the earlier run's log\n"
    assert json.loads((earlier / "STATUS.json").read_text())["source"] == {"sha256": "abc", "files": 1}


@pytest.mark.parametrize("failure", [7, RuntimeError("demo launch failed")])
def test_demo_failures_are_committed_and_propagated(runner, monkeypatch, failure):
    _fake_tee(runner, monkeypatch, {"demo": failure})
    with pytest.raises(RuntimeError):
        runner.demo()
    out = Path(runner.RUNS) / "demo"
    assert (out / "demo.txt").exists()
    assert json.loads((out / "STATUS.json").read_text())["passed"] is False
    assert runner.volume_commits


def test_demo_success_is_explicit(runner, monkeypatch):
    _fake_tee(runner, monkeypatch, {})
    assert "exit code 0" in runner.demo()
    assert json.loads((Path(runner.RUNS) / "demo/STATUS.json").read_text())["passed"] is True


@pytest.mark.parametrize("phase", ["vllm", "demo", "machete_diag"])
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


@pytest.mark.parametrize("phase", ["vllm", "demo", "machete_diag", "fetch"])
def test_successful_remote_and_download_complete(runner, monkeypatch, phase, capsys):
    monkeypatch.setattr(runner, "vllm", _remote("vllm finished"))
    monkeypatch.setattr(runner, "demo", _remote("demo finished"))
    monkeypatch.setattr(runner, "machete_diag", _remote({"message": "machete_diag finished", "fix": None}))
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
    spec = importlib.util.spec_from_file_location("vllm_report_under_test", path)
    experiment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment)
    torch.manual_seed(927)
    layer = torch.nn.Linear(16, 32).bfloat16()
    guard = ServeGuard().attach(torch.nn.ModuleDict({"layer": layer}))
    monkeypatch.setitem(vllm_plugin._STATE, "guard", guard)
    x = torch.randn(4, 16).bfloat16()
    try:
        startup = experiment._guard_report(None)
        with torch.no_grad():
            layer(x)
        clean = experiment._guard_report(None)
        with torch.no_grad():
            layer.weight[0, 0].add_(100)
            layer(x)
        after = experiment._guard_report(None)
        assert experiment._guarded_failures({"startup": [startup], "clean": [clean], "after_flip": [after],
                                            "flipped": ["layer"]}) == []
    finally:
        guard.detach()


def test_machete_diag_runs_the_diagnostic_and_records_its_exit_and_fix(runner, monkeypatch):
    calls = []

    def tee(cmd, log_path, **_):
        calls.append(Path(log_path).stem)
        assert cmd[cmd.index("--model") + 1] == "Qwen/Qwen2.5-7B-Instruct-AWQ"     # round 3 (docs/METHOD.md)
        assert cmd[cmd.index("--prompts") + 1] == "256"
        Path(log_path).write_text("log\n")
        Path(cmd[cmd.index("--out") + 1], "DIAG.json").write_text(json.dumps({"fix": "fused"}))
        return 0
    monkeypatch.setattr(runner, "_tee", tee)
    r = runner.machete_diag()
    assert r["fix"] == "fused" and "finished" in r["message"] and "met the rule: fused" in r["message"]
    assert calls == ["diag"]
    status = json.loads((Path(runner.RUNS) / "machete_diag/STATUS.json").read_text())
    assert status["passed"] is True and status["fix"] == "fused"
    _fake_tee(runner, monkeypatch, {"diag": 1})
    with pytest.raises(RuntimeError, match="exit code 1"):
        runner.machete_diag(force=True)
    status = json.loads((Path(runner.RUNS) / "machete_diag/STATUS.json").read_text())
    assert status["passed"] is False and status["fix"] is None


@pytest.mark.parametrize("fix,quant_runs", [("fused", True), ("fused_declared", False), (None, False)])
def test_machete_fix_runs_the_int4_check_only_when_the_shipped_fix_met_the_rule(runner, monkeypatch, capsys, fix,
                                                                                 quant_runs):
    ran = []
    monkeypatch.setattr(runner, "machete_diag", _remote({"message": "machete_diag finished", "fix": fix}))
    monkeypatch.setattr(runner, "quant", SimpleNamespace(remote=lambda *a: ran.append(a) or "quant finished"))
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    if quant_runs:
        runner.main("machete_fix")
        out = capsys.readouterr().out
        assert "starting the INT4 runs" in out and "quant finished" in out and "Done." in out
    else:
        with pytest.raises(SystemExit, match="INT4 runs weren't started"):
            runner.main("machete_fix")
        assert "Done." not in capsys.readouterr().out
    assert len(ran) == (1 if quant_runs else 0)
    assert (runner.LOCAL / "runs/gpu_r3/log.txt").exists()            # results downloaded either way


def test_a_failed_quant_after_the_fix_is_a_failure(runner, monkeypatch):
    monkeypatch.setattr(runner, "machete_diag", _remote({"message": "machete_diag finished", "fix": "fused"}))
    monkeypatch.setattr(runner, "quant", _remote(RuntimeError("quantized-model validation failed")))
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    with pytest.raises(SystemExit, match="quantized-model validation failed"):
        runner.main("machete_fix")


def _spawned(value):
    def get():
        if isinstance(value, Exception):
            raise value
        return value
    return SimpleNamespace(spawn=lambda *a, **k: SimpleNamespace(get=get))


def test_the_serve_phase_passes_only_when_all_three_server_runs_pass(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    for tag in ("serve032", "serve032_tp2", "serve031"):
        monkeypatch.setattr(runner, tag, _spawned(f"{tag}: exit code 0 in 12.0 minutes"))
    runner.main("serve")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "serve032_tp2", _spawned("serve032_tp2: exit code 1 in 12.0 minutes"))
    with pytest.raises(SystemExit, match="serve032_tp2: exit code 1"):
        runner.main("serve")
    monkeypatch.setattr(runner, "serve031", _spawned(RuntimeError("container lost")))
    with pytest.raises(SystemExit, match="serve031 did not complete"):
        runner.main("serve")


def test_the_l4_phase_needs_both_models_to_pass(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "quant_l4", _remote(
        "quant_l4: {'awq1.5b_marlin_l4': 0, 'gptq1.5b_marlin_l4': 0} in 20.0 minutes"))
    runner.main("quant_l4")
    monkeypatch.setattr(runner, "quant_l4", _remote(
        "quant_l4: {'awq1.5b_marlin_l4': 0, 'gptq1.5b_marlin_l4': 1} in 20.0 minutes"))
    with pytest.raises(SystemExit, match="gptq1.5b_marlin_l4': 1"):
        runner.main("quant_l4")


def test_the_serve_run_records_its_status_and_gpus(runner, monkeypatch, tmp_path):
    import torch
    seen = []
    monkeypatch.setattr(runner, "_tee", lambda cmd, log, **k: seen.append(cmd) or 0)
    monkeypatch.setattr(runner, "_locked", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA H100 80GB HBM3")
    msg = runner._serve("serve032_tp2", 2, False, {"sha256": "x"}, runner.RATE_TP2, "2 H100s")
    st = json.load(open(os.path.join(runner.RUNS, "serve032_tp2", "STATUS.json")))
    assert "exit code 0" in msg and st["passed"] and st["gpus"] == ["NVIDIA H100 80GB HBM3"] * 2
    assert seen[0][-2:] == ["--tp", "2"] and "vllm_serve_check.py" in seen[0][1]


def test_the_2_gpu_server_check_runs_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "serve032_tp2", _remote("serve032_tp2: exit code 0 in 9.0 minutes"))
    runner.main("serve_tp2")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "serve032_tp2", _remote("serve032_tp2: exit code 1 in 4.8 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("serve_tp2")


def test_the_readiness_and_attribution_check_runs_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "ready_attrib", _remote("ready_attrib: exit code 0 in 41.0 minutes"))
    runner.main("ready_attrib")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "ready_attrib", _remote("ready_attrib: exit code 1 in 39.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("ready_attrib")


def test_the_live_catch_runs_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "live_catch", _remote("live_catch: exit code 0 in 52.0 minutes"))
    runner.main("live_catch")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "live_catch", _remote("live_catch: exit code 1 in 50.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("live_catch")


def test_the_decisions_run_alone_and_fail_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "decisions", _remote("decisions: exit code 0 in 41.0 minutes"))
    runner.main("decisions")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "decisions", _remote("decisions: exit code 1 in 40.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("decisions")


def test_the_isolation_runs_both_versions_and_only_031_decides(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))

    class _Call:
        def __init__(self, m):
            self.m = m

        def get(self):
            return self.m

    class _Fn:
        def __init__(self, m):
            self.m = m

        def spawn(self, *a):
            return _Call(self.m)
    monkeypatch.setattr(runner, "isolation031", _Fn("isolation0310: exit code 0 in 18.0 minutes"))
    monkeypatch.setattr(runner, "isolation032", _Fn("isolation0320: exit code 1 in 17.0 minutes"))
    runner.main("isolation")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "isolation031", _Fn("isolation0310: exit code 1 in 18.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("isolation")


def test_the_sensitivity_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "sensitivity", _remote("sensitivity: exit code 0 in 22.0 minutes"))
    runner.main("sensitivity")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "sensitivity", _remote("sensitivity: exit code 1 in 22.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("sensitivity")


def test_the_fp8_residuals_run_alone(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "fp8_residuals", _remote("fp8_residuals: exit code 0 in 6.0 minutes"))
    runner.main("fp8_residuals")
    assert "Done." in capsys.readouterr().out


def test_the_norms_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "norms", _remote("norms: exit code 0 in 41.0 minutes"))
    runner.main("norms")
    out = capsys.readouterr().out
    assert "Done." in out and "RMSNorm" in out
    monkeypatch.setattr(runner, "norms", _remote("norms: exit code 1 in 41.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("norms")


def test_the_qualify_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "qualify", _remote("qualify: exit code 0 in 33.0 minutes"))
    runner.main("qualify")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "qualify", _remote("qualify: exit code 1 in 33.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("qualify")


def test_the_fp8kv_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "fp8kv", _remote("fp8kv: exit code 0 in 41.0 minutes"))
    runner.main("fp8kv")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "fp8kv", _remote("fp8kv: exit code 1 in 41.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("fp8kv")


def test_the_fp8_moe_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "fp8_moe", _remote("fp8_moe: exit code 0 in 71.0 minutes"))
    runner.main("fp8_moe")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "fp8_moe", _remote("fp8_moe: exit code 1 in 71.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("fp8_moe")


def test_the_moe_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "moe", _remote("moe: exit code 0 in 28.0 minutes"))
    runner.main("moe")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "moe", _remote("moe: exit code 1 in 28.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("moe")


def test_the_graphs_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "graphs", _remote("graphs: exit code 0 in 9.0 minutes"))
    runner.main("graphs")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "graphs", _remote("graphs: exit code 1 in 9.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("graphs")


def test_the_graphs_run_makes_a_missing_gpu_a_failure_and_records_its_status(runner, monkeypatch):
    seen = {}

    def tee(cmd, log_path, env=None, **_):
        seen["cmd"], seen["env"] = cmd, env
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("fixture\n")
        return 0

    monkeypatch.setattr(runner, "_tee", tee)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda i: "NVIDIA L4")))
    msg = runner.graphs()
    assert msg.startswith("graphs: exit code 0 ")
    assert seen["env"]["SDCGUARD_GPU_TESTS"] == "1" and "tests/test_gpu_graphs.py" in seen["cmd"]
    status = json.loads((Path(runner.RUNS) / "graphs" / "STATUS.json").read_text())
    assert status["passed"] is True and status["gpus"] == ["NVIDIA L4"]
