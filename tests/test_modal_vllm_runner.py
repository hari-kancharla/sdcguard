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

        def function(self, **kw):
            def keep(f):
                f._modal = kw                       # (what the decorator asked for: GPU, time limit)
                return f
            return keep

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


class _ModalError(Exception):
    pass


class ConnectionError(_ModalError):  # noqa: A001  (shaped like modal.exception.ConnectionError: not Python's own)
    pass


class FunctionTimeoutError(_ModalError):
    pass


@pytest.mark.parametrize("err", [ConnectionError("Deadline exceeded"), OSError(8, "nodename nor servname provided")])
def test_a_broken_connection_triggers_nothing_more(runner, monkeypatch, err):
    """When the call to Modal breaks off (this computer's network), the client calls no other function: a detached
    run keeps only the last function the client triggered alive, so a download now could stop the run."""
    calls = []
    monkeypatch.setattr(runner, "run1", _remote(err))
    monkeypatch.setattr(runner, "bundle", SimpleNamespace(remote=lambda *a, **k: calls.append(1) or _archive()))
    with pytest.raises(SystemExit) as exc:
        runner.main("run1")
    assert calls == [] and "connection to Modal broke off" in str(exc.value) and "--phase fetch" in str(exc.value)


def test_a_run_that_fails_on_modal_still_downloads(runner, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run1", _remote(FunctionTimeoutError("ran past its limit")))
    monkeypatch.setattr(runner, "bundle", SimpleNamespace(remote=lambda *a, **k: calls.append(1) or _archive()))
    with pytest.raises(SystemExit):
        runner.main("run1")
    assert calls == [1] and not runner._connection_broke(FunctionTimeoutError("x"))


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


def test_a_skipped_capture_test_fails_the_graphs_run(runner, monkeypatch):
    def tee(cmd, log_path, env=None, **_):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("SKIPPED [6] tests/test_gpu_graphs.py:24: needs an NVIDIA GPU\n"
                                  "SKIPPED [1] tests/test_serve.py:383: no graph-replay hook\n"
                                  "30 passed, 7 skipped\n")
        return 0

    monkeypatch.setattr(runner, "_tee", tee)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda i: "NVIDIA L4")))
    msg = runner.graphs()
    assert msg.startswith("graphs: FAILED") and "exit code 0 " not in msg
    status = json.loads((Path(runner.RUNS) / "graphs" / "STATUS.json").read_text())
    assert status["passed"] is False and len(status["skipped_required"]) == 1      # test_serve.py's skip is allowed
    assert status["skipped_other"] == ["SKIPPED [1] tests/test_serve.py:383: no graph-replay hook"]


def test_an_allowed_skip_passes_the_graphs_run_and_is_said(runner, monkeypatch):
    def tee(cmd, log_path, env=None, **_):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text("SKIPPED [1] tests/test_serve.py:383: no graph-replay hook\n63 passed, 1 skipped\n")
        return 0

    monkeypatch.setattr(runner, "_tee", tee)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 1, get_device_name=lambda i: "NVIDIA L4")))
    msg = runner.graphs()
    assert msg.startswith("graphs: exit code 0 ") and "skipped, not required: SKIPPED [1] tests/test_serve.py" in msg
    status = json.loads((Path(runner.RUNS) / "graphs" / "STATUS.json").read_text())
    assert status["passed"] is True and status["skipped_required"] == [] and len(status["skipped_other"]) == 1


def test_the_fp8mm_calibration_runs_on_both_gpus_and_fails_if_either_does(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))

    class Call:
        def __init__(self, msg):
            self.msg = msg

        def get(self):
            return self.msg

    ok = SimpleNamespace(spawn=lambda *a: Call("fp8mm_cal_h100: exit code 0 in 12.0 minutes"))
    ok_l4 = SimpleNamespace(spawn=lambda *a: Call("fp8mm_cal_l4: exit code 0 in 9.0 minutes"))
    monkeypatch.setattr(runner, "fp8mm_cal_h100", ok)
    monkeypatch.setattr(runner, "fp8mm_cal_l4", ok_l4)
    runner.main("fp8mm_cal")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "fp8mm_cal_l4", SimpleNamespace(spawn=lambda *a: Call("fp8mm_cal_l4: exit code 1 in 2.0 minutes")))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("fp8mm_cal")


def test_the_attention_check_runs_on_both_gpus_and_fails_if_either_does(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))

    class Call:
        def __init__(self, msg):
            self.msg = msg

        def get(self):
            return self.msg

    def fn(msg):
        return SimpleNamespace(spawn=lambda *a: Call(msg))

    monkeypatch.setattr(runner, "attn_h100", fn("attn_h100: exit code 0 in 60.0 minutes"))
    monkeypatch.setattr(runner, "attn_l4", fn("attn_l4: exit code 0 in 25.0 minutes"))
    runner.main("attn")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "attn_h100", fn("attn_h100: exit code 1 in 9.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("attn")
    assert "attn" in runner.ESTIMATES and "$" in runner.ESTIMATES["attn"]


def test_the_operator_screen_runs_on_both_gpus(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))

    class Call:
        def __init__(self, msg):
            self.msg = msg

        def get(self):
            return self.msg

    monkeypatch.setattr(runner, "screen_ops_h100", SimpleNamespace(spawn=lambda *a: Call("screen_ops_h100: exit code 0 "
                                                                                          "in 9.0 minutes")))
    monkeypatch.setattr(runner, "screen_ops_l4", SimpleNamespace(spawn=lambda *a: Call("screen_ops_l4: exit code 1 in "
                                                                                        "9.0 minutes")))
    with pytest.raises(SystemExit, match="screen_ops_l4: exit code 1"):
        runner.main("screen_ops")
    assert "screen_ops_h100: exit code 0" in capsys.readouterr().out


def test_a_rerun_keeps_the_earlier_runs_folder_and_log(runner, tmp_path):
    out = tmp_path / "remote-results" / "moe"
    out.mkdir(parents=True)
    (out / "SUMMARY.json").write_text("{}")
    (tmp_path / "remote-results" / "moe.txt").write_text("earlier log")
    runner._set_aside(str(out))
    kept = sorted(p.name for p in (tmp_path / "remote-results").iterdir())
    assert len(kept) == 2 and all(k.startswith("moe_before_") for k in kept)
    assert kept[0][:-4] == kept[1] or kept[1][:-4] == kept[0]                  # the same time on both
    assert (tmp_path / "remote-results" / [k for k in kept if k.endswith(".txt")][0]).read_text() == "earlier log"
    runner._set_aside(str(out))                                                # nothing there: no error


def test_the_weight_scan_run_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "weight_scan", _remote("weight_scan: exit code 0 in 28.0 minutes"))
    runner.main("weight_scan")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "weight_scan", _remote("weight_scan: exit code 1 in 28.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("weight_scan")


def test_the_moe_bench_runs_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "moe_bench", _remote("moe_bench: exit code 0 in 5.0 minutes"))
    runner.main("moe_bench")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "moe_bench", _remote("moe_bench: exit code 1 in 5.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("moe_bench")
    assert "$" in runner.ESTIMATES["moe_bench"] and "15-minute limit" in runner.ESTIMATES["moe_bench"]


def test_the_norm_diagnosis_runs_alone_and_fails_on_a_nonzero_exit(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    monkeypatch.setattr(runner, "norm_diag", _remote("norm_diag: exit code 0 in 30.0 minutes"))
    runner.main("norm_diag")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "norm_diag", _remote("norm_diag: exit code 1 in 30.0 minutes"))
    with pytest.raises(SystemExit, match="exit code 1"):
        runner.main("norm_diag")
    assert "60-minute limit" in runner.ESTIMATES["norm_diag"]


def test_the_spot_run_spawns_both_gpus_and_fails_if_either_does(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))

    class Spawned:
        def __init__(self, msg):
            self.msg = msg

        def get(self):
            return self.msg

    class F:
        def __init__(self, msg):
            self.msg = msg

        def spawn(self, *a):
            return Spawned(self.msg)
    monkeypatch.setattr(runner, "spot_h100", F("spot_h100: exit code 0 in 30.0 minutes"))
    monkeypatch.setattr(runner, "spot_l4", F("spot_l4: exit code 0 in 12.0 minutes"))
    runner.main("spot")
    assert "Done." in capsys.readouterr().out
    monkeypatch.setattr(runner, "spot_l4", F("spot_l4: exit code 1 in 12.0 minutes"))
    with pytest.raises(SystemExit, match="spot_l4: exit code 1"):
        runner.main("spot")


def test_run1_is_one_call_and_reports_every_part(runner, monkeypatch, capsys):
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    ok = {t: f"{t}: exit code 0 in 20.0 minutes" for t in runner.RUN1}
    monkeypatch.setattr(runner, "run1", _remote(ok))
    runner.main("run1")
    out = capsys.readouterr().out
    assert "Done." in out and all(m in out for m in ok.values())
    bad = dict(ok, norm_diag="norm_diag: exit code 1 in 30.0 minutes")
    bad.pop("spot_l4")
    monkeypatch.setattr(runner, "run1", _remote(bad))
    with pytest.raises(SystemExit) as e:
        runner.main("run1")
    assert "norm_diag: exit code 1" in str(e.value) and "spot_l4: no result" in str(e.value)
    assert set(runner.RUN1) == {"weight_scan", "moe_bench", "norm_diag", "moe", "spot_h100", "spot_l4",
                                "screen_ops_h100", "screen_ops_l4", "fp8_moe"}


def test_the_source_fingerprint_carries_the_criteria(runner, monkeypatch, tmp_path):
    """A result's STATUS.json records the code and the criteria (docs/METHOD.md) it ran under."""
    import hashlib
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "METHOD.md").write_text("criteria\n")
    monkeypatch.setattr(runner, "LOCAL", tmp_path)
    s = runner._source()
    assert s["files"] == 1 and s["criteria"] == {"docs/METHOD.md": hashlib.sha256(b"criteria\n").hexdigest()}
    (tmp_path / "docs" / "METHOD.md").write_text("criteria changed\n")
    assert runner._source()["criteria"] != s["criteria"] and runner._source()["sha256"] == s["sha256"]


def test_the_smoke_test_is_one_call_over_every_part(runner, monkeypatch, capsys):
    """Run 1's smoke test: one call starts every part (6 on H100s, 2 on L4s) from inside Modal; a part that fails
    or gives no result fails the phase; each part's limit fits inside its function's."""
    monkeypatch.setattr(runner, "bundle", _remote(_archive()))
    parts = (*runner.SMOKE_H100, *runner.SMOKE_L4)
    assert set(parts) == set(runner.SMOKE_LIMITS) and len(parts) == 8
    assert all(runner.SMOKE_LIMITS[p] <= 28 for p in runner.SMOKE_H100)          # (function: 30 minutes)
    assert all(runner.SMOKE_LIMITS[p] <= 15 for p in runner.SMOKE_L4)            # (function: 17 minutes)
    ok = {p: f"smoke_{p}: exit code 0 in 9.0 minutes" for p in parts}
    monkeypatch.setattr(runner, "smoke1", _remote(ok))
    runner.main("smoke1")
    assert "Done." in capsys.readouterr().out
    bad = dict(ok, moe="smoke_moe: exit code 1 in 12.0 minutes")
    bad.pop("screen_l4")
    monkeypatch.setattr(runner, "smoke1", _remote(bad))
    with pytest.raises(SystemExit) as e:
        runner.main("smoke1")
    assert "smoke_moe: exit code 1" in str(e.value) and "smoke_screen_l4: no result" in str(e.value)
    seen = {}

    class One:
        @staticmethod
        def remote(force, src, parts):
            seen["parts"], seen["force"] = parts, force
            return {"moe": "smoke_moe: exit code 0 in 9.0 minutes"}
    monkeypatch.setattr(runner, "smoke1", One)
    runner.main("smoke1", True, "moe")                    # a fixed part again, alone
    assert seen == {"parts": ["moe"], "force": True}
    with pytest.raises(SystemExit, match="--parts goes with --phase smoke1"):
        runner.main("run1", False, "moe")
    with pytest.raises(SystemExit, match="--parts goes with"):
        runner.main("smoke1", False, "nope")


def test_the_coordinators_outlast_their_parts_even_after_a_gpu_queue(runner):
    """run1 and smoke1 wait for their parts; if one ran out of time first, the client would finish, download and stop
    the app, stopping every part still running. Each coordinator's limit is its longest part's plus at least 2 hours
    of queueing."""
    fns = {"weight_scan": runner.weight_scan, "moe_bench": runner.moe_bench, "norm_diag": runner.norm_diag,
           "moe": runner.moe, "spot_h100": runner.spot_h100, "spot_l4": runner.spot_l4,
           "screen_ops_h100": runner.screen_ops_h100, "screen_ops_l4": runner.screen_ops_l4,
           "fp8_moe": runner.fp8_moe}
    assert set(fns) == set(runner.RUN1)
    longest = max(f._modal["timeout"] for f in fns.values())
    assert longest == 190 * 60 and runner.run1._modal["timeout"] >= longest + 2 * 3600
    sfn = {p: getattr(runner, runner.SMOKE_FN[p]) for p in (*runner.SMOKE_H100, *runner.SMOKE_L4)}
    smoke = max(f._modal["timeout"] for f in sfn.values())
    assert runner.smoke1._modal["timeout"] >= smoke + 2 * 3600
    for p, f in sfn.items():          # each part's function: just over the part's own limit, on the right GPU
        assert runner.SMOKE_LIMITS[p] * 60 <= f._modal["timeout"] - 120 <= runner.SMOKE_LIMITS[p] * 60 + 180, p
        assert (f._modal["gpu"] == "L4") == (p in runner.SMOKE_L4), p
    assert runner.run1._modal["timeout"] <= 24 * 3600 and "gpu" not in runner.run1._modal    # (CPU only)


def test_both_steps_at_every_functions_limit_stay_under_the_40_dollar_budget(runner):
    """The worst case the brief states: the smoke test and Run 1 with every function, coordinators included, run to
    its own time limit at its requested resources (Modal's other charges aside)."""
    rate = lambda f: runner.RATE_L4 if f._modal.get("gpu") == "L4" else (                    # noqa: E731
        runner.RATE_PER_HOUR if f._modal.get("gpu") else 0.0472 + 2 * 0.008)
    cost = lambda f: f._modal["timeout"] / 3600 * rate(f)                                    # noqa: E731
    run1 = [runner.weight_scan, runner.moe_bench, runner.norm_diag, runner.moe, runner.spot_h100, runner.spot_l4,
            runner.screen_ops_h100, runner.screen_ops_l4, runner.fp8_moe, runner.run1]
    smoke = [getattr(runner, runner.SMOKE_FN[p]) for p in (*runner.SMOKE_H100, *runner.SMOKE_L4)] + [runner.smoke1]
    r1, sm = sum(map(cost, run1)), sum(map(cost, smoke))
    assert 25 < r1 < 27.5 and 11 < sm < 12.5 and r1 + sm < 40, (r1, sm)


def test_a_preempted_call_restarted_by_modal_gets_through_its_own_lock(runner, monkeypatch, tmp_path):
    """Modal restarts a preempted GPU call on the same input. The restart met the lock its first attempt wrote and
    returned NOT STARTED: the partial run was paid for and lost. The lock now keeps the call's id."""
    spec = importlib.util.spec_from_file_location("modal_vllm_lock", Path(__file__).resolve().parents[1] /
                                                  "scripts/modal_vllm.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(m, "V", str(tmp_path))
    monkeypatch.setattr(m, "vol", SimpleNamespace(commit=lambda: None, reload=lambda: None))
    ids = iter(["call-1", "call-1", "call-2"])
    monkeypatch.setattr(m, "_call_id", lambda: next(ids))
    assert m._locked("issue59686", False) is None                        # first attempt takes the lock
    assert m._locked("issue59686", False) is None                        # Modal's restart of the same call
    assert m._lock_info("issue59686")["restarted"] == 1
    assert m._locked("issue59686", False).startswith("NOT STARTED")      # a second launch is still refused


def _lifetime(runner, monkeypatch, tmp_path, status=None, cand=True, patch=b"the reviewed fix patch",
              expected=None, code=0):
    import hashlib
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(runner, "vol", SimpleNamespace(commit=lambda: None, reload=lambda: None))
    monkeypatch.setattr(runner, "R", str(tmp_path / "container-root"))       # no container copy here
    exp = tmp_path / "experiments"
    exp.mkdir(exist_ok=True)
    (exp / "pr62065_lease_fix.patch").write_bytes(patch)
    monkeypatch.setattr(runner, "_PATCH_SHA256", expected or hashlib.sha256(b"the reviewed fix patch").hexdigest())
    src = Path(runner.RUNS) / "lifetime_pd"
    src.mkdir(parents=True, exist_ok=True)
    if cand:
        (src / "candidate.json").write_text('{"spec": {"point": "read_submit"}}')
    if status is not None:
        (src / "STATUS.json").write_text(json.dumps(status))
    seen = []
    monkeypatch.setattr(runner, "_tee", lambda cmd, log, **k: seen.append(cmd) or code)
    return seen


HARM = {"exit_code": 1, "verdict": "harm_reproduced", "source": {"sha256": "s1"}}


def test_the_fixed_lifetime_run_needs_a_saved_schedule(runner, monkeypatch, tmp_path):
    seen = _lifetime(runner, monkeypatch, tmp_path, status=HARM, cand=False)
    assert "NOT STARTED" in runner._lifetime_pd("lifetime_pd_fixed", "fixed", False, {}, candidate_from="lifetime_pd")
    assert seen == []


@pytest.mark.parametrize("status", [None, {"exit_code": 5, "verdict": "harm_found_in_discovery_not_reproduced_in_"
                                                                     "replays"},
                                    {"exit_code": 3, "verdict": "control_broke"}])
def test_the_fixed_lifetime_run_needs_confirmed_harm_on_the_stock_build(runner, monkeypatch, tmp_path, status):
    seen = _lifetime(runner, monkeypatch, tmp_path, status=status)
    msg = runner._lifetime_pd("lifetime_pd_fixed", "fixed", False, {}, candidate_from="lifetime_pd")
    assert "NOT STARTED" in msg and "did not confirm harm" in msg and seen == []


def test_the_fixed_lifetime_run_refuses_an_unreviewed_patch(runner, monkeypatch, tmp_path):
    seen = _lifetime(runner, monkeypatch, tmp_path, status=HARM, patch=b"something else")
    msg = runner._lifetime_pd("lifetime_pd_fixed", "fixed", False, {}, candidate_from="lifetime_pd")
    assert "NOT STARTED" in msg and "does not match" in msg and seen == []


def test_the_fixed_lifetime_run_records_what_it_replayed(runner, monkeypatch, tmp_path):
    import hashlib
    seen = _lifetime(runner, monkeypatch, tmp_path, status=HARM)
    msg = runner._lifetime_pd("lifetime_pd_fixed", "fixed", False, {"sha256": "s2"}, candidate_from="lifetime_pd")
    assert "exit code 0" in msg and "--candidate" in seen[0]
    st = json.load(open(os.path.join(runner.RUNS, "lifetime_pd_fixed", "STATUS.json")))
    src = st["replayed_candidate_from"]
    assert src["verdict"] == "harm_reproduced" and src["source"] == {"sha256": "s1"}
    assert src["candidate_sha256"] == hashlib.sha256(b'{"spec": {"point": "read_submit"}}').hexdigest()
    assert st["provenance"]["patch_matches_expected"] and st["verdict"] == "no_result_file"


def test_a_timed_out_lifetime_run_says_so(runner, monkeypatch, tmp_path):
    _lifetime(runner, monkeypatch, tmp_path, code=124)
    runner._lifetime_pd("lifetime_pd", "unpatched", False, {})
    st = json.load(open(os.path.join(runner.RUNS, "lifetime_pd", "STATUS.json")))
    assert st["timed_out"] and st["verdict"] == "harness_timed_out" and st["harness_failed"]
