"""Release orchestration must preserve failures. These tests never contact Modal or rent a GPU."""
import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load(relative, monkeypatch):
    class Image:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: self

    class App:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    modal = types.SimpleNamespace(
        Image=types.SimpleNamespace(debian_slim=lambda **kwargs: Image()),
        App=App,
        Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: Image()),
    )
    monkeypatch.setitem(sys.modules, "modal", modal)
    spec = importlib.util.spec_from_file_location("_release_test_" + Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(result):
    def get(**kwargs):
        if isinstance(result, Exception):
            raise result
        return copy.deepcopy(result)
    return types.SimpleNamespace(spawn=lambda: types.SimpleNamespace(get=get), remote=get)


@pytest.mark.parametrize("bad", [
    {"as_shipped": {"criteria_met": False}},
    {"error": "CUDA execution failed"},
    RuntimeError("remote call failed"),
])
def test_precise_runner_exits_unsuccessfully_and_preserves_results(tmp_path, monkeypatch, bad):
    module = _load("scripts/modal_precise_validate.py", monkeypatch)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    good = {"as_shipped": {"criteria_met": True}, "markdown": "fixture results"}
    for name in ("h100", "b200", "a100", "l4", "t4"):
        monkeypatch.setattr(module, name, _job(bad if name == "h100" else good))
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code != 0
    dest = tmp_path / "results" / "precise_validate"
    assert len(list(dest.glob("*.json"))) == 5
    assert "Met on 4 of 5" in (dest / "SUMMARY.md").read_text()


def test_precise_runner_accepts_complete_success(tmp_path, monkeypatch):
    module = _load("scripts/modal_precise_validate.py", monkeypatch)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    for name in ("h100", "b200", "a100", "l4", "t4"):
        monkeypatch.setattr(module, name, _job({"as_shipped": {"criteria_met": True}, "markdown": "fixture"}))
    module.main()
    assert "Met on 5 of 5" in (tmp_path / "results/precise_validate/SUMMARY.md").read_text()


@pytest.mark.parametrize("bad", [
    {"exit": 2, "cards": {}},
    {"exit": 0, "cards": {}},
    {"exit": 0, "cards": {"gpu": {"result": "FAIL", "gpu": {"name": "fixture"}}}},
    RuntimeError("worker did not complete"),
])
def test_sweep_failure_is_not_a_successful_command(tmp_path, monkeypatch, bad):
    module = _load("scripts/modal_check.py", monkeypatch)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    monkeypatch.setattr(module, "lock", types.SimpleNamespace(remote=lambda *args: ""))
    good = {"label": "fixture", "minutes": 0, "exit": 0, "output": "", "errors": "",
            "cards": {"gpu": {"result": "PASS", "gpu": {"name": "fixture"}}}}
    failed = bad if isinstance(bad, Exception) else {**good, **bad}
    monkeypatch.setattr(module, "SWEEP", [("first", _job(failed)), ("second", _job(good))])
    downloaded = []
    monkeypatch.setattr(module, "_download", lambda: downloaded.append(True))
    with pytest.raises(SystemExit) as exc:
        module.main(phase="sweep")
    assert exc.value.code != 0
    assert downloaded == [True], "Preserve diagnostics even when a worker failed."


def test_sweep_reports_board_build_failure(monkeypatch):
    module = _load("scripts/modal_check.py", monkeypatch)
    monkeypatch.setattr(module, "lock", types.SimpleNamespace(remote=lambda *args: ""))
    good = {"label": "fixture", "minutes": 0, "exit": 0, "output": "", "errors": "",
            "cards": {"gpu": {"result": "PASS", "gpu": {"name": "fixture"}}}}
    monkeypatch.setattr(module, "SWEEP", [("first", _job(good))])
    def fail_download():
        raise RuntimeError("board build failed")
    monkeypatch.setattr(module, "_download", fail_download)
    with pytest.raises(SystemExit, match="download or board"):
        module.main(phase="sweep")


@pytest.mark.parametrize("invalid", [
    {"unjudged": 1}, {"checks": 0}, {"worst": -1.0}, {"worst": float("nan")},
])
def test_held_out_validation_rejects_missing_or_unjudged_measurements(monkeypatch, invalid):
    module = _load("experiments/precise_models.py", monkeypatch)
    good = {"checks": 10, "alarms": 0, "unjudged": 0, "worst": 1.0}
    measured = iter([good, {**good, **invalid}])
    monkeypatch.setattr(module, "VALIDATE", {"cpu": [{"name": "clean", "steps": 1},
                                                    {"name": "invalid", "steps": 1}]})
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: {
        "models": {"current_as_shipped": next(measured)}, "expected_checks": 10,
        "guard_stats": {"checked": 10, "checked_fp32": 10, "unjudged": 0,
                        "skipped_oom": 0, "skipped_dtype": 0, "alerts": 0}})
    monkeypatch.setattr(module, "overhead", lambda *args: {"fixture": True})
    result = module.validate(torch.device("cpu"), say=lambda _: None)
    assert result["as_shipped"]["criteria_met"] is False


@pytest.mark.parametrize("actual", [
    {"checked": 0, "checked_fp32": 0, "skipped_oom": 10},
    {"checked": 9, "checked_fp32": 9}, {"unjudged": 1}, {"alerts": 1},
])
def test_held_out_validation_requires_actual_guard_execution(monkeypatch, actual):
    module = _load("experiments/precise_models.py", monkeypatch)
    stats = {"checked": 10, "checked_fp32": 10, "unjudged": 0, "skipped_oom": 0,
             "skipped_dtype": 0, "alerts": 0, **actual}
    monkeypatch.setattr(module, "VALIDATE", {"cpu": [{"name": "fixture", "steps": 1}]})
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: {
        "models": {"current_as_shipped": {"checks": 10, "alarms": 0, "unjudged": 0, "worst": 1.0}},
        "guard_stats": stats, "expected_checks": 10})
    monkeypatch.setattr(module, "overhead", lambda *args: {})
    assert module.validate(torch.device("cpu"), say=lambda _: None)["as_shipped"]["criteria_met"] is False


def test_tiny_held_out_workload_validates_real_production_checks(monkeypatch):
    module = _load("experiments/precise_models.py", monkeypatch)
    cfg = dict(name="tiny real execution", vocab=256, layers=1, width=128, heads=4,
               seq=8, batch=2, seed=7, lr=1e-3, steps=1)
    monkeypatch.setattr(module, "VALIDATE", {"cpu": [cfg]})
    monkeypatch.setattr(module, "overhead", lambda *args: {})
    result = module.validate(torch.device("cpu"), say=lambda _: None)
    assert result["as_shipped"]["criteria_met"] is True
    assert result["as_shipped"]["checks"] == 15
    assert result["workloads"][0]["guard_stats"]["checked_fp32"] == 15


class _Pending:
    """A spawned call that isn't finished for `polls` polls, then returns `result` (or raises it)."""

    def __init__(self, result, polls=0, flaky=0):
        self.result, self.polls, self.flaky = result, polls, flaky

    def get(self, timeout=None):
        if self.flaky:
            self.flaky -= 1
            raise ConnectionError("connection dropped")
        if self.polls:
            self.polls -= 1
            raise TimeoutError()
        if isinstance(self.result, Exception):
            raise self.result
        return copy.deepcopy(self.result)


def test_the_tests_phase_waits_for_every_gpu_and_reports_them_as_they_finish(tmp_path, monkeypatch):
    module = _load("scripts/modal_check.py", monkeypatch)
    monkeypatch.setattr(module, "LOCAL", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    ok = {"text": "fixture", "pytest": 0, "fast_compare": 0, "serve_compare": 0, "wheel": 0, "check": 0,
          "attention_calibration": 0}

    class FunctionTimeoutError(Exception):
        pass
    spawns = {"tests": _Pending(ok, polls=50), "tests_h100": _Pending(ok, polls=5, flaky=2),
              "tests_b200": _Pending(FunctionTimeoutError("hit its time limit"), polls=3)}
    for name, call in spawns.items():
        monkeypatch.setattr(module, name, types.SimpleNamespace(spawn=lambda c=call: c))
    with pytest.raises(SystemExit) as exc:
        module.main(phase="tests")
    assert exc.value.code == 1
    saved = sorted(p.name.split("_")[0] for p in (tmp_path / "results" / "gpu_tests").glob("*.txt"))
    assert saved == ["b200", "h100", "l4"]            # all three reported, the slow L4 included
    b200 = next((tmp_path / "results" / "gpu_tests").glob("b200_*.txt")).read_text()
    assert "FunctionTimeoutError" in b200 and "modal volume get" in b200


def _wait_module(monkeypatch):
    module = _load("scripts/modal_check.py", monkeypatch)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    return module


def test_wait_all_gives_up_only_after_the_cap(monkeypatch):
    module = _wait_module(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])

    class Never:
        def get(self, timeout=None):
            clock[0] += 600
            raise TimeoutError()
    out = list(module._wait_all([("h100", Never())], cap=3600, say=lambda *a, **k: None))
    assert len(out) == 1 and out[0][0] == "h100" and out[0][1]["run"] == 2
    assert "still not finished after 1.0 hours" in out[0][1]["text"]
