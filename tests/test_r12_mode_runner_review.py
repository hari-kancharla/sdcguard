"""Mode-comparison orchestration tests use fake calls and local processes only."""
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path



def _load(monkeypatch):
    class Image:
        def __getattr__(self, _):
            return lambda *a, **kw: self

    class App:
        def __init__(self, *a, **kw):
            pass

        def function(self, **kw):
            return lambda f: f

        def local_entrypoint(self):
            return lambda f: f

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(
        App=App, Image=types.SimpleNamespace(debian_slim=lambda **kw: Image())))
    path = Path(__file__).resolve().parents[1] / "scripts" / "modal_mode_compare.py"
    spec = importlib.util.spec_from_file_location("_r12_mode_runner", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_mode_runner_polls_all_calls_instead_of_abandoning_a_queued_gpu(monkeypatch, tmp_path):
    m = _load(monkeypatch)
    monkeypatch.setattr(m, "LOCAL", tmp_path)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    events = []

    class Call:
        def __init__(self, name, remaining):
            self.name, self.remaining = name, remaining

        def get(self, timeout):
            events.append((self.name, timeout))
            if self.remaining:
                self.remaining -= 1
                raise TimeoutError("still queued")
            return {"complete": True, "overhead_status": "complete", "markdown": "fixture"}

    for name, waits in (("h100", 3), ("a100", 0), ("l4", 0)):
        call = Call(name, waits)
        monkeypatch.setattr(m, name, types.SimpleNamespace(spawn=lambda c=call: c))
    m.main()
    assert events[:3] == [("h100", 0), ("a100", 0), ("l4", 0)]
    assert len(list((tmp_path / "results/mode_compare").glob("*.json"))) == 3
    assert all("error" not in json.loads(p.read_text()) for p in (tmp_path / "results/mode_compare").glob("*.json"))


def test_mode_worker_has_a_soft_deadline_and_returns_saved_partial_results(monkeypatch, tmp_path):
    m = _load(monkeypatch)
    monkeypatch.setattr(m, "R", str(tmp_path))
    calls, killed = [], []

    class Process:
        pid = 987654

        def __init__(self, cmd, **kw):
            dest = Path(cmd[cmd.index("--out") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "gpu.json").write_text(json.dumps({"matmuls": 7, "complete": True,
                                                       "overhead_status": "pending"}))
            (dest / "gpu.md").write_text("Completed sensitivity fixture")
            kw["stdout"].write("partial worker output\n")

        def wait(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", Process)
    import os
    monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a))
    r = m._work("fixture")
    assert calls and 0 < calls[0] <= m.LIMIT - 120
    assert killed
    assert r["matmuls"] == 7
    assert r["error"]
    assert "partial worker output" in r["output"]
    assert "Completed sensitivity fixture" in r["markdown"]


def test_mode_poll_transport_errors_respect_the_global_deadline(monkeypatch):
    m = _load(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(m.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt))

    class Broken:
        def get(self, timeout):
            raise ConnectionError("offline fixture")

    out = list(m._wait_all([("fixture", Broken())], cap=40, every=20))
    assert clock[0] == 40
    assert out[0][1]["error"]
