"""The paid runners' local entry points, with Modal faked: failures and incomplete runs exit with an error,
partial results and raw logs are kept, and costs are recorded. No Modal or GPU calls happen."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
needs_fused = pytest.mark.skipif(not (ROOT / "scripts" / "modal_fused.py").exists(), reason="next branch only")


class _Builder:
    def __getattr__(self, _):
        return lambda *a, **k: self


class _App:
    def __init__(self, *a, **k):
        pass

    def function(self, **_):
        return lambda f: f

    def local_entrypoint(self):
        return lambda f: f


def _load(monkeypatch, name, vol):
    fake = SimpleNamespace(Image=SimpleNamespace(debian_slim=lambda **_: _Builder(),
                                                 from_registry=lambda *a, **k: _Builder()),
                           App=_App, Volume=SimpleNamespace(from_name=lambda *a, **k: vol))
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location(f"entry_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Call:
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _ok(label):
    good = {"criteria": {"all met": True}, "markdown": f"## {label} ok\n"}
    return {"label": label, "fused": dict(good), "recompute": dict(good), "function_seconds": 600.0,
            "estimated_cost_usd": 0.5, "log": [f"{label} line 1", f"{label} line 2"]}


def _fused(monkeypatch, tmp_path, h100, l4, vol=None):
    mod = _load(monkeypatch, "modal_fused", vol or _Builder())
    monkeypatch.setattr(mod, "LOCAL", tmp_path)
    monkeypatch.setattr(mod, "_source", lambda: {"sha256": "f" * 64, "files": 1})
    monkeypatch.setattr(mod, "h100", SimpleNamespace(spawn=lambda run_id: _Call(h100)))
    monkeypatch.setattr(mod, "l4", SimpleNamespace(spawn=lambda run_id: _Call(l4)))
    return mod


@needs_fused
def test_fused_runner_passes_only_when_every_criterion_is_met(monkeypatch, tmp_path, capsys):
    mod = _fused(monkeypatch, tmp_path, _ok("h100"), _ok("l4"))
    mod.main()
    out = capsys.readouterr().out
    assert "Every criterion met on both GPUs. Done." in out
    logs = sorted((tmp_path / "results" / "fused").glob("*_log.txt"))
    assert len(logs) == 2 and "line 2" in logs[0].read_text()


@needs_fused
def test_fused_runner_fails_when_one_criterion_is_not_met(monkeypatch, tmp_path, capsys):
    bad = _ok("l4")
    bad["recompute"]["criteria"] = {"all met": False}
    mod = _fused(monkeypatch, tmp_path, _ok("h100"), bad)
    with pytest.raises(SystemExit):
        mod.main()
    assert "NOT ALL CRITERIA MET (l4 recompute)" in capsys.readouterr().out


@needs_fused
def test_fused_runner_keeps_partial_results_after_a_timeout(monkeypatch, tmp_path, capsys):
    saved = dict(_ok("l4"), function_seconds=None)

    class Vol:
        def read_file_into_fileobj(self, path, f):
            assert path.endswith("/l4.json")
            f.write(json.dumps(saved).encode())
    mod = _fused(monkeypatch, tmp_path, _ok("h100"), RuntimeError("FunctionTimeoutError"), vol=Vol())
    with pytest.raises(SystemExit):
        mod.main()
    l4 = json.loads(next((tmp_path / "results" / "fused").glob("l4_*.json")).read_text())
    assert l4["partial"] is True and "FunctionTimeoutError" in l4["error"] and l4["fused"]["criteria"]["all met"]
    assert "NOT ALL CRITERIA MET" in capsys.readouterr().out          # a partial run is never a pass


class _H2HVol:
    def __init__(self, files):
        self.files = files

    def listdir(self, path, recursive=False):
        import enum

        class FileEntryType(enum.IntEnum):          # what Modal 1.x returns (str() of it is just the number)
            FILE = 1
            DIRECTORY = 2
        dirs = sorted({p.rsplit("/", 1)[0] for p in self.files})
        return ([SimpleNamespace(path=d, type=FileEntryType.DIRECTORY) for d in dirs] +
                [SimpleNamespace(path=p, type=FileEntryType.FILE) for p in self.files])

    def read_file_into_fileobj(self, path, f):
        f.write(self.files[path])


def _h2h(monkeypatch, tmp_path, result, files=None):
    mod = _load(monkeypatch, "modal_h2h", _H2HVol(files or {}))
    monkeypatch.setattr(mod, "LOCAL", tmp_path)
    monkeypatch.setattr(mod, "_source", lambda: {"sha256": "a" * 64, "files": 1})
    spawn = SimpleNamespace(spawn=lambda run_id: _Call(dict(result, run_id=run_id)))
    monkeypatch.setattr(mod, "h200", spawn)
    monkeypatch.setattr(mod, "h200_preflight", spawn)
    monkeypatch.setattr(mod, "_wait", lambda call, **k: call.get())
    return mod


def _cells(skip=()):
    return [{"share": s, "tool": t, "flag": "passed", "cell": "NOT EXERCISED"} for s in (100, 40, 75, 14)
            for t in ("dcgm", "gpu_burn_tc", "sdcguard_check") if (s, t) not in skip]


def test_h2h_full_run_with_every_cell_measured_passes_and_downloads_logs(monkeypatch, tmp_path, capsys):
    files = {"h200_x/share_40/dcgm.stdout.txt": b"raw dcgm output\n"}
    mod = _h2h(monkeypatch, tmp_path, {"cells": _cells(), "markdown": "| table |", "function_seconds": 3600.0,
                                       "estimated_cost_usd": 4.92}, files)
    monkeypatch.setattr(mod.time, "strftime", lambda fmt, *a: "h200_x" if fmt.startswith("h200") else "t")
    mod.main("run")
    out = capsys.readouterr().out
    assert "All 12 planned cells measured. Done." in out and "about $4.92" in out
    assert (tmp_path / "results" / "h2h" / "h200_x_logs" / "share_40" / "dcgm.stdout.txt").read_bytes() == \
        b"raw dcgm output\n"


def test_h2h_full_run_with_a_missing_cell_is_incomplete(monkeypatch, tmp_path, capsys):
    cells = _cells(skip={(14, "dcgm")}) + [{"share": 14, "tool": "dcgm", "cell": "COULD NOT TEST", "why": "not run"}]
    mod = _h2h(monkeypatch, tmp_path, {"cells": cells, "markdown": ""})
    with pytest.raises(SystemExit):
        mod.main("run")
    assert "INCOMPLETE: 1 of 12" in capsys.readouterr().out


def test_h2h_preflight_no_go_exits_with_an_error(monkeypatch, tmp_path, capsys):
    mod = _h2h(monkeypatch, tmp_path, {"decisions": {"go: the full head-to-head can answer its question": False,
                                                     "mps_works (...)": True}})
    with pytest.raises(SystemExit):
        mod.main("preflight")
    out = capsys.readouterr().out
    assert "NO GO" in out and "[yes] mps_works" in out and "[NO] go:" in out
