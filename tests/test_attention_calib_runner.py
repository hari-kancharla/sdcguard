"""Calibration must distinguish measured safety from missing measurements (CPU only)."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "attention_calib.py"
    spec = importlib.util.spec_from_file_location("_loaded_attention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good(cfg):
    return {"config": cfg, "backend": "cpu flash", "injected": {},
            "clean": {"checks": 1, "forward_checks": 1, "unjudged": 0, "alarms": 0,
                      "worst": 1.0, "skipped": {}}}


@pytest.mark.parametrize("invalid", [
    {"error": "out of memory"},
    {"checks": 0}, {"unjudged": 1}, {"skipped": {"unsupported backward": 1}},
    {"forward_checks": 0}, {"worst": float("nan")},
])
def test_attention_partial_or_unjudged_run_cannot_meet_criteria(monkeypatch, invalid):
    m = _module()
    cfgs = [{"name": name, "layers": 1, "steps": 1} for name in ("first", "second")]
    monkeypatch.setattr(m, "CONFIGS", {"cpu": cfgs})
    good, bad = _good(cfgs[0]), _good(cfgs[1])
    if "error" in invalid:
        bad = {"config": cfgs[1], "backend": "cpu flash", **invalid}
    else:
        bad["clean"].update(invalid)
    results = iter([good, bad])
    monkeypatch.setattr(m, "run", lambda *a, **kw: copy.deepcopy(next(results)))
    r = m.calibrate(torch.device("cpu"), say=lambda _: None)
    assert not r["criteria_met"], r
    assert r["incomplete"]
    assert len(r["runs"]) == 2
    assert "INCOMPLETE" in m.markdown(r)
    monkeypatch.setattr(m, "calibrate", lambda *a: r)
    assert m.main([]) == 2


def test_attention_all_complete_cases_still_meet_criteria(monkeypatch):
    m = _module()
    cfg = {"name": "complete", "layers": 1, "steps": 1}
    monkeypatch.setattr(m, "CONFIGS", {"cpu": [cfg]})
    monkeypatch.setattr(m, "run", lambda *a, **kw: _good(cfg))
    r = m.calibrate(torch.device("cpu"), say=lambda _: None)
    assert r["criteria_met"]


def test_attention_margin_uses_unrounded_measurement(monkeypatch):
    """A score above 4 must not pass because a display value was rounded to 4.000."""
    m = _module()
    cfg = dict(name="tiny", layers=1, width=16, heads=2, kv=2, seq=4,
               batch=1, vocab=8, steps=1, seed=1)
    monkeypatch.setattr(m, "CONFIGS", {"cpu": [cfg]})
    monkeypatch.setattr(m, "MULTS", ())
    monkeypatch.setattr(m, "POW2", ())
    original_guard = m.SDCGuard

    def guard(**kw):
        callback = kw["on_result"]

        def measured(detail, result):
            if detail.get("kind") == "attention_backward":
                result = copy.copy(result)
                result.max_z = 4.0004
            callback(detail, result)

        return original_guard(**{**kw, "on_result": measured})

    monkeypatch.setattr(m, "SDCGuard", guard)
    r = m.calibrate(torch.device("cpu"), say=lambda _: None)
    assert r["worst"] == 4.0004
    assert not r["criteria_met"]
