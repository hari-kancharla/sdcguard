"""The attention calibration on a GPU tries every fused kernel PyTorch has (flash, memory-efficient, cuDNN). A kernel
the GPU doesn't offer is PyTorch's own answer ("No available kernel"), given before any check runs: recorded, not
counted as missing evidence. Any other error (a CUDA fault, out of memory) still makes the calibration incomplete."""
import copy
import importlib.util
from pathlib import Path

import torch


def _module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "attention_calib.py"
    spec = importlib.util.spec_from_file_location("_attention_calib_backends", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good(cfg, backend):
    return {"config": cfg, "backend": backend, "injected": {},
            "clean": {"checks": 2, "forward_checks": 2, "unjudged": 0, "alarms": 0, "worst": 1.5, "skipped": {}}}


def _calibrate(monkeypatch, failure):
    m = _module()
    cfg = {"name": "w", "layers": 1, "steps": 2}
    monkeypatch.setattr(m, "CONFIGS", {"cpu": [cfg]})
    monkeypatch.setattr(m, "backends", lambda dev: [("flash", "F"), ("efficient", "E"), ("cudnn", "C")])

    def run(dev, cfg, backend, low, say=print):
        if backend == "C":
            raise failure
        return copy.deepcopy(_good(cfg, {"F": "flash", "E": "efficient"}[backend]))
    monkeypatch.setattr(m, "run", run)
    return m, m.calibrate(torch.device("cpu"), say=lambda _: None)


def test_a_kernel_the_gpu_does_not_offer_is_recorded_not_counted_against_completeness(monkeypatch):
    m, r = _calibrate(monkeypatch, RuntimeError("No available kernel. Aborting execution."))
    assert r["criteria_met"] and not r["incomplete"]
    assert any(run.get("unavailable") for run in r["runs"])
    assert "not offered on this GPU" in m.markdown(r)


def test_any_other_kernel_error_still_makes_the_calibration_incomplete(monkeypatch):
    m, r = _calibrate(monkeypatch, RuntimeError("CUDA error: misaligned address"))
    assert not r["criteria_met"] and r["incomplete"]
    monkeypatch.setattr(m, "calibrate", lambda *a: r)
    assert m.main([]) == 2
