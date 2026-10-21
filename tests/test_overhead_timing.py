"""The timing experiment releases its instrumentation after each mode, including a failed mode."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import weakref

import pytest
import torch


@pytest.fixture
def experiment(monkeypatch):
    root = Path(os.environ.get("SDCGUARD_REVIEW_REPO", Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location("loaded_overhead_experiment", root / "experiments/precise_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real = module.SDCGuard
    guards = []

    class ObservedGuard(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            guards.append(self)

    monkeypatch.setattr(module, "SDCGuard", ObservedGuard)
    return module, guards


CONFIG = dict(vocab=16, layers=1, width=8, heads=2, seq=4, batch=1)


def test_overhead_releases_each_guard_after_a_successful_measurement(experiment):
    module, guards = experiment
    result = module.overhead(torch.device("cpu"), CONFIG, steps=1)
    assert set(result["seconds_per_step"]) == {"unchecked", "default check, every step", "precise mode, every step"}
    assert len(guards) == 2
    assert all(not guard._protected for guard in guards)


def test_overhead_releases_its_guard_when_the_training_step_fails(experiment, monkeypatch):
    module, guards = experiment
    original = torch.optim.AdamW.step

    def fail_checked_step(optimizer, *args, **kwargs):
        if guards:
            raise RuntimeError("injected optimizer failure")
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", fail_checked_step)
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        module.overhead(torch.device("cpu"), CONFIG, steps=1)
    assert len(guards) == 1
    assert not guards[0]._protected


def test_previous_mode_model_is_released_before_allocating_the_next(experiment, monkeypatch):
    module, _ = experiment
    original = module.ScreenModel
    models = []

    def create(*args, **kwargs):
        assert all(ref() is None for ref in models), "The next mode must not allocate beside the previous model."
        model = original(*args, **kwargs)
        models.append(weakref.ref(model))
        return model

    monkeypatch.setattr(module, "ScreenModel", create)
    module.overhead(torch.device("cpu"), CONFIG, steps=1)
    assert len(models) == 3 and all(ref() is None for ref in models)
