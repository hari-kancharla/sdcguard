"""The serving release gate measures each layer after the observed warm-up boundary."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from sdcguard.serve import ServeGuard


@pytest.fixture
def experiment():
    root = Path(os.environ.get("SDCGUARD_REVIEW_REPO", Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location("loaded_serving_experiment", root / "experiments/vllm_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result():
    startup_counts = {n: {"checks": 3, "unjudged": 0} for n in ("a", "b")}
    clean_counts = {n: {"checks": 5, "unjudged": 0} for n in ("a", "b")}
    return {"startup": [{"layers": 2, "layer_counts": startup_counts, "checks_executed": 6,
                         "unjudged_checksums": 0}],
            "clean": [{"layers": 2, "layer_counts": clean_counts, "checks_executed": 10,
                       "layers_with_executed_checks": 2, "layers_checked_since_last_read": 2,
                       "unjudged_checksums": 0, "alerts": [], "changed_weights": []}],
            "flipped": ["a"],
            "after_flip": [{"alerts": [{"where": "a", "captured": {"row": 0, "col_block": 0, "with_bias": False,
                                                                   "z": 99.0},
                                        "recheck": {"verdict": "weights_changed"}}],
                            "changed_weights": ["a"]}]}


@pytest.mark.parametrize("change", ["no capture", "another verdict"])
def test_vllm_gate_needs_the_flipped_layers_row_kept_and_rechecked(experiment, change):
    """Original-event capture inside vLLM's compiled graphs (added October 21, 2026): the alert that names the
    corrupted layer must keep its row, and the recheck must find the weights changed since load."""
    result = _result()
    alert = result["after_flip"][0]["alerts"][0]
    if change == "no capture":
        alert["captured"] = None
    else:
        alert["recheck"] = {"verdict": "output_wrong"}
    errors = experiment._guarded_failures(result)
    assert errors and any("kept no row" in e for e in errors), errors


@pytest.mark.parametrize("stopped", [("a",), ("a", "b")])
def test_vllm_gate_rejects_layers_only_checked_during_startup(experiment, stopped):
    result = _result()
    for name in stopped:
        result["clean"][0]["layer_counts"][name]["checks"] = 3
    result["clean"][0]["checks_executed"] = sum(v["checks"] for v in result["clean"][0]["layer_counts"].values())
    errors = experiment._guarded_failures(result)
    assert errors and any("after warm-up" in error for error in errors), errors


@pytest.mark.parametrize("missing", ["startup", "layer_counts"])
def test_vllm_gate_requires_the_observed_startup_boundary(experiment, missing):
    result = _result()
    if missing == "startup":
        result.pop("startup")
    else:
        result["clean"][0].pop("layer_counts")
    assert experiment._guarded_failures(result)


def test_vllm_gate_accepts_executed_checks_with_exactly_zero_residual(experiment):
    result = _result()
    result["clean"][0]["layers_checked_since_last_read"] = 0
    assert experiment._guarded_failures(result) == []


def test_vllm_gate_preserves_clean_and_startup_unjudged_controls(experiment):
    result = _result()
    for phase in ("startup", "clean"):
        result[phase][0]["unjudged_checksums"] = 2
        result[phase][0]["layer_counts"]["a"]["unjudged"] = 2
    assert experiment._guarded_failures(result) == []
    result["clean"][0]["unjudged_checksums"] = 3
    result["clean"][0]["layer_counts"]["a"]["unjudged"] = 3
    assert experiment._guarded_failures(result)


def test_vllm_worker_report_contains_observed_per_layer_counts(experiment, monkeypatch):
    from sdcguard import vllm_plugin
    layer = torch.nn.Linear(4, 8, bias=False).bfloat16()
    guard = ServeGuard().attach(torch.nn.ModuleDict({"layer": layer}))
    monkeypatch.setitem(vllm_plugin._STATE, "guard", guard)
    try:
        with torch.no_grad():
            layer(torch.ones(2, 4, dtype=torch.bfloat16))
        report = experiment._guard_report(None)
        assert report.get("layer_counts") == {"layer": {"checks": 1, "unjudged": 0}}
    finally:
        guard.detach()


@pytest.mark.parametrize("captured", [{"row": 0}, {"row": -1, "col_block": 0, "with_bias": False, "z": 99.0},
                                      {"row": 0, "col_block": "7", "with_bias": False, "z": 99.0},
                                      {"row": 0, "col_block": 0, "with_bias": False, "z": None},
                                      {"row": 0, "col_block": 0, "with_bias": False, "z": float("nan")}])
def test_vllm_gate_needs_a_kept_row_with_a_place_and_a_score(experiment, captured):
    """Any non-empty capture record passed the gate (outside review, October 21, 2026): a row with no block, a
    negative row, a block that isn't a number, or no score isn't a row kept."""
    result = _result()
    result["after_flip"][0]["alerts"][0]["captured"] = captured
    assert any("kept no row" in e for e in experiment._guarded_failures(result)), captured


def test_vllm_gate_takes_the_row_a_flipped_exponent_bit_gives():
    """A flipped high exponent bit makes outputs the inputs can't produce: the kept row's score is infinite."""
    import json
    spec = importlib.util.spec_from_file_location("vllm_gate_inf", Path(__file__).resolve().parents[1]
                                                  / "experiments/vllm_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = json.loads(json.dumps(_result()).replace("99.0", "Infinity"))       # as the report file holds it
    assert result["after_flip"][0]["alerts"][0]["captured"]["z"] == float("inf")
    assert module._guarded_failures(result) == []

