"""The gradients follow-up (experiments/recompute_grad_control.py): the comparison sees a one-bit difference, the
decision needs a deterministic control and a check that compared every block, and a check that changes gradients
comes out CHANGED, not UNCHANGED."""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "experiments"))

pytest.importorskip("transformers")
import recompute_grad_control as gc  # noqa: E402


def _run(grads, loss=1.0):
    return {"grads": grads, "loss": loss, "check_stats": None}


def test_compare_sees_one_bit():
    a = torch.tensor([1.0, 2.0, 4.0])
    b = a.clone()
    b[1] = torch.nextafter(b[1], torch.tensor(3.0))
    same = gc.compare(_run({"w": a}), _run({"w": a.clone()}))
    diff = gc.compare(_run({"w": a}), _run({"w": b}))
    assert same["bit_identical"] and same["parameters_that_differ"] == 0
    assert not diff["bit_identical"] and diff["parameters_that_differ"] == 1
    assert 0 < diff["max_relative_difference"] < 1e-6 and "w" in diff["worst_parameters"]


def _det(ab=True, ac=True, ad=True, compared=8, mismatches=0):
    return {"check_stats": {"recomputes_compared": compared, "mismatches": mismatches},
            "A vs B": {"bit_identical": ab}, "A vs C": {"bit_identical": ac}, "A vs D": {"bit_identical": ad}}


def test_decision_rules():
    assert gc.decide(_det(), 8) == "UNCHANGED"
    assert gc.decide(_det(ac=False), 8) == "CHANGED"
    assert gc.decide(_det(ad=False), 8) == "CHANGED"
    assert gc.decide(_det(ab=False), 8) == "INCONCLUSIVE"              # no deterministic control
    assert gc.decide(_det(ab=False, ac=False), 8) == "INCONCLUSIVE"
    assert gc.decide(_det(compared=0), 8) == "INCONCLUSIVE"            # the check compared nothing
    assert gc.decide(_det(compared=7), 8) == "INCONCLUSIVE"
    assert gc.decide(_det(mismatches=1), 8) == "INCONCLUSIVE"


def test_criteria_need_both_models_unchanged():
    def r(d1, d2):
        return {"settings": {"deterministic": {"models": {"gpt2": {"decision": d1}, "llama": {"decision": d2}}}}}
    key = "3 (follow-up) gradients unchanged by the check, against a deterministic control"
    assert gc.criteria(r("UNCHANGED", "UNCHANGED"))[key]
    assert not gc.criteria(r("UNCHANGED", "INCONCLUSIVE"))[key]
    assert not gc.criteria(r("CHANGED", "UNCHANGED"))[key]
    assert not gc.criteria({"settings": {"deterministic": {"error": "exit 1"}}})[key]
    assert not gc.criteria({"settings": {}})[key]


def test_a_check_that_changes_gradients_is_caught(monkeypatch):
    real = gc._grads

    def sabotaged(m, dev, sz, setting, check):
        out = real(m, dev, sz, setting, check)
        if check:                                   # the kind of change the follow-up exists to find
            name = next(iter(out["grads"]))
            g = out["grads"][name]
            g.view(-1)[0] = torch.nextafter(g.view(-1)[0], torch.tensor(float("inf")))
        return out
    monkeypatch.setattr(gc, "_grads", sabotaged)
    try:
        res = gc.one_setting("deterministic", torch.device("cpu"), dict(gc.rc_run.SIZES["cpu"]))
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    for kind in gc.MODELS:
        m = res["models"][kind]
        assert m["A vs B"]["bit_identical"] and not m["A vs C"]["bit_identical"]
        assert m["check_stats"]["recomputes_compared"] == gc.rc_run.SIZES["cpu"]["layers"]
        assert m["decision"] == "CHANGED"


def test_end_to_end_on_cpu_in_child_processes():
    r = gc.run_all(torch.device("cpu"), say=lambda s: None)
    assert set(r["settings"]) == {"default", "deterministic"}
    for s in r["settings"].values():
        assert "error" not in s, s
        for m in s["models"].values():
            assert m["A vs B"]["bit_identical"] and m["A vs C"]["bit_identical"] and m["A vs D"]["bit_identical"]
    assert r["criteria"]["decisions"] == {"gpt2": "UNCHANGED", "llama": "UNCHANGED"}
    assert "UNCHANGED" in gc.markdown(r)
