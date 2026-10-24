"""The vLLM plugin's MoE checks with stand-ins for vLLM's RoutedExperts and model runner (tests/_vllm_standins.py):
the patch at register() (SDCGUARD_MOE=1; off by default), registration right after load_model, layers taking turns
step by step, a wrong layer caught and named, a failing check turning its layer off without breaking serving."""
import logging

import torch

from _vllm_standins import MoEDecoder, install_fake_moe, install_fake_vllm, load, plugin_env


def _setup(monkeypatch, tmp_path, **env):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    moe = install_fake_moe(monkeypatch.setitem)

    def load_model(self):
        self.model = load(self.factory())

    monkeypatch.setattr(GM, "__init__", lambda self, factory: setattr(self, "factory", factory), raising=False)
    monkeypatch.setattr(GM, "load_model", load_model, raising=False)
    vp = plugin_env(monkeypatch, tmp_path, **{"SDCGUARD_MOE": "1", **env})
    return vp, UL, GM, moe


def _x():
    return (torch.randn(6, 128) * 2).to(torch.bfloat16)


def test_moe_layers_are_registered_after_load_and_take_turns(monkeypatch, tmp_path, caplog):
    vp, UL, GM, moe = _setup(monkeypatch, tmp_path)
    vp.register()
    assert vp._STATE["moe_classes"] == ["RoutedExperts"]
    runner = GM(lambda: MoEDecoder(moe, UL))
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        runner.load_model()
    g = vp.guard()
    assert g.moe is not None and len(g.moe.layers) == 3
    assert any("checking 3 of 3 MoE layers" in r.getMessage() for r in caplog.records)
    for _ in range(6):
        runner.execute_model(_x())
    assert vp.poll(wait=True) == []
    cov = g.coverage_report()["moe"]
    assert cov["layers_checked"] == 3 and cov["active_calls"] == 6
    # one layer starts giving another row's output: named when its turn comes
    runner.model.experts[2].fault = lambda out: torch.roll(out, 1, 0)
    ev = []
    for _ in range(3):
        runner.execute_model(_x())
        ev += vp.poll(wait=True)
    assert len(ev) == 1 and ev[0].detector == "serve_moe" and ev[0].detail["where"] == "experts.2"
    assert ev[0].detail["recheck"]["verdict"] == "output_wrong"
    assert any("MoE experts" in u for u in ev[0].detail["unknown_impact"])


def test_moe_checks_are_off_unless_sdcguard_moe_is_1(monkeypatch, tmp_path):
    vp, UL, GM, moe = _setup(monkeypatch, tmp_path, SDCGUARD_MOE="0")
    vp.register()
    assert vp._STATE["moe_classes"] == [] and getattr(moe.RoutedExperts, "_sdcguard_moe_orig", None) is None
    runner = GM(lambda: MoEDecoder(moe, UL))
    runner.load_model()
    runner.execute_model(_x())
    assert vp.guard().moe is None


def test_a_failing_moe_check_turns_that_layer_off_without_breaking_serving(monkeypatch, tmp_path, caplog):
    vp, UL, GM, moe = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(lambda: MoEDecoder(moe, UL))
    runner.load_model()
    bad = runner.model.experts[0]
    bad._sdcguard_moe = ("not", "its", "arguments")
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        out = runner.execute_model(_x())
        runner.execute_model(_x())
    assert out.shape == (6, 128) and "_sdcguard_moe" not in vars(bad)
    assert vp.guard().stats["moe_check_errors"] == 1
    assert sum("an MoE check failed" in r.getMessage() for r in caplog.records) == 1
