"""The vLLM plugin's norm checks with stand-ins for vLLM's RMSNorm classes and model runner (tests/_vllm_standins.py):
the patch at register(), registration right after load_model, checks that run and alert through the guard, norms the
check doesn't know listed, a reload taking new weights, SDCGUARD_NORMS=0, and the compile-cache key."""
import logging

import torch

from _vllm_standins import NormDecoder, install_fake_layernorm, install_fake_vllm, load, plugin_env


def _setup(monkeypatch, tmp_path, custom_op_enabled=True, **env):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    ln = install_fake_layernorm(monkeypatch.setitem, custom_op_enabled=custom_op_enabled)

    def load_model(self):
        self.model = load(self.factory())

    monkeypatch.setattr(GM, "__init__", lambda self, factory: setattr(self, "factory", factory), raising=False)
    monkeypatch.setattr(GM, "load_model", load_model, raising=False)
    vp = plugin_env(monkeypatch, tmp_path, **{"SDCGUARD_NORMS": "1", **env})
    return vp, UL, GM, ln


def _x():
    return (torch.randn(5, 256) * 3).to(torch.bfloat16)


def test_norms_are_registered_after_load_and_checked_every_call(monkeypatch, tmp_path, caplog):
    vp, UL, GM, ln = _setup(monkeypatch, tmp_path)
    vp.register()
    assert vp._STATE["norm_classes"] == ["RMSNorm", "GemmaRMSNorm"]
    runner = GM(lambda: NormDecoder(ln, UL))
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        runner.load_model()
    g = vp.guard()
    nc = g.norms
    assert nc is not None and nc.finalized and len(nc.layers) == 7          # 3 input, 3 post-attention, 1 final
    assert nc.unsupported == {"vision_norm": "LayerNorm: not an RMSNorm the check knows"}
    assert any("checking 7 of 8 norm layers" in r.getMessage() for r in caplog.records)
    for _ in range(3):
        runner.execute_model(_x())
    assert vp.poll(wait=True) == []
    cov = g.coverage_report()["norms"]
    assert cov["norms_checked"] == 7 and cov["never_checked"] == [] and cov["calls_checked"] == 21
    assert max(nc.worst_seen.values()) <= 4.05
    # a norm weight changed in memory: the alert names the layer, through the guard's log and attribution
    with torch.no_grad():
        runner.model.pnorm[1].weight.data.view(torch.int16)[40] ^= (1 << 13)
    runner.execute_model(_x())
    ev = vp.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["where"] == "pnorm.1" and ev[0].detail["col"] == 40
    assert ev[0].detail["recheck"]["verdict"] == "weights_changed"
    assert "attribution" in ev[0].detail and any("RMSNorm" in u for u in ev[0].detail["unknown_impact"])
    assert ev[0] in g.log.events


def test_a_weight_reload_makes_the_norms_take_the_new_weights(monkeypatch, tmp_path):
    vp, UL, GM, ln = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(lambda: NormDecoder(ln, UL))
    runner.load_model()
    with torch.no_grad():
        runner.model.norm.weight.data *= 1.5                                   # an intended update (an RL sync)
    load(runner.model)                                                         # vLLM's reload: linear layers again
    assert vp._STATE["norms_stale"] is True
    runner.execute_model(_x())                                                 # the step refreshes the norms first
    assert vp.poll(wait=True) == [] and vp._STATE["norms_stale"] is False


def test_a_norm_bound_before_the_patch_is_reported_as_never_checked(monkeypatch, tmp_path):
    """A layer whose forward vLLM bound to the unpatched function (built before register(), custom op disabled) is
    registered but never runs its check, and coverage says so instead of counting it as checked."""
    vp, UL, GM, ln = _setup(monkeypatch, tmp_path, custom_op_enabled=False)
    early = NormDecoder(ln, UL)
    vp.register()
    runner = GM(lambda: early)
    runner.load_model()
    runner.execute_model(_x())
    vp.poll(wait=True)
    cov = vp.guard().coverage_report()["norms"]
    assert cov["norms_checked"] == 0 and len(cov["never_checked"]) == 7


def test_norms_are_off_unless_sdcguard_norms_is_1(monkeypatch, tmp_path):
    vp, UL, GM, ln = _setup(monkeypatch, tmp_path, SDCGUARD_NORMS="0")
    vp.register()
    assert vp._STATE["norm_classes"] == [] and getattr(ln.RMSNorm, "_sdcguard_norm_orig", None) is None
    runner = GM(lambda: NormDecoder(ln, UL))
    runner.load_model()
    runner.execute_model(_x())
    assert vp.guard().norms is None


def test_the_compile_cache_key_follows_the_norm_settings(monkeypatch, tmp_path):
    vp = plugin_env(monkeypatch, tmp_path)
    default = vp._graph_key(1.0, 0, True)
    monkeypatch.setenv("SDCGUARD_NORMS", "1")
    on = vp._graph_key(1.0, 0, True)
    monkeypatch.setenv("SDCGUARD_NORM_Z", "8")
    z8 = vp._graph_key(1.0, 0, True)
    monkeypatch.setenv("SDCGUARD_NORMS", "0")
    off = vp._graph_key(1.0, 0, True)
    assert len({on, z8, off}) == 3 and off == default
    monkeypatch.setenv("SDCGUARD_NORMS", "1")
    monkeypatch.setenv("SDCGUARD_NORM_Z", "3")                                 # too low: 6 is used, so the key too
    assert vp._norm_line() == 6.0 and vp._graph_key(1.0, 0, True) == on


def test_a_failing_norm_check_turns_that_layer_off_without_breaking_serving(monkeypatch, tmp_path, caplog):
    vp, UL, GM, ln = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(lambda: NormDecoder(ln, UL))
    runner.load_model()
    bad = runner.model.inorm[2]
    bad._sdcguard_norm = ("not", "the", "check's", "arguments")             # its check raises
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        out = runner.execute_model(_x())
        out2 = runner.execute_model(_x())
    assert out.shape == out2.shape == (5, 256) and torch.isfinite(out2.float()).all()
    assert "_sdcguard_norm" not in vars(bad) and vp.guard().stats["norm_check_errors"] == 1
    assert sum("a norm check failed" in r.getMessage() for r in caplog.records) == 1
    assert vp.poll(wait=True) == []
