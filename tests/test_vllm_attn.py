"""The vLLM plugin's attention check with stand-ins for vLLM's attention backends, layers and model runner
(tests/_vllm_standins.py): the patch at register() with SDCGUARD_ATTN=1, registration right after load_model, layers
the check doesn't take listed, one layer checked a step through the guard, a wrong attention kernel caught and named,
a wrong KV write caught, a failing check turning its layer off without breaking serving, and the summary line."""
import logging

import torch

from _vllm_standins import AttnDecoder, attn_batches, install_fake_attention, install_fake_vllm, plugin_env


def _setup(monkeypatch, tmp_path, **env):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    impls = install_fake_attention(monkeypatch.setitem)

    def load_model(self):
        self.model = self.factory()

    monkeypatch.setattr(GM, "__init__", lambda self, factory: setattr(self, "factory", factory), raising=False)
    monkeypatch.setattr(GM, "load_model", load_model, raising=False)
    monkeypatch.setattr(GM, "device", torch.device("cpu"), raising=False)
    vp = plugin_env(monkeypatch, tmp_path, **{"SDCGUARD_ATTN": "1", **env})
    return vp, GM, impls


def _model(impls, sinks_on=None):
    FA, TA = impls["FlashAttentionImpl"], impls["TritonAttentionImpl"]
    ims = [FA(4, 64, 0.125, 2), TA(4, 64, 0.125, 2), FA(4, 64, 0.125, 2, sinks=torch.zeros(4))]
    return lambda: AttnDecoder(ims)


def test_attention_layers_are_registered_after_load_and_checked_one_a_step(monkeypatch, tmp_path, caplog):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    assert vp._STATE["attn_classes"] == ["FlashAttentionImpl", "TritonAttentionImpl"]
    runner = GM(_model(impls))
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        runner.load_model()
    g = vp.guard()
    ac = g.attn
    assert ac is not None and ac.finalized and len(ac.layers) == 2
    assert ac.unsupported == {"model.layers.2.self_attn.attn": "attention sinks"}
    assert any("checking 2 of 3 attention layers" in r.getMessage() for r in caplog.records)
    batches = attn_batches(4, 2, 64, 3, steps=7)
    for b in batches:
        outs = runner.execute_model(b)
        assert all(torch.isfinite(o.float()).all() for o in outs)
    assert vp.poll(wait=True) == []
    cov = g.coverage_report()["attention"]
    assert cov["layers_checked"] == 2 and cov["calls_checked"] == 7 and cov["never_checked"] == []
    assert vp._STATE["steps"] == vp._STATE["token_steps"] == 7          # each step with tokens checks one layer
    assert ac.stats["samples"] == 7 * 4 and ac.stats["writes_checked"] == 7 * 4
    assert max(ac.worst_seen.values()) <= 1.0
    assert "attention" in vp._STATE["stack"]["checks"] if vp._STATE.get("stack") else True


def test_a_wrong_attention_kernel_is_caught_and_named(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    TA = impls["TritonAttentionImpl"]
    good = TA.forward

    def faulty(self, layer, query, key, value, kv_cache, md, output, *a, **k):     # one head's value 40 wrong
        out = good(self, layer, query, key, value, kv_cache, md, output, *a, **k)
        if md is not None:
            out[:, 2, 40] = out[:, 2, 40].float() * 2 + 1
        return out

    monkeypatch.setattr(TA, "forward", faulty)          # the backend as installed: register() wraps this one
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    for b in attn_batches(4, 2, 64, 3, steps=40):
        runner.execute_model(b)
    ev = [e for e in vp.poll(wait=True) if e.detector == "serve_attn"]
    assert ev and {e.detail["where"] for e in ev} == {"model.layers.1.self_attn.attn"}
    d = ev[0].detail
    assert d["head"] == 2 and d["dim"] == 40 and d["max_z"] > 4 and d["finding"] == "output_wrong_or_misread"
    assert d["captured"]["head"] == 2 and d["recheck"]["verdict"] == "not_rechecked"      # (the PyTorch check)
    assert any("sampled attention outputs" in u for u in d["unknown_impact"]) and "attribution" in d


def test_a_wrong_kv_write_is_caught(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    FA = impls["FlashAttentionImpl"]
    orig = FA.do_kv_cache_update

    def lossy(self, layer, key, value, kv_cache, slot_mapping):            # drops one dimension of every key
        k2 = key.clone()
        k2[:, :, 7] = 0
        orig(self, layer, k2, value, kv_cache, slot_mapping)

    monkeypatch.setattr(FA, "do_kv_cache_update", lossy)
    for b in attn_batches(4, 2, 64, 3, steps=4):
        runner.execute_model(b)
    ev = [e for e in vp.poll(wait=True) if e.detail.get("finding") == "kv_write_wrong"]
    assert len(ev) == 1 and ev[0].detail["where"] == "model.layers.0.self_attn.attn"
    lt = ev[0].detail["latest"]
    assert lt["content_wrong"] and not lt["slot_wrong"] and lt["dim"] == 7 and lt["part"] == "key"


def test_a_failing_attention_check_turns_that_layer_off_without_breaking_serving(monkeypatch, tmp_path, caplog):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    bad = runner.model.attn[0]
    bad._sdcguard_attn = ("not the check", 0)                       # its check raises
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        for b in attn_batches(4, 2, 64, 3, steps=3):
            outs = runner.execute_model(b)
    assert all(torch.isfinite(o.float()).all() for o in outs)
    assert "_sdcguard_attn" not in vars(bad) and vp.guard().stats["attn_check_errors"] == 1
    assert sum("an attention check failed" in r.getMessage() for r in caplog.records) == 1


def test_attention_is_off_unless_sdcguard_attn_is_1(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path, SDCGUARD_ATTN="0")
    vp.register()
    assert vp._STATE["attn_classes"] == [] and not hasattr(impls["FlashAttentionImpl"], "_sdcguard_attn_orig")
    runner = GM(_model(impls))
    runner.load_model()
    runner.execute_model(attn_batches(4, 2, 64, 3, steps=1)[0])
    assert vp.guard() is None or vp.guard().attn is None


def test_the_summary_line_and_the_line_setting(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path, SDCGUARD_ATTN_Z="1.2")
    assert vp._attn_line() == 4.0                                   # below 1.5: the default, with a warning
    monkeypatch.setenv("SDCGUARD_ATTN_Z", "6")
    assert vp._attn_line() == 6.0
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    r = {"checked": 0, "total": 0, "weight_share_checked": 0.0, "unchecked_by_method": {}, "quantized_or_moe": 0,
         "never_checked": []}
    msg = vp._summary(r)
    assert "attention checked by a sampled recompute on 2 layers (SDCGUARD_ATTN=1), not on 1" in msg


def test_a_dummy_batch_is_not_a_model_step(monkeypatch, tmp_path):
    """vLLM's V2 runner makes its dummy batches (memory profiling, compile warm-up, a data-parallel rank with nothing
    scheduled) through execute_model(dummy_run=True): not counted, logged or checked by attention (it stays off,
    as the last model step left it), while the next model steps are."""
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(GM, "execute_model", lambda self, x, dummy_run=False, **kw: self.model(x))
    monkeypatch.setattr(GM, "_dummy_run", lambda self, b: self.execute_model(b, dummy_run=True), raising=False)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    ac = vp.guard().attn
    batches = attn_batches(4, 2, 64, 3, steps=6)
    for b in batches[:3]:
        runner.execute_model(b)
    runner._dummy_run(batches[3])
    runner._dummy_run(batches[3])
    assert vp._STATE["steps"] == vp._STATE["token_steps"] == 3 and vp._STATE["dummy_runs"] == 2
    assert list(vp._STATE["dummy_after"]) == [3] and [e[0] for e in vp._STATE["step_log"]] == [1, 2, 3]
    for b in batches[3:]:
        runner.execute_model(b)
    assert vp.poll(wait=True) == []
    assert vp._STATE["steps"] == vp._STATE["token_steps"] == 6
    assert ac.stats["calls_checked"] == 6 and ac.stats["samples"] == 6 * 4
