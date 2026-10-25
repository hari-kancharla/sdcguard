"""The vLLM plugin's attention check with stand-ins for vLLM's attention backends, layers and model runner
(tests/_vllm_standins.py): the patch at register() (on unless SDCGUARD_ATTN=0), registration after load_model, layers
the check doesn't take listed, one layer checked a step through the guard, a wrong attention kernel caught and named,
a wrong KV write caught, a failing check turning its layer off without breaking serving, and the summary line, the
stack and alerts saying where attention is checked (inside CUDA graphs only with its Triton kernels)."""
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
    lt, w = ev[0].detail["latest"], ev[0].detail["a_wrong_write"]
    assert set(lt) == {"step", "row"} and w["content_wrong"] and not w["slot_wrong"] and w["dim"] == 7 and \
        w["part"] == "key"


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


def test_sdcguard_attn_0_turns_attention_off(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path, SDCGUARD_ATTN="0")
    vp.register()
    assert vp._STATE["attn_classes"] == [] and not hasattr(impls["FlashAttentionImpl"], "_sdcguard_attn_orig")
    runner = GM(_model(impls))
    runner.load_model()
    runner.execute_model(attn_batches(4, 2, 64, 3, steps=1)[0])
    assert vp.guard() is None or vp.guard().attn is None


def test_attention_is_checked_when_sdcguard_attn_is_unset(monkeypatch, tmp_path, caplog):
    """On by default (its GPU run met its criteria, docs/METHOD.md): unset, the layers are registered and checked one
    a step; SDCGUARD_ATTN=false or off turns it off too, and an unknown value keeps it on with a warning."""
    import os
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.delenv("SDCGUARD_ATTN")
    assert "SDCGUARD_ATTN" not in os.environ
    vp.register()
    assert vp._STATE["attn_classes"] == ["FlashAttentionImpl", "TritonAttentionImpl"]
    runner = GM(_model(impls))
    runner.load_model()
    for b in attn_batches(4, 2, 64, 3, steps=4):
        runner.execute_model(b)
    assert vp.poll(wait=True) == [] and vp.guard().attn.stats["calls_checked"] == 4
    assert "attention=outside_graphs" in vp._STATE["stack"]["checks"]         # on the CPU: no Triton kernels
    for v, on in (("false", False), ("Off", False), ("no", False), ("1", True), ("yes", True)):
        monkeypatch.setenv("SDCGUARD_ATTN", v)
        assert vp._attn_on() is on, v
    monkeypatch.setenv("SDCGUARD_ATTN", "maybe")
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        assert vp._attn_on() is True
    assert any("isn't 0 or 1" in r.getMessage() for r in caplog.records)


def test_a_server_whose_attention_isnt_checkable_doesnt_say_it_is(monkeypatch, tmp_path):
    """Backends patched but no layer registered (here: microbatching): the stack and the alerts say attention isn't
    checked."""
    import types
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(GM, "parallel_config", types.SimpleNamespace(ubatch_size=2), raising=False)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    assert vp._STATE["attn_classes"] and vp._STATE["attn"]["checked"] == 0
    assert "attention=off" in vp._STATE["stack"]["checks"]
    ev = types.SimpleNamespace(detector="serve", detail={})
    vp._attribute(ev)
    assert not any("sampled attention outputs" in u for u in ev.detail.get("unknown_impact", []))
    r = {"checked": 0, "total": 0, "weight_share_checked": 0.0, "unchecked_by_method": {}, "quantized_or_moe": 0,
         "never_checked": []}
    assert "attention not checked on any of its 3 layers" in vp._summary(r)


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
    assert "attention checked by a sampled recompute on 2 layers outside CUDA graphs only (its Triton kernels aren't " \
           "in use: cpu bfloat16: the Triton kernels aren't available here) (SDCGUARD_ATTN=0 turns it off), not on 1" \
           in msg


_R = {"checked": 0, "total": 0, "weight_share_checked": 0.0, "unchecked_by_method": {}, "quantized_or_moe": 0,
      "never_checked": []}


def test_without_its_triton_kernels_attention_isnt_said_to_be_checked_inside_cuda_graphs(monkeypatch, tmp_path):
    """The Triton kernels failed their first-use comparison (here: a CPU, where they aren't used): the PyTorch check
    runs outside CUDA graphs only, so a compiled server's decode steps aren't checked, and the summary, the stack and
    alerts say so."""
    import types
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    assert vp._attn_kernels() == (False, "cpu bfloat16: the Triton kernels aren't available here")
    assert "attention=outside_graphs" in vp._STATE["stack"]["checks"]
    assert "outside CUDA graphs only" in vp._summary(dict(_R))
    ev = types.SimpleNamespace(detector="serve", detail={})
    vp._attribute(ev)
    u = " ".join(ev.detail["unknown_impact"])
    assert "sampled attention outputs outside CUDA graphs and those tokens' KV writes" in u
    assert "attention outputs not sampled or run inside CUDA graphs" in u


def test_with_its_triton_kernels_attention_is_reported_on(monkeypatch, tmp_path):
    import types
    from sdcguard import serve_attn
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setitem(serve_attn.STATUS, (serve_attn._dev_key(torch.device("cpu")), torch.bfloat16), (True, ""))
    monkeypatch.setattr(serve_attn.AttnChecks, "_precompile", lambda self, *a: None)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    assert vp._attn_kernels() == (True, "")
    assert vp._STATE["stack"]["checks"].startswith("attention=on,")      # (the keys are sorted)
    msg = vp._summary(dict(_R))
    assert "attention checked by a sampled recompute on 2 layers (SDCGUARD_ATTN=0 turns it off)" in msg
    ev = types.SimpleNamespace(detector="serve", detail={})
    vp._attribute(ev)
    assert "sampled attention outputs and those tokens' KV writes" in " ".join(ev.detail["unknown_impact"])


def test_a_model_without_attention_layers_says_so(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    runner = GM(lambda: torch.nn.Sequential(torch.nn.Linear(4, 4)))
    runner.load_model()
    assert vp._STATE["attn"] == {"checked": 0, "not_checked": {}}
    assert "attention not checked: no attention layers found in this model" in vp._summary(dict(_R))
    assert "attention=off" in vp._STATE["stack"]["checks"]


def test_a_model_that_cant_be_found_or_a_failed_registration_says_attention_isnt_checked(monkeypatch, tmp_path):
    from sdcguard import serve
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    vp.register()
    GM(lambda: None).load_model()
    assert "attention not checked: the model to register its attention layers wasn't found" in vp._summary(dict(_R))
    vp, GM, impls = _setup(monkeypatch, tmp_path)

    def broken(self, line=None):
        raise RuntimeError("no attention checks")
    monkeypatch.setattr(serve.ServeGuard, "attn_checks", broken)
    vp.register()
    GM(_model(impls)).load_model()
    assert "attention not checked: registering its layers failed (RuntimeError)" in vp._summary(dict(_R))
    assert "attention=off" in vp._STATE["stack"]["checks"]


def test_layers_registered_stay_reported_as_checked_when_reading_their_kernel_status_fails(monkeypatch, tmp_path):
    from sdcguard import serve_attn

    def broken(self):
        raise RuntimeError("no status")
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(serve_attn.AttnChecks, "kernel_status", broken)
    vp.register()
    runner = GM(_model(impls))
    runner.load_model()
    for b in attn_batches(4, 2, 64, 3, steps=4):
        runner.execute_model(b)
    assert vp.poll(wait=True) == [] and vp.guard().attn.stats["calls_checked"] == 4
    assert vp._STATE["attn"]["checked"] == 2 and not vp._STATE.get("attn_why")
    msg = vp._summary(dict(_R))
    assert "attention checked by a sampled recompute on 2 layers possibly outside CUDA graphs only (its kernel " \
           "status couldn't be read (RuntimeError))" in msg
    assert "attention=on_unknown_where" in vp._STATE["stack"]["checks"]
    monkeypatch.setattr(serve_attn.AttnChecks, "kernel_status", lambda self: {})
    assert vp._attn_kernels() == (None, "no kernel status")
    assert "possibly outside CUDA graphs only (no kernel status)" in vp._summary(dict(_R))
    import types
    ev = types.SimpleNamespace(detector="serve", detail={})
    vp._attribute(ev)
    assert "sampled attention outputs (perhaps only outside CUDA graphs)" in " ".join(ev.detail["unknown_impact"])


def test_a_vllm_without_either_attention_backend_says_attention_isnt_checked(monkeypatch, tmp_path):
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(vp, "_patch_attn", lambda: [])
    vp.register()
    assert vp._STATE["attn_classes"] == []
    assert "attention not checked: neither of vLLM's FlashAttention and Triton attention backends was found" in \
        vp._summary(dict(_R))


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


def test_attention_is_listed_not_checked_under_microbatching(monkeypatch, tmp_path, caplog):
    """vLLM 0.32 splits steps into microbatches with dual-batch overlap or --ubatch-size above 1; the attention check
    hasn't been run that way, so its layers are listed as not checked, and the model serves."""
    import types
    vp, GM, impls = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(GM, "parallel_config", types.SimpleNamespace(enable_dbo=False, ubatch_size=2), raising=False)
    vp.register()
    runner = GM(_model(impls))
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        runner.load_model()
    g = vp.guard()
    assert g is None or g.attn is None
    assert all("microbatching" in w for w in vp._STATE["attn"]["not_checked"].values())
    assert len(vp._STATE["attn"]["not_checked"]) == 3 and vp._STATE["attn"]["checked"] == 0
    assert any("attention isn't checked: microbatching" in r.getMessage() for r in caplog.records)
    outs = runner.execute_model(attn_batches(4, 2, 64, 3, steps=1)[0])
    assert all(torch.isfinite(o.float()).all() for o in outs)
