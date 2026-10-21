"""What the vLLM plugin counts as checked, what SDCGUARD_REQUIRE asks for, and which layers a share selects.

The startup count, sdcguard_layers_unchecked and SDCGUARD_REQUIRE are only as good as the denominator: layers the
plugin can never select (the LM head, a convolution kept in a linear layer) must not make REQUIRE=1 refuse, while
linear-layer work it can't check (fused MoE experts, quantized layers) and layers whose check never runs must not
count as checked. vLLM is not installed: the stand-ins are in _vllm_standins.py."""
import logging

import pytest

from _vllm_standins import (ConvAsLinear, Decoder, GateLinear, Lin, RoutedExperts, install_fake_vllm, load,
                            plugin_env, x_in)


def _fake_vllm(monkeypatch):
    return install_fake_vllm(monkeypatch.setitem)


# ---------------------------------------------------------------- SDCGUARD_REQUIRE
def test_require_1_serves_a_model_with_an_lm_head_when_every_selected_layer_is_checked(monkeypatch, tmp_path):
    """REQUIRE=1 is "every selected layer checked". vLLM's ParallelLMHead (UnquantizedEmbeddingMethod, never
    selectable, documented as not checked) must stay out of that count: in it, checked/total = 4/5 < 1 refused
    every Llama or Qwen model with all of its linear layers checked."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1")
    plugin.register()
    model = load(Decoder(M, n_layers=4))
    assert len(plugin.guard().layers) == 4
    Runner(model).execute_model(x_in())                       # serves
    r = plugin._STATE["layers"]
    assert r["checked"] == r["selected"] == 4 and r["unchecked_by_method"] == {"LM head": 1}


def test_require_1_with_share_does_not_refuse_some_replica_offsets(monkeypatch, tmp_path):
    """With SDCGUARD_SHARE=1/R the golden-ratio selection picks floor or ceil of N/R layers per offset. REQUIRE
    compares the checked layers with those the offset selected, not with N x SHARE: otherwise offsets that drew
    fewer refuse to serve although every layer they selected is checked."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1", SDCGUARD_SHARE="0.25", SDCGUARD_OFFSET="1")
    plugin.register()
    model = load(Decoder(M, n_layers=12, lm_head=False))
    assert len(plugin.guard().layers) == 2                    # 2 of 12 selected (select_layer), both checked
    Runner(model).execute_model(x_in())
    assert plugin._STATE["layers"]["unchecked_by_method"] == {"not selected by SDCGUARD_SHARE": 10}


def test_require_1_is_not_refused_by_a_convolution_and_its_reason_is_not_share(monkeypatch, tmp_path):
    """A conv1d held in a ColumnParallelLinear (Mamba, Jamba, Nemotron-H, LFM2 ShortConv) can never be selected
    (3-D weight). It is reported by that cause, not as "not selected by SDCGUARD_SHARE" with the share unset, and
    REQUIRE=1 doesn't refuse the model for it."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1")
    plugin.register()
    model = load(Decoder(M, n_layers=4, lm_head=False, extra=[ConvAsLinear(M)]))
    r = plugin.layer_report(model)
    assert list(r["unchecked_by_method"]) == ["3-D weight (a convolution kept in a linear layer)"], r
    Runner(model).execute_model(x_in())


class _Fp32Gate(Lin):
    """An fp32 router gate (GLM-4.5, MiniMax-M2, ERNIE-4.5 MoE: params_dtype=float32) in a bf16 model."""

    def __init__(self, M):
        super().__init__(M, 256, 16, "model.layers.0.mlp.gate")
        self.weight.data = self.weight.data.float()

    def forward(self, x):
        return self.quant_method.apply(self, x.float(), None)


def test_an_fp32_layer_is_reported_by_its_dtype_and_an_fp32_model_is_refused(monkeypatch, tmp_path):
    """An fp32 router gate in a bf16 model is reported as fp32, not as unselected, and doesn't stop REQUIRE=1. A
    model with nothing the plugin can check (all fp32) is refused: no REQUIRE is met with nothing checked."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1")
    plugin.register()
    model = load(Decoder(M, n_layers=2, lm_head=False, extra=[_Fp32Gate(M)]))
    assert plugin.layer_report(model)["unchecked_by_method"] == {
        "float32 weights (the check takes bf16 and fp16)": 1}
    Runner(model).execute_model(x_in())
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1")
    M, Runner = _fake_vllm(monkeypatch)
    plugin.register()
    fp32 = Decoder(M, n_layers=2, lm_head=False)
    for lay in fp32.layers:
        lay.weight.data = lay.weight.data.float()
    load(fp32)
    with pytest.raises(RuntimeError, match="SDCGUARD_REQUIRE"):
        Runner(fp32).execute_model(x_in().float())


def test_the_lm_head_is_not_reported_as_a_quantized_layer(monkeypatch, tmp_path, caplog):
    """A case-sensitive startswith("unquantized") made an unquantized bf16 model log a WARNING "not checked
    (quantized layers aren't supported yet): 1 UnquantizedEmbeddingMethod". It's the LM head, at info level."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(Decoder(M, n_layers=4))
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        Runner(model).execute_model(x_in())
    recs = [r for r in caplog.records if "of their weights" in r.getMessage()]
    assert recs and all(r.levelno == logging.INFO for r in recs), [r.getMessage() for r in recs]
    assert not any("quantized" in r.getMessage() for r in recs) and "1 LM head" in recs[0].getMessage()


# ---------------------------------------------------------------- linear-layer work the plugin can't check
def test_layer_report_counts_fused_moe_experts_as_unchecked(monkeypatch, tmp_path):
    """Fused MoE experts have no `weight` (w13_weight, w2_weight; w13_qweight / w13_weight_packed when quantized),
    and were neither checked nor counted: a Mixtral/Qwen3-MoE/DeepSeek model logged "checking N of N linear
    layers". They are counted, and the share of weights checked shows they are most of the matmul work."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(Decoder(M, n_layers=4, lm_head=False, extra=[RoutedExperts()]))
    r = plugin.layer_report(model)
    assert r["total"] == 5 and r["checked"] == 4, r
    assert r["unchecked_by_method"] == {"fused MoE experts (UnquantizedFusedMoEMethod)": 1}
    assert r["weight_share_checked"] < 0.1             # 4 x 256 x 256 of 3.4 million weights
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    met = Metrics()
    met.layer_report(r)
    text = generate_latest(met.registry).decode()
    assert 'sdcguard_layers_unchecked{method="fused MoE experts (UnquantizedFusedMoEMethod)"} 1.0' in text
    assert f"sdcguard_weights_checked_ratio {r['weight_share_checked']}" in text


def test_require_1_refuses_a_moe_model_whose_experts_are_unchecked(monkeypatch, tmp_path):
    """Same gap, seen through the gate: SDCGUARD_REQUIRE=1 must not serve a MoE model with every expert
    unchecked."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1")
    plugin.register()
    model = load(Decoder(M, n_layers=4, lm_head=False, extra=[RoutedExperts()]))
    with pytest.raises(RuntimeError, match="SDCGUARD_REQUIRE"):
        Runner(model).execute_model(x_in())


def test_a_router_gate_that_vllm_computes_with_its_own_gemm_is_not_registered(monkeypatch, tmp_path):
    """GateLinear with a bf16 weight and fp32 output (allow_cublas_router_gemm or allow_ll_bf16_gemm) never calls
    apply on NVIDIA GPUs. Registering it would count a check that never runs; it's reported by that cause."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(Decoder(M, n_layers=2, lm_head=False, extra=[GateLinear(M, flags=True)]))
    assert [lay["name"] for lay in plugin.guard().layers] == ["model.layers.0.proj", "model.layers.1.proj"]
    r = plugin.layer_report(model)
    assert r["unchecked_by_method"] == {"router gate computed by GateLinear's own GEMM": 1} and r["cant_check"] == 1


def test_layer_report_does_not_count_a_layer_whose_forward_bypasses_apply_as_checked(monkeypatch, tmp_path):
    """A registered layer whose forward never calls apply (a GateLinear-like gate without the flags above, MLA's
    decode) runs no check: guard.counts() stays 0. Once checks have run, layer_report reads the counts and
    reports such a layer as never checked instead of checked."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(Decoder(M, n_layers=4, lm_head=False, extra=[GateLinear(M)]))
    r = Runner(model)
    for _ in range(3):
        r.execute_model(x_in())
    executed = {n for n, v in plugin.guard().counts().items() if v["checks"] > 0}
    assert "model.layers.0.mlp.gate" not in executed
    rep = plugin.layer_report(model)
    assert rep["checked"] == len(executed) == 4 and rep["never_checked"] == ["model.layers.0.mlp.gate"], rep


def test_the_first_read_reports_a_layer_that_never_ran_its_check_and_require_refuses(monkeypatch, tmp_path, caplog):
    """At the first read of results (no warm-up ran here, so the startup count can't know yet) the plugin counts
    again from what was read back: the gate whose check never ran is logged, exported in stats, and REQUIRE=1
    refuses to go on serving."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_POLL_EVERY="2")
    plugin.register()
    model = load(Decoder(M, n_layers=4, lm_head=False, extra=[GateLinear(M)]))
    r = Runner(model)
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        r.execute_model(x_in())
        assert plugin._STATE["layers"]["checked"] == 5                 # nothing known yet at the start
        r.execute_model(x_in())                                        # the first read
    assert plugin._STATE["layers"]["never_checked"] == ["model.layers.0.mlp.gate"]
    assert plugin.guard().stats["layers_never_checked"] == 1
    assert any("never ran their check: model.layers.0.mlp.gate" in m.getMessage() for m in caplog.records)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_POLL_EVERY="2", SDCGUARD_REQUIRE="1")
    M, Runner = _fake_vllm(monkeypatch)
    plugin.register()
    r = Runner(load(Decoder(M, n_layers=4, lm_head=False, extra=[GateLinear(M)])))
    r.execute_model(x_in())
    with pytest.raises(RuntimeError, match="SDCGUARD_REQUIRE"):
        r.execute_model(x_in())


def test_require_1_refuses_a_model_none_of_whose_layers_ever_runs_its_check(monkeypatch, tmp_path):
    """The count again from what was read back waited for a read with some check in it. In a model where no
    registered layer reaches its check (here the only one is a gate whose forward bypasses apply), no read ever
    has one, so REQUIRE=1 served unchecked for good (outside review, October 21, 2026). It now refuses at the
    first read."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1", SDCGUARD_POLL_EVERY="1")
    plugin.register()
    model = load(Decoder(M, n_layers=0, lm_head=False, extra=[GateLinear(M)]))
    assert [lay["name"] for lay in plugin.guard().layers] == ["model.layers.0.mlp.gate"]
    with pytest.raises(plugin.RequirementNotMet, match="SDCGUARD_REQUIRE"):
        Runner(model).execute_model(x_in())
    assert plugin.guard().counts()["model.layers.0.mlp.gate"]["checks"] == 0


def test_require_1_refuses_when_the_results_cant_be_read_to_count(monkeypatch, tmp_path):
    """The strict count waits for the GPU once. If that read fails, coverage can't be shown, so REQUIRE=1 refuses
    with the reason (not an unrelated error out of execute_model)."""
    M, Runner = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_REQUIRE="1", SDCGUARD_POLL_EVERY="1")
    plugin.register()
    model = load(Decoder(M, n_layers=2, lm_head=False))
    g = plugin.guard()
    real = g.poll

    def poll(wait=False):
        if wait:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real(wait=False)
    monkeypatch.setattr(g, "poll", poll)
    with pytest.raises(plugin.RequirementNotMet, match="couldn't be read to count what ran.*illegal memory access"):
        Runner(model).execute_model(x_in())


# ---------------------------------------------------------------- weight reload
def test_reloading_weights_keeps_the_share_selection(monkeypatch, tmp_path):
    """vLLM calls process_weights_after_loading again on a weight reload (model_loader/reload/layerwise.py).
    Registered layers refresh; unselected ones must keep their place in the order, or each reload selects more
    layers (2 -> 4 of 12 at 1/4) and the share and the replica partition by SDCGUARD_OFFSET no longer hold."""
    M, _ = _fake_vllm(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_SHARE="0.25", SDCGUARD_OFFSET="1")
    plugin.register()
    model = load(Decoder(M, n_layers=12, lm_head=False))
    before = [lay["name"] for lay in plugin.guard().layers]
    load(model)                                                  # reload_weights()
    load(model)
    assert [lay["name"] for lay in plugin.guard().layers] == before
