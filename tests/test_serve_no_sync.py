"""The serving checks that run inside a model step (the linear, FP8 quantization, MoE, routing and norm checks) must
never read a GPU value on the CPU: under CUDA graph capture that invalidates the capture and vLLM fails to start
(an FP8 MoE model on an L4, October 24, 2026: `x[i]` with a 0-d tensor index `i` calls .item()); outside capture it
stalls every step. Each check's PyTorch version runs here on fake tensors (shapes only, no data), where any operation
that needs a value on the CPU (.item(), indexing with a 0-d tensor, bool() of a tensor, nonzero, a boolean mask)
raises. Every recipe, policy and option that changes which operations run is covered. (The Triton kernels' launchers
read shapes and strides only; tests/test_serve_capture.py captures every check in a real CUDA graph on a GPU.)"""
import pytest
import torch

from _vllm_standins import install_fake_moe, vllm_grouped_topk, vllm_topk
from sdcguard import _serve_fast
from sdcguard import serve_fp8 as sf
from sdcguard import serve_moe as sm
from sdcguard import serve_norm as sn
from sdcguard import serve_route as sr
from sdcguard.serve import STATE_COLS, LayerChecksums, ServeGuard, check_static_impl

fake_tensor = pytest.importorskip("torch._subclasses.fake_tensor")


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def _fake(mode, v):
    if isinstance(v, torch.Tensor):
        return mode.from_tensor(v)
    if isinstance(v, (tuple, list)):
        return type(v)(_fake(mode, x) for x in v)
    return v


def on_fake_tensors(fn, *args, **kwargs):
    """fn(*args, **kwargs) with every tensor argument made a fake tensor, inside FakeTensorMode."""
    mode = fake_tensor.FakeTensorMode()
    fa, fk = _fake(mode, args), {k: _fake(mode, v) for k, v in kwargs.items()}
    with mode, torch.no_grad():
        fn(*fa, **fk)


def test_a_host_read_is_caught_here():
    def leaky(x, i):
        return x[i.argmax()]                       # the FP8 check's bug: index with a 0-d tensor
    with pytest.raises(Exception):
        on_fake_tensors(leaky, torch.zeros(4), torch.zeros(4))
    with pytest.raises(Exception):
        on_fake_tensors(lambda x: x[x > 0], torch.zeros(4))      # a boolean mask
    with pytest.raises(Exception):
        on_fake_tensors(lambda x: bool(x.any()), torch.zeros(4))
    on_fake_tensors(lambda x, i: sn.pick(x, i.argmax()), torch.zeros(4), torch.zeros(4))


def test_pick_is_indexing():
    t = torch.arange(12.0).reshape(4, 3)
    for i in range(4):
        assert torch.equal(sn.pick(t, torch.tensor(i)), t[i])
        assert torch.equal(sn.pick(t.reshape(-1), torch.tensor(i * 2)), t.reshape(-1)[i * 2])
    assert sn.pick(torch.arange(5), torch.tensor(3)).shape == ()


# ----------------------------------------------------------------------------------------------------- linear check
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_the_linear_check(bias, dt):
    g = torch.Generator().manual_seed(0)
    k, n = 64, 300                                  # 300: a remainder block
    W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dt)
    b = torch.randn(n, generator=g).to(dt) if bias else None
    x = torch.randn(8, k, generator=g).to(dt)
    y = torch.nn.functional.linear(x, W, b)
    cs = LayerChecksums(W, b, 128)
    st = torch.zeros(1, STATE_COLS, dtype=torch.int64)
    on_fake_tensors(check_static_impl, x, y, *cs.tensors(True), st, 0, 6.0, 1e-20)
    cm, cd = _serve_fast._capture_buffers(x.device, k)
    on_fake_tensors(check_static_impl, x, y, *cs.tensors(True), st, 0, 6.0, 1e-20, cm, cd, 128, 2)


# ------------------------------------------------------------------------------------------------- FP8 quantization
class _Quant:
    def __init__(self, static=False, per_token=True):
        self.static, self.use_per_token_if_dynamic, self.is_group_quant = static, per_token, False
        self.group_shape = type("G", (), {"name": "PER_TOKEN" if per_token else "PER_TENSOR"})()


FP8_CASES = {   # name: (static, per_token, rows marked, scale bound, rows of the FP8 input (padded), mult)
    "per_token": (False, True, 12, False, 12, 1),
    "per_token_bound": (False, True, 12, True, 12, 1),
    "per_token_padded": (False, True, 9, False, 16, 1),
    "per_tensor": (False, False, 12, False, 12, 1),
    "per_tensor_padded": (False, False, 9, False, 16, 1),
    "static": (True, False, 12, False, 12, 1),
    "moe_top_k": (False, False, 5, False, 12, 2),
}


@pytest.mark.parametrize("case", list(FP8_CASES))
def test_the_fp8_quantization_check(case):
    static, per_token, rows, bound, padded, mult = FP8_CASES[case]
    g = ServeGuard()
    fc = g.fp8_checks()
    x = torch.randn(12, 256, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
    s = (x.float().abs().amax(-1, keepdim=True) / sf.FP8_MAX).clamp(min=sf.MIN_SCALE)
    if not per_token:
        s = s.amax().reshape(1)
    s0 = torch.tensor([0.02]) if static else None
    if static:
        s = s0
    q = (x.float() / s).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
    q = torch.cat([q, q[: padded - 12]]) if padded > 12 else q
    a = fc.add(torch.nn.Module(), "model.layers.0.mlp.down_proj", _Quant(static, per_token), s0, torch.device("cpu"))
    want = "static" if static else ("dynamic_token" if per_token else "dynamic_tensor")
    assert a is not None and a[0] == sf.RECIPES[want]
    fc.mark_step(3, rows=rows)
    on_fake_tensors(sf.check_call, a, x, q, s, torch.tensor(1.0) if bound else None, mult)
    fc.end_step()


# ----------------------------------------------------------------------------------------------------------- MoE
@pytest.mark.parametrize("act", [0, 1, 2])
@pytest.mark.parametrize("expert_map", [False, True])
@pytest.mark.parametrize("routed_first", [False, True])
def test_the_moe_check(monkeypatch, act, expert_map, routed_first):
    mod = install_fake_moe(monkeypatch.setitem)
    lay = mod.RoutedExperts(seed=0)
    g = ServeGuard()
    mc = g.moe_checks()
    assert mc.add(lay, "model.layers.0.mlp.experts") == 0
    mc.finalize()
    e = lay.w13_weight.shape[0]
    x = torch.randn(7, 128).to(torch.bfloat16)
    gen = torch.Generator().manual_seed(1)
    w, ids = torch.topk(torch.softmax(torch.randn(7, e, generator=gen), -1), lay.top_k, -1)
    w, ids = w / w.sum(-1, keepdim=True), ids.to(torch.int32)
    out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
    w13, w2, _emap, _act, _rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg, line = lay._sdcguard_moe
    emap = torch.tensor([i if i < e // 2 else -1 for i in range(e)], dtype=torch.int32) if expert_map else None
    g.mark_step(0)
    on_fake_tensors(sm.moe_check_torch, x, out, w, ids, w13, w2, emap, act, routed_first, st, slot, sb, nl, idx, cm,
                    cx, cy, ci, cg, line)


# ------------------------------------------------------------------------------------------------------- routing
BIAS = torch.linspace(-0.05, 0.05, 64)
ROUTERS = {
    "softmax top-k, renormalized": (dict(top_k=4, scoring_func="softmax", renormalize=True), "FusedTopKRouter",
                                    lambda lg: vllm_topk(lg, 4, True)),
    "sigmoid top-k": (dict(top_k=6, scoring_func="sigmoid", renormalize=False), "FusedTopKRouter",
                      lambda lg: vllm_topk(lg, 6, False, "sigmoid")),
    "grouped, biased, scaled": (
        dict(top_k=8, scoring_func="sigmoid", renormalize=True, num_expert_group=8, topk_group=4,
             routed_scaling_factor=2.5, global_num_experts=64, e_score_correction_bias=BIAS),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 8, True, 8, 4, "sigmoid", 2.5, BIAS)),
    "grouped softmax, no bias": (
        dict(top_k=6, scoring_func="softmax", renormalize=False, num_expert_group=4, topk_group=2,
             routed_scaling_factor=1.0, global_num_experts=64),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 6, False, 4, 2, "softmax")),
    "biased top-k": (
        dict(top_k=4, scoring_func="sigmoid", renormalize=True, routed_scaling_factor=1.5,
             e_score_correction_bias=BIAS, _hash_indices_table=None, bias_vl=None, num_fused_shared_experts=0),
        "FusedTopKBiasRouter", lambda lg: vllm_topk(lg, 4, True, "sigmoid")),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", list(ROUTERS))
def test_the_routing_check(case, dtype):
    kw, cls, vllm = ROUTERS[case]
    router = type(cls, (), dict(kw, _select_experts=lambda *a, **k: None))()
    g = ServeGuard()
    rc = g.route_checks()
    assert rc.add(router, "model.layers.0.mlp.experts.router", torch.device("cpu")) is not None
    rc.finalize()
    lg = (torch.randn(16, 64, generator=torch.Generator().manual_seed(0)) * 2).to(dtype)
    w, ids = vllm(lg.float())
    rc.mark_step(1, rows=16)
    on_fake_tensors(sr.route_check_torch, lg, w, ids, *router._sdcguard_route)
    rc.end_step()


# ---------------------------------------------------------------------------------------------------------- norms
@pytest.mark.parametrize("kind", ["rmsnorm", "gemma_rmsnorm"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_norm_check(kind, dtype):
    h = 896
    mod = torch.nn.Module()
    mod.weight = torch.nn.Parameter((1 + 0.3 * torch.randn(h)).to(dtype), requires_grad=False)
    mod.variance_epsilon = 1e-6
    nc = sn.NormChecks(None)
    assert nc.add(mod, "model.layers.0.input_layernorm", kind) == 0
    nc.finalize(dtypes=())
    x = torch.randn(9, h).to(dtype)
    s = x.float()
    y = (s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6) * nc.layers[0]["w_ref"]).to(dtype)
    nc.mark_step(0)
    on_fake_tensors(sn.norm_check_torch, x, y, *mod._sdcguard_norm)
