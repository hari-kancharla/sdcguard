"""Every Triton kernel of the serving checks compiles for the NVIDIA GPUs they run on (A100, L4, H100/H200), from the
launch code with the argument types it passes, on a machine without a GPU (tests/_triton_aot.py). The CPU tests run
these kernels in Triton's interpreter, which doesn't type-check: the routing kernel failed to compile on its first
GPU (October 24, 2026) after passing every interpreted test, and its MoE server fell back to the PyTorch check."""
import pytest
import torch

pytest.importorskip("triton")

from _triton_aot import compiling  # noqa: E402
from _vllm_standins import install_fake_moe, vllm_grouped_topk, vllm_topk  # noqa: E402
from sdcguard import _serve_fast  # noqa: E402
from sdcguard import serve_moe as sm  # noqa: E402
from sdcguard import serve_norm as sn  # noqa: E402
from sdcguard import serve_route as sr  # noqa: E402
from sdcguard.serve import STATE_COLS, LayerChecksums, ServeGuard  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def _compiled(done, *names):
    got = {n for n, _ in done}
    assert set(names) <= got, (names, done)
    assert {cc for _, cc in done} == {80, 89, 90}


BIAS = torch.linspace(-0.05, 0.05, 64)
ROUTERS = {
    "softmax top-k": (dict(top_k=4, scoring_func="softmax", renormalize=True), "FusedTopKRouter",
                      lambda lg: vllm_topk(lg, 4, True)),
    "grouped, biased, scaled": (
        dict(top_k=8, scoring_func="sigmoid", renormalize=True, num_expert_group=8, topk_group=4,
             routed_scaling_factor=2.5, global_num_experts=64, e_score_correction_bias=BIAS),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 8, True, 8, 4, "sigmoid", 2.5, BIAS)),
    "biased top-k": (
        dict(top_k=4, scoring_func="sigmoid", renormalize=True, routed_scaling_factor=1.5,
             e_score_correction_bias=BIAS, _hash_indices_table=None, bias_vl=None, num_fused_shared_experts=0),
        "FusedTopKBiasRouter", lambda lg: vllm_topk(lg, 4, True, "sigmoid")),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", list(ROUTERS))
def test_the_routing_kernel_compiles(case, dtype):
    kw, cls, vllm = ROUTERS[case]
    router = type(cls, (), dict(kw, _select_experts=lambda *a, **k: None))()
    rc = ServeGuard().route_checks()
    assert rc.add(router, "model.layers.0.mlp.experts.router", torch.device("cpu")) is not None
    rc.finalize()
    lg = (torch.randn(16, 64) * 2).to(dtype)
    w, ids = vllm(lg.float())
    with compiling() as done:
        sr.route_check_triton(lg, w, ids, *router._sdcguard_route)
    _compiled(done, "_route_kernel")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("act", [0, 1, 2])
def test_the_moe_kernels_compile(monkeypatch, dtype, act):
    mod = install_fake_moe(monkeypatch.setitem)
    lay = mod.RoutedExperts(seed=0, dtype=dtype)
    mc = ServeGuard().moe_checks()
    assert mc.add(lay, "model.layers.0.mlp.experts") == 0
    mc.finalize()
    e = lay.w13_weight.shape[0]
    x = torch.randn(7, 128).to(dtype)
    w, ids = vllm_topk(torch.randn(7, e), lay.top_k, True)
    out = torch.zeros_like(x)
    w13, w2, _emap, _act, _rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg, line = lay._sdcguard_moe
    for emap, rf in ((None, False), (torch.arange(e, dtype=torch.int32), True)):
        with compiling() as done:
            sm.moe_check_triton(x, out, w, ids, w13, w2, emap, act, rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg,
                                line)
        _compiled(done, "_moe_up", "_moe_down", "_moe_compare")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("h", [64, 896, 4097, 8192])
def test_the_norm_kernel_compiles(dtype, h):
    mod = torch.nn.Module()
    mod.weight = torch.nn.Parameter((1 + 0.3 * torch.randn(h)).to(dtype), requires_grad=False)
    mod.variance_epsilon = 1e-6
    nc = sn.NormChecks(None)
    assert nc.add(mod, "model.layers.0.input_layernorm", "rmsnorm") == 0
    nc.finalize(dtypes=())
    x = torch.randn(9, h).to(dtype)
    with compiling() as done:
        sn.norm_check_triton(x, x.clone(), *mod._sdcguard_norm)
    _compiled(done, "_norm_check_kernel")


@pytest.mark.parametrize("kern", ["v1", "v3"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
def test_the_linear_check_kernels_compile(kern, dtype, bias):
    k, n = 512, 300
    W = (torch.randn(n, k) / k ** 0.5).to(dtype)
    b = torch.randn(n).to(dtype) if bias else None
    x = torch.randn(64, k).to(dtype)
    y = torch.nn.functional.linear(x, W, b)
    cs = LayerChecksums(W, b, 128)
    st = torch.zeros(1, STATE_COLS, dtype=torch.int64)
    cm, cd = _serve_fast._capture_buffers(x.device, k)
    with compiling() as done:
        _serve_fast.check(x, y, *cs.tensors(bias), st, 0, 6.0, 1e-20, kern=kern)
        _serve_fast.check(x, y, *cs.tensors(bias), st, 0, 6.0, 1e-20, kern=kern, cap_meta=cm, cap_data=cd,
                          cap_w=128, cap_tag=2)
    assert done and {cc for _, cc in done} == {80, 89, 90}
