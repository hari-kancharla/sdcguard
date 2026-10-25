"""Every Triton kernel of the serving checks compiles for the NVIDIA GPUs they run on (A100, L4, H100/H200), from the
launch code with the arguments it passes, specialized as Triton specializes a real launch, on a machine without a GPU
(tests/_triton_aot.py). The CPU tests run these kernels in Triton's interpreter, which doesn't type-check: the
routing kernel failed to compile on its first GPU (October 24, 2026) after passing every interpreted test, and its
MoE server fell back to the PyTorch check. The variants cover what changes the compiled code: dtypes, policies,
sizes like a real model's, a single row, a single expert per token, a router or layer other than the first, and
the linear check's row tiles, and attention's head sizes and head counts, with and without the KV-write check.
(The Triton here may not be the one a deployment has: vLLM 0.32's image has 3.7.1.)"""
import pytest
import torch

pytest.importorskip("triton")

import _triton_aot  # noqa: E402
from _triton_aot import compiling  # noqa: E402
from _vllm_standins import install_fake_moe, vllm_grouped_topk, vllm_topk  # noqa: E402
from sdcguard import _serve_fast  # noqa: E402
from sdcguard import serve_attn as sa  # noqa: E402
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


def test_the_compile_follows_tritons_own_specialization():
    assert _triton_aot.FAITHFUL, "this Triton has no binder to specialize like a real launch"


BIAS = torch.linspace(-0.05, 0.05, 64)
ROUTERS = {
    "softmax top-k": (dict(top_k=4, scoring_func="softmax", renormalize=True), "FusedTopKRouter",
                      lambda lg: vllm_topk(lg, 4, True)),
    "sigmoid top-k, not renormalized": (dict(top_k=6, scoring_func="sigmoid", renormalize=False), "FusedTopKRouter",
                                        lambda lg: vllm_topk(lg, 6, False, "sigmoid")),
    "top-1": (dict(top_k=1, scoring_func="softmax", renormalize=False), "FusedTopKRouter",
              lambda lg: vllm_topk(lg, 1, False)),
    "grouped, biased, scaled": (
        dict(top_k=8, scoring_func="sigmoid", renormalize=True, num_expert_group=8, topk_group=4,
             routed_scaling_factor=2.5, global_num_experts=64, e_score_correction_bias=BIAS),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 8, True, 8, 4, "sigmoid", 2.5, BIAS)),
    "biased top-k": (
        dict(top_k=4, scoring_func="sigmoid", renormalize=True, routed_scaling_factor=1.5,
             e_score_correction_bias=BIAS, _hash_indices_table=None, bias_vl=None, num_fused_shared_experts=0),
        "FusedTopKBiasRouter", lambda lg: vllm_topk(lg, 4, True, "sigmoid")),
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", list(ROUTERS))
def test_the_routing_kernel_compiles(case, dtype):
    kw, cls, vllm = ROUTERS[case]
    routers = [type(cls, (), dict(kw, _select_experts=lambda *a, **k: None))() for _ in range(3)]
    rc = ServeGuard().route_checks()
    for i, r in enumerate(routers):                  # three routers: slots 0 to 2, taking turns
        assert rc.add(r, f"model.layers.{i}.mlp.experts.router", torch.device("cpu")) is not None
    rc.finalize()
    for rows in (16, 1):
        lg = (torch.randn(rows, 64) * 2).to(dtype)
        w, ids = vllm(lg.float())
        with compiling() as done:
            for r in routers:
                sr.route_check_triton(lg, w, ids, *r._sdcguard_route)
        _compiled(done, "_route_kernel")


MOE_SHAPES = {"small": dict(experts=8, hidden=128, inter=64, top_k=2),
              "granite-3.1-1b-a400m": dict(experts=32, hidden=1024, inter=512, top_k=8),
              "top-1": dict(experts=8, hidden=128, inter=64, top_k=1)}
MOE_CASES = [("granite-3.1-1b-a400m", dt, act) for dt in (torch.bfloat16, torch.float16) for act in (0, 1, 2)] + \
    [("small", torch.bfloat16, 0), ("top-1", torch.bfloat16, 0)]


@pytest.mark.parametrize("shape,dtype,act", MOE_CASES)
def test_the_moe_kernels_compile(monkeypatch, shape, dtype, act):
    mod = install_fake_moe(monkeypatch.setitem)
    sh = MOE_SHAPES[shape]
    lays = [mod.RoutedExperts(seed=i, dtype=dtype, **sh) for i in range(2)]
    mc = ServeGuard().moe_checks()
    for i, lay in enumerate(lays):                   # two layers: slots and indices 0 and 1
        assert mc.add(lay, f"model.layers.{i}.mlp.experts") == i
    mc.finalize()
    e, h = sh["experts"], sh["hidden"]
    for rows in (7, 1):
        x = torch.randn(rows, h).to(dtype)
        w, ids = vllm_topk(torch.randn(rows, e), sh["top_k"], True)
        out = torch.zeros_like(x)
        for emap, rf in ((None, False), (torch.arange(e, dtype=torch.int32), True)):
            with compiling() as done:
                for lay in lays:
                    w13, w2, _emap, _act, _rf, st, slot, sb, nl, idx, cm, cx, cy, ci, cg, line = lay._sdcguard_moe
                    sm.moe_check_triton(x, out, w, ids, w13, w2, emap, act, rf, st, slot, sb, nl, idx, cm, cx, cy,
                                        ci, cg, line)
            _compiled(done, "_moe_up", "_moe_down", "_moe_compare")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("h", [64, 896, 2048, 4097, 8192])
def test_the_norm_kernel_compiles(dtype, h):
    mods = []
    nc = sn.NormChecks(None)
    for i in range(2):
        mod = torch.nn.Module()
        mod.weight = torch.nn.Parameter((1 + 0.3 * torch.randn(h)).to(dtype), requires_grad=False)
        mod.variance_epsilon = 1e-6
        assert nc.add(mod, f"model.layers.{i}.input_layernorm", "rmsnorm") == i
        mods.append(mod)
    nc.finalize(dtypes=())
    for rows in (9, 1):
        x = torch.randn(rows, h).to(dtype)
        with compiling() as done:
            for mod in mods:
                sn.norm_check_triton(x, x.clone(), *mod._sdcguard_norm)
        _compiled(done, "_norm_check_kernel")


@pytest.mark.parametrize("kern", ["v1", "v3"])
@pytest.mark.parametrize("bm", [1, 2, 4, 8, 16])
def test_the_linear_check_kernels_compile_for_every_row_tile(kern, bm):
    k, n = 512, 300
    W = (torch.randn(n, k) / k ** 0.5).to(torch.bfloat16)
    x = torch.randn(64, k).to(torch.bfloat16)
    y = torch.nn.functional.linear(x, W)
    cs = LayerChecksums(W, None, 128)
    st = torch.zeros(2, STATE_COLS, dtype=torch.int64)
    cm, cd = _serve_fast._capture_buffers(x.device, k, slots=2)
    with compiling() as done:
        for slot in (0, 1):
            _serve_fast.check(x, y, *cs.tensors(False), st, slot, 6.0, 1e-20, kern=kern, _bm=bm)
            _serve_fast.check(x, y, *cs.tensors(False), st, slot, 6.0, 1e-20, kern=kern, _bm=bm, cap_meta=cm,
                              cap_data=cd, cap_w=128, cap_tag=2)
    assert done and {cc for _, cc in done} == {80, 89, 90}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("m", [1, 64, 4096])
def test_the_linear_check_kernels_compile(dtype, bias, m):
    k, n = 512, 300
    W = (torch.randn(n, k) / k ** 0.5).to(dtype)
    b = torch.randn(n).to(dtype) if bias else None
    x = torch.randn(m, k).to(dtype)
    y = torch.nn.functional.linear(x, W, b)
    cs = LayerChecksums(W, b, 128)
    st = torch.zeros(1, STATE_COLS, dtype=torch.int64)
    with compiling() as done:
        _serve_fast.check(x, y, *cs.tensors(bias), st, 0, 6.0, 1e-20)
    assert done and {cc for _, cc in done} == {80, 89, 90}


ATTN_SHAPES = {"llama-like": (128, 8, 2), "qwen2.5-1.5b": (128, 12, 2), "head size 64, MHA": (64, 4, 4),
               "head size 80": (80, 4, 1), "head size 256": (256, 8, 4)}
ATTN_CASES = [("qwen2.5-1.5b", torch.bfloat16), ("qwen2.5-1.5b", torch.float16), ("llama-like", torch.float16),
              ("head size 64, MHA", torch.bfloat16), ("head size 80", torch.bfloat16),
              ("head size 256", torch.bfloat16)]


@pytest.mark.parametrize("shape,dtype", ATTN_CASES)
def test_the_attention_kernels_compile(shape, dtype):
    D, Hq, Hkv = ATTN_SHAPES[shape]
    calls = [sa.make_case("cpu", dtype, D=D, Hq=Hq, Hkv=Hkv, lens=((40, 1), (17, 5), (0, 0)), NB=12),
             sa.make_case("cpu", dtype, D=D, Hq=Hq, Hkv=Hkv, lens=((5, 1),), NB=2, padded_rows=0)]   # one row
    b = sa.buffers("cpu", 2, D, 64, dtype)
    with compiling() as done:
        for c in calls:
            for has_write, keep in ((True, True), (False, False)):
                for slot in (0, 1):
                    sa.run_case(sa.attn_check_triton, c, b, slot=slot, idx=slot, nl=2, has_write=has_write,
                                keep_kv=keep)
            sa.run_case(sa.attn_check_triton, c, b, left=8)
    _compiled(done, "_attn_part", "_attn_judge")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.uint8])
@pytest.mark.parametrize("n", [1, 4095, 4096, 2 * 4096 + 17, 5_000_000])
def test_the_weight_scan_kernel_compiles(dtype, n):
    from sdcguard import serve_weights as sw
    t = torch.zeros(n, dtype=dtype)
    row = torch.zeros(2, dtype=torch.int64)
    with compiling() as done:
        sw._launch(t, row)
    _compiled(done, "_fingerprint_kernel")
