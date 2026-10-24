"""On a GPU: each serving check run inside a captured CUDA graph, the way vLLM captures them (the call recorded once,
replayed every step). For each check, its PyTorch version and its Triton kernels where it has them (the FP8
quantization check has only the PyTorch one): the capture succeeds (a check that reads a GPU value on the CPU
invalidates it, which is how an FP8 MoE server failed to start on October 24, 2026); a replay leaves exactly the
state an eager call leaves; a wrong value is found inside the replayed graph; and the step and real-row count marked
between replays are read at replay time, not frozen at capture. Skipped without a GPU; run on one by
`modal run scripts/modal_vllm.py --phase graphs`."""
import os

import pytest
import torch

import _fp8_standins as fs
from _vllm_standins import install_fake_moe, vllm_topk
from sdcguard import _serve_fast
from sdcguard import serve_attn as sa
from sdcguard import serve_fp8 as sf
from sdcguard import serve_moe as sm
from sdcguard import serve_norm as sn
from sdcguard import serve_route as sr
from sdcguard.serve import STATE_COLS, LayerChecksums, ServeGuard, check_static_impl

# SDCGUARD_GPU_TESTS=1 (the graphs phase): no GPU is a failure, not a skip
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() and os.environ.get("SDCGUARD_GPU_TESTS") != "1",
                                reason="needs an NVIDIA GPU")
DEV = torch.device("cuda")


def capture(fn):
    """fn captured in a CUDA graph, warmed up first on a side stream as PyTorch asks."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def close_counts(e, r, count_cols, z_of, rel=1e-3):
    """State that went through cuBLAS (the MoE recompute, the linear check's sums): the same counts exactly and the
    same score to within rel (a replay may run another cuBLAS algorithm than the eager call: the last bits can
    differ), on the same side of the line."""
    assert torch.equal(e[:, count_cols], r[:, count_cols]), (e, r)
    ze, zr = z_of(e), z_of(r)
    assert abs(ze - zr) <= rel * max(abs(ze), 1.0), (ze, zr)


def eager_and_replayed(fn, state, kept=None):
    """(the state after one eager call, the state after one replay of the captured call, the graph). state and the
    kept-row flags (kept) are zeroed before each, so the replay has to write them itself."""
    def zero():
        state.zero_()
        if kept is not None:
            kept.zero_()
    zero()
    fn()
    torch.cuda.synchronize()
    eager = (state.clone(), None if kept is None else kept.clone())
    g = capture(fn)
    zero()
    g.replay()
    torch.cuda.synchronize()
    replayed = (state.clone(), None if kept is None else kept.clone())
    if kept is None:
        return eager[0], replayed[0], g
    return eager, replayed, g


# ------------------------------------------------------------------------------------------------- FP8 quantization
class _Quant:
    def __init__(self, static=False, per_token=True):
        self.static, self.use_per_token_if_dynamic, self.is_group_quant = static, per_token, False
        self.group_shape = fs.PER_TOKEN if per_token else fs.PER_TENSOR       # vLLM 0.32's GroupShape


@pytest.mark.parametrize("through", ["call", "op"])
def test_a_per_tensor_scale_from_padding_rows_is_found_in_a_replayed_graph(through):
    """vLLM #62124's mechanism inside a graph: rows 10..15 are padding holding larger values; the scale over every
    row is named as taken from outside the batch. Through check_call (eager code, as inside vLLM's MoE op) and through
    the registered op (what torch.compile's code calls)."""
    g = ServeGuard()
    fc = g.fp8_checks()
    a = fc.add(torch.nn.Module(), "model.layers.0.mlp.experts", _Quant(per_token=False), None, DEV)
    st, slot = a[4], a[5]
    x = torch.randn(16, 256, device=DEV).to(torch.bfloat16)
    x[10:] *= 8
    s = (x.float().abs().amax() / sf.FP8_MAX).reshape(1)
    q = (x.float() * (1.0 / s)).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
    op = sf.quant_op()
    fn = (lambda: sf.check_call(a, x, q, s)) if through == "call" else \
        (lambda: op(x, q, s, a[0], None, a[1], a[2], a[3], st, slot, 1))
    fc.mark_step(3, rows=10)
    eager, replayed, graph = eager_and_replayed(fn, st)
    assert torch.equal(eager, replayed)
    assert replayed[slot, sf.C_CALLS] == 1 and replayed[slot, sf.C_OUTSIDE] == 1
    ev = g.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["finding"] == "scale_from_outside_rows" and ev[0].detail["captured"]["step"] == 3
    fc.mark_step(4, rows=0)                 # outside a model step (rows 0): nothing judged at replay
    st.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(st[slot, sf.C_CALLS]) == 0
    fc.mark_step(5, rows=16)                # every row real: the maximum over them is the right scale
    st.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(st[slot, sf.C_CALLS]) == 1 and int(st[slot, sf.C_SCALE]) == 0 and int(st[slot, sf.C_OUTSIDE]) == 0
    assert g.poll(wait=True) == []


@pytest.mark.parametrize("recipe", ["per_token", "static"])
def test_wrong_fp8_scales_are_found_in_a_replayed_graph(recipe):
    g = ServeGuard()
    fc = g.fp8_checks()
    static = torch.tensor([0.02], device=DEV) if recipe == "static" else None
    a = fc.add(torch.nn.Module(), "model.layers.0.mlp.down_proj", _Quant(static=static is not None,
               per_token=recipe == "per_token"), static, DEV)
    st, slot = a[4], a[5]
    x = torch.randn(12, 256, device=DEV).to(torch.bfloat16)
    if recipe == "per_token":
        s = (x.float().abs().amax(-1, keepdim=True) / sf.FP8_MAX).clamp(min=sf.MIN_SCALE)
        wrong = s * 1.5
        q = (x.float() / wrong).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
        col = sf.C_SCALE
    else:
        wrong = torch.tensor([0.03], device=DEV)
        q = (x.float() * (1.0 / wrong)).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
        col = sf.C_STATIC
    fc.mark_step(7, rows=12)
    eager, replayed, _ = eager_and_replayed(lambda: sf.check_call(a, x, q, wrong), st)
    assert torch.equal(eager, replayed) and int(replayed[slot, col]) >= 1
    assert g.poll(wait=True)


# ----------------------------------------------------------------------------------------------------------- MoE
@pytest.mark.parametrize("path", ["torch", "triton"])
def test_the_moe_check_in_a_replayed_graph(monkeypatch, path):
    mod = install_fake_moe(monkeypatch.setitem)
    lay = mod.RoutedExperts(seed=0).to(DEV)
    other = mod.RoutedExperts(seed=1).to(DEV)
    g = ServeGuard()
    mc = g.moe_checks()
    assert mc.add(lay, "model.layers.0.mlp.experts") == 0
    assert mc.add(other, "model.layers.1.mlp.experts") == 1          # two layers: they take turns by step
    mc.finalize()
    if path == "triton":
        ok, why = sm.STATUS.get((sm._dev_key(DEV), torch.bfloat16, 0), (False, "not verified"))
        assert ok, why
    fn_check = sm.moe_check_torch if path == "torch" else sm.moe_check_triton
    x = torch.randn(7, 128, device=DEV).to(torch.bfloat16)
    w, ids = vllm_topk(torch.randn(7, lay.w13_weight.shape[0], device=DEV), lay.top_k, True)
    out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
    a = lay._sdcguard_moe
    st, slot, cm = a[5], a[6], a[10]
    z_of = lambda t: sm._f32(int(t[slot, 0]) >> 32)  # noqa: E731
    for wrong in (False, True):
        y = out + 1.0 if wrong else out           # every row 1.0 off: far over the line
        mc.mark_step(2)                           # layer 0's turn (2 % 2)
        (e_st, e_cm), (r_st, r_cm), graph = eager_and_replayed(lambda: fn_check(x, y, w, ids, *a), st, cm)
        close_counts(e_st, r_st, slice(sm.C_CALLS, None), z_of)
        assert int(r_st[slot, sm.C_CALLS]) == 1 and (z_of(r_st) > mc.line) == wrong, z_of(r_st)
        assert int(e_cm[slot, 0]) == int(r_cm[slot, 0]) == int(wrong)       # the row kept, by the replay too
    mc.mark_step(3)                               # layer 1's turn: layer 0's captured check must stay inactive
    st.zero_()
    cm.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(st[slot, sm.C_CALLS]) == 0 and int(cm[slot, 0]) == 0


# ------------------------------------------------------------------------------------------------------- routing
@pytest.mark.parametrize("path", ["torch", "triton"])
def test_the_routing_check_in_a_replayed_graph(path):
    router = type("FusedTopKRouter", (), dict(top_k=4, scoring_func="softmax", renormalize=True,
                                              _select_experts=lambda *a, **k: None))()
    g = ServeGuard()
    rc = g.route_checks()
    assert rc.add(router, "model.layers.0.mlp.experts.router", DEV) is not None
    rc.finalize()
    lg = torch.randn(16, 64, device=DEV) * 2
    w, ids = vllm_topk(lg, 4, True)
    if path == "triton":
        ok, why = sr.verify_kernel(DEV, lg.dtype)
        assert ok, why
    fn_check = sr.route_check_torch if path == "torch" else sr.route_check_triton
    a = router._sdcguard_route
    st, slot, cm = a[9], a[10], a[13]
    bad = ids.clone()
    bad[:, -1] = lg.argmin(-1).to(bad.dtype)
    for chosen, wrong in ((ids, False), (bad, True)):
        rc.mark_step(1, rows=16)
        (e_st, e_cm), (r_st, r_cm), graph = eager_and_replayed(lambda: fn_check(lg, w, chosen, *a), st, cm)
        assert torch.equal(e_st, r_st), (e_st, r_st)
        assert int(r_st[slot, sr.C_CALLS]) == 1 and (int(r_st[slot, sr.C_SET]) > 0) == wrong
        assert int(e_cm[slot, 0]) == int(r_cm[slot, 0]) == int(wrong)
    rc.mark_step(2, rows=0)                   # a step with no real rows: nothing judged at replay
    st.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(st[slot, sr.C_SET]) == 0 and int(st[slot, sr.C_ROWS]) == 0


# ---------------------------------------------------------------------------------------------------------- norms
@pytest.mark.parametrize("path", ["torch", "triton"])
def test_the_norm_check_in_a_replayed_graph(path):
    h = 896
    mod = torch.nn.Module()
    mod.weight = torch.nn.Parameter((1 + 0.3 * torch.randn(h, device=DEV)).to(torch.bfloat16), requires_grad=False)
    mod.variance_epsilon = 1e-6
    nc = sn.NormChecks(None)
    assert nc.add(mod, "model.layers.0.input_layernorm", "rmsnorm") == 0
    nc.finalize()
    if path == "triton":
        ok, why = sn.STATUS.get((sn._dev_key(DEV), torch.bfloat16), (False, "not verified"))
        assert ok, why
    fn_check = sn.norm_check_torch if path == "torch" else sn.norm_check_triton
    x = torch.randn(9, h, device=DEV).to(torch.bfloat16)
    s = x.float()
    y = ((s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16) * mod.weight).to(torch.bfloat16)
    a = mod._sdcguard_norm
    st, slot, cm = a[2], a[3], a[5]
    for wrong in (False, True):
        yy = y.clone()
        if wrong:
            yy[4, 100] += 5.0
        nc.mark_step(3)
        (e_st, e_cm), (r_st, r_cm), graph = eager_and_replayed(lambda: fn_check(x, yy, *a), st, cm)
        assert torch.equal(e_st, r_st), (e_st, r_st)
        z = sn._f32(int(r_st[slot, 0]) >> 32)
        assert int(r_st[slot, sn.C_CALLS]) == 1 and (z > nc.line) == wrong, z
        assert int(e_cm[slot, 0]) == int(r_cm[slot, 0]) == int(wrong)
    nc.mark_step(7)                               # read at replay, not frozen at capture
    st.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert int(st[slot, sn.C_STEP]) & 0xFFFFFFFF == 7


# ----------------------------------------------------------------------------------------------------- linear check
@pytest.mark.parametrize("path", ["torch", "triton"])
def test_the_linear_check_in_a_replayed_graph(path):
    m, k, n = 16, 512, 300
    W = (torch.randn(n, k, device=DEV) / k ** 0.5).to(torch.bfloat16)
    x = torch.randn(m, k, device=DEV).to(torch.bfloat16)
    mm = torch.backends.cuda.matmul
    saved = mm.allow_bf16_reduced_precision_reduction
    mm.allow_bf16_reduced_precision_reduction = False        # fp32 partial sums, as ServeGuard serves with
    try:
        y = x @ W.t()
    finally:
        mm.allow_bf16_reduced_precision_reduction = saved
    cs = LayerChecksums(W, None, 128)
    if path == "triton":
        assert _serve_fast.verified(DEV, torch.bfloat16), _serve_fast.WHY
    st = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=DEV)
    for wrong in (False, True):
        yy = y.clone()
        if wrong:
            yy[m // 2, n // 3] += 30.0
        if path == "torch":
            fn = lambda: check_static_impl(x, yy, *cs.tensors(True), st, 0, 6.0, 1e-20)  # noqa: E731
        else:
            fn = lambda: _serve_fast.check(x, yy, *cs.tensors(True), st, 0, 6.0, 1e-20)  # noqa: E731
        eager, replayed, _ = eager_and_replayed(fn, st)
        close_counts(eager, replayed, slice(2, None), lambda t: _serve_fast.decode(*t[0].tolist())[0], rel=2e-2)
        z = _serve_fast.decode(*replayed[0].tolist())[0]
        assert (z > 6.0) == wrong, z


# ------------------------------------------------------------------------------------------------------ attention
@pytest.mark.parametrize("path", ["torch", "triton"])
def test_the_attention_check_in_a_replayed_graph(path):
    """As vLLM's FULL CUDA graphs run it: captured once, replayed with the next steps' batches written into the same
    tensors (queries, outputs, cache, query_start_loc, seq_lens, block table, slots). A replay leaves the state an
    eager call on that batch leaves, a wrong output is found, and inactive steps and other layers' turns count
    nothing."""
    if path == "triton":
        ok, why = sa.verify_kernel(DEV, torch.bfloat16)
        assert ok, why
    fn = sa.attn_check_torch if path == "torch" else sa.attn_check_triton
    lens = ((300, 1), (40, 1), (64, 12), (0, 0))
    c = sa.make_case(DEV, torch.bfloat16, D=128, Hq=8, Hkv=2, lens=lens, NB=40, padded_rows=4, seed=1)
    b = sa.buffers(DEV, 2, 128, 512, torch.bfloat16)
    st, cm = b["state"], b["capm"]
    counts = [sa.C_CALLS, sa.C_SAMPLES, sa.C_SKIP, sa.C_UNJ, sa.C_BADBLK, sa.C_KEYS, sa.C_WN, sa.C_WBAD, sa.C_WSLOT]
    z_of = lambda t: sa._f32(int(t[1, sa.C_ZT]) >> 32)  # noqa: E731
    o = c["o"].clone()

    def call():
        args = (c["q"], o, c["kv"], c["key"], c["val"], c["qsl"], c["seq_lens"], c["bt"], c["slots"], st, 1,
                b["ctrl"], 2, 1, c["scale"], -1, sa.ATTN_Z, cm, b["capq"], b["capo"])
        if path == "torch":
            fn(*args, True)
        else:
            fn(*args, b["capkv"], b["capkvm"], b["ws"], True, True)

    b["ctrl"][0], b["ctrl"][1] = 3, 1                   # step 3: the turn of the layer with index 1 of 2
    for wrong in (False, True):
        o.copy_(c["o"])
        if wrong:
            o[:, :, 3] += 1.0                           # every output wrong at dimension 3
        (e_st, e_cm), (r_st, r_cm), graph = eager_and_replayed(call, st, cm)
        close_counts(e_st, r_st, counts, z_of)
        assert int(r_st[1, sa.C_CALLS]) == 1 and int(r_st[1, sa.C_SAMPLES]) == sa.SAMPLES
        assert (z_of(r_st) > sa.ATTN_Z) == wrong and int(r_cm[1, 0]) == int(wrong), z_of(r_st)
        assert int(r_st[1, sa.C_WN]) == sa.SAMPLES and int(r_st[1, sa.C_WBAD]) == 0 and r_st[0].abs().sum() == 0
    # the next batch in the same tensors: other lengths, cache, queries, outputs and slots, read at replay
    nxt = sa.make_case(DEV, torch.bfloat16, D=128, Hq=8, Hkv=2, lens=((301, 1), (41, 1), (66, 12), (0, 0)), NB=40,
                       padded_rows=4, seed=2)
    for k in ("q", "o", "kv", "key", "val", "qsl", "seq_lens", "bt", "slots"):
        assert c[k].shape == nxt[k].shape, k
        c[k].copy_(nxt[k])
    o.copy_(nxt["o"])
    b["ctrl"][0] = 5
    st.zero_()
    cm.zero_()
    graph.replay()
    torch.cuda.synchronize()
    replayed = st.clone()
    st.zero_()
    call()
    torch.cuda.synchronize()
    close_counts(st.clone(), replayed, counts, z_of)
    assert z_of(replayed) <= 1.0 and int(replayed[1, sa.C_KEYS]) == int(st[1, sa.C_KEYS])
    for step, active in ((4, 1), (5, 0)):               # another layer's turn; outside a model step
        b["ctrl"][0], b["ctrl"][1] = step, active
        st.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert int(st.abs().sum()) == 0
