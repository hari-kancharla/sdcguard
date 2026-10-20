"""The coverage and numerics reviews of Oct 19, 2026, each finding turned into a test built from its
reproduction: a checked model computes exactly what it would unchecked (copies, replicas, weight updates in
place, keyword calls, pickling), every check that ran is counted and every alert is kept (exceptions, failing
callbacks), layers whose forward never runs are listed instead of reported as checked, and the checks handle
empty matmuls, wide blocks, cancelling huge outputs and infinities."""
import copy
import io
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import SDCGuard, guard as gmod
from sdcguard import checksum as cs
from sdcguard.serve import ServeGuard


def _mlp(dt=None):
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
    return m.to(dt) if dt is not None else m


def _zero(m):
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m


# --- copies and replicas compute with their own weights (H1) ----------------------------------------------

def test_a_deep_copy_of_a_protected_model_uses_its_own_weights():
    m = _mlp()
    g = SDCGuard().protect(m)
    c = _zero(copy.deepcopy(m))
    x = torch.randn(4, 64)
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        out = c(x)
    assert float(out.abs().sum()) == 0.0 and type(c[0]) is nn.Linear and type(m[0]) is not nn.Linear


def test_an_averaged_model_of_a_protected_model_is_its_own_model():
    from torch.optim.swa_utils import AveragedModel
    m = _mlp()
    SDCGuard().protect(m)
    ema = AveragedModel(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    x = torch.randn(4, 64)
    assert not torch.equal(ema(x), m(x))


def test_a_data_parallel_style_replica_runs_its_own_weights():
    m = _mlp()
    g = SDCGuard().protect(m)
    rep = m[0]._replicate_for_data_parallel()            # what nn.DataParallel does per device
    for key, p in m[0]._parameters.items():              # (replicate() sets them as plain attributes)
        setattr(rep, key, torch.zeros_like(p))
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        out = rep(torch.randn(4, 64))
    assert float(out.abs().sum()) == 0.0 and g.stats["checked"] == 1


def test_a_deep_copy_of_an_attached_serving_model_uses_its_own_weights():
    m = _mlp(torch.bfloat16)
    sg = ServeGuard().attach(m)
    c = _zero(copy.deepcopy(m))
    with torch.no_grad():
        assert float(c(torch.randn(4, 64).bfloat16()).abs().sum()) == 0.0
    assert sg.poll(wait=True) == []


def test_protected_and_attached_models_pickle_as_plain_models_and_take_keywords():
    m = _mlp()
    g = SDCGuard().protect(m)
    buf = io.BytesIO()
    torch.save(m, buf)
    buf.seek(0)
    loaded = torch.load(buf, weights_only=False)
    assert type(loaded[0]) is nn.Linear and not [k for k in loaded[0].__dict__ if k.startswith("_sdcguard")]
    x = torch.randn(4, 64)
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        m[0](input=x)
    s = _mlp(torch.bfloat16)
    ServeGuard().attach(s)
    with torch.no_grad():
        s[0](input=x.bfloat16())
    g.unprotect()
    assert type(m[0]) is nn.Linear and repr(m) == repr(_mlp())


# --- exceptions and callbacks don't lose checks (H2, M5) ----------------------------------------------------

@pytest.mark.parametrize("mode", ["protect", "dispatch"])
def test_a_step_that_raises_keeps_its_alerts(mode):
    m = _mlp()
    g = SDCGuard()
    if mode == "protect":
        g.protect(m)
    real = F.linear if mode == "protect" else None

    class Corrupt(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func._schema.name in ("aten::addmm", "aten::mm") and out.dtype == torch.bfloat16:
                out.view(-1)[3] += 1e4
            return out

    def bad_linear(*a, **k):
        y = real(*a, **k)
        y.view(-1)[3] += 1e4
        return y

    with pytest.raises(ValueError) as info:
        if mode == "protect":
            F.linear = bad_linear
            try:
                with g.at_step(5), torch.autocast("cpu", dtype=torch.bfloat16):
                    m(torch.randn(4, 64))
                    raise ValueError("non-finite gradient norm")
            finally:
                F.linear = real
        else:
            with Corrupt(), g.at_step(5), torch.autocast("cpu", dtype=torch.bfloat16):
                m(torch.randn(4, 64))
                raise ValueError("non-finite gradient norm")
    assert g.stats["alerts"] >= 1 and len(g.log.events) >= 1 and g.log.events[0].step == 5
    assert any("sdcguard" in n for n in getattr(info.value, "__notes__", []))
    assert not g._pending and g.stats["checked"] >= 1


def test_an_alert_callback_that_raises_doesnt_lose_later_alerts():
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
    seen = []

    def page(ev):
        seen.append(ev)
        raise ConnectionError("pager down")

    g = SDCGuard(on_alert=page).protect(layers)
    real = F.linear

    def bad(*a, **k):
        y = real(*a, **k)
        y.view(-1)[3] += 1e4
        return y

    F.linear = bad
    try:
        with pytest.raises(ConnectionError):
            with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
                for lin in layers:
                    lin(torch.randn(4, 64))
    finally:
        F.linear = real
    assert g.stats["alerts"] == 3 and len(seen) == 3


def test_a_serving_callback_that_raises_doesnt_lose_other_layers_alerts():
    torch.manual_seed(0)
    a, b = nn.Linear(128, 128).bfloat16(), nn.Linear(128, 128).bfloat16()
    seen = []

    def page(ev):
        seen.append(ev.detail["where"])
        raise ConnectionError("pager down")

    sg = ServeGuard(on_alert=page).attach(nn.ModuleDict({"a": a, "b": b}))
    x = torch.randn(4, 128).bfloat16()
    with torch.no_grad():
        a.weight.view(torch.int16)[1, 1] ^= 1 << 14          # both layers' weights corrupted after attach
        b.weight.view(torch.int16)[1, 1] ^= 1 << 14
        a(x)
        b(x)
    with pytest.raises(ConnectionError):
        sg.poll(wait=True)
    assert sg.stats["alerts"] == 2 and sorted(seen) == ["a", "b"]


# --- the step computes what it would unchecked (H3, M4) ----------------------------------------------------

def _awp_step(m, guard=None):
    """Weight perturbation (AWP): a forward, weights changed through .data, a second forward in its own
    autocast region, then the weights restored."""
    x = torch.randn(8, 64, generator=torch.Generator().manual_seed(1))
    ctx = guard.at_step(0) if guard is not None else torch.autograd.set_grad_enabled(True)
    with ctx:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = m(x).float().pow(2).mean()
        loss.backward()
        backup = [p.data.clone() for p in m.parameters()]
        for p in m.parameters():
            p.data.add_(0.5 * p.grad.sign())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            adv = m(x).float().pow(2).mean()
        adv.backward()
        for p, b in zip(m.parameters(), backup):
            p.data = b
    return float(loss), float(adv), [p.grad.clone() for p in m.parameters()]


def test_weights_changed_in_place_between_autocast_regions_are_cast_again():
    plain = _awp_step(_mlp())
    m = _mlp()
    checked = _awp_step(m, SDCGuard().protect(m))
    assert plain[:2] == checked[:2]
    assert all(torch.equal(a, b) for a, b in zip(plain[2], checked[2]))


def test_cache_disabled_autocast_stays_bit_identical():
    def step(m, g=None):
        x = torch.randn(8, 64, generator=torch.Generator().manual_seed(2))
        lin = m[0]
        with (g.at_step(0) if g else torch.autograd.set_grad_enabled(True)):
            with torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
                loss = (lin(lin(x)[:, :64]) if lin.out_features >= 64 else lin(x)).float().pow(2).mean()
            loss.backward()
        return [p.grad.clone() for p in m.parameters() if p.grad is not None]

    a = step(_mlp())
    m = _mlp()
    b = step(m, SDCGuard().protect(m))
    assert all(torch.equal(x, y) for x, y in zip(a, b))


# --- layers whose forward never runs are listed (H4) --------------------------------------------------------

def test_multihead_attention_out_proj_is_listed_not_protected():
    mha = nn.MultiheadAttention(64, 4, batch_first=True)
    g = SDCGuard().protect(nn.ModuleDict({"attn": mha}))
    assert "attn.out_proj" in g.skipped and not g._protected


def test_serving_lists_the_linears_the_fused_transformer_path_skips():
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, batch_first=True), 2).bfloat16().eval()
    sg = ServeGuard().attach(enc)
    assert sg.stats["layers"] == 0 and len(sg.unsupported) == 6
    assert all("never runs" in w or "fused inference path" in w for w in sg.unsupported.values())


# --- more matmul ops in the default mode (M1) ---------------------------------------------------------------

@pytest.mark.parametrize("op", ["baddbmm", "addbmm", "mv", "addmv"])
def test_more_matmul_ops_are_checked_and_counted(op):
    dt = torch.bfloat16
    b, m, k, n = 2, 16, 64, 128
    calls = {"baddbmm": lambda: torch.baddbmm(torch.randn(b, m, n, dtype=dt), torch.randn(b, m, k, dtype=dt),
                                              torch.randn(b, k, n, dtype=dt), beta=0.5, alpha=2.0),
             "addbmm": lambda: torch.addbmm(torch.randn(m, n, dtype=dt), torch.randn(b, m, k, dtype=dt),
                                            torch.randn(b, k, n, dtype=dt)),
             "mv": lambda: torch.mv(torch.randn(m, k, dtype=dt), torch.randn(k, dtype=dt)),
             "addmv": lambda: torch.addmv(torch.randn(m, dtype=dt), torch.randn(m, k, dtype=dt),
                                          torch.randn(k, dtype=dt))}
    g = SDCGuard()
    with g:
        calls[op]()
    assert g.stats["checked"] == 1 and g.stats["alerts"] == 0, (g.stats, g.ops_seen)


def test_a_corrupted_baddbmm_alerts_and_coverage_counts_it():
    dt = torch.bfloat16

    class Corrupt(torch.utils._python_dispatch.TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func._schema.name == "aten::baddbmm":
                out.view(-1)[5] += 1e3
            return out

    g = SDCGuard()
    with Corrupt(), g:
        torch.baddbmm(torch.randn(2, 16, 128, dtype=dt), torch.randn(2, 16, 64, dtype=dt),
                      torch.randn(2, 64, 128, dtype=dt))
    assert g.stats["alerts"] == 1
    lin = nn.Linear(64, 64)
    g2 = SDCGuard().protect(lin)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cov = g2.coverage(lambda: (lin(torch.randn(16, 64)), torch.baddbmm(torch.randn(16, 64, 64), torch.randn(16, 64, 8),
                                                                              torch.randn(16, 8, 64))))
    assert 0 < cov["share_checked_in_fast_mode"] < 1


# --- process-wide settings with overlapping guards (M2) --------------------------------------------------

def test_overlapping_guards_restore_the_cublas_settings_once(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m = torch.backends.cuda.matmul
    saved = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    try:
        m.allow_bf16_reduced_precision_reduction = m.allow_fp16_reduced_precision_reduction = True
        a, b = ServeGuard().attach(_mlp(torch.bfloat16)), ServeGuard().attach(_mlp(torch.bfloat16))
        a.detach()
        assert not m.allow_bf16_reduced_precision_reduction           # b is still attached
        g = SDCGuard()
        with g.at_step(0):
            pass
        assert not m.allow_bf16_reduced_precision_reduction           # a step ending doesn't undo b
        b.detach()
        assert m.allow_bf16_reduced_precision_reduction and m.allow_fp16_reduced_precision_reduction
    finally:
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved


def test_a_serving_guard_dropped_without_detach_lets_go_of_the_cublas_settings(monkeypatch):
    # On a GPU every ServeGuard holds the settings while attached. One that's simply dropped (with its model)
    # used to keep them changed for the rest of the process: the GPU test run on Oct 19 failed this way.
    import gc
    from sdcguard import guard as guard_mod
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m = torch.backends.cuda.matmul
    saved = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    try:
        gc.collect()
        holds = guard_mod._FLAGS["holds"]
        m.allow_bf16_reduced_precision_reduction = True
        model = _mlp(torch.bfloat16)
        g = ServeGuard().attach(model)
        assert not m.allow_bf16_reduced_precision_reduction and guard_mod._FLAGS["holds"] == holds + 1
        del g, model
        gc.collect()
        assert guard_mod._FLAGS["holds"] == holds
        if holds == 0:
            assert m.allow_bf16_reduced_precision_reduction
    finally:
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved


# --- serving lifecycle (M6, L2-L5) ------------------------------------------------------------------------

def test_a_device_hook_added_while_attached_survives_detach():
    lin = nn.Linear(64, 64).bfloat16()
    sg = ServeGuard().attach(nn.ModuleDict({"l": lin}))
    calls = []
    old = lin.forward

    def hooked(*a, **k):
        calls.append(1)
        return lin._old_forward(*a, **k)

    lin._old_forward, lin._hf_hook = old, object()
    lin.forward = hooked
    x = torch.randn(4, 64).bfloat16()
    with torch.no_grad():
        lin(x)
    sg.detach()
    with torch.no_grad():
        y = lin(x)
    assert calls == [1, 1] and torch.equal(y, F.linear(x, lin.weight, lin.bias)) and type(lin) is nn.Linear


def test_a_layer_cast_or_reshaped_after_attach_is_unchecked_until_refresh():
    m = _mlp(torch.bfloat16)
    sg = ServeGuard().attach(m)
    m.float()
    with torch.no_grad():
        m(torch.randn(4, 64))                                     # used to raise KeyError
    assert sg.poll(wait=True) == [] and set(sg.verify_weights()) == {"0", "2"}
    sg.refresh()
    assert set(sg.unsupported) == {"0", "2"}                       # listed while they're fp32
    with torch.no_grad():
        m(torch.randn(4, 64))
    m.bfloat16()
    with torch.no_grad():
        m[0].weight.data = torch.randn(80, 64).bfloat16()          # a new shape
        m[0].bias.data = torch.randn(80).bfloat16()
    sg.refresh()
    assert sg.unsupported == {}
    with torch.no_grad():
        y = m[0](torch.randn(4, 64).bfloat16())
    assert y.shape == (4, 80) and sg.poll(wait=True) == [] and sg.worst_seen["0"] > 0


def test_a_second_serving_guard_lists_the_layers_the_first_has():
    m = _mlp(torch.bfloat16)
    ServeGuard().attach(m)
    b = ServeGuard().attach(m)
    assert b.stats["layers"] == 0 and set(b.unsupported) == {"0", "2"}


def test_attach_that_fails_partway_leaves_nothing_behind(monkeypatch):
    m = _mlp(torch.bfloat16)
    sg = ServeGuard()
    real = sg.add_layer
    n = {"c": 0}

    def fail_second(*a, **k):
        n["c"] += 1
        if n["c"] == 2:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(sg, "add_layer", fail_second)
    with pytest.raises(RuntimeError):
        sg.attach(m)
    assert type(m[0]) is nn.Linear and not sg.layers and sg._restore is None


def test_nested_steps_check_once_and_keep_the_outer_step():
    for protect in (False, True):
        m = _mlp()
        g = SDCGuard()
        if protect:
            g.protect(m)
        with g.at_step(100):
            with g.at_step(7):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    m(torch.randn(4, 64)).sum().backward()
        assert g.step == 100
        single = SDCGuard()
        m2 = _mlp()
        if protect:
            single.protect(m2)
        with single.at_step(1), torch.autocast("cpu", dtype=torch.bfloat16):
            m2(torch.randn(4, 64)).sum().backward()
        assert g.stats["checked"] == single.stats["checked"]


# --- numerics --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m,k,n", [(0, 64, 256), (8, 64, 0), (8, 0, 256)])
def test_empty_matmuls_check_cleanly(m, k, n):
    A, B = torch.randn(m, k).bfloat16(), torch.randn(k, n).bfloat16()
    C = (A.float() @ B.float()).bfloat16()
    assert cs.check_matmul(A, B, C).ok
    assert cs.check_matmul_acc(A, B, A.float() @ B.float()).ok
    if k == 0 and m and n:
        bad = C.clone()
        bad[0, 0] = 1.0
        assert not cs.check_matmul(A, B, bad).ok                 # k = 0: the output must be all zeros


@pytest.mark.parametrize("precise", [False, True])
def test_an_moe_expert_with_no_tokens_doesnt_break_the_step(precise):
    expert = nn.Linear(64, 128)
    g = SDCGuard(precise=precise).protect(expert)
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        expert(torch.randn(0, 64, requires_grad=True)).sum().backward()
    assert g.stats["alerts"] == 0 and g.stats.get("empty", 0) >= 1
    d = SDCGuard()
    with d, torch.autocast("cpu", dtype=torch.bfloat16):
        nn.Linear(64, 128)(torch.randn(0, 64, requires_grad=True)).sum().backward()
    assert d.stats["alerts"] == 0


def test_serving_blocks_wider_than_128_use_the_pytorch_check():
    from sdcguard import _serve_fast
    from sdcguard.serve import LayerChecksums
    m = nn.Sequential(nn.Linear(32, 300)).bfloat16()
    sg = ServeGuard(block=256).attach(m)
    assert not sg.uses_fast(0)
    with torch.no_grad():
        m(torch.randn(4, 32).bfloat16())
    assert sg.poll(wait=True) == [] and sg.worst_seen["0"] > 0
    lc = LayerChecksums(m[0].weight, None, 256)
    with pytest.raises(ValueError):
        _serve_fast.check(torch.randn(4, 32).bfloat16(), m[0](torch.randn(4, 32).bfloat16()).detach(),
                          *lc.tensors(False), torch.zeros(1, 2, dtype=torch.int64), 0, 6.0, 1e-20)


def test_huge_outputs_that_cancel_in_both_checksums_are_caught():
    torch.manual_seed(0)
    A, B = torch.randn(8, 64).bfloat16(), torch.randn(64, 256).bfloat16()
    s0 = cs._signs(128, "cpu", 0)
    j = next(j for j in range(1, 128) if s0[j] == s0[0])
    for big in (1e18, 1e6):
        C = (A.float() @ B.float()).bfloat16()
        C[2, 0], C[2, j] = big, -big
        r = cs.check_matmul(A, B, C)
        assert not r.ok and r.row == 2 and r.col_block == 0
        C32 = A.float() @ B.float()
        C32[2, 0], C32[2, j] = big, -big
        assert not cs.check_matmul_acc(A, B, C32).ok
        lin = nn.Linear(64, 256, bias=False).bfloat16()
        lin.weight.data = B.t().contiguous()
        sg = ServeGuard().attach(nn.ModuleDict({"l": lin}))
        y = (A.float() @ B.float()).bfloat16()
        y[2, 0], y[2, j] = big, -big
        sg.check(0, A, y)
        assert sg.poll(wait=True) and math.isinf(sg.worst_seen["l"])
        sg.detach()


def test_outputs_at_the_input_bound_are_not_called_impossible():
    """The rule allows the accumulation's worst-case rounding: an output at exactly the bound is legitimate."""
    for k in (64, 4096, 65536):
        A = torch.ones(4, k).bfloat16()
        B = torch.ones(k, 128).bfloat16()
        C = (A.float() @ B.float()).bfloat16()
        assert cs.check_matmul(A, B, C).ok, k


def test_serving_inputs_that_make_at_linear_round_twice_dont_alarm():
    """at::linear rounds the matmul, then adds the bias (rounded again) for non-contiguous 3-D inputs; with a
    bias that cancels most of the output the single-rounding model alarmed (worst z 19)."""
    worst = 0.0
    for seed in range(10):
        torch.manual_seed(seed)
        lin = nn.Linear(256, 256).bfloat16()
        x = torch.randn(4, 6, 256).bfloat16()
        with torch.no_grad():
            lin.bias.copy_((-0.99 * (x[:, -2:, :].float() @ lin.weight.float().t()).mean((0, 1))).bfloat16())
        sg = ServeGuard().attach(nn.ModuleDict({"l": lin}))
        with torch.no_grad():
            y = lin(x[:, -2:, :])
            assert torch.equal(y, F.linear(x[:, -2:, :], lin.weight, lin.bias))      # same values as unchecked
        assert sg.poll(wait=True) == []
        worst = max(worst, sg.worst_seen["l"])
        sg.detach()
    assert worst < 6


def test_the_canary_doesnt_call_an_infinity_close():
    from sdcguard.canary import Canary
    c = Canary.__new__(Canary)
    c.tol = 1e-3
    a, b = torch.tensor([0.41, 6.96], dtype=torch.float64), torch.tensor([float("inf"), float("inf")],
                                                                         dtype=torch.float64)
    assert not c._agree(a, b) and c._agree(b, b.clone()) and c._agree(a, a + 1e-6)


def test_replica_offsets_split_the_layers_between_them():
    """With share 1/R, offsets 0 to R-1 check every layer exactly once between them (offsets used to shift the
    layer number, which left about 10% of the layers unchecked on every replica at share 0.125)."""
    from sdcguard.serve import select_layer
    for r in (2, 4, 8):
        for n in (7, 112, 196, 1000):
            assert all(sum(select_layer(i, 1 / r, o) for o in range(r)) == 1 for i in range(n)), (r, n)
