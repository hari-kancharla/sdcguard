"""Attention checks (attention.py): clean fused attention passes, corrupted dV and outputs are caught."""
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sdcguard import SDCGuard
from sdcguard import attention as att
from sdcguard.checksum import resolve


def _sdpa(dt=torch.bfloat16, B=2, H=4, KV=4, T=128, D=32, scale=1.0, causal=True, seed=0, exact=False):
    """dO, dV, O and V of one attention call in dt. exact: computed in float64 from the same dt inputs, then rounded
    to dt once (what an fp32-accumulating fused kernel gives, to an fp32 rounding)."""
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(B, H, T, D, generator=g) * scale).to(dt)
    k = (torch.randn(B, KV, T, D, generator=g) * scale).to(dt)
    v = torch.randn(B, KV, T, D, generator=g).to(dt)
    ct = torch.float64 if exact else dt
    q, k, v = (t.to(ct).requires_grad_() for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=H != KV)
    do = (torch.randn(o.shape, generator=g) * 1e-3).to(dt)
    o.backward(do.to(ct))
    return do, v.grad.to(dt), o.detach().to(dt), v.detach().to(dt)


def _accurate(dt, dv, **kw) -> bool:
    """This PyTorch's own attention backward gives dV within a few roundings of float64 here. PyTorch 2.8's fp16
    attention on aarch64 CPUs doesn't (2.6% of the largest dV off, about 50 roundings, on October 21, 2026; 2.14.1:
    about one), and the check rightly flags that."""
    ref = _sdpa(dt, exact=True, **kw)[1].double()
    u = 2.0 ** -8 if dt == torch.bfloat16 else 2.0 ** -11
    return float((dv.double() - ref).abs().max()) <= 4 * u * float(ref.abs().max())


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [0.02, 1.0, 5.0])          # near-uniform, typical, near one-hot attention
@pytest.mark.parametrize("kv", [4, 1])
def test_clean_attention_scores_low(dt, scale, kv):
    """Clean results score low: computed exactly and rounded once, and from this PyTorch's own kernels where
    they are accurate (see _accurate)."""
    kw = dict(KV=kv, scale=scale, causal=scale != 1.0)
    runs = [_sdpa(dt, exact=True, **kw)]
    own = _sdpa(dt, **kw)
    if _accurate(dt, own[1], **kw):
        runs.append(own)
    for do, dv, o, v in runs:
        (b,) = resolve([att.backward_stat(do, dv)])
        (f,) = resolve([att.forward_stat(o, v)])
        assert b.ok and 0 <= b.max_z < 3 and f.ok and f.max_z == 0


def test_near_uniform_attention_needs_the_worst_case_bound():
    # every dV of a column is nearly the same number and rounds the same way: the random-walk estimate alone
    # scored 9.1 here on the CPU (a false alarm); the worst-case bound keeps it quiet
    do, dv, _, _ = _sdpa(scale=0.02, causal=False, T=256, D=64)
    (b,) = resolve([att.backward_stat(do, dv)])
    assert b.ok


def test_a_corrupted_dv_value_is_caught_and_located():
    do, dv, _, _ = _sdpa(B=2, H=4, KV=2, T=128, D=32)
    bad = dv.clone()
    bad[1, 1, 40, 7] += 100 * dv.float().pow(2).mean().sqrt().to(dv.dtype)
    (b,) = resolve([att.backward_stat(do, bad)])
    assert not b.ok and b.row == 1 * 2 + 1 and b.col_block == 7          # (batch 1, key/value head 1), feature 7


def test_an_output_outside_the_values_range_is_impossible():
    _, _, o, v = _sdpa()
    bad = o.clone()
    bad[0, 2, 9, 3] = v[0, 2].float().abs().max().to(o.dtype) * 4
    (f,) = resolve([att.forward_stat(bad, v)])
    assert not f.ok and f.max_z == float("inf") and f.row == (0 * 4 + 2) * 128 + 9 and f.col_block == 3


def test_inf_and_nan():
    do, dv, o, v = _sdpa()
    bad = dv.clone()
    bad[0, 0, 0, 0] = float("inf")
    assert not resolve([att.backward_stat(do, bad)])[0].ok                     # dO finite: impossible
    do_bad = do.clone()
    do_bad[0, 0, 0, 0] = float("nan")
    r = resolve([att.backward_stat(do_bad, bad)])[0]
    assert r.ok and not r.judged      # one column can't be judged (dO isn't finite): the check is incomplete
    worse = bad.clone()
    worse[1, 2, 5, 9] += 100 * dv.float().pow(2).mean().sqrt().to(dv.dtype)
    assert not resolve([att.backward_stat(do_bad, worse)])[0].ok      # corruption elsewhere still shows
    r = resolve([att.backward_stat(torch.full_like(do, float("nan")), bad)])[0]
    assert r.ok and not r.judged
    o_bad = o.clone()
    o_bad[0, 0, 0, 0] = float("nan")
    assert not resolve([att.forward_stat(o_bad, v)])[0].ok


def test_other_dtypes_and_layouts_are_not_judged():
    do, dv, o, v = _sdpa()
    assert att.backward_stat(do.float(), dv.float()) is None
    assert att.backward_stat(do[..., 0], dv[..., 0]) is None
    assert att.forward_stat(o, v[:, :3]) is None                                # 4 heads over 3: not a grouping


def test_unsupported_calls_are_named():
    flash = "aten::_scaled_dot_product_flash_attention_backward"
    args = [None] * 14
    assert att.why_not(flash, args, {}) is None
    args[10] = 0.1
    assert att.why_not(flash, args, {}) == "dropout"
    args[10] = 0.0
    args[6] = torch.tensor([0, 5, 9])
    assert att.why_not(flash, args, {}) == "variable-length batch"
    eff = "aten::_scaled_dot_product_efficient_attention"
    assert att.why_not(eff, [None, None, None, torch.zeros(2, 2), True, 0.0], {}) == "mask or bias tensor"
    cpu = "aten::_scaled_dot_product_flash_attention_for_cpu"
    assert att.why_not(cpu, [None] * 5, {"attn_mask": torch.zeros(2, 2)}) == "mask or bias tensor"
    assert att.why_not("aten::mm", [], {}) == "not an attention op"


def _model_step(guard, inject=None, kv=2):
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 64 + 2 * 16 * kv).to(torch.bfloat16)
    x = torch.randn(2, 96, 64).to(torch.bfloat16)

    def run():
        y = lin(x)
        q = y[..., :64].reshape(2, 96, 4, 16).transpose(1, 2)
        k, v = y[..., 64:].reshape(2, 96, 2, kv, 16).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=kv != 4)
        o.float().pow(2).mean().backward()

    if inject is None:
        with guard:
            run()
    else:
        with inject, guard:          # the injector sits below the guard: the guard sees what it wrote
            run()


class _Corrupt(TorchDispatchMode):
    def __init__(self, backward):
        super().__init__()
        self.backward = backward

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func._schema.name
        if self.backward and name in att.BACKWARD_OPS:
            dv = out[2]
            dv[0, 1, 50, 5] += 200 * dv.float().pow(2).mean().sqrt().to(dv.dtype)
        if not self.backward and name in att.FORWARD_OPS:
            out[0][1, 3, 20, 2] = out[0][1, 3, 20, 2] * 4096 + 16
        return out


def _runs_fused_attention() -> bool:
    """Whether this PyTorch runs the test model's causal GQA attention as one fused op on the CPU (2.8 runs it
    as plain matmuls, which the guard checks as matmuls instead)."""
    from torch.utils._python_dispatch import TorchDispatchMode
    seen = []

    class Names(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(func._schema.name)
            return func(*args, **(kwargs or {}))
    with Names():
        _model_step(SDCGuard(attention=False))
    return any(n in att.FORWARD_OPS for n in seen)


def test_the_guard_checks_fused_attention_only_when_asked():
    g = SDCGuard()
    _model_step(g)
    assert g.stats["attention_checked"] == 0
    g = SDCGuard(attention=True)
    _model_step(g)
    if not _runs_fused_attention():
        pytest.skip("this PyTorch doesn't run this attention as a fused op on the CPU")
    assert g.stats["attention_checked"] == 2 and g.stats["alerts"] == 0


@pytest.mark.parametrize("backward", [True, False])
def test_the_guard_alerts_on_corrupted_attention(backward):
    if not _runs_fused_attention():
        pytest.skip("this PyTorch doesn't run this attention as a fused op on the CPU")
    g = SDCGuard(attention=True)
    _model_step(g, inject=_Corrupt(backward))
    kinds = [e.detail["kind"] for e in g.log.events]
    assert kinds == ["attention_backward" if backward else "attention_forward"]
    ev = g.log.events[0].detail
    assert ev["op"] in (att.BACKWARD_OPS if backward else att.FORWARD_OPS) and ev["check"].startswith("attention")


def test_the_calibration_experiment_runs_on_the_cpu():
    if not _runs_fused_attention():
        pytest.skip("this PyTorch doesn't run GQA attention as a fused op on the CPU")
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
    import attention_calib
    res = attention_calib.calibrate(torch.device("cpu"), say=lambda s: None)
    (run,) = res["runs"]
    assert res["criteria_met"] and run["clean"]["checks"] > 0 and run["clean"]["skipped"] == {}
    assert run["injected"]["dV + 256 x rms"]["alarms"] == run["injected"]["dV + 256 x rms"]["corrupted_calls"] > 0
    assert "| tiny, GQA 4:2 | cpu flash |" in attention_calib.markdown(res)


@pytest.mark.parametrize("bad", [7e4, float("nan")])          # 7e4 overflows fp16 to inf
def test_attention_check_does_not_blame_the_gpu_for_a_non_finite_key(bad):
    """A key (or query) that isn't finite breaks the identities' premise (every row of P sums to 1), whatever
    the GPU computes: those heads are unjudged, never an alert. The backward check used to report max_z 3.4e38."""
    torch.manual_seed(0)
    dt = torch.float16
    q = torch.randn(1, 2, 16, 8).to(dt).requires_grad_()
    k = torch.randn(1, 2, 16, 8)
    k[0, 0, 3, 0] = bad
    k = k.to(dt).requires_grad_()
    v = torch.randn(1, 2, 16, 8).to(dt).requires_grad_()
    do = torch.randn(1, 2, 16, 8).to(dt)
    alerts = []
    g = SDCGuard(attention=True, on_alert=alerts.append)
    with g:
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o.backward(do)
    assert g.stats["attention_checked"] == 2
    assert alerts == [], [(a.detail["kind"], a.detail["max_z"]) for a in alerts]
    assert g.stats["unjudged"] == 2                    # incomplete, not clean either


def test_non_finite_queries_leave_only_their_heads_unjudged():
    """Head 0's queries hold a NaN: head 0 can't be judged, but a wrong dV in head 1 is still caught."""
    do, dv, o, v = _sdpa(B=1, H=2, KV=2)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 2, 128, 32, generator=g).bfloat16()
    k = torch.randn(1, 2, 128, 32, generator=g).bfloat16()
    q[0, 0, 5, 1] = float("nan")
    (r,) = resolve([att.backward_stat(do, dv, q=q, k=k)])
    assert r.ok and not r.judged
    bad = dv.clone()
    bad[0, 1, 4, 2] += 1.0
    (r,) = resolve([att.backward_stat(do, bad, q=q, k=k)])
    assert not r.ok and r.row == 1
    (f,) = resolve([att.forward_stat(o, v, q=q, k=k)])
    assert f.ok and not f.judged
    oo = o.clone()
    oo[0, 1, 3, 0] = 1e4
    (f,) = resolve([att.forward_stat(oo, v, q=q, k=k)])
    assert not f.ok


def test_attention_backward_reports_an_inf_dv_as_nonfinite():
    """An Inf dV the finite dO can't produce is scored inf (nonfinite), not as the largest float32."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 8).bfloat16().requires_grad_()
    k = torch.randn(1, 2, 32, 8).bfloat16().requires_grad_()
    v = torch.randn(1, 2, 32, 8).bfloat16().requires_grad_()
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    do = torch.randn_like(o) * 1e-3
    o.backward(do)
    dv = v.grad.clone()
    dv[0, 1, 4, 2] = float("inf")
    (r,) = resolve([att.backward_stat(do, dv)])
    assert not r.ok
    assert r.nonfinite and r.max_z == float("inf"), r


def test_attention_forward_partial_unjudged_is_incomplete():
    """One head and feature with a non-finite value can't be judged: the forward check is incomplete (as the
    backward check and checksum._complete_score treat such cases), not clean."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 32, 8).bfloat16()
    k = torch.randn(1, 2, 32, 8).bfloat16()
    v = torch.randn(1, 2, 32, 8).bfloat16()
    v[0, 1, 7, 3] = float("nan")                 # head 1, feature 3 can't be judged
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    (r,) = resolve([att.forward_stat(o, v)])
    assert r.ok
    assert not r.judged, r
