"""Attention checks (attention.py): clean fused attention passes, corrupted dV and outputs are caught."""
import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sdcguard import SDCGuard
from sdcguard import attention as att
from sdcguard.checksum import resolve


def _sdpa(dt=torch.bfloat16, B=2, H=4, KV=4, T=128, D=32, scale=1.0, causal=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(B, H, T, D, generator=g) * scale).to(dt).requires_grad_()
    k = (torch.randn(B, KV, T, D, generator=g) * scale).to(dt).requires_grad_()
    v = torch.randn(B, KV, T, D, generator=g).to(dt).requires_grad_()
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=H != KV)
    do = (torch.randn(o.shape, generator=g) * 1e-3).to(dt)
    o.backward(do)
    return do, v.grad, o.detach(), v.detach()


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [0.02, 1.0, 5.0])          # near-uniform, typical, near one-hot attention
@pytest.mark.parametrize("kv", [4, 1])
def test_clean_attention_scores_low(dt, scale, kv):
    do, dv, o, v = _sdpa(dt, KV=kv, scale=scale, causal=scale != 1.0)
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


def test_the_guard_checks_fused_attention_only_when_asked():
    g = SDCGuard()
    _model_step(g)
    assert g.stats["attention_checked"] == 0
    g = SDCGuard(attention=True)
    _model_step(g)
    if "aten::_scaled_dot_product_flash_attention_for_cpu" not in str(att.FORWARD_OPS):
        pytest.skip("no fused CPU attention op in this PyTorch")
    assert g.stats["attention_checked"] == 2 and g.stats["alerts"] == 0


@pytest.mark.parametrize("backward", [True, False])
def test_the_guard_alerts_on_corrupted_attention(backward):
    g = SDCGuard(attention=True)
    _model_step(g, inject=_Corrupt(backward))
    kinds = [e.detail["kind"] for e in g.log.events]
    assert kinds == ["attention_backward" if backward else "attention_forward"]
    ev = g.log.events[0].detail
    assert ev["op"] in (att.BACKWARD_OPS if backward else att.FORWARD_OPS) and ev["check"].startswith("attention")


def test_the_calibration_experiment_runs_on_the_cpu():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
    import attention_calib
    res = attention_calib.calibrate(torch.device("cpu"), say=lambda s: None)
    (run,) = res["runs"]
    assert res["criteria_met"] and run["clean"]["checks"] > 0 and run["clean"]["skipped"] == {}
    assert run["injected"]["dV + 256 x rms"]["alarms"] == run["injected"]["dV + 256 x rms"]["corrupted_calls"] > 0
    assert "| tiny, GQA 4:2 | cpu flash |" in attention_calib.markdown(res)
