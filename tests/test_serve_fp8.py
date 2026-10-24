"""The FP8 quantization check (serve_fp8.py) on the CPU, against vLLM's quantization as its kernels do it: correct
per-token and per-tensor scales pass; a per-tensor scale taken over padding rows that hold an earlier step's larger
values is reported as taken from outside the batch (vLLM issue #62124's mechanism); a wrong scale, FP8 values two
steps off, a changed static scale are alarms; one step off (fast math) is counted, not an alarm; nothing is judged
outside a model step; an MoE layer's second quantization counts top_k rows per token."""
import pytest
import torch

from sdcguard import serve_fp8 as sf
from sdcguard.serve import ServeGuard


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def vllm_per_token(x):
    amax = x.float().abs().amax(-1, keepdim=True)
    s = (amax / sf.FP8_MAX).clamp(min=sf.MIN_SCALE)
    q = (x.float() / s).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


def vllm_per_tensor(x):
    """Over every row it's given (as vLLM's kernel: padding rows included)."""
    s = (x.float().abs().amax() / sf.FP8_MAX).reshape(1)
    q = (x.float() * (1.0 / s)).clamp(-sf.FP8_MAX, sf.FP8_MAX).to(torch.float8_e4m3fn)
    return q, s


class _Quant:
    def __init__(self, static=False, per_token=True, group=False):
        self.static, self.use_per_token_if_dynamic, self.is_group_quant = static, per_token, group
        self.group_shape = type("G", (), {"name": "PER_TENSOR" if not per_token else "PER_TOKEN"})()


def _setup(recipe_quant, static_scale=None):
    g = ServeGuard()
    fc = g.fp8_checks()
    lay = torch.nn.Module()
    a = fc.add(lay, "model.layers.0.mlp.down_proj", recipe_quant, static_scale, torch.device("cpu"))
    return g, fc, lay, a


def _x(m=12, k=256, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g) * scale).to(torch.bfloat16)


def test_correct_per_token_and_per_tensor_quantization_pass():
    for quant, fn in ((_Quant(per_token=True), vllm_per_token), (_Quant(per_token=False), vllm_per_tensor)):
        g, fc, lay, a = _setup(quant)
        x = _x()
        for step in range(5):
            fc.mark_step(step, rows=12)
            q, s = fn(x)
            sf.check_call(a, x, q, s)
            fc.end_step()
        assert g.poll(wait=True) == [] and fc.stats["calls"] == 5 and fc.stats["scale_wrong"] == 0


def test_a_per_tensor_scale_from_padding_rows_is_named():
    g, fc, lay, a = _setup(_Quant(per_token=False))
    x = _x(16)
    x[10:] = x[10:] * 8                   # rows 10.. are padding left by an earlier, larger request
    fc.mark_step(3, rows=10)
    q, s = vllm_per_tensor(x)             # vLLM's kernel takes every row
    sf.check_call(a, x, q, s)
    fc.end_step()
    ev = g.poll(wait=True)
    assert len(ev) == 1 and ev[0].detector == "serve_fp8"
    d = ev[0].detail
    assert d["finding"] == "scale_from_outside_rows" and d["calls_scale_from_outside_rows"] == 1
    assert d["scale_ratio"] > 7 and d["captured"]["step"] == 3
    # the same padding with the real rows' own maximum larger: nothing to report
    x2 = x.clone()
    x2[:10] = x2[:10] * 20
    fc.mark_step(4, rows=10)
    q, s = vllm_per_tensor(x2)
    sf.check_call(a, x2, q, s)
    fc.end_step()
    assert g.poll(wait=True) == []


def test_wrong_scales_and_values_are_alarms_and_one_step_is_counted():
    g, fc, lay, a = _setup(_Quant(per_token=True))
    x = _x()
    q, s = vllm_per_token(x)
    fc.mark_step(1, rows=12)
    sf.check_call(a, x, q, s * 1.5)       # every row's scale wrong (sampled rows included)
    fc.end_step()
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["finding"] == "scale_wrong"
    bad = q.view(torch.uint8).clone()
    bad ^= 0x20                           # two exponent steps off, every value
    fc.mark_step(2, rows=12)
    sf.check_call(a, x, bad.view(torch.float8_e4m3fn), s)
    fc.end_step()
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["finding"] == "values_wrong" and ev[0].detail["sampled_rows_with_wrong_values"] == 2
    one = q.view(torch.uint8).clone()
    nz = (one & 0x7F) > 1
    one = torch.where(nz, one - 1, one)  # one representable step towards zero
    fc.mark_step(3, rows=12)
    sf.check_call(a, x, one.view(torch.float8_e4m3fn), s)
    fc.end_step()
    assert g.poll(wait=True) == [] and fc.stats["fp8_one_step"] > 0


def test_nothing_is_judged_outside_a_model_step():
    g, fc, lay, a = _setup(_Quant(per_token=False))
    x = _x(8)
    x[4:] *= 50
    q, s = vllm_per_tensor(x)
    sf.check_call(a, x, q, s)             # a warm-up or capture run: no step marked
    fc.mark_step(1, rows=4)
    fc.end_step()
    sf.check_call(a, x, q, s)             # after the step ended
    assert g.poll(wait=True) == [] and fc.stats["calls"] == 0


def test_a_changed_static_scale_is_reported():
    s0 = torch.tensor([0.02])
    g, fc, lay, a = _setup(_Quant(static=True, per_token=False), static_scale=s0)
    x = _x()
    fc.mark_step(1, rows=12)
    q = (x.float() * (1.0 / s0)).clamp(-448, 448).to(torch.float8_e4m3fn)
    sf.check_call(a, x, q, s0)
    fc.end_step()
    assert g.poll(wait=True) == []
    s1 = torch.tensor([0.03])
    fc.mark_step(2, rows=12)
    q = (x.float() * (1.0 / s1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    sf.check_call(a, x, q, s1)
    fc.end_step()
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["finding"] == "static_scale_changed"


def test_an_moe_second_quantization_counts_top_k_rows_per_token():
    g, fc, lay, a = _setup(_Quant(per_token=False))
    top_k, tokens, padded = 4, 3, 5
    h = _x(padded * top_k, 64)
    h[tokens * top_k:] *= 9               # the padded tokens' expert rows
    fc.mark_step(7, rows=tokens)
    q, s = vllm_per_tensor(h)
    sf.check_call(a, h, q, s, mult=top_k)
    fc.end_step()
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["finding"] == "scale_from_outside_rows"


def test_group_scales_are_listed_not_checked():
    g = ServeGuard()
    fc = g.fp8_checks()
    assert fc.add(torch.nn.Module(), "x", _Quant(group=True), None, torch.device("cpu")) is None
    assert "group" in fc.unsupported["x"]
