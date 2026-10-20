"""The CPU's independent confirmation (screen.exact_block): legitimate rounding never counts as a wrong output,
including underflow to zero and overflow to infinity; real corruption still does."""
import torch

from sdcguard.screen import cpu_recheck, exact_block


def _mm(a, b, dt):
    return (a.float() @ b.float()).to(dt)


def test_an_fp16_product_that_underflows_to_zero_is_correct():
    # the Oct 19 outside review's case: fp16(1e-4) x fp16(1e-4) = 1.00033e-8, which fp16 rounds to 0
    a = torch.tensor([[1e-4]], dtype=torch.float16)
    b = torch.tensor([[1e-4]], dtype=torch.float16)
    c = _mm(a, b, torch.float16)
    assert c.item() == 0.0
    r = exact_block(a, b, c, None, 1.0, 1.0)
    assert r["outputs_outside_rounding_envelope"] == 0 and r["worst_error_over_envelope"] < 1


def test_tiny_subnormal_outputs_and_a_tiny_bias_are_correct():
    g = torch.Generator().manual_seed(3)
    a = (torch.randn(1, 64, generator=g) * 3e-4).half()
    b = (torch.randn(64, 32, generator=g) * 3e-4).half()
    bias = (torch.randn(32, generator=g) * 1e-7).half()
    c = (a.float() @ b.float() + bias.float()).half()
    assert exact_block(a, b, c, bias, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 0


def test_an_fp16_result_beyond_its_range_is_correctly_infinite():
    a = torch.full((1, 4), 200.0, dtype=torch.float16)
    b = torch.full((4, 2), 100.0, dtype=torch.float16)
    b[:, 1] = -100.0
    c = _mm(a, b, torch.float16)
    assert torch.isinf(c).all()
    assert exact_block(a, b, c, None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 0
    assert exact_block(a, b, -c, None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 2     # wrong sign
    small = torch.full((4, 2), 1.0, dtype=torch.float16)
    inf_out = torch.full((1, 2), float("inf"), dtype=torch.float16)
    assert exact_block(a, small, inf_out, None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 2  # 800 isn't inf


def test_real_corruption_is_still_outside_the_envelope():
    g = torch.Generator().manual_seed(0)
    for dt in (torch.float16, torch.bfloat16):
        a = torch.randn(1, 512, generator=g).to(dt)
        b = torch.randn(512, 64, generator=g).to(dt)
        c = _mm(a, b, dt)
        assert exact_block(a, b, c, None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 0
        for change in ("add", "zero", "nan"):
            bad = c.clone()
            if change == "add":
                bad[0, 3] += 1.0
            elif change == "zero":
                bad[0, 3] = 0.0 if abs(float(c[0, 3])) > 0.5 else 5.0
            else:
                bad[0, 3] = float("nan")
            assert exact_block(a, b, bad, None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 1, (dt, change)


def test_the_cpu_recheck_of_a_correct_underflow_doesnt_confirm_a_fault():
    a = torch.full((1, 8), 1e-4, dtype=torch.float16)
    b = torch.full((8, 128), 1e-4, dtype=torch.float16)
    c = _mm(a, b, torch.float16)                   # 8.0e-8: rounds to a subnormal
    r = cpu_recheck(a, b, c, None, 1.0, 1.0, row=0, col_block=0)
    assert r["done"] and r["outputs_outside_rounding_envelope"] == 0 and not r["output_mismatch_confirmed"]
