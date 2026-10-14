import math
import random

import pytest
import torch
import torch.nn.functional as F

from sdcguard import check_linear, check_matmul


def make_layer(T, din, dout, dtype=torch.bfloat16, outliers=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, din, generator=g)
    if outliers:  # LLM-style outlier input channels
        ch = torch.randperm(din, generator=g)[: max(1, din // 256)]
        x[:, ch] *= 40
    W = torch.randn(dout, din, generator=g) / math.sqrt(din)
    b = torch.randn(dout, generator=g) * 0.02
    x, W, b = x.to(dtype), W.to(dtype), b.to(dtype)
    return x, W, b, F.linear(x, W, b)


SHAPES = [(256, 512, 1024), (512, 1024, 256), (128, 768, 3072), (64, 2048, 1000)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("outliers", [False, True])
def test_no_false_alarms_on_clean_layers(dtype, outliers):
    for i, (T, din, dout) in enumerate(SHAPES):
        x, W, b, y = make_layer(T, din, dout, dtype, outliers, seed=i)
        res = check_linear(x, W, b, y)
        assert res is not None and res.ok, f"false alarm: shape {(T, din, dout)} z={res.max_z:.2f}"


def _rate(y0, x, W, b, corrupt, trials=60):
    caught = 0
    for t in range(trials):
        y = y0.clone()
        corrupt(y, random.Random(t))
        caught += not check_linear(x, W, b, y).ok
    return caught / trials


def test_catches_zeroed_warp_chunk():
    x, W, b, y0 = make_layer(256, 1024, 2048, outliers=True)

    def zero_chunk(y, rng):
        r, c = rng.randrange(y.shape[0]), rng.randrange(y.shape[1] // 32) * 32
        y[r, c:c + 32] = 0

    assert _rate(y0, x, W, b, zero_chunk) >= 0.9


def test_catches_nan_and_inf():
    x, W, b, y0 = make_layer(128, 512, 512)
    for bad in (float("nan"), float("inf")):
        y = y0.clone()
        y[3, 7] = bad
        res = check_linear(x, W, b, y)
        assert not res.ok and res.nonfinite


def test_catches_top_exponent_bit_flip():
    x, W, b, y0 = make_layer(128, 512, 1024)

    def flip14(y, rng):
        i = rng.randrange(y.numel())
        v = y.view(-1)[i:i + 1].contiguous()
        y.view(-1)[i] = _flip_bits_exact(v, 14)[0]

    assert _rate(y0, x, W, b, flip14) == 1.0


def _flip_bits_exact(v, bit):
    ints = v.view(torch.int16).clone()
    ints ^= (1 << bit) if bit < 15 else -32768
    return ints.view(v.dtype)


def test_remainder_block_is_checked():
    x, W, b, y0 = make_layer(64, 256, 1000)   # 1000 = 7*128 + 104
    y = y0.clone()
    y[5, 960:992] = 0                          # inside the 104-wide remainder block
    assert not check_linear(x, W, b, y).ok


def test_bmm_clean_and_corrupted():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(4, 64, 128, generator=g).bfloat16()
    B = (torch.randn(4, 128, 256, generator=g) / math.sqrt(128)).bfloat16()
    C = torch.bmm(A, B)
    assert check_matmul(A, B, C).ok
    C2 = C.clone()
    C2[2, 10, 32:64] = 0
    res = check_matmul(A, B, C2)
    assert not res.ok and res.batch == 2 and res.row == 10


def test_addmm_with_2d_bias_and_scaling():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(64, 128, generator=g).bfloat16()
    B = (torch.randn(128, 256, generator=g) / math.sqrt(128)).bfloat16()
    bias = torch.randn(64, 256, generator=g).bfloat16()
    C = torch.addmm(bias, A, B, beta=0.5, alpha=2.0)
    assert check_matmul(A, B, C, bias, alpha=2.0, beta=0.5).ok


def test_fp32_is_skipped():
    A, B = torch.randn(8, 8), torch.randn(8, 8)
    assert check_matmul(A, B, A @ B) is None


@pytest.mark.parametrize("scale", [1e-12, 1e-18, 1e-20, 1e-22, 1e-24, 1e-26, 1e-30, 1e-36])
def test_tiny_outputs_never_false_alarm(scale):
    # Found on an H100: GPT-2 small without LR warmup had attention gradients ~1e-24. Their squares
    # underflow fp32, the noise estimate collapsed, and clean checks scored z ~ 1e4 to 1e5.
    g = torch.Generator().manual_seed(0)
    W = (torch.randn(768, 768, generator=g) * 0.02).bfloat16()
    for i in range(4):
        dY = (torch.randn(256, 768, generator=g) * scale).bfloat16()
        dX = (dY.float() @ W.float()).bfloat16()
        res = check_matmul(dY, W, dX)
        assert res.ok, f"false alarm at scale {scale}: z={res.max_z}"


def test_corruption_still_caught_down_to_tiny_scales():
    g = torch.Generator().manual_seed(1)
    W = (torch.randn(768, 768, generator=g) * 0.02).bfloat16()
    for scale in (1.0, 1e-8, 1e-15, 1e-18):
        dY = (torch.randn(256, 768, generator=g) * scale).bfloat16()
        dX = (dY.float() @ W.float()).bfloat16()
        bad = dX.clone()
        bad[9, 0:32] = 0
        res = check_matmul(dY, W, bad)
        assert not res.ok and res.row == 9, f"missed zeroed chunk at scale {scale}"
    # a tiny value whose exponent gets corrupted into a big one is still caught
    dY = (torch.randn(256, 768, generator=g) * 1e-24).bfloat16()
    dX = (dY.float() @ W.float()).bfloat16()
    bad = dX.clone()
    bad[3, 5] = 1e-3
    assert not check_matmul(dY, W, bad).ok


def test_result_reports_noise_scale():
    x, W, b, y = make_layer(64, 256, 256)
    res = check_linear(x, W, b, y)
    assert res.noise is not None and 0 < res.noise < 1


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_identical_columns_do_not_false_alarm(dtype):
    """B with identical columns (a matrix of ones, an expanded vector) makes every output in a
    block equal, so they all round the same way. That is not corruption."""
    g = torch.Generator().manual_seed(0)
    for k in (8, 64, 768, 4096):
        A = torch.randn(64, k, generator=g).to(dtype)
        for B in (torch.ones(k, 512, dtype=dtype), torch.randn(k, 1, generator=g).to(dtype).expand(k, 512)):
            res = check_matmul(A, B, A @ B)
            assert res.ok, (k, res)


def test_extreme_but_legitimate_bf16_values_do_not_false_alarm():
    """Finite outputs whose fp32 checksum arithmetic overflows are not corruption."""
    g = torch.Generator().manual_seed(2)
    cases = []
    A, B = torch.ones(4, 1, dtype=torch.bfloat16), torch.full((1, 128), 1e37, dtype=torch.bfloat16)
    cases.append((A, B))                                         # block sums overflow fp32
    B = torch.full((2, 128), 1e37, dtype=torch.bfloat16)
    B[1] = -1e37
    cases.append((torch.full((4, 2), 1e-30, dtype=torch.bfloat16), B))      # B's block sums overflow
    cases.append((torch.full((4, 8), 1e20, dtype=torch.bfloat16),
                  (torch.randn(8, 256, generator=g) * 1e-12).bfloat16()))   # |a_row|^2 overflows
    cases.append(((torch.randn(4, 8, generator=g) * 1e-15).bfloat16(),
                  (torch.randn(8, 128, generator=g) * 1e18).bfloat16()))    # checksum column^2 overflows
    B = torch.randn(8, 256, generator=g).bfloat16()
    B[:, 128:] = 0
    cases.append((torch.full((4, 8), 1e20, dtype=torch.bfloat16), B))       # huge A, a zero block
    for A, B in cases:
        C = (A.float() @ B.float()).bfloat16()
        assert torch.isfinite(C).all()
        res = check_matmul(A, B, C)
        assert res.ok, res


@pytest.mark.parametrize("scale", [1e-2, 3e-3, 1e-3, 3e-4])
def test_tiny_fp16_outputs_do_not_false_alarm(scale):
    """Small fp16 gradients land below fp16's smallest normal number (6e-5), where rounding is a
    fixed step (6e-8) whatever the size. That is not corruption."""
    g = torch.Generator().manual_seed(3)
    for _ in range(5):
        A = (torch.randn(128, 256, generator=g) * scale).half()
        B = (torch.randn(256, 300, generator=g) * scale).half()
        C = (A.float() @ B.float()).half()
        res = check_matmul(A, B, C)
        assert res.ok, (scale, res)
    C[17, 40] += 0.01                         # a real change at this scale is still caught
    res = check_matmul(A, B, C)
    assert not res.ok and (res.row, res.col_block) == (17, 0)
