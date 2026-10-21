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


def _rounded_once(A, B, C) -> bool:
    """C is A @ B rounded once from the exact product (what an fp32-accumulating GEMM gives, to an fp32 rounding)."""
    exact = A.double() @ B.double()
    u = 2.0 ** -8 if C.dtype == torch.bfloat16 else 2.0 ** -11
    return bool(((C.double() - exact).abs() <= 1.01 * u * exact.abs() + 2.0 ** -126).all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_identical_columns_do_not_false_alarm(dtype):
    """B with identical columns (a matrix of ones, an expanded vector) makes every output in a
    block equal, so they all round the same way. That is not corruption. Checked on the exact product rounded
    once, and on this PyTorch's own CPU matmul where that matmul is accurate: PyTorch 2.8's bf16 matmul on aarch64
    CPUs accumulates at low precision (up to 14% off at k = 4096 on October 21, 2026; 2.14.1 rounds correctly), and
    the check rightly flags that (test_a_matmul_that_accumulates_in_low_precision_is_caught)."""
    g = torch.Generator().manual_seed(0)
    for k in (8, 64, 768, 4096):
        A = torch.randn(64, k, generator=g).to(dtype)
        for B in (torch.ones(k, 512, dtype=dtype), torch.randn(k, 1, generator=g).to(dtype).expand(k, 512)):
            res = check_matmul(A, B, (A.double() @ B.double()).to(dtype))
            assert res.ok, (k, res)
            C = A @ B
            if _rounded_once(A, B, C):
                res = check_matmul(A, B, C)
                assert res.ok, (k, "this PyTorch's matmul", res)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_matmul_that_accumulates_in_low_precision_is_caught(dtype):
    """A GEMM that keeps its running sums in bf16/fp16 instead of fp32 gives results farther from the exact product
    than one rounding allows. PyTorch 2.8's bf16 matmul on aarch64 CPUs did that (measured against float64 on
    October 21, 2026: up to 5.6% off at k = 768 and 14% at k = 4096; 2.14.1 is correct), and the check flagged it
    on clean inputs. Emulated here: a running sum kept in the output's dtype, rounded after every product (z 47 in
    bf16, 25 in fp16 on these inputs). Milder versions can pass: partial sums of 64 products, rounded after each,
    score 7.4 in bf16 but 3.0 in fp16, under the line, since the check compares with the output's own rounding."""
    g = torch.Generator().manual_seed(1)
    k = 4096
    A = torch.randn(64, k, generator=g).to(dtype)
    B = (torch.randn(k, 512, generator=g) / k ** 0.5).to(dtype)
    acc = torch.zeros(64, 512, dtype=dtype)
    for i in range(k):
        acc = (acc.double() + A[:, i:i + 1].double() @ B[i:i + 1].double()).to(dtype)
    assert not _rounded_once(A, B, acc)
    assert check_matmul(A, B, (A.double() @ B.double()).to(dtype)).ok          # the same product, rounded once
    assert not check_matmul(A, B, acc).ok


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


def test_floor_does_not_hide_an_error_50x_the_outputs_in_attention_sink_pv(monkeypatch):
    """The floor for the fp32 rounding of the expected side must stay near that rounding. P @ V of attention
    with a sink (98% of every query's probability on key 0, whose value row is zero, T = 2048): the rounding is
    about 1e-8 and the outputs' bf16 noise about 2e-5, but the old floor, 1e-5 x |a_row|_2 x |checksum
    column|_2, was 5e-3 and let one output off by 50 times a typical output score 2.9. The bound
    |a_row|_1 x max|checksum column| is about ten times smaller here."""
    from sdcguard import checksum as cs
    monkeypatch.setattr(cs, "USE_FAST", False)
    torch.manual_seed(0)
    T, d = 2048, 128
    p = torch.full((4, T), 0.02 / (T - 1))
    p[:, 0] = 0.98
    V = torch.randn(T, d)
    V[0] = 0.0
    P, V = p.bfloat16(), V.bfloat16()
    C = (P.float() @ V.float()).bfloat16()
    assert check_matmul(P, V, C).ok                          # clean
    typical = C.float().abs().mean().item()                  # ~3e-4
    bad = C.clone()
    bad[1, 5] += 50 * typical
    r = check_matmul(P, V, bad)
    assert not r.ok and (r.row, r.col_block) == (1, 0), f"missed: z={r.max_z:.2f}, noise={r.noise:.2e}"


def test_expected_side_floor_takes_the_smaller_bound_and_grows_with_long_sums():
    """The floor is min(1e-5 |a|_2 |col|_2, l1_floor_coef(k) |a|_1 max|col|): never above the Cauchy-Schwarz
    floor it replaced, and the second coefficient grows with k past 1,342 terms (tensor cores truncate each
    product against the running sum, an error that grows with k when outputs nearly cancel)."""
    from sdcguard.checksum import l1_floor_coef
    assert l1_floor_coef(128) == l1_floor_coef(1342) == 1e-5
    assert l1_floor_coef(131072) == 131072 * 2.0 ** -27 > 9e-4
    # outputs that cancel exactly, with a peaked row, at every length up to 131072: no false alarm
    g = torch.Generator().manual_seed(5)
    for k in (1024, 16384, 131072):
        a = torch.randn(4, k, generator=g) * 1e-3
        a[:, 0], a[:, k - 1] = 1.0, -1.0
        B = torch.randn(k, 128, generator=g)
        B[k - 1] = B[0] + (a[:, 1:k - 1] @ B[1:k - 1]).mean(0)
        A, Bq = a.bfloat16(), B.bfloat16()
        C = (A.double() @ Bq.double()).bfloat16()
        assert check_matmul(A, Bq, C).ok, k


def _tensor_core_dot(a, b, n=8, kept=25):
    """a @ b (float64 arrays holding bf16 and fp32 values) as tensor cores add it up (Khattak and Mikaitis,
    arXiv 2512.07004; the model in modes.py): each step adds n exact products to the fp32 running sum, every
    term cut toward zero to `kept` bits below the step's largest term, and the step's sum cut to fp32."""
    import numpy as np
    s = np.zeros(a.shape[0])
    for i in range(0, a.shape[1], n):
        terms = np.concatenate([s[:, None], a[:, i:i + n] * b[i:i + n]], axis=1)
        top = np.abs(terms).max(1)
        step = np.exp2(np.floor(np.log2(np.where(top > 0, top, 1.0))) - kept)[:, None]
        r = (np.trunc(terms / step) * step).sum(1)
        ulp = np.exp2(np.floor(np.log2(np.where(r != 0, np.abs(r), 1.0))) - 23)
        s = np.trunc(r / ulp) * ulp
    return s


def test_expected_side_floor_covers_truncating_tensor_cores_on_long_sums():
    """On GPUs the expected side is a tensor-core product, and truncation toward zero makes its error grow with
    k when the running sum dwarfs the products: dQ = dS K of an attention backward with a sink (the sink's dS
    carries the running sum) and keys that share a large mean, whose block sums cancel against dS's zero row
    sums, at k = 131,072. With a fixed 1e-5 in front of |a_row|_1 max|column| that error alone scores 7.8,
    a false alarm on a clean result; l1_floor_coef(k) keeps it near 0.2."""
    import numpy as np
    from sdcguard.checksum import l1_floor_coef
    g = torch.Generator().manual_seed(0)
    k = 131072
    p = torch.rand(4, k, generator=g) + 0.5
    p[:, 1:] *= 0.02 / p[:, 1:].sum(1, keepdim=True)
    p[:, 0] = 0.98
    dP = torch.randn(4, k, generator=g, dtype=torch.float64)
    dS = (p.double() * (dP - (p.double() * dP).sum(1, keepdim=True))).bfloat16()       # rows sum to 0
    K = (torch.randn(k, 128, generator=g) + 16).bfloat16()
    a, col = dS.double().numpy(), K.float().sum(1).double().numpy()     # one plain checksum column
    err = np.abs(_tensor_core_dot(a, col) - a @ col)
    noise = 2.0 ** -8 * (dS.double() @ K.double()).bfloat16().float().norm(dim=1).double().numpy()
    holder = np.abs(a).sum(1) * np.abs(col).max()
    cauchy = 1e-5 * np.linalg.norm(a, axis=1) * np.linalg.norm(col)
    z_fixed = err / (noise + np.minimum(cauchy, 1e-5 * holder))
    z_used = err / (noise + np.minimum(cauchy, l1_floor_coef(k) * holder))
    assert z_fixed.max() > 6 and z_used.max() < 1, (z_fixed.max(), z_used.max())
