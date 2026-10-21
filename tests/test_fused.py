"""The fused precise check (fused.py): block sums of the fp32 accumulator taken inside the matmul.

The kernel cases run tests/_fused_compare.py in a separate process with Triton's CPU interpreter
(TRITON_INTERPRET=1 has to be set before Triton is imported). On a GPU, experiments/fused_check.py runs the
same comparison on the real hardware.
"""
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn as nn

from sdcguard import SDCGuard
from sdcguard import fused

HERE = os.path.dirname(os.path.abspath(__file__))


def test_kernel_and_guard_match_the_plain_math_in_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_fused_compare.py"), "cpu"], env=env,
                       capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0 and "ALL AGREE" in p.stdout, p.stdout[-4000:] + p.stderr[-4000:]


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="checks the case without the interpreter")
def test_not_available_on_a_cpu_without_the_interpreter():
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    assert not fused.available(x) and not fused.available(torch.randn(4, 8))
    with pytest.raises(RuntimeError, match="Triton"):
        fused.fused_mm(x, x.t())


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="checks the case without the interpreter")
def test_precise_fused_falls_back_to_precise_where_the_kernel_cant_run():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))
    g = SDCGuard(every=1, precise="fused").protect(model)
    try:
        with g.on_step(0):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                loss = model(torch.randn(64, 256)).float().pow(2).mean()
            loss.backward()
    finally:
        g.unprotect()
    assert g.stats["checked_fused"] == 0 and g.stats["checked_fp32"] == g.stats["checked"] == 5
    assert g.stats["alerts"] == 0 and g.stats["fused_fallbacks"] == 0


def test_fused_mm_rejects_bad_shapes_before_launching():
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="shape mismatch"):
        fused.fused_mm(x, torch.randn(9, 3, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="bias"):
        fused.fused_mm(x, torch.randn(8, 3, dtype=torch.bfloat16), torch.randn(4))


def _interp(code: str) -> subprocess.CompletedProcess:
    """Runs code in a new process under Triton's CPU interpreter (it has to be set before Triton is imported)."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", OMP_NUM_THREADS="1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                          timeout=600)


def _result(p: subprocess.CompletedProcess) -> list:
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
    assert line, p.stdout[-2000:] + p.stderr[-2000:]
    return line[0].split()[1:]


def test_fused_selfcheck_judges_fp16_layers_whose_block_sums_exceed_fp16_range():
    """fp16 stops at 65504. Block sums of W above that made the three fp16 parts of the expected side infinite,
    so the default in-kernel check left an ordinary finite matmul unjudged; the parts are now made from the
    block sums divided by a power of two (exact), and the narrow dot's result is multiplied back."""
    finite, fused_judged, fz, precise_judged, pz = _result(_interp("""
        import torch
        from sdcguard import checksum as cs, fused
        torch.manual_seed(0)
        A = torch.randn(16, 64).half()
        B = (torch.rand(64, 256) * 900 + 100).half()        # finite, outputs ~4e4; block sums of B ~7e4
        C, st = fused.fused_mm_checked(A, B)
        r = cs.resolve([st])[0]
        rp = cs.check_matmul_acc(A, B, A.float() @ B.float())
        print("RESULT", bool(torch.isfinite(C).all()), r.judged, r.max_z, rp.judged, rp.max_z)
    """))
    assert finite == "True" and precise_judged == "True"     # an ordinary matmul precise=True judges
    assert fused_judged == "True" and float(fz) < 3, f"fused z={fz}, precise=True z={pz}"


def test_fused_expected_side_term_is_the_frozen_model_and_fp16_parts_stay_exact():
    """The in-kernel noise of the expected side x . (block sums) must be the frozen model for that dot product,
    C1 sqrt(k) u |x| |block sums|_2 (as for any output column). It used max|block sums| instead, a lower bound up
    to sqrt(k) times smaller (0.64 against 11.2 here). And in fp16 the scaled parts must still add up to the
    fp32 block sums: small sums too, which a fixed scale would push below fp16's normal range."""
    out = _result(_interp("""
        import torch, triton
        import triton.language as tl
        from sdcguard import fused
        torch.manual_seed(0)
        K, N = 4096, 128
        B = (torch.randn(K, N) / 64).bfloat16()
        signs = fused._signs_for(N, "cpu", 0)
        BX = torch.zeros((K, 1, fused.NX), dtype=torch.bfloat16)
        AUX = torch.zeros((1, 4), dtype=torch.float32)
        fused._bside_kernel[(triton.cdiv(K, 32), 1)](B, signs, BX, AUX, K, N, B.stride(0), B.stride(1),
            BX.stride(0), BX.stride(1), BX.stride(2), BKP=32, BN=128, SPLIT_DT=tl.bfloat16, STATS=True,
            PARTS=True, SCALED=False)
        bp = B.float().sum(1)
        used, model, mx = float(AUX[0, 1]) ** 0.5, float(bp.norm()), float(bp.abs().max())
        # fp16: rows of very different sizes, statistics first, then the scaled parts
        W = torch.randn(K, N)
        W[: K // 2] *= 1e-3
        W = W.half()
        BX = torch.zeros((K, 1, fused.NX), dtype=torch.float16)
        AUX = torch.zeros((1, 4), dtype=torch.float32)
        for st, pa, sc in ((True, False, False), (False, True, True)):
            fused._bside_kernel[(triton.cdiv(K, 32), 1)](W, signs, BX, AUX, K, N, W.stride(0), W.stride(1),
                BX.stride(0), BX.stride(1), BX.stride(2), BKP=32, BN=128, SPLIT_DT=tl.float16, STATS=st,
                PARTS=pa, SCALED=sc)
        top = 128 * float(AUX[0, 3])                      # the largest block sum this block could have
        scale = 2.0 ** (int(torch.tensor(top).log2().floor()) - 14)
        parts = BX[:, 0, :3].double().sum(1) * scale
        # allowed: the kernel's own fp32 summation of the 128 values, and in the last part half of fp16's
        # smallest step (2^-24) times the scale, which is at most 2^-14 of `top`
        allowed = 128 * 2.0 ** -24 * W.double().abs().sum(1) + 2.0 ** -38 * top
        worst = float(((parts - W.double().sum(1)).abs() / allowed).max())
        print("RESULT", used, model, mx, worst)
    """))
    used, model, mx, worst = map(float, out)
    assert abs(used - model) <= 1e-5 * model and model > 5 * mx, (used, model, mx)
    assert worst <= 1.0, worst
