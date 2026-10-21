"""The fused precise check (fused.py) against the plain PyTorch math, in Triton's CPU interpreter.

Run by tests/test_fused.py in a subprocess with TRITON_INTERPRET=1 (it has to be set before Triton is
imported), and on GPUs by experiments/fused_check.py. Prints ALL AGREE when every case passes.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

from sdcguard import SDCGuard
from sdcguard import checksum as cs
from sdcguard import fused

DEV = sys.argv[1] if len(sys.argv) > 1 else "cpu"
fails = []


def check(cond, what):
    if not cond:
        fails.append(what)
        print("FAIL", what)


def ulp(x, dtype):
    """One step of `dtype` at each value of x (subnormal steps near zero)."""
    mant, emin = (7, -126) if dtype == torch.bfloat16 else (10, -14)
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-300))).clamp_min(emin)
    return torch.exp2(e - mant)


def operands(m, k, n, dtype, bias, layout, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(m, k, generator=g).to(dtype).to(DEV)
    B = (torch.randn(k, n, generator=g) / k ** 0.5).to(dtype).to(DEV)
    if layout == "w.t()":                        # a linear layer's forward: B = w.t(), w stored (n, k)
        B = B.t().contiguous().t()
    elif layout == "gy.t()":                     # a weight gradient: A = gy.t(), gy stored (k, m)
        A = A.t().contiguous().t()
    b = (torch.randn(n, generator=g) * 0.5).to(dtype).to(DEV) if bias else None
    return A, B, b


CASES = [(16, 96, 200, True, "w.t()"), (33, 64, 128, False, "plain"), (8, 40, 77, True, "w.t()"),
         (5, 256, 300, True, "gy.t()"), (70, 130, 129, False, "w.t()"), (1, 32, 256, True, "plain")]

# 1. The output and the block statistics are those of the fp32 accumulator
for dtype in (torch.bfloat16, torch.float16):
    for i, (m, k, n, bias, layout) in enumerate(CASES):
        A, B, b = operands(m, k, n, dtype, bias, layout, i)
        C, st = fused.fused_mm(A, B, b)
        ref = A.double() @ B.double() + (b.double() if bias else 0)
        tag = f"{dtype} {m}x{k}x{n} bias={bias} {layout}"
        check(C.dtype == dtype and C.shape == (m, n), f"{tag}: output dtype/shape")
        # correctly rounded, give or take the fp32 accumulation's own rounding (summation order differs)
        half = 0.5 * ulp(ref, dtype)
        acc_err = k * 2.0 ** -24 * ((A.double().abs() @ B.double().abs()) + (b.double().abs() if bias else 0))
        check(bool(((C.double() - ref).abs() <= half * 1.0001 + acc_err).all()),
              f"{tag}: output is the fp32 result rounded to nearest")
        nb = -(-n // 128)
        check(st.shape == (m, nb, 4) and st.dtype == torch.float64, f"{tag}: stats shape")
        pad = torch.zeros(m, nb * 128, dtype=torch.float64, device=DEV)
        pad[:, :n] = ref
        blocks = pad.view(m, nb, 128)
        signs = torch.zeros(nb * 128, dtype=torch.float64, device=DEV)
        signs[:n] = fused._signs_for(n, DEV, 0)
        want = torch.stack([blocks.sum(-1), (blocks * signs.view(nb, 128)).sum(-1), (blocks ** 2).sum(-1),
                            blocks.abs().sum(-1)], -1)
        tol = 1e-5 * (blocks.abs().sum(-1, keepdim=True) + blocks.pow(2).sum(-1, keepdim=True) + 1)
        check(bool(((st - want).abs() <= tol).all()), f"{tag}: block statistics match the fp32 result")

        # 2. Clean: the check from the statistics scores like precise mode's check of a stored fp32 result
        r = cs.resolve([fused.zscores_from_stats(A, B, st, b)])[0]
        C32 = A.float() @ B.float() + (b.float() if bias else 0)
        r_ref = cs.check_matmul_acc(A, B, C32, b)
        check(r.ok and 0 <= r.max_z < 1.0, f"{tag}: clean, z={r.max_z:.3f}")
        check(r_ref.ok and abs(r.max_z - r_ref.max_z) < 0.5, f"{tag}: z {r.max_z:.3f} vs stored fp32 result "
                                                              f"{r_ref.max_z:.3f}")

        # 3. Errors in the accumulator, before rounding: located, down to below bf16's resolution
        row, col = m // 2, n - 1
        for delta in (1.0, 1e-2):
            _, st2 = fused.fused_mm(A, B, b, inject=(row, col, delta))
            r2 = cs.resolve([fused.zscores_from_stats(A, B, st2, b)])[0]
            check(not r2.ok and r2.row == row and r2.col_block == col // 128,
                  f"{tag}: error {delta} caught and located (z={r2.max_z:.1f}, row {r2.row}, block {r2.col_block})")
        for bad in (float("inf"), float("nan")):
            _, st3 = fused.fused_mm(A, B, b, inject=(0, 0, bad))
            r3 = cs.resolve([fused.zscores_from_stats(A, B, st3, b)])[0]
            check(not r3.ok, f"{tag}: {bad} in the accumulator flagged (z={r3.max_z})")

# 4. An error smaller than bf16 rounding: the fused check sees it, the ordinary check of the output can't
A, B, b = operands(32, 128, 256, torch.bfloat16, True, "w.t()", 99)
Cc, stc = fused.fused_mm(A, B, b)
typical = Cc.float().abs().median().item()
delta = 2e-3 * typical                       # well under half a bf16 step (0.2% vs about 0.4% to 0.8%)
Cd, std = fused.fused_mm(A, B, b, inject=(7, 200, delta))
rf = cs.resolve([fused.zscores_from_stats(A, B, std, b)])[0]
ro = cs.check_matmul(A, B, Cd, b)
print(f"sub-rounding error {delta:.2e} ({delta / typical:.1%} of a typical output): fused z={rf.max_z:.1f}, "
      f"ordinary check of the bf16 output z={ro.max_z:.2f}, outputs changed: {int((Cd != Cc).sum())}")
check(not rf.ok and rf.row == 7 and rf.col_block == 1, "sub-rounding error caught by the fused check")
check(ro.ok, "the ordinary check of the rounded output can't see it (the reason for precise mode)")

# 5. A fresh sign pattern per epoch: still clean, and the statistics follow the new pattern
r_e = cs.resolve([fused.zscores_from_stats(A, B, fused.fused_mm(A, B, b, epoch=3)[1], b, epoch=3)])[0]
r_x = cs.resolve([fused.zscores_from_stats(A, B, stc, b, epoch=3)])[0]
check(r_e.ok and r_e.max_z < 1.0, f"epoch 3 clean (z={r_e.max_z:.3f})")
check(not torch.equal(fused._signs_for(256, DEV, 0), fused._signs_for(256, DEV, 3)) and not r_x.ok,
      "statistics from another epoch's pattern don't pass")

# 6. In a model: SDCGuard(precise="fused") against precise=True
def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 384), nn.GELU(), nn.Linear(384, 256)).to(DEV)


def step(net, g, seed, step_no=0):
    torch.manual_seed(seed)
    x = torch.randn(48, 256, device=DEV)
    with g.on_step(step_no):
        with torch.autocast(DEV, dtype=torch.bfloat16):
            loss = net(x).float().pow(2).mean()
        loss.backward()
    grads = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    return loss.detach(), grads


net_p, net_f = model(), model()
gp = SDCGuard(every=1, precise=True).protect(net_p)
gf = SDCGuard(every=1, precise="fused").protect(net_f)
for s in range(2):
    lp, grp = step(net_p, gp, s, s)
    lf, grf = step(net_f, gf, s, s)
    check(torch.allclose(lp, lf, rtol=1e-3), f"model step {s}: loss {lf.item()} vs precise=True {lp.item()}")
    for a, c in zip(grp, grf):
        check((a - c).abs().max() <= 2.0 ** -6 * a.abs().max(), f"model step {s}: gradients agree")
print("fused guard stats:", gf.stats)
check(gf.stats["checked_fused"] == gf.stats["checked"] == 10 and gf.stats["alerts"] == 0,
      "every linear matmul (forward, dX, dW) went through the fused check, clean")
check(gp.stats["alerts"] == 0, "precise=True clean on the same steps")

hits = []
gf.fused_inject = lambda name, pass_, shape: (3, 5, 0.05) if (name, pass_) == ("2", "backward dW") else None
gf.on_alert = hits.append
step(net_f, gf, 7, 2)
gf.fused_inject = None
check(len(hits) == 1 and hits[0].detail["where"] == "2 (backward dW)" and "fused" in hits[0].detail["dtype"],
      f"a fault in one weight gradient's accumulator: one alert, named ({[h.detail['where'] for h in hits]})")
gp.unprotect()
gf.unprotect()

net_s = model()
g_s = SDCGuard(every=1, precise="fused", sample_rate=0.5, seed=1).protect(net_s)
step(net_s, g_s, 3)
check(0 < g_s.stats["checked_fused"] < g_s.stats["matmuls_seen"] == 5 and g_s.stats["alerts"] == 0,
      f"sampling: unsampled matmuls run unchecked ({g_s.stats})")
g_s.unprotect()

# 5. The fp32 self-check (fused_mm_checked): no float64, scored in the kernel
for dtype in (torch.bfloat16, torch.float16):
    for i, (m, k, n, bias, layout) in enumerate(CASES):
        A, B, b = operands(m, k, n, dtype, bias, layout, 100 + i)
        tag = f"self-check {dtype} {m}x{k}x{n} bias={bias} {layout}"
        C, st = fused.fused_mm_checked(A, B, b)
        C0, _ = fused.fused_mm(A, B, b)
        r = cs.resolve([st])[0]
        check(torch.equal(C, C0), f"{tag}: same output as the first version")
        check(r.ok and r.judged and r.max_z < 1.0, f"{tag}: clean (z {r.max_z:.3f})")
        typical = float(C.float().abs().median()) or 1.0
        _, sti = fused.fused_mm_checked(A, B, b, inject=(m - 1, n - 1, 0.01 * typical))
        ri = cs.resolve([sti])[0]
        check(not ri.ok and ri.row == m - 1, f"{tag}: a 1% error in one accumulator caught and located "
                                             f"(z {ri.max_z:.1f}, row {ri.row})")
A, B, b = operands(64, 256, 256, torch.bfloat16, True, "w.t()", 7)
C, _ = fused.fused_mm_checked(A, B, b)
typical = float(C.float().abs().median())
_, st = fused.fused_mm_checked(A, B, b, inject=(5, 7, 0.002 * typical))
check(not cs.resolve([st])[0].ok, "self-check: an error of 0.2% of a typical output (under half a bf16 step) caught")
_, st = fused.fused_mm_checked(A, B, b, inject=(5, 7, float("inf")))
r = cs.resolve([st])[0]
check(not r.ok and r.judged, f"self-check: an Inf the inputs can't produce is a failure ({r})")
A2 = A.clone()
A2[3, 4] = float("inf")
_, st = fused.fused_mm_checked(A2, B, b)
r = cs.resolve([st])[0]
check(r.ok and not r.judged, f"self-check: an Inf input can't be judged, and isn't called corruption ({r})")
_, st = fused.fused_mm_checked(A[:, :0], B[:0], b)
check(cs.resolve([st])[0].ok, "self-check: k = 0")

print("ALL AGREE" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
