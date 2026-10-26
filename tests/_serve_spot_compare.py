"""Run by tests/test_serve_spot.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the spot check's Triton kernel
against its PyTorch version on the CPU. Prints ALL MATCH."""
import torch

from sdcguard import serve_spot as sp


def run(fn, x, y, w, b, step, real=None, nl=3, idx=1):
    k = x.shape[1]
    st = torch.zeros(2, sp.COLS, dtype=torch.int64)
    caps = sp.new_caps("cpu", k + 3)
    sb = torch.full((1,), sp.pack_step(step, real), dtype=torch.int64)
    fn(x, y, w, b, sb, st, 1, nl, idx, *caps, sp.SPOT_Z)
    return st, caps


def main():
    for dt in (torch.bfloat16, torch.float16):
        ok, why = sp.verify_kernel(torch.device("cpu"), dt)
        assert ok, (dt, why)
    g = torch.Generator().manual_seed(5)
    n = 0
    for dt in (torch.bfloat16, torch.float16):
        for m, nout, k in ((1, 8, 64), (7, 33, 129), (50, 300, 1000)):
            x = torch.randn(m, k, generator=g).to(dt)
            w = (torch.randn(nout, k, generator=g) * 0.05).to(dt)
            bias = (torch.randn(nout, generator=g) * 0.1).to(dt)
            for b in (None, bias):
                y = torch.nn.functional.linear(x, w, b)
                for step, case in ((4, "clean"), (7, "wrong"), (5, "inactive"), (4, "real"), (10, "strided"),
                                   (4294967293, "late")):
                    yy = y.clone() if case != "strided" else y.t().contiguous().t()
                    real = min(3, m) if case == "real" else None
                    if case in ("wrong", "late"):
                        rows, rmask, cols = sp.sampled(torch.tensor(step), 3, m, nout)
                        yy[int(rows[min(m, sp.SPOT_ROWS) - 1]), int(cols[2])] *= 1.25
                    a = run(sp.spot_check_torch, x, yy, w, b, step, real)
                    c = run(sp.spot_check_triton, x, yy, w, b, step, real)
                    za, zc = sp._f32(int(a[0][1, 0]) >> 32), sp._f32(int(c[0][1, 0]) >> 32)
                    assert a[0][1, sp.C_CALLS:].tolist() == c[0][1, sp.C_CALLS:].tolist(), (dt, m, case)
                    assert a[0][0].abs().sum() == 0 and c[0][0].abs().sum() == 0
                    if case == "inactive":
                        assert a[0].abs().sum() == 0 and c[0].abs().sum() == 0
                        assert a[1][0].abs().sum() == 0 and c[1][0].abs().sum() == 0
                    elif case in ("clean", "real", "strided"):
                        assert za <= 1.0 and zc <= 1.0 and abs(za - zc) < 0.25, (dt, m, case, za, zc)
                    else:
                        assert za > sp.SPOT_Z and zc > sp.SPOT_Z, (dt, m, za, zc)
                        for col in (sp.C_ROW, sp.C_COL, sp.C_STEP):
                            assert int(a[0][1, col]) & sp.M32 == int(c[0][1, col]) & sp.M32, (dt, m, col)
                        same = [0, 1, 2, 3, 5, 6]                # the kept output (not the score's bits)
                        assert a[1][0][same].tolist() == c[1][0][same].tolist() and int(c[1][0][6]) == 1
                        assert torch.equal(a[1][1], c[1][1]) and torch.equal(a[1][2], c[1][2])
                        assert torch.equal(a[1][3][:2], c[1][3][:2])
                    n += 1
    # inputs scaled by 2^100 and weights by 2^-100 (the same products): float32 squares overflow and underflow, and
    # the kernel's scaled norms must score like the PyTorch check's float64 ones; then a cancelling row whose bound
    # overflows float32 (both count it unjudged, never clean)
    x = (torch.randn(9, 300, generator=g) * 2.0 ** 100).to(torch.bfloat16)
    w = (torch.randn(40, 300, generator=g) * 0.05 * 2.0 ** -100).to(torch.bfloat16)
    y = torch.nn.functional.linear(x, w)
    for step, case in ((4, "clean"), (7, "wrong")):
        yy = y.clone()
        if case == "wrong":
            rows, rmask, cols = sp.sampled(torch.tensor(step), 3, 9, 40)
            yy[int(rows[0]), int(cols[3])] *= 2.0
        a = run(sp.spot_check_torch, x, yy, w, None, step)
        c = run(sp.spot_check_triton, x, yy, w, None, step)
        za, zc = sp._f32(int(a[0][1, 0]) >> 32), sp._f32(int(c[0][1, 0]) >> 32)
        assert a[0][1, sp.C_CALLS:].tolist() == c[0][1, sp.C_CALLS:].tolist(), case
        assert int(a[0][1, sp.C_JUDGED]) == 9 * sp.SPOT_COLS, case
        if case == "clean":
            assert za <= 1.0 and zc <= 1.0 and abs(za - zc) < 0.25, (za, zc)
        else:
            assert za > sp.SPOT_Z and zc > sp.SPOT_Z, (za, zc)
            assert int(a[0][1, sp.C_COL]) & sp.M32 == int(c[0][1, sp.C_COL]) & sp.M32
        n += 1
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[:, :2] = 1e20
    w = torch.zeros(16, 64, dtype=torch.bfloat16)
    w[:, 0], w[:, 1] = 1e20, -1e20
    y = torch.zeros(2, 16, dtype=torch.bfloat16)
    y[0, :] = 5.0                                       # (wrong: the exact outputs are 0)
    a = run(sp.spot_check_torch, x, y, w, None, 4)
    c = run(sp.spot_check_triton, x, y, w, None, 4)
    for st in (a[0], c[0]):
        assert int(st[1, sp.C_JUDGED]) == 0 and int(st[1, sp.C_UNJ]) == 2 * sp.SPOT_COLS, st[1].tolist()
        assert int(st[1, 0]) == 0
    n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
