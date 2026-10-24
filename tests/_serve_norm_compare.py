"""Run by tests/test_serve_norm.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the norm check's Triton kernel
against its PyTorch version on the CPU. Prints ALL MATCH when every case agrees."""
import torch

from sdcguard import serve_norm as sn


def run(fn, x, y, w, line=sn.NORM_Z):
    h = x.shape[-1]
    st = torch.zeros(2, sn.COLS, dtype=torch.int64)
    cm = torch.zeros(2, sn.CAP_COLS, dtype=torch.int32)
    ci = torch.zeros(2, h + 3)
    co = torch.zeros(2, h + 3)
    sb = torch.full((1,), (5 << 32) | 1234, dtype=torch.int64)
    fn(x, y, w, 1e-6, st, 1, sb, cm, ci, co, line)
    return st, cm, ci, co


def main():
    for dt in (torch.bfloat16, torch.float16):
        ok, why = sn.verify_kernel(torch.device("cpu"), dt)
        assert ok, why
    g = torch.Generator().manual_seed(1)
    n = 0
    for dt in (torch.bfloat16, torch.float16):
        for m, h in ((1, 128), (17, 64), (40, 896), (3, 2048), (2, 4100), (1, 9000)):
            x = (torch.randn(m, h, generator=g) * 3).to(dt)
            w = 1 + 0.3 * torch.randn(h, generator=g)
            xf = x.float()
            y = ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dt) * w.to(dt)).to(dt)
            cases = [("clean", y)]
            yb = y.clone()
            yb[m - 1, h - 1] = yb[m - 1, h - 1] * 2 + 1
            cases.append(("wrong last", yb))
            yi = y.clone()
            yi[m // 2, h // 3] = float("inf")
            cases.append(("inf", yi))
            for name, yy in cases:
                a = run(sn.norm_check_torch, x, yy, w)
                b = run(sn.norm_check_triton, x, yy, w)
                za, zb = sn._f32(int(a[0][1, 0]) >> 32), sn._f32(int(b[0][1, 0]) >> 32)
                assert a[0][1, sn.C_CALLS:].tolist() == b[0][1, sn.C_CALLS:].tolist(), (dt, m, h, name)
                assert abs(za - zb) <= 1e-3 * max(1.0, za) or (za == zb == float("inf")), (dt, m, h, name, za, zb)
                assert a[0][0].tolist() == [0] * sn.COLS and b[0][0].tolist() == [0] * sn.COLS
                if name != "clean":
                    for col in (sn.C_ROW, sn.C_COL, sn.C_STEP):
                        assert int(a[0][1, col]) & sn.M32 == int(b[0][1, col]) & sn.M32, (dt, m, h, name, col)
                    assert int(b[0][1, sn.C_STEP]) & sn.M32 == 1234
                    assert a[1][1, :3].tolist() == b[1][1, :3].tolist(), (a[1][1], b[1][1])
                    ka, kb = sn._f32(int(a[1][1, 3])), sn._f32(int(b[1][1, 3]))   # (the mean's order differs)
                    assert abs(ka - kb) <= 1e-3 * max(1.0, ka) or ka == kb, (ka, kb)
                    assert torch.equal(a[2][1], b[2][1]) and torch.equal(a[3][1], b[3][1])
                    assert a[1][0].tolist() == [0] * 4 and b[2][0].abs().sum() == 0
                else:
                    assert za <= 4.05 and b[1][1, 0] == 0
                n += 1
            # a strided view (a per-head norm on q split from qkv)
            if h <= 2048:
                big = (torch.randn(m, 3 * h, generator=g) * 3).to(dt)
                xs = big[:, h:2 * h]
                xf = xs.float()
                ys = ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)).to(dt) * w.to(dt)).to(dt)
                a, b = run(sn.norm_check_torch, xs, ys, w), run(sn.norm_check_triton, xs, ys, w)
                assert a[0][1, sn.C_CALLS:].tolist() == b[0][1, sn.C_CALLS:].tolist()
                n += 1
    for dt in (torch.bfloat16, torch.float16):                     # rows near fp32 overflow: unjudged on both
        x = torch.full((3, 512), 3e18 if dt == torch.bfloat16 else 6e4, dtype=dt)
        y = torch.zeros(3, 512, dtype=dt)
        w = torch.ones(512)
        a, b = run(sn.norm_check_torch, x, y, w), run(sn.norm_check_triton, x, y, w)
        assert a[0][1, sn.C_CALLS:].tolist() == b[0][1, sn.C_CALLS:].tolist(), (a[0][1], b[0][1])
        n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
