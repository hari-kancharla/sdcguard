"""Run by tests/test_serve_moe.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the MoE check's Triton kernels
against its PyTorch version on the CPU. Prints ALL MATCH when every case agrees."""
import torch

from sdcguard import serve_moe as sm


def run(fn, x, y, w, ids, w13, w2, emap, step, nl=3, idx=1, act=0, routed_first=False):
    h, k = x.shape[1], ids.shape[1]
    st = torch.zeros(2, sm.COLS, dtype=torch.int64)
    cm = torch.zeros(2, sm.CAP_COLS, dtype=torch.int32)
    cx, cy = torch.zeros(2, h + 5), torch.zeros(2, h + 5)
    ci, cg = torch.zeros(2, 16, dtype=torch.int64), torch.zeros(2, 16)
    sb = torch.full((1,), (9 << 32) | step, dtype=torch.int64)
    fn(x, y, w, ids, w13, w2, emap, act, routed_first, st, 1, sb, nl, idx, cm, cx, cy, ci, cg, sm.MOE_Z)
    return st, cm, cx, cy, ci, cg


def main():
    for dt in (torch.bfloat16, torch.float16):
        ok, why = sm.verify_kernel(torch.device("cpu"), dt, 0)
        assert ok, (dt, why)
    g = torch.Generator().manual_seed(3)
    n = 0
    for dt in (torch.bfloat16, torch.float16):
        for E, H, I, K, M in ((4, 64, 32, 2, 1), (8, 300, 96, 3, 9), (6, 128, 200, 4, 33)):
            w13 = (torch.randn(E, 2 * I, H, generator=g) * 0.05).to(dt)
            w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(dt)
            x = torch.randn(M, H, generator=g).to(dt)
            w, ids = torch.topk(torch.softmax(torch.randn(M, E, generator=g), -1), K, -1)
            ids = ids.to(torch.int32)
            for emap in (None, torch.tensor([0] + [-1] + list(range(1, E - 1)), dtype=torch.int32)):
                ww13, ww2 = (w13, w2) if emap is None else (w13[:E - 1].contiguous(), w2[:E - 1].contiguous())
                y = sm._reference_layer(x, ids, w, ww13, ww2, emap, 0, dt)
                for step, case in ((4, "clean"), (7, "wrong"), (5, "inactive")):
                    yy = y.clone()
                    if case == "wrong":
                        rows = sm.sample_rows(torch.tensor(step), M, "cpu")
                        yy[int(rows[-1]), H - 1] = yy[int(rows[-1]), H - 1] * 8 + 3
                    a = run(sm.moe_check_torch, x, yy, w, ids, ww13, ww2, emap, step)
                    b = run(sm.moe_check_triton, x, yy, w, ids, ww13, ww2, emap, step)
                    za, zb = sm._f32(int(a[0][1, 0]) >> 32), sm._f32(int(b[0][1, 0]) >> 32)
                    assert a[0][1, sm.C_CALLS:].tolist() == b[0][1, sm.C_CALLS:].tolist(), (dt, M, case)
                    assert a[0][0].abs().sum() == 0 and b[0][0].abs().sum() == 0
                    if case == "inactive":
                        assert a[0].abs().sum() == 0 and b[0].abs().sum() == 0, (dt, M)
                    elif case == "clean":
                        assert za < 1.0 and zb < 1.0 and abs(za - zb) < 0.5, (dt, M, za, zb)
                    else:
                        assert za > 6 and zb > 6, (dt, M, za, zb)
                        for col in (sm.C_ROW, sm.C_COL, sm.C_STEP):
                            assert int(a[0][1, col]) & sm.M32 == int(b[0][1, col]) & sm.M32, (dt, M, col)
                        assert a[1][1, :3].tolist() == b[1][1, :3].tolist()
                        assert torch.equal(a[2][1], b[2][1]) and torch.equal(a[3][1], b[3][1])
                        assert torch.equal(a[4][1, :K], b[4][1, :K]) and torch.equal(a[5][1, :K], b[5][1, :K])
                    n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
