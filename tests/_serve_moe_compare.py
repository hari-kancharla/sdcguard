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
                for step, case in ((4, "clean"), (7, "wrong"), (5, "inactive"), (7, "padding"), (4, "partial")):
                    yy, cids = y.clone(), ids
                    rows = sm.sample_rows(torch.tensor(step), min(9, M), "cpu", 3)    # (9 real rows packed in sb)
                    if case == "wrong":
                        yy[int(rows[-1]), H - 1] = yy[int(rows[-1]), H - 1] * 8 + 3
                    elif case == "padding":                     # the row vLLM marked padding: garbage, not judged
                        cids = ids.clone()
                        cids[int(rows[-1])] = -1
                        yy[int(rows[-1])] = 1000.0
                    elif case == "partial":                     # one expert of the last real row missing
                        cids = ids.clone()
                        cids[min(9, M) - 1, K - 1] = -1
                        yy = sm._reference_layer(x, cids, w, ww13, ww2, emap, 0, dt)
                    a = run(sm.moe_check_torch, x, yy, w, cids, ww13, ww2, emap, step)
                    b = run(sm.moe_check_triton, x, yy, w, cids, ww13, ww2, emap, step)
                    za, zb = sm._f32(int(a[0][1, 0]) >> 32), sm._f32(int(b[0][1, 0]) >> 32)
                    assert a[0][1, sm.C_CALLS:].tolist() == b[0][1, sm.C_CALLS:].tolist(), (dt, M, case)
                    assert a[0][0].abs().sum() == 0 and b[0][0].abs().sum() == 0
                    if case == "inactive":
                        assert a[0].abs().sum() == 0 and b[0].abs().sum() == 0, (dt, M)
                    elif case in ("clean", "padding", "partial"):
                        assert za < 1.0 and zb < 1.0 and abs(za - zb) < 0.5, (dt, M, za, zb)
                        routed = sum(case != "padding" or int(r) != int(rows[-1]) for r in rows)
                        assert int(a[0][1, sm.C_ROWS]) == routed, (dt, M, case, a[0][1].tolist(), routed)
                        # the real rows are given (9, packed in sb): every one with experts -1 is counted, and the
                        # last of them kept with the step
                        want = {"clean": [0, 0, 0], "padding": [1, 0, (step << 32) | (int(rows[-1]) + 1)],
                                "partial": [0, 1, (step << 32) | min(9, M)]}[case]
                        got = [int(v) for v in a[0][1, sm.C_UNROUTED:].tolist()]
                        assert got == want, (dt, M, case, got, want)
                    else:
                        assert za > 6 and zb > 6, (dt, M, za, zb)
                        for col in (sm.C_ROW, sm.C_COL, sm.C_STEP):
                            assert int(a[0][1, col]) & sm.M32 == int(b[0][1, col]) & sm.M32, (dt, M, col)
                        assert a[1][1, :3].tolist() == b[1][1, :3].tolist()
                        assert torch.equal(a[2][1], b[2][1]) and torch.equal(a[3][1], b[3][1])
                        assert torch.equal(a[4][1, :K], b[4][1, :K]) and torch.equal(a[5][1, :K], b[5][1, :K])
                    n += 1
    # the routing-validity scan over more real rows than one block of the kernel's loop (ROUTE_BLOCK)
    E, H, I, K = 6, 64, 32, 2
    w13 = (torch.randn(E, 2 * I, H, generator=g) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(torch.bfloat16)
    for M, real, bad in ((150, 130, (3, 70, 129)), (200, 65, (64,)), (300, 300, (0, 150, 299))):
        x = torch.randn(M, H, generator=g).to(torch.bfloat16)
        w, ids = torch.topk(torch.softmax(torch.randn(M, E, generator=g), -1), K, -1)
        ids = ids.to(torch.int32)
        for i, r in enumerate(bad):
            if i % 2:
                ids[r, 0] = -1                                  # missing one expert
            else:
                ids[r] = -1                                     # no experts
        y = sm._reference_layer(x, ids, w, w13, w2, None, 0, torch.bfloat16)
        res = []
        for fn in (sm.moe_check_torch, sm.moe_check_triton):
            st = torch.zeros(2, sm.COLS, dtype=torch.int64)
            cm = torch.zeros(2, sm.CAP_COLS, dtype=torch.int32)
            cx, cy = torch.zeros(2, H), torch.zeros(2, H)
            ci, cg = torch.zeros(2, 16, dtype=torch.int64), torch.zeros(2, 16)
            sb = torch.full((1,), sm.pack_step(4, real), dtype=torch.int64)
            fn(x, y, w, ids, w13, w2, None, 0, False, st, 1, sb, 3, 1, cm, cx, cy, ci, cg, sm.MOE_Z)
            res.append([int(v) for v in st[1, sm.C_UNROUTED:].tolist()])
        inside = [r for r in bad if r < real]
        want = [sum(1 for i, r in enumerate(bad) if r < real and not i % 2),
                sum(1 for i, r in enumerate(bad) if r < real and i % 2), (4 << 32) | (max(inside) + 1)]
        assert res[0] == res[1] == want, (M, real, res, want)
        n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
