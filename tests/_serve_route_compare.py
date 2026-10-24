"""Run by tests/test_serve_route.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the routing check's Triton
kernel against its PyTorch version on the CPU. Prints ALL MATCH when every case agrees."""
import torch

from sdcguard import serve_route as sr

CNT = [sr.C_CALLS, sr.C_SET, sr.C_WEIGHT, sr.C_ROWS, sr.C_LAST]


def run(fn, lg, w, ids, pol, step, real, nl=3, idx=1):
    st = torch.zeros(2, sr.COLS, dtype=torch.int64)
    sb = torch.full((1,), (5 << 32) | step, dtype=torch.int64)
    rb = torch.full((1,), real, dtype=torch.int64)
    caps = sr.new_caps("cpu", 2, 300)
    fn(lg, w, ids, pol[0], pol[1], pol[2], pol[3], pol[4], pol[5], pol[6], rb, sb, st, 1, nl, idx, *caps)
    return st, caps


def main():
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        ok, why = sr.verify_kernel(torch.device("cpu"), dt)
        assert ok, (dt, why)
    g = torch.Generator().manual_seed(5)
    n = 0
    for dt in (torch.float32, torch.bfloat16):
        for E, pol in ((8, (2, 0, True, None, 0, 0, 1.0)), (60, (4, 0, True, None, 0, 0, 1.0)),
                       (128, (8, 1, False, None, 0, 0, 1.0)), (64, (6, 1, True, "bias", 0, 0, 2.5)),
                       (256, (8, 1, True, "bias", 8, 4, 2.5)), (64, (6, 0, False, None, 4, 2, 1.0)),
                       (96, (4, 1, True, "bias", 3, 2, 1.0))):
            M = 11
            bias = torch.randn(E, generator=g) * 0.03 if pol[3] == "bias" else None
            pol = (*pol[:3], bias, *pol[4:])
            lg = (torch.randn(M, E, generator=g) * 2).to(dt)
            w, ids = sr._reference_route(lg.float(), *pol)
            for case, step in (("clean", 4), ("expert", 7), ("weight", 10), ("dup", 13), ("pad", 16),
                               ("inactive", 5)):
                ww, ii = w.clone(), ids.clone()
                real = M
                if case == "expert":
                    ii[:, -1] = lg.float().argmin(-1).to(ii.dtype)
                elif case == "weight":
                    ww[:, 0] = ww[:, 0] * 0.5
                elif case == "dup":
                    ii[:, 1] = ii[:, 0]
                elif case == "pad":
                    ii[:] = -1
                a, ka = run(sr.route_check_torch, lg, ww, ii, pol, step, real)
                b, kb = run(sr.route_check_triton, lg, ww, ii, pol, step, real)
                assert all(torch.equal(x, y) for x, y in zip(ka, kb)), (dt, E, case, ka[0], kb[0])
                assert int(ka[0][1, 0]) == int(case in ("expert", "weight", "dup")) and ka[0][0].abs().sum() == 0
                if int(ka[0][1, 0]):
                    m = ka[0][1].tolist()
                    assert m[3] == (sr.KIND_WEIGHT if case == "weight" else sr.KIND_SET) and m[4] == E
                    assert m[5] == sr.DTYPE_CODE[dt] and torch.equal(ka[1][1, :E], lg[m[1]].float())
                    kp = pol[0]
                    assert torch.equal(ka[2][1, :kp], ii[m[1]].long()) and torch.equal(ka[3][1, :kp], ww[m[1]].float())
                    rc = sr.recheck(ka[1][1, :E], ka[2][1, :kp], ka[3][1, :kp],
                                    dict(zip(("k", "scoring", "renormalize", "bias", "groups", "topk_group", "scale"),
                                             pol)), dt)
                    assert rc["verdict"] == "output_wrong", (dt, E, case, rc)
                ca, cb = [int(a[1, c]) for c in CNT], [int(b[1, c]) for c in CNT]
                assert ca == cb, (dt, E, pol[:3], pol[4:], case, ca, cb)
                assert a[0].abs().sum() == 0 and b[0].abs().sum() == 0, "another router's row changed"
                expect = {"clean": (0, 0), "expert": (2, 0), "weight": (0, 2), "dup": (2, 0), "pad": (0, 0),
                          "inactive": (0, 0)}[case]
                assert (ca[1], ca[2]) == expect, (dt, E, case, ca)
                rows = {"inactive": 0, "pad": 0}.get(case, sr.SAMPLES)
                assert ca[3] == rows, (dt, E, case, ca)
                wa, wb = sr._f32(int(a[1, sr.C_WDIFF])), sr._f32(int(b[1, sr.C_WDIFF]))
                assert abs(wa - wb) <= 1e-5 + 1e-3 * abs(wa), (case, wa, wb)
                n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
