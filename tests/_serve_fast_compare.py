"""The Triton serving check vs the PyTorch one, in Triton's CPU interpreter.

Run by tests/test_serve.py in a subprocess with TRITON_INTERPRET=1 and SDCGUARD_FAST_CPU_TEST=1.
Prints one JSON line per case and ALL AGREE at the end (exit code 0), or exits 1."""
import json
import sys

import torch

from sdcguard import _serve_fast
from sdcguard.checksum import _split3
from sdcguard.serve import LayerChecksums, ServeGuard, check_static_impl


def run(x, y, cs):
    ref = torch.zeros(1, 2, dtype=torch.int64)
    fast = torch.zeros(1, 2, dtype=torch.int64)
    check_static_impl(x, y, *cs.tensors(True), ref, 0, 6.0, 1e-20)
    _serve_fast.check(x, y, *cs.tensors(True), fast, 0, 6.0, 1e-20)
    return _serve_fast.decode(*ref[0].tolist()), _serve_fast.decode(*fast[0].tolist())


def main():
    ok_all = True
    g = torch.Generator().manual_seed(3)
    cases = [(64, 256, 300, True), (1, 512, 1000, False), (200, 96, 40, True), (7, 300, 5632 // 4, False),
             (33, 130, 129, True), (300, 64, 2000, False), (5, 40, 1, True), (2, 1, 300, True), (1, 1, 1, False)]
    for dt in (torch.bfloat16, torch.float16):
        for split in (False, True):
            for m, k, n, with_bias in cases:
                W = (torch.randn(n, k, generator=g) / max(1, k) ** 0.5).to(dt)
                b = torch.randn(n, generator=g).to(dt) if with_bias else None
                x = torch.randn(m, k, generator=g).to(dt)
                y = torch.nn.functional.linear(x, W, b)
                cs = LayerChecksums(W, b, 128)
                if split:      # the GPU layout: three low-precision parts of the block sums
                    B2 = cs.parts                                   # (on the CPU: plain fp32 block sums)
                    cs.parts = _split3(B2 / cs.scale.view(1, -1), dt).contiguous()
                for corrupt in (None, "big", "nan", "inf", "zero_block", "small", "huge_pair"):
                    yy = y.clone()
                    r, c = (m * 2) // 3, (n * 3) // 5
                    if corrupt == "big":
                        yy[r, c] += 50.0
                    elif corrupt == "nan":
                        yy[r, c] = float("nan")
                    elif corrupt == "inf":
                        yy[r, c] = float("inf")
                    elif corrupt == "zero_block":
                        yy[r, :] = 0
                    elif corrupt == "small":
                        yy[r, c] += yy[r, c].abs() * 0.001        # below the noise: both must pass
                    elif corrupt == "huge_pair" and n >= 2:
                        # two huge outputs of opposite sign where the sign pattern agrees: they cancel in both
                        # checksums; only "larger than the inputs allow" catches them
                        from sdcguard.checksum import _signs
                        s0 = _signs(min(128, n), "cpu", 0)
                        j = next((j for j in range(1, min(128, n)) if s0[j] == s0[0]), None)
                        if j is not None:
                            yy[r, 0], yy[r, j] = 1e4 if dt == torch.float16 else 1e18, \
                                -1e4 if dt == torch.float16 else -1e18
                    (rz, rf, rn), (fz, ff, fn) = run(x, yy, cs)
                    if corrupt in (None, "small"):
                        agree = rz <= 6.0 and fz <= 6.0 and abs(fz - rz) <= 0.05 + 0.02 * rz
                    else:
                        changed = not torch.equal(yy, y)
                        agree = (rz > 6.0) == (fz > 6.0) and (not (rz > 6.0) or rf // 2 == ff // 2) and \
                            (not changed or rz > 6.0 or n == 1 or k == 1)
                    row = {"dtype": str(dt)[6:], "split": split, "m": m, "k": k, "n": n, "bias": with_bias,
                           "corrupt": corrupt, "plain": [rz, rf], "fast": [fz, ff], "agree": bool(agree)}
                    print(json.dumps(row), flush=True)
                    ok_all &= bool(agree)
    # end to end: a guard on a small model uses the fast check after its first-use comparison
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(64, 300), torch.nn.GELU(), torch.nn.Linear(300, 40)).to(torch.bfloat16)
    gd = ServeGuard().attach(model)
    uses = [gd.uses_fast(s) for s in range(len(gd.layers))]
    model(torch.randn(5, 64).to(torch.bfloat16))
    ev = gd.poll(wait=True)
    zs = gd.worst_seen
    e2e = all(uses) and ev == [] and all(z > 0 for z in zs.values())
    print(json.dumps({"case": "guard end to end", "uses_fast": uses, "alerts": len(ev), "worst": zs,
                      "status": {str(k): v for k, v in _serve_fast.STATUS.items()}, "agree": e2e}), flush=True)
    ok_all &= e2e
    print("ALL AGREE" if ok_all else "DISAGREE")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
