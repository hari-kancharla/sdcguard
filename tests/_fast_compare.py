"""Run the Triton check and the plain PyTorch check on the same inputs and compare.

Run by tests/test_fast.py in a subprocess with TRITON_INTERPRET=1 (Triton's CPU interpreter),
and by the GPU self-test on CUDA. Prints one JSON line per case.
"""
import json
import sys

import torch

from sdcguard import checksum as cs


def both(A, B, C, bias=None, alpha=1.0, beta=1.0):
    cs.USE_FAST = True
    fast = cs.check_matmul(A, B, C, bias, alpha=alpha, beta=beta)
    cs.USE_FAST = False
    plain = cs.check_matmul(A, B, C, bias, alpha=alpha, beta=beta)
    cs.USE_FAST = True
    return fast, plain


def cases(device):
    g = torch.Generator().manual_seed(0)

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(device=device, dtype=dtype)

    out = []
    for dt in (torch.bfloat16, torch.float16):
        A, B = r(37, 50, dtype=dt), r(50, 300, dtype=dt)
        out.append((f"mm {dt} n=300 (partial last block)", A, B, A @ B, None, 1.0, 1.0))
        A, B = r(64, 96, dtype=dt), r(96, 40, dtype=dt)
        out.append((f"mm {dt} n=40 (< one block)", A, B, A @ B, None, 1.0, 1.0))
    A, B, bias = r(64, 96), r(96, 256), r(256)
    out.append(("addmm with bias", A, B, torch.addmm(bias, A, B), bias, 1.0, 1.0))
    out.append(("addmm alpha=0.5 beta=2", A, B, torch.addmm(bias, A, B, beta=2.0, alpha=0.5), bias, 0.5, 2.0))
    X, W = r(48, 64), r(200, 64)
    out.append(("linear forward: B = W^T (strided view)", X, W.t(), X @ W.t(), None, 1.0, 1.0))
    dY = r(48, 200)
    out.append(("dW = dY^T X (strided A)", dY.t(), X, dY.t() @ X, None, 1.0, 1.0))
    A3, B3 = r(3, 20, 30), r(3, 30, 130)
    out.append(("bmm batch 3", A3, B3, torch.bmm(A3, B3), None, 1.0, 1.0))
    B1 = r(30, 130)
    C = torch.matmul(A3, B1)
    out.append(("batched A, shared B", A3, B1.unsqueeze(0), C, None, 1.0, 1.0))
    A, B = r(16, 3000), r(3000, 64)
    out.append(("long sum over k=3000 (split across programs)", A, B, A @ B, None, 1.0, 1.0))
    A, B = r(64, 96, scale=1e-12), r(96, 256, scale=1e-12)
    out.append(("tiny values (1e-24 products)", A, B, A @ B, None, 1.0, 1.0))
    A, B = r(70, 33), r(33, 1500)
    out.append(("n=1500 (24 checksum columns per tile)", A, B, A @ B, None, 1.0, 1.0))
    A, B = r(70, 33), r(33, 2600)
    out.append(("n=2600 (64 checksum columns per tile)", A, B, A @ B, None, 1.0, 1.0))
    A, B, bias = r(40, 50), r(50, 9000), r(9000)
    out.append(("n=9000 with bias (128 per tile, 2 tiles across)", A, B, torch.addmm(bias, A, B), bias, 1.0, 1.0))
    # sizes of 1 (Triton compiles an argument equal to 1 as a constant: every kernel must allow it)
    for dt in (torch.bfloat16, torch.float16):
        A, B = r(1, 96, dtype=dt), r(96, 300, dtype=dt)
        out.append((f"{dt} one output row (m=1)", A, B, A @ B, None, 1.0, 1.0))
        A, B = r(64, 96, dtype=dt), r(96, 1, dtype=dt)
        out.append((f"{dt} one output column (n=1)", A, B, A @ B, None, 1.0, 1.0))
        A, B, bias = r(40, 1, dtype=dt), r(1, 300, dtype=dt), r(300, dtype=dt)
        out.append((f"{dt} inner size 1 with a bias (k=1)", A, B, torch.addmm(bias, A, B), bias, 1.0, 1.0))
        A, B = r(1, 1, dtype=dt), r(1, 1, dtype=dt)
        out.append((f"{dt} 1 x 1 x 1", A, B, A @ B, None, 1.0, 1.0))
    A3, B3 = r(4, 1, 64), r(4, 64, 200)
    out.append(("bmm, one row per batch", A3, B3, torch.bmm(A3, B3), None, 1.0, 1.0))
    # Inf/NaN outputs the inputs explain are not corruption (fp16 loss scaling overflows on purpose)
    A, B = r(48, 64, dtype=torch.float16), r(64, 300, dtype=torch.float16)
    A[5] = 30000.0                                         # row 5 overflows fp16 (true values ~2e5)
    out.append(("fp16 overflow from large inputs (not corruption)", A, B, A @ B, None, 1.0, 1.0))
    A = r(48, 64)
    A[7, 3] = float("nan")                                 # NaN input: row 7 of the output is NaN
    B = r(64, 300)
    out.append(("NaN in an input (not corruption)", A, B, A @ B, None, 1.0, 1.0))
    A, B = r(32, 64, scale=1e10), r(64, 200, scale=1e10)
    out.append(("bf16 outputs ~1e21 from large inputs (too big to square, not corruption)", A, B, A @ B,
                None, 1.0, 1.0))
    # equal outputs round the same way (B with identical columns): not corruption
    for dt in (torch.bfloat16, torch.float16):
        A = r(64, 768, dtype=dt)
        B = torch.ones(768, 512, device=device, dtype=dt)
        out.append((f"{dt} B = ones (identical columns)", A, B, A @ B, None, 1.0, 1.0))
    A = r(64, 4096)
    B = r(4096, 1).expand(4096, 512)
    out.append(("B = an expanded vector (identical columns, stride 0)", A, B, A @ B, None, 1.0, 1.0))
    # extreme but legitimate bf16 values: the fp32 checksum arithmetic overflows, the outputs don't
    out.append(("bf16 outputs 1e37 (block sums overflow fp32)", torch.ones(4, 1, device=device, dtype=torch.bfloat16),
                torch.full((1, 128), 1e37, device=device, dtype=torch.bfloat16), None, None, 1.0, 1.0))
    B = torch.full((2, 128), 1e37, device=device, dtype=torch.bfloat16)
    B[1] = -1e37
    out.append(("bf16 B's block sums overflow fp32, outputs ~0", torch.full((4, 2), 1e-30, device=device,
                dtype=torch.bfloat16), B, None, None, 1.0, 1.0))
    out.append(("bf16 |a_row|^2 overflows fp32", torch.full((4, 8), 1e20, device=device, dtype=torch.bfloat16),
                r(8, 256, scale=1e-12), None, None, 1.0, 1.0))
    out.append(("bf16 checksum column^2 overflows fp32", r(4, 8, scale=1e-15), r(8, 128, scale=1e18), None,
                None, 1.0, 1.0))
    B = r(8, 256)
    B[:, 128:] = 0
    out.append(("bf16 huge A and a zero block of B", torch.full((4, 8), 1e20, device=device, dtype=torch.bfloat16),
                B, None, None, 1.0, 1.0))
    # small fp16 values: below 6e-5 rounding is a fixed step (6e-8), whatever the size
    for sc in (1e-3, 3e-4):
        out.append((f"fp16 tiny outputs (inputs ~{sc:g})", r(128, 256, dtype=torch.float16, scale=sc),
                    r(256, 300, dtype=torch.float16, scale=sc), None, None, 1.0, 1.0))
    # C=None above: the exact product rounded once (what a correct GPU matmul returns)
    return [(nm, A, B, (A.float() @ B.float()).to(A.dtype) if C is None else C, bias, al, be)
            for nm, A, B, C, bias, al, be in out]


def corrupted(device):
    g = torch.Generator().manual_seed(1)
    A = torch.randn(64, 96, generator=g).to(device=device, dtype=torch.bfloat16)
    B = torch.randn(96, 300, generator=g).to(device=device, dtype=torch.bfloat16)
    res = []
    C = A @ B
    C[17, 130:162:4] = 0                                   # zero 8 values in row 17, block 1
    res.append(("zeroed chunk", A, B, C, True))
    C = A @ B
    C[5, 290] = float("nan")                               # NaN in the partial last block
    res.append(("NaN", A, B, C, True))
    C = A @ B
    C[40, 7] = C[40, 7] + 40                               # one value off by 40 in block 0
    res.append(("one value off by 40", A, B, C, True))
    C = A @ B
    C[40, 7] = C[40, 7] * 2 + 3                            # a small change: below the alarm line
    res.append(("small change (should pass in both)", A, B, C, False))
    A2 = torch.randn(70, 40, generator=g).to(device=device, dtype=torch.bfloat16)
    B2 = torch.randn(40, 9000, generator=g).to(device=device, dtype=torch.bfloat16)
    C = A2 @ B2
    C[66, 8500] = 0                                        # second column tile, second row tile
    res.append(("wide output, error far from the start", A2, B2, C, True))
    A3 = torch.randn(3, 20, 30, generator=g).to(device=device, dtype=torch.bfloat16)
    B3 = torch.randn(3, 30, 130, generator=g).to(device=device, dtype=torch.bfloat16)
    C = torch.bmm(A3, B3)
    C[2, 11, 129] = 1000                                   # last batch, last (1-wide) block
    res.append(("bmm, error in batch 2", A3, B3, C, True))
    A16 = torch.randn(48, 64, generator=g).to(device=device, dtype=torch.float16)
    B16 = torch.randn(64, 300, generator=g).to(device=device, dtype=torch.float16)
    C = A16 @ B16
    C[9, 140] = float("inf")                               # fp16, small inputs: Inf can't be real
    res.append(("fp16 Inf from small inputs", A16, B16, C, True))
    A16b = A16.clone()
    A16b[2] = 30000.0                                      # row 2 overflows for real...
    C = A16b @ B16
    C[30, 260] = float("nan")                              # ...and row 30 gets a NaN it can't have
    res.append(("fp16 real overflow in one row, NaN corruption in another", A16b, B16, C, True))
    C = A @ B
    C[9, 50] = 1e30                                        # exponent flip: a huge value from small inputs
    res.append(("bf16 value 1e30 from small inputs", A, B, C, True))
    At = (torch.randn(128, 256, generator=g) * 1e-3).to(device=device, dtype=torch.float16)
    Bt = (torch.randn(256, 300, generator=g) * 1e-3).to(device=device, dtype=torch.float16)
    C = (At.float() @ Bt.float()).half()
    C[17, 40] += 0.01                                      # fp16, tiny outputs: a real change is still caught
    res.append(("fp16 tiny outputs, one changed by 0.01", At, Bt, C, True))
    return res


def guard_end_to_end(device):
    """A training step under SDCGuard with a faulty layer: both paths must flag the same outputs."""
    import torch.nn as nn
    from sdcguard import FaultInjector, SDCGuard
    from sdcguard.inject import FaultModel

    results = {}
    for fast in (True, False):
        cs.USE_FAST = fast
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 160), nn.GELU(), nn.Linear(160, 64), nn.GELU(),
                              nn.Linear(64, 2600), nn.GELU(), nn.Linear(2600, 32)).to(device)
        x = torch.randn(24, 64, device=device)
        inj = FaultInjector(rate=1.0, seed=3, fault_model=FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0),
                            dtypes=(torch.bfloat16,)).arm_on([model[2]])
        g = SDCGuard()
        with inj, g, torch.autocast(device if device != "cuda" else "cuda", dtype=torch.bfloat16):
            model(x).float().pow(2).mean().backward()
        results[fast] = (g.stats["checked"], sorted((e.detail["where"], e.detail["row"], e.detail["col_block"])
                                                     for e in g.log.events))
    cs.USE_FAST = True
    same = results[True] == results[False] and len(results[True][1]) > 0
    print(json.dumps({"case": "guard end to end (mixed widths, one faulty layer)", "fast": results[True],
                      "plain": results[False], "agree": same}))
    return same


def inplace_safety(device):
    """Inputs and outputs changed in place, or freed, after the matmul must not cause false alarms:
    every check reads A, B and C right after its matmul (training code does all of this: in-place
    activations, gradient accumulation, FSDP freeing gathered weights, reused buffers)."""
    from sdcguard import SDCGuard
    g = SDCGuard()
    torch.manual_seed(1)
    x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    W = torch.randn(64, 200, device=device, dtype=torch.bfloat16)
    with g:
        y = x @ W
        y.mul_(3).add_(1)               # changes C in place, like an in-place activation would
        z = y @ W.t()
        y.zero_()                       # changes the A of the second matmul
        W2 = W.clone()
        z2 = x @ W2
        W2.untyped_storage().resize_(0)          # frees B's memory, like FSDP after a layer's forward
        junk = torch.full((64, 200), 7.0, device=device, dtype=torch.bfloat16)   # may reuse that memory
    ok = g.stats["alerts"] == 0 and g.stats["checked"] == 3
    print(json.dumps({"case": "in-place changes and freed inputs after the matmul", "stats": g.stats,
                      "agree": ok}))
    del z, z2, junk
    return ok


def first_use_check_ignores_other_hooks(device):
    """The first-use comparison runs inside the guard. A fault injector wrapped around the guard
    must not see (or corrupt) the comparison's own matmuls."""
    from sdcguard import FaultInjector, SDCGuard, _group
    from sdcguard.inject import FaultModel
    dev = torch.device(device)
    saved = (dict(_group.STATUS), dict(_group._BROKEN), _group.FALLBACKS[0], _group.USED[0])
    try:
        _group.STATUS.clear()
        _group._BROKEN.clear()
        inj = FaultInjector(rate=1.0, seed=3, fault_model=FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0),
                            dtypes=(torch.bfloat16,))
        g = SDCGuard()
        x = torch.randn(16, 64, device=dev, dtype=torch.bfloat16)
        w = torch.randn(64, 200, device=dev, dtype=torch.bfloat16)
        with inj, g:
            x @ w
        st = _group.status(dev, torch.bfloat16)
        ok = st is not None and st[0] and len(inj.events) == 1 and _group.USED[0] == saved[3] + 1
    finally:
        _group.STATUS.clear(); _group.STATUS.update(saved[0])
        _group._BROKEN.clear(); _group._BROKEN.update(saved[1])
        _group.FALLBACKS[0] = saved[2]
    print(json.dumps({"case": "first-use comparison inside a guard wrapped by a fault injector",
                      "status": st, "injected": len(inj.events), "agree": ok}))
    return ok


def side_stream(device):
    """Matmuls on a side CUDA stream: the read-back must wait for their checks."""
    if device != "cuda":
        return True
    from sdcguard import SDCGuard
    g = SDCGuard()
    side = torch.cuda.Stream()
    x = torch.randn(4096, 1024, device=device, dtype=torch.bfloat16)
    w = torch.randn(1024, 4096, device=device, dtype=torch.bfloat16)
    side.wait_stream(torch.cuda.current_stream())
    with g:
        with torch.cuda.stream(side):
            ys = [x @ w for _ in range(8)]
    torch.cuda.synchronize()
    ok = g.stats["checked"] == 8 and g.stats["alerts"] == 0
    print(json.dumps({"case": "matmuls on a side CUDA stream", "stats": g.stats, "agree": ok}))
    del ys
    return ok


def out_of_memory(device):
    """A check that doesn't fit in memory is skipped and counted; training goes on."""
    from sdcguard import SDCGuard, _group
    real_add = _group.Group.add

    def oom(self, *a, **k):
        raise torch.cuda.OutOfMemoryError("pretend the GPU is full")
    _group.Group.add = oom
    try:
        g = SDCGuard()
        x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
        w = torch.randn(64, 200, device=device, dtype=torch.bfloat16)
        with g:
            x @ w
    finally:
        _group.Group.add = real_add
    ok = g.stats["skipped_oom"] == 1 and g.stats["checked"] == 0 and not _group._BROKEN
    print(json.dumps({"case": "out of memory during a check", "stats": g.stats, "agree": ok}))
    return ok


def probe_rejects_a_wrong_kernel(device):
    """If the Triton check ever answers differently from the plain check (say, a Triton version
    that compiles a kernel differently), the first-use comparison must switch it off."""
    from sdcguard import _group
    dev = torch.device(device)
    real_row = _group.Group.row
    _group.Group.row = lambda self, idx, r: (lambda z, b, row, blk, d: (z, b, row + 1, blk, d))(*real_row(self, idx, r))
    saved = (dict(_group.STATUS), dict(_group._BROKEN), _group.FALLBACKS[0], list(_group.WHY))
    try:
        _group.STATUS.clear()
        _group._BROKEN.clear()
        used = _group.available(torch.empty(1, 1, device=dev, dtype=torch.bfloat16))
        why = _group.status(dev, torch.bfloat16)[1]
    finally:
        _group.Group.row = real_row
        _group.STATUS.clear(); _group.STATUS.update(saved[0])
        _group._BROKEN.clear(); _group._BROKEN.update(saved[1])
        _group.FALLBACKS[0] = saved[2]
        _group.WHY[:] = saved[3]
    ok = not used
    print(json.dumps({"case": "first-use comparison switches off a kernel that disagrees", "why": why, "agree": ok}))
    return ok


def main(device="cpu"):
    from sdcguard import _group
    ok = True
    for name, A, B, C, bias, alpha, beta in cases(device):
        fb = _group.FALLBACKS[0]
        f, p = both(A, B, C, bias, alpha, beta)
        agree = f.ok and p.ok and abs(f.max_z - p.max_z) <= 0.02 + 0.01 * p.max_z
        if _group.FALLBACKS[0] != fb:          # the Triton check failed here: the plain one answered
            agree = False
            print(json.dumps({"case": name, "fallback": _group.WHY[-1:]}))
        ok &= agree
        print(json.dumps({"case": name, "fast_z": f.max_z, "plain_z": p.max_z, "both_clean": f.ok and p.ok,
                          "agree": agree}))
    for name, A, B, C, flagged in corrupted(device):
        fb = _group.FALLBACKS[0]
        f, p = both(A, B, C)
        same = (f.ok == p.ok == (not flagged)) and f.row == p.row and f.col_block == p.col_block \
            and f.nonfinite == p.nonfinite and f.batch == p.batch and _group.FALLBACKS[0] == fb
        ok &= same
        print(json.dumps({"case": name, "fast": [f.ok, f.batch, f.row, f.col_block, f.nonfinite, f.max_z],
                          "plain": [p.ok, p.batch, p.row, p.col_block, p.nonfinite, p.max_z], "agree": same}))
    ok &= guard_end_to_end(device)
    ok &= inplace_safety(device)
    ok &= side_stream(device)
    ok &= out_of_memory(device)
    ok &= first_use_check_ignores_other_hooks(device)
    print(json.dumps({"fast_fallbacks": _group.FALLBACKS[0], "why": _group.WHY}))
    ok &= _group.FALLBACKS[0] == 0
    ok &= probe_rejects_a_wrong_kernel(device)
    print("ALL AGREE" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "cpu"))
