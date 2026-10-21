"""The one-pass fingerprint kernel (sdcguard/_fingerprint.py) against the PyTorch sums, run in Triton's CPU
interpreter by tests/test_recompute.py, or directly on a GPU:

    TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1 python tests/_fingerprint_compare.py cpu
    python tests/_fingerprint_compare.py cuda

Prints ALL AGREE when every case matches bit for bit and the recompute check, running on the kernel, still
catches a fault and names its block.
"""
import sys

import torch
import torch.nn as nn
import torch.utils.checkpoint as tuc

from sdcguard import _fingerprint
from sdcguard.recompute import RecomputeCheck, _fingerprint_torch, fingerprint


def bits_of(t):
    x = t.contiguous().view(-1)
    return x.view(torch.int16) if x.element_size() % 2 == 0 else x.view(torch.int8)


def main(device: str) -> int:
    dev = torch.device(device)
    torch.manual_seed(0)
    bad = []
    cases = []
    # (the last size is past one step of the PyTorch version, recompute._FP_CHUNK)
    for n in (1, 7, 250, 251, 252, 251 * 3, 251 * 1000 + 7, 4096, 4097, 300_001, (1 << 20) + 5):
        cases.append((f"bf16 n={n}", torch.randn(n, dtype=torch.bfloat16)))
    cases += [("fp16 2-D", torch.randn(37, 129, dtype=torch.float16)),
              ("fp32 3-D", torch.randn(3, 17, 61)),
              ("uint8", torch.randint(0, 256, (251 * 4 + 5,), dtype=torch.uint8)),
              ("bool", torch.rand(1000) > 0.5),
              ("int64", torch.randint(-2**40, 2**40, (999,), dtype=torch.int64)),
              ("transposed bf16", torch.randn(64, 300, dtype=torch.bfloat16).t()),
              ("extremes", torch.tensor([65504.0, -65504.0, float("inf"), float("-inf"), float("nan")] * 120,
                                        dtype=torch.float16))]
    for name, t in cases:
        t = t.to(dev)
        fast = _fingerprint.fingerprint_bits(bits_of(t)).cpu()
        slow = _fingerprint_torch(bits_of(t)).cpu()
        if not torch.equal(fast, slow):
            bad.append(f"{name}: kernel {fast.tolist()} vs torch {slow.tolist()}")
        if not torch.equal(fingerprint(t).cpu(), slow):
            bad.append(f"{name}: fingerprint() didn't use the same sums")
    one = torch.randn(251 * 50 + 3, dtype=torch.bfloat16, device=dev)
    for i in (0, 250, 251, 251 * 50 + 2):                 # a one-element change always changes the sums
        two = one.clone()
        two[i] = two[i] * 1.5 + 1
        if torch.equal(fingerprint(one).cpu(), fingerprint(two).cpu()):
            bad.append(f"change at {i} not seen")
    swap = one.clone()
    swap[[3, 900]] = swap[[900, 3]]
    if torch.equal(fingerprint(one).cpu(), fingerprint(swap).cpu()) and one[3] != one[900]:
        bad.append("swap not seen")
    st = _fingerprint.status(dev)        # ("cuda" here, "cuda:0" where fingerprint() ran: the same device)
    if not (st and st[0]):
        bad.append(f"kernel not in use: {st}")

    # the recompute check on the kernel: clean, then a fault in the recompute of the second block
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(64, 64), nn.GELU()) for _ in range(3)]).to(dev)
    x = torch.randn(8, 64, device=dev, requires_grad=True)

    def step():
        h = x
        for b in blocks:
            h = tuc.checkpoint(b, h, use_reentrant=False)
        h.sum().backward()
    rc = RecomputeCheck()
    cal = rc.calibrate(step, steps=2)
    if cal["mismatches"] or cal["recomputes_compared"] != 6:
        bad.append(f"clean run: {cal}")
    calls = {"n": 0}

    def hook(mod, inp, out):
        calls["n"] += 1
        if calls["n"] == 2:                        # the recompute of block 2
            return out + 1e-3
        return out
    h = blocks[1][0].register_forward_hook(hook)
    seen = []
    with RecomputeCheck(on_alert=seen.append) as rc2:
        step()
        rc2.flush()
    h.remove()
    if len(seen) != 1 or seen[0].detail["block_call"] != 2:
        bad.append(f"fault in the recompute: {[(e.detail['where'], e.detail['block_call']) for e in seen]}")
    if bad:
        print("DISAGREE:\n" + "\n".join(bad))
        return 1
    print(f"ALL AGREE ({len(cases)} tensors, kernel status {st})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "cpu"))
