"""Does fast mode change training numbers on a GPU? It shouldn't: the checks only read.

Compares the gradients of a few bf16 training steps of GPT-2-shaped model (random data):
  1. no sdcguard            vs  fast mode with exact_reductions=False   -> must be bit-identical
  2. operator-level guard   vs  fast mode (both with exact reductions)  -> must be bit-identical
(exact_reductions=True, the default, keeps bf16 matmul partial sums in fp32 on checked steps, which
changes rounding slightly on those steps, by design; see guard.py.)

    python experiments/gpu_identity.py
"""
from __future__ import annotations

import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from harm_study import GPT  # noqa: E402
from sdcguard import SDCGuard  # noqa: E402


def grads(model, guard, steps=3, seed=0):
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    out = []
    for s in range(steps):
        T = model.block
        x = torch.randint(0, 50304, (8, T), generator=g).to(dev)
        y = torch.randint(0, 50304, (8, T), generator=g).to(dev)
        ctx = guard.at_step(s) if guard is not None else torch.enable_grad()
        with ctx:
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
        out.append({k: p.grad.detach().clone() for k, p in model.named_parameters()})
        model.zero_grad(set_to_none=True)
    return out


def same(a, b):
    diff = [k for ga, gb in zip(a, b) for k in ga if not torch.equal(ga[k], gb[k])]
    return len(diff) == 0, diff[:5]


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = GPT(50304, 256, 12, 12, 768).to(dev) if dev == "cuda" else GPT(50304, 32, 2, 2, 64)
    res = {}
    ref = grads(copy.deepcopy(base), None)
    m = copy.deepcopy(base)
    fast_noexact = SDCGuard(exact_reductions=False).protect(m)
    ok, diff = same(ref, grads(m, fast_noexact))
    res["no guard vs fast mode (exact_reductions=False)"] = {"bit_identical": ok, "differing": diff,
                                                             "checks": fast_noexact.stats["checked"],
                                                             "alerts": fast_noexact.stats["alerts"]}
    op = SDCGuard()
    ref2 = grads(copy.deepcopy(base), op)
    m = copy.deepcopy(base)
    fast = SDCGuard().protect(m)
    ok, diff = same(ref2, grads(m, fast))
    res["operator-level guard vs fast mode"] = {"bit_identical": ok, "differing": diff,
                                                "checks_operator_level": op.stats["checked"],
                                                "checks_fast": fast.stats["checked"],
                                                "alerts": op.stats["alerts"] + fast.stats["alerts"]}
    print(json.dumps(res, indent=1))
    with open(os.environ.get("IDENTITY_OUT", "identity.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
