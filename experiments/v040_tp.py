"""Tensor parallel on GPUs (0.4.0): every rank checks its own shard of each column- and row-parallel matmul.
Run with torchrun on 2 GPUs (NCCL); criteria in docs/METHOD.md ("0.4.0 on GPUs"), written before the run.

    torchrun --nproc_per_node 2 experiments/v040_tp.py --out results/v040

For both modes (protect() and the default operator-level mode): 50 clean steps of a bf16 MLP stack with
Colwise/Rowwise parallel layers (no alarm, every step checked on both ranks, every local matmul of every step
checked and none skipped); with exact_reductions=False, the weight gradients bit-identical to the same
tensor-parallel model without sdcguard; then one wrong value in rank 1's local output, changing its 128-column
block by 10% of that block's L2 norm (the harm study's measure of a fault's size), on 10 checked steps (caught on
rank 1 every time, never on rank 0).

And the same model with protect() under torch.compile(fullgraph=True) (added October 21, before any run): the
same clean and fault criteria, the wrong value made inside the checked operator (where the GPU computes it);
no gradient comparison, since Inductor's own kernels may round differently from the operator's cuBLAS calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from sdcguard import SDCGuard  # noqa: E402
from v040_check import add_fault_  # noqa: E402  (the same wrong value as the single-GPU runner)


class CorruptLocal(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if str(func).startswith(("aten.mm", "aten.addmm")) and hasattr(out, "to_local"):
            with torch.no_grad():                # one wrong value (add_fault_)
                loc = out.to_local()
                add_fault_(loc.view(-1, loc.shape[-1])[0], 17)
        return out


LAYERS = 4
MODES = ("fast", "default", "compiled")


def checks_per_step(layers: int = LAYERS) -> int:
    """The matmuls one training step of build()'s model runs on each rank, all of which must be checked: each of
    its 2 * layers nn.Linear layers has a forward and a dW matmul, and a dX matmul except the first (its input is
    the data, which needs no gradient)."""
    linears = 2 * layers
    return 3 * linears - 1


def build(mesh, width=4096, layers=LAYERS, dev="cuda"):
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
    torch.manual_seed(0)
    mods = []
    for _ in range(layers):
        mods += [torch.nn.Linear(width, 4 * width), torch.nn.GELU(), torch.nn.Linear(4 * width, width)]
    m = torch.nn.Sequential(*mods).to(torch.bfloat16).to(dev)
    plan = {}
    for i in range(layers):
        plan[str(3 * i)] = ColwiseParallel()
        plan[str(3 * i + 2)] = RowwiseParallel()
    return parallelize_module(m, mesh, plan)


class CorruptInsideOperator:
    """Inside a compiled graph no dispatch mode sees the matmul: the wrong value is made where the checked operator
    computes the layer (its F.linear call), on this rank's local output."""

    def __enter__(self):
        import torch.nn.functional as F
        self.real = F.linear

        def wrong(x, w, b=None):
            y = self.real(x, w, b).clone()
            with torch.no_grad():
                add_fault_(y.view(-1, y.shape[-1])[0], 17)
            return y
        F.linear = wrong
        return self

    def __exit__(self, *a):
        import torch.nn.functional as F
        F.linear = self.real
        return False


def run(mode: str, mesh, rank: int, dev: str = "cuda", width: int = 4096, steps: int = 50) -> dict:
    x_gen = torch.Generator().manual_seed(1)
    xs = [torch.randn(8, 512 if dev == "cuda" else 16, width, generator=x_gen).to(torch.bfloat16).to(dev)
          for _ in range(2)]
    ref = build(mesh, width, dev=dev)
    m = build(mesh, width, dev=dev)
    g = SDCGuard(every=1, exact_reductions=False)
    if mode in ("fast", "compiled"):
        g.protect(m)
    f = torch.compile(m, fullgraph=True) if mode == "compiled" else m
    for n in range(steps):
        with g.on_step(n):
            f(xs[n % 2]).float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True)
    clean = {k: g.stats.get(k, 0) for k in ("checked", "checked_dtensor", "skipped_dtensor", "skipped_oom",
                                             "skipped_dtype", "skipped_dtensor_compiled", "alerts", "unjudged")}
    same = None
    if mode != "compiled":
        with g.on_step(1000):
            m(xs[0]).float().pow(2).mean().backward()
        ref(xs[0]).float().pow(2).mean().backward()
        same = all(torch.equal(a.grad.full_tensor(), b.grad.full_tensor()) for a, b in zip(m.parameters(),
                                                                                           ref.parameters()))
        m.zero_grad(set_to_none=True)
    caught = 0
    for t in range(10):
        before = g.stats["alerts"]
        inj = (CorruptInsideOperator() if mode == "compiled" else CorruptLocal()) if rank == 1 else None
        if inj is not None:
            inj.__enter__()
        try:
            with g.on_step(2000 + t):
                f(xs[0]).float().pow(2).mean().backward()
        finally:
            if inj is not None:
                inj.__exit__(None, None, None)
        m.zero_grad(set_to_none=True)
        caught += g.stats["alerts"] > before
    return {"clean": clean, "grads_identical_to_unguarded_tp": same, "fault_steps": 10, "steps_with_alert": caught}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true", help="a smoke test over gloo on the CPU, small sizes")
    a = ap.parse_args(argv)
    dev = "cpu" if a.cpu else "cuda"
    dist.init_process_group("gloo" if a.cpu else "nccl")
    rank = dist.get_rank()
    if not a.cpu:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh(dev, (dist.get_world_size(),))
    res = {"rank": rank, "gpu": torch.cuda.get_device_name() if not a.cpu else "cpu", "torch": torch.__version__}
    steps = 4 if a.cpu else 50
    for mode in MODES:
        try:
            res[mode] = run(mode, mesh, rank, dev, 256 if a.cpu else 4096, steps)
        except Exception as e:  # noqa: BLE001
            import traceback
            res[mode] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}
        print(json.dumps({"rank": rank, mode: res[mode]}), flush=True)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, res)
    ok = True
    if rank == 0:
        # "every matmul checked as a local shard (none skipped)": every local matmul of every clean step went
        # through the check, on both ranks. checked == checked_dtensor holds whether checked_dtensor counts the
        # tensor-parallel matmuls that reached the check or only the checks that ran, once nothing was skipped.
        expected = steps * checks_per_step()
        for mode in MODES:
            for r in gathered:
                d = r.get(mode) or {}
                c = d.get("clean") or {}
                expect = 10 if r["rank"] == 1 else 0
                grads = d.get("grads_identical_to_unguarded_tp") is True if mode != "compiled" else True
                ok &= ("error" not in d and c.get("alerts") == 0 and c.get("unjudged") == 0
                       and c.get("checked") == expected and c.get("checked_dtensor") == expected
                       and all(c.get(k) == 0 for k in ("skipped_dtensor", "skipped_oom", "skipped_dtype",
                                                       "skipped_dtensor_compiled"))
                       and grads and d.get("steps_with_alert") == expect)
        summary = {"ranks": gathered, "clean steps": steps, "matmuls checked per step and rank": checks_per_step(),
                   "all met": ok}
        print(json.dumps({"criteria all met": ok}), flush=True)
        if a.out:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "v040_tp.json"), "w") as f:
                json.dump(summary, f, indent=2, default=str)
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
