"""How small an error each check catches, on the same matmuls: the check of bf16 results that training and
serving use (checksum.py), and the check of the fp32 accumulator that `sdcguard check` runs (modes.py).

    python experiments/sensitivity.py                          # CPU, small: a smoke test
    python experiments/sensitivity.py --device cuda --out results/sensitivity

Each trial: bf16 inputs A (m x k) and B (k x n), C = A @ B with an fp32 result on the device (the tensor
cores' accumulator). One output gets an error of e times the typical (root-mean-square) output of its
128-column block, with a random sign, before rounding. The bf16 check sees C rounded to bf16, as a bf16
matmul returns it; the fp32 check sees C itself. Per error size it records how often each check flagged
the result, and how often the error still changed the bf16 output (when it rounds away, the model never
saw it). Trials without an error count false alarms.

What this isolates is the effect of keeping the accumulator: both checks look at the same matmul. It says
nothing about faults that aren't in a matmul's result, and on the CPU there are no tensor cores, so CPU
numbers are a smoke test, not a result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import checksum, modes  # noqa: E402

SIZES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _fp32_result(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_cuda:
        return torch.ops.aten.mm.dtype(a, b, torch.float32)
    return a.float() @ b.float()


def run(dev: torch.device, shape: tuple, trials: int, seed: int = 0, say=print) -> dict:
    m, k, n = shape
    gen = torch.Generator().manual_seed(seed)
    nfma = modes.n_fma("bf16", torch.cuda.get_device_capability(dev)) if dev.type == "cuda" else None
    hits = {"bf16": [0] * len(SIZES), "fp32": [0] * len(SIZES)}
    changed = [0] * len(SIZES)
    clean = {"bf16": {"false_alarms": 0, "worst_z": 0.0}, "fp32": {"false_alarms": 0, "worst_z": 0.0}}
    per_variant = {}
    done = 0
    for dist, scale in modes.VARIANTS:
        vname = f"{dist} x{scale:g}"
        per_variant[vname] = {"bf16": [0] * len(SIZES), "fp32": [0] * len(SIZES), "trials": 0}
        for _ in range(trials):
            a, b, a64, b64 = modes._inputs("bf16", shape, dist, scale, gen)
            a, b = a.to(dev), b.to(dev)
            c32 = _fp32_result(a, b)
            c16 = c32.to(torch.bfloat16)
            r_old = checksum.check_matmul(a, b, c16)
            r_new = modes.judge("bf16", a64, b64, c32, nfma)
            clean["bf16"]["false_alarms"] += not r_old.ok
            clean["fp32"]["false_alarms"] += r_new["wrong_blocks"] > 0
            if math.isfinite(r_old.max_z):
                clean["bf16"]["worst_z"] = max(clean["bf16"]["worst_z"], float(r_old.max_z))
            clean["fp32"]["worst_z"] = max(clean["fp32"]["worst_z"], r_new["worst_z"])
            i = int(torch.randint(0, m, (1,), generator=gen))
            j = int(torch.randint(0, n, (1,), generator=gen))
            sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
            blk = c32[i, (j // 128) * 128:(j // 128 + 1) * 128].double()
            typical = float(blk.norm()) / math.sqrt(128)
            for s_i, e in enumerate(SIZES):
                c = c32.clone()
                c[i, j] += sign * e * typical
                c16f = c.to(torch.bfloat16)
                changed[s_i] += bool(c16f[i, j] != c16[i, j])
                old = not checksum.check_matmul(a, b, c16f).ok
                new = modes.judge("bf16", a64, b64, c, nfma)["wrong_blocks"] > 0
                hits["bf16"][s_i] += old
                hits["fp32"][s_i] += new
                per_variant[vname]["bf16"][s_i] += old
                per_variant[vname]["fp32"][s_i] += new
            per_variant[vname]["trials"] += 1
            done += 1
        say(f"  {vname}: {trials} trials")
    rate = {key: [h / done for h in v] for key, v in hits.items()}
    return {"sizes": list(SIZES), "trials_per_size": done, "flagged": rate,
            "changed_bf16_output": [c / done for c in changed], "clean": clean,
            "half_caught_at": {key: _half(SIZES, v) for key, v in rate.items()},
            "per_variant": per_variant, "tensor_core_steps": nfma}


def _half(sizes, rates):
    """The error size at which the check flags half the trials (log-linear between measured sizes)."""
    for (s0, r0), (s1, r1) in zip(zip(sizes, rates), zip(sizes[1:], rates[1:])):
        if r0 < 0.5 <= r1:
            t = (0.5 - r0) / (r1 - r0)
            return float(f"{math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0))):.2g}")
    return sizes[0] if rates and rates[0] >= 0.5 else None


def markdown(res: dict) -> str:
    def pct(x):
        return f"{100 * x:.0f}%"

    def size(e):
        return f"{100 * e:.4g}%"
    out = [f"# Sensitivity: bf16 result vs fp32 accumulator ({res['device']})", "",
           f"{res['gpu']}, torch {res['torch']}, matmuls of {res['shape'][0]} x {res['shape'][1]} by "
           f"{res['shape'][1]} x {res['shape'][2]}, {res['trials_per_size']} trials per error size "
           f"({len(modes.VARIANTS)} input distributions). An error is relative to a typical output of its block.", "",
           "| error size | bf16 check flags | fp32 check flags | changed the bf16 output |", "|---|---|---|---|"]
    for i, e in enumerate(res["sizes"]):
        out.append(f"| {size(e)} | {pct(res['flagged']['bf16'][i])} | {pct(res['flagged']['fp32'][i])} | "
                   f"{pct(res['changed_bf16_output'][i])} |")
    h = res["half_caught_at"]
    out += ["", f"Half the errors are caught from about {size(h['bf16']) if h['bf16'] else '?'} (bf16 check) and "
                f"{size(h['fp32']) if h['fp32'] else '?'} (fp32 check).",
            f"Clean trials: bf16 check {res['clean']['bf16']['false_alarms']} false alarms (worst score "
            f"{res['clean']['bf16']['worst_z']:.2f}), fp32 check {res['clean']['fp32']['false_alarms']} "
            f"(worst score {res['clean']['fp32']['worst_z']:.2f}); the alarm is at 6.", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", default="cpu")
    p.add_argument("--trials", type=int, default=None, help="trials per input distribution (default 10 GPU, 2 CPU)")
    p.add_argument("--shape", default=None, help="m,k,n (default 2048,1024,2048 on GPUs, 256,256,256 on the CPU)")
    p.add_argument("--out", default=None, help="folder for <device>.json and .md")
    a = p.parse_args(argv)
    dev = torch.device(a.device)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor() or "CPU"
    shape = tuple(int(x) for x in a.shape.split(",")) if a.shape else modes.SHAPE["cuda" if dev.type == "cuda" else "cpu"]
    trials = a.trials or (10 if dev.type == "cuda" else 2)
    t0 = time.time()
    res = run(dev, shape, trials)
    res.update(device=str(dev), gpu=gpu, torch=torch.__version__, shape=list(shape), seconds=round(time.time() - t0, 1),
               date=time.strftime("%Y-%m-%d"))
    md = markdown(res)
    print(md)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = gpu.replace("NVIDIA ", "").replace(" ", "_").replace("/", "_") if dev.type == "cuda" else "cpu"
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        with open(os.path.join(a.out, f"{name}.md"), "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
