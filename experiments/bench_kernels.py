"""How long the Triton check takes on each matmul shape of GPT-2 small, next to the matmul itself.

One training step of GPT-2 small (12 layers, batch 32 x 512 tokens = 16,384 rows) runs these
matmuls; every one of them is checked on a checked step. For each shape this prints the matmul's
time, the check's time (all its kernels), the plain PyTorch check's time, and the GPU time per
Triton kernel, then the totals for a whole step.

    python experiments/bench_kernels.py            (NVIDIA GPU)
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import _group  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402

T, D, F, V, L = 16384, 768, 3072, 50257, 12        # tokens, width, MLP width, vocabulary, layers
VP = 50304                                          # the vocabulary padded to a multiple of 64


def cases():
    """(name, how many per step, A, B) with the strides training really produces. The output layer
    is timed twice: GPT-2's own vocabulary (50,257 words, an odd row length) and padded to 50,304
    as nanoGPT does (rows that start on 16-byte boundaries, which lets the GPU read them faster)."""
    g = torch.Generator(device="cuda").manual_seed(0)

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=g) * scale).to(torch.bfloat16)

    x, h = r(T, D), r(T, F)
    out = []
    for name, d_in, d_out, inp, count in (("attn q/k/v/proj", D, D, x, 4), ("mlp up", D, F, x, 1),
                                          ("mlp down", F, D, h, 1)):
        w = r(d_out, d_in, scale=0.02)
        dy = r(T, d_out, scale=1e-3)
        out.append((f"{name}: forward", count * L, inp, w.t()))
        out.append((f"{name}: input grad", count * L, dy, w))
        out.append((f"{name}: weight grad", count * L, dy.t(), inp))
    for v, tag, count in ((V, "head", 1), (VP, "head padded", 0)):
        wv = r(v, D, scale=0.02)
        dl = r(T, v, scale=1e-5)
        out.append((f"{tag}: forward", count, x, wv.t()))
        out.append((f"{tag}: input grad", count, dl, wv))
        out.append((f"{tag}: weight grad", count, dl.t(), x))
    return out


def timed(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def host_us(fn, reps=20):
    """CPU microseconds to queue one call (Python, Triton's launcher), not waiting for the GPU."""
    import time
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    dt = (time.perf_counter() - t) / reps * 1e6
    torch.cuda.synchronize()
    return round(dt, 1)


def kernel_ms(fn, reps=5):
    """GPU milliseconds per call of each Triton kernel inside fn."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    out = {}
    for e in prof.key_averages():
        if e.key in ("_stage", "_compare", "_compare_part", "_finish"):
            t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
            out[e.key] = round(t / 1000 / reps, 3)
    return out


def main():
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False   # as the guard does
    print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, "
          f"Triton check available: {_group.available(torch.empty(1, 1, device='cuda', dtype=torch.bfloat16))}",
          flush=True)
    rows, tot = [], {"matmul": 0.0, "check": 0.0, "plain": 0.0}
    print(f"{'shape':28s} {'A':>16s} {'B':>14s} {'n/step':>6s} {'matmul ms':>9s} {'check ms':>9s} "
          f"{'plain ms':>9s} {'splits':>6s}  kernels (ms per call)", flush=True)
    for name, count, A, B in cases():
        C = A @ B
        J = 2 * _group._layout(C.shape[-1], 128)[2]
        bj = _group._bj(J)
        tiles = _group._cdiv(C.shape[0], _group.BM) * _group._cdiv(J, bj)
        splits = _group._splits(tiles, A.shape[-1], _group._cfg(bj)[0], C.device)[0]
        t_mm = timed(lambda: A @ B)

        def check(A=A, B=B, C=C):
            g = _group.Group(C.device, C.dtype, 128, cs.ABS_FLOOR)
            g.add(A, B, C, None, 1.0, 1.0)
        t_ck = timed(check)
        t_pl = timed(lambda: cs.matmul_zscores(A, B, C))
        ks = kernel_ms(check)
        cpu = host_us(check)
        # how fast _stage reads C and B (it reads each once): H100 HBM3 peaks near 3,350 GB/s
        stage_gbs = round((C.numel() + B.numel()) * C.element_size() / (ks["_stage"] * 1e6)) if ks.get("_stage") else None
        rows.append({"shape": name, "A": list(A.shape), "B": list(B.shape), "per_step": count, "matmul_ms": t_mm,
                     "check_ms": t_ck, "plain_ms": t_pl, "splits": splits, "kernels_ms": ks,
                     "stage_GB_per_s": stage_gbs, "host_us": cpu})
        for key, t in (("matmul", t_mm), ("check", t_ck), ("plain", t_pl)):
            tot[key] += count * t
        print(f"{name:28s} {str(tuple(A.shape)):>16s} {str(tuple(B.shape)):>14s} {count:6d} {t_mm:9.3f} "
              f"{t_ck:9.3f} {t_pl:9.3f} {splits:6d}  {ks} stage {stage_gbs} GB/s, CPU {cpu} us", flush=True)
    print(f"\nPer checked step (all {sum(r['per_step'] for r in rows)} matmuls, vocabulary {V}): matmuls "
          f"{tot['matmul']:.1f} ms, Triton check {tot['check']:.1f} ms, plain check {tot['plain']:.1f} ms", flush=True)
    pad = {k: sum(r[k] * (r["per_step"] or (1 if r["shape"].startswith("head padded") else 0))
                  for r in rows if not r["shape"].startswith("head:")) for k in ("matmul_ms", "check_ms")}
    tot["padded"] = pad
    print(f"Same with the vocabulary padded to {VP}: matmuls {pad['matmul_ms']:.1f} ms, "
          f"Triton check {pad['check_ms']:.1f} ms", flush=True)
    out = os.environ.get("BENCH_OUT")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"gpu": torch.cuda.get_device_name(), "rows": rows, "totals_ms": tot}, f, indent=1)


if __name__ == "__main__":
    main()
