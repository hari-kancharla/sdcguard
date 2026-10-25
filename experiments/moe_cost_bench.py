"""Where the MoE checks' serving cost goes, measured without vLLM (docs/METHOD.md, "The MoE checks' cost", written
before the run).

The October 25 MoE run measured 6.4% fewer output tokens per second with the MoE and routing checks on (16 requests
at once; 7.6% at 256). Each model step, every one of the 24 MoE layers launches 3 Triton kernels (up, down, compare)
and every router 1, and all but one layer's return at once. This measures, on one GPU, in CUDA graphs as vLLM
replays them:

  exit     a graph of 96 kernels that each read the step buffer and return, at grids of 1, 2, 16, 132, 512 and 4096
           programs (the MoE kernels' grids are 16 x 32 = 512 at granite's shapes): the cost of a kernel that does
           nothing, by grid
  checks   granite-3.1-1b-a400m's shapes (24 layers, 32 experts, hidden 1024, intermediate 512, top 8), 16 and 128
           rows: a graph of the 24 layers' MoE checks and routing checks (sdcguard's own kernels, the same calls the
           plugin makes) with no layer active and with one active, and each kind alone
  work     the same checks interleaved with a stand-in for the layers' own kernels (24 bf16 matmuls), against the
           stand-in alone: what the checks add when they sit between other kernels

Every figure is microseconds per graph replay (CUDA events over 200 replays after 20 left out, the median of 5 such
measurements). Nothing here is a criterion of the checks; the run is complete when every figure was measured.

    python experiments/moe_cost_bench.py --out runs/moe_bench
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

E, H, I, K, L = 32, 1024, 512, 8, 24
GRIDS = (1, 2, 16, 132, 512, 4096)
EXITS = 96
ROWS = (16, 128)
REPS, WARM, TIMES = 200, 20, 5
DEV = "cuda"                            # (the CPU test runs the calls once through Triton's interpreter)


def graph_us(fn) -> float:
    """Median microseconds per replay of a CUDA graph of fn (fn run once outside the graph first)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(WARM):
        g.replay()
    torch.cuda.synchronize()
    vals = []
    for _ in range(TIMES):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(REPS):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0 / REPS)
    del g
    return round(statistics.median(vals), 3)


def exit_costs() -> dict:
    import triton
    import triton.language as tl

    @triton.jit
    def _exit_kernel(stepbuf, out, nl, idx):
        raw = tl.load(stepbuf)
        if (raw % nl) == idx:                 # never here: no index matches
            tl.store(out + tl.program_id(0), raw)

    sb = torch.full((1,), 5, dtype=torch.int64, device=DEV)
    out = torch.zeros(4096, dtype=torch.int64, device=DEV)
    res = {}
    for g in GRIDS:
        def fn(g=g):
            for _ in range(EXITS):
                _exit_kernel[(g,)](sb, out, 7, 6, num_warps=4)
        t = graph_us(fn)
        res[str(g)] = {"graph_us": t, "us_per_kernel": round(t / EXITS, 3)}
    return res


def _layers(rows: int, active: bool):
    """24 layers' check arguments at granite's shapes (random weights and routing), as the plugin arms them."""
    from sdcguard import serve_moe as sm
    from sdcguard import serve_route as sr
    dev = torch.device(DEV)
    gen = torch.Generator(device=dev).manual_seed(0)
    nl = L if active else L + 1                # with L + 1 turns and step L, no layer is active
    step = 3 if active else L
    msb = torch.full((1,), sm.pack_step(step, rows), dtype=torch.int64, device=dev)
    rsb = torch.full((1,), step, dtype=torch.int64, device=dev)
    rrows = torch.full((1,), rows, dtype=torch.int64, device=dev)
    mst = torch.zeros(L, sm.COLS, dtype=torch.int64, device=dev)
    rst = torch.zeros(256, sr.COLS, dtype=torch.int64, device=dev)
    mcap = (torch.zeros(L, sm.CAP_COLS, dtype=torch.int32, device=dev), torch.zeros(L, H, device=dev),
            torch.zeros(L, H, device=dev), torch.zeros(L, 64, dtype=torch.int64, device=dev),
            torch.zeros(L, 64, device=dev))
    rcap = sr.new_caps(dev)
    out = []
    for i in range(L):
        w13 = (torch.randn(E, 2 * I, H, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(E, H, I, generator=gen, device=dev) * 0.02).to(torch.bfloat16)
        x = torch.randn(rows, H, generator=gen, device=dev).to(torch.bfloat16)
        logits = torch.randn(rows, E, generator=gen, device=dev).to(torch.bfloat16)
        w, ids = torch.topk(torch.softmax(logits.float(), -1), K, -1)
        w = w / w.sum(-1, keepdim=True)
        y = torch.randn(rows, H, generator=gen, device=dev).to(torch.bfloat16)
        moe = (x, y, w, ids.to(torch.int32), w13, w2, None, 0, False, mst, i, msb, nl, i, *mcap, sm.MOE_Z)
        route = (logits, w, ids.to(torch.int32), K, sr.SCORING["softmax"], True, None, 0, 0, 1.0, rrows, rsb, rst, i,
                 nl, i, *rcap)
        out.append((moe, route))
    return out


def check_costs() -> dict:
    from sdcguard import serve_moe as sm
    from sdcguard import serve_route as sr
    res = {}
    for rows in ROWS:
        for active in (False, True):
            lays = _layers(rows, active)
            both = graph_us(lambda: [(sm.moe_check_triton(*m), sr.route_check_triton(*r)) for m, r in lays])
            moe = graph_us(lambda: [sm.moe_check_triton(*m) for m, _ in lays])
            route = graph_us(lambda: [sr.route_check_triton(*r) for _, r in lays])
            wa = torch.randn(rows, H, device=DEV).to(torch.bfloat16)
            wb = torch.randn(H, H, device=DEV).to(torch.bfloat16)
            wc = torch.empty(rows, H, device=DEV, dtype=torch.bfloat16)
            work = graph_us(lambda: [torch.mm(wa, wb, out=wc) for _ in range(L)])

            def interleaved():
                for m, r in lays:
                    torch.mm(wa, wb, out=wc)
                    sr.route_check_triton(*r)
                    sm.moe_check_triton(*m)
            mixed = graph_us(interleaved)
            res[f"{rows}_{'one_active' if active else 'none_active'}"] = {
                "checks_us": both, "moe_us": moe, "route_us": route, "work_us": work, "work_and_checks_us": mixed,
                "added_to_work_us": round(mixed - work, 3)}
            del lays
            if DEV == "cuda":
                torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res: dict = {"gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else DEV, "torch": torch.__version__}
    try:
        import triton
        res["triton"] = triton.__version__
        res["exit"] = exit_costs()
        res["checks"] = check_costs()
    except Exception as e:  # noqa: BLE001  (what was measured is written)
        res["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    res["complete"] = "error" not in res and len(res.get("exit") or {}) > len(GRIDS) - 1 and \
        len(res.get("checks") or {}) == 2 * len(ROWS)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
