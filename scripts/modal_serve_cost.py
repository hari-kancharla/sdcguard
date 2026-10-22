"""What the serving check costs, kernel by kernel, and whether v5 agrees with the plain check, on one H100.

    modal run scripts/modal_serve_cost.py

On every linear-layer shape of Qwen2.5-1.5B and Qwen2.5-7B (bf16), at 1 to 4,096 rows (tokens in the batch), with
more row counts below 256 (the batch sizes vLLM captures CUDA graphs for when decoding):
  - the layer's own matmul, and the checked layer with v1 (cuBLAS for E, then _check), v3 (cuBLAS for E, then _sums
    and _score) and v5 (one kernel, _fused) with 16-row and with 64-row tiles, and cuBLAS's E alone, each as a CUDA
    graph of up to 20 calls replayed (as vLLM runs them), median of 7 replays. Unlike the October 22 runs, the calls
    go round several copies of the layer, enough that its weights and checksums can't stay in the H100's 50 MB L2
    cache from one call to the next, as in a model, where every other layer passes through L2 in between;
  - each kernel's own GPU time, from the profiler (cuBLAS's GEMM, _check; _sums, _score; _fused), so the cost can be
    placed exactly; and a plain sum of y, as the speed of just reading it;
  - on the same input, clean and with one output changed by 1% of the layer's largest output and by 50: each
    setting's scores against the plain PyTorch check's (clean: within 0.05 + 2%; changed: the same alarm, same row
    and block).
Before any of that, each design's first-use comparison (bf16 and fp16) and tests/_serve_fast_compare.py on the GPU
(must print ALL AGREE). summarize() applies the rules docs/METHOD.md sets for v5 before the run.

About 8 to 12 minutes: about $0.80 at Modal's prices on Oct 20, 2026 ($3.95 an hour for the H100 plus about $0.45
for CPU and memory); at most about $2.20 (a 30-minute limit). Results go to results/serve_cost/.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 30 * 60
RATE = 3.95 + 4 * 0.0472 + 32 * 0.008
SHAPES = {
    "qwen1.5b": {"qkv_proj": (1536, 2048, True), "o_proj": (1536, 1536, False),
                 "gate_up_proj": (1536, 17920, False), "down_proj": (8960, 1536, False)},
    "qwen7b": {"qkv_proj": (3584, 4608, True), "o_proj": (3584, 3584, False),
               "gate_up_proj": (3584, 37888, False), "down_proj": (18944, 3584, False)},
}
ROWS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 1024, 4096]
KERNELS = ("v1", "v3", "v5_t16", "v5_t64")       # v5 with 16-row and with 64-row tiles
DESIGNS = ("v1", "v3", "v5")
L2_BYTES = 50 * 1024 * 1024
MARGIN = 0.95                                      # v5 must cost at most 95% of what it would replace


def _digest(root: Path) -> dict:
    """Fingerprint of the code under root (src, experiments, tests, scripts), the same one modal_check.py uses."""
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "experiments", "tests", "scripts"):
        for p in sorted((root / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    return {"sha256": h.hexdigest(), "files": n}


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs", "fixtures") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("vllm==0.31.0")
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_dir(LOCAL / "scripts", f"{R}/scripts", ignore=_skip)
)
app = modal.App("sdcguard-serve-cost", image=image)


def copies_for(k: int, n: int) -> int:
    """Copies of a layer the timed calls go round: enough that weights plus checksums are at least twice L2."""
    per = n * k * 2 + k * 6 * math.ceil(n / 128) * 2
    return max(2, math.ceil(2 * L2_BYTES / per))


_SIDE = []


def _graph_ms(fn, calls: int = 20, replays: int = 7) -> float:
    """Milliseconds per call of fn(i) (i = 0, 1, ...: the call's number, to pick a copy), as a CUDA graph. The warm-up
    runs on one side stream made once (a new stream each time would take a new set of v5's counters each time)."""
    import torch
    fn(0)
    torch.cuda.synchronize()
    if not _SIDE:
        _SIDE.append(torch.cuda.Stream())
    s = _SIDE[0]
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        out.append(a.elapsed_time(b) / calls)
    return sorted(out)[len(out) // 2]


def _group_of(name: str) -> str:
    group = next((g for g in ("_check", "_sums", "_score", "_fused") if g in name), None)
    if group is None:
        low = name.lower()
        group = "gemm" if any(w in low for w in ("gemm", "cutlass", "sm90", "xmma", "cublas", "nvjet")) else "other"
    return group


def _kernel_us(fn, iters: int = 10) -> dict:
    """GPU time per call of each kernel fn(i) launches, from the profiler, in microseconds, grouped: cuBLAS's GEMMs
    ("gemm"), sdcguard's Triton kernels by name, and everything else ("other")."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    fn(0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(iters):
            fn(i)
        torch.cuda.synchronize()
    out: dict = {}
    for ev in prof.key_averages():
        t = getattr(ev, "device_time_total", None)
        if t is None:
            t = getattr(ev, "cuda_time_total", 0.0)
        if not t:
            continue
        group = _group_of(ev.key)
        out[group] = round(out.get(group, 0.0) + t / iters, 2)
    return out


def _setting(kern: str) -> dict:
    """check() keywords for one timed setting."""
    if kern.startswith("v5_t"):
        return {"kern": "v5", "_bm": int(kern[4:])}
    return {"kern": kern}


@app.function(gpu="H100!", cpu=4, memory=32768, timeout=LIMIT)
def h100(expected: str) -> dict:
    t0 = time.monotonic()
    src = _digest(Path(R))
    out = {"source_on_gpu": src, "matches_launcher": src["sha256"] == expected}
    if not out["matches_launcher"]:
        out["stopped"] = (f"the code here ({src['sha256'][:12]}) isn't the code the launcher fingerprinted "
                          f"({expected[:12]}): nothing was run")
        return out
    import torch
    import torch.nn.functional as F
    from sdcguard import _serve_fast
    from sdcguard.serve import STATE_COLS, LayerChecksums, check_static_impl
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dev = torch.device("cuda")
    out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    import triton
    out["triton"] = triton.__version__
    # 1. agreement: the first-use comparison, then the comparison set, with each design forced
    out["agreement"] = {}
    for kern in DESIGNS:
        env = dict(os.environ, SDCGUARD_SERVE_KERNEL=kern)
        p = subprocess.run([sys.executable, f"{R}/tests/_serve_fast_compare.py", "cuda"], capture_output=True,
                           text=True, timeout=900, cwd=R, env=env)
        lines = p.stdout.strip().splitlines()
        os.environ["SDCGUARD_SERVE_KERNEL"] = kern
        probes = {}
        for dt in (torch.bfloat16, torch.float16):
            try:
                probes[str(dt)[6:]] = list(_serve_fast._probe(dev, dt))
            except Exception as e:  # noqa: BLE001
                probes[str(dt)[6:]] = [False, f"{type(e).__name__}: {str(e)[:300]}"]
        out["agreement"][kern] = {"compare_exit": p.returncode, "compare_last": lines[-1] if lines else None,
                                  "cases": len(lines), "disagree": [ln for ln in lines if '"agree": false' in ln][:10],
                                  "stderr": p.stderr[-1500:], "first_use": probes}
        print(kern, "compare:", lines[-1] if lines else None, "first use:", probes, flush=True)
    os.environ.pop("SDCGUARD_SERVE_KERNEL", None)
    # 2. cost and scores, layer by layer
    out["models"] = {}
    for model, layers in SHAPES.items():
        res = {}
        for name, (k, n, has_bias) in layers.items():
            g = torch.Generator(device="cpu").manual_seed(0)
            ncopy = copies_for(k, n)
            Ws, bs_, css = [], [], []
            for _ in range(ncopy):
                W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dev, torch.bfloat16)
                b = torch.randn(n, generator=g).to(dev, torch.bfloat16) if has_bias else None
                Ws.append(W)
                bs_.append(b)
                css.append(LayerChecksums(W, b, 128))
            argss = [cs.tensors(True) for cs in css]
            for cs in css:                                       # (as warm() does at load time)
                _serve_fast._reserve_counters(dev, cs.parts.data_ptr(), cs.nb)
            by_rows = {}
            for m in ROWS:
                r = {"copies": ncopy}
                calls = max(ncopy, 20 if m * max(n, k) <= 64 * 1024 * 1024 else 5)     # (every copy used)
                try:
                    x = torch.randn(m, k, generator=g).to(dev, torch.bfloat16)
                    ys = [F.linear(x, Ws[c], bs_[c]) for c in range(ncopy)]
                    r["linear_ms"] = _graph_ms(lambda i: F.linear(x, Ws[i % ncopy], bs_[i % ncopy]), calls)
                    r["e_cublas_ms"] = _graph_ms(
                        lambda i: torch.ops.aten.mm.dtype(x, argss[i % ncopy][0], torch.float32), calls)
                    r["read_y_ms"] = _graph_ms(lambda i: torch.sum(ys[i % ncopy], dtype=torch.float32), calls)
                    st = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                    cm, cd = _serve_fast._capture_buffers(dev, k)
                    for kern in KERNELS:
                        kw = _setting(kern)
                        fb = _serve_fast.V5_FALLBACKS[0]

                        def checked(i, kw=kw):
                            c = i % ncopy
                            yy = F.linear(x, Ws[c], bs_[c])
                            _serve_fast.check(x, yy, *argss[c], st, 0, 6.0, 1e-20, cap_meta=cm, cap_data=cd,
                                              cap_w=128, cap_tag=2, **kw)
                        r[f"{kern}_checked_ms"] = _graph_ms(checked, calls)
                        added = r[f"{kern}_checked_ms"] - r["linear_ms"]
                        r[f"{kern}_check_pct"] = round(100 * added / r["linear_ms"], 1)
                        r[f"{kern}_kernels_us"] = _kernel_us(
                            lambda i, kw=kw: _serve_fast.check(x, ys[i % ncopy], *argss[i % ncopy], st, 0, 6.0, 1e-20,
                                                               cap_meta=cm, cap_data=cd, cap_w=128, cap_tag=2, **kw))
                        r[f"{kern}_fallbacks"] = _serve_fast.V5_FALLBACKS[0] - fb
                    # scores on the same input; clean, a 1% change and a large one
                    y = ys[0]
                    big = float(y.float().abs().max())
                    for case, delta in (("clean", 0.0), ("one_pct", 0.01 * big), ("large", 50.0)):
                        yy = y.clone()
                        rr, cc = (m * 2) // 3, (n * 3) // 5
                        yy[rr, cc] += delta
                        z = {}
                        for kern in ("plain",) + KERNELS:
                            s2 = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                            if kern == "plain":
                                check_static_impl(x, yy, *argss[0], s2, 0, 6.0, 1e-20)
                            else:
                                _serve_fast.check(x, yy, *argss[0], s2, 0, 6.0, 1e-20, **_setting(kern))
                            zz, flat, _ = _serve_fast.decode(*s2[0, :2].tolist())
                            z[kern] = [zz, flat // 2, s2[0, 2:].tolist()]
                        r[f"scores_{case}"] = z
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                by_rows[str(m)] = r
                print(model, name, m, {kk: v for kk, v in r.items() if not kk.startswith("scores")}, flush=True)
            res[name] = {"k": k, "n": n, "bias": has_bias, "by_rows": by_rows}
            del Ws, bs_, css, argss
            torch.cuda.empty_cache()
        out["models"][model] = res
    torch.cuda.synchronize()
    out["v5_counters_clean"] = _serve_fast.counters_clean(dev)      # every tile's counter back at zero
    minutes = (time.monotonic() - t0) / 60
    out["minutes"] = round(minutes, 1)
    out["estimated_cost_usd"] = round(minutes / 60 * RATE, 2)
    return out


def _agrees(r: dict, kern: str) -> bool:
    """One kernel setting agrees with the plain check on this row count: clean within 0.05 + 2%; with a changed
    output, the same verdict, and the same row and block when it alarms; the same counts."""
    ok = True
    for case in ("clean", "one_pct", "large"):
        z = r.get(f"scores_{case}")
        if not z or kern not in z:
            return False
        pz, pb, pc = z["plain"]
        fz, fb, fc = z[kern]
        if case == "clean":
            ok &= pz <= 6.0 and fz <= 6.0 and abs(fz - pz) <= 0.05 + 0.02 * pz
        else:
            ok &= (pz > 6.0) == (fz > 6.0) and (pz <= 6.0 or pb == fb)
        ok &= pc == fc
    return ok


def _added_us(layers: dict, m: int, kern: str):
    """Time the check adds at m rows, summed over a model's layers, in microseconds (None if any is missing)."""
    tot = 0.0
    for dd in layers.values():
        r = dd["by_rows"].get(str(m), {})
        if f"{kern}_checked_ms" not in r:
            return None
        tot += r[f"{kern}_checked_ms"] - r["linear_ms"]
    return round(tot * 1000, 1)


def summarize(res: dict) -> dict:
    """The rules METHOD.md sets for v5 (section "The serving kernel", written before this run): criteria 1 and 2, the
    tile height, the row counts v5 is used for, and the added time per row count summed over each model's layers, per
    setting."""
    out = {"criteria": {}, "added_us": {}}
    ag = res.get("agreement", {}).get("v5", {})
    out["criteria"]["1 first-use comparison and comparison set"] = bool(
        ag.get("compare_exit") == 0 and ag.get("compare_last") == "ALL AGREE"
        and ag.get("first_use") and all(v and v[0] is True for v in ag["first_use"].values()))
    agree = True
    fallbacks = 0
    for model, layers in res.get("models", {}).items():
        out["added_us"][model] = {str(m): {kern: _added_us(layers, m, kern) for kern in KERNELS} for m in ROWS}
        for dd in layers.values():
            for m in ROWS:
                r = dd["by_rows"].get(str(m), {})
                agree &= "error" not in r and all(_agrees(r, kern) for kern in ("v5_t16", "v5_t64"))
                fallbacks += sum(r.get(f"{kern}_fallbacks", 1) for kern in ("v5_t16", "v5_t64"))
    out["v5_fallbacks"] = fallbacks
    out["criteria"]["2 scores agree with the plain check on every shape and row count, both tile heights"] = (
        agree and fallbacks == 0 and res.get("v5_counters_clean") is True)
    a7 = out["added_us"].get("qwen7b", {})

    def cost(m, kern):
        return a7.get(str(m), {}).get(kern)
    # tile height: 64-row tiles from the smallest row count from which they are cheaper at every larger one
    t64_from = None
    for m in reversed(ROWS):
        c16, c64 = cost(m, "v5_t16"), cost(m, "v5_t64")
        if c16 is None or c64 is None or not c64 < c16:
            break
        t64_from = m
    out["v5_t64_from_rows"] = t64_from

    def v5(m):
        return cost(m, "v5_t64" if t64_from is not None and m >= t64_from else "v5_t16")

    def now(m):
        return cost(m, "v3" if m >= 256 else "v1")
    # v5 is used from 1 row up to the largest row count of the run up to which it costs at most MARGIN of the design
    # it replaces at every row count
    upto = None
    for m in ROWS:
        a, b = v5(m), now(m)
        if a is None or b is None or not a <= MARGIN * b:
            break
        upto = m
    out["v5_up_to_rows"] = upto
    out["criteria"]["3 at 1 row, at most 95% of v1's cost over the four 7B layers"] = upto is not None
    out["v5_used"] = all(out["criteria"].values())
    out["v5_below_rows"] = (upto + 1) if out["v5_used"] else 0
    out["v5_vs_now"] = {str(m): {"v5": v5(m), "now": now(m),
                                 "ratio": round(v5(m) / now(m), 3) if v5(m) and now(m) else None} for m in ROWS}
    return out


@app.local_entrypoint()
def main() -> None:
    src = _digest(LOCAL)
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files). One H100, about 8 to 12 minutes, "
          "about $0.80.", flush=True)
    res = h100.remote(src["sha256"])
    d = LOCAL / "results" / "serve_cost"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"h100_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"launcher_source": src, **res}, indent=1, default=str))
    if res.get("stopped"):
        print("Stopped:", res["stopped"])
        return
    summary = summarize(res)
    path.write_text(json.dumps({"launcher_source": src, **res, "summary": summary}, indent=1, default=str))
    for kern, a in res["agreement"].items():
        print(f"{kern}: comparison set {a['compare_last']} (exit {a['compare_exit']}); first use {a['first_use']}")
    for model, layers in res["models"].items():
        print(f"\n{model}: time added over the {len(layers)} layers, microseconds")
        print("    rows  " + "".join(f"{k:>10}" for k in KERNELS) + "   layers' own")
        for m in ROWS:
            own = sum(dd["by_rows"].get(str(m), {}).get("linear_ms", float("nan")) for dd in layers.values()) * 1000
            print(f"  {m:>6}  " + "".join(f"{summary['added_us'][model][str(m)][k]!s:>10}" for k in KERNELS)
                  + f"   {own:9.1f}")
    print("\nv5 (tiles as chosen) against the design used now (v1 below 256 rows, v3 from 256), 7B:")
    for m in ROWS:
        v = summary["v5_vs_now"][str(m)]
        print(f"  {m:>6} rows: v5 {v['v5']}  now {v['now']}  ratio {v['ratio']}")
    for name, ok in summary["criteria"].items():
        print(f"  {'met' if ok else 'NOT met'}: {name}")
    print(f"v5 used: {summary['v5_used']} (below {summary['v5_below_rows']} rows; 64-row tiles from "
          f"{summary['v5_t64_from_rows']} rows). Fallbacks to v1/v3 while timing: {summary['v5_fallbacks']}. "
          f"Saved {path}. Estimated cost ${res.get('estimated_cost_usd')}.")
