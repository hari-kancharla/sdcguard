"""What the serving check costs, kernel by kernel, and whether the v2 kernels agree with the plain check, on one H100.

    modal run scripts/modal_serve_cost.py

On every linear-layer shape of Qwen2.5-1.5B and Qwen2.5-7B (bf16), at 1 to 16,384 rows (tokens in the batch):
  - the layer's own matmul, and the checked layer with the v1 kernel (cuBLAS for E, then _check) and with v2 (_emm,
    then _pairs), each as a CUDA graph of up to 20 calls replayed (as vLLM runs them), median of 7 replays; v2 also
    with 16 and 64 blocks per _pairs program instead of 32 (PAIRS_P), to pick that setting from the same run;
  - on the same input, clean and with one output changed by 1% of the layer's largest output and by 50: both
    kernels' scores against the plain PyTorch check's (clean: within 0.05 + 2%; changed: the same alarm, same row
    and block).
Before any of that, both kernels' first-use comparison (bf16 and fp16) and tests/_serve_fast_compare.py on the GPU
(must print ALL AGREE) with each kernel.

About 10 to 15 minutes: about $1 at Modal's prices on Oct 20, 2026 ($3.95 an hour for the H100 plus about $0.45 for
CPU and memory); at most about $2.20 (a 30-minute limit). Results go to results/serve_cost/.
"""
from __future__ import annotations

import json
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
ROWS = [1, 16, 64, 256, 1024, 4096, 16384]
KERNELS = ("v1", "v2", "v2_p16", "v2_p64")      # v2 with 32 (its default), 16 and 64 blocks per _pairs program


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


def _graph_ms(fn, calls: int = 20, replays: int = 7) -> float:
    import torch
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(calls):
            fn()
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
    # 1. agreement: the first-use comparison, then the comparison set, with each kernel
    out["agreement"] = {}
    for kern in ("v1", "v2"):
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
    # 2. cost and scores, layer by layer
    out["models"] = {}
    for model, layers in SHAPES.items():
        res = {}
        for name, (k, n, has_bias) in layers.items():
            g = torch.Generator(device="cpu").manual_seed(0)
            W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dev, torch.bfloat16)
            b = torch.randn(n, generator=g).to(dev, torch.bfloat16) if has_bias else None
            cs = LayerChecksums(W, b, 128)
            args = cs.tensors(True)
            for kern in ("v1", "v2"):
                os.environ["SDCGUARD_SERVE_KERNEL"] = kern
                _serve_fast.warm(*args, cs.k, cs.n, cs.dtype)
            by_rows = {}
            for m in ROWS:
                r = {}
                calls = 20 if m * max(n, k) <= 64 * 1024 * 1024 else 5
                try:
                    x = torch.randn(m, k, generator=g).to(dev, torch.bfloat16)
                    y = F.linear(x, W, b)
                    r["linear_ms"] = _graph_ms(lambda: F.linear(x, W, b), calls)
                    st = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                    cm, cd = _serve_fast._capture_buffers(dev, k)
                    default_p = _serve_fast.PAIRS_P
                    for kern, pairs_p in (("v1", None), ("v2", None), ("v2_p16", 16), ("v2_p64", 64)):
                        os.environ["SDCGUARD_SERVE_KERNEL"] = kern[:2]
                        _serve_fast.PAIRS_P = pairs_p or default_p

                        def checked():
                            yy = F.linear(x, W, b)
                            _serve_fast.check(x, yy, *args, st, 0, 6.0, 1e-20, cap_meta=cm, cap_data=cd, cap_w=128,
                                              cap_tag=2)
                        try:
                            r[f"{kern}_checked_ms"] = _graph_ms(checked, calls)
                            added = r[f"{kern}_checked_ms"] - r["linear_ms"]
                            r[f"{kern}_check_pct"] = round(100 * added / r["linear_ms"], 1)
                        finally:
                            _serve_fast.PAIRS_P = default_p
                    r["v2_config"] = list(_serve_fast.config_v2(m, k, args[0].shape[1], dev))
                    # scores on the same input: plain, v1, v2; clean, a 1% change and a large one
                    big = float(y.float().abs().max())
                    for case, delta in (("clean", 0.0), ("one_pct", 0.01 * big), ("large", 50.0)):
                        yy = y.clone()
                        rr, cc = (m * 2) // 3, (n * 3) // 5
                        yy[rr, cc] += delta
                        z = {}
                        for kern in ("plain",) + KERNELS:
                            s2 = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                            if kern == "plain":
                                check_static_impl(x, yy, *args, s2, 0, 6.0, 1e-20)
                            else:
                                os.environ["SDCGUARD_SERVE_KERNEL"] = kern[:2]
                                _serve_fast.PAIRS_P = {"v2_p16": 16, "v2_p64": 64}.get(kern, default_p)
                                try:
                                    _serve_fast.check(x, yy, *args, s2, 0, 6.0, 1e-20)
                                finally:
                                    _serve_fast.PAIRS_P = default_p
                            zz, flat, _ = _serve_fast.decode(*s2[0, :2].tolist())
                            z[kern] = [zz, flat // 2, s2[0, 2:].tolist()]
                        r[f"scores_{case}"] = z
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    r["error"] = f"{type(e).__name__}: {str(e)[:400]}"
                by_rows[str(m)] = r
                print(model, name, m, {kk: v for kk, v in r.items() if not kk.startswith("scores")}, flush=True)
            res[name] = {"k": k, "n": n, "bias": has_bias, "by_rows": by_rows}
            del W, b, cs, args
            torch.cuda.empty_cache()
        out["models"][model] = res
    minutes = (time.monotonic() - t0) / 60
    out["minutes"] = round(minutes, 1)
    out["estimated_cost_usd"] = round(minutes / 60 * RATE, 2)
    return out


def _agrees(r: dict, kern: str) -> bool:
    """One kernel agrees with the plain check on this row count: clean within 0.05 + 2%; with a changed output,
    the same verdict, and the same row and block when it alarms; the same counts."""
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



def summarize(res: dict) -> dict:
    """The three criteria METHOD.md sets for v2 (section "The serving kernel"), and the added time per row count
    summed over each model's layers, per kernel setting."""
    out = {"criteria": {}, "added_ms": {}}
    ag = res.get("agreement", {}).get("v2", {})
    out["criteria"]["1 first-use comparison and comparison set"] = bool(
        ag.get("compare_exit") == 0 and ag.get("compare_last") == "ALL AGREE"
        and all(v and v[0] is True for v in (ag.get("first_use") or {}).values()) and ag.get("first_use"))
    agree, cheaper = True, True
    for model, layers in res.get("models", {}).items():
        added = {}
        for m in ROWS:
            row = {}
            for kern in KERNELS:
                tot = 0.0
                for dd in layers.values():
                    r = dd["by_rows"].get(str(m), {})
                    if f"{kern}_checked_ms" not in r:
                        tot = None
                        break
                    tot += r[f"{kern}_checked_ms"] - r["linear_ms"]
                row[kern] = None if tot is None else round(tot * 1000, 1)        # microseconds
            for dd in layers.values():
                r = dd["by_rows"].get(str(m), {})
                agree &= "error" not in r and _agrees(r, "v2")
            if model == "qwen7b":
                cheaper &= row["v1"] is not None and row["v2"] is not None and row["v2"] < row["v1"]
            added[str(m)] = row
        out["added_ms"][model] = added
    out["criteria"]["2 scores agree with the plain check on every shape and row count"] = agree
    out["criteria"]["3 cheaper than v1 over the four 7B layers at every row count"] = cheaper
    out["v2_becomes_default"] = all(out["criteria"].values())
    # 16 or 64 blocks per _pairs program instead of 32: used if its scores agree everywhere and it is cheaper over the
    # 7B layers at most row counts (the faster of the two if both qualify)
    out["pairs_p"] = 32
    best = None
    for kern, p in (("v2_p16", 16), ("v2_p64", 64)):
        ok = all("error" not in dd["by_rows"].get(str(m), {"error": 1}) and _agrees(dd["by_rows"][str(m)], kern)
                 for layers in res.get("models", {}).values() for dd in layers.values() for m in ROWS)
        rows7 = out["added_ms"].get("qwen7b", {})
        wins = sum(1 for m in ROWS if rows7.get(str(m), {}).get(kern) is not None
                   and rows7[str(m)].get("v2") is not None and rows7[str(m)][kern] < rows7[str(m)]["v2"])
        total = sum(rows7.get(str(m), {}).get(kern) or 0.0 for m in ROWS)
        if ok and wins > len(ROWS) // 2 and (best is None or total < best[1]):
            best = (p, total)
    if best is not None:
        out["pairs_p"] = best[0]
    return out


@app.local_entrypoint()
def main() -> None:
    src = _digest(LOCAL)
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files). One H100, about 10 to 15 minutes, about $1.",
          flush=True)
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
        print(f"\n{model}: check cost as % of the layer's own matmul (v1 -> v2), by rows")
        print("  layer           " + "".join(f"{m:>14}" for m in ROWS))
        for name, dd in layers.items():
            cells = []
            for m in ROWS:
                r = dd["by_rows"][str(m)]
                if "error" in r or "v2_check_pct" not in r:
                    cells.append("error")
                    continue
                cells.append(f"{r['v1_check_pct']:.0f}->{r['v2_check_pct']:.0f}{'' if _agrees(r, 'v2') else '!'}")
            print(f"  {name:<16}" + "".join(f"{c:>14}" for c in cells))
        print(f"  added time over the {len(layers)} layers, microseconds (" + ", ".join(KERNELS) + ")")
        for m in ROWS:
            print(f"    {m:>6} rows: " + "  ".join(f"{k} {summary['added_ms'][model][str(m)][k]}" for k in KERNELS))
    print("\n(! marks a row count where v2's scores didn't agree with the plain check)")
    for name, ok in summary["criteria"].items():
        print(f"  {'met' if ok else 'NOT met'}: {name}")
    print(f"v2 becomes the default: {summary['v2_becomes_default']}; blocks per _pairs program: {summary['pairs_p']}. "
          f"Saved {path}. Estimated cost ${res.get('estimated_cost_usd')}.")
