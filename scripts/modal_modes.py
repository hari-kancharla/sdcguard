"""The math-mode check and the sensitivity comparison on real GPUs: a T4, L4, A100, H100 and B200 at once.

    modal run scripts/modal_modes.py

For each GPU: each of the six math modes, many times over every input distribution, to see how far clean
results stay from the alarm (calibration); experiments/sensitivity.py (bf16 check vs fp32 check on the same
matmuls); the 20-second math-mode screen exactly as `sdcguard check` runs it; and experiments/precise_gpu.py
(precise mode in training: false alarms, what the model sees, cost, injected faults caught). Results go to
results/math_modes/.

About 7 minutes per GPU, about $2 in total at Modal's prices on Oct 18, 2026 (per hour: B200 $6.25,
H100 $3.95, A100 80GB $2.50, L4 $0.80, T4 $0.59, plus about $0.38 per machine for CPU and memory). Every
function stops at 15 minutes, so it can't cost more than about $4.
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
LIMIT = 15 * 60
CALIBRATE_SECONDS = 120


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy", "pytest")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-modes", image=image)


def _work(label: str) -> dict:
    import torch
    from sdcguard import modes
    import sensitivity

    dev = torch.device("cuda")
    out = {"label": label, "gpu": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0)),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,ecc.mode.current",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    print(label, out["gpu"], out["cc"], flush=True)

    # 1. calibration: clean results only, every mode and distribution, as many rounds as fit
    shape = modes.SHAPE["cuda"]
    gen = torch.Generator().manual_seed(7)
    cal = {}
    for m in modes.MODES:
        why = modes.available(m, dev)
        cal[m] = {"skipped": why} if why else {"matmuls": 0, "wrong": 0, "worst_z": {}, "typical_caught": {},
                                                "tensor_core_steps": modes.n_fma(m, tuple(out["cc"]))}
    t_end = time.time() + CALIBRATE_SECONDS
    rounds = 0
    while time.time() < t_end:
        for dist, scale in modes.VARIANTS:
            v = f"{dist} x{scale:g}"
            for m, rec in cal.items():
                if "skipped" in rec or "error" in rec:
                    continue
                try:
                    r = modes.check_once(m, dev, shape, dist, scale, gen)
                except Exception as e:  # noqa: BLE001
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    continue
                rec["matmuls"] += 1
                rec["wrong"] += bool(r["wrong_blocks"])
                if r["worst_z"] is not None:
                    z = r["worst_z"] if math.isfinite(r["worst_z"]) else 1e9
                    rec["worst_z"][v] = round(max(rec["worst_z"].get(v, 0.0), z), 3)
                if r["typical_caught"] is not None:
                    rec["typical_caught"][v] = float(f"{max(rec['typical_caught'].get(v, 0.0), r['typical_caught']):.3g}")
        rounds += 1
    out["calibration"] = {"shape": list(shape), "rounds": rounds, "modes": cal}
    print(label, "calibration done:", rounds, "rounds", flush=True)

    # 2. the bf16 check vs the fp32 check on the same matmuls (fp16 on GPUs without bf16)
    try:
        t0 = time.time()
        low = "bf16" if out["cc"][0] >= 8 else "fp16"
        sens = sensitivity.run(dev, shape, trials=8, say=lambda s: None, low=low)
        sens.update(device="cuda", gpu=out["gpu"], torch=out["torch"], shape=list(shape),
                    seconds=round(time.time() - t0, 1), date=time.strftime("%Y-%m-%d"))
        out["sensitivity"] = sens
        out["sensitivity_md"] = sensitivity.markdown(sens)
    except Exception as e:  # noqa: BLE001
        out["sensitivity_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(label, "sensitivity done", flush=True)

    # 3. the screen as `sdcguard check` runs it
    try:
        out["screen"] = modes.screen(dev, 20.0)
    except Exception as e:  # noqa: BLE001
        out["screen_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(label, "screen done", flush=True)

    # 4. precise mode in training: false alarms, what the model sees, cost, injected faults caught
    try:
        import precise_gpu
        t0 = time.time()
        prec = precise_gpu.run(dev, clean_steps=30, fault_steps=20, rate=0.05, say=lambda s: None)
        prec.update(gpu=out["gpu"], torch=out["torch"], seconds=round(time.time() - t0, 1))
        out["precise"] = prec
        out["precise_md"] = precise_gpu.markdown(prec)
    except Exception as e:  # noqa: BLE001
        out["precise_error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(label, "precise mode done", flush=True)
    return out


KW = dict(cpu=4, memory=16384, timeout=LIMIT)


@app.function(gpu="T4", **KW)
def t4() -> dict:
    return _work("t4")


@app.function(gpu="L4", **KW)
def l4() -> dict:
    return _work("l4")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _work("a100_80gb")


@app.function(gpu="H100!", **KW)
def h100() -> dict:
    return _work("h100")


@app.function(gpu="B200", **KW)
def b200() -> dict:
    return _work("b200")


def _summary(results: list) -> str:
    lines = ["# Math modes on real GPUs", "",
             "Clean results only. Worst score: the largest distance from the CPU's float64 answer, in units of the "
             "noise model (the alarm is at 6). Typical sensitivity: the size of an error in one output that the check "
             "typically catches (median over blocks), relative to a typical output, for the least sensitive input "
             "distribution. Not a guarantee for every position or every error pattern.", "",
             "| GPU | mode | matmuls | wrong | worst score | typical sensitivity |", "|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['label']} | could not run: {r['error']} | | | | |")
            continue
        for m, rec in r["calibration"]["modes"].items():
            if "skipped" in rec or "error" in rec:
                lines.append(f"| {r['gpu']} | {m} | not run: {rec.get('skipped') or rec.get('error')} | | | |")
                continue
            wz = max(rec["worst_z"].values()) if rec["worst_z"] else None
            sc = max(rec["typical_caught"].values()) if rec["typical_caught"] else None
            lines.append(f"| {r['gpu']} | {m} | {rec['matmuls']} | {rec['wrong']} | "
                         f"{'exact' if m == 'int8' else f'{wz:.2f}'} | "
                         f"{'exact' if m == 'int8' else f'{100 * sc:.3g}%'} |")
    lines.append("")
    for r in results:
        if "sensitivity_md" in r:
            lines += ["", r["sensitivity_md"]]
        if "precise_md" in r:
            lines += ["", r["precise_md"]]
        for k in ("sensitivity_error", "screen_error", "precise_error"):
            if k in r:
                lines += ["", f"{r['label']}: {k.replace('_', ' ')}: {r[k]}"]
    return "\n".join(lines) + "\n"


@app.local_entrypoint()
def main() -> None:
    jobs = [("t4", t4), ("l4", l4), ("a100_80gb", a100), ("h100", h100), ("b200", b200)]
    started = [(name, fn.spawn()) for name, fn in jobs]
    print(f"Started {len(started)} GPUs. Keep this terminal open until it prints Done.", flush=True)
    results = []
    for name, call in started:
        try:
            results.append(call.get(timeout=LIMIT + 300))
            print(f"{name}: finished", flush=True)
        except Exception as e:  # noqa: BLE001
            results.append({"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            print(f"{name}: could not finish ({type(e).__name__})", flush=True)
    dest = LOCAL / "results" / "math_modes"
    dest.mkdir(parents=True, exist_ok=True)
    for r in results:
        (dest / f"{r['label']}.json").write_text(json.dumps(r, indent=2, default=str))
    (dest / "SUMMARY.md").write_text(_summary(results))
    print(_summary(results))
    print(f"Done. Results in {dest}", flush=True)
