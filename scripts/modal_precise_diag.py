"""Precise mode's false alarms, taken apart on an H100, an A100 and an L4 at once (experiments/precise_diag.py).

    modal run scripts/modal_precise_diag.py

About 4 minutes per GPU, about $0.50 in total at Modal's prices on Oct 18, 2026 (per hour: H100 $3.95,
A100 80GB $2.50, L4 $0.80, plus about $0.38 per machine for CPU and memory). Every function stops at 12
minutes, so it can't cost more than about $1.70. Results go to results/precise_diag/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 12 * 60


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-precise-diag", image=image)
KW = dict(cpu=4, memory=32768, timeout=LIMIT)


def _work(label: str) -> dict:
    import torch
    import precise_diag
    t0 = time.time()
    try:
        res = precise_diag.run(torch.device("cuda"), steps=8, say=lambda s: print(label, s, flush=True))
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:]}
    res["label"] = label
    res["seconds"] = round(time.time() - t0, 1)
    return res


@app.function(gpu="H100!", **KW)
def h100() -> dict:
    return _work("h100")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _work("a100_80gb")


@app.function(gpu="L4", **KW)
def l4() -> dict:
    return _work("l4")


def _summary(r: dict) -> str:
    if "error" in r:
        return f"## {r['label']}: could not run: {r['error']}\n"
    lines = [f"## {r.get('gpu')} ({r['dtype']}): {r['checks']} checks over {r['steps']} steps, {r['alarms']} alarms", "",
             "| layer, pass | m x k x n | tensor-core steps of | checks | over 6 | worst |", "|---|---|---|---|---|---|"]
    for key, b in r["by_layer_and_pass"].items():
        lines.append(f"| {key} | {' x '.join(map(str, b['m_k_n']))} | {b['nfma']} | {b['checks']} | {b['over_6']} | "
                     f"{b['worst']:.2f} |")
    lines.append("")
    for d in r["detailed"][:6]:
        lines.append(f"- {d['layer']} {d['pass']} step {d['step']}: score {d['z']} (plain {d['z_plain']}, signed "
                     f"{d['z_signed']}); coherence {d['worst_block_coherence']} (median {d['all_blocks_coherence_median']}); "
                     f"toward zero {d['worst_block_toward_zero']}; error in u x |output| median "
                     f"{d['worst_block_err_in_u_times_output']['median']}; same bits rerun {d['same_bits_when_rerun']}; "
                     f"plain fp32 FMA score {d['z_with_plain_fp32_fma']}; kernels {d['kernels'][:2]}")
    lines.append(f"\nmath modes: {r.get('math_modes')}\n")
    return "\n".join(lines)


@app.local_entrypoint()
def main() -> None:
    jobs = [("h100", h100), ("a100_80gb", a100), ("l4", l4)]
    started = [(name, fn.spawn()) for name, fn in jobs]
    print(f"Started {len(started)} GPUs (at most 12 minutes each). Keep this terminal open until it prints Done.",
          flush=True)
    dest = LOCAL / "results" / "precise_diag"
    dest.mkdir(parents=True, exist_ok=True)
    text = []
    for name, call in started:
        try:
            r = call.get(timeout=LIMIT + 300)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        text.append(_summary(r))
        print(f"{name}: finished", flush=True)
    (dest / "SUMMARY.md").write_text("# Precise mode, check by check\n\n" + "\n".join(text))
    print("\n".join(text))
    print(f"Done. Results in {dest}", flush=True)
