"""Precise mode's noise model and two candidate fixes, scored on 30 clean training steps on five GPUs at once
(experiments/precise_models.py): H100, B200, A100, L4 (bf16) and T4 (fp16).

    modal run scripts/modal_precise_models.py

About 6 minutes per GPU, about $2 in total at Modal's prices on Oct 18, 2026 (per hour: B200 $6.25, H100
$3.95, A100 80GB $2.50, L4 $0.80, T4 $0.59, plus about $0.38 per machine for CPU and memory). Every function
stops at 20 minutes, so it can't cost more than about $5.40. Results go to results/precise_models/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent


def _source() -> dict:
    """Fingerprint of the code this run used (src, experiments, tests, scripts), to match it to a commit."""
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "experiments", "tests", "scripts"):
        for p in sorted((LOCAL / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(LOCAL)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    return {"sha256": h.hexdigest(), "files": n}


R = "/root/sdcguard"
LIMIT = 20 * 60


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

app = modal.App("sdcguard-precise-models", image=image)
KW = dict(cpu=4, memory=32768, timeout=LIMIT)


def _work(label: str) -> dict:
    import torch
    import precise_models
    t0 = time.time()
    try:
        res = precise_models.run(torch.device("cuda"), steps=30, say=lambda s: print(label, s, flush=True))
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:]}
    res.update(label=label, seconds=round(time.time() - t0, 1), markdown=precise_models.markdown(res))
    return res


@app.function(gpu="H100!", **KW)
def h100() -> dict:
    return _work("h100")


@app.function(gpu="B200", **KW)
def b200() -> dict:
    return _work("b200")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _work("a100_80gb")


@app.function(gpu="L4", **KW)
def l4() -> dict:
    return _work("l4")


@app.function(gpu="T4", **KW)
def t4() -> dict:
    return _work("t4")


@app.local_entrypoint()
def main() -> None:
    jobs = [("h100", h100), ("b200", b200), ("a100_80gb", a100), ("l4", l4), ("t4", t4)]
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    started = [(name, fn.spawn()) for name, fn in jobs]
    print(f"Started {len(started)} GPUs (at most 20 minutes each). Keep this terminal open until it prints Done.",
          flush=True)
    dest = LOCAL / "results" / "precise_models"
    dest.mkdir(parents=True, exist_ok=True)
    text = []
    for name, call in started:
        try:
            r = call.get(timeout=LIMIT + 300)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["source"] = src
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        text.append(r.get("markdown") or f"## {name}: could not run: {r.get('error')}\n")
        print(f"{name}: finished", flush=True)
    (dest / "SUMMARY.md").write_text("# Precise mode: noise models on real training matmuls\n\n" + "\n".join(text))
    print("\n".join(text))
    print(f"Done. Results in {dest}", flush=True)
