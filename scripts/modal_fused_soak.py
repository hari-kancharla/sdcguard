"""The fused precise check on long clean training (experiments/fused_soak.py), on an H100 and an L4 at once.

    modal run scripts/modal_fused_soak.py

40 minutes of training per GPU: about $3 (H100) + $0.80 (L4) at Modal's prices on Oct 20, 2026 (per hour: H100
$3.95, L4 $0.80, plus about $0.45 per machine for 4 cores and 32 GiB). Every function stops at 55 minutes, so the
requested resources can't cost more than about $4.50; startup and the image build add a little. Each result records
its measured function time and the cost that implies. Criteria: docs/METHOD.md ("The fused precise check, a long
clean run"), written before the run. Results go to results/fused_soak/. Exits with an error unless both GPUs met
every criterion.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
MINUTES = 40.0
LIMIT = 55 * 60
RATES = {"h100": 3.95 + 4 * 0.0472 + 32 * 0.008, "l4": 0.80 + 4 * 0.0472 + 32 * 0.008}


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


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs", "fixtures") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-fused-soak", image=image)
KW = dict(cpu=4, memory=32768, timeout=LIMIT)


def _work(label: str, source: dict) -> dict:
    import os
    import tempfile
    import fused_soak
    import torch
    t0 = time.monotonic()
    out = tempfile.mkdtemp()
    try:
        rec = fused_soak.soak(torch.device("cuda"), MINUTES, out)
        rec["failures"] = fused_soak.judge(rec)
    except Exception as e:  # noqa: BLE001  (a crash is a result too, never a pass)
        partial = os.path.join(out, "soak_cuda.json")
        rec = json.load(open(partial)) if os.path.exists(partial) else {}
        rec["failures"] = [f"crashed: {type(e).__name__}: {str(e)[:300]}"]
    minutes = (time.monotonic() - t0) / 60
    rec.update(label=label, source=source, function_minutes=round(minutes, 1),
               estimated_cost_usd=round(minutes / 60 * RATES[label], 2))
    return rec


@app.function(gpu="H100!", **KW)
def h100(source: dict) -> dict:
    return _work("h100", source)


@app.function(gpu="L4", **KW)
def l4(source: dict) -> dict:
    return _work("l4", source)


@app.local_entrypoint()
def main() -> None:
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files). {MINUTES:.0f} minutes of clean training "
          "per GPU on an H100 and an L4 at once: about $3.80 in all.", flush=True)
    calls = {"h100": h100.spawn(src), "l4": l4.spawn(src)}
    dest = LOCAL / "results" / "fused_soak"
    dest.mkdir(parents=True, exist_ok=True)
    failed = []
    for label, call in calls.items():
        try:
            rec = call.get()
        except Exception as e:  # noqa: BLE001
            rec = {"label": label, "failures": [f"did not complete: {type(e).__name__}: {e}"]}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (dest / f"{label}_{stamp}.json").write_text(json.dumps(rec, indent=1, default=str))
        print(f"{label}: {rec.get('fused_checks')} fused checks, {rec.get('false_alarms')} alarms, worst "
              f"{rec.get('worst_clean_z')}, failures {rec.get('failures')}", flush=True)
        if rec.get("failures"):
            failed.append(label)
    if failed:
        raise SystemExit(f"CRITERIA NOT MET on {', '.join(failed)}: see results/fused_soak/")
    print("Done: every criterion met on both GPUs.", flush=True)
