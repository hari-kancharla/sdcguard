"""0.5.0's serving kernel on a GPU: the checks the new loop has to pass before it ships.

    modal run scripts/modal_v050.py

One H100:
  - tests/_serve_fast_compare.py on the GPU: the Triton check against the plain PyTorch check on every case of
    the comparison set (scores, positions, counts and captured rows); must print ALL AGREE;
  - from experiments/v040_check.py, the serving capture checks of 0.4.0 with their criteria as written then
    (docs/METHOD.md, "0.4.0 on GPUs"): 6 (a one-time fault kept as it alarmed inside a CUDA graph, a checker fault
    not blamed), 7 (capture costs the clean check at most 5%) and 8 (capture under contention).

About 10 minutes: about $0.75 at Modal's prices on Oct 20, 2026 ($3.95 an hour for the H100 plus about $0.45 for
CPU and memory); at most about $2.20 (a 30-minute limit). Results go to results/v050/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 30 * 60
RATE = 3.95 + 4 * 0.0472 + 32 * 0.008


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
    .uv_pip_install("torch==2.14.1", "numpy", "transformers==5.20.1")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_dir(LOCAL / "scripts", f"{R}/scripts", ignore=_skip)        # (part of the fingerprint)
)
app = modal.App("sdcguard-v050", image=image)


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
    out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    p = subprocess.run([sys.executable, f"{R}/tests/_serve_fast_compare.py", "cuda"], capture_output=True, text=True,
                       timeout=900, cwd=R)
    lines = p.stdout.strip().splitlines()
    out["compare"] = {"exit": p.returncode, "cases": len(lines), "last": lines[-1] if lines else None,
                      "disagree": [ln for ln in lines if '"agree": false' in ln][:10],
                      "stderr": p.stderr[-1500:]}
    import v040_check
    dev = torch.device("cuda")
    rep = {}
    for name, fn in (("capture", lambda: v040_check.capture_check(dev, 200)),
                     ("capture_stress", lambda: v040_check.capture_stress(dev))):
        try:
            rep[name] = fn()
        except Exception as e:  # noqa: BLE001
            rep[name] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
    c = v040_check.criteria(rep)
    out["report"] = rep
    out["criteria"] = {k: v for k, v in c.items() if k[:2] in ("6 ", "7 ", "8 ")}
    out["all met"] = (out["compare"]["exit"] == 0 and out["compare"]["last"] == "ALL AGREE"
                      and all(v is True for v in out["criteria"].values()))
    minutes = (time.monotonic() - t0) / 60
    out["minutes"] = round(minutes, 1)
    out["estimated_cost_usd"] = round(minutes / 60 * RATE, 2)
    return out


@app.local_entrypoint()
def main() -> None:
    src = _digest(LOCAL)
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files). One H100, about 10 minutes, about $0.75.",
          flush=True)
    res = h100.remote(src["sha256"])
    d = LOCAL / "results" / "v050"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"h100_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"launcher_source": src, **res}, indent=2, default=str))
    if res.get("stopped"):
        print("Stopped:", res["stopped"])
    print("compare on the GPU:", (res.get("compare") or {}).get("last"), "exit", (res.get("compare") or {}).get("exit"))
    for k, v in (res.get("criteria") or {}).items():
        print(f"  {k}: {v}")
    print(f"ALL MET: {res.get('all met')}. Saved {path}. Estimated cost ${res.get('estimated_cost_usd')}.")
    if not res.get("all met"):
        raise SystemExit("0.5.0 serving kernel: not every check passed")
