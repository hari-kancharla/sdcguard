"""Find the Triton training-check kernel that faults on Blackwell (experiments/triton_blackwell.py) on a B200.

    modal run scripts/modal_blackwell.py

About 5 minutes on one B200, about $0.55 at Modal's prices on Oct 18, 2026 ($6.25 an hour for the GPU plus
about $0.38 for CPU and memory); it stops at 15 minutes, so at most about $1.70. Results go to
results/blackwell_triton/.
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
LIMIT = 15 * 60


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-blackwell", image=image)


@app.function(gpu="B200", cpu=4, memory=16384, timeout=LIMIT)
def b200() -> dict:
    t0 = time.time()
    p = subprocess.run([sys.executable, f"{R}/experiments/triton_blackwell.py"], capture_output=True, text=True,
                       cwd=R, timeout=LIMIT - 60)
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("RESULT ")), None)
    return {"seconds": round(time.time() - t0, 1), "exit": p.returncode, "head": p.stdout[:600],
            "result": json.loads(line[7:]) if line else None, "stderr": p.stderr[-3000:]}


@app.local_entrypoint()
def main() -> None:
    print("Starting one B200 (at most 15 minutes). Keep this terminal open until it prints Done.", flush=True)
    r = b200.remote()
    dest = LOCAL / "results" / "blackwell_triton"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"b200_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(r, indent=2, default=str))
    print(r["head"])
    print(json.dumps(r["result"], indent=1, default=str)[:4000] if r["result"] else r["stderr"])
    print(f"Saved {path}. Done.", flush=True)
