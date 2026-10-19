"""NVIDIA's MPS cuBLASLt bug, live on an H200: does sdcguard catch it?

    modal run scripts/modal_mps.py

Starts the MPS daemon on one H200, then runs experiments/mps_unwritten.py as an MPS client at several
active-thread shares, each in a fresh process (a client reads its share when it starts). In the bug report
(NVIDIA developer forum thread 384873, Oct 2, 2026) shares that give an H200 18 to 28, 36 to 60 or 68 to 116
SMs leave part of a 16x2048x1000 fp16/bf16 output unwritten, and 100% is always right. So 100% and 50% (66
SMs) are the controls, and 14%, 20%, 40% and 75% are expected to show the bug. If Modal's sandbox doesn't
let the MPS daemon or its clients run, this says so and stops.

One H200 for at most 10 minutes: at most about $0.82 at Modal's prices on Oct 18, 2026 ($4.54 an hour for
the GPU plus about $0.38 for CPU and memory); usually 3 to 5 minutes. Results go to results/mps_bug/.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 10 * 60
SHARES = (100, 14, 20, 40, 50, 75)


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-mps", image=image)


def _tail(path: str, n: int = 3000) -> str:
    try:
        with open(path, errors="replace") as f:
            return f.read()[-n:]
    except OSError as e:
        return f"({e})"


def _versions() -> dict:
    from importlib import metadata
    return {d.metadata["Name"]: d.version for d in metadata.distributions()
            if d.metadata["Name"] and ("cublas" in d.metadata["Name"].lower() or d.metadata["Name"] == "torch")}


@app.function(gpu="H200", cpu=4, memory=16384, timeout=LIMIT)
def h200() -> dict:
    t0 = time.time()
    out = {"nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_mode",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "packages": _versions(), "runs": []}
    mps = shutil.which("nvidia-cuda-mps-control")
    out["mps_control"] = mps
    if not mps:
        out["stopped"] = "nvidia-cuda-mps-control isn't in this container, so MPS can't be started here"
        return out
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_pipe", CUDA_MPS_LOG_DIRECTORY="/tmp/mps_log")
    os.makedirs(env["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
    os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    p = subprocess.run([mps, "-d"], env=env, capture_output=True, text=True, timeout=60)
    out["mps_start"] = {"exit": p.returncode, "output": (p.stdout + p.stderr)[-1000:]}
    if p.returncode != 0:
        out["stopped"] = "the MPS daemon didn't start"
        return out
    try:
        for share in SHARES:
            e = dict(env, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(share))
            try:
                c = subprocess.run([sys.executable, f"{R}/experiments/mps_unwritten.py"], env=e, capture_output=True,
                                   text=True, timeout=150, cwd=R)
                line = next((ln for ln in reversed(c.stdout.splitlines()) if ln.startswith("{")), None)
                rec = json.loads(line) if line else {"share": str(share), "exit": c.returncode,
                                                     "error": (c.stdout + c.stderr)[-2000:]}
            except subprocess.TimeoutExpired:
                rec = {"share": str(share), "error": "timed out after 150 s"}
            out["runs"].append(rec)
            print(f"share {share}%: done", flush=True)
            if share == 100 and "error" in rec:
                out["stopped"] = "the first MPS client (100%) couldn't run, so the rest would fail too"
                break
    finally:
        subprocess.run([mps], input="quit\n", env=env, capture_output=True, text=True, timeout=60)
        out["mps_logs"] = {"control": _tail("/tmp/mps_log/control.log"), "server": _tail("/tmp/mps_log/server.log")}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _row(r: dict) -> str:
    if "error" in r:
        return f"| {r.get('share')}% | | could not run: {r['error'][-200:]!r} | |"
    cells = []
    for dt in ("fp16", "bf16"):
        d = r.get(dt) or {}
        if "error" in d:
            cells.append(f"{dt}: error")
            continue
        cells.append(f"{dt}: {d['wrong_outputs']:,}/{d['outputs']:,} wrong (NaN-filled buffer: "
                     f"{d['wrong_outputs_nan_prefilled']:,}); training check {d['training_check']['alerts']} alert(s), "
                     f"serving check {d['serving_check']['alerts']}")
    mm = r.get("math_modes") or {}
    wrong = sum(v.get("wrong", 0) for v in mm.values() if isinstance(v, dict))
    return f"| {r['share']}% | {r['sms_reported']} | {' ; '.join(cells)} | math modes: {wrong} wrong |"


@app.local_entrypoint()
def main() -> None:
    print("Starting one H200 (at most 10 minutes). Keep this terminal open until it prints Done.", flush=True)
    res = h200.remote()
    d = LOCAL / "results" / "mps_bug"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"h200_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    print(res.get("nvidia_smi"), res.get("packages"))
    if res.get("stopped"):
        print("Stopped:", res["stopped"])
    print("| share | SMs reported | layer 16x2048x1000 | sdcguard check |")
    print("|---|---|---|---|")
    for r in res.get("runs", []):
        print(_row(r))
    print(f"Saved {path}. Done.")
