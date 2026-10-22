"""The upgrade gate on a real library bug: cuBLAS known issue 6862950 on an H200 under MPS.

    modal run scripts/modal_gate.py

One H200 with the MPS daemon running; every step is `sdcguard gate`, each run of a case in a fresh process:
  1. capture   a bf16 linear layer of 16 tokens x 2048 -> 1000 with bias (the shape of NVIDIA forum thread 384873)
               as an MPS client at 40%, on PyTorch 2.14.1 with the cuBLAS its wheel brings
  2. again     the same stack at 40%: expected UNCHANGED, still wrong (the gate's control)
  3. full      the same stack at 100%: expected INCONCLUSIVE although every output is right (other conditions:
               a clean rerun somewhere else proves nothing)
  4. newest    PyTorch 2.14.1 with the newest cuBLAS wheel (nvidia-cublas 13.8.1.7) at 40%: NVIDIA lists 6862950 as
               not fixed yet, so expected UNCHANGED (still wrong, or avoided if the heuristic now picks another kernel)
  5. pin       `gate pin`: a TunableOp file forcing algorithm 66 with a 1x1x1 cluster (NVIDIA's workaround), then
               `gate workaround` with it: expected WORKS, with the log showing that kernel on every call
  6. not pinned the same TunableOp switches with a file that doesn't exist: expected DOESN'T WORK (the default kernel
               runs), so a gate that only looked at the switches would be fooled (INCONCLUSIVE since the change below)

These are expectations written before the run (October 22, 2026), not results. Every verdict is saved as it came
out, including any that differ from them.

Changed after the first run (October 22, before the second; docs/METHOD.md has both runs): that run met 1 to 4 and
6, and not 5, because `gate pin` wrote pin.csv while PyTorch, which puts the GPU's number in the file name, looked
for pin0.csv; the default kernel ran and `gate workaround` said DOESN'T WORK instead of the INCONCLUSIVE written for
a pin that isn't applied. `gate pin` now writes the file PyTorch reads, and `gate workaround` says INCONCLUSIVE
whenever the workaround wasn't what ran (the file PyTorch reads isn't there, or the log shows another kernel), so
step 6 is now expected INCONCLUSIVE (still never WORKS), and 1 to 5 stay as written.

One H200 for at most 30 minutes: at most about $2.50 at Modal's prices on Oct 20, 2026 ($4.54 an hour for the GPU
plus about $0.45 for CPU and memory); usually 10 to 15 minutes. Building the image (two PyTorch installs) runs on
CPU first. Results go to results/gate_h200/.
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
LIMIT = 30 * 60
CAND = "/opt/cand"                  # a second environment: the same PyTorch with the newest cuBLAS
NEWEST_CUBLAS = "13.8.1.7"


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
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .run_commands("pip install uv",
                  f"uv venv --python 3.12 {CAND}",
                  f"uv pip install --python {CAND}/bin/python torch==2.14.1 numpy",
                  f"uv pip install --python {CAND}/bin/python --no-deps nvidia-cublas=={NEWEST_CUBLAS}")
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
)

app = modal.App("sdcguard-gate", image=image)


def _run(cmd, env, timeout=600) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return {"cmd": " ".join(cmd), "exit": p.returncode, "seconds": round(time.time() - t0, 1),
                "output": (p.stdout + p.stderr)[-6000:]}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "exit": None, "error": f"timed out after {timeout} s"}


def _read(path):
    try:
        with open(path) as f:
            return json.load(f) if path.endswith(".json") else f.read()
    except (OSError, ValueError) as e:
        return {"missing": str(e)}


def _versions(python) -> str:
    code = ("import importlib.metadata as m, torch; print(torch.__version__, torch.version.cuda, "
            "[(d.metadata['Name'], d.version) for d in m.distributions() if 'cublas' in (d.metadata['Name'] or '')])")
    p = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=120)
    return (p.stdout + p.stderr).strip()[-500:]


@app.function(gpu="H200", cpu=4, memory=32768, timeout=LIMIT)
def h200() -> dict:
    t0 = time.time()
    out = {"nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
           "stack_main": _versions(sys.executable), "stack_newest": _versions(f"{CAND}/bin/python"), "steps": {}}
    mps = shutil.which("nvidia-cuda-mps-control")
    if not mps:
        out["stopped"] = "nvidia-cuda-mps-control isn't in this container"
        return out
    base = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_pipe", CUDA_MPS_LOG_DIRECTORY="/tmp/mps_log")
    os.makedirs(base["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
    os.makedirs(base["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    p = subprocess.run([mps, "-d"], env=base, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        out["stopped"] = f"the MPS daemon didn't start: {(p.stdout + p.stderr)[-500:]}"
        return out
    w = "/tmp/gate"
    pack = f"{w}/pack"
    os.makedirs(w, exist_ok=True)
    s40 = dict(base, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="40")
    s100 = dict(base, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE="100")
    py, cand = sys.executable, f"{CAND}/bin/python"
    g = ["-m", "sdcguard", "gate"]
    steps = out["steps"]
    try:
        steps["capture"] = _run([py, *g, "capture", "--shape", "1000,2048,16", "--bias", "--dtype", "bfloat16",
                                 "--pack", pack, "--repeats", "20", "--name", "h200-mps40-linear-bf16-16x2048x1000"],
                                s40)
        case = f"{pack}/h200-mps40-linear-bf16-16x2048x1000"
        steps["case"] = _read(f"{case}/case.json")
        steps["run_base40"] = _run([py, *g, "run", pack, "--out", f"{w}/base40.json", "--repeats", "20"], s40)
        steps["run_again40"] = _run([py, *g, "run", pack, "--out", f"{w}/again40.json", "--repeats", "20"], s40)
        steps["run_full100"] = _run([py, *g, "run", pack, "--out", f"{w}/full100.json", "--repeats", "20"], s100)
        steps["run_newest40"] = _run([cand, *g, "run", pack, "--out", f"{w}/newest40.json", "--repeats", "20"],
                                     dict(s40, PYTHONPATH=f"{R}/src"))
        for name, other in (("again40", "again40"), ("full100", "full100"), ("newest40", "newest40")):
            steps[f"compare_{name}"] = _run([py, *g, "compare", f"{w}/base40.json", f"{w}/{other}.json", "--json",
                                             f"{w}/verdict_{name}.json"], s40)
            steps[f"verdict_{name}"] = _read(f"{w}/verdict_{name}.json")
        steps["pin"] = _run([py, *g, "pin", case, "--out", f"{w}/pin.csv"], s40)
        # what `gate pin` wrote: pin<the GPU's number>.csv, which PyTorch reads for PYTORCH_TUNABLEOP_FILENAME=pin.csv
        steps["pin_file"] = {f: _read(f"{w}/{f}") for f in sorted(os.listdir(w))
                             if f.startswith("pin") and f.endswith(".csv")}
        tun = ["--env", "PYTORCH_TUNABLEOP_ENABLED=1", "--env", "PYTORCH_TUNABLEOP_TUNING=0"]
        steps["workaround_pin"] = _run([py, *g, "workaround", case, *tun, "--env",
                                        f"PYTORCH_TUNABLEOP_FILENAME={w}/pin.csv", "--repeats", "50",
                                        "--expect-kernel", "algoId=66 cluster=1x1x1", "--json",
                                        f"{w}/verdict_workaround_pin.json"], s40, timeout=900)
        steps["verdict_workaround_pin"] = _read(f"{w}/verdict_workaround_pin.json")
        steps["workaround_no_file"] = _run([py, *g, "workaround", case, *tun, "--env",
                                            f"PYTORCH_TUNABLEOP_FILENAME={w}/does_not_exist.csv", "--repeats", "50",
                                            "--expect-kernel", "algoId=66 cluster=1x1x1", "--json",
                                            f"{w}/verdict_workaround_no_file.json"], s40, timeout=900)
        steps["verdict_workaround_no_file"] = _read(f"{w}/verdict_workaround_no_file.json")
        for f in ("base40", "again40", "full100", "newest40"):
            steps[f"result_{f}"] = _read(f"{w}/{f}.json")
    finally:
        subprocess.run([mps], input="quit\n", env=base, capture_output=True, text=True, timeout=60)
    out["seconds"] = round(time.time() - t0, 1)
    out["cost_estimate_usd"] = round(out["seconds"] / 3600 * (4.54 + 0.45), 2)
    return out


@app.local_entrypoint()
def main() -> None:
    print("Starting one H200 (at most 30 minutes). Keep this terminal open until it prints Done.", flush=True)
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    res = h200.remote()
    res["source"] = src
    d = LOCAL / "results" / "gate_h200"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"h200_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    print(res.get("nvidia_smi"))
    print("main stack:", res.get("stack_main"))
    print("newest cuBLAS stack:", res.get("stack_newest"))
    if res.get("stopped"):
        print("Stopped:", res["stopped"])
    steps = res.get("steps", {})
    cap = steps.get("capture", {})
    print("\ncapture at 40%:", cap.get("exit"), (cap.get("output") or "")[-600:])
    for name in ("again40", "full100", "newest40"):
        print(f"\nbase40 vs {name}:", (steps.get(f"compare_{name}") or {}).get("output", "")[-900:])
    print("\npin:", (steps.get("pin") or {}).get("output", "")[-900:])
    for name in ("workaround_pin", "workaround_no_file"):
        print(f"\n{name}:", (steps.get(name) or {}).get("output", "")[-900:])
    print(f"\nSaved {path}. Done. Estimated cost: ${res.get('cost_estimate_usd')}")
