"""Round 3 GPU runs on a Modal H100: fast training mode, the screen, the serving check.

    modal run scripts/modal_v2.py --phase check   # about 35 min, about $2.40
    modal run scripts/modal_v2.py --phase serve   # about 25 min, about $1.65
    modal run scripts/modal_v2.py --phase fetch   # download the results again

check: tests, self-test, Triton vs plain check comparison, bit-identity check; training slowdown (the old
       operator-level hook vs fast mode) on GPT-2 small and a 0.9B model; a 3-minute screen, a 30-second
       screen with simulated faults and a replay of its evidence; the serving check on a 1.1B model.
serve: experiments/bench_serve.py: the serving check's Triton kernel vs its PyTorch version vs the training
       check, and its cost on 1.1B and 7B Llama-shaped models under CUDA graphs.

vLLM and NVBit runs live in modal_vllm.py and modal_nvbit.py. Results are saved to the Modal volume
sdcguard-runs and downloaded to runs/gpu_r3/. Every function has a hard time limit, and a phase won't start
again within an hour of its last start unless you pass --force.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

GPU = "H100!"                 # exactly an H100 (without "!" Modal may upgrade to an H200)
LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
V = "/vol"
ROUND = "r3"
DATA, RUNS = f"{V}/data", f"{V}/runs_{ROUND}"
LOCK_HOURS = 1.5           # longer than any phase's time limit


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")                       # Triton compiles a small launcher once
    .uv_pip_install("torch==2.14.1", "numpy", "matplotlib", "tiktoken", "pytest", "transformers==5.20.1")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1", "TIKTOKEN_CACHE_DIR": f"{DATA}/tiktoken"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml")
)

app = modal.App("sdcguard-v2", image=image)
vol = modal.Volume.from_name("sdcguard-runs", create_if_missing=True)
VOLS = {V: vol}


def _tee(cmd, log_path, env=None, cwd=R):
    """Run a command, stream its output, and write each line to the log as it arrives (Modal commits the
    volume every few seconds, so a hard time limit loses almost nothing). Never raises: every step runs."""
    print("$ " + " ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    code = 127
    with open(log_path, "w") as f:
        try:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 env={**os.environ, **(env or {})})
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            p.wait()
            code = p.returncode
        except OSError as e:
            f.write(f"(could not run: {e})\n")
        f.write(f"\n[exit code {code}]\n")
    return code


def _locked(phase: str, force: bool) -> str | None:
    vol.reload()
    path = f"{V}/locks/{ROUND}_{phase}.json"
    if os.path.exists(path) and not force:
        try:
            age = (time.time() - json.load(open(path))["started"]) / 3600
        except (OSError, ValueError, KeyError):
            age = 0.0
        if age < LOCK_HOURS:
            return (f"NOT STARTED: phase '{phase}' was started {age * 60:.0f} minutes ago and may still be running. "
                    f"Starting it twice would pay twice. Wait, then run --phase fetch.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"started": time.time()}, f)
    vol.commit()
    return None


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=60 * 60)
def check(force: bool = False) -> str:
    msg = _locked("check", force)
    if msg:
        return msg
    import shutil
    out = f"{RUNS}/check"
    if os.path.isdir(out):
        shutil.rmtree(out)
    py = sys.executable
    t0 = time.time()
    try:
        _tee(["nvidia-smi"], f"{out}/nvidia-smi.txt")
        _tee([py, "-m", "pytest", "-q", "tests", "--ignore", "tests/test_fast.py"], f"{out}/pytest.txt")
        _tee([py, "-m", "sdcguard.selftest"], f"{out}/selftest.txt")
        _tee([py, "tests/_fast_compare.py", "cuda"], f"{out}/fast_compare.txt")
        _tee([py, "experiments/gpu_identity.py"], f"{out}/identity.txt", env={"IDENTITY_OUT": f"{out}/identity.json"})
        vol.commit()
        common = ["--timing-steps", "30", "--rounds", "3", "--data-dir", DATA]
        _tee([py, "experiments/calibrate.py", "--preset", "ts-gpt2", "--overhead", "--fast",
              "--sweep", "1:1,2:1,5:1,10:1,20:1,50:1", "--out", f"{out}/runs_fast", *common],
             f"{out}/overhead_fast_gpt2.txt")
        vol.commit()
        _tee([py, "experiments/calibrate.py", "--preset", "ts-gpt2", "--overhead",
              "--sweep", "1:1,10:1,20:1,50:1", "--out", f"{out}/runs_dispatch", *common],
             f"{out}/overhead_dispatch_gpt2.txt")
        vol.commit()
        _tee([py, "experiments/calibrate.py", "--preset", "ts-900m", "--overhead", "--fast",
              "--sweep", "1:1,2:1,5:1,10:1,20:1", "--out", f"{out}/runs_fast", "--timing-steps", "20",
              "--rounds", "2", "--data-dir", DATA], f"{out}/overhead_fast_900m.txt")
        vol.commit()
        _tee([py, "-m", "sdcguard.screen", "--minutes", "3", "--size", "1b", "--out", f"{out}/screen"],
             f"{out}/screen.txt")
        _tee([py, "-m", "sdcguard.screen", "--minutes", "0.5", "--size", "1b", "--simulate-faults", "0.002",
              "--out", f"{out}/screen_simulated"], f"{out}/screen_simulated.txt")
        ev = sorted(str(p) for p in Path(f"{out}/screen_simulated").rglob("evidence_0.pt"))
        if ev:
            _tee([py, "-m", "sdcguard.screen", "--replay", ev[0], "--times", "200"], f"{out}/replay.txt")
        vol.commit()
        # last, because it compiles a 1.1B model several times (if it runs long, the rest is already saved)
        _tee([py, "experiments/bench_serve.py"], f"{out}/serve.txt", env={"SERVE_OUT": f"{out}/serve.json"})
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.time() - t0) / 60:.1f} minutes on {GPU}\n")
        vol.commit()
    return f"check finished in {(time.time() - t0) / 60:.1f} minutes"


@app.function(volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=75 * 60)
def serve(force: bool = False) -> str:
    """The serving check after round 3: the Triton kernel (one per layer) and layer shares."""
    msg = _locked("serve", force)
    if msg:
        return msg
    import shutil
    out = f"{RUNS}/serve"
    if os.path.isdir(out):
        shutil.rmtree(out)
    py = sys.executable
    t0 = time.time()
    try:
        _tee(["nvidia-smi"], f"{out}/nvidia-smi.txt")
        _tee([py, "-m", "pytest", "-q", "tests", "--ignore", "tests/test_fast.py"], f"{out}/pytest.txt")
        _tee([py, "-m", "sdcguard.selftest"], f"{out}/selftest.txt")
        vol.commit()
        _tee([py, "experiments/bench_serve.py"], f"{out}/serve.txt", env={"SERVE_OUT": f"{out}/serve.json"})
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.time() - t0) / 60:.1f} minutes on {GPU}\n")
        vol.commit()
    return f"serve finished in {(time.time() - t0) / 60:.1f} minutes"


@app.function(volumes=VOLS, cpu=1, memory=4096, timeout=15 * 60)
def bundle() -> bytes:
    vol.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if os.path.isdir(RUNS):
            # evidence .pt files (matmul operands, up to hundreds of MB each) stay on the cloud volume
            tar.add(RUNS, arcname=f"gpu_{ROUND}", filter=lambda ti: None if ti.name.endswith(".pt") else ti)
    return buf.getvalue()


ESTIMATES = {
    "check": "about 35 minutes on 1 H100: roughly $2.40 (hard limit 60 minutes, about $4.20)",
    "serve": "about 25 minutes on 1 H100: roughly $1.65 (hard limit 75 minutes, about $4.95)",
    "fetch": "download only: a few cents",
}


@app.local_entrypoint()
def main(phase: str = "check", force: bool = False):
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    try:
        if phase == "check":
            print(check.remote(force), flush=True)
        elif phase == "serve":
            print(serve.remote(force), flush=True)
    finally:
        try:
            data = bundle.remote()
            dest = LOCAL / "runs"
            dest.mkdir(exist_ok=True)
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(dest, filter="data")
            print(f"Results are in {dest / f'gpu_{ROUND}'} ({len(data) / 1e6:.1f} MB)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
    print("\nDone.", flush=True)
