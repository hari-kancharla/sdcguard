"""The sdcguard vLLM plugin on real vLLM 0.31, on a Modal H100.

    modal run scripts/modal_vllm.py --phase vllm   # about 25 min, about $1.65
    modal run scripts/modal_vllm.py --phase demo   # 1-minute live demo (experiments/demo_vllm.py), about $0.40

vllm: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct without sdcguard, with it on every layer and on 1/8 of
      the layers: outputs, tokens/s, alerts on clean traffic, and a flipped weight bit caught and located.

Results are saved to the Modal volume sdcguard-runs and downloaded to runs/gpu_r3/. Every function has a
hard time limit, and a phase won't start again within an hour of its last start unless you pass --force.
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


# vLLM brings its own PyTorch (2.13 for vLLM 0.31). sdcguard is pip-installed here, so vLLM finds its
# plugin entry point the way it would on a user's machine.
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("vllm==0.31.0")
    # VLLM_USE_FLASHINFER_SAMPLER=0: vLLM 0.31's warm-up JIT-compiles FlashInfer's sampler with nvcc, which a
    # pip install doesn't have (round 3: every compiled run died there); 0 = vLLM's own PyTorch/Triton sampler
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{V}/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip, copy=True)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip, copy=True)
    .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml", copy=True)
    .add_local_file(LOCAL / "README.md", f"{R}/README.md", copy=True)
    .add_local_file(LOCAL / "LICENSE", f"{R}/LICENSE", copy=True)
    .run_commands(f"pip install --no-deps {R}")
)

app = modal.App("sdcguard-vllm", image=vllm_image)
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


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=75 * 60)
def vllm(force: bool = False) -> str:
    msg = _locked("vllm", force)
    if msg:
        return msg
    import shutil
    out = f"{RUNS}/vllm"
    if os.path.isdir(out):
        shutil.rmtree(out)
    t0 = time.time()
    try:
        _tee([sys.executable, "-c", "import importlib.metadata as m; print([e for e in m.entry_points("
              "group='vllm.general_plugins')])"], f"{out}/entry_points.txt")
        # a small model (fast) and a 7B one (per-layer costs like real deployments); each: without sdcguard,
        # with it on every layer, and on 1/8 of the layers, under vLLM's default torch.compile + CUDA graphs
        for tag, model in (("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"), ("qwen7b", "Qwen/Qwen2.5-7B-Instruct")):
            _tee([sys.executable, f"{R}/experiments/vllm_check.py", "--out", f"{out}/{tag}", "--model", model,
                  "--shares", "0.125"], f"{out}/{tag}.txt", cwd="/root")
            vol.commit()
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.time() - t0) / 60:.1f} minutes on {GPU}\n")
        vol.commit()
    return f"vllm finished in {(time.time() - t0) / 60:.1f} minutes"


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=15 * 60)
def demo(force: bool = False) -> str:
    """The 1-minute vLLM demo (experiments/demo_vllm.py), for a screen recording of your terminal."""
    out = f"{RUNS}/demo"
    os.makedirs(out, exist_ok=True)
    code = _tee([sys.executable, f"{R}/experiments/demo_vllm.py"], f"{out}/demo.txt", cwd="/root")
    vol.commit()
    return f"demo finished (exit code {code})"


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
    "vllm": "about 25 minutes on 1 H100: roughly $1.65 (hard limit 75 minutes, about $4.95)",
    "demo": "about 5 minutes on 1 H100: roughly $0.40 (hard limit 15 minutes, about $1.05)",
    "fetch": "download only: a few cents",
}


@app.local_entrypoint()
def main(phase: str = "vllm", force: bool = False):
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    try:
        if phase == "vllm":
            print(vllm.remote(force), flush=True)
        elif phase == "demo":
            print(demo.remote(force), flush=True)
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
