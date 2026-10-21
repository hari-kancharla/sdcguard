"""The sdcguard vLLM plugin on real vLLM 0.31, on a Modal H100.

    modal run scripts/modal_vllm.py --phase vllm   # about 25 min, about $2 (a 75-minute limit on the whole run)
    modal run scripts/modal_vllm.py --phase demo   # 1-minute live demo (experiments/demo_vllm.py), about $0.40

vllm: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct without sdcguard, with it on every layer and on 1/8 of
      the layers: outputs, tokens/s, alerts on clean traffic, and a flipped weight bit caught and located.

Costs use Modal's prices (H100 $3.95 an hour; CPU $0.0472 a core-hour and memory $0.008 a GiB-hour, so
these 8-core, 64 GiB machines cost about $4.84 an hour). If vLLM's default compiled mode fails, the experiment
also tries two fallback modes, which takes longer (all inside the one function call and its 75-minute limit), and it
then reports a failure: a fallback doesn't validate the default mode.
Costs assume requested resources. Higher CPU/memory usage, startup/idle time and execution restarts can add
charges; the roughly $6 of requested resources for a run that uses the whole limit is not a spending cap.

Results are saved to the Modal volume sdcguard-runs and downloaded to runs/gpu_r3/. Every function has a
hard time limit, and a phase won't start again within an hour of its last start unless you pass --force.
A new run never deletes an earlier one: the earlier folder is renamed vllm_before_<time>. STATUS.json records the
code fingerprint the run used, its exit codes, its measured minutes and the cost those minutes imply at the rates
above (Modal's usage page has the billed amount).
"""
from __future__ import annotations

import io
import json
import os
import signal
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
RATE_PER_HOUR = 3.95 + 8 * 0.0472 + 64 * 0.008         # H100 + 8 cores + 64 GiB, Modal prices on Oct 20, 2026


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


def _cost(minutes: float) -> dict:
    return {"minutes": round(minutes, 1), "estimated_cost_usd": round(minutes / 60 * RATE_PER_HOUR, 2),
            "cost_note": f"function time x ${RATE_PER_HOUR:.2f}/hour (H100, 8 cores, 64 GiB at Oct 20 prices); "
                         "image pull and startup are billed too and not in this number"}


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


def _tee(cmd, log_path, env=None, cwd=R, timeout: float = 15 * 60):
    """Run a command, stream its output, and write each line to the log as it arrives (Modal commits the
    volume periodically). A local deadline ends the process group before the function's limit, preserving
    the log and allowing the next model and final status to run. Output goes directly to the log file so a
    child retaining a pipe cannot stall timeout cleanup. An escaped process group is not covered by SIGKILL."""
    print("$ " + " ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    code = 127
    with open(log_path, "w") as f:
        if timeout <= 0:
            f.write("not started: no time remains in this function\n[exit code 125]\n")
            return 125
        try:
            deadline = time.monotonic() + timeout
            p = subprocess.Popen(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, **(env or {}),
                                      "SDCGUARD_RUN_DEADLINE_MONOTONIC": str(deadline - 10)})
            with open(log_path, errors="replace") as reader:
                while p.poll() is None:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end="", flush=True)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        try:
                            os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            p.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                        code = 124
                        break
                    time.sleep(min(0.2, remaining))
                else:
                    code = p.returncode
                print(reader.read(), end="", flush=True)
            if code == 124:
                f.write(f"\n(stopped after {timeout / 60:.1f} minutes; process group termination requested)\n")
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
def vllm(force: bool = False, source: dict | None = None) -> str:
    t0 = time.monotonic()
    deadline = t0 + 75 * 60 - 120     # final status and volume commit must precede the hard function limit
    msg = _locked("vllm", force)
    if msg:
        raise RuntimeError(msg)
    import shutil
    out = f"{RUNS}/vllm"
    if os.path.isdir(out):              # keep the earlier run's evidence; this run starts from an empty folder
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    codes = {}
    try:
        codes["entry_points"] = _tee([sys.executable, "-c", "import importlib.metadata as m; "
              "entries=list(m.entry_points(group='vllm.general_plugins')); print(entries); "
              "assert any(e.name == 'sdcguard' for e in entries), 'sdcguard plugin entry point missing'"],
              f"{out}/entry_points.txt", timeout=min(120, max(0, deadline - time.monotonic())))
        # a small model (fast) and a 7B one (per-layer costs like real deployments); each: without sdcguard,
        # with it on every layer, and on 1/8 of the layers, under vLLM's default torch.compile + CUDA graphs
        if codes["entry_points"] == 0:
            models = (("qwen1.5b", "Qwen/Qwen2.5-1.5B-Instruct"), ("qwen7b", "Qwen/Qwen2.5-7B-Instruct"))
            for index, (tag, model) in enumerate(models):
                # A hung first model must leave time to collect the second model's diagnostics.
                budget = max(0, deadline - time.monotonic()) / (len(models) - index)
                codes[tag] = _tee([sys.executable, f"{R}/experiments/vllm_check.py", "--out", f"{out}/{tag}",
                                   "--model", model, "--shares", "0.125"], f"{out}/{tag}.txt", cwd="/root",
                                  timeout=budget)
                vol.commit()
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.monotonic() - t0) / 60:.1f} minutes on {GPU}\n")
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_codes": codes, "passed": len(codes) == 3 and all(c == 0 for c in codes.values()),
                       "source": source, **_cost((time.monotonic() - t0) / 60)}, f)
        vol.commit()
    if len(codes) != 3 or any(code != 0 for code in codes.values()):
        raise RuntimeError(f"vLLM validation failed: {codes}; diagnostics were saved to {out}")
    return f"vllm finished in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=15 * 60)
def demo(force: bool = False, source: dict | None = None) -> str:
    """The 1-minute vLLM demo (experiments/demo_vllm.py), for a screen recording of your terminal."""
    t0 = time.monotonic()
    deadline = t0 + 15 * 60 - 120
    msg = _locked("demo", force)
    if msg:
        raise RuntimeError(msg)
    out = f"{RUNS}/demo"
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/demo_vllm.py"], f"{out}/demo.txt", cwd="/root",
                    timeout=max(0, deadline - time.monotonic()))
    finally:
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       **_cost((time.monotonic() - t0) / 60)}, f)
        vol.commit()
    if code != 0:
        raise RuntimeError(f"vLLM demo failed (exit code {code}); diagnostics were saved to {out}")
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
    "vllm": "about 25 minutes on 1 H100: roughly $2 (75-minute function limit; about $6 at requested resources "
            "for a run that uses the whole limit, not a spending cap)",
    "demo": "about 5 minutes on 1 H100: roughly $0.40 (15-minute function limit; about $1.20 at requested "
            "resources for a run that uses the whole limit, not a spending cap)",
    "fetch": "download only: a few cents",
}


@app.local_entrypoint()
def main(phase: str = "vllm", force: bool = False):
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    failures = []
    try:
        if phase == "vllm":
            message = vllm.remote(force, src)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED"):
                failures.append(str(message))
        elif phase == "demo":
            message = demo.remote(force, src)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED"):
                failures.append(str(message))
    except Exception as e:  # noqa: BLE001 (download diagnostics before reporting the remote failure)
        failures.append(f"{phase} did not complete: {type(e).__name__}: {e}")
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
            failures.append(f"result download failed: {type(e).__name__}: {e}")
    if failures:
        raise SystemExit("VALIDATION NOT COMPLETED: " + "; ".join(failures))
    print("\nDone.", flush=True)
