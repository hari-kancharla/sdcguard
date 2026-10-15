"""The October 14-15 GPU runs on Modal H100s: harm study, false-alarm soak, slowdown, the 4-GPU demo.

    modal setup                                       # once
    modal run scripts/modal_gpu.py --phase check      # self-test, Triton vs plain comparison, false-alarm diagnosis
    modal run scripts/modal_gpu.py --phase bench      # per-shape timing of the Triton check on GPT-2 small
    modal run scripts/modal_gpu.py --phase speed      # slowdown sweep on GPT-2 small
    modal run scripts/modal_gpu.py --phase harm-clean # the harm study's 4 clean runs
    modal run scripts/modal_gpu.py --phase harm-faulty  # its 24 runs with a simulated bad GPU, then the report
    modal run scripts/modal_gpu.py --phase soak       # 5 GPU-hours of clean bf16 training, every matmul checked
    modal run scripts/modal_gpu.py --phase soak16     # 1 GPU-hour of clean fp16 training with loss scaling
    modal run scripts/modal_gpu.py --phase demo       # 4 GPUs, one goes bad
    modal run scripts/modal_gpu.py --phase main       # speed, harm-clean, harm-faulty, soak and soak16
    modal run scripts/modal_gpu.py --phase fetch      # download the results again

Results are saved to a Modal volume and downloaded to runs/gpu_r2/. Each phase saves as it goes and skips
finished work, so a stopped phase picks up where it left off. Add --detach to keep a single phase running if
the terminal closes, then use --phase fetch. A phase won't start while an earlier start of it is less than
3.5 hours old (--force overrides), so the same work is never paid for twice.
"""
from __future__ import annotations

import io
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

GPU = "H100!"                 # exactly an H100 (without "!" Modal may upgrade to an H200)
H100_PER_HOUR = 3.95          # modal.com/pricing, checked Oct 14 2026 ($0.001097/s)
LOCAL = Path(__file__).resolve().parent.parent      # this repo
R = "/root/sdcguard"          # the project folder inside the cloud machine
V = "/vol"                    # persistent cloud folder for data and results
ROUND = "r2"                  # round 1 results stay untouched in /vol/runs and runs/gpu
DATA, RUNS = f"{V}/data", f"{V}/runs_{ROUND}"

HARM_PRESET = "ts-gpt2"       # GPT-2 small (124M) on TinyStories
BIG_PRESET = "ts-900m"        # ~0.9B parameters: how the slowdown changes with model size
# The harm study checks every 10th step; its report also shows detection at every 20 and 50
# (same runs, keeping only those checks). The demo checks every 10 and says so.
GUARD_EVERY = 10
LOCK_HOURS = 3.5
CHECK_PRESET, CHECK_STEPS = "ts-gpu", 300
# The GPU goes bad at step 205: halfway between two checks (every 10) and not on a canary step
# (every 50), so the catch can't come from the fault happening to start on a checked step.
# The first demo run used 200 and is kept in runs/gpu/demo_first_run_step200.
DEMO_PRESET, DEMO_STEPS, DEMO_BAD_FROM = "ts-gpu", 1000, 205
DEMO_TRIAGE_STEPS = 30        # the every-50 demo: steps checked one by one after a restart
SOAK_TARGET_GPU_HOURS = 5.0   # kill criterion #1
SOAK16_TARGET_GPU_HOURS = 1.0  # fp16 with loss scaling (GradScaler), reported separately
SOAK_WORKERS = 5
SOAK_MINUTES = 62


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache") or part.startswith("._") or part.endswith(".egg-info")
               for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")      # Triton compiles a small C launcher the first time it runs
    .uv_pip_install("torch==2.14.1", "numpy", "matplotlib", "tiktoken", "pytest")
    # TIKTOKEN_CACHE_DIR: keep GPT-2's tokenizer files on the volume (prep_data downloads them once),
    # so no training container depends on downloading them itself
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1", "TIKTOKEN_CACHE_DIR": f"{DATA}/tiktoken"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml")
)
app = modal.App("sdcguard", image=image)
vol = modal.Volume.from_name("sdcguard-runs", create_if_missing=True)
VOLS = {V: vol}


def _tee(cmd, log_path=None, check=True, env=None):
    """Run a command, stream its output live, and save a copy to log_path."""
    print("$ " + " ".join(cmd), flush=True)
    lines = []
    try:
        p = subprocess.Popen(cmd, cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env={**os.environ, **(env or {})})
    except OSError as e:
        if check:
            raise
        print(f"(could not run {cmd[0]}: {e})", flush=True)
        return 127, ""
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    p.wait()
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.writelines(lines)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed with code {p.returncode}: {' '.join(cmd)}")
    return p.returncode, "".join(lines)


def _imports():
    for p in (f"{R}/src", f"{R}/experiments"):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------ cloud functions
@app.function(volumes=VOLS, cpu=4, memory=8192, timeout=45 * 60)
def prep_data():
    """Download and tokenize TinyStories once (CPU only, no GPU billed)."""
    _imports()
    import harm_study
    harm_study.load_dataset("tinystories", DATA, 300)
    vol.commit()


@app.function(volumes=VOLS, gpu=GPU, cpu=4, memory=32768, timeout=15 * 60)
def bench_gpu():
    """Per-shape timing of the Triton check on GPT-2 small's matmuls (experiments/bench_kernels.py)."""
    out = f"{RUNS}/bench"
    _tee([sys.executable, "experiments/bench_kernels.py"], f"{out}/bench.txt", check=False,
         env={"BENCH_OUT": f"{out}/bench.json"})
    vol.commit()
    # and that the kernels being timed give the plain check's answers on this GPU
    _tee([sys.executable, "tests/_fast_compare.py", "cuda"], f"{out}/fast_compare.txt", check=False)
    vol.commit()


@app.function(volumes=VOLS, gpu=GPU, cpu=4, memory=32768, timeout=40 * 60)
def check_gpu():
    try:
        _check_gpu()
    finally:
        vol.commit()


def _check_gpu():
    import shutil
    out = f"{RUNS}/check"
    if os.path.isdir(out):
        shutil.rmtree(out)                 # every check starts clean (no leftovers from an older run)
    py = sys.executable
    # Every step runs even if an earlier one fails, so one check run tells us everything.
    _tee(["nvidia-smi"], f"{out}/nvidia-smi.txt", check=False)
    # (the Triton-interpreter test runs on the CPU for minutes; tests/_fast_compare.py cuda below runs
    # the same comparison on the GPU itself)
    _tee([py, "-m", "pytest", "-q", "tests", "--ignore", "tests/test_fast.py"], f"{out}/pytest.txt", check=False)
    _tee([py, "-m", "sdcguard.selftest"], f"{out}/selftest.txt", check=False)
    # Triton check vs plain check on the real GPU: every shape case, corrupted outputs, a guarded
    # step, inputs changed or freed right after the matmul
    _tee([py, "tests/_fast_compare.py", "cuda"], f"{out}/fast_compare.txt", check=False)
    vol.commit()
    # false alarms on GPT-2 small, every matmul checked, 150 steps each: with LR warmup; without warmup
    # and the old noise floor (should reproduce check run #2's alarms); without warmup and the fix
    _tee([py, "experiments/calibrate.py", "--preset", HARM_PRESET, "--diag", "--diag-steps", "150",
          "--out", f"{out}/runs", "--data-dir", DATA], f"{out}/diag.txt", check=False)
    vol.commit()
    # a small end-to-end experiment on the GPU: 2 clean runs, faulty runs, the report
    _tee([py, "experiments/harm_study.py", "--preset", CHECK_PRESET, "--steps", str(CHECK_STEPS), "--rates", "0.05",
          "--fault-scale", "warp,tile", "--fault-seeds", "1", "--clean-seeds", "2",
          "--out", f"{out}/runs", "--data-dir", DATA], f"{out}/harm_mini.txt", check=False)
    vol.commit()
    # slowdown on GPT-2 small: per-step timing plus aligned windows (whole multiples of k steps)
    _tee([py, "experiments/calibrate.py", "--preset", HARM_PRESET, "--overhead",
          "--sweep", "1:1,10:1,20:1,50:1,100:1,1:0.1", "--timing-steps", "30", "--rounds", "3",
          "--out", f"{out}/runs", "--data-dir", DATA], f"{out}/overhead.txt", check=False)
    vol.commit()
    # the same with the Triton GPU check switched off (the plain PyTorch check): before vs after
    _tee([py, "experiments/calibrate.py", "--preset", HARM_PRESET, "--overhead",
          "--sweep", "1:1,10:1,20:1,50:1", "--timing-steps", "30", "--rounds", "3",
          "--out", f"{out}/runs_plain_check", "--data-dir", DATA], f"{out}/overhead_plain_check.txt",
         check=False, env={"SDCGUARD_FAST": "0"})
    vol.commit()
    # the same with the vocabulary padded to a multiple of 64, as nanoGPT does and as Llama, Mistral
    # and Qwen vocabularies already are (GPT-2's 50,257 is an odd shape for the GPU's matmul kernels)
    _tee([py, "experiments/calibrate.py", "--preset", HARM_PRESET, "--overhead", "--vocab-pad", "64",
          "--sweep", "1:1,10:1,20:1,50:1", "--timing-steps", "30", "--rounds", "3",
          "--out", f"{out}/runs_vocab_padded", "--data-dir", DATA], f"{out}/overhead_vocab_padded.txt",
         check=False)
    vol.commit()
    # the same on a ~0.9B-parameter model: does the slowdown shrink as the model grows?
    _tee([py, "experiments/calibrate.py", "--preset", BIG_PRESET, "--overhead",
          "--sweep", "1:1,10:1,20:1,50:1", "--timing-steps", "12", "--rounds", "2",
          "--out", f"{out}/runs", "--data-dir", DATA], f"{out}/overhead_big.txt", check=False)
    vol.commit()


# a full ts-gpt2 run takes about 7 minutes; 20 caps the cost if something ever crawls. One retry
# (a failed or preempted container); finished runs are skipped, so a retry only redoes the failed one.
@app.function(volumes=VOLS, gpu=GPU, cpu=2, memory=16384, timeout=20 * 60, retries=1)
def harm_run(spec: dict) -> dict:
    _imports()
    import harm_study
    path = harm_study.run_from_spec(spec, DATA, RUNS, skip_existing=True)
    vol.commit()
    r = harm_study.load_run(path)
    return {"name": r.name, "minutes": round(r.seconds / 60, 1), "crash_step": r.nan_step,
            "first_alert": r.first_alert_step, "ms_per_step": r.ms_per_step, "gpu": r.env.get("gpu")}


@app.function(volumes=VOLS, gpu=GPU, cpu=4, memory=32768, timeout=30 * 60)
def overhead_run():
    _tee([sys.executable, "experiments/calibrate.py", "--preset", HARM_PRESET, "--overhead",
          "--sweep", "1:1,10:1,20:1,30:1,40:1,50:1,100:1,1:0.1", "--timing-steps", "40", "--rounds", "3",
          "--out", RUNS, "--data-dir", DATA], f"{RUNS}/calibration/{HARM_PRESET}/overhead_log.txt")
    vol.commit()


@app.function(volumes=VOLS, gpu=GPU, cpu=2, memory=16384, timeout=80 * 60)
def soak_worker(worker: str, minutes: float, seed: int, dtype_name: str = "bf16"):
    _imports()
    import calibrate
    import harm_study
    import torch
    cfg = dict(harm_study.PRESETS[HARM_PRESET], name=HARM_PRESET)
    device = harm_study.pick_device("auto")
    dtype = torch.float16 if dtype_name == "fp16" else harm_study.resolve_dtype(device, torch.bfloat16)
    ds = harm_study.load_dataset(cfg["dataset"], DATA, cfg.get("ts_train_mb", 200))
    model, opt = calibrate.make(cfg, ds.vocab, device, seed)
    gen = torch.Generator().manual_seed(seed)
    path = f"{RUNS}/calibration/{HARM_PRESET}/{calibrate._soak_dir(dtype)}/{worker}_{int(time.time())}.json"
    print(f"soak worker {worker}: {minutes:.0f} min of clean {dtype_name} training, checking every matmul",
          flush=True)
    rec = calibrate.soak(model, opt, ds.train, cfg, device, dtype, gen, minutes, path, on_save=vol.commit)
    vol.commit()
    return {"worker": worker, "gpu_hours": rec["gpu_hours"], "checks": rec["checks"],
            "false_alarms": rec["false_alarms"], "worst_z": rec["worst_clean_z"]}


@app.function(volumes=VOLS, cpu=2, memory=8192, timeout=30 * 60)
def analyze_all():
    _imports()
    import harm_study
    vol.reload()
    out = []
    for preset in (HARM_PRESET,):
        if os.path.isdir(f"{RUNS}/{preset}/raw"):
            try:
                harm_study.analyze(RUNS, preset)
                out.append(preset)
            except (Exception, SystemExit) as e:  # noqa: BLE001  (still write CALIBRATION.md below)
                import traceback
                traceback.print_exc()
                with open(f"{RUNS}/{preset}/ANALYSIS_FAILED.txt", "w") as f:
                    f.write(f"harm_study.analyze failed: {e!r}\n{traceback.format_exc()}")
    cal = f"{RUNS}/calibration/{HARM_PRESET}"
    if os.path.isdir(cal):
        _tee([sys.executable, "experiments/calibrate.py", "--preset", HARM_PRESET, "--summarize", "--out", RUNS],
             f"{cal}/summarize_log.txt", check=False)
    vol.commit()
    return out


def harm_specs():
    """The harm study: 4 clean runs + 24 faulty runs on GPT-2 small (TinyStories)."""
    # triage_steps: after a crash, restart from the newest snapshot (every 50 steps) on the same bad
    # GPU and check every step for up to 50 steps (clean runs never crash, so it costs them nothing)
    base = {"preset": HARM_PRESET, "guard_every": GUARD_EVERY, "triage_steps": 50, "snapshot_every": 50}
    specs = [{**base, "kind": "clean", "seed": s} for s in (0, 1, 2)]
    specs.append({**base, "kind": "clean", "seed": 0, "repeat": True})        # GPU run-to-run noise
    for scale in ("warp", "tile"):
        for rate in (0.001, 0.01):
            for f in (1, 2, 3):
                specs.append({**base, "kind": "faulty", "seed": 0, "rate": rate, "target": "qk",
                              "mix": "nvidia", "scale": scale, "fault_seed": f})
    for scale in ("warp", "tile"):
        for f in (1, 2, 3):
            specs.append({**base, "kind": "faulty", "seed": 0, "rate": 0.01, "target": "qk",
                          "mix": "zero", "scale": scale, "fault_seed": f})
    for f in (1, 2, 3):
        specs.append({**base, "kind": "faulty", "seed": 0, "rate": 0.01, "target": "qk",
                      "mix": "mantissa", "scale": "warp", "fault_seed": f})
    for f in (1, 2, 3):        # a bad GPU hits every layer, not just attention Q/K
        specs.append({**base, "kind": "faulty", "seed": 0, "rate": 0.001, "target": "qkvprojmlp",
                      "mix": "nvidia", "scale": "warp", "fault_seed": f})
    return specs


def _lock_path(phase):
    return f"{V}/locks/{ROUND}_{phase}.json"


@app.function(volumes=VOLS, cpu=1, memory=2048, timeout=3 * 3600)
def conductor(phase: str, force: bool = False) -> str:
    """Runs a whole phase in the cloud, so it keeps going even if your terminal closes."""
    t0 = time.time()
    vol.reload()
    lock = _lock_path(phase)
    me = modal.current_function_call_id()
    if os.path.exists(lock) and not force:
        try:
            with open(lock) as f:
                held = json.load(f)
            age_h = (time.time() - held["started"]) / 3600
            mine = held.get("call") == me and me is not None      # Modal restarted this same call
        except (OSError, ValueError, KeyError):
            age_h, mine = 0.0, False
        if age_h < LOCK_HOURS and not mine:
            return (f"NOT STARTED: phase '{phase}' was started {age_h * 60:.0f} minutes ago and may still be "
                    f"running in the cloud. Starting it twice would pay twice. Wait for it, then run "
                    f"--phase fetch to download the results.")
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as f:
        json.dump({"started": time.time(), "call": me}, f)
    vol.commit()
    try:
        _conduct(phase)
    finally:
        try:
            os.remove(lock)
            vol.commit()
        except OSError:
            pass
    return f"phase {phase} finished in {(time.time() - t0) / 60:.1f} min"


def _soak_topup(dtype_name: str, target_h: float, max_workers: int):
    """Start soak workers for whatever part of target_h GPU-hours isn't saved yet."""
    _imports()
    import harm_study
    sub = "soak_fp16" if dtype_name == "fp16" else "soak"
    now = harm_study.code_version(("calibrate.py",))
    vol.reload()
    done_h, n_files, stale = 0.0, 0, 0
    soak_dir = f"{RUNS}/calibration/{HARM_PRESET}/{sub}"
    if os.path.isdir(soak_dir):
        for fn in sorted(os.listdir(soak_dir)):
            if fn.endswith(".json"):
                with open(os.path.join(soak_dir, fn)) as f:
                    rec = json.load(f)
                if rec.get("code") != now:          # made by older code: set aside, not counted
                    os.makedirs(os.path.join(soak_dir, "stale"), exist_ok=True)
                    os.replace(os.path.join(soak_dir, fn), os.path.join(soak_dir, "stale", fn))
                    stale += 1
                    continue
                done_h += rec.get("gpu_hours", 0.0)
                n_files += 1
    vol.commit()
    need = max(0.0, target_h - done_h)
    workers = min(max_workers, math.ceil(need / (SOAK_MINUTES / 60 * 0.97))) if need > 0.05 else 0
    print(f"{dtype_name} false-alarm soak: {done_h:.2f} GPU-hours already saved by this code ({now}) in "
          f"{n_files} files; {stale} older files set aside; starting {workers} more workers", flush=True)
    seed0 = 100 if dtype_name == "bf16" else 500
    return [soak_worker.spawn(f"{dtype_name}_{int(time.time()) % 100000}_{i}", SOAK_MINUTES, seed0 + n_files + i,
                              dtype_name) for i in range(workers)]


def _wait(calls, what):
    for fc in calls:
        try:
            print(f"{what} done: {fc.get()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{what} failed: {e}", flush=True)


def _conduct(phase):
    prep_data.remote()
    if phase == "check":
        check_gpu.remote()
        return
    if phase == "bench":
        bench_gpu.remote()
        return
    if phase == "demo":
        demo.remote()
        return
    if phase in ("speed", "main"):
        # slowdown alone on its GPU (the checked step is CPU-bound, so no neighbours)
        try:
            overhead_run.remote()
        except Exception as e:  # noqa: BLE001
            print(f"overhead run failed: {e}", flush=True)
    soaks = []
    if phase in ("soak", "main"):
        soaks += _soak_topup("bf16", SOAK_TARGET_GPU_HOURS, SOAK_WORKERS)
    if phase in ("soak16", "main"):
        soaks += _soak_topup("fp16", SOAK16_TARGET_GPU_HOURS, 1)
    kinds = {"harm-clean": ("clean",), "harm-faulty": ("faulty",), "main": ("clean", "faulty")}.get(phase, ())
    if kinds:
        # up to 10 GPUs at once, shared with the soak; finished runs are skipped
        for r in harm_run.map([sp for sp in harm_specs() if sp["kind"] in kinds], return_exceptions=True):
            print(f"harm run done: {r}", flush=True)
    _wait(soaks, "soak")
    if phase != "speed":
        analyze_all.remote()


@app.function(volumes=VOLS, gpu=f"{GPU}:4", cpu=8, memory=65536, timeout=40 * 60)
def demo():
    import shutil
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 1) checks every 10 steps: caught before the crash. 2) checks every 50 steps (cheaper): the crash
    #    can come first, and the restart with every step checked names the GPU (after_restart).
    for name, every, triage in (("demo", GUARD_EVERY, 0), ("demo_every50", 50, DEMO_TRIAGE_STEPS)):
        local = f"/tmp/{name}"
        _tee([sys.executable, "experiments/ddp_demo.py", "--device", dev, "--gpus", "4", "--bad-gpu", "2",
              "--bad-from", str(DEMO_BAD_FROM), "--rate", "0.05", "--preset", DEMO_PRESET, "--steps", str(DEMO_STEPS),
              "--batch", "32", "--ckpt-every", "50", "--canary-every", "50", "--replica-every", "50",
              "--guard-every", str(every), "--triage-steps", str(triage),
              "--data-dir", DATA, "--out", local], f"{local}/demo_log.txt", check=False)
        dest = f"{RUNS}/{name}"
        if os.path.exists(dest):
            shutil.rmtree(dest)
        for root, _dirs, files in os.walk(local):
            for fn in files:
                if fn.endswith(".pt"):
                    continue                       # checkpoints stay behind (big, not needed)
                src = os.path.join(root, fn)
                dst = os.path.join(dest, os.path.relpath(src, local))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        vol.commit()


@app.function(volumes=VOLS, cpu=1, memory=4096, timeout=15 * 60)
def bundle() -> bytes:
    vol.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if os.path.isdir(RUNS):
            tar.add(RUNS, arcname=f"gpu_{ROUND}", filter=lambda ti: None if ti.name.endswith(".pt") else ti)
    return buf.getvalue()


# ------------------------------------------------------------------------------ local side
def _fetch():
    print("\nDownloading results...", flush=True)
    data = bundle.remote()
    dest = LOCAL / "runs"
    dest.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"Results are in {dest / f'gpu_{ROUND}'} ({len(data) / 1e6:.1f} MB)", flush=True)


# H100 $3.95/hour plus CPU and memory; the harm-study times are from round 1 (Oct 14 2026)
ESTIMATES = {
    "bench": "about 7 minutes on 1 H100: roughly $0.55 (hard limit 15 minutes, about $1.10)",
    "check": "about 15 minutes on 1 H100: roughly $1.20 (hard limit 40 minutes, about $3)",
    "speed": "about 10 minutes on 1 H100: roughly $0.80 (hard limit 30 minutes, about $2.20)",
    "harm-clean": "4 runs of about 8 minutes on 4 H100s at once: roughly $2.30",
    "harm-faulty": "24 runs on up to 10 H100s at once (most crash early): roughly $9",
    "soak": f"{SOAK_TARGET_GPU_HOURS:.0f} GPU-hours on {SOAK_WORKERS} H100s for about 65 minutes: roughly $22",
    "soak16": "about 65 minutes on 1 H100: roughly $4.50",
    "demo": "about 15 minutes on 4 H100s (two demos): roughly $4 (hard limit 40 minutes, about $11)",
    "main": "speed + harm-clean + harm-faulty + soak + soak16: about 80 minutes, roughly $39",
    "fetch": "download only: a few cents",
}


@app.local_entrypoint()
def main(phase: str = "check", force: bool = False):
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    try:
        if phase != "fetch":
            print(conductor.remote(phase, force), flush=True)
        else:
            analyze_all.remote()
    finally:
        try:
            _fetch()
        except Exception as e:  # noqa: BLE001
            print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
    print("\nDone.", flush=True)
