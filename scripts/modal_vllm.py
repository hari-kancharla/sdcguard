"""The sdcguard vLLM plugin on real vLLM 0.31, on a Modal H100.

    modal run scripts/modal_vllm.py --phase vllm   # about 25 min, about $2 (a 75-minute limit on the whole run)
    modal run scripts/modal_vllm.py --phase demo   # 1-minute live demo (experiments/demo_vllm.py), about $0.40
    modal run scripts/modal_vllm.py --phase quant  # INT4 models, about 26 min, about $2.10 (75-minute limit)
    modal run scripts/modal_vllm.py --phase outputs  # the gate's output check, about 12 min, about $1
    modal run scripts/modal_vllm.py --phase machete_diag  # the Machete patch against vLLM's own completions and
                                                          # compiled kernels, about 20 min, about $1.60
    modal run scripts/modal_vllm.py --phase machete_fix   # machete_diag, then quant if the fix meets the rule in
                                                          # docs/METHOD.md: about 45 to 50 min, about $3.70 (20
                                                          # and $1.60 if it doesn't)
    modal run scripts/modal_vllm.py --phase serve     # real `vllm serve`, plain and guarded: vLLM 0.32 on 1 and 2
                                                      # H100s, 0.31 on 1, at once: about 15 min each, about $5
    modal run scripts/modal_vllm.py --phase quant_l4  # INT4 on Marlin on an L4: about 20 min, about $0.50

vllm: Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct without sdcguard, with it on every layer and on 1/8 of
      the layers: outputs, tokens/s (256 requests at a time, and 16), alerts on clean traffic, and a flipped weight
      bit caught and located.
quant: the same for INT4 models: Qwen2.5-1.5B-Instruct-AWQ (zero points) and -GPTQ-Int4 (no zero points), each on
      Machete (vLLM's choice on an H100) and on Marlin (VLLM_DISABLED_KERNELS=MacheteLinearKernel, the kernel every
      other NVIDIA GPU gets), then Qwen2.5-7B-Instruct-AWQ on Machete with 1/8 of the layers too (the cost). Every
      checked layer must be quantized and run by the expected kernel, and only the LM head may go unchecked.
outputs: `sdcguard gate model-ref` on Qwen2.5-1.5B-Instruct (bf16), then `sdcguard gate model-check` against it: the
      same stack with other batching (the noise floor), another attention backend (Triton), the AWQ model, and the
      bf16 model with rms_norm_eps set to 1.0 (a config change no matmul check can see, run with sdcguard's matmul
      checks on to show they stay quiet). docs/METHOD.md has what each is expected to show.

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
from typing import Optional

import modal

GPU = "H100!"                 # exactly an H100 (without "!" Modal may upgrade to an H200)
LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
V = "/vol"
ROUND = "r3"
DATA, RUNS = f"{V}/data", f"{V}/runs_{ROUND}"
LOCK_HOURS = 1.5           # longer than any phase's time limit
RATE_PER_HOUR = 3.95 + 8 * 0.0472 + 64 * 0.008         # H100 + 8 cores + 64 GiB, Modal prices on Oct 20, 2026
RATE_TP2 = 2 * 3.95 + 16 * 0.0472 + 128 * 0.008        # 2 H100s + 16 cores + 128 GiB
RATE_L4 = 0.80 + 8 * 0.0472 + 32 * 0.008               # L4 + 8 cores + 32 GiB


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


def _cost(minutes: float, rate: float = RATE_PER_HOUR, what: str = "H100, 8 cores, 64 GiB") -> dict:
    return {"minutes": round(minutes, 1), "estimated_cost_usd": round(minutes / 60 * rate, 2),
            "cost_note": f"function time x ${rate:.2f}/hour ({what} at Oct 20 prices); "
                         "image pull and startup are billed too and not in this number"}


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


# vLLM brings its own PyTorch (2.13 for vLLM 0.31). sdcguard is pip-installed here, so vLLM finds its
# plugin entry point the way it would on a user's machine.
def _vllm_image(version: str):
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("build-essential")
        .uv_pip_install(f"vllm=={version}")
        # VLLM_USE_FLASHINFER_SAMPLER=0: vLLM 0.31's warm-up JIT-compiles FlashInfer's sampler with nvcc, which a
        # pip install doesn't have (every compiled run on October 18 died there); 0 = vLLM's own sampler
        .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{V}/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
        .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip, copy=True)
        .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip, copy=True)
        .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml", copy=True)
        .add_local_file(LOCAL / "README.md", f"{R}/README.md", copy=True)
        .add_local_file(LOCAL / "LICENSE", f"{R}/LICENSE", copy=True)
        .run_commands(f"pip install --no-deps {R}")
    )


vllm_image = _vllm_image("0.31.0")
vllm032_image = _vllm_image("0.32.0")       # (October 18; the server checks run on it as well as on 0.31)

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


QUANT_RUNS = (
    ("awq1.5b_machete", "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "MacheteLinearKernel", {}, ()),
    ("awq1.5b_marlin", "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "MarlinLinearKernel",
     {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}, ()),
    ("gptq1.5b_machete", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "MacheteLinearKernel", {}, ()),
    ("gptq1.5b_marlin", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "MarlinLinearKernel",
     {"VLLM_DISABLED_KERNELS": "MacheteLinearKernel"}, ()),
    ("awq7b_machete", "Qwen/Qwen2.5-7B-Instruct-AWQ", "MacheteLinearKernel", {}, ("--shares", "0.125")),
)


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=75 * 60)
def quant(force: bool = False, source: dict | None = None) -> str:
    t0 = time.monotonic()
    deadline = t0 + 75 * 60 - 120
    msg = _locked("quant", force)
    if msg:
        raise RuntimeError(msg)
    out = f"{RUNS}/quant"
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    codes = {}
    try:
        for index, (tag, model, kernel, env, extra) in enumerate(QUANT_RUNS):
            budget = max(0, deadline - time.monotonic()) / (len(QUANT_RUNS) - index)
            codes[tag] = _tee([sys.executable, f"{R}/experiments/vllm_check.py", "--out", f"{out}/{tag}",
                               "--model", model, "--expect-quant", kernel, *extra], f"{out}/{tag}.txt", env=env,
                              cwd="/root", timeout=budget)
            vol.commit()
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.monotonic() - t0) / 60:.1f} minutes on {GPU}\n")
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_codes": codes, "passed": len(codes) == len(QUANT_RUNS) and
                       all(c == 0 for c in codes.values()), "source": source,
                       **_cost((time.monotonic() - t0) / 60)}, f)
        vol.commit()
    if len(codes) != len(QUANT_RUNS) or any(code != 0 for code in codes.values()):
        raise RuntimeError(f"quantized-model validation failed: {codes}; diagnostics were saved to {out}")
    return f"quant finished in {(time.monotonic() - t0) / 60:.1f} minutes"


OUTPUT_CHECKS = (
    ("same_batch4", [], {}, ["--vllm-arg", "max_num_seqs=4"]),
    ("triton_attention", [], {}, ["--vllm-arg", "attention_backend=TRITON_ATTN"]),
    ("awq", ["--model", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"], {}, []),
    ("rms_norm_eps_1", [], {"SDCGUARD": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "SDCGUARD_POLL_EVERY": "1000000000"},
     ["--vllm-arg", 'hf_overrides={"rms_norm_eps": 1.0}']),
)


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=40 * 60)
def outputs(force: bool = False, source: dict | None = None) -> str:
    t0 = time.monotonic()
    deadline = t0 + 40 * 60 - 120
    msg = _locked("outputs", force)
    if msg:
        raise RuntimeError(msg)
    out = f"{RUNS}/outputs"
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    codes = {}
    g = [sys.executable, "-m", "sdcguard", "gate"]
    try:
        codes["ref"] = _tee([*g, "model-ref", "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--out", f"{out}/ref.json"],
                            f"{out}/ref.txt", cwd="/root", timeout=min(600, max(0, deadline - time.monotonic())))
        vol.commit()
        if codes["ref"] == 0:
            for index, (tag, model, env, extra) in enumerate(OUTPUT_CHECKS):
                noise = [] if tag == "same_batch4" else ["--noise", f"{out}/same_batch4.json"]
                budget = max(0, deadline - time.monotonic()) / (len(OUTPUT_CHECKS) - index)
                codes[tag] = _tee([*g, "model-check", f"{out}/ref.json", "--out", f"{out}/{tag}.json", *model, *noise,
                                   *extra], f"{out}/{tag}.txt", env=env, cwd="/root", timeout=min(600, budget))
                vol.commit()
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.monotonic() - t0) / 60:.1f} minutes on {GPU}\n")
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_codes": codes, "source": source, **_cost((time.monotonic() - t0) / 60)}, f)
        vol.commit()
    return f"outputs finished in {(time.monotonic() - t0) / 60:.1f} minutes: {codes}"


# round 3 (docs/METHOD.md): the model whose completions still changed in the INT4 runs after round 2, as many
# requests at once as vllm_check sends; its runs are the script's default list
MACHETE_DIAG_ARGS = ("--model", "Qwen/Qwen2.5-7B-Instruct-AWQ", "--prompts", "256")


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=60 * 60)
def machete_diag(force: bool = False, source: dict | None = None) -> dict:
    """sdcguard's Machete patch against vLLM without it (experiments/machete_diag.py): one model, every completion
    and every compiled graph's code compared. Returns {"message", "fix"}: the variant that met the rule in
    docs/METHOD.md, or None."""
    t0 = time.monotonic()
    deadline = t0 + 60 * 60 - 120
    msg = _locked("machete_diag", force)
    if msg:
        raise RuntimeError(msg)
    out = f"{RUNS}/machete_diag"
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    code, fix = None, None
    try:
        code = _tee([sys.executable, f"{R}/experiments/machete_diag.py", "--out", out, *MACHETE_DIAG_ARGS],
                    f"{out}/diag.txt",
                    cwd="/root", timeout=max(0, deadline - time.monotonic()))
        try:
            fix = json.load(open(f"{out}/DIAG.json")).get("fix") if code == 0 else None
        except (OSError, ValueError):
            fix = None
    finally:
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "fix": fix, "source": source,
                       **_cost((time.monotonic() - t0) / 60)}, f)
        vol.commit()
    if code != 0:
        raise RuntimeError(f"Machete diagnostic failed (exit code {code}); diagnostics were saved to {out}")
    return {"message": f"machete_diag finished in {(time.monotonic() - t0) / 60:.1f} minutes; the fix that met the "
                       f"rule: {fix or 'none'}", "fix": fix}


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


# the server checks (experiments/vllm_serve_check.py, docs/METHOD.md): real `vllm serve`, plain and guarded, on
# vLLM 0.32 with 1 and 2 GPUs and on 0.31 with 1; the three run at once
SERVE_RUNS = {"serve032": ("0.32.0", 1), "serve032_tp2": ("0.32.0", 2), "serve031": ("0.31.0", 1)}
# (2 GPUs as "H100:2": Modal's documented count syntax; with it Modal may hand out H200s, which STATUS.json records)


def _serve(tag: str, tp: int, force: bool, source: Optional[dict], rate: float, what: str) -> str:
    t0 = time.monotonic()
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_serve_check.py", "--out", out, "--tp", str(tp)],
                    f"{out}.txt", cwd="/root", timeout=40 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=45 * 60)
def serve032(force: bool = False, source: Optional[dict] = None) -> str:
    return _serve("serve032", 1, force, source, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")


@app.function(image=vllm032_image, volumes=VOLS, gpu="H100:2", cpu=16, memory=131072, timeout=45 * 60)
def serve032_tp2(force: bool = False, source: Optional[dict] = None) -> str:
    return _serve("serve032_tp2", 2, force, source, RATE_TP2, "2 H100s, 16 cores, 128 GiB")


@app.function(image=vllm_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=45 * 60)
def serve031(force: bool = False, source: Optional[dict] = None) -> str:
    return _serve("serve031", 1, force, source, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")


# INT4 on another GPU than the H100: on an L4 (sm89) vLLM has no Machete and runs Marlin by itself, with other
# shapes of its kernels than the H100's
QUANT_L4_RUNS = (
    ("awq1.5b_marlin_l4", "Qwen/Qwen2.5-1.5B-Instruct-AWQ", "MarlinLinearKernel", {}, ()),
    ("gptq1.5b_marlin_l4", "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4", "MarlinLinearKernel", {}, ()),
)


@app.function(image=vllm_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=60 * 60)
def quant_l4(force: bool = False, source: Optional[dict] = None) -> str:
    t0 = time.monotonic()
    deadline = t0 + 60 * 60 - 120
    msg = _locked("quant_l4", force)
    if msg:
        return msg
    out = f"{RUNS}/quant_l4"
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    codes = {}
    try:
        for index, (tag, model, kernel, env, extra) in enumerate(QUANT_L4_RUNS):
            budget = max(0, deadline - time.monotonic()) / (len(QUANT_L4_RUNS) - index)
            codes[tag] = _tee([sys.executable, f"{R}/experiments/vllm_check.py", "--out", f"{out}/{tag}",
                               "--model", model, "--expect-quant", kernel, *extra], f"{out}/{tag}.txt", env=env,
                              cwd="/root", timeout=budget)
            vol.commit()
    finally:
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_codes": codes, "passed": len(codes) == len(QUANT_L4_RUNS) and
                       all(c == 0 for c in codes.values()), "source": source,
                       **_cost((time.monotonic() - t0) / 60, RATE_L4, "L4, 8 cores, 32 GiB")}, f)
        vol.commit()
    return f"quant_l4: {codes} in {(time.monotonic() - t0) / 60:.1f} minutes"


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
    "quant": "about 26 minutes on 1 H100: roughly $2.10 (24.6 and 26.1 minutes on October 22; 75-minute function "
             "limit, about $6 at requested resources for a run that uses the whole limit, not a spending cap)",
    "outputs": "about 12 minutes on 1 H100: roughly $1 (40-minute function limit, about $3.20 at requested "
               "resources for a run that uses the whole limit, not a spending cap)",
    "demo": "about 5 minutes on 1 H100: roughly $0.40 (15-minute function limit; about $1.20 at requested "
            "resources for a run that uses the whole limit, not a spending cap)",
    "machete_diag": "about 20 minutes on 1 H100: roughly $1.60 (60-minute function limit; about $4.80 at requested "
                    "resources for a run that uses the whole limit, not a spending cap)",
    "machete_fix": "the Machete diagnostic (about 20 minutes, roughly $1.60), then, only if the fix meets the rule "
                   "written in docs/METHOD.md, the INT4 runs (about 26 minutes, roughly $2.10): about 45 to 50 "
                   "minutes and $3.70 in all on 1 H100 (limits 60 and 75 minutes; at most about $10.90 at requested "
                   "resources, not a spending cap)",
    "serve": "real `vllm serve`, plain and guarded: vLLM 0.32 on 1 H100 and on 2, vLLM 0.31 on 1, the three at once; "
             "about 15 minutes each, roughly $1.20 + $2.40 + $1.20 = about $5 (45-minute limits; at most about $18 at "
             "requested resources, not a spending cap; building the 0.32 image the first time adds a few minutes)",
    "quant_l4": "INT4 on Marlin on 1 L4, two 1.5B models: about 20 minutes, roughly $0.50 (60-minute limit, at most "
                "about $1.50)",
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
        elif phase == "quant":
            message = quant.remote(force, src)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED"):
                failures.append(str(message))
        elif phase == "outputs":
            message = outputs.remote(force, src)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED"):
                failures.append(str(message))
        elif phase in ("machete_diag", "machete_fix"):
            result = machete_diag.remote(force, src)
            message = result.get("message") if isinstance(result, dict) else str(result)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED"):
                failures.append(str(message))
            elif phase == "machete_fix":
                if isinstance(result, dict) and result.get("fix") == "fused":
                    print("The shipped fix met the rule written in docs/METHOD.md: starting the INT4 runs "
                          f"(--phase quant: {ESTIMATES['quant']}).", flush=True)
                    message = quant.remote(force, src)
                    print(message, flush=True)
                    if str(message).startswith("NOT STARTED"):
                        failures.append(str(message))
                else:
                    fix = result.get("fix") if isinstance(result, dict) else None
                    failures.append(f"the shipped Machete fix didn't meet the rule in docs/METHOD.md (the variant "
                                    f"that did: {fix or 'none'}; runs/gpu_r3/machete_diag/DIAG.json has why), so the "
                                    "INT4 runs weren't started")
        elif phase == "serve":
            calls = {tag: fn.spawn(force, src) for tag, fn in (("serve032", serve032), ("serve032_tp2", serve032_tp2),
                                                               ("serve031", serve031))}
            for tag, call in calls.items():
                try:
                    message = call.get()
                except Exception as e:  # noqa: BLE001
                    message = f"{tag} did not complete: {type(e).__name__}: {e}"
                print(message, flush=True)
                if str(message).startswith("NOT STARTED") or "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "quant_l4":
            message = quant_l4.remote(force, src)
            print(message, flush=True)
            if str(message).startswith("NOT STARTED") or any(f"'{tag}': 0" not in str(message)
                                                             for tag, *_ in QUANT_L4_RUNS):
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
