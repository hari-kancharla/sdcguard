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
    modal run scripts/modal_vllm.py --phase serve_tp2  # the 2-GPU server check alone: about 10 min, about $1.60
    modal run scripts/modal_vllm.py --phase ready_attrib   # `sdcguard ready` and request naming: about 20 min, about $1.60
    modal run scripts/modal_vllm.py --phase live_catch     # cuBLAS 6862950 live in vllm serve on an H200: about $5-6.50
    modal run scripts/modal_vllm.py --phase decisions      # `sdcguard decide` on four server changes: about $4.40-5.70
    modal run scripts/modal_vllm.py --phase isolation      # vLLM #62124 and controls on 0.31 and 0.32, 2 L4s: under $1
    modal run scripts/modal_vllm.py --phase sensitivity    # 80 weight bit flips in a served model, 1 H100: about $2
    modal run scripts/modal_vllm.py --phase weight_scan    # the weight scan in vllm serve, 66 changes, 1 H100: about $2.40-3.65
    modal run scripts/modal_vllm.py --phase fp8_residuals  # vLLM's FP8 GEMM against exact arithmetic, 1 H100: under $1
    modal run scripts/modal_vllm.py --phase norms          # RMSNorm checks in vllm serve, 1 H100: about $3 to $4
    modal run scripts/modal_vllm.py --phase qualify        # `sdcguard qualify` with its defaults on 1 L4: about $1
    modal run scripts/modal_vllm.py --phase moe            # MoE checks in vllm serve, 1 H100: about $2 to $3
    modal run scripts/modal_vllm.py --phase graphs         # every check inside a captured CUDA graph, 1 L4: about $0.25
    modal run scripts/modal_vllm.py --phase fp8mm_cal      # FP8 GEMM block sums for an FP8 matmul check, H100 + L4: about $1.50
    modal run scripts/modal_vllm.py --phase attn           # attention checks in vllm serve, H100 + L4: $4.50 to $7
    modal run scripts/modal_vllm.py --phase screen_ops     # exact operator tests for screening GPUs, L4 + H100: under $1.30
    modal run scripts/modal_vllm.py --phase moe_bench      # where the MoE checks' cost goes, 1 H100, no server: under $0.65
    modal run scripts/modal_vllm.py --phase norm_diag      # why norm checks changed answers, 7 servers, 1 H100: $2-2.85
    modal run scripts/modal_vllm.py --phase spot           # spot checks in vllm serve, H100 + L4: $3-4
    modal run --detach scripts/modal_vllm.py --phase smoke1  # Run 1's parts scaled down to minutes, before it: $5-6.50
    modal run --detach scripts/modal_vllm.py --phase run1  # weight_scan, moe_bench, norm_diag, moe, spot, screen_ops
                                                           # and fp8_moe at once, each on its own GPU: about
                                                           # $12.30-17.50, about 2 hours (fp8_moe the longest)

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
hard time limit, and a phase won't start again within an hour and a half of its last start unless you pass
--force.
A new run never deletes an earlier one: the earlier folder (and the log beside it) is renamed <phase>_before_<time>. STATUS.json records the
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
LOCK_HOURS = 3.25          # longer than any phase's time limit (fp8_moe: 190 minutes)
RATE_PER_HOUR = 3.95 + 8 * 0.0472 + 64 * 0.008         # H100 + 8 cores + 64 GiB, Modal prices on Oct 20, 2026
RATE_TP2 = 2 * 3.95 + 16 * 0.0472 + 128 * 0.008        # 2 H100s + 16 cores + 128 GiB
RATE_L4 = 0.80 + 8 * 0.0472 + 32 * 0.008               # L4 + 8 cores + 32 GiB


def _source() -> dict:
    """Fingerprint of the code this run used (src, experiments, tests, scripts), to match it to a commit, and of the
    criteria it is judged by (docs/METHOD.md, written before the run), so a result carries the criteria it met."""
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "experiments", "tests", "scripts"):
        for p in sorted((LOCAL / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(LOCAL)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    method = LOCAL / "docs" / "METHOD.md"
    crit = hashlib.sha256(method.read_bytes()).hexdigest() if method.is_file() else None
    return {"sha256": h.hexdigest(), "files": n, "criteria": {"docs/METHOD.md": crit}}


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
        # VLLM_ALLREDUCE_USE_FLASHINFER=0: vLLM 0.32 reduces across GPUs with FlashInfer's fused all-reduce, which
        # JIT-compiles with nvcc too, and both 2-GPU servers of October 23 died there before serving, plain vLLM
        # as well as guarded; 0 = vLLM's own all-reduce (one GPU never reduces)
        .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{V}/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0",
              "VLLM_ALLREDUCE_USE_FLASHINFER": "0"})
        .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip, copy=True)
        .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip, copy=True)
        .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml", copy=True)
        .add_local_file(LOCAL / "README.md", f"{R}/README.md", copy=True)
        .add_local_file(LOCAL / "LICENSE", f"{R}/LICENSE", copy=True)
        .run_commands(f"pip install --no-deps {R}")
    )


vllm_image = _vllm_image("0.31.0")
vllm032_image = _vllm_image("0.32.0")       # (October 18; the server checks run on it as well as on 0.31)
# the tests too, for the GPU tests that run on vLLM 0.32's own PyTorch and Triton (the graphs phase)
tests_image = vllm032_image.uv_pip_install("pytest==9.1.1").add_local_dir(LOCAL / "tests", f"{R}/tests",
                                                                          ignore=_skip, copy=True)

app = modal.App("sdcguard-vllm", image=vllm_image)
vol = modal.Volume.from_name("sdcguard-runs", create_if_missing=True)
VOLS = {V: vol}


def _set_aside(out: str) -> None:
    """A rerun (--force) keeps the earlier run: its folder and its log, both renamed with the same time."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    if os.path.isdir(out):
        os.rename(out, f"{out}_before_{stamp}")
    if os.path.exists(f"{out}.txt"):
        os.rename(f"{out}.txt", f"{out}_before_{stamp}.txt")


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
    _set_aside(out)
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
    ("rms_norm_eps_1", [], {"SDCGUARD": "1", "SDCGUARD_ATTN": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                            "SDCGUARD_POLL_EVERY": "1000000000"},
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
    _set_aside(out)
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
    _set_aside(out)
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
    _set_aside(out)
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


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=60 * 60)
def ready_attrib(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_ready_attrib_check.py: `sdcguard ready`'s verdicts and the requests alerts name, on a real
    `vllm serve` (vLLM 0.32, one H100)."""
    t0 = time.monotonic()
    tag = "ready_attrib"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_ready_attrib_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=55 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=90 * 60)
def decisions(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_change_decisions.py: `sdcguard decide` on four changes to a real `vllm serve` (vLLM 0.32,
    Qwen2.5-7B-Instruct, one H100)."""
    t0 = time.monotonic()
    tag = "decisions"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_change_decisions.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=85 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=45 * 60)
def sensitivity(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_sensitivity.py: which weight bit flips the plugin catches while serving, and whether the ones
    it misses change the answers (vLLM 0.32, Qwen2.5-1.5B-Instruct, every layer checked, one H100)."""
    t0 = time.monotonic()
    tag = "sensitivity"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_sensitivity.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=40 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=25 * 60)
def weight_repair(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_weight_repair_check.py: the plugin's weight repair in real `vllm serve` (vLLM 0.32,
    Qwen2.5-1.5B-Instruct, one H100): 8 weight bit flips left in place, each alerted, written back and verified,
    identical answers after."""
    t0 = time.monotonic()
    tag = "weight_repair"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_weight_repair_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=22 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=58 * 60)
def weight_scan(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_weight_scan_check.py: the plugin's weight scan in real `vllm serve` (vLLM 0.32,
    Qwen2.5-1.5B-Instruct, one layer in eight checked, one H100): coverage, no clean alert, flips named, identical
    answers, the cost."""
    t0 = time.monotonic()
    tag = "weight_scan"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_weight_scan_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=55 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=20 * 60)
def fp8_residuals(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/fp8_residuals.py: vLLM's FP8 GEMM on an H100 against exact arithmetic on the same operands."""
    t0 = time.monotonic()
    tag = "fp8_residuals"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/fp8_residuals.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=15 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=75 * 60)
def norms(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_norm_check.py: the plugin's RMSNorm checks in real `vllm serve` (vLLM 0.32, Qwen2.5-1.5B and
    Qwen3-1.7B, one H100): coverage, identical answers, clean scores against the bound, norm weight flips, exact
    request naming in eager mode, and the cost."""
    t0 = time.monotonic()
    tag = "norms"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_norm_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=70 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=40 * 60)
def moe(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_moe_check.py: the plugin's MoE checks in real `vllm serve` (vLLM 0.32, granite-3.1-1b-a400m,
    one H100): coverage, identical answers, clean scores, wrong kernels caught in eager mode, and the cost."""
    t0 = time.monotonic()
    tag = "moe"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_moe_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=37 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=75 * 60)
def qualify(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_qualify_check.py: one `sdcguard qualify` with its defaults and the plugin on (vLLM 0.32,
    Qwen2.5-1.5B-Instruct, one L4): the baseline twice, then the same stack, another attention kernel and a wrong
    rms_norm_eps; then `sdcguard ready --qualified` on the same stack and on the other attention kernel."""
    t0 = time.monotonic()
    tag = "qualify"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_qualify_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=72 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_L4, "L4, 8 cores, 32 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=190 * 60)
def fp8_moe(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_fp8_moe_check.py: one `sdcguard qualify` on vLLM issue #62124 (FP8 per-tensor MoE with CUDA
    graphs, granite-3.1-1b-a400m on one L4) and two remedies, with the FP8 quantization check on; then the rejected
    candidate's repro folder run on its own."""
    t0 = time.monotonic()
    tag = "fp8_moe"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_fp8_moe_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=187 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_L4, "L4, 8 cores, 32 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


def _fp8mm_cal(tag: str, models: str, rate: float, what: str, source: Optional[dict], force: bool) -> str:
    t0 = time.monotonic()
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/fp8mm_calibration.py", "--out", out, "--models", models],
                    f"{out}.txt", cwd="/root", timeout=40 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source, "models": models,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=45 * 60)
def fp8mm_cal_h100(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/fp8mm_calibration.py on an H100: Qwen2.5-1.5B (every linear layer) and 7B (5 blocks)."""
    return _fp8mm_cal("fp8mm_cal_h100", "1.5b,7b", RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=45 * 60)
def fp8mm_cal_l4(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/fp8mm_calibration.py on an L4 (Ada, sm89): Qwen2.5-1.5B (every linear layer)."""
    return _fp8mm_cal("fp8mm_cal_l4", "1.5b", RATE_L4, "L4, 8 cores, 32 GiB", source, force)


def _attn(tag: str, roles: str, rate: float, what: str, minutes: int, source: Optional[dict], force: bool) -> str:
    t0 = time.monotonic()
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_attn_check.py", "--out", out, "--roles", roles],
                    f"{out}.txt", cwd="/root", timeout=minutes * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source, "roles": roles,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=95 * 60)
def attn_h100(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_attn_check.py on an H100 (FlashAttention 3, and vLLM's Triton attention): the plugin's
    attention checks in real `vllm serve` (Qwen2.5-1.5B): coverage, identical answers, clean scores against the bound,
    checks inside CUDA graphs, faults caught in eager mode, and the cost."""
    return _attn("attn_h100", "plain,guarded,noattn,eager,tplain,tguarded", RATE_PER_HOUR, "H100, 8 cores, 64 GiB",
                 90, source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=50 * 60)
def attn_l4(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_attn_check.py on an L4 (FlashAttention 2): plain and guarded only."""
    return _attn("attn_l4", "plain,guarded", RATE_L4, "L4, 8 cores, 32 GiB", 45, source, force)


def _screen_ops(tag: str, rate: float, what: str, source: Optional[dict], force: bool) -> str:
    t0 = time.monotonic()
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/screen_ops_check.py", "--out", out, "--minutes", "5"],
                    f"{out}.txt", cwd="/root", timeout=20 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=15 * 60)
def moe_bench(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/moe_cost_bench.py on an H100, with vLLM 0.32's PyTorch and Triton (no server): what the MoE and
    routing checks' kernels cost in CUDA graphs, by kind, active or not, and what a kernel that returns at once costs
    by grid."""
    t0 = time.monotonic()
    tag = "moe_bench"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/moe_cost_bench.py", "--out", out], f"{out}.txt", cwd="/root",
                    timeout=12 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=60 * 60)
def norm_diag(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_norm_diag.py: why the compiled server's answers changed with norm checks on (vLLM 0.32,
    Qwen2.5-1.5B-Instruct, one H100): 7 servers, each compiled from an empty cache, answers and compiled norm
    kernels compared pair by pair."""
    t0 = time.monotonic()
    tag = "norm_diag"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_norm_diag.py", "--out", out], f"{out}.txt", cwd="/root",
                    timeout=57 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


RUN1 = ("weight_scan", "moe_bench", "norm_diag", "moe", "spot_h100", "spot_l4", "screen_ops_h100", "screen_ops_l4",
        "fp8_moe")


# The coordinators wait for parts that can sit in Modal's GPU queue before their own limits even start. If a
# coordinator ran out of time first, the local client would finish, download and stop the app, and that stops every
# part still running: so they get hours (a CPU container that only waits costs about 6 cents an hour, and returns as
# soon as its parts do).
COORDINATOR_S = {"run1": 8 * 3600, "smoke1": 4 * 3600}


@app.function(image=vllm032_image, volumes=VOLS, cpu=1, memory=2048, timeout=COORDINATOR_S["run1"])
def run1(force: bool = False, source: Optional[dict] = None) -> dict:
    """Run 1 from one call (so `modal run --detach` keeps all of it alive: the parts are started from here, inside
    Modal, not by the local client): the weight scan, the MoE cost bench, the norm diagnosis, the MoE checks with
    M6 (real rows given no experts), the spot checks (H100 and L4), the screen's operator tests with their varied inputs and realistic-data matmuls (H100 and L4), and the
    FP8 MoE run (L4, the longest: up to its 190-minute limit), each on its own GPU at the same time, each with its own
    time limit. {tag: its message}."""
    fns = {"weight_scan": weight_scan, "moe_bench": moe_bench, "norm_diag": norm_diag, "moe": moe, "spot_h100": spot_h100,
           "spot_l4": spot_l4, "screen_ops_h100": screen_ops_h100, "screen_ops_l4": screen_ops_l4,
           "fp8_moe": fp8_moe}
    calls = {tag: fns[tag].spawn(force, source) for tag in RUN1}
    out = {}
    for tag, call in calls.items():
        try:
            out[tag] = str(call.get())
        except Exception as e:  # noqa: BLE001
            out[tag] = f"{tag} did not complete: {type(e).__name__}: {e}"
    return out


# Run 1's smoke test (experiments/smoke_run1.py): each part scaled down to minutes on its own GPU, through the same
# code, so what breaks only on a real GPU shows before the full run. Each part's own time limit (minutes), inside its
# function's (SMOKE_FN, below: each function's limit just over its parts').
SMOKE_LIMITS = {"weight_scan": 27, "moe_bench": 8, "norm_diag": 20, "moe": 27, "spot_h100": 28, "spot_l4": 15,
                "screen_h100": 9, "screen_l4": 9}
SMOKE_H100 = ("weight_scan", "moe_bench", "norm_diag", "moe", "spot_h100", "screen_h100")
SMOKE_L4 = ("spot_l4", "screen_l4")


def _smoke(part: str, rate: float, what: str, source: Optional[dict], force: bool) -> str:
    t0 = time.monotonic()
    tag = f"smoke_{part}"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/smoke_run1.py", "--part", part, "--out", out], f"{out}.txt",
                    cwd="/root", timeout=SMOKE_LIMITS[part] * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


# One function per GPU and length class, each limit just over its parts' own (SMOKE_LIMITS), so a part that hangs
# costs its own limit, not the longest part's.
@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=30 * 60)
def smoke_h100(part: str, force: bool = False, source: Optional[dict] = None) -> str:
    """A long H100 part of Run 1's smoke test (weight_scan, moe, spot_h100)."""
    return _smoke(part, RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=23 * 60)
def smoke_h100_mid(part: str, force: bool = False, source: Optional[dict] = None) -> str:
    """norm_diag's part of Run 1's smoke test."""
    return _smoke(part, RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=12 * 60)
def smoke_h100_short(part: str, force: bool = False, source: Optional[dict] = None) -> str:
    """A short H100 part of Run 1's smoke test (moe_bench, screen_h100)."""
    return _smoke(part, RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=17 * 60)
def smoke_l4(part: str, force: bool = False, source: Optional[dict] = None) -> str:
    """spot_l4's part of Run 1's smoke test."""
    return _smoke(part, RATE_L4, "L4, 8 cores, 32 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=12 * 60)
def smoke_l4_short(part: str, force: bool = False, source: Optional[dict] = None) -> str:
    """screen_l4's part of Run 1's smoke test."""
    return _smoke(part, RATE_L4, "L4, 8 cores, 32 GiB", source, force)


SMOKE_FN = {"weight_scan": "smoke_h100", "moe": "smoke_h100", "spot_h100": "smoke_h100", "norm_diag": "smoke_h100_mid",
            "moe_bench": "smoke_h100_short", "screen_h100": "smoke_h100_short", "spot_l4": "smoke_l4",
            "screen_l4": "smoke_l4_short"}


@app.function(image=vllm032_image, volumes=VOLS, cpu=1, memory=2048, timeout=COORDINATOR_S["smoke1"])
def smoke1(force: bool = False, source: Optional[dict] = None, parts: Optional[list] = None) -> dict:
    """Run 1's smoke test from one call (as run1: the parts are started from here, inside Modal), each part on its
    own GPU at the same time (parts: only those, to run a fixed part again). {part: its message}."""
    want = set(parts or (*SMOKE_H100, *SMOKE_L4))
    fns = {"smoke_h100": smoke_h100, "smoke_h100_mid": smoke_h100_mid, "smoke_h100_short": smoke_h100_short,
           "smoke_l4": smoke_l4, "smoke_l4_short": smoke_l4_short}
    calls = {p: fns[SMOKE_FN[p]].spawn(p, force, source) for p in (*SMOKE_H100, *SMOKE_L4) if p in want}
    out = {}
    for part, call in calls.items():
        try:
            out[part] = str(call.get())
        except Exception as e:  # noqa: BLE001
            out[part] = f"smoke_{part} did not complete: {type(e).__name__}: {e}"
    return out


def _spot(tag: str, gpu: str, rate: float, what: str, source: Optional[dict], force: bool, limit_s: float) -> str:
    """limit_s: the experiment's own time limit, inside the function's (vllm_spot_check.RUN_S is shorter still)."""
    t0 = time.monotonic()
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_spot_check.py", "--out", out, "--gpu", gpu], f"{out}.txt",
                    cwd="/root", timeout=limit_s)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, rate, what)}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=55 * 60)
def spot_h100(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_spot_check.py on an H100: the spot checks in real `vllm serve` (Qwen2.5-1.5B, every linear
    layer): answers, clean traffic, cost, wrong layers and sampled wrong values caught, and a fault compiled into
    the graph caught."""
    return _spot("spot_h100", "h100", RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force, 52 * 60)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=30 * 60)
def spot_l4(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_spot_check.py --gpu l4: the spot server's clean traffic on an L4."""
    return _spot("spot_l4", "l4", RATE_L4, "L4, 8 cores, 32 GiB", source, force, 27 * 60)


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=25 * 60)
def screen_ops_h100(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/screen_ops_check.py on an H100: sdcguard.screen_ops clean for 5 minutes, then each test with one
    value made wrong in software."""
    return _screen_ops("screen_ops_h100", RATE_PER_HOUR, "H100, 8 cores, 64 GiB", source, force)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=25 * 60)
def screen_ops_l4(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/screen_ops_check.py on an L4."""
    return _screen_ops("screen_ops_l4", RATE_L4, "L4, 8 cores, 32 GiB", source, force)


GRAPH_TESTS = ("tests/test_gpu_graphs.py", "tests/test_serve_no_sync.py",
               "tests/test_serve.py::test_gpu_fast_serving_check_agrees_and_is_used",
               "tests/test_serve.py::test_gpu_compiled_cuda_graphs_catch_a_flip")


@app.function(image=tests_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=30 * 60)
def graphs(force: bool = False, source: Optional[dict] = None) -> str:
    """tests/test_gpu_graphs.py and the serving checks' other GPU tests on one L4, with vLLM 0.32's PyTorch and
    Triton: every check (linear, FP8 quantization, MoE, routing, norms; PyTorch and Triton versions) inside a captured
    CUDA graph (attention too). SDCGUARD_GPU_TESTS=1 makes a missing GPU a failure, not a skip."""
    t0 = time.monotonic()
    tag = "graphs"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code, skipped, other = None, [], []
    try:
        code = _tee([sys.executable, "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", *GRAPH_TESTS],
                    f"{out}/pytest.txt", env={**os.environ, "SDCGUARD_GPU_TESTS": "1"}, cwd=R, timeout=25 * 60)
        skipped = _required_skips(f"{out}/pytest.txt")
        other = [s for s in _required_skips(f"{out}/pytest.txt", any_file=True) if s not in skipped]
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0 and not skipped, "skipped_required": skipped,
                       "skipped_other": other, "source": source, "tests": list(GRAPH_TESTS),
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_L4, "L4, 8 cores, 32 GiB")}, f)
        vol.commit()
    if skipped:      # a skipped capture test shows nothing: the phase fails (criterion G5)
        return (f"{tag}: FAILED, {len(skipped)} required GPU tests skipped ({'; '.join(skipped)[:300]}) in "
                f"{(time.monotonic() - t0) / 60:.1f} minutes")
    # skips outside the required files (G5) are allowed, and said: the compiled-model test of tests/test_serve.py
    # skips where PyTorch has no CUDA graph replay hook (2.13 hasn't), so it can't prove its replays there
    note = f" (skipped, not required: {'; '.join(other)[:300]})" if other else ""
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes{note}"


REQUIRED_GPU_TESTS = ("tests/test_gpu_graphs.py", "tests/test_serve_no_sync.py")


def _required_skips(log_path: str, any_file: bool = False) -> list:
    """The skips pytest reported (-rs) in the test files that must run in full on the GPU (any_file: all of them)."""
    try:
        lines = open(log_path, errors="replace").read().splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.lstrip().startswith("SKIPPED") and
            (any_file or any(t in ln for t in REQUIRED_GPU_TESTS))]


@app.function(image=vllm032_image, volumes=VOLS, gpu=GPU, cpu=8, memory=65536, timeout=75 * 60)
def fp8kv(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_fp8kv_check.py: one `sdcguard qualify` on vLLM issue #54035 (an FP8 KV cache's decode and
    prefill differ on Hopper's FlashAttention 3; Qwen3-4B on one H100) and the same cache on FlashInfer."""
    t0 = time.monotonic()
    tag = "fp8kv"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_fp8kv_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=72 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_PER_HOUR, "H100, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


def _isolation(version: str, force: bool, source: Optional[dict]) -> str:
    t0 = time.monotonic()
    tag = f"isolation{version.replace('.', '')}"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code = None
    try:
        code = _tee([sys.executable, f"{R}/experiments/vllm_isolation_check.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=40 * 60)
    finally:
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source, "vllm": version,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_L4, "L4, 8 cores, 32 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


@app.function(image=vllm_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=45 * 60)
def isolation031(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_isolation_check.py on vLLM 0.31.0 (the version vLLM issue #62124 was reported on), one L4."""
    return _isolation("0.31.0", force, source)


@app.function(image=vllm032_image, volumes=VOLS, gpu="L4", cpu=8, memory=32768, timeout=45 * 60)
def isolation032(force: bool = False, source: Optional[dict] = None) -> str:
    """The same on vLLM 0.32.0: whether #62124 is still there is reported as found."""
    return _isolation("0.32.0", force, source)


RATE_H200 = 4.54 + 8 * 0.0472 + 64 * 0.008             # H200 + 8 cores + 64 GiB, Modal prices on Oct 20, 2026


@app.function(image=vllm032_image, volumes=VOLS, gpu="H200", cpu=8, memory=65536, timeout=90 * 60)
def live_catch(force: bool = False, source: Optional[dict] = None) -> str:
    """experiments/vllm_live_catch.py on one H200 with the MPS daemon running: cuBLAS issue 6862950 inside the real
    server, its replay at the alert's real shape, and the pinned workaround."""
    import shutil
    t0 = time.monotonic()
    tag = "live_catch"
    msg = _locked(tag, force)
    if msg:
        return msg
    out = f"{RUNS}/{tag}"
    _set_aside(out)
    os.makedirs(out, exist_ok=True)
    code, mps = None, shutil.which("nvidia-cuda-mps-control")
    base = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_pipe", CUDA_MPS_LOG_DIRECTORY="/tmp/mps_log")
    started = False
    try:
        if not mps:
            raise RuntimeError("nvidia-cuda-mps-control isn't in this container")
        os.makedirs(base["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
        os.makedirs(base["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
        p = subprocess.run([mps, "-d"], env=base, capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            raise RuntimeError(f"the MPS daemon didn't start: {(p.stdout + p.stderr)[-500:]}")
        started = True
        os.environ.update(CUDA_MPS_PIPE_DIRECTORY=base["CUDA_MPS_PIPE_DIRECTORY"],
                          CUDA_MPS_LOG_DIRECTORY=base["CUDA_MPS_LOG_DIRECTORY"])
        code = _tee([sys.executable, f"{R}/experiments/vllm_live_catch.py", "--out", out], f"{out}.txt",
                    cwd="/root", timeout=85 * 60)
    except Exception as e:  # noqa: BLE001  (a run that couldn't start is a result too, never a pass)
        with open(f"{out}.txt", "a") as f:
            f.write(f"\nNOT RUN: {type(e).__name__}: {e}\n")
    finally:
        if started:
            subprocess.run([mps], input="quit\n", env=base, capture_output=True, text=True, timeout=60)
        import torch
        with open(f"{out}/STATUS.json", "w") as f:
            json.dump({"exit_code": code, "passed": code == 0, "source": source,
                       "verdict": {0: "CAUGHT", 3: "NOT EXERCISED"}.get(code, "see SUMMARY.json"),
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                       **_cost((time.monotonic() - t0) / 60, RATE_H200, "H200, 8 cores, 64 GiB")}, f)
        vol.commit()
    return f"{tag}: exit code {code} in {(time.monotonic() - t0) / 60:.1f} minutes"


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
    _set_aside(out)
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
    "serve_tp2": "the 2-GPU server check alone (vLLM 0.32 on 2 H100s): about 10 minutes, roughly $1.60 (45-minute "
                 "limit, at most about $7.30)",
    "quant_l4": "INT4 on Marlin on 1 L4, two 1.5B models: about 20 minutes, roughly $0.50 (60-minute limit, at most "
                "about $1.50)",
    "ready_attrib": "`sdcguard ready` and request naming on real `vllm serve` (vLLM 0.32, 1 H100, four server starts): "
                    "about 15 to 25 minutes, roughly $1.20 to $2 (60-minute limit, at most about $4.90 at requested "
                    "resources, not a spending cap)",
    "live_catch": "cuBLAS issue 6862950 inside real `vllm serve` on 1 H200 as an MPS client (100% and 40%, then "
                  "40% with the pinned workaround), with replays at the alert's real shape and `kernels test` on "
                  "both inventories: about 55 to 70 minutes, roughly $5 to $6.50 (90-minute limit, at most about "
                  "$8 at requested resources, not a spending cap)",
    "decisions": "`sdcguard decide` on four changes to a real `vllm serve` of Qwen2.5-7B-Instruct on 1 H100, then "
                 "vLLM's Trace Replay offline on three of them (6 server launches, 3 engine loads: about 55 to 70 "
                 "minutes, roughly $4.40 to $5.70; 90-minute limit, at most about $7.30 at requested resources, not a "
                 "spending cap)",
    "isolation": "`sdcguard isolation` on vLLM issue #62124 and two controls, on vLLM 0.31.0 and 0.32.0 at once (2 L4s, "
                 "3 server launches each: about 15 to 25 minutes, roughly $0.40 to $0.70 together; 45-minute limit, "
                 "at most about $1.80 at requested resources, not a spending cap)",
    "sensitivity": "80 weight bit flips in a served Qwen2.5-1.5B on 1 H100, each with the plugin's verdict and a quality "
                   "check: about 20 to 30 minutes, roughly $1.60 to $2.40 (45-minute limit, at most about $3.60 at "
                   "requested resources, not a spending cap)",
    "weight_repair": "the plugin's weight repair in real `vllm serve` on 1 H100: Qwen2.5-1.5B with the weight scan "
                     "and repair on, 8 weight bit flips left in place, each to be alerted, written back and verified, "
                     "and the 64 answers after identical to those before: about 10 to 15 minutes, roughly $0.80 to "
                     "$1.20 (25-minute limit, at most about $2 at requested resources, not a spending cap)",
    "weight_scan": "the plugin's weight scan in real `vllm serve` on 1 H100: Qwen2.5-1.5B with one layer in eight "
                   "checked, without and with the scan (identical answers, the cost), then 48 weight bit flips and 12 "
                   "changes outside the linear layers each named or not within 120 s, then granite-3.1-1b-a400m with "
                   "6 changes in MoE experts' weights: about 30 to 45 minutes, roughly $2.40 to $3.65 (58-minute "
                   "limit, at most about $4.70 at requested resources, not a spending cap)",
    "fp8_residuals": "vLLM's FP8 GEMM against exact arithmetic on 16 layers of a 1.5B model, 1 H100: about 5 to 10 "
                     "minutes, roughly $0.40 to $0.80 (20-minute limit, at most about $1.60 at requested resources, "
                     "not a spending cap)",
    "norms": "the plugin's RMSNorm checks in real `vllm serve` on 1 H100: Qwen2.5-1.5B plain, guarded, without norm "
             "checks and eager, then Qwen3-1.7B plain and guarded (6 server launches, 44 weight flips, 5 naming "
             "trials, the cost): about 35 to 50 minutes, roughly $2.80 to $4 (75-minute limit, at most about $6 at "
             "requested resources, not a spending cap)",
    "qualify": "one `sdcguard qualify` with its defaults on 1 L4 (Qwen2.5-1.5B: the baseline twice, the same stack, "
               "Triton attention, a wrong rms_norm_eps), then `sdcguard ready --qualified` on two more launches (7 "
               "server launches): about 35 to 50 minutes, roughly $0.85 to $1.20 (75-minute limit, at most about "
               "$1.80 at requested resources, not a spending cap)",
    "moe": "the plugin's MoE checks in real `vllm serve` on 1 H100: granite-3.1-1b-a400m plain, guarded, without MoE "
           "checks and eager (4 server launches, 6 wrong-kernel trials, 2 wrong routers, M6's real rows given no "
           "experts, the cost): about 12 to 20 minutes (the fourth run took 10.4), roughly $1 to $1.60 (40-minute "
           "limit, about $3.25 at requested resources if it runs to its limit; an estimate, not a spending cap)",
    "fp8_moe": "one `sdcguard qualify` on vLLM issue #62124 and two remedies on 1 L4 (granite-3.1-1b-a400m: bf16 "
               "twice, FP8 per-tensor, the same eager, FP8 per-channel; 5 server launches, then the repro folder "
               "once): about 110 to 130 minutes, roughly $2.60 to $3.10 (190-minute limit, at most about $4.50 at "
               "requested resources, not a spending cap)",
    "fp8kv": "one `sdcguard qualify` on vLLM issue #54035 on 1 H100 (Qwen3-4B: bf16 KV cache, FP8 KV cache, FP8 on "
             "FlashInfer; 3 server launches): about 35 to 50 minutes, roughly $2.80 to $4 (75-minute limit, at most "
             "about $6 at requested resources, not a spending cap)",
    "graphs": "every serving check inside a captured CUDA graph (pytest on 1 L4 with vLLM 0.32's PyTorch): about 5 to "
              "15 minutes with the image build, roughly $0.15 to $0.40 (30-minute limit, at most about $0.70 at "
              "requested resources, not a spending cap)",
    "fp8mm_cal": "FP8 GEMM block sums and an emulation of the tensor core's FP8 accumulation on real layers, on 1 H100 "
                 "(Qwen2.5-1.5B and 7B) and 1 L4 (1.5B) at once: about 10 to 20 minutes each, roughly $0.80 to $1.60 "
                 "plus $0.25 to $0.50 (45-minute limits, at most about $3.60 and $1.10 at requested resources, not a "
                 "spending cap)",
    "attn": "the plugin's attention checks in real `vllm serve` (Qwen2.5-1.5B) on 1 H100 (FlashAttention 3 and "
            "vLLM's Triton attention: 6 servers, with fault trials in eager mode) and 1 L4 (FlashAttention 2: plain "
            "and guarded) at once: about 45 to 75 minutes and 20 to 35 minutes, roughly $3.60 to $6 plus $0.50 to "
            "$0.85 (90- and 45-minute limits, at most about $7.30 and $1.10 at requested resources, not a spending "
            "cap)",
    "screen_ops": "sdcguard.screen_ops (exact FP64, FP32, TF32, BF16, FP16, FP8 and INT8 matmuls in varied shapes, "
                  "BF16 and FP16 matmuls of realistic data, CUDA-core FMAs, SM consistency and a memory test) clean "
                  "for 5 minutes and then with one value made wrong per test, on "
                  "1 H100 and 1 L4 at once: about 8 to 12 minutes each, roughly $0.65 to $1 plus $0.20 to $0.30 "
                  "(25-minute limits, at most about $2 and $0.60 at requested resources, not a spending cap)",
    "moe_bench": "where the MoE checks' cost goes, measured without vLLM on 1 H100 (kernels in CUDA graphs): about 4 "
                 "to 8 minutes, roughly $0.35 to $0.65 (15-minute limit, at most about $1.25 at requested resources, "
                 "not a spending cap)",
    "norm_diag": "why the compiled server's answers changed with norm checks on, 1 H100: Qwen2.5-1.5B on 7 servers "
                 "each compiled afresh (plain twice, plain with the plugin's cuBLAS setting, the plugin without and "
                 "with norm checks, and both in Inductor's deterministic mode): about 25 to 35 minutes, roughly $2 "
                 "to $2.85 (60-minute limit, so 7 slow compiles still fit: about $4.84 at requested resources if it "
                 "runs to its limit; an estimate, not a spending cap)",
    "spot": "the spot checks in real `vllm serve`, Qwen2.5-1.5B with every linear layer checked: on 1 H100 four "
            "servers (answers, clean traffic, cost, 17 eager trials, a fault compiled into the graph), and on 1 L4 at "
            "the same time the spot server's clean traffic: about 35 to 45 minutes, roughly $2.80 to $3.65 (H100) "
            "and $0.25 to $0.35 (L4) (55- and 30-minute limits, about $4.44 and $0.72 at requested resources if a "
            "part runs to its limit; an estimate, not a spending cap)",
    "run1": "Run 1, each part on its own GPU at the same time: weight_scan, moe_bench, norm_diag, moe, spot (H100 "
            "and L4), screen_ops (H100 and L4) and fp8_moe (L4) as above, about 45 minutes for all but fp8_moe and "
            "about 2 hours for it: roughly $12.30 to $17.50 together (about $26.80 if every part and the "
            "coordinator ran to their functions' limits at requested resources; an estimate, not a spending cap: "
            "Modal also bills image pulls and startup, CPU and memory used above the request, and a part rerun "
            "after its machine is preempted)",
    "smoke1": "Run 1's smoke test, each part scaled down to minutes on its own GPU at the same time (6 H100s, 2 L4s): "
              "about 25 minutes, roughly $5 to $6.50 together with startups (about $12 if every part and the "
              "coordinator ran to their functions' limits at requested resources; an estimate, not a spending cap)",
    "fetch": "download only: a few cents",
}


def _connection_broke(e: BaseException) -> bool:
    """Did the client lose its connection to Modal (this computer's network or sleep), rather than the run fail?
    modal.exception.ConnectionError ("Deadline exceeded") and ClientClosed don't derive from Python's own
    ConnectionError, so they are told by name; socket errors (OSError, e.g. "nodename nor servname provided") by type."""
    names = {c.__name__ for c in type(e).__mro__}
    return isinstance(e, OSError) or bool(names & {"ConnectionError", "ClientClosed"})


@app.local_entrypoint()
def main(phase: str = "vllm", force: bool = False, parts: str = ""):
    """parts: with --phase smoke1, a comma-separated subset of its parts (to run a fixed part again, with --force)."""
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    chosen = [p for p in parts.split(",") if p] if parts else None
    if chosen is not None and (phase != "smoke1" or not set(chosen) <= {*SMOKE_H100, *SMOKE_L4}):
        raise SystemExit(f"--parts goes with --phase smoke1, from {', '.join((*SMOKE_H100, *SMOKE_L4))}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    failures = []
    broken = False
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
        elif phase == "serve_tp2":
            message = serve032_tp2.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "ready_attrib":
            message = ready_attrib.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "isolation":
            calls = {"isolation031": isolation031.spawn(force, src), "isolation032": isolation032.spawn(force, src)}
            for tag, call in calls.items():
                try:
                    message = call.get()
                except Exception as e:  # noqa: BLE001
                    message = f"{tag} did not complete: {type(e).__name__}: {e}"
                print(message, flush=True)
                # 0.32.0 is reported as found (the bug may be fixed there); 0.31.0 carries the criteria
                if tag == "isolation031" and "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "fp8_residuals":
            message = fp8_residuals.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "norm_diag":
            message = norm_diag.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "moe_bench":
            message = moe_bench.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "moe":
            message = moe.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "qualify":
            message = qualify.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "fp8kv":
            message = fp8kv.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase in ("fp8mm_cal", "attn", "screen_ops"):
            calls = ({"fp8mm_cal_h100": fp8mm_cal_h100.spawn(force, src),
                      "fp8mm_cal_l4": fp8mm_cal_l4.spawn(force, src)} if phase == "fp8mm_cal" else
                     {"attn_h100": attn_h100.spawn(force, src), "attn_l4": attn_l4.spawn(force, src)}
                     if phase == "attn" else
                     {"screen_ops_h100": screen_ops_h100.spawn(force, src),
                      "screen_ops_l4": screen_ops_l4.spawn(force, src)})
            for tag, call in calls.items():
                try:
                    message = call.get()
                except Exception as e:  # noqa: BLE001
                    message = f"{tag} did not complete: {type(e).__name__}: {e}"
                print(message, flush=True)
                if "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "smoke1":
            result = smoke1.remote(force, src, chosen)
            for part in chosen or (*SMOKE_H100, *SMOKE_L4):
                message = (result or {}).get(part, f"smoke_{part}: no result") if isinstance(result, dict) else \
                    str(result)
                print(message, flush=True)
                if "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "run1":
            result = run1.remote(force, src)
            for tag in RUN1:
                message = (result or {}).get(tag, f"{tag}: no result") if isinstance(result, dict) else str(result)
                print(message, flush=True)
                if "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "spot":
            calls = {"spot_h100": spot_h100.spawn(force, src), "spot_l4": spot_l4.spawn(force, src)}
            for tag, call in calls.items():
                try:
                    message = call.get()
                except Exception as e:  # noqa: BLE001
                    message = f"{tag} did not complete: {type(e).__name__}: {e}"
                print(message, flush=True)
                if "exit code 0 " not in str(message):
                    failures.append(str(message))
        elif phase == "graphs":
            message = graphs.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "fp8_moe":
            message = fp8_moe.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "norms":
            message = norms.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "sensitivity":
            message = sensitivity.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "weight_repair":
            message = weight_repair.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "weight_scan":
            message = weight_scan.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "decisions":
            message = decisions.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
                failures.append(str(message))
        elif phase == "live_catch":
            message = live_catch.remote(force, src)
            print(message, flush=True)
            if "exit code 0 " not in str(message):
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
        broken = _connection_broke(e)
    finally:
        try:
            if broken:
                # the connection to Modal broke off (this computer's network, sleep): calling another function now
                # would make it the last one this client triggered, and a detached run keeps only that one alive
                # (Modal's note), which stopped the October 25 FP8 MoE run after 48 minutes. Download later instead.
                raise RuntimeError("not downloaded: the connection to Modal broke off, so the run is left alone "
                                   "(check it on modal.com, then run --phase fetch)")
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
