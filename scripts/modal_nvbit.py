"""Hardware-level fault injection (NVBit) on a Modal L4, with sdcguard checking every step.

    modal run scripts/modal_nvbit.py --phase nvbit   # about 12 min, about $0.20

Faults go into the tensor-core instructions of one SM lane while a small transformer trains and serves
(experiments/nvbit_check.py). Results are saved to the Modal volume sdcguard-runs and downloaded to
runs/gpu_r3/. Every function has a hard time limit, and a phase won't start again within an hour of its
last start unless you pass --force.
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


# NVBit instruction-level fault injection on an L4 (Ada, sm_89: its cuBLAS GEMMs use HMMA, which the
# injector targets). The injector is TU Berlin's (github.com/aaltenbernd/llm-sdc-training), built against
# NVBit 1.8 (their repo bundles 1.7.4, which supports drivers up to 555). Round 3 ran it on Modal's L4
# (driver 580) without trouble. Two changes to the injector:
#   - with TARGET_FUNC=-1 it injects into every instrumented GEMM kernel, not only one chosen kernel ID
#     (a bad tensor-core lane would corrupt every kernel that runs on it)
#   - it is switched on while the file /tmp/sdcguard_fault_on exists, checked at every kernel launch,
#     instead of by SIGUSR1/SIGUSR2. In round 3 one of three runs never switched on (its positive control
#     changed nothing); a signal handler installed in the wrong order would do exactly that.
NVB = "/root/nvbit_release_x86_64/tools/fault_injection"
nvbit_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "bc", "curl", "bzip2")
    .uv_pip_install("torch==2.14.1", "numpy")
    .run_commands(
        "cd /root && curl -sSL -o nvbit.tar.bz2 https://github.com/NVlabs/NVBit/releases/download/v1.8/"
        "nvbit-Linux-x86_64-1.8.tar.bz2 && tar xjf nvbit.tar.bz2 && rm nvbit.tar.bz2",
        "git clone --depth 1 https://github.com/aaltenbernd/llm-sdc-training /root/tub",
        f"cp -r /root/tub/nvbit/1.7.4_nvbit_release/tools/fault_injection {NVB} && rm -f {NVB}/*.o {NVB}/*.so",
        f"cd {NVB} && sed -i 's/enable = (fid == active_target_func.load());/enable = (active_target_func.load() < 0) || "
        f"(fid == active_target_func.load());/' fault_injection.cu && grep -n 'active_target_func.load() < 0' fault_injection.cu",
        f"cd {NVB} && sed -i 's|bool global_on = g_on.load(std::memory_order_relaxed);|bool global_on = "
        f"(access(\"/tmp/sdcguard_fault_on\", F_OK) == 0);|' fault_injection.cu && sed -i '1i #include <unistd.h>' "
        f"fault_injection.cu && grep -n 'sdcguard_fault_on' fault_injection.cu && grep -n 'unistd' fault_injection.cu",
        f"cd {NVB} && make ARCH=sm_89 && ls -la fault_injection.so",
    )
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-nvbit", image=nvbit_image)
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


@app.function(image=nvbit_image, volumes=VOLS, gpu="L4", cpu=4, memory=32768, timeout=40 * 60)
def nvbit(force: bool = False) -> str:
    msg = _locked("nvbit", force)
    if msg:
        return msg
    import shutil
    out = f"{RUNS}/nvbit"
    if os.path.isdir(out):
        shutil.rmtree(out)
    t0 = time.time()
    # CUDA_INJECTION64_PATH rather than LD_PRELOAD: NVBit's README says PyTorch can override LD_PRELOAD
    base = {"CUDA_INJECTION64_PATH": f"{NVB}/fault_injection.so", "TARGET_OP": "HMMA", "TARGET_REGISTER": "1",
            "TARGET_SMID": "0", "TARGET_LANEID": "0", "TARGET_FUNC": "-1", "TARGET_INSTR": "-1",
            "TARGET_FUNC_CONTAINS": "gemm", "TOOL_VERBOSE": "0"}
    try:
        _tee(["nvidia-smi"], f"{out}/nvidia-smi.txt")
        # The register holds two bf16 inputs; bits 0-15 are the first: sign bit 15, exponent bits 14-7,
        # mantissa bits 6-0. Bit 14 (top exponent bit): values near 1 become ~1e38, Inf or NaN. Bit 4
        # (mantissa, worth 2^-3): the input changes by 6 to 12 percent. Bit 1 (mantissa, 2^-6): by 0.8
        # to 1.6 percent, near the smallest error the check is built to see.
        for mask in (1 << 14, 1 << 4, 1 << 1):
            _tee([sys.executable, "experiments/nvbit_check.py"], f"{out}/mask_{mask}.txt",
                 env={**base, "TARGET_BITMASK": str(mask), "NVBIT_OUT": f"{out}/mask_{mask}.json"})
            vol.commit()
    finally:
        with open(f"{out}/TIME.txt", "w") as f:
            f.write(f"{(time.time() - t0) / 60:.1f} minutes on L4\n")
        vol.commit()
    return f"nvbit finished in {(time.time() - t0) / 60:.1f} minutes"


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
    "nvbit": "about 12 minutes on 1 L4: roughly $0.20 (hard limit 40 minutes, about $0.60)",
    "fetch": "download only: a few cents",
}


@app.local_entrypoint()
def main(phase: str = "nvbit", force: bool = False):
    if phase not in ESTIMATES:
        raise SystemExit(f"--phase must be one of {', '.join(ESTIMATES)}")
    print(f"Phase '{phase}': {ESTIMATES[phase]}. Everything shuts down by itself when it's done.", flush=True)
    try:
        if phase == "nvbit":
            print(nvbit.remote(force), flush=True)
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
