"""Run `sdcguard check` on every Modal GPU type at once. These are the results the public board starts with.

    modal run scripts/modal_check.py --phase tests   # test suite on an L4, about 13 min, about $0.25
    modal run scripts/modal_check.py                 # the sweep, about 12 min, about $4.40
    modal run scripts/modal_check.py --phase b200    # B200 rerun plus a diagnostic, about 10 min, about $1.10
    modal run scripts/modal_check.py --phase fetch   # download the sweep's results again

Keep the terminal connected until it prints "Done.". If the client goes away, Modal only keeps the last
started function alive (even with --detach), which would stop most of the sweep.

The sweep runs a 5-minute check on a B200, H200, two H100s (separate machines), A100 80GB, L40S, L4, an L4
pair (2 GPUs in one machine) and a T4, all in parallel: 10 GPUs, the most a Modal Starter plan runs at once.
Full output goes to results/board_seed/ (not committed), the shareable cards to docs/board/seed/, and the
board in docs/board/ is rebuilt. Every function has a hard 15-minute limit, and the sweep won't start again
within an hour of its last start unless you pass --force.

Cost estimates use Modal's GPU prices on Oct 18, 2026 (per hour: B200 $6.25, H200 $4.54, H100 $3.95,
A100 80GB $2.50, L40S $1.95, L4 $0.80, T4 $0.59) plus about $0.38 an hour per machine for CPU and memory.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
V = "/vol"
OUT = f"{V}/check_sweep"
LOCK_HOURS = 1.0
LIMIT = 15 * 60


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")                       # Triton compiles a small launcher once
    .uv_pip_install("torch==2.14.1", "numpy", "matplotlib", "tiktoken", "pytest", "transformers==5.20.1",
                    "cryptography", "prometheus-client")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1", "TIKTOKEN_CACHE_DIR": f"{V}/data/tiktoken"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml")
    # two board tests read the seed cards
    .add_local_dir(LOCAL / "docs" / "board" / "seed", f"{R}/docs/board/seed",
                   ignore=lambda p: p.name.startswith("._"))
)

app = modal.App("sdcguard-check", image=image)
vol = modal.Volume.from_name("sdcguard-runs", create_if_missing=True)
VOLS = {V: vol}


def _run(label: str, minutes: float = 5.0) -> dict:
    out = f"{OUT}/{label}"
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    smi += subprocess.run(["nvidia-smi", "--help-query-gpu"], capture_output=True, text=True).stdout[-6000:]
    smi += subprocess.run([sys.executable, "-c", "import torch; print('torch', torch.__version__, torch.version.cuda, "
                           "torch.cuda.get_arch_list())"], capture_output=True, text=True).stdout
    p = subprocess.run([sys.executable, "-m", "sdcguard", "check", "--minutes", str(minutes), "--out", out,
                        "--offline"], cwd=R, capture_output=True, text=True)
    with open(f"{out}/check_output.txt", "w") as f:
        f.write(smi + "\n" + p.stdout + "\n" + p.stderr + f"\n[exit code {p.returncode}]\n")
    vol.commit()
    cards = {}
    for d in sorted(Path(out).glob("*/result.json")):
        cards[d.parent.name] = json.loads(d.read_text())
    return {"label": label, "exit": p.returncode, "minutes": round((time.time() - t0) / 60, 1),
            "output": p.stdout[-6000:], "errors": p.stderr[-3000:], "cards": cards}


KW = dict(volumes=VOLS, cpu=4, memory=24576, timeout=LIMIT)


@app.function(gpu="B200", **KW)
def b200() -> dict:
    return _run("b200")


@app.function(gpu="H200", **KW)
def h200() -> dict:
    return _run("h200")


@app.function(gpu="H100!", **KW)
def h100_a() -> dict:
    return _run("h100_a")


@app.function(gpu="H100!", **KW)
def h100_b() -> dict:
    return _run("h100_b")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _run("a100_80gb")


@app.function(gpu="L40S", **KW)
def l40s() -> dict:
    return _run("l40s")


@app.function(gpu="L4", **KW)
def l4_a() -> dict:
    return _run("l4_a")


@app.function(gpu="L4:2", **KW)
def l4_pair() -> dict:
    return _run("l4_pair")


@app.function(gpu="T4", **KW)
def t4() -> dict:
    return _run("t4")


SWEEP = [("b200", b200), ("h200", h200), ("h100_a", h100_a), ("h100_b", h100_b), ("a100_80gb", a100),
         ("l40s", l40s), ("l4_a", l4_a), ("l4_pair", l4_pair), ("t4", t4)]


@app.function(gpu="L4", volumes=VOLS, cpu=4, memory=16384, timeout=30 * 60)
def tests() -> dict:
    """The test suite on a GPU (includes a 30-second `check` on the GPU); the Triton checks compared with the
    plain ones on this GPU (tests/_fast_compare.py, the comparison test_fast.py runs in Triton's CPU
    interpreter); then the `sdcguard check` command itself for 1.5 minutes, exactly as a user would run it.
    Returns the output and each part's exit code: a finished run isn't a passed one."""
    os.makedirs(OUT, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "tests", "--ignore", "tests/test_fast.py"],
                       cwd=R, capture_output=True, text=True)
    with open(f"{OUT}/pytest_l4.txt", "w") as f:
        f.write(p.stdout + "\n" + p.stderr + f"\n[exit code {p.returncode}]\n")
    vol.commit()
    fc = subprocess.run([sys.executable, "tests/_fast_compare.py", "cuda"], cwd=R, capture_output=True, text=True,
                        env={**os.environ, "PYTHONPATH": f"{R}/src"})
    with open(f"{OUT}/fast_compare_l4.txt", "w") as f:
        f.write(fc.stdout + "\n" + fc.stderr + f"\n[exit code {fc.returncode}]\n")
    if os.path.isdir(f"{OUT}/tests_cli"):
        shutil.rmtree(f"{OUT}/tests_cli")
    c = subprocess.run([sys.executable, "-m", "sdcguard", "check", "--minutes", "1.5", "--offline", "--out",
                        f"{OUT}/tests_cli"], cwd=R, capture_output=True, text=True)
    with open(f"{OUT}/tests_cli_output.txt", "w") as f:
        f.write(c.stdout + "\n" + c.stderr + f"\n[exit code {c.returncode}]\n")
    vol.commit()
    mismatches = [ln for ln in fc.stdout.splitlines() if '"agree": false' in ln]
    text = (p.stdout[-3500:] + f"\n[pytest exit code {p.returncode}]\n\n"
            + "Triton vs plain checks on this GPU: " + (fc.stdout.strip().splitlines() or ["(no output)"])[-1]
            + "".join("\n  " + ln[:300] for ln in mismatches) + (fc.stderr[-1500:] if fc.returncode else "")
            + f"\n[fast compare exit code {fc.returncode}]\n\n"
            + c.stdout[-2500:] + c.stderr[-1500:] + f"\n[sdcguard check exit code {c.returncode}]")
    return {"text": text, "pytest": p.returncode, "fast_compare": fc.returncode, "check": c.returncode}


DIAG = r"""
import os, sys, torch
step = sys.argv[1]
torch.manual_seed(0)
n = 8192
A = torch.randn(n, n, device="cuda").bfloat16()
B = torch.randn(n, n, device="cuda").bfloat16()
from sdcguard.guard import _exact_reductions_on
if step != "cublas_default":
    _exact_reductions_on()
C = A @ B
torch.cuda.synchronize()
if step in ("plain", "triton_train"):
    from sdcguard.checksum import check_matmul
    from sdcguard import _group
    r = check_matmul(A, B, C)
    torch.cuda.synchronize()
    print("check:", r, "triton status:", _group.STATUS, "why:", _group.WHY)
if step == "triton_serve":
    from sdcguard import _serve_fast
    print("serve kernel verified:", _serve_fast.verified(torch.device("cuda"), torch.bfloat16), _serve_fast.STATUS)
torch.cuda.synchronize()
print("OK", step)
"""


def _diag() -> None:
    """Which part faults on this GPU: each step in its own process (a fault ends the process), launches
    synchronous so the error points at the kernel that caused it."""
    lines = []
    for step, extra in (("cublas_default", {}), ("cublas_exact", {}), ("plain", {"SDCGUARD_FAST": "0"}),
                        ("triton_train", {"SDCGUARD_TRITON_NEW_ARCH": "1"}),
                        ("triton_serve", {"SDCGUARD_TRITON_NEW_ARCH": "1"})):
        env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1", **extra}
        try:
            p = subprocess.run([sys.executable, "-c", DIAG, step], cwd=R, env=env, capture_output=True, text=True,
                               timeout=300)
            out = (p.stdout + p.stderr)[-3000:]
            code = p.returncode
        except subprocess.TimeoutExpired:
            out, code = "(timed out)", "timeout"
        lines.append(f"===== {step}: exit {code}\n{out}\n")
        print(f"diagnostic {step}: exit {code}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/b200_diagnostic.txt", "w") as f:
        f.write("\n".join(lines))
    vol.commit()


@app.function(gpu="B200", volumes=VOLS, cpu=4, memory=24576, timeout=20 * 60)
def b200_again() -> dict:
    _diag()
    return _run("b200")


@app.function(volumes=VOLS, cpu=1, memory=2048, timeout=5 * 60)
def lock(force: bool, name: str = "check_sweep") -> str:
    vol.reload()
    path = f"{V}/locks/{name}.json"
    if os.path.exists(path) and not force:
        try:
            age = (time.time() - json.load(open(path))["started"]) / 3600
        except (OSError, ValueError, KeyError):
            age = 0.0
        if age < LOCK_HOURS:
            return (f"NOT STARTED: the sweep was started {age * 60:.0f} minutes ago and may still be running. "
                    "Starting it twice would pay twice. Wait, then run --phase fetch.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"started": time.time()}, f)
    vol.commit()
    return ""


@app.function(volumes=VOLS, cpu=1, memory=4096, timeout=10 * 60)
def bundle() -> bytes:
    vol.reload()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        if os.path.isdir(OUT):
            tar.add(OUT, arcname="board_seed", filter=lambda ti: None if ti.name.endswith(".pt") else ti)
    return buf.getvalue()


def _download() -> None:
    data = bundle.remote()
    dest = LOCAL / "results"
    dest.mkdir(exist_ok=True)
    if (dest / "board_seed").exists():
        shutil.rmtree(dest / "board_seed")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    sys.path.insert(0, str(LOCAL / "src"))
    from sdcguard import check as _check            # rewrite the cards with this version's card format
    for full in sorted((dest / "board_seed").glob("*/*/check.json")):
        try:
            _check.refresh(str(full.parent))
        except Exception as e:  # noqa: BLE001
            print(f"(could not refresh {full.parent}: {e})", flush=True)
    seed = LOCAL / "docs" / "board" / "seed"
    seed.mkdir(parents=True, exist_ok=True)
    for old in seed.glob("modal_*.json"):
        old.unlink()
    n = 0
    for card in sorted((dest / "board_seed").glob("*/*/result.json")):
        label, gpu = card.parent.parent.name, card.parent.name
        if label == "tests_cli":                    # the 1.5-minute test run, not part of the sweep
            continue
        shutil.copy(card, seed / f"modal_{label}_{gpu[-8:]}.json")
        n += 1
    print(f"Results: {dest / 'board_seed'} ({len(data) / 1e6:.1f} MB); {n} result cards in {seed}", flush=True)
    pngs = sorted((dest / "board_seed").glob("h100_a/*/card.png")) or sorted((dest / "board_seed").glob("*/*/card.png"))
    if pngs:
        (LOCAL / "docs" / "img").mkdir(parents=True, exist_ok=True)
        shutil.copy(pngs[0], LOCAL / "docs" / "img" / "example_card.png")
        print(f"README example card: docs/img/example_card.png (from {pngs[0].parent.parent.name})", flush=True)
    subprocess.run([sys.executable, str(LOCAL / "src" / "sdcguard" / "board.py"), "--dir", str(seed), "--out",
                    str(LOCAL / "docs" / "board")])


@app.local_entrypoint()
def main(phase: str = "sweep", force: bool = False):
    if phase == "tests":
        print("Tests on 1 L4: about 13 minutes, about $0.25 (hard limit 30 minutes, about $0.60).", flush=True)
        r = tests.remote()
        print(r["text"], flush=True)
        dest = LOCAL / "results" / "gpu_tests"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / f"l4_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        path.write_text(r["text"])
        failed = {k: v for k, v in r.items() if k != "text" and v != 0}
        print(f"\nSaved {path}.", flush=True)
        if failed:
            print(f"GPU TESTS FAILED: {failed} (exit codes; sdcguard check: 1 FAIL or SUSPECT, 2 INCOMPLETE)", flush=True)
            raise SystemExit(1)
        print("All passed. Done.", flush=True)
        return
    if phase == "fetch":
        _download()
        print("\nDone.", flush=True)
        return
    if phase == "b200":
        print("B200 again, with a diagnostic first: about 10 minutes, about $1.10 (hard limit 20 minutes, about "
              "$2.20).", flush=True)
        msg = lock.remote(force, "b200")
        if msg:
            print(msg)
            return
        r = b200_again.remote()
        verdicts = ", ".join(f"{c['gpu']['name']}: {c['result']}" for c in r["cards"].values()) or "no result"
        print(f"\n=== {r['label']} ({r['minutes']} min, exit {r['exit']}): {verdicts}\n{r['output'][-2500:]}", flush=True)
        if r["exit"] not in (0, 1) and r["errors"]:
            print(r["errors"][-1500:], flush=True)
        try:
            _download()
        except Exception as e:  # noqa: BLE001
            print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
        print("\nDone.", flush=True)
        return
    if phase != "sweep":
        raise SystemExit("--phase must be sweep, tests, b200 or fetch")
    print("Sweep: 10 GPUs on 9 machines in parallel, about 12 minutes, about $4.40 (hard cap about $7.40). "
          "Everything shuts down by itself.", flush=True)
    msg = lock.remote(force)
    if msg:
        print(msg)
        return
    calls = [(label, f.spawn()) for label, f in SWEEP]
    for label, call in calls:
        try:
            r = call.get()
            verdicts = ", ".join(f"{c['gpu']['name']}: {c['result']}" for c in r["cards"].values()) or "no result"
            print(f"\n=== {r['label']} ({r['minutes']} min, exit {r['exit']}): {verdicts}\n{r['output'][-2500:]}", flush=True)
            if r["exit"] not in (0, 1) and r["errors"]:
                print(r["errors"][-1500:], flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {label}: did not finish ({type(e).__name__}: {e})", flush=True)
    try:
        _download()
    except Exception as e:  # noqa: BLE001
        print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
    print("\nDone.", flush=True)
