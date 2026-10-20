"""Run `sdcguard check` on every Modal GPU type at once. These are the results the public board starts with.

    modal run scripts/modal_check.py --phase tests   # tests on an L4, an H100 and a B200, about 30 to 35 min, about $5
    modal run scripts/modal_check.py                 # the sweep, about 12 min, about $4.40
    modal run scripts/modal_check.py --phase b200    # B200 rerun plus a diagnostic, about 10 min, about $1.10
    modal run scripts/modal_check.py --phase fetch   # download the sweep's results again

Keep the terminal connected until it prints "Done.". This command uses an ordinary ephemeral app: exiting
or disconnecting the client stops its functions. Detached execution has different app-lifetime semantics,
but it does not replace this entrypoint's result collection.

The sweep runs a 5-minute check on a B200, H200, two H100s (separate machines), A100 80GB, L40S, L4, an L4
pair (2 GPUs in one machine) and a T4, all in parallel: 10 GPUs, the most a Modal Starter plan runs at once.
Full output goes to results/board_seed/ (not committed), the shareable cards to docs/board/seed/, and the
board in docs/board/ is rebuilt. Every function has a hard 15-minute limit, and the sweep won't start again
within an hour of its last start unless you pass --force.

Cost estimates use Modal's GPU prices on Oct 18, 2026 (per hour: B200 $6.25, H200 $4.54, H100 $3.95,
A100 80GB $2.50, L40S $1.95, L4 $0.80, T4 $0.59) plus about $0.38 an hour per machine for CPU and memory.
They are estimates at the requested allocations, not spending caps: usage above the CPU/memory request,
container startup/idle time and restarted execution attempts can add charges.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent


def _digest(root: Path) -> dict:
    """Fingerprint of the code under root (src, experiments, tests, scripts), to match it to a commit."""
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "experiments", "tests", "scripts"):
        for p in sorted((root / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    return {"sha256": h.hexdigest(), "files": n}


def _source() -> dict:
    """Fingerprint of the code this run uploads (on the user's machine)."""
    return _digest(LOCAL)


def _fingerprint(root: Path) -> str:
    """The same fingerprint, short, of the code a GPU actually has (the copy in its image)."""
    d = _digest(root)
    return f"{d['sha256'][:12]} ({d['files']} files)"


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
    .add_local_dir(LOCAL / "scripts", f"{R}/scripts", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_file(LOCAL / "pyproject.toml", f"{R}/pyproject.toml")
    .add_local_file(LOCAL / "README.md", f"{R}/README.md")             # (the wheel's description)
    .add_local_file(LOCAL / "LICENSE", f"{R}/LICENSE")
)
# Seed cards are historical artifacts, not prerequisites for testing a source archive.
if (LOCAL / "docs" / "board" / "seed").is_dir():
    image = image.add_local_dir(LOCAL / "docs" / "board" / "seed", f"{R}/docs/board/seed",
                                ignore=lambda p: p.name.startswith("._"))

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


TESTS_LIMIT = {"l4": 45 * 60, "h100": 30 * 60, "b200": 30 * 60}     # each tests function's hard limit (seconds)
TESTS_WAIT = 3 * 3600      # how long the terminal waits for all of them, GPU queues included (queued time is free)


# a result, not a connection problem: the function hit its own time limit, its output expired, or it raised
FINAL_ERRORS = ("FunctionTimeoutError", "OutputExpiredError", "RemoteError", "ExecutionError", "InputCancellation")


def _wait_all(calls, cap: float, every: float = 20.0, say=print, retry_for: float = 600.0):
    """Yield (label, result) for each spawned call as it finishes, in whatever order they finish, printing what's
    still running every 5 minutes. Waiting one call at a time with a fixed timeout gave up on the H100 and the
    B200 on Oct 19 while they were still queued or running (TimeoutError after 35, 70 and 105 minutes). Modal's
    Built-in TimeoutError from a pending get(timeout=0) means "not finished yet". Our worker wrappers return
    user-code exceptions as error records, so a worker's own TimeoutError cannot masquerade as a pending poll.
    Any other error that isn't in FINAL_ERRORS (a
    dropped connection, say) is asked again for up to `retry_for` seconds before it counts, since giving up
    ends the app and stops the GPUs still running. A call still pending after `cap` seconds is reported as not
    finished, with where its logs are."""
    pending = dict(calls)
    failing_since: dict = {}
    t0 = last = time.time()
    while pending:
        for label, call in list(pending.items()):
            try:
                r = call.get(timeout=0)
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                expired = time.time() - t0 >= cap
                if name == "TimeoutError" and not expired:
                    failing_since.pop(label, None)
                    continue                      # pending poll; platform execution timeout has another class
                if name not in FINAL_ERRORS and name != "TimeoutError":
                    since = failing_since.setdefault(label, time.time())
                    if not expired and time.time() - since < retry_for:
                        continue                  # asked again later: a dropped connection isn't a result
                r = {"text": f"===== {label}\ncould not run: {name}: {e}"
                             + (f" (still not finished after {cap / 3600:.1f} hours)" if name == "TimeoutError" else
                                "" if name in FINAL_ERRORS else
                                f" (overall wait limit {cap / 3600:.1f} hours reached)" if expired else
                                f" (the same error for {retry_for / 60:.0f} minutes)")
                             + f"\nLogs written before it stopped: modal volume get sdcguard-runs "
                             f"/check_sweep/pytest_{label}.txt results/gpu_tests/ (also started_, fast_compare_, "
                             f"serve_compare_, wheel_, tests_cli_output_; each starts with the code fingerprint "
                             f"it ran)", "run": 2}
            del pending[label]
            yield label, r
        if pending:
            if time.time() - last >= 300:
                last = time.time()
                say(f"... {(last - t0) / 60:.0f} minutes in, still running or waiting for a GPU: "
                    f"{', '.join(pending)}", flush=True)
            time.sleep(min(every, max(0.0, cap - (time.time() - t0))))


def _step(cmd, budget: float, floor: float = 30.0, **kw) -> subprocess.CompletedProcess:
    """A step with its own time limit: if it runs over, it's stopped (exit code 124, its output so far kept)
    and the steps after it still run. The step runs in its own process group and the whole group is killed
    on a timeout: killing only the direct child left its children (a `sdcguard check` under bash, pytest's
    subprocesses) holding the GPU while the next step ran."""
    budget = max(floor, budget)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
                         **kw)
    try:
        out, err = p.communicate(timeout=budget)
        return subprocess.CompletedProcess(cmd, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = p.communicate(timeout=60)      # what it wrote before it was stopped (nothing is lost)
        except subprocess.TimeoutExpired as e:        # something outside the group still holds the pipes
            p.kill()
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return subprocess.CompletedProcess(cmd, 124, out or "", (err or "") +
                                           f"\n(stopped after {budget / 60:.1f} minutes; its processes were killed)")


def _tests(label: str, with_bench: bool = False) -> dict:
    """The test suite on a GPU (includes a 30-second `check` on the GPU); the Triton checks compared with the
    plain ones on this GPU (tests/_fast_compare.py and tests/_serve_fast_compare.py, which test_fast.py and
    test_serve.py run in Triton's CPU interpreter); the package built as a wheel, installed in a fresh virtual
    environment and run from outside the source tree (what `pip install sdcguard` gets); then the `sdcguard
    check` command itself for 1.5 minutes, exactly as a user would run it; with_bench, per-shape kernel timings;
    last, the attention check's calibration (an experiment with its own criteria, reported separately).
    Returns the output and each part's exit code: a finished run isn't a passed one. Every step has its own time
    limit inside the function's, and the optional steps at the end only start when there's time left (on Oct 19
    none of the three returned in time and nothing came back; the logs each step saves are on the volume)."""
    t0 = time.monotonic()

    def left() -> float:                     # seconds before the hard limit, keeping 2 minutes to save and return
        return TESTS_LIMIT[label] - (time.monotonic() - t0) - 120

    def run(cmd, budget: float, **kw) -> subprocess.CompletedProcess:
        # The initial volume commit and earlier timeout cleanups consume this function's limit too.
        # A minimum per-step budget must never restart the clock or spend the final save/return reserve.
        remaining = left()
        if remaining < 30.0:
            return subprocess.CompletedProcess(cmd, 125, "", "not started: insufficient time before the "
                                               "function's save/return reserve")
        return _step(cmd, min(max(30.0, budget), remaining), floor=0.0, **kw)

    def stamp(path: str, text: str) -> None:
        """Write a step's log with the code fingerprint and the time it was written on top (an old log left on the
        volume can't pass for this run's)."""
        with open(path, "w") as f:
            f.write(f"code fingerprint {fingerprint}, written {time.strftime('%Y-%m-%d %H:%M:%S %Z')}, "
                    f"{(time.monotonic() - t0) / 60:.1f} minutes in\n" + text)

    fingerprint = _fingerprint(Path(R))

    os.makedirs(OUT, exist_ok=True)
    for name in ("pytest", "fast_compare", "serve_compare", "bench", "wheel", "tests_cli_output", "attention_output"):
        try:                                   # an earlier run's logs must not be read as this run's
            os.remove(f"{OUT}/{name}_{label}.txt")
        except OSError:
            pass
    shutil.rmtree(f"{OUT}/attention_{label}", ignore_errors=True)
    stamp(f"{OUT}/started_{label}.txt", "started\n")
    vol.commit()
    # The suite normally gets at most half the function's time, reserving time for comparisons, the wheel and
    # `check`. If setup or an earlier cleanup exhausts that allocation, unstarted mandatory steps fail visibly.
    # -rfEs names failed, errored and skipped tests in the summary; --durations names the slowest tests, and a
    # test still running after 4 minutes gets every thread's stack printed (pytest's faulthandler_timeout)
    p = run([sys.executable, "-m", "pytest", "-q", "-rfEs", "--durations=25", "-o", "faulthandler_timeout=240",
             "tests", "--ignore", "tests/test_fast.py"],
            min(TESTS_LIMIT[label] * 0.5, left() - 11.5 * 60), cwd=R)   # (11.5 minutes kept for the mandatory steps after)
    stamp(f"{OUT}/pytest_{label}.txt", p.stdout + "\n" + p.stderr + f"\n[exit code {p.returncode}]\n")
    failed_tests = [ln for ln in p.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))]
    vol.commit()
    env = {**os.environ, "PYTHONPATH": f"{R}/src"}
    fc = run([sys.executable, "tests/_fast_compare.py", "cuda"], min(8 * 60, left() - 10 * 60), cwd=R, env=env)
    stamp(f"{OUT}/fast_compare_{label}.txt", fc.stdout + "\n" + fc.stderr + f"\n[exit code {fc.returncode}]\n")
    vol.commit()
    sc = run([sys.executable, "tests/_serve_fast_compare.py", "cuda"], min(8 * 60, left() - 8 * 60), cwd=R, env=env)
    stamp(f"{OUT}/serve_compare_{label}.txt", sc.stdout + "\n" + sc.stderr + f"\n[exit code {sc.returncode}]\n")
    vol.commit()
    bench, bench_note = None, ""
    if with_bench and left() > 15 * 60:      # per-shape time of the Triton check next to the matmul and the plain check
        bench = run([sys.executable, "experiments/bench_kernels.py"], min(8 * 60, left() - 7 * 60), cwd=R, env=env)
        stamp(f"{OUT}/bench_{label}.txt", bench.stdout + "\n" + bench.stderr + f"\n[exit code {bench.returncode}]\n")
    elif with_bench:
        bench_note = f"Kernel timings: not run, {left() / 60:.1f} minutes left (they need 15).\n\n"
    # The wheel, in a clean environment that sees torch but not the source tree. Every step runs without
    # PYTHONPATH and builds from a copy: with the image's PYTHONPATH (the source tree) and the egg-info an
    # in-tree build leaves there, pip sees sdcguard 0.3.0 as "already installed" and skips the wheel
    # (what broke this step on Oct 19).
    whl = run(
        "set -e; unset PYTHONPATH; rm -rf /tmp/w /tmp/venv /tmp/run /tmp/srccopy; mkdir -p /tmp/w /tmp/run; "
        f"mkdir /tmp/srccopy && cp -r {R}/src {R}/pyproject.toml {R}/README.md {R}/LICENSE /tmp/srccopy/; "
        "rm -rf /tmp/srccopy/src/*.egg-info; "
        f"{sys.executable} -m pip wheel --no-deps --no-build-isolation -q -w /tmp/w /tmp/srccopy; "
        f"{sys.executable} -m zipfile -l /tmp/w/sdcguard-*.whl | tail -n +2 | wc -l | sed 's/^/files in the wheel: /'; "
        f"{sys.executable} -m venv --system-site-packages /tmp/venv; "
        "/tmp/venv/bin/python -m pip install --no-deps --disable-pip-version-check /tmp/w/sdcguard-*.whl; "
        "cd /tmp/run; "
        "/tmp/venv/bin/python -c 'import sdcguard, sys; print(sdcguard.__file__); "
        "sys.exit(0 if \"/tmp/venv/\" in sdcguard.__file__ else 1)'; "
        "/tmp/venv/bin/sdcguard selftest; "
        "/tmp/venv/bin/sdcguard check --minutes 0.5 --offline --out /tmp/run/wheel_check",
        min(6 * 60, left() - 4 * 60), shell=True, executable="/bin/bash")
    stamp(f"{OUT}/wheel_{label}.txt", whl.stdout + "\n" + whl.stderr + f"\n[exit code {whl.returncode}]\n")
    if os.path.isdir(f"{OUT}/tests_cli_{label}"):
        shutil.rmtree(f"{OUT}/tests_cli_{label}")
    c = run([sys.executable, "-m", "sdcguard", "check", "--minutes", "1.5", "--offline", "--out",
             f"{OUT}/tests_cli_{label}"], min(6 * 60, left()), cwd=R)
    stamp(f"{OUT}/tests_cli_output_{label}.txt", c.stdout + "\n" + c.stderr + f"\n[exit code {c.returncode}]\n")
    vol.commit()
    # Last, within the time left (so the rest is never lost to it): the attention check's calibration
    # (experiments/attention_calib.py; its criteria are in its docstring). Exit 3: not enough time to start;
    # 124: stopped at its time limit; 2: the attention calibration was incomplete.
    budget = min(9 * 60, left())
    if budget < 4 * 60:
        ac = subprocess.CompletedProcess([], 3, "", f"not started: {max(0, budget) / 60:.1f} minutes left")
    else:
        ac = run([sys.executable, "experiments/attention_calib.py", "--device", "cuda", "--out",
                  f"{OUT}/attention_{label}"], budget, cwd=R, env=env)
    att_json = None
    stamp(f"{OUT}/attention_output_{label}.txt", ac.stdout + "\n" + ac.stderr + f"\n[exit code {ac.returncode}]\n")
    if ac.returncode in (0, 1, 2):             # completed process, including an explicit incomplete calibration
        try:
            with open(f"{OUT}/attention_{label}/attention_calib.json") as f:
                att_json = f.read()
        except OSError:
            pass
    vol.commit()
    mismatches = [ln for ln in fc.stdout.splitlines() if '"agree": false' in ln]
    text = (f"===== {label}\ncode fingerprint on the GPU: {fingerprint}\n" + p.stdout[-3500:]
            + (f"\n{len(failed_tests)} failed or errored tests:\n" + "\n".join(failed_tests[:60]) if failed_tests else "")
            + f"\n[pytest exit code {p.returncode}]\n\n"
            + "Triton vs plain checks on this GPU: " + (fc.stdout.strip().splitlines() or ["(no output)"])[-1]
            + "".join("\n  " + ln[:300] for ln in mismatches) + (fc.stderr[-1500:] if fc.returncode else "")
            + f"\n[fast compare exit code {fc.returncode}]\n\n"
            + "Triton vs plain serving checks on this GPU: "
            + (sc.stdout.strip().splitlines() or ["(no output)"])[-1]
            + "".join("\n  " + ln[:300] for ln in sc.stdout.splitlines() if '"agree": false' in ln)
            + (sc.stderr[-1500:] if sc.returncode else "") + f"\n[serve compare exit code {sc.returncode}]\n\n"
            + ("" if bench is None else "Kernel timings (experiments/bench_kernels.py):\n" + bench.stdout[-3000:]
               + bench.stderr[-800:] + f"\n[bench exit code {bench.returncode}"
               + (": stopped at its time limit, timing data only, not a failure" if bench.returncode == 124 else "")
               + "]\n\n") + bench_note
            + "Installed wheel (fresh venv, outside the source tree):\n" + whl.stdout[-2500:] + whl.stderr[-1500:]
            + f"\n[wheel exit code {whl.returncode}]\n\n"
            + c.stdout[-2500:] + c.stderr[-1500:] + f"\n[sdcguard check exit code {c.returncode}]\n\n"
            + "Attention check calibration (experiments/attention_calib.py):\n" + ac.stdout[-4000:]
            + (ac.stderr[-1500:] if ac.returncode not in (0, 1) else "")
            + f"\n[attention calibration exit code {ac.returncode}: 0 criteria met, 1 not met, 2 INCOMPLETE, "
            f"3 not started (no time left), 124 stopped at its time limit, other: error]")
    # (kernel timings stopped at their limit are timing data, not a test failure; a crash is)
    bench_code = {} if bench is None or bench.returncode == 124 else {"bench": bench.returncode}
    return {"text": text, "pytest": p.returncode, "fast_compare": fc.returncode, "serve_compare": sc.returncode,
            "wheel": whl.returncode, "check": c.returncode, **bench_code,
            "attention_calibration": ac.returncode, "attention_json": att_json}


def _tests_result(label: str, with_bench: bool = False) -> dict:
    """Keep a worker exception distinct from the SDK's pending-output polling exception."""
    try:
        return _tests(label, with_bench=with_bench)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"run": 2, "text": f"===== {label}\nworker failed: {type(e).__name__}: {e}\n"
                                 + traceback.format_exc()[-6000:]}


@app.function(gpu="L4", volumes=VOLS, cpu=4, memory=16384, timeout=TESTS_LIMIT["l4"])
def tests() -> dict:
    return _tests_result("l4")


@app.function(gpu="H100!", volumes=VOLS, cpu=4, memory=32768, timeout=TESTS_LIMIT["h100"])
def tests_h100() -> dict:
    return _tests_result("h100")


@app.function(gpu="B200", volumes=VOLS, cpu=4, memory=32768, timeout=TESTS_LIMIT["b200"])
def tests_b200() -> dict:
    return _tests_result("b200", with_bench=True)


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
        if label == "tests_cli" or label.startswith("tests_cli_"):   # release smoke tests, not the sweep
            continue
        shutil.copy(card, seed / f"modal_{label}_{gpu[-8:]}.json")
        n += 1
    print(f"Results: {dest / 'board_seed'} ({len(data) / 1e6:.1f} MB); {n} result cards in {seed}", flush=True)
    pngs = sorted((dest / "board_seed").glob("h100_a/*/card.png")) or [
        p for p in sorted((dest / "board_seed").glob("*/*/card.png"))
        if p.parent.parent.name != "tests_cli" and not p.parent.parent.name.startswith("tests_cli_")]
    if pngs:
        (LOCAL / "docs" / "img").mkdir(parents=True, exist_ok=True)
        shutil.copy(pngs[0], LOCAL / "docs" / "img" / "example_card.png")
        print(f"README example card: docs/img/example_card.png (from {pngs[0].parent.parent.name})", flush=True)
    subprocess.run([sys.executable, str(LOCAL / "src" / "sdcguard" / "board.py"), "--dir", str(seed), "--out",
                    str(LOCAL / "docs" / "board")], check=True)


def _passed(result: dict) -> bool:
    """A completed remote call is useful evidence only if it returned successful checks and actual cards."""
    cards = result.get("cards")
    return result.get("exit") == 0 and isinstance(cards, dict) and bool(cards) and all(
        card.get("result") == "PASS" for card in cards.values()
    )


@app.local_entrypoint()
def main(phase: str = "sweep", force: bool = False):
    if phase == "tests":
        print("Tests on an L4, an H100 and a B200 at once: about 30 to 35 minutes once they have GPUs, about $5 "
              "(function limits 45 minutes for the L4 and 30 for the others; about $6.40 at requested resources "
              "for one full-length attempt, not a spending cap).", flush=True)
        src = _source()
        print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
        calls = [("l4", tests.spawn()), ("h100", tests_h100.spawn()), ("b200", tests_b200.spawn())]
        print("Started. Each GPU's result prints as soon as it's done; time spent waiting for a GPU isn't billed. "
              "Keep this terminal open until it prints Done.", flush=True)
        dest = LOCAL / "results" / "gpu_tests"
        dest.mkdir(parents=True, exist_ok=True)
        failed, att = {}, {}
        for label, r in _wait_all(calls, TESTS_WAIT):
            r["text"] = f"code fingerprint {src['sha256']} ({src['files']} files)\n" + r["text"]
            print(r["text"], flush=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            path = dest / f"{label}_{stamp}.txt"
            path.write_text(r["text"])
            print(f"\nSaved {path}.", flush=True)
            if r.get("attention_json"):
                adir = LOCAL / "results" / "attention_calib"
                adir.mkdir(parents=True, exist_ok=True)
                (adir / f"{label}_{stamp}.json").write_text(r["attention_json"])
            # the attention calibration is an experiment with its own criteria: reported, not a test failure
            att[label] = r.get("attention_calibration")
            failed.update({f"{label} {k}": v for k, v in r.items()
                           if k not in ("text", "attention_json", "attention_calibration") and v != 0})
        verdicts = {0: "MET", 1: "NOT MET", 2: "INCOMPLETE", 3: "not started (no time left)",
                    124: "stopped at its time limit", None: "didn't run"}
        print("Attention check criteria (0 clean alarms, worst at most 4): " + ", ".join(
            f"{k} {verdicts.get(v, f'error ({v})')}" for k, v in att.items()), flush=True)
        if failed:
            print(f"GPU TESTS FAILED: {failed} (exit codes; sdcguard check: 1 FAIL or SUSPECT, 2 INCOMPLETE). Done.",
                  flush=True)
            raise SystemExit(1)
        print("All passed. Done.", flush=True)
        return
    if phase == "fetch":
        _download()
        print("\nDone.", flush=True)
        return
    if phase == "b200":
        print("B200 again, with a diagnostic first: about 10 minutes, about $1.10 (20-minute function limit, "
              "about $2.20 at requested resources for one attempt, not a spending cap).", flush=True)
        msg = lock.remote(force, "b200")
        if msg:
            print(msg)
            return
        r = b200_again.remote()
        verdicts = ", ".join(f"{c['gpu']['name']}: {c['result']}" for c in r["cards"].values()) or "no result"
        print(f"\n=== {r['label']} ({r['minutes']} min, exit {r['exit']}): {verdicts}\n{r['output'][-2500:]}", flush=True)
        if r["exit"] not in (0, 1) and r["errors"]:
            print(r["errors"][-1500:], flush=True)
        failed = not _passed(r)
        try:
            _download()
        except Exception as e:  # noqa: BLE001
            failed = True
            print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
        if failed:
            raise SystemExit("B200 VALIDATION FAILED OR INCOMPLETE. Inspect the saved results.")
        print("\nDone.", flush=True)
        return
    if phase != "sweep":
        raise SystemExit("--phase must be sweep, tests, b200 or fetch")
    print("Sweep: 10 GPUs on 9 machines in parallel, about 12 minutes, about $4.40 (about $7.40 at requested "
          "resources for one full-length attempt, not a spending cap). "
          "Everything shuts down by itself.", flush=True)
    msg = lock.remote(force)
    if msg:
        print(msg)
        return
    calls = [(label, f.spawn()) for label, f in SWEEP]
    failed = []
    for label, call in calls:
        try:
            r = call.get()
            if not _passed(r):
                failed.append(label)
            verdicts = ", ".join(f"{c['gpu']['name']}: {c['result']}" for c in r["cards"].values()) or "no result"
            print(f"\n=== {r['label']} ({r['minutes']} min, exit {r['exit']}): {verdicts}\n{r['output'][-2500:]}", flush=True)
            if r["exit"] not in (0, 1) and r["errors"]:
                print(r["errors"][-1500:], flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(label)
            print(f"\n=== {label}: did not finish ({type(e).__name__}: {e})", flush=True)
    try:
        _download()
    except Exception as e:  # noqa: BLE001
        failed.append("result download or board build")
        print(f"(could not download results: {e}. Run --phase fetch later.)", flush=True)
    if failed:
        raise SystemExit(f"SWEEP FAILED OR INCOMPLETE: {', '.join(failed)}. Inspect the saved results.")
    print("\nDone.", flush=True)
