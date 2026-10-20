"""The default check against precise mode on the same training matmuls, on one revision
(experiments/mode_compare.py): how small an injected error each flags, and the cost per training step of
each, on an H100, an A100 and an L4 at once (the A100 and the L4 are where precise mode missed its held-out
criteria on Oct 19).

    modal run scripts/modal_mode_compare.py

About 5 to 15 minutes per GPU, about $1.50 in total at Modal's prices (per hour: H100 $3.95, A100 80GB $2.50,
L4 $0.80, plus about $0.45 per machine for 4 CPU cores at $0.0472 and 32 GiB at $0.008 a GiB). Every function
has a 30-minute execution limit: about $4.30 at requested CPU/RAM usage, not a total bill cap. Startup, higher
actual resource usage and restarts can add cost. There is no pass line (the reporting rules are in the experiment's
docstring and in docs/METHOD.md, written before the run); the exit code says whether every GPU ran. Results go
to results/mode_compare/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent


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


R = "/root/sdcguard"
LIMIT = 30 * 60


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-mode-compare", image=image)
KW = dict(cpu=4, memory=32768, timeout=LIMIT)


def _work(label: str) -> dict:
    """Return saved sensitivity and logs even if timing fails or a child reaches its soft deadline."""
    import os
    import signal
    import subprocess
    import sys
    import tempfile

    t0 = time.monotonic()
    res = {"label": label}
    try:
        with tempfile.TemporaryDirectory(prefix="sdcguard-mode-") as td:
            dest = Path(td) / "results"
            with open(Path(td) / "output.txt", "w+") as log:
                budget = LIMIT - 120 - (time.monotonic() - t0)
                if budget <= 0:
                    return {"label": label, "error": "no execution budget remains"}
                cmd = [sys.executable, f"{R}/experiments/mode_compare.py", "--device", "cuda", "--out", str(dest)]
                p = subprocess.Popen(cmd, cwd=R, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = p.wait(timeout=budget)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                    code = 124
                log.seek(0)
                output = log.read()
            files = sorted(dest.glob("*.json"))
            if len(files) == 1:
                res.update(json.loads(files[0].read_text()))
                md = files[0].with_suffix(".md")
                res["markdown"] = md.read_text() if md.exists() else ""
            if code != 0:
                res["error"] = (f"worker exited {code}; " + res.get("error", "see partial output"))
            elif not res.get("complete") or res.get("overhead_status") != "complete":
                res["error"] = "missing or incomplete sensitivity/timing results"
            res["output"] = output
    except Exception as e:  # noqa: BLE001
        import traceback
        res.update(error=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-2000:])
    res.update(label=label, seconds=round(time.monotonic() - t0, 1))
    if res.get("error"):
        res["markdown"] = f"## {label}: INCOMPLETE: {res['error']}\n\n" + res.get("markdown", "")
    return res


def _wait_all(calls, cap=3 * 3600, every=20.0, retry_for=600.0):
    """Poll every call. Queue time is separate from the worker's execution limit.

    Worker failures return error records. Explicit platform terminal errors stop polling; unknown client
    exceptions remain visible after a bounded retry period. A pending poll can raise built-in TimeoutError.
    """
    pending, failing_since = dict(calls), {}
    start = last = time.monotonic()
    final = {"FunctionTimeoutError", "OutputExpiredError", "RemoteError", "ExecutionError", "InputCancellation"}
    while pending:
        for name, call in list(pending.items()):
            try:
                result = call.get(timeout=0)
            except Exception as e:  # noqa: BLE001
                now, kind = time.monotonic(), type(e).__name__
                expired = now - start >= cap
                if kind == "TimeoutError" and not expired:
                    failing_since.pop(name, None)
                    continue
                if kind not in final and kind != "TimeoutError" and not expired:
                    if now - failing_since.setdefault(name, now) < retry_for:
                        continue
                result = {"label": name, "error": f"{kind}: {e}; result unavailable within the wait/retry limit"}
            if not isinstance(result, dict):
                result = {"label": name, "error": "worker returned an invalid result"}
            del pending[name]
            yield name, result
        if pending:
            now = time.monotonic()
            if now - last >= 300:
                print(f"{(now - start) / 60:.0f} minutes: still waiting for {', '.join(pending)}", flush=True)
                last = now
            time.sleep(min(every, max(0.0, cap - (now - start))))


@app.function(gpu="H100!", **KW)
def h100() -> dict:
    return _work("h100")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _work("a100_80gb")


@app.function(gpu="L4", **KW)
def l4() -> dict:
    return _work("l4")


@app.local_entrypoint()
def main() -> None:
    jobs = [("h100", h100), ("a100_80gb", a100), ("l4", l4)]
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    started = [(name, fn.spawn()) for name, fn in jobs]
    print(f"Started {len(started)} GPUs (at most 30 minutes each). Keep this terminal open until it prints Done.",
          flush=True)
    dest = LOCAL / "results" / "mode_compare"
    dest.mkdir(parents=True, exist_ok=True)
    text, failed = [], []
    for name, r in _wait_all(started):
        r["source"] = src
        if not r.get("error") and (not r.get("complete") or r.get("overhead_status") != "complete"):
            r["error"] = "worker returned incomplete measurements"
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        if r.get("error"):
            failed.append(name)
        text.append(r.get("markdown") or f"## {name}: could not run: {r.get('error')}\n")
        print(f"{name}: finished", flush=True)
    head = (f"# Default check vs precise mode, same matmuls, code fingerprint {src['sha256'][:12]}\n\n"
            "No pass line: the reporting rules were written before the run (docs/METHOD.md). Precise mode stays "
            "experimental whatever this shows.\n\n")
    (dest / "SUMMARY.md").write_text(head + "\n".join(text))
    print(head + "\n".join(text))
    print(f"Results saved in {dest}", flush=True)
    if failed:
        print(f"{len(failed)} of {len(jobs)} GPUs didn't run: {', '.join(failed)}. Done.", flush=True)
        raise SystemExit(1)
    print("Every GPU ran. Done.", flush=True)
