"""Head-to-head on a real NVIDIA bug: NVIDIA's DCGM diagnostic vs gpu-burn vs `sdcguard check` on one H200, against
cuBLAS known issue 6862950 (silently wrong FP16/BF16/FP8 GEMMs under MPS). What runs and how each result is
classified: experiments/h2h_run.py and docs/METHOD.md ("Head-to-head on a real cuBLAS bug"), written before the run.

    modal run scripts/modal_h2h.py --phase build       # CPU only, a few cents: builds the image, checks DCGM,
                                                        # gpu-burn and the probe are there and the probe's CPU self-check
    modal run scripts/modal_h2h.py --phase preflight   # one H200, about 10 minutes: can the full run answer its
                                                        # question? (MPS, the bug as published, DCGM under MPS)
    modal run scripts/modal_h2h.py --phase run         # one H200, about 60 minutes

Cost at Modal's prices (Oct 20, 2026: H200 $4.54 an hour, CPU $0.0472 a core-hour, memory $0.008 a GiB-hour, so
$4.92 an hour with 4 cores and 24 GiB): the preflight about $1 (its 20-minute limit is about $1.65); the full run
about $5 for a typical run, and its 80-minute execution limit is about $6.60 at the requested CPU and memory. Limits
are per attempt and exclude queueing; actual CPU or memory use above the request, startup and a restart after
preemption can add cost. Image builds may add a few cents. Each run prints its measured function time and the cost
that implies at these rates; Modal's usage page has the billed amount.
Results go to results/h2h/: the summary JSON and Markdown, and every step's full output (downloaded from the
Modal volume sdcguard-h2h, where it also stays).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 80 * 60
PREFLIGHT_LIMIT = 20 * 60
RATE_PER_HOUR = 4.54 + 4 * 0.0472 + 24 * 0.008          # H200 + 4 cores + 24 GiB, Modal prices on Oct 20, 2026
GPU_BURN_COMMIT = "0d19c94dcb2e9b6858b7586a4c327674cd67113b"     # github.com/wilicc/gpu-burn, Oct 2, 2026
DCGM_PACKAGE = "datacenter-gpu-manager-4-cuda13=1:4.7.0-2"         # newest in NVIDIA's ubuntu2204 repo on Oct 20, 2026


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


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "procps", "build-essential", DCGM_PACKAGE)
    .run_commands(
        "git clone https://github.com/wilicc/gpu-burn /opt/gpu-burn",
        f"cd /opt/gpu-burn && git checkout {GPU_BURN_COMMIT} "
        "&& make COMPUTE= NVCCFLAGS='-gencode=arch=compute_90,code=sm_90'",
    )
    .uv_pip_install("torch==2.14.1", "numpy")
    .add_local_file(LOCAL / "experiments" / "h2h_probe.cpp", "/opt/h2h/h2h_probe.cpp", copy=True)
    .run_commands(
        "cd /opt/h2h && g++ -O2 -std=c++17 h2h_probe.cpp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 "
        "-lcublasLt -lcublas -lcudart -o probe && ./probe cpu:hgemm:256:hole cpu:exbf16:128",
    )
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
)

app = modal.App("sdcguard-h2h", image=image)
vol = modal.Volume.from_name("sdcguard-h2h", create_if_missing=True)


@app.function(cpu=2, memory=4096, timeout=10 * 60)
def build_check() -> dict:
    """No GPU: are DCGM, gpu-burn, the probe, torch and sdcguard all present in the image?"""
    import subprocess

    def sh(cmd):
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return {"cmd": cmd, "exit": p.returncode, "output": (p.stdout + p.stderr)[-2500:]}
    checks = [
        sh("dcgmi --version"),
        sh("ls /usr/libexec/datacenter-gpu-manager-4/ | head -20"),
        sh("dpkg -l | grep -i datacenter-gpu-manager | awk '{print $2, $3}'"),
        sh("ls -la /opt/gpu-burn/gpu_burn /opt/gpu-burn/compare.fatbin && ldd /opt/gpu-burn/gpu_burn | grep -i cublas"),
        sh("ldd /opt/h2h/probe | grep -i -E 'cublas|cudart' && /opt/h2h/probe cpu:hgemm:256:hole cpu:sgemmtc:128"),
        sh("ldd /opt/h2h/probe | grep -c cublasLt"),
        sh("which strings pgrep getent && (find /usr -name 'libdcgm_cublas_proxy*' 2>/dev/null | head -5)"),
        sh("which nvidia-cuda-mps-control || echo 'nvidia-cuda-mps-control is not in the image (Modal adds it on GPU "
           "machines; the Oct 19 run found it there)'"),
        sh(f"cd {R} && python -c \"import torch, sdcguard, h2h_run; print(torch.__version__, torch.version.cuda, "
           "sdcguard.__file__)\""),
    ]
    return {"checks": checks}


def _timed(fn, run_id: str, limit: int) -> dict:
    t0 = time.monotonic()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        res = fn(f"/out/{run_id}", limit - 240)
    except Exception as e:  # noqa: BLE001
        import traceback
        res = {"error": f"{type(e).__name__}: {str(e)[:500]}", "trace": traceback.format_exc()[-3000:]}
    secs = round(time.monotonic() - t0, 1)
    res.update(run_id=run_id, function_started_utc=started, function_seconds=secs,
               estimated_cost_usd=round(secs / 3600 * RATE_PER_HOUR, 2),
               cost_note=f"function time x ${RATE_PER_HOUR:.2f}/hour (H200, 4 cores, 24 GiB at Oct 20 prices); "
                         "image pull and startup are billed too and not in this number; Modal's usage page has the "
                         "billed amount")
    try:
        vol.commit()
    except Exception:  # noqa: BLE001
        pass
    return res


@app.function(gpu="H200", cpu=4, memory=24576, timeout=LIMIT, volumes={"/out": vol})
def h200(run_id: str) -> dict:
    import h2h_run

    def go(out, deadline):
        res = h2h_run.run_all(out, deadline_s=deadline, commit=vol.commit, repo=R)
        res["markdown"] = h2h_run.markdown(res)
        return res
    return _timed(go, run_id, LIMIT)


@app.function(gpu="H200", cpu=4, memory=24576, timeout=PREFLIGHT_LIMIT, volumes={"/out": vol})
def h200_preflight(run_id: str) -> dict:
    import h2h_run
    return _timed(lambda out, deadline: h2h_run.preflight(out, deadline_s=deadline, commit=vol.commit, repo=R),
                  run_id, PREFLIGHT_LIMIT)


def _download(run_id: str, dest: Path) -> str:
    """Every file the run wrote on the volume, to dest (the raw logs stay on the volume too)."""
    try:
        entries = vol.listdir(f"/{run_id}", recursive=True)
    except Exception as e:  # noqa: BLE001
        return (f"could not list the volume ({type(e).__name__}: {e}); get the logs with: modal volume get "
                f"sdcguard-h2h /{run_id} {dest}")
    n = 0
    for e in entries:
        kind = getattr(getattr(e, "type", None), "name", str(getattr(e, "type", "")))
        if kind != "FILE":                 # (FileEntryType is an IntEnum: str() of it is just "2")
            continue
        rel = str(e.path).lstrip("/")
        rel = rel[len(run_id) + 1:] if rel.startswith(run_id + "/") else rel
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(target, "wb") as f:
                vol.read_file_into_fileobj(e.path, f)
            n += 1
        except Exception as ex:  # noqa: BLE001
            return (f"downloaded {n} files, then failed ({type(ex).__name__}: {ex}); the rest: modal volume get "
                    f"sdcguard-h2h /{run_id} {dest}")
    return f"downloaded {n} raw log files to {dest}"


def _wait(call, cap=3 * 3600, every=20.0, retry_for=600.0):
    """Poll one spawned call. A pending poll raises the built-in TimeoutError; platform terminal errors end it."""
    start = last = time.monotonic()
    failing_since = None
    final = {"FunctionTimeoutError", "OutputExpiredError", "RemoteError", "ExecutionError", "InputCancellation"}
    while True:
        try:
            return call.get(timeout=0)
        except Exception as e:  # noqa: BLE001
            now, kind = time.monotonic(), type(e).__name__
            if now - start >= cap:
                return {"error": f"{kind}: still not finished after {cap / 3600:.1f} hours"}
            if kind in final:
                return {"error": f"{kind}: {e}"}
            if kind != "TimeoutError":
                failing_since = failing_since or now
                if now - failing_since >= retry_for:
                    return {"error": f"{kind}: {e} (for {retry_for / 60:.0f} minutes)"}
            else:
                failing_since = None
        if now - last >= 300:
            print(f"{(now - start) / 60:.0f} minutes: still running (a run takes about an hour)", flush=True)
            last = now
        time.sleep(every)


@app.local_entrypoint()
def main(phase: str = "build") -> None:
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    if phase == "build":
        print("Building the image and checking it on a CPU-only machine (a few cents).", flush=True)
        res = build_check.remote()
        bad = 0
        for c in res["checks"]:
            ok = c["exit"] == 0
            bad += not ok
            print(f"\n[{'ok' if ok else 'FAILED'}] {c['cmd']}\n{c['output'].strip()}")
        print("\nBuild check passed. Done." if not bad else f"\n{bad} check(s) failed: don't start the GPU run. Done.")
        if bad:
            raise SystemExit(1)
        return
    if phase == "preflight":
        run_id = time.strftime("preflight_h200_%Y%m%d_%H%M%S")
        print("Starting one H200 for the preflight (about 10 minutes, at most 20). Keep this terminal open until it "
              "prints Done.", flush=True)
        res = _wait(h200_preflight.spawn(run_id), cap=PREFLIGHT_LIMIT + 1800)
        res["source"] = src
        dest = LOCAL / "results" / "h2h"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{run_id}.json").write_text(json.dumps(res, indent=1, default=str))
        print(_download(run_id, dest / f"{run_id}_logs"), flush=True)
        print(f"\nPreflight, code fingerprint {src['sha256'][:12]}:")
        for k, v in (res.get("decisions") or {}).items():
            print(f"  [{'yes' if v is True else 'NO' if v is False else 'not measured'}] {k}")
        print(f"\nFunction time {res.get('function_seconds')} s, about ${res.get('estimated_cost_usd')} at the rates "
              f"above ({res.get('cost_note', '')}).")
        print(f"Saved {dest / (run_id + '.json')}.", flush=True)
        if res.get("error") or not (res.get("decisions") or {}).get("go: the full head-to-head can answer its question"):
            print(f"Preflight says NO GO{': ' + res['error'] if res.get('error') else ''}. Send these results "
                  "before running the full head-to-head. Done.")
            raise SystemExit(1)
        print("Preflight says GO for the full head-to-head. Done.")
        return
    if phase != "run":
        raise SystemExit("--phase must be build, preflight or run")
    run_id = time.strftime("h200_%Y%m%d_%H%M%S")
    print("Starting one H200 for the head-to-head (about 60 minutes, at most 80). Keep this terminal open until it "
          "prints Done.", flush=True)
    res = _wait(h200.spawn(run_id))
    res["source"] = src
    dest = LOCAL / "results" / "h2h"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{run_id}.json"
    path.write_text(json.dumps(res, indent=1, default=str))
    md = res.get("markdown") or ""
    head = (f"# Head-to-head on cuBLAS known issue 6862950, H200, code fingerprint {src['sha256'][:12]}\n\n"
            "Rules and predictions were written before the run (docs/METHOD.md). Each cell is judged against that "
            "tool's own GEMMs at that share (experiments/h2h_probe.cpp).\n\n")
    notes = "\n".join(f"- {n}" for n in res.get("notes", []))
    (dest / f"{run_id}.md").write_text(head + md + ("\n\nNotes:\n" + notes if notes else "") + "\n")
    print(head + md)
    if notes:
        print("\nNotes:\n" + notes)
    print(_download(run_id, dest / f"{run_id}_logs"), flush=True)
    print(f"\nSaved {path}. Function time {res.get('function_seconds')} s, about ${res.get('estimated_cost_usd')} "
          f"({res.get('cost_note', '')}).", flush=True)
    planned = {(s, t) for s in (100, 40, 75, 14) for t in ("dcgm", "gpu_burn_tc", "sdcguard_check")}
    measured = {(c["share"], c["tool"]) for c in res.get("cells", []) if "flag" in c and c.get("cell") != "COULD NOT TEST"}
    missing = sorted(planned - measured)
    if res.get("error"):
        print(f"The run reported an error: {res['error']}. Done.")
        raise SystemExit(1)
    if missing:
        print(f"INCOMPLETE: {len(missing)} of {len(planned)} planned cells have no measured result "
              f"({', '.join(f'{t} at {s}%' for s, t in missing)}). They are COULD NOT TEST, not passes or misses. "
              "Done.")
        raise SystemExit(1)
    print("All 12 planned cells measured. Done.")
