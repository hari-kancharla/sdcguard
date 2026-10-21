"""The two experimental checks on the `next` branch, on an H100 and an L4, in one container per GPU:

  - the fused precise check (experiments/fused_check.py): correctness on the real GPU, clean training steps
    scored by the fused check and by precise mode on the same matmuls, sensitivity, the Triton kernel's speed
    against cuBLAS, the cost per training step, and gradients against precise=True;
  - the recompute check (experiments/recompute_check.py): Hugging Face GPT-2 and Llama configurations with
    gradient checkpointing, clean steps, injected faults, gradients with and without the check, coverage, cost,
    and a nondeterministic kernel.

Criteria: docs/METHOD.md ("The fused precise check" and "The recompute check on GPUs"), written before the run.

    modal run scripts/modal_fused.py

About 15 to 20 minutes per GPU, about $2 in total at Modal's prices on Oct 20, 2026 (per hour: H100 $3.95, L4
$0.80, plus about $0.45 per machine for 4 cores and 32 GiB). Every function stops at 40 minutes, so the requested
resources can't cost more than about $3.80; startup and image builds add a little. Each result records its
measured function time and the cost that implies at these rates (Modal's usage page has the billed amount).
Results go to results/fused/, with every line each GPU printed. The command exits with an error unless every
criterion was met on both GPUs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 40 * 60
RATES = {"h100": 3.95 + 4 * 0.0472 + 32 * 0.008, "l4": 0.80 + 4 * 0.0472 + 32 * 0.008}


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
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs", "fixtures") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential")
    .uv_pip_install("torch==2.14.1", "numpy", "transformers==5.20.1")
    .env({"PYTHONPATH": f"{R}/src:{R}/experiments", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
    .add_local_dir(LOCAL / "experiments", f"{R}/experiments", ignore=_skip)
    .add_local_dir(LOCAL / "tests", f"{R}/tests", ignore=_skip)
)

app = modal.App("sdcguard-fused", image=image)
vol = modal.Volume.from_name("sdcguard-fused", create_if_missing=True)
KW = dict(cpu=4, memory=32768, timeout=LIMIT, volumes={"/out": vol})


def _work(label: str, run_id: str) -> dict:
    import torch
    import fused_check
    import recompute_check
    t0 = time.monotonic()
    deadline = t0 + LIMIT - 300          # leave time to save; parts that can't start are recorded as not run
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log: list = []

    def say(s):
        line = f"{time.monotonic() - t0:7.1f}s {s}"
        log.append(line)
        print(label, line, flush=True)
    res = {"label": label, "function_started_utc": started}

    def save():                          # partial results survive a timeout or a crash later on
        import os
        os.makedirs(f"/out/{run_id}", exist_ok=True)
        with open(f"/out/{run_id}/{label}.json", "w") as f:
            json.dump(dict(res, log=log), f, indent=2, default=str)
        vol.commit()
    for name, mod in (("fused", fused_check), ("recompute", recompute_check)):
        try:
            res[name] = mod.run_all(torch.device("cuda"), say=say, deadline=deadline)
            res[name]["markdown"] = mod.markdown(res[name])
        except Exception as e:  # noqa: BLE001
            import traceback
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:],
                         "criteria": {"all met": False}}
        save()
    secs = round(time.monotonic() - t0, 1)
    res.update(function_seconds=secs, estimated_cost_usd=round(secs / 3600 * RATES[label], 2),
               cost_note=f"function time x ${RATES[label]:.2f}/hour ({label}, 4 cores, 32 GiB at Oct 20 prices); "
                         "image pull and startup are billed too and not in this number",
               log=log)
    save()
    return res


@app.function(gpu="H100!", **KW)
def h100(run_id: str) -> dict:
    return _work("h100", run_id)


@app.function(gpu="L4", **KW)
def l4(run_id: str) -> dict:
    return _work("l4", run_id)


@app.local_entrypoint()
def main() -> None:
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    started = [(name, fn.spawn(stamp)) for name, fn in (("h100", h100), ("l4", l4))]
    print("Started 2 GPUs (at most 40 minutes each). Keep this terminal open until it prints Done.", flush=True)
    dest = LOCAL / "results" / "fused"
    dest.mkdir(parents=True, exist_ok=True)
    text, failed, cost = [], [], 0.0
    for name, call in started:
        try:
            r = call.get(timeout=LIMIT + 600)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
            try:                         # what it saved before it stopped
                import io
                buf = io.BytesIO()
                vol.read_file_into_fileobj(f"/{stamp}/{name}.json", buf)
                r = dict(json.loads(buf.getvalue()), error=r["error"], partial=True)
            except Exception:  # noqa: BLE001
                pass
        r["source"] = src
        (dest / f"{name}_{stamp}.json").write_text(json.dumps(r, indent=2, default=str))
        (dest / f"{name}_{stamp}_log.txt").write_text("\n".join(r.get("log") or []) + "\n")
        cost += r.get("estimated_cost_usd") or 0.0
        if r.get("error"):                # stopped early (timeout, crash): never a pass, whatever it saved
            failed.append(f"{name} ({r['error'][:80]})")
        for part in ("fused", "recompute"):
            p = r.get(part) or {}
            text.append(p.get("markdown") or f"## {name} {part}: could not run: {p.get('error') or r.get('error')}\n")
            if not (p.get("criteria") or {}).get("all met"):
                failed.append(f"{name} {part}")
        print(f"{name}: finished in {r.get('function_seconds')} s, about ${r.get('estimated_cost_usd')}", flush=True)
    summary = (f"# Experimental checks, {stamp}\n\nCode fingerprint {src['sha256'][:12]}. Criteria not all met on: "
               f"{', '.join(failed) or 'none'}. Measured function time implies about ${cost:.2f} in total.\n\n"
               + "\n".join(text))
    (dest / f"SUMMARY_{stamp}.md").write_text(summary)
    print(summary)
    print(f"Results in {dest}.", flush=True)
    if failed:
        print(f"NOT ALL CRITERIA MET ({', '.join(failed)}): these checks stay experimental, and the failures are "
              "written up as they are. Done.")
        raise SystemExit(1)
    print("Every criterion met on both GPUs. Done.")
