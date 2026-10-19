"""The held-out validation of precise mode's frozen noise model (experiments/precise_models.py --validate) on
five GPUs at once: H100, B200, A100, L4 (bf16) and T4 (fp16). Three workloads none of the candidate models
was chosen on, the pre-registered criteria (0 alarms, worst score at most 4), and the cost per training step
of the default check and of precise mode.

    modal run scripts/modal_precise_validate.py

About 5 minutes on the H100, B200 and A100 and up to 25 on the L4 and T4 (slow float64), about $2.50 in total
at Modal's prices on Oct 18, 2026 (per hour: B200 $6.25, H100 $3.95, A100 80GB $2.50, L4 $0.80, T4 $0.59, plus
about $0.38 per machine for CPU and memory). Every function stops at 30 minutes, so it can't cost more than
about $8. Results go to results/precise_validate/.
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

app = modal.App("sdcguard-precise-validate", image=image)
KW = dict(cpu=4, memory=32768, timeout=LIMIT)


def _work(label: str) -> dict:
    import torch
    import precise_models
    t0 = time.time()
    try:
        res = precise_models.validate(torch.device("cuda"), say=lambda s: print(label, s, flush=True))
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:]}
    res.update(label=label, seconds=round(time.time() - t0, 1), markdown=precise_models.validate_markdown(res))
    return res


@app.function(gpu="H100!", **KW)
def h100() -> dict:
    return _work("h100")


@app.function(gpu="B200", **KW)
def b200() -> dict:
    return _work("b200")


@app.function(gpu="A100-80GB", **KW)
def a100() -> dict:
    return _work("a100_80gb")


@app.function(gpu="L4", **KW)
def l4() -> dict:
    return _work("l4")


@app.function(gpu="T4", **KW)
def t4() -> dict:
    return _work("t4")


@app.local_entrypoint()
def main() -> None:
    jobs = [("h100", h100), ("b200", b200), ("a100_80gb", a100), ("l4", l4), ("t4", t4)]
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    started = [(name, fn.spawn()) for name, fn in jobs]
    print(f"Started {len(started)} GPUs (at most 30 minutes each). Keep this terminal open until it prints Done.",
          flush=True)
    dest = LOCAL / "results" / "precise_validate"
    dest.mkdir(parents=True, exist_ok=True)
    text = []
    for name, call in started:
        try:
            r = call.get(timeout=LIMIT + 300)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["source"] = src
        (dest / f"{name}.json").write_text(json.dumps(r, indent=2, default=str))
        text.append(r.get("markdown") or f"## {name}: could not run: {r.get('error')}\n")
        print(f"{name}: finished", flush=True)
    met = [n for n in ("h100", "b200", "a100_80gb", "l4", "t4")
           if (json.loads((dest / f"{n}.json").read_text()).get("as_shipped") or {}).get("criteria_met")]
    head = (f"# Precise mode, held-out validation\n\nCriteria (pre-registered): 0 alarms and worst score at most 4 "
            f"on every GPU. Met on {len(met)} of 5: {', '.join(met) or 'none'}.\n\n")
    (dest / "SUMMARY.md").write_text(head + "\n".join(text))
    print(head + "\n".join(text))
    print(f"Done. Results in {dest}", flush=True)
    if len(met) < len(jobs):        # the exit code says whether the pre-registered criteria were met everywhere
        raise SystemExit(1)
