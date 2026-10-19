"""The default check against precise mode on the same training matmuls, on one revision
(experiments/mode_compare.py): how small an injected error each flags, and the cost per training step of
each, on an H100, an A100 and an L4 at once (the A100 and the L4 are where precise mode missed its held-out
criteria on Oct 19).

    modal run scripts/modal_mode_compare.py

About 5 to 15 minutes per GPU, about $1.50 in total at Modal's prices on Oct 18, 2026 (per hour: H100 $3.95,
A100 80GB $2.50, L4 $0.80, plus about $0.38 per machine for CPU and memory). Every function stops at 30 minutes,
so it can't cost more than about $4.20. There is no pass line (the reporting rules are in the experiment's
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
    import platform

    import torch
    import mode_compare
    import precise_models
    t0 = time.time()
    dev = torch.device("cuda")
    try:
        cfg = dict(mode_compare.CONFIG["cuda"])
        res = mode_compare.run(dev, cfg, say=lambda s: print(label, s, flush=True))
        res.update(device="cuda", torch=torch.__version__, code_sha256=mode_compare.code_digest(),
                   gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)),
                   python=platform.python_version())
        res["overhead"] = precise_models.overhead(dev, cfg)
    except Exception as e:  # noqa: BLE001
        import traceback
        return {"label": label, "error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:]}
    res.update(label=label, seconds=round(time.time() - t0, 1), markdown=mode_compare.markdown(res))
    return res


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
    for name, call in started:
        try:
            r = call.get(timeout=LIMIT + 300)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["source"] = src
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
