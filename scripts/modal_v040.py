"""0.4.0 on GPUs: the new checks, against criteria written in docs/METHOD.md before the run ("0.4.0 on GPUs").

    modal run scripts/modal_v040.py

Three functions in parallel:
  - one H100: the fused check with its scoring in the kernel (experiments/fused_check.py), the recompute check
    with the Triton fingerprint kernel (experiments/recompute_check.py), and experiments/v040_check.py
    (fingerprint kernel bit for bit, torch.compile fullgraph, MoE grouped GEMMs, original-event capture in a
    CUDA graph, cuBLASLt sweep);
  - two H100s in one machine: tensor parallel over NCCL (experiments/v040_tp.py, with torchrun);
  - one H200 under MPS: `sdcguard blas` and the known-issue report at 100% and 40% of the SMs (issue 6862950).

Costs at Modal's prices on Oct 20, 2026 (per hour: H100 $3.95, H200 $4.54, plus 4 cores at $0.0472 and memory at
$0.008 a GiB: about $0.45 with 32 GiB, $0.70 with the pair's 64 GiB): the H100 runs up to 60 minutes (at most
about $4.40, usually 35 to 45 minutes, about $3), the pair up to 20 minutes (at most about $2.90), the H200 up to
15 minutes (at most about $1.25). So at most about $8.50, usually about $5. Startup and image builds add a little.
Results go to results/v040/; the command exits with an error unless every criterion was met.

Each function fingerprints the code it actually has (as scripts/modal_check.py does) and stops before any GPU
work if that isn't the code the launcher fingerprinted; the launcher also records the git commit it ran from and
whether the tree had changes (or that it isn't a git checkout).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
# GPU, plus 4 cores at $0.0472 and memory at $0.008 a GiB an hour (32 GiB, the pair 64)
RATES = {"h100": 3.95 + 4 * 0.0472 + 32 * 0.008, "h100x2": 2 * 3.95 + 4 * 0.0472 + 64 * 0.008,
         "h200": 4.54 + 4 * 0.0472 + 32 * 0.008}
LIMITS = {"h100": 60 * 60, "h100x2": 20 * 60, "h200": 15 * 60}


def _digest(root: Path) -> dict:
    """Fingerprint of the code under root (src, experiments, tests, scripts), the same one modal_check.py uses."""
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "experiments", "tests", "scripts"):
        for p in sorted((root / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    return {"sha256": h.hexdigest(), "files": n}


def _git(root: Path) -> dict:
    """The commit the launcher's tree is at and whether it has uncommitted changes. A tree that isn't a git
    checkout of its own (an extracted source archive) is recorded as such, never a reason to stop."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != root.resolve():
            return {"git": "not a git checkout"}
        head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError) as e:
        return {"git": f"not a git checkout (git didn't run: {type(e).__name__})"}
    return {"git_head": head.stdout.strip() if head.returncode == 0 else None,
            "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None}


def _source() -> dict:
    """What the launcher uploads: its code fingerprint and git state (on the user's machine)."""
    return {**_digest(LOCAL), **_git(LOCAL)}


def _on_gpu(expected: str) -> dict:
    """The fingerprint of the code this function actually has (its copy in the image), against the launcher's."""
    d = _digest(Path(R))
    return {**d, "matches_launcher": d["sha256"] == expected}


def _mismatch(src: dict, expected: str) -> str:
    return (f"the code here (fingerprint {src['sha256'][:12]}, {src['files']} files) isn't the code the launcher "
            f"fingerprinted ({expected[:12]}): nothing was run")


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
    .add_local_dir(LOCAL / "scripts", f"{R}/scripts", ignore=_skip)        # (part of the fingerprint)
)
app = modal.App("sdcguard-v040", image=image)
vol = modal.Volume.from_name("sdcguard-v040", create_if_missing=True)


def _save(run_id: str, label: str, res: dict) -> None:
    os.makedirs(f"/out/{run_id}", exist_ok=True)
    with open(f"/out/{run_id}/{label}.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    vol.commit()


@app.function(gpu="H100!", cpu=4, memory=32768, timeout=LIMITS["h100"], volumes={"/out": vol})
def h100(run_id: str, expected: str) -> dict:
    src = _on_gpu(expected)
    if not src["matches_launcher"]:
        return {"label": "h100", "source_on_gpu": src, "stopped": _mismatch(src, expected)}
    import torch
    import fused_check
    import recompute_check
    import recompute_grad_control
    import v040_check
    t0 = time.monotonic()
    deadline = t0 + LIMITS["h100"] - 300
    log: list = []

    def say(s):
        line = f"{time.monotonic() - t0:7.1f}s {s}"
        log.append(line)
        print("h100", line, flush=True)
    res = {"label": "h100", "source_on_gpu": src, "log": log}
    dev = torch.device("cuda")
    for name, fn in (("v040", lambda: v040_check.run_all(dev, say=say)),
                     ("fused", lambda: fused_check.run_all(dev, say=say, deadline=deadline)),
                     ("recompute", lambda: recompute_check.run_all(dev, say=say, deadline=deadline)),
                     ("grad_control", lambda: recompute_grad_control.run_all(dev, say=say))):
        try:
            res[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            res[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-2000:],
                         "criteria": {"all met": False}}
        _save(run_id, "h100", res)
    res["function_seconds"] = round(time.monotonic() - t0, 1)
    _save(run_id, "h100", res)
    return res


@app.function(gpu="H100!:2", cpu=4, memory=65536, timeout=LIMITS["h100x2"], volumes={"/out": vol})
def h100x2(run_id: str, expected: str) -> dict:
    src = _on_gpu(expected)
    if not src["matches_launcher"]:
        return {"label": "h100x2", "source_on_gpu": src, "stopped": _mismatch(src, expected)}
    t0 = time.monotonic()
    p = subprocess.run(["torchrun", "--nproc_per_node", "2", f"{R}/experiments/v040_tp.py", "--out", "/tmp/tp"],
                       capture_output=True, text=True, timeout=LIMITS["h100x2"] - 120, cwd=R)
    # its own lines (one JSON line per mode and rank) kept whole, and the end of the error output: on October 21 the
    # two together were cut to their last 12,000 characters, which lost which mode was running when the job died
    res = {"label": "h100x2", "source_on_gpu": src, "exit": p.returncode, "output": p.stdout[-60000:],
           "errors": p.stderr[-12000:]}
    try:
        res["tp"] = json.loads(Path("/tmp/tp/v040_tp.json").read_text())
    except (OSError, ValueError) as e:
        res["tp"] = {"error": f"no result file ({e})", "all met": False}
    res["function_seconds"] = round(time.monotonic() - t0, 1)
    _save(run_id, "h100x2", res)
    return res


@app.function(gpu="H200", cpu=4, memory=32768, timeout=LIMITS["h200"], volumes={"/out": vol})
def h200(run_id: str, expected: str) -> dict:
    src = _on_gpu(expected)
    if not src["matches_launcher"]:
        return {"label": "h200", "source_on_gpu": src, "stopped": _mismatch(src, expected)}
    t0 = time.monotonic()
    res = {"label": "h200", "source_on_gpu": src, "runs": []}
    mps = shutil.which("nvidia-cuda-mps-control")
    if not mps:
        res["stopped"] = "nvidia-cuda-mps-control isn't in this container"
        return res
    env = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_pipe", CUDA_MPS_LOG_DIRECTORY="/tmp/mps_log")
    os.makedirs(env["CUDA_MPS_PIPE_DIRECTORY"], exist_ok=True)
    os.makedirs(env["CUDA_MPS_LOG_DIRECTORY"], exist_ok=True)
    p = subprocess.run([mps, "-d"], env=env, capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        res["stopped"] = f"the MPS daemon didn't start: {(p.stdout + p.stderr)[-500:]}"
        return res
    try:
        for share in (100, 40):
            e = dict(env, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(share))
            rec = {"share": share}
            out = f"/tmp/blas_{share}.json"
            b = subprocess.run([sys.executable, "-m", "sdcguard", "blas", "--shape", "1000x2048x16", "--shape",
                                "4096x4096x2048", "--json", out], env=e, capture_output=True, text=True,
                               timeout=300, cwd=R)
            rec["blas_exit"] = b.returncode
            rec["blas_output"] = (b.stdout + b.stderr)[-4000:]
            try:
                rec["blas"] = json.loads(Path(out).read_text())
            except (OSError, ValueError) as err:
                rec["blas"] = {"error": str(err)}
            k = subprocess.run([sys.executable, "-m", "sdcguard", "issues", "--json"], env=e, capture_output=True,
                               text=True, timeout=120, cwd=R)
            try:
                rec["known_issues"] = json.loads(k.stdout)
            except ValueError:
                rec["known_issues"] = {"error": (k.stdout + k.stderr)[-2000:]}
            # each wrong algorithm with the listed cuBLAS issues that could explain it, so a library bug NVIDIA
            # already lists is reported as that, apart from sdcguard's own results
            try:
                import v040_check
                rec["wrong_with_listed_issues"] = v040_check.wrong_with_listed_issues(
                    rec["blas"] if isinstance(rec["blas"], list) else [], rec["known_issues"].get("matches"))
            except Exception as err:  # noqa: BLE001
                rec["wrong_with_listed_issues"] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
            res["runs"].append(rec)
            print(f"h200 share {share}%: blas exit {b.returncode}", flush=True)
    finally:
        subprocess.run([mps], input="quit\n", env=env, capture_output=True, text=True, timeout=60)
    res["function_seconds"] = round(time.monotonic() - t0, 1)
    _save(run_id, "h200", res)
    return res


def _recompute_criteria(res: dict) -> dict:
    """The recompute check with the fingerprint kernel (docs/METHOD.md, "0.4.0 on GPUs"): clean, faults caught and
    named, gradients unchanged against a deterministic control, and at most 10% per step with every step checked."""
    rc = (res.get("recompute") or {}).get("criteria") or {}
    gc = (res.get("grad_control") or {}).get("criteria") or {}
    models = [m for m in (res.get("recompute") or {}).get("models", []) if isinstance(m, dict) and "error" not in m]
    costs = []
    for m in models:
        costs.append((m.get("overhead_pct") or {}).get("full check"))
    c4 = bool(costs) and all(isinstance(x, (int, float)) and x <= 10.0 for x in costs)
    out = {"1 clean": rc.get("1 clean: no mismatch on any clean step"),
           "2 faults caught and named": rc.get("2 every injected fault caught and its block named"),
           "3 gradients unchanged (deterministic control)":
               gc.get("3 (follow-up) gradients unchanged by the check, against a deterministic control"),
           "4 at most 10% per step, every step checked": c4, "cost_pct": costs}
    out["all met"] = bool(out["1 clean"] and out["2 faults caught and named"]
                          and out["3 gradients unchanged (deterministic control)"] and c4)
    return out


def _mps_criteria(res: dict) -> dict:
    runs = {r["share"]: r for r in res.get("runs", [])}
    full, part = runs.get(100) or {}, runs.get(40) or {}

    def sweeps(r):
        b = r.get("blas")
        return b if isinstance(b, list) else []

    def wrong(r):
        return [w for s in sweeps(r) for w in (s.get("wrong_algorithms") or [])]

    def workaround_right(r):
        # right on at least one configuration the issue concerns: a 1x1x1 rerun of algorithm 66 whose own run (the
        # plain one with the same index in the same sweep) had a cluster of 4 or more CTAs. A workaround run on
        # any other algorithm says nothing about the workaround.
        v, affected = [], 0
        for s in sweeps(r):
            algos = s.get("algorithms") or []
            plain = {a.get("index"): a for a in algos if not a.get("variant")}
            for a in algos:
                if not a.get("variant"):
                    continue
                v.append(a)
                orig = plain.get(a.get("index")) or {}
                affected += (a.get("algo_id") == 66 and orig.get("algo_id") == 66
                             and (orig.get("cluster_ctas") or 0) >= 4)
        return affected > 0 and all(a.get("correct") is True for a in v)

    def issue(r, iid):
        ki = r.get("known_issues") or {}
        return next((m.get("status") for m in ki.get("matches", []) if m.get("id") == iid), None)
    w40 = wrong(part)
    # an empty or partial sweep isn't evidence that every algorithm was right: both shapes, both dtypes, with
    # algorithms run and every one of them right
    want = {("1000x2048x16", "bfloat16"), ("1000x2048x16", "float16"), ("4096x4096x2048", "bfloat16"),
            ("4096x4096x2048", "float16")}
    seen = {("x".join(str(v) for v in sw.get("shape", [])), sw.get("dtype")) for sw in sweeps(full)
            if sw.get("algorithms")}
    plain = [a for sw in sweeps(full) for a in (sw.get("algorithms") or []) if not a.get("variant")]
    c1 = (full.get("blas_exit") == 0 and want <= seen and bool(plain) and all(a.get("correct") is True for a in plain)
          and not wrong(full))
    c2 = (bool(w40) and all(w.get("algo_id") == 66 and (w.get("cluster_ctas") or 0) >= 4 for w in w40)
          and workaround_right(part))
    c3 = issue(part, "6862950") == "applies" and issue(full, "6862950") == "not active"
    return {"1 at 100% every algorithm right": c1,
            "2 at 40% the wrong ones are algorithm 66 with clusters of 4 or more CTAs, and the 1x1x1 cluster "
            "workaround is right": c2,
            "3 the known-issue report says 6862950 applies at 40% and is not active at 100%": c3,
            "all met": bool(c1 and c2 and c3)}


def _library_lines(label: str, wrong) -> list:
    """Summary lines for wrong cuBLASLt algorithms: those a listed NVIDIA cuBLAS issue accounts for (a library bug
    NVIDIA already lists, not sdcguard's) apart from those none accounts for. They still fail their criterion. The
    configurations of one algorithm on one shape and data type with the same listed issues are one entry: how
    many there were, their cluster sizes and their split-K."""
    if not isinstance(wrong, list) or not wrong:
        return []
    groups: dict = {}
    for w in wrong:
        key = ("x".join(map(str, w.get("shape") or [])), w.get("dtype"), w.get("algo_id"),
               tuple(w.get("listed_issues") or ()))
        groups.setdefault(key, []).append(w)

    def show(key, ws) -> str:
        ctas = sorted({w["cluster_ctas"] for w in ws if w.get("cluster_ctas")})
        splits = sorted({w["splitk"] for w in ws if w.get("splitk") is not None})
        detail = ["clusters of " + " and ".join(map(str, ctas)) + " CTAs"] if ctas else []
        if splits:
            detail.append("no split-K" if splits == [1] else "split-K " + ", ".join(map(str, splits)))
        n = f"{len(ws)} configurations" if len(ws) > 1 else "1 configuration"
        return f"{key[0]} {key[1]} algorithm {key[2]}, {n}" + (f" ({', '.join(detail)})" if detail else "")
    listed = [f"{show(k, ws)}: {', '.join(k[3])}" for k, ws in groups.items() if k[3]]
    other = [show(k, ws) for k, ws in groups.items() if not k[3]]
    out = []
    if listed:
        out.append(f"{label}: wrong cuBLASLt algorithms that a listed NVIDIA cuBLAS issue accounts for: "
                   + "; ".join(listed))
    if other:
        out.append(f"{label}: wrong cuBLASLt algorithms that no listed cuBLAS issue accounts for: " + "; ".join(other))
    return out


def _git_line(src: dict) -> str:
    if src.get("git"):
        return src["git"]
    return f"git {(src.get('git_head') or 'no commit')[:12]}" + (", with uncommitted changes" if src.get("git_dirty")
                                                                 else "")


@app.local_entrypoint()
def main() -> None:
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files), {_git_line(src)}", flush=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    calls = [("h100", h100.spawn(stamp, src["sha256"])), ("h100x2", h100x2.spawn(stamp, src["sha256"])),
             ("h200", h200.spawn(stamp, src["sha256"]))]
    print("Started an H100, two H100s and an H200. Keep this terminal open until it prints Done.", flush=True)
    dest = LOCAL / "results" / "v040"
    dest.mkdir(parents=True, exist_ok=True)
    failed, cost, lines = [], 0.0, []
    for name, call in calls:
        try:
            r = call.get(timeout=LIMITS[name] + 600)
        except Exception as e:  # noqa: BLE001
            r = {"label": name, "error": f"{type(e).__name__}: {str(e)[:300]}"}
        r["source"] = src
        (dest / f"{name}_{stamp}.json").write_text(json.dumps(r, indent=2, default=str))
        secs = r.get("function_seconds") or 0
        cost += secs / 3600 * RATES[name]
        if name == "h100":
            for part in ("v040", "fused"):
                crit = (r.get(part) or {}).get("criteria") or {}
                lines.append(f"h100 {part}: " + "; ".join(f"{k}: {v}" for k, v in crit.items()))
                if not crit.get("all met"):
                    failed.append(f"h100 {part}")
            lines += _library_lines("h100 blas (no MPS)",
                                    ((r.get("v040") or {}).get("blas") or {}).get("wrong_with_listed_issues"))
            crit = _recompute_criteria(r)
            r["recompute_criteria_v040"] = crit
            (dest / f"{name}_{stamp}.json").write_text(json.dumps(r, indent=2, default=str))
            lines.append("h100 recompute: " + "; ".join(f"{k}: {v}" for k, v in crit.items()))
            if not crit.get("all met"):
                failed.append("h100 recompute")
        elif name == "h100x2":
            ok = r.get("exit") == 0 and bool((r.get("tp") or {}).get("all met"))   # a crashed run never passes
            lines.append(f"h100x2 tensor parallel: all met {ok}")
            if not ok:
                failed.append("h100x2 tensor parallel")
        else:
            crit = _mps_criteria(r)
            r["criteria"] = crit
            (dest / f"{name}_{stamp}.json").write_text(json.dumps(r, indent=2, default=str))
            lines.append("h200 MPS: " + "; ".join(f"{k}: {v}" for k, v in crit.items()))
            for run in r.get("runs", []):
                lines += _library_lines(f"h200 at {run.get('share')}%", run.get("wrong_with_listed_issues"))
            if not crit.get("all met"):
                failed.append("h200 MPS")
        if r.get("error") or r.get("stopped"):
            failed.append(f"{name} ({(r.get('error') or r.get('stopped'))[:80]})")
        elif (r.get("source_on_gpu") or {}).get("matches_launcher") is not True:
            failed.append(f"{name} (the code it ran isn't confirmed to be the code launched)")
        print(f"{name}: finished in {secs:.0f} s", flush=True)
    summary = (f"# 0.4.0 on GPUs, {stamp}\n\nCode fingerprint {src['sha256'][:12]} ({_git_line(src)}). Function "
               f"time implies about ${cost:.2f}. Criteria not all met on: {', '.join(failed) or 'none'}.\n\n"
               + "\n".join(f"- {ln}" for ln in lines) + "\n")
    (dest / f"SUMMARY_{stamp}.md").write_text(summary)
    print(summary)
    if failed:
        print("NOT ALL CRITERIA MET: written up as they are. Done.")
        raise SystemExit(1)
    print("Every criterion met. Done.")
