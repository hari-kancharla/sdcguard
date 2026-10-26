"""A second real NVIDIA library bug: cuBLAS issue 6580689 on an H100.

    modal run scripts/modal_nvbug.py

NVIDIA's CUDA 13.4 release notes list, as a known issue of cuBLAS 13.7.0: "cublasLtMatmul() can return incorrect
output when using split-K if the number of K splits (SPLITK_NUM) does not evenly divide the K dimension and the floor
division of K / SPLITK_NUM is an exact multiple of the stage size", for CUBLASLT_ALGO_CONFIG_ID 66 on compute
capability 9.0, 10.x and 11.x, "first introduced in CUDA Toolkit 12.6 Update 2 (cuBLAS 12.6.3)"; resolved in CUDA 13.4
Update 1 (cuBLAS 13.8.0). The stage size is the first number of the configuration's stages id (64 for 64xAUTO).

One H100, three environments that differ only in the cuBLAS wheel, each with PyTorch 2.14.1: its own pinned cuBLAS
13.1.1.3 (what PyTorch 2.14.1 and vLLM 0.32's PyTorch 2.13.0 install by default, inside the affected range), 13.7.0.27
(the release the notes name) and 13.8.1.7 (fixed). In each, `sdcguard blas --splitk 9,11,13,15 --workspace-mib 256` on
eight shapes whose K meets the condition for one of the counts: K = 768 * s + 8 for s in 9, 11, 13 and 15 (6920, 8456,
9992 and 11528: K // s = 768, a multiple of every stage size cuBLASLt defines, and K % s = 8), each as a prefill-like
4096xKx256 and a decode-like 1024xKx16, in bf16 and fp16. The first 4 algorithm-66 configurations the heuristic offers
are forced to each count with each reduction scheme cuBLASLt accepts; the counts that don't meet a shape's condition
are controls. Each forced run records its stages id, whether it meets the condition, and whether the split shows
(output bits unlike the unforced configuration's, or a reduction that needs workspace). Then, without forcing and
with the default 32 MiB workspace, six linear-layer shapes of real models (down projections of Qwen2.5-1.5B and 7B,
Llama 3 8B and 70B, at 1 and 16 tokens), to see whether the heuristic picks a configuration in the condition on its
own. Each environment has 7 minutes for its sweeps and 2 for its version probe.

Expectations written before the run (October 26, 2026; revised the same day after an outside review, before any run:
the first design's shapes, K = 4096 and 11008 with counts 3 to 9, never met the condition):
  N1  with cuBLAS 13.7.0, at least one forced algorithm-66 run in the issue's condition, with evidence the split was
      applied (output bits that differ from the unforced configuration, or a reduction that needs workspace),
      computes wrong (reproduced).
  N2  with cuBLAS 13.8.1, every run computes right: forced (in the condition or not), offered by the heuristic, and
      the cluster workaround (fixed).
  N3  each of those two environments loads the cuBLAS it should (cublasLtGetVersion 130700 and 130801), runs on an H100
      (compute capability 9.0), and tests at least 8 forced runs in the condition with that evidence; otherwise the
      run says nothing.
Reported, not criteria: whether the default cuBLAS (13.1.1) computes wrong in the condition too; forced runs outside
the condition that compute wrong (the condition would not be the whole story); configurations the heuristic offers on
its own in the condition, and whether they compute wrong (whether a plain library call can hit it); for each wrong run
with 13.7.0, whether the same configuration ran right with 13.8.1; refused and not-run counts. If N1 isn't met, the
result is recorded as not reproduced, as it stands.

One H100 for at most 30 minutes: at most about $2.10 at Modal's prices on Oct 20, 2026; usually 10 to 20 minutes.
Results go to results/nvbug_6580689/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

LOCAL = Path(__file__).resolve().parent.parent
R = "/root/sdcguard"
LIMIT = 30 * 60
ENV_S = 420                       # each environment's sweeps
VERSIONS_S = 120                  # each environment's version probe (the function's limit leaves room for all three)
# name -> (venv, cuBLAS wheel installed over PyTorch's own or None for its own, cublasLtGetVersion expected)
ENVS = {"default": ("/opt/cdef", None, 130101), "affected": ("/opt/c137", "13.7.0.27", 130700),
        "fixed": ("/opt/c138", "13.8.1.7", 130801)}
JUDGED = ("affected", "fixed")
SPLITS = (9, 11, 13, 15)
MIN_IN_CONDITION = 8


def k_for(s: int) -> int:
    """K // s = 768 (a multiple of every stage size: 8 to 256, and 768) and K % s = 8 (not a divisor; K stays a
    multiple of 8, so 16-byte aligned rows in fp16 and bf16)."""
    return 768 * s + 8


FORCED_SHAPES = tuple(f"{m}x{k_for(s)}x{n}" for s in SPLITS for m, n in ((4096, 256), (1024, 16)))
MODEL_SHAPES = ("1536x8960x16", "3584x18944x1", "3584x18944x16", "4096x14336x1", "4096x14336x16", "8192x28672x16")


def _skip(p: Path) -> bool:
    return any(part in ("__pycache__", ".pytest_cache", "runs", "data", "results", "private", "docs") or
               part.startswith("._") or part.endswith(".egg-info") for part in p.parts)


def _source() -> dict:
    import hashlib
    h, n = hashlib.sha256(), 0
    for d in ("src", "scripts"):
        for p in sorted((LOCAL / d).rglob("*.py")):
            if "__pycache__" in p.parts or p.name.startswith("._"):
                continue
            h.update(str(p.relative_to(LOCAL)).encode() + b"\0" + p.read_bytes() + b"\0")
            n += 1
    return {"sha256": h.hexdigest(), "files": n}


def _installs() -> list:
    cmds = ["pip install uv"]
    for path, ver, _ in ENVS.values():
        cmds += [f"uv venv --python 3.12 {path}", f"uv pip install --python {path}/bin/python torch==2.14.1 numpy"]
        if ver:
            cmds.append(f"uv pip install --python {path}/bin/python --no-deps nvidia-cublas=={ver}")
    return cmds


image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(*_installs())
    .env({"PYTHONPATH": f"{R}/src", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(LOCAL / "src", f"{R}/src", ignore=_skip)
)

app = modal.App("sdcguard-nvbug", image=image)


def _versions(python) -> str:
    code = ("import importlib.metadata as m, torch; from sdcguard import blas; lib = blas._library(); "
            "print(torch.__version__, torch.version.cuda, lib.cublasLtGetVersion() if lib else None, "
            "[(d.metadata['Name'], d.version) for d in m.distributions() if 'cublas' in (d.metadata['Name'] or '')])")
    try:
        p = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=VERSIONS_S)
    except subprocess.TimeoutExpired:
        return f"the version probe timed out after {VERSIONS_S} s"
    return (p.stdout + p.stderr).strip()[-600:]


def _forced(sweeps: list) -> list:
    return [dict(a, shape=x.get("shape"), dtype=x.get("dtype")) for x in sweeps for a in x.get("algorithms") or []
            if str(a.get("variant") or "").startswith("splitk_")]


def _ran(a: dict) -> bool:
    return not a.get("status") and a.get("correct") is not None


def _key(a: dict) -> tuple:
    return (tuple(a.get("shape") or ()), a.get("dtype"), a.get("tile_id"), a.get("stages_id"),
            a.get("cluster_shape_id"), a.get("requested_splitk"), a.get("reduction_scheme"))


def summarize_env(sweeps: list) -> dict:
    forced = _forced(sweeps)
    ran = [a for a in forced if _ran(a)]
    cond = [a for a in ran if a.get("split_applied") and a.get("issue_6580689_condition")]
    outside = [a for a in ran if a.get("issue_6580689_condition") is False]
    plain = [a for x in sweeps for a in x.get("algorithms") or [] if not a.get("variant")]
    return {"cublaslt_version": sorted({x.get("cublaslt_version") for x in sweeps} - {None}),
            "gpus": sorted({x.get("gpu") for x in sweeps if x.get("gpu")}),
            "compute_capability": sorted({tuple(x.get("compute_capability") or ()) for x in sweeps}),
            "sweeps": len(sweeps), "errors": [x.get("error") for x in sweeps if x.get("error")],
            "algo66_offered": sum(int(x.get("algo66_offered") or 0) for x in sweeps),
            "forced_ran": len(ran), "forced_in_condition": len(cond),
            "forced_wrong_in_condition": sum(1 for a in cond if not a.get("correct")),
            "forced_wrong_outside_condition": sum(1 for a in outside if not a.get("correct")),
            "forced_without_split_evidence": sum(1 for a in ran if not a.get("split_applied")),
            "forced_refused": sum(len(x.get("splitk_refused") or []) for x in sweeps),
            "forced_not_run": sum(len(x.get("splitk_not_run") or []) for x in sweeps),
            "heuristic_wrong": sum(len(x.get("wrong_algorithms") or []) for x in sweeps),
            "workaround_wrong": sum(len(x.get("workaround_wrong") or []) for x in sweeps),
            "heuristic_in_condition": sum(len(x.get("heuristic_in_condition") or []) for x in sweeps),
            "heuristic_in_condition_wrong": sum(1 for x in sweeps for h in x.get("heuristic_in_condition") or []
                                                if h.get("correct") is False),
            "plain_ran": sum(1 for a in plain if _ran(a))}


def judge(res: dict) -> tuple:
    """(met, failures, summary) for N1 to N3 from the environments' sweep results."""
    fails, s = [], {}
    for name, (_path, _ver, want) in ENVS.items():
        r = res.get(name) or {}
        s[name] = summarize_env(r.get("sweeps") or [])
        if name not in JUDGED:
            continue
        e = s[name]
        ok = (r.get("sweeps") and e["cublaslt_version"] == [want] and e["gpus"] and
              all("H100" in g for g in e["gpus"]) and e["compute_capability"] == [(9, 0)] and
              e["forced_in_condition"] >= MIN_IN_CONDITION)
        if not ok:
            fails.append(f"N3 ({name}): cuBLASLt {e['cublaslt_version']} (want {want}), GPUs {e['gpus']} "
                         f"(capability {e['compute_capability']}, driver {res.get('nvidia_smi')}), "
                         f"{e['forced_in_condition']} forced runs in the condition (want {MIN_IN_CONDITION}), "
                         f"{e['algo66_offered']} algorithm-66 configurations offered, {e['forced_refused']} forced "
                         f"runs refused, {e['forced_not_run']} not run, errors {e['errors'][:2]}")
    a, f = s.get("affected") or {}, s.get("fixed") or {}
    if not a.get("forced_wrong_in_condition"):
        fails.append(f"N1: with cuBLAS 13.7.0, none of {a.get('forced_in_condition')} forced runs in the condition "
                     "computed wrong (not reproduced)")
    wrong_fixed = (f.get("forced_wrong_in_condition", 0) + f.get("forced_wrong_outside_condition", 0) +
                   sum(1 for x in _forced((res.get("fixed") or {}).get("sweeps") or [])
                       if _ran(x) and not x.get("correct") and x.get("issue_6580689_condition") is None) +
                   f.get("heuristic_wrong", 0) + f.get("workaround_wrong", 0))
    if wrong_fixed or not f.get("plain_ran"):
        fails.append(f"N2: with cuBLAS 13.8.1, {wrong_fixed} runs computed wrong (forced in the condition "
                     f"{f.get('forced_wrong_in_condition')}, outside it {f.get('forced_wrong_outside_condition')}, "
                     f"heuristic {f.get('heuristic_wrong')}, workaround {f.get('workaround_wrong')}), "
                     f"{f.get('plain_ran')} heuristic runs")
    # reported: each wrong run with 13.7.0 against the same configuration with 13.8.1
    fixed_runs = {_key(x): x for x in _forced((res.get("fixed") or {}).get("sweeps") or []) if _ran(x)}
    pairs = {"right_with_13_8_1": 0, "wrong_with_13_8_1": 0, "not_run_with_13_8_1": 0}
    for x in _forced((res.get("affected") or {}).get("sweeps") or []):
        if _ran(x) and not x.get("correct"):
            y = fixed_runs.get(_key(x))
            pairs["not_run_with_13_8_1" if y is None else
                  "right_with_13_8_1" if y.get("correct") else "wrong_with_13_8_1"] += 1
    s["pairing"] = pairs
    d = s.get("default") or {}
    s["default_affected"] = (None if not d.get("forced_in_condition") else
                             bool(d.get("forced_wrong_in_condition")))
    return not fails, fails, s


@app.function(gpu="H100", cpu=4, memory=32768, timeout=LIMIT)
def h100() -> dict:
    t0 = time.time()
    out = {"nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()}
    for name, (path, _ver, _want) in ENVS.items():
        py = f"{path}/bin/python"
        res = {"stack": _versions(py), "sweeps": [], "runs": []}
        end = time.time() + ENV_S
        # the forced runs with a workspace large enough for split-K reductions in the compute type; the model shapes
        # with sdcguard blas's default 32 MiB, near what a plain library call gets
        for tag, shapes, extra in (("forced", FORCED_SHAPES, ["--splitk", ",".join(map(str, SPLITS)),
                                                              "--workspace-mib", "256"]),
                                   ("models", MODEL_SHAPES, [])):
            left = end - time.time()
            if left < 30:
                res["runs"].append({"part": tag, "skipped": "the environment's time ran out"})
                continue
            js = f"/tmp/{name}_{tag}.json"
            cmd = [py, "-m", "sdcguard", "blas", "--dtype", "both", "--json", js, *extra]
            for sh in shapes:
                cmd += ["--shape", sh]
            run = {"part": tag}
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=left, cwd="/tmp")
                run.update(exit=p.returncode, output=(p.stdout + p.stderr)[-3000:])
            except subprocess.TimeoutExpired:
                run["error"] = f"timed out after {round(left)} s"
            try:
                res["sweeps"] += json.loads(Path(js).read_text())
            except (OSError, ValueError) as e:
                run["sweeps_missing"] = str(e)
            res["runs"].append(run)
        out[name] = res
    out["seconds"] = round(time.time() - t0, 1)
    out["cost_estimate_usd"] = round(out["seconds"] / 3600 * (3.95 + 4 * 0.0472 + 32 * 0.008), 2)
    return out


@app.local_entrypoint()
def main() -> None:
    print("Starting one H100 (at most 30 minutes). Keep this terminal open until it prints Done.", flush=True)
    src = _source()
    print(f"Code fingerprint {src['sha256'][:12]} ({src['files']} files)", flush=True)
    res = h100.remote()
    res["source"] = src
    met, fails, s = judge(res)
    res.update(met=met, failures=fails, summary=s)
    d = LOCAL / "results" / "nvbug_6580689"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"h100_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(res, indent=2, default=str))
    print(res.get("nvidia_smi"))
    for name in ENVS:
        r = res.get(name) or {}
        print(f"\n{name}: {r.get('stack')}")
        for run in r.get("runs") or []:
            print(f"  [{run.get('part')}] {(run.get('output') or run.get('error') or run.get('skipped') or '')[-1200:]}")
    print("\n" + json.dumps({"met": met, "failures": fails, "summary": s}, indent=1, default=str))
    print(f"\nSaved {path}. Done. Estimated cost: ${res.get('cost_estimate_usd')}")
    sys.exit(0 if met else 1)
