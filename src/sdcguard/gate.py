"""Upgrade gate: keep a matmul that once computed wrong as a test case, and run it again on every new stack.

A wrong result found in production (or by `sdcguard check`) is worth keeping. `sdcguard gate` turns one into a case
(its inputs, its exact call and the conditions it ran under), runs the cases on whatever stack is installed now,
and compares two runs to say what an upgrade changed:

    sdcguard gate capture --shape 1000,2048,16 --bias --pack gates/    # run a linear layer here, keep it as a case
    sdcguard gate add evidence_0.pt --pack gates/                      # a case from saved evidence
    sdcguard gate run gates/ --out before.json                         # every case on this stack
    ... upgrade PyTorch, CUDA, cuBLAS or the driver ...
    sdcguard gate run gates/ --out after.json
    sdcguard gate compare before.json after.json                       # a verdict per case
    sdcguard gate workaround gates/<case> --env KEY=VALUE ...          # does a workaround fix it, here?
    sdcguard gate pin gates/<case> --out pin.csv                       # a TunableOp file that pins the workaround
    sdcguard gate model-ref --model M --out ref.json                   # the model's outputs, on a trusted stack
    sdcguard gate model-check ref.json --out new.json                  # ... and on a new one (gate_model.py)

Each run of a case is a fresh process (cuBLAS, cuBLASLt and PyTorch read their environment variables once), with
cuBLASLt's API log on, so the kernel that ran (algorithm id, tile, stages, cluster shape) is read from the library
itself. The output is filled with NaN before the call, so an output the kernel never wrote is seen as such, and
every output is judged against a float64 reference on the CPU with a rounding bound (one rounding of the output
format plus fp32 accumulation over k terms, a worst case).

Verdicts never claim more than was shown:
  PASSED        right every time on both stacks (for these cases, shapes and conditions only)
  REGRESSED     right every time before, wrong at least once after
  UNCHANGED     wrong before and after ("still wrong"), or right after only because another kernel ran
                ("avoided": the failing kernel wasn't run, so nothing shows it was fixed)
  FIXED         wrong before, right every time after, the same kernel ran after, and the difference in failure
                rates is significant (one-sided Fisher exact test, p < 0.01)
  INCONCLUSIVE  the two runs didn't run under the same conditions (GPU, SMs visible, MPS share, the environment
                variables and PyTorch flags that steer GEMM kernels: unknown is never a match), too few runs, a
                run failed, or no kernel log where the verdict needs one

Exit codes: 0 when every verdict is PASSED or FIXED (or the workaround WORKS); 1 when one is REGRESSED or
UNCHANGED (or the workaround DOESN'T WORK); 2 when one is INCONCLUSIVE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Optional

CASE_SCHEMA = "sdcguard.gate/1"
RUN_SCHEMA = "sdcguard.gate-run/1"
MIN_RUNS = 5                       # fewer runs on either side of a comparison: INCONCLUSIVE
FIXED_P = 0.01                     # FIXED needs the drop in failures to be this significant

# environment variables that steer which GEMM kernel runs or how: recorded with every run, and part of the
# conditions two runs must share (logging switches excepted)
STEER_PREFIXES = ("CUBLAS", "CUBLASLT_", "PYTORCH_TUNABLEOP_", "TORCH_CUBLASLT_", "CUDA_MPS_ACTIVE_THREAD")
STEER_NAMES = ("TORCH_BLAS_PREFER_CUBLASLT", "DISABLE_ADDMM_CUDA_LT", "NVIDIA_TF32_OVERRIDE", "CUDA_DEVICE_MAX_CONNECTIONS")
LOGGING = ("CUBLASLT_LOG_LEVEL", "CUBLASLT_LOG_FILE", "CUBLASLT_LOG_MASK", "CUBLAS_LOGINFO_DBG", "CUBLAS_LOGDEST_DBG",
           "PYTORCH_TUNABLEOP_VERBOSE", "PYTORCH_TUNABLEOP_VERBOSE_FILENAME")
_U32 = 2.0 ** -24
_UNIT = {"bfloat16": 2.0 ** -8, "float16": 2.0 ** -11, "float32": 2.0 ** -24}


def steering_env(env=None) -> dict:
    env = os.environ if env is None else env
    return {k: v for k, v in sorted(env.items())
            if (k.startswith(STEER_PREFIXES) or k in STEER_NAMES) and k not in LOGGING}


def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------- cuBLASLt's log: which kernel ran
_ALGO = re.compile(r"algoId=(\d+)")
_TILE = re.compile(r"\btile=(?:MATMUL_TILE_)?(\w+)")
_STAGES = re.compile(r"\bstages=(?:MATMUL_STAGES_)?(\w+)")
_CLUSTER = re.compile(r"clusterShape=(?:CLUSTER_SHAPE_)?(\w+)|CLUSTER_SHAPE_(\w+)")


def parse_lt_log(text: str) -> dict:
    """The cuBLASLt matmul calls in an API log (CUBLASLT_LOG_LEVEL=5): how many, and the kernel each ran, as
    "algoId=66 tile=64x8 stages=... cluster=4x1x1" with the parts the log prints."""
    kernels: dict = {}
    calls = 0
    for line in text.splitlines():
        if "cublasLtMatmul]" not in line or "algoId=" not in line:
            continue
        calls += 1
        a = _ALGO.search(line)
        t, s, c = _TILE.search(line), _STAGES.search(line), _CLUSTER.search(line)
        tag = f"algoId={a.group(1)}"
        if t:
            tag += f" tile={t.group(1)}"
        if s:
            tag += f" stages={s.group(1)}"
        tag += f" cluster={(c.group(1) or c.group(2)) if c else '?'}"
        kernels[tag] = kernels.get(tag, 0) + 1
    return {"calls": calls, "kernels": kernels}


def kernel_key(tag: str) -> str:
    """The parts of a kernel tag that name the kernel: algorithm and cluster shape (tile and stages too when known)."""
    return tag


def _algo_cluster(tag: str) -> tuple:
    a = re.search(r"algoId=(\d+)", tag)
    c = re.search(r"cluster=(\S+)", tag)
    return (a.group(1) if a else None, c.group(1) if c else None)


# ---------------------------------------------------------------- one case, run in this process (the worker)
def _align(t) -> int:
    p = t.data_ptr()
    return 256 if p % 256 == 0 else p & -p


def _judge(y, ref, bound, prefilled: bool) -> dict:
    import torch
    yd = y.double().cpu()
    nan = torch.isnan(yd)
    finite = torch.isfinite(yd)
    err = (yd - ref).abs()
    ok = finite & (err <= bound)
    wrong = int((~ok).sum())
    over = (err / bound).nan_to_num(nan=float("inf"), posinf=float("inf"))
    return {"wrong": wrong, "unwritten": int(nan.sum()) if prefilled else None, "nonfinite": int((~finite).sum()),
            "worst_error_over_bound": float(over.max()) if over.numel() else 0.0, "prefilled": prefilled}


def run_here(case_dir: str, repeats: int, device: Optional[str] = None) -> dict:
    """Run one case `repeats` times in this process: the call the case recorded, on inputs laid out as recorded,
    into an output filled with NaN first, each result judged against float64. Records the conditions."""
    import torch
    import torch.nn.functional as F
    case = json.load(open(os.path.join(case_dir, "case.json")))
    if case.get("schema") != CASE_SCHEMA:
        raise ValueError(f"{case_dir}: not a {CASE_SCHEMA} case")
    op = case["op"]
    if op.get("call") != "linear":
        raise ValueError(f"{case_dir}: call {op.get('call')!r} isn't supported by this sdcguard")
    t = torch.load(os.path.join(case_dir, "inputs.pt"), map_location="cpu", weights_only=True)
    dt = getattr(torch, op["dtype"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x, w = t["x"].to(dt), t["w"].to(dt)
    b = t.get("bias")
    b = None if b is None else b.to(dt)
    m, k = x.shape
    n = w.shape[0]
    x64, w64 = x.double(), w.double()
    ref = x64 @ w64.t() + (0 if b is None else b.double())
    # the output format's rounding, fp32 accumulation over k products (a worst case), and the bias: added in fp32
    # before the one rounding (cuBLASLt's epilogue), or to the rounded product (the legacy path), which rounds twice
    acc = x64 @ w64.t()
    bound = (_UNIT[op["dtype"]] * (ref.abs() + (0 if b is None else acc.abs())) + k * _U32 * (x64.abs() @ w64.abs().t())
             + (0 if b is None else 2 * _U32 * b.double().abs()) + 1e-30)
    xd, wd = x.to(dev), w.to(dev)
    bd = None if b is None else b.to(dev)
    runs = []
    with torch.no_grad():
        for _ in range(repeats):
            # NaN first, in the block the call is about to get: the caching allocator hands the freed block of the
            # same size back to the next allocation, which is checked by its address
            pre = torch.full((m, n), float("nan"), dtype=dt, device=dev)
            ptr = pre.data_ptr()
            del pre
            y = F.linear(xd, wd, bd)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            runs.append(_judge(y, ref, bound, y.data_ptr() == ptr))
            del y
    from .casefile import software_stack
    from .check import gemm_settings
    sw = software_stack(dev)
    cuda = dev.type == "cuda"
    cond = {"gpu_name": sw.get("gpu_name") if cuda else "cpu",
            "compute_capability": sw.get("compute_capability") if cuda else "cpu",
            "sms_visible": sw.get("sms_visible") if cuda else 0, "env": steering_env(), "matmul": gemm_settings(),
            # what cuBLASLt's heuristic is told about each pointer: its alignment, up to 256 bytes (PyTorch passes it
            # as the minimum alignment preference), on the GPU only
            "alignment": {"x": _align(xd), "w": _align(wd), "bias": None if bd is None else _align(bd)} if cuda else None,
            "device_type": dev.type}
    pin = os.environ.get("PYTORCH_TUNABLEOP_FILENAME")
    if pin:
        cond["tunableop_file_sha256"] = _sha256_file(pin)
    stack = {key: sw.get(key) for key in ("torch", "cuda", "cublas", "cublaslt_loaded", "driver", "sdcguard")}
    return {"case": case["id"], "runs": runs, "n": len(runs), "conditions": cond, "stack": stack,
            "outputs": m * n}


def _worker_main(argv) -> int:
    case_dir, out, repeats = argv[0], argv[1], int(argv[2])
    device = argv[3] if len(argv) > 3 and argv[3] else None
    try:
        res = run_here(case_dir, repeats, device)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"}
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return 0


def run_case(case_dir: str, repeats: int = 20, env: Optional[dict] = None, device: Optional[str] = None,
             timeout: float = 600.0) -> dict:
    """Run one case in a fresh process with cuBLASLt's API log on; the result with the kernels that ran."""
    case_dir = os.path.abspath(case_dir)
    with tempfile.TemporaryDirectory(prefix="sdcguard_gate_") as tmp:
        out, log = os.path.join(tmp, "result.json"), os.path.join(tmp, "lt.log")
        e = {**os.environ, **(env or {}), "CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_FILE": log}
        e = {k: v for k, v in e.items() if v is not None}
        cmd = [sys.executable, "-m", "sdcguard.gate", "--worker", case_dir, out, str(repeats), device or ""]
        try:
            p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"case": os.path.basename(case_dir), "error": f"timed out after {timeout:.0f} s"}
        try:
            res = json.load(open(out))
        except (OSError, ValueError):
            return {"case": os.path.basename(case_dir),
                    "error": f"the run exited {p.returncode} without a result: {(p.stdout + p.stderr)[-800:]}"}
        try:
            with open(log, errors="replace") as f:
                lt = parse_lt_log(f.read())
            lt["log"] = True
        except OSError:
            lt = {"log": False, "calls": 0, "kernels": {}}
        res["kernel_log"] = lt
        res["applied_env"] = steering_env(env or {})
        return res


# ---------------------------------------------------------------- cases
def _case_id(op: dict, cond: dict) -> str:
    gpu = re.sub(r"[^A-Za-z0-9]+", "", str(cond.get("gpu_name") or "gpu").replace("NVIDIA", ""))[:12].lower() or "gpu"
    mps = (cond.get("env") or {}).get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    return (f"{gpu}{'-mps' + mps if mps else ''}-{op['call']}-{op['dtype']}-{op['n']}x{op['k']}x{op['m']}"
            f"{'-bias' if op['bias'] else ''}")


def _write_case(pack: str, op: dict, tensors: dict, conditions: dict, stack: dict, observed: dict, kind: str,
                source: dict, kernels: Optional[dict], name: Optional[str] = None) -> str:
    import torch
    cid = name or _case_id(op, conditions)
    d = os.path.join(pack, cid)
    os.makedirs(d, exist_ok=True)
    torch.save({k: (None if v is None else v.detach().cpu().contiguous()) for k, v in tensors.items()},
               os.path.join(d, "inputs.pt"))
    case = {"schema": CASE_SCHEMA, "id": cid, "kind": kind, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": source, "op": op, "inputs": {"file": "inputs.pt",
                                                     "sha256": _sha256_file(os.path.join(d, "inputs.pt"))},
            "conditions": conditions, "stack": stack, "kernels": kernels or {}, "observed": observed,
            "expected": {"reference": "float64 on the CPU", "bound": "output rounding + k x 2^-24 x sum|x||w|",
                         "prefill": "nan"}}
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(case, f, indent=1, default=str)
    return d


def capture(shape: tuple, dtype: str = "bfloat16", bias: bool = True, pack: str = "gates", seed: int = 0,
            repeats: int = 20, name: Optional[str] = None, device: Optional[str] = None) -> dict:
    """Run a linear layer of this shape (out features, in features, tokens) here, with seeded inputs, and keep it
    as a case: an incident if any run was wrong, else a pin (a shape known right here, to catch a regression)."""
    import torch
    n, k, m = shape
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    w = torch.randn(n, k, generator=g) / math.sqrt(k)
    b = torch.randn(n, generator=g) if bias else None
    op = {"call": "linear", "dtype": dtype, "m": m, "n": n, "k": k, "bias": bias}
    with tempfile.TemporaryDirectory(prefix="sdcguard_gate_new_") as tmp:
        d = _write_case(tmp, op, {"x": x, "w": w, "bias": b}, {}, {}, {}, "pending", {"kind": "capture", "seed": seed},
                        None, name="case")
        res = run_case(d, repeats, device=device)
    if "error" in res:
        raise RuntimeError(res["error"])
    bad = sum(r["wrong"] > 0 or (r.get("unwritten") or 0) > 0 for r in res["runs"])
    observed = {"runs": res["n"], "runs_wrong": bad, "wrong_per_run": [r["wrong"] for r in res["runs"]],
                "unwritten_per_run": [r.get("unwritten") for r in res["runs"]], "outputs": res["outputs"]}
    d = _write_case(pack, op, {"x": x, "w": w, "bias": b}, res["conditions"], res["stack"], observed,
                    "incident" if bad else "pin", {"kind": "capture", "seed": seed}, res["kernel_log"].get("kernels"),
                    name)
    return {"case": d, "kind": "incident" if bad else "pin", "runs_wrong": bad, "runs": res["n"],
            "kernels": res["kernel_log"].get("kernels")}


def add(evidence: str, pack: str = "gates", name: Optional[str] = None) -> dict:
    """A case from a saved evidence file (sdcguard check, the training guard): its operands and the conditions it
    recorded. Serving evidence holds one row and one 128-column block of a layer, not the layer's whole matmul, so
    its case runs that slice: another shape, which cuBLAS may give another kernel (the case says so)."""
    import torch
    ev = torch.load(evidence, map_location="cpu", weights_only=False)
    info_path = evidence[:-3] + ".json" if evidence.endswith(".pt") else None
    info = {}
    if info_path and os.path.exists(info_path):
        info = json.load(open(info_path))
    A, B, C, bias = ev["A"], ev["B"], ev["C"], ev.get("bias")
    if A.dim() != 2 or B.dim() != 2 or ev.get("batch") is not None:
        raise ValueError("only 2-D matmul evidence (not batched) can be a case")
    if float(ev.get("alpha", 1.0)) != 1.0 or float(ev.get("beta", 1.0)) not in (0.0, 1.0):
        raise ValueError("only alpha = 1 and beta = 0 or 1 (a linear layer) can be a case")
    # evidence holds C = A @ B (+ bias): as a linear layer, x = A, w = B^T
    m, k = A.shape
    n = B.shape[1]
    op = {"call": "linear", "dtype": str(C.dtype).replace("torch.", ""), "m": m, "n": n, "k": k,
          "bias": bias is not None}
    sw = info.get("software") or {}
    cond = {"gpu_name": (info.get("gpu") or {}).get("gpu_name") or sw.get("gpu_name"),
            "compute_capability": sw.get("compute_capability"), "sms_visible": sw.get("sms_visible"),
            "env": dict(sw.get("environment") or {}) or None, "matmul": sw.get("matmul"), "alignment": None,
            "device_type": "cuda"}
    stack = {key: sw.get(key) for key in ("torch", "cuda", "cublas", "cublaslt_loaded", "driver", "sdcguard")}
    observed = {"from_evidence": True, "max_z": info.get("max_z"), "cpu_recheck": info.get("cpu_recheck")}
    slice_note = "a slice of the layer's matmul (one row, one block)" if info.get("phase") == "serving" else None
    d = _write_case(pack, op, {"x": A, "w": B.t(), "bias": bias}, cond, stack, observed, "incident",
                    {"kind": "evidence", "file": os.path.abspath(evidence), "sha256": _sha256_file(evidence),
                     "note": slice_note}, None, name)
    return {"case": d}


def cases(pack: str) -> list:
    out = []
    for name in sorted(os.listdir(pack)):
        d = os.path.join(pack, name)
        if os.path.isfile(os.path.join(d, "case.json")):
            out.append(d)
    return out


def run_pack(pack: str, repeats: int = 20, env: Optional[dict] = None, device: Optional[str] = None) -> dict:
    results = {}
    for d in cases(pack):
        res = run_case(d, repeats, env, device)
        results[os.path.basename(d)] = res
    return {"schema": RUN_SCHEMA, "pack": os.path.abspath(pack), "repeats": repeats,
            "when": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": results}


# ---------------------------------------------------------------- verdicts
def _bad_runs(res: dict) -> int:
    return sum(r.get("wrong", 0) > 0 or (r.get("unwritten") or 0) > 0 for r in res.get("runs", []))


# the matmul settings an environment variable changes: a workaround that sets the variable changes them on purpose
_ENV_MATMUL = {"PYTORCH_TUNABLEOP_ENABLED": ("tunableop",), "TORCH_BLAS_PREFER_CUBLASLT": ("blas_library",)}


def condition_differences(a: dict, b: dict, ignore_env: tuple = ()) -> Optional[list]:
    """What two runs' conditions differ in ([] when nothing does); None when either didn't record one of them.
    ignore_env: variables a workaround sets on purpose (with the PyTorch settings they change). A TunableOp file is
    compared by its contents, not its path."""
    from .check import gemm_effective
    out = []
    ignore_mm = {m for k in ignore_env for m in _ENV_MATMUL.get(k, ())}
    for key in ("gpu_name", "compute_capability", "sms_visible", "device_type"):
        if a.get(key) is None or b.get(key) is None:
            return None
        if a[key] != b[key]:
            out.append(f"{key} {a[key]} vs {b[key]}")
    ea, eb = a.get("env"), b.get("env")
    if ea is None or eb is None:
        return None
    for k in sorted(set(ea) | set(eb)):
        if k in ignore_env:
            continue
        if k == "PYTORCH_TUNABLEOP_FILENAME" and k in ea and k in eb:
            if a.get("tunableop_file_sha256") != b.get("tunableop_file_sha256") or a.get("tunableop_file_sha256") is None:
                out.append("the TunableOp file's contents differ (or couldn't be read)")
            continue
        if ea.get(k) != eb.get(k):
            out.append(f"{k} {ea.get(k)!r} vs {eb.get(k)!r}")
    ga, gb = gemm_effective(a.get("matmul")), gemm_effective(b.get("matmul"))
    if ga is None or gb is None or set(ga) != set(gb):
        return None
    out += [f"matmul {k} {ga[k]} vs {gb[k]}" for k in sorted(ga) if k != "env" and k not in ignore_mm and ga[k] != gb[k]]
    al, bl = a.get("alignment"), b.get("alignment")
    if isinstance(al, dict) and isinstance(bl, dict) and al != bl:
        out.append(f"input alignment {al} vs {bl}")
    return out


def fisher_one_sided(bad_a: int, n_a: int, bad_b: int, n_b: int) -> float:
    """P(B has this few failures or fewer | the same rate on both sides, totals fixed): one-sided Fisher exact."""
    total_bad, n = bad_a + bad_b, n_a + n_b

    def h(x):        # P(B gets x of the failures)
        return math.exp(math.lgamma(total_bad + 1) - math.lgamma(x + 1) - math.lgamma(total_bad - x + 1)
                        + math.lgamma(n - total_bad + 1) - math.lgamma(n_b - x + 1)
                        - math.lgamma(n - total_bad - n_b + x + 1) - (math.lgamma(n + 1) - math.lgamma(n_b + 1)
                                                                       - math.lgamma(n - n_b + 1)))
    lo = max(0, n_b - (n - total_bad))
    return min(1.0, sum(h(x) for x in range(lo, bad_b + 1)))


def verdict(base: dict, cand: dict) -> dict:
    """One case: the baseline run (before the change) against the candidate run (after)."""
    from .casefile import rate_interval
    for side, r in (("baseline", base), ("candidate", cand)):
        if not isinstance(r, dict) or "error" in r or not r.get("runs"):
            return {"verdict": "INCONCLUSIVE", "why": f"the {side} run has no result: "
                    f"{(r or {}).get('error', 'missing') if isinstance(r, dict) else 'missing'}"}
    diffs = condition_differences(base["conditions"], cand["conditions"])
    if diffs is None:
        return {"verdict": "INCONCLUSIVE", "why": "a condition wasn't recorded on one side, so the two runs can't be "
                "shown to have run the same way (unknown is never a match)"}
    if diffs:
        return {"verdict": "INCONCLUSIVE", "why": "the runs didn't run under the same conditions: " + "; ".join(diffs),
                "differences": diffs}
    nb, nc = base["n"], cand["n"]
    bb, bc = _bad_runs(base), _bad_runs(cand)
    out = {"baseline": {"runs": nb, "wrong": bb, "rate_95": rate_interval(bb, nb),
                        "kernels": (base.get("kernel_log") or {}).get("kernels")},
           "candidate": {"runs": nc, "wrong": bc, "rate_95": rate_interval(bc, nc),
                         "kernels": (cand.get("kernel_log") or {}).get("kernels")},
           "stack_changes": {k: [base["stack"].get(k), cand["stack"].get(k)] for k in sorted(base.get("stack", {}))
                             if base["stack"].get(k) != (cand.get("stack") or {}).get(k)}}
    if min(nb, nc) < MIN_RUNS:
        return {**out, "verdict": "INCONCLUSIVE", "why": f"fewer than {MIN_RUNS} runs on a side"}
    if bb == 0 and bc == 0:
        return {**out, "verdict": "PASSED", "why": f"right in all {nb} and {nc} runs (this case's shape and "
                "conditions only)"}
    if bb == 0:
        return {**out, "verdict": "REGRESSED", "why": f"right in all {nb} runs before, wrong in {bc} of {nc} after"}
    if bc > 0:
        return {**out, "verdict": "UNCHANGED", "flavor": "still wrong",
                "why": f"wrong in {bb} of {nb} runs before and {bc} of {nc} after"}
    kb, kc = out["baseline"]["kernels"] or {}, out["candidate"]["kernels"] or {}
    if not kb or not kc:
        return {**out, "verdict": "INCONCLUSIVE", "why": "right in every run after, but without cuBLASLt's log on both "
                "sides nothing shows the kernel that failed before ran after (a heuristic that picks another kernel "
                "avoids a bug without fixing it)"}
    missing = sorted(t for t in kb if t not in kc)
    if missing:
        return {**out, "verdict": "UNCHANGED", "flavor": "avoided",
                "why": f"right in every run after, but the kernel{'s' if len(missing) > 1 else ''} that ran before "
                       f"({', '.join(missing)}) didn't run after ({', '.join(sorted(kc))}): avoided, not shown fixed"}
    p = fisher_one_sided(bb, nb, bc, nc)
    if p >= FIXED_P:
        return {**out, "verdict": "INCONCLUSIVE", "p": p, "why": f"wrong in {bb} of {nb} before and 0 of {nc} after: "
                f"not enough runs to tell the rates apart (p = {p:.3g}); run more"}
    return {**out, "verdict": "FIXED", "p": p,
            "why": f"the same kernel{'s' if len(kb) > 1 else ''} ran, wrong in {bb} of {nb} runs before and 0 of {nc} "
                   f"after (p = {p:.2g}; failure rate after below {out['candidate']['rate_95'][1]:.1%} at 95%)"}


def compare(base_run: dict, cand_run: dict) -> dict:
    out = {}
    for cid in sorted(set(base_run.get("cases", {})) | set(cand_run.get("cases", {}))):
        out[cid] = verdict(base_run["cases"].get(cid), cand_run["cases"].get(cid))
    return out


def workaround(case_dir: str, treatment_env: dict, repeats: int = 50, expect_kernel: Optional[str] = None,
               device: Optional[str] = None) -> dict:
    """Does a workaround fix this case here? Two fresh processes under the same conditions: control (as the case
    ran) and treatment (with treatment_env). WORKS only if the control still fails, the treatment is right in every
    run with no output left unwritten, and (expect_kernel, e.g. "algoId=66 cluster=1x1x1") the log shows that kernel
    on every call."""
    from .casefile import rate_interval
    ctl = run_case(case_dir, repeats, None, device)
    trt = run_case(case_dir, repeats, treatment_env, device)
    out = {"case": os.path.basename(case_dir), "treatment_env": treatment_env, "control": None, "treatment": None}
    for side, r in (("control", ctl), ("treatment", trt)):
        if "error" in r:
            return {**out, "verdict": "INCONCLUSIVE", "why": f"the {side} run failed: {r['error']}"}
        out[side] = {"runs": r["n"], "wrong": _bad_runs(r), "rate_95": rate_interval(_bad_runs(r), r["n"]),
                     "kernels": (r.get("kernel_log") or {}).get("kernels"),
                     "calls_logged": (r.get("kernel_log") or {}).get("calls"),
                     "unwritten": sum((x.get("unwritten") or 0) for x in r["runs"])}
    diffs = condition_differences(ctl["conditions"], trt["conditions"], ignore_env=tuple(treatment_env))
    if diffs is None or diffs:
        return {**out, "verdict": "INCONCLUSIVE", "why": "control and treatment didn't run under the same conditions "
                "besides the workaround: " + ("unknown" if diffs is None else "; ".join(diffs))}
    if out["control"]["wrong"] == 0:
        return {**out, "verdict": "INCONCLUSIVE", "why": f"the failure didn't come back without the workaround (0 of "
                f"{ctl['n']} runs wrong), so a clean treatment shows nothing"}
    if out["treatment"]["wrong"] > 0:
        return {**out, "verdict": "DOESN'T WORK", "why": f"wrong in {out['treatment']['wrong']} of {trt['n']} runs "
                "with the workaround"}
    if expect_kernel:
        want = _algo_cluster(expect_kernel)
        ks = out["treatment"]["kernels"] or {}
        on = sum(c for t, c in ks.items() if _algo_cluster(t) == want)
        calls = out["treatment"]["calls_logged"] or 0
        if calls < trt["n"] or on != calls:
            return {**out, "verdict": "INCONCLUSIVE", "why": f"right in every run, but the log shows {ks or 'no kernel'} "
                    f"instead of {expect_kernel} on every one of {trt['n']} calls: the workaround may not be what ran"}
    up = out["treatment"]["rate_95"][1]
    return {**out, "verdict": "WORKS", "why": f"wrong in {out['control']['wrong']} of {ctl['n']} runs without it and 0 "
            f"of {trt['n']} with it (failure rate with it below {up:.1%} at 95%)"
            + (f"; the log shows {expect_kernel} on every call" if expect_kernel else "")}


# ---------------------------------------------------------------- TunableOp pin for the 6862950 workaround
def pin(case_dir: str, out_csv: str, device: Optional[str] = None) -> dict:
    """A PyTorch TunableOp results file (PyTorch 2.14 and later on CUDA) that makes this case's GEMM run cuBLASLt
    algorithm 66 with a 1x1x1 cluster (NVIDIA's workaround for known issue 6862950) without changing any code:
        PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=<out_csv>
    The configuration is the first algorithm-66 candidate cuBLASLt offers for this shape with 4 or more CTAs per
    cluster that accepts a 1x1x1 cluster and computes right (run and judged here first). The rest of the file
    (PyTorch's validators and the GEMM's signature) is written by TunableOp itself, so PyTorch accepts it; it
    rejects the whole file after any change of PyTorch, cuBLASLt or GPU, so rerun this after an upgrade, and check
    with `gate workaround` that the pin is what runs."""
    with tempfile.TemporaryDirectory(prefix="sdcguard_pin_") as tmp:
        out = os.path.join(tmp, "pin.json")
        cmd = [sys.executable, "-m", "sdcguard.gate", "--pin-worker", os.path.abspath(case_dir), out, device or ""]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        try:
            res = json.load(open(out))
        except (OSError, ValueError):
            raise RuntimeError(f"no pin was made: {(p.stdout + p.stderr)[-800:]}")
        tuned = os.path.join(tmp, "tuned.csv")
        e = {**os.environ, "PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "1",
             "PYTORCH_TUNABLEOP_FILENAME": tuned, "PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS": "3",
             "PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS": "1"}
        if "error" in res:
            return res
        q = subprocess.run([sys.executable, "-m", "sdcguard.gate", "--worker", os.path.abspath(case_dir),
                            os.path.join(tmp, "tune_run.json"), "1", device or ""], env=e, capture_output=True,
                           text=True, timeout=600)
        files = [f for f in os.listdir(tmp) if f.startswith("tuned") and f.endswith(".csv")]
        if not files:
            return {**res, "error": f"TunableOp wrote no results file: {(q.stdout + q.stderr)[-800:]}"}
        text = open(os.path.join(tmp, sorted(files)[0])).read()
    pinned, rows = rewrite_pin(text, res["bias"], res["name"])
    if rows != 1:
        return {**res, "error": f"expected one {'GemmAndBias' if res['bias'] else 'Gemm'}TunableOp row in TunableOp's "
                f"file, found {rows}", "tunableop_file": text}
    with open(out_csv, "w") as f:
        f.write(pinned)
    return {**res, "file": os.path.abspath(out_csv), "sha256": _sha256_file(out_csv),
            "use": f"PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME={out_csv}"}


def rewrite_pin(text: str, bias: bool, name: str) -> tuple:
    """TunableOp's own results file with the solution of the case's GEMM row replaced by `name` (its validators and
    the GEMM's signature stay as PyTorch wrote them). Returns (the file, how many rows were replaced)."""
    want = "GemmAndBiasTunableOp" if bias else "GemmTunableOp"
    out, rows = [], 0
    for ln in text.splitlines():
        parts = ln.split(",")
        if parts and parts[0].startswith(want + "_") and len(parts) >= 3:
            parts[2] = name
            rows += 1
            ln = ",".join(parts)
        out.append(ln)
    return "\n".join(out) + "\n", rows


def _pin_worker(argv) -> int:
    case_dir, out = argv[0], argv[1]
    try:
        res = _pin_config(case_dir, argv[2] if len(argv) > 2 and argv[2] else None)
    except Exception as e:  # noqa: BLE001
        res = {"error": f"{type(e).__name__}: {e}"}
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    return 0


def _pin_config(case_dir: str, device: Optional[str]) -> dict:
    """Find, run and judge the 1x1x1-cluster version of each algorithm-66 candidate for this case's GEMM; the first
    right one's TunableOp name."""
    import ctypes
    import torch
    from . import blas
    case = json.load(open(os.path.join(case_dir, "case.json")))
    op = case["op"]
    dt = getattr(torch, op["dtype"])
    if dt not in blas._DTYPES:
        return {"error": f"{op['dtype']} isn't a cuBLASLt half-precision type here"}
    ok, why = blas.available()
    if not ok:
        return {"error": why}
    t = torch.load(os.path.join(case_dir, "inputs.pt"), map_location="cpu", weights_only=True)
    m, n, k = op["m"], op["n"], op["k"]
    lib = blas._library()
    call = blas._Call(lib)
    dev = torch.device(device or "cuda")
    W, X = t["w"].to(dt), t["x"].to(dt)
    ref = X.double() @ W.double().t()
    tol = blas._UNIT[dt] * ref.abs() + k * blas._U32 * (X.double().abs() @ W.double().abs().t()) + 1e-30
    Wd, Xd = W.to(dev).contiguous(), X.to(dev).contiguous()
    Y = torch.empty(m, n, dtype=dt, device=dev)
    ws_bytes = 1 << 20                     # PyTorch's default cuBLASLt workspace: a pinned config must fit in it
    ws = torch.empty(ws_bytes, dtype=torch.uint8, device=dev)
    ty = blas._DTYPES[dt]
    vp = ctypes.c_void_p
    handle, desc, la, lb, ld, pref = vp(), vp(), vp(), vp(), vp(), vp()
    attrs = (("id", 0, ctypes.c_int32), ("tile", 1, ctypes.c_uint32), ("splitk", 2, ctypes.c_int32),
             ("red", 3, ctypes.c_uint32), ("swizzle", 4, ctypes.c_uint32), ("custom", 5, ctypes.c_uint32),
             ("stages", 6, ctypes.c_uint32), ("inner", 7, ctypes.c_uint16), ("cluster", 8, ctypes.c_uint16))
    tried = []
    try:
        with torch.cuda.device(dev):
            call("cublasLtCreate", ctypes.byref(handle))
            call("cublasLtMatmulDescCreate", ctypes.byref(desc), blas._CUBLAS_COMPUTE_32F, blas._CUDA_R_32F)
            ta, tb = ctypes.c_int32(blas._OP_T), ctypes.c_int32(blas._OP_N)
            call("cublasLtMatmulDescSetAttribute", desc, blas._DESC_TRANSA, ctypes.byref(ta), ctypes.sizeof(ta))
            call("cublasLtMatmulDescSetAttribute", desc, blas._DESC_TRANSB, ctypes.byref(tb), ctypes.sizeof(tb))
            call("cublasLtMatrixLayoutCreate", ctypes.byref(la), ty, ctypes.c_uint64(k), ctypes.c_uint64(n),
                 ctypes.c_int64(k))
            call("cublasLtMatrixLayoutCreate", ctypes.byref(lb), ty, ctypes.c_uint64(k), ctypes.c_uint64(m),
                 ctypes.c_int64(k))
            call("cublasLtMatrixLayoutCreate", ctypes.byref(ld), ty, ctypes.c_uint64(n), ctypes.c_uint64(m),
                 ctypes.c_int64(n))
            call("cublasLtMatmulPreferenceCreate", ctypes.byref(pref))
            wsize = ctypes.c_size_t(ws_bytes)
            call("cublasLtMatmulPreferenceSetAttribute", pref, blas._PREF_MAX_WORKSPACE_BYTES, ctypes.byref(wsize),
                 ctypes.sizeof(wsize))
            heur = (blas._Heuristic * 16)()
            found = ctypes.c_int(0)
            call("cublasLtMatmulAlgoGetHeuristic", handle, desc, la, lb, ld, ld, pref, 16, heur, ctypes.byref(found))
            stream = vp(torch.cuda.current_stream(dev).cuda_stream)
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)

            def get(algo, which, ctype):
                v, got = ctype(0), ctypes.c_size_t(0)
                lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), which, ctypes.byref(v),
                                                         ctypes.sizeof(v), ctypes.byref(got))
                return int(v.value)

            for i in range(found.value):
                if heur[i].state != 0 or get(heur[i].algo, 0, ctypes.c_int32) != 66 or \
                        blas.cluster_ctas(get(heur[i].algo, 8, ctypes.c_uint16)) < 4:
                    continue
                a = blas._Algo()
                ctypes.memmove(ctypes.byref(a), ctypes.byref(heur[i].algo), ctypes.sizeof(blas._Algo))
                c1 = ctypes.c_uint16(blas._CLUSTER_1x1x1)
                chk = blas._Heuristic()
                rec = {"candidate": i, "cluster_before": get(heur[i].algo, 8, ctypes.c_uint16)}
                if lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), 8, ctypes.byref(c1), ctypes.sizeof(c1)) \
                        or lib.cublasLtMatmulAlgoCheck(handle, desc, la, lb, ld, ld, ctypes.byref(a), ctypes.byref(chk)):
                    tried.append({**rec, "accepted": False})
                    continue
                if chk.workspaceSize > ws_bytes:
                    tried.append({**rec, "accepted": False, "why": "needs more workspace than PyTorch gives cuBLASLt"})
                    continue
                Y.fill_(float("nan"))
                torch.cuda.synchronize(dev)
                st = lib.cublasLtMatmul(handle, desc, ctypes.byref(alpha), vp(Wd.data_ptr()), la, vp(Xd.data_ptr()),
                                        lb, ctypes.byref(beta), vp(Y.data_ptr()), ld, vp(Y.data_ptr()), ld,
                                        ctypes.byref(a), vp(ws.data_ptr()), wsize, stream)
                torch.cuda.synchronize(dev)
                got = Y.double().cpu()
                right = st == 0 and bool(torch.isfinite(got).all()) and bool(((got - ref).abs() <= tol).all())
                cfg = {name: get(a, which, ctype) for name, which, ctype in attrs}
                tried.append({**rec, "accepted": True, "status": int(st), "right": right, "config": cfg})
                if right:
                    name = ("Gemm_Cublaslt_id_{id}_tile_{tile}_stages_{stages}_splitk_{splitk}_red_{red}_swizzle_"
                            "{swizzle}_custom_{custom}_inner_{inner}_cluster_{cluster}").format(**cfg)
                    return {"name": name, "config": cfg, "bias": bool(op["bias"]), "tried": tried,
                            "note": "the bias is added by the same kernel's epilogue in PyTorch's call; this check ran "
                                    "it without (the same main loop)"}
    finally:
        for fn, h in (("cublasLtMatmulPreferenceDestroy", pref), ("cublasLtMatrixLayoutDestroy", ld),
                      ("cublasLtMatrixLayoutDestroy", lb), ("cublasLtMatrixLayoutDestroy", la),
                      ("cublasLtMatmulDescDestroy", desc), ("cublasLtDestroy", handle)):
            if h.value:
                getattr(lib, fn)(h)
    return {"error": "no algorithm-66 candidate with 4+ CTAs per cluster here accepted a 1x1x1 cluster and computed "
                     "right (or cuBLASLt offered none: this shape may not need the workaround here)", "tried": tried}


# ---------------------------------------------------------------- command line
_EXIT = {"PASSED": 0, "FIXED": 0, "WORKS": 0, "REGRESSED": 1, "UNCHANGED": 1, "DOESN'T WORK": 1, "INCONCLUSIVE": 2}


def _env_pairs(items) -> dict:
    out = {}
    for it in items or ():
        if "=" not in it:
            raise SystemExit(f"--env takes KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def _print_verdicts(vs: dict) -> int:
    worst = 0
    for cid, v in vs.items():
        print(f"{v['verdict']:<13} {cid}\n              {v['why']}")
        worst = max(worst, _EXIT.get(v["verdict"], 2))
    return worst


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "--worker":
        return _worker_main(argv[1:])
    if argv and argv[0] == "--pin-worker":
        return _pin_worker(argv[1:])
    if argv and argv[0] in ("model-ref", "model-check"):
        from .gate_model import main as model_main
        return model_main(argv)
    p = argparse.ArgumentParser(prog="sdcguard gate", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="run a linear layer of a shape here and keep it as a case")
    c.add_argument("--shape", required=True, help="out features,in features,tokens, e.g. 1000,2048,16")
    c.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    c.add_argument("--bias", action="store_true")
    c.add_argument("--pack", default="gates")
    c.add_argument("--name", default=None)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--repeats", type=int, default=20)
    a_ = sub.add_parser("add", help="a case from a saved evidence file")
    a_.add_argument("evidence")
    a_.add_argument("--pack", default="gates")
    a_.add_argument("--name", default=None)
    r = sub.add_parser("run", help="run every case here")
    r.add_argument("pack")
    r.add_argument("--out", required=True)
    r.add_argument("--repeats", type=int, default=20)
    r.add_argument("--env", action="append", help="KEY=VALUE for every run (repeatable)")
    r.add_argument("--device", default=None)
    cm = sub.add_parser("compare", help="a verdict per case: a run before a change against one after")
    cm.add_argument("baseline")
    cm.add_argument("candidate")
    cm.add_argument("--json", default=None, help="also write the verdicts here")
    w = sub.add_parser("workaround", help="does a workaround fix a case here?")
    w.add_argument("case")
    w.add_argument("--env", action="append", required=True, help="KEY=VALUE the workaround sets (repeatable)")
    w.add_argument("--repeats", type=int, default=50)
    w.add_argument("--expect-kernel", default=None, help='e.g. "algoId=66 cluster=1x1x1"')
    w.add_argument("--json", default=None)
    pn = sub.add_parser("pin", help="a TunableOp file pinning a case's GEMM to algorithm 66 with a 1x1x1 cluster")
    pn.add_argument("case")
    pn.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "capture":
        shape = tuple(int(v) for v in a.shape.replace("x", ",").split(","))
        if len(shape) != 3:
            raise SystemExit("--shape takes three numbers: out features, in features, tokens")
        res = capture(shape, a.dtype, a.bias, a.pack, a.seed, a.repeats, a.name)
        print(json.dumps(res, indent=1, default=str))
        return 0
    if a.cmd == "add":
        print(json.dumps(add(a.evidence, a.pack, a.name), indent=1, default=str))
        return 0
    if a.cmd == "run":
        res = run_pack(a.pack, a.repeats, _env_pairs(a.env), a.device)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)
        for cid, r in res["cases"].items():
            if "error" in r:
                print(f"{cid}: could not run: {r['error']}")
            else:
                print(f"{cid}: wrong in {_bad_runs(r)} of {r['n']} runs; kernels {r['kernel_log'].get('kernels')}")
        return 2 if any("error" in r for r in res["cases"].values()) else 0
    if a.cmd == "compare":
        vs = compare(json.load(open(a.baseline)), json.load(open(a.candidate)))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(vs, f, indent=1, default=str)
        return _print_verdicts(vs)
    if a.cmd == "workaround":
        v = workaround(a.case, _env_pairs(a.env), a.repeats, a.expect_kernel)
        if a.json:
            with open(a.json, "w") as f:
                json.dump(v, f, indent=1, default=str)
        return _print_verdicts({v["case"]: v})
    res = pin(a.case, a.out)
    print(json.dumps(res, indent=1, default=str))
    return 0 if "error" not in res else 2


if __name__ == "__main__":
    sys.exit(main())
