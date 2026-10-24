"""A wrong matmul kept as a case: does it follow a GPU, a condition, or the software path?

    sdcguard gate investigate gates/<case> [--repeats 20]

The first question after a wrong result decides what to do next: drain a GPU (a bad board), or change the software
or its conditions (a library bug that moving the job won't fix). This runs the case again in fresh processes, each
with cuBLASLt's log on, and changes one thing at a time:

  device     the case as it was recorded (its MPS share and the variables that steer GEMM kernels) on every GPU here
  condition  when the case ran under MPS below 100%, the same on every GPU at 100%
  path       on the first GPU, with TORCH_BLAS_PREFER_CUBLASLT=0 (PyTorch's cuBLAS path instead of cuBLASLt)

Verdicts:
  DEVICE          wrong on some GPUs and right on the others, under the case's conditions and with the same kernel
                  on all of them: it follows a GPU (keep the case as evidence for an RMA; drain that GPU)
  CONDITION       wrong on every GPU that ran it under the recorded condition and right on each at 100%: it follows
                  the MPS share, not a GPU (avoid the condition, or pin another kernel with `gate pin`)
  SOFTWARE PATH   wrong on every GPU (two or more), with the condition too, right with the other library path: it
                  follows the software (pin, change the path, or upgrade; moving the job won't fix it)
  NOT ONE GPU     wrong on every one of two or more GPUs, and neither factor cleared it
  NOT REPRODUCED  right everywhere here: a transient fault, one that needs the original data or memory layout, or
                  conditions this machine doesn't reproduce
  INCONCLUSIVE    a run failed, a run's conditions didn't match the case's, GPUs ran different kernels, a run's kernel
                  log is empty (so whether they ran the same kernel is unknown: a case whose kernel cuBLASLt doesn't
                  log can't be called DEVICE), or there was one GPU and no factor cleared it (one GPU can't tell a
                  board from the software on its own)
"""
from __future__ import annotations

import json
import os
from typing import Optional

_MPS = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def _kernels(r: dict) -> frozenset:
    return frozenset(((r.get("kernel_log") or {}).get("kernels") or {}).keys())


def _summary(r: dict, label: str, device: Optional[str]) -> dict:
    from .gate import _bad_runs
    if "error" in r:
        return {"label": label, "device": device, "error": r["error"]}
    c = r.get("conditions") or {}
    return {"label": label, "device": device, "runs": r.get("n"), "wrong": _bad_runs(r),
            "kernels": sorted(_kernels(r)), "gpu": c.get("gpu_uuid") or c.get("gpu_name"),
            "sms_visible": c.get("sms_visible")}


def investigate(case_dir: str, repeats: int = 20, devices: Optional[list] = None, runner=None) -> dict:
    from .gate import condition_differences, run_case
    run = runner or run_case
    case = json.load(open(os.path.join(case_dir, "case.json")))
    rec = case.get("conditions") or {}
    env_rec = {k: v for k, v in (rec.get("env") or {}).items() if v is not None}
    if devices is None:
        try:
            import torch
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        except Exception:  # noqa: BLE001
            devices = []
    out: dict = {"case": os.path.basename(os.path.abspath(case_dir)), "recorded": {
        "gpu": rec.get("gpu_name"), "sms_visible": rec.get("sms_visible"), "env": env_rec}, "runs": []}
    if not devices:
        return {**out, "verdict": "INCONCLUSIVE", "why": "no GPU here"}
    recorded, usable, problems = {}, {}, []
    for d in devices:
        r = run(case_dir, repeats, env_rec, d)
        s = _summary(r, "device", d)
        out["runs"].append(s)
        recorded[d] = r
        if "error" in r:
            problems.append(f"{d}: the run failed: {r['error']}")
            continue
        diffs = condition_differences(rec, r.get("conditions") or {})
        if diffs is None or diffs:
            problems.append(f"{d}: didn't run under the case's conditions: " + ("unknown" if diffs is None else
                                                                                "; ".join(diffs)))
            s["conditions_match"] = False
            continue
        s["conditions_match"] = True
        usable[d] = r
    from .gate import _bad_runs
    if not usable:
        return {**out, "verdict": "INCONCLUSIVE", "why": "no GPU here ran the case under its recorded conditions: "
                + "; ".join(problems)}
    wrong = [d for d, r in usable.items() if _bad_runs(r)]
    right = [d for d, r in usable.items() if not _bad_runs(r)]
    out["wrong_on"], out["right_on"] = wrong, right
    if not wrong:
        return {**out, "verdict": "NOT REPRODUCED", "why": f"right in every run on {len(right)} GPU(s) under the "
                "case's conditions: a transient fault, one that needs the original data or memory layout, or "
                "conditions this machine doesn't reproduce" + (f" ({'; '.join(problems)})" if problems else "")}
    if right:
        ks = {_kernels(usable[d]) for d in wrong + right}
        if any(not _kernels(usable[d]) for d in wrong + right):
            return {**out, "verdict": "INCONCLUSIVE", "why": "wrong on some GPUs and right on others, but the kernel "
                    "log of a run is missing, so whether they ran the same kernel is unknown (and the difference can "
                    "be the kernel, not the board)"}
        if len(ks) != 1:
            return {**out, "verdict": "INCONCLUSIVE", "why": "wrong on some GPUs and right on others, but they ran "
                    "different kernels, so the difference can be the kernel and not the board"}
        gpus = [f"{d} ({next(s for s in out['runs'] if s['device'] == d).get('gpu')})" for d in wrong]
        return {**out, "verdict": "DEVICE", "why": f"wrong on {', '.join(gpus)} and right on {len(right)} "
                "other GPU(s) with the same kernel and conditions: it follows the GPU (keep this case as evidence "
                "for an RMA, and drain that GPU)"}
    # wrong on every usable GPU: change one thing at a time
    factors = {}
    share = env_rec.get(_MPS)
    if share is not None and str(share).strip() not in ("100", "100.0"):
        at100 = {}
        for d in wrong:
            r = run(case_dir, repeats, {**env_rec, _MPS: "100"}, d)
            out["runs"].append(_summary(r, "condition: MPS 100%", d))
            at100[d] = r
        clean = all("error" not in r and not _bad_runs(r) for r in at100.values())
        factors["condition"] = {"share": share, "right_at_100_on_every_gpu": clean}
        if clean:
            many = len(wrong) > 1
            return {**out, "factors": factors, "verdict": "CONDITION", "why": f"wrong at MPS {share}% on "
                    f"{'every one of ' + str(len(wrong)) + ' GPUs' if many else 'this GPU'} and right at 100%: it "
                    "follows the MPS share" + (", not a GPU" if many else " (one GPU here: that a second board "
                                               "behaves the same is untested)") + "; avoid the share or pin another "
                    "kernel (`sdcguard gate pin`)"}
    r = run(case_dir, repeats, {**env_rec, "TORCH_BLAS_PREFER_CUBLASLT": "0"}, wrong[0])
    out["runs"].append(_summary(r, "path: cuBLAS instead of cuBLASLt", wrong[0]))
    path_clean = "error" not in r and not _bad_runs(r)
    factors["path"] = {"right_with_cublas": path_clean, "kernels": sorted(_kernels(r)) if "error" not in r else None}
    out["factors"] = factors
    if len(wrong) > 1:
        if path_clean:
            return {**out, "verdict": "SOFTWARE PATH", "why": f"wrong on every one of {len(wrong)} GPUs with the "
                    "same conditions, and right through PyTorch's cuBLAS path: it follows the cuBLASLt kernel, not a "
                    "GPU; moving the job won't fix it (pin, change the path, or upgrade)"}
        return {**out, "verdict": "NOT ONE GPU", "why": f"wrong on every one of {len(wrong)} GPUs, and neither the "
                "condition nor the library path cleared it: not a single board; the software or something every GPU "
                "here shares"}
    return {**out, "verdict": "INCONCLUSIVE", "why": "wrong on the one GPU here" + (
        ", and right through PyTorch's cuBLAS path: the cuBLASLt kernel is implicated, but one GPU can't tell a "
        "board from the software (run this where a second GPU is)" if path_clean else
        ", and no factor cleared it: one GPU can't tell a board from the software (run this where a second GPU is)")}
