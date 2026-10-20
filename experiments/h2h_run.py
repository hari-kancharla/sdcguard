"""Head-to-head: do NVIDIA's DCGM diagnostic, gpu-burn and `sdcguard check` notice a real NVIDIA bug that makes
GPU matrix math silently wrong? Runs inside one GPU container (scripts/modal_h2h.py starts it on an H200).

The bug is cuBLAS known issue 6862950 (CUDA 13.4 Update 2 release notes, Oct 12, 2026): on Hopper, FP16, BF16 and
FP8 GEMMs can leave part of the output unwritten when the process is an MPS client with
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100. It is deterministic and needs no faulty hardware, so every tool can be
pointed at the same, reproducible wrong math. sdcguard reproduced it on a Modal H200 on Oct 19, 2026
(results/mps_bug).

At each MPS share (100 as the control, then 40, 75 and 14) the steps are:
  1. ground truth: experiments/mps_unwritten.py (the Oct 19 probe: the 16x2048x1000 layer and the math modes,
     PyTorch, checked against float64 on the CPU)
  2. ground truth for the tools' own calls: experiments/h2h_probe.cpp makes DCGM's and gpu-burn's exact cuBLAS calls
     (Hgemm/Sgemm/Sgemm with tensor-op math/Dgemm, their shapes and inputs) plus FP16/BF16 GemmEx, into buffers
     pre-filled with NaN, and checks them against float64; it also measures whether fresh device memory is zeroed
  3. DCGM: `dcgmi diag -r diagnostic` (its only test that checks GEMM results)
  4. gpu-burn -tc for 60 s (its only mode that could reach FP16 math; the default FP32 and -d FP64 can't)
  5. `sdcguard check` for 5 minutes, offline
The MPS daemon is restarted at the share before every tool, so each starts with a fresh MPS server, and the share
is set both at the daemon and in every tool's environment. At 40% (the share where square FP16/BF16 GEMMs went
wrong in the public reproduction) it also runs DCGM with FP16 only, a short DCGM run with cuBLASLt logging (to see
which kernel DCGM's cuBLAS picks), and gpu-burn's default FP32 mode. A last short 100% recheck repeats steps 1-2.

Classification rules (written before the run, docs/METHOD.md "Head-to-head on a real cuBLAS bug"): each (tool,
share) cell is DETECTED, MISSED, NOT EXERCISED, FALSE ALARM, FAIL (NOT ATTRIBUTABLE) or COULD NOT TEST, judged
against that tool's own ground truth at that share. Nothing is rerun because its result was unwelcome.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

SHARES = (100, 40, 75, 14)
PROBE_CASES = ("hgemm:2048", "sgemm:2048", "sgemmtc:2048", "sgemm:8192", "sgemmtc:8192", "dgemm:8192",
               "ex16:2048", "exbf16:2048")
# which probe cases stand for which tool's GEMMs (DCGM's diagnostic on an H200: 2048, half then single then double,
# tensor-op math switched on halfway; gpu-burn: 8192)
EXERCISE = {
    "dcgm": ("hgemm:2048", "sgemm:2048", "sgemmtc:2048"),
    "dcgm_fp16_only": ("hgemm:2048",),
    "gpu_burn_tc": ("sgemmtc:8192",),
    "gpu_burn_default": ("sgemm:8192",),
}
DCGM_FAULTY_MEMORY, DCGM_NAN_VALUE = 52, 111            # dcgmlib/dcgm_errors.h (DCGM 4.6.1)
MPS_PIPE, MPS_LOG = "/tmp/nvidia-mps", "/tmp/nvidia-mps-log"
TAIL = 4000
MPS_SETTLE = (1.0, 0.5)                                 # seconds after "quit" and after pkill (tests set 0)


# ---------------------------------------------------------------- parsing (pure functions, tested on CPU)

def _json_objects(text: str):
    """Every top-level JSON object in text (one per line, or one spanning many lines)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    if not out and "{" in text:
        try:
            out.append(json.loads(text[text.index("{"):text.rindex("}") + 1]))
        except ValueError:
            pass
    return out


def parse_probe(text: str) -> dict:
    """{'hgemm:2048': {...}, 'scrub': {...}} from h2h_probe output."""
    res = {}
    for o in _json_objects(text):
        if "case" not in o:
            continue
        key = o["case"] if o["case"] == "scrub" else f"{o['case']}:{o.get('n')}"
        res[key] = o
    return res


def probe_wrong(probe: dict, cases) -> bool | None:
    """True if any listed probe case computed wrong results, False if all ran and were right, None if unknown."""
    seen = [probe.get(c) for c in cases]
    if any(s and s.get("correct") is False for s in seen):
        return True
    if seen and all(s and s.get("correct") is True for s in seen):
        return False
    return None


def _walk(node, path=()):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, path + (str(i),))


def parse_dcgm(text: str, exit_code: int | None = None, timed_out: bool = False) -> dict:
    """The diagnostic test's verdict from `dcgmi diag -j` output (schema-tolerant), with error codes and messages."""
    res = {"status": "unknown", "codes": [], "messages": [], "faulty_memory_elements": None, "nan_elements": None}
    m = re.search(r"Found (\d+) faulty memory elements", text)
    if m:
        res["faulty_memory_elements"] = int(m.group(1))
    m = re.search(r"Found (\d+) NaN-value memory elements", text)
    if m:
        res["nan_elements"] = int(m.group(1))
    statuses = []
    for obj in _json_objects(text):
        for _, node in _walk(obj):
            if not isinstance(node, dict):
                continue
            name = str(node.get("name", node.get("test_name", ""))).strip().lower()
            if name != "diagnostic":
                continue
            for _, sub in _walk(node):
                if isinstance(sub, dict):
                    for k, v in sub.items():
                        lk = k.lower()
                        if lk in ("status", "result") and isinstance(v, str):
                            statuses.append(v.strip().lower())
                        elif lk in ("error_id", "error_code", "code") and isinstance(v, int):
                            res["codes"].append(v)
                        elif lk in ("message", "msg", "warning", "info") and isinstance(v, str):
                            res["messages"].append(v[:300])
    if any("fail" in s for s in statuses) or res["faulty_memory_elements"] or res["nan_elements"]:
        res["status"] = "fail"
    elif any(s.startswith("pass") for s in statuses):
        res["status"] = "pass"
    elif any(s.startswith("warn") for s in statuses):
        res["status"] = "warn"
    elif any("skip" in s or "not run" in s or "not_run" in s for s in statuses):
        res["status"] = "skip"
    if timed_out:
        res["status"] = "timeout"
    elif res["status"] == "unknown" and exit_code not in (None, 0):
        res["status"] = "error"
    res["codes"] = sorted(set(res["codes"]))
    res["messages"] = res["messages"][:20]
    return res


def parse_burn(text: str, timed_out: bool = False) -> dict:
    """gpu-burn's verdict: 'GPU 0: OK' or 'GPU 0: FAULTY', and the largest error count it printed."""
    verdicts = re.findall(r"GPU \d+: (OK|FAULTY)", text)
    errors = [int(x) for x in re.findall(r"errors: (\d+)", text)]
    verdict = "FAULTY" if "FAULTY" in verdicts else "OK" if verdicts else "unknown"
    if "DIED" in text:
        verdict = "DIED"
    if timed_out and verdict == "unknown":
        verdict = "timeout"
    return {"verdict": verdict, "max_errors": max(errors) if errors else None}


def parse_check(text: str, out_dir: Path | None = None) -> dict:
    """`sdcguard check --json` result: verdict and wrong counts (stdout first, then result.json files)."""
    cands = [o for o in _json_objects(text) if "result" in o and "wrong" in o]
    if not cands and out_dir is not None and Path(out_dir).exists():
        for f in sorted(Path(out_dir).rglob("result.json")):
            try:
                o = json.loads(f.read_text())
            except ValueError:
                continue
            if "result" in o:
                cands.append(o)
    if not cands:
        return {"result": "unknown", "wrong": {}}
    o = cands[-1]
    return {"result": o.get("result"), "wrong": o.get("wrong") or {}, "notes": (o.get("notes") or [])[:10]}


def parse_ground(text: str) -> dict:
    """The last JSON line mps_unwritten.py printed, reduced to wrong counts."""
    objs = [o for o in _json_objects(text) if "share" in o or "fp16" in o]
    if not objs:
        return {"error": "no result", "tail": text[-1500:]}
    o = objs[-1]
    out = {"sms_reported": o.get("sms_reported"), "layer_wrong": {}, "math_modes_wrong": None, "in_job_alerts": {}}
    for dt in ("fp16", "bf16"):
        d = o.get(dt) or {}
        if "error" in d:
            out["layer_wrong"][dt] = "error"
            continue
        out["layer_wrong"][dt] = {k: d.get(k) for k in ("wrong_outputs", "wrong_outputs_nan_prefilled",
                                                         "backward_dX_wrong_outputs", "backward_dW_wrong_outputs",
                                                         "checked_outputs_wrong") if k in d}
        out["in_job_alerts"][dt] = {"training": (d.get("training_check") or {}).get("alerts"),
                                    "serving": (d.get("serving_check") or {}).get("alerts")}
    mm = o.get("math_modes") or {}
    if isinstance(mm, dict) and mm:
        out["math_modes_wrong"] = {k: v.get("wrong") for k, v in mm.items() if isinstance(v, dict)}
    return out


def ground_wrong(g: dict) -> bool | None:
    """Did PyTorch's own matmuls (what sdcguard uses) go wrong at this share?"""
    if not g or "error" in g:
        return None
    vals = []
    for d in (g.get("layer_wrong") or {}).values():
        if isinstance(d, dict):
            vals += [v for v in d.values() if isinstance(v, int)]
    vals += [v for v in (g.get("math_modes_wrong") or {}).values() if isinstance(v, int)]
    if not vals:
        return None
    return any(v > 0 for v in vals)


# ---------------------------------------------------------------- classification (rules in docs/METHOD.md)

DETECTED, MISSED, NOT_EXERCISED, FALSE_ALARM = "DETECTED", "MISSED", "NOT EXERCISED", "FALSE ALARM"
NOT_ATTRIBUTABLE, COULD_NOT_TEST = "FAIL (NOT ATTRIBUTABLE)", "COULD NOT TEST"
FLAGGED_NO_TRUTH, PASSED_NO_TRUTH = "FLAGGED (GROUND TRUTH UNKNOWN)", "PASSED (GROUND TRUTH UNKNOWN)"


def tool_flag(tool: str, parsed: dict) -> str:
    """'flagged', 'passed', 'not_attributable' or 'could_not_test' from one tool's parsed output."""
    if tool.startswith("dcgm"):
        st = parsed.get("status")
        if st == "fail":
            hit = (DCGM_FAULTY_MEMORY in parsed.get("codes", []) or DCGM_NAN_VALUE in parsed.get("codes", [])
                   or parsed.get("faulty_memory_elements") or parsed.get("nan_elements"))
            return "flagged" if hit else "not_attributable"
        return "passed" if st in ("pass", "warn") else "could_not_test"
    if tool.startswith("gpu_burn"):
        v = parsed.get("verdict")
        if v == "FAULTY" and (parsed.get("max_errors") or 0) > 0:
            return "flagged"
        return "passed" if v == "OK" else "could_not_test"
    if tool == "sdcguard_check":
        r = parsed.get("result")
        w = parsed.get("wrong") or {}
        if r == "FAIL":
            math = sum(int(w.get(k) or 0) for k in ("workload_training", "workload_serving", "stress", "math_modes"))
            return "flagged" if math > 0 else "not_attributable"
        if r == "PASS":
            return "passed"
        if r == "SUSPECT":
            return "not_attributable"          # recorded separately; a repeatability mismatch isn't a wrong count
        return "could_not_test"
    raise ValueError(tool)


def classify(flag: str, exercised: bool | None) -> str:
    if flag == "could_not_test":
        return COULD_NOT_TEST
    if flag == "not_attributable":
        return NOT_ATTRIBUTABLE
    if exercised is None:
        return FLAGGED_NO_TRUTH if flag == "flagged" else PASSED_NO_TRUTH
    if flag == "flagged":
        return DETECTED if exercised else FALSE_ALARM
    return MISSED if exercised else NOT_EXERCISED


def summarize_lt_log(path: Path, limit: int = 60) -> dict:
    """Unique cuBLASLt (computeType, types, shape, algo, cluster) lines with counts, from CUBLASLT_LOG_FILE."""
    if not path.exists():
        return {"lines": 0, "unique": [], "note": "no cuBLASLt log was written"}
    counts, total = {}, 0
    with open(path, errors="replace") as f:
        for line in f:
            if "algoId=" not in line and "algo=" not in line:
                continue
            total += 1
            key = re.sub(r"\[\d{4}-\d\d-\d\d[^\]]*\]|\bA=0x[0-9a-f]+|\bB=0x[0-9a-f]+|\bC=0x[0-9a-f]+|\bD=0x[0-9a-f]+"
                         r"|0x[0-9a-f]{6,}|\bthreadId=\d+|\bprocessId=\d+", "", line).strip()
            key = re.sub(r"\s+", " ", key)[:400]
            counts[key] = counts.get(key, 0) + 1
    uniq = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return {"lines": total, "unique": [{"count": c, "line": k} for k, c in uniq],
            "algo66_cluster4plus": sum(c for k, c in counts.items() if re.search(r"algoId=66\b", k)
                                       and re.search(r"CLUSTER_SHAPE_(?!1x1x1|2x1x1|1x2x1)\d", k))}


# ---------------------------------------------------------------- running (GPU container only)

def _env(share: int | None) -> dict:
    e = dict(os.environ, CUDA_MPS_PIPE_DIRECTORY=MPS_PIPE, CUDA_MPS_LOG_DIRECTORY=MPS_LOG, PYTHONUNBUFFERED="1")
    if share is not None:
        e["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(share)
    return e


def run_step(name: str, argv: list, out: Path, budget: float, env=None, cwd=None, floor: float = 20.0) -> dict:
    """Run argv in its own process group, output to files, killed (whole group) at the time limit."""
    rec = {"step": name, "argv": argv, "budget_s": round(budget, 1)}
    if budget < floor:
        rec.update(status="not started (no time left)")
        return rec
    out.mkdir(parents=True, exist_ok=True)
    so, se = out / f"{name}.stdout.txt", out / f"{name}.stderr.txt"
    t0 = time.monotonic()
    with open(so, "w") as fo, open(se, "w") as fe:
        try:
            p = subprocess.Popen(argv, stdout=fo, stderr=fe, env=env, cwd=cwd, start_new_session=True)
        except OSError as e:
            rec.update(status="could not start", error=str(e))
            return rec
        try:
            rec["exit"] = p.wait(timeout=budget)
            rec["timed_out"] = False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            rec["exit"], rec["timed_out"] = None, True
    rec["seconds"] = round(time.monotonic() - t0, 1)
    rec["stdout"], rec["stderr"] = so.read_text(errors="replace"), se.read_text(errors="replace")
    rec["status"] = "timed out" if rec["timed_out"] else "ran"
    return rec


def _tail(rec: dict) -> dict:
    r = dict(rec)
    for k in ("stdout", "stderr"):
        if k in r and isinstance(r[k], str) and len(r[k]) > TAIL:
            r[k] = "...(cut)...\n" + r[k][-TAIL:]
    return r


class MPS:
    """One MPS control daemon in this container; restarted with the share before every tool."""

    def __init__(self, out: Path):
        self.out, self.log = out, []
        os.makedirs(MPS_PIPE, exist_ok=True)
        os.makedirs(MPS_LOG, exist_ok=True)

    def stop(self):
        try:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", env=_env(None), capture_output=True,
                           text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            pass
        time.sleep(MPS_SETTLE[0])
        subprocess.run(["pkill", "-9", "-f", "nvidia-cuda-mps"], capture_output=True)
        time.sleep(MPS_SETTLE[1])

    def start(self, share: int) -> bool:
        self.stop()
        p = subprocess.run(["nvidia-cuda-mps-control", "-d"], env=_env(share), capture_output=True, text=True,
                           timeout=60)
        q = subprocess.run(["nvidia-cuda-mps-control"], input="get_default_active_thread_percentage\n",
                           env=_env(None), capture_output=True, text=True, timeout=30)
        self.log.append({"share": share, "start_exit": p.returncode, "start_output": (p.stdout + p.stderr)[-500:],
                         "default_share_reported": (q.stdout + q.stderr).strip()[-200:]})
        return p.returncode == 0


def run_all(out_root: str, deadline_s: float, commit=lambda: None, repo: str = "/root/sdcguard",
            probe: str = "/opt/h2h/probe", burn_dir: str = "/opt/gpu-burn") -> dict:
    t_start = time.monotonic()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    def left(reserve: float = 90.0) -> float:
        return deadline_s - (time.monotonic() - t_start) - reserve

    res = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "shares": {},
           "extras_40": {}, "recheck_100": {}, "cells": [], "notes": []}

    def save():
        (out / "progress.json").write_text(json.dumps(res, indent=1, default=str))
        commit()

    # ---- setup, no MPS
    for name, argv in (("nvidia_smi", ["nvidia-smi", "--query-gpu=name,uuid,pci.device_id,driver_version,"
                                       "compute_mode,memory.total", "--format=csv"]),
                       ("dcgmi_version", ["dcgmi", "--version"]),
                       ("torch", [sys.executable, "-c", "import torch, sdcguard; print(torch.__version__, "
                                  "torch.version.cuda, sdcguard.__file__)"])):
        r = run_step(name, argv, out / "setup", min(120.0, left()), env=_env(None))
        res["setup"][name] = _tail(r)
    dcgm_extra = ""
    smoke = run_step("dcgm_smoke", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j", "-p",
                                    "diagnostic.test_duration=30"], out / "setup", min(240.0, left()), env=_env(None))
    sp = parse_dcgm(smoke.get("stdout", "") + smoke.get("stderr", ""), smoke.get("exit"), smoke.get("timed_out", False))
    if sp["status"] == "skip":
        dcgm_extra = ";diagnostic.is_allowed=true"
        smoke = run_step("dcgm_smoke_allowed", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j", "-p",
                                                "diagnostic.test_duration=30" + dcgm_extra], out / "setup",
                         min(240.0, left()), env=_env(None))
        sp = parse_dcgm(smoke.get("stdout", "") + smoke.get("stderr", ""), smoke.get("exit"),
                        smoke.get("timed_out", False))
    dcgm_params_ok = sp["status"] in ("pass", "warn", "fail")
    if not dcgm_params_ok:                      # maybe it was the -p syntax: try once with DCGM's own defaults
        smoke = run_step("dcgm_smoke_defaults", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j"],
                         out / "setup", min(420.0, left()), env=_env(None))
        sp = parse_dcgm(smoke.get("stdout", "") + smoke.get("stderr", ""), smoke.get("exit"),
                        smoke.get("timed_out", False))
        dcgm_extra = ""
    dcgm_ok = sp["status"] in ("pass", "warn", "fail")
    res["setup"]["dcgm_smoke"] = {"step": _tail(smoke), "parsed": sp, "usable": dcgm_ok, "extra_params": dcgm_extra,
                                  "test_parameters_accepted": dcgm_params_ok}
    if not dcgm_ok:
        res["notes"].append("DCGM's diagnostic did not run in this container without MPS, so every DCGM cell is "
                            "COULD NOT TEST (not a miss).")
    save()

    mps = MPS(out)
    check_cmd = [sys.executable, "-m", "sdcguard.check", "--minutes", "5", "--offline", "--json", "--device", "cuda:0"]

    def tools_at(share: int, d: Path) -> dict:
        rec = {}
        mps.start(share)
        rec["ground"] = run_step("ground_truth", [sys.executable, f"{repo}/experiments/mps_unwritten.py"], d,
                                 min(200.0, left()), env=_env(share), cwd=repo)
        rec["ground_parsed"] = parse_ground(rec["ground"].get("stdout", ""))
        save()
        lt = d / "probe_cublaslt.log"
        penv = dict(_env(share), CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=str(lt))
        rec["probe"] = run_step("probe", [probe, "scrub", *PROBE_CASES], d, min(360.0, left()), env=penv)
        rec["probe_parsed"] = parse_probe(rec["probe"].get("stdout", ""))
        rec["probe_lt"] = summarize_lt_log(lt)
        save()
        if dcgm_ok:
            mps.start(share)
            rec["dcgm"] = run_step("dcgm", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j"] +
                                   (["-p", dcgm_extra.lstrip(";")] if dcgm_extra else []), d, min(480.0, left()),
                                   env=_env(share))
            rec["dcgm_parsed"] = parse_dcgm(rec["dcgm"].get("stdout", "") + rec["dcgm"].get("stderr", ""),
                                            rec["dcgm"].get("exit"), rec["dcgm"].get("timed_out", False))
            save()
        mps.start(share)
        rec["gpu_burn_tc"] = run_step("gpu_burn_tc", ["./gpu_burn", "-tc", "60"], d, min(170.0, left()),
                                      env=_env(share), cwd=burn_dir)
        rec["gpu_burn_tc_parsed"] = parse_burn(rec["gpu_burn_tc"].get("stdout", ""),
                                               rec["gpu_burn_tc"].get("timed_out", False))
        save()
        mps.start(share)
        rec["sdcguard_check"] = run_step("sdcguard_check", check_cmd + ["--out", str(d / "check")], d,
                                         min(660.0, left()), env=_env(share), cwd=repo)
        rec["sdcguard_check_parsed"] = parse_check(rec["sdcguard_check"].get("stdout", ""), d / "check")
        save()
        return rec

    for share in SHARES:
        if left() < 300:
            res["notes"].append(f"share {share}% not run: no time left")
            continue
        d = out / f"share_{share}"
        rec = tools_at(share, d)
        res["shares"][str(share)] = {k: (_tail(v) if isinstance(v, dict) and "argv" in v else v) for k, v in rec.items()}
        save()
        if share == 40 and left() > 400:
            x = {}
            if dcgm_ok and dcgm_params_ok:
                mps.start(40)
                x["dcgm_fp16_only"] = run_step("dcgm_fp16_only", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j",
                                               "-p", "diagnostic.precision=h;diagnostic.test_duration=60" + dcgm_extra],
                                               d, min(300.0, left()), env=_env(40))
                x["dcgm_fp16_only_parsed"] = parse_dcgm(x["dcgm_fp16_only"].get("stdout", "") +
                                                        x["dcgm_fp16_only"].get("stderr", ""),
                                                        x["dcgm_fp16_only"].get("exit"),
                                                        x["dcgm_fp16_only"].get("timed_out", False))
                save()
                mps.start(40)
                lt = d / "dcgm_cublaslt.log"
                x["dcgm_traced"] = run_step("dcgm_traced", ["dcgmi", "diag", "-r", "diagnostic", "-i", "0", "-j", "-p",
                                            "diagnostic.test_duration=20" + dcgm_extra], d, min(240.0, left()),
                                            env=dict(_env(40), CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=str(lt)))
                x["dcgm_traced_lt"] = summarize_lt_log(lt)
                save()
            mps.start(40)
            x["gpu_burn_default"] = run_step("gpu_burn_default", ["./gpu_burn", "60"], d, min(170.0, left()),
                                             env=_env(40), cwd=burn_dir)
            x["gpu_burn_default_parsed"] = parse_burn(x["gpu_burn_default"].get("stdout", ""),
                                                      x["gpu_burn_default"].get("timed_out", False))
            res["extras_40"] = {k: (_tail(v) if isinstance(v, dict) and "argv" in v else v) for k, v in x.items()}
            save()

    if left() > 200:
        d = out / "recheck_100"
        mps.start(100)
        r = run_step("probe", [probe, *PROBE_CASES], d, min(300.0, left()), env=_env(100))
        res["recheck_100"] = {"probe": _tail(r), "probe_parsed": parse_probe(r.get("stdout", ""))}
    mps.stop()
    res["mps_log"] = mps.log
    for name in ("control.log", "server.log"):
        p = Path(MPS_LOG) / name
        res.setdefault("mps_logs", {})[name] = p.read_text(errors="replace")[-6000:] if p.exists() else "(missing)"
    res["cells"] = cells(res)
    res["seconds"] = round(time.monotonic() - t_start, 1)
    save()
    return res


def cells(res: dict) -> list:
    """One row per (tool, share): the tool's verdict, its ground truth, and the pre-registered classification."""
    rows = []
    for share, rec in (res.get("shares") or {}).items():
        probe = rec.get("probe_parsed") or {}
        g = rec.get("ground_parsed") or {}
        torch_wrong = ground_wrong(g)
        ex16 = probe_wrong(probe, ("ex16:2048", "exbf16:2048"))
        sdc_ex = True if (torch_wrong or ex16) else (False if torch_wrong is False else None)
        for tool, key, ex in (("dcgm", "dcgm_parsed", probe_wrong(probe, EXERCISE["dcgm"])),
                              ("gpu_burn_tc", "gpu_burn_tc_parsed", probe_wrong(probe, EXERCISE["gpu_burn_tc"])),
                              ("sdcguard_check", "sdcguard_check_parsed", sdc_ex)):
            parsed = rec.get(key)
            if parsed is None:
                rows.append({"share": int(share), "tool": tool, "cell": COULD_NOT_TEST, "why": "not run"})
                continue
            flag = tool_flag(tool, parsed)
            rows.append({"share": int(share), "tool": tool, "flag": flag, "exercised": ex,
                         "cell": classify(flag, ex), "verdict": parsed})
        alerts = g.get("in_job_alerts") or {}
        if alerts:
            rows.append({"share": int(share), "tool": "sdcguard_in_job (Oct 19 probe)", "alerts": alerts,
                         "layer_wrong": g.get("layer_wrong"), "cell": "see layer_wrong vs alerts"})
    x = res.get("extras_40") or {}
    p40 = ((res.get("shares") or {}).get("40") or {}).get("probe_parsed") or {}
    for tool, key, cases in (("dcgm_fp16_only", "dcgm_fp16_only_parsed", EXERCISE["dcgm_fp16_only"]),
                             ("gpu_burn_default", "gpu_burn_default_parsed", EXERCISE["gpu_burn_default"])):
        if key in x:
            flag = tool_flag(tool, x[key])
            ex = probe_wrong(p40, cases)
            rows.append({"share": 40, "tool": tool, "flag": flag, "exercised": ex, "cell": classify(flag, ex),
                         "verdict": x[key]})
    return rows


def markdown(res: dict) -> str:
    lines = ["| share | tool | result | its own GEMMs wrong at this share? | verdict |", "|---|---|---|---|---|"]
    for r in res.get("cells", []):
        if "flag" not in r:
            if r.get("cell") == COULD_NOT_TEST:
                lines.append(f"| {r['share']}% | {r['tool']} | {COULD_NOT_TEST} | | {r.get('why', '')} |")
            continue
        v = r.get("verdict") or {}
        short = (v.get("status") or v.get("verdict") or v.get("result") or "")
        if r["tool"].startswith("dcgm") and v.get("faulty_memory_elements"):
            short += f" ({v['faulty_memory_elements']} 'faulty memory' elements)"
        if r["tool"].startswith("gpu_burn") and v.get("max_errors") is not None:
            short += f" (errors {v['max_errors']})"
        if r["tool"] == "sdcguard_check" and v.get("wrong"):
            short += f" (wrong {v['wrong']})"
        ex = {True: "yes", False: "no", None: "unknown"}[r.get("exercised")]
        lines.append(f"| {r['share']}% | {r['tool']} | **{r['cell']}** | {ex} | {short} |")
    return "\n".join(lines)


if __name__ == "__main__":                       # local dry use: python h2h_run.py OUT_DIR [minutes]
    mins = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
    r = run_all(sys.argv[1], mins * 60)
    print(markdown(r))
