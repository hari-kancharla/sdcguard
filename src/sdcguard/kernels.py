"""sdcguard kernels: which GPU matmul kernels a workload really runs, and how many of them a test ever ran.

A GPU's health isn't one yes or no. cuBLAS chooses a kernel per call from the shape, the data types, the GPU and
the conditions the process runs under (an MPS client below 100% sees fewer SMs and gets other kernels), and a
library defect or a faulty unit can sit in one kernel and not the others. On October 20, 2026, on an H200 under
MPS, DCGM's diagnostic and gpu-burn passed while cuBLAS known issue 6862950 made the job's own matmuls wrong:
their GEMMs never ran the kernel that was broken (docs/METHOD.md, head-to-head). Passing a test says nothing about
a kernel the test didn't run.

This reads cuBLASLt's own log (CUBLASLT_LOG_LEVEL=5: one Trace line per matmul call, with its shapes, data types
and the exact kernel configuration: algorithm, tile, stages, split-K, custom option, cluster shape) and turns it
into an inventory: the distinct kernels a process ran, with the shapes it ran them on and their share of the
matmul work (2 m n k per call). Two inventories compare:

    sdcguard kernels record --out serve.json -- vllm serve Qwen/Qwen2.5-1.5B-Instruct     (Ctrl-C to stop)
    sdcguard kernels record --out burn.json --seconds 120 -- ./gpu_burn 60
    sdcguard kernels coverage serve.json --test burn.json      # which of the workload's kernels the test ran
    sdcguard kernels diff before.json after.json               # which kernels an upgrade changed
    sdcguard kernels show serve.json
    sdcguard kernels parse cublaslt.*.log --out inv.json       # logs you already have

The coverage report counts a workload kernel as exercised when a test ran the same configuration on the same data
types, and, more strictly, on the same shape too; it weights both by matmul work, lists the kernels no test ran,
and marks the ones whose configuration matches one of NVIDIA's listed wrong-result issues (known_issues.py).
Exercised is not the same as checked: a test that ran a kernel may still not have looked at its results (gpu-burn
compares the GPU with itself).

What it can't see: matmuls outside cuBLAS/cuBLASLt (Triton, CUTLASS, vLLM's Marlin and Machete INT4 kernels,
FlashAttention), and processes whose environment the logging variables don't reach (DCGM runs its diagnostic in
nv-hostengine, so the variables must be set where that process starts). Under CUDA graphs cuBLAS is called when a
graph is captured, not when it is replayed, so a server's inventory comes from its warm-up and capture plus the
calls it makes outside graphs; the counts are calls logged, not kernel launches. The log's format is NVIDIA's and
undocumented line by line; lines this can't read are counted, not guessed. The method of reading which kernel ran
from cuBLASLt's log follows Sangho Yeo's public reproduction of issue 6862950
(github.com/Yeosangho/mps-cublaslt-fp16-corruption-repro, October 2, 2026) and sdcguard's own gate.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Iterable, Optional

FORMAT = "sdcguard-kernels/1"
_TRACE = re.compile(r"\[Trace\]\[(cublasLt\w*Matmul)\]\s?(.*)$")
_FIELD = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")
_PAIR = re.compile(r"(\w+)=([^\s\]]+)")
_PREFIX = ("MATMUL_TILE_", "MATMUL_STAGES_", "CLUSTER_SHAPE_", "REDUCTION_SCHEME_", "CUBLASLT_", "EPILOGUE_")
# the parts of a call that name its kernel, in this order (others the log prints are kept as they come)
_ALGO_ORDER = ("algoId", "tile", "stages", "numSplitsK", "reductionScheme", "customOption", "clusterShape",
               "swizzle", "innerShape")
_TYPES = {"R_16F": "fp16", "R_16BF": "bf16", "R_32F": "fp32", "R_64F": "fp64", "R_8F_E4M3": "fp8e4m3",
          "R_8F_E5M2": "fp8e5m2", "R_8I": "int8", "R_32I": "int32", "R_4F_E2M1": "fp4e2m1"}
HALF = {"fp16", "bf16", "fp8e4m3", "fp8e5m2"}


def _short(v: str) -> str:
    for p in _PREFIX:
        if v.startswith(p):
            return v[len(p):]
    return v


def _bracket(text: str) -> dict:
    return {k: _short(v) for k, v in _PAIR.findall(text[1:-1])}


def parse_line(line: str) -> Optional[dict]:
    """One Trace line of a cuBLASLt matmul call as a dict, or None for any other line. Raises ValueError for a
    Trace line of a matmul that doesn't have the parts a call needs (descriptors of A and D, the kernel)."""
    m = _TRACE.search(line)
    if not m:
        return None
    fn, rest = m.groups()
    rec: dict = {"fn": fn}
    for key, val in _FIELD.findall(rest):
        rec[key] = _bracket(val) if val.startswith("[") else val
    a, d, algo = rec.get("Adesc"), rec.get("Ddesc"), rec.get("algo")
    if not (isinstance(a, dict) and isinstance(d, dict) and isinstance(algo, dict) and "algoId" in algo):
        raise ValueError("a cuBLASLt matmul Trace line without A and D descriptors or a kernel")
    comp = rec.get("computeDesc") if isinstance(rec.get("computeDesc"), dict) else {}
    try:
        m_, n_ = int(d["rows"]), int(d["cols"])
        k_ = int(a["rows"]) if comp.get("transa", "OP_N") == "OP_T" else int(a["cols"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"a cuBLASLt matmul Trace line with unreadable dimensions ({e})") from None
    batch = 1
    for desc in ("Adesc", "Ddesc"):
        bc = rec.get(desc, {}).get("batchCount") if isinstance(rec.get(desc), dict) else None
        if bc and bc.isdigit():
            batch = max(batch, int(bc))
    b, c = rec.get("Bdesc") if isinstance(rec.get("Bdesc"), dict) else {}, \
        rec.get("Cdesc") if isinstance(rec.get("Cdesc"), dict) else {}
    types = [_TYPES.get(x.get("type", "?"), x.get("type", "?")) for x in (a, b, c, d)]
    return {"fn": fn, "m": m_, "n": n_, "k": k_, "batch": batch, "types": types,
            "transa": comp.get("transa", "OP_N"), "transb": comp.get("transb", "OP_N"),
            "lda": a.get("ld"), "ldb": b.get("ld"), "ldd": d.get("ld"),
            "compute": comp.get("computeType", "?"), "scale": comp.get("scaleType", "?"),
            "epilogue": comp.get("epilogue", "DEFAULT"), "sm_count_target": comp.get("smCountTarget"),
            "algo": algo, "workspace": rec.get("workSpaceSizeInBytes")}


def kernel_key(call: dict) -> str:
    """The kernel a call ran: data types, compute type, epilogue and the full kernel configuration."""
    algo = call["algo"]
    parts = [f"{k}={algo[k]}" for k in _ALGO_ORDER if k in algo]
    parts += [f"{k}={algo[k]}" for k in sorted(algo) if k not in _ALGO_ORDER]
    t = call["types"]
    return (f"{t[0]}x{t[1]}->{t[3]} compute={call['compute']} epilogue={call['epilogue']} "
            f"{call['fn']} " + " ".join(parts))


def call_key(call: dict) -> str:
    """The kernel and the exact call: shape, batch, transposes, leading dimensions, the SM count it targeted."""
    return (f"{kernel_key(call)} | m={call['m']} n={call['n']} k={call['k']} batch={call['batch']} "
            f"{call['transa']}/{call['transb']} ld={call['lda']},{call['ldb']},{call['ldd']} "
            f"sms={call['sm_count_target'] or 'all'}")


def cluster_ctas(shape: Optional[str]) -> int:
    """CTAs per cluster from a shape such as 4x1x1 (0 when unknown or automatic)."""
    if not shape or not re.fullmatch(r"\d+x\d+x\d+", shape):
        return 0
    x, y, z = (int(v) for v in shape.split("x"))
    return x * y * z


def issue_flags(kernel: dict) -> list:
    """NVIDIA's listed wrong-result issues whose kernel conditions this kernel meets (known_issues.py has the
    version and GPU conditions; this only matches what the kernel configuration shows)."""
    algo, types = kernel["algo"], kernel["types"]
    flags = []
    if algo.get("algoId") == "66" and cluster_ctas(algo.get("clusterShape")) >= 4 and set(types[:2]) & HALF:
        sms = sorted({s for s in kernel.get("sm_count_targets", []) if s})
        flags.append({"issue": "6862950", "match": "kernel",
                      "why": "algorithm 66 with a cluster of 4 or more CTAs on FP16/BF16/FP8: wrong on Hopper when the "
                             "process is an MPS client below 100% of the SMs" +
                             (f" (the log shows it run with an SM count target of {', '.join(sms)})" if sms else "")})
    splits = algo.get("numSplitsK")
    if algo.get("algoId") == "66" and splits and splits.isdigit() and int(splits) > 1:
        ks = sorted({s["k"] for s in kernel["shapes"] if s["k"] % int(splits)})
        if ks:
            flags.append({"issue": "6580689", "match": "kernel and shape",
                          "why": f"algorithm 66 split-K into {splits} with K not divisible by it (K = "
                                 f"{', '.join(map(str, ks[:5]))}): wrong when K / {splits} is a multiple of the stage "
                                 "size; fixed in cuBLAS 13.8.0"})
    if "Grouped" in kernel["fn"]:
        flags.append({"issue": "6681084, 6798616", "match": "kind of call",
                      "why": "a cuBLASLt grouped GEMM: two listed issues give wrong grouped-GEMM results (Hopper with "
                             "host pointer mode and more than 2 waves; Blackwell and Rubin)"})
    return flags


def inventory(lines: Iterable[str], source: Optional[dict] = None) -> dict:
    """The inventory of the matmul calls in cuBLASLt log lines."""
    kernels: dict = {}
    calls = unreadable = 0
    for line in lines:
        try:
            c = parse_line(line)
        except ValueError:
            unreadable += 1
            continue
        if c is None:
            continue
        calls += 1
        work = 2 * c["m"] * c["n"] * c["k"] * c["batch"]
        kk = kernel_key(c)
        kern = kernels.setdefault(kk, {"key": kk, "fn": c["fn"], "types": c["types"], "compute": c["compute"],
                                       "epilogue": c["epilogue"], "algo": c["algo"], "calls": 0, "work": 0,
                                       "shapes": {}, "sm_count_targets": set()})
        kern["calls"] += 1
        kern["work"] += work
        kern["sm_count_targets"].add(c["sm_count_target"] or "")
        ck = call_key(c)
        s = kern["shapes"].setdefault(ck, {"key": ck, "m": c["m"], "n": c["n"], "k": c["k"], "batch": c["batch"],
                                           "transa": c["transa"], "transb": c["transb"],
                                           "ld": [c["lda"], c["ldb"], c["ldd"]],
                                           "sm_count_target": c["sm_count_target"], "calls": 0, "work": 0})
        s["calls"] += 1
        s["work"] += work
    out = []
    for kern in kernels.values():
        kern["sm_count_targets"] = sorted(kern["sm_count_targets"])
        kern["shapes"] = sorted(kern["shapes"].values(), key=lambda s: -s["work"])
        kern["issues"] = issue_flags(kern)
        out.append(kern)
    out.sort(key=lambda k: -k["work"])
    return {"format": FORMAT, "source": source or {}, "calls": calls, "unreadable_lines": unreadable,
            "kernels": out, "work": sum(k["work"] for k in out)}


def _keys(inv: dict, level: str) -> dict:
    """key -> work, at 'kernel' or 'call' level."""
    if level == "kernel":
        return {k["key"]: k["work"] for k in inv["kernels"]}
    return {s["key"]: s["work"] for k in inv["kernels"] for s in k["shapes"]}


def coverage(workload: dict, tests: list) -> dict:
    """How much of the workload's kernels (and exact calls) the tests ran, by count and by matmul work."""
    res: dict = {"workload_calls": workload["calls"], "tests": []}
    for t in tests:
        res["tests"].append({"source": t.get("source", {}), "calls": t["calls"], "kernels": len(t["kernels"]),
                             "unreadable_lines": t.get("unreadable_lines", 0)})
    for level in ("kernel", "call"):
        w = _keys(workload, level)
        seen: set = set()
        for t in tests:
            seen |= set(_keys(t, level))
        hit = [k for k in w if k in seen]
        total = sum(w.values())
        res[level] = {"workload": len(w), "exercised": len(hit),
                      "work_share_exercised": (sum(w[k] for k in hit) / total) if total else None}
    tested = set().union(*(_keys(t, "kernel") for t in tests)) if tests else set()
    res["not_exercised"] = [{"key": k["key"], "work_share": (k["work"] / workload["work"]) if workload["work"] else None,
                             "calls": k["calls"], "shapes": len(k["shapes"]), "issues": k["issues"]}
                            for k in workload["kernels"] if k["key"] not in tested]
    res["notes"] = []
    if workload["calls"] == 0:
        res["notes"].append("the workload's log has no cuBLASLt matmul calls: nothing to cover (was the logging set "
                            "in the process that ran the matmuls?)")
    for t in res["tests"]:
        if t["calls"] == 0:
            res["notes"].append(f"a test's log ({t['source'].get('command') or t['source'].get('logs') or '?'}) has no "
                                "cuBLASLt matmul calls: either it ran none, or the logging didn't reach the process "
                                "that did (DCGM runs its diagnostic in nv-hostengine)")
    return res


def diff(before: dict, after: dict) -> dict:
    """Kernels the 'after' inventory runs that 'before' didn't (new, unverified paths), and the reverse."""
    b, a = _keys(before, "kernel"), _keys(after, "kernel")
    total = sum(a.values())
    new = [k for k in a if k not in b]
    gone = [k for k in b if k not in a]
    issues = {k["key"]: k["issues"] for k in after["kernels"]}
    return {"kernels_before": len(b), "kernels_after": len(a), "same": len(a) - len(new),
            "new": [{"key": k, "work_share": a[k] / total if total else None, "issues": issues.get(k, [])} for k in new],
            "gone": gone, "work_share_on_new_kernels": (sum(a[k] for k in new) / total) if total else None}


# ---------------------------------------------------------------- recording
def _gpus() -> list:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [dict(zip(("index", "name", "uuid", "driver"), [v.strip() for v in row.split(",")]))
            for row in out.strip().splitlines() if row.count(",") >= 3]


def read_logs(paths: list) -> list:
    lines: list = []
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines.extend(f.read().splitlines())
    return lines


def record(cmd: list, log_dir: str, seconds: Optional[float] = None, level: str = "5") -> dict:
    """Run cmd with cuBLASLt's log on (one file per process: %i is the process id), until it exits, Ctrl-C, or
    `seconds` (then SIGINT to its process group, SIGTERM after 30 s, SIGKILL after 30 more). Returns its inventory."""
    os.makedirs(log_dir, exist_ok=True)
    env = {**os.environ, "CUBLASLT_LOG_LEVEL": level, "CUBLASLT_LOG_FILE": os.path.join(log_dir, "cublaslt.%i.log")}
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    stopped = None
    try:
        p.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        stopped = "time"
    except KeyboardInterrupt:
        stopped = "ctrl-c"
    if p.poll() is None:
        for sig, wait in ((signal.SIGINT, 30), (signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                break
            try:
                p.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
    logs = sorted(glob.glob(os.path.join(log_dir, "cublaslt.*.log")))
    source = {"command": " ".join(cmd), "exit_code": p.returncode, "stopped_by": stopped,
              "seconds": round(time.time() - t0, 1), "logs": logs, "gpus": _gpus(),
              "env": {k: os.environ.get(k) for k in ("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "CUDA_VISIBLE_DEVICES")
                      if os.environ.get(k) is not None}, "log_level": level,
              "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return inventory(read_logs(logs), source)


# ---------------------------------------------------------------- command line
def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _print_inventory(inv: dict, top: int = 15) -> None:
    print(f"{inv['calls']:,} cuBLASLt matmul calls logged, {len(inv['kernels'])} distinct kernels, "
          f"{sum(len(k['shapes']) for k in inv['kernels'])} distinct calls (kernel and shape)"
          + (f"; {inv['unreadable_lines']} unreadable lines" if inv.get("unreadable_lines") else ""))
    for k in inv["kernels"][:top]:
        share = k["work"] / inv["work"] if inv["work"] else None
        print(f"  {_pct(share):>6} of the work, {k['calls']:,} calls, {len(k['shapes'])} shapes: {k['key']}")
        for f in k["issues"]:
            print(f"         matches issue {f['issue']} ({f['match']}): {f['why']}")
    if len(inv["kernels"]) > top:
        print(f"  ... and {len(inv['kernels']) - top} more (--json for all)")


def _print_coverage(r: dict) -> None:
    k, c = r["kernel"], r["call"]
    print(f"The tests ran {k['exercised']} of the workload's {k['workload']} kernels "
          f"({_pct(k['work_share_exercised'])} of its matmul work), and {c['exercised']} of its {c['workload']} exact "
          f"calls ({_pct(c['work_share_exercised'])} of the work).")
    for t in r["tests"]:
        print(f"  test: {t['source'].get('command') or t['source'].get('logs') or '?'}: {t['calls']:,} calls, "
              f"{t['kernels']} kernels")
    if r["not_exercised"]:
        print("Kernels no test ran (largest share of the work first):")
        for n in r["not_exercised"][:20]:
            print(f"  {_pct(n['work_share']):>6}, {n['calls']:,} calls: {n['key']}")
            for f in n["issues"]:
                print(f"         matches issue {f['issue']} ({f['match']}): {f['why']}")
        if len(r["not_exercised"]) > 20:
            print(f"  ... and {len(r['not_exercised']) - 20} more (--json for all)")
    for note in r["notes"]:
        print(f"note: {note}")
    print("Exercised means a test ran the same kernel configuration, not that it checked the results. Only cuBLAS and "
          "cuBLASLt matmuls are listed.")


def _load(path: str) -> dict:
    with open(path) as f:
        inv = json.load(f)
    if inv.get("format") != FORMAT:
        raise ValueError(f"{path} isn't a {FORMAT} inventory")
    return inv


def _save(inv: dict, path: Optional[str]) -> None:
    if path:
        with open(path, "w") as f:
            json.dump(inv, f, indent=1)
        print(f"saved {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard kernels", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="run a command with cuBLASLt's log on and save its kernel inventory")
    r.add_argument("--out", required=True)
    r.add_argument("--logs", help="folder for the raw logs (default: next to --out)")
    r.add_argument("--seconds", type=float, help="stop the command after this long (SIGINT, then SIGTERM)")
    r.add_argument("--level", default="5", help="CUBLASLT_LOG_LEVEL (default 5, which prints the Trace lines)")
    r.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    pa = sub.add_parser("parse", help="an inventory from cuBLASLt logs you already have")
    pa.add_argument("logs", nargs="+")
    pa.add_argument("--out")
    s = sub.add_parser("show", help="print an inventory")
    s.add_argument("inventory")
    s.add_argument("--json", action="store_true")
    c = sub.add_parser("coverage", help="how much of a workload's kernels the tests ran")
    c.add_argument("workload")
    c.add_argument("--test", action="append", required=True, help="a test's inventory (repeat for several)")
    c.add_argument("--json", action="store_true")
    d = sub.add_parser("diff", help="kernels an upgrade or change added or dropped")
    d.add_argument("before")
    d.add_argument("after")
    d.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "record":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not cmd:
            p.error("give the command after --")
        inv = record(cmd, a.logs or (os.path.splitext(a.out)[0] + "_logs"), a.seconds, a.level)
        _save(inv, a.out)
        _print_inventory(inv)
        return 0 if inv["calls"] else 2
    if a.cmd == "parse":
        inv = inventory(read_logs(a.logs), {"logs": list(a.logs)})
        _save(inv, a.out)
        _print_inventory(inv)
        return 0 if inv["calls"] else 2
    if a.cmd == "show":
        inv = _load(a.inventory)
        print(json.dumps(inv, indent=1)) if a.json else _print_inventory(inv)
        return 0
    if a.cmd == "coverage":
        res = coverage(_load(a.workload), [_load(t) for t in a.test])
        print(json.dumps(res, indent=1)) if a.json else _print_coverage(res)
        return 0 if res["workload_calls"] else 2
    res = diff(_load(a.before), _load(a.after))
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"{res['kernels_after']} kernels after, {res['same']} the same as before, {len(res['new'])} new "
              f"({_pct(res['work_share_on_new_kernels'])} of the work now runs on kernels the old stack never ran), "
              f"{len(res['gone'])} no longer used.")
        for n in res["new"][:20]:
            print(f"  new, {_pct(n['work_share'])}: {n['key']}")
            for f in n["issues"]:
                print(f"         matches issue {f['issue']} ({f['match']}): {f['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
