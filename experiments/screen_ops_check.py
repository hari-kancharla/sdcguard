"""sdcguard.screen_ops on a real GPU (docs/METHOD.md, "Exact operator tests for screening GPUs", written before the
run): a clean run of every test for some minutes, then one short run per test with one value made wrong in software,
which must be found and located.

    python experiments/screen_ops_check.py --out runs/screen_ops --minutes 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from sdcguard import screen_ops as so

ALL = [f"gemm_{p}" for p in so.GEMM] + ["fma_fp32", "fma_fp64", "sm_consistency", "memory"]
SIM_ELEMENT = {"fma_fp32": 5, "fma_fp64": 5, "sm_consistency": 3}


def run(out: str, minutes: float) -> dict:
    dev = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(dev), "capability": list(torch.cuda.get_device_capability(dev))}
    res["clean"] = so.run(dev, minutes, os.path.join(out, "clean"), seed=0)
    res["simulated"] = {}
    for t in [n for n in ALL if n in res["clean"]["tests"]]:
        r = so.run(dev, 0.0, os.path.join(out, f"simulate_{t}"), seed=5, simulate=t, tests=[t])
        res["simulated"][t] = {"result": r["result"], "wrong": r["wrong"],
                               "first_wrong": (r["tests"].get(t) or {}).get("first_wrong"), "error": r.get("error")}
    return res


def expected_skips(cap) -> set:
    """Tests a GPU of this compute capability can't run (Ada and newer run them all)."""
    cap = tuple(cap)
    out = set()
    if cap < (8, 9):
        out.add("gemm_fp8")
    if cap < (8, 0):
        out |= {"gemm_tf32", "gemm_bf16"}
    if cap < (7, 0):
        out.add("gemm_int8")
    return out


def judge(res: dict) -> tuple:
    fails = []
    c = res.get("clean") or {}
    skipped = set((c.get("skipped") or {}))
    if skipped != expected_skips(res.get("capability") or (0, 0)):
        fails.append(f"S1: tests skipped {sorted(skipped)}, expected {sorted(expected_skips(res.get('capability')))}: "
                     f"{c.get('skipped')}")
    if c.get("result") != "PASS" or c.get("wrong"):
        bad = {n: t["wrong"] for n, t in (c.get("tests") or {}).items() if t.get("wrong")}
        fails.append(f"S1: the clean run was {c.get('result')} ({c.get('wrong')} wrong values: {bad}; error "
                     f"{c.get('error')})")
    for n, t in (c.get("tests") or {}).items():
        if not t.get("values_checked") or not t.get("rounds"):
            fails.append(f"S1: {n} checked nothing")
    sims = res.get("simulated") or {}
    for n in (c.get("tests") or {}):
        s = sims.get(n) or {}
        fw = (s.get("first_wrong") or [{}])[0]
        if s.get("result") != "FAIL" or s.get("wrong") != 1:
            fails.append(f"S2: {n} with one value made wrong: {s.get('result')}, {s.get('wrong')} wrong")
        elif n == "memory":
            if fw.get("chunk") != 0 or fw.get("offset", 0) + fw.get("element", -1) != 12345:
                fails.append(f"S2: memory located the changed value at {fw}")
        elif fw.get("element") != SIM_ELEMENT.get(n, 777):
            fails.append(f"S2: {n} located the wrong value at {fw}")
    return not fails, fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=5.0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res = run(a.out, a.minutes)
    ok, fails = judge(res)
    res.update(passed=ok, failures=fails)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    c = res["clean"]
    print(json.dumps({"passed": ok, "failures": fails, "gpu": res["gpu"], "values_checked": c.get("values_checked"),
                      "rounds": c.get("rounds"), "skipped": c.get("skipped"),
                      "tests": {n: (t["rounds"], t["values_checked"], t["wrong"]) for n, t in c["tests"].items()}},
                     indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
