"""sdcguard.screen_ops on a real GPU (docs/METHOD.md, "Exact operator tests for screening GPUs" and its second
version, "Varied inputs and realistic-data matmuls in the screen", each written before its run): a clean run of every
test for some minutes, then one short run per test with one value made wrong in software, which must be found and
located; the realistic-data matmuls' worst clean score must stay under REAL_CLEAN_LIMIT.

    python experiments/screen_ops_check.py --out runs/screen_ops --minutes 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from sdcguard import screen_ops as so

ALL = [f"gemm_{p}" for p in so.GEMM] + list(so.REAL_DTYPE) + ["fma_fp32", "fma_fp64", "sm_consistency", "memory"]
SIM_ELEMENT = {"fma_fp32": 5, "fma_fp64": 5, "sm_consistency": 3}
REAL_CLEAN_LIMIT = 1.5         # the realistic-data matmuls' worst clean score (the line is 2; a correct output: <= 1)
SIM_SEED = 5                   # the simulations' seed: a realistic-data test's round 0 uses SIM_SEED * 13, REAL[0]


def run(out: str, minutes: float) -> dict:
    dev = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(dev), "capability": list(torch.cuda.get_device_capability(dev))}
    res["clean"] = so.run(dev, minutes, os.path.join(out, "clean"), seed=0)
    res["simulated"] = {}
    for t in sims(res["clean"]["tests"]):
        base = t.split(".")[0]
        r = so.run(dev, 0.0, os.path.join(out, f"simulate_{t}"), seed=SIM_SEED, simulate=t, tests=[base])
        res["simulated"][t] = {"result": r["result"], "wrong": r["wrong"],
                               "first_wrong": (r["tests"].get(base) or {}).get("first_wrong"), "error": r.get("error")}
    return res


def sims(tests) -> list:
    """The simulations for the tests a clean run ran: one per test, and for each realistic-data test also one whose
    value changes in its second run only (name.repeat), which the comparison of the two runs must find."""
    out = [n for n in ALL if n in tests]
    return out + [f"{n}.repeat" for n in so.REAL_DTYPE if n in tests]


def expected_skips(cap) -> set:
    """Tests a GPU of this compute capability can't run (Ada and newer run them all)."""
    cap = tuple(cap)
    out = set()
    if cap < (8, 9):
        out.add("gemm_fp8")
    if cap < (8, 0):
        out |= {"gemm_tf32", "gemm_bf16", "gemm_bf16_real"}
    if cap < (7, 0):
        out.add("gemm_int8")
    return out


def expected_cases(name: str) -> set:
    """The (input set or shape, transposed) cases a clean run of a matmul test must cover (none for the others)."""
    if name in so.REAL_DTYPE:
        return {(tuple(shape), tr) for shape in so.REAL for tr in (False, True)}
    if name.startswith("gemm_"):
        return {(s, tr) for s in range(so.SETS) for tr in (False, True)}
    return set()


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
    tests = c.get("tests") or {}
    want = set(ALL) - skipped
    if set(tests) != want:
        fails.append(f"S1: the clean run's tests {sorted(tests)} aren't every test not skipped: missing "
                     f"{sorted(want - set(tests))}, unexpected {sorted(set(tests) - want)}")
    refused = c.get("unsupported") or {}
    for n, t in tests.items():
        # a case cuBLAS refuses on this GPU (its own NOT_SUPPORTED, in both layouts) is listed, not run: covered
        listed = {(u.get("set"), bool(u.get("transposed"))) for u in refused.get(n) or []
                  if "NOT_SUPPORTED" in str(u.get("why"))}
        miss = sorted(map(str, expected_cases(n) - listed - {(tuple(k) if isinstance(k, list) else k, tr)
                                                             for k, tr in t.get("cases") or []}))
        if miss:
            fails.append(f"S1: {n} didn't run every input set and shape, plain and transposed: missing {miss}")
    for n, t in tests.items():
        if not t.get("values_checked") or not t.get("rounds"):
            fails.append(f"S1: {n} checked nothing")
        if n in so.REAL_DTYPE:
            w = t.get("worst_score")
            if not isinstance(w, (int, float)) or not w <= REAL_CLEAN_LIMIT:
                fails.append(f"S3: {n}'s worst clean score was {w}, above {REAL_CLEAN_LIMIT}")
    simulated = res.get("simulated") or {}
    for n in sims(c.get("tests") or {}):
        s = simulated.get(n) or {}
        fw = (s.get("first_wrong") or [{}])[0]
        if s.get("result") != "FAIL" or s.get("wrong") != 1:
            fails.append(f"S2: {n} with one value made wrong: {s.get('result')}, {s.get('wrong')} wrong")
        elif n == "memory":
            if fw.get("chunk") != 0 or fw.get("offset", 0) + fw.get("element", -1) != 12345:
                fails.append(f"S2: memory located the changed value at {fw}")
        elif n.split(".")[0] in so.REAL_DTYPE:
            rows, cols = so.real_samples(SIM_SEED * 13, so.REAL[0])     # where the simulation moved a value
            m_, n_, _ = so.REAL[0]
            r0, c0 = int(rows[0]), int(cols[0])
            if fw.get("seed") != SIM_SEED * 13 or fw.get("shape") != list(so.REAL[0]) or fw.get("round") != 0:
                fails.append(f"S2: {n} found a value somewhere else than the simulation's round, seed and shape: {fw}")
            elif n.endswith(".repeat"):
                if fw.get("check") != "repeat" or fw.get("element") != r0 * n_ + c0:
                    fails.append(f"S2: {n} located the wrong value at {fw} (made wrong at row {r0}, col {c0})")
            elif (fw.get("row"), fw.get("col")) != (r0, c0) or fw.get("check") != "reference":
                fails.append(f"S2: {n} located the wrong value at {fw} (made wrong at row {r0}, col {c0})")
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
                      "tests": {n: (t["rounds"], t["values_checked"], t["wrong"], *([t["worst_score"]] if "worst_score" in t
                                                                                    else []))
                                for n, t in c["tests"].items()}},
                     indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
