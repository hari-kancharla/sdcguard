"""One `sdcguard qualify` on vLLM issue #62124 and two remedies, with the FP8 quantization check on (docs/METHOD.md,
"An FP8 MoE bug from detection to remedy", written before the run).

#62124 (reported October 17, 2026): with FP8 per-tensor MoE and CUDA graphs, the padding rows of a graph's batch
carry an earlier request's data into the per-tensor activation scale, so a request's answer depends on the request
before it. One L4 (sm_89, FP8 in hardware, as the reporter's L40S), vLLM 0.32.0, ibm-granite/granite-3.1-1b-a400m-
instruct (a 1B MoE), vLLM's Triton MoE kernel, the reporter's compilation settings (CUDA graphs for batches 1, 2, 4
and 8). The baseline (bf16, twice: the noise floor) and three candidates:

  fp8pt     --quantization fp8_per_tensor                     the reported configuration
  eager     --quantization fp8_per_tensor --enforce-eager     remedy 1: no CUDA graphs (the reporter's control)
  fp8pc     --quantization fp8_per_channel                    remedy 2: per-token activation scales, CUDA graphs kept

every server with the plugin and SDCGUARD_FP8=1 (its FP8 quantization check: each MoE layer's expert inputs and
each FP8 linear layer's input, against the recipe recomputed over the step's real rows), and qualify's checks:
quality against the baseline, isolation, consistency, speed. The policy accepts a WARN (FP8 is an approximation)
and requires the isolation check. Then the rejected candidate's repro folder is run once on its own.

    python experiments/vllm_fp8_moe_check.py --out runs/fp8_moe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
COMPILE = json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8], "cudagraph_mode": "FULL_AND_PIECEWISE"})
BASE = f"{MODEL} --max-model-len 2048 --gpu-memory-utilization 0.85 --seed 0 --moe-backend triton " \
       f"--compilation-config '{COMPILE}'"
CANDIDATES = {"fp8pt": f"{BASE} --quantization fp8_per_tensor",
              "eager": f"{BASE} --quantization fp8_per_tensor --enforce-eager",
              "fp8pc": f"{BASE} --quantization fp8_per_channel"}
POLICY = {"quality": {"accept_warn": True}, "isolation": {"required": True}}
MOE_LAYERS = 24


def _alerts(path: str) -> list:
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _metric(out: str, name: str, metric: str):
    """The largest value of a metric over a server's metrics pages (None when none has it)."""
    vals = []
    for i in range(16):
        path = os.path.join(out, f"{name}_metrics{i}.txt")
        if not os.path.exists(path):
            continue
        for line in open(path):
            if line.startswith(metric + " "):
                try:
                    vals.append(float(line.split()[1]))
                except (IndexError, ValueError):
                    pass
    return max(vals) if vals else None


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    pol = os.path.join(out, "policy.json")
    with open(pol, "w") as f:
        json.dump(POLICY, f)
    cmd = [sys.executable, "-m", "sdcguard", "qualify", "--out", out, "--baseline", BASE, "--repeat-baseline",
           "--policy", pol, "--env", "SDCGUARD_FP8=1", "--port", str(port), "--metrics-port", str(metrics_port),
           "--stack", "gpu=L4"]
    for name, cfg in CANDIDATES.items():
        cmd += ["--candidate", f"{name}={cfg}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=5000)
    res: dict = {"cmd": cmd, "exit_code": p.returncode, "output": (p.stdout + p.stderr)[-6000:]}
    try:
        q = json.load(open(os.path.join(out, "QUALIFY.json")))
    except (OSError, ValueError) as e:
        res["error"] = f"no QUALIFY.json: {e}"
        return res
    res["runs"], res["decisions"] = q.get("runs") or {}, q.get("decisions") or {}
    res["fp8_alerts"] = {}
    for name in ("baseline", *CANDIDATES):
        al = [a for a in _alerts(os.path.join(out, f"{name}_alerts.jsonl")) if a.get("detector") == "serve_fp8"]
        res["fp8_alerts"][name] = {
            "count": len(al), "findings": sorted({(a.get("detail") or {}).get("finding") for a in al} - {None}),
            "moe_layers": sorted({(a.get("detail") or {}).get("where") for a in al
                                  if "(expert inputs)" in str((a.get("detail") or {}).get("where"))}),
            "implicated": [(a.get("detail") or {}).get("directly_implicated") for a in al[:5]]}
    # reported, not criteria: how many FP8 layers each server's check took, and the quantizations it judged
    res["fp8_coverage"] = {name: {"layers": _metric(out, name, "sdcguard_fp8_layers_checked"),
                                  "calls_judged": _metric(out, name, "sdcguard_fp8_calls_judged_total")}
                           for name in ("baseline", *CANDIDATES)}
    rp = (res["decisions"].get("fp8pt") or {}).get("repro")
    if rp and os.path.exists(os.path.join(rp, "rerun.sh")):
        r = subprocess.run(["sh", os.path.join(rp, "rerun.sh")], capture_output=True, text=True, timeout=1500,
                           env=dict(os.environ, PORT=str(port), MPORT=str(metrics_port), PY=sys.executable))
        iso = os.path.join(rp, "rerun_isolation.json")
        res["repro"] = {"exit_code": r.returncode, "output": (r.stdout + r.stderr)[-3000:],
                        "isolation": json.load(open(iso)).get("verdict") if os.path.exists(iso) else None,
                        "fp8_alerts": len([a for a in _alerts(os.path.join(rp, "rerun_alerts.jsonl"))
                                           if a.get("detector") == "serve_fp8"])}
    return res


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "An FP8 MoE bug from detection to remedy"."""
    fails = []
    runs, dec, fa = res.get("runs") or {}, res.get("decisions") or {}, res.get("fp8_alerts") or {}
    for name in ("baseline", "fp8pt", "eager"):
        if not (runs.get(name) or {}).get("ready"):
            fails.append(f"F1: {name} didn't start")
    d = dec.get("fp8pt") or {}
    if d.get("decision") != "REJECT" or d.get("isolation") != "LEAK":
        fails.append(f"F2: fp8pt was {d.get('decision')} with isolation {d.get('isolation')}, not REJECT on a LEAK")
    f = fa.get("fp8pt") or {}
    if "scale_from_outside_rows" not in (f.get("findings") or []) or not f.get("moe_layers"):
        fails.append(f"F3: the FP8 check didn't name an MoE layer's scale as taken from outside rows on fp8pt: {f}")
    if (fa.get("eager") or {}).get("count"):
        fails.append(f"F4: FP8 alerts on eager (no padding rows): {fa['eager']}")
    if (dec.get("eager") or {}).get("isolation") != "NO LEAK SEEN":
        fails.append(f"F4: eager's isolation was {(dec.get('eager') or {}).get('isolation')}, not NO LEAK SEEN")
    if (fa.get("baseline") or {}).get("count"):
        fails.append(f"F5: FP8 alerts on the bf16 baseline: {fa['baseline']}")
    rp = res.get("repro") or {}
    if rp.get("isolation") != "LEAK":
        fails.append(f"F6: the repro folder's rerun didn't reproduce the leak ({rp.get('isolation')})")
    s = {"decisions": {n: {k: (dec.get(n) or {}).get(k) for k in ("decision", "approximation", "isolation",
                                                                 "consistency", "throughput_change")} for n in dec},
         "fp8_alerts": {n: {"count": v["count"], "findings": v["findings"]} for n, v in fa.items()},
         "fp8pc": {"started": (runs.get("fp8pc") or {}).get("ready"), "fp8_alerts": (fa.get("fp8pc") or {}).get("count"),
                   "isolation": (dec.get("fp8pc") or {}).get("isolation")},
         "repro": {k: rp.get(k) for k in ("isolation", "fp8_alerts", "exit_code")},
         "fp8_coverage": res.get("fp8_coverage")}
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    res = run(a.out, a.port, a.metrics_port)
    ok, fails, s = judge(res)
    res.update(passed=ok, failures=fails, summary=s)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": s}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
