"""One `sdcguard qualify` on vLLM issue #54035, open on the current release (docs/METHOD.md, "An FP8 KV-cache decode
bug through qualify", written before the run).

#54035: with an FP8 KV cache on Hopper, FlashAttention 3 tiles decode and prefill differently (kBlockN 96 against
192 for head size 128), so from 97 tokens of KV on, a token's log-probability when decoded differs from the same
token's when the text is scored in one prefill (the reporter: up to 0.5 nats on an H100 with Qwen3-4B, none with a
bf16 cache). A quality check scores prefill only, and a bf16-vs-FP8 comparison differs anyway; `sdcguard
consistency`'s decode-vs-prefill probe compares the two paths of one server, and qualify judges the candidate's gap
against the baseline's own.

One H100, vLLM 0.32.0, Qwen/Qwen3-4B in bf16, prefix caching on (vLLM's default). The baseline (bf16 KV cache, the
FlashAttention backend) and two candidates:

  fp8kv     --kv-cache-dtype fp8                                   the reported configuration
  fp8kv_fi  --kv-cache-dtype fp8 --attention-backend FLASHINFER    another kernel for the same FP8 cache

with the plugin on every server (its linear-layer checks; the KV cache isn't something it checks), and qualify's
checks: quality, isolation, consistency, speed. The policy accepts a WARN (an FP8 cache is an approximation).

    python experiments/vllm_fp8kv_check.py --out runs/fp8kv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

MODEL = "Qwen/Qwen3-4B"
BASE = f"{MODEL} --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0"
CANDIDATES = {"fp8kv": f"{BASE} --kv-cache-dtype fp8",
              "fp8kv_fi": f"{BASE} --kv-cache-dtype fp8 --attention-backend FLASHINFER"}
POLICY = {"quality": {"accept_warn": True}}


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    pol = os.path.join(out, "policy.json")
    with open(pol, "w") as f:
        json.dump(POLICY, f)
    cmd = [sys.executable, "-m", "sdcguard", "qualify", "--out", out, "--baseline", BASE, "--policy", pol,
           "--port", str(port), "--metrics-port", str(metrics_port), "--stack", "gpu=H100"]
    for name, cfg in CANDIDATES.items():
        cmd += ["--candidate", f"{name}={cfg}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=4200)
    res: dict = {"cmd": cmd, "exit_code": p.returncode, "output": (p.stdout + p.stderr)[-6000:]}
    try:
        q = json.load(open(os.path.join(out, "QUALIFY.json")))
    except (OSError, ValueError) as e:
        res["error"] = f"no QUALIFY.json: {e}"
        return res
    res["runs"], res["decisions"] = q.get("runs") or {}, q.get("decisions") or {}
    res["consistency"] = {}
    for name in ("baseline", *CANDIDATES):
        try:
            c = json.load(open(os.path.join(out, f"{name}_consistency.json")))
        except (OSError, ValueError):
            continue
        dp = c.get("decode_prefill") or {}
        res["consistency"][name] = {
            "verdict": c.get("verdict"), "decode_largest_gap": dp.get("largest_gap"),
            "first_over_min_gap_kv": [((x.get("first_over_min_gap") or {}).get("kv_length")) for x in
                                      dp.get("cases") or []],
            "kv_length_at_max": [x.get("kv_length_at_max") for x in dp.get("cases") or []],
            "prefix_cache": (c.get("prefix_cache") or {}).get("verdict"),
            "cache_hits_confirmed": (c.get("prefix_cache") or {}).get("cache_hits_confirmed")}
    return res


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "An FP8 KV-cache decode bug through qualify"."""
    fails = []
    runs, dec, con = res.get("runs") or {}, res.get("decisions") or {}, res.get("consistency") or {}
    for name in ("baseline", "fp8kv"):
        if not (runs.get(name) or {}).get("ready"):
            fails.append(f"K1: {name} didn't start")
    b = con.get("baseline") or {}
    if b.get("verdict") != "CONSISTENT":
        fails.append(f"K2: the baseline's own consistency was {b.get('verdict')} (decode gap "
                     f"{b.get('decode_largest_gap')}), not CONSISTENT: the probe can't judge a candidate against it")
    d = dec.get("fp8kv") or {}
    if d.get("consistency") != "INCONSISTENT" or d.get("decision") != "REJECT":
        fails.append(f"K3: fp8kv's consistency was {d.get('consistency')} and its decision {d.get('decision')}, not "
                     f"INCONSISTENT and REJECT (decode gap {(con.get('fp8kv') or {}).get('decode_largest_gap')} "
                     f"against the baseline's {b.get('decode_largest_gap')})")
    s = {"decisions": {n: {k: (dec.get(n) or {}).get(k) for k in ("decision", "approximation", "isolation",
                                                                 "consistency", "throughput_change")} for n in dec},
         "consistency": con}
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
