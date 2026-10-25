"""`sdcguard isolation` on an open vLLM bug where a request's answer depends on the request before it (vLLM issue
#62124, reported October 17, 2026: FP8 per-tensor MoE with CUDA graphs; padding rows of the graph's batch carry
stale data into the per-tensor activation scale), and on two controls (docs/METHOD.md, "Answers that depend on the
request before", written before the run).

One L4 (sm_89, FP8 in hardware, as the reporter's L40S), ibm-granite/granite-3.1-1b-a400m-instruct (a 1B MoE), the
reporter's compilation settings, three servers one after another:

  bf16      no quantization                                        (the trusted stack)
  fp8pt     --quantization fp8_per_tensor --moe-backend triton     (the reported configuration)
  eager     the same with --enforce-eager                          (the reporter's control: 0 of 114 differed)

Each: `sdcguard isolation` (the primer probe: 8 short probes x 2 primer lengths x 6 primers, run alone one after
another, each request with its own cache_salt; and 64 canaries with shared prefixes, 32 at a time, a quarter
cancelled), and `gate model-check --url` against bf16's reference. Then `sdcguard decide` on bf16 -> fp8pt and
bf16 -> eager, with a policy that requires the isolation check and accepts a WARN (FP8 is an approximation). No
sdcguard plugin: the MoE experts aren't matmuls it checks, and this run is about isolation. After the servers,
vLLM's own Trace Replay (offline) scores bf16's recorded completions under each configuration, as its documentation
suggests for comparing quantization schemes: reported only.

    python experiments/vllm_isolation_check.py --out runs/isolation
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
COMPILE = json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8], "cudagraph_mode": "FULL_AND_PIECEWISE"})
ROLES = {
    "bf16": [],
    "fp8pt": ["--quantization", "fp8_per_tensor", "--moe-backend", "triton"],
    "eager": ["--quantization", "fp8_per_tensor", "--moe-backend", "triton", "--enforce-eager"],
}
POLICY = {"quality": {"accept_warn": True}, "isolation": {"required": True}}


FILE = {"bf16": "baseline", "fp8pt": "fp8pt", "eager": "eager"}     # role -> the name sdcguard qualify gives it
SERVE = (f"{MODEL} --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 "
         f"--compilation-config '{json.dumps(json.loads(COMPILE), separators=(',', ':'))}'")


def config(role: str) -> str:
    return " ".join([SERVE, *ROLES[role]])


def run_qualify(out: str, port: int) -> dict:
    """The whole run is one `sdcguard qualify` (the product's own command), without the plugin."""
    with open(os.path.join(out, "policy.json"), "w") as f:
        json.dump(POLICY, f)
    cmd = [sys.executable, "-m", "sdcguard", "qualify", "--out", out, "--policy", os.path.join(out, "policy.json"),
           "--baseline", config("bf16"), "--candidate", f"fp8pt={config('fp8pt')}",
           "--candidate", f"eager={config('eager')}", "--no-plugin", "--port", str(port), "--rounds", "2",
           "--stack", "gpu=L4"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    res = {"qualify_exit_code": p.returncode, "qualify_output": (p.stdout + p.stderr)[-4000:]}
    try:
        q = json.load(open(os.path.join(out, "QUALIFY.json")))
    except (OSError, ValueError):
        return res
    for role, name in FILE.items():
        res[role] = q["runs"].get(name) or {}
    res["decisions"] = q.get("decisions") or {}
    res["qualify_complete"] = q.get("complete", True)          # False: qualify stopped partway (its record so far)
    return res


TRACE_ARGS = {"bf16": [], "fp8pt": ["quantization=fp8_per_tensor", "moe_backend=triton"],
              "eager": ["quantization=fp8_per_tensor", "moe_backend=triton", "enforce_eager=true"]}


def traces(out: str) -> dict:
    """vLLM's own Trace Replay on bf16's recorded completions, offline, per configuration (reported only)."""
    res = {}
    for role, args in TRACE_ARGS.items():
        cmd = [sys.executable, os.path.join(vs.HERE, "trace_check.py"), os.path.join(out, "ref.json"), "--out",
               os.path.join(out, f"{role}_trace.json"), "--model", MODEL, "--max-len", "4096",
               "--vllm-arg", f"compilation_config={COMPILE}"]
        for x in args:
            cmd += ["--vllm-arg", x]
        if role != "bf16":
            cmd += ["--noise", os.path.join(out, "bf16_trace.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        r = {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-1500:]}
        try:
            t = json.load(open(os.path.join(out, f"{role}_trace.json")))
            r.update(verdict=t["verdict"], mean_kl=t["summary"].get("mean_kl"),
                     top1=t["summary"].get("top1_agreement"))
        except (OSError, ValueError, KeyError):
            pass
        res[role] = r
    return res


def judge(res: dict, out: str) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Answers that depend on the request before"."""
    fails, s = [], {}
    if res.get("qualify_complete") is False:
        fails.append("run: qualify stopped partway; its record so far isn't the run's")
    for role in ROLES:
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"1: the {role} server didn't start")
    if fails:
        return False, fails, s
    iso = {}
    for role in ROLES:
        try:
            iso[role] = json.load(open(os.path.join(out, f"{FILE[role]}_isolation.json")))
        except (OSError, ValueError):
            iso[role] = {}
        p, c = iso[role].get("primer") or {}, iso[role].get("canary") or {}
        s[role] = {"verdict": iso[role].get("verdict"), "primer": p.get("verdict"),
                   "primer_pairs": p.get("pairs"), "primer_pairs_differing": p.get("pairs_differing"),
                   "primer_cases_judged": p.get("judged"), "primer_cases": len(p.get("cases") or []),
                   "canary": c.get("verdict"), "canary_answered": c.get("answered"),
                   "canary_leaks": len(c.get("leaks") or [])}
    if s["fp8pt"]["primer"] != "LEAK":
        fails.append(f"2: the reported configuration's primer probe says {s['fp8pt']['primer']}, not LEAK")
    for role in ("eager", "bf16"):
        if s[role]["verdict"] != "NO LEAK SEEN":
            fails.append(f"3: {role} says {s[role]['verdict']}, not NO LEAK SEEN")
    for role in ROLES:
        if s[role]["canary_leaks"]:
            fails.append(f"4: {s[role]['canary_leaks']} canary leaks on {role}")
    dec = res.get("decisions") or {}
    fd = dec.get("fp8pt") or {}
    if fd.get("decision") != "REJECT" or not any(w.startswith("isolation") for w in fd.get("why") or []):
        fails.append(f"5: the decision on bf16 -> fp8pt is {fd.get('decision')}, not REJECT for isolation")
    s["decisions"] = dec
    s["trace_replay"] = {k: {x: v.get(x) for x in ("verdict", "mean_kl", "top1")}
                         for k, v in (res.get("traces") or {}).items()}
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res: dict = {"model": MODEL, "compile": COMPILE, "policy": POLICY}
    res.update(run_qualify(a.out, a.port))
    if (res.get("bf16") or {}).get("ready"):
        res["traces"] = traces(a.out)
    ok, fails, summary = judge(res, a.out)
    res.update(passed=ok, failures=fails, summary=summary)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": summary}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
