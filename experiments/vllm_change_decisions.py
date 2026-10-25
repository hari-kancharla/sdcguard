"""`sdcguard decide` on real changes to a real `vllm serve` (docs/METHOD.md, "Decisions on four changes to a vLLM
server", written before the run).

One H100, vLLM 0.32.0, Qwen2.5-7B-Instruct in bf16 as the trusted stack, sdcguard on in every server (SDCGUARD=1,
SDCGUARD_REQUIRE=1) on every linear layer, as qualify runs it: a decision counts the plugin's silence only when it
checks at least half the weights, and a share of 1/8 checks the same 15 layers every step (the first run's fault
layer wasn't one of them; docs/METHOD.md, amendment). Each role is one server launch:

  base        the trusted stack: the quality reference (`gate model-ref --url`), the perf runs, the plugin's metrics
  plain       the trusted stack without sdcguard: perf only, for what sdcguard on every layer costs (reported)
  same        the same stack started again: the control (and the quality check's noise run)
  triton      --attention-backend TRITON_ATTN
  awq         Qwen/Qwen2.5-7B-Instruct-AWQ (INT4 weights, vLLM's Machete kernels on Hopper)
  fault       the trusted stack with one weight of model.layers.10.mlp.down_proj multiplied by 2^8 after start (what
              flipping bit 10 of a bf16 weight does when it is 0), through /collective_rpc

Per role, after the server is up: `gate model-check --url` against the reference (teacher-forced next-token
distributions, with the same role as noise), `sdcguard isolation` (reported; no policy here requires it),
`sdcguard perf` at 16 requests at a time (128 prompts, 128 tokens, 3
rounds, the decision's load) and at 1 at a time (16 prompts, reported only), then each worker's sdcguard metrics and
the alerts from SDCGUARD_LOG. The fault is injected before the quality check and perf. Then `sdcguard decide` under
two policies written before the run (and, after the servers, vLLM's own Trace Replay on the recorded completions
offline: the trusted stack, the faulty one and the AWQ model; reported only):

  hold      a change may cost neither quality nor speed: quality PASS, throughput no lower (within the noise)
  speed     a change made for speed: at least 10% more throughput, quality WARN accepted

    python experiments/vllm_change_decisions.py --out runs/decisions
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B-Instruct"
FAULT_LAYER = "model.layers.10.mlp.down_proj"
FAULT_EXP = 8
ROLES = {                      # role: (model, extra vllm serve arguments)
    "base": (MODEL, []),
    "plain": (MODEL, []),      # the trusted stack without sdcguard: what sdcguard at 1 layer in 8 costs (reported)
    "same": (MODEL, []),
    "triton": (MODEL, ["--attention-backend", "TRITON_ATTN"]),
    "awq": ("Qwen/Qwen2.5-7B-Instruct-AWQ", []),
    "fault": (MODEL, []),
}
POLICIES = {
    "hold": {"quality": {"accept_warn": False},
             "performance": {"required": True, "min_throughput_gain": 0.0}},
    "speed": {"quality": {"accept_warn": True},
              "performance": {"required": True, "min_throughput_gain": 0.10}},
}
EXPECTED = {               # (policy, role) -> decision written before the run; None: reported as found
    ("hold", "same"): "ACCEPT",
    ("hold", "fault"): "REJECT",
    ("speed", "fault"): "REJECT",
}


def _sdc(*args, timeout: int = 1800) -> dict:
    p = subprocess.run([sys.executable, "-m", "sdcguard", *args], capture_output=True, text=True, timeout=timeout)
    return {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-3000:]}


def _start(role: str, out: str, port: int, metrics_port: int):
    model, extra = ROLES[role]
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if role == "plain":
        for k in [k for k in env if k.startswith("SDCGUARD")]:
            del env[k]
    else:
        # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
        env["SDCGUARD_ATTN"] = "0"
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_SHARE="1", SDCGUARD_METRICS_PORT=str(metrics_port),
                   SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
                   SDCGUARD_EVIDENCE_DIR=os.path.join(out, f"{role}_evidence"))
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port),
           "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
           "--seed", "0", "--worker-extension-cls", "sdcguard_test_ext.FaultInjection", *extra]
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def run_role(role: str, out: str, port: int, metrics_port: int) -> dict:
    url = f"http://127.0.0.1:{port}"
    res: dict = {"role": role, "model": ROLES[role][0], "args": ROLES[role][1]}
    proc, log = _start(role, out, port, metrics_port)
    try:
        t0 = time.monotonic()
        res["ready"] = vs.wait_ready(port, proc, timeout=1800)
        res["startup_s"] = round(time.monotonic() - t0, 1)
        if not res["ready"]:
            res["exit_code"] = proc.poll()
            return res
        stack = ["--stack", "gpu=H100", "--stack", f"role={role}", "--stack", f"model_id={ROLES[role][0]}",
                 "--stack", "args=" + json.dumps(ROLES[role][1])]
        if role == "fault":
            status, raw = vs._request(port, "POST", "/collective_rpc",
                                      {"method": "sdcguard_test_scale", "args": [FAULT_LAYER, FAULT_EXP],
                                       "timeout": 60})
            res["fault"] = {"status": status, "results": json.loads(raw).get("results") if status == 200 else
                            raw.decode("utf-8", "replace")[:300], "injected_at": time.time()}
        ref = os.path.join(out, "ref.json")
        if role == "plain":
            pass
        elif role == "base":
            res["model_ref"] = _sdc("gate", "model-ref", "--url", url, "--out", ref, *stack)
        else:
            noise = [] if role == "same" else ["--noise", os.path.join(out, "same_quality.json")]
            res["model_check"] = _sdc("gate", "model-check", ref, "--url", url, "--out",
                                      os.path.join(out, f"{role}_quality.json"), *noise, *stack)
        if role != "plain":
            res["isolation"] = _sdc("isolation", "--url", url, "--out", os.path.join(out, f"{role}_isolation.json"))
        res["perf16"] = _sdc("perf", "--url", url, "--concurrency", "16", "--rounds", "3", "--out",
                             os.path.join(out, f"{role}_perf16.json"))
        prompts = os.path.join(out, "prompts16.txt")
        if not os.path.exists(prompts):
            from sdcguard.perf import default_prompts
            with open(prompts, "w") as f:
                f.write("\n".join(p.replace("\n", " ") for p in default_prompts(16, seed=1)) + "\n")
        res["perf1"] = _sdc("perf", "--url", url, "--concurrency", "1", "--rounds", "3", "--warmup", "2",
                            "--prompts-file", prompts, "--out", os.path.join(out, f"{role}_perf1.json"))
        if role == "plain":
            return res
        time.sleep(5)
        vs.complete(port, ROLES[role][0], "One more request, so the last results are read.", stream=False)
        time.sleep(3)
        try:
            status, raw = vs._request(metrics_port, "GET", "/metrics", timeout=10)
            with open(os.path.join(out, f"{role}_metrics.txt"), "wb") as f:
                f.write(raw)
            res["metrics_status"] = status
        except Exception as e:  # noqa: BLE001
            res["metrics_status"] = f"{type(e).__name__}: {e}"
    finally:
        res["stop"] = vs.stop(proc)
        log.close()
    return res


TRACES = {   # vLLM's own Trace Replay on the recorded completions, offline, after the servers (reported only)
    "base": (MODEL, None, None),
    "fault": (MODEL, f"{FAULT_LAYER}:{FAULT_EXP}", "base"),
    "awq": ("Qwen/Qwen2.5-7B-Instruct-AWQ", None, "base"),
}


def traces(out: str) -> dict:
    res = {}
    ref = os.path.join(out, "ref.json")
    for role, (model, fault, noise) in TRACES.items():
        cmd = [sys.executable, os.path.join(vs.HERE, "trace_check.py"), ref, "--out",
               os.path.join(out, f"{role}_trace.json"), "--model", model, "--max-len", "4096"]
        if fault:
            cmd += ["--fault", fault]
        if noise:
            cmd += ["--noise", os.path.join(out, f"{noise}_trace.json")]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        r = {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-1500:]}
        try:
            t = json.load(open(os.path.join(out, f"{role}_trace.json")))
            r.update(verdict=t["verdict"], mean_kl=t["summary"].get("mean_kl"),
                     top1=t["summary"].get("top1_agreement"), fault=t.get("fault"))
        except (OSError, ValueError, KeyError):
            pass
        res[role] = r
    return res


def decide_all(out: str, roles: list) -> dict:
    """Every role but base against base, under each policy (`sdcguard decide`, files only)."""
    from sdcguard import decide as dc
    rd = lambda p: json.load(open(p)) if os.path.exists(p) else None  # noqa: E731
    base_perf = rd(os.path.join(out, "base_perf16.json"))
    decisions: dict = {}
    for pol_name, pol in POLICIES.items():
        for role in roles:
            if role in ("base", "plain"):
                continue
            alerts_path = os.path.join(out, f"{role}_alerts.jsonl")
            alerts = dc._read_jsonl(alerts_path) if os.path.exists(alerts_path) else []
            met = os.path.join(out, f"{role}_metrics.txt")
            cov = dc.plugin_coverage([open(met).read()]) if os.path.exists(met) else None
            d = dc.decide(pol, quality=rd(os.path.join(out, f"{role}_quality.json")), alerts=alerts,
                          perf_before=base_perf, perf_after=rd(os.path.join(out, f"{role}_perf16.json")), plugin=cov,
                          isolation=rd(os.path.join(out, f"{role}_isolation.json")))
            name = f"{pol_name}_{role}"
            with open(os.path.join(out, f"decision_{name}.json"), "w") as f:
                json.dump(d, f, indent=1, default=str)
            with open(os.path.join(out, f"decision_{name}.txt"), "w") as f:
                f.write(dc.render(d) + "\n")
            decisions[name] = {"decision": d["decision"], "why": d["why"],
                               "arithmetic": d["arithmetic"]["verdict"],
                               "weights": (d.get("weights") or {}).get("verdict"),
                               "approximation": d["approximation"]["verdict"],
                               "quality_check": d["approximation"].get("check"),
                               "performance": d["performance"]["verdict"],
                               "isolation": d["isolation"]["verdict"],
                               "throughput_change": ((d["performance"].get("metrics") or {})
                                                     .get("output_tokens_per_s") or {}).get("change")}
    return decisions


def _kv_tokens(out: str, role: str):
    """The KV cache size vLLM reports at startup (what sdcguard's memory takes from serving shows up here)."""
    try:
        text = open(os.path.join(out, f"{role}_server.log"), errors="replace").read()
    except OSError:
        return None
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    return int(m[-1].replace(",", "")) if m else None


def cost(res: dict, out: str) -> dict:
    """sdcguard at 1 layer in 8 against the same stack without it: throughput and latency at both loads, KV cache
    size, and for the fault, seconds from injection to the first alert (reported only)."""
    from sdcguard.perf import compare
    rd = lambda p: json.load(open(p)) if os.path.exists(p) else None  # noqa: E731
    c: dict = {}
    for load in ("perf16", "perf1"):
        a, b = rd(os.path.join(out, f"plain_{load}.json")), rd(os.path.join(out, f"base_{load}.json"))
        c[load] = compare(a, b)["metrics"] if a and b else None
    c["kv_cache_tokens"] = {"plain": _kv_tokens(out, "plain"), "base": _kv_tokens(out, "base")}
    t0 = ((res.get("fault") or {}).get("fault") or {}).get("injected_at")
    first = None
    try:
        from sdcguard.decide import _read_jsonl
        times = [a.get("time") for a in _read_jsonl(os.path.join(out, "fault_alerts.jsonl")) if a.get("time")]
        first = min(times) if times else None
    except OSError:
        pass
    c["fault_seconds_to_first_alert"] = round(first - t0, 2) if (first and t0) else None
    return c


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Decisions on four changes to a vLLM server"."""
    fails, s = [], {}
    for role in ROLES:
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"1: the {role} server didn't start")
    if fails:
        return False, fails, s
    dec = res.get("decisions") or {}
    for (pol, role), want in EXPECTED.items():
        got = (dec.get(f"{pol}_{role}") or {}).get("decision")
        if got != want:
            fails.append(f"2: {pol} on {role} decided {got}, expected {want}")
    why = " ".join((dec.get("hold_fault") or {}).get("why") or [])
    # the injected fault is a changed weight: decide reports it under weights (amendment, docs/METHOD.md); a run
    # on the code before that reported it under arithmetic
    if not ("weights:" in why or "arithmetic" in why) or FAULT_LAYER not in why:
        fails.append(f"3: the fault's rejection doesn't give the weight (or arithmetic) reason naming {FAULT_LAYER}")
    for role in ("same", "triton", "awq"):
        a = (dec.get(f"hold_{role}") or {}).get("arithmetic")
        if a == "FAULT":
            fails.append(f"4: a FAULT on {role}, which has none injected")
        if (dec.get(f"hold_{role}") or {}).get("weights") == "CHANGED":
            fails.append(f"4: weights CHANGED on {role}, which has none injected")
    for name, d in dec.items():
        if d.get("isolation") == "LEAK":
            fails.append(f"4: an isolation LEAK on {name}, where none is expected")
    for name, d in dec.items():
        if not all(d.get(k) for k in ("arithmetic", "approximation", "performance")):
            fails.append(f"5: decision {name} is missing a section")
    s["decisions"] = dec
    s["fault_quality_check"] = (dec.get("hold_fault") or {}).get("quality_check")
    s["trace_replay"] = {k: {x: v.get(x) for x in ("verdict", "mean_kl", "top1")}
                         for k, v in (res.get("traces") or {}).items()}
    s["cost"] = res.get("cost")
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    res: dict = {"model": MODEL, "policies": POLICIES, "expected": {f"{p}_{r}": v for (p, r), v in EXPECTED.items()}}
    for role in ROLES:
        res[role] = run_role(role, a.out, a.port, a.metrics_port)
        with open(os.path.join(a.out, "PROGRESS.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
        if role == "base" and not res[role].get("ready"):
            break
    res["decisions"] = decide_all(a.out, [r for r in ROLES if (res.get(r) or {}).get("ready")])
    if (res.get("base") or {}).get("ready"):
        res["traces"] = traces(a.out)
    res["cost"] = cost(res, a.out)
    ok, fails, summary = judge(res)
    res.update(passed=ok, failures=fails, summary=summary)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": summary}, indent=1, default=str))
    for path in sorted(glob.glob(os.path.join(a.out, "decision_hold_*.txt"))):
        print(f"\n== {os.path.basename(path)}\n" + open(path).read())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
