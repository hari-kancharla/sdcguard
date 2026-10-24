"""`sdcguard qualify` as someone would run it, on a real GPU with the plugin on (docs/METHOD.md, "Qualify on a real
server", written before the run).

One L4, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16, one `sdcguard qualify` with its defaults (every layer checked,
decide's default policy, no policy file) and --repeat-baseline (the baseline started twice: the noise floor):

  same      the baseline's own configuration                     a change that changes nothing
  triton    --attention-backend TRITON_ATTN                      another attention kernel (PASS on an H100, Oct 23)
  badeps    --hf-overrides '{"rms_norm_eps": 1.0}'               a configuration error no arithmetic check sees
                                                                 (FAIL on an H100, Oct 23: mean KL 11.2)

Each server: quality against the baseline's reference, `sdcguard isolation`, `sdcguard perf`, the plugin's metrics
(with the stack it read from inside the server) and alerts; then `sdcguard decide` per candidate. Then the deployment
side: the `same` configuration started again and `sdcguard ready --qualified decision_same.json` (it is the stack the
decision accepted), and the `triton` configuration with the same decision (it isn't).

    python experiments/vllm_qualify_check.py --out runs/qualify
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SERVE = f"{MODEL} --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0"
CANDIDATES = {"same": SERVE, "triton": f"{SERVE} --attention-backend TRITON_ATTN",
              "badeps": SERVE + " --hf-overrides '{\"rms_norm_eps\": 1.0}'"}


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "sdcguard", "qualify", "--out", out, "--baseline", SERVE, "--repeat-baseline",
           "--port", str(port), "--metrics-port", str(metrics_port), "--stack", "gpu=L4"]
    for name, cfg in CANDIDATES.items():
        cmd += ["--candidate", f"{name}={cfg}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)
    res = {"cmd": cmd, "exit_code": p.returncode, "output": (p.stdout + p.stderr)[-6000:]}
    try:
        q = json.load(open(os.path.join(out, "QUALIFY.json")))
    except (OSError, ValueError) as e:
        res["error"] = f"no QUALIFY.json: {e}"
        return res
    res["runs"] = q.get("runs") or {}
    res["decisions"] = q.get("decisions") or {}
    res["report_exists"] = os.path.exists(os.path.join(out, "REPORT.md"))
    res["alerts"] = {name: _count(os.path.join(out, f"{name}_alerts.jsonl")) for name in res["runs"]}
    dec = os.path.join(out, "decision_same.json")
    if os.path.exists(dec):
        res["ready_same"] = ready_check(out, "same", CANDIDATES["same"], dec, port, metrics_port)
        res["ready_triton"] = ready_check(out, "triton", CANDIDATES["triton"], dec, port, metrics_port)
    return res


def ready_check(out: str, name: str, config: str, decision: str, port: int, metrics_port: int) -> dict:
    """Start one configuration as a deployment would (the plugin on, its metrics), send a few requests, and ask
    `sdcguard ready --qualified` whether it is the stack the decision accepted."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("SDCGUARD")}
    env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_METRICS_PORT=str(metrics_port))
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", *shlex.split(config), "--port", str(port)]
    log = open(os.path.join(out, f"ready_{name}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    res: dict = {"name": name}
    try:
        import vllm_serve_check as vs
        res["ready"] = vs.wait_ready(port, proc, timeout=1500)
        if not res["ready"]:
            return res
        for i in range(4):
            try:
                vs._request(port, "POST", "/v1/completions", {"model": MODEL, "prompt": f"Count to {i + 3}.",
                                                               "max_tokens": 16, "temperature": 0.0})
            except OSError:
                pass
        time.sleep(3)
        r = subprocess.run([sys.executable, "-m", "sdcguard", "ready", "--port", str(metrics_port), "--workers", "1",
                            "--vllm", f"http://127.0.0.1:{port}/metrics", "--probe", f"http://127.0.0.1:{port}",
                            "--qualified", decision, "--json"], capture_output=True, text=True, timeout=300)
        res["exit_code"] = r.returncode
        try:
            res["result"] = json.loads(r.stdout)
        except ValueError:
            res["output"] = (r.stdout + r.stderr)[-2000:]
    finally:
        import vllm_serve_check as vs
        res["stop"] = vs.stop(proc)
        log.close()
    return res


def _count(path: str) -> int:
    """Alerts the plugin logged for one server (SDCGUARD_LOG: one JSON event per line)."""
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Qualify on a real server"."""
    fails = []
    runs, dec = res.get("runs") or {}, res.get("decisions") or {}
    for name in ("baseline", "again", *CANDIDATES):
        if not (runs.get(name) or {}).get("ready"):
            fails.append(f"Q1: {name} didn't start")
    if res.get("exit_code") != 1:
        fails.append(f"Q1: qualify exited {res.get('exit_code')}, not 1 (a REJECT expected)")
    if not res.get("report_exists"):
        fails.append("Q1: no REPORT.md")
    same = dec.get("same") or {}
    if same.get("decision") != "ACCEPT" or same.get("arithmetic") != "NO FAULT OBSERVED" or \
            same.get("isolation") != "NO LEAK SEEN":
        fails.append(f"Q2: same is {same.get('decision')} (arithmetic {same.get('arithmetic')}, isolation "
                     f"{same.get('isolation')}, quality {same.get('approximation')}): {same.get('why')}")
    bad = dec.get("badeps") or {}
    if bad.get("decision") != "REJECT" or bad.get("approximation") != "OUTSIDE" or \
            bad.get("arithmetic") != "NO FAULT OBSERVED":
        fails.append(f"Q3: badeps is {bad.get('decision')} (quality {bad.get('approximation')}, arithmetic "
                     f"{bad.get('arithmetic')}): {bad.get('why')}")
    alerts = res.get("alerts") or {}
    if any(alerts.values()):
        fails.append(f"Q4: alerts with no fault: {alerts}")
    rs, rt = res.get("ready_same") or {}, res.get("ready_triton") or {}
    if rs.get("exit_code") != 0 or (rs.get("result") or {}).get("verdict") not in ("PROTECTED",):
        fails.append(f"Q5: ready --qualified on the accepted stack said {(rs.get('result') or {}).get('verdict')} "
                     f"(exit {rs.get('exit_code')}), not PROTECTED")
    if rt.get("exit_code") != 1 or (rt.get("result") or {}).get("verdict") != "NOT THE QUALIFIED STACK":
        fails.append(f"Q6: ready --qualified on the Triton-attention stack said "
                     f"{(rt.get('result') or {}).get('verdict')} (exit {rt.get('exit_code')}), not NOT THE QUALIFIED "
                     "STACK")
    s = {"decisions": {n: {k: d.get(k) for k in ("decision", "arithmetic", "approximation", "isolation",
                                                 "performance")} for n, d in dec.items()},
         "triton": (dec.get("triton") or {}).get("decision"), "alerts": alerts,
         "deployed_stacks": {n: (r or {}).get("deployed_stack", {}).get("fingerprint") for n, r in runs.items()},
         "ready": {"same": (rs.get("result") or {}).get("verdict"), "triton": (rt.get("result") or {}).get("verdict"),
                   "triton_differences": ((rt.get("result") or {}).get("qualified") or {}).get("differences")}}
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
