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
import signal
import subprocess
import sys
import time

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
COMPILE = json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8], "cudagraph_mode": "FULL_AND_PIECEWISE"})
BASE = f"{MODEL} --max-model-len 2048 --gpu-memory-utilization 0.85 --seed 0 --moe-backend triton " \
       f"--compilation-config '{COMPILE}'"
CANDIDATES = {"fp8pt": f"{BASE} --quantization fp8_per_tensor",
              "eager": f"{BASE} --quantization fp8_per_tensor --enforce-eager",
              "fp8pc": f"{BASE} --quantization fp8_per_channel"}
POLICY = {"quality": {"accept_warn": True}, "isolation": {"required": True}}
MOE_LAYERS = 24
# within the Modal function's 190 minutes (its log is cut at 187): a server that never comes up costs 15 minutes, not
# qualify's default 30, and a run past either limit is still judged on what it saved (qualify writes its record after
# each server). In the October 25 run qualify would have needed about 100 to 105 minutes (the FP8 and eager servers'
# speed loads on an L4 are slow) and was stopped at the 90 it had.
STARTUP_S = 900
QUALIFY_S = 150 * 60
RERUN_S = 25 * 60


def _cmdline(pid) -> list:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode(errors="replace").split("\0")
    except OSError:
        return []


def servers_on(port: int) -> list:
    """The pids of vLLM processes: API servers whose command line serves on --port `port`, and vLLM's engine
    processes, which rename themselves ("VLLM::EngineCore") and so keep no port in their command line, from /proc.
    (stop_servers stops only those that weren't running before: other vLLM processes on the machine are left alone.)"""
    out = []
    for d in os.listdir("/proc") if os.path.isdir("/proc") else []:
        if not d.isdigit() or int(d) == os.getpid():
            continue
        args = _cmdline(d)
        if not args:
            continue
        api = "serve" in args and any(a == "--port" and args[i + 1:i + 2] == [str(port)] for i, a in enumerate(args))
        if api or args[0].startswith("VLLM::"):
            out.append(int(d))
    return out


def _alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except OSError:
        return False


def stop_servers(port: int, grace: float = 30.0, before=()) -> list:
    """Stop every vLLM process still running (servers_on) that isn't in before (those running before the command
    began): their process groups, SIGTERM, then SIGKILL after grace seconds, waiting until no process of those groups
    is left (the engine holds the GPU's memory until it exits); the pids found."""
    pids = [p for p in servers_on(port) if p not in set(before)]
    groups = set()
    for pid in pids:
        try:
            groups.add(os.getpgid(pid))
        except OSError:
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for g in groups:
            try:
                os.killpg(g, sig)
            except OSError:
                pass
        t0 = time.monotonic()
        while any(_alive(g) for g in groups) and time.monotonic() - t0 < grace:
            time.sleep(1)
        if not any(_alive(g) for g in groups):
            break
    return pids


def run_bounded(cmd, timeout: float, port: int, env=None) -> dict:
    """Run cmd (in its own session) for at most timeout seconds. Past it: SIGINT first, so qualify's own cleanup
    stops its servers and its record so far stays written (Python runs its finally blocks), SIGKILL after 120 s,
    then any server still on port stopped too (qualify starts each server in a session of its own, which a kill of
    qualify's group doesn't reach). Its output goes to a file, not a pipe, so a child left holding it can't stall
    the wait. {"code", "output", "timed_out", "stray": the pids of servers left running}."""
    import tempfile
    timed_out = False
    before = servers_on(port)
    with tempfile.TemporaryFile("w+", errors="replace") as f:
        # SIGINT back to its default in the child: a process started from a background job inherits it ignored, and
        # Python then never raises KeyboardInterrupt, so no cleanup would run
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                             preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig, wait in ((signal.SIGINT, 120), (signal.SIGKILL, 30)):
                try:
                    os.killpg(p.pid, sig)
                except OSError:
                    pass
                try:
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    pass
        f.seek(0)
        out = f.read()
    stray = stop_servers(port, before=before)
    return {"code": None if timed_out else p.returncode, "output": out[-6000:], "timed_out": timed_out,
            "stray": stray}


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
           "--stack", "gpu=L4", "--startup-timeout", str(STARTUP_S)]
    for name, cfg in CANDIDATES.items():
        cmd += ["--candidate", f"{name}={cfg}"]
    q = run_bounded(cmd, QUALIFY_S, port)
    res: dict = {"cmd": cmd, "exit_code": q["code"], "output": q["output"], "stray_servers": q["stray"]}
    if q["timed_out"]:                          # still judged (and written): what qualify saved by then
        res["error"] = f"qualify ran past {QUALIFY_S} s"
    try:
        q = json.load(open(os.path.join(out, "QUALIFY.json")))
    except (OSError, ValueError) as e:
        res["error"] = (f"{res['error']}; " if res.get("error") else "") + f"no QUALIFY.json: {e}"
        return res
    res["runs"], res["decisions"] = q.get("runs") or {}, q.get("decisions") or {}
    # a qualify cut short: its record so far (complete: false), and the servers it never reached
    res["qualify_complete"], res["not_yet_run"] = q.get("complete", True), q.get("not_yet_run") or []
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
        r = run_bounded(["sh", os.path.join(rp, "rerun.sh")], RERUN_S, port,
                        env=dict(os.environ, PORT=str(port), MPORT=str(metrics_port), PY=sys.executable))
        code, text = r["code"], (f"rerun.sh ran past {RERUN_S} s; " if r["timed_out"] else "") + r["output"][-3000:]
        iso = os.path.join(rp, "rerun_isolation.json")
        res["repro"] = {"exit_code": code, "output": text, "stray_servers": r["stray"],
                        "isolation": json.load(open(iso)).get("verdict") if os.path.exists(iso) else None,
                        "fp8_alerts": len([a for a in _alerts(os.path.join(rp, "rerun_alerts.jsonl"))
                                           if a.get("detector") == "serve_fp8"])}
    return res


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "An FP8 MoE bug from detection to remedy"."""
    fails = []
    runs, dec, fa = res.get("runs") or {}, res.get("decisions") or {}, res.get("fp8_alerts") or {}
    later = res.get("not_yet_run") or []
    for name in ("baseline", "fp8pt", "eager"):
        if name in later:
            fails.append(f"F1: qualify stopped before it finished {name}")
        elif not (runs.get(name) or {}).get("ready"):
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
    if res.get("qualify_complete") is False:
        fails.append("F6: qualify didn't finish, so it wrote no repro folder to rerun")
    elif rp.get("isolation") != "LEAK":
        fails.append(f"F6: the repro folder's rerun didn't reproduce the leak ({rp.get('isolation')})")
    s = {"decisions": {n: {k: (dec.get(n) or {}).get(k) for k in ("decision", "approximation", "isolation",
                                                                 "consistency", "throughput_change")} for n in dec},
         "fp8_alerts": {n: {"count": v["count"], "findings": v["findings"]} for n, v in fa.items()},
         "fp8pc": {"started": (runs.get("fp8pc") or {}).get("ready"), "fp8_alerts": (fa.get("fp8pc") or {}).get("count"),
                   "isolation": (dec.get("fp8pc") or {}).get("isolation")},
         "repro": {k: rp.get(k) for k in ("isolation", "fp8_alerts", "exit_code")},
         "fp8_coverage": res.get("fp8_coverage"), "qualify_complete": res.get("qualify_complete"),
         "not_yet_run": later}
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
