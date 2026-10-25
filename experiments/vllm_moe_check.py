"""The MoE check inside a real `vllm serve` (docs/METHOD.md, "MoE checks in serving", written before the run).

One H100, vLLM 0.32.0, ibm-granite/granite-3.1-1b-a400m-instruct (24 fused MoE layers, 32 experts, top 8) in bf16,
vLLM's Triton MoE kernel on every server (--moe-backend triton), prefix caching off. Servers, one after another:

  plain      SDCGUARD=0: the reference answers and speed
  guarded    SDCGUARD=1 SDCGUARD_REQUIRE=1 SDCGUARD_MOE=1, compiled with CUDA graphs (vLLM's default)
  nomoe      guarded with SDCGUARD_MOE=0 and without SDCGUARD_REQUIRE (which refuses a model whose MoE layers
             nothing checks): the speed without MoE checks
  eager      guarded with --enforce-eager

plain and guarded answer 64 prompts greedily (32 tokens); plain, guarded and nomoe take the speed load (sdcguard.perf:
128 prompts, 128 tokens, 3 rounds, 16 and 256 requests at once). Faults can't be put into a kernel inside captured
CUDA graphs from Python, and a weight flipped in memory is read by the kernel and the recompute alike (outside what
this check sees), so the compiled server is run clean only. eager runs 6 trials, each making one MoE layer's kernel
wrong (sdcguard_test_corrupt_moe: its output 1.5 times too large, the kernel given the wrong experts, or each row
given another row's output) and sending up to 16 rounds of 4 requests at once (a batch of one row hides a row
mix-up) until an alert names the layer; then 2 trials of the routing check (SDCGUARD_MOE=1 turns it on with the MoE
checks; amendment of October 24), each making one router wrong before the check sees its result (each row's last
expert replaced by the next expert number; every weight 1.5 times too large), the same way until an alert names
the router.

    python experiments/vllm_moe_check.py --out runs/moe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_norm_check as vn  # noqa: E402
import vllm_serve_check as vs  # noqa: E402

MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
MOE_LAYERS = 24
MODES = ("scale", "expert", "row", "scale", "expert", "row")
CLEAN_LIMIT = 2.0                        # the worst clean score allowed (CPU: 0.25; line 6)
MAX_REQUESTS = 16                        # rounds of BURST requests
BURST = 4                                # requests at once: the layer sees batches of several rows
CORRUPT_TRIALS = len(MODES)
ROUTE_MODES = ("experts", "weights")
COST_LIMIT = 0.03
IDENTITY = 64


def server(role: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    if role != "plain":
        # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
        env["SDCGUARD_ATTN"] = "0"
        env.update(SDCGUARD="1", SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
                   SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_MOE="0" if role == "nomoe" else "1")
        if role != "nomoe":
            env["SDCGUARD_REQUIRE"] = "1"
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--moe-backend", "triton", "--no-enable-prefix-caching",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role == "eager":
        cmd.append("--enforce-eager")
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def _until_alert(port: int, log_path: str, n0: int, layer: str, prompts: list) -> tuple:
    """Rounds of BURST requests at once until an alert names the layer (or MAX_REQUESTS rounds): (alerts on it, rounds
    sent). A layer checks when its turn comes (one layer a step), so the rows it sees are the batch of that step: with
    requests one at a time a decode batch is one row, and each row given another row's output looks right."""
    for i in range(MAX_REQUESTS):
        with ThreadPoolExecutor(BURST) as ex:
            list(ex.map(lambda k: vn._try(port, MODEL, prompts[(i * BURST + k) % len(prompts)]), range(BURST)))
        time.sleep(0.5)
        vn._try(port, MODEL, prompts[0], 4)
        hit = [a for a in vn.alerts(log_path)[n0:] if (a.get("detail") or {}).get("where") == layer]
        if hit:
            return hit, i + 1
    time.sleep(3)
    vn._try(port, MODEL, prompts[0], 4)
    return [a for a in vn.alerts(log_path)[n0:] if (a.get("detail") or {}).get("where") == layer], MAX_REQUESTS


def corrupt_trials(port: int, log_path: str, names: list, prompts: list) -> list:
    rows = []
    for k, mode in enumerate(MODES):
        layer = sorted(names)[(5 * k + 2) % len(names)]
        for p in prompts[:2]:
            vn._try(port, MODEL, p, 8)
        time.sleep(3)
        vn._try(port, MODEL, prompts[0], 4)
        n0 = len(vn.alerts(log_path))
        mark = vn.rpc(port, "sdcguard_test_corrupt_moe", [layer, mode])
        hit, sent = _until_alert(port, log_path, n0, layer, prompts)
        unmark = vn.rpc(port, "sdcguard_test_uncorrupt_moe")
        rows.append({"layer": layer, "mode": mode, "mark": mark, "unmark": unmark, "detected": bool(hit),
                     "rounds_to_detect": sent if hit else None,
                     "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict")
                                         for a in hit} - {None})})
    return rows


def route_trials(port: int, log_path: str, names: list, prompts: list) -> list:
    rows = []
    for k, mode in enumerate(ROUTE_MODES):
        router = sorted(names)[(7 * k + 4) % len(names)]
        for p in prompts[:2]:
            vn._try(port, MODEL, p, 8)
        time.sleep(3)
        vn._try(port, MODEL, prompts[0], 4)
        n0 = len(vn.alerts(log_path))
        mark = vn.rpc(port, "sdcguard_test_corrupt_route", [router, mode])
        hit, sent = _until_alert(port, log_path, n0, router, prompts)
        unmark = vn.rpc(port, "sdcguard_test_uncorrupt_route")
        rows.append({"router": router, "mode": mode, "mark": mark, "unmark": unmark, "detected": bool(hit),
                     "rounds_to_detect": sent if hit else None,
                     "findings": sorted({(a.get("detail") or {}).get("finding") for a in hit} - {None}),
                     "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict")
                                         for a in hit} - {None})})
    return rows


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    res: dict = {"model": MODEL}
    ps = vs.prompts(IDENTITY + 8)
    tp = ps[IDENTITY:IDENTITY + 8]
    for role in ("plain", "guarded", "nomoe", "eager"):
        proc, log = server(role, out, port, metrics_port)
        r: dict = {}
        res[role] = r
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=1500)
            if not r["ready"]:
                continue
            if role in ("guarded", "eager"):
                r["moe_at_start"] = vn.rpc(port, "sdcguard_test_moe_stats")
            if role in ("plain", "guarded"):
                r["identity"] = [vn._try(port, MODEL, p).get("text") for p in ps[:IDENTITY]]
            if role in ("plain", "guarded", "nomoe"):
                r["speed"] = vn._speed(port, MODEL)
            if role in ("guarded", "eager"):
                time.sleep(3)
                vn._try(port, MODEL, ps[0], 8)
                r["clean_alerts"] = len(vn.alerts(log_path))
                r["moe_clean"] = vn.rpc(port, "sdcguard_test_moe_stats")
            names = (r.get("moe_at_start") or {}).get("names") or []
            if role == "eager" and names:
                r["corrupt"] = corrupt_trials(port, log_path, names, tp)
            rnames = ((r.get("moe_at_start") or {}).get("routing") or {}).get("names") or []
            if role == "eager" and rnames:
                r["route_corrupt"] = route_trials(port, log_path, rnames, tp)
            r["alerts_total"] = len(vn.alerts(log_path))
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
    return res


def summarize(res: dict) -> dict:
    g = res.get("guarded") or {}
    cost = {}
    for c in ("16", "256"):
        on = (((g.get("speed") or {}).get(c) or {}).get("median") or {}).get("output_tokens_per_s")
        off = (((((res.get("nomoe") or {}).get("speed") or {}).get(c) or {}).get("median")) or {}).get(
            "output_tokens_per_s")
        if on and off:
            cost[c] = {"moe_on": on, "moe_off": off, "moe_cost": round(1 - on / off, 4)}
    return {"corrupt": [(t["mode"], t["layer"], t["detected"], t.get("rounds_to_detect"))
                        for t in (res.get("eager") or {}).get("corrupt") or []], "cost": cost,
            "route_corrupt": [(t["mode"], t["router"], t["detected"], t.get("rounds_to_detect"), t.get("verdicts"))
                              for t in (res.get("eager") or {}).get("route_corrupt") or []],
            "clean_worst": {k: ((res.get(k) or {}).get("moe_clean") or {}).get("worst") for k in ("guarded", "eager")},
            "backends": ((res.get("guarded") or {}).get("moe_at_start") or {}).get("backends")}


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "MoE checks in serving"."""
    fails = []
    for role in ("plain", "guarded", "nomoe", "eager"):
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"M1: {role} didn't start")
    for role in ("guarded", "eager"):
        st = (res.get(role) or {}).get("moe_at_start") or {}
        if (res.get(role) or {}).get("ready"):
            if st.get("layers") != MOE_LAYERS:
                fails.append(f"M1 ({role}): {st.get('layers')} MoE layers checked, not {MOE_LAYERS}: "
                             f"{st.get('not_checked')}")
            kern = st.get("kernel") or {}
            if not kern or not all(v and v[0] for v in kern.values()):
                fails.append(f"M1 ({role}): the Triton kernels aren't in use: {kern}")
            rt = st.get("routing") or {}
            if rt.get("routers") != MOE_LAYERS:
                fails.append(f"M1 ({role}): the routing of {rt.get('routers')} routers checked, not {MOE_LAYERS}: "
                             f"{rt.get('not_checked')}")
    for role in ("guarded", "eager"):
        r = res.get(role) or {}
        rk = (((r.get("moe_clean") or {}).get("routing") or {}).get("coverage") or {}).get("kernel") or {}
        if r.get("ready") and (not rk or not all(v == "Triton" for v in rk.values())):
            fails.append(f"M1 ({role}): the routing check's Triton kernel isn't in use: {rk}")
        rs = ((r.get("moe_clean") or {}).get("routing") or {}).get("stats") or {}
        if r.get("ready") and not rs.get("calls"):
            fails.append(f"M3 ({role}): no routing check ran")
    x, y = (res.get("plain") or {}).get("identity"), (res.get("guarded") or {}).get("identity")
    if x is None or y is None or x != y or any(t is None for t in x):
        fails.append(f"M2: the guarded server's completions differ from plain's "
                     f"({sum(p != q for p, q in zip(x or [], y or []))} differ)")
    for role in ("guarded", "eager"):
        r = res.get(role) or {}
        if not r.get("ready"):
            continue
        if r.get("clean_alerts"):
            fails.append(f"M3 ({role}): {r['clean_alerts']} alerts with no fault")
        w = (r.get("moe_clean") or {}).get("worst")
        if not vn._finite(w) or float(w) > CLEAN_LIMIT:
            fails.append(f"M3 ({role}): the worst clean MoE score was {w}, above {CLEAN_LIMIT}")
        if not ((r.get("moe_clean") or {}).get("stats") or {}).get("active_calls"):
            fails.append(f"M3 ({role}): no MoE check ran")
    e = res.get("eager") or {}
    if e.get("ready"):
        rows = e.get("corrupt") or []
        if len(rows) != CORRUPT_TRIALS or not all(t["detected"] and t.get("mark") for t in rows):
            fails.append(f"M4: wrong kernels caught in {sum(t['detected'] for t in rows)} of {len(rows)} trials: "
                         f"{[(t['mode'], t['layer'], t['detected']) for t in rows]}")
        rr = e.get("route_corrupt") or []
        want = {"experts": "wrong_experts", "weights": "wrong_weights"}
        if len(rr) != len(ROUTE_MODES) or not all(t["detected"] and t.get("mark") and want[t["mode"]] in
                                                  (t.get("findings") or []) for t in rr):
            fails.append(f"M5: wrong routers caught in {sum(t['detected'] for t in rr)} of {len(rr)} trials: "
                         f"{[(t['mode'], t['router'], t['detected'], t.get('findings')) for t in rr]}")
    s = summarize(res)
    c16 = (s["cost"].get("16") or {}).get("moe_cost")
    s["on_by_default"] = None if c16 is None else c16 <= COST_LIMIT
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
