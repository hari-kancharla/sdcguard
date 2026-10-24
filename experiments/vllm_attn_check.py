"""Attention checks inside a real `vllm serve` (docs/METHOD.md, "Attention checks in serving", written before the run).

vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16 (28 attention layers, 12 query heads, 2 KV heads, head size 128),
--max-model-len 4096. On one H100 (FlashAttention 3 there), servers one after another:

  plain      SDCGUARD=0, --attention-backend FLASH_ATTN: the reference answers and speed
  guarded    SDCGUARD=1 SDCGUARD_REQUIRE=1 SDCGUARD_ATTN=1, FLASH_ATTN, compiled with CUDA graphs (vLLM's default:
             FULL graphs for decode batches, so the check runs inside them), prefix caching on (vLLM's default)
  noattn     guarded with SDCGUARD_ATTN=0: the speed without attention checks
  eager      guarded with --enforce-eager: the fault trials
  tplain     plain with --attention-backend TRITON_ATTN
  tguarded   guarded with TRITON_ATTN

and on one L4 (FlashAttention 2) plain and guarded only (--roles plain,guarded).

Each plain and guarded server answers 64 prompts greedily (32 tokens) and 8 long prompts (1,378 to 3,295 tokens in
Qwen's tokenizer, all starting with the same 1,024-token text, so later ones hit the prefix cache) with 64 tokens.
Guarded servers then report their attention checks: layers, kernels, calls, samples, scores, KV writes checked. plain,
guarded and noattn take the speed load (sdcguard.perf: 128 prompts, 128 tokens, 3 rounds, 16 and 256 requests at
once). Faults can't be put into the attention kernel inside captured CUDA graphs from Python, so compiled servers run
clean and the eager one runs the trials: one layer's attention made wrong before the check sees it
(sdcguard_test_corrupt_attn: "value", "head", "other", "short", "kv"), rounds of 4 requests at once until an alert
names the layer, or 16 rounds.

    python experiments/vllm_attn_check.py --out runs/attn [--roles plain,guarded]
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

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = 28
ROLES = ("plain", "guarded", "noattn", "eager", "tplain", "tguarded")
GUARDED = ("guarded", "noattn", "eager", "tguarded")
BACKEND = {"plain": "FLASH_ATTN", "guarded": "FLASH_ATTN", "noattn": "FLASH_ATTN", "eager": "FLASH_ATTN",
           "tplain": "TRITON_ATTN", "tguarded": "TRITON_ATTN"}
IMPL = {"FLASH_ATTN": "FlashAttentionImpl", "TRITON_ATTN": "TritonAttentionImpl"}
PAIRS = (("plain", "guarded"), ("tplain", "tguarded"))
IDENTITY = 64
LONG = 8
MODES = ("value", "head", "other", "kv", "short")
MUST_CATCH = ("value", "head", "other", "kv")      # "short" is reported: how much a query's newest key moves its
#                                                    output depends on the attention, not on the check
BURST, MAX_ROUNDS = 4, 16
BOUND = 1.0                                      # a correct output's score is at most 1 (the line is 4)
STEP_SHARE = 0.98                                # checked calls at least this share of model steps (one layer a step)
COST_LIMIT = 0.03


def long_prompts(n: int = LONG) -> list:
    """n long prompts sharing their first ~1,000 tokens."""
    words = ("signal", "kernel", "matrix", "vector", "cache", "block", "token", "layer", "value", "query")
    shared = " ".join(f"Note {i}: the {words[i % 10]} {words[(3 * i) % 10]} holds {i * 7 % 101}." for i in range(80))
    out = []
    for k in range(n):
        tail = " ".join(f"Entry {k}.{i}: {words[(i + k) % 10]} {i * 13 % 97}." for i in range(30 + 22 * k))
        out.append(f"{shared}\n{tail}\nSummarize the entries above in one sentence.")
    return out


def server(role: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    if role in GUARDED:
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
                   SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_ATTN="0" if role == "noattn" else "1")
    else:       # plain vLLM with the cuBLAS setting the plugin serves with (sdcguard_test_ext; docs/METHOD.md, A2)
        env.update(SDCGUARD="0", SDCGUARD_TEST_EXACT="1")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "4096", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--attention-backend", BACKEND[role], "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role == "eager":
        cmd.append("--enforce-eager")
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def stats(port: int) -> dict:
    s = vn.rpc(port, "sdcguard_test_attn_stats")
    st = vn.rpc(port, "sdcguard_test_steps")
    st = st if isinstance(st, dict) else {}
    # steps with scheduled tokens (each checks one layer); all marked steps if vLLM didn't say how many tokens
    return {**s, "steps": st.get("token_steps") or st.get("steps"), "marked_steps": st.get("steps")}


def _until_alert(port: int, log_path: str, n0: int, layer: str, prompts: list) -> tuple:
    for i in range(MAX_ROUNDS):
        with ThreadPoolExecutor(BURST) as ex:
            list(ex.map(lambda k: vn._try(port, MODEL, prompts[(i * BURST + k) % len(prompts)]), range(BURST)))
        time.sleep(0.5)
        vn._try(port, MODEL, prompts[0], 4)
        hit = [a for a in vn.alerts(log_path)[n0:] if (a.get("detail") or {}).get("where") == layer]
        if hit:
            return hit, i + 1
    time.sleep(3)
    vn._try(port, MODEL, prompts[0], 4)
    return [a for a in vn.alerts(log_path)[n0:] if (a.get("detail") or {}).get("where") == layer], MAX_ROUNDS


def fault_trials(port: int, log_path: str, names: list, prompts: list) -> list:
    rows = []
    for k, mode in enumerate(MODES):
        layer = sorted(names, key=lambda n: int(n.split(".")[2]) if n.split(".")[2].isdigit() else 0)[
            (9 * k + 4) % len(names)]
        for p in prompts[:2]:
            vn._try(port, MODEL, p, 8)
        time.sleep(3)
        vn._try(port, MODEL, prompts[0], 4)
        n0 = len(vn.alerts(log_path))
        mark = vn.rpc(port, "sdcguard_test_corrupt_attn", [layer, mode])
        hit, rounds = _until_alert(port, log_path, n0, layer, prompts)
        unmark = vn.rpc(port, "sdcguard_test_uncorrupt_attn")
        time.sleep(3)
        vn._try(port, MODEL, prompts[0], 4)
        new = vn.alerts(log_path)[n0:]
        # A4: any alert (any check) naming another layer, unless its score isn't finite (a non-finite value the
        # fault made can reach later layers' inputs); alerts with no score (a wrong KV write) count
        other = [a for a in new if (a.get("detail") or {}).get("where") != layer
                 and ((a.get("detail") or {}).get("max_z") is None or vn._finite((a.get("detail") or {})["max_z"]))]
        rows.append({"layer": layer, "mode": mode, "mark": mark, "unmark": unmark, "detected": bool(hit),
                     "rounds_to_detect": rounds if hit else None,
                     "findings": sorted({(a.get("detail") or {}).get("finding") for a in hit} - {None}),
                     "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict")
                                         for a in hit} - {None}),
                     "where": [((a.get("detail") or {}).get("head"), (a.get("detail") or {}).get("dim"),
                                ((a.get("detail") or {}).get("latest") or {}).get("dim"),
                                ((a.get("detail") or {}).get("latest") or {}).get("part")) for a in hit[:4]],
                     "alerts_other_layers": len(other),
                     "other_layers": sorted({f'{a.get("detector")}:{(a.get("detail") or {}).get("where")}'
                                             for a in other})})
    return rows


def run(out: str, port: int, metrics_port: int, roles=ROLES) -> dict:
    os.makedirs(out, exist_ok=True)
    res: dict = {"model": MODEL, "roles": list(roles)}
    ps = vs.prompts(IDENTITY + 8)
    tp = ps[IDENTITY:IDENTITY + 8]
    lp = long_prompts()
    for role in roles:
        proc, log = server(role, out, port, metrics_port)
        r: dict = {"backend": BACKEND[role]}
        res[role] = r
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=1500)
            if not r["ready"]:
                continue
            r["reductions"] = vn.rpc(port, "sdcguard_test_reductions")
            checked = role in GUARDED and role != "noattn"
            if checked:
                r["attn_at_start"] = stats(port)
            if role in ("plain", "guarded", "tplain", "tguarded"):
                r["identity"] = [vn._try(port, MODEL, p).get("text") for p in ps[:IDENTITY]]
                r["long"] = [vn._try(port, MODEL, p, 64).get("text") for p in lp]
            if role in ("plain", "guarded", "noattn"):
                r["speed"] = vn._speed(port, MODEL)
            if role in GUARDED:
                time.sleep(3)
                vn._try(port, MODEL, ps[0], 8)
                r["clean_alerts"] = len(vn.alerts(log_path))
            if checked:
                r["attn_clean"] = stats(port)
            names = (r.get("attn_at_start") or {}).get("names") or []
            if role == "eager" and names:
                r["faults"] = fault_trials(port, log_path, names, tp)
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
        off = (((((res.get("noattn") or {}).get("speed") or {}).get(c) or {}).get("median")) or {}).get(
            "output_tokens_per_s")
        if on and off:
            cost[c] = {"attn_on": on, "attn_off": off, "attn_cost": round(1 - on / off, 4)}
    clean = {}
    for role in ("guarded", "eager", "tguarded"):
        s = (res.get(role) or {}).get("attn_clean") or {}
        st = s.get("stats") or {}
        clean[role] = {"worst": s.get("worst"), "calls": st.get("calls_checked"), "steps": s.get("steps"),
                       "samples": st.get("samples"), "keys": st.get("keys"), "writes": st.get("writes_checked")}
    return {"cost": cost, "clean": clean,
            "faults": [(t["mode"], t["layer"], t["detected"], t.get("rounds_to_detect"), t.get("findings"),
                        t.get("verdicts")) for t in (res.get("eager") or {}).get("faults") or []]}


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Attention checks in serving". Criteria about a role apply
    when the run planned it (res["roles"])."""
    roles = res.get("roles") or list(ROLES)
    fails = []
    for role in roles:
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"A1: {role} didn't start")
    for role in ("guarded", "eager", "tguarded"):
        r = res.get(role) or {}
        if role not in roles or not r.get("ready"):
            continue
        st = r.get("attn_at_start") or {}
        if st.get("layers") != LAYERS:
            fails.append(f"A1 ({role}): {st.get('layers')} attention layers checked, not {LAYERS}: "
                         f"{st.get('not_checked')}")
        kern = st.get("kernel") or {}
        if not kern or not all(v and v[0] for v in kern.values()):
            fails.append(f"A1 ({role}): the Triton kernels aren't in use: {kern}")
        if st.get("backends") != [IMPL[BACKEND[role]]]:
            fails.append(f"A1 ({role}): backends {st.get('backends')}, not {IMPL[BACKEND[role]]}")
    for a, b in PAIRS:
        if a not in roles or b not in roles:
            continue
        for role in (a, b):                 # both with bf16/fp16 cuBLAS partial sums in fp32 (the plugin's setting)
            red = (res.get(role) or {}).get("reductions")
            if (res.get(role) or {}).get("ready") and (not isinstance(red, dict) or red.get("bf16") is not False):
                fails.append(f"A2: {role} didn't keep cuBLAS partial sums in fp32 ({red}), so its answers aren't "
                             "compared on the same setting")
        for what in ("identity", "long"):
            x, y = (res.get(a) or {}).get(what), (res.get(b) or {}).get(what)
            if x is None or y is None or x != y or any(t is None for t in x):
                fails.append(f"A2: {b}'s {what} completions differ from {a}'s "
                             f"({sum(p != q for p, q in zip(x or [], y or []))} differ)")
    for role in ("guarded", "eager", "tguarded", "noattn"):
        r = res.get(role) or {}
        if role not in roles or not r.get("ready"):
            continue
        if r.get("clean_alerts"):
            fails.append(f"A3 ({role}): {r['clean_alerts']} alerts with no fault")
        if role == "noattn":
            continue
        s = r.get("attn_clean") or {}
        st = s.get("stats") or {}
        w = s.get("worst")
        if not vn._finite(w) or float(w) > BOUND:
            fails.append(f"A3 ({role}): the worst clean attention score was {w}, above the bound of {BOUND}")
        for k in ("writes_wrong", "slots_wrong", "bad_blocks", "unjudged"):
            if st.get(k):
                fails.append(f"A3 ({role}): {st[k]} {k.replace('_', ' ')} with no fault")
        if not st.get("samples") or not st.get("writes_checked"):
            fails.append(f"A3 ({role}): no samples or writes checked ({st.get('samples')}, {st.get('writes_checked')})")
        if s.get("calls_not_checked"):
            fails.append(f"A3 ({role}): calls not checked: {s['calls_not_checked']}")
        steps, calls = s.get("steps"), st.get("calls_checked")
        if role != "eager" and (not steps or not calls or calls < STEP_SHARE * steps):
            fails.append(f"A6 ({role}): {calls} calls checked in {steps} model steps (one layer a step: inside "
                         "CUDA graphs too)")
    e = res.get("eager") or {}
    if "eager" in roles and e.get("ready"):
        rows = e.get("faults") or []
        want = {"value": "output_wrong_or_misread", "head": "output_wrong_or_misread",
                "other": "output_wrong_or_misread", "kv": "kv_write_wrong", "short": "output_wrong_or_misread"}
        missed = [t for t in rows if t["mode"] in MUST_CATCH and
                  not (t["detected"] and t.get("mark") and want[t["mode"]] in (t.get("findings") or []))]
        if len(rows) != len(MODES) or missed:
            fails.append(f"A4: faults caught in {sum(t['detected'] for t in rows)} of {len(rows)} trials; missed: "
                         f"{[(t['mode'], t['layer']) for t in missed]}")
        stray = [t for t in rows if t.get("alerts_other_layers")]
        if stray:
            fails.append(f"A4: alerts on other layers: {[(t['mode'], t['other_layers']) for t in stray]}")
        z_rows = [t for t in rows if t["mode"] in ("value", "head", "other") and t["detected"]]
        if any("output_wrong" not in (t.get("verdicts") or []) for t in z_rows):
            fails.append(f"A5: rechecks of caught outputs: {[(t['mode'], t.get('verdicts')) for t in z_rows]}")
    s = summarize(res)
    c16 = (s["cost"].get("16") or {}).get("attn_cost")
    s["on_by_default"] = None if c16 is None else c16 <= COST_LIMIT
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    ap.add_argument("--roles", default=",".join(ROLES))
    a = ap.parse_args(argv)
    roles = [r for r in a.roles.split(",") if r]
    bad = [r for r in roles if r not in ROLES]
    if bad:
        raise SystemExit(f"unknown roles {bad}; known: {ROLES}")
    res = run(a.out, a.port, a.metrics_port, roles)
    ok, fails, s = judge(res)
    res.update(passed=ok, failures=fails, summary=s)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": s}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
