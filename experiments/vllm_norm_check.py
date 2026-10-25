"""RMSNorm checks inside a real `vllm serve` (docs/METHOD.md, "Norm checks in serving", written before the run).

One H100, vLLM 0.32.0, bf16, prefix caching off on every server (a weight flipped and put back must not leave its KV
blocks for later requests). Servers, one after another:

  plain        Qwen2.5-1.5B-Instruct, SDCGUARD=0: the reference answers and speed
  guarded      the same with SDCGUARD=1 SDCGUARD_REQUIRE=1 SDCGUARD_NORMS=1, compiled with CUDA graphs
               (vLLM's default: its native PyTorch norm, compiled by Inductor)
  nonorms      guarded with SDCGUARD_NORMS=0: the speed without norm checks
  eager        guarded with --enforce-eager (vLLM's CUDA norm kernel instead)
  q3plain      Qwen3-1.7B, SDCGUARD=0 (its attention has per-head q/k norms)
  q3guarded    Qwen3-1.7B guarded

What each does: plain and q3plain answer 64 prompts greedily (32 tokens). guarded answers them too, then the 4 prompts
the trials use, then a speed load (sdcguard.perf: 128 prompts, 128 tokens, 3 rounds, 16 and 256 requests at once,
also on plain and nonorms), then the trials: one bit of one norm weight flipped (through /collective_rpc), the 4
prompts answered, a wait of 3 s and one more request (so the plugin reads its results back), the bit flipped back, 2
requests, 3 s, 1 request. Bits 15 (sign), 14, 10, 7 (exponent) 3 trials each; 6 to 0 (mantissa) 4 each; at norm layers
and positions drawn with seed 0. eager runs 5 trials of exact naming: 32 requests at once, one holding a marker token
whose row of one norm's output is made wrong after the norm (sdcguard_test_mark_norm_rows). q3guarded answers the 64
prompts and runs 4 flips of bit 14 in q_norm/k_norm layers.

    python experiments/vllm_norm_check.py --out runs/norms
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
Q3 = "Qwen/Qwen3-1.7B"
NORMS = {MODEL: 57, Q3: 113}             # 28 blocks x 2 (x 4 with q/k norms) + the final norm
SURE_BITS = (15, 14, 10, 7, 6, 5, 4)     # the bound says: caught every time (a change of more than 10 roundoffs)
REPORT_BITS = (3, 2, 1, 0)               # between the lines; bit 0 (at most 2 roundoffs): a score of at most 6.06
PER_BIT = {15: 3, 14: 3, 10: 3, 7: 3, 6: 4, 5: 4, 4: 4, 3: 4, 2: 4, 1: 4, 0: 4}
BOUND = 4.05                             # a correct norm's worst score: 4 roundings, 4.03 with their products
LINE = 6.0
U = 2.0 ** -8
IDENTITY = 64
TOKENS = 32
MARK_TRIALS = 5
COST_LIMIT = 0.05                        # norm checks costing more than this at 16 at once: not on by default


def _post(port: int, path: str, body: dict, timeout: float = 600.0):
    status, raw = vs._request(port, "POST", path, body, timeout=timeout)
    return status, (json.loads(raw) if status == 200 else raw.decode("utf-8", "replace")[:300])


def complete(port: int, model: str, prompt, tokens: int = TOKENS) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "temperature": 0.0, "ignore_eos": True}
    status, r = _post(port, "/v1/completions", body)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {r!r}")
    return {"id": r["id"], "text": r["choices"][0]["text"], "t": time.time()}


def _try(port: int, model: str, prompt, tokens: int = TOKENS) -> dict:
    try:
        return complete(port, model, prompt, tokens)
    except Exception as e:  # noqa: BLE001  (a request a broken weight makes fail is a result too)
        return {"error": str(e)[:200], "t": time.time()}


def rpc(port: int, method: str, args=()) -> dict:
    status, r = _post(port, "/collective_rpc", {"method": method, "args": list(args), "timeout": 120})
    if status != 200:
        return {"error": f"HTTP {status}: {r!r}"}
    res = r.get("results") or [{}]
    return res[0] if isinstance(res, list) and res else {}


def alerts(path: str) -> list:
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def server(role: str, model: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    if role not in ("plain", "q3plain"):
        # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
        env["SDCGUARD_ATTN"] = "0"
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
                   SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_NORMS="0" if role == "nonorms" else "1")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--no-enable-prefix-caching", "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role == "eager":
        cmd.append("--enforce-eager")
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def trials(names: list, bits=None, per_bit=None, seed: int = 0) -> list:
    """The flips: per_bit trials of each bit (a number, or a dict by bit; PER_BIT by default), each at a norm layer
    and a position drawn with the seed."""
    rng = random.Random(seed)
    names = sorted(names)
    counts = per_bit if isinstance(per_bit, dict) else PER_BIT if per_bit is None else None
    out = []
    for bit in (bits or (*SURE_BITS, *REPORT_BITS)):
        for _ in range(counts[bit] if counts is not None else per_bit):
            out.append({"bit": bit, "layer": rng.choice(names), "col_frac": rng.random()})
    return out


def _speed(port: int, model: str) -> dict:
    from sdcguard import perf
    url = f"http://127.0.0.1:{port}"
    out = {}
    for c in (16, 256):
        try:
            out[str(c)] = perf.measure(url, model, concurrency=c, max_tokens=128, rounds=3)
        except Exception as e:  # noqa: BLE001
            out[str(c)] = {"error": str(e)[:300]}
    return out


def flip_trial(port: int, model: str, log_path: str, t: dict, prompts: list, clean: list) -> dict:
    n0 = len(alerts(log_path))
    t0 = time.time()
    flip = rpc(port, "sdcguard_test_flip_norm_bit", [t["layer"], t["col_frac"], t["bit"]])
    answers = [_try(port, model, p) for p in prompts]
    time.sleep(3)
    _try(port, model, prompts[0], 8)
    back = rpc(port, "sdcguard_test_flip_norm_bit", [t["layer"], t["col_frac"], t["bit"]])
    t1 = time.time()
    for p in prompts[:2]:
        _try(port, model, p, 8)
    time.sleep(3)
    _try(port, model, prompts[0], 8)
    new = [a for a in alerts(log_path)[n0:] if t0 <= (a.get("time") or 0) <= t1 + 8]
    hit = [a for a in new if (a.get("detail") or {}).get("where") == t["layer"]]
    other = [a for a in new if (a.get("detail") or {}).get("where") != t["layer"]]
    finite_other = [a for a in other if _finite((a.get("detail") or {}).get("max_z"))]
    rel = relative_change((flip or {}).get("before"), (flip or {}).get("after"))
    return {**t, "flip": flip, "back": back, "relative_change": rel, "detected": bool(hit), "alerts": len(hit),
            "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict") for a in hit} - {None}),
            "alerts_other_layers": len(other), "finite_alerts_other_layers": len(finite_other),
            "other_layers": sorted({(a.get("detail") or {}).get("where") for a in finite_other}),
            "answers_changed": sum(a.get("text") != c for a, c in zip(answers, clean)),
            "requests_failed": sum("error" in a for a in answers)}


def relative_change(before, after):
    """|after - before| / |before| for a flipped weight, or None when either isn't a finite number (the extension
    sends inf and NaN as names) or the weight was 0."""
    if not (_finite(before) and _finite(after)) or not before:
        return None
    return abs(float(after) - float(before)) / abs(float(before))


def _finite(z) -> bool:
    try:
        return float(z) == float(z) and abs(float(z)) != float("inf")
    except (TypeError, ValueError):
        return False


def exact_naming(port: int, model: str, log_path: str) -> dict:
    """32 requests at once, one holding a marker token whose row of one norm's output is made wrong; each alert with a
    kept row must name exactly that request (the A4 design of vllm_ready_attrib_check, on norms)."""
    ps = vs.prompts(40)
    st, r = _post(port, "/tokenize", {"model": model, "prompt": "龘"})
    marker = r["tokens"][-1] if st == 200 else None
    ids = []
    for p in ps[:32]:
        st, r = _post(port, "/tokenize", {"model": model, "prompt": p})
        ids.append(r["tokens"] if st == 200 else [])
    res: dict = {"marker": marker, "prompts_holding_marker_naturally": [i for i, x in enumerate(ids) if marker in x],
                 "trials": []}
    for k in range(MARK_TRIALS):
        layer = f"model.layers.{16 + k}.post_attention_layernorm"
        target = (7 * k + 3) % 32
        batch = [list(x) for x in ids]
        batch[target] = batch[target][:5] + [marker] + batch[target][5:]
        for p in ps[32:36]:
            _try(port, model, p, 8)
        time.sleep(3)
        _try(port, model, ps[36], 8)
        n0 = len(alerts(log_path))
        mark = rpc(port, "sdcguard_test_mark_norm_rows", [layer, marker])
        out: list = [None] * 32
        threads = [threading.Thread(target=lambda j=j: out.__setitem__(j, _try(port, model, batch[j], 16)))
                   for j in range(32)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        unmark = rpc(port, "sdcguard_test_unmark_norm_rows")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 60 and not any((a.get("detail") or {}).get("where") == layer
                                                     for a in alerts(log_path)[n0:]):
            _try(port, model, ps[37], 8)
            time.sleep(1)
        time.sleep(3)
        got = [a for a in alerts(log_path)[n0:] if (a.get("detail") or {}).get("where") == layer]
        res["trials"].append({"layer": layer, "target": (out[target] or {}).get("id"), "mark": mark,
                              "unmark": unmark, "alerts": len(got),
                              "named": [(a.get("detail") or {}).get("directly_implicated") for a in got],
                              "attribution": [(a.get("detail") or {}).get("attribution") for a in got],
                              "kept_rows": [((a.get("detail") or {}).get("captured") or {}).get("row") for a in got]})
    return res


def run(out: str, port: int, metrics_port: int, only=None) -> dict:
    os.makedirs(out, exist_ok=True)
    res: dict = {"models": [MODEL, Q3]}
    ps = vs.prompts(IDENTITY + 8)
    tp = ps[IDENTITY:IDENTITY + 4]                     # the trials' prompts
    for role, model in (("plain", MODEL), ("guarded", MODEL), ("nonorms", MODEL), ("eager", MODEL),
                        ("q3plain", Q3), ("q3guarded", Q3)):
        if only and role not in only:
            continue
        proc, log = server(role, model, out, port, metrics_port)
        r: dict = {"model": model}
        res[role] = r
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=1500)
            if not r["ready"]:
                continue
            guarded = role not in ("plain", "q3plain")
            if guarded:
                r["norms_at_start"] = rpc(port, "sdcguard_test_norm_stats")
            if role != "nonorms":
                r["identity"] = [_try(port, model, p).get("text") for p in ps[:IDENTITY]]
            if role in ("plain", "guarded", "nonorms"):
                r["speed"] = _speed(port, model)
            if role == "guarded":
                r["trial_clean"] = [_try(port, model, p).get("text") for p in tp]
            if guarded:
                time.sleep(3)
                _try(port, model, ps[0], 8)
                r["clean_alerts"] = len(alerts(log_path))
                r["norms_clean"] = rpc(port, "sdcguard_test_norm_stats")
            if role == "guarded":
                rows = []
                names = (r["norms_at_start"] or {}).get("names") or []
                for i, t in enumerate(trials(names)):
                    rows.append(flip_trial(port, model, log_path, t, tp, r["trial_clean"]))
                    with open(os.path.join(out, "PROGRESS.json"), "w") as f:
                        json.dump({"done": len(rows), "rows": rows}, f, indent=1, default=str)
                r["trials"] = rows
            if role == "eager":
                r["exact"] = exact_naming(port, model, log_path)
                r["norms_after"] = rpc(port, "sdcguard_test_norm_stats")
            if role == "q3guarded":
                names = [n for n in (r["norms_at_start"] or {}).get("names") or [] if n.endswith(("q_norm", "k_norm"))]
                clean = [_try(port, model, p).get("text") for p in tp]
                r["trials"] = [flip_trial(port, model, log_path, t, tp, clean)
                               for t in trials(names, bits=(14,), per_bit=4, seed=1)]
            r["alerts_total"] = len(alerts(log_path))
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
    return res


def _norm_line_ok(stats: dict, model: str) -> list:
    """N1 for one guarded server: every RMSNorm layer checked, with the Triton kernel."""
    fails = []
    n = (stats or {}).get("norms")
    if n != NORMS[model]:
        fails.append(f"{n} norm layers checked, not {NORMS[model]}")
    kern = (stats or {}).get("kernel") or {}
    if not kern or not all(v and v[0] for v in kern.values()):
        fails.append(f"the Triton kernel isn't in use: {kern}")
    return fails


def summarize(res: dict) -> dict:
    g = res.get("guarded") or {}
    by_bit: dict = {}
    for t in g.get("trials") or []:
        b = by_bit.setdefault(t["bit"], {"trials": 0, "counted": 0, "detected": 0, "answers_changed": 0,
                                         "changed_and_missed": 0, "zero_weight": 0})
        b["trials"] += 1
        if not t.get("flip"):
            continue
        if not (t.get("flip") or {}).get("before"):
            b["zero_weight"] += 1
        b["counted"] += 1
        b["detected"] += t["detected"]
        b["answers_changed"] += bool(t["answers_changed"])
        b["changed_and_missed"] += bool(t["answers_changed"]) and not t["detected"]
    cost = {}
    for c in ("16", "256"):
        on = (((g.get("speed") or {}).get(c) or {}).get("median") or {})
        off = ((((res.get("nonorms") or {}).get("speed") or {}).get(c) or {}).get("median") or {})
        plain = ((((res.get("plain") or {}).get("speed") or {}).get(c) or {}).get("median") or {})
        key = "output_tokens_per_s"
        if on.get(key) and off.get(key):
            cost[c] = {"metric": key, "norms_on": on[key], "norms_off": off[key], "plain": plain.get(key),
                       "norm_cost": round(1 - on[key] / off[key], 4)}
    worst = {role: ((res.get(role) or {}).get("norms_clean") or {}).get("worst")
             for role in ("guarded", "eager", "q3guarded")}
    return {"by_bit": by_bit, "cost": cost, "clean_worst": worst}


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Norm checks in serving"."""
    fails = []
    for role in ("plain", "guarded", "nonorms", "eager", "q3plain", "q3guarded"):
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"N1: {role} didn't start")
    for role, model in (("guarded", MODEL), ("eager", MODEL), ("q3guarded", Q3)):
        r = res.get(role) or {}
        if r.get("ready"):
            fails += [f"N1 ({role}): {f}" for f in _norm_line_ok(r.get("norms_at_start"), model)]
    for a, b in (("plain", "guarded"), ("q3plain", "q3guarded")):
        x, y = (res.get(a) or {}).get("identity"), (res.get(b) or {}).get("identity")
        if x is None or y is None or x != y or any(t is None for t in x):
            diff = sum(p != q for p, q in zip(x or [], y or []))
            fails.append(f"N2: {b}'s {IDENTITY} completions differ from {a}'s ({diff} differ)")
    for role in ("guarded", "eager", "q3guarded"):
        r = res.get(role) or {}
        if not r.get("ready"):
            continue
        if r.get("clean_alerts"):
            fails.append(f"N3 ({role}): {r['clean_alerts']} alerts with no fault")
        w = (r.get("norms_clean") or {}).get("worst")
        if w is None or not w <= BOUND:
            fails.append(f"N3 ({role}): the worst clean norm score was {w}, above the bound of {BOUND}")
    g = res.get("guarded") or {}
    sure = [t for t in g.get("trials") or [] if t["bit"] in SURE_BITS]
    counted = [t for t in sure if t.get("flip") and (t.get("flip") or {}).get("before")]
    missed = [t for t in counted if not t["detected"]]
    if g.get("ready") and (not counted or missed):
        fails.append(f"N4: {len(missed)} of {len(counted)} flips of bits {SURE_BITS} missed: "
                     f"{[(t['bit'], t['layer'], t.get('relative_change')) for t in missed]}")
    wrong_layer = [t for t in g.get("trials") or [] if t.get("finite_alerts_other_layers")]
    if wrong_layer:
        fails.append(f"N4: finite-score alerts on other layers in {len(wrong_layer)} trials: "
                     f"{[(t['bit'], t['layer'], t['other_layers']) for t in wrong_layer]}")
    not_back = [t for t in g.get("trials") or [] if t.get("flip") and
                (t.get("back") or {}).get("after") != (t.get("flip") or {}).get("before")]
    if not_back:
        fails.append(f"N4: {len(not_back)} flips weren't undone")
    ex = (res.get("eager") or {}).get("exact") or {}
    if (res.get("eager") or {}).get("ready"):
        ok = 0
        for t in ex.get("trials") or []:
            named = [n for n in t.get("named") or [] if n is not None]
            ok += bool(t["alerts"] and named and t.get("target") and
                       all(len(n) == 1 and (n[0] == t["target"] or n[0].startswith(t["target"] + "-")) for n in named))
        if ex.get("prompts_holding_marker_naturally"):
            fails.append(f"N5: the marker token is in other prompts too ({ex['prompts_holding_marker_naturally']})")
        if ok != MARK_TRIALS or len(ex.get("trials") or []) != MARK_TRIALS:
            fails.append(f"N5: exact naming in {ok} of {len(ex.get('trials') or [])} trials")
    q3 = res.get("q3guarded") or {}
    if q3.get("ready"):
        tr = q3.get("trials") or []
        if len(tr) != 4 or not all(t["detected"] for t in tr if (t.get("flip") or {}).get("before")) or \
                any(t.get("finite_alerts_other_layers") for t in tr):
            fails.append(f"N6: Qwen3 q/k norm flips: {[(t['layer'], t['detected'], t['other_layers']) for t in tr]}")
    s = summarize(res)
    c16 = (s["cost"].get("16") or {}).get("norm_cost")
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
