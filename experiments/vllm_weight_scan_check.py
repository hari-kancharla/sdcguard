"""The plugin's weight scan in a real `vllm serve` (docs/METHOD.md, "Scheduled weight scan in serving", written before
the run).

One H100, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16 (112 linear layers), sdcguard checking one linear layer in eight
(SDCGUARD_SHARE=0.125, the cheap setting the README shows), attention checks off (they have their own run). Two
servers, one after the other, the same but for the scan:

  noscan   SDCGUARD_WEIGHT_SCAN=0
  scan     SDCGUARD_WEIGHT_SCAN=1 (64 MB of weights compared at each read of the results, the default)

Each answers 64 prompts greedily (32 tokens). The scan server then takes the speed load with the scan on and paused
in turn on the same server (sdcguard_test_scan_pause; sdcguard.perf: 128 prompts, 128 tokens, one round at a time,
4 rounds each way in the order on, off, off, on, on, off, off, on, at 16 and at 256 requests at once), so the cost
isn't a difference between two server starts. Then the trials, one after another: one bit of one weight flipped
(through /collective_rpc, at a random layer, row and column, seed 0; bits 0 to 15, 3 trials each) and the prefix
cache emptied, rounds of 4 requests at once (the scan moves on only while the server runs model steps) until an
alert of the scan names the layer or 120 s have passed, then the bit flipped back, the layer's fingerprint taken
again (sdcguard_test_scan_refresh) and the prefix cache emptied, so the next trial starts from a clean model. The
trials stop early after 3 misses in a row or when the run's time is nearly up (they are then judged as not all run),
and what has been measured is written after each phase and each trial.

Amendment of October 25 (after the first run's result): the scan also holds the model's other parameters (the
embedding, which the LM head shares, and the norms' weights), and its fingerprint keys every value by its position.
After the 48 flips, 12 more trials on scan, the same way, each named by the scan entry that holds the changed value
(sdcguard_test_change_param): bits 0, 5, 10, 15 of the embedding, bits 0 and 14 of the final norm, bit 3 of an input
norm and bit 9 of a post-attention norm, two values swapped in two linear layers and in the embedding (seed 1). Then a
third server, moe_scan (granite-3.1-1b-a400m-instruct in bf16, 24 MoE layers, the scan on, SDCGUARD_REQUIRE off since
nothing checks its MoE layers here), answers 64 prompts and runs 6 trials in MoE experts' weights: bits 0 and 7 of
the first matmul's weights, 12 and 15 of the second's, one swap in each (seed 2). Each scan server reports the
model's parameters no scan entry holds (sdcguard_test_scan_uncovered).

    python experiments/vllm_weight_scan_check.py --out runs/weight_scan
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_norm_check as vn  # noqa: E402
import vllm_serve_check as vs  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCKS = 28
KINDS = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
LINEAR_LAYERS = BLOCKS * len(KINDS)
SHARE = "0.125"
BITS = tuple(range(16))
PER_BIT = 3
DETECT_S = 120.0
BURST = 4
IDENTITY = 64
COST_LIMIT = 0.01
ALTERNATIONS = ("on", "off", "off", "on", "on", "off", "off", "on")
MAX_MISSES_IN_A_ROW = 3
RUN_S = 50 * 60                          # inside the Modal phase's 55 minutes: the rest is for writing and stopping
MOE_MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
MOE_BLOCKS = 24
MOE_PROMPTS = 64                         # (enough model steps for a few passes before the trials)
WARM_S = 240.0                           # the most clean traffic warm_scan sends for the scan to compare every entry


def param_trials(seed: int = 1) -> list:
    """The amendment's trials outside the linear layers' weights (Qwen2.5-1.5B): bits flipped in the embedding (which
    the LM head shares), the final norm and two layer norms, and two values swapped in linear layers and in the
    embedding (a swap inside one block, which the earlier fingerprint couldn't see)."""
    rng = random.Random(seed)
    out = [{"prefix": "model.embed_tokens", "suffix": "weight", "frac": rng.random(), "op": "flip", "bit": b}
           for b in (0, 5, 10, 15)]
    out += [{"prefix": "model.norm", "suffix": "weight", "frac": rng.random(), "op": "flip", "bit": b} for b in (0, 14)]
    out += [{"prefix": f"model.layers.{rng.randrange(BLOCKS)}.{part}", "suffix": "weight", "frac": rng.random(),
             "op": "flip", "bit": b} for part, b in (("input_layernorm", 3), ("post_attention_layernorm", 9))]
    out += [{"prefix": f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}", "suffix": "weight",
             "frac": rng.random(), "op": "swap", "bit": 0} for _ in range(2)]
    out += [{"prefix": "model.embed_tokens", "suffix": "weight", "frac": rng.random(), "op": "swap", "bit": 0}
            for _ in range(2)]
    return out


def moe_trials(seed: int = 2) -> list:
    """MoE experts' weights (granite-3.1-1b-a400m): bits flipped in the first and second expert matmuls' weights of
    random layers, and one swap in each."""
    rng = random.Random(seed)
    out = [{"prefix": f"model.layers.{rng.randrange(MOE_BLOCKS)}.", "suffix": w, "frac": rng.random(), "op": "flip",
            "bit": b} for w, b in (("w13_weight", 0), ("w13_weight", 7), ("w2_weight", 12), ("w2_weight", 15))]
    out += [{"prefix": f"model.layers.{rng.randrange(MOE_BLOCKS)}.", "suffix": w, "frac": rng.random(), "op": "swap",
             "bit": 0} for w in ("w13_weight", "w2_weight")]
    return out


def trials(seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for bit in BITS:
        for _ in range(PER_BIT):
            out.append({"bit": bit, "layer": f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}",
                        "row_frac": rng.random(), "col_frac": rng.random()})
    return out


def server(role: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
    env["SDCGUARD_ATTN"] = "0"
    env.update(SDCGUARD="1", SDCGUARD_SHARE=SHARE, SDCGUARD_METRICS_PORT=str(metrics_port),
               SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
               SDCGUARD_WEIGHT_SCAN="0" if role == "noscan" else "1")
    if role != "moe_scan":            # (SDCGUARD_REQUIRE refuses a model whose MoE layers nothing checks)
        env["SDCGUARD_REQUIRE"] = "1"
    model = MOE_MODEL if role == "moe_scan" else MODEL
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role == "moe_scan":
        cmd += ["--dtype", "bfloat16", "--moe-backend", "triton"]
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def _scan_alerts(path: str) -> list:
    return [a for a in vn.alerts(path) if a.get("detector") == "serve_weights"]


def _empty_prefix_cache(port: int, tries: int = 5) -> bool:
    """POST /reset_prefix_cache (vLLM 0.32 dev mode), which answers {"success": false} while blocks are still held, so
    it is tried again for a few seconds; False if the cache could not be emptied."""
    for k in range(tries):
        try:
            status, r = vn._post(port, "/reset_prefix_cache", {}, timeout=60.0)
            if status == 200 and (not isinstance(r, dict) or r.get("success", True)):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5 * (k + 1))
    return False


def _pause(port: int, paused: bool) -> bool:
    """Pause or resume the scan; True only if the worker says it took."""
    try:
        r = vn.rpc(port, "sdcguard_test_scan_pause", [paused])
    except Exception:  # noqa: BLE001
        return False
    return isinstance(r, dict) and r.get("scan") is True and r.get("paused") is bool(paused)


def _speed_alternating(port: int) -> dict:
    """Output tokens per second with the scan on and paused in turn, on the same server: {concurrency: {"on": [...],
    "off": [...], "pause_failed": [...], "errors": [...]}} (one value per round, after one round left out so every
    measured round meets a warm server and prefix cache)."""
    from sdcguard import perf
    url = f"http://127.0.0.1:{port}"
    out: dict = {}
    for c in (16, 256):
        vals: dict = {"on": [], "off": [], "pause_failed": [], "errors": []}
        try:
            perf.measure(url, MODEL, concurrency=c, max_tokens=128, rounds=1, warmup=2)     # left out
        except Exception as e:  # noqa: BLE001
            vals["errors"].append(f"warm-up: {str(e)[:200]}")
        for k, mode in enumerate(ALTERNATIONS):
            if not _pause(port, mode == "off"):
                vals["pause_failed"].append(f"round {k} ({mode})")
                continue
            try:
                r = perf.measure(url, MODEL, concurrency=c, max_tokens=128, rounds=1, warmup=2)
                v = (r.get("median") or {}).get("output_tokens_per_s")
                if v is not None and r.get("complete"):
                    vals[mode].append(v)
                else:
                    vals["errors"].append(f"round {k} ({mode}): incomplete")
            except Exception as e:  # noqa: BLE001
                vals["errors"].append(f"round {k} ({mode}): {str(e)[:200]}")
        if not _pause(port, False):
            vals["pause_failed"].append("resume after the rounds")
        out[str(c)] = vals
    return out


def warm_scan(port: int, model: str, prompts: list, deadline: float, limit_s: float = WARM_S) -> dict:
    """Clean traffic, BURST requests at a time, until the scan has compared every entry at least once (none in its
    coverage's never_compared), for at most limit_s seconds and never past the deadline less 3 minutes: the coverage
    requirement holds the scan to every entry compared before any fault, which a short clean load (the smoke test's)
    may not reach. {"rounds", "seconds", "left": entries still never compared}."""
    t0 = time.monotonic()
    rounds, left = 0, None
    while True:
        st = vn.rpc(port, "sdcguard_test_scan_stats")
        cov = (st.get("coverage") or {}) if isinstance(st, dict) else {}
        left = len(cov.get("never_compared") or []) if isinstance(cov.get("never_compared"), list) else None
        if left == 0 or time.monotonic() - t0 > limit_s or time.monotonic() + 180 > deadline:
            break
        with ThreadPoolExecutor(BURST) as ex:
            list(ex.map(lambda k: vn._try(port, model, prompts[(rounds * BURST + k) % len(prompts)], 16),
                        range(BURST)))
        rounds += 1
    return {"rounds": rounds, "seconds": round(time.monotonic() - t0, 1), "left": left}


def _write(out: str, name: str, data) -> None:
    with open(os.path.join(out, name), "w") as f:
        json.dump(data, f, indent=1, default=str)


def _refreshed(r) -> Optional[int]:
    """How many scan entries the refresh RPC refreshed (None if it didn't answer as expected)."""
    v = r.get("refreshed") if isinstance(r, dict) else None
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _checksum_late(rows: list, log_path: str, end: float) -> None:
    """The arithmetic checks' alerts on each flipped layer from its flip to the next trial's flip (the guard reads
    their results a read late, so some land after the trial's own window). Reported only."""
    al = [a for a in vn.alerts(log_path) if a.get("detector") == "serve_checksum"]
    tr = [r for r in rows if "bit" in r and isinstance(r.get("t_flip"), (int, float))]
    for k, r in enumerate(tr):
        t1 = tr[k + 1]["t_flip"] if k + 1 < len(tr) else end
        r["checksum_alerts_on_layer_until_next"] = sum(
            1 for a in al if (a.get("detail") or {}).get("where") == r["layer"] and
            isinstance(a.get("time"), (int, float)) and r["t_flip"] <= a["time"] < t1)


def run_trials(port: int, log_path: str, prompts: list, deadline: float, out: str) -> list:
    rows: list = []
    try:
        _run_trials(port, log_path, prompts, deadline, out, rows)
    except Exception as e:  # noqa: BLE001  (a server that died mid-trial: what was measured is kept and judged)
        rows.append({"stopped": f"error during trial {sum(1 for r in rows if 'bit' in r)}: {type(e).__name__}: "
                                f"{str(e)[:300]}"})
    try:
        _checksum_late(rows, log_path, time.time())
    except Exception:  # noqa: BLE001
        pass
    _write(out, "PROGRESS.json", {"done": len(rows), "rows": rows})
    return rows


def _run_trials(port: int, log_path: str, prompts: list, deadline: float, out: str, rows: list) -> None:
    misses = 0
    for i, t in enumerate(trials()):
        if time.monotonic() + DETECT_S + 30 > deadline:
            rows.append({"stopped": f"the run's time was nearly up before trial {i}"})
            break
        if misses >= MAX_MISSES_IN_A_ROW:
            rows.append({"stopped": f"{misses} misses in a row before trial {i}"})
            break
        n0 = len(vn.alerts(log_path))
        t0 = time.time()                       # before the flip: seconds to detect count from here
        flip = vn.rpc(port, "sdcguard_test_flip_bit", [t["layer"], t["row_frac"], t["col_frac"], t["bit"]])
        emptied = [_empty_prefix_cache(port)]
        hit, rounds = None, 0
        while time.time() - t0 < DETECT_S:
            with ThreadPoolExecutor(BURST) as ex:
                list(ex.map(lambda k: vn._try(port, MODEL, prompts[(rounds * BURST + k) % len(prompts)], 16),
                            range(BURST)))
            rounds += 1
            new = [a for a in vn.alerts(log_path)[n0:] if a.get("detector") == "serve_weights"]
            mine = [a for a in new if (a.get("detail") or {}).get("where") == t["layer"]]
            if mine:
                hit = mine[0]
                break
        detect_s = (hit.get("time") - t0) if hit is not None and isinstance(hit.get("time"), (int, float)) else None
        in_time = detect_s is not None and detect_s <= DETECT_S
        back = vn.rpc(port, "sdcguard_test_flip_bit", [t["layer"], t["row_frac"], t["col_frac"], t["bit"]])
        refreshed = _refreshed(vn.rpc(port, "sdcguard_test_scan_refresh", [t["layer"]]))
        emptied.append(_empty_prefix_cache(port))
        misses = 0 if in_time else misses + 1
        new = vn.alerts(log_path)[n0:]
        rows.append({**t, "t_flip": t0, "flip": flip, "back": back, "refreshed": refreshed, "cache_emptied": emptied,
                     "detected": in_time, "alerted": hit is not None,
                     "seconds_to_detect": round(detect_s, 3) if detect_s is not None else None, "rounds": rounds,
                     "kind": (hit.get("detail") or {}).get("kind") if hit is not None else None,
                     "scan_alerts_other_layers": sum(1 for a in new if a.get("detector") == "serve_weights"
                                                     and (a.get("detail") or {}).get("where") != t["layer"]),
                     "checksum_alerts_on_layer": sum(1 for a in new if a.get("detector") == "serve_checksum"
                                                     and (a.get("detail") or {}).get("where") == t["layer"])})
        _write(out, "PROGRESS.json", {"done": len(rows), "rows": rows})


def run_param_trials(port: int, log_path: str, model: str, prompts: list, trials_: list, deadline: float, out: str,
                     name: str) -> list:
    rows: list = []
    try:
        _run_param_trials(port, log_path, model, prompts, trials_, deadline, out, name, rows)
    except Exception as e:  # noqa: BLE001  (a server that died mid-trial: what was measured is kept and judged)
        rows.append({"stopped": f"error during trial {sum(1 for r in rows if 'op' in r)}: {type(e).__name__}: "
                                f"{str(e)[:300]}"})
    _write(out, f"PROGRESS_{name}.json", {"done": len(rows), "rows": rows})
    return rows


def _run_param_trials(port: int, log_path: str, model: str, prompts: list, trials_: list, deadline: float, out: str,
                      name: str, rows: list) -> None:
    """Each trial: the change made through /collective_rpc (the worker says which scan entry holds the value), rounds
    of 4 requests until a scan alert names that entry or DETECT_S have passed since just before the change, the
    change undone, that entry's fingerprint taken again and the prefix cache emptied."""
    misses = 0
    for i, t in enumerate(trials_):
        if time.monotonic() + DETECT_S + 30 > deadline:
            rows.append({"stopped": f"the run's time was nearly up before trial {i}"})
            break
        if misses >= MAX_MISSES_IN_A_ROW:
            rows.append({"stopped": f"{misses} misses in a row before trial {i}"})
            break
        n0 = len(vn.alerts(log_path))
        t0 = time.time()
        ch = vn.rpc(port, "sdcguard_test_change_param", [t["prefix"], t["suffix"], t["frac"], t["op"], t["bit"]])
        ch = ch if isinstance(ch, dict) else {}
        entry = ch.get("entry") or ""
        emptied = [_empty_prefix_cache(port)]
        hit, rounds = None, 0
        while entry and time.time() - t0 < DETECT_S:
            with ThreadPoolExecutor(BURST) as ex:
                list(ex.map(lambda k: vn._try(port, model, prompts[(rounds * BURST + k) % len(prompts)], 16),
                            range(BURST)))
            rounds += 1
            mine = [a for a in vn.alerts(log_path)[n0:] if a.get("detector") == "serve_weights" and
                    (a.get("detail") or {}).get("where") == entry]
            if mine:
                hit = mine[0]
                break
        detect_s = (hit.get("time") - t0) if hit is not None and isinstance(hit.get("time"), (int, float)) else None
        in_time = detect_s is not None and detect_s <= DETECT_S
        back = {}
        if isinstance(ch.get("index"), int):
            back = vn.rpc(port, "sdcguard_test_change_param", [t["prefix"], t["suffix"], t["frac"], t["op"], t["bit"],
                                                               ch["index"]])
            back = back if isinstance(back, dict) else {}
        refreshed = _refreshed(vn.rpc(port, "sdcguard_test_scan_refresh_entry", [entry])) if entry else None
        emptied.append(_empty_prefix_cache(port))
        misses = 0 if in_time else misses + 1
        new = vn.alerts(log_path)[n0:]
        rows.append({**t, "t_change": t0, "change": ch, "back": back, "entry": entry, "refreshed": refreshed,
                     "cache_emptied": emptied, "detected": in_time, "alerted": hit is not None,
                     "seconds_to_detect": round(detect_s, 3) if detect_s is not None else None, "rounds": rounds,
                     "undone": bool(ch.get("before")) and back.get("after") == ch.get("before"),
                     "scan_alerts_other_entries": sum(1 for a in new if a.get("detector") == "serve_weights"
                                                      and (a.get("detail") or {}).get("where") != entry)})
        _write(out, f"PROGRESS_{name}.json", {"done": len(rows), "rows": rows})


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    deadline = time.monotonic() + RUN_S
    res: dict = {"model": MODEL, "share": SHARE}
    ps = vs.prompts(IDENTITY + 16)
    tp = ps[IDENTITY:]
    for role in ("noscan", "scan", "moe_scan"):
        if role == "moe_scan" and time.monotonic() + 10 * 60 > deadline:
            res[role] = {"skipped": "the run's time was nearly up"}
            continue
        proc, log = server(role, out, port, metrics_port)
        r: dict = {}
        res[role] = r
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        model = MOE_MODEL if role == "moe_scan" else MODEL
        try:
            # (a slow start can't take the run past the Modal phase's limit: SUMMARY.json is always written)
            r["ready"] = vs.wait_ready(port, proc, timeout=max(60.0, min(1500.0, deadline - time.monotonic() - 300)))
            if not r["ready"]:
                continue
            n_prompts = MOE_PROMPTS if role == "moe_scan" else IDENTITY
            texts = [vn._try(port, model, p).get("text") for p in ps[:n_prompts]]
            if role != "moe_scan":
                r["identity"] = texts
            if role == "scan":
                r["speed"] = _speed_alternating(port)
            if role != "noscan":                # every entry compared before the clean baseline and any fault
                r["warm"] = warm_scan(port, model, ps, deadline)
            time.sleep(3)
            vn._try(port, model, ps[0], 8)
            r["clean_scan_alerts"] = len(_scan_alerts(log_path))
            r["clean_alerts"] = len(vn.alerts(log_path))
            if role != "noscan":
                r["scan_at_clean"] = vn.rpc(port, "sdcguard_test_scan_stats")
                r["uncovered"] = vn.rpc(port, "sdcguard_test_scan_uncovered")
                _write(out, "PARTIAL.json", res)
            if role == "scan":
                r["trials"] = run_trials(port, log_path, tp, deadline, out)
                r["param_trials"] = run_param_trials(port, log_path, MODEL, tp, param_trials(), deadline, out,
                                                     "params")
            if role == "moe_scan":
                r["param_trials"] = run_param_trials(port, log_path, MOE_MODEL, tp, moe_trials(), deadline, out,
                                                     "moe")
            if role != "noscan":
                r["scan_at_end"] = vn.rpc(port, "sdcguard_test_scan_stats")
            r["alerts_total"] = len(vn.alerts(log_path))
        except Exception as e:  # noqa: BLE001  (a server that died: the run is still judged on what was measured)
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
            _write(out, "PARTIAL.json", res)
    return res


def _median(vals: list):
    v = sorted(vals)
    if not v:
        return None
    return v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2


def _trials(s: dict) -> list:
    return [t for t in s.get("trials") or [] if "bit" in t]


def summarize(res: dict) -> dict:
    s = res.get("scan") or {}
    tr = [t for t in _trials(s) if (t.get("flip") or {}).get("layer")]
    secs = sorted(t["seconds_to_detect"] for t in tr if t.get("seconds_to_detect") is not None)
    cost = {}
    each_way = ALTERNATIONS.count("on")
    for c, vals in (s.get("speed") or {}).items():
        on, off = _median(vals.get("on") or []), _median(vals.get("off") or [])
        if on and off:
            # measured: every pause and resume took and all rounds each way came back (else not used to decide)
            cost[c] = {"scan_on": on, "scan_off": off, "scan_cost": round(1 - on / off, 4),
                       "rounds_on": vals.get("on"), "rounds_off": vals.get("off"),
                       "measured": not vals.get("pause_failed") and len(vals.get("on") or []) == each_way and
                       len(vals.get("off") or []) == each_way,
                       "pause_failed": vals.get("pause_failed") or [], "errors": vals.get("errors") or []}
    by_bit = {}
    for t in tr:
        b = by_bit.setdefault(t["bit"], {"trials": 0, "detected": 0})
        b["trials"] += 1
        b["detected"] += bool(t["detected"])
    return {"trials": len(tr), "detected": sum(bool(t["detected"]) for t in tr), "by_bit": by_bit,
            "alerted_after_the_limit": sum(1 for t in tr if t.get("alerted") and not t.get("detected")),
            "checksum_detections_until_next_trial": sum(1 for t in tr if t.get("checksum_alerts_on_layer_until_next")),
            "stopped": [t["stopped"] for t in s.get("trials") or [] if "stopped" in t],
            "seconds_to_detect": {"median": _median(secs), "max": secs[-1] if secs else None},
            "detected_in_unchecked_layers": sum(1 for t in tr if t["detected"] and t.get("kind") == "unchecked"),
            "checksum_detections": sum(1 for t in tr if t.get("checksum_alerts_on_layer")),
            "scan_alerts_other_layers": sum(t.get("scan_alerts_other_layers") or 0 for t in tr),
            "prefix_cache_not_emptied": sum(1 for t in tr for e in t.get("cache_emptied") or [] if not e),
            "cost": cost, "coverage": (s.get("scan_at_clean") or {}).get("coverage"),
            "passes": ((s.get("scan_at_end") or {}).get("stats") or {}).get("passes"),
            "kernel": ((s.get("scan_at_clean") or {}).get("coverage") or {}).get("kernel"),
            "last_pass_seconds": ((s.get("scan_at_end") or {}).get("stats") or {}).get("last_pass_seconds"),
            "oldest_verification_age_seconds": (((s.get("scan_at_end") or {}).get("coverage") or {})
                                                .get("oldest_verification_age_seconds")),
            "params": _param_summary(s), "moe": _param_summary(res.get("moe_scan") or {}),
            "moe_coverage": ((res.get("moe_scan") or {}).get("scan_at_clean") or {}).get("coverage"),
            "moe_kernel": (((res.get("moe_scan") or {}).get("scan_at_clean") or {}).get("coverage") or {})
            .get("kernel")}


def _ptrials(r: dict) -> list:
    return [t for t in r.get("param_trials") or [] if "op" in t]


def _param_summary(r: dict) -> dict:
    tr = _ptrials(r)
    secs = sorted(t["seconds_to_detect"] for t in tr if t.get("seconds_to_detect") is not None)
    return {"trials": len(tr), "detected": sum(bool(t.get("detected")) for t in tr),
            "by_trial": [(t["prefix"] + "." + t["suffix"], t["op"], t["bit"], t.get("entry"), t.get("detected"),
                          t.get("seconds_to_detect")) for t in tr],
            "stopped": [t["stopped"] for t in r.get("param_trials") or [] if "stopped" in t],
            "seconds_to_detect": {"median": _median(secs), "max": secs[-1] if secs else None}}


def _param_fails(r: dict, tag: str, want: int) -> list:
    """The trials outside the linear layers (W6, W7): all ran, each change made where a scan entry holds it, named
    by a scan alert on that entry within DETECT_S, undone and its entry's fingerprint taken again, and no scan alert
    on another entry."""
    fails = []
    tr = _ptrials(r)
    if len(tr) != want:
        stopped = [t["stopped"] for t in r.get("param_trials") or [] if "stopped" in t]
        fails.append(f"{tag}: {len(tr)} trials ran, not {want}" + (f" ({stopped[0]})" if stopped else ""))
    nowhere = [(t["prefix"], t["suffix"], t.get("change")) for t in tr if not t.get("entry")]
    if nowhere:
        fails.append(f"{tag}: {len(nowhere)} changes weren't made where a scan entry holds them: {nowhere[:3]}")
    missed = [(t["prefix"], t["suffix"], t["op"], t["bit"]) for t in tr if t.get("entry") and not t.get("detected")]
    if missed:
        fails.append(f"{tag}: {len(missed)} changes not named within {DETECT_S:g} s: {missed[:6]}")
    undone = [t["entry"] for t in tr if t.get("entry") and not t.get("undone")]
    if undone:
        fails.append(f"{tag}: {len(undone)} changes weren't undone: {undone[:3]}")
    unref = [t["entry"] for t in tr if t.get("entry") and (not isinstance(t.get("refreshed"), int) or
                                                            t["refreshed"] < 1)]
    if unref:
        fails.append(f"{tag}: {len(unref)} entries' fingerprints weren't taken again: {unref[:3]}")
    other = sum(t.get("scan_alerts_other_entries") or 0 for t in tr)
    if other:
        fails.append(f"{tag}: {other} scan alerts named an entry with no change")
    return fails


def _coverage_fails(r: dict, tag: str) -> list:
    """Every parameter of the model is held by a scan entry, nothing is listed as not scanned, and every entry was
    compared before the trials."""
    u = r.get("uncovered") if isinstance(r.get("uncovered"), dict) else {}     # (vLLM gives a string for a non-dict)
    cov = (r.get("scan_at_clean") or {}).get("coverage") or {}
    fails = []
    if not isinstance(u.get("uncovered"), list) or u.get("uncovered") or u.get("not_scanned"):
        fails.append(f"{tag}: parameters no scan entry holds: {(u.get('uncovered') or [])[:5]} (not scanned: "
                     f"{u.get('not_scanned')}, of {u.get('parameters')} parameters)")
    if cov.get("never_compared"):
        fails.append(f"{tag}: {len(cov['never_compared'])} entries never compared before the trials: "
                     f"{cov['never_compared'][:3]}")
    return fails


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Scheduled weight scan in serving". The scan goes on by
    default only if every criterion is met and its measured cost is at most COST_LIMIT at 16 and at 256 at once
    (medians)."""
    fails = []
    for role in ("noscan", "scan"):
        if not (res.get(role) or {}).get("ready"):
            fails.append(f"W1: {role} didn't start")
        if (res.get(role) or {}).get("error"):
            fails.append(f"run: {role} stopped with an error ({res[role]['error']})")
    s = res.get("scan") or {}
    cov = (s.get("scan_at_clean") or {}).get("coverage") or {}
    kinds = cov.get("by_kind") or {}
    linear = kinds.get("checked", 0) + kinds.get("unchecked", 0)
    if s.get("ready"):
        if linear != LINEAR_LAYERS or not kinds.get("other"):
            fails.append(f"W1: the scan has {linear} linear layers (not {LINEAR_LAYERS}) and {kinds.get('other', 0)} "
                         f"entries of other parameters ({kinds})")
        fails += _coverage_fails(s, "W1")
    for role in ("noscan", "scan"):
        r = res.get(role) or {}
        if r.get("ready") and r.get("clean_alerts"):
            fails.append(f"W2: {r.get('clean_alerts')} alerts on {role} with no fault")
    tr = _trials(s)
    if s.get("ready"):
        if len(tr) != len(BITS) * PER_BIT:
            stopped = [t["stopped"] for t in s.get("trials") or [] if "stopped" in t]
            fails.append(f"W3: {len(tr)} trials ran, not {len(BITS) * PER_BIT}" + (f" ({stopped[0]})" if stopped else ""))
        bad = [t for t in tr if not (t.get("flip") or {}).get("layer")]
        if bad:
            fails.append(f"W3: {len(bad)} flips didn't happen: {[t.get('flip') for t in bad][:3]}")
        unrefreshed = [t["layer"] for t in tr if not isinstance(t.get("refreshed"), int) or t["refreshed"] < 1]
        if unrefreshed:
            fails.append(f"W3: {len(unrefreshed)} layers' fingerprints weren't taken again after their trial: "
                         f"{unrefreshed[:3]}")
        missed = [(t["bit"], t["layer"]) for t in tr if (t.get("flip") or {}).get("layer") and not t.get("detected")]
        if missed:
            fails.append(f"W3: {len(missed)} flips not named within {DETECT_S:g} s of the flip: {missed[:6]}")
        other = sum(t.get("scan_alerts_other_layers") or 0 for t in tr)
        if other:
            fails.append(f"W3: {other} scan alerts named a layer with no flip")
        undone = [t for t in tr if (t.get("flip") or {}).get("layer") and
                  (t.get("back") or {}).get("after") != (t.get("flip") or {}).get("before")]
        if undone:
            fails.append(f"W3: {len(undone)} flips weren't undone")
    m = res.get("moe_scan") or {}
    for r, who in ((s, "scan"), (m, "moe_scan")):   # W5 (amendment of October 25): the scans used the Triton kernel
        if not r.get("ready"):
            continue
        for when in ("scan_at_clean", "scan_at_end"):   # (at the end too: a launch that failed turns it off)
            kern = (((r.get(when) or {}) if isinstance(r.get(when), dict) else {}).get("coverage") or {}).get(
                "kernel") or {}
            if not any(k.startswith("cuda") and v == "Triton" for k, v in kern.items()) or \
                    any(k.startswith("cuda") and v != "Triton" for k, v in kern.items()):
                fails.append(f"W5: {who}'s fingerprint didn't use the Triton kernel on the GPU ({when}): {kern}")
    if s.get("ready"):                         # W6: parameters outside the linear layers, and swaps
        fails += _param_fails(s, "W6", len(param_trials()))
    if not m.get("ready"):                     # W7: MoE experts' weights (granite-3.1-1b-a400m)
        fails.append(f"W7: moe_scan didn't start ({m.get('skipped') or m.get('error') or 'not ready'})")
    else:
        if m.get("error"):
            fails.append(f"run: moe_scan stopped with an error ({m['error']})")
        mk = (((m.get("scan_at_clean") or {}).get("coverage") or {}).get("by_kind")) or {}
        if not mk.get("other"):
            fails.append(f"W7: no entries of the MoE model's other parameters ({mk})")
        fails += _coverage_fails(m, "W7")
        if m.get("clean_alerts"):
            fails.append(f"W7: {m['clean_alerts']} alerts on moe_scan with no fault")
        fails += _param_fails(m, "W7", len(moe_trials()))
    x, y = (res.get("noscan") or {}).get("identity"), s.get("identity")
    if x is None or y is None or x != y or any(t is None for t in x):
        fails.append(f"W4: the scan server's completions differ from noscan's "
                     f"({sum(p != q for p, q in zip(x or [], y or []))} differ)")
    summ = summarize(res)
    # on by default only with every criterion met and a measured cost of at most COST_LIMIT at 16 and at 256 at once
    # (the amendment of October 25; the first run decided on 16 alone)
    costs = [((summ["cost"].get(c) or {}).get("scan_cost") if (summ["cost"].get(c) or {}).get("measured") else None)
             for c in ("16", "256")]
    summ["on_by_default"] = (not fails) and all(c is not None and c <= COST_LIMIT for c in costs)
    # (W1 to W7, the amendment of October 25 written after the first run's result)
    return not fails, fails, summ


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
