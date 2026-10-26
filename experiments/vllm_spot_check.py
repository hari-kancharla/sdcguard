"""Sampled higher-precision verification inside a real `vllm serve` (docs/METHOD.md, "Spot checks in serving",
written before the run).

vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16 (112 linear layers), the checksum check on every layer
(SDCGUARD_SHARE=1, so every linear layer is spot-checked too), attention checks off (their own run), prefix caching
off. On one H100, servers one after another:

  nospot   SDCGUARD_SPOT=0, compiled with CUDA graphs: the reference answers and speed
  spot     SDCGUARD_SPOT=1, compiled with CUDA graphs: its answers, speed, and clean traffic
  eager    SDCGUARD_SPOT=1 with --enforce-eager: the trials
  faulted  SDCGUARD_SPOT=1, compiled with CUDA graphs, one layer's outputs 2% too large from model load on (in the
           compiled graph: SDCGUARD_TEST_COMPILED_FAULT, its own compile cache): the compiled path's positive control

nospot and spot answer 64 prompts greedily (32 tokens) and take the speed load (sdcguard.perf: 128 prompts, 128
tokens, 3 rounds, 16 and 256 requests at once). Every spot-checked layer's turns read back must go up during that
traffic (spot) or during eager's 16 prompts (eager), not just at warm-up. eager answers 16 prompts, then runs the
trials, one after another, each on a linear layer drawn with seed 0, rounds of 4 requests at once until a spot alert
names the layer or ROUNDS rounds have passed, then undone:
  criteria   every output 2% too large (3 trials) and 5% too large (3 trials) (sdcguard_test_scale_outputs: its outputs
             multiplied after its GEMM, before the checks); one sampled output 2% too large at each of the layer's
             turns, the sampled output of largest magnitude (3 trials; sdcguard_test_spot_one: a known sampled single
             wrong value, the rest of the layer right)
  reported   every output 1% too large (2 trials); a band of 64 columns 3% too large (3 trials); one sampled output 2%
             too large, the sampled output of median magnitude (3 trials)
faulted runs the same rounds until a spot alert names its layer (drawn with seed 1), with no fault undone (it is in the
graph). A round is about 33 model steps, so the faulty layer gets about 11 turns in ROUNDS rounds. Each turn samples
a band of 64 columns about once every N / (8 x 64) turns (the column order spreads them): within 11 turns from any
starting turn for the qkv band drawn (N = 2,048) and from 31% of starting turns for each gate_up band (N = 17,920),
computed exactly from the order (band_odds); each trial records the layer's turns and those odds. A speed load
counts only when sdcguard.perf calls it complete (every request of every round answered). With --gpu l4 only the
spot server runs (its clean traffic on another GPU generation), in RUN_S["l4"], inside that function's shorter limit.

    python experiments/vllm_spot_check.py --out runs/spot [--gpu l4]
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
IDENTITY = 64
ROUNDS = 40
BURST = 4
CLEAN_LIMIT = 1.5
CRITERIA_TRIALS = (("all", 1.02), ("all", 1.02), ("all", 1.02), ("all", 1.05), ("all", 1.05), ("all", 1.05))
REPORTED_TRIALS = (("all", 1.01), ("all", 1.01), ("band", 1.03), ("band", 1.03), ("band", 1.03))
# the sampled single wrong values (amendment of October 26), drawn with a seed of their own so the trials above keep
# their layers (and the band odds computed for them)
ONE_CRITERIA = (("one", 1.02), ("one", 1.02), ("one", 1.02))
ONE_REPORTED = (("one_median", 1.02), ("one_median", 1.02), ("one_median", 1.02))
COMPILED_FACTOR = 1.02
BAND = 64
COST_LIMIT = 0.02
# each inside its Modal function's limit (scripts/modal_vllm.py: 55 and 30 minutes), with time left to write results
RUN_S = {"h100": 50 * 60, "l4": 25 * 60}
FAULTED_RESERVE = 10 * 60     # held back from the eager trials for the faulted server: eager's stop, a fresh compile
MIN_START = {"faulted": 7 * 60}               # (2 to 4 minutes), its trial; a server isn't started with less time left
                                              # (the others: 5 minutes)


def compiled_layer(seed: int = 1) -> str:
    """The layer the faulted server's fault is compiled into."""
    rng = random.Random(seed)
    return f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}"


def trials(seed: int = 0, one_seed: int = 2) -> list:
    """The eager trials, the criteria's first (S4's, then S5's), then the reported ones; each group drawn as before
    (S4's and the first reported ones with seed, S5's and theirs with one_seed), only the running order changed."""
    drawn = {}
    for sd, groups in ((seed, (("s4", True, CRITERIA_TRIALS), ("rep", False, REPORTED_TRIALS))),
                       (one_seed, (("s5", True, ONE_CRITERIA), ("one_rep", False, ONE_REPORTED)))):
        rng = random.Random(sd)
        for name, crit, group in groups:
            drawn[name] = [{"criterion": crit, "kind": kind, "factor": factor,
                            "layer": f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}",
                            "band_frac": rng.random()} for kind, factor in group]
    return drawn["s4"] + drawn["s5"] + drawn["rep"] + drawn["one_rep"]


def server(role: str, out: str, port: int, metrics_port: int, cache: Optional[str] = None):
    env = dict(os.environ)
    for k in ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "SDCGUARD_TEST_COMPILED_FAULT"):
        env.pop(k, None)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
    env["SDCGUARD_ATTN"] = "0"
    env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_SHARE="1",
               SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
               SDCGUARD_SPOT="0" if role == "nospot" else "1")
    if role == "faulted":
        # the fault is traced into the compiled graph at load: a compile cache of its own, so no clean graph is
        # loaded in its place and its graph is never loaded by another server
        # (local disk, not the results folder on the network volume: thousands of small files, removed after)
        cache = cache or os.path.join(out, "faulted_cache")
        env.update(SDCGUARD_TEST_COMPILED_FAULT=f"{compiled_layer()}:{COMPILED_FACTOR}", SDCGUARD_SHARED_CACHE="1",
                   VLLM_CACHE_ROOT=os.path.join(cache, "vllm"), TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"))
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--no-enable-prefix-caching", "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    if role == "eager":
        cmd.append("--enforce-eager")
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def _spot_alerts(path: str, n0: int = 0) -> list:
    return [a for a in vn.alerts(path)[n0:] if a.get("detector") == "serve_spot"]


def band_odds(width: int, c0: int, w: int, turns: int):
    """The share of starting turns from which the spot check's column order (serve_spot.sampled, deterministic)
    samples the band of columns [c0, c0 + w) of a layer N = width wide within `turns` of its turns, exactly, over one
    period of the order (N / SPOT_COLS turns when SPOT_COLS divides N, else N)."""
    import torch
    from sdcguard import serve_spot as sp
    if not turns or turns <= 0:
        return None
    period = width // sp.SPOT_COLS if width % sp.SPOT_COLS == 0 else width
    t = torch.arange(period, dtype=torch.int64)
    off = (t % width) * sp.col_step(width) % width
    cols = (off[:, None] + torch.arange(sp.SPOT_COLS)[None, :] * max(width // sp.SPOT_COLS, 1)) % width
    hit = ((cols - c0) % width < w).any(1).to(torch.int64)
    if turns >= period:
        return 1.0 if bool(hit.any()) else 0.0
    c = torch.cat([torch.zeros(1, dtype=torch.int64), torch.cat([hit, hit]).cumsum(0)])
    within = c[torch.arange(period) + turns] - c[torch.arange(period)]
    return round(float((within > 0).double().mean()), 4)


def _turns(port: int, layer: str):
    """The layer's spot-checked turns read back so far (None if not known)."""
    st = vn.rpc(port, "sdcguard_test_spot_stats")
    return (st.get("turns") or {}).get(layer) if isinstance(st, dict) else None


def run_trials(port: int, log_path: str, prompts: list, deadline: float) -> list:
    rows = []
    for i, t in enumerate(trials()):
        if time.monotonic() + 120 > deadline:
            rows.append({"stopped": f"the run's time was nearly up before trial {i}"})
            break
        width = None
        turns0 = _turns(port, t["layer"])
        n0 = len(vn.alerts(log_path))
        start = 0
        if t["kind"] == "band":
            probe = vn.rpc(port, "sdcguard_test_scale_outputs", [t["layer"], 1.0, 0, 1])
            vn.rpc(port, "sdcguard_test_unmark_rows")
            width = (probe or {}).get("width") if isinstance(probe, dict) else None
            start = int(t["band_frac"] * max(1, (width or BAND) - BAND))
        if t["kind"] in ("one", "one_median"):
            mark = vn.rpc(port, "sdcguard_test_spot_one", [t["layer"], t["factor"],
                                                           "largest" if t["kind"] == "one" else "median"])
        else:
            mark = vn.rpc(port, "sdcguard_test_scale_outputs", [t["layer"], t["factor"], start,
                                                                BAND if t["kind"] == "band" else 0])
        hit, sent, cut = [], 0, False
        try:
            for r in range(ROUNDS):
                if time.monotonic() + 60 > deadline:      # (leave time to undo the fault and stop the server)
                    cut = True
                    break
                with ThreadPoolExecutor(BURST) as ex:
                    list(ex.map(lambda k: vn._try(port, MODEL, prompts[(r * BURST + k) % len(prompts)], 32),
                                range(BURST)))
                sent = r + 1
                hit = [a for a in _spot_alerts(log_path, n0) if (a.get("detail") or {}).get("where") == t["layer"]]
                if hit:
                    break
        finally:
            changed = vn.rpc(port, "sdcguard_test_spot_one_log") if t["kind"] in ("one", "one_median") else None
            unmark = vn.rpc(port, "sdcguard_test_unmark_rows")
        time.sleep(3)
        vn._try(port, MODEL, prompts[0], 4)
        new = vn.alerts(log_path)[n0:]
        # every alert naming the layer, those of the last round too (they can arrive with the read after it)
        hit = [a for a in new if a.get("detector") == "serve_spot" and (a.get("detail") or {}).get("where") == t["layer"]]
        turns1 = _turns(port, t["layer"])
        turns = turns1 - turns0 if isinstance(turns0, int) and isinstance(turns1, int) else None
        w = _d(mark).get("width") or width
        odds = None
        if t["kind"] == "band" and w:
            odds = {"per_turn": round(min(1.0, BAND * 8 / w), 4), "in_turns": band_odds(w, start, BAND, turns)}
        rows.append({**t, "mark": mark, "unmark": unmark, "changed": changed if isinstance(changed, list) else None,
                     "detected": bool(hit), "rounds": sent if hit else None,
                     "rounds_sent": sent, "cut_by_deadline": cut, "turns": turns, "band_odds": odds,
                     "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict")
                                         for a in hit} - {None}),
                     "relative_errors": [(a.get("detail") or {}).get("relative_error") for a in hit[:3]],
                     "spot_alerts_other_layers": sum(1 for a in new if a.get("detector") == "serve_spot" and
                                                     (a.get("detail") or {}).get("where") != t["layer"]),
                     "checksum_alerts": sum(1 for a in new if a.get("detector") == "serve_checksum")})
    return rows


def served_stack(metrics_port: int) -> dict:
    """How the server says it runs (its sdcguard_stack_info): enforce_eager, CUDA graph and compilation modes."""
    from sdcguard.stack import from_metrics
    try:
        status, raw = vs._request(metrics_port, "GET", "/metrics", timeout=10)
        st = (from_metrics(raw.decode(errors="replace")) or {}) if status == 200 else {}
    except Exception:  # noqa: BLE001
        st = {}
    return {k: st.get(k) for k in ("eager", "cudagraph", "compilation")}


def compiled_trial(port: int, log_path: str, prompts: list, deadline: float, layer: str) -> dict:
    """Rounds of BURST requests on the faulted server until a spot alert names its layer (or ROUNDS rounds). Only
    alerts after the trial starts count (the fault is there from load, so warm-up can alert too: those are counted
    apart), with the layer's checked turns read before and after."""
    n0 = len(vn.alerts(log_path))
    before = sum(1 for a in vn.alerts(log_path) if a.get("detector") == "serve_spot" and
                 (a.get("detail") or {}).get("where") == layer)
    turns0 = _turns(port, layer)
    hit, sent, cut = [], 0, False
    for r in range(ROUNDS):
        if time.monotonic() + 60 > deadline:
            cut = True
            break
        with ThreadPoolExecutor(BURST) as ex:
            list(ex.map(lambda k: vn._try(port, MODEL, prompts[(r * BURST + k) % len(prompts)], 32), range(BURST)))
        sent = r + 1
        hit = [a for a in _spot_alerts(log_path, n0) if (a.get("detail") or {}).get("where") == layer]
        if hit:
            break
    time.sleep(3)
    vn._try(port, MODEL, prompts[0], 4)
    new = vn.alerts(log_path)[n0:]
    hit = [a for a in new if a.get("detector") == "serve_spot" and (a.get("detail") or {}).get("where") == layer]
    turns1 = _turns(port, layer)
    return {"layer": layer, "factor": COMPILED_FACTOR, "detected": bool(hit), "rounds": sent if hit else None,
            "rounds_sent": sent, "cut_by_deadline": cut, "alerts_before_trial": before,
            "alert_steps": [a.get("step") for a in hit[:5]],
            "turns": turns1 - turns0 if isinstance(turns0, int) and isinstance(turns1, int) else None,
            "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict") for a in hit} - {None}),
            "relative_errors": [(a.get("detail") or {}).get("relative_error") for a in hit[:3]],
            "spot_alerts_other_layers": sum(1 for a in new if a.get("detector") == "serve_spot" and
                                            (a.get("detail") or {}).get("where") != layer),
            "checksum_alerts": sum(1 for a in new if a.get("detector") == "serve_checksum")}


def run(out: str, port: int, metrics_port: int, gpu: str) -> dict:
    os.makedirs(out, exist_ok=True)
    deadline = time.monotonic() + RUN_S[gpu]
    res: dict = {"model": MODEL, "gpu": gpu}
    ps = vs.prompts(IDENTITY + 16)
    tp = ps[IDENTITY:]
    import shutil
    import tempfile
    for role in (("spot",) if gpu == "l4" else ("nospot", "spot", "eager", "faulted")):
        r: dict = {}
        res[role] = r
        left = deadline - time.monotonic()
        if left < MIN_START.get(role, 5 * 60):
            r["skipped"] = f"the run's time was nearly up ({left / 60:.1f} minutes left)"
            continue
        cache = tempfile.mkdtemp(prefix="sdcguard_faulted_") if role == "faulted" else None
        proc, log = server(role, out, port, metrics_port, cache)
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=max(60.0, min(1500.0, deadline - time.monotonic() - 240)))
            if not r["ready"]:
                continue
            if role != "nospot":
                # the baseline for "every layer checked during the traffic": read after a short request and a read
                # of the results 2 s later, so the start-up runs (step 0, the first layer's turn) are already in it
                vn._try(port, MODEL, ps[0], 1)
                time.sleep(3)
                vn._try(port, MODEL, ps[0], 1)
                r["spot_at_start"] = vn.rpc(port, "sdcguard_test_spot_stats")
            if role == "faulted":
                r["compiled_fault"] = vn.rpc(port, "sdcguard_test_compiled_fault")
                r["served"] = served_stack(metrics_port)
                r["trial"] = compiled_trial(port, log_path, tp, deadline, compiled_layer())
                r["spot_after"] = vn.rpc(port, "sdcguard_test_spot_stats")
                r["alerts_total"] = len(vn.alerts(log_path))
                continue
            r["identity"] = [vn._try(port, MODEL, p).get("text") for p in ps[:IDENTITY if role != "eager" else 16]]
            if role in ("nospot", "spot"):
                r["speed"] = vn._speed(port, MODEL)
            time.sleep(3)
            vn._try(port, MODEL, ps[0], 8)
            r["clean_alerts"] = len(vn.alerts(log_path))
            if role != "nospot":
                r["spot_clean"] = vn.rpc(port, "sdcguard_test_spot_stats")
            if role == "eager":         # (the faulted server's time held back on the H100)
                r["trials"] = run_trials(port, log_path, tp, deadline - (FAULTED_RESERVE if gpu != "l4" else 0))
            r["alerts_total"] = len(vn.alerts(log_path))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
            if cache:
                shutil.rmtree(cache, ignore_errors=True)
            with open(os.path.join(out, "PARTIAL.json"), "w") as f:
                json.dump(res, f, indent=1, default=str)
    return res


def _d(v) -> dict:
    return v if isinstance(v, dict) else {}


def summarize(res: dict) -> dict:
    cost = {}
    for c in ("16", "256"):
        ron = _d(_d(_d(res.get("spot")).get("speed")).get(c))
        roff = _d(_d(_d(res.get("nospot")).get("speed")).get(c))
        on, off = _d(ron.get("median")).get("output_tokens_per_s"), _d(roff.get("median")).get("output_tokens_per_s")
        complete = bool(ron.get("complete")) and bool(roff.get("complete"))
        if on and off:
            cost[c] = {"spot_on": on, "spot_off": off, "spot_cost": round(1 - on / off, 4), "complete": complete}
    tr = [t for t in _d(res.get("eager")).get("trials") or [] if "factor" in t]
    return {"cost": cost,
            "clean_worst": {k: _d(_d(res.get(k)).get("spot_clean")).get("worst") for k in ("spot", "eager")},
            "outputs_judged": {k: _d(_d(_d(res.get(k)).get("spot_clean")).get("stats")).get("outputs_judged")
                               for k in ("spot", "eager")},
            "not_checked_in_traffic": {k: unchecked_in_traffic(_d(res.get(k))) for k in ("spot", "eager")
                                       if _d(res.get(k)).get("ready")},
            "trials": [(t["kind"], t["factor"], t["layer"], t["detected"], t.get("rounds"), t.get("verdicts"),
                        t.get("checksum_alerts"), t.get("turns"), t.get("band_odds")) for t in tr],
            "compiled": {k: _d(_d(res.get("faulted")).get("trial")).get(k)
                         for k in ("layer", "detected", "rounds", "turns", "verdicts", "checksum_alerts")}}


def unchecked_in_traffic(r: dict) -> Optional[list]:
    """The spot-checked layers whose turns read back didn't go up between the read after startup and the read after
    the measured traffic (None if either read is missing)."""
    a, b = _d(r.get("spot_at_start")), _d(r.get("spot_clean"))
    names, t0, t1 = a.get("names"), a.get("turns"), b.get("turns")
    if not isinstance(names, list) or not isinstance(t0, dict) or not isinstance(t1, dict):
        return None
    return [n for n in names if not (isinstance(t1.get(n), int) and t1[n] > (t0.get(n) or 0))]


def _triton_everywhere(st: dict) -> bool:
    kern = _d(_d(st.get("coverage")).get("kernel"))
    return bool(kern) and any(k.startswith("cuda") for k in kern) and \
        all(v == "Triton" for k, v in kern.items() if k.startswith("cuda"))


def compiled_served(served: dict) -> bool:
    """The server's own stack says it runs compiled with CUDA graphs: enforce_eager off and a CUDA graph mode."""
    return str(served.get("eager")).lower() == "false" and served.get("cudagraph") not in (None, "", "NONE", "None")


def _caught(t: dict) -> bool:
    return bool(t.get("detected") and _d(t.get("mark")).get("layer") and "output_wrong" in (t.get("verdicts") or []))


def judge(res: dict) -> tuple:
    fails = []
    roles = ("spot",) if res.get("gpu") == "l4" else ("nospot", "spot", "eager", "faulted")
    for role in roles:
        r = _d(res.get(role))
        if not r.get("ready"):
            fails.append(f"{'S6' if role == 'faulted' else 'S1'}: {role} didn't start" +
                         (f" ({r['skipped']})" if r.get("skipped") else ""))
        if r.get("error"):
            fails.append(f"run: {role} stopped with an error ({r['error']})")
    for role in [x for x in roles if x not in ("nospot", "faulted")]:
        r = _d(res.get(role))
        if not r.get("ready"):
            continue
        st = _d(r.get("spot_at_start"))
        if st.get("layers") != LINEAR_LAYERS:
            fails.append(f"S1 ({role}): {st.get('layers')} layers spot-checked, not {LINEAR_LAYERS}")
        for when, x in (("at startup", st), ("after the traffic", _d(r.get("spot_clean")))):
            if not _triton_everywhere(x):
                fails.append(f"S1 ({role}): the Triton kernel isn't in use {when}: "
                             f"{_d(x.get('coverage')).get('kernel')}")
        miss = unchecked_in_traffic(r)
        if miss is None or miss:
            fails.append(f"S1 ({role}): " + ("the layers' turns weren't read back" if miss is None else
                                             f"{len(miss)} layers not spot-checked during the traffic: {miss[:5]}"))
        if r.get("clean_alerts"):
            fails.append(f"S3 ({role}): {r['clean_alerts']} alerts with no fault")
        sc = _d(r.get("spot_clean"))
        w = sc.get("worst")
        if not vn._finite(w) or float(w) > CLEAN_LIMIT:
            fails.append(f"S3 ({role}): the worst clean spot score was {w}, above {CLEAN_LIMIT}")
        if not _d(sc.get("stats")).get("outputs_judged"):
            fails.append(f"S3 ({role}): no output was spot-checked")
    if res.get("gpu") != "l4":
        x, y = _d(res.get("nospot")).get("identity"), _d(res.get("spot")).get("identity")
        if x is None or y is None or x != y or any(t is None for t in x):
            fails.append(f"S2: the spot server's completions differ from nospot's "
                         f"({sum(p != q for p, q in zip(x or [], y or []))} differ)")
        e = _d(res.get("eager"))
        if e.get("ready"):
            tr = [t for t in e.get("trials") or [] if "factor" in t and t.get("criterion")]
            if len(tr) != len(CRITERIA_TRIALS) + len(ONE_CRITERIA):
                fails.append(f"S4/S5: {len(tr)} of {len(CRITERIA_TRIALS) + len(ONE_CRITERIA)} trials ran")
            bad = [(t["factor"], t["layer"]) for t in tr if t["kind"] == "all" and not _caught(t)]
            if bad:
                fails.append(f"S4: {len(bad)} wrong layers not caught within {ROUNDS} rounds (or not rechecked as "
                             f"wrong): {bad}")
            bad = [(t["factor"], t["layer"]) for t in tr if t["kind"] == "one" and
                   not (_caught(t) and any("row" in c for c in t.get("changed") or []))]
            if bad:
                fails.append(f"S5: {len(bad)} sampled wrong outputs not caught within {ROUNDS} rounds (or never made "
                             f"wrong, or not rechecked as wrong): {bad}")
            other = sum(t.get("spot_alerts_other_layers") or 0 for t in e.get("trials") or [] if "factor" in t)
            if other:
                fails.append(f"S4/S5: {other} spot alerts named a layer with no fault")
        f = _d(res.get("faulted"))
        if f.get("ready"):
            cf, t = _d(f.get("compiled_fault")), _d(f.get("trial"))
            st = _d(f.get("spot_at_start"))
            if not cf.get("applied") or cf.get("layer") != compiled_layer():
                fails.append(f"S6: the compiled fault wasn't applied: {cf}")
            elif st.get("layers") != LINEAR_LAYERS or not _triton_everywhere(st) or \
                    not _triton_everywhere(_d(f.get("spot_after"))):
                fails.append(f"S6: the faulted server spot-checked {st.get('layers')} layers, kernels "
                             f"{_d(st.get('coverage')).get('kernel')} at startup and "
                             f"{_d(_d(f.get('spot_after')).get('coverage')).get('kernel')} after its trial")
            elif not compiled_served(_d(f.get("served"))):
                fails.append(f"S6: the faulted server can't be shown to serve compiled with CUDA graphs: "
                             f"{f.get('served')}")
            elif not (t.get("detected") and "output_wrong" in (t.get("verdicts") or [])):
                fails.append(f"S6: the compiled server's wrong layer wasn't caught within {ROUNDS} rounds of the trial "
                             f"(or not rechecked as wrong): {t.get('verdicts')}, {t.get('rounds_sent')} rounds")
            elif not (isinstance(t.get("turns"), int) and t["turns"] > 0):
                fails.append(f"S6: the faulted layer's checked turns didn't go up during the trial ({t.get('turns')})")
            if t.get("spot_alerts_other_layers"):
                fails.append(f"S6: {t['spot_alerts_other_layers']} spot alerts named a layer with no fault")
    s = summarize(res)
    costs = [s["cost"].get(c) or {} for c in ("16", "256")]
    s["on_by_default"] = (not fails) and res.get("gpu") != "l4" and \
        all(c.get("complete") and c.get("spot_cost") is not None and c["spot_cost"] <= COST_LIMIT for c in costs)
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="h100", choices=("h100", "l4"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    res = run(a.out, a.port, a.metrics_port, a.gpu)
    ok, fails, s = judge(res)
    res.update(passed=ok, failures=fails, summary=s)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": s}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
