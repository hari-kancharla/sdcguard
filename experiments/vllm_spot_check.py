"""Sampled higher-precision verification inside a real `vllm serve` (docs/METHOD.md, "Spot checks in serving",
written before the run).

vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16 (112 linear layers), the checksum check on every layer
(SDCGUARD_SHARE=1, so every linear layer is spot-checked too), attention checks off (their own run), prefix caching
off. On one H100, servers one after another:

  nospot   SDCGUARD_SPOT=0, compiled with CUDA graphs: the reference answers and speed
  spot     SDCGUARD_SPOT=1, compiled with CUDA graphs: its answers, speed, and clean traffic
  eager    SDCGUARD_SPOT=1 with --enforce-eager: the trials

nospot and spot answer 64 prompts greedily (32 tokens) and take the speed load (sdcguard.perf: 128 prompts, 128
tokens, 3 rounds, 16 and 256 requests at once). eager answers 16 prompts, then runs the trials, one after another, each
on a linear layer drawn with seed 0 (sdcguard_test_scale_outputs: its outputs multiplied after its GEMM, before the
checks), rounds of 4 requests at once until a spot alert names the layer or ROUNDS rounds have passed, then undone:
  criteria   every output 2% too large (3 trials) and 5% too large (3 trials)
  reported   every output 1% too large (2 trials); a band of 64 columns 3% too large (3 trials)
A round is about 33 model steps, so the faulty layer gets about 11 turns in ROUNDS rounds. Each turn samples a band of
64 columns about once every N / (8 x 64) turns (the column order spreads them): within 11 turns from any starting
turn for the qkv band drawn (N = 2,048) and from 31% of starting turns for each gate_up band (N = 17,920), computed
exactly from the order (band_odds); each trial records the layer's turns and those odds. With --gpu l4 only the
spot server runs (its clean traffic on another GPU generation).

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
BAND = 64
COST_LIMIT = 0.02
RUN_S = 40 * 60


def trials(seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for crit, group in ((True, CRITERIA_TRIALS), (False, REPORTED_TRIALS)):
        for kind, factor in group:
            out.append({"criterion": crit, "kind": kind, "factor": factor,
                        "layer": f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}", "band_frac": rng.random()})
    return out


def server(role: str, out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
    env["SDCGUARD_ATTN"] = "0"
    env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_SHARE="1",
               SDCGUARD_METRICS_PORT=str(metrics_port), SDCGUARD_LOG=os.path.join(out, f"{role}_alerts.jsonl"),
               SDCGUARD_SPOT="0" if role == "nospot" else "1")
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
        rows.append({**t, "mark": mark, "unmark": unmark, "detected": bool(hit), "rounds": sent if hit else None,
                     "rounds_sent": sent, "cut_by_deadline": cut, "turns": turns, "band_odds": odds,
                     "verdicts": sorted({((a.get("detail") or {}).get("recheck") or {}).get("verdict")
                                         for a in hit} - {None}),
                     "relative_errors": [(a.get("detail") or {}).get("relative_error") for a in hit[:3]],
                     "spot_alerts_other_layers": sum(1 for a in new if a.get("detector") == "serve_spot" and
                                                     (a.get("detail") or {}).get("where") != t["layer"]),
                     "checksum_alerts": sum(1 for a in new if a.get("detector") == "serve_checksum")})
    return rows


def run(out: str, port: int, metrics_port: int, gpu: str) -> dict:
    os.makedirs(out, exist_ok=True)
    deadline = time.monotonic() + RUN_S
    res: dict = {"model": MODEL, "gpu": gpu}
    ps = vs.prompts(IDENTITY + 16)
    tp = ps[IDENTITY:]
    for role in (("spot",) if gpu == "l4" else ("nospot", "spot", "eager")):
        proc, log = server(role, out, port, metrics_port)
        r: dict = {}
        res[role] = r
        log_path = os.path.join(out, f"{role}_alerts.jsonl")
        try:
            r["ready"] = vs.wait_ready(port, proc, timeout=max(60.0, min(1500.0, deadline - time.monotonic() - 300)))
            if not r["ready"]:
                continue
            if role != "nospot":
                r["spot_at_start"] = vn.rpc(port, "sdcguard_test_spot_stats")
            r["identity"] = [vn._try(port, MODEL, p).get("text") for p in ps[:IDENTITY if role != "eager" else 16]]
            if role in ("nospot", "spot"):
                r["speed"] = vn._speed(port, MODEL)
            time.sleep(3)
            vn._try(port, MODEL, ps[0], 8)
            r["clean_alerts"] = len(vn.alerts(log_path))
            if role != "nospot":
                r["spot_clean"] = vn.rpc(port, "sdcguard_test_spot_stats")
            if role == "eager":
                r["trials"] = run_trials(port, log_path, tp, deadline)
            r["alerts_total"] = len(vn.alerts(log_path))
        except Exception as e:  # noqa: BLE001
            r["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r["stop"] = vs.stop(proc)
            log.close()
            with open(os.path.join(out, "PARTIAL.json"), "w") as f:
                json.dump(res, f, indent=1, default=str)
    return res


def _d(v) -> dict:
    return v if isinstance(v, dict) else {}


def summarize(res: dict) -> dict:
    cost = {}
    for c in ("16", "256"):
        on = ((_d(_d(res.get("spot")).get("speed")).get(c) or {}).get("median") or {}).get("output_tokens_per_s")
        off = ((_d(_d(res.get("nospot")).get("speed")).get(c) or {}).get("median") or {}).get("output_tokens_per_s")
        if on and off:
            cost[c] = {"spot_on": on, "spot_off": off, "spot_cost": round(1 - on / off, 4)}
    tr = [t for t in _d(res.get("eager")).get("trials") or [] if "factor" in t]
    return {"cost": cost,
            "clean_worst": {k: _d(_d(res.get(k)).get("spot_clean")).get("worst") for k in ("spot", "eager")},
            "outputs_judged": {k: _d(_d(_d(res.get(k)).get("spot_clean")).get("stats")).get("outputs_judged")
                               for k in ("spot", "eager")},
            "trials": [(t["kind"], t["factor"], t["layer"], t["detected"], t.get("rounds"), t.get("verdicts"),
                        t.get("checksum_alerts"), t.get("turns"), t.get("band_odds")) for t in tr]}


def judge(res: dict) -> tuple:
    fails = []
    roles = ("spot",) if res.get("gpu") == "l4" else ("nospot", "spot", "eager")
    for role in roles:
        r = _d(res.get(role))
        if not r.get("ready"):
            fails.append(f"S1: {role} didn't start")
        if r.get("error"):
            fails.append(f"run: {role} stopped with an error ({r['error']})")
    for role in [x for x in roles if x != "nospot"]:
        r = _d(res.get(role))
        if not r.get("ready"):
            continue
        st = _d(r.get("spot_at_start"))
        kern = _d(_d(st.get("coverage")).get("kernel"))
        if st.get("layers") != LINEAR_LAYERS:
            fails.append(f"S1 ({role}): {st.get('layers')} layers spot-checked, not {LINEAR_LAYERS}")
        if not kern or not all(v == "Triton" for k, v in kern.items() if k.startswith("cuda")) or \
                not any(k.startswith("cuda") for k in kern):
            fails.append(f"S1 ({role}): the Triton kernel isn't in use: {kern}")
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
            if len(tr) != len(CRITERIA_TRIALS):
                fails.append(f"S4: {len(tr)} of {len(CRITERIA_TRIALS)} trials ran")
            bad = [(t["factor"], t["layer"]) for t in tr if not (t["detected"] and _d(t.get("mark")).get("layer") and
                                                                "output_wrong" in (t.get("verdicts") or []))]
            if bad:
                fails.append(f"S4: {len(bad)} wrong layers not caught within {ROUNDS} rounds (or not rechecked as "
                             f"wrong): {bad}")
            other = sum(t.get("spot_alerts_other_layers") or 0 for t in e.get("trials") or [] if "factor" in t)
            if other:
                fails.append(f"S4: {other} spot alerts named a layer with no fault")
    s = summarize(res)
    costs = [(s["cost"].get(c) or {}).get("spot_cost") for c in ("16", "256")]
    s["on_by_default"] = (not fails) and res.get("gpu") != "l4" and all(c is not None and c <= COST_LIMIT
                                                                        for c in costs)
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
