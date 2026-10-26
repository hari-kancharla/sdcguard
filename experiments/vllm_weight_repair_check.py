"""The plugin's weight repair in a real `vllm serve` (docs/METHOD.md, "Weight repair in serving", written before the
run).

One H100, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16, sdcguard checking one linear layer in eight
(SDCGUARD_SHARE=0.125), attention checks off, the weight scan on with repair (SDCGUARD_WEIGHT_SCAN=1,
SDCGUARD_WEIGHT_REPAIR=1: a host-RAM copy of the scanned weights, and a changed layer written back from it at the
next read, then compared again). The server answers 64 prompts greedily (32 tokens), gets clean traffic until the
scan has compared every entry, then 8 trials one after another: one bit of one weight flipped (through
/collective_rpc, at a random layer, row and column, seed 3; bits 14, 13, 10 and 7, two trials each) and NOT flipped
back: rounds of 4 requests until the alert log has the scan's alert naming the layer and a repair record for it (or
DETECT_S seconds have passed), then the weight is read back. After the trials the prefix cache is emptied (its blocks
were computed while a weight was wrong) and the 64 prompts are answered again.

R1: every trial's flip is alerted by the scan (weights_changed, naming its layer) and restored (a weights_restored
    record naming it, one changed value written back), and the weight reads back with exactly its value before the
    flip.
R2: no repair failed (no weights_restore_failed record), and no scan alert names a layer no trial changed.
R3: the prefix cache was emptied after the trials, and the 64 answers after them are identical to the 64 before.
R4: from the flip to its repair record, at most 5 s median and 30 s at most.
R5: at the end, the scan finds no layer changed, and every repaired layer has had a scheduled comparison come back
    clean since its repair (clean traffic is sent for up to RECHECK_S seconds for that).

    python experiments/vllm_weight_repair_check.py --out runs/weight_repair
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
import vllm_weight_scan_check as ws  # noqa: E402

MODEL = ws.MODEL
SHARE = ws.SHARE
BITS = (14, 13, 10, 7)
PER_BIT = 2
DETECT_S = 60.0
BURST = 4
IDENTITY = 64
RUN_S = 18 * 60            # inside the Modal phase's 25 minutes
RECHECK_S = 120.0
MEDIAN_S, MAX_S = 5.0, 30.0


def trials(seed: int = 3) -> list:
    rng = random.Random(seed)
    return [{"bit": bit, "layer": f"model.layers.{rng.randrange(ws.BLOCKS)}.{rng.choice(ws.KINDS)}",
             "row_frac": rng.random(), "col_frac": rng.random()} for bit in BITS for _ in range(PER_BIT)]


def server(out: str, port: int, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    env.update({"SDCGUARD": "1", "SDCGUARD_SHARE": SHARE, "SDCGUARD_ATTN": "0", "SDCGUARD_REQUIRE": "1",
                "SDCGUARD_METRICS_PORT": str(metrics_port), "SDCGUARD_LOG": os.path.join(out, "repair_alerts.jsonl"),
                "SDCGUARD_WEIGHT_SCAN": "1", "SDCGUARD_WEIGHT_REPAIR": "1"})
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    log = open(os.path.join(out, "repair_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def _records(path: str, layer: str, finding: str) -> list:
    return [a for a in vn.alerts(path) if a.get("detector") == "serve_weights" and
            (a.get("detail") or {}).get("where") == layer and (a.get("detail") or {}).get("finding") == finding]


def run_trials(port: int, log_path: str, prompts: list, deadline: float) -> list:
    out = []
    for t in trials():
        if time.monotonic() + DETECT_S + 30 > deadline:
            out.append({**t, "skipped": "the run's time was nearly up"})
            continue
        r = dict(t)
        # (only this trial's records: an earlier trial may have changed and restored the same layer)
        a0 = len(_records(log_path, t["layer"], "weights_changed"))
        n0 = len(_records(log_path, t["layer"], "weights_restored"))
        r["value_before"] = vn.rpc(port, "sdcguard_test_weight_value", [t["layer"], t["row_frac"], t["col_frac"]])
        t0 = time.monotonic()
        r["flip"] = vn.rpc(port, "sdcguard_test_flip_bit", [t["layer"], t["row_frac"], t["col_frac"], t["bit"]])
        k = 0
        while time.monotonic() - t0 < DETECT_S and len(_records(log_path, t["layer"], "weights_restored")) <= n0:
            with ThreadPoolExecutor(BURST) as ex:
                list(ex.map(lambda j: vn._try(port, MODEL, prompts[(k * BURST + j) % len(prompts)], 16),
                            range(BURST)))
            k += 1
        restored = len(_records(log_path, t["layer"], "weights_restored")) > n0
        r["seconds"] = round(time.monotonic() - t0, 2) if restored else None
        r["alerted"] = len(_records(log_path, t["layer"], "weights_changed")) > a0
        r["restored"] = restored
        r["value_after"] = vn.rpc(port, "sdcguard_test_weight_value", [t["layer"], t["row_frac"], t["col_frac"]])
        out.append(r)
    return out


def recheck_repaired(port: int, prompts: list, deadline: float) -> dict:
    """Clean traffic until every repaired layer has had a scheduled comparison come back since its repair (none in
    the coverage's repair.not_compared_since_repair), for at most RECHECK_S seconds and never within 60 s of the
    deadline. {"rounds", "seconds", "left"}."""
    t0, rounds, left = time.monotonic(), 0, None
    while True:
        st = vn.rpc(port, "sdcguard_test_scan_stats")
        rep = (((st or {}).get("coverage") or {}).get("repair") or {}) if isinstance(st, dict) else {}
        pending = rep.get("not_compared_since_repair")
        left = len(pending) if isinstance(pending, list) else None
        if left == 0 or time.monotonic() - t0 > RECHECK_S or time.monotonic() + 60 > deadline:
            break
        with ThreadPoolExecutor(BURST) as ex:
            list(ex.map(lambda j: vn._try(port, MODEL, prompts[(rounds * BURST + j) % len(prompts)], 16),
                        range(BURST)))
        rounds += 1
    return {"rounds": rounds, "seconds": round(time.monotonic() - t0, 1), "left": left}


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    deadline = time.monotonic() + RUN_S
    res: dict = {"model": MODEL}
    ps = vs.prompts(IDENTITY + 16)
    tp = ps[IDENTITY:]
    proc, log = server(out, port, metrics_port)
    log_path = os.path.join(out, "repair_alerts.jsonl")
    try:
        res["ready"] = vs.wait_ready(port, proc, timeout=max(60.0, min(1500.0, deadline - time.monotonic() - 300)))
        if res["ready"]:
            res["identity_before"] = [vn._try(port, MODEL, p).get("text") for p in ps[:IDENTITY]]
            res["warm"] = ws.warm_scan(port, MODEL, ps, deadline)
            res["scan_at_clean"] = vn.rpc(port, "sdcguard_test_scan_stats")
            res["clean_scan_alerts"] = len([a for a in vn.alerts(log_path) if a.get("detector") == "serve_weights"])
            res["trials"] = run_trials(port, log_path, tp, deadline)
            res["prefix_cache_emptied"] = ws._empty_prefix_cache(port)
            after = []
            for p in ps[:IDENTITY]:
                if time.monotonic() + 60 > deadline:          # (judged as not answered: R3)
                    res["identity_after_cut"] = f"the run's time ran out after {len(after)} answers"
                    break
                after.append(vn._try(port, MODEL, p).get("text"))
            res["identity_after"] = after
            res["recheck"] = recheck_repaired(port, tp, deadline)
            res["scan_at_end"] = vn.rpc(port, "sdcguard_test_scan_stats")
            res["records"] = [a for a in vn.alerts(log_path) if a.get("detector") == "serve_weights"]
    except Exception as e:  # noqa: BLE001  (a server that died: judged on what was measured)
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    finally:
        res["stop"] = vs.stop(proc)
        log.close()
    return res


def judge(res: dict) -> tuple:
    fails = []
    if not res.get("ready"):
        return False, ["run: the server didn't start"], {}
    if res.get("error"):
        fails.append(f"run: stopped with an error ({res['error']})")
    tr = res.get("trials") or []
    run_ = [t for t in tr if "skipped" not in t]
    if len(run_) != len(trials()):
        fails.append(f"R1: {len(run_)} of {len(trials())} trials ran")
    recs = res.get("records") or []
    for t in run_:
        same = (t.get("value_after") or {}).get("bits") == (t.get("value_before") or {}).get("bits") and \
            (t.get("value_before") or {}).get("bits") is not None
        if not (t.get("alerted") and t.get("restored") and same and (t.get("flip") or {}).get("layer")):
            fails.append(f"R1: {t['layer']} bit {t['bit']}: alerted {t.get('alerted')}, restored {t.get('restored')}, "
                         f"value {t.get('value_before')} -> {t.get('value_after')}")
    wrote = [(r.get("detail") or {}).get("values_changed") for r in recs
             if (r.get("detail") or {}).get("finding") == "weights_restored"]
    if any(v != 1 for v in wrote):
        fails.append(f"R1: a repair wrote back other than one value: {wrote}")
    failed = [r for r in recs if (r.get("detail") or {}).get("finding") == "weights_restore_failed"]
    if failed:
        fails.append(f"R2: {len(failed)} repairs failed: {[(r.get('detail') or {}).get('where') for r in failed]}")
    changed = {t["layer"] for t in run_}
    other = [r for r in recs if (r.get("detail") or {}).get("finding") == "weights_changed" and
             (r.get("detail") or {}).get("where") not in changed]
    if other or res.get("clean_scan_alerts"):
        fails.append(f"R2: scan alerts on layers no trial changed: {len(other)} (clean: {res.get('clean_scan_alerts')})")
    before, after = res.get("identity_before") or [], res.get("identity_after") or []
    diff = sum(1 for a, b in zip(before, after) if a != b) + abs(len(before) - len(after))
    if len(before) != IDENTITY or diff or None in before:
        fails.append(f"R3: {diff} of {IDENTITY} answers differ after the trials (or weren't answered)")
    if not res.get("prefix_cache_emptied"):
        fails.append("R3: the prefix cache wasn't emptied after the trials (its blocks were computed while a weight "
                     "was wrong), so the answers after them don't test the repaired weights alone")
    secs = sorted(t["seconds"] for t in run_ if isinstance(t.get("seconds"), (int, float)))
    med = secs[len(secs) // 2] if secs else None
    if not secs or med > MEDIAN_S or secs[-1] > MAX_S:
        fails.append(f"R4: flip to repair took a median {med} s, at most {secs[-1] if secs else None} s")
    cov = (res.get("scan_at_end") or {}).get("coverage") or {}
    rep = cov.get("repair") or {}
    if cov.get("changed") != [] or rep.get("not_compared_since_repair") != []:
        fails.append(f"R5: at the end the scan finds changed {cov.get('changed')}, and repaired layers not compared "
                     f"since their repair {rep.get('not_compared_since_repair')}")
    s = {"trials": len(run_), "restored": sum(1 for t in run_ if t.get("restored")), "median_seconds": med,
         "max_seconds": secs[-1] if secs else None, "answers_identical_after": not diff and len(before) == IDENTITY,
         "host_copy_bytes": rep.get("host_copy_bytes"), "repaired": rep.get("repaired"), "failed": rep.get("failed")}
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
