"""Which weight bit flips the vLLM plugin catches while serving, and whether the ones it misses change the model's
answers (docs/METHOD.md, "Bit flips in a served model's weights", written before the run).

One H100, vLLM 0.32.0, Qwen2.5-1.5B-Instruct in bf16, sdcguard checking every linear layer (SDCGUARD_SHARE=1), one
`vllm serve`. First the clean phase: the quality reference and two clean quality checks (the noise), and 64
requests with no fault (the false alarms). Then the trials, one after another: one bit of one weight flipped
(through /collective_rpc), 8 requests of 32 tokens one at a time, a wait of 3 s and one more request (so the plugin
reads its results back), the quality check (`gate model-check --url`: teacher-forced next-token distributions on 32
prompts, against the reference, with the clean run as noise) with the flip still in place, then the bit flipped
back and two clean requests. A trial counts as detected when an alert names its layer between the flip and 5 s
after the flip back. Bits 0 to 15 of a bf16 weight (0 to 6 mantissa, 7 to 14 exponent, 15 sign), 5 trials each,
at random layers (q/k/v, o, gate/up and down projections of all 28 blocks) and positions, seed 0.

    python experiments/vllm_sensitivity.py --out runs/sensitivity
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCKS = 28
KINDS = ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")
BITS = tuple(range(16))
PER_BIT = 5


def _sdc(*args, timeout: int = 600) -> dict:
    p = subprocess.run([sys.executable, "-m", "sdcguard", *args], capture_output=True, text=True, timeout=timeout)
    return {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-1500:]}


def _alerts(path: str) -> list:
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


def _rpc(port: int, method: str, args: list) -> dict:
    status, raw = vs._request(port, "POST", "/collective_rpc", {"method": method, "args": args, "timeout": 60})
    if status != 200:
        return {"error": f"HTTP {status}: {raw[:200]!r}"}
    res = json.loads(raw).get("results") or [{}]
    return res[0] if isinstance(res, list) and res else {}


def trials(seed: int = 0) -> list:
    rng = random.Random(seed)
    out = []
    for bit in BITS:
        for _ in range(PER_BIT):
            out.append({"bit": bit, "layer": f"model.layers.{rng.randrange(BLOCKS)}.{rng.choice(KINDS)}",
                        "row_frac": rng.random(), "col_frac": rng.random()})
    return out


def _quality(path: str) -> dict:
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        return {}
    return {"verdict": (d.get("verdict") or {}).get("verdict"), "mean_kl": (d.get("summary") or {}).get("mean_kl"),
            "top1": (d.get("summary") or {}).get("top1_agreement")}


def run(out: str, port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "alerts.jsonl")
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_SHARE="1", SDCGUARD_LOG=log_path)
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    log = open(os.path.join(out, "server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    res: dict = {"model": MODEL, "share": 1}
    try:
        res["ready"] = vs.wait_ready(port, proc, timeout=1500)
        if not res["ready"]:
            return res
        ref = os.path.join(out, "ref.json")
        res["model_ref"] = _sdc("gate", "model-ref", "--url", url, "--out", ref, "--tokens", "32")
        res["noise"] = _sdc("gate", "model-check", ref, "--url", url, "--out", os.path.join(out, "noise.json"))
        res["noise_quality"] = _quality(os.path.join(out, "noise.json"))
        ps = vs.prompts(96)
        for p in ps[:64]:
            vs.complete(port, MODEL, p, stream=False)
        time.sleep(3)
        vs.complete(port, MODEL, ps[64], stream=False)
        res["clean_alerts"] = len(_alerts(log_path))
        res["clean_check"] = _quality(os.path.join(out, "noise.json"))
        rows = []
        for i, t in enumerate(trials()):
            n0 = len(_alerts(log_path))
            t0 = time.time()
            flip = _rpc(port, "sdcguard_test_flip_bit", [t["layer"], t["row_frac"], t["col_frac"], t["bit"]])
            for p in ps[(i * 8) % 88:(i * 8) % 88 + 8]:
                try:
                    vs.complete(port, MODEL, p, stream=False)
                except Exception:  # noqa: BLE001  (a request a broken weight makes fail is a result too)
                    pass
            time.sleep(3)
            try:
                vs.complete(port, MODEL, ps[90], stream=False)
            except Exception:  # noqa: BLE001
                pass
            qpath = os.path.join(out, f"trial{i:03d}_quality.json")
            _sdc("gate", "model-check", ref, "--url", url, "--out", qpath, "--noise", os.path.join(out, "noise.json"))
            back = _rpc(port, "sdcguard_test_flip_bit", [t["layer"], t["row_frac"], t["col_frac"], t["bit"]])
            t1 = time.time()
            for p in ps[92:94]:
                try:
                    vs.complete(port, MODEL, p, stream=False)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(3)
            new = _alerts(log_path)[n0:]
            hit = [a for a in new if (a.get("detail") or {}).get("where") == t["layer"]
                   and t0 <= (a.get("time") or 0) <= t1 + 5]
            other = [a for a in new if (a.get("detail") or {}).get("where") != t["layer"]]
            rows.append({**t, "flip": flip, "back": back, "detected": bool(hit), "alerts": len(hit),
                         "alerts_other_layers": len(other),
                         "other_layers_nonfinite": sum(bool((a.get("detail") or {}).get("nonfinite")) for a in other),
                         "quality": _quality(qpath)})
            with open(os.path.join(out, "PROGRESS.json"), "w") as f:
                json.dump({"done": len(rows), "rows": rows}, f, indent=1, default=str)
        res["trials"] = rows
    finally:
        res["stop"] = vs.stop(proc)
        log.close()
    return res


def summarize(res: dict) -> dict:
    by_bit = {}
    for r in res.get("trials") or []:
        b = by_bit.setdefault(r["bit"], {"trials": 0, "detected": 0, "changed_answers": 0, "changed_and_missed": 0,
                                         "flip_failed": 0})
        b["trials"] += 1
        if not r.get("flip"):
            b["flip_failed"] += 1
            continue
        changed = (r.get("quality") or {}).get("verdict") in ("WARN", "FAIL")
        b["detected"] += r["detected"]
        b["changed_answers"] += changed
        b["changed_and_missed"] += changed and not r["detected"]
    tr = [r for r in res.get("trials") or [] if r.get("flip")]
    changed = [r for r in tr if (r.get("quality") or {}).get("verdict") in ("WARN", "FAIL")]
    return {"by_bit": by_bit, "trials": len(tr), "detected": sum(r["detected"] for r in tr),
            "changed_answers": len(changed), "changed_and_detected": sum(r["detected"] for r in changed),
            "alerts_on_other_layers": sum(r["alerts_other_layers"] for r in tr),
            "clean_alerts": res.get("clean_alerts"), "noise": res.get("noise_quality")}


def judge(res: dict) -> tuple:
    """(criteria met, failures, summary): docs/METHOD.md, "Bit flips in a served model's weights"."""
    fails = []
    if not res.get("ready"):
        return False, ["1: the server didn't start"], {}
    s = summarize(res)
    if s["clean_alerts"]:
        fails.append(f"2: {s['clean_alerts']} alerts with no fault")
    if (s.get("noise") or {}).get("verdict") != "PASS":
        fails.append(f"3: the clean quality check was {(s.get('noise') or {}).get('verdict')}, not PASS")
    bad_back = [r for r in res.get("trials") or [] if r.get("flip") and
                (r.get("back") or {}).get("after") != (r.get("flip") or {}).get("before")]
    if bad_back:
        fails.append(f"4: {len(bad_back)} flips weren't undone (the next trials would start from a faulty model)")
    return not fails, fails, s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)
    res = run(a.out, a.port)
    ok, fails, s = judge(res)
    res.update(passed=ok, failures=fails, summary=s)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": s}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
