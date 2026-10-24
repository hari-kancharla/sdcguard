"""`sdcguard ready` and request attribution on a real `vllm serve` (docs/METHOD.md, "Verifiable protection status
and the requests an alert concerns", written before the run).

Servers, one after another, on one GPU (Qwen2.5-1.5B-Instruct, vLLM's default compiled mode with CUDA graphs):
  guarded     SDCGUARD=1, SDCGUARD_REQUIRE=1, metrics on SDCGUARD_METRICS_PORT, alerts to a local webhook
  excluded    SDCGUARD=1 but VLLM_PLUGINS names no plugin, so vLLM never loads sdcguard (what a plugin that fails to
              import looks like from outside: vLLM skips it and serves)
  disabled    SDCGUARD=0
Each `sdcguard ready` verdict below is asked 3 times (a fresh process each, --window 5):
  R1 guarded, while requests are being served, with --vllm                     expected PROTECTED
  R2 guarded, idle, without --vllm                                              expected INCONCLUSIVE
  R3 guarded, idle, with --vllm                                                 expected PROTECTED
  R4 guarded, its result reads stopped (sdcguard_test_stop_reads), serving, --vllm   expected NOT PROTECTED
  R5 excluded, serving, with --vllm                                             expected NOT PROTECTED
  R6 disabled, serving, with --vllm                                             expected NOT PROTECTED
  R7 any server stopped (SIGINT), with --vllm                                   expected NOT PROTECTED
On the guarded server, before R4, request attribution:
  A1 5 times, each in its own layer (model.layers.10 to 14 down_proj): one request alone (512 tokens), a weight bit
     flipped (sdcguard_test_flip) 0.6 s into it and flipped back 0.4 s later (during its decoding), then clean
     requests until the alert is in SDCGUARD_LOG: at least one alert for that layer names exactly that request
     (the client's id, as the response gave it) with "exact" attribution, none names another request (an alert on
     a CUDA-graph padding row names none and says so), and the requests since the layer's last clean read
     include it
  A2 once (model.layers.15 down_proj): the bit flipped, 32 requests sent 8 at a time, the bit flipped back:
     every request an alert names was one of those 32, and the requests since the last clean read include all 32
  A3 the 64 one-at-a-time completions before any flip are identical to the 'disabled' server's (the plugin's step
     bookkeeping doesn't change what vLLM computes)

    python experiments/vllm_ready_attrib_check.py --out runs/ready_attrib
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_serve_check as vs  # noqa: E402

MODEL = vs.MODEL
FLIP = vs.FLIP
REPEATS = 3


def _complete(port: int, prompt: str, tokens: int = 32) -> dict:
    """A greedy completion: its id (the client's view of the request) and text."""
    body = {"model": MODEL, "prompt": prompt, "max_tokens": tokens, "temperature": 0.0, "ignore_eos": True}
    status, raw = vs._request(port, "POST", "/v1/completions", body)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
    r = json.loads(raw)
    return {"id": r["id"], "text": r["choices"][0]["text"], "t": time.time()}


def _ready(metrics_port: int, vllm_port, window: float = 5.0) -> dict:
    cmd = [sys.executable, "-m", "sdcguard", "ready", "--port", str(metrics_port), "--workers", "1",
           "--window", str(window), "--json"]
    if vllm_port is not None:
        cmd += ["--vllm", f"http://127.0.0.1:{vllm_port}/metrics"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        r = {"verdict": None, "raw": (p.stdout + p.stderr)[-500:]}
    r["exit_code"] = p.returncode
    return r


def _background_load(port: int, stop: threading.Event) -> threading.Thread:
    ps = vs.prompts(64)

    def run():
        i = 0
        while not stop.is_set():
            try:
                _complete(port, ps[i % len(ps)], tokens=64)
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
            i += 1
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _verdicts(metrics_port: int, vllm_port, serving_port=None) -> list:
    out = []
    for _ in range(REPEATS):
        stop = threading.Event()
        threads = [_background_load(serving_port, stop) for _ in range(4)] if serving_port else []
        if threads:
            time.sleep(3)
        out.append(_ready(metrics_port, vllm_port))
        stop.set()
        for t in threads:
            t.join(timeout=120)
    return out


def _rpc(port: int, method: str, args=()) -> dict:
    status, raw = vs._request(port, "POST", "/collective_rpc", {"method": method, "args": list(args), "timeout": 60})
    return {"status": status, "results": json.loads(raw).get("results") if status == 200 else
            raw.decode("utf-8", "replace")[:300]}


def _log_alerts(path: str) -> list:
    """Every alert the guarded server wrote to SDCGUARD_LOG (one JSON event per line, attribution included: the
    webhook posts at most one alert per layer and GPU every 10 minutes, the log keeps them all)."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def _wait_alert(path: str, n0: int, layer: str, port: int, seconds: float = 90) -> list:
    t0 = time.monotonic()
    ps = vs.prompts(8)
    while time.monotonic() - t0 < seconds and not any(layer in json.dumps(b) for b in _log_alerts(path)[n0:]):
        _complete(port, ps[0], tokens=8)
        time.sleep(1)
    time.sleep(3)
    return [b for b in _log_alerts(path)[n0:] if (b.get("detail") or {}).get("where") == layer]


def _alert_detail(body: dict) -> dict:
    return body.get("detail") or (body.get("event") or {}).get("detail") or body


def _named(body: dict) -> tuple:
    """(directly implicated requests, how they were found, requests since the layer's last clean read)."""
    d = _alert_detail(body)
    since = ((d.get("potentially_exposed") or {}).get("since_last_clean_read") or {}).get("requests") or []
    return d.get("directly_implicated"), d.get("attribution") or "", since


def _matches(named: str, client_id: str) -> bool:
    """vLLM's engine id for a one-prompt completion is the response id plus "-0" (serving_completion)."""
    return named == client_id or named.startswith(client_id + "-")


def attribution(port: int, log_path: str) -> dict:
    """A1 and A2 (see the docstring). Each trial flips a bit in its own layer (model.layers.10 to 15 down_proj),
    so every layer had a clean read before its flip."""
    res: dict = {"single": [], "concurrent": None}
    ps = vs.prompts(64)
    for i in range(5):
        layer = f"model.layers.{10 + i}.mlp.down_proj"
        for p in ps[:4]:                                   # clean steps, then a read (2 s), so the layer has a clean read
            _complete(port, p, tokens=8)
        time.sleep(3)
        _complete(port, ps[5], tokens=8)
        n0 = len(_log_alerts(log_path))
        # the flip lands in the request's decoding, where one request runs in the batch-of-1 CUDA graph (no
        # padding rows): every row the check sees is that request's
        box: dict = {}
        t = threading.Thread(target=lambda: box.update(r=_complete(port, ps[10 + i], tokens=512)))
        t.start()
        time.sleep(0.6)
        flip = _rpc(port, "sdcguard_test_flip", [layer])
        time.sleep(0.4)
        back = _rpc(port, "sdcguard_test_flip", [layer])
        t.join()
        r = box["r"]
        alerts = _wait_alert(log_path, n0, layer, port)
        named = [_named(b) for b in alerts]
        res["single"].append({"layer": layer, "request": r["id"], "flip": flip, "back": back, "alerts": len(alerts),
                              "named": [n[0] for n in named], "attribution": [n[1] for n in named],
                              "since_includes": any(any(_matches(x, r["id"]) for x in n[2]) for n in named)})
    layer = "model.layers.15.mlp.down_proj"
    for p in ps[:4]:
        _complete(port, p, tokens=8)
    time.sleep(3)
    _complete(port, ps[5], tokens=8)
    n0 = len(_log_alerts(log_path))
    flip = _rpc(port, "sdcguard_test_flip", [layer])
    t_on = time.time()
    done, lock = [], threading.Lock()

    def worker(k):
        for j in range(4):
            r = _complete(port, ps[(8 * j + k) % 64], tokens=32)
            with lock:
                done.append(r)
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    back = _rpc(port, "sdcguard_test_flip", [layer])
    t_off = time.time()
    alerts = _wait_alert(log_path, n0, layer, port)
    named = [_named(b) for b in alerts]
    res["concurrent"] = {"layer": layer, "flip": flip, "back": back, "requests": [r["id"] for r in done],
                         "flipped_seconds": round(t_off - t_on, 1), "alerts": len(alerts),
                         "named": [n[0] for n in named], "attribution": [n[1] for n in named],
                         "since": sorted({x for n in named for x in n[2]})}
    return res


def _start(role: str, out: str, port: int, sink, metrics_port: int):
    """vllm_serve_check.start's server, with the environment of an unprotected role: 'excluded' asks for
    protection (SDCGUARD=1, SDCGUARD_REQUIRE=1, a metrics port) but VLLM_PLUGINS names no installed plugin, so vLLM
    never loads sdcguard; 'disabled' has SDCGUARD=0 (and no REQUIRE, which would refuse to start)."""
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["PYTHONPATH"] = vs.HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for k in [k for k in env if k.startswith("SDCGUARD")]:
        del env[k]
    env["SDCGUARD_METRICS_PORT"] = str(metrics_port)
    if role == "excluded":
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", VLLM_PLUGINS="none_installed")
    else:
        env["SDCGUARD"] = "0"
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", MODEL, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def run(out: str, port: int, metrics_port: int) -> dict:
    os.makedirs(out, exist_ok=True)
    res: dict = {"model": MODEL}
    # guarded
    sink = vs.Sink()
    log_path = os.path.join(out, "alerts.jsonl")
    os.environ["SDCGUARD_LOG"] = log_path              # vs.start copies the environment into the guarded server's
    try:
        proc, log = vs.start("guarded", out, MODEL, port, 1, sink, metrics_port)
    finally:
        os.environ.pop("SDCGUARD_LOG", None)
    try:
        res["guarded_ready"] = vs.wait_ready(port, proc, timeout=1500)
        if res["guarded_ready"]:
            ps = vs.prompts(256)
            res["identity_guarded"] = [vs.complete(port, MODEL, p, stream=False)["text"]
                                       for p in ps[:vs.IDENTITY_PROMPTS]]
            res["R1"] = _verdicts(metrics_port, port, serving_port=port)
            time.sleep(8)
            res["R2"] = _verdicts(metrics_port, None)
            res["R3"] = _verdicts(metrics_port, port)
            res["attribution"] = attribution(port, log_path)
            res["stop_reads"] = _rpc(port, "sdcguard_test_stop_reads")
            res["R4"] = _verdicts(metrics_port, port, serving_port=port)
            res["alerts_total"] = len(sink.bodies)
    finally:
        res["guarded_stop"] = vs.stop(proc)
        log.close()
        sink.server.shutdown()
    res["R7_after_guarded"] = [_ready(metrics_port, port) for _ in range(REPEATS)]
    # excluded and disabled
    for role in ("excluded", "disabled"):
        sink = vs.Sink()
        proc, log = _start(role, out, port, sink, metrics_port)
        try:
            res[f"{role}_ready"] = vs.wait_ready(port, proc, timeout=1500)
            if res[f"{role}_ready"]:
                if role == "disabled":
                    ps = vs.prompts(256)
                    res["identity_disabled"] = [vs.complete(port, MODEL, p, stream=False)["text"]
                                                for p in ps[:vs.IDENTITY_PROMPTS]]
                res["R5" if role == "excluded" else "R6"] = _verdicts(metrics_port, port, serving_port=port)
        finally:
            res[f"{role}_stop"] = vs.stop(proc)
            log.close()
            sink.server.shutdown()
    return res


EXPECT = {"R1": "PROTECTED", "R2": "INCONCLUSIVE", "R3": "PROTECTED", "R4": "NOT PROTECTED", "R5": "NOT PROTECTED",
          "R6": "NOT PROTECTED", "R7_after_guarded": "NOT PROTECTED"}


def judge(res: dict) -> tuple:
    fails, summary = [], {}
    for k, want in EXPECT.items():
        got = [v.get("verdict") for v in res.get(k) or []]
        summary[k] = got
        if len(got) != REPEATS or any(g != want for g in got):
            fails.append(f"{k}: expected {want} {REPEATS} times, got {got}")
    protected_wrongly = sum(g == "PROTECTED" for k in ("R2", "R4", "R5", "R6", "R7_after_guarded")
                            for g in summary.get(k, []))
    summary["protected_when_not"] = protected_wrongly
    a = res.get("attribution") or {}
    singles = a.get("single") or []
    ok_single = 0
    for s in singles:
        # every alert for the layer names that request exactly (an alert on a CUDA-graph padding row names none
        # and says so; it can't name a wrong one), at least one does, and "since the last clean read" includes it
        named_ok = all((n is not None and len(n) == 1 and _matches(n[0], s["request"]) and t.startswith("exact"))
                       or (n == [] and "padding" in t) for n, t in zip(s["named"], s["attribution"]))
        some = any(n and _matches(n[0], s["request"]) for n in s["named"])
        ok_single += bool(s["alerts"] > 0 and named_ok and some and s["since_includes"])
    summary["single_exact"] = f"{ok_single}/{len(singles)}"
    if len(singles) != 5 or ok_single != 5:
        fails.append(f"A1: the alert named exactly the one request in {ok_single} of {len(singles)} trials")
    c = a.get("concurrent") or {}
    sent = set(c.get("requests") or [])
    named = [x for n in (c.get("named") or []) if n for x in n]
    wrong = [x for x in named if not any(_matches(x, r) for r in sent)]
    missing = [r for r in sent if not any(_matches(x, r) for x in c.get("since") or [])]
    summary["concurrent"] = {"alerts": c.get("alerts"), "named": len(named), "named_wrong": len(wrong),
                             "since_missing": len(missing)}
    if not c.get("alerts") or wrong or missing:
        fails.append(f"A2: {c.get('alerts')} alerts, {len(wrong)} named requests not sent during the flip, "
                     f"{len(missing)} flipped requests missing from 'since the last clean read'")
    same = res.get("identity_guarded") is not None and res.get("identity_guarded") == res.get("identity_disabled")
    summary["identity"] = same
    if not same:
        fails.append("A3: the guarded server's 64 completions differ from the disabled server's")
    return not fails, fails, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    res = run(a.out, a.port, a.metrics_port)
    ok, fails, summary = judge(res)
    res.update(passed=ok, failures=fails, summary=summary)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": summary}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
