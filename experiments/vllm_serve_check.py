"""sdcguard in a real `vllm serve`: the HTTP server, with its engine and GPU workers in processes of
their own, as people deploy it. experiments/vllm_check.py runs the offline API with the engine in one process;
this runs the server twice, plain and guarded, and talks to it over HTTP only.

Per role (plain, then guarded, same flags, same port layout):
  1. start `vllm serve MODEL` (guarded: SDCGUARD=1 and SDCGUARD_REQUIRE=1, alerts posted as JSON to a local
     webhook sink, Prometheus metrics on SDCGUARD_METRICS_PORT + the worker's GPU index), wait for /health
  2. identity: 64 prompts one at a time (greedy, 128 tokens, ignore_eos), so batching can't differ between roles
  3. load: streamed requests at 1, 16 and 256 at a time; time to first token and request latency, p50/p90/p99,
     and output tokens per second
  4. guarded only: the metrics of every worker; then one weight bit flipped in every worker through vLLM's
     /collective_rpc (sdcguard_test_ext.FaultInjection) and 16 more requests; the alert must reach the webhook
     naming the layer, and the metrics must count it
  5. SIGINT, and the server must exit

Criteria (docs/METHOD.md, written before the run), for the guarded server:
  1. it starts and serves with SDCGUARD_REQUIRE=1 (every registered layer checked, so it doesn't refuse)
  2. the 64 one-at-a-time completions are identical to the plain server's
  3. every worker's metrics: layers checked > 0 and the same on every worker (EXPECT_LAYERS when known), none whose
     check never ran, results read back, and no alert before the flip
  4. no alert reaches the webhook during the clean load
  5. after the flip, an alert naming the flipped layer reaches the webhook within 90 seconds, and the metrics
     count at least one alert
  6. both servers exit within 120 seconds of SIGINT
Latency and throughput are reported against the plain server; they decide nothing.

    python experiments/vllm_serve_check.py --out runs/serve --model Qwen/Qwen2.5-1.5B-Instruct [--tp 2]
"""
from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EXPECT_LAYERS = {"Qwen/Qwen2.5-1.5B-Instruct": 112, "Qwen/Qwen2.5-7B-Instruct": 112}
FLIP = "model.layers.10.mlp.down_proj"
IDENTITY_PROMPTS = 64
LOAD = ((1, 16), (16, 128), (256, 256))          # (requests at a time, requests)
TOKENS = 128
HERE = os.path.dirname(os.path.abspath(__file__))


def prompts(n: int) -> list:
    """The prompts experiments/vllm_check.py sends (the same words, the same lengths)."""
    words = ("the quick brown fox jumps over a lazy dog while engineers check every matrix multiply "
             "on thousands of graphics processors to make sure the numbers are right").split()
    out = []
    for i in range(n):
        k = 20 + (i * 37) % 400
        out.append("Summarize this in one sentence: " + " ".join(words[(i + j) % len(words)] for j in range(k)))
    return out


def percentile(xs: list, q: float):
    """Nearest-rank percentile (q in 0..100); None for no values."""
    if not xs:
        return None
    s = sorted(xs)
    return s[max(0, min(len(s) - 1, int(-(-q * len(s) // 100)) - 1))]


def parse_metrics(text: str) -> dict:
    """Prometheus text format -> {metric name: sum over its label sets} (comments and _created series dropped)."""
    out: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_labels, _, value = line.rpartition(" ")
        name = name_labels.split("{", 1)[0]
        if name.endswith("_created"):
            continue
        try:
            out[name] = out.get(name, 0.0) + float(value)
        except ValueError:
            continue
    return out


class Sink:
    """A local HTTP endpoint that keeps every JSON body POSTed to it: sdcguard's webhook target."""

    def __init__(self):
        bodies = self.bodies = []

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n)
                try:
                    bodies.append(json.loads(raw))
                except ValueError:
                    bodies.append({"raw": raw.decode("utf-8", "replace")})
                self.send_response(200)
                self.end_headers()

            def log_message(self, *a):
                pass
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


def _request(port: int, method: str, path: str, body=None, timeout: float = 600.0):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request(method, path, body=None if body is None else json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def complete(port: int, model: str, prompt: str, stream: bool) -> dict:
    """One greedy completion of TOKENS tokens. Streamed: time to first token and the whole request's latency."""
    body = {"model": model, "prompt": prompt, "max_tokens": TOKENS, "temperature": 0.0, "ignore_eos": True,
            "stream": stream}
    t0 = time.perf_counter()
    if not stream:
        status, raw = _request(port, "POST", "/v1/completions", body)
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {raw[:300]!r}")
        return {"text": json.loads(raw)["choices"][0]["text"], "latency": time.perf_counter() - t0}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        c.request("POST", "/v1/completions", body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
        ttft, parts = None, []
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            text = "".join(ch.get("text") or "" for ch in chunk.get("choices") or [])
            if text and ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(text)
        return {"text": "".join(parts), "ttft": ttft, "latency": time.perf_counter() - t0}
    finally:
        c.close()


def load(port: int, model: str, texts: list, at_once: int) -> dict:
    """Send every prompt, at_once at a time (each worker sends its next one as soon as the last finished)."""
    q: queue.Queue = queue.Queue()
    for p in texts:
        q.put(p)
    results, errors = [], []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                p = q.get_nowait()
            except queue.Empty:
                return
            try:
                r = complete(port, model, p, stream=True)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(e).__name__}: {e}"[:300])
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(at_once)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in results if r.get("ttft") is not None]
    lat = [r["latency"] for r in results]
    return {"at_once": at_once, "requests": len(texts), "completed": len(results), "errors": errors[:5],
            "seconds": round(wall, 3), "output_tokens_per_s": round(len(results) * TOKENS / wall, 1),
            **{f"ttft_p{q}_ms": None if percentile(ttft, q) is None else round(1000 * percentile(ttft, q), 1)
               for q in (50, 90, 99)},
            **{f"latency_p{q}_ms": None if percentile(lat, q) is None else round(1000 * percentile(lat, q), 1)
               for q in (50, 90, 99)}}


def scrape(base: int, workers: int) -> list:
    """Every worker's sdcguard metrics (one exporter per GPU: base + its index)."""
    out = []
    for i in range(workers):
        try:
            status, raw = _request(base + i, "GET", "/metrics", timeout=10)
            out.append(parse_metrics(raw.decode()) if status == 200 else {"error": f"HTTP {status}"})
        except Exception as e:  # noqa: BLE001
            out.append({"error": f"{type(e).__name__}: {e}"})
    return out


def start(role: str, out: str, model: str, port: int, tp: int, sink: Sink, metrics_port: int):
    env = dict(os.environ)
    env["VLLM_SERVER_DEV_MODE"] = "1"                       # /collective_rpc, for the flip
    env["PYTHONPATH"] = HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if role == "guarded":
        # attention checks have their own run (vllm_attn_check.py); this run's criteria were written without them
        env["SDCGUARD_ATTN"] = "0"
        env.update(SDCGUARD="1", SDCGUARD_REQUIRE="1", SDCGUARD_WEBHOOK=f"http://127.0.0.1:{sink.port}/hook",
                   SDCGUARD_WEBHOOK_FORMAT="json", SDCGUARD_METRICS_PORT=str(metrics_port))
    else:
        for k in [k for k in env if k.startswith("SDCGUARD")]:
            del env[k]
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model, "--port", str(port),
           "--dtype", "bfloat16", "--max-model-len", "2048", "--gpu-memory-utilization", "0.6", "--seed", "0",
           "--tensor-parallel-size", str(tp), "--worker-extension-cls", "sdcguard_test_ext.FaultInjection"]
    log = open(os.path.join(out, f"{role}_server.log"), "w")
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log


def wait_ready(port: int, proc, timeout: float) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if _request(port, "GET", "/health", timeout=5)[0] == 200:
                return True
        except OSError:
            pass
        time.sleep(3)
    return False


def stop(proc, timeout: float = 120.0) -> dict:
    t0 = time.monotonic()
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return {"exited_in_s": None, "killed": True, "code": proc.returncode}
    return {"exited_in_s": round(time.monotonic() - t0, 1), "killed": False, "code": proc.returncode}


def run_role(role: str, out: str, model: str, tp: int, port: int, metrics_port: int) -> dict:
    sink = Sink()
    proc, log = start(role, out, model, port, tp, sink, metrics_port)
    res: dict = {"role": role, "model": model, "tp": tp}
    try:
        t0 = time.monotonic()
        res["ready"] = wait_ready(port, proc, timeout=1500)
        res["ready_s"] = round(time.monotonic() - t0, 1)
        if not res["ready"]:
            res["exit_code"] = proc.poll()
            return res
        ps = prompts(256)
        res["identity"] = [complete(port, model, p, stream=False)["text"] for p in ps[:IDENTITY_PROMPTS]]
        res["load"] = [load(port, model, ps[:n], at_once) for at_once, n in LOAD]
        if role == "guarded":
            time.sleep(5)                                   # a poll after the last step: results read back
            res["metrics_clean"] = scrape(metrics_port, tp)
            res["alerts_clean"] = list(sink.bodies)
            status, raw = _request(port, "POST", "/collective_rpc",
                                   {"method": "sdcguard_test_flip", "args": [FLIP], "timeout": 60})
            res["flip"] = {"status": status, "results": json.loads(raw).get("results") if status == 200 else
                           raw.decode("utf-8", "replace")[:300]}
            n0 = len(res["alerts_clean"])
            for p in ps[:16]:
                complete(port, model, p, stream=False)
            t1 = time.monotonic()
            while time.monotonic() - t1 < 90 and not any(FLIP in json.dumps(b) for b in sink.bodies[n0:]):
                complete(port, model, ps[0], stream=False)  # steps, so the plugin reads results and posts
                time.sleep(1)
            res["alerts_after_flip"] = sink.bodies[n0:]
            res["alert_seconds"] = round(time.monotonic() - t1, 1)
            res["metrics_after_flip"] = scrape(metrics_port, tp)
    finally:
        res["stop"] = stop(proc)
        log.close()
        sink.server.shutdown()
    return res


def judge(plain: dict, guarded: dict) -> tuple:
    """(criteria met, failures, summary)."""
    fails = []
    model = guarded.get("model")
    if not guarded.get("ready"):
        fails.append(f"1: the guarded server didn't start (exit code {guarded.get('exit_code')})")
    if not plain.get("ready"):
        fails.append(f"the plain server didn't start (exit code {plain.get('exit_code')})")
    same = None
    if plain.get("identity") and guarded.get("identity"):
        same = sum(a == b for a, b in zip(plain["identity"], guarded["identity"]))
        if same != IDENTITY_PROMPTS or len(guarded["identity"]) != IDENTITY_PROMPTS:
            fails.append(f"2: {same} of {IDENTITY_PROMPTS} one-at-a-time completions the same")
    elif guarded.get("ready") and plain.get("ready"):
        fails.append("2: no completions to compare")
    m = guarded.get("metrics_clean") or []
    if guarded.get("ready"):
        layers = [w.get("sdcguard_layers_checked") for w in m]
        want = EXPECT_LAYERS.get(model) if guarded.get("tp", 1) == 1 else None
        if not m or any("error" in w for w in m):
            fails.append(f"3: metrics not read from every worker: {m}")
        elif not all(v and v > 0 for v in layers) or len(set(layers)) != 1 or (want and layers[0] != want):
            fails.append(f"3: layers checked per worker {layers}" + (f", expected {want}" if want else ""))
        elif any(w.get("sdcguard_layers_never_checked", 0) for w in m) or not all(
                w.get("sdcguard_reads_total", w.get("sdcguard_reads", 0)) > 0 for w in m):
            fails.append(f"3: never-checked layers or no reads: {m}")
        elif any(w.get("sdcguard_all_alerts_total", w.get("sdcguard_all_alerts", 0)) for w in m):
            fails.append("3: alerts counted before the flip")
        if guarded.get("alerts_clean"):
            fails.append(f"4: {len(guarded['alerts_clean'])} alerts during clean traffic")
        hits = [b for b in guarded.get("alerts_after_flip") or [] if FLIP in json.dumps(b)]
        after = guarded.get("metrics_after_flip") or []
        counted = sum(w.get("sdcguard_all_alerts_total", w.get("sdcguard_all_alerts", 0)) for w in after
                      if "error" not in w)
        if not hits:
            fails.append(f"5: no alert naming {FLIP} reached the webhook within 90 s (flip: {guarded.get('flip')})")
        elif counted < 1:
            fails.append("5: the alert reached the webhook but the metrics didn't count it")
    for r in (plain, guarded):
        s = r.get("stop") or {}
        if s.get("killed") or s.get("exited_in_s") is None:
            fails.append(f"6: the {r.get('role')} server didn't exit within 120 s of SIGINT")
    summary = {"model": model, "tp": guarded.get("tp"), "same_one_at_a_time": f"{same}/{IDENTITY_PROMPTS}",
               "layers_checked_per_worker": [w.get("sdcguard_layers_checked") for w in m],
               "alerts_clean": len(guarded.get("alerts_clean") or []),
               "alert_after_flip_s": guarded.get("alert_seconds"), "flip": guarded.get("flip"),
               "load": []}
    for a, b in zip(plain.get("load") or [], guarded.get("load") or []):
        row = {"at_once": a["at_once"], "requests": a["requests"],
               "plain": {k: a[k] for k in a if k.endswith("_ms") or k == "output_tokens_per_s"},
               "guarded": {k: b[k] for k in b if k.endswith("_ms") or k == "output_tokens_per_s"}}
        if a["output_tokens_per_s"] and b["output_tokens_per_s"]:
            row["throughput_change_pct"] = round(100 * (b["output_tokens_per_s"] / a["output_tokens_per_s"] - 1), 2)
        for k in ("latency_p50_ms", "latency_p99_ms", "ttft_p50_ms", "ttft_p99_ms"):
            if a.get(k) and b.get(k):
                row[k.replace("_ms", "_change_pct")] = round(100 * (b[k] / a[k] - 1), 2)
        if a.get("errors") or b.get("errors") or a["completed"] != a["requests"] or b["completed"] != b["requests"]:
            fails.append(f"load at {a['at_once']}: requests failed (plain {a.get('errors')}, guarded "
                         f"{b.get('errors')})")
        summary["load"].append(row)
    return not fails, fails, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--metrics-port", type=int, default=9400)
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    try:
        import vllm
        version = vllm.__version__
    except Exception:  # noqa: BLE001
        version = None
    roles = {}
    for role in ("plain", "guarded"):
        roles[role] = run_role(role, a.out, a.model, a.tp, a.port, a.metrics_port)
        with open(os.path.join(a.out, f"{role}.json"), "w") as f:
            json.dump(roles[role], f, indent=1)
        print(f"{role}: ready={roles[role].get('ready')} stop={roles[role].get('stop')}", flush=True)
    ok, fails, summary = judge(roles["plain"], roles["guarded"])
    summary.update(vllm=version, passed=ok, failures=fails)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
