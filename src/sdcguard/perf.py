"""How fast a running vLLM server (or any OpenAI-compatible one) serves a fixed load, measured the same way on
each side of a change:

    sdcguard perf --url http://127.0.0.1:8000 --out before.json      # the stack you run now
    ... change the stack, start it again ...
    sdcguard perf --url http://127.0.0.1:8000 --out after.json
    sdcguard perf --compare before.json after.json

The load is a fixed prompt set (the same every run, or your own with --prompts-file, one prompt per line), each a
greedy completion of --max-tokens tokens with ignore_eos, so every request asks for the same number of tokens on
every stack. --concurrency requests run at a time, after --warmup requests that aren't measured, and the whole
load runs --rounds times. Per request, from the stream: time to first token, the gaps between later chunks
(inter-token latency, per chunk) and the whole latency. Per round: output tokens per second (the tokens the
server says it generated, from the stream's usage, over the round's wall time). The report gives each round and
their median.

A comparison says how the candidate's medians differ from the baseline's, and calls a difference noise when it is
smaller than the spread between rounds on either side. Two runs with another load (prompts, concurrency, tokens)
or another model aren't compared. Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import queue
import random
import statistics
import sys
import threading
import time
from typing import Optional
from urllib.parse import urlparse

SCHEMA = "sdcguard.perf/1"
_WORDS = ("cache", "river", "matrix", "garden", "protocol", "history", "engine", "letter", "planet", "theory",
          "market", "recipe", "signal", "bridge", "kernel", "forest", "contract", "lesson", "orbit", "language")
_ASKS = ("Explain {a} and {b} to a new engineer.", "Write a short story about a {a} and a {b}.",
         "Compare {a} with {b} in a list of points.", "Summarize what a {a} has to do with a {b}.",
         "Give step by step instructions that involve a {a} and a {b}.")


def default_prompts(n: int = 128, seed: int = 0) -> list:
    """A fixed, varied prompt set: short asks and long ones (a padded context of 0 to 600 words)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        a, b = rng.sample(_WORDS, 2)
        ctx = " ".join(rng.choice(_WORDS) for _ in range(rng.choice((0, 0, 40, 150, 600))))
        out.append((f"Context: {ctx}\n\n" if ctx else "") + rng.choice(_ASKS).format(a=a, b=b))
    return out


def percentile(xs: list, q: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))]


def _conn(url: str, timeout: float):
    u = urlparse(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def _get(url: str, path: str, timeout: float = 30.0):
    c, base = _conn(url, timeout)
    try:
        c.request("GET", base + path)
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def server_model(url: str) -> Optional[str]:
    try:
        status, raw = _get(url, "/v1/models")
        return json.loads(raw)["data"][0]["id"] if status == 200 else None
    except Exception:  # noqa: BLE001
        return None


def server_version(url: str) -> Optional[str]:
    try:
        status, raw = _get(url, "/version")
        return json.loads(raw).get("version") if status == 200 else None
    except Exception:  # noqa: BLE001
        return None


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: float = 600.0) -> dict:
    """One streamed greedy completion: time to first token, gaps between later chunks, latency, tokens."""
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    c, base = _conn(url, timeout)
    t0 = time.perf_counter()
    try:
        c.request("POST", base + "/v1/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
        ttft, last, gaps, chunks, tokens = None, None, [], 0, None
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                tokens = chunk["usage"].get("completion_tokens", tokens)
            if any(ch.get("text") for ch in chunk.get("choices") or []):
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    gaps.append(now - last)
                last = now
                chunks += 1
        return {"ttft": ttft, "gaps": gaps, "latency": time.perf_counter() - t0, "chunks": chunks,
                "tokens": tokens}
    finally:
        c.close()


def run_round(url: str, model: str, texts: list, concurrency: int, max_tokens: int) -> dict:
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
                r = one_request(url, model, p, max_tokens)
                with lock:
                    results.append(r)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"{type(e).__name__}: {e}"[:300])
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ttft = [r["ttft"] for r in results if r["ttft"] is not None]
    gaps = [g for r in results for g in r["gaps"]]
    lat = [r["latency"] for r in results]
    counted = [r["tokens"] for r in results if r["tokens"] is not None]
    tokens = sum(counted) if len(counted) == len(results) else None
    ms = lambda v: None if v is None else round(1000 * v, 2)  # noqa: E731
    return {"requests": len(texts), "completed": len(results), "errors": len(errors), "error_samples": errors[:3],
            "seconds": round(wall, 3), "output_tokens": tokens,
            "output_tokens_per_s": round(tokens / wall, 1) if tokens is not None and wall > 0 else None,
            "tokens_counted_by_server": tokens is not None,
            **{f"ttft_p{q}_ms": ms(percentile(ttft, q)) for q in (50, 99)},
            **{f"itl_p{q}_ms": ms(percentile(gaps, q)) for q in (50, 99)},
            **{f"latency_p{q}_ms": ms(percentile(lat, q)) for q in (50, 99)}}


METRICS = ("output_tokens_per_s", "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms", "latency_p50_ms",
           "latency_p99_ms")
HIGHER_IS_BETTER = {"output_tokens_per_s"}


def measure(url: str, model: Optional[str] = None, texts: Optional[list] = None, concurrency: int = 16,
            max_tokens: int = 128, rounds: int = 3, warmup: int = 8) -> dict:
    model = model or server_model(url)
    if not model:
        raise RuntimeError(f"{url}: no model given and /v1/models didn't name one")
    texts = texts or default_prompts()
    for p in texts[:warmup]:
        one_request(url, model, p, max_tokens)
    rs = [run_round(url, model, texts, concurrency, max_tokens) for _ in range(rounds)]
    med = {}
    for m in METRICS:
        vals = [r[m] for r in rs if r.get(m) is not None]
        med[m] = statistics.median(vals) if len(vals) == len(rs) and vals else None
    return {"schema": SCHEMA, "url": url, "model": model, "server_version": server_version(url),
            "load": {"prompts": len(texts), "prompts_sha256": hashlib.sha256("\0".join(texts).encode()).hexdigest(),
                     "concurrency": concurrency, "max_tokens": max_tokens, "rounds": rounds, "warmup": warmup},
            "rounds": rs, "median": med, "complete": all(r["errors"] == 0 and r["completed"] == r["requests"]
                                                         for r in rs),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def compare(base: dict, cand: dict) -> dict:
    """The candidate's medians against the baseline's: the relative change per metric (positive = more), whether
    it's better or worse for that metric, and whether it's inside the spread between rounds (noise)."""
    why = []
    if base.get("load") != cand.get("load"):
        why.append("the two runs didn't send the same load (prompts, concurrency, tokens or rounds)")
    if base.get("model") != cand.get("model"):
        why.append(f"another model ({base.get('model')} against {cand.get('model')})")
    for side, r in (("baseline", base), ("candidate", cand)):
        if not r.get("complete"):
            why.append(f"the {side} run had failed or missing requests")
    out: dict = {"comparable": not why, "why": why, "metrics": {}}
    if why:
        return out
    for m in METRICS:
        b, c = base["median"].get(m), cand["median"].get(m)
        if b is None or c is None or b == 0:
            out["metrics"][m] = None
            continue
        spread = 0.0
        for r in (base, cand):
            vals = [x[m] for x in r["rounds"] if x.get(m) is not None]
            med = statistics.median(vals)
            if med:
                spread = max(spread, (max(vals) - min(vals)) / med)
        change = (c - b) / b
        better = change > 0 if m in HIGHER_IS_BETTER else change < 0
        out["metrics"][m] = {"baseline": b, "candidate": c, "change": round(change, 4),
                             "direction": "better" if better else ("same" if change == 0 else "worse"),
                             "round_spread": round(spread, 4), "noise": abs(change) <= spread}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard perf", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None, help="default: the one the server names at /v1/models")
    p.add_argument("--prompts-file", default=None, help="one prompt per line (default: a fixed built-in set of 128)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    a = p.parse_args(argv)
    if a.compare:
        res = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        print(json.dumps(res, indent=1))
        return 0 if res["comparable"] else 2
    texts = None
    if a.prompts_file:
        texts = [line.rstrip("\n") for line in open(a.prompts_file) if line.strip()]
    res = measure(a.url, a.model, texts, a.concurrency, a.max_tokens, a.rounds, a.warmup)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    m = res["median"]
    print(f"{res['model']}: {m['output_tokens_per_s']} output tokens/s, time to first token p50 {m['ttft_p50_ms']} ms "
          f"p99 {m['ttft_p99_ms']} ms, inter-token p50 {m['itl_p50_ms']} ms p99 {m['itl_p99_ms']} ms (median of "
          f"{a.rounds} rounds" + ("" if res["complete"] else "; some requests failed") + ")")
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
