"""How fast a running vLLM server (or any server with the same completions API) serves a fixed load, measured the
same way on each side of a change:

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
aren't compared; another model is (a quantized model is one of the changes this measures), and both are named.
Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
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
        model = json.loads(raw)["data"][0]["id"] if status == 200 else None
        return model if isinstance(model, str) and model else None
    except Exception:  # noqa: BLE001
        return None


def server_version(url: str) -> Optional[str]:
    try:
        status, raw = _get(url, "/version")
        version = json.loads(raw).get("version") if status == 200 else None
        return version if isinstance(version, str) and version else None
    except Exception:  # noqa: BLE001
        return None


class _RequestFailure(RuntimeError):
    """A failed request with its observations retained for the round report."""
    def __init__(self, message, result):
        super().__init__(message)
        self.request_result = result


class _RoundInterrupted(KeyboardInterrupt):
    def __init__(self, result):
        super().__init__("measurement round interrupted")
        self.round_result = result


def one_request(url: str, model: str, prompt: str, max_tokens: int, timeout: float = 600.0) -> dict:
    """One streamed completion; EOF alone is never evidence of success.

    Failed requests raise with their partial observations attached. A successful
    request requires [DONE] and an explicitly recorded nonnegative integer usage.
    """
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    c, base = _conn(url, timeout)
    t0 = time.perf_counter()
    result = {"ttft": None, "gaps": [], "latency": None, "chunks": 0, "tokens": None,
              "complete": False, "terminal_observed": False, "usage_valid": False,
              "http_status": None, "error": None}
    last = None
    try:
        c.request("POST", base + "/v1/completions", body=json.dumps(body),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        result["http_status"] = r.status
        if r.status != 200:
            raise ValueError(f"HTTP {r.status}: {r.read()[:300]!r}")
        for raw in r:
            line = raw.decode("utf-8", "strict").strip()
            if line.startswith("event:") and line[6:].strip().lower() == "error":
                raise ValueError("SSE error event")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                result["terminal_observed"] = True
                break
            # Keep the malformed event as text so even JSON NaN remains portable
            # evidence rather than poisoning the result's numeric fields/digest.
            result["last_event"] = data
            def invalid_constant(value):
                raise ValueError(f"nonfinite JSON value: {value}")
            def unique_object(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError(f"duplicate SSE JSON key: {key}")
                    value[key] = item
                return value
            chunk = json.loads(data, parse_constant=invalid_constant, object_pairs_hook=unique_object)
            if not isinstance(chunk, dict):
                raise ValueError("SSE data must be a JSON object")
            if "error" in chunk:
                raise ValueError("SSE error payload")
            usage = chunk.get("usage")
            if usage is not None:
                if not isinstance(usage, dict):
                    raise ValueError("malformed completion-token usage object")
                tokens = usage.get("completion_tokens")
                if type(tokens) is not int or tokens < 0:
                    raise ValueError("completion_tokens must be a nonnegative integer (bool is invalid)")
                try:
                    finite_tokens = math.isfinite(tokens)
                except OverflowError:
                    finite_tokens = False
                if not finite_tokens:
                    raise ValueError("completion_tokens must have a finite numeric representation")
                if result["usage_valid"] and tokens != result["tokens"]:
                    raise ValueError("conflicting completion-token usage records")
                result["tokens"], result["usage_valid"] = tokens, True
            choices = chunk.get("choices")
            if choices is not None and (not isinstance(choices, list) or any(
                    not isinstance(ch, dict) or ch.get("text") is not None and not isinstance(ch["text"], str)
                    for ch in choices)):
                raise ValueError("malformed SSE completion choices")
            if any(ch.get("finish_reason") == "error" for ch in choices or []):
                raise ValueError("SSE completion reported an error finish reason")
            if any(ch.get("text") for ch in choices or []):
                now = time.perf_counter()
                if result["ttft"] is None:
                    result["ttft"] = now - t0
                else:
                    result["gaps"].append(now - last)
                last = now
                result["chunks"] += 1
        if not result["terminal_observed"]:
            raise ValueError("stream ended without the required SSE [DONE] terminal")
        if not result["usage_valid"]:
            raise ValueError("completion-token usage was not recorded")
        result["complete"] = True
        result.pop("last_event", None)
        return result
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
        if isinstance(exc, KeyboardInterrupt):
            exc.request_result = result
            raise
        raise _RequestFailure(result["error"], result) from exc
    finally:
        result["latency"] = time.perf_counter() - t0
        c.close()


def run_round(url: str, model: str, texts: list, concurrency: int, max_tokens: int) -> dict:
    q: queue.Queue = queue.Queue()
    for index, prompt in enumerate(texts):
        q.put((index, prompt))
    records = {}
    lock = threading.Lock()
    stop = threading.Event()
    interrupted = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                index, prompt = q.get_nowait()
            except queue.Empty:
                return
            with lock:
                records[index] = {"request_index": index, "complete": False,
                                  "error": "request was still in progress at interruption"}
            try:
                record = one_request(url, model, prompt, max_tokens)
            except (Exception, KeyboardInterrupt) as exc:
                record = getattr(exc, "request_result", {"complete": False})
                record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                if isinstance(exc, KeyboardInterrupt):
                    interrupted.set()
                    stop.set()
            with lock:
                records[index] = {**record, "request_index": index}

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        interrupted.set()
        stop.set()  # Active requests can finish; no additional requests are dequeued.
    wall = time.perf_counter() - t0
    with lock:
        request_results = [dict(records[index]) for index in sorted(records)]
    results = [row for row in request_results if row.get("complete") is True]
    errors = [row.get("error") or "request did not complete" for row in request_results
              if row.get("complete") is not True]
    ttft = [row["ttft"] for row in results if row["ttft"] is not None]
    gaps = [gap for row in results for gap in row["gaps"]]
    latencies = [row["latency"] for row in results]
    tokens = (sum(row["tokens"] for row in results) if len(results) == len(texts) and results
              and all(type(row.get("tokens")) is int and row["tokens"] >= 0
                      and row.get("usage_valid") is True for row in results) else None)
    aggregation_error = None
    try:
        throughput = round(tokens / wall, 1) if tokens is not None and wall > 0 else None
        if throughput is not None and not math.isfinite(throughput):
            raise ValueError("nonfinite output-token throughput")
    except (ArithmeticError, ValueError) as exc:
        throughput = None
        aggregation_error = f"{type(exc).__name__}: {exc}"
    ms = lambda value: None if value is None else round(1000 * value, 2)  # noqa: E731
    result = {"requests": len(texts), "completed": len(results), "errors": len(errors), "error_samples": errors[:3],
              "started": len(request_results), "unstarted": len(texts) - len(request_results),
              "request_results": request_results, "interrupted": interrupted.is_set(),
              "seconds": round(wall, 3), "output_tokens": tokens,
              "output_tokens_per_s": throughput,
              "tokens_counted_by_server": tokens is not None,
              **{f"ttft_p{q}_ms": ms(percentile(ttft, q)) for q in (50, 99)},
              **{f"itl_p{q}_ms": ms(percentile(gaps, q)) for q in (50, 99)},
              **{f"latency_p{q}_ms": ms(percentile(latencies, q)) for q in (50, 99)}}
    if aggregation_error is not None:
        result["aggregation_error"] = aggregation_error
    if interrupted.is_set():
        raise _RoundInterrupted(result)
    return result


METRICS = ("output_tokens_per_s", "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms", "latency_p50_ms",
           "latency_p99_ms")
HIGHER_IS_BETTER = {"output_tokens_per_s"}


def _bound_measure(url, model, texts, concurrency, max_tokens, rounds, warmup, identity, checks_enabled):
    from .performance_identity import (MEASUREMENT_SCHEMA, read_json, validate_identity,
                                       measurement_digest)

    if type(checks_enabled) is not bool or identity is None:
        raise ValueError("prepared identity and explicit checks on/off state must be supplied together")
    for name, value, minimum in (("concurrency", concurrency, 1), ("max_tokens", max_tokens, 1),
                                 ("rounds", rounds, 1), ("warmup", warmup, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}")
    texts = default_prompts() if texts is None else texts
    if not isinstance(texts, list) or not texts or any(not isinstance(text, str) or not text for text in texts):
        raise ValueError("a nonempty prompt list is required")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("an HTTP(S) completion endpoint is required")
    identity_path = Path(identity) if isinstance(identity, (str, Path)) else None
    prepared = read_json(identity_path) if identity_path else json.loads(json.dumps(identity, allow_nan=False))
    validate_identity(prepared)  # All local validation precedes any HTTP query.
    allowed_models = {prepared["model"], prepared["local_inputs"]["model_snapshot"]}
    if model is not None and model not in allowed_models:
        raise ValueError("requested model differs from the prepared model identity")

    def metadata():
        return {"model": server_model(url), "version": server_version(url)}

    before = metadata()
    result = {"schema": SCHEMA, "url": url, "model": model or before["model"],
              "server_version": before["version"], "identity": dict(prepared["identity"]),
              "checks_enabled": checks_enabled,
              "load": {"prompts": len(texts), "prompts_sha256": hashlib.sha256("\0".join(texts).encode()).hexdigest(),
                       "concurrency": concurrency, "max_tokens": max_tokens, "rounds": rounds, "warmup": warmup},
              "rounds": [], "warmup_results": [], "median": {name: None for name in METRICS}, "complete": False,
              "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    receipt = {"schema": MEASUREMENT_SCHEMA, "status": "inconclusive", "missing": [],
               "interrupted": False,
               "prepared_identity": prepared, "checks_enabled": checks_enabled,
               "checks_state_basis": prepared["basis"]["checks_enabled"],
               "local_validation": {"before": True, "after": False},
               "server_metadata": {"before": before, "after": None,
                                   "basis": "HTTP self-reports; live weights, engine bytes and GPU are not attested"},
               "timing_note": "Local hashing and metadata queries are outside the timed measurement rounds"}
    missing = receipt["missing"]
    if before["model"] not in allowed_models or result["model"] != before["model"]:
        missing.append("server model self-report does not match the prepared/requested model")
    if before["version"] != prepared["engine"]["version"]:
        missing.append("server version self-report does not match the pinned engine")
    if not missing:
        try:
            for prompt in texts[:warmup]:
                try:
                    result["warmup_results"].append(one_request(url, result["model"], prompt, max_tokens))
                except (Exception, KeyboardInterrupt) as exc:
                    if hasattr(exc, "request_result"):
                        result["warmup_results"].append(exc.request_result)
                    raise
            for _ in range(rounds):
                try:
                    result["rounds"].append(run_round(url, result["model"], texts, concurrency, max_tokens))
                except (Exception, KeyboardInterrupt) as exc:
                    if hasattr(exc, "round_result"):
                        result["rounds"].append(exc.round_result)
                    raise
        except (Exception, KeyboardInterrupt) as exc:  # Retain completed and partial request/round observations.
            missing.append(f"measurement did not finish: {type(exc).__name__}: {exc}")
            receipt["interrupted"] = isinstance(exc, KeyboardInterrupt)
        if not receipt["interrupted"]:
            try:
                receipt["server_metadata"]["after"] = metadata()
                if receipt["server_metadata"]["after"] != before:
                    missing.append("server model/version self-reports changed during measurement")
            except KeyboardInterrupt:
                receipt["interrupted"] = True
                missing.append("post-measurement metadata query interrupted")
    if not receipt["interrupted"]:
        try:
            validate_identity(prepared)
            if identity_path and read_json(identity_path) != prepared:
                raise ValueError("prepared identity file changed during measurement")
            receipt["local_validation"]["after"] = True
        except (Exception, KeyboardInterrupt) as exc:
            missing.append(f"post-measurement local identity check failed: {type(exc).__name__}: {exc}")
            receipt["interrupted"] = isinstance(exc, KeyboardInterrupt)
    raw = result["rounds"]
    try:
        for name in METRICS:
            values = [row[name] for row in raw if row.get(name) is not None]
            result["median"][name] = statistics.median(values) if len(values) == len(raw) and values else None
        if len(raw) != rounds or any(row["errors"] or row["completed"] != row["requests"] for row in raw):
            missing.append("not all requested measurement rounds/requests completed successfully")
        cost_metrics = ("output_tokens_per_s", "ttft_p99_ms", "itl_p99_ms", "latency_p99_ms")
        if any(row.get("tokens_counted_by_server") is not True for row in raw):
            missing.append("server completion-token usage was not recorded for every request")
        if any(type(result["median"].get(name)) not in (int, float)
               or not math.isfinite(result["median"][name]) or result["median"][name] <= 0 for name in cost_metrics):
            missing.append("complete positive throughput and tail-latency measurements required")
    except (Exception, KeyboardInterrupt) as exc:
        missing.append(f"measurement analysis failed: {type(exc).__name__}: {exc}")
        receipt["interrupted"] = receipt["interrupted"] or isinstance(exc, KeyboardInterrupt)

    # Invalid numbers remain explicitly tagged evidence, never zero or a usable
    # measurement. This also keeps late analysis/binding failures serializable.
    invalid = []
    def portable(value, path="result"):
        if isinstance(value, float) and not math.isfinite(value):
            invalid.append(path)
            return {"invalid_numeric_evidence": repr(value)}
        if isinstance(value, dict):
            return {str(key): portable(item, f"{path}.{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [portable(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if value is None or type(value) in (str, bool, int, float):
            return value
        invalid.append(path)
        return {"invalid_evidence": repr(value)}
    result = portable(result)
    receipt = portable(receipt, "measurement_receipt")
    if invalid:
        missing.append("invalid nonnumeric/nonfinite evidence at " + ", ".join(invalid))
    receipt["missing"] = list(dict.fromkeys(missing))
    result["complete"] = not receipt["missing"]
    receipt["status"] = "complete" if result["complete"] else "inconclusive"
    try:
        receipt["measurement_sha256"] = measurement_digest(result)
    except (Exception, KeyboardInterrupt) as exc:
        receipt["missing"].append(f"measurement binding failed: {type(exc).__name__}: {exc}")
        receipt["interrupted"] = receipt["interrupted"] or isinstance(exc, KeyboardInterrupt)
        receipt["status"], result["complete"] = "inconclusive", False
        receipt["measurement_sha256"] = None
    result["measurement_receipt"] = receipt
    return result


def measure(url: str, model: Optional[str] = None, texts: Optional[list] = None, concurrency: int = 16,
            max_tokens: int = 128, rounds: int = 3, warmup: int = 8, *, identity=None,
            checks_enabled=None) -> dict:
    if identity is not None or checks_enabled is not None:
        return _bound_measure(url, model, texts, concurrency, max_tokens, rounds, warmup, identity, checks_enabled)
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
    for side, r in (("baseline", base), ("candidate", cand)):
        if not r.get("complete"):
            why.append(f"the {side} run had failed or missing requests")
    out: dict = {"comparable": not why, "why": why, "metrics": {},
                 "models": [base.get("model"), cand.get("model")]}   # another model (a quantized one) is a change too
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
    p.add_argument("--identity", type=Path, help="offline identity made by sdcguard overhead prepare")
    p.add_argument("--checks", choices=("on", "off"), help="operator-declared check state; does not change the server")
    p.add_argument("--compare", nargs=2, metavar=("BASELINE", "CANDIDATE"))
    a = p.parse_args(argv)
    if (a.identity is None) != (a.checks is None):
        p.error("--identity and --checks must be supplied together")
    if a.identity is not None and (not a.out or a.compare):
        p.error("bound measurements require a new --out file and cannot use --compare")
    if a.compare:
        res = compare(json.load(open(a.compare[0])), json.load(open(a.compare[1])))
        print(json.dumps(res, indent=1))
        return 0 if res["comparable"] else 2
    texts = None
    if a.prompts_file:
        texts = [line.rstrip("\n") for line in open(a.prompts_file) if line.strip()]
    if a.identity is not None:
        try:
            # Create the result exclusively before starting any request. A failure
            # receipt is retained instead of overwriting an earlier measurement.
            with Path(a.out).open("x", encoding="utf-8") as output:
                try:
                    res = measure(a.url, a.model, texts, a.concurrency, a.max_tokens, a.rounds, a.warmup,
                                  identity=a.identity, checks_enabled=a.checks == "on")
                except (Exception, KeyboardInterrupt) as exc:
                    from .performance_identity import MEASUREMENT_SCHEMA
                    res = {"schema": SCHEMA, "url": a.url, "model": a.model, "complete": False,
                           "rounds": [], "median": {name: None for name in METRICS},
                           "measurement_receipt": {"schema": MEASUREMENT_SCHEMA, "status": "inconclusive",
                               "interrupted": isinstance(exc, KeyboardInterrupt),
                               "missing": [f"measurement preflight failed: {type(exc).__name__}: {exc}"]}}
                json.dump(res, output, indent=2, allow_nan=False)
                output.write("\n")
        except (OSError, ValueError) as exc:
            print(f"INCONCLUSIVE: result file was not created: {exc}")
            return 2
        if not res["complete"]:
            print("INCONCLUSIVE: " + "; ".join(res["measurement_receipt"]["missing"]))
            print(f"Retained result: {a.out}")
            return 130 if res["measurement_receipt"].get("interrupted") else 2
    else:
        res = measure(a.url, a.model, texts, a.concurrency, a.max_tokens, a.rounds, a.warmup)
    if a.out and a.identity is None:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    m = res["median"]
    print(f"{res['model']}: {m['output_tokens_per_s']} output tokens/s, time to first token p50 {m['ttft_p50_ms']} ms "
          f"p99 {m['ttft_p99_ms']} ms, SSE chunk gap p50 {m['itl_p50_ms']} ms p99 {m['itl_p99_ms']} ms (median of "
          f"{a.rounds} rounds" + ("" if res["complete"] else "; some requests failed") + ")")
    if a.identity is not None:
        print(f"Checks {a.checks}: operator-declared; this command did not change the server's check state")
    return 0 if res["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
