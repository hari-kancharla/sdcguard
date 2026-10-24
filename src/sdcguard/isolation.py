"""Does one request's answer depend on other requests? Two probes against a running vLLM server (or any server
with the same completions API that takes token-id prompts and vLLM's cache_salt):

    sdcguard isolation --url http://127.0.0.1:8000 --out iso.json

A server can compute every matmul right and still answer one request with another's data: a stale buffer, a KV
block reused too early, a prefix-cache hit on the wrong blocks. vLLM issues in October 2026 alone report answers
that change with the previous request (#62124), decode against another request's KV prefix (#61673), and responses
containing another request's context under prefix caching (#61160). Checksums of the arithmetic can't see these.

1. Primer dependence. A short probe prompt (short enough for CUDA-graph padding), run alone, greedy, right after a
   primer request that also ran alone: on an isolated server the probe's tokens and log-probabilities don't depend
   on which primer came before. Each request gets its own cache_salt, so the prefix cache isn't shared between them.
   The control: the probe after the same primer twice, which must agree exactly (otherwise the server isn't
   deterministic at batch size 1 and this probe can't judge anything).
2. Canaries. Requests with a shared document prefix (the prefix cache shares its blocks) and a unique random code
   each, sent at once, some cancelled after their first chunk: no answer may contain another request's code, which
   only another request's context could have put there.

Verdicts: LEAK (a probe's answer changed with the primer while its control agreed, or an answer holds another
request's code); NO LEAK SEEN (neither, with both probes judged); INCONCLUSIVE (a control disagreed, requests
failed, or a probe was skipped). NO LEAK SEEN covers these prompts, this load and this server only. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
import string
import sys
import threading
import time
import uuid
from typing import Optional

SCHEMA = "sdcguard.isolation/1"
PROBES = ("import numpy as", "The capital of France is", "def add(a, b):", "1, 2, 3, 4,", "Once upon a time",
          "SELECT name FROM", "The opposite of hot is", "Water boils at")
PRIMERS = {   # name: text; each is cycled or cut to an exact number of tokens
    "plain_a": "The weather today is mild and the market was quiet all afternoon long.",
    "plain_b": "Our team shipped the new release on Friday after a long week of testing.",
    "digits": "4817 2930 5561 0042 7719 3305 8826 1164 9073 2258",
    "symbols": "#$%&*@!?<>{}[]^~|\\/;:=+-_",
    "caps": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG AGAIN",
    "repeat": "zx",
}
PRIMER_TOKENS = (8, 16)
DOCS = (
    "The harbor town keeps a ledger of every ship that docks, with its cargo, its crew and the weather on arrival. ",
    "A recipe for bread begins with flour, water, salt and yeast, and the dough rests for an hour before shaping. ",
    "The committee met on Tuesday to review the budget, and the minutes list each vote with the members present. ",
    "In the observatory, the night log records the telescope's target, the exposure time and the sky conditions. ",
)


def _post(url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    from .perf import _conn
    c, base = _conn(url, timeout)
    try:
        c.request("POST", base + path, body=json.dumps(body), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path}: HTTP {r.status}: {raw[:300]!r}")
        return json.loads(raw)
    finally:
        c.close()


def _tokenize(url: str, model: str, text: str) -> list:
    return list(_post(url, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True})["tokens"])


def _exact(ids: list, n: int) -> list:
    return [ids[i % len(ids)] for i in range(n)]


def _greedy(url: str, model: str, prompt, max_tokens: int, logprobs: Optional[int] = None) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "cache_salt": uuid.uuid4().hex[:16], "return_tokens_as_token_ids": True, "return_token_ids": True}
    if logprobs:
        body["logprobs"] = logprobs
    ch = _post(url, "/v1/completions", body)["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = ch.get("token_ids") or [int(t.split(":", 1)[1]) for t in lp.get("tokens") or []]
    return {"tokens": list(toks), "logprobs": lp.get("token_logprobs"), "text": ch.get("text")}


def primer_probe(url: str, model: str, repeats: int = 2, max_tokens: int = 8, probes=PROBES,
                 primers: Optional[dict] = None) -> dict:
    primers = primers or PRIMERS
    pids = {name: _tokenize(url, model, text) for name, text in primers.items()}
    ref_name = next(iter(primers))
    cases, errors = [], []
    for probe in probes:
        pr = _tokenize(url, model, probe)
        for n in PRIMER_TOKENS:
            runs: dict = {}
            try:
                for name in primers:
                    for _ in range(repeats + (1 if name == ref_name else 0)):
                        _greedy(url, model, _exact(pids[name], n), 2)
                        runs.setdefault(name, []).append(_greedy(url, model, pr, max_tokens, logprobs=1))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{probe!r} after {n}-token primers: {type(e).__name__}: {e}"[:300])
                continue
            ref = runs[ref_name]
            control = all(r["tokens"] == ref[0]["tokens"] and r["logprobs"] == ref[0]["logprobs"] for r in ref[1:])
            diffs = []
            for name, rs in runs.items():
                for r in (rs if name != ref_name else rs[1:]):
                    if r["tokens"] != ref[0]["tokens"]:
                        at = next(i for i, (a, b) in enumerate(zip(r["tokens"] + [None] * max_tokens,
                                                                   ref[0]["tokens"])) if a != b)
                        diffs.append({"primer": name, "first_token_differing": at, "tokens": r["tokens"],
                                      "reference_tokens": ref[0]["tokens"], "text": r["text"],
                                      "reference_text": ref[0]["text"]})
            cases.append({"probe": probe, "probe_tokens": len(pr), "primer_tokens": n, "control_agrees": control,
                          "pairs": sum(len(rs) for rs in runs.values()) - 1, "differing": diffs})
    judged = [c for c in cases if c["control_agrees"]]
    leaks = [c for c in judged if c["differing"]]
    if leaks:
        v = "LEAK"
    elif not judged or errors or len(judged) < len(cases):
        v = "INCONCLUSIVE"
    else:
        v = "NO LEAK SEEN"
    return {"verdict": v, "cases": cases, "errors": errors, "judged": len(judged), "with_leak": len(leaks),
            "pairs": sum(c["pairs"] for c in judged),
            "pairs_differing": sum(len(c["differing"]) for c in judged)}


def _code(rng: random.Random) -> str:
    a = string.ascii_uppercase + string.digits
    return "-".join("".join(rng.choice(a) for _ in range(4)) for _ in range(3))


def _stream_cancel(url: str, model: str, prompt: str, max_tokens: int) -> None:
    """Start a streamed request and drop the connection after its first chunk."""
    from .perf import _conn
    c, base = _conn(url, 120)
    try:
        c.request("POST", base + "/v1/completions", headers={"Content-Type": "application/json"},
                  body=json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
                                   "stream": True, "ignore_eos": True}))
        r = c.getresponse()
        for raw in r:
            if raw.strip().startswith(b"data:"):
                break
    finally:
        c.close()


def canary_probe(url: str, model: str, requests: int = 64, concurrency: int = 32, cancel_share: float = 0.25,
                 max_tokens: int = 24, seed: int = 0, doc_repeats: int = 6) -> dict:
    rng = random.Random(seed)
    reqs = []
    for i in range(requests):
        doc = DOCS[i % len(DOCS)] * doc_repeats
        code = _code(rng)
        reqs.append({"i": i, "group": i % len(DOCS), "code": code, "cancel": rng.random() < cancel_share,
                     "prompt": f"{doc}\n\nAccess code: {code}. Repeat the access code exactly, and nothing else.\n"
                               "Access code:"})
    codes = {r["code"]: r["i"] for r in reqs}
    results, errors = {}, []
    lock = threading.Lock()
    todo = list(reqs)

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                r = todo.pop()
            try:
                if r["cancel"]:
                    _stream_cancel(url, model, r["prompt"], max_tokens)
                    continue
                out = _post(url, "/v1/completions", {"model": model, "prompt": r["prompt"], "max_tokens": max_tokens,
                                                     "temperature": 0.0, "ignore_eos": True})
                with lock:
                    results[r["i"]] = out["choices"][0]["text"]
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(f"request {r['i']}: {type(e).__name__}: {e}"[:300])
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    leaks, echoed = [], 0
    for r in reqs:
        text = results.get(r["i"])
        if text is None:
            continue
        echoed += r["code"] in text
        for code, j in codes.items():
            if j != r["i"] and code in text:
                leaks.append({"request": r["i"], "found_code_of": j, "same_prefix": reqs[j]["group"] == r["group"],
                              "other_was_cancelled": reqs[j]["cancel"], "text": text[:200]})
    answered = len(results)
    expected = sum(not r["cancel"] for r in reqs)
    if leaks:
        v = "LEAK"
    elif errors or answered < expected:
        v = "INCONCLUSIVE"
    else:
        v = "NO LEAK SEEN"
    return {"verdict": v, "requests": requests, "cancelled": requests - expected, "answered": answered,
            "echoed_own_code": echoed, "leaks": leaks, "errors": errors[:10]}


def run(url: str, model: Optional[str] = None, repeats: int = 2, canaries: int = 64, concurrency: int = 32) -> dict:
    from .perf import server_model, server_version
    model = model or server_model(url)
    if not model:
        raise RuntimeError(f"{url}: no model given and /v1/models didn't name one")
    t0 = time.time()
    p = primer_probe(url, model, repeats)
    c = canary_probe(url, model, canaries, concurrency)
    why = []
    if p["verdict"] == "LEAK":
        ex = next(c_ for c_ in p["cases"] if c_["control_agrees"] and c_["differing"])
        d = ex["differing"][0]
        why.append(f"primer dependence: {p['pairs_differing']} of {p['pairs']} probe answers changed with the request "
                   f"before them, e.g. {ex['probe']!r} answered {d['text']!r} after the {d['primer']!r} primer and "
                   f"{d['reference_text']!r} after the reference one, whose own repeat agreed exactly")
    if c["verdict"] == "LEAK":
        lk = c["leaks"][0]
        why.append(f"canaries: {len(c['leaks'])} answers hold another request's code, e.g. request {lk['request']} "
                   f"answered with request {lk['found_code_of']}'s")
    if why:
        v = "LEAK"
    elif p["verdict"] == "NO LEAK SEEN" and c["verdict"] == "NO LEAK SEEN":
        v, why = "NO LEAK SEEN", [f"{p['pairs']} primer pairs agreed and {c['answered']} canary answers held no "
                                  "other request's code"]
    else:
        v = "INCONCLUSIVE"
        why = [f"primer probe {p['verdict']} ({p['judged']} of {len(p['cases'])} cases had a control that agreed)",
               f"canary probe {c['verdict']}"]
    return {"schema": SCHEMA, "url": url, "model": model, "server_version": server_version(url), "verdict": v,
            "why": why, "primer": p, "canary": c, "seconds": round(time.time() - t0, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sdcguard isolation", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--canaries", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.url, a.model, a.repeats, a.canaries, a.concurrency)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    print(f"{r['verdict']}: " + "; ".join(r["why"]))
    return {"NO LEAK SEEN": 0, "LEAK": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
