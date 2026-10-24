"""Does a server give the same answer along its different serving paths? Two probes against a running vLLM server
(or any server with the same completions API that takes token-id prompts, prompt_logprobs and vLLM's cache_salt):

    sdcguard consistency --url http://127.0.0.1:8000 --out cons.json

A server reaches an answer by different code depending on its state: a prompt's prefix read from the prefix cache
instead of computed, a token decoded one at a time against the KV cache instead of scored in one prefill. Each path
has its own kernels and bookkeeping, and bugs live in one path only: answers that degrade with prefix caching (vLLM
issues #5543, #22808, #60174), a decode kernel that tiles an FP8 KV cache differently from prefill (#54035). The
teacher-forced quality check (`gate model-check`) scores prefill only, and a stack compared with itself agrees there.

1. Prefix cache. Each probe prompt is a long shared document plus a question. It is answered cold twice (its own
   cache_salt each time: nothing cached), then answered right after another prompt with the same document filled the
   prefix cache (a shared cache_salt: the document's blocks are reused). The cached answer is compared with the cold
   one, and the second cold answer with the first (the floor: what running again alone moves). The server's prefix
   cache counters (its /metrics) say whether the cached run hit the cache.
2. Decode vs prefill. A greedy completion's log-probability of each token it chose (decode) against the same text
   scored in one prefill (prompt_logprobs). Completions run past 128 tokens of KV, where decode kernels change
   tiling.

A path is INCONSISTENT when, at a position where the two answers so far agree, the chosen token's log-probability
moved by more than MIN_GAP nats, or the greedy token changed where the reference answer was sure of it (its top two
log-probabilities MIN_MARGIN nats apart) or where the reference's token lost more than MIN_GAP nats in the other
answer (from the other answer's top log-probabilities: at least that much when the token isn't among them). Batch shapes alone move log-probabilities by hundredths of a nat; these
lines are for wrong paths, not for rounding. With a baseline (sdcguard qualify), decode-vs-prefill gaps are judged
against the baseline's own: a candidate whose largest gap is DECODE_FACTOR times the baseline's and at least
DECODE_FLOOR nats is INCONSISTENT. CONSISTENT covers these prompts and this server only; a probe that couldn't run,
or whose cached run didn't hit the cache, is INCONCLUSIVE. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from typing import Optional

SCHEMA = "sdcguard.consistency/1"
MIN_GAP = 0.5            # nats
MIN_MARGIN = 1.0         # nats
DECODE_FACTOR = 4.0
DECODE_FLOOR = 0.1       # nats
KV_SPLIT = 96            # the decode gaps are also reported for KV lengths up to this and past it
DOCS = (
    "The harbor town keeps a ledger of every ship that docks, with its cargo, its crew and the weather on arrival. "
    "Each entry is signed by the harbor master, and the ledger has been kept without a gap for two hundred years. ",
    "A recipe for bread begins with flour, water, salt and yeast, and the dough rests for an hour before shaping. "
    "Bakers adjust the water for the flour's age and the kitchen's warmth, and they judge the crumb by its sound. ",
    "The committee met on Tuesday to review the budget, and the minutes list each vote with the members present. "
    "Three items were deferred to the next meeting, and the treasurer was asked to bring last year's accounts. ",
    "In the observatory, the night log records the telescope's target, the exposure time and the sky conditions. "
    "Clouds ended the session early twice this month, and the dome motor was serviced on the second Friday. ",
)
QUESTIONS = (("Who signs each entry?", "What is recorded on arrival?"),
             ("How long does the dough rest?", "What do bakers adjust?"),
             ("When did the committee meet?", "What was the treasurer asked to bring?"),
             ("What does the night log record?", "What was serviced?"))
DECODE_PROMPTS = ("Explain how a hash map handles collisions.", "Write a short story about a lighthouse keeper.",
                  "Describe the water cycle step by step.", "List the steps to make a cup of tea, with reasons.")


def _post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
    from .isolation import _post as post
    return post(url, path, body, timeout)


def _ids(url: str, model: str, text: str, chat: bool = False) -> list:
    if chat:
        return list(_post(url, "/tokenize", {"model": model, "messages": [{"role": "user", "content": text}],
                                             "add_generation_prompt": True})["tokens"])
    return list(_post(url, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True})["tokens"])


def _complete(url: str, model: str, prompt: list, max_tokens: int, salt: str, top: int = 2,
              prompt_logprobs: Optional[int] = None) -> dict:
    from .gate_model import NEUTRAL       # greedy whatever the server's defaults (its model's generation_config)
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True,
            "cache_salt": salt, "logprobs": top, "return_tokens_as_token_ids": True, "return_token_ids": True,
            **NEUTRAL}
    if prompt_logprobs is not None:
        body["prompt_logprobs"] = prompt_logprobs
    ch = _post(url, "/v1/completions", body)["choices"][0]
    lp = ch.get("logprobs") or {}
    toks = ch.get("token_ids") or [int(t.split(":", 1)[1]) for t in lp.get("tokens") or []]
    maps = [{int(str(k).split(":", 1)[-1]): float(v) for k, v in (pos or {}).items()}
            for pos in lp.get("top_logprobs") or []]
    tops = [sorted(m.values(), reverse=True) for m in maps]
    plp = ch.get("prompt_logprobs")
    return {"tokens": list(toks), "logprobs": [float(v) for v in lp.get("token_logprobs") or []], "tops": tops,
            "top_maps": maps,
            "prompt_logprobs": None if plp is None else [None if p is None else {int(k): (v["logprob"] if
                                                          isinstance(v, dict) else float(v)) for k, v in p.items()}
                                                         for p in plp]}


def compare(ref: dict, other: dict) -> dict:
    """Where two greedy answers part: the largest move of a chosen token's log-probability while they agree, and a
    changed token with the reference's margin there."""
    gap, at, flip = 0.0, None, None
    for t, (a, b) in enumerate(zip(ref["tokens"], other["tokens"])):
        if a != b:
            tops = ref["tops"][t] if t < len(ref["tops"]) else []
            margin = (tops[0] - tops[1]) if len(tops) >= 2 else None
            drop = None
            om = (other.get("top_maps") or [])[t] if t < len(other.get("top_maps") or []) else {}
            if t < len(ref["logprobs"]) and om:
                theirs = om.get(a, min(om.values()))       # not among its top: at most its lowest top value
                drop = ref["logprobs"][t] - theirs
            flip = {"position": t, "reference_token": a, "token": b, "reference_margin": margin,
                    "reference_token_drop": None if drop is None else round(drop, 6)}
            break
        if t < len(ref["logprobs"]) and t < len(other["logprobs"]):
            d = abs(ref["logprobs"][t] - other["logprobs"][t])
            if d > gap:
                gap, at = d, t
    sure_flip = flip is not None and ((flip["reference_margin"] is not None and flip["reference_margin"] > MIN_MARGIN)
                                      or (flip["reference_token_drop"] is not None and
                                          flip["reference_token_drop"] > MIN_GAP))
    return {"max_gap": round(gap, 6), "at": at, "flip": flip, "significant": gap > MIN_GAP or sure_flip}


def _prefix_hits(url: str) -> Optional[float]:
    """vLLM's prefix cache hit counter (tokens), from the server's /metrics, or None when it isn't there."""
    base = url.rstrip("/")
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return None
    total = None
    for line in text.splitlines():
        if line.startswith(("vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits{", "vllm:prefix_cache_hits ")) \
                and not line.startswith("#"):
            try:
                total = (total or 0.0) + float(line.rsplit(None, 1)[-1])
            except ValueError:
                pass
    return total


def prefix_probe(url: str, model: str, doc_tokens: int = 512, max_tokens: int = 24) -> dict:
    cases, errors = [], []
    for i, doc in enumerate(DOCS):
        try:
            d = _ids(url, model, doc)
            d = [d[k % len(d)] for k in range(doc_tokens)]
            qa, qb = (_ids(url, model, " Question: " + q + " Answer:")[1:] for q in QUESTIONS[i])
            cold = _complete(url, model, d + qa, max_tokens, uuid.uuid4().hex[:16])
            cold2 = _complete(url, model, d + qa, max_tokens, uuid.uuid4().hex[:16])
            shared = uuid.uuid4().hex[:16]
            h0 = _prefix_hits(url)
            _complete(url, model, d + qb, 1, shared)
            warm = _complete(url, model, d + qa, max_tokens, shared)
            h1 = _prefix_hits(url)
        except Exception as e:  # noqa: BLE001
            errors.append(f"document {i}: {type(e).__name__}: {e}"[:300])
            continue
        hit = None if h0 is None or h1 is None else (h1 - h0) >= doc_tokens // 2
        cases.append({"document": i, "prompt_tokens": len(d) + len(qa), "cache_hit": hit,
                      "floor": compare(cold, cold2), "cached": compare(cold, warm),
                      "cold_tokens": cold["tokens"], "cached_tokens": warm["tokens"]})
    # a verdict about the cached path needs the cache hit shown (vLLM's counter): without it, either way, the
    # cached run may not have taken that path
    bad = [c for c in cases if c["cached"]["significant"] and not c["floor"]["significant"] and c["cache_hit"] is True]
    noisy = [c for c in cases if c["floor"]["significant"]]
    if bad:
        v = "INCONSISTENT"
    elif errors or not cases or noisy or any(c["cache_hit"] is not True for c in cases):
        v = "INCONCLUSIVE"
    else:
        v = "CONSISTENT"
    return {"verdict": v, "cases": cases, "errors": errors, "inconsistent": len(bad),
            "cache_hits_confirmed": None if any(c["cache_hit"] is None for c in cases) else
            all(c["cache_hit"] for c in cases), "largest_cached_gap": max((c["cached"]["max_gap"] for c in cases),
                                                                          default=None),
            "largest_floor_gap": max((c["floor"]["max_gap"] for c in cases), default=None)}


def decode_probe(url: str, model: str, max_tokens: int = 160, prompts=DECODE_PROMPTS) -> dict:
    cases, errors = [], []
    for p in prompts:
        try:
            ids = _ids(url, model, p, chat=True)
            dec = _complete(url, model, ids, max_tokens, uuid.uuid4().hex[:16], top=1)
            pre = _complete(url, model, ids + dec["tokens"], 1, uuid.uuid4().hex[:16], top=1, prompt_logprobs=1)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{p!r}: {type(e).__name__}: {e}"[:300])
            continue
        plp = pre["prompt_logprobs"] or []
        gaps = []
        for t, tok in enumerate(dec["tokens"]):
            pos = len(ids) + t
            if pos < len(plp) and plp[pos] is not None and tok in plp[pos] and t < len(dec["logprobs"]):
                gaps.append(abs(dec["logprobs"][t] - plp[pos][tok]))
            else:
                gaps.append(None)
        got = [g for g in gaps if g is not None]
        if not got:
            errors.append(f"{p!r}: no prompt_logprobs for the completion's tokens")
            continue
        big = next((t for t, g in enumerate(gaps) if g is not None and g > MIN_GAP), None)
        early = [g for t, g in enumerate(gaps) if g is not None and len(ids) + t <= KV_SPLIT]
        late = [g for t, g in enumerate(gaps) if g is not None and len(ids) + t > KV_SPLIT]
        cases.append({"prompt": p, "prompt_tokens": len(ids), "tokens": len(dec["tokens"]), "max_gap": round(max(got),
                      6), "mean_gap": round(sum(got) / len(got), 6), "kv_length_at_max": len(ids) + gaps.index(
                      max(got)), "first_over_min_gap": None if big is None else {"position": big,
                                                                                 "kv_length": len(ids) + big},
                      "missing": sum(g is None for g in gaps),
                      # by KV length: decode kernels change tiling with it (#54035: from 97 tokens of KV on)
                      "mean_gap_kv_upto_split": round(sum(early) / len(early), 6) if early else None,
                      "mean_gap_kv_past_split": round(sum(late) / len(late), 6) if late else None,
                      "gaps": [None if g is None else round(g, 6) for g in gaps]})
    largest = max((c["max_gap"] for c in cases), default=None)
    partial = [c for c in cases if c["missing"]]
    if cases and any(c["first_over_min_gap"] for c in cases):
        v = "INCONSISTENT"
    elif errors or not cases or partial:
        v = "INCONCLUSIVE"          # a position not measured is a position not shown consistent
    else:
        v = "CONSISTENT"
    return {"verdict": v, "cases": cases, "errors": errors, "largest_gap": largest, "kv_split": KV_SPLIT,
            "partial_cases": len(partial)}


def judge_against(cand: dict, base: Optional[dict]) -> dict:
    """A candidate's consistency against the baseline's (sdcguard qualify): the prefix-cache probe stands on its own
    (a cached answer must match a cold one on any stack); decode-vs-prefill gaps are judged against the baseline's."""
    out = {"prefix_cache": (cand.get("prefix_cache") or {}).get("verdict", "INCONCLUSIVE"), "why": []}
    cd, bd = cand.get("decode_prefill") or {}, (base or {}).get("decode_prefill") or {}
    cg, bg = cd.get("largest_gap"), bd.get("largest_gap")
    if bd.get("verdict") not in ("CONSISTENT", "INCONSISTENT"):
        bg = None                   # a baseline whose own probe was incomplete sets no line
    if cg is None or cd.get("verdict") not in ("CONSISTENT", "INCONSISTENT"):
        out["decode_prefill"] = "INCONCLUSIVE"      # partial or failed: a measurement with gaps decides nothing
    elif bg is None:
        out["decode_prefill"] = cd.get("verdict", "INCONCLUSIVE")
    else:
        line = max(DECODE_FACTOR * bg, DECODE_FLOOR)
        out["decode_prefill"] = "INCONSISTENT" if cg > line else "CONSISTENT"
        out["decode_line"] = round(line, 6)
        if cg > line:
            out["why"].append(f"decode vs prefill: the candidate's largest log-probability gap is {cg:.3f} nats, the "
                              f"baseline's {bg:.3f} (line {line:.3f})")
    if out["prefix_cache"] == "INCONSISTENT":
        pc = cand["prefix_cache"]
        c = next(c for c in pc["cases"] if c["cached"]["significant"])
        out["why"].append(f"prefix cache: an answer with its prefix read from the cache differs from the same prompt "
                          f"answered cold (largest gap {c['cached']['max_gap']:.3f} nats, "
                          f"{'a sure token changed at position ' + str(c['cached']['flip']['position']) if c['cached']['flip'] else 'no token changed'}"
                          f"), while two cold answers agreed within {c['floor']['max_gap']:.3f}")
    vs = (out["prefix_cache"], out["decode_prefill"])
    out["verdict"] = "INCONSISTENT" if "INCONSISTENT" in vs else ("CONSISTENT" if vs == ("CONSISTENT", "CONSISTENT")
                                                                 else "INCONCLUSIVE")
    return out


def run(url: str, model: Optional[str] = None, prefix: bool = True, decode: bool = True) -> dict:
    from .perf import server_model, server_version
    model = model or server_model(url)
    if not model:
        raise RuntimeError(f"{url}: no model given and /v1/models didn't name one")
    t0 = time.time()
    res = {"schema": SCHEMA, "url": url, "model": model, "server_version": server_version(url),
           "lines": {"min_gap": MIN_GAP, "min_margin": MIN_MARGIN, "decode_factor": DECODE_FACTOR,
                     "decode_floor": DECODE_FLOOR}}
    if prefix:
        res["prefix_cache"] = prefix_probe(url, model)
    if decode:
        res["decode_prefill"] = decode_probe(url, model)
    j = judge_against(res, None)
    res.update(verdict=j["verdict"], why=j["why"] or [
        f"prefix cache {res.get('prefix_cache', {}).get('verdict', 'not run')}, decode vs prefill "
        f"{res.get('decode_prefill', {}).get('verdict', 'not run')} (largest gap "
        f"{res.get('decode_prefill', {}).get('largest_gap')})"], seconds=round(time.time() - t0, 1),
        created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sdcguard consistency", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-prefix", action="store_true", help="skip the prefix-cache probe")
    ap.add_argument("--no-decode", action="store_true", help="skip the decode-vs-prefill probe")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = run(a.url, a.model, prefix=not a.no_prefix, decode=not a.no_decode)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1)
    print(f"{r['verdict']}: " + "; ".join(r["why"]))
    return {"CONSISTENT": 0, "INCONSISTENT": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
