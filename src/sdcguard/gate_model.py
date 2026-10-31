"""Output check for upgrades of an inference stack: does the new stack give the model the same next-token
distributions as the old one?

The matmul cases of `sdcguard gate` test one library call. Most upgrade breakages are elsewhere (attention kernels,
fused MoE, quantization kernels, a config or logic change), and they show up in what the model predicts. This
check records, on the stack you trust, greedy completions of a fixed prompt set with the top 20 log-probabilities
at every position; on the new stack it feeds each prompt plus its recorded completion back in (teacher forcing:
the new stack scores the same tokens, so one near-tie flip can't make the rest of a completion diverge) and
compares the distributions position by position:

    sdcguard gate model-ref --model Qwen/Qwen2.5-1.5B-Instruct --out ref.json       # on the trusted stack
    sdcguard gate model-check ref.json --out same.json                              # same stack: the noise floor
    ... upgrade vLLM, PyTorch, CUDA, the driver, the quantization ...
    sdcguard gate model-check ref.json --out new.json --noise same.json             # the verdict

Per position: whether the new stack's most likely token is the reference's most likely one (top-1 agreement), the
KL divergence from the recorded distribution to the new one over the union of both top-20 sets (each renormalized
over that union; a token one side doesn't list gets that side's smallest listed probability, an upper bound), and
the change in the recorded token's log-probability. Against a server the reference is recorded with every sampling
setting neutral in the request (no repetition, frequency or presence penalty, no top-k, top-p or min-p), because vLLM
applies a model's generation_config.json to requests that don't set them (Qwen2.5's has a repetition penalty), and
the completion would then not be the greedy one; the reference keeps how often its recorded token was its own most
likely one (recorded_is_top1), and top-1 agreement is measured against the reference's most likely token either
way. Free-running greedy text isn't compared: across GPUs and batch sizes it
diverges for a large share of prompts on correct stacks.

The reference is scored the way the candidate is: since October 26 a reference keeps, for each recorded completion,
the reference stack's own prompt_logprobs over it (the prefill path, `logprobs`) beside the decode path's
(`decode_logprobs`), and a candidate scored with prompt_logprobs is compared with the first, one scored through Trace
Replay with the second. (Before, the reference was the decode path's and the candidate the prefill path's, so a
same-stack repeat measured the stack's own decode-vs-prefill gap: 1.8e-3 mean KL on an L4, above the pass line.) An
older reference has only decode-path distributions and is used as before; the summary says when the two paths differ.

Verdict (the thresholds are a starting point taken from published numbers, not calibrated for every model; a
same-stack run with --noise moves each line to what that run measured):
  PASS   mean KL <= max(1e-3, 3 x noise), top-1 agreement >= 99% (or, with a noise run, no more disagreement than
         the noise run's: a two-proportion test, z < 3), and no prompt's mean KL above max(0.05, 3 x noise's worst)
  WARN   mean KL up to max(7e-3, 3 x noise) (about where quantized models land) or top-1 agreement from 97% to 99%
  FAIL   anything further: mean KL above that, top-1 agreement under 97% and (with a noise run) significantly below
         the noise run's, or a prompt's mean KL above max(0.1, 3 x noise's worst). A result past these lines fails
         even when the noise run is too noisy to certify a pass.
  INCONCLUSIVE  a recorded position wasn't scored (a prompt without a result counts as all of its positions), fewer
         than 200 positions were scored, the --noise run is incomplete or of another reference or model, or the
         noise run is so noisy (its pass line at the warn line, or its own top-1 agreement under 97%) that a result
         inside it can't be told from noise
Fireworks, for example, reports KL around 1e-6 between two FP16 runs and calls 1e-3 the line between equivalent
and not; vLLM issue #61100 (an upgrade that made a model print commas) would be KL of order 1 or more.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Optional

REF_SCHEMA = "sdcguard.gate-model-ref/1"
CHECK_SCHEMA = "sdcguard.gate-model-check/1"
TOP = 20
PASS_KL, WARN_KL, FAIL_PROMPT_KL, PASS_PROMPT_KL = 1e-3, 7e-3, 0.1, 0.05
PASS_TOP1, WARN_TOP1 = 0.99, 0.97
MIN_POSITIONS = 200                     # fewer scored positions: no verdict (one flip in 24 was a FAIL before)
Z_EXCESS = 3.0                          # top-1 disagreement significantly above the noise run's (one-sided)
# sampling settings a server request sets so that a server's defaults (vLLM applies the model's generation_config.json
# to what a request leaves out) can't change the greedy completion
NEUTRAL = {"repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0, "top_p": 1.0, "top_k": 0,
           "min_p": 0.0}

# A fixed, varied prompt set (chat, explanation, code, arithmetic, lists, translation, long instructions): the same
# prompts every run, so two runs score the same positions.
PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "What is 17 multiplied by 23? Show the steps.",
    "List five common uses of a hash map.",
    "Translate into French: The meeting is moved to Thursday afternoon.",
    "Summarize the plot of a story about a lighthouse keeper who finds a message in a bottle.",
    "Write a haiku about a GPU that never sleeps.",
    "What are the differences between TCP and UDP?",
    "Give three tips for writing clear technical documentation.",
    "Solve for x: 3x + 7 = 25.",
    "Write a SQL query that returns the ten most recent orders with their customer names.",
    "Describe how a binary search works and its time complexity.",
    "What causes the seasons on Earth?",
    "Write a short email asking a colleague to review a pull request by Friday.",
    "Convert 72 degrees Fahrenheit to Celsius and explain the formula.",
    "Name the planets of the solar system in order from the Sun.",
    "Write a bash one-liner that counts the lines in every .py file in a directory.",
    "Explain what a matrix multiplication computes, with a 2 by 2 example.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Write a JSON object describing a book with a title, an author and a year.",
    "Explain overfitting in machine learning to a beginner.",
    "Write a regular expression that matches a US ZIP code.",
    "What is the derivative of x squared times sine of x?",
    "Give a recipe outline for a simple tomato soup.",
    "Explain the difference between a process and a thread.",
    "Write a limerick about a cat who learned to code.",
    "What are the main causes of inflation?",
    "Write a C function that reverses a string in place.",
    "Describe the water cycle in four steps.",
    "Explain what a checksum is and why it is useful.",
    "Continue the sequence and explain the rule: 2, 6, 12, 20, 30, ...",
    ("You are a careful assistant. Read the following instructions and follow them exactly. First, restate the "
     "task in one sentence. Second, list the assumptions you are making. Third, give a short answer. The task: "
     "estimate how many piano tuners work in a city of one million people."),
]


# ---------------------------------------------------------------- the metrics (no vLLM needed)
def _lp(v) -> float:
    if isinstance(v, dict):                     # a server's JSON: {"logprob": ..., "rank": ..., "decoded_token": ...}
        return float(v["logprob"])
    return float(getattr(v, "logprob", v))


def _usable(lps: dict) -> bool:
    """A top list that is a distribution's: no NaN or +inf, and at least one finite log-probability (-inf, a token
    with probability 0, is allowed)."""
    return (bool(lps) and not any(math.isnan(v) or v == math.inf for v in lps.values())
            and any(math.isfinite(v) for v in lps.values()))


def position_metrics(ref: dict, cand: dict, token: int) -> Optional[dict]:
    """One position: ref and cand map token id -> log-probability (each side's top list, which includes `token`,
    the recorded token, on the candidate side). Returns top-1 agreement, KL(ref || cand) over the union of both
    lists, and the change in the recorded token's log-probability.

    Non-finite numbers never make a position look agreed: a candidate list with a NaN or +inf (or nothing finite)
    is no distribution, so the position scores KL inf, no top-1 agreement, and `nonfinite`; a reference list like
    that returns None (the position can't be scored: compare() counts it as missing). A -inf is probability 0;
    where the reference gives a token probability and the candidate gives it 0, KL is inf, as the definition says."""
    ref = {int(t): _lp(v) for t, v in ref.items()}
    cand = {int(t): _lp(v) for t, v in cand.items()}
    if not _usable(ref):
        return None
    if not _usable(cand):
        return {"top1": False, "kl": math.inf, "dlogprob": None, "nonfinite": True}
    rmax, cmax = max(ref.values()), max(cand.values())
    union = set(ref) | set(cand)
    floor_r = min(v for v in ref.values() if math.isfinite(v))      # a token one side doesn't list: its smallest
    floor_c = min(v for v in cand.values() if math.isfinite(v))     # listed probability (an upper bound)
    p = {t: math.exp(ref.get(t, floor_r)) for t in union}
    q = {t: math.exp(cand.get(t, floor_c)) for t in union}
    sp, sq = sum(p.values()), sum(q.values())
    kl = 0.0
    for t in union:
        if p[t] > 0:
            kl = math.inf if q[t] == 0 else kl + (p[t] / sp) * (math.log(p[t] / sp) - math.log(q[t] / sq))
            if kl == math.inf:
                break
    # the candidate's most likely token is the reference's (where either side ties for the top, any shared one)
    top1 = bool({t for t, v in cand.items() if v == cmax} & {t for t, v in ref.items() if v == rmax})
    d = cand.get(token, math.nan) - ref.get(token, math.nan)
    return {"top1": top1, "kl": max(0.0, kl), "dlogprob": d if math.isfinite(d) else None}


def recorded_is_top1(records: list) -> Optional[float]:
    """How often a reference's recorded token is its own most likely one (1.0 for a greedy completion with nothing
    else changing the choice)."""
    n = agree = 0
    for rec in records:
        for tok, top in zip(rec.get("completion_ids") or [], rec.get("decode_logprobs") or rec.get("logprobs") or []):
            vals = {int(t): _lp(v) for t, v in (top or {}).items()}
            if not _usable(vals):
                continue
            m = max(vals.values())
            n += 1
            agree += vals.get(int(tok), -math.inf) == m
    return agree / n if n else None


def summarize(per_prompt: list) -> dict:
    """per_prompt: one list of position_metrics per prompt."""
    allpos = [m for ps in per_prompt for m in ps]
    if not allpos:
        return {"positions": 0}
    kls = sorted(m["kl"] for m in allpos)
    dl = [abs(m["dlogprob"]) for m in allpos if m["dlogprob"] is not None]
    prompt_kl = [sum(m["kl"] for m in ps) / len(ps) for ps in per_prompt if ps]
    return {"positions": len(allpos), "prompts": len(per_prompt),
            "nonfinite_positions": sum(bool(m.get("nonfinite")) for m in allpos),
            "mean_kl": sum(kls) / len(kls), "p99_kl": kls[min(len(kls) - 1, int(0.99 * len(kls)))],
            "max_kl": kls[-1], "top1_agreement": sum(m["top1"] for m in allpos) / len(allpos),
            "max_prompt_mean_kl": max(prompt_kl) if prompt_kl else 0.0,
            "mean_abs_dlogprob": sum(dl) / len(dl) if dl else None}


def _incomplete(s: dict) -> Optional[str]:
    """Why a summary doesn't cover every recorded position, or None when it does. A summary without its coverage
    counts can't be shown complete (unknown is never complete)."""
    if "expected_positions" not in s:
        return "the summary doesn't say how many positions were expected, so it can't be shown complete"
    if s.get("missing_positions") or s.get("missing_prompts") or s.get("extra_results"):
        return (f"scored {s.get('positions', 0)} of {s['expected_positions']} recorded positions"
                f" ({s.get('missing_prompts', 0)} prompts had no result, {s.get('extra_results', 0)} results matched"
                " no prompt)")
    return None


def _excess_z(f_c: float, n_c: int, f_n: float, n_n: int) -> float:
    """One-sided two-proportion z: how far the candidate's top-1 disagreement rate f_c (over n_c positions) is above
    the noise run's f_n (over n_n). 0 when it isn't above; inf when the noise run never disagreed and the candidate
    did more than chance allows (the pooled rate's variance is then from the candidate alone)."""
    if not (n_c and n_n) or not (math.isfinite(f_c) and math.isfinite(f_n)) or f_c <= f_n:
        return 0.0
    p = (f_c * n_c + f_n * n_n) / (n_c + n_n)
    se = math.sqrt(p * (1 - p) * (1 / n_c + 1 / n_n))
    return (f_c - f_n) / se if se > 0 else math.inf


def verdict(s: dict, noise: Optional[dict] = None, min_positions: int = MIN_POSITIONS) -> dict:
    """PASS / WARN / FAIL from a summary (and the summary of a same-stack run, when there is one). INCONCLUSIVE when
    either doesn't cover every recorded position (a verdict on part of the prompts would pass a stack that drops the
    rest, and an incomplete noise run would set the line from a part too), when too few positions were scored, and
    when a result within the noise of a very noisy stack can't be told apart from it. With a noise run every line
    moves with what it measured: the mean-KL and per-prompt lines to 3x its values, the top-1 line by a
    two-proportion test against its disagreement rate."""
    if not s.get("positions"):
        return {"verdict": "INCONCLUSIVE", "why": "no positions were scored"}
    why = _incomplete(s)
    if why:
        return {"verdict": "INCONCLUSIVE", "why": why}
    if s.get("nonfinite_positions"):
        return {"verdict": "FAIL", "why": f"at {s['nonfinite_positions']} of {s['positions']} positions the new "
                "stack's log-probabilities weren't a distribution (NaN, +inf, or nothing finite)"}
    if s.get("max_kl") == math.inf:
        return {"verdict": "FAIL", "why": f"at some of {s['positions']} positions the new stack gave a token the "
                "reference stack found likely a probability of 0 (KL infinite)"}
    if s["positions"] < min_positions:
        return {"verdict": "INCONCLUSIVE", "why": f"only {s['positions']} positions were scored; a verdict needs at "
                f"least {min_positions} (one flipped token would otherwise decide it)"}
    pass_kl, warn_kl, pass_prompt, fail_prompt = PASS_KL, WARN_KL, PASS_PROMPT_KL, FAIL_PROMPT_KL
    kl, top1, worst, n = s["mean_kl"], s["top1_agreement"], s["max_prompt_mean_kl"], s["positions"]
    top1_pass, top1_warn = top1 >= PASS_TOP1, top1 >= WARN_TOP1
    noisy, z, nline = None, None, ""
    if noise is not None:
        nwhy = "it scored no positions" if not noise.get("positions") else _incomplete(noise)
        if nwhy:
            return {"verdict": "INCONCLUSIVE", "why": f"the --noise run can't set the line: {nwhy}"}
        nkl, ntop1, nworst = noise.get("mean_kl"), noise.get("top1_agreement"), noise.get("max_prompt_mean_kl")
        if noise.get("nonfinite_positions") or not (isinstance(nkl, (int, float)) and 0 <= nkl < math.inf):
            return {"verdict": "INCONCLUSIVE", "why": f"the --noise run can't set the line: its mean KL is {nkl}"}
        if not (isinstance(ntop1, (int, float)) and 0 <= ntop1 <= 1):
            ntop1 = 1.0                          # (an older noise summary without it: the fixed top-1 lines)
        if not (isinstance(nworst, (int, float)) and 0 <= nworst < math.inf):
            nworst = 0.0
        pass_kl, warn_kl = max(PASS_KL, 3 * nkl), max(WARN_KL, 3 * nkl)
        pass_prompt, fail_prompt = max(PASS_PROMPT_KL, 3 * nworst), max(FAIL_PROMPT_KL, 3 * nworst)
        z = _excess_z(1 - top1, n, 1 - ntop1, int(noise["positions"]))
        not_worse = z < Z_EXCESS                 # (NaN-safe: a NaN top-1 makes z 0 here, and the KL lines fail it)
        top1_pass, top1_warn = top1_pass or not_worse, top1_warn or not_worse
        nline = (f"; the same stack's own run: mean KL {nkl:.2e}, top-1 agreement {ntop1:.2%}, worst prompt "
                 f"{nworst:.2e}; top-1 disagreement {z:.1f} standard errors above it")
        if not pass_kl < WARN_KL or ntop1 < WARN_TOP1:
            noisy = (f"the same stack's own run (mean KL {nkl:.2e}, top-1 agreement {ntop1:.2%}) is too noisy for a "
                     "pass: a result within its noise can't be told from it")
    line = (f"mean KL {kl:.2e} (pass up to {pass_kl:.1e}, fail above {warn_kl:.1e}), top-1 agreement {top1:.2%}, "
            f"worst prompt's mean KL {worst:.2e} (pass up to {pass_prompt:.2g}, fail above {fail_prompt:.2g}), over "
            f"{n} positions of {s['prompts']} prompts{nline}")
    # written so a NaN (every comparison with it is False) can only fail: PASS needs each bound shown met
    if not (kl <= warn_kl and top1_warn and worst <= fail_prompt and top1 == top1):
        return {"verdict": "FAIL", "why": line}
    if noisy:
        return {"verdict": "INCONCLUSIVE", "why": f"{noisy}; {line}"}
    if not (kl <= pass_kl and top1_pass and worst <= pass_prompt):
        return {"verdict": "WARN", "why": line}
    return {"verdict": "PASS", "why": line}


def _ref_key(ref: dict, path: str) -> str:
    """Which recorded distributions a candidate scored on `path` ("prefill": prompt_logprobs; "decode": Trace Replay)
    is compared with. A reference recorded since October 26 keeps both (`logprobs` from the prefill path,
    `decode_logprobs` from the decode); an older one has only the decode path's, under `logprobs`."""
    if ref.get("logprobs_path") == "prefill":
        return "decode_logprobs" if path == "decode" else "logprobs"
    return "logprobs"


def compare(ref: dict, cand_logprobs: list, path: str = "prefill") -> dict:
    """ref: a model-ref file's records; cand_logprobs: per prompt, the candidate's log-probabilities over the prompt
    plus the recorded completion (position i scores token i), scored on `path` (prefill: prompt_logprobs; decode:
    Trace Replay). The summary counts the recorded positions and how many weren't scored (a prompt without a result,
    a position either side lacks), and verdict() won't judge a part. It also says which path each side was scored
    on: a comparison across paths measures the stack's own decode-vs-prefill gap too."""
    records, cand = ref["records"], list(cand_logprobs)
    key = _ref_key(ref, path)
    ref_path = ("decode" if key == "decode_logprobs" else "prefill") if ref.get("logprobs_path") == "prefill" \
        else "decode"
    per_prompt, missing_prompts = [], 0
    for i, rec in enumerate(records):
        plp = cand[i] if i < len(cand) else None
        missing_prompts += plp is None
        p_len, comp = len(rec["prompt_ids"]), rec["completion_ids"]
        ms = []
        for t, tok in enumerate(comp):
            cpos = plp[p_len + t] if plp is not None and p_len + t < len(plp) else None
            rlist = rec.get(key) or []
            rpos = rlist[t] if t < len(rlist) else None
            if not cpos or not rpos:
                continue
            m = position_metrics({int(k): v for k, v in rpos.items()}, cpos, tok)
            if m is not None:                    # None: the reference's list isn't usable, so not scored
                ms.append(m)
        per_prompt.append(ms)
    summary = summarize(per_prompt)
    expected = sum(len(r["completion_ids"]) for r in records)
    summary.update(expected_positions=expected, missing_positions=expected - summary["positions"],
                   missing_prompts=missing_prompts, extra_results=max(0, len(cand) - len(records)),
                   ref_path=ref_path, candidate_path=path, same_path=ref_path == path)
    return {"per_prompt": per_prompt, "summary": summary}


def ref_digest(ref: dict) -> str:
    """What a model-check scored against: the reference's records (prompts, completions, log-probabilities). A
    --noise run must have scored the same ones."""
    import hashlib
    blob = json.dumps(ref["records"], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------- against a running server (what is deployed)
def _post(url: str, path: str, body: dict, timeout: float = 600.0) -> dict:
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


def _server_stack(url: str, model: str, extra_stack: Optional[dict]) -> dict:
    from .perf import server_version
    return {"vllm": server_version(url), "served_model": model, "url": url, **(extra_stack or {})}


def make_ref_url(url: str, out: str, tokens: int = 64, model: Optional[str] = None,
                 extra_stack: Optional[dict] = None, prompts: Optional[list] = None) -> dict:
    """The reference from a running vLLM server: the chat template applied by its /tokenize, greedy completions of
    `tokens` tokens with ignore_eos, and the top-20 log-probabilities at every position, by token id (the server
    needs --max-logprobs of at least 20, vLLM's default)."""
    from .perf import server_model
    model = model or server_model(url)
    records = []
    for p in (prompts or PROMPTS):
        ids = _post(url, "/tokenize", {"model": model, "messages": [{"role": "user", "content": p}],
                                       "add_generation_prompt": True})["tokens"]
        r = _post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": tokens, "temperature": 0.0,
                                           "logprobs": TOP, "ignore_eos": True, "return_tokens_as_token_ids": True,
                                           "return_token_ids": True, **NEUTRAL})
        ch = r["choices"][0]
        lp = ch["logprobs"]
        comp = ch.get("token_ids") or [int(t.split(":", 1)[1]) for t in lp["tokens"]]
        tops = [{k.split(":", 1)[1]: float(v) for k, v in (pos or {}).items()} for pos in lp["top_logprobs"]]
        records.append({"prompt_ids": list(ids), "completion_ids": list(comp), "decode_logprobs": tops})
    path_note = _rescore_url(url, model, records)
    ref = {"schema": REF_SCHEMA, "model": model, "dtype": None, "max_len": None, "tokens": tokens, "extra": {},
           "via": "server", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stack": _server_stack(url, model, extra_stack), **path_note,
           "recorded_is_top1": recorded_is_top1(records), "records": records}
    with open(out, "w") as f:
        json.dump(ref, f)
    return {"file": out, "prompts": len(records), "positions": sum(len(r["completion_ids"]) for r in records)}


def _prompt_logprobs_url(url: str, model: str, ids: list) -> Optional[list]:
    """The server's top-20 prompt log-probabilities over `ids` (the prefill path), one dict per position (None at
    position 0), or None when the request fails."""
    try:
        r = _post(url, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0,
                                           "prompt_logprobs": TOP})
        plp = r["choices"][0].get("prompt_logprobs")
    except Exception:  # noqa: BLE001  (the caller says what it couldn't score)
        return None
    return None if plp is None else [None if pos is None else {int(k): v for k, v in pos.items()} for pos in plp]


def _rescore(records: list, scored: list) -> dict:
    """Keep each record's prefill-path distributions (`logprobs`) beside its decode-path ones (`decode_logprobs`).
    scored: per record, prompt log-probabilities over prompt plus completion, or None. When any record couldn't be
    scored the reference stays a decode-path one (its `logprobs` are the decode's, as before October 26), and says
    why: a reference with some records of each path would compare a candidate on two paths at once."""
    prefill = []
    for rec, plp in zip(records, scored):
        p_len, n = len(rec["prompt_ids"]), len(rec["completion_ids"])
        pos = None if plp is None else plp[p_len:p_len + n]
        if pos is None or len(pos) != n or any(not x for x in pos):
            prefill = None
            break
        prefill.append([{str(int(k)): _lp(v) for k, v in x.items()} for x in pos])
    if prefill is None:
        for rec in records:
            rec["logprobs"] = rec["decode_logprobs"]
        return {"logprobs_path": "decode", "path_note": "the completions couldn't all be re-scored with prompt "
                "log-probabilities, so this reference is the decode path's and a candidate scored with prompt "
                "log-probabilities is compared across paths"}
    for rec, tops in zip(records, prefill):
        rec["logprobs"] = tops
    return {"logprobs_path": "prefill"}


def _rescore_url(url: str, model: str, records: list) -> dict:
    return _rescore(records, [_prompt_logprobs_url(url, model, rec["prompt_ids"] + rec["completion_ids"])
                              for rec in records])


def check_url(ref_path: str, out: str, url: str, model: Optional[str] = None, noise_path: Optional[str] = None,
              extra_stack: Optional[dict] = None) -> dict:
    """model-check against a running server: each prompt plus its recorded completion sent as token ids, scored with
    prompt_logprobs (teacher forcing, as the offline check does). `model` is the name the server serves; the noise
    run must come from the reference's own model, as offline."""
    from .perf import server_model
    ref = json.load(open(ref_path))
    if ref.get("schema") != REF_SCHEMA:
        raise ValueError(f"{ref_path} isn't a {REF_SCHEMA} file")
    served = model or server_model(url)
    # (a prompt without a result: compare() counts its positions as missing)
    cand = [_prompt_logprobs_url(url, served, rec["prompt_ids"] + rec["completion_ids"]) for rec in ref["records"]]
    return _finish(ref, ref_path, cand, out, served, ref.get("extra") or {}, noise_path,
                   _server_stack(url, served, extra_stack))


# ---------------------------------------------------------------- the vLLM runs
def _stack() -> dict:
    out = {}
    try:
        import torch
        out.update(torch=torch.__version__, cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:  # noqa: BLE001
        pass
    try:
        import vllm
        out["vllm"] = vllm.__version__
    except Exception:  # noqa: BLE001
        pass
    from .gate import steering_env
    out["env"] = steering_env()
    return out


def _llm(model: str, dtype: str, max_len: int, extra: dict):
    from vllm import LLM
    return LLM(model=model, dtype=dtype, seed=0, max_model_len=max_len, max_logprobs=TOP,
               gpu_memory_utilization=float(os.environ.get("SDCGUARD_GATE_GPU_MEMORY", "0.6")), **extra)


def make_ref(model: str, out: str, tokens: int = 64, dtype: str = "bfloat16", max_len: int = 2048,
             extra: Optional[dict] = None) -> dict:
    from vllm import SamplingParams
    llm = _llm(model, dtype, max_len, extra or {})
    tok = llm.get_tokenizer()
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             if getattr(tok, "chat_template", None) else p for p in PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=tokens, logprobs=TOP, ignore_eos=True)
    outs = llm.generate(chats, sp, use_tqdm=False)
    records = []
    for o in outs:
        c = o.outputs[0]
        records.append({"prompt_ids": list(o.prompt_token_ids), "completion_ids": list(c.token_ids),
                        "decode_logprobs": [{str(t): _lp(v) for t, v in pos.items()} for pos in c.logprobs]})
    # the same completions re-scored on this stack's prefill path, as a candidate is scored
    from vllm.inputs import TokensPrompt
    rescored = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["completion_ids"]) for r in records],
                            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOP), use_tqdm=False)
    path_note = _rescore(records, [o.prompt_logprobs for o in rescored])
    ref = {"schema": REF_SCHEMA, "model": model, "dtype": dtype, "max_len": max_len, "tokens": tokens,
           "extra": extra or {}, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stack": _stack(),
           **path_note, "recorded_is_top1": recorded_is_top1(records), "records": records}
    with open(out, "w") as f:
        json.dump(ref, f)
    return {"file": out, "prompts": len(records), "positions": sum(len(r["completion_ids"]) for r in records)}


def check(ref_path: str, out: str, model: Optional[str] = None, extra: Optional[dict] = None,
          noise_path: Optional[str] = None) -> dict:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    ref = json.load(open(ref_path))
    if ref.get("schema") != REF_SCHEMA:
        raise ValueError(f"{ref_path} isn't a {REF_SCHEMA} file")
    llm = _llm(model or ref["model"], ref["dtype"], ref["max_len"], extra if extra is not None else ref["extra"])
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOP)
    inputs = [TokensPrompt(prompt_token_ids=r["prompt_ids"] + r["completion_ids"]) for r in ref["records"]]
    outs = llm.generate(inputs, sp, use_tqdm=False)
    cand = [o.prompt_logprobs for o in outs]
    return _finish(ref, ref_path, cand, out, model or ref["model"], extra if extra is not None else ref["extra"],
                   noise_path, _stack())


def check_trace(ref_path: str, out: str, model: Optional[str] = None, extra: Optional[dict] = None,
                noise_path: Optional[str] = None, llm=None) -> dict:
    """model-check through vLLM's own Trace Replay (vLLM 0.32 and later, model runner V2): the engine is made to emit
    the recorded completion token by token through its decode path, CUDA graphs and all, while the log-probabilities
    come from the real logits at each step. The default check scores the same tokens through the prefill path
    (prompt_logprobs); a fault that only shows in decode is seen here and not there. `llm` is an engine already made
    (with enable_trace_replay=True), for a caller that changes it first."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    ref = json.load(open(ref_path))
    if ref.get("schema") != REF_SCHEMA:
        raise ValueError(f"{ref_path} isn't a {REF_SCHEMA} file")
    extra = dict(extra if extra is not None else ref.get("extra") or {})
    if llm is None:
        llm = _llm(model or ref["model"], ref.get("dtype") or "bfloat16", ref.get("max_len") or 2048,
                   {**extra, "enable_trace_replay": True})
    # as many tokens as the trace, and on past an end-of-sequence token in it (the default max_tokens is 16: the
    # first runs stopped there, so no trace matched and no position was scored). vLLM refuses an empty trace: a
    # record without completion tokens has no position to score and isn't sent.
    recs = ref["records"]
    sent = [i for i, r in enumerate(recs) if r["completion_ids"]]
    sps = [SamplingParams(temperature=0.0, logprobs=TOP, trace_decode_token_ids=list(recs[i]["completion_ids"]),
                          max_tokens=len(recs[i]["completion_ids"]), ignore_eos=True, **NEUTRAL) for i in sent]
    got = llm.generate([TokensPrompt(prompt_token_ids=recs[i]["prompt_ids"]) for i in sent], sps,
                       use_tqdm=False) if sent else []
    outs = [None] * len(recs)
    for i, o in zip(sent, got):
        outs[i] = o
    cand = []
    for r, o in zip(recs, outs):
        if o is None:                            # not sent: no completion token, so nothing expected there either
            cand.append([None] * len(r["prompt_ids"]))
            continue
        c = o.outputs[0]
        if list(c.token_ids) != list(r["completion_ids"]):
            cand.append(None)                    # the trace wasn't what ran: none of its positions are scored
            continue
        cand.append([None] * len(r["prompt_ids"]) + [dict(pos) for pos in (c.logprobs or [])])
    run = _finish(ref, ref_path, cand, out, model or ref["model"], extra, noise_path, {**_stack(), "via": "trace"},
                  path="decode")
    return run


def _finish(ref: dict, ref_path: str, cand: list, out: str, model: str, extra, noise_path: Optional[str],
            stack: dict, path: str = "prefill") -> dict:
    res = compare(ref, cand, path)
    digest = ref_digest(ref)
    noise, noise_why = None, None
    if noise_path:
        nrun = json.load(open(noise_path))
        noise = nrun.get("summary") or {}
        if nrun.get("ref_digest") != digest:
            noise_why = "the --noise run scored another reference (or doesn't say which)"
        elif nrun.get("model") != ref["model"]:
            # other vLLM arguments (batching, say) are what a noise run measures; another model isn't noise
            noise_why = "the --noise run scored another model than the reference's, so its differences aren't noise"
    v = ({"verdict": "INCONCLUSIVE", "why": noise_why} if noise_why else verdict(res["summary"], noise))
    run = {"schema": CHECK_SCHEMA, "ref": os.path.abspath(ref_path), "ref_digest": digest,
           "model": model, "extra": extra, "stack": stack,
           "ref_stack": ref.get("stack"),
           "summary": res["summary"], "verdict": v, "noise": noise,
           "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if os.environ.get("SDCGUARD") == "1":
        # the matmul checks ran too (the vLLM plugin): what they saw in the same traffic, when the engine ran in
        # this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
        try:
            from . import vllm_plugin
            vllm_plugin.poll(wait=True)
            g = vllm_plugin.guard()
            run["sdcguard_matmul_checks"] = ({"in_process": False} if g is None else
                                             {"in_process": True, "layers": len(g.layers),
                                              "checks_executed": g.stats.get("checks_executed"),
                                              "alerts": len(g.log.events)})
        except Exception as e:  # noqa: BLE001
            run["sdcguard_matmul_checks"] = {"error": f"{type(e).__name__}: {e}"}
    with open(out, "w") as f:
        json.dump(run, f, indent=1)
    return run


def _extra(items) -> dict:
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="sdcguard gate model-ref|model-check")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("model-ref", help="record reference completions and their top-20 log-probabilities")
    r.add_argument("--model", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--tokens", type=int, default=64)
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--max-len", type=int, default=2048)
    r.add_argument("--vllm-arg", action="append", help="KEY=VALUE passed to vllm.LLM (JSON values allowed)")
    r.add_argument("--url", default=None, help="record from this running vLLM server instead of loading the model here")
    r.add_argument("--stack", action="append", help="KEY=VALUE recorded as part of the server's stack (gpu, say)")
    c = sub.add_parser("model-check", help="score the recorded completions on this stack and compare")
    c.add_argument("ref")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default=None, help="another model than the reference's (a quantized one, say)")
    c.add_argument("--noise", default=None, help="a model-check file from the reference's own stack")
    c.add_argument("--vllm-arg", action="append", help="KEY=VALUE passed to vllm.LLM (JSON values allowed)")
    c.add_argument("--url", default=None, help="score on this running vLLM server instead of loading the model here")
    c.add_argument("--trace-replay", action="store_true",
                   help="score through vLLM's Trace Replay (the decode path) instead of prompt_logprobs (prefill)")
    c.add_argument("--stack", action="append", help="KEY=VALUE recorded as part of the server's stack (gpu, say)")
    a = p.parse_args(argv)
    if a.cmd == "model-ref":
        if a.url:
            print(json.dumps(make_ref_url(a.url, a.out, a.tokens, a.model, _extra(a.stack)), indent=1))
            return 0
        if not a.model:
            p.error("model-ref needs --model (or --url)")
        print(json.dumps(make_ref(a.model, a.out, a.tokens, a.dtype, a.max_len, _extra(a.vllm_arg)), indent=1))
        return 0
    if a.url:
        run = check_url(a.ref, a.out, a.url, a.model, a.noise, _extra(a.stack))
    elif a.trace_replay:
        run = check_trace(a.ref, a.out, a.model, _extra(a.vllm_arg) if a.vllm_arg else None, a.noise)
    else:
        run = check(a.ref, a.out, a.model, _extra(a.vllm_arg) if a.vllm_arg else None, a.noise)
    print(json.dumps({"summary": run["summary"], "verdict": run["verdict"]}, indent=1))
    return {"PASS": 0, "WARN": 1, "FAIL": 1}.get(run["verdict"]["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
