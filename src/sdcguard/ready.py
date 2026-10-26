"""sdcguard ready: is this vLLM server's math being checked right now? Asked from outside the server.

The plugin's own SDCGUARD_REQUIRE refuses to serve when it can't check, but it can only do that once it runs: a
plugin that fails to import is logged and skipped by vLLM's loader, and the server serves unchecked. This command
doesn't trust the plugin to report its own absence. It reads every GPU worker's sdcguard metrics
(SDCGUARD_METRICS_PORT, one port per worker) twice, a few seconds apart, and says PROTECTED only if, on every
worker it was told to expect:

  - the metrics are there (no answer: the plugin didn't load, the port isn't set, or the exporter failed);
  - linear layers are checked, none whose check never ran, the same number on every worker (and as many as
    --layers, a share of the weights of at least --min-weights, when given);
  - the alarm line is at most --max-z (default 6: a server set to alert at z=1000 never alerts);
  - checks have run and their results have been read back at least once;
  - each worker is on its own GPU (with --gpus, exactly the GPUs given: the answer is then bound to this
    server's workers, not to whatever process exports metrics on those ports), and none restarted between the
    two reads;
  - every worker's executed checks went up between the two reads. If vLLM's own metrics (--vllm, the API
    server's /metrics) show it produced tokens and a worker's checks didn't go up, that worker is serving
    unchecked. If they show no tokens in between, the server is idle and the answer is IDLE: everything else
    holds, but nothing shows the checks would run for the next request (stopped checks look the same on an idle
    server); --probe sends two short requests and reads again, which turns IDLE into PROTECTED or NOT PROTECTED
    (and does the same for a server that hasn't served a request yet, whose checks have never run: as a readiness
    probe, without --probe, a new server would stay INCONCLUSIVE and never get traffic).
    Without --vllm, checks that didn't go up can mean an idle server or stopped checks, so the answer is
    INCONCLUSIVE: pass --vllm (and --probe) for a definite one.

A worker that has confirmed a fault makes the answer FAULT FOUND, whatever else holds: an alert whose own evidence
shows a wrong result (its kept output rechecked in float64 outside the rounding envelope, or weights that no longer
match their fingerprint at load; sdcguard_confirmed_alerts_total). Used as a readiness probe, that takes the server
out of rotation until it restarts (the count starts again at 0): screen or replace the GPU first (`sdcguard check`),
and see the alert log for where and which requests. Alerts that aren't confirmed (no row kept, or a recheck that
agrees or can't judge) are listed as notes; --strict-alerts makes any alert FAULT FOUND.

With --qualified DECISION.json (a decision `sdcguard qualify` or `sdcguard decide` wrote), it also says whether this
server is the stack that decision accepted: every worker's sdcguard_stack_info (what the plugin read from inside the
server: model, its resolved revision and a digest of its weights as loaded, dtype, quantization, LoRA, attention and
MoE backends, compilation and CUDA graph modes, parallelism, versions, GPU, kernel-steering environment) against the
stack recorded in the decision for that worker. A different stack is NOT THE QUALIFIED STACK, listing what differs;
a decision that wasn't ACCEPT is NOT QUALIFIED; a server or decision with no stack recorded, or a decision older than
the weights' digest (or a server that doesn't publish one), makes the answer INCONCLUSIVE.

The checks are held to what the qualification ran with, too (checker settings aren't the stack, so they're compared
here): the share of weights checked must reach the larger of --min-weights and the decision's policy
(arithmetic.min_weights_checked), the alarm line must be at most the smaller of --max-z and the line the
qualification ran at; every detector it ran with (norms, MoE, FP8, spot checks, attention, the weight scan) must be on
now, at a line no higher than then (norm_z, attn_z, the weight scan's age target). Weaker checks are NOT PROTECTED,
listing what's weaker; another sdcguard version than the qualification's is INCONCLUSIVE (its checks may differ); a
requirement the decision doesn't record is left to the command line, with a note. --max-age-days (or the decision's
policy, ready.max_age_days; the smaller applies) makes a decision older than that INCONCLUSIVE. On the qualified
stack it also says where the server goes beyond what the qualification exercised (the decision records it): contexts
longer than any it tested, and with --vllm, requests served with longer prompts than it sent; notes, or INCONCLUSIVE
with --strict-conditions.

Exit codes: 0 PROTECTED or IDLE (a readiness probe on an idle server must pass, or no traffic ever reaches it;
use --probe to have IDLE checked), 1 FAULT FOUND or NOT PROTECTED (or, with --qualified, NOT THE QUALIFIED STACK or
NOT QUALIFIED),
2 INCONCLUSIVE (couldn't tell: no checks read back yet, a worker
restarted between the reads, checks that didn't go up with no vLLM metrics to say the server was idle). For a
Kubernetes readiness probe any nonzero code keeps traffic away, so the probe's timeoutSeconds must be longer than
--window. Only the Python standard library is used, so the check runs in a small sidecar as well as in the
server's own image. PROTECTED means the configured checks are running and their results are being read, on every
expected worker; it doesn't mean every answer is right (see what the checks cover, and SDCGUARD_SHARE: a share
checks the same subset of layers every step).

    sdcguard ready --port 9400 --workers 2 --vllm http://127.0.0.1:8000/metrics --probe http://127.0.0.1:8000
    sdcguard ready --url http://10.0.0.5:9400/metrics --url http://10.0.0.5:9401/metrics --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+(\S+)")
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
INFO = "sdcguard_protection_info"
VLLM_TOKENS = ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")


def parse(text: str) -> tuple:
    """(values, info): every series' value summed over its labels (series named *_created are left out), and
    the labels of sdcguard_protection_info (the last one seen)."""
    values: dict = {}
    info: dict = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name.endswith("_created"):
            continue
        try:
            v = float(value)
        except ValueError:
            continue
        values[name] = values.get(name, 0.0) + v
        if name == INFO and labels:
            info = dict(_LABEL.findall(labels))
    return values, info


def scrape(url: str, timeout: float = 2.0) -> tuple:
    """(values, info, error): error is None when the page was read."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            text = r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return {}, {}, f"{type(e).__name__}: {getattr(e, 'reason', e)}"
    values, info = parse(text)
    return values, info, None


def _histogram(text: str, name: str) -> list:
    """[(upper bound, cumulative count)] of a Prometheus histogram on a metrics page, summed over its label sets."""
    out: dict = {}
    for line in (text or "").splitlines():
        if not line.startswith(name + "_bucket"):
            continue
        m = re.search(r'le="([^"]+)"', line)
        try:
            le = float("inf") if m.group(1) in ("+Inf", "Inf") else float(m.group(1))
            out[le] = out.get(le, 0.0) + float(line.rsplit(None, 1)[-1])
        except (AttributeError, ValueError):
            continue
    return sorted(out.items())


def outside_tested(tested: dict, live: Optional[dict], vllm_text: Optional[str]) -> list:
    """Where a deployment can go, or went, beyond what its qualification exercised."""
    out = []
    ctx = (tested or {}).get("max_context_tokens")
    try:
        allowed = int(float((live or {}).get("max_model_len") or 0))
    except ValueError:
        allowed = 0
    if ctx and allowed > ctx:
        out.append(f"this server accepts contexts up to {allowed} tokens; the qualification tested up to {ctx}")
    pmax = (tested or {}).get("max_prompt_tokens")
    if pmax and vllm_text:
        h = _histogram(vllm_text, "vllm:request_prompt_tokens")
        if h:
            total = h[-1][1]
            le = next(((b, c) for b, c in h if b >= pmax), None)
            above = total - le[1] if le else 0
            if above > 0:
                out.append(f"at least {int(above)} of {int(total)} requests served had prompts longer than {le[0]:g} "
                           f"tokens; the qualification's longest prompt was {pmax}")
    if (tested or {}).get("prefix_cache_hit_tested") is False:
        out.append("the qualification couldn't show a prefix-cache hit (prefix caching off then, or not hit)")
    return out


# The detectors a qualification runs with (sdcguard.stack's checks, as the vLLM plugin publishes them) and how strong
# each setting is: a server running one weaker than its qualification did isn't checked as it was qualified.
DETECTORS = ("norms", "moe", "fp8", "spot", "attention", "weights")
_STRENGTH = {"off": 0, "outside_graphs": 1, "on_unknown_where": 1, "on": 2}
_LINES = {"norm_z": ("norms", "the norm check's line"), "attn_z": ("attention", "the attention check's line"),
          "weight_scan_age": ("weights", "the weight scan's verification-age target, in seconds")}


def _checks(text) -> dict:
    """sdcguard_stack_info's checks label ("attention=off,moe=on,z=6,...") as a dict."""
    out = {}
    for part in str(text or "").split(","):
        k, sep, v = part.partition("=")
        if sep and k.strip():
            out[k.strip()] = v.strip()
    return out


def _deployed(d: dict) -> tuple:
    """(the stack the plugin read on the qualified server's first worker, every worker's), from a decision. Each
    worker keeps its position: one whose record is missing or unreadable is None there."""
    cand = ((d.get("scope") or {}).get("stacks") or {}).get("candidate") or {}
    workers = [w if isinstance(w, dict) and w else None for w in cand.get("deployed_workers")] \
        if isinstance(cand.get("deployed_workers"), list) else []
    return cand.get("deployed") or None, workers


def requirements(d: Optional[dict]) -> dict:
    """What a decision required of the checks, for `ready --qualified` to hold a server to: the policy's least share
    of weights checked, the alarm line and detectors the qualification ran with (its stack's checks), the checker's
    version, and how many days the decision stands (policy ready.max_age_days). A requirement the decision doesn't
    record is None and named in "unrecorded" (the command line's value applies)."""
    d = d if isinstance(d, dict) else {}
    pol = d.get("policy") if isinstance(d.get("policy"), dict) else {}
    was, _ = _deployed(d)
    checks = _checks((was or {}).get("checks"))
    out = {"min_weights": _num((pol.get("arithmetic") or {}).get("min_weights_checked")),
           "max_z": _num(checks.get("z")), "checks": checks, "sdcguard": (was or {}).get("sdcguard") or None,
           "max_age_days": _num((pol.get("ready") or {}).get("max_age_days")), "created": d.get("created")}
    out["unrecorded"] = [what for key, what in (
        ("min_weights", "the least share of weights checked (policy arithmetic.min_weights_checked)"),
        ("max_z", "the alarm line"), ("sdcguard", "the checker's version")) if out[key] is None]
    if not any(k in checks for k in DETECTORS):
        out["unrecorded"].append("the detectors the qualification ran with")
    return out


def _weaker(want: dict, have: dict) -> tuple:
    """(what the live checks `have` run weaker than the qualification's `want`, what can't be compared)."""
    weaker, unknown = [], []
    for k in DETECTORS:
        if k not in want:
            continue
        w = _STRENGTH.get(want[k])
        if k not in have:
            if w != 0:
                unknown.append(f"{k}: the server doesn't say whether these checks run; the qualification's were "
                               f"{want[k]}")
            continue
        h = _STRENGTH.get(have[k])
        if w is None or h is None:
            unknown.append(f"{k}: {have[k]!r} now, {want[k]!r} in the qualification (a setting this doesn't know)")
        elif h < w:
            weaker.append(f"{k} checks {have[k]} now, {want[k]} in the qualification")
    for k, (detector, what) in _LINES.items():
        if k not in want or not _STRENGTH.get(have.get(detector)) or not _STRENGTH.get(want.get(detector)):
            continue                          # (a detector off on either side is judged above)
        w, h = _num(want[k]), _num(have.get(k))
        if h is None and k == "weight_scan_age":
            weaker.append(f"{what}: none now, {want[k]} in the qualification")
        elif h is None:
            unknown.append(f"{what}: not published now, {want[k]} in the qualification")
        elif w is not None and h > w:
            weaker.append(f"{what}: {have[k]} now, {want[k]} in the qualification")
    return weaker, unknown


def _age_days(created) -> Optional[float]:
    import calendar
    try:
        return (time.time() - calendar.timegm(time.strptime(str(created), "%Y-%m-%dT%H:%M:%SZ"))) / 86400.0
    except (TypeError, ValueError, OverflowError):
        return None


def qualified(decision_path: str, texts: list, vllm_text: Optional[str] = None,
              max_age_days: Optional[float] = None) -> dict:
    """Is the server whose workers' metrics pages are `texts` the stack the decision at decision_path accepted, and
    are its checks at least those the qualification ran with?
    {"verdict": None (it is) | "NOT THE QUALIFIED STACK" | "NOT PROTECTED" (weaker checks) | "NOT QUALIFIED" |
    "INCONCLUSIVE", "why", "differences", "weaker", "notes", "outside_tested": where it goes beyond what the
    qualification exercised (vllm_text: vLLM's own metrics)}. max_age_days: the command line's; the decision's
    policy (ready.max_age_days) can make it smaller."""
    from .stack import compare, from_metrics
    try:
        with open(decision_path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return {"verdict": "INCONCLUSIVE", "why": f"the decision {decision_path} couldn't be read ({e})",
                "differences": []}
    if not isinstance(d, dict):
        return {"verdict": "INCONCLUSIVE", "why": f"the decision {decision_path} isn't a decision", "differences": []}
    if d.get("decision") != "ACCEPT":
        return {"verdict": "NOT QUALIFIED", "why": f"the decision given is {d.get('decision')}, not ACCEPT",
                "differences": []}
    was, recorded = _deployed(d)
    if not was:
        return {"verdict": "INCONCLUSIVE", "differences": [],
                "why": "the decision doesn't record the stack the plugin saw (made without the plugin, or by an older "
                       "sdcguard)"}
    req = requirements(d)
    diffs, unknown, weaker, notes = [], [], [], []
    # every worker the qualification ran on must be read: one worker's page can't stand for another's (its own
    # weights shard, its own checks)
    try:
        layout = int(float(was.get("tensor_parallel") or 1)) * int(float(was.get("pipeline_parallel") or 1))
    except (TypeError, ValueError):
        layout = None
    want_n = len(recorded) if recorded else layout
    missing = [i for i, w in enumerate(recorded) if w is None]
    if missing:
        unknown.append(f"the decision has no readable stack recorded for worker{'s' if len(missing) > 1 else ''} "
                       f"{', '.join(map(str, missing))}: what ran there can't be compared")
    if recorded and layout is not None and len(recorded) != layout:
        unknown.append(f"the decision records {len(recorded)} workers, but the stack's tensor and pipeline "
                       f"parallelism make {layout}")
    if want_n is not None and len(texts) != want_n:
        unknown.append(f"{len(texts)} workers read, the qualification ran on {want_n} (" +
                       ("one stack recorded for each" if recorded else "its tensor and pipeline parallelism") +
                       "): read every worker (--workers, or one --url each)")
    for i, t in enumerate(texts):
        live = from_metrics(t or "")
        if i < len(recorded) and recorded[i] is None:
            continue                                  # (no record to compare it with: unknown above)
        # each worker against its own record (with tensor parallelism, its own shard's weights)
        want = recorded[i] if i < len(recorded) and recorded[i] else was
        c = compare(want, live)
        if c["same"] is None:
            unknown.append(f"worker {i}: {c['why']}")
        elif not c["same"]:
            diffs += [f"worker {i}: {x}" for x in c["differences"]]
        if not live:
            continue
        if req["checks"]:
            have = _checks(live.get("checks"))
            if not have:
                if any(_STRENGTH.get(req["checks"].get(k)) != 0 for k in DETECTORS if k in req["checks"]):
                    unknown.append(f"worker {i}: the server doesn't publish which checks it runs (an older plugin), "
                                   "so they can't be held to the qualification's")
            else:
                wk, uk = _weaker(req["checks"], have)
                weaker += [f"worker {i}: {x}" for x in wk]
                unknown += [f"worker {i}: {x}" for x in uk]
                if (_num(have.get("share")) or 0) < (_num(req["checks"].get("share")) or 0):
                    notes.append(f"worker {i}: checks a share {have.get('share')} of the layers, the qualification "
                                 f"{req['checks'].get('share')} (what's held is the share of weights checked)")
        if req["sdcguard"] and (live.get("sdcguard") or None) != req["sdcguard"]:
            unknown.append(f"worker {i}: sdcguard {live.get('sdcguard') or 'unknown'} now, {req['sdcguard']} in the "
                           "qualification: its checks can't be shown to be the ones qualified (qualify again)")
    for what in req["unrecorded"]:
        notes.append(f"the decision doesn't record (or set) {what}: left to the command line")
    limits = []
    if max_age_days is not None:
        limits.append((max_age_days, f"--max-age-days {max_age_days:g}"))
    if req["max_age_days"] is not None:
        limits.append((req["max_age_days"], f"the decision's policy ready.max_age_days {req['max_age_days']:g}"))
    age = _age_days(req["created"])
    if limits:
        lim, src = min(limits)
        if age is None:
            unknown.append(f"the decision doesn't say when it was made (created: {req['created']!r}), so it can't be "
                           f"shown to be within {src}")
        elif age > lim:
            unknown.append(f"the decision was made {age:.0f} days ago ({req['created']}), past {src}: qualify again")
    if age is not None:
        notes.append(f"the decision was made {age:.0f} days ago ({req['created']})")
    base = {"weaker": weaker, "notes": notes, "requirements": req}
    if diffs:
        return {**base, "verdict": "NOT THE QUALIFIED STACK", "differences": diffs,
                "why": f"{len(diffs)} difference(s) from the stack the decision accepted"}
    if weaker:
        return {**base, "verdict": "NOT PROTECTED", "differences": [],
                "why": "the qualified stack, but checked weaker than its qualification was: " + "; ".join(weaker)}
    if unknown or not texts:
        return {**base, "verdict": "INCONCLUSIVE", "differences": [], "why": "; ".join(unknown) or "no worker read"}
    live = from_metrics(texts[0] or "") if texts else None
    return {**base, "verdict": None, "differences": [], "fingerprint": was.get("fingerprint"),
            "outside_tested": outside_tested((d.get("scope") or {}).get("tested") or {}, live, vllm_text),
            "why": f"the stack the decision accepted (fingerprint {was.get('fingerprint')}) on every worker, checked "
                   "at least as its qualification was"}


def inherited(path: Optional[str], max_z: float, min_weights: Optional[float]) -> dict:
    """The stricter of the command line's alarm line and least share of weights checked and the decision's at path:
    {"max_z", "z_from", "min_weights", "weights_from", "note"}."""
    out = {"max_z": max_z, "z_from": "--max-z", "min_weights": min_weights, "weights_from": "--min-weights",
           "note": None}
    if not path:
        return out
    try:
        with open(path) as f:
            req = requirements(json.load(f))
    except (OSError, ValueError):
        return out
    held = []
    if req["max_z"] is not None:
        held.append(f"alarm line at most {req['max_z']:g}")
        if req["max_z"] < max_z:
            out.update(max_z=req["max_z"], z_from="the qualification's line")
    if req["min_weights"] is not None:
        held.append(f"at least {req['min_weights']:.0%} of the weights checked")
        if min_weights is None or req["min_weights"] > min_weights:
            out.update(min_weights=req["min_weights"],
                       weights_from="the qualification's policy (arithmetic.min_weights_checked)")
    on = [k for k in DETECTORS if _STRENGTH.get(req["checks"].get(k), 0) > 0]
    if on:
        held.append("these detectors on: " + ", ".join(on))
    if req["sdcguard"]:
        held.append(f"sdcguard {req['sdcguard']}")
    if held:
        out["note"] = ("held to the qualification's requirements (the stricter of each and the command line): "
                       + "; ".join(held))
    return out


def _raw(url: str, timeout: float = 2.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError):
        return ""


def _num(text) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def judge(first: list, second: list, vllm_first: Optional[dict], vllm_second: Optional[dict], *,
          max_z: float = 6.0, layers: Optional[int] = None, min_weights: Optional[float] = None,
          gpus: Optional[list] = None, z_from: str = "--max-z", weights_from: str = "--min-weights",
          strict_alerts: bool = False) -> dict:
    """first, second: per worker (url, values, info, error) from the two reads; vllm_*: vLLM's values or None.
    z_from, weights_from: where max_z and min_weights came from, for the problems' text. strict_alerts: any alert,
    confirmed or not, is a fault found.
    Returns {"verdict", "problems", "unsure", "notes", "workers", "faults"}."""
    problems, unsure, notes, workers, faults = [], [], [], [], []
    idle = fresh = False
    for (url, a, ia, ea), (_u, b, ib, eb) in zip(first, second):
        w = {"url": url, "gpu": ia.get("UUID") or None, "pid": ia.get("pid") or None,
             "layers_checked": a.get("sdcguard_layers_checked"),
             "checks_executed": b.get("sdcguard_checks_executed_total", a.get("sdcguard_checks_executed_total"))}
        workers.append(w)
        faults += _alerts(url, b or a, w, notes, strict_alerts)
        for err in (ea, eb):
            if err:
                problems.append(f"{url}: no answer ({err}): the plugin didn't load, SDCGUARD_METRICS_PORT isn't "
                                "set for this worker, or its exporter failed")
                break
        else:
            if "sdcguard_layers_checked" not in a or "sdcguard_layers_checked" not in b:
                problems.append(f"{url}: answers, but has no sdcguard metrics")
                continue
            if not ia:
                problems.append(f"{url}: no {INFO} (sdcguard older than this check, or not the vLLM plugin)")
                continue
            z = _num(ia.get("z"))
            if z is None or not z <= max_z:
                problems.append(f"{url}: alarm line z={ia.get('z')}, above {z_from} {max_z:g}")
            n = a.get("sdcguard_layers_checked", 0.0)
            if not n > 0:
                problems.append(f"{url}: no linear layer is checked")
            elif layers is not None and n != layers:
                problems.append(f"{url}: {n:g} layers checked, expected {layers}")
            if a.get("sdcguard_layers_never_checked", 0.0) > 0 or b.get("sdcguard_layers_never_checked", 0.0) > 0:
                problems.append(f"{url}: {max(a.get('sdcguard_layers_never_checked', 0.0), b.get('sdcguard_layers_never_checked', 0.0)):g} "
                                "registered layers whose check never ran")
            ws = a.get("sdcguard_weights_checked_ratio")
            if min_weights is not None and not (ws is not None and ws >= min_weights):
                problems.append(f"{url}: share of weights checked {ws}, below {weights_from} {min_weights:g}")
            if ia.get("pid") != ib.get("pid") or ia.get("UUID") != ib.get("UUID"):
                unsure.append(f"{url}: the worker changed between the two reads (pid {ia.get('pid')} -> "
                              f"{ib.get('pid')}): restarted")
            if not (b.get("sdcguard_reads_total", 0.0) > 0 and b.get("sdcguard_checks_executed_total", 0.0) > 0):
                fresh = True
                unsure.append(f"{url}: no check results read back yet (a server that hasn't served a request: "
                              "--probe sends two)")
    reported = [w["gpu"] for w in workers if w["gpu"]]
    if len(reported) != len(set(reported)):
        twice = sorted({g for g in reported if reported.count(g) > 1})
        problems.append(f"two workers report the same GPU ({', '.join(twice)}): "
                        "a worker is missing or answering twice")
    if gpus is not None:
        seen = set(reported)
        if seen != set(gpus):
            problems.append(f"the workers report GPUs {sorted(seen)}, not the server's {sorted(set(gpus))}: these "
                            "metrics may come from another process")
    counts = {w["layers_checked"] for w in workers if w["layers_checked"] is not None}
    if len(counts) > 1:
        problems.append(f"workers check different numbers of layers: {sorted(counts)}")
    grew = [b.get("sdcguard_checks_executed_total", 0.0) > a.get("sdcguard_checks_executed_total", 0.0)
            for (_u, a, _i, _e), (_v, b, _j, _f) in zip(first, second)]
    if vllm_first is not None and vllm_second is not None:
        t1 = sum(vllm_first.get(k, 0.0) for k in VLLM_TOKENS)
        t2 = sum(vllm_second.get(k, 0.0) for k in VLLM_TOKENS)
        if not any(k in vllm_second for k in VLLM_TOKENS):
            unsure.append("vLLM's metrics have no token counters (" + ", ".join(VLLM_TOKENS) + ")")
        elif t2 > t1:
            for w, g in zip(workers, grew):
                if not g:
                    problems.append(f"{w['url']}: vLLM produced {t2 - t1:g} tokens between the reads but this "
                                    "worker's executed checks didn't go up: it is serving unchecked")
        else:
            idle = True
            notes.append("idle: vLLM produced no tokens between the reads, so the checks weren't shown running now "
                         "(--probe sends a short request to show it)")
    elif not all(grew):
        unsure.append("executed checks didn't go up between the reads on every worker: the server is idle, or its "
                      "checks stopped, and without --vllm these can't be told apart")
    verdict = "FAULT FOUND" if faults else ("NOT PROTECTED" if problems else
                                            ("INCONCLUSIVE" if unsure else ("IDLE" if idle else "PROTECTED")))
    return {"verdict": verdict, "problems": faults + problems, "unsure": unsure, "notes": notes, "workers": workers,
            "faults": faults, "fresh": fresh}


def _alerts(url: str, v: dict, w: dict, notes: list, strict: bool) -> list:
    """A worker's alerts so far (the later read's counters): the faults it has confirmed, as problems; alerts it
    hasn't confirmed, as notes (problems with strict)."""
    total, conf = v.get("sdcguard_all_alerts_total"), v.get("sdcguard_confirmed_alerts_total")
    w["alerts"], w["confirmed_alerts"] = total, conf
    out = []
    if conf is not None and conf > 0:
        out.append(f"{url}: {conf:g} confirmed fault{'s' if conf != 1 else ''} (an output rechecked as wrong in "
                   "float64, or weights changed since load): this worker computed wrong results; take it out of "
                   "service and screen its GPU (the alert log says where, and which requests)")
    rest = (total or 0.0) - (conf or 0.0)
    if rest > 0:
        why = (f"{url}: {rest:g} alert{'s' if rest != 1 else ''} not confirmed by {'their' if rest != 1 else 'its'} "
               "own evidence (no row kept, or a recheck that agrees or couldn't judge)" if conf is not None else
               f"{url}: {rest:g} alert{'s' if rest != 1 else ''}, from a plugin too old to tell confirmed ones apart")
        (out if strict else notes).append(why + ("" if strict else "; see the alert log (--strict-alerts counts any "
                                                 "alert as a fault)"))
    return out


def probe(base: str, model: Optional[str] = None, timeout: float = 60.0) -> Optional[str]:
    """Two short greedy completions 2.5 s apart (the plugin reads its results back at the first step after 2 s
    idle), so the checks have to run, and be read, for this request. None when they were answered, else why not."""
    import http.client
    from urllib.parse import urlparse
    u = urlparse(base if "://" in base else "http://" + base)
    try:
        if model is None:
            with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/v1/models", timeout=timeout) as r:
                model = json.loads(r.read())["data"][0]["id"]
        for i in range(2):
            c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
            try:
                c.request("POST", "/v1/completions", headers={"Content-Type": "application/json"},
                          body=json.dumps({"model": model, "prompt": "sdcguard readiness probe", "max_tokens": 8,
                                           "temperature": 0.0, "ignore_eos": True}))
                r = c.getresponse()
                r.read()
                if r.status != 200:
                    return f"the probe request got HTTP {r.status}"
            finally:
                c.close()
            if i == 0:
                time.sleep(2.5)
    except (OSError, ValueError, KeyError, IndexError) as e:
        return f"the probe request failed: {type(e).__name__}: {e}"
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard ready", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, help="SDCGUARD_METRICS_PORT of the server (its workers use port, port+1, ...)")
    p.add_argument("--workers", type=int, default=1, help="GPU workers to expect (tensor x pipeline parallel size)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--url", action="append", default=[], help="a worker's metrics URL (instead of --port; repeat)")
    p.add_argument("--vllm", help="vLLM's own /metrics URL, to tell an idle server from one whose checks stopped")
    p.add_argument("--window", type=float, default=5.0, help="seconds between the two reads (default 5)")
    p.add_argument("--max-z", type=float, default=6.0, help="highest alarm line accepted (default 6)")
    p.add_argument("--layers", type=int, help="layers each worker must check")
    p.add_argument("--min-weights", type=float, help="lowest share of the model's linear-layer weights checked")
    p.add_argument("--gpus", help="the GPUs this server must run on (UUIDs, comma-separated, as nvidia-smi -L "
                   "prints them): binds the answer to this server's workers, not any process exporting metrics")
    p.add_argument("--probe", metavar="URL", help="vLLM's API base (http://host:8000): when the server is idle, send "
                   "two short requests and read again, so the checks are shown running now (PROTECTED) or not "
                   "(NOT PROTECTED) instead of IDLE")
    p.add_argument("--qualified", metavar="DECISION", help="a decision file (decision_<name>.json from sdcguard "
                   "qualify): also require this server to be the stack it accepted")
    p.add_argument("--strict-conditions", action="store_true", help="with --qualified: a server that goes beyond "
                   "what the qualification tested (longer contexts, longer prompts) is INCONCLUSIVE, not a note")
    p.add_argument("--max-age-days", type=float, metavar="DAYS", help="with --qualified: a decision made longer ago "
                   "than this is INCONCLUSIVE (the decision's policy, ready.max_age_days, can set a smaller limit)")
    p.add_argument("--strict-alerts", action="store_true", help="any alert is FAULT FOUND, not only ones whose own "
                   "evidence confirms a wrong result")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    urls = list(a.url) or ([f"http://{a.host}:{a.port + i}/metrics" for i in range(a.workers)]
                           if a.port is not None else [])
    if not urls:
        p.error("give --port (with --workers) or --url")
    if not a.window >= 0:
        p.error("--window must be 0 or more")
    first = [(u, *scrape(u)) for u in urls]
    v1 = scrape(a.vllm) if a.vllm else None
    time.sleep(a.window)
    second = [(u, *scrape(u)) for u in urls]
    v2 = scrape(a.vllm) if a.vllm else None
    vllm_first = vllm_second = None
    if a.vllm:
        if v1[2] or v2[2]:
            print(f"sdcguard ready: vLLM metrics at {a.vllm} not read ({v1[2] or v2[2]})", file=sys.stderr)
        else:
            vllm_first, vllm_second = v1[0], v2[0]
    gpus = [g.strip() for g in a.gpus.split(",") if g.strip()] if a.gpus else None
    # with --qualified, the stricter of the command line and what the qualification required
    lim = inherited(a.qualified, a.max_z, a.min_weights)
    held = dict(max_z=lim["max_z"], min_weights=lim["min_weights"], z_from=lim["z_from"],
                weights_from=lim["weights_from"], strict_alerts=a.strict_alerts)
    r = judge(first, second, vllm_first, vllm_second, layers=a.layers, gpus=gpus, **held)
    if a.vllm and vllm_first is None:
        r["unsure"].append(f"vLLM's metrics at {a.vllm} couldn't be read")
        if r["verdict"] in ("PROTECTED", "IDLE"):
            r["verdict"] = "INCONCLUSIVE"
    # an idle server, or one that hasn't served a request yet (as a readiness probe sees every new server: no traffic
    # reaches it until the probe passes), is shown its checks running by two short requests
    if a.probe and vllm_first is not None and (r["verdict"] == "IDLE" or (r["verdict"] == "INCONCLUSIVE"
                                                                          and r["fresh"] and not r["problems"])):
        why = probe(a.probe)
        if why:
            r["unsure"].append(why)
            r["verdict"] = "INCONCLUSIVE"
        else:
            time.sleep(1.0)
            third = [(u, *scrape(u)) for u in urls]
            v3 = scrape(a.vllm)
            if v3[2]:
                r["unsure"].append(f"vLLM's metrics at {a.vllm} couldn't be read after the probe")
                r["verdict"] = "INCONCLUSIVE"
            else:
                r2 = judge(second, third, vllm_second, v3[0], layers=a.layers, gpus=gpus, **held)
                r2["notes"].insert(0, "probed: two short requests were sent because the server was idle" +
                                   (" and hadn't served a request yet" if r["fresh"] else ""))
                if r2["verdict"] == "IDLE":   # vLLM counted no tokens for the probe: it can't show anything
                    r2["verdict"] = "INCONCLUSIVE"
                    r2["unsure"].append("vLLM counted no tokens for the probe requests")
                r = r2
    if a.qualified:
        q = qualified(a.qualified, [_raw(u) for u in urls], _raw(a.vllm) if a.vllm else None, a.max_age_days)
        r["qualified"] = q
        if lim["note"]:
            r["notes"].append(lim["note"])
        r["notes"] += q.get("notes") or []
        if q["verdict"] is None:
            r["notes"].append(q["why"])
            for x in q.get("outside_tested") or []:
                if a.strict_conditions:
                    r["unsure"].append("outside what was qualified: " + x)
                    if r["verdict"] in ("PROTECTED", "IDLE"):
                        r["verdict"] = "INCONCLUSIVE"
                else:
                    r["notes"].append("outside what was qualified: " + x)
        elif q["verdict"] == "INCONCLUSIVE":
            r["unsure"].append(q["why"])
            if r["verdict"] in ("PROTECTED", "IDLE"):
                r["verdict"] = "INCONCLUSIVE"
        else:                       # NOT THE QUALIFIED STACK, NOT QUALIFIED, NOT PROTECTED (weaker checks)
            r["problems"] += [q["why"], *q["differences"]] + \
                ([f"weaker than the qualification: {x}" for x in q.get("weaker") or []] if q["differences"] else [])
            if r["verdict"] not in ("FAULT FOUND", "NOT PROTECTED"):
                r["verdict"] = q["verdict"]
    if a.json:
        print(json.dumps(r, indent=1))
    else:
        print(f"{r['verdict']}: {len(urls)} worker{'s' if len(urls) != 1 else ''}")
        for line in r["problems"]:
            print(f"  problem: {line}")
        for line in r["unsure"]:
            print(f"  unsure: {line}")
        for line in r["notes"]:
            print(f"  note: {line}")
    return {"PROTECTED": 0, "IDLE": 0, "FAULT FOUND": 1, "NOT PROTECTED": 1, "NOT THE QUALIFIED STACK": 1,
            "NOT QUALIFIED": 1}.get(r["verdict"], 2)


if __name__ == "__main__":
    sys.exit(main())
