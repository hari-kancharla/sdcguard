"""One decision on one proposed change to a serving stack (a vLLM, PyTorch, CUDA or driver upgrade, a quantization, a
kernel backend, a parallelism or batching setting), with its questions kept apart:

    sdcguard decide --out decision.json \\
        --policy policy.json                     # what the team accepts (below; every key optional)
        --quality new.json                       # `gate model-check` of the candidate against the trusted reference
        --gate verdicts.json                     # `gate compare --json`: the kept matmul cases, before and after
        --kernels before.json after.json         # `sdcguard kernels parse` of each stack's cuBLASLt log
        --kernel-test tested.json                # `sdcguard kernels test` of the candidate's inventory, on it
        --alerts alerts.jsonl                    # the vLLM plugin's alerts while the candidate served
        --plugin-metrics w0.txt w1.txt           # each worker's sdcguard /metrics at the end: what it checked
        --perf before_perf.json after_perf.json  # `sdcguard perf` against each
        --isolation iso.json                     # `sdcguard isolation` of the candidate
        --consistency cons.json base_cons.json   # `sdcguard consistency` of the candidate (and of the baseline)
    sdcguard decide --applies decision.json      # does that decision still apply to the stack installed here?

1. Arithmetic: did the operations that were checked compute correctly? Against float64, never against the old
   stack, so a faulty baseline can't make a faulty candidate look right.
     FAULT              an output shown wrong: a plugin alert whose float64 recheck says output_wrong (or the layer's
                        weights changed in memory), a kept case REGRESSED or still wrong, a kernel test call WRONG
     NO FAULT OBSERVED  none of those, from at least one source; with what each source covered (the plugin's alerts
                        count only with its metrics showing checks ran on every worker)
     INCONCLUSIVE       an alert the recheck couldn't confirm or clear, a case INCONCLUSIVE, or no source
1b. Quantization: were FP8 layers' inputs quantized from the step's own rows, by their recipe? From the plugin's FP8
   quantization check (SDCGUARD_FP8=1): WRONG (a scale taken over rows outside the batch, a wrong scale, wrong FP8
   values), RIGHT, or UNMEASURED. Kept apart from the arithmetic: the GEMM can be right on wrongly made operands.
2. Approximation: is the change's effect on the model's outputs inside the team's budget? From `gate model-check`
   (teacher-forced next-token distributions against the trusted stack's):
     WITHIN, OUTSIDE or UNMEASURED (no check, or an INCONCLUSIVE one). WARN counts as WITHIN only when the policy says
     quality.accept_warn; the policy can also set its own max_mean_kl and min_top1.
3. Performance: does the change buy what it is for? From two `sdcguard perf` runs of the same load:
     MEETS, MISSES (a required gain not reached, a latency limit passed, or the gain within the noise between rounds)
     or UNMEASURED.
4. Isolation: does one request's answer depend on another request? From `sdcguard isolation` (primer dependence
   and canaries): LEAK, NO LEAK SEEN or UNMEASURED. A matmul check can't see this: the arithmetic can be right on
   another request's data.
4b. Consistency: does the candidate give the same answer along its serving paths (a prefix read from the prefix
   cache vs computed; decode vs prefill)? From `sdcguard consistency` of the candidate, judged against the
   baseline's own decode-vs-prefill gaps when given: CONSISTENT, INCONSISTENT or UNMEASURED. Like isolation, a
   serving-state finding: the arithmetic can be right on each path.
5. Scope: where the answer holds: the stacks (from the sources' own records), the kernels the change added and
   dropped, added kernels that match a known cuBLAS issue, added kernels no test ran, and what no source checks.

The decision: REJECT when the arithmetic shows a FAULT, the quantization is WRONG, the approximation is OUTSIDE,
the isolation check shows a LEAK, the consistency check is INCONSISTENT, or a required performance target MISSES;
ACCEPT when the arithmetic
shows NO FAULT OBSERVED, the approximation is WITHIN, and performance MEETS (or isn't required); otherwise NOT
DECIDED, with what is missing. ACCEPT means no reason to reject was found within the scope listed, not that the
stack is correct everywhere.

Policy (JSON), with these defaults:
    {"arithmetic": {"min_weights_checked": 0.5},
     "quality": {"accept_warn": false, "max_mean_kl": null, "min_top1": null},
     "performance": {"required": false, "min_throughput_gain": 0.0, "max_ttft_p99_increase": null,
                     "max_itl_p99_increase": null},
     "isolation": {"required": false},
     "consistency": {"required": false}}
Stdlib only: this runs anywhere the files are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional

SCHEMA = "sdcguard.decision/1"
DEFAULT_POLICY = {"arithmetic": {"min_weights_checked": 0.5},
                  "quality": {"accept_warn": False, "max_mean_kl": None, "min_top1": None},
                  "performance": {"required": False, "min_throughput_gain": 0.0, "max_ttft_p99_increase": None,
                                  "max_itl_p99_increase": None},
                  "isolation": {"required": False},
                  "consistency": {"required": False}}
# what none of the sources can see, whatever they say (the plugin checks linear layers; the cases and the kernel test
# rerun GEMMs; the quality check sees the model's output distributions)
NOT_CHECKED = ("attention, normalization, sampling and other non-GEMM kernels: only through the quality check",
               "requests unlike the quality check's prompts and the perf load",
               "faults that need the workload's own data or memory layout to show (the kernel test uses random inputs)")
_FAULT_RECHECK = {"output_wrong"}
_WEIGHTS_RECHECK = {"weights_changed"}
_CLEAR_RECHECK = {"output_agrees"}


def policy_with_defaults(p: Optional[dict]) -> dict:
    out = json.loads(json.dumps(DEFAULT_POLICY))
    for sec, vals in (p or {}).items():
        if sec not in out or not isinstance(vals, dict):
            raise ValueError(f"policy: unknown section {sec!r} (known: {', '.join(out)})")
        for k, v in vals.items():
            if k not in out[sec]:
                raise ValueError(f"policy: unknown key {sec}.{k}")
            out[sec][k] = v
    return out


# ---------------------------------------------------------------- 1. arithmetic
def plugin_coverage(metrics: list) -> dict:
    """What the vLLM plugin checked, from each worker's /metrics text read at the end of the run."""
    from .ready import parse
    workers = []
    for text in metrics:
        v, info = parse(text)
        workers.append({"layers_checked": v.get("sdcguard_layers_checked"),
                        "layers_unchecked": v.get("sdcguard_layers_unchecked"),
                        "weights_checked_ratio": v.get("sdcguard_weights_checked_ratio"),
                        "checks_executed": v.get("sdcguard_checks_executed_total", v.get("sdcguard_checks_executed")),
                        "reads": v.get("sdcguard_reads_total", v.get("sdcguard_reads")),
                        "fp8_layers_checked": v.get("sdcguard_fp8_layers_checked"),
                        "fp8_calls_judged": v.get("sdcguard_fp8_calls_judged_total",
                                                  v.get("sdcguard_fp8_calls_judged")),
                        "share": info.get("share"), "gpu": info.get("UUID")})
    ratios = [w["weights_checked_ratio"] for w in workers]
    return {"workers": workers,
            "checked": bool(workers) and all((w["layers_checked"] or 0) > 0 and (w["checks_executed"] or 0) > 0
                                             for w in workers),
            "min_weights_checked": min(ratios) if ratios and None not in ratios else None}


def arithmetic(alerts: Optional[list] = None, gate: Optional[dict] = None, kernel_test: Optional[dict] = None,
               plugin: Optional[dict] = None, min_weights_checked: Optional[float] = 0.5) -> dict:
    """plugin: plugin_coverage() of the candidate's run. Alerts count as a source of 'no fault' only when the
    coverage shows checks ran on every worker; a confirmed wrong output is a fault whatever the coverage."""
    faults, unsure, notes, sources, weights = [], [], [], {}, []
    if alerts is not None:
        n = {"alerts": len(alerts), "confirmed_wrong": 0, "weights_changed": 0, "cleared": 0, "unconfirmed": 0,
             "coverage": plugin}
        for ev in alerts:
            if ev.get("detector") == "serve_fp8":   # quantization findings: their own section (quantization)
                n["alerts"] -= 1
                continue
            d = ev.get("detail") or {}
            rv = (d.get("recheck") or {}).get("verdict")
            where = d.get("where") or d.get("layer") or "?"
            step = ev.get("step") if ev.get("step") is not None else (d.get("captured") or {}).get("step")
            at = f" at step {step}" if step is not None else ""
            if rv in _FAULT_RECHECK:
                n["confirmed_wrong"] += 1
                faults.append(f"the plugin's alert on {where}{at}: the float64 recheck says {rv}")
            elif rv in _WEIGHTS_RECHECK or d.get("weights_changed_since_load") is True:
                # the layer's weights no longer match their load-time checksums: a changed weight (in GPU memory,
                # or updated without refresh()), which the GPU may multiply correctly. Its own finding, not arithmetic
                n["weights_changed"] += 1
                weights.append({"where": where, "step": step})
            elif rv in _CLEAR_RECHECK:
                n["cleared"] += 1
            else:
                n["unconfirmed"] += 1
                unsure.append(f"the plugin's alert on {where}{at}: recheck {rv or 'missing (no row kept)'}")
        low = (plugin is not None and min_weights_checked is not None and
               not ((plugin.get("min_weights_checked") or 0.0) >= min_weights_checked))
        if plugin is None or not plugin.get("checked") or low:
            why = ("its coverage wasn't given (--plugin-metrics)" if plugin is None else
                   "its metrics show a worker with no checked layer or no check run" if not plugin.get("checked") else
                   f"it checks {plugin.get('min_weights_checked') or 0:.0%} of the linear-layer weights on some "
                   f"worker, under the policy's {min_weights_checked:.0%}")
            if not n["confirmed_wrong"]:
                unsure.append(f"the plugin's {len(alerts)} alerts can't stand for no fault: {why}")
            n["counts_as_source"] = False
        else:
            n["counts_as_source"] = True
        sources["plugin_alerts"] = n
    if gate is not None:
        n = {"cases": len(gate), "verdicts": {}}
        for cid, v in gate.items():
            vv = (v or {}).get("verdict", "INCONCLUSIVE")
            n["verdicts"][vv] = n["verdicts"].get(vv, 0) + 1
            if vv == "REGRESSED" or (vv == "UNCHANGED" and (v or {}).get("flavor") != "avoided"):
                faults.append(f"kept case {cid}: {vv} ({(v or {}).get('why', '')[:160]})")
            elif vv == "UNCHANGED":
                # right on the candidate, on another kernel: no fault on it here, and the old one isn't shown fixed
                notes.append(f"kept case {cid}: right on the candidate only because another kernel ran (avoided, not "
                             "shown fixed): it can come back if the old kernel is picked again")
            elif vv == "INCONCLUSIVE":
                unsure.append(f"kept case {cid}: INCONCLUSIVE ({(v or {}).get('why', '')[:160]})")
        sources["gate_cases"] = n
    if kernel_test is not None:
        c = kernel_test.get("counts") or {}
        sources["kernel_test"] = {"counts": c, "gpu": kernel_test.get("gpu")}
        for row in kernel_test.get("calls") or []:
            if row.get("verdict") == "WRONG":
                faults.append(f"kernel test: {row['runs_wrong']} of {row['runs']} runs wrong for {row['call']}")
        if kernel_test.get("warning"):
            unsure.append("kernel test: " + kernel_test["warning"])
    if faults:
        v = "FAULT"
    elif not sources:
        v, unsure = "INCONCLUSIVE", ["no arithmetic source was given"]
    elif unsure:
        v = "INCONCLUSIVE"
    else:
        v = "NO FAULT OBSERVED"
    return {"verdict": v, "faults": faults, "unresolved": unsure, "notes": notes, "sources": sources,
            "weights_changed": weights}


def weights_section(a: dict, plugin: Optional[dict]) -> dict:
    """Weight integrity: do the checked layers' weights still match their checksums at load? CHANGED (an alert
    showed a layer's weights changed since load: a corrupted or overwritten weight, whatever the arithmetic did with
    it), UNCHANGED (the plugin's checks ran on every worker and none showed it), UNMEASURED (no plugin coverage)."""
    changed = a.get("weights_changed") or []
    if changed:
        layers: dict = {}
        for w in changed:
            layers.setdefault(w["where"], []).append(w["step"])
        names = sorted(layers)
        first = [s for s in layers[names[0]] if s is not None]
        why = (f"the weights of {', '.join(names[:3])}" + (f" and {len(names) - 3} more layers" if len(names) > 3
                                                            else "")
               + f" changed since load ({len(changed)} alert{'s' if len(changed) != 1 else ''}"
               + (f", first at step {min(first)}" if first else "")
               + "): a corrupted or overwritten weight, not wrong arithmetic")
        return {"verdict": "CHANGED", "layers": names, "alerts": len(changed), "why": why}
    if plugin is not None and plugin.get("checked"):
        return {"verdict": "UNCHANGED", "why": "no checked layer's weights differed from their checksums at load"}
    return {"verdict": "UNMEASURED", "why": "no plugin coverage was given" if plugin is None else
            "the plugin's metrics show a worker with no checked layer or no check run"}


# ---------------------------------------------------------------- 2. approximation
def approximation(quality: Optional[dict], pol: dict) -> dict:
    if quality is None:
        return {"verdict": "UNMEASURED", "why": "no model-check was given"}
    v = (quality.get("verdict") or {}).get("verdict")
    why = (quality.get("verdict") or {}).get("why")
    s = quality.get("summary") or {}
    q = pol["quality"]
    if v not in ("PASS", "WARN", "FAIL"):
        return {"verdict": "UNMEASURED", "why": f"the model-check was {v or 'unreadable'}: {why}"}
    extra = []
    if q["max_mean_kl"] is not None and not (s.get("mean_kl", float("inf")) <= q["max_mean_kl"]):
        extra.append(f"mean KL {s.get('mean_kl')} above the policy's {q['max_mean_kl']}")
    if q["min_top1"] is not None and not (s.get("top1_agreement", -1.0) >= q["min_top1"]):
        extra.append(f"top-1 agreement {s.get('top1_agreement')} below the policy's {q['min_top1']}")
    if v == "FAIL" or extra or (v == "WARN" and not q["accept_warn"]):
        reason = "; ".join(extra) if extra else (why if v == "FAIL" else f"WARN, and the policy doesn't accept WARN: "
                                                                          f"{why}")
        return {"verdict": "OUTSIDE", "check": v, "why": reason, "summary": s}
    return {"verdict": "WITHIN", "check": v, "why": why, "summary": s}


# ---------------------------------------------------------------- 3. performance
def performance(base: Optional[dict], cand: Optional[dict], pol: dict) -> dict:
    p = pol["performance"]
    if base is None or cand is None:
        return {"verdict": "UNMEASURED", "required": p["required"], "why": "no pair of perf runs was given"}
    from .perf import compare
    c = compare(base, cand)
    if not c["comparable"]:
        return {"verdict": "UNMEASURED", "required": p["required"], "why": "; ".join(c["why"])}
    m = c["metrics"]
    misses = []
    tp = m.get("output_tokens_per_s")
    if tp is None:
        return {"verdict": "UNMEASURED", "required": p["required"], "why": "no throughput in one of the runs"}
    g = p["min_throughput_gain"]
    if g is not None:
        # a gain asked for has to show past the noise; "no slower" (g <= 0) tolerates a drop within the noise
        if g > 0 and tp["change"] >= g and tp["noise"]:
            misses.append(f"throughput changed {tp['change']:+.1%}, within the {tp['round_spread']:.1%} spread between "
                          "rounds, so the gain isn't shown")
        elif tp["change"] < g and not (g <= 0 and tp["noise"]):
            misses.append(f"throughput changed {tp['change']:+.1%}, the policy asks for at least {g:+.1%}")
    unmeasured = []
    for key, metric in (("max_ttft_p99_increase", "ttft_p99_ms"), ("max_itl_p99_increase", "itl_p99_ms")):
        lim, x = p[key], m.get(metric)
        if lim is None:
            continue
        ch = (x or {}).get("change")
        if not isinstance(ch, (int, float)) or ch != ch or ch in (float("inf"), float("-inf")):
            unmeasured.append(f"{metric}: the policy limits it, and one of the runs has no value for it")
        elif ch > lim and not x["noise"]:
            misses.append(f"{metric} rose {ch:+.1%}, past the policy's {lim:+.1%}")
    if misses:
        return {"verdict": "MISSES", "required": p["required"], "misses": misses, "metrics": m}
    if unmeasured:                     # a limit that can't be read isn't met
        return {"verdict": "UNMEASURED", "required": p["required"], "why": "; ".join(unmeasured), "metrics": m}
    return {"verdict": "MEETS", "required": p["required"], "misses": [], "metrics": m}


# ---------------------------------------------------------------- 5. isolation
def isolation_section(iso: Optional[dict], pol: dict) -> dict:
    req = pol["isolation"]["required"]
    if iso is None:
        return {"verdict": "UNMEASURED", "required": req, "why": "no isolation check was given"}
    v = iso.get("verdict")
    if v not in ("LEAK", "NO LEAK SEEN"):
        return {"verdict": "UNMEASURED", "required": req, "why": f"the isolation check was {v}: "
                + "; ".join(iso.get("why") or [])}
    return {"verdict": v, "required": req, "why": "; ".join(iso.get("why") or []),
            "primer_pairs": (iso.get("primer") or {}).get("pairs"),
            "primer_pairs_differing": (iso.get("primer") or {}).get("pairs_differing"),
            "canary_answers": (iso.get("canary") or {}).get("answered"),
            "canary_leaks": len((iso.get("canary") or {}).get("leaks") or [])}


# ---------------------------------------------------------------- 1b. quantization
_QUANT_FINDINGS = {"scale_from_outside_rows": "the FP8 scale was taken over rows outside the batch (an earlier "
                                              "request's data decided this batch's quantization)",
                   "scale_wrong": "an FP8 scale differs from what its recipe gives for the step's rows",
                   "values_wrong": "FP8 values the GEMM got are two or more steps from the input quantized",
                   "static_scale_changed": "a static FP8 input scale changed since load"}


def quantization_section(alerts: Optional[list], plugin: Optional[dict]) -> dict:
    """FP8 layers: were their inputs quantized from the step's own rows by their recipe? From the plugin's FP8
    quantization check (SDCGUARD_FP8=1): WRONG (a finding, whatever else), RIGHT (it judged calls on every worker
    and found nothing), UNMEASURED (it didn't run: no FP8 layers, or the check off)."""
    fp8 = [ev for ev in (alerts or []) if ev.get("detector") == "serve_fp8"]
    findings: dict = {}
    for ev in fp8:
        d = ev.get("detail") or {}
        f = d.get("finding") or "unknown"
        findings.setdefault(f, []).append(d.get("where") or "?")
    workers = (plugin or {}).get("workers") or []
    judged = [w.get("fp8_calls_judged") for w in workers]
    ran = bool(workers) and all((j or 0) > 0 for j in judged)
    if findings:
        why = "; ".join(f"{_QUANT_FINDINGS.get(f, f)}: {', '.join(sorted(set(w))[:3])}"
                        + (f" and {len(set(w)) - 3} more layers" if len(set(w)) > 3 else "")
                        for f, w in sorted(findings.items()))
        return {"verdict": "WRONG", "findings": {f: sorted(set(w)) for f, w in findings.items()}, "why": why,
                "alerts": len(fp8)}
    if ran:
        return {"verdict": "RIGHT", "calls_judged": sum(j or 0 for j in judged),
                "why": f"the FP8 quantization check judged {sum(j or 0 for j in judged)} quantizations and found "
                       "nothing"}
    return {"verdict": "UNMEASURED", "why": "the FP8 quantization check didn't run (no FP8 layers, or SDCGUARD_FP8 "
                                            "off)" if plugin is not None else "no plugin coverage was given"}


# ---------------------------------------------------------------- 4b. consistency
def consistency_section(cons: Optional[dict], base: Optional[dict], pol: dict) -> dict:
    req = pol["consistency"]["required"]
    if cons is None:
        return {"verdict": "UNMEASURED", "required": req, "why": "no consistency check was given"}
    from .consistency import judge_against
    j = judge_against(cons, base)
    v = j["verdict"]
    detail = {"prefix_cache": j["prefix_cache"], "decode_prefill": j["decode_prefill"],
              "decode_largest_gap": (cons.get("decode_prefill") or {}).get("largest_gap"),
              "baseline_decode_largest_gap": ((base or {}).get("decode_prefill") or {}).get("largest_gap"),
              "prefix_cache_hits_confirmed": (cons.get("prefix_cache") or {}).get("cache_hits_confirmed")}
    if v not in ("CONSISTENT", "INCONSISTENT"):
        return {"verdict": "UNMEASURED", "required": req, **detail,
                "why": f"the consistency check was INCONCLUSIVE (prefix cache {j['prefix_cache']}, decode vs prefill "
                       f"{j['decode_prefill']})"}
    return {"verdict": v, "required": req, **detail,
            "why": "; ".join(j["why"]) or "cached and cold answers agreed, and decode matched prefill"}


# ---------------------------------------------------------------- 4. scope
def scope(kernels_before: Optional[dict], kernels_after: Optional[dict], kernel_test: Optional[dict],
          stacks: dict) -> dict:
    out: dict = {"stacks": stacks, "not_checked": list(NOT_CHECKED)}
    if kernels_before is not None and kernels_after is not None:
        from .kernels import diff
        d = diff(kernels_before, kernels_after)
        out["kernels"] = {"before": d["kernels_before"], "after": d["kernels_after"], "same": d["same"],
                          "added": [k["key"] for k in d["new"]], "dropped": d["gone"],
                          "work_share_on_added": d["work_share_on_new_kernels"]}
        out["added_kernels_with_known_issues"] = [{"kernel": k["key"], "issues": [i["issue"] for i in k["issues"]]}
                                                  for k in d["new"] if k.get("issues")]
        tested = set()
        if kernel_test is not None:
            tested = {r["call"].split(" | ")[0] for r in kernel_test.get("calls") or [] if r.get("runs")}
        out["added_kernels_not_tested"] = [k["key"] for k in d["new"] if k["key"] not in tested]
    else:
        out["kernels"] = None
    out["fingerprint"] = fingerprint(stacks.get("candidate") or {})
    return out


_STACK_KEYS = ("gpu", "torch", "cuda", "vllm", "cublas", "cublaslt", "driver", "env")


def fingerprint(stack: dict) -> str:
    """The parts of a stack a decision depends on, hashed: another value of any of them and the decision no longer
    applies as made."""
    blob = json.dumps({k: stack.get(k) for k in _STACK_KEYS}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def still_applies(decision: dict, stack_now: dict) -> dict:
    was = ((decision.get("scope") or {}).get("stacks") or {}).get("candidate") or {}
    if not was:
        return {"applies": None, "why": "the decision doesn't record the stack it was made on"}
    if was.get("deployed") and stack_now.get("deployed"):
        from .stack import compare
        c = compare(was["deployed"], stack_now["deployed"])
        return {"applies": c["same"], "differences": c["differences"], "why": c["why"]}
    diffs = [f"{k}: {was.get(k)!r} then, {stack_now.get(k)!r} now" for k in _STACK_KEYS
             if k in was and was.get(k) != stack_now.get(k)]
    return {"applies": not diffs, "differences": diffs,
            "why": "the stack is the one the decision was made on" if not diffs else
                   "the stack changed since the decision: " + "; ".join(diffs)}


# ---------------------------------------------------------------- the decision
def decide(pol: Optional[dict] = None, quality=None, gate=None, kernels_before=None, kernels_after=None,
           kernel_test=None, alerts=None, perf_before=None, perf_after=None, plugin=None, isolation=None,
           consistency=None, consistency_baseline=None) -> dict:
    pol = policy_with_defaults(pol)
    a = arithmetic(alerts, gate, kernel_test, plugin, pol["arithmetic"]["min_weights_checked"])
    wt = weights_section(a, plugin) if alerts is not None else {"verdict": "UNMEASURED", "why": "no alerts were given"}
    qz = quantization_section(alerts, plugin)
    q = approximation(quality, pol)
    p = performance(perf_before, perf_after, pol)
    iso = isolation_section(isolation, pol)
    con = consistency_section(consistency, consistency_baseline, pol)
    stacks = {"baseline": (quality or {}).get("ref_stack"), "candidate": (quality or {}).get("stack")}
    s = scope(kernels_before, kernels_after, kernel_test, stacks)
    reject, missing = [], []
    if a["verdict"] == "FAULT":
        reject.append("arithmetic: " + a["faults"][0] + (f" (and {len(a['faults']) - 1} more)" if len(a["faults"]) > 1
                                                          else ""))
    if wt["verdict"] == "CHANGED":
        reject.append("weights: " + wt["why"])
    if qz["verdict"] == "WRONG":
        reject.append("quantization: " + qz["why"])
    if q["verdict"] == "OUTSIDE":
        reject.append("approximation: " + (q.get("why") or "outside the budget"))
    if p["verdict"] == "MISSES" and p["required"]:
        reject.append("performance: " + "; ".join(p["misses"]))
    if iso["verdict"] == "LEAK":
        reject.append("isolation: " + iso["why"])
    if con["verdict"] == "INCONSISTENT":
        reject.append("consistency: " + con["why"])
    if a["verdict"] != "NO FAULT OBSERVED" and a["verdict"] != "FAULT":
        missing.append("arithmetic: " + "; ".join(a["unresolved"][:3]))
    if q["verdict"] == "UNMEASURED":
        missing.append("approximation: " + q["why"])
    if p["required"] and p["verdict"] == "UNMEASURED":
        missing.append("performance: " + p["why"])
    if iso["required"] and iso["verdict"] == "UNMEASURED":
        missing.append("isolation: " + iso["why"])
    if con["required"] and con["verdict"] == "UNMEASURED":
        missing.append("consistency: " + con["why"])
    if reject:
        d, why = "REJECT", reject
    elif missing:
        d, why = "NOT DECIDED", missing
    else:
        d, why = "ACCEPT", ["no reason to reject was found within the scope listed"]
    return {"schema": SCHEMA, "decision": d, "why": why, "arithmetic": a, "weights": wt, "quantization": qz,
            "approximation": q,
            "performance": p,
            "isolation": iso, "consistency": con, "scope": s, "policy": pol, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def render(r: dict) -> str:
    """The decision as text a person reads first."""
    a, q, p, s = r["arithmetic"], r["approximation"], r["performance"], r["scope"]
    lines = [f"DECISION: {r['decision']}"] + [f"  {w}" for w in r["why"]] + [""]
    lines.append(f"1. Arithmetic (against float64): {a['verdict']}")
    lines += [f"   fault: {f}" for f in a["faults"][:5]] + [f"   unresolved: {u}" for u in a["unresolved"][:5]]
    lines += [f"   note: {n}" for n in a["notes"][:5]]
    for name, src in a["sources"].items():
        lines.append(f"   {name}: {json.dumps(src, default=str)}")
    wt = r.get("weights") or {}
    if wt:
        lines.append(f"1a. Weight integrity (the checked layers' weights against their checksums at load): "
                     f"{wt.get('verdict')}")
        lines.append(f"   {wt.get('why')}")
    qz = r.get("quantization") or {}
    if qz and qz.get("verdict") != "UNMEASURED":
        lines.append(f"1b. FP8 quantization (inputs quantized from the step's own rows, by their recipe): "
                     f"{qz['verdict']}")
        lines.append(f"   {qz.get('why')}")
    lines.append(f"2. Approximation (model outputs against the trusted stack): {q['verdict']}"
                 + (f", model-check {q['check']}" if q.get("check") else ""))
    if q.get("why"):
        lines.append(f"   {q['why']}")
    lines.append(f"3. Performance: {p['verdict']}" + (" (required)" if p.get("required") else " (not required)"))
    for miss in p.get("misses") or []:
        lines.append(f"   {miss}")
    for k, x in (p.get("metrics") or {}).items():
        if x:
            lines.append(f"   {k}: {x['baseline']} -> {x['candidate']} ({x['change']:+.1%}, {x['direction']}"
                         + (", within noise" if x["noise"] else "") + ")")
    if p.get("why"):                       # why it couldn't be judged (no runs, or a limited metric missing)
        lines.append(f"   {p['why']}")
    iso = r.get("isolation") or {}
    lines.append(f"4. Isolation (does one request's answer depend on another's?): {iso.get('verdict')}"
                 + (" (required)" if iso.get("required") else " (not required)"))
    if iso.get("why"):
        lines.append(f"   {iso['why']}")
    con = r.get("consistency") or {}
    if con:
        lines.append(f"4b. Consistency (the same answer from the prefix cache and cold; decode and prefill): "
                     f"{con.get('verdict')}" + (" (required)" if con.get("required") else " (not required)"))
        if con.get("why"):
            lines.append(f"   {con['why']}")
    lines.append(f"5. Scope (stack fingerprint {s['fingerprint']})")
    for side in ("baseline", "candidate"):
        st = (s.get("stacks") or {}).get(side)
        if st:
            lines.append(f"   {side}: " + ", ".join(f"{k} {st[k]}" for k in ("gpu", "vllm", "torch", "cuda")
                                                     if st.get(k)))
    if s.get("kernels") is not None:
        lines.append(f"   kernels the change added: {len(s['kernels'].get('added') or [])}, dropped: "
                     f"{len(s['kernels'].get('dropped') or [])}")
        for k in s.get("added_kernels_with_known_issues") or []:
            lines.append(f"   added, matches known issue {', '.join(k['issues'])}: {k['kernel']}")
        if s.get("added_kernels_not_tested"):
            lines.append(f"   added kernels no test ran: {len(s['added_kernels_not_tested'])}")
    lines += [f"   not checked: {n}" for n in s["not_checked"]]
    return "\n".join(lines)


def _read_jsonl(path: str) -> list:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="sdcguard decide", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy")
    p.add_argument("--quality")
    p.add_argument("--gate")
    p.add_argument("--kernels", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--kernel-test")
    p.add_argument("--alerts")
    p.add_argument("--plugin-metrics", nargs="+", metavar="METRICS",
                   help="each worker's sdcguard /metrics text, read at the end of the candidate's run")
    p.add_argument("--perf", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--isolation", help="`sdcguard isolation` of the candidate")
    p.add_argument("--consistency", nargs="+", metavar=("CANDIDATE", "BASELINE"),
                   help="`sdcguard consistency` of the candidate (and of the baseline, to judge decode gaps against)")
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.add_argument("--applies", metavar="DECISION", help="does this decision apply to the stack installed here?")
    a = p.parse_args(argv)
    if a.applies:
        from .gate_model import _stack
        r = still_applies(json.load(open(a.applies)), _stack())
        print(r["why"])
        return {True: 0, False: 1}.get(r["applies"], 2)
    ld = lambda x: json.load(open(x)) if x else None  # noqa: E731
    r = decide(ld(a.policy), ld(a.quality), ld(a.gate), *(map(ld, a.kernels) if a.kernels else (None, None)),
               ld(a.kernel_test), _read_jsonl(a.alerts) if a.alerts else None,
               *(map(ld, a.perf) if a.perf else (None, None)),
               plugin_coverage([open(x).read() for x in a.plugin_metrics]) if a.plugin_metrics else None,
               ld(a.isolation), ld(a.consistency[0]) if a.consistency else None,
               ld(a.consistency[1]) if a.consistency and len(a.consistency) > 1 else None)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(r, f, indent=1, default=str)
        with open(os.path.splitext(a.out)[0] + ".txt", "w") as f:
            f.write(render(r) + "\n")
    print(json.dumps(r, indent=1, default=str) if a.json else render(r))
    return {"ACCEPT": 0, "REJECT": 1}.get(r["decision"], 2)


if __name__ == "__main__":
    sys.exit(main())
