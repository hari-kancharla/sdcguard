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

1. Arithmetic: did the operations that were checked compute correctly? Against a recompute (in float64, or on the
   CPU with the kernel's own rounding steps for MoE experts), never against the old stack, so a faulty baseline
   can't make a faulty candidate look right.
     FAULT              an output shown wrong: a plugin alert whose CPU recheck says output_wrong, a kept case
                        REGRESSED or still wrong, a kernel test call WRONG
     NO FAULT OBSERVED  none of those, from at least one source that tested something; with what each source covered.
                        A source counts only with evidence of checks that ran and were judged: the plugin's alerts with
                        its metrics showing checks ran on every worker, a gate report with at least one case judged, a
                        kernel test with at least one call run and judged on the workload's own kernel (an empty
                        report, or one whose calls were all not run, tests nothing)
     INCONCLUSIVE       an alert the recheck couldn't confirm or clear, a case INCONCLUSIVE, a verdict this doesn't
                        know, a kernel test whose counts disagree with its rows, or no source that tested anything
1a. Weights: do the model's weights still match what was loaded? CHANGED (an alert, or the weight scan's metrics, show
   a weight changed since load), VERIFIED (the scheduled weight scan, SDCGUARD_WEIGHT_SCAN=1, compared every entry it
   holds on every worker with its fingerprint at load and found no mismatch; with how long ago the least recently
   verified one was, and what it leaves out), NO CHANGE SEEN (only the arithmetic checks ran: no alert showed a
   change, but their checksums see a changed weight only once it moves an output past their line, so that is no
   verification), UNMEASURED (no plugin coverage, or no alerts).
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
DECIDED, with what is missing. A changed weight REJECTs too. ACCEPT means no reason to reject was found within the
scope listed, not that the stack is correct everywhere.

Policy (JSON), with these defaults:
    {"arithmetic": {"min_weights_checked": 0.5},
     "quality": {"accept_warn": false, "max_mean_kl": null, "min_top1": null},
     "performance": {"required": false, "min_throughput_gain": 0.0, "max_ttft_p99_increase": null,
                     "max_itl_p99_increase": null},
     "isolation": {"required": false},
     "consistency": {"required": false},
     "ready": {"max_age_days": null}}
ready.max_age_days: how many days the decision stands for `sdcguard ready --qualified` (older: INCONCLUSIVE).
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
                  "consistency": {"required": False},
                  "ready": {"max_age_days": None}}
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
_GATE_JUDGED = {"PASSED", "FIXED", "REGRESSED", "UNCHANGED"}       # the gate's verdicts on a case run on both sides
_KERNEL_VERDICTS = ("PASSED", "WRONG", "OTHER KERNEL", "NOT RUN")   # kernels.test_calls: a row's verdicts, its counts


def _finite(v) -> Optional[float]:
    return v if isinstance(v, (int, float)) and v == v and v not in (float("inf"), float("-inf")) else None


def _count(v) -> Optional[float]:
    """A counter's value, or None when it isn't a finite count (absent, NaN, infinite or negative)."""
    v = _finite(v)
    return v if v is not None and v >= 0 else None


def plugin_coverage(metrics: list) -> dict:
    """What the vLLM plugin checked, from each worker's /metrics text read at the end of the run: its arithmetic
    checks, and its scheduled weight scan (SDCGUARD_WEIGHT_SCAN=1: the sdcguard_weight_scan_* series, absent when the
    scan is off)."""
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
                        "weight_scan_layers": v.get("sdcguard_weight_scan_layers"),
                        "weight_scan_passes": v.get("sdcguard_weight_scan_passes_total",
                                                    v.get("sdcguard_weight_scan_passes")),
                        "weight_scan_never_compared": v.get("sdcguard_weight_scan_never_compared"),
                        "weight_scan_changed": v.get("sdcguard_weight_scan_changed"),
                        "weight_scan_oldest_age_seconds": _finite(v.get("sdcguard_weight_scan_oldest_age_seconds")),
                        "weight_scan_left_out": v.get("sdcguard_weight_scan_left_out"),
                        "weight_scan_target_age_seconds": _finite(v.get("sdcguard_weight_scan_target_age_seconds")),
                        "alerts_counted": _count(v.get("sdcguard_all_alerts_total", v.get("sdcguard_all_alerts"))),
                        "share": info.get("share"), "gpu": info.get("UUID")})
    ratios = [w["weights_checked_ratio"] for w in workers]
    counted = [w["alerts_counted"] for w in workers]
    # checked: on every worker layers are checked, checks ran, and their results were read back at least once (an
    # alert can only be seen at a read)
    return {"workers": workers,
            "checked": bool(workers) and all((w["layers_checked"] or 0) > 0 and (w["checks_executed"] or 0) > 0 and
                                             (w["reads"] or 0) > 0 for w in workers),
            "min_weights_checked": min(ratios) if ratios and None not in ratios else None,
            # every alert the workers counted (each is also one record in SDCGUARD_LOG): None when a worker's metrics
            # don't show it
            "alerts_counted": sum(counted) if counted and None not in counted else None,
            "weight_scan": _scan_coverage(workers)}


def _scan_coverage(workers: list) -> dict:
    """The weight scan over all workers: whether it ran on every one, and compared every entry it holds there."""
    gaps = []
    for i, w in enumerate(workers):
        if not (w.get("weight_scan_layers") or 0) > 0:
            gaps.append(f"worker {i} shows no weight scan")
        elif w.get("weight_scan_never_compared") is None or w.get("weight_scan_oldest_age_seconds") is None:
            gaps.append(f"worker {i}'s metrics don't show what its weight scan compared (an older sdcguard)")
        elif w["weight_scan_never_compared"] > 0:
            gaps.append(f"worker {i}'s weight scan had {w['weight_scan_never_compared']:g} of its entries never "
                        "compared")
    ages = [w.get("weight_scan_oldest_age_seconds") for w in workers]
    # the measured oldest age against the age target where one is set (not the schedule's prediction of it)
    late = []
    for i, w in enumerate(workers):
        age, target = w.get("weight_scan_oldest_age_seconds"), w.get("weight_scan_target_age_seconds")
        if age is not None and target is not None and age > target:
            late.append(f"worker {i}: {age:.0f} s, its target {target:g} s")
    targets = [w for w in workers if w.get("weight_scan_target_age_seconds") is not None]
    return {"ran": bool(workers) and all((w.get("weight_scan_layers") or 0) > 0 for w in workers),
            "complete": bool(workers) and not gaps, "gaps": gaps,
            "oldest_within_target": (not late) if targets else None, "past_target": late,
            "entries": sum(w.get("weight_scan_layers") or 0 for w in workers),
            "changed": sum(w.get("weight_scan_changed") or 0 for w in workers),
            "left_out": sum(w.get("weight_scan_left_out") or 0 for w in workers),
            "oldest_age_seconds": max(ages) if ages and None not in ages else None}


def _gate_source(gate, faults: list, unsure: list, notes: list) -> tuple:
    """`gate compare --json` (case -> verdict): (its summary, why it doesn't count as a source or None). A case counts
    when it has a verdict on a case that ran on both sides; one this doesn't know is unresolved, never clean."""
    cases = gate if isinstance(gate, dict) else {}
    n = {"cases": len(cases), "verdicts": {}, "judged": 0}
    if not isinstance(gate, dict):
        unsure.append(f"the gate report isn't a map of cases to verdicts ({type(gate).__name__})")
    for cid, v in cases.items():
        v = v if isinstance(v, dict) else {}
        vv = v.get("verdict")
        n["verdicts"][str(vv)] = n["verdicts"].get(str(vv), 0) + 1
        if vv in _GATE_JUDGED:
            n["judged"] += 1
        if vv == "REGRESSED" or (vv == "UNCHANGED" and v.get("flavor") != "avoided"):
            faults.append(f"kept case {cid}: {vv} ({str(v.get('why', ''))[:160]})")
        elif vv == "UNCHANGED":
            # right on the candidate, on another kernel: no fault on it here, and the old one isn't shown fixed
            notes.append(f"kept case {cid}: right on the candidate only because another kernel ran (avoided, not "
                         "shown fixed): it can come back if the old kernel is picked again")
        elif vv == "INCONCLUSIVE":
            unsure.append(f"kept case {cid}: INCONCLUSIVE ({str(v.get('why', ''))[:160]})")
        elif vv not in ("PASSED", "FIXED"):
            unsure.append(f"kept case {cid}: verdict {vv!r}, not one the gate gives (PASSED, FIXED, REGRESSED, "
                          "UNCHANGED, INCONCLUSIVE), so it can't count as clean")
    n["counts_as_source"] = n["judged"] > 0
    why = None if n["judged"] else ("the gate report has no case" if not cases else
                                    f"the gate report has no case judged on both stacks ({n['cases']} cases)")
    return n, why


def _kernel_source(kt, faults: list, unsure: list, notes: list) -> tuple:
    """`sdcguard kernels test` (kernels.test_calls): (its summary, why it doesn't count as a source or None). It counts
    when at least one call ran and was judged on the kernel the workload ran (PASSED or WRONG with runs); calls not
    run, or right only on another kernel, test nothing of the workload's. Counts that disagree with the rows, and row
    verdicts this doesn't know, are unresolved."""
    kt = kt if isinstance(kt, dict) else {}
    c = kt.get("counts") if isinstance(kt.get("counts"), dict) else {}
    rows = [r for r in (kt.get("calls") or []) if isinstance(r, dict)]
    seen = dict.fromkeys(_KERNEL_VERDICTS, 0)
    judged = 0
    for row in rows:
        v, runs = row.get("verdict"), row.get("runs")
        ran = isinstance(runs, int) and runs > 0
        if v == "WRONG":
            faults.append(f"kernel test: {row.get('runs_wrong')} of {runs} runs wrong for {row.get('call')}")
        if v not in seen:
            unsure.append(f"kernel test: verdict {v!r} for {row.get('call')}, not one the kernel test gives "
                          f"({', '.join(_KERNEL_VERDICTS)}), so it can't count as clean")
            continue
        seen[v] += 1
        if v in ("PASSED", "WRONG") and not ran:
            unsure.append(f"kernel test: {row.get('call')} is {v} with no run recorded")
        elif v == "PASSED" and row.get("runs_wrong"):
            unsure.append(f"kernel test: {row.get('call')} is PASSED with {row.get('runs_wrong')} runs wrong")
        elif v in ("PASSED", "WRONG"):
            judged += 1
    want = {**seen, "NOT RUN": seen["NOT RUN"] + len(kt.get("not_run") or [])}
    off = [f"{k} {c.get(k, 0)} counted, {want.get(k, 0)} in its rows" for k in sorted(set(want) | set(c))
           if c.get(k, 0) != want.get(k, 0)]
    if off:
        unsure.append("kernel test: its counts disagree with its rows (" + "; ".join(off) + ")")
    if seen["OTHER KERNEL"]:
        notes.append(f"kernel test: {seen['OTHER KERNEL']} calls were right here only on another kernel than the "
                     "workload ran (other conditions: run it where the workload runs): the workload's kernel wasn't "
                     "tested for them")
    if kt.get("warning"):
        unsure.append("kernel test: " + kt["warning"])
    n = {"counts": c, "gpu": kt.get("gpu"), "judged": judged, "counts_as_source": judged > 0}
    why = None if judged else (f"the kernel test ran and judged no call on the workload's kernel ({want['PASSED']} "
                               f"passed, {want['WRONG']} wrong, {want['OTHER KERNEL']} on another kernel, "
                               f"{want['NOT RUN']} not run)")
    return n, why


def arithmetic(alerts: Optional[list] = None, gate: Optional[dict] = None, kernel_test: Optional[dict] = None,
               plugin: Optional[dict] = None, min_weights_checked: Optional[float] = 0.5) -> dict:
    """plugin: plugin_coverage() of the candidate's run. Alerts count as a source of 'no fault' only when the
    coverage shows checks ran on every worker, the gate's cases only when one was judged, the kernel test only when
    one call ran and was judged; a confirmed wrong output is a fault whatever the rest shows."""
    faults, unsure, notes, sources, weights, empty = [], [], [], {}, [], []
    if alerts is not None:
        n = {"alerts": len(alerts), "confirmed_wrong": 0, "weights_changed": 0, "cleared": 0, "unconfirmed": 0,
             "unreadable": 0, "coverage": plugin}
        records = len(alerts)
        for ev in alerts:
            if ev.get("unreadable"):                # a line of the alert file that isn't a record (_read_jsonl)
                n["unreadable"] += 1
                n["alerts"] -= 1
                continue
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
                faults.append(f"the plugin's alert on {where}{at}: its recheck on the CPU says {rv}")
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
        # the alerts given must be the alerts the workers counted: missing or unreadable records (an alert file cut
        # short, lost, or from another run) are evidence that isn't there, never a clean one
        counted = plugin.get("alerts_counted") if plugin is not None else None
        n["records_vs_counted"] = [records - n["unreadable"], counted]
        gap = None
        if n["unreadable"]:
            gap = f"{n['unreadable']} lines of the alert file aren't readable records"
        elif plugin is not None and counted is None:
            gap = ("a worker's metrics don't show how many alerts it counted (sdcguard_all_alerts_total: an older "
                   "plugin, or metrics cut short), so the alert records can't be checked against them")
        elif counted is not None and counted > records:
            gap = f"the workers counted {counted:g} alerts but only {records} alert records were given (an alert file " \
                  "cut short or lost)"
        elif counted is not None and counted < records:
            # more records than counted: alerts reported after the metrics were read (the plugin reads once more as
            # the server stops), or a file that also holds another run's; none is missing, and each is judged above
            notes.append(f"{records} alert records for {counted:g} counted in the workers' metrics (read before the "
                         "server stopped)")
        if plugin is None or not plugin.get("checked") or low or gap:
            why = ("its coverage wasn't given (--plugin-metrics)" if plugin is None else
                   "its metrics show a worker with no checked layer, no check run, or no result read back"
                   if not plugin.get("checked") else gap if gap else
                   f"it checks {plugin.get('min_weights_checked') or 0:.0%} of the linear-layer weights on some "
                   f"worker, under the policy's {min_weights_checked:.0%}")
            if not n["confirmed_wrong"]:
                unsure.append(f"the plugin's {len(alerts)} alerts can't stand for no fault: {why}")
            n["counts_as_source"] = False
        else:
            n["counts_as_source"] = True
        sources["plugin_alerts"] = n
    if gate is not None:
        sources["gate_cases"], why = _gate_source(gate, faults, unsure, notes)
        if why:
            empty.append(why + ": it tested nothing, so it isn't a source")
    if kernel_test is not None:
        sources["kernel_test"], why = _kernel_source(kernel_test, faults, unsure, notes)
        if why:
            empty.append(why + ": it isn't a source")
    counting = [k for k, src in sources.items() if src.get("counts_as_source")]
    if faults:
        v = "FAULT"
        notes += empty
    elif not sources:
        v, unsure = "INCONCLUSIVE", ["no arithmetic source was given"]
    elif not counting:
        v, unsure = "INCONCLUSIVE", empty + unsure
    elif unsure:
        v = "INCONCLUSIVE"
        notes += empty
    else:
        v = "NO FAULT OBSERVED"
        notes += empty
    return {"verdict": v, "faults": faults, "unresolved": unsure, "notes": notes, "sources": sources,
            "weights_changed": weights}


def weights_section(a: dict, plugin: Optional[dict], alerts_given: bool = True) -> dict:
    """Weight integrity: do the model's weights still match what was loaded?
      CHANGED        an alert showed a layer's weights changed since load (a corrupted or overwritten weight,
                     whatever the arithmetic did with it), or the weight scan's metrics count a mismatch
      VERIFIED       fingerprints verified: the scheduled weight scan (SDCGUARD_WEIGHT_SCAN=1) ran on every worker,
                     compared every entry it holds with its fingerprint at load and found no mismatch; with how long
                     before the metrics were read the least recently verified entry was found clean, and how many
                     tensors it leaves out (each worker's coverage report lists them)
      NO CHANGE SEEN only the arithmetic checks ran (or the scan didn't cover every entry on every worker) and no alert
                     showed a change: their checksums see a changed weight only once it moves an output past the
                     line, so this verifies no weight
      UNMEASURED     no plugin coverage, or no alerts given."""
    changed = a.get("weights_changed") or []
    scan = (plugin or {}).get("weight_scan") or {}
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
    if scan.get("changed"):
        return {"verdict": "CHANGED", "layers": [], "alerts": 0,
                "why": f"the weight scan's metrics count {scan['changed']:g} entries whose tensors no longer match "
                       "their fingerprint at load (their alerts aren't in the alerts given)"}
    if plugin is None:
        return {"verdict": "UNMEASURED", "why": "no plugin coverage was given"}
    if not alerts_given:
        return {"verdict": "UNMEASURED", "why": "no alerts were given"}
    if scan.get("complete"):
        age = scan.get("oldest_age_seconds")
        late = scan.get("past_target") or []
        return {"verdict": "VERIFIED", "entries": scan.get("entries"), "left_out": scan.get("left_out"),
                "oldest_verification_age_seconds": age, "oldest_within_target": scan.get("oldest_within_target"),
                "why": f"fingerprints verified: the weight scan compared every one of its {scan.get('entries'):g} "
                       "entries on every worker with its fingerprint at load and found no mismatch; the least "
                       f"recently verified was found clean {age:.0f} s before the metrics were read"
                       + (f", older than the verification-age target ({'; '.join(late)})" if late else "")
                       + f"; it leaves out {scan.get('left_out') or 0:g} tensors (listed in each worker's coverage "
                       "report)"}
    if plugin.get("checked"):
        how = ("the scheduled weight scan (SDCGUARD_WEIGHT_SCAN=1) didn't run" if not any(
            (w.get("weight_scan_layers") or 0) > 0 for w in plugin.get("workers") or []) else
            "the weight scan didn't verify every entry: " + "; ".join(scan.get("gaps") or []))
        return {"verdict": "NO CHANGE SEEN", "why": "no alert showed a changed weight, but the weights were not "
                f"verified: {how}, and the arithmetic checks' checksums see a changed weight only once it moves an "
                "output past their line"}
    return {"verdict": "UNMEASURED", "why": "the plugin's metrics show a worker with no checked layer or no check run, "
            "and no weight scan that verified every entry"}


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
            tested = {str(r.get("call", "")).split(" | ")[0] for r in kernel_test.get("calls") or []
                      if isinstance(r, dict) and r.get("runs")}
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
    wt = weights_section(a, plugin, alerts is not None)
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
        lines.append(f"1a. Weight integrity (the model's weights against what was loaded): {wt.get('verdict')}")
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
    """The records of a JSONL file; a line that isn't a JSON object is kept as {"unreadable": its line number}, so
    a cut-short or corrupted alert file shows instead of reading as fewer alerts."""
    out = []
    with open(path) as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                except ValueError:
                    rec = None
                out.append(rec if isinstance(rec, dict) else {"unreadable": k})
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
