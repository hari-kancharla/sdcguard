"""An incident casefile: what one evidence file and its replays support, and what they don't.

An alert says a result failed its check. It doesn't say why. The same symptom can come from a GPU that computed
wrong once, a GPU that keeps computing wrong, a library or configuration defect (cuBLAS known issue 6862950 made
healthy H200s compute wrong under MPS), a software bug that reads memory it shouldn't (AEGIS, OSDI '26, section 8,
caught one), or the check itself. A casefile collects the evidence that separates those:

  - the recorded output against the CPU: is it outside the rounding envelope of a float64 recomputation of the
    same inputs (screen.cpu_recheck), or did the CPU not reproduce the mismatch?
  - replays on the same GPU (same UUID): does the wrong result come back from the same inputs?
  - replays on peer GPUs: does it come back elsewhere, and did the peer run the same software?

and scores each explanation as supported, contradicted or not tested, from that evidence only. A replay counts
only if it replayed this exact file (its SHA-256), once, and ran the way the incident did: the same versions,
MPS share, visible SMs, cuBLAS environment and PyTorch matmul flags (`sdcguard replay` applies the recorded flags;
the MPS share and the environment are the operator's to match). Anything unrecorded is unknown, never a match.
The incident's setup is also matched against NVIDIA's listed cuBLAS wrong-result issues, as leads. The outcome is
"most consistent with" one explanation when exactly one is supported and none of the evidence contradicts it, and
INCONCLUSIVE otherwise, with the replays that would settle it. It never says a cause is proven: a transient
hardware fault and a rare software race look the same to replays, which is why the alternatives stay listed.
(Megatron Core's rerun state machine sorts a loss or gradient-norm anomaly into correct, transient or persistent
by rerunning the step on the same GPU and then another; this does the same for one saved matmul, with a CPU
reference and the software stack of every run, outside any one training framework.)

    sdcguard casefile evidence_0.pt --replay-here 200                 # replay here (the same GPU or a peer)
    sdcguard casefile evidence_0.pt --replays peer_replay.json ...    # add replays run elsewhere
    sdcguard replay evidence_0.pt --save peer_replay.json            # on a peer GPU, to bring back
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

SUPPORTED, CONTRADICTED, NOT_TESTED = "supported", "contradicted", "not tested"
INCONCLUSIVE = "INCONCLUSIVE"
STACK_KEYS = ("torch", "cuda", "cublas", "driver", "gpu_name")
WEAK_REPLAYS = 20                 # fewer recomputations than this on one side make its result weak evidence

EXPLANATIONS = {
    "transient": "a transient fault on the original GPU (computed wrong once, right when replayed)",
    "recurring": "a recurring fault on the original GPU (wrong again there, right on a peer with the same software)",
    "software": "a software, library or configuration defect (wrong on a different GPU too)",
    "check": "the check, not the output (the CPU finds the recorded output within its rounding envelope)",
}


def software_stack(device=None) -> dict:
    """What a replay ran with: library versions, the GPU model, its driver, and settings that change GEMMs."""
    import torch
    from .check import software
    sw = dict(software())
    ident = {}
    try:
        from .events import device_identity
        ident = device_identity(device)
        if device is not None and getattr(device, "type", None) == "cuda":
            from .screen import _nvidia_smi
            ident.update(_nvidia_smi(ident.get("gpu_uuid") or ident.get("gpu_index")))
    except Exception:  # noqa: BLE001
        pass
    sw["gpu_name"] = ident.get("gpu_name") or ("cpu" if device is None or getattr(device, "type", None) == "cpu"
                                               else None)
    sw["driver"] = ident.get("driver")
    from .screen import execution_settings
    sw.update(execution_settings(device))
    return sw


EXECUTION_KEYS = ("environment", "sms_visible", "matmul")
# compared when both sides recorded them (older files don't): the cuBLASLt actually loaded, which can differ from
# the installed package
OPTIONAL_KEYS = ("cublaslt_loaded",)


def _settings_differ(a: dict, b: dict) -> Optional[list]:
    """The execution settings two runs recorded differently ([] when they match), None when either side didn't
    record one of them (an older file, or a field that couldn't be read): unknown is never a match, because two
    runs that both lack the MPS share may still have run with different shares. PyTorch's matmul flags are
    compared by what they do (check.gemm_effective), so the same TF32 setting made through the old and the new
    API matches."""
    from .check import gemm_effective
    out = []
    for k in EXECUTION_KEYS:
        if a.get(k) is None or b.get(k) is None:
            return None
        if k == "matmul":
            ea, eb = gemm_effective(a[k]), gemm_effective(b[k])
            if ea is None or eb is None or set(ea) != set(eb):
                return None
            out += [f"matmul {n} {ea[n]} vs {eb[n]}" for n in sorted(ea) if ea[n] != eb[n]]
        elif a[k] != b[k]:
            out.append(f"{k} {a[k]} vs {b[k]}")
    for k in OPTIONAL_KEYS:
        if a.get(k) is not None and b.get(k) is not None and str(a[k]) != str(b[k]):
            out.append(f"{k} {a[k]} vs {b[k]}")
    return out


def same_stack(a: Optional[dict], b: Optional[dict]) -> Optional[bool]:
    """Whether a replay ran the same software in the same way as the incident: the versions in STACK_KEYS and the
    execution settings (MPS share, visible SMs, cuBLAS environment, PyTorch's matmul flags). False when anything
    recorded differs, None when either side didn't record something (unknown: not a matched control either)."""
    if not a or not b or any(a.get(k) is None or b.get(k) is None for k in STACK_KEYS):
        return None
    if not all(str(a[k]) == str(b[k]) for k in STACK_KEYS):
        return False
    diff = _settings_differ(a, b)
    if diff is None:
        return None
    return not diff


def _differences(a: Optional[dict], b: Optional[dict]) -> list:
    """What two recorded stacks differ in, in words (versions first, then settings); [] when nothing recorded
    differs."""
    a, b = a or {}, b or {}
    out = [f"{k} {a.get(k)} vs {b.get(k)}" for k in STACK_KEYS
           if a.get(k) is not None and b.get(k) is not None and str(a[k]) != str(b[k])]
    return out + (_settings_differ(a, b) or [])


def _unknown(a: Optional[dict], b: Optional[dict]) -> list:
    """What one side didn't record (so the two can't be shown to match)."""
    a, b = a or {}, b or {}
    out = [k for k in STACK_KEYS + EXECUTION_KEYS if a.get(k) is None or b.get(k) is None]
    from .check import gemm_effective
    if "matmul" not in out:
        ea, eb = gemm_effective(a.get("matmul")), gemm_effective(b.get("matmul"))
        if ea is None or eb is None or set(ea) != set(eb):
            out.append("matmul")
    return out


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def load_incident(path: str) -> dict:
    """The evidence file's facts: where it was recorded, with what, and what the CPU says about its output."""
    info_path = path[:-3] + ".json" if path.endswith(".pt") else path + ".json"
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    digest = _sha256(path)
    stale = None
    if info.get("evidence_sha256") and info["evidence_sha256"] != digest:
        # the .json beside it describes another .pt (a run stopped between the two writes, a copied file): its
        # facts aren't this file's, so everything is judged again from the .pt
        stale, info = info_path, {}
    gpu = info.get("gpu") or {}
    sw = info.get("software") or {}                      # recorded with the evidence since Oct 20, 2026
    inc = {"evidence": path, "evidence_sha256": digest, "info": info_path if info else None,
           "info_ignored": stale, "info_data": {"in_job_replays": info.get("in_job_replays")},
           "what": info.get("what"), "op": info.get("op") or info.get("mode"), "where": info.get("where"),
           "max_z": info.get("max_z", info.get("worst_z")), "gpu_uuid": gpu.get("gpu_uuid"),
           "host": gpu.get("host"), "device": gpu.get("device"),
           "stack": {"torch": sw.get("torch") or info.get("torch"), "cuda": sw.get("cuda"), "cublas": sw.get("cublas"),
                     "sdcguard": sw.get("sdcguard") or info.get("sdcguard"), "driver": gpu.get("driver"),
                     "gpu_name": gpu.get("gpu_name") or ("cpu" if str(gpu.get("device") or "").startswith("cpu")
                                                         else None),
                     **{k: sw[k] for k in EXECUTION_KEYS + OPTIONAL_KEYS + ("compute_capability",) if k in sw}}}
    cpu = info.get("cpu_recheck")
    try:
        import torch
        ev = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(ev, dict) and ev.get("kind") == "math_mode":
            # judged again from the .pt (the .json may be missing): against the CPU's float64 answer. Only a
            # result these inputs can't produce (non-finite, beyond Cauchy-Schwarz) or a wrong INT8 result is
            # wrong for certain; a score above the alarm line is a statistical mismatch
            from .modes import judge
            inc["math_mode"] = ev.get("mode")
            r = judge(ev["mode"], ev["A"].double(), ev["B"].double(), ev["C"], ev.get("nfma"), ev.get("signs"),
                      ev.get("unit"))
            wrong = bool(r["wrong_blocks"])
            certain = wrong and (ev["mode"] == "int8" or r["worst_z"] == float("inf"))
            cpu = {"done": True, "judged": True, "output_wrong": wrong, "output_mismatch_confirmed": certain,
                   "wrong_blocks": r["wrong_blocks"], "worst_z": r["worst_z"],
                   "reading": "math-mode evidence judged again against the CPU's float64 answer"}
        elif cpu is None and isinstance(ev, dict) and ev.get("row") is not None:
            from .screen import cpu_recheck
            cfg = ev.get("check_config") or {}
            cpu = cpu_recheck(ev["A"], ev["B"], ev["C"], ev.get("bias"), ev["alpha"], ev["beta"], ev["row"],
                              ev["col_block"], ev.get("batch"), ev.get("epoch") or 0, block=cfg.get("block") or 128,
                              seed=ev.get("sign_seed"), z=cfg.get("z") or 6.0, abs_floor=cfg.get("abs_floor"),
                              nfma=cfg.get("nfma"))
    except Exception as e:  # noqa: BLE001
        cpu = cpu or {"done": False, "why": f"{type(e).__name__}: {str(e)[:200]}"}
    inc["cpu"] = cpu or {"done": False, "why": "no CPU judgment recorded"}
    if info.get("captured"):            # serving evidence: from a second pass, not the output that alarmed
        inc["recaptured"] = info["captured"]
        inc["weights_changed_since_load"] = info.get("weights_changed_since_load")
    elif info.get("captured_at_check"):  # serving evidence the check kept when it alarmed: the original numbers
        inc["weights_changed_since_load"] = info.get("weights_changed_since_load")
        inc["weights_rewritten_since_load"] = info.get("weights_rewritten_since_load")
    return inc


def cpu_state(cpu: dict) -> str:
    """'outside envelope' (the recorded output is wrong beyond rounding), 'statistical' (the CPU checksum flags it
    within the envelope), 'not reproduced' (the CPU finds nothing), or 'unknown'."""
    if not cpu or not cpu.get("done") or not cpu.get("judged", True):
        return "unknown"
    if cpu.get("output_mismatch_confirmed"):
        return "outside envelope"
    if cpu.get("output_wrong"):
        return "statistical"
    return "not reproduced"


def classify_replay(inc: dict, rep: dict) -> str:
    """'same' (the original GPU), 'peer' (another GPU), 'cpu', or 'unknown' (not enough identity to tell)."""
    on = rep.get("replayed_on") or {}
    dev = str(on.get("device") or "")
    if dev.startswith("cpu"):
        return "cpu"
    a, b = inc.get("gpu_uuid"), on.get("gpu_uuid")
    if a and b:
        return "same" if a == b else "peer"
    return "unknown"


def _judged(rep: dict) -> int:
    return max(0, int(rep.get("recomputations") or 0) - int(rep.get("recomputations_that_couldnt_be_judged") or 0))


def _fails(rep: dict) -> int:
    return int(rep.get("recomputations_that_failed_the_check") or 0)


def rate_interval(failed: int, n: int, confidence: float = 0.95) -> Optional[list]:
    """Clopper-Pearson interval for how often this error comes back, from `failed` of `n` recomputations: exact,
    and conservative. With none of 200 wrong, the rate is below 1.8% (95%); a GPU whose error recurs once in a
    million recomputations (SDCHunter, OSDI '26, saw rates that low) passes far more replays than anyone runs."""
    if n <= 0:
        return None
    import math
    a = (1 - confidence) / 2

    def cdf(k, p):                     # P(X <= k), X ~ Binomial(n, p)
        if p <= 0:
            return 1.0
        if p >= 1:
            return 1.0 if k >= n else 0.0
        lp, lq = math.log(p), math.log1p(-p)
        return min(1.0, sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                                     + i * lp + (n - i) * lq) for i in range(k + 1)))

    def solve(f, target):              # f decreasing in p on [0, 1]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) > target else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if failed == 0 else solve(lambda p: cdf(failed - 1, p), 1 - a)
    upper = 1.0 if failed >= n else solve(lambda p: cdf(failed, p), a)
    return [round(lower, 6), round(upper, 6)]


def _replay_key(r: dict) -> str:
    """One replay result's identity: its replay id, else (an older result) its contents."""
    rid = r.get("replay_id")
    if rid:
        return f"id:{rid}"
    return "json:" + hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()


def _bind(inc: dict, replays: list) -> tuple:
    """(the replays of this evidence file, each counted once; how many were of another file or didn't say which;
    how many were the same result passed again)."""
    expected = inc.get("evidence_sha256")
    bound, seen, other, dup = [], set(), 0, 0
    for r in replays:
        if not isinstance(r, dict):
            other += 1
            continue
        if not expected or r.get("evidence_sha256") != expected:
            other += 1          # a replay of another .pt, or one that doesn't say which (sdcguard before 0.4.0)
            continue
        key = _replay_key(r)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        bound.append(r)
    return bound, other, dup


def known_issue_leads(stack: Optional[dict]) -> list:
    """NVIDIA's listed cuBLAS wrong-result issues that the incident's recorded setup matches (known_issues.py):
    leads to test with a replay, not evidence of a cause."""
    stack = stack or {}
    try:
        from . import known_issues
        cc = stack.get("compute_capability")
        share = (stack.get("environment") or {}).get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
        try:
            share = float(share) if share is not None else None
        except ValueError:
            share = None
        env = {"cublas": stack.get("cublaslt_loaded") or stack.get("cublas"), "gpu": stack.get("gpu_name"),
               "cc": f"{cc[0]}.{cc[1]}" if isinstance(cc, (list, tuple)) and len(cc) == 2 else None,
               "sms": stack.get("sms_visible"), "mps_share": share}
        env["sms_full"], env["mps"] = known_issues.mps_state(env["gpu"], env["sms"], share)
        return [{"id": m["id"], "status": m["status"], "what": m["what"], "workaround": m.get("workaround"),
                 "why": m["why"]} for m in known_issues.match(env) if m["status"] in ("applies", "may apply")]
    except Exception:  # noqa: BLE001  (a lead, never in the way of the casefile)
        return []


def assess(inc: dict, replays: list) -> dict:
    """Score every explanation against the evidence (pure: no GPU, no files).

    Only replays of this evidence file count (their evidence SHA-256 matches), each once. A replay on the original
    GPU counts toward "transient" or "recurring" only when it ran the incident's software the same way (versions,
    MPS share, visible SMs, cuBLAS environment, PyTorch's matmul flags): a clean replay at another MPS share says
    nothing about a fault that needs the original share, as cuBLAS known issue 6862950 does. A wrong replay on the
    original GPU contradicts "transient" whatever it ran with. A peer is a matched control only when it ran the
    incident's software the same way, and with the sdcguard version of the original GPU's replays."""
    state = cpu_state(inc.get("cpu") or {})
    recaptured = bool(inc.get("recaptured"))
    if recaptured and state != "outside envelope":
        # the saved output came from one more pass after the alert: a clean one says nothing about the output
        # that alarmed (a transient fault isn't in it; a corrupted weight makes it agree with that weight)
        state = "unknown"
    unjudged = "the CPU couldn't judge the recorded output"
    if not recaptured and (inc.get("weights_changed_since_load") or inc.get("weights_rewritten_since_load")):
        # serving evidence kept when the check alarmed holds the weights as they were when the alert was read:
        # changed or written in place since load, they may not be the ones the matmul used, so the CPU's judgment
        # of the output says nothing about its cause (a corrupted weight makes the output agree with it; one put
        # back makes a right output look wrong)
        state = "unknown"
        unjudged = ("the weights this evidence holds changed or were written in place since load, so they may not "
                    "be the ones the matmul used: the CPU's judgment of the output can't be taken")
    stack = inc.get("stack") or {}
    bound, other_file, duplicates = _bind(inc, replays)
    groups = {"same": [], "peer": [], "cpu": [], "unknown": []}
    for r in bound:
        groups[classify_replay(inc, r)].append(r)
    same_m = [r for r in groups["same"] if same_stack(stack, r.get("software")) is True]
    same_u = [r for r in groups["same"] if same_stack(stack, r.get("software")) is not True]
    same_n, same_f = sum(_judged(r) for r in same_m), sum(_fails(r) for r in same_m)
    other_n, other_f = sum(_judged(r) for r in same_u), sum(_fails(r) for r in same_u)
    all_same_f = same_f + other_f
    ref = {(r.get("software") or {}).get("sdcguard") for r in same_m}

    def peer_row(r):
        sw = r.get("software") or {}
        st = same_stack(stack, sw)
        ver = sw.get("sdcguard")
        if st is True and ref and ver not in ref:       # replayed by other sdcguard code than the original GPU's
            st = None if ver is None else False
        diff = _differences(stack, sw)
        if st is not True and ref and ver is not None and ver not in ref:
            diff = diff + [f"sdcguard {ver} vs {', '.join(sorted(str(v) for v in ref))} on the original GPU"]
        return {"gpu_uuid": (r.get("replayed_on") or {}).get("gpu_uuid"), "judged": _judged(r), "failed": _fails(r),
                "same_stack": st, "settings_differ": diff or None,
                "not_recorded": None if st is not None else (_unknown(stack, sw) or ["sdcguard"]),
                "same_model": (sw.get("gpu_name") == stack.get("gpu_name")) if stack.get("gpu_name") else None}
    peers = [peer_row(r) for r in groups["peer"]]
    peer_clean_same_stack = [p for p in peers if p["judged"] and not p["failed"] and p["same_stack"] is True]
    # a peer that ran other software or settings and failed shows its own setup computes this wrong, not why the
    # incident did (a peer under an MPS share the incident didn't have can fail from cuBLAS 6862950 alone)
    peer_failed = [p for p in peers if p["failed"] and p["same_stack"] is True]
    peer_failed_other = [p for p in peers if p["failed"] and p["same_stack"] is not True]
    matched_peer_reps = [r for r, p in zip(groups["peer"], peers) if p["same_stack"] is True]
    # Only an output outside the CPU's rounding envelope is shown wrong. A statistical mismatch (the CPU's checksum,
    # with the alert's sign patterns, flags it too, but every element is within rounding) is either a small fault
    # or the noise model misjudging these inputs; replays judged with the same patterns separate the two.
    wrong_recorded = state == "outside envelope"
    statistical = state == "statistical"
    same_signs = bool(same_m) and all(r.get("sign_patterns") == "recorded" for r in same_m)
    peer_signs = bool(matched_peer_reps) and all(r.get("sign_patterns") == "recorded" for r in matched_peer_reps)
    unmatched_rows = [{"judged": _judged(r), "failed": _fails(r),
                       "settings_differ": _differences(stack, r.get("software")) or None,
                       "not_recorded": None if same_stack(stack, r.get("software")) is False
                       else _unknown(stack, r.get("software"))} for r in same_u]
    for p in peers:
        p["error_rate_95"] = rate_interval(p["failed"], p["judged"])
    obs = {"recorded output vs CPU": state,
           "same-GPU replays": {"runs": len(same_m), "judged": same_n, "failed": same_f,
                                "error_rate_95": rate_interval(same_f, same_n)},
           "same-GPU replays with other software or settings (not counted)": unmatched_rows,
           "peer-GPU replays": peers,
           "replays that couldn't be placed": len(groups["unknown"]),
           "replays of another evidence file, or not saying which (not counted)": other_file,
           "the same replay result passed more than once (counted once)": duplicates}
    hyp = {}
    other_note = (f" ({len(same_u)} replay(s) there ran with other software or settings and don't count)"
                  if same_u else "")
    # transient: wrong when recorded, right when replayed on the same GPU the same way
    if all_same_f:
        hyp["transient"] = (CONTRADICTED, [f"{all_same_f} of {same_n + other_n} replays on the original GPU were "
                                           "wrong again"])
    elif not same_m or same_n == 0:
        hyp["transient"] = (NOT_TESTED, ["no judged replay on the original GPU with the incident's software and "
                                         "settings" + other_note])
    elif state == "not reproduced":
        hyp["transient"] = (CONTRADICTED, ["the CPU finds the recorded output within its rounding envelope"])
    elif wrong_recorded:
        hyp["transient"] = (SUPPORTED, [f"recorded output {state}; {same_n} judged replays on the same GPU, with "
                                        "the incident's software and settings, were right"])
    elif statistical and same_signs:
        hyp["transient"] = (SUPPORTED, [f"the recorded output fails the checksum and {same_n} judged replays on the "
                                        "same GPU, with the same sign patterns, software and settings, pass: the "
                                        "GPU computes something else now. The error is within the rounding "
                                        "envelope, so it was small"])
    elif statistical:
        hyp["transient"] = (NOT_TESTED, ["a statistical mismatch, and the replays were judged with other sign "
                                         "patterns than the alert, so a false alarm tied to those patterns would "
                                         "not have come back"])
    else:
        hyp["transient"] = (NOT_TESTED, [unjudged])
    # recurring: wrong again on the original GPU, right on a peer with the same software, both run the same way
    if not same_m or same_n == 0:
        hyp["recurring"] = (NOT_TESTED, ["no judged replay on the original GPU with the incident's software and "
                                         "settings" + (f" ({other_f} of {other_n} replays under other settings "
                                                       "were wrong)" if other_f else other_note)])
    elif not same_f:
        hyp["recurring"] = (CONTRADICTED, [f"all {same_n} judged replays on the original GPU were right"])
    elif peer_failed:
        hyp["recurring"] = (CONTRADICTED, [f"wrong on {len(peer_failed)} peer GPU(s) too"])
    elif peer_clean_same_stack and (not statistical or peer_signs):
        hyp["recurring"] = (SUPPORTED, [f"{same_f} of {same_n} replays wrong on the original GPU; right on "
                                        f"{len(peer_clean_same_stack)} peer(s) with the same software and "
                                        "settings"])
    else:
        hyp["recurring"] = (NOT_TESTED, [f"{same_f} of {same_n} replays wrong on the original GPU, but no peer with "
                                         "the same recorded software and settings was replayed"
                                         + (" with the alert's sign patterns" if peer_clean_same_stack else "")])
    # software / library / configuration: wrong on a different GPU too
    if peer_failed and statistical:
        hyp["software"] = (NOT_TESTED, ["a peer fails the same checksum, but every element of the recorded output is "
                                        "within the CPU's rounding envelope: that points at the check, not at a "
                                        "wrong result"])
    elif peer_failed:
        hyp["software"] = (SUPPORTED, [f"wrong on {len(peer_failed)} peer GPU(s): it follows the inputs and the "
                                       "software, not one chip"])
    elif peer_clean_same_stack and same_f:
        hyp["software"] = (CONTRADICTED, ["a peer with the same software and settings computed it right"])
    elif peer_failed_other:
        hyp["software"] = (NOT_TESTED, [f"{len(peer_failed_other)} peer(s) that ran other software or settings were "
                                        "wrong: that says their setup computes this wrong, not why the incident "
                                        "did; a wrong replay on a peer run the incident's way would"])
    else:
        hyp["software"] = (NOT_TESTED, ["no peer replay that was wrong, and no same-software peer to compare"])
    # the check: the output isn't shown wrong
    if state == "not reproduced":
        notes = ["the CPU finds the recorded output within its rounding envelope"]
        if all_same_f:
            hyp["check"] = (CONTRADICTED, notes + [f"but replays on the original GPU failed {all_same_f} times"])
        else:
            hyp["check"] = (SUPPORTED, notes + ["the alert may come from the check's own GPU arithmetic (on Oct 19, "
                                                "2026 an H200 under the cuBLAS MPS bug alerted this way) or a checker "
                                                "error; the output itself was not shown wrong"])
    elif wrong_recorded:
        hyp["check"] = (CONTRADICTED, [f"the CPU finds the recorded output {state}"])
    elif statistical:
        notes = ["every element of the recorded output is within the CPU's rounding envelope (a statistical "
                 "mismatch)"]
        if same_signs and same_n and not same_f:
            hyp["check"] = (CONTRADICTED, notes + ["but replays on the original GPU with the same sign patterns, "
                                                   "software and settings pass, so the recorded output differs from "
                                                   "what it computes now"])
        elif peer_failed and peer_signs:
            hyp["check"] = (SUPPORTED, notes + ["and the same checksum flags the recomputation on another GPU too: "
                                                "consistent with the noise model misjudging these inputs (a false "
                                                "alarm), not a wrong result"])
        elif peer_clean_same_stack and peer_signs and same_f:
            hyp["check"] = (CONTRADICTED, notes + ["but a peer with the same software passes the same checksum"])
        else:
            hyp["check"] = (NOT_TESTED, notes + ["a replay on a peer GPU with the alert's sign patterns would "
                                                 "separate a small fault from a false alarm"])
    else:
        hyp["check"] = (NOT_TESTED, [unjudged])
    supported = [k for k, (s, _) in hyp.items() if s == SUPPORTED]
    weak = []
    if same_m and same_n < WEAK_REPLAYS:
        weak.append(f"only {same_n} judged recomputations on the original GPU")
    for p in peers:
        if p["judged"] and p["judged"] < WEAK_REPLAYS:
            weak.append(f"only {p['judged']} judged recomputations on peer {p['gpu_uuid']}")
    # a clean result rests on how many recomputations were judged: a handful of clean replays can miss an
    # intermittent fault, so they don't decide the outcome on their own
    thin = None
    if supported == ["transient"] and same_n < WEAK_REPLAYS:
        thin = f"the original GPU ({same_n} judged)"
    elif supported == ["recurring"] and all(p["judged"] < WEAK_REPLAYS for p in peer_clean_same_stack):
        thin = "the peer GPU (" + ", ".join(str(p["judged"]) for p in peer_clean_same_stack) + " judged)"
    if len(supported) == 1 and not thin:
        outcome = f"MOST CONSISTENT WITH: {EXPLANATIONS[supported[0]]}"
    else:
        outcome = INCONCLUSIVE
    nxt = []
    ev = inc.get("evidence") or "<evidence.pt>"
    if inc.get("info_ignored"):
        nxt.append(f"{inc['info_ignored']} describes another evidence file (its SHA-256 doesn't match), so it was "
                   "ignored and the evidence judged again from the .pt alone")
    if not inc.get("evidence_sha256"):
        nxt.append(f"the evidence file {ev} couldn't be read, so no replay can be tied to it")
    elif other_file:
        nxt.append(f"{other_file} replay result(s) are of another evidence file, or don't record which file they "
                   f"replayed (sdcguard before 0.4.0): replay {ev} itself with sdcguard 0.4.0 or later")
    if thin:
        nxt.append(f"the clean replays on {thin} are too few to settle it: replay at least {WEAK_REPLAYS} times "
                   f"(sdcguard replay {ev} --times 200)")
    if same_u and not same_m:
        why = "; ".join(sorted({d for row in unmatched_rows for d in (row["settings_differ"] or [])})) or \
            "not recorded: " + ", ".join(sorted({d for row in unmatched_rows for d in (row["not_recorded"] or [])}))
        nxt.append(f"the replays on the original GPU ran with other software or settings ({why}): replay there "
                   f"the way the incident ran (the same MPS share and cuBLAS environment variables; sdcguard replay "
                   f"applies the recorded PyTorch matmul flags itself): sdcguard casefile {ev} --replay-here 200")
    elif not groups["same"]:
        nxt.append(f"replay on the original GPU ({inc.get('gpu_uuid') or 'its UUID is in the .json'}): "
                   f"sdcguard casefile {ev} --replay-here 200")
    if not peer_clean_same_stack and not peer_failed:
        nxt.append(f"replay on another GPU of the same model with the same driver, CUDA, cuBLAS, PyTorch and "
                   f"settings: sdcguard replay {ev} --times 200 --save peer.json, then add --replays peer.json")
    for p in peers:
        if p.get("same_stack") is False and p.get("settings_differ"):
            nxt.append(f"peer {p['gpu_uuid']} ran with other software or settings "
                       f"({'; '.join(p['settings_differ'])}): it isn't a matched control. Replay it the way the "
                       "original ran (an MPS share below 100% changes which cuBLAS kernels run, as in known issue "
                       "6862950), or replay the original with the peer's")
        elif p.get("same_stack") is None and p.get("not_recorded"):
            nxt.append(f"peer {p['gpu_uuid']}'s replay doesn't record {', '.join(p['not_recorded'])}, so it can't "
                       "be shown to run the same way: replay it with this sdcguard version")
    if recaptured:
        nxt.append("this evidence was captured from a second pass after the alert (serving), not the output that "
                   "alarmed: a clean recapture doesn't clear the GPU, and only replays can say more")
        if inc.get("weights_changed_since_load"):
            nxt.append("the serving guard found the layer's weights changed since load: GPU memory corruption, or "
                       "an update without refresh(). The recapture computed with the changed weights")
    elif inc.get("weights_changed_since_load"):
        nxt.append("the serving guard found the layer's weights changed since load: GPU memory corruption, or an "
                   "update without refresh(). This evidence holds the weights as they were when the alert was read, "
                   "so the CPU's judgment is against the changed weights")
    elif inc.get("weights_rewritten_since_load"):
        nxt.append("the serving guard found the layer's weights written in place since load, though their values "
                   "match load time now: the matmul may have run with other weights than this evidence holds (those "
                   "at the alert), so the CPU's judgment can't say whether its output was right")
    elif state == "unknown":
        nxt.append("the recorded output couldn't be judged on the CPU; the operands may hold values that aren't "
                   "finite")
    missing = [k for k in STACK_KEYS + EXECUTION_KEYS if stack.get(k) is None]
    if missing:
        nxt.append(f"the incident's {', '.join(missing)} weren't recorded with the evidence, so no replay can be "
                   "shown to run the same way (evidence saved by sdcguard 0.4.0 records them)")
    leads = known_issue_leads(stack)
    for m in leads:
        nxt.append(f"the incident's recorded setup matches NVIDIA cuBLAS known issue {m['id']} ({m['status']}: "
                   f"{m['what']}). Replays with and without its conditions test it"
                   + (f"; NVIDIA's workaround: {m['workaround']}" if m.get("workaround") else ""))
    alternatives = {
        "transient": "a rare race or nondeterministic software path that didn't recur; the replay ran in a fresh "
                     "process, outside the original job's memory state and configuration",
        "recurring": "a software or configuration difference between the original machine and the peer that the "
                     "recorded versions and settings don't capture (firmware, other processes on the GPU)",
        "software": "the same defect in two chips of one model is possible but unlikely; replays on another model "
                    "or with another library version separate them",
        "check": "an output error too small for the CPU's envelope but large enough for the GPU checksum; the "
                 "envelope is conservative",
    }
    return {"observations": obs,
            "explanations": {k: {"what": EXPLANATIONS[k], "status": s, "because": why,
                                 "also_consistent_with": alternatives[k] if s == SUPPORTED else None}
                             for k, (s, why) in hyp.items()},
            "outcome": outcome, "weak_evidence": weak, "next": nxt, "known_issue_leads": leads,
            "note": "Supported means consistent with the evidence above, not proven. The Megatron Core rerun "
                    "machine's own advice applies here too: confirm a suspected GPU with the vendor's diagnostics "
                    "before an RMA. A casefile is built from files anyone can edit: it is evidence for a person to "
                    "weigh, not a signed record to quarantine on by itself."}


def markdown(case: dict) -> str:
    inc, a = case["incident"], case["assessment"]
    lines = ["# sdcguard casefile", "", f"**Outcome: {a['outcome']}**", "",
             f"- evidence: `{inc['evidence']}` (sha256 {str(inc.get('evidence_sha256'))[:16]})",
             f"- what: {inc.get('what')}; op {inc.get('op')}, where {inc.get('where')}, score {inc.get('max_z')}",
             f"- recorded on GPU {inc.get('gpu_uuid')} ({(inc.get('stack') or {}).get('gpu_name')}, driver "
             f"{(inc.get('stack') or {}).get('driver')}, torch {(inc.get('stack') or {}).get('torch')})",
             f"- recorded output vs CPU: {a['observations']['recorded output vs CPU']}",
             f"- same-GPU replays: {a['observations']['same-GPU replays']}",
             f"- peer-GPU replays: {a['observations']['peer-GPU replays'] or 'none'}"]
    o = a["observations"]
    other = o.get("same-GPU replays with other software or settings (not counted)")
    if other:
        lines.append(f"- same-GPU replays with other software or settings (not counted): {other}")
    for key in ("replays of another evidence file, or not saying which (not counted)",
                "the same replay result passed more than once (counted once)"):
        if o.get(key):
            lines.append(f"- {key}: {o[key]}")
    for m in a.get("known_issue_leads") or []:
        lines.append(f"- lead: NVIDIA cuBLAS known issue {m['id']} {m['status']} to the recorded setup ({m['what']})")
    lines += ["",
             "| explanation | status | because |", "|---|---|---|"]
    for k, e in a["explanations"].items():
        lines.append(f"| {e['what']} | {e['status']} | {'; '.join(e['because'])} |")
    if a["weak_evidence"]:
        lines += ["", "Weak evidence: " + "; ".join(a["weak_evidence"]) + "."]
    alts = [f"- {e['what']}: also consistent with {e['also_consistent_with']}"
            for e in a["explanations"].values() if e.get("also_consistent_with")]
    if alts:
        lines += ["", "Still possible:"] + alts
    if a["next"]:
        lines += ["", "To settle it:"] + [f"- {n}" for n in a["next"]]
    lines += ["", a["note"], ""]
    return "\n".join(lines)


def build(evidence: str, replays: list, replay_here: int = 0, device: Optional[str] = None) -> dict:
    inc = load_incident(evidence)
    # the guard's own recomputations right after the alert (SDCGuard(confirm=...)), kept in the evidence file
    reps = [r for r in ((inc.get("info_data") or {}).get("in_job_replays") or []) if isinstance(r, dict)]
    reps += list(replays)
    if replay_here:
        from .screen import replay
        r = replay(evidence, replay_here, device)         # records the software and settings it ran with
        reps.append(r)
    return {"incident": inc, "replays": reps, "assessment": assess(inc, reps)}


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="sdcguard casefile", description=__doc__.split("\n\n")[0])
    p.add_argument("evidence", help="an evidence_N.pt (or math_mode_evidence_N.pt) file")
    p.add_argument("--replays", nargs="*", default=[], help="replay results saved with sdcguard replay --save")
    p.add_argument("--replay-here", type=int, default=0, help="also replay N times on this machine's GPU")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="folder for casefile.json and casefile.md (default: beside the evidence)")
    a = p.parse_args(argv)
    reps = []
    for path in a.replays:
        with open(path) as f:
            reps.append(json.load(f))
    case = build(a.evidence, reps, a.replay_here, a.device)
    out = a.out or os.path.dirname(os.path.abspath(a.evidence))
    os.makedirs(out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.evidence))[0]
    with open(os.path.join(out, f"{stem}_casefile.json"), "w") as f:
        json.dump(case, f, indent=2, default=str)
    md = markdown(case)
    with open(os.path.join(out, f"{stem}_casefile.md"), "w") as f:
        f.write(md)
    print(md)
    return 0 if not case["assessment"]["outcome"].startswith(INCONCLUSIVE) else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
