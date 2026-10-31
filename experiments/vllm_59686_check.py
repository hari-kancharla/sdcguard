"""A silent wrong-answer bug in vLLM 0.32.0, the current release, and the alternatives that remove it, judged by
`sdcguard qualify` (docs/METHOD.md, "vLLM issue 59686 in vLLM 0.32.0", written before the run).

vLLM issue #59686 (opened October 1, 2026): with the CUTLASS FP8 MoE backend (CutlassExpertsFp8), valid tokens get
the wrong MoE output whenever a CUDA-graph batch has padding rows, because moe_permute computes wrong per-expert offsets
when the router writes expert id -1 for padding (VLLM_MOE_SKIP_PADDING=1, the default). The issue's reporter found it
by comparing next-token distributions one request at a time; the same FP8 checkpoint scored within one standard error
of bf16 on MMLU-Pro. Fixed by PR #59717, merged October 23 as commit e4db46c, after 0.32.0 (October 18).

One H100, RedHatAI/Qwen3-30B-A3B-FP8-dynamic (a compressed-tensors FP8 dynamic checkpoint, 128 experts, 8 a token).

Step 0, the kernel (no weights): moe_permute on every token count from 1 to 512, padded with -1 routes to the CUDA-graph
size vLLM would pad it to (1, 2, 4, then multiples of 8 to 248, then of 16 to 512), 128 experts at top 4 (the issue's)
and top 8 (this model's), called both ways vLLM calls it (plain, and with the reusable scratch serving uses), in vLLM
0.32.0 and in vLLM built at the fix (its wheel for commit e4db46c, in its own environment). Each token's hidden row
carries its own index, so for every valid route the check reads which row the expert's segment really holds: a route
counts as misrouted when its row isn't inside its expert's segment or isn't its own token's. A wrong offset that
misroutes no valid row (the last offset counting padding rows) is reported, not counted. Counts that need no padding
are the probe's own control: no route may be misrouted there, in either build.

Step 1, `sdcguard qualify --no-plugin --repeat-baseline` (the plugin's arithmetic checks don't cover FP8 matmuls; this
run is about qualify's checks: the quality check against the baseline's reference, isolation, consistency, speed):
  baseline  the model with --moe-backend triton (the issue reports this backend correct)
  again     the baseline started a second time (the noise floor)
  cutlass   --moe-backend cutlass on vLLM 0.32.0: the issue's backend
  nopad     the same with VLLM_MOE_SKIP_PADDING=0 (padding rows routed to real experts: the bug's trigger removed)
  fixed     --moe-backend cutlass on vLLM built at e4db46c (the fix); left out when step 0 can't show that build fixed
  auto      no backend flag: what vLLM 0.32.0 picks for this model on its own (reported, not judged)

Criteria:
  V   the run says something only if: step 0 shows vLLM 0.32.0's moe_permute misrouting valid routes at top 8 and
      none at any count without padding; the baseline, again, cutlass and nopad servers start, captured CUDA graphs
      (their logs say "Graph capturing finished": padding happens only in graphs), and their logs say vLLM used the
      TRITON FP8 MoE backend (baseline, again) and VLLM_CUTLASS (cutlass, nopad).
  C1  qualify REJECTs cutlass, with its quality, consistency or isolation check among the reasons (caught).
  C2  for nopad and for fixed: qualify doesn't REJECT it, and each check that rejected cutlass (the quality check
      when none did) was measured on it and passed (approximation WITHIN, consistency CONSISTENT, isolation NO LEAK
      SEEN): the alternatives remove what was caught. NOT DECIDED is expected otherwise, since without the plugin
      the arithmetic isn't measured. For fixed, also: step 0 shows its environment is the build at e4db46c with no
      misrouted route, and its server started, captured graphs and used VLLM_CUTLASS.
Reported, not criteria: auto's backend and decision (whether vLLM 0.32.0's own choice is the affected backend); each
run's quality numbers (mean KL, top-1 agreement, worst prompt) beside again's, the noise floor, and cutlass's against
nopad's (the same kernel with the trigger removed); each candidate's checks and speed against the baseline (what each
alternative costs); step 0 at top 4 and with each call. If C1 isn't met, the result is recorded as not caught, as it
stands.

    python experiments/vllm_59686_check.py --out runs/issue59686
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

MODEL = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
FIX_COMMIT = "e4db46c79bca354fef07197465faea14ca563c09"
FIX_PY = os.environ.get("SDCGUARD_FIX_PY", "/opt/vfix/bin/python")
COMMON = "--max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0"
QUALIFY_S = 85 * 60                 # (with step 0, inside the Modal phase's 96 minutes)
STEP0_S = 300
BACKEND = re.compile(r"Using (\w+) Fp8 MoE backend")
GRAPHS = "Graph capturing finished"
CANDIDATES = {"cutlass": "--moe-backend cutlass", "nopad": "--moe-backend cutlass",
              "fixed": "--moe-backend cutlass", "auto": ""}
WANT_BACKEND = {"baseline": "TRITON", "again": "TRITON", "cutlass": "VLLM_CUTLASS", "nopad": "VLLM_CUTLASS",
                "fixed": "VLLM_CUTLASS"}
CORE = ("baseline", "again", "cutlass", "nopad")     # what V needs; fixed is part of C2 only
CAUGHT_BY = ("approximation", "consistency", "isolation")
PASSING = {"approximation": "WITHIN", "consistency": "CONSISTENT", "isolation": "NO LEAK SEEN"}
VARIANTS = ("plain", "scratch")
TOPKS = ("top4", "top8")

# where moe_permute puts each valid route, against where its expert's segment is: the fix's own regression test reads
# the permuted rows the same way (a row carries its token's index)
PROBE = r"""
import json, torch
from vllm.model_executor.layers.fused_moe import moe_permute_unpermute as mpu
import vllm
CAPTURE = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
dev = torch.device("cuda")


def one(E, topk, n_valid, scratch, gen):
    n = next(c for c in CAPTURE if c >= n_valid)
    valid = torch.rand(n_valid, E, device=dev, generator=gen).argsort(dim=1)[:, :topk].to(torch.int32)
    ids = torch.cat([valid, torch.full((n - n_valid, topk), -1, dtype=torch.int32, device=dev)])
    t = torch.arange(n, device=dev)
    hidden = torch.zeros(n, 64, device=dev, dtype=torch.bfloat16)
    hidden[:, 0], hidden[:, 1] = (t // 32).to(torch.bfloat16), (t % 32).to(torch.bfloat16)
    kw = {"scratch": mpu.MoEPermuteScratch(num_experts=E, num_local_experts=E, device=dev)} if scratch else {}
    perm, _, off, inv, _ = mpu.moe_permute(hidden, None, ids, E, E, None, **kw)
    off, inv = off.long(), inv.long().view(n, topk)[:n_valid]
    counts = torch.bincount(valid.reshape(-1).long(), minlength=E)
    expected = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)])
    e = valid.long()
    inside = (inv >= 0) & (inv < perm.shape[0])
    row = perm[inv.clamp(0, perm.shape[0] - 1)]
    own = (row[..., 0].float() * 32 + row[..., 1].float()) == t[:n_valid, None].float()
    right = inside & (inv >= off[e]) & (inv < off[e + 1]) & own
    return n - n_valid, bool((off != expected).any()), int((~right).sum())


out = {"vllm": vllm.__version__}
for variant in ("plain", "scratch"):
    out[variant] = {}
    for E, topk in ((128, 4), (128, 8)):
        gen = torch.Generator(device=dev).manual_seed(0)
        r = {"sizes": 0, "padded_sizes": 0, "sizes_offsets_wrong": 0, "sizes_misrouting": 0, "misrouted_routes": 0,
             "max_misrouted_in_a_size": 0, "unpadded_sizes_misrouting": 0, "examples": []}
        try:
            for n_valid in range(1, 513):
                pad, off_wrong, bad = one(E, topk, n_valid, variant == "scratch", gen)
                r["sizes"] += 1
                r["padded_sizes"] += pad > 0
                r["sizes_offsets_wrong"] += off_wrong
                r["sizes_misrouting"] += bad > 0
                r["misrouted_routes"] += bad
                r["max_misrouted_in_a_size"] = max(r["max_misrouted_in_a_size"], bad)
                r["unpadded_sizes_misrouting"] += pad == 0 and bad > 0
                if bad and len(r["examples"]) < 5:
                    r["examples"].append({"tokens": n_valid, "padding": pad, "misrouted": bad, "of": n_valid * topk})
            torch.cuda.synchronize()
        except Exception as ex:  # noqa: BLE001  (one call's failure is reported; the other still runs)
            r["error"] = f"{type(ex).__name__}: {ex}"[:400]
        out[variant][f"top{topk}"] = r
print("PROBE " + json.dumps(out))
"""


def probe(python: str) -> dict:
    try:
        p = subprocess.run([python, "-c", PROBE], capture_output=True, text=True, timeout=STEP0_S)
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {STEP0_S} s"}
    except OSError as e:
        return {"error": f"{type(e).__name__}: {e}"}
    for line in p.stdout.splitlines():
        if line.startswith("PROBE "):
            try:
                return json.loads(line[6:])
            except ValueError:
                break
    return {"error": f"exit {p.returncode}: {(p.stdout + p.stderr)[-1500:]}"}


def qualify_cmd(out: str, fixed: bool = True) -> list:
    cmd = [sys.executable, "-m", "sdcguard", "qualify", "--out", out, "--no-plugin", "--repeat-baseline",
           "--baseline", f"{MODEL} {COMMON} --moe-backend triton", "--tokens", "64", "--rounds", "2",
           "--startup-timeout", "1500", "--config-env", "nopad:VLLM_MOE_SKIP_PADDING=0"]
    if fixed:
        cmd += ["--config-serve-cmd", f"fixed:{FIX_PY} -m vllm.entrypoints.cli.main serve"]
    for name, flags in CANDIDATES.items():
        if name != "fixed" or fixed:
            cmd += ["--candidate", f"{name}={MODEL} {COMMON} {flags}".strip()]
    return cmd


def _logs(qdir: str):
    for name in ("baseline", "again", *CANDIDATES):
        try:
            with open(os.path.join(qdir, f"{name}_server.log"), errors="replace") as f:
                yield name, f.read()
        except OSError:
            yield name, None


def backends(qdir: str) -> dict:
    """Each server's FP8 MoE backend, from vLLM's own log line ("Using X Fp8 MoE backend out of ...")."""
    return {n: (None if text is None else sorted(set(BACKEND.findall(text)))) for n, text in _logs(qdir)}


def graphs(qdir: str) -> dict:
    """Whether each server captured CUDA graphs (vLLM's "Graph capturing finished in ..."): only a graph pads."""
    return {n: (None if text is None else GRAPHS in text) for n, text in _logs(qdir)}


def quality(qdir: str) -> dict:
    """Each run's quality numbers against the baseline's reference, from its {name}_quality.json."""
    out = {}
    for name in ("again", *CANDIDATES):
        try:
            with open(os.path.join(qdir, f"{name}_quality.json")) as f:
                j = json.load(f)
        except (OSError, ValueError):
            out[name] = None
            continue
        s = j.get("summary") or {}
        out[name] = {k: s.get(k) for k in ("mean_kl", "top1_agreement", "max_prompt_mean_kl", "positions")}
        out[name]["verdict"] = (j.get("verdict") or {}).get("verdict")
    return out


def self_checks(qdir: str) -> dict:
    """The baseline's and its repeat's own isolation and consistency verdicts (a candidate's are read beside them: a
    line the baseline itself fails can't separate a candidate)."""
    out = {}
    for n in ("baseline", "again"):
        out[n] = {}
        for c in ("isolation", "consistency"):
            try:
                with open(os.path.join(qdir, f"{n}_{c}.json")) as f:
                    out[n][c] = json.load(f).get("verdict")
            except (OSError, ValueError):
                out[n][c] = None
    return out


def run(out: str) -> dict:
    os.makedirs(out, exist_ok=True)
    res: dict = {"model": MODEL, "fix_commit": FIX_COMMIT}
    t0 = time.monotonic()
    res["step0"] = {"0.32.0": probe(sys.executable), "fixed": probe(FIX_PY)}
    _save(out, "step0.json", res["step0"])          # (kept even if the rest of the run is cut short)
    qdir = os.path.join(out, "qual")
    why = step0_problem(res["step0"])
    if why:                                  # (the servers would say nothing about the bug: not started, not paid for)
        res["skipped"] = f"qualify not run: {why}"
        res["backends"] = {}
        res["seconds"] = round(time.monotonic() - t0, 1)
        return res
    res["fixed_left_out"] = fixed_problem(res["step0"])    # (then its server isn't started either)
    cmd = qualify_cmd(qdir, fixed=not res["fixed_left_out"])
    res["qualify_command"] = cmd
    try:
        with open(os.path.join(out, "qualify.log"), "w") as log:
            p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=QUALIFY_S)
        res["qualify_exit"] = p.returncode
    except subprocess.TimeoutExpired:
        res["qualify_exit"] = None
        res["qualify_error"] = f"timed out after {QUALIFY_S} s"
    for f, key in (("QUALIFY.json", "decisions"), ("RUNS.json", "runs")):
        try:
            with open(os.path.join(qdir, f)) as fh:
                res[key] = json.load(fh).get(key)
        except (OSError, ValueError) as e:
            res[key] = None
            res.setdefault("missing", []).append(f"{f}: {e}")
    res["backends"] = backends(qdir)
    res["graphs"] = graphs(qdir)
    res["quality"] = quality(qdir)
    res["self_checks"] = self_checks(qdir)
    res["seconds"] = round(time.monotonic() - t0, 1)
    _save(out, "PARTIAL.json", res)
    return res


def _save(out: str, name: str, obj) -> None:
    try:
        with open(os.path.join(out, name), "w") as f:
            json.dump(obj, f, indent=1, default=str)
    except OSError:
        pass


def _cells(env: dict):
    """(variant, topk, result) for each call the probe made in one environment."""
    for v in VARIANTS:
        for k in TOPKS:
            r = ((env or {}).get(v) or {}).get(k)
            if r is not None:
                yield v, k, r


def step0_problem(s0: dict):
    """Why step 0 makes the rest of the run say nothing (None when it doesn't): the probe failed, it misroutes where
    nothing was padded (it misreads the kernel), or vLLM 0.32.0 misroutes no valid route at this model's top 8."""
    old = s0.get("0.32.0") or {}
    if old.get("error"):
        return f"step 0 couldn't run in vLLM 0.32.0: {old['error']}"
    cells = list(_cells(old))
    if not cells:
        return "step 0 returned nothing for vLLM 0.32.0"
    ctl = [(v, k) for v, k, r in cells if r.get("unpadded_sizes_misrouting")]
    if ctl:
        return f"the probe misroutes routes with no padding in vLLM 0.32.0 ({ctl}): it misreads the kernel"
    if not any(r.get("sizes_misrouting") for v, k, r in cells if k == "top8"):
        return ("vLLM 0.32.0's moe_permute misroutes no valid route at top 8 (this model's): "
                + json.dumps({v: (old.get(v) or {}).get("top8") for v in VARIANTS})[:600])
    return None


def fixed_problem(s0: dict):
    """Why the fixed build can't be judged (None when it can): its probe failed, it isn't the build at e4db46c, the
    plain call didn't run, or a route is still misrouted."""
    new = s0.get("fixed") or {}
    if new.get("error"):
        return f"step 0 couldn't run in the fixed environment: {new['error']}"
    if "e4db46c" not in str(new.get("vllm") or ""):
        return f"the fixed environment's vLLM is {new.get('vllm')!r}, not the build at e4db46c"
    cells = list(_cells(new))
    plain = [r for v, k, r in cells if v == "plain"]
    if len(plain) != len(TOPKS) or any(r.get("error") for r in plain):
        return f"the plain call didn't run in the fixed build: {new.get('plain')}"
    still = [(v, k, r.get("misrouted_routes")) for v, k, r in cells if r.get("sizes_misrouting")]
    if still:
        return f"the fixed build still misroutes routes: {still}"
    return None


def _reasons(row: dict) -> list:
    return [w.split(":", 1)[0].strip() for w in (row or {}).get("why") or []]


def judge(res: dict) -> tuple:
    fails = []
    s0 = res.get("step0") or {}
    why = step0_problem(s0)
    if why:
        fails.append(f"V: step 0: {why}")
    runs = res.get("runs") or {}
    be, cap = res.get("backends") or {}, res.get("graphs") or {}

    def server(n):
        """What's wrong with a server's run for this experiment (an empty list when nothing is)."""
        bad = []
        if not (runs.get(n) or {}).get("ready"):
            bad.append("didn't start")
        if be.get(n) != [WANT_BACKEND[n]]:
            bad.append(f"FP8 MoE backend {be.get(n)}, not {WANT_BACKEND[n]}")
        if cap.get(n) is not True:
            bad.append("no CUDA graphs captured (nothing padded)")
        return bad

    core = {n: server(n) for n in CORE}
    if any(core.values()):
        fails.append(f"V: servers not as the run needs them: { {n: b for n, b in core.items() if b} }")
    rows = res.get("decisions") or {}
    cut = rows.get("cutlass") or {}
    caught_by = [r for r in CAUGHT_BY if r in _reasons(cut)]
    caught = cut.get("decision") == "REJECT" and bool(caught_by)
    if not caught:
        fails.append(f"C1: cutlass was {cut.get('decision')} ({cut.get('why')}), not rejected by the quality, "
                     "consistency or isolation check (not caught)")
    need = caught_by or ["approximation"]
    not_run, same_lines = [], []
    timed_out = res.get("qualify_exit") is None and bool(res.get("qualify_error"))
    for n in ("nopad", "fixed"):
        if n == "fixed":
            fwhy = fixed_problem(s0)
            if fwhy:
                fails.append(f"C2: fixed can't be judged: {fwhy}")
                continue
            if timed_out and n not in runs:
                not_run.append(n)                  # cut by qualify's time limit: not run, not a failed criterion
                continue
            bad = server("fixed")
            if bad:
                fails.append(f"C2: the fixed server: {bad}")
                continue
        elif timed_out and n not in runs:
            not_run.append(n)
            fails.append("C2: nopad wasn't run before qualify's time limit (the run says nothing about the "
                         "alternatives)")
            continue
        row = rows.get(n) or {}
        if row.get("decision") in (None, "REJECT"):
            fails.append(f"C2: {n} was {row.get('decision')} ({row.get('why')})")
        unmet = {c: row.get(c) for c in need if row.get(c) != PASSING[c]}
        if unmet:
            fails.append(f"C2: on {n}, the checks that caught cutlass weren't measured and passed: {unmet}")
            if caught and any(row.get(c) not in (None, "UNMEASURED") for c in unmet):
                same_lines.append(n)               # the same line failed on the same kernel without the trigger
    # How a C2 failure is read (written before the run): when nopad or fixed fails the very check that caught
    # cutlass, qualify's lines can't separate the CUTLASS backend change from the bug on this stack; the result is
    # "not separated", never "the alternative doesn't remove the bug" and never "caught".
    reading = None
    if caught and same_lines:
        reading = ("not separated: " + ", ".join(same_lines) + " failed the same check that caught cutlass, so qualify's "
                   "lines can't tell the CUTLASS backend change from the bug here")
    elif caught and not fails:
        reading = "caught, and both alternatives remove it"
    q = res.get("quality") or {}
    kl = {n: (q.get(n) or {}).get("mean_kl") for n in ("again", "cutlass", "nopad", "fixed")}
    ratio = (kl["cutlass"] / kl["nopad"] if isinstance(kl["cutlass"], (int, float)) and
             isinstance(kl["nopad"], (int, float)) and kl["nopad"] > 0 else None)
    summary = {"step0": s0, "backends": be, "graphs": cap, "quality": q, "cutlass_over_nopad_mean_kl": ratio,
               "fixed_left_out": res.get("fixed_left_out"), "not_run_for_time": not_run, "reading": reading,
               "baseline_self_checks": res.get("self_checks"),
               "decisions": {n: {"decision": (rows.get(n) or {}).get("decision"),
                                 "approximation": (rows.get(n) or {}).get("approximation"),
                                 "consistency": (rows.get(n) or {}).get("consistency"),
                                 "isolation": (rows.get(n) or {}).get("isolation"),
                                 "performance": (rows.get(n) or {}).get("performance"),
                                 "why": (rows.get(n) or {}).get("why")} for n in CANDIDATES},
               "auto_backend": be.get("auto"), "caught": caught}
    return not fails, fails, summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.out)
    ok, fails, s = judge(res)
    res.update(passed=ok, failures=fails, summary=s)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"passed": ok, "failures": fails, "summary": s}, indent=1, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
