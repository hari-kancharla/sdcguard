"""Run 1's smoke test (docs/METHOD.md, "Run 1's smoke test", written before it): each part of Run 1 on its own GPU,
scaled down to minutes, through the same code and judged by its own judge, so what breaks only on a real GPU inside a
real `vllm serve` (a server that doesn't start, a kernel that doesn't compile or fails its first-use comparison, a
fault-injection hook vLLM doesn't reach, a compiled graph without the compiled fault, a timeout) shows for a few
dollars before the full run. Each part runs in its own process, so the settings changed here change nothing else.

  weight_scan   vllm_weight_scan_check: its 3 servers, 2 weight bit flips (bits 3 and 14), 2 other-parameter changes,
                1 MoE expert change, one alternation of the scan paused and on
  moe_bench     moe_cost_bench: every figure, 20 replays each
  norm_diag     vllm_norm_diag: plain_a and guarded only, 4 prompts each
  moe           vllm_moe_check: its 4 servers, 4 prompts each, 1 wrong kernel, 1 wrong router, 1 real row without
                experts (M6)
  spot_h100     vllm_spot_check: its 4 servers (the compiled fault among them), 8 prompts each, one trial of each
                kind (every output 5% too large, one sampled value 2% too large, a band)
  spot_l4       vllm_spot_check --gpu l4: the spot server, 8 prompts
  screen_h100   screen_ops_check: the clean run for 0.2 minutes (its 6 rounds' coverage still), the 15 simulations
  screen_l4     the same on an L4

Every speed load is cut to 16 prompts of 16 tokens, at most 8 at once, one round. The part passes when its own judge
passes on these scaled-down counts (the cost decision isn't part of it); the norm diagnosis when it is complete.
Nothing here is evidence for a criterion of Run 1: the smoke test's results are reported as a check of the run, not
as measurements.

    python experiments/smoke_run1.py --part spot_h100 --out runs/smoke_spot_h100
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PARTS = ("weight_scan", "moe_bench", "norm_diag", "moe", "spot_h100", "spot_l4", "screen_h100", "screen_l4")
L4_PARTS = ("spot_l4", "screen_l4")
SPEED_PROMPTS, SPEED_TOKENS, SPEED_AT_ONCE = 16, 16, 8


def small_speed() -> None:
    """Every sdcguard.perf.measure in this process cut to SPEED_PROMPTS prompts of SPEED_TOKENS tokens, at most
    SPEED_AT_ONCE at once, one round after one left out."""
    from sdcguard import perf
    if getattr(perf.measure, "_smoke", False):
        return
    full = perf.measure

    def measure(url, model=None, texts=None, concurrency=16, max_tokens=128, rounds=3, warmup=8):
        return full(url, model, texts=(texts or perf.default_prompts())[:SPEED_PROMPTS],
                    concurrency=min(concurrency, SPEED_AT_ONCE), max_tokens=min(max_tokens, SPEED_TOKENS), rounds=1,
                    warmup=1)
    measure._smoke = True
    perf.measure = measure


def setup(part: str):
    """The part's experiment module, scaled down (the module's settings changed in place)."""
    small_speed()
    if part == "weight_scan":
        import vllm_weight_scan_check as m
        m.BITS, m.PER_BIT, m.IDENTITY, m.MOE_PROMPTS = (3, 14), 1, 4, 8
        m.ALTERNATIONS = ("on", "off")
        full_p, full_m = m.param_trials, m.moe_trials
        m.param_trials = lambda seed=1: full_p(seed)[:2]
        m.moe_trials = lambda seed=2: full_m(seed)[:1]
        m.RUN_S = 22 * 60
    elif part == "moe_bench":
        import moe_cost_bench as m
        m.REPS, m.WARM, m.TIMES = 20, 5, 1
    elif part == "norm_diag":
        import vllm_norm_diag as m
        m.ROLES, m.PAIRS, m.IDENTITY = ("plain_a", "guarded"), (("plain_a", "guarded"),), 4
        m.RUN_S = 17 * 60
    elif part == "moe":
        import vllm_moe_check as m
        m.MODES, m.CORRUPT_TRIALS, m.ROUTE_MODES, m.DROP_TRIALS, m.IDENTITY = ("scale",), 1, ("experts",), 1, 4
        m.RUN_S = 22 * 60
    elif part in ("spot_h100", "spot_l4"):
        import vllm_spot_check as m
        m.IDENTITY = 8
        m.CRITERIA_TRIALS, m.REPORTED_TRIALS = (("all", 1.05),), (("band", 1.03),)
        m.ONE_CRITERIA, m.ONE_REPORTED = (("one", 1.02),), ()
        m.RUN_S = {"h100": 25 * 60, "l4": 12 * 60}
        m.FAULTED_RESERVE = 8 * 60
    elif part in ("screen_h100", "screen_l4"):
        import screen_ops_check as m
    else:
        raise ValueError(f"no such part: {part} (one of {', '.join(PARTS)})")
    return m


def run(part: str, out: str, port: int = 8000, metrics_port: int = 9400) -> dict:
    os.makedirs(out, exist_ok=True)
    m = setup(part)
    res: dict = {"part": part}
    try:
        if part == "moe_bench":
            code = m.main(["--out", out])
            s = json.load(open(os.path.join(out, "SUMMARY.json")))
            ok, fails = code == 0, ([] if code == 0 else
                                    [f"moe_bench incomplete (fixtures_ok {s.get('fixtures_ok')}, error {s.get('error')})"])
        elif part == "norm_diag":
            r = m.run(out, port, metrics_port)
            s = m.summarize(r)
            ok, fails = bool(s["complete"]), list(s.get("incomplete_because") or [])
            res["tuned_by_kernel"] = {k: v.get("tuned_by_kernel") for k, v in s["roles"].items()}
            res["result"] = r
        elif part.startswith("screen"):
            r = m.run(out, 0.2)
            ok, fails = m.judge(r)
            res["result"] = r
        else:
            r = m.run(out, port, metrics_port, part.split("_")[1]) if part.startswith("spot") else \
                m.run(out, port, metrics_port)
            ok, fails, s = m.judge(r)
            res["result"], res["summary"] = r, s
    except Exception as e:  # noqa: BLE001  (a part that crashed: that is the smoke test's finding)
        ok, fails = False, [f"crashed: {type(e).__name__}: {str(e)[:500]}"]
    res.update(passed=bool(ok), failures=fails)
    with open(os.path.join(out, "SMOKE.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(json.dumps({"part": part, "passed": bool(ok), "failures": fails}, indent=1, default=str))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=PARTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--metrics-port", type=int, default=9400)
    a = ap.parse_args(argv)
    return 0 if run(a.part, a.out, a.port, a.metrics_port)["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
