"""How much of the harm study's catch rate was luck, and what checking more often would buy.

The H100 harm study (results/harm_study_gpt2) checked every 10th step and caught a fault before the
crash in 14 of 15 crashing runs. Whether a fault lands on a checked step is partly chance, so this
recomputes the expected catch rate from the same runs' fault logs:

  - checks are read-only, so a run's faults and its crash step don't depend on which steps are checked;
  - a fault on a checked step (every 10th) has a known outcome (caught or not);
  - a fault on another step is caught, if checked, with the catch rate measured for faults of its size
    (how much it changed its 128-value block) among the faults that were checked.

Then: every k steps with a random starting offset (averaged over all offsets), and checking a random
share of matmuls on every step.

    python experiments/schedule_analysis.py [results/harm_study_gpt2/raw]
"""
from __future__ import annotations

import glob
import json
import os
import sys

BUCKETS = [(0, .01), (.01, .02), (.02, .05), (.05, .1), (.1, float("inf"))]


def _rel(r):
    return float("inf") if r == "inf" else (None if r is None else float(r))


def _bucket(rel):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= rel < hi:
            return i
    return len(BUCKETS) - 1


def main(raw):
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(raw, "faulty_*.json")))]
    checked = [(_rel(e[2]), e[4]) for r in runs for e in r["fault_events"] if e[3] and _rel(e[2]) is not None]
    rate = []
    for lo, hi in BUCKETS:
        s = [c for rel, c in checked if lo <= rel < hi]
        rate.append(sum(s) / len(s) if s else 0.0)
    crash = [r for r in runs if r["nan_step"] is not None]
    on_checked = sum(1 for r in runs for e in r["fault_events"] if e[0] % 10 == 0)
    total = sum(len(r["fault_events"]) for r in runs)

    def p_catch(e):
        if e[3]:
            return 1.0 if e[4] else 0.0
        rel = _rel(e[2])
        return rate[_bucket(rel)] if rel is not None else 0.0

    def every(k):
        tot = 0.0
        for r in crash:
            hit = 0.0
            for phase in range(k):
                miss = 1.0
                for e in r["fault_events"]:
                    if e[0] < r["nan_step"] and e[0] % k == phase:
                        miss *= 1 - p_catch(e)
                hit += 1 - miss
            tot += hit / k
        return tot

    def actual(k):
        return sum(any(e[3] and e[4] and e[0] % k == 0 and e[0] < r["nan_step"] for e in r["fault_events"])
                   for r in crash)

    def random_share(p):
        tot = 0.0
        for r in crash:
            miss = 1.0
            for e in r["fault_events"]:
                if e[0] < r["nan_step"]:
                    miss *= 1 - p * p_catch(e)
            tot += 1 - miss
        return tot

    print(f"{len(runs)} faulty runs, {len(crash)} crashed; {on_checked} of {total} faults "
          f"({100 * on_checked / total:.1f}%) landed on checked steps (every 10th)")
    print("catch rate of checked faults by size: " +
          ", ".join(f"{lo:g}-{hi:g}: {100 * p:.0f}%" for (lo, hi), p in zip(BUCKETS, rate)))
    print("\ncaught before the crash (of %d crashing runs):" % len(crash))
    print("  schedule                     this study   expected (random start)")
    for k in (1, 2, 5, 10, 20, 40, 50):
        a = actual(k) if k % 10 == 0 else None
        print(f"  every {k:2d} steps                {'' if a is None else a:>6}       {every(k):5.1f}")
    for p in (0.5, 0.2, 0.1):
        print(f"  random {100 * p:.0f}% of matmuls, every step            {random_share(p):5.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results",
                                                             "harm_study_gpt2", "raw"))
