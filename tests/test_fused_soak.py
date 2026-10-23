"""experiments/fused_soak.py's criteria and histogram (the soak itself needs a GPU; its CPU smoke run is in the
experiment's docstring)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import fused_soak as fs  # noqa: E402


def _rec(**kw):
    r = {"finished": True, "false_alarms": 0, "fused_checks": 1000, "scored": 1000, "unjudged": 0,
         "stats": {"checked_fused": 1000, "fused_fallbacks": 0}}
    r.update(kw)
    return r


def test_a_clean_finished_run_meets_every_criterion():
    assert fs.judge(_rec()) == []


def test_each_criterion_can_fail():
    assert fs.judge(_rec(finished=False))
    assert fs.judge(_rec(false_alarms=1))
    assert fs.judge(_rec(scored=500))
    assert fs.judge(_rec(fused_checks=0, scored=0))
    assert fs.judge(_rec(unjudged=3))
    assert fs.judge(_rec(stats={"checked_fused": 900, "fused_fallbacks": 100}))


def test_the_histogram_puts_each_score_in_one_bin():
    h = fs.histogram([0.1, 0.5, 2.9, 5.99, 6.0, math.inf])
    assert sum(h) == 6 and h[0] == 1 and h[1] == 1 and h[-1] == 2
