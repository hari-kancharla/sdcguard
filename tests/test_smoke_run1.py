"""experiments/smoke_run1.py without a GPU: each part's scaled-down settings are the ones its own judge counts, every
speed load is cut, and a part that crashes or fails is written as such. In subprocesses: the smoke test changes the
experiment modules' settings in place, which must not leak into the other tests."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _py(code: str, tmp_path=None) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "experiments"),
                                                       HERE, os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300,
                       cwd=str(tmp_path) if tmp_path else None)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout


def test_each_part_is_scaled_down_to_what_its_judge_counts():
    out = _py(r"""
import smoke_run1 as s
m = s.setup("weight_scan")
assert len(m.trials()) == len(m.BITS) * m.PER_BIT == 2 and [t["bit"] for t in m.trials()] == [3, 14]
assert len(m.param_trials()) == 2 and len(m.moe_trials()) == 1 and m.ALTERNATIONS == ("on", "off")
m = s.setup("moe")
assert m.MODES == ("scale",) and m.CORRUPT_TRIALS == 1 and m.ROUTE_MODES == ("experts",) and m.DROP_TRIALS == 1
m = s.setup("norm_diag")
assert m.ROLES == ("plain_a", "guarded") and m.PAIRS == (("plain_a", "guarded"),) and m.IDENTITY == 4
assert m.RUN_S >= 2 * m.MIN_START_S
m = s.setup("spot_h100")
t = m.trials()
assert [x["kind"] for x in t] == ["all", "one", "band"] and sum(x["criterion"] for x in t) == 2
assert len(m.CRITERIA_TRIALS) + len(m.ONE_CRITERIA) == 2 and m.RUN_S["h100"] - m.FAULTED_RESERVE >= 15 * 60
m = s.setup("moe_bench")
assert (m.REPS, m.WARM, m.TIMES) == (20, 5, 1)
s.setup("screen_l4")
from sdcguard import perf
seen = {}
def full(url, model=None, texts=None, concurrency=16, max_tokens=128, rounds=3, warmup=8):
    seen.update(n=len(texts), c=concurrency, t=max_tokens, r=rounds, w=warmup)
    return {}
assert perf.measure._smoke                                  # (setup wrapped it once)
import importlib
importlib.reload(perf)
perf_full = perf.measure
perf.measure = full
s.small_speed()
perf.measure("http://x", "m", concurrency=256, max_tokens=128, rounds=3)
assert seen == {"n": s.SPEED_PROMPTS, "c": s.SPEED_AT_ONCE, "t": s.SPEED_TOKENS, "r": 1, "w": 1}, seen
print("OK")
""")
    assert "OK" in out


def test_a_part_that_crashes_or_fails_is_written_as_such(tmp_path):
    _py(r"""
import json, os, smoke_run1 as s
import vllm_moe_check as m
s.setup = lambda part: m
def boom(*a):
    raise RuntimeError("server died")
m.run = boom
r = s.run("moe", "out1")
assert r["passed"] is False and r["failures"] == ["crashed: RuntimeError: server died"]
assert json.load(open("out1/SMOKE.json"))["passed"] is False
m.run = lambda *a: {"plain": {"ready": True}}
m.judge = lambda res: (False, ["M1: guarded didn't start"], {})
r = s.run("moe", "out2")
assert r["passed"] is False and r["failures"] == ["M1: guarded didn't start"]
m.judge = lambda res: (True, [], {"on_by_default": False})
assert s.run("moe", "out3")["passed"] is True and s.main(["--part", "moe", "--out", "out4"]) == 0
print("OK")
""", tmp_path)
