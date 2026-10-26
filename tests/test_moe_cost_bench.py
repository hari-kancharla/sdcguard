"""experiments/moe_cost_bench.py without a GPU: every call it times runs once through Triton's interpreter with the
arguments it gives sdcguard's kernels, and the summary is complete."""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(HERE), "experiments")

CODE = r"""
import json, sys, tempfile
import moe_cost_bench as b
b.DEV = "cpu"
b.E, b.H, b.I, b.L = 8, 64, 32, 3
b.ROWS = (5,)
b.GRIDS = (1, 2)
b.EXITS = 2
calls = []
def once(fn):
    fn()
    calls.append(1)
    return 1.0
b.graph_us = once
d = tempfile.mkdtemp()
code = b.main(["--out", d])
s = json.load(open(d + "/SUMMARY.json"))
assert code == 0 and s["complete"], s
assert set(s["checks"]) == {"5_none_active", "5_one_active", "5_one_active_wrong"}, s["checks"]
assert s["exit"]["2"]["us_per_kernel"] == 0.5 and s["fixtures_ok"]
ok = s["checks"]["5_one_active"]
assert ok["quiet"] and ok["judged"] and ok["state_after"]["moe_calls"] > 0, ok
assert s["checks"]["5_none_active"]["state_after"]["moe_calls"] == 0
bad = s["checks"]["5_one_active_wrong"]
assert bad["alarmed"] and bad["state_after"]["moe_kept"] == 1, bad
# a healthy fixture that alarms makes the run incomplete
real = b._layers
b._layers = lambda rows, active, wrong=False: real(rows, active, wrong=active or wrong)
d2 = tempfile.mkdtemp()
assert b.main(["--out", d2]) == 1 and not json.load(open(d2 + "/SUMMARY.json"))["complete"]
print("OK", len(calls))
"""


def test_the_bench_calls_the_kernels_as_the_plugin_does():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_ALLOW_CPU="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), EXP,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0 and "OK" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
