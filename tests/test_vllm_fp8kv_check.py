"""experiments/vllm_fp8kv_check.py's criteria (docs/METHOD.md, "An FP8 KV-cache decode bug through qualify") on
made-up results, and the qualify command it runs."""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import vllm_fp8kv_check as vk  # noqa: E402

GOOD = {"runs": {n: {"ready": True} for n in ("baseline", "fp8kv", "fp8kv_fi")},
        "decisions": {"fp8kv": {"decision": "REJECT", "consistency": "INCONSISTENT"},
                      "fp8kv_fi": {"decision": "ACCEPT", "consistency": "CONSISTENT"}},
        "consistency": {"baseline": {"verdict": "CONSISTENT", "decode_largest_gap": 0.03},
                        "fp8kv": {"verdict": "INCONSISTENT", "decode_largest_gap": 0.45},
                        "fp8kv_fi": {"verdict": "CONSISTENT", "decode_largest_gap": 0.04}}}


def test_the_bug_found_through_qualify_passes():
    ok, fails, s = vk.judge(GOOD)
    assert ok, fails


def test_each_missing_piece_fails_its_criterion():
    cases = {"K1": lambda r: r["runs"]["fp8kv"].update(ready=False),
             "K2": lambda r: r["consistency"]["baseline"].update(verdict="INCONSISTENT"),
             "K3": lambda r: r["decisions"]["fp8kv"].update(consistency="CONSISTENT", decision="ACCEPT")}
    for crit, change in cases.items():
        r = copy.deepcopy(GOOD)
        change(r)
        ok, fails, _ = vk.judge(r)
        assert not ok and any(f.startswith(crit) for f in fails), (crit, fails)


def test_the_qualify_command(monkeypatch, tmp_path):
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, "", ""

    monkeypatch.setattr(vk.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and Done())
    vk.run(str(tmp_path), 8000, 9400)
    cmd = seen["cmd"]
    assert "--no-consistency" not in cmd and "--kv-cache-dtype fp8" in " ".join(cmd)
    assert json.load(open(cmd[cmd.index("--policy") + 1]))["quality"]["accept_warn"] is True
