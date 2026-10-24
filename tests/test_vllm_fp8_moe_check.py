"""experiments/vllm_fp8_moe_check.py's criteria (docs/METHOD.md, "An FP8 MoE bug from detection to remedy") on
made-up results: a run that shows the whole path passes; each missing piece fails the criterion it should."""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))
import vllm_fp8_moe_check as vf  # noqa: E402

GOOD = {
    "runs": {n: {"ready": True} for n in ("baseline", "again", "fp8pt", "eager", "fp8pc")},
    "decisions": {"fp8pt": {"decision": "REJECT", "isolation": "LEAK", "quantization": "WRONG"},
                  "eager": {"decision": "NOT DECIDED", "isolation": "NO LEAK SEEN", "throughput_change": -0.4},
                  "fp8pc": {"decision": "NOT DECIDED", "isolation": "NO LEAK SEEN", "throughput_change": 0.1}},
    "fp8_alerts": {"baseline": {"count": 0, "findings": [], "moe_layers": []},
                   "fp8pt": {"count": 7, "findings": ["scale_from_outside_rows"],
                             "moe_layers": ["model.layers.3.block_sparse_moe.experts (expert inputs)"]},
                   "eager": {"count": 0, "findings": [], "moe_layers": []},
                   "fp8pc": {"count": 0, "findings": [], "moe_layers": []}},
    "repro": {"isolation": "LEAK", "fp8_alerts": 3, "exit_code": 0},
}


def test_the_whole_path_passes():
    ok, fails, s = vf.judge(GOOD)
    assert ok, fails
    assert s["fp8pc"]["isolation"] == "NO LEAK SEEN" and s["repro"]["isolation"] == "LEAK"


def test_each_missing_piece_fails_its_criterion():
    cases = {
        "F1": lambda r: r["runs"]["eager"].update(ready=False),
        "F2": lambda r: r["decisions"]["fp8pt"].update(decision="NOT DECIDED", isolation="NO LEAK SEEN"),
        "F3": lambda r: r["fp8_alerts"]["fp8pt"].update(findings=["values_wrong"], moe_layers=[]),
        "F4": lambda r: r["fp8_alerts"]["eager"].update(count=2),
        "F5": lambda r: r["fp8_alerts"]["baseline"].update(count=1),
        "F6": lambda r: r["repro"].update(isolation="NO LEAK SEEN"),
    }
    for crit, change in cases.items():
        r = copy.deepcopy(GOOD)
        change(r)
        ok, fails, _ = vf.judge(r)
        assert not ok and any(f.startswith(crit) for f in fails), (crit, fails)


def test_fp8pc_not_starting_is_reported_not_failed():
    r = copy.deepcopy(GOOD)
    r["runs"]["fp8pc"]["ready"] = False
    ok, fails, s = vf.judge(r)
    assert ok and s["fp8pc"]["started"] is False


def test_the_qualify_command_has_the_check_on_and_the_policy(monkeypatch, tmp_path):
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setattr(vf.subprocess, "run", fake_run)
    res = vf.run(str(tmp_path), 8000, 9400)
    cmd = seen["cmd"]
    assert "no QUALIFY.json" in res["error"]
    assert cmd[cmd.index("--env") + 1] == "SDCGUARD_FP8=1" and "--repeat-baseline" in cmd
    assert sum(c.startswith(("fp8pt=", "eager=", "fp8pc=")) for c in cmd) == 3
    import json
    pol = json.load(open(cmd[cmd.index("--policy") + 1]))
    assert pol["isolation"]["required"] and pol["quality"]["accept_warn"]
    assert "fp8_per_tensor" in vf.CANDIDATES["fp8pt"] and "--enforce-eager" in vf.CANDIDATES["eager"]
    assert "fp8_per_channel" in vf.CANDIDATES["fp8pc"] and "cudagraph_capture_sizes" in vf.BASE
