"""experiments/vllm_fp8_moe_check.py's criteria (docs/METHOD.md, "An FP8 MoE bug from detection to remedy") on
made-up results: a run that shows the whole path passes; each missing piece fails the criterion it should."""
import copy
import json
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

    def fake_run(cmd, timeout, port, env=None):
        seen["cmd"] = cmd
        return {"code": 0, "output": "", "timed_out": False, "stray": []}

    monkeypatch.setattr(vf, "run_bounded", fake_run)
    res = vf.run(str(tmp_path), 8000, 9400)
    cmd = seen["cmd"]
    assert "no QUALIFY.json" in res["error"]
    assert cmd[cmd.index("--env") + 1] == "SDCGUARD_FP8=1" and "--repeat-baseline" in cmd
    assert cmd[cmd.index("--startup-timeout") + 1] == str(vf.STARTUP_S) == "900"
    assert sum(c.startswith(("fp8pt=", "eager=", "fp8pc=")) for c in cmd) == 3
    import json
    pol = json.load(open(cmd[cmd.index("--policy") + 1]))
    assert pol["isolation"]["required"] and pol["quality"]["accept_warn"]
    assert "fp8_per_tensor" in vf.CANDIDATES["fp8pt"] and "--enforce-eager" in vf.CANDIDATES["eager"]
    assert "fp8_per_channel" in vf.CANDIDATES["fp8pc"] and "cudagraph_capture_sizes" in vf.BASE


def test_each_servers_fp8_coverage_is_read_from_its_metrics(tmp_path):
    out = str(tmp_path)
    with open(os.path.join(out, "fp8pt_metrics0.txt"), "w") as f:
        f.write("# HELP sdcguard_fp8_layers_checked x\nsdcguard_fp8_layers_checked 72.0\n"
                "sdcguard_fp8_calls_judged_total 1234.0\n")
    assert vf._metric(out, "fp8pt", "sdcguard_fp8_layers_checked") == 72.0
    assert vf._metric(out, "fp8pt", "sdcguard_fp8_calls_judged_total") == 1234.0
    assert vf._metric(out, "baseline", "sdcguard_fp8_layers_checked") is None
    r = copy.deepcopy(GOOD)
    r["fp8_coverage"] = {"fp8pt": {"layers": 72.0, "calls_judged": 1234.0}}
    ok, fails, s = vf.judge(r)
    assert ok and s["fp8_coverage"]["fp8pt"]["layers"] == 72.0


def test_a_qualify_past_its_limit_still_gives_a_judged_result(monkeypatch, tmp_path):
    """qualify and the repro folder fit inside the Modal function's 190 minutes, and one that runs past its limit
    leaves an error to judge instead of no SUMMARY.json."""
    def late(cmd, timeout, port, env=None):
        return {"code": None, "output": "started", "timed_out": True, "stray": []}
    monkeypatch.setattr(vf, "run_bounded", late)
    res = vf.run(str(tmp_path), 8000, 9400)
    assert res["exit_code"] is None and res["error"].startswith(f"qualify ran past {vf.QUALIFY_S} s; no QUALIFY.json")
    assert res["output"] == "started"
    ok, fails, _ = vf.judge(res)
    assert not ok and fails
    assert vf.QUALIFY_S + vf.RERUN_S < 187 * 60 and vf.STARTUP_S < vf.QUALIFY_S     # (servers start inside qualify)
    assert vf.QUALIFY_S >= 140 * 60       # the October 25 run needed about 105 minutes in qualify


def test_a_cut_short_qualify_is_judged_on_its_record_so_far(tmp_path, monkeypatch):
    """qualify writes QUALIFY.json after each server (complete: false): the judge reads it, says which servers
    qualify didn't finish and that no repro folder was written, instead of calling every server not started."""
    out = str(tmp_path)

    def late(cmd, timeout, port, env=None):
        with open(os.path.join(out, "QUALIFY.json"), "w") as f:
            json.dump({"runs": {"baseline": {"ready": True}, "again": {"ready": True}, "fp8pt": {"ready": True}},
                       "decisions": {"fp8pt": {"decision": "REJECT", "isolation": "LEAK"}}, "complete": False,
                       "not_yet_run": ["eager", "fp8pc"]}, f)
        return {"code": None, "output": "", "timed_out": True, "stray": []}
    monkeypatch.setattr(vf, "run_bounded", late)
    res = vf.run(out, 8000, 9400)
    assert res["qualify_complete"] is False and res["not_yet_run"] == ["eager", "fp8pc"]
    ok, fails, s = vf.judge(res)
    assert not ok and "F1: qualify stopped before it finished eager" in fails
    assert not any(f.startswith("F1: baseline") or f.startswith("F1: fp8pt") or f.startswith("F2") for f in fails)
    assert "F6: qualify didn't finish, so it wrote no repro folder to rerun" in fails
    assert s["qualify_complete"] is False and s["not_yet_run"] == ["eager", "fp8pc"]


def test_a_command_past_its_limit_is_interrupted_and_its_escaped_servers_stopped(tmp_path):
    """The real thing, on processes: a stand-in for qualify that starts a stand-in server in a session of its own
    (as qualify does), cleans up on SIGINT only if told to, and outlives its limit. Past the limit it gets SIGINT
    (its finally block runs), and a server it left behind on the port is found and stopped."""
    import subprocess
    import textwrap
    port = 18431
    srv = tmp_path / "srv.py"
    srv.write_text("import time\nwhile True:\n    time.sleep(1)\n")
    for clean in (True, False):
        mark = tmp_path / f"cleaned_{clean}"
        q = tmp_path / f"q_{clean}.py"
        q.write_text(textwrap.dedent(f"""
            import subprocess, sys, time
            p = subprocess.Popen([sys.executable, {str(srv)!r}, "serve", "--port", "{port}"], start_new_session=True)
            try:
                time.sleep(600)
            finally:
                if {clean}:
                    p.kill()
                open({str(mark)!r}, "w").write("1")
        """))
        r = vf.run_bounded([sys.executable, str(q)], 3, port)
        assert r["timed_out"] and r["code"] is None and mark.exists()        # its finally block ran
        assert (r["stray"] == []) if clean else (len(r["stray"]) == 1)      # an escaped server: found
        assert vf.servers_on(port) == []                                      # and nothing left on the port
    r = vf.run_bounded([sys.executable, "-c", "print('hi')"], 30, port)
    assert r == {"code": 0, "output": "hi\n", "timed_out": False, "stray": []}


def test_the_sweep_finds_renamed_engines_and_leaves_earlier_ones_alone(tmp_path):
    """vLLM's engine renames itself (VLLM::EngineCore), so the sweep finds it by that name too; one that was already
    running before the command began isn't the command's and is left alone."""
    import subprocess
    import textwrap
    port = 18432
    # a stand-in with "VLLM::EngineCore" as its command line, as setproctitle leaves it
    old = subprocess.Popen(["bash", "-c", "exec -a VLLM::EngineCore sleep 600"], start_new_session=True)
    import time
    t0 = time.monotonic()
    while old.pid not in vf.servers_on(port) and time.monotonic() - t0 < 10:      # (until bash has exec'd)
        time.sleep(0.05)
    try:
        q = tmp_path / "q.py"
        q.write_text("import subprocess\nsubprocess.Popen(['bash', '-c', 'exec -a VLLM::EngineCore sleep 600'], "
                     "start_new_session=True)\n")
        r = vf.run_bounded([sys.executable, str(q)], 30, port)
        assert len(r["stray"]) == 1 and old.pid not in r["stray"] and old.poll() is None
        assert old.pid in vf.servers_on(port)
    finally:
        old.kill()
        old.wait()
