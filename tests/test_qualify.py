"""sdcguard qualify end to end on a CPU, with stand-in servers (tests/_fake_vllm.py) for the baseline and five
candidates: the same stack, a quality change, an answer that depends on the request before, a changed weight the
plugin reports, and a server that doesn't start."""
import json
import os
import socket
import sys

from sdcguard import qualify

HERE = os.path.dirname(os.path.abspath(__file__))


def _free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_one_decision_per_candidate_with_its_reason(tmp_path, capsys):
    out = str(tmp_path / "q")
    code = qualify.main([
        "--out", out, "--baseline", "fake/m", "--repeat-baseline",
        "--candidate", "same=fake/m", "--candidate", "shifted=fake/m --shift 4", "--candidate", "leaky=fake/m --leak",
        "--candidate", "faulty=fake/m --fault", "--candidate", "dead=fake/m --fail",
        "--serve-cmd", f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}",
        "--port", str(_free()), "--metrics-port", str(_free()), "--tokens", "8", "--concurrency", "4",
        "--max-tokens", "8", "--rounds", "2", "--startup-timeout", "30", "--stack", "gpu=cpu"])
    rows = json.load(open(os.path.join(out, "QUALIFY.json")))["decisions"]
    assert code == 1
    assert rows["same"]["decision"] == "ACCEPT", rows["same"]
    assert rows["same"]["arithmetic"] == "NO FAULT OBSERVED" and rows["same"]["isolation"] == "NO LEAK SEEN"
    assert rows["shifted"]["decision"] == "REJECT" and rows["shifted"]["approximation"] == "OUTSIDE"
    assert rows["leaky"]["decision"] == "REJECT" and rows["leaky"]["isolation"] == "LEAK"
    assert any(w.startswith("isolation") for w in rows["leaky"]["why"])
    assert rows["faulty"]["decision"] == "REJECT" and rows["faulty"]["arithmetic"] == "FAULT"
    assert rows["dead"]["decision"] == "NOT DECIDED"
    report = open(os.path.join(out, "REPORT.md")).read()
    assert "| leaky | REJECT |" in report and "Servers that didn't start: dead" in report
    assert os.path.exists(os.path.join(out, "decision_faulty.txt"))
    q = json.load(open(os.path.join(out, "same_quality.json")))
    assert q["stack"]["gpu"] == "cpu" and q["verdict"]["verdict"] == "PASS"


def test_names_and_tensor_parallel_size_are_read_from_the_configuration():
    assert qualify._parse_config("awq=Qwen/Q --x 1", "c1") == ("awq", "Qwen/Q --x 1")
    assert qualify._parse_config("Qwen/Q --x=1", "c1") == ("c1", "Qwen/Q --x=1")
    assert qualify._tp(["m", "--tensor-parallel-size", "2"]) == 2 and qualify._tp(["m", "-tp=4"]) == 4
    assert qualify._tp(["m"]) == 1


def test_the_sdcguard_command_runs_qualify(capsys):
    from sdcguard import cli
    try:
        cli.main(["qualify", "--help"])
    except SystemExit as e:
        assert e.code == 0
    assert "--candidate" in capsys.readouterr().out


def test_a_share_too_small_for_the_policy_is_not_accepted(tmp_path):
    """With one layer in eight checked, the plugin's silence can't stand for no fault under decide's default policy
    (at least half the weights): the unchanged stack isn't ACCEPTed on that silence."""
    out = str(tmp_path / "q")
    code = qualify.main([
        "--out", out, "--baseline", "fake/m", "--candidate", "same=fake/m", "--share", "1/8", "--no-isolation",
        "--no-perf", "--serve-cmd", f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}",
        "--port", str(_free()), "--metrics-port", str(_free()), "--tokens", "8", "--startup-timeout", "30"])
    row = json.load(open(os.path.join(out, "QUALIFY.json")))["decisions"]["same"]
    assert code != 0 and row["decision"] != "ACCEPT" and row["arithmetic"] != "NO FAULT OBSERVED", row
