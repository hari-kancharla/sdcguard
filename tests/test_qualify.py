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
    assert rows["faulty"]["decision"] == "REJECT" and rows["faulty"]["weights"] == "CHANGED"   # a changed weight
    assert rows["faulty"]["arithmetic"] != "FAULT" and any(w.startswith("weights: ") for w in rows["faulty"]["why"])
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


def test_a_rejected_candidate_gets_a_repro_folder_that_reruns_what_failed(tmp_path):
    import subprocess
    out = str(tmp_path / "q")
    serve = f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}"
    qport = _free()
    code = qualify.main([
        "--out", out, "--baseline", "fake/m", "--candidate", "same=fake/m", "--candidate", "shifted=fake/m --shift 4",
        "--candidate", "leaky=fake/m --leak", "--serve-cmd", serve, "--port", str(qport),
        "--metrics-port", str(_free()), "--tokens", "8", "--concurrency", "4", "--max-tokens", "8", "--rounds", "1",
        "--startup-timeout", "30"])
    rows = json.load(open(os.path.join(out, "QUALIFY.json")))["decisions"]
    assert code == 1 and "repro" not in rows["same"]
    shifted, leaky = os.path.join(out, "repro_shifted"), os.path.join(out, "repro_leaky")
    assert rows["shifted"]["repro"] == shifted and rows["leaky"]["repro"] == leaky
    sh = open(os.path.join(shifted, "rerun.sh")).read()
    assert "gate model-check" in sh and "isolation" not in sh and "--shift 4" in sh
    assert "sdcguard perf" not in sh                     # performance isn't required by the policy: not a reason
    assert "isolation --url" in open(os.path.join(leaky, "rerun.sh")).read()
    readme = open(os.path.join(shifted, "README.md")).read()
    assert "approximation" in readme and "rerun.sh" in readme and "decision_shifted.json" in readme
    assert os.access(os.path.join(shifted, "rerun.sh"), os.X_OK) and os.path.exists(os.path.join(shifted, "ref.json"))
    assert "repro_<name>/" in open(os.path.join(out, "REPORT.md")).read()
    # the rerun reproduces the failure on its own
    port = _free()
    env = dict(os.environ, PORT=str(port), MPORT=str(_free()), PY=sys.executable,
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run(["sh", os.path.join(shifted, "rerun.sh")], env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode in (0, 1), p.stdout[-2000:] + p.stderr[-2000:]
    q = json.load(open(os.path.join(shifted, "rerun_quality.json")))
    assert q["verdict"]["verdict"] == "FAIL"
    import socket                                         # and the server it started is stopped
    with socket.socket() as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0


def test_a_required_speed_target_missed_reruns_the_same_load_and_only_that(tmp_path):
    import subprocess
    from types import SimpleNamespace as NS
    out = str(tmp_path)
    with open(os.path.join(out, "baseline_perf.json"), "w") as f:
        f.write("{}")
    pf = tmp_path / "load.txt"
    pf.write_text("a\nb\n")
    a = NS(serve_cmd=["vllm", "serve"], share="1", env_map={"VLLM_X": "a b"}, no_plugin=False, no_isolation=False,
           concurrency=8, max_tokens=64, rounds=2, prompts_file=str(pf))
    row = {"decision": "REJECT", "approximation": "WITHIN", "isolation": "NO LEAK SEEN",
           "arithmetic": "NO FAULT OBSERVED", "performance": "MISSES", "performance_required": True,
           "why": ["performance: throughput -12%"]}
    d = qualify.write_repro(out, "slow", {"baseline": "m", "slow": "m --x"}, {}, row, a)
    path = os.path.join(d, "rerun.sh")
    sh = open(path).read()
    assert "--concurrency 8 --max-tokens 64 --rounds 2" in sh and '"$HERE/perf_prompts.txt"' in sh
    assert "VLLM_X='a b'" in sh and "SDCGUARD_SHARE=1" in sh and "m --x --port $PORT" in sh
    assert "gate model-check" not in sh and "isolation --url" not in sh
    assert os.path.exists(os.path.join(d, "perf_prompts.txt")) and os.path.exists(os.path.join(d, "baseline_perf.json"))
    assert subprocess.run(["sh", "-n", path]).returncode == 0
    row["performance_required"] = False                  # a speed target the policy doesn't require isn't a reason
    d2 = qualify.write_repro(out, "slow2", {"baseline": "m", "slow2": "m"}, {}, row, a)
    assert "sdcguard perf" not in open(os.path.join(d2, "rerun.sh")).read()
    assert "nothing to rerun" in open(os.path.join(d2, "rerun.sh")).read()


def test_your_own_prompts_make_the_quality_reference(tmp_path):
    out = str(tmp_path / "q")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("What is our refund policy?\nSummarize this ticket.\n\nTranslate: hello\n")
    qualify.main([
        "--out", out, "--baseline", "fake/m", "--candidate", "same=fake/m", "--quality-prompts", str(prompts),
        "--no-isolation", "--no-perf", "--serve-cmd", f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}",
        "--port", str(_free()), "--metrics-port", str(_free()), "--tokens", "8", "--startup-timeout", "30"])
    ref = json.load(open(os.path.join(out, "ref.json")))
    assert len(ref["records"]) == 3
