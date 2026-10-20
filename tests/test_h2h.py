"""The head-to-head's parsing and pre-registered classification (experiments/h2h_run.py), on CPU.

The GPU run itself is scripts/modal_h2h.py; these tests pin down how each tool's output is read and how each
(tool, share) cell is classified, so the rules can't drift after results come in.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import h2h_run as h  # noqa: E402

PROBE_OUT = """\
{"case": "scrub", "allocated_bytes": 75000000000, "sampled_bytes": 268435456, "nonzero_bytes": 0, "fresh_memory_zeroed": true}
{"case": "hgemm", "n": 2048, "source": "gpu", "cublas_version": 130100, "elements": 4194304, "unwritten": 491520, "first_unwritten_index": 1048576, "nonfinite_written": 0, "sampled": 65536, "wrong_sampled": 7680, "worst_rel_err_finite": 0.00033, "gemm_ms": 1.2, "correct": false}
{"case": "sgemm", "n": 2048, "source": "gpu", "cublas_version": 130100, "elements": 4194304, "unwritten": 0, "first_unwritten_index": -1, "nonfinite_written": 0, "sampled": 65536, "wrong_sampled": 0, "worst_rel_err_finite": 4e-08, "gemm_ms": 1.0, "correct": true}
{"case": "sgemmtc", "n": 2048, "error": "cublasSgemm", "code": 13}
{"case": "sgemm", "n": 8192, "source": "gpu", "cublas_version": 130100, "elements": 67108864, "unwritten": 0, "first_unwritten_index": -1, "nonfinite_written": 0, "sampled": 262144, "wrong_sampled": 0, "worst_rel_err_finite": 4e-08, "gemm_ms": 8.0, "correct": true}
{"case": "sgemmtc", "n": 8192, "source": "gpu", "cublas_version": 130100, "elements": 67108864, "unwritten": 0, "first_unwritten_index": -1, "nonfinite_written": 0, "sampled": 262144, "wrong_sampled": 0, "worst_rel_err_finite": 0.0003, "gemm_ms": 9.0, "correct": true}
"""

DCGM_FAIL = {"DCGM Diagnostic": {"version": "4.7.0", "test_categories": [
    {"category": "Deployment", "tests": [{"name": "software", "results": [{"status": "Pass"}]}]},
    {"category": "Hardware", "tests": [{"name": "diagnostic", "results": [
        {"entity_group": "GPU", "entity_id": 0, "status": "Fail",
         "warnings": [{"warning": "Found 491520 faulty memory elements on GPU 0", "error_id": 52}]}],
        "test_summary": {"status": "Fail"}}]}]}}
DCGM_PASS = {"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [
    {"name": "Diagnostic", "results": [{"entity_group": "GPU", "entity_id": 0, "status": "Pass"}],
     "test_summary": {"status": "Pass"}}]}]}}
DCGM_SKIP = {"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [
    {"name": "diagnostic", "results": [{"entity_group": "GPU", "entity_id": 0, "status": "Skip"}]}]}]}}
DCGM_OTHER_FAIL = {"DCGM Diagnostic": {"test_categories": [{"category": "Hardware", "tests": [
    {"name": "diagnostic", "results": [{"status": "Fail", "warnings": [
        {"warning": "cuBLAS returned an error", "error_id": 13}]}]}]}]}}
DCGM_RUNTIME = {"DCGM Diagnostic": {"version": "4.7.0", "runtime_error": "Unable to initialize NVML"}}

BURN_OK = "Burning for 60 seconds.\n10.0%  proc'd: 400 (38000 Gflop/s)   errors: 0   temps: 41 C\n\nTested 1 GPUs:\n\tGPU 0: OK\n"
BURN_BAD = "100.0%  proc'd: 900 (38000 Gflop/s)   errors: 12 (WARNING!)  temps: 50 C\nTested 1 GPUs:\n\tGPU 0: FAULTY\n"


def test_probe_parsing_and_exercise():
    p = h.parse_probe(PROBE_OUT)
    assert p["scrub"]["fresh_memory_zeroed"] is True
    assert p["hgemm:2048"]["unwritten"] == 491520
    assert h.probe_wrong(p, ("hgemm:2048", "sgemm:2048")) is True
    assert h.probe_wrong(p, ("sgemm:2048",)) is False
    assert h.probe_wrong(p, ("sgemmtc:2048",)) is None              # the call errored: ground truth unknown
    assert h.probe_wrong(p, ("dgemm:8192",)) is None                # not run
    assert h.probe_wrong(p, ("sgemm:2048", "sgemmtc:8192")) is False


def test_dcgm_parsing():
    f = h.parse_dcgm(json.dumps(DCGM_FAIL), 1)
    assert f["status"] == "fail" and 52 in f["codes"] and f["faulty_memory_elements"] == 491520
    assert h.tool_flag("dcgm", f) == "flagged"
    p = h.parse_dcgm(json.dumps(DCGM_PASS), 0)
    assert p["status"] == "pass" and h.tool_flag("dcgm", p) == "passed"
    s = h.parse_dcgm(json.dumps(DCGM_SKIP), 0)
    assert s["status"] == "skip" and h.tool_flag("dcgm", s) == "could_not_test"
    o = h.parse_dcgm(json.dumps(DCGM_OTHER_FAIL), 1)
    assert o["status"] == "fail" and h.tool_flag("dcgm", o) == "not_attributable"
    r = h.parse_dcgm(json.dumps(DCGM_RUNTIME), 226)
    assert r["status"] == "error" and h.tool_flag("dcgm", r) == "could_not_test"
    t = h.parse_dcgm("", None, timed_out=True)
    assert t["status"] == "timeout" and h.tool_flag("dcgm", t) == "could_not_test"
    # the software plugin passing must not make a failing diagnostic look like a pass
    assert h.parse_dcgm(json.dumps(DCGM_FAIL), 1)["status"] == "fail"
    # pretty-printed JSON spanning many lines
    assert h.parse_dcgm(json.dumps(DCGM_PASS, indent=2), 0)["status"] == "pass"


def test_burn_parsing():
    ok = h.parse_burn(BURN_OK)
    assert ok == {"verdict": "OK", "max_errors": 0} and h.tool_flag("gpu_burn_tc", ok) == "passed"
    bad = h.parse_burn(BURN_BAD)
    assert bad["verdict"] == "FAULTY" and bad["max_errors"] == 12 and h.tool_flag("gpu_burn_tc", bad) == "flagged"
    died = h.parse_burn("GPU 0: DIED")
    assert died["verdict"] == "DIED" and h.tool_flag("gpu_burn_tc", died) == "could_not_test"
    assert h.parse_burn("", timed_out=True)["verdict"] == "timeout"


def test_check_parsing(tmp_path):
    card = json.loads((ROOT / "docs" / "board" / "seed" / "modal_h100_a_45b2fbe6.json").read_text())
    c = h.parse_check("progress line\n" + json.dumps(card) + "\n")
    assert c["result"] == "PASS" and h.tool_flag("sdcguard_check", c) == "passed"
    bad = dict(card, result="FAIL", wrong=dict(card["wrong"], math_modes=24))
    assert h.tool_flag("sdcguard_check", h.parse_check(json.dumps(bad))) == "flagged"
    mem_only = dict(card, result="FAIL", wrong=dict(card["wrong"], memory_words=3))
    assert h.tool_flag("sdcguard_check", h.parse_check(json.dumps(mem_only))) == "not_attributable"
    assert h.tool_flag("sdcguard_check", {"result": "SUSPECT", "wrong": {}}) == "not_attributable"
    assert h.tool_flag("sdcguard_check", {"result": "INCOMPLETE", "wrong": {}}) == "could_not_test"
    # falls back to the result.json the check wrote when stdout has no JSON
    d = tmp_path / "check" / "GPU-x"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(bad))
    assert h.parse_check("no json here", tmp_path / "check")["result"] == "FAIL"


def _oct19_runs():
    data = json.loads((ROOT / "results" / "mps_bug" / "h200_20261019_101053.json").read_text())
    return {str(r["share"]): r for r in data["runs"] if "share" in r}


def test_ground_truth_on_the_oct19_h200_runs():
    runs = _oct19_runs()
    for share, expect in (("100", False), ("50", False), ("40", True), ("14", True), ("75", True)):
        g = h.parse_ground(json.dumps(runs[share]))
        assert g["sms_reported"] in (132, 66, 52, 18, 98)
        assert h.ground_wrong(g) is expect, (share, g)
    # at 20% the layer and the math modes were right on Oct 19 (the alert hit the check's own matmul)
    g20 = h.parse_ground(json.dumps(runs["20"]))
    assert h.ground_wrong(g20) is False
    assert any((a or {}).get("training") for a in g20["in_job_alerts"].values())


@pytest.mark.parametrize("flag,exercised,cell", [
    ("flagged", True, h.DETECTED), ("passed", True, h.MISSED), ("passed", False, h.NOT_EXERCISED),
    ("flagged", False, h.FALSE_ALARM), ("could_not_test", True, h.COULD_NOT_TEST),
    ("not_attributable", True, h.NOT_ATTRIBUTABLE), ("flagged", None, h.FLAGGED_NO_TRUTH),
    ("passed", None, h.PASSED_NO_TRUTH)])
def test_classification_table(flag, exercised, cell):
    assert h.classify(flag, exercised) == cell


def test_cells_end_to_end():
    runs = _oct19_runs()
    probe = h.parse_probe(PROBE_OUT)
    res = {"shares": {
        "40": {"probe_parsed": probe, "ground_parsed": h.parse_ground(json.dumps(runs["40"])),
               "dcgm_parsed": h.parse_dcgm(json.dumps(DCGM_PASS), 0),
               "gpu_burn_tc_parsed": h.parse_burn(BURN_OK),
               "sdcguard_check_parsed": {"result": "FAIL", "wrong": {"math_modes": 72, "stress": 0}}},
        "100": {"probe_parsed": {k: dict(v, correct=True) for k, v in probe.items() if k != "sgemmtc:2048"},
                "ground_parsed": h.parse_ground(json.dumps(runs["100"])),
                "gpu_burn_tc_parsed": h.parse_burn(BURN_OK),
                "sdcguard_check_parsed": {"result": "PASS", "wrong": {}}}},
        "extras_40": {"gpu_burn_default_parsed": h.parse_burn(BURN_OK)}}
    cells = {(r["share"], r["tool"]): r["cell"] for r in h.cells(res)}
    assert cells[(40, "dcgm")] == h.MISSED                  # its Hgemm was wrong (probe) and it passed
    assert cells[(40, "gpu_burn_tc")] == h.NOT_EXERCISED    # its 8192 tensor-op Sgemm was right
    assert cells[(40, "sdcguard_check")] == h.DETECTED
    assert cells[(40, "gpu_burn_default")] == h.NOT_EXERCISED
    assert cells[(100, "dcgm")] == h.COULD_NOT_TEST         # not run (as when DCGM can't start)
    assert cells[(100, "sdcguard_check")] == h.NOT_EXERCISED
    md = h.markdown(dict(res, cells=h.cells(res)))
    assert "MISSED" in md and "DETECTED" in md and "\u2014" not in md


def test_lt_log_summary(tmp_path):
    log = tmp_path / "lt.log"
    log.write_text(
        "[2026-10-19 10:00:00][cublasLt][1][Api][cublasLtMatmul] A=0x7f00aa Adesc=[type=R_16F rows=1000 cols=2048] "
        "computeDesc=[computeType=COMPUTE_32F scaleType=R_32F smCountTarget=18] algo=[algoId=66 tile=MATMUL_TILE_64x16 "
        "stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_4x1x1]\n"
        "[2026-10-19 10:00:01][cublasLt][1][Api][cublasLtMatmul] A=0x7f00bb Adesc=[type=R_16F rows=1000 cols=2048] "
        "computeDesc=[computeType=COMPUTE_32F scaleType=R_32F smCountTarget=18] algo=[algoId=66 tile=MATMUL_TILE_64x16 "
        "stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_4x1x1]\n"
        "[2026-10-19 10:00:02][cublasLt][1][Api][cublasLtMatmul] algo=[algoId=21 clusterShape=CLUSTER_SHAPE_1x1x1]\n"
        "unrelated line\n")
    s = h.summarize_lt_log(log)
    assert s["lines"] == 3 and s["unique"][0]["count"] == 2 and s["algo66_cluster4plus"] == 2
    assert h.summarize_lt_log(tmp_path / "missing.log")["lines"] == 0


def test_run_step_kills_the_whole_group_at_its_limit(tmp_path):
    rec = h.run_step("sleeper", [sys.executable, "-c", "import subprocess, sys, time; "
                                 "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                                 "print('started', flush=True); time.sleep(60)"], tmp_path, budget=2.0, floor=1.0)
    assert rec["timed_out"] is True and rec["status"] == "timed out" and "started" in rec["stdout"]
    assert h.run_step("x", ["true"], tmp_path, budget=5.0, floor=20.0)["status"].startswith("not started")


def test_files_say_what_runs_and_cost_without_dashes():
    for f in ("experiments/h2h_run.py", "experiments/h2h_probe.cpp", "scripts/modal_h2h.py"):
        text = (ROOT / f).read_text()
        assert "\u2014" not in text and "\u2013" not in text, f
    assert "about $5" in (ROOT / "scripts" / "modal_h2h.py").read_text()


@pytest.mark.skipif(os.environ.get("H2H_PROBE_BIN") is None, reason="set H2H_PROBE_BIN to a built probe")
def test_probe_cpu_selfcheck():
    import subprocess
    out = subprocess.run([os.environ["H2H_PROBE_BIN"], "cpu:hgemm:128", "cpu:hgemm:128:hole", "cpu:exbf16:64:hole"],
                         capture_output=True, text=True, timeout=60).stdout.splitlines()
    rows = [json.loads(x) for x in out]
    assert rows[0]["correct"] is True and rows[1]["unwritten"] == 128 * 128 // 4 and rows[2]["correct"] is False


def _script(path: Path, body: str):
    path.write_text("#!" + sys.executable + "\n" + body)
    path.chmod(0o755)


def test_run_all_with_stand_in_tools(tmp_path, monkeypatch):
    """The whole orchestration with stand-ins for nvidia-smi, MPS, DCGM, gpu-burn, the probe, the ground-truth
    script and sdcguard check: every share and extra runs, progress is saved, and the cells come out as the rules
    say. (A crash halfway through a paid run would lose its money; this is the dress rehearsal.)"""
    bin_, repo, burn, pkg = tmp_path / "bin", tmp_path / "repo", tmp_path / "burn", tmp_path / "pkg"
    for d in (bin_, repo / "experiments", burn, pkg / "sdcguard"):
        d.mkdir(parents=True)
    share = "import os\nS = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')\n"
    _script(bin_ / "nvidia-smi", "print('name, uuid\\nNVIDIA H200, GPU-fake')\n")
    _script(bin_ / "nvidia-cuda-mps-control", "import sys\nif '-d' not in sys.argv: print(sys.stdin.read() and '40')\n")
    _script(bin_ / "dcgmi", share + "import sys, json\n"
            f"FAIL = {json.dumps(DCGM_FAIL)!r}\nPASS = {json.dumps(DCGM_PASS)!r}\n"
            "if '--version' in sys.argv: print('dcgmi version: 4.7.0'); sys.exit(0)\n"
            "print(FAIL if S == '40' else PASS); sys.exit(1 if S == '40' else 0)\n")
    good = "\n".join(json.dumps(dict({k: v for k, v in json.loads(x).items() if k not in ("error", "code")},
                                      correct=True, unwritten=0)) for x in PROBE_OUT.splitlines())
    _script(tmp_path / "probe", share + f"BAD = {PROBE_OUT!r}\nGOOD = {good!r}\nprint(BAD if S == '40' else GOOD)\n")
    _script(burn / "gpu_burn", f"print({BURN_OK!r})\n")
    runs = _oct19_runs()
    _script(repo / "experiments" / "mps_unwritten.py", share + "import json\n"
            f"R = {json.dumps({k: runs[k] for k in ('100', '40', '75', '14')})!r}\n"
            "print(json.dumps(json.loads(R)[S]))\n")
    card = json.loads((ROOT / "docs" / "board" / "seed" / "modal_h100_a_45b2fbe6.json").read_text())
    bad = dict(card, result="FAIL", wrong=dict(card["wrong"], math_modes=24))
    (pkg / "sdcguard" / "__init__.py").write_text("")
    (pkg / "sdcguard" / "check.py").write_text(share + "import json\n"
                                               f"print(json.dumps({bad!r} if S in ('40', '75') else {card!r}))\n")
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setenv("PYTHONPATH", str(pkg))
    monkeypatch.setattr(h, "MPS_SETTLE", (0.0, 0.0))
    monkeypatch.setattr(h, "MPS_PIPE", str(tmp_path / "pipe"))
    monkeypatch.setattr(h, "MPS_LOG", str(tmp_path / "mpslog"))
    commits = []
    res = h.run_all(str(tmp_path / "out"), deadline_s=900, commit=lambda: commits.append(1), repo=str(repo),
                    probe=str(tmp_path / "probe"), burn_dir=str(burn))
    assert set(res["shares"]) == {"100", "40", "75", "14"} and res["setup"]["dcgm_smoke"]["usable"]
    assert "dcgm_fp16_only_parsed" in res["extras_40"] and "gpu_burn_default_parsed" in res["extras_40"]
    assert res["recheck_100"]["probe_parsed"] and len(commits) > 20
    assert (tmp_path / "out" / "progress.json").exists() and (tmp_path / "out" / "share_40" / "dcgm.stdout.txt").exists()
    cells = {(r["share"], r["tool"]): r["cell"] for r in res["cells"] if "flag" in r}
    assert cells[(40, "dcgm")] == h.DETECTED and cells[(40, "sdcguard_check")] == h.DETECTED
    assert cells[(40, "gpu_burn_tc")] == h.NOT_EXERCISED and cells[(100, "dcgm")] == h.NOT_EXERCISED
    assert cells[(100, "sdcguard_check")] == h.NOT_EXERCISED
    assert cells[(75, "sdcguard_check")] == h.DETECTED      # PyTorch's own matmuls were wrong at 75% on Oct 19
    assert cells[(40, "dcgm_fp16_only")] == h.DETECTED
    assert "DETECTED" in h.markdown(res)


def _lt_rows(share: str) -> str:
    """Probe output for the cuBLASLt cases: algorithm 66 with a 4-CTA cluster wrong below 100%, as published."""
    sms = {"100": 132, "40": 53, "14": 18}[share]
    rows = []
    for case in ("lt16", "ltbf16"):
        for i, (algo, shape, ctas) in enumerate(((66, 4, 4), (66, 3, 2), (21, 2, 1))):
            bad = share != "100" and ctas >= 4
            rows.append({"case": case, "n": f"1000x2048x16#{i}", "variant": "", "algo_id": algo,
                         "cluster_shape_id": shape, "cluster_ctas": ctas, "sms_visible": sms,
                         "cublaslt_version": 130101, "unwritten": 11000 if bad else 0, "wrong_sampled": 300 if bad else 0,
                         "status": 0, "cuda_error": 0, "correct": not bad})
        rows.append({"case": case, "n": "1000x2048x16#0w", "variant": "cluster_1x1x1", "algo_id": 66,
                     "cluster_shape_id": 2, "cluster_ctas": 1, "sms_visible": sms, "correct": True})
    return "\n".join(json.dumps(r) for r in rows)


def test_lt_summary_matches_the_published_description_or_says_not():
    s = h.lt_summary(h.parse_probe(_lt_rows("40")))
    assert s["algorithms_run"] == 6 and s["wrong"] == 2 and s["all_wrong_are_algo66_cluster4plus"]
    assert s["wrong_with_1_or_2_cta_clusters"] == 0 and s["workaround_runs"] == 2 and s["workaround_wrong"] == 0
    assert s["sms_visible"] == [53]
    other = [json.loads(x) for x in _lt_rows("40").splitlines()]
    for r in other:
        if r["cluster_ctas"] == 2:
            r["correct"] = False                         # a 2-CTA cluster wrong: not what NVIDIA describes
    s2 = h.lt_summary(h.parse_probe("\n".join(json.dumps(r) for r in other)))
    assert not s2["all_wrong_are_algo66_cluster4plus"] and s2["wrong_with_1_or_2_cta_clusters"] == 2
    assert h.lt_summary({})["wrong"] == 0 and not h.lt_summary({})["all_wrong_are_algo66_cluster4plus"]


def test_a_share_without_mps_is_could_not_test_for_every_tool():
    rows = h.cells({"shares": {"40": {"mps_started": False}}})
    assert {(r["tool"], r["cell"]) for r in rows} == {(t, h.COULD_NOT_TEST)
                                                      for t in ("dcgm", "gpu_burn_tc", "sdcguard_check")}


def _preflight_tools(tmp_path, dcgm_connects: bool):
    bin_, burn = tmp_path / "bin", tmp_path / "burn"
    for d in (bin_, burn):
        d.mkdir(parents=True, exist_ok=True)
    share = "import os\nS = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')\n"
    connect = ("L = os.environ.get('CUDA_MPS_LOG_DIRECTORY')\n"
               "if L and 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE' in os.environ:\n"
               "    os.makedirs(L, exist_ok=True)\n"
               "    open(os.path.join(L, 'control.log'), 'a').write('[x] Control 8] Accepting connection...\\n')\n")
    _script(bin_ / "nvidia-smi", "print('name, uuid\\nNVIDIA H200, GPU-fake')\n")
    _script(bin_ / "nvidia-cuda-mps-control", "import sys\nif '-d' not in sys.argv: print(sys.stdin.read() and '40')\n")
    _script(bin_ / "dcgmi", share + (connect if dcgm_connects else "") + "import sys, json\n"
            f"PASS = {json.dumps(DCGM_PASS)!r}\n"
            "if '--version' in sys.argv: print('dcgmi version: 4.7.0'); sys.exit(0)\nprint(PASS)\n")
    rows = {s: _lt_rows(s) for s in ("100", "40", "14")}
    _script(tmp_path / "probe", share + f"R = {rows!r}\nprint(R[S])\n")
    _script(burn / "gpu_burn", share + connect + f"print({BURN_OK!r})\n")
    return bin_, burn


@pytest.mark.parametrize("dcgm_connects", [True, False])
def test_preflight_with_stand_in_tools(tmp_path, monkeypatch, dcgm_connects):
    bin_, burn = _preflight_tools(tmp_path, dcgm_connects)
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setattr(h, "MPS_SETTLE", (0.0, 0.0))
    monkeypatch.setattr(h, "MPS_PIPE", str(tmp_path / "pipe"))
    monkeypatch.setattr(h, "MPS_LOG", str(tmp_path / "mpslog"))
    res = h.preflight(str(tmp_path / "out"), deadline_s=600, probe=str(tmp_path / "probe"), burn_dir=str(burn))
    d = res["decisions"]
    assert d["go: the full head-to-head can answer its question"] is True, (d, {k: (v.get("mps_started"), v.get(
        "algorithms")) for k, v in res["shares"].items()}, res["notes"], res["setup"].get("nvidia_smi"))
    assert d["bug reproduced as published (wrong only with algorithm 66 and 4+ CTAs per cluster)"] is True
    assert d["documented workaround (1x1x1 cluster) gives right results"] is True
    assert d["gpu-burn runs under MPS with a readable verdict"] is True
    assert d["DCGM connected as an MPS client (new MPS connections during its run)"] is dcgm_connects
    assert d["DCGM cells will be meaningful"] is dcgm_connects
    assert (tmp_path / "out" / "preflight.json").exists()
    assert (tmp_path / "out" / "share_40" / "dcgm_under_mps.stdout.txt").exists()


def test_preflight_says_no_go_when_the_bug_does_not_look_as_published(tmp_path, monkeypatch):
    bin_, burn = _preflight_tools(tmp_path, True)
    rows = {s: _lt_rows("100").replace('"sms_visible": 132', '"sms_visible": ' + str({"100": 132, "40": 53,
                                                                                         "14": 18}[s]))
            for s in ("100", "40", "14")}                 # nothing goes wrong at any share
    _script(tmp_path / "probe", "import os\nS = os.environ.get('CUDA_MPS_ACTIVE_THREAD_PERCENTAGE', '100')\n"
            f"R = {rows!r}\nprint(R[S])\n")
    monkeypatch.setenv("PATH", f"{bin_}:{os.environ['PATH']}")
    monkeypatch.setattr(h, "MPS_SETTLE", (0.0, 0.0))
    monkeypatch.setattr(h, "MPS_PIPE", str(tmp_path / "pipe"))
    monkeypatch.setattr(h, "MPS_LOG", str(tmp_path / "mpslog"))
    d = h.preflight(str(tmp_path / "out"), deadline_s=600, probe=str(tmp_path / "probe"), burn_dir=str(burn))["decisions"]
    assert d["bug reproduced as published (wrong only with algorithm 66 and 4+ CTAs per cluster)"] is False
    assert d["go: the full head-to-head can answer its question"] is False


def test_dcgm_outside_mps_is_not_exercised_rather_than_missed():
    probe = h.parse_probe(PROBE_OUT)                      # its Hgemm was wrong as an MPS client
    base = {"probe_parsed": probe, "ground_parsed": {}, "dcgm_parsed": h.parse_dcgm(json.dumps(DCGM_PASS), 0),
            "gpu_burn_tc_parsed": h.parse_burn(BURN_OK), "sdcguard_check_parsed": {"result": "PASS", "wrong": {}}}
    seen = dict(base, mps_clients={"dcgm": {"connections": 0}, "gpu_burn_tc": {"connections": 1},
                                   "sdcguard_check": {"connections": 2}})
    cells = {r["tool"]: r for r in h.cells({"shares": {"40": seen}})}
    assert cells["dcgm"]["cell"] == h.NOT_EXERCISED and "not an MPS client" in cells["dcgm"]["exercised_note"]
    # no evidence either way (the logs show no connection for known clients either): the probe decides
    blind = dict(base, mps_clients={"dcgm": {"connections": 0}, "gpu_burn_tc": {"connections": 0}})
    assert {r["tool"]: r["cell"] for r in h.cells({"shares": {"40": blind}})}["dcgm"] == h.MISSED
    connected = dict(base, mps_clients={"dcgm": {"connections": 1}, "gpu_burn_tc": {"connections": 1}})
    assert {r["tool"]: r["cell"] for r in h.cells({"shares": {"40": connected}})}["dcgm"] == h.MISSED
