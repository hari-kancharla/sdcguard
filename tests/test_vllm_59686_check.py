"""experiments/vllm_59686_check.py without a GPU: its qualify command, its reading of vLLM's log lines, its step-0
stop and its leaving out the fixed build, and its criteria (docs/METHOD.md, "vLLM issue 59686 in vLLM 0.32.0") on
made-up results."""
import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_59686_check as ex  # noqa: E402


def _row(decision, *why, approximation="WITHIN", consistency="CONSISTENT", isolation="NO LEAK SEEN"):
    return {"decision": decision, "why": list(why), "approximation": approximation, "consistency": consistency,
            "isolation": isolation, "performance": "MEETS"}


def _cell(misrouting=0, control=0):
    return {"sizes": 512, "padded_sizes": 470, "sizes_offsets_wrong": 470 if misrouting else 0,
            "sizes_misrouting": misrouting, "misrouted_routes": 30 * misrouting, "max_misrouted_in_a_size": 40,
            "unpadded_sizes_misrouting": control, "examples": []}


def _env(version, misrouting):
    return {"vllm": version, **{v: {k: _cell(misrouting) for k in ex.TOPKS} for v in ex.VARIANTS}}


def passing():
    names = ("baseline", "again", "cutlass", "nopad", "fixed", "auto")
    return {"step0": {"0.32.0": _env("0.32.0", 300), "fixed": _env("0.32.1rc1.dev431+ge4db46c79", 0)},
            "runs": {n: {"ready": True} for n in names},
            "backends": {"baseline": ["TRITON"], "again": ["TRITON"], "cutlass": ["VLLM_CUTLASS"],
                         "nopad": ["VLLM_CUTLASS"], "fixed": ["VLLM_CUTLASS"], "auto": ["VLLM_CUTLASS"]},
            "graphs": {n: True for n in names},
            "quality": {"again": {"mean_kl": 1e-4}, "cutlass": {"mean_kl": 5e-2}, "nopad": {"mean_kl": 2e-4}},
            "decisions": {"cutlass": _row("REJECT", "approximation: outside the budget", approximation="OUTSIDE"),
                          "nopad": _row("NOT DECIDED", "arithmetic: no plugin"),
                          "fixed": _row("NOT DECIDED", "arithmetic: no plugin"),
                          "auto": _row("REJECT", "approximation: outside the budget", approximation="OUTSIDE")}}


def test_a_caught_bug_with_working_alternatives_passes():
    ok, fails, s = ex.judge(passing())
    assert ok, fails
    assert s["caught"] and s["auto_backend"] == ["VLLM_CUTLASS"] and s["decisions"]["cutlass"]["decision"] == "REJECT"
    assert s["cutlass_over_nopad_mean_kl"] == pytest.approx(250)


def _fails(r, tag):
    ok, fails, _ = ex.judge(r)
    return not ok and any(f.startswith(tag) for f in fails), fails


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing(); r["step0"]["0.32.0"] = _env("0.32.0", 0); cases.append((r, "V"))              # no bug here
    r = passing(); r["step0"]["0.32.0"]["plain"]["top8"] = _cell(0)
    r["step0"]["0.32.0"]["scratch"]["top8"] = _cell(0); cases.append((r, "V"))                   # not at this top-k
    r = passing(); r["step0"]["0.32.0"]["plain"]["top4"]["unpadded_sizes_misrouting"] = 2; cases.append((r, "V"))
    r = passing(); r["step0"]["0.32.0"] = {"error": "ImportError"}; cases.append((r, "V"))
    r = passing(); r["runs"]["nopad"]["ready"] = False; cases.append((r, "V"))
    r = passing(); r["backends"]["cutlass"] = ["TRITON"]; cases.append((r, "V"))                 # not the backend
    r = passing(); r["backends"]["baseline"] = None; cases.append((r, "V"))
    r = passing(); r["graphs"]["cutlass"] = False; cases.append((r, "V"))                       # eager: nothing padded
    r = passing(); r["decisions"]["cutlass"] = _row("NOT DECIDED", "arithmetic: no plugin"); cases.append((r, "C1"))
    r = passing(); r["decisions"]["cutlass"] = _row("REJECT", "performance: slower"); cases.append((r, "C1"))
    r = passing(); r["decisions"]["nopad"] = _row("REJECT", "approximation: outside"); cases.append((r, "C2"))
    r = passing(); del r["decisions"]["fixed"]; cases.append((r, "C2"))
    # NOT DECIDED with the catching check unmeasured is not a working alternative
    r = passing(); r["decisions"]["nopad"] = _row("NOT DECIDED", "approximation: unmeasured",
                                                  approximation="UNMEASURED"); cases.append((r, "C2"))
    r = passing(); r["step0"]["fixed"] = _env("0.32.1rc1.dev431+ge4db46c79", 5); cases.append((r, "C2"))  # not fixed
    r = passing(); r["step0"]["fixed"]["vllm"] = "0.32.0"; cases.append((r, "C2"))               # not the fix
    r = passing(); r["graphs"]["fixed"] = False; cases.append((r, "C2"))
    r = passing(); r["runs"]["fixed"]["ready"] = False; cases.append((r, "C2"))
    for r, tag in cases:
        hit, fails = _fails(r, tag)
        assert hit, (tag, fails)
    r = passing(); r["decisions"]["auto"] = _row("ACCEPT", "x"); r["backends"]["auto"] = ["TRITON"]
    r["graphs"]["auto"] = False
    assert ex.judge(r)[0]                                        # auto is reported, never judged


def test_the_checks_that_caught_cutlass_are_the_ones_held_against_the_alternatives():
    r = passing()
    r["decisions"]["cutlass"] = _row("REJECT", "consistency: differs across batch sizes", consistency="INCONSISTENT")
    r["decisions"]["nopad"] = _row("NOT DECIDED", "arithmetic: no plugin", approximation="UNMEASURED")
    assert ex.judge(r)[0]                                        # only consistency caught it: that one must pass
    r["decisions"]["nopad"]["consistency"] = "UNMEASURED"
    hit, fails = _fails(r, "C2")
    assert hit and "consistency" in " ".join(fails)


def test_the_fixed_build_failing_step0_costs_only_its_own_part():
    r = passing()
    r["step0"]["fixed"] = {"error": "exit 1: ModuleNotFoundError"}
    del r["runs"]["fixed"], r["decisions"]["fixed"]
    ok, fails, _ = ex.judge(r)
    assert not ok and fails and all(f.startswith("C2: fixed") for f in fails), fails   # V and C1 stand


def test_the_qualify_command_sets_each_alternative_apart():
    cmd = ex.qualify_cmd("/tmp/q")
    text = " ".join(cmd)
    assert "--no-plugin" in cmd and "--repeat-baseline" in cmd
    assert f"--baseline {ex.MODEL} {ex.COMMON} --moe-backend triton" in text
    assert f"cutlass={ex.MODEL} {ex.COMMON} --moe-backend cutlass" in cmd
    assert "nopad:VLLM_MOE_SKIP_PADDING=0" in cmd and f"fixed:{ex.FIX_PY} -m vllm.entrypoints.cli.main serve" in cmd
    assert f"auto={ex.MODEL} {ex.COMMON}" in cmd
    without = " ".join(ex.qualify_cmd("/tmp/q", fixed=False))
    assert "fixed" not in without and "cutlass=" in without and "nopad:" in without
    from sdcguard import qualify                                  # the options it uses exist and parse
    with pytest.raises(SystemExit) as e:
        qualify.main([*cmd[4:], "--help"])
    assert e.value.code == 0


def test_backends_and_graphs_are_read_from_vllms_own_log_lines(tmp_path):
    q = tmp_path / "qual"
    q.mkdir()
    (q / "baseline_server.log").write_text("INFO ... Using TRITON Fp8 MoE backend out of potential backends: "
                                           "['AITER', 'TRITON'].\nINFO Graph capturing finished in 31 secs, took "
                                           "0.62 GiB\n")
    (q / "cutlass_server.log").write_text("Using VLLM_CUTLASS Fp8 MoE backend out of potential backends: [...]\n"
                                          "Using VLLM_CUTLASS Fp8 MoE backend out of potential backends: [...]\n")
    b, g = ex.backends(str(q)), ex.graphs(str(q))
    assert b["baseline"] == ["TRITON"] and b["cutlass"] == ["VLLM_CUTLASS"] and b["nopad"] is None
    assert g["baseline"] is True and g["cutlass"] is False and g["nopad"] is None
    (q / "nopad_quality.json").write_text(json.dumps({"summary": {"mean_kl": 2e-4, "top1_agreement": 0.995,
                                                                  "max_prompt_mean_kl": 1e-3, "positions": 900},
                                                      "verdict": {"verdict": "PASS"}}))
    qq = ex.quality(str(q))
    assert qq["nopad"]["verdict"] == "PASS" and qq["nopad"]["mean_kl"] == 2e-4 and qq["cutlass"] is None


def test_step0_stops_the_run_before_any_server_when_it_says_nothing(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ex, "probe", lambda py: _env("0.32.0", 0))
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: seen.append(a) or None)
    res = ex.run(str(tmp_path / "o"))
    assert "skipped" in res and not seen and "misroutes no valid route" in res["skipped"]
    ok, fails, _ = ex.judge(res)
    assert not ok and fails[0].startswith("V: step 0")


def test_a_broken_fixed_build_is_left_out_of_qualify(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ex, "probe", lambda py: _env("0.32.0", 300) if py == sys.executable else {"error": "x"})

    def fake_run(cmd, **k):
        seen.append(cmd)

        class P:
            returncode = 1
        return P()
    monkeypatch.setattr(ex.subprocess, "run", fake_run)
    res = ex.run(str(tmp_path / "o"))
    assert res["fixed_left_out"] and seen and not any(part.startswith(("fixed=", "fixed:")) for part in seen[0])


def test_the_probe_reads_its_own_line(monkeypatch):
    class P:
        returncode, stderr = 0, ""
        stdout = "noise\nPROBE " + json.dumps(_env("0.32.0", 42)) + "\n"
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: P())
    assert ex.probe("python")["plain"]["top8"]["sizes_misrouting"] == 42
    P.stdout = "Traceback: no module"
    assert "error" in ex.probe("python")
    r = copy.deepcopy(passing())
    assert ex.step0_problem(r["step0"]) is None and ex.fixed_problem(r["step0"]) is None


def test_the_probe_script_compiles_and_reads_rows_by_their_own_index():
    compile(ex.PROBE, "probe", "exec")
    assert "MoEPermuteScratch" in ex.PROBE and "unpadded_sizes_misrouting" in ex.PROBE
    assert "range(1, 513)" in ex.PROBE and "list(range(256, 513, 16))" in ex.PROBE


def test_a_c2_failure_on_the_same_check_reads_as_not_separated_never_as_caught():
    r = passing()
    r["decisions"]["nopad"] = _row("REJECT", "approximation: outside", approximation="OUTSIDE")
    ok, fails, sm = ex.judge(r)
    assert not ok and sm["reading"].startswith("not separated: nopad") and sm["caught"]
    ok, fails, sm = ex.judge(passing())
    assert ok and sm["reading"] == "caught, and both alternatives remove it"


def test_a_fixed_server_cut_by_the_time_limit_is_not_run_not_a_failed_criterion():
    r = passing()
    del r["runs"]["fixed"], r["decisions"]["fixed"]
    r.update(qualify_exit=None, qualify_error="timed out after 5100 s")
    ok, fails, sm = ex.judge(r)
    assert ok, fails
    assert sm["not_run_for_time"] == ["fixed"]


def test_step0_and_the_partial_result_are_saved_as_they_come(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "probe", lambda py: _env("0.32.0", 300) if py == sys.executable else {"error": "x"})

    def fake_run(cmd, **k):
        class P:
            returncode = 1
        return P()
    monkeypatch.setattr(ex.subprocess, "run", fake_run)
    out = tmp_path / "o"
    ex.run(str(out))
    assert json.load(open(out / "step0.json"))["0.32.0"]["vllm"] == "0.32.0"
    part = json.load(open(out / "PARTIAL.json"))
    assert part["self_checks"]["baseline"] == {"isolation": None, "consistency": None}
