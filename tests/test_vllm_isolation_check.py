"""experiments/vllm_isolation_check.py without a GPU: its criteria (docs/METHOD.md) on files like the run writes."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_isolation_check as ic  # noqa: E402


def _iso(verdict="NO LEAK SEEN", primer="NO LEAK SEEN", leaks=()):
    return {"verdict": verdict, "why": ["primer dependence: 88 of 192 changed"] if primer == "LEAK" else ["ok"],
            "primer": {"verdict": primer, "pairs": 192, "pairs_differing": 88 if primer == "LEAK" else 0,
                       "judged": 16, "cases": [{}] * 16},
            "canary": {"verdict": "LEAK" if leaks else "NO LEAK SEEN", "answered": 48, "leaks": list(leaks)}}


def _run(tmp_path, fp8=None, eager=None, bf16=None):
    out = str(tmp_path)
    q = {"verdict": {"verdict": "WARN", "why": "x"}, "summary": {"mean_kl": 3e-3, "top1_agreement": 0.98}}
    for role, iso in (("fp8pt", fp8 or _iso("LEAK", "LEAK")), ("eager", eager or _iso()), ("bf16", bf16 or _iso())):
        json.dump(iso, open(os.path.join(out, f"{ic.FILE[role]}_isolation.json"), "w"))
        if role != "bf16":
            json.dump(q, open(os.path.join(out, f"{role}_quality.json"), "w"))
    res = {r: {"ready": True} for r in ic.ROLES}
    from sdcguard import qualify
    res["decisions"] = qualify.decide_all(out, ["fp8pt", "eager"], ic.POLICY, plugin=False)
    return res, out


def test_the_reported_bug_seen_and_both_controls_clean_meets_the_criteria(tmp_path):
    res, out = _run(tmp_path)
    ok, fails, s = ic.judge(res, out)
    assert ok, fails
    assert s["decisions"]["fp8pt"]["decision"] == "REJECT" and s["decisions"]["eager"]["decision"] == "NOT DECIDED"
    assert s["fp8pt"]["primer_pairs_differing"] == 88


def test_the_bug_not_showing_fails_criterion_2(tmp_path):
    res, out = _run(tmp_path, fp8=_iso())
    ok, fails, _ = ic.judge(res, out)
    assert not ok and any(f.startswith("2:") for f in fails) and any(f.startswith("5:") for f in fails)


def test_a_leak_on_a_control_fails(tmp_path):
    res, out = _run(tmp_path, eager=_iso("LEAK", "LEAK"))
    ok, fails, _ = ic.judge(res, out)
    assert not ok and any(f.startswith("3:") for f in fails)
    res, out = _run(tmp_path, bf16=_iso("LEAK", leaks=[{"request": 1, "found_code_of": 2}]))
    ok, fails, _ = ic.judge(res, out)
    assert any(f.startswith("4:") for f in fails)


def test_the_configurations_keep_the_compilation_json_in_one_argument():
    import shlex
    args = shlex.split(ic.config("eager"))
    i = args.index("--compilation-config")
    import json as j
    assert j.loads(args[i + 1])["cudagraph_capture_sizes"] == [1, 2, 4, 8]
    assert args[-1] == "--enforce-eager" and "fp8_per_tensor" in args
