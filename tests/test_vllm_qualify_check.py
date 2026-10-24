"""experiments/vllm_qualify_check.py without a GPU: the command it runs and its criteria (docs/METHOD.md, "Qualify on
a real server") on made-up results."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_qualify_check as vq  # noqa: E402


def passing():
    ok_run = {"ready": True}
    row = {"decision": "ACCEPT", "arithmetic": "NO FAULT OBSERVED", "approximation": "WITHIN",
           "isolation": "NO LEAK SEEN", "performance": "MEETS", "why": []}
    return {"exit_code": 1, "report_exists": True,
            "runs": {n: dict(ok_run) for n in ("baseline", "again", "same", "triton", "badeps")},
            "alerts": {n: 0 for n in ("baseline", "again", "same", "triton", "badeps")},
            "decisions": {"same": dict(row), "triton": dict(row),
                          "badeps": {**row, "decision": "REJECT", "approximation": "OUTSIDE",
                                     "why": ["approximation: mean KL 11"]}}}


def test_the_configurations_are_what_vllm_serve_takes():
    import shlex
    for cfg in vq.CANDIDATES.values():
        args = shlex.split(cfg)
        assert args[0] == vq.MODEL
    bad = shlex.split(vq.CANDIDATES["badeps"])
    assert bad[bad.index("--hf-overrides") + 1] == '{"rms_norm_eps": 1.0}'


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = vq.judge(passing())
    assert ok, fails
    assert s["triton"] == "ACCEPT"


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing()
    r["runs"]["again"]["ready"] = False
    cases.append((r, "Q1: again didn't start"))
    r = passing()
    r["exit_code"] = 2
    cases.append((r, "qualify exited 2"))
    r = passing()
    r["decisions"]["same"]["decision"] = "NOT DECIDED"
    r["decisions"]["same"]["arithmetic"] = "INCONCLUSIVE"
    cases.append((r, "Q2: same is NOT DECIDED"))
    r = passing()
    r["decisions"]["badeps"]["decision"] = "ACCEPT"
    cases.append((r, "Q3: badeps is ACCEPT"))
    r = passing()
    r["decisions"]["badeps"]["arithmetic"] = "FAULT"
    cases.append((r, "Q3: badeps is REJECT"))
    r = passing()
    r["alerts"]["triton"] = 2
    cases.append((r, "Q4: alerts with no fault"))
    for res, want in cases:
        ok, fails, _ = vq.judge(res)
        assert not ok and any(want in f for f in fails), (want, fails)


def test_alerts_are_counted_from_each_servers_log(tmp_path):
    p = tmp_path / "same_alerts.jsonl"
    p.write_text('{"detector": "serve_norm"}\n\n{"detector": "serve_checksum"}\n')
    assert vq._count(str(p)) == 2 and vq._count(str(tmp_path / "none.jsonl")) == 0
