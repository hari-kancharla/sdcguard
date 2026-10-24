"""experiments/vllm_ready_attrib_check.py's judge, on results made here: each criterion as written in METHOD."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_ready_attrib_check as ra  # noqa: E402


def _v(verdict):
    return [{"verdict": verdict, "exit_code": 0} for _ in range(ra.REPEATS)]


def _good():
    res = {k: _v(v) for k, v in ra.EXPECT.items()}
    res["attribution"] = {
        "single": [{"request": f"cmpl-{i}", "alerts": 1, "named": [[f"cmpl-{i}-0"]],
                    "attribution": ["exact: row 0 of step 9 ..."], "since_includes": True} for i in range(5)],
        "concurrent": {"requests": [f"cmpl-c{i}" for i in range(32)], "alerts": 1, "named": [["cmpl-c3-0"]],
                       "attribution": ["exact"], "since": [f"cmpl-c{i}-0" for i in range(32)]}}
    res["identity_guarded"] = res["identity_disabled"] = ["a", "b"]
    res["exact_concurrent"] = {"marker": 9, "prompts_holding_marker_naturally": [], "trials": [
        {"target": f"cmpl-m{k}", "alerts": 2, "named": [[f"cmpl-m{k}-0"], [f"cmpl-m{k}-0"]],
         "attribution": ["exact"] * 2} for k in range(ra.MARK_TRIALS)]}
    return res


def test_a_run_meeting_every_criterion_passes():
    ok, fails, s = ra.judge(_good())
    assert ok and not fails and s["single_exact"] == "5/5" and s["protected_when_not"] == 0


def test_any_protected_verdict_where_it_isnt_fails():
    r = _good()
    r["R2"][1]["verdict"] = "PROTECTED"
    ok, fails, s = ra.judge(r)
    assert not ok and s["protected_when_not"] == 1 and any(f.startswith("R2") for f in fails)
    r = _good()
    r["R5"] = r["R5"][:2]
    assert not ra.judge(r)[0]


def test_naming_another_request_or_missing_one_fails_but_a_padding_row_does_not():
    r = _good()
    r["attribution"]["single"][2]["named"] = [["cmpl-9-0"]]
    assert not ra.judge(r)[0]
    r = _good()
    r["attribution"]["single"][0].update(alerts=2, named=[[], ["cmpl-0-0"]],
                                         attribution=["row 9 of step 4 is CUDA-graph padding", "exact: ..."])
    assert ra.judge(r)[0]
    r = _good()
    r["attribution"]["concurrent"]["named"] = [["cmpl-zz-0"]]
    assert not ra.judge(r)[0]
    r = _good()
    r["attribution"]["concurrent"]["since"] = r["attribution"]["concurrent"]["since"][:-1]
    assert not ra.judge(r)[0]


def test_changed_outputs_fail():
    r = _good()
    r["identity_disabled"] = ["a", "c"]
    ok, fails, _ = ra.judge(r)
    assert not ok and any(f.startswith("A3") for f in fails)


def test_client_ids_match_vllms_per_prompt_ids():
    assert ra._matches("cmpl-abc-0", "cmpl-abc") and ra._matches("cmpl-abc", "cmpl-abc")
    assert not ra._matches("cmpl-abcd-0", "cmpl-abc")


def test_exact_naming_under_concurrency_needs_the_marked_request_every_time():
    r = _good()
    r["exact_concurrent"]["trials"][2]["named"] = [["cmpl-m2-0"], ["cmpl-other-0"]]
    ok, fails, s = ra.judge(r)
    assert not ok and s["exact_concurrent"] == "4/5" and any(f.startswith("A4") for f in fails)
    r = _good()
    r["exact_concurrent"]["trials"][0].update(alerts=0, named=[])
    assert not ra.judge(r)[0]
    r = _good()
    r["exact_concurrent"]["prompts_holding_marker_naturally"] = [4]
    assert not ra.judge(r)[0]
    r = _good()
    r["exact_concurrent"]["trials"][1]["named"] = [None, ["cmpl-m1-0"]]    # an alert without a captured row
    assert ra.judge(r)[0]
