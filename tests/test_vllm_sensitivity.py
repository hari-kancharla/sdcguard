"""experiments/vllm_sensitivity.py without a GPU: its trials, its summary by bit and its criteria."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_sensitivity as vsn  # noqa: E402


def test_the_trials_cover_every_bit_the_same_way_every_time():
    t = vsn.trials()
    assert len(t) == 16 * vsn.PER_BIT and t == vsn.trials()
    assert {x["bit"] for x in t} == set(range(16)) and all(x["layer"].startswith("model.layers.") for x in t)


def _row(bit, detected, verdict, ok_back=True):
    flip = {"before": 0.01, "after": 0.02}
    return {"bit": bit, "flip": flip, "back": {"after": 0.01 if ok_back else 0.02}, "detected": detected,
            "alerts_other_layers": 0, "quality": {"verdict": verdict}}


def test_the_summary_counts_detected_changed_and_missed():
    res = {"ready": True, "clean_alerts": 0, "noise_quality": {"verdict": "PASS"},
           "trials": [_row(14, True, "FAIL"), _row(14, True, "FAIL"), _row(3, False, "PASS"), _row(9, False, "WARN"),
                      {"bit": 5, "flip": {}, "back": {}, "detected": False, "alerts_other_layers": 0,
                       "quality": {}}]}
    ok, fails, s = vsn.judge(res)
    assert ok, fails
    assert s["trials"] == 4 and s["detected"] == 2 and s["changed_answers"] == 3 and s["changed_and_detected"] == 2
    assert s["by_bit"][9]["changed_and_missed"] == 1 and s["by_bit"][5]["flip_failed"] == 1


def test_clean_alerts_a_noisy_reference_or_a_flip_not_undone_fail():
    base = {"ready": True, "clean_alerts": 0, "noise_quality": {"verdict": "PASS"}, "trials": [_row(3, False, "PASS")]}
    assert vsn.judge(dict(base, clean_alerts=1))[1][0].startswith("2:")
    assert vsn.judge(dict(base, noise_quality={"verdict": "WARN"}))[1][0].startswith("3:")
    assert vsn.judge(dict(base, trials=[_row(3, False, "PASS", ok_back=False)]))[1][0].startswith("4:")
    assert not vsn.judge({"ready": False})[0]
