"""The lifetime ledger's rules, the explorer, the protocol models and the CLI."""

import json

import pytest

from sdcguard.lifetime import Ledger, explore
from sdcguard.lifetime import models as M
from sdcguard.lifetime.cli import main as cli_main
from sdcguard.lifetime.explore import replay
from sdcguard.lifetime.report import text

D0 = ("decode", 0)
P0 = ("prefill", 0)


def rules(led):
    return [v.rule for v in led.violations]


# -- ledger rules ---------------------------------------------------------------------------------------------
def test_r1_submit_on_freed_or_reallocated_handle():
    led = Ledger()
    h = led.alloc(D0, "A")
    led.free(D0, "A")
    led.submit("w", "write", "prefill", dst=h, payload=("A", 0, "kv"))
    assert rules(led) == ["R1"] and "freed" in led.violations[0].message
    led2 = Ledger()
    h = led2.alloc(D0, "A")
    led2.free(D0, "A")
    led2.alloc(D0, "B")
    led2.submit("w", "write", "prefill", dst=h, payload=("A", 0, "kv"))
    assert rules(led2) == ["R1"] and "now gen 2 owned by B" in led2.violations[0].message


def test_r1_stale_descriptor_accept():
    led = Ledger()
    h = led.alloc(P0, "X")
    led.publish("d", [h], "X")
    assert led.accept("d", "decode") and not led.violations
    led.free(P0, "X")
    assert not led.accept("d", "decode")
    assert rules(led) == ["R1"]


def test_r2_reuse_with_outstanding_write_and_read():
    led = Ledger()
    h = led.alloc(D0, "A")
    led.submit("w", "write", "prefill", dst=h, payload=("A", 0, "kv"))
    led.cancel("w", "prefill")                       # a cancel request does not stop it
    led.free(D0, "A")
    led.alloc(D0, "B")
    assert rules(led) == ["R2"] and "write into" in led.violations[0].message
    assert "cancel requested: True" in led.violations[0].message
    led2 = Ledger()
    src = led2.alloc(P0, "X")
    dst = led2.alloc(D0, "X")
    led2.submit("r", "read", "decode", src=src, dst=dst)
    led2.free(P0, "X")
    led2.alloc(P0, "Y")
    assert rules(led2) == ["R2"] and "read from" in led2.violations[0].message


def test_no_r2_once_stopped_or_completed():
    led = Ledger()
    h = led.alloc(D0, "A")
    led.submit("w", "write", "prefill", dst=h, payload=("A", 0, "kv"))
    led.cancel("w", "prefill")
    led.stopped("w")
    led.free(D0, "A")
    led.alloc(D0, "B")
    h2 = led.alloc(("decode", 1), "C")
    led.submit("w2", "write", "prefill", dst=h2, payload=("C", 0, "kv"))
    led.complete("w2")
    led.free(("decode", 1), "C")
    led.alloc(("decode", 1), "D")
    assert led.violations == []


def test_late_write_lands_in_new_owner_and_consume_catches_it():
    led = Ledger()
    h = led.alloc(D0, "A")
    led.submit("w", "write", "prefill", dst=h, payload=("A", 0, "A's KV"))
    led.free(D0, "A")
    hb = led.alloc(D0, "B")                          # R2 here
    led.complete("w")
    assert any(e.kind == "effect" and "meant for gen 1" in e.text for e in led.events)
    led.consume("B", [hb])
    assert rules(led) == ["R2", "R3"] and "A's data" in led.violations[1].message


def test_r3_consume_while_writing_and_never_written_and_retired():
    led = Ledger()
    h = led.alloc(D0, "X")
    led.submit("l", "load", "load stream", dst=h, payload=("X", 0, "prefix"))
    led.consume("X", [h])
    assert rules(led) == ["R3"] and "still writing" in led.violations[0].message
    led2 = Ledger()
    h = led2.alloc(D0, "X")
    led2.consume("X", [h])
    assert rules(led2) == ["R3"] and "never written" in led2.violations[0].message
    led3 = Ledger()
    h = led3.alloc(D0, "X")
    led3.write_local(D0, "kv")
    led3.free(D0, "X")
    led3.consume("X", [h])
    assert rules(led3) == ["R3"] and "retired" in led3.violations[0].message


def test_permitted_sharing_is_not_a_violation_but_unshared_use_is():
    led = Ledger()
    h = led.alloc(D0, "A")
    led.write_local(D0, "shared prefix")
    led.share(D0, "B")
    assert led.consume("B", [h]) and led.violations == []
    assert not led.consume("C", [h])
    assert rules(led) == ["R3"] and "does not own or share" in led.violations[0].message


def test_r4_completion_applied_to_newer_allocation():
    led = Ledger()
    a = led.alloc(D0, "A")
    led.submit("w", "write", "prefill", dst=a, payload=("A", 0, "kv"))
    led.land_only("w")
    led.free(D0, "A")
    led.alloc(("decode", 1), "B")
    led.apply_completion("w", ("decode", 1))
    assert rules(led) == ["R4"]
    led2 = Ledger()
    a = led2.alloc(D0, "A")
    led2.submit("w", "write", "prefill", dst=a, payload=("A", 0, "kv"))
    led2.complete("w", marks_ready=D0)
    assert led2.violations == []


# -- explorer and models ---------------------------------------------------------------------------------------
EXPECTED = {"pd-abort": {"R1", "R2"}, "pd-lease": {"R1", "R2"}, "pd-notify": {"R4"}, "offload-load": {"R3"}}


@pytest.mark.parametrize("key", sorted(M.MODELS))
def test_buggy_variants_break_their_rule_and_fixed_ones_do_not(key):
    bad = explore(M.model(key, "buggy"), max_delays=3)
    assert {f.rule for f in bad.findings} == EXPECTED[key]
    assert bad.coverage["exhaustive_within_bounds"]
    assert all(f.delays <= 2 for f in bad.findings)
    good = explore(M.model(key, "fixed"), max_delays=5)
    assert good.findings == [] and good.coverage["exhaustive_within_bounds"]


@pytest.mark.parametrize("key", sorted(M.MODELS))
def test_partial_order_reduction_finds_the_same_violations(key):
    a = explore(M.model(key, "buggy"), max_delays=3, reduce_por=True)
    b = explore(M.model(key, "buggy"), max_delays=3, reduce_por=False)
    assert sorted((f.rule, f.message) for f in a.findings) == sorted((f.rule, f.message) for f in b.findings)
    assert a.coverage["states"] <= b.coverage["states"]


@pytest.mark.parametrize("key", sorted(M.MODELS))
def test_the_reduced_schedule_replays_to_the_same_violation(key):
    for f in explore(M.model(key, "buggy"), max_delays=3).findings:
        state, taken, _ = replay(M.model(key, "buggy"), f.schedule, strict=True)
        assert taken == f.schedule
        assert any(v.rule == f.rule and v.message == f.message for v in state.ledger.violations)


def test_the_normal_order_is_clean_for_every_model():
    for key in M.MODELS:
        for variant in ("buggy", "fixed"):
            state, _, departures = replay(M.model(key, variant), [])
            assert departures == [] and state.ledger.violations == [], (key, variant)


def test_a_release_timeout_fallback_permits_reuse_under_a_cancelled_write():
    res = explore(M.model("pd-abort", "release-timeout"), max_delays=4)
    r2 = next(f for f in res.findings if f.rule == "R2")
    assert "cancel requested: True" in r2.message
    assert any("drain timeout" in step for step in r2.schedule)
    assert any("landed in decode block 0 gen 2 (B)" in t for t in r2.then)


def test_consequences_follow_the_violation():
    res = explore(M.model("pd-lease", "buggy"), max_delays=3)
    r2 = next(f for f in res.findings if f.rule == "R2")
    assert any("carried Y's data" in t for t in r2.then)
    assert any("R3" in t and "Y's data" in t for t in r2.then)
    abort = explore(M.model("pd-abort", "buggy"), max_delays=3)
    assert any(f.rule == "R2" and "cancel requested" in f.message for f in abort.findings)


def test_fixed_lease_model_reaches_the_allowed_failure_without_violation():
    """The fixed pull protocol may fail the request after the lease is lost; that is allowed, not a violation."""
    m = M.model("pd-lease", "fixed")
    state, taken, _ = replay(m, ["prefill runs X and publishes its block",
                                 "decode gets X's descriptor and allocates its block", "time passes",
                                 "prefill's lease on X's block expires"])
    assert state.ledger.violations == []
    assert state.req["X"]["phase"] == "failed"


def test_report_and_json():
    res = explore(M.model("pd-notify", "buggy"), max_delays=3)
    out = text(res)
    assert "R4 stale_completion" in out and "instead of:" in out and "candidate ordering" in out
    d = json.loads(json.dumps(res.as_dict()))
    assert d["verdict"] == "VIOLATION" and d["findings"][0]["rule"] == "R4"
    clean = text(explore(M.model("pd-notify", "fixed"), max_delays=3))
    assert clean.startswith("sdcguard lifetime: pd-notify:fixed\nNONE FOUND")


def test_cli_exit_codes(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert cli_main(["explore", "offload-load", "--json", str(out)]) == 1
    assert json.loads(out.read_text())["findings"][0]["rule"] == "R3"
    assert cli_main(["explore", "offload-load", "--variant", "fixed"]) == 0
    assert cli_main(["selfcheck", "--max-delays", "3"]) == 0
    assert cli_main(["list"]) == 0
    assert "pd-lease" in capsys.readouterr().out
    with pytest.raises(ValueError):
        M.model("nope")
    with pytest.raises(ValueError):
        M.model("pd-abort", "sometimes")
    with pytest.raises(ValueError):
        M.model("pd-lease", "release-timeout")


def test_top_level_cli_dispatches_lifetime(capsys):
    from sdcguard.cli import main
    assert main(["lifetime", "explore", "offload-load", "--variant", "fixed"]) == 0
    assert "NONE FOUND" in capsys.readouterr().out
