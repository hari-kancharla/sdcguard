"""The model-free search: given only make() and run() for real threaded code, it finds the lifetime-breaking
delays in the buggy implementations and none in the fixed ones; findings are confirmed by replay, carry a holds spec
that replays them later, and a run that stops finishing is a liveness finding."""

import json

import pytest

from sdcguard.lifetime.adapter import LocalSink, Probe, holds_from_spec, until_event
from sdcguard.lifetime.realsearch import candidates, footprints, rank, replay, search, text
from sdcguard.lifetime.toys import ToyPull, ToyPush, ToyWakeup

SLOW = dict(hold_timeout=0.6, run_timeout=5.0)


def make(cls, fixed, **kw):
    return lambda p: cls(p, fixed=fixed, **kw)


def run(s):
    s.run()


def baseline_events(cls, fixed=False, **kw):
    sink = LocalSink()
    s = cls(Probe(sink, record_checkpoints=True), fixed=fixed, **kw)
    s.run()
    return sink


@pytest.mark.parametrize("cls,kw", [(ToyPull, {"lease_s": 0.3}), (ToyPush, {"timeout_s": 0.3}), (ToyWakeup, {})])
def test_the_normal_timing_is_clean(cls, kw):
    for fixed in (False, True):
        sink = baseline_events(cls, fixed, **kw)
        assert sink.ledger.violations == []


def test_footprints_resolve_descriptors_to_blocks():
    sink = baseline_events(ToyPull, lease_s=0.3)
    fps = {cp["args"]["point"]: regs for cp, regs, _ in footprints(sink.events)}
    assert ("prefill", 0) in fps["read_submit"] and ("decode", 0) in fps["read_submit"]
    cs = candidates(sink.events)
    assert {c.kind for c in cs if c.source == "rule"} == {"free", "realloc", "consume", "complete"}
    flips = [c for c in cs if c.source == "flip" and c.point == "read_submit"]
    assert any(c.kind == "alloc" and c.spec["args"].get("owner") == "Y" for c in flips)   # Y's later allocation
    assert all(json.loads(json.dumps(c.hold_spec(1.0)))["until"] == c.spec for c in cs)   # plain JSON


def test_rank_puts_never_flipped_reuse_first():
    sink = baseline_events(ToyPull, lease_s=0.3)
    ranked = rank(candidates(sink.events), [sink.events])
    assert ranked[0].seen_flipped == 0 and ranked[0].kind in ("realloc", "alloc")


def test_hold_operators_match_like_the_rules_need():
    e = {"kind": "alloc", "args": {"region": ["P", 0], "owner": "Y"}}
    assert until_event("alloc", region=["P", "*"], owner={"not": "X"})(e)
    assert not until_event("alloc", owner={"not": "Y"})(e)
    c = {"kind": "consume", "args": {"request": "Y", "regions": [["P", 0], ["P", 1]]}}
    assert until_event("consume", regions={"has": ["P", 1]})(c)
    assert not until_event("consume", regions={"has": ["P", 2]})(c)
    s = {"kind": "submit", "args": {"op": "w", "kind": "write"}}
    assert until_event("submit", kind="write")(s)          # an arg named kind is matched, not confused


def test_pull_buggy_found_without_a_model_confirmed_and_fixed_clean():
    bad = search(make(ToyPull, False, lease_s=0.3), run, **SLOW)
    assert bad.verdict == "VIOLATION"
    rules = {f.rule for f in bad.findings}
    assert "R1" in rules
    r1 = next(f for f in bad.findings if f.rule == "R1")
    assert all(r1.reached) and "read_submit" in r1.holds[0]
    assert r1.repro[1] == 3 and r1.repro[0] >= 2 and r1.confirmed
    assert any("Y's data" in t for f in bad.findings for t in [f.message, *f.then])
    good = search(make(ToyPull, True, lease_s=0.3), run, **SLOW)
    assert good.verdict == "NONE_FOUND"


def test_push_buggy_found_without_a_model_and_fixed_clean():
    bad = search(make(ToyPush, False, timeout_s=0.3), run, **SLOW)
    rules = {f.rule for f in bad.findings}
    assert {"R1", "R2"} <= rules
    assert any("write_land" in f.holds[0] and f.rule == "R2" for f in bad.findings)
    good = search(make(ToyPush, True, timeout_s=0.3), run, **SLOW)
    assert good.verdict == "NONE_FOUND"
    out = text(bad, "push")
    assert "condition reached" in out and "R2 reuse" in out and "reproduced 3 of 3" in out


def test_a_finding_replays_from_its_spec_as_a_regression_check():
    bad = search(make(ToyPush, False, timeout_s=0.3), run, **SLOW)
    r2 = next(f for f in bad.findings if f.rule == "R2")
    spec = json.loads(json.dumps(bad.as_dict()))["findings"][[f.rule for f in bad.findings].index("R2")]["spec"]
    assert spec == json.loads(json.dumps(r2.spec)) and len(holds_from_spec(spec)) == 1
    assert all(k is not None and k[0] == "R2" for k in replay(make(ToyPush, False, timeout_s=0.3), run, spec,
                                                               times=2))
    assert replay(make(ToyPush, True, timeout_s=0.3), run, spec, times=2) == [None, None]


def test_a_lost_wakeup_is_a_liveness_finding_and_the_fix_is_clean():
    bad = search(make(ToyWakeup, False), run, hold_timeout=0.6, run_timeout=2.0)
    assert bad.verdict == "LIVENESS" and bad.findings == []
    f = bad.liveness[0]
    assert f.message.startswith("hang") and "before_wait" in f.holds[0] and all(f.reached) and f.confirmed
    assert "LIVENESS hang" in text(bad)
    good = search(make(ToyWakeup, True), run, hold_timeout=0.6, run_timeout=2.0)
    assert good.verdict == "NONE_FOUND"


def test_random_pauses_runs_are_counted_and_clean_on_fixed_code():
    good = search(make(ToyPush, True, timeout_s=0.3), run, hold_timeout=0.3, run_timeout=5.0, random_runs=2,
                  max_runs=2, confirm=1)
    assert good.runs == 4 and good.budget_hit          # 2 targeted (the budget) + the 2 random runs asked for
    assert good.verdict == "NONE_FOUND"


def test_symbolic_holds_name_blocks_relative_to_the_checkpoint_and_still_find_the_bugs():
    sink = baseline_events(ToyPull, lease_s=0.3)
    cs = candidates(sink.events, symbolic=True)
    assert cs and all(c.key is None for c in cs)
    specs = [json.dumps(c.spec) for c in cs]
    assert any('"desc_block": 0' in x for x in specs) and not any('"Y"' in x for x in specs)   # no baseline ids
    bad = search(make(ToyPull, False, lease_s=0.3), run, symbolic=True, **SLOW)
    assert "R1" in {f.rule for f in bad.findings}
    assert search(make(ToyPull, True, lease_s=0.3), run, symbolic=True, **SLOW).verdict == "NONE_FOUND"


def test_a_symbolic_condition_is_resolved_against_this_runs_blocks():
    from sdcguard.lifetime.adapter import resolve
    hist = [{"kind": "publish", "args": {"desc": "x@P", "regions": [["P", 7], ["P", 9]]}},
            {"kind": "alloc", "args": {"region": ["D", 3], "owner": "X"}}]
    spec = {"kind": "alloc", "args": {"region": {"desc_block": 1}}}
    assert resolve(spec, "X", {"desc": "x@P"}, hist) == {"kind": "alloc", "args": {"region": ["P", 9]}}
    assert resolve({"kind": "free", "args": {"region": {"key_block": 0}}}, "X", {}, hist)["args"]["region"] == ["D", 3]
    assert resolve(spec, "X", {"desc": "other"}, hist) is None


def test_replay_command_reruns_a_holds_file(tmp_path, capsys):
    from sdcguard.lifetime.cli import main
    spec = [{"point": "write_land", "key": "write-A", "timeout": 0.8,
             "until": {"kind": "alloc", "args": {"region": ["decode", 0], "owner": {"not": "A"}}}}]
    path = tmp_path / "holds.json"
    path.write_text(json.dumps(spec))
    assert main(["replay-toy", "push", str(path), "--times", "1"]) == 1
    assert "R2 reuse" in capsys.readouterr().out
    assert main(["replay-toy", "push", str(path), "--times", "1", "--fixed"]) == 0
