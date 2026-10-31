"""Adversarial evidence histories: certainty must come from observed facts."""
from copy import deepcopy

import pytest

from sdcguard.lifetime.ledger import Violation
from sdcguard.lifetime.obligations import (CONSUMER_CHECKS, PRODUCER_HOLDS, RULES, VERDICTS,
                                           Facts, evaluate, lease_valid_at_use)


class Recording:
    def __init__(self):
        self.events = []
        self.seqs = {}

    def event(self, event_kind, mono, pid, **args):
        seq = self.seqs.get(pid, 0) + 1
        self.seqs[pid] = seq
        e = dict(kind=event_kind, args=args, t=mono + 1000, mono=mono, pid=pid,
                 session=f"stream-{pid}", seq=seq, dropped=0, host="host", clock_id="host/boot")
        self.events.append(e)
        return e

    def evidence(self, what, mono, pid, **args):
        return self.event("evidence", mono, pid, what=what, **args)

    def health(self, mono=100):
        for pid in list(self.seqs):
            self.evidence("recording_health", mono, pid, healthy=True)
        return self.events


def recording(*, rejected=False, check=None, stale=False, reuse_after_notify=False,
              uncertain=False, op="pull:decoder-X:0", free_only=False):
    r = Recording()
    r.event("note", 1, "P", hooks=["KVCacheManager.free", "KVCacheManager.allocate_slots"])
    r.event("alloc", 2, "P", region=["producer", 4], owner="R")
    r.event("publish", 3, "P", desc="R@producer", regions=[["producer", 4]], by="R")
    r.evidence("lease_granted", 3.1, "P", request="R", engine="producer", lease_nonce=7, desc="R@producer")
    # An attempt can be observed before a hold; only the actual submit is a bound.
    r.evidence("attempt", 4, "D", request="X", remote_request="R", remote_engine="producer",
               desc="R@producer", blocks=1, op_ids=[op], mapping_complete=True, reader_agent="D-agent",
               source_regions=[["producer", 4]], source_indices=[0], published_count=1,
               destination_regions=[["decode", 8]],
               lease_fields=True, lease_slot=0, lease_nonce=7)
    if stale or free_only:
        r.event("free", 5, "P", region=["producer", 4], by="R")
        if stale:
            r.event("alloc", 6, "P", region=["producer", 4], owner="another")
    r.event("submit", 10, "D", op=op, kind="read", by="decode X",
            src={"desc": "R@producer", "i": 0}, dst=["decode", 8])
    r.evidence("read_submission_result", 11, "D", request="X", op_ids=[op], submitted=True)
    if reuse_after_notify:
        r.evidence("released_by_read_notification", 12, "PW", request="R", engine="producer",
                   transport_complete=True, exact_mapping=True, sender_agent="D-agent", current_nonce=7,
                   notification_nonce=7)
    if reuse_after_notify or uncertain:
        r.event("free", 13, "P", region=["producer", 4], by="R")
        r.event("alloc", 14, "P", region=["producer", 4], owner="another")
    r.evidence("transfer_done", 20, "D", request="X", op_ids=[op])
    if check:
        r.evidence("lease_check", 21, "D", request="X", op_ids=[op],
                   result="passed" if check == "matched" else "failed", cause=check)
    if rejected:
        r.evidence("load_failed", 22, "DS", request="X", policy="recompute")
    r.evidence("admit", 23, "DS", request="X", transferred=not rejected, closed=False)
    return r


def lease(events, violations=()):
    return lease_valid_at_use(Facts(events), violations, "X")


def replace(events, what, **changes):
    e = next(e for e in events if e["args"].get("what") == what)
    e["args"].update(changes)
    return e


def drop(events, *, what=None, kind=None):
    # Renumber to distinguish a semantically missing hook from a detectable trace gap.
    events[:] = [e for e in events if not (e["args"].get("what") == what if what else e["kind"] == kind)]
    counts = {}
    for e in events:
        counts[e["pid"]] = counts.get(e["pid"], 0) + 1
        e["seq"] = counts[e["pid"]]


def test_rules_and_verdicts():
    assert set(RULES) == {"lease_valid_at_use"}
    assert "recovered" in VERDICTS and "unknown" in VERDICTS


@pytest.mark.parametrize("check", [None, "matched"])
def test_clean_source_lifetime_is_satisfied(check):
    f = lease(recording(check=check).health())
    assert f.verdict == "satisfied", f.as_dict()
    assert f.protection == (CONSUMER_CHECKS if check else PRODUCER_HOLDS)
    assert "admission" in f.not_observed[0]


def test_h1_notification_reuse_before_completion_poll_is_clean():
    events = recording(reuse_after_notify=True).health()
    v = Violation("R2", 99, "delayed completion poll", regions=(("producer", 4),), ops=("pull:decoder-X:0",))
    assert lease(events, [v]).verdict == "satisfied"


def test_h1_without_attributable_notification_overlap_is_unknown():
    events = recording(uncertain=True).health()
    assert lease(events).verdict == "unknown"


@pytest.mark.parametrize("what", ["attempt", "read_submission_result", "transfer_done", "admit"])
def test_m1_missing_required_evidence_is_unknown(what):
    events = recording().health()
    drop(events, what=what)
    assert lease(events).verdict == "unknown"


@pytest.mark.parametrize("kind", ["submit", "publish", "note", "alloc"])
def test_m1_missing_lifecycle_evidence_is_unknown(kind):
    events = recording().health()
    drop(events, kind=kind)
    assert lease(events).verdict == "unknown"


@pytest.mark.parametrize("args", [dict(blocks=0), dict(blocks=2), dict(op_ids=[]), dict(mapping_complete=False)])
def test_m1_missing_or_partial_mapping_is_unknown(args):
    events = recording().health()
    replace(events, "attempt", **args)
    assert lease(events).verdict == "unknown"


def test_empty_publication_is_not_safe():
    events = recording().health()
    next(e for e in events if e["kind"] == "publish")["args"]["regions"] = []
    assert lease(events).verdict == "unknown"


def test_m2_arbitrary_operation_names_work_and_string_violations_are_not_evidence():
    events = recording(op="opaque-op-141", stale=True).health()
    assert lease(events).verdict == "violated"
    clean = recording(op="opaque-op-141").health()
    assert lease(clean, ["R1 opaque-op-141 stale source"]).verdict == "satisfied"


def test_m3_shared_nonce_does_not_attribute_notification():
    r = recording(reuse_after_notify=True)
    r.evidence("attempt", 9, "D2", request="X2", remote_request="R", remote_engine="producer",
               desc="R@producer", blocks=1, op_ids=["other"], mapping_complete=True, reader_agent="D2-agent",
               lease_nonce=7)
    assert lease(r.health()).verdict == "unknown"


def test_m4_retirement_during_hold_is_before_actual_submission():
    f = lease(recording(stale=True).health())
    assert f.verdict == "violated"
    assert "before read submission" in f.relationship


@pytest.mark.parametrize("clock", [None, "other-host/boot"])
def test_m5_cross_clock_is_unknown_regardless_of_wall_clock_order(clock):
    events = recording(stale=True).health()
    for e in events:
        if e["pid"] == "P":
            e["clock_id"] = clock
            e["t"] += 1e9
    assert lease(events).verdict == "unknown"


def test_wall_clock_skew_does_not_change_same_clock_result():
    events = recording().health()
    for i, e in enumerate(events):
        e["t"] = 10e9 if i % 2 else -10e9
    assert lease(events).verdict == "satisfied"


def test_m6_policy_without_completion_is_correctly_rejected_not_recovered():
    f = lease(recording(check="lease_mismatch", rejected=True, stale=True).health())
    assert f.verdict == "correctly_rejected" and "completion not observed" in f.detail


def test_m6_recompute_completion_is_required_and_independent_of_output():
    r = recording(check="lease_mismatch", rejected=True, stale=True)
    r.evidence("recompute_done", 25, "DS", request="X", rejected_op_ids=["pull:decoder-X:0"])
    events = r.health()
    for e in events:
        e["answer"] = "incorrect text that evaluator must never inspect"
    assert lease(events).verdict == "recovered"


def test_fail_policy_without_observed_terminal_closure_is_unknown():
    events = recording(check="lease_mismatch", rejected=True).health()
    replace(events, "load_failed", policy="fail")
    replace(events, "admit", closed=True)
    assert lease(events).verdict == "unknown"


def test_closed_request_cannot_be_recovered():
    r = recording(check="lease_mismatch", rejected=True)
    replace(r.events, "admit", closed=True)
    r.evidence("recompute_done", 25, "DS", request="X", rejected_op_ids=["pull:decoder-X:0"])
    assert lease(r.health()).verdict == "unknown"




def test_l3_destination_violation_does_not_become_stale_source():
    events = recording().health()
    v = Violation("R1", 99, "uses destination", regions=(("decode", 8),), ops=("pull:decoder-X:0",))
    assert lease(events, [v]).verdict == "satisfied"


def test_l4_metadata_without_check_does_not_claim_a_check_ran():
    f = lease(recording(stale=True).health())
    assert f.verdict == "violated" and f.protection == PRODUCER_HOLDS
    assert "no lease check" in f.detail


@pytest.mark.parametrize("cause", ["check_unavailable", "unknown", None])
def test_failed_check_without_established_invalidity_is_unknown(cause):
    r = recording(check="check_unavailable", rejected=True)
    replace(r.events, "lease_check", cause=cause)
    assert lease(r.health()).verdict == "unknown"


def test_admitting_after_a_failed_check_is_a_violation():
    assert lease(recording(check="lease_mismatch").health()).verdict == "violated"


def test_wrong_english_reason_cannot_drive_a_verdict():
    r = recording(check="check_unavailable", rejected=True)
    replace(r.events, "lease_check", reason="released the KV blocks before the read completed")
    assert lease(r.health()).verdict == "unknown"


def test_recompute_without_a_reason_is_not_satisfied():
    assert lease(recording(rejected=True).health()).verdict == "unknown"


def test_unobserved_transport_submit_is_unknown():
    events = recording(stale=True).health()
    replace(events, "read_submission_result", submitted=False)
    assert lease(events).verdict == "unknown"


def test_duplicate_publications_are_unknown_not_latest_wins():
    r = recording(stale=True)
    r.event("publish", 7, "P", desc="R@producer", regions=[["producer", 4]], by="R")
    assert lease(r.health()).verdict == "unknown"


def test_conflicting_admission_is_unknown():
    r = recording()
    r.evidence("admit", 24, "DS", request="X", transferred=False, closed=False)
    assert lease(r.health()).verdict == "unknown"


def test_conflicting_check_results_are_unknown():
    r = recording(check="matched")
    r.evidence("lease_check", 22, "D", request="X", result="failed", cause="lease_mismatch",
               op_ids=["pull:decoder-X:0"])
    assert lease(r.health()).verdict == "unknown"


@pytest.mark.parametrize("pid", ["P", "D", "DS"])
def test_each_relevant_stream_needs_a_health_receipt(pid):
    events = recording().health()
    events[:] = [e for e in events if not (e["pid"] == pid and e["args"].get("what") == "recording_health")]
    assert lease(events).verdict == "unknown"


def test_trace_gap_cannot_turn_missing_free_into_satisfied():
    events = recording().health()
    events[:] = [e for e in events if e["kind"] != "alloc"]
    assert lease(events).verdict == "unknown"


def test_drop_count_and_parse_errors_fail_unknown():
    events = recording().health()
    events[0]["dropped"] = 1
    assert lease(events).verdict == "unknown"
    events = recording().health()
    events.append({"kind": "evidence", "args": {"what": "recording_error", "source": "truncated.jsonl"}})
    assert lease(events).verdict == "unknown"


def test_zero_false_violations_on_notification_poll_delay_permutations():
    for shift in (0.01, 1, 100):
        r = recording(reuse_after_notify=True)
        for e in r.events:
            if e["args"].get("what") in ("transfer_done", "admit"):
                e["mono"] += shift
        f = lease(r.health(200))
        assert f.verdict == "satisfied", f.as_dict()


def test_exact_request_selection_and_missing_attempt():
    events = recording().health()
    findings = evaluate(events, [], requests=["X"])
    assert len(findings) == 1 and all(f.request == "X" for f in findings)
    missing = evaluate(events, [], requests=["other"])
    assert len(missing) == 1 and all(f.verdict == "unknown" for f in missing)
    assert all(isinstance(f.as_dict(), dict) for f in findings)


def test_free_only_is_unknown_when_no_completion_witness_exists():
    assert lease(recording(free_only=True).health()).verdict == "unknown"


@pytest.mark.parametrize("field,value", [("args", None), ("args", []), ("kind", None)])
def test_malformed_record_does_not_crash_or_certify(field, value):
    events = recording().health()
    events[0][field] = value
    assert lease(events).verdict == "unknown"


def test_non_event_record_is_not_silently_ignored():
    events = recording().health()
    events.append("a corrupted record")
    assert lease(events).verdict == "unknown"


@pytest.mark.parametrize("what,field,value", [
    ("attempt", "remote_request", None),
    ("attempt", "remote_engine", None),
    ("attempt", "op_ids", [["nested"]]),
    ("attempt", "request", {"bad": "id"}),
    ("read_submission_result", "op_ids", None),
    ("transfer_done", "op_ids", [False]),
])
def test_malformed_required_identity_is_unknown(what, field, value):
    events = recording().health()
    replace(events, what, **{field: value})
    assert lease(events).verdict == "unknown"


def test_identical_attempts_are_still_ambiguous():
    r = recording(stale=True)
    a = deepcopy(next(e["args"] for e in r.events if e["args"].get("what") == "attempt"))
    r.event("evidence", 22, "D", **a)
    assert lease(r.health()).verdict == "unknown"


def test_failed_unavailable_check_is_not_invalidity_proof():
    assert lease(recording(check="check_unavailable").health()).verdict == "unknown"


def test_success_label_with_unrecognized_cause_is_not_a_verified_check():
    r = recording(check="matched")
    replace(r.events, "lease_check", cause="unknown")
    assert lease(r.health()).verdict == "unknown"


def test_another_worker_cannot_supply_this_attempt_completion():
    events = recording().health()
    replace(events, "transfer_done")["session"] = "another-process"
    assert lease(events).verdict == "unknown"


def test_missing_source_index_does_not_default_to_zero():
    events = recording().health()
    next(e for e in events if e["kind"] == "submit")["args"]["src"].pop("i")
    assert lease(events).verdict == "unknown"


def test_publish_owner_must_match_remote_request():
    events = recording().health()
    next(e for e in events if e["kind"] == "publish")["args"]["by"] = "another-request"
    assert lease(events).verdict == "unknown"
















def test_receipt_before_terminal_is_not_final_coverage():
    r = recording()
    assert lease(r.health(22.5)).verdict == "unknown"


@pytest.mark.parametrize("changes", [
    dict(source_regions=[["producer", 5]]), dict(source_regions=[]), dict(source_indices=[]),
    dict(published_count=2), dict(destination_regions=[["decode", 9]]),
])
def test_actual_source_and_destination_mapping_must_match_publication(changes):
    events = recording(stale=True).health()
    replace(events, "attempt", **changes)
    assert lease(events).verdict == "unknown"




def test_equal_timestamp_retirement_and_submit_cannot_establish_order():
    r = recording(stale=True)
    # The later producer allocation equals the worker's submission clock value.
    next(e for e in r.events if e["kind"] == "alloc" and e["args"]["owner"] == "another")["mono"] = 10
    assert lease(r.health()).verdict == "unknown"


def test_conflicting_recovery_policies_do_not_certify_recovery():
    r = recording(check="lease_mismatch", rejected=True)
    r.evidence("load_failed", 24, "DS", request="X", policy="fail")
    r.evidence("recompute_done", 25, "DS", request="X", rejected_op_ids=["pull:decoder-X:0"])
    assert lease(r.health()).verdict == "unknown"


def test_malformed_source_lifecycle_event_cannot_disappear_from_history():
    events = recording(stale=True).health()
    next(e for e in events if e["kind"] == "free")["args"]["region"] = None
    assert lease(events).verdict == "unknown"


def test_alloc_without_observed_retirement_is_not_a_false_violation():
    events = recording(stale=True).health()
    drop(events, kind="free")
    assert lease(events).verdict == "unknown"


def test_free_of_another_sharer_is_not_retirement_of_remote_owner():
    events = recording(stale=True).health()
    next(e for e in events if e["kind"] == "free")["args"]["by"] = "some-other-reference"
    assert lease(events).verdict == "unknown"


@pytest.mark.parametrize("kind,owner_args", [
    ("alloc", {"owner": "another"}), ("free", {"by": "R"}), ("share", {"request": "another"}),
])
def test_colliding_allocator_stream_cannot_hide_source_lifecycle(kind, owner_args):
    r = recording()
    r.event(kind, 6, "P2", region=["producer", 4], **owner_args)
    f = lease(r.health())
    assert f.verdict == "unknown"
    assert "one allocator recording incarnation" in " ".join(f.missing)


def test_allocator_restart_same_pid_new_recording_session_is_unknown():
    r = recording()
    event = r.event("alloc", 6, "P2", region=["producer", 4], owner="another")
    event["pid"] = "P"
    # A restarted allocator can reuse the old PID; session is part of identity.
    event["session"] = "stream-P-after-restart"
    assert lease(r.health()).verdict == "unknown"


def test_foreign_allocator_using_different_source_region_is_irrelevant():
    r = recording()
    r.event("alloc", 6, "P2", region=["other-engine", 4], owner="another")
    assert lease(r.health()).verdict == "satisfied"


def test_notification_tied_with_source_free_does_not_prove_safe_ordering():
    r = recording(reuse_after_notify=True)
    next(e for e in r.events if e["kind"] == "free")["mono"] = 12
    assert lease(r.health()).verdict == "unknown"


def test_notification_tied_with_transfer_completion_is_not_a_causal_witness():
    r = recording(reuse_after_notify=True)
    replace(r.events, "released_by_read_notification")["mono"] = 20
    assert lease(r.health()).verdict == "unknown"


def test_recovery_from_unrelated_process_is_not_this_request_recovery():
    r = recording(check="lease_mismatch", rejected=True)
    r.evidence("recompute_done", 25, "UNRELATED", request="X", rejected_op_ids=["pull:decoder-X:0"])
    assert lease(r.health()).verdict == "unknown"


@pytest.mark.parametrize("rejected_ops", [None, [], ["another-attempt-op"]])
def test_recovery_must_name_the_exact_rejected_transfer(rejected_ops):
    r = recording(check="lease_mismatch", rejected=True)
    r.evidence("recompute_done", 25, "DS", request="X", rejected_op_ids=rejected_ops)
    assert lease(r.health()).verdict == "unknown"


def test_recompute_policy_from_another_scheduler_cannot_bind_recovery():
    r = recording(check="lease_mismatch", rejected=True)
    drop(r.events, what="load_failed")
    r.seqs = {pid: max(e["seq"] for e in r.events if e["pid"] == pid) for pid in r.seqs}
    r.evidence("load_failed", 22, "UNRELATED", request="X", policy="recompute")
    r.evidence("recompute_done", 25, "DS", request="X", rejected_op_ids=["pull:decoder-X:0"])
    assert lease(r.health()).verdict == "unknown"


def test_notification_tied_with_submission_does_not_prove_completion_after_read():
    r = recording(reuse_after_notify=True)
    replace(r.events, "released_by_read_notification")["mono"] = 10
    assert lease(r.health()).verdict == "unknown"
