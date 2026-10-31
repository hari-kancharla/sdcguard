"""experiments/lifetime_pd_check.py without GPUs: the prefill/decode request flow, the metric reading from each
server, the decode-side decline attribution, the outcome classification, and the whole discover, save and replay
flow against a stand-in prefill/decode pair."""

import importlib.util
import io
import json
import os
import urllib.error
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "lifetime_pd_check", os.path.join(os.path.dirname(HERE), "experiments", "lifetime_pd_check.py"))
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

R1 = "R1 stale handle submitted: read-d1-0 (read by decode d1) uses source P block 3 gen 1 was retired"
BASE = " PLUM. Again: PLUM"


def test_prefill_then_decode_follows_vllms_toy_proxy(monkeypatch):
    calls = []

    def post(url, body, timeout=120):
        calls.append((url, json.loads(json.dumps(body))))
        if url.endswith(":8100/v1/completions"):
            return {"kv_transfer_params": {"remote_engine_id": "P", "remote_block_ids": [3, 4]}}
        return {"choices": [{"text": " red, green and blue"}], "id": "cmpl-abc123-0-9"}

    monkeypatch.setattr(X, "post", post)
    out = X.pd_complete("m", "prompt", max_tokens=5)
    assert out == " red, green and blue"
    (u1, b1), (u2, b2) = calls
    assert u1.startswith("http://localhost:8100") and b1["max_tokens"] == 1
    assert b1["kv_transfer_params"]["do_remote_decode"] is True and b1["temperature"] == 0
    assert u2.startswith("http://localhost:8200") and b2["kv_transfer_params"]["remote_block_ids"] == [3, 4]
    assert b2["max_tokens"] == 5
    # decode returns the completion id so a decline line can be matched to this request
    text, err, rid = X.decode("m", "prompt", {"remote_block_ids": [3, 4]})
    assert text == " red, green and blue" and err is None and rid == "cmpl-abc123-0-9"
    calls.clear()
    kv = X.pd_complete("m", "prompt", prefill_only=True)
    assert kv["remote_engine_id"] == "P" and len(calls) == 1


def test_events_are_read_from_every_process_file_in_time_order(tmp_path):
    for name, evs in (("1.jsonl", [{"t": 2.0, "kind": "free"}]), ("2.jsonl", [{"t": 1.0, "kind": "alloc"}])):
        (tmp_path / name).write_text("".join(json.dumps(e) + "\n" for e in evs))
    (tmp_path / "other.txt").write_text("ignored")
    assert [e["kind"] for e in X.events(str(tmp_path))] == ["alloc", "free"]
    assert X.tail(str(tmp_path / "missing.log")) == []
    assert X.loglines(str(tmp_path / "missing.log")) == []


def test_counters_summed_and_a_failed_scrape_is_unknown_not_zero(tmp_path):
    text = ("# HELP vllm:nixl_num_notifications_after_expiry_total x\n"
            'vllm:nixl_num_notifications_after_expiry_total{engine="0",model_name="m"} 2.0\n'
            'vllm:nixl_num_notifications_after_expiry_total{engine="1",model_name="m"} 1.0\n'
            'vllm:nixl_num_kv_expired_reqs_total{engine="0"} 4.0\n'
            'vllm:nixl_num_kv_expired_reqs_created{engine="0"} 1.7e9\n')
    assert X.counter(text, "vllm:nixl_num_notifications_after_expiry") == 3.0
    assert X.counter(text, "vllm:nixl_num_kv_expired_reqs") == 4.0 and X.counter(text, "absent") == 0.0
    # nothing is listening on port 1: every counter comes back unknown (None), never silently 0
    s = X.scrape(1, X.CONSUMER_SIGNALS)
    assert "metrics_error" in s and all(s[n] is None for n in X.CONSUMER_SIGNALS)
    log = tmp_path / "p.log"
    log.write_text(f"ok\nERROR {X.VLLM_LOG_LINE} abc were retrieved\n")
    assert X.log_count(str(log), X.VLLM_LOG_LINE) == 1 and X.log_count(str(tmp_path / "missing"), "x") == 0
    # a counter this build does not export is unknown, not 0
    assert X.counter_or_none(text, "vllm:nixl_num_lease_check_failures") is None
    assert X.counter_or_none(text, "vllm:nixl_num_kv_expired_reqs") == 4.0
    # delta: a failed scrape makes every metric unknown (the log count, read from a file, still counts); a counter
    # that first appears during the window counts from 0; an unknown never turns into a number
    assert X.delta({"a": 1.0, "b": None, "metrics_error": "e"}, {"a": 3.0, "b": 5.0}) == {"a": None, "b": None}
    assert X.delta({"a": 1.0, "b": None, "c": None},
                   {"a": 3.0, "b": 5.0, "c": None}) == {"a": 2.0, "b": 5.0, "c": None}
    assert X.delta({"a": 1.0, "log_lines": 1},
                   {"a": None, "log_lines": 3, "metrics_error": "e"}) == {"a": None, "log_lines": 2}


def test_decode_decline_lines_are_parsed_and_matched_to_the_request():
    lines = [
        "(EngineCore pid=253) WARNING 10-27 06:49:35 [base_worker.py:3107] Declining KV pulled for request "
        "cmpl-9a7877809f0ce00a-0-856257e6: engine P rank 0 released the KV blocks before the read completed. "
        "The KV load fails.",
        "INFO:     127.0.0.1 - unrelated line",
        "(EngineCore pid=253) WARNING [pull_worker.py:160] Declining KV pull for request cmpl-dead00-0-1 from "
        "engine P: the lease slot of engine P rank 0 could not be read.",
    ]
    lines.append("(EngineCore pid=253) WARNING [base_worker.py:3206] Declining KV pulled for request "
                 "cmpl-beef01-0-2: the lease slot of engine P rank 0 could not be read. The KV load fails.")
    ds = X.declines(lines)
    assert len(ds) == 3
    assert ds[0]["kind"] == "post_read" and "released the KV blocks before the read completed" in ds[0]["reason"]
    assert ds[0]["cause"] == "lease_mismatch"                       # the #61229 catch
    assert ds[1]["kind"] == "pre_read" and "lease slot" in ds[1]["reason"] and ds[1]["cause"] == "cannot_check"
    # the SAME post-read line format, but the check could not run: a refusal, never a detection
    assert ds[2]["kind"] == "post_read" and ds[2]["cause"] == "verification_unavailable"
    assert X.decline_cause("post_read", "something new") == "other_post_read"
    assert X._id_core("cmpl-9a7877809f0ce00a-0-856257e6") == "9a7877809f0ce00a"
    # the completion id (no engine/seq suffix) matches the fuller id in the decline line
    assert X.match_decline(ds, ["cmpl-9a7877809f0ce00a"])["kind"] == "post_read"
    assert X.match_decline(ds, [None, "cmpl-notpresent"]) is None


def test_harm_is_a_wrong_or_foreign_answer_independent_of_the_counter():
    none = X.classify([], BASE, BASE, "p1", {}, {}, [])
    assert none["level"] == 0 and not none["lease_expired"] and not none["harm"] and none["answer"] == "correct"
    assert none["terminal"] == "clean"
    expired = X.classify([], BASE, BASE, "p1", {"vllm:nixl_num_kv_expired_reqs": 1.0}, {}, [])
    assert expired["level"] == 1 and not expired["exposure"] and expired["lease_expired"]
    stale = X.classify([R1], BASE, BASE, "p1", {}, {}, ["lease expired for p1 on P"])
    assert stale["level"] == 2 and stale["exposure"] and not stale["harm"]      # a stale read, output unchanged
    assert stale["access"] == "read issued after the blocks were reclaimed"
    assert stale["terminal"] == "exposed_answer_unchanged"                     # never reported as clean
    garbled = X.classify([R1], "::::", BASE, "p1", {"vllm:nixl_num_kv_expired_reqs": 1.0}, {}, [])
    assert garbled["harm"] and garbled["answer"] == "differs" and garbled["terminal"] == "harm_garbled"
    # X's word inside a garbled answer is NOT correct: no substring acceptance
    worded = X.classify([R1], " PLUM :::::::::::::", BASE, "p1", {}, {}, [])
    assert worded["answer"] == "differs" and worded["says_own_word"] and worded["harm"]
    assert not garbled["rejected"]
    foreign = X.classify([R1], " MANGO. Again: MANGO", BASE, "p1", {}, {}, [])
    assert foreign["level"] == 4 and foreign["harm"] and foreign["foreign_words"] == ["MANGO"]
    assert foreign["terminal"] == "harm_foreign"
    r2 = "R2 reuse while read-d1-0 can still read from it: P block 3 reallocated from gen 1 to gen 2 (q)"
    assert X.classify([r2], BASE, BASE, "p1", {}, {}, [])["access"] == "read in flight when the blocks were reclaimed"
    # a changed answer with NO recorded stale read is nondeterminism, reported apart, not harm
    odd = X.classify([], " banana", BASE, "p1", {}, {}, [])
    assert odd["answer"] == "differs" and odd["changed_without_stale_access"] and not odd["harm"] and odd["level"] == 0
    # other phrasing with the right word is not correct unless it equals an agreed reference
    variant = X.classify([], " PLUM. The letters are P, L, U, M.", BASE, "p1", {}, {}, [])
    assert variant["answer"] == "differs" and variant["terminal"] == "changed_nondeterministic"
    # ...such as the agreed local recompute reference (X computed on the decode engine, no KV transfer)
    local = X.classify([], " PLUM. The letters are P, L, U, M.", BASE, "p1", {}, {}, [],
                       local_answer=" PLUM. The letters are P, L, U, M.")
    assert local["answer"] == "correct_recompute" and local["terminal"] == "clean" and not local["harm"]
    assert X.CANARY_X in X.PROMPT_X and not any(w in X.PROMPT_X for w in X.CANARIES)


def test_rejection_recovery_fail_closed_and_the_hidden_leak_case():
    # fix declines the stale read (decode decline line matched to the request), request recomputes correctly
    mism = "engine P rank 0 released the KV blocks before the read completed"
    rec = X.classify([R1], BASE, BASE, "p1", {"vllm:nixl_num_kv_expired_reqs": 1.0},
                     {"vllm:nixl_num_lease_check_failures": 1.0}, ["lease expired for p1 on P"],
                     rejected=True, reject_kind="post_read", reject_reason=mism)
    assert rec["exposure"] and rec["rejected"] and rec["reject_counter_rose"] and not rec["harm"]
    assert rec["terminal"] == "recovered" and rec["level"] == 2 and rec["caught_stale_lease"]
    assert rec["reject_cause"] == "lease_mismatch"
    # fail-closed: the request errored rather than returning a wrong answer, so not harm
    closed = X.classify([R1], None, BASE, "p1", {}, {"vllm:nixl_num_lease_check_failures": 1.0}, [],
                        x_error="HTTP 500", rejected=True, reject_kind="post_read", reject_cause="lease_mismatch")
    assert closed["answer"] == "failed_closed" and not closed["harm"] and closed["terminal"] == "failed_closed"
    # an error with no matched decline is not the fix failing the request closed
    err = X.classify([R1], None, BASE, "p1", {}, {}, [], x_error="HTTP 500")
    assert err["terminal"] == "failed_without_decline" and not err["harm"]
    # the same post-read line, but the lease slot could not be read: never counted as recovery
    unv = X.classify([R1], BASE, BASE, "p1", {}, {}, [], rejected=True, reject_kind="post_read",
                     reject_reason="the lease slot of engine P rank 0 could not be read")
    assert unv["reject_cause"] == "verification_unavailable" and unv["terminal"] == "declined_unverifiable"
    assert not unv["caught_stale_lease"]
    # a decline that matched X, then a garbled answer: the decline does not erase it, and it is never success
    bad = X.classify([R1], " PLUM ::::::::", BASE, "p1", {}, {}, [], rejected=True, reject_kind="post_read",
                     reject_reason=mism)
    assert bad["bad_answer_after_decline"] and bad["terminal"] == "declined_answer_differs" and not bad["harm"]
    # a decline in the window that could not be tied to X: unknown, never credited
    unk_dec = X.classify([R1], BASE, BASE, "p1", {}, {}, [], reject_unknown=True)
    assert unk_dec["terminal"] == "inconclusive_decline_unattributed"
    # ...and it blocks a harm claim on a differing answer too (the bytes may not have been admitted)
    unk_bad = X.classify([R1], "::::", BASE, "p1", {}, {}, [], reject_unknown=True)
    assert not unk_bad["harm"] and unk_bad["terminal"] == "inconclusive_decline_unattributed"
    # the leak the old logic hid: a rejection was recorded AND a foreign answer came back. Still harm.
    leak = X.classify([R1], " MANGO. Again: MANGO", BASE, "p1", {}, {"vllm:nixl_num_lease_check_failures": 1.0}, [],
                      rejected=True, reject_kind="post_read", reject_reason=mism)
    assert leak["answer"] == "foreign" and leak["harm"] and leak["rejected"] and leak["terminal"] == "harm_foreign"
    # an unfinished request is inconclusive, never counted as correct or as harm
    hung = X.classify([], None, BASE, "p1", {}, {}, [], x_finished=False)
    assert hung["answer"] == "unresolved" and not hung["harm"] and hung["terminal"] == "inconclusive_unresolved"
    # an unknown fix counter is reported as unknown, not as zero
    unk = X.classify([R1], BASE, BASE, "p1", {}, {"vllm:nixl_num_lease_check_failures": None}, [])
    assert unk["reject_counter_unknown"] and not unk["reject_counter_rose"]
    # a decline before any read ("cannot verify the lease") is not the #61229 catch and never counts as recovery
    pre = X.classify([R1], BASE, BASE, "p1", {}, {}, [], rejected=True, reject_kind="pre_read")
    assert pre["terminal"] == "declined_before_read" and not pre["harm"]


def test_a_server_that_does_not_start_is_a_harness_failure_not_a_finding(tmp_path, monkeypatch):
    class Dead:                                      # a vLLM server that exits during startup
        def __init__(self, *a, **k):
            pass

        def poll(self):
            return 1

        def terminate(self):
            pass

        def wait(self, t=None):
            return 1

    monkeypatch.setattr(X.subprocess, "Popen", Dead)
    out = tmp_path / "out"
    assert X.main(["--out", str(out)]) == 4          # 1 would mean a confirmed rule break
    res = json.loads((out / "RESULT.json").read_text())
    assert res["error"].startswith("RuntimeError: server on 8100 exited")
    assert res["verdict"] == "harness_failed"
    assert res["nixl_ucx_params"] == {"ucx_error_handling_mode": "none"}


class _Cand:                                     # a stand-in for realsearch's Candidate
    source, kind = "rule", "alloc"

    def describe(self):
        return "hold read_submit until block 0 is reallocated"

    def hold_spec(self, timeout):
        return {"point": "read_submit", "key": None,
                "until": {"kind": "alloc", "args": {"region": {"key_block": 0}}}, "timeout": timeout}


def test_hold_outcome_and_stall_are_read_from_the_release_lines():
    expired = ["hold condition for read_submit r1: {}", "released read_submit r1 (hold expired) after 14.006s"]
    reached = ["released read_submit r2 (condition reached) after 5.315s"]
    assert X.hold_outcome(expired) == "expired" and X.stall_seconds(expired) == 14.006
    assert X.hold_outcome(reached) == "condition_reached" and X.stall_seconds(reached) == 5.315
    assert X.hold_outcome([]) == "not_held" and X.stall_seconds([]) is None


def test_a_discovered_candidate_is_saved_and_replayed_exactly(tmp_path):
    found = {"hold_outcome": "expired", "stall_s": 14.0, "x": "::::", "outcome": {"terminal": "harm_garbled"}}
    sc = X.SavedCandidate.from_candidate(_Cand(), 14, discovered=found)
    assert sc.timeout == 14 and sc.hold_spec()["timeout"] == 14 and sc.hold_spec(2)["timeout"] == 2
    assert sc.hold_spec(2)["until"] == _Cand().hold_spec(14)["until"]       # only the stall length changes
    rec = sc.record()
    assert rec["found_by"]["hold_outcome"] == "expired" and "posting its KV read" in rec["intervention"]
    path = tmp_path / "candidate.json"
    X.save_candidate(str(path), rec)
    again = X.SavedCandidate(X.load_candidate(str(path)))
    assert X._spec_key(again.hold_spec()) == X._spec_key(sc.hold_spec())


def test_an_unusable_saved_candidate_fails_before_any_server_starts(tmp_path, monkeypatch):
    def no_servers(*a, **k):
        raise AssertionError("a server was started")

    monkeypatch.setattr(X.subprocess, "Popen", no_servers)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--variant", "fixed", "--candidate", str(tmp_path / "missing.json")]) == 4
    res = json.loads((out / "RESULT.json").read_text())
    assert "run the unpatched variant first" in res["error"] and res["verdict"] == "harness_failed"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"label": "no spec"}))
    try:
        X.load_candidate(str(bad))
    except ValueError as e:
        assert "no usable hold spec" in str(e)
    else:
        raise AssertionError("a candidate without a hold spec was accepted")


def test_an_error_finish_reason_in_a_200_body_is_fail_closed_not_an_answer(monkeypatch):
    monkeypatch.setattr(X, "post", lambda url, body, timeout=120: {
        "choices": [{"text": "", "finish_reason": "error"}], "id": "cmpl-e1"})
    assert X.decode("m", "p", {}) == (None, "finish_reason=error", "cmpl-e1")


def test_verdicts_never_report_success_on_a_hollow_or_broken_run():
    un = SimpleNamespace(variant="unpatched")
    fx = SimpleNamespace(variant="fixed", kv_load_failure_policy="recompute", confirm=3)
    ff = SimpleNamespace(variant="fixed", kv_load_failure_policy="fail", confirm=3)

    def res(**kw):
        return {"harm_present": False, "runs_total": 5, "runs_planned": 5, "exercised_terminals": {},
                "unexercised_terminals": {}, "discovered_harm_once": False} | kw

    cases = [  # (result fields, variant, controls_ok, baseline_ok, trigger, unresolved, no_candidate) -> (code, verdict)
        (res(), un, True, False, True, False, False, 5, "inconclusive_bad_baseline"),
        (res(), un, False, True, True, False, False, 3, "control_broke"),
        (res(), un, False, True, True, True, False, 5, "inconclusive_unresolved"),
        (res(), fx, True, True, True, True, False, 5, "inconclusive_unresolved"),
        (res(), un, True, True, False, False, True, 5, "inconclusive_no_reproducing_candidate"),
        (res(harm_present=True), un, True, True, True, False, False, 1, "harm_reproduced"),
        (res(harm_present=True), fx, True, True, True, False, False, 1, "fix_failed_harm_present"),
        (res(), fx, True, True, False, False, False, 5, "inconclusive_no_trigger"),
        (res(exercised_terminals={"recovered": 5}), fx, True, True, True, False, False, 0, "recovery_demonstrated"),
        (res(exercised_terminals={"failed_closed": 5}), ff, True, True, True, False, False, 0,
         "fail_closed_as_configured"),
        # one recovered plus four failed requests is NOT recovery under recompute
        (res(exercised_terminals={"recovered": 1, "failed_closed": 4}), fx, True, True, True, False, False, 5,
         "inconclusive_mixed_outcomes"),
        # and recovery is not what the fail policy promised
        (res(exercised_terminals={"recovered": 5}), ff, True, True, True, False, False, 5,
         "inconclusive_mixed_outcomes"),
        (res(exercised_terminals={"recovered": 4, "exposed_answer_unchanged": 1}), fx, True, True, True, False, False,
         5, "inconclusive_mixed_outcomes"),
        (res(exercised_terminals={"recovered": 4, "declined_answer_differs": 1}), fx, True, True, True, False, False,
         5, "inconclusive_mixed_outcomes"),
        (res(exercised_terminals={"recovered": 4}, unexercised_terminals={"clean": 1}), fx, True, True, True, False,
         False, 0, "recovery_demonstrated"),
        (res(exercised_terminals={"recovered": 4}, unexercised_terminals={"changed_nondeterministic": 1}), fx, True,
         True, True, False, False, 5, "inconclusive_mixed_outcomes"),
        (res(exercised_terminals={"recovered": 2}, unexercised_terminals={"clean": 3}), fx, True, True, True, False,
         False, 5, "inconclusive_too_few_replays_exercised"),
        (res(exercised_terminals={"recovered": 4}, runs_total=4), fx, True, True, True, False, False, 5,
         "inconclusive_incomplete_replays"),
        (res(exercised_terminals={"declined_before_read": 5}), fx, True, True, True, False, False, 5,
         "inconclusive_declined_before_read"),
        (res(exercised_terminals={"recovered": 4, "declined_unverifiable": 1}), fx, True, True, True, False, False,
         5, "inconclusive_lease_check_could_not_run"),
        (res(exercised_terminals={"recovered": 4, "inconclusive_decline_unattributed": 1}), fx, True, True, True,
         False, False, 5, "inconclusive_decline_unattributed"),
        (res(discovered_harm_once=True), un, True, True, True, False, False, 5,
         "harm_found_in_discovery_not_reproduced_in_replays"),
        (res(), un, True, True, True, False, False, 0, "no_corruption_despite_trigger"),
    ]
    for r, variant, ctl, base, trig, unres, nocand, code, verdict in cases:
        assert X._verdict(r, variant, ctl, base, trig, unres, no_candidate=nocand) == code, verdict
        assert r["verdict"] == verdict
    r = res(exercised_terminals={"recovered": 5})
    assert X._verdict(r, fx, True, True, True, False, any_unclean=True) == 5
    assert r["verdict"] == "inconclusive_unclean_start"


def test_idle_is_true_false_or_unknown(monkeypatch):
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    clock = iter(range(0, 1000))
    monkeypatch.setattr(X.time, "time", lambda: next(clock))
    monkeypatch.setattr(X, "num_active", lambda port: 0)
    assert X.wait_idle(timeout=5) is True
    monkeypatch.setattr(X, "num_active", lambda port: 1)
    assert X.wait_idle(timeout=5) is False                 # busy throughout: the run would not start clean
    monkeypatch.setattr(X, "num_active", lambda port: None)
    assert X.wait_idle(timeout=5) is None                  # counts never readable: unknown, not idle


def test_decode_sends_the_request_id_it_will_be_matched_by(monkeypatch):
    seen = {}

    def post(url, body, timeout=120):
        seen.update(body)
        return {"choices": [{"text": " PLUM", "finish_reason": "length"}], "id": "cmpl-abc123-0"}

    monkeypatch.setattr(X, "post", post)
    assert X.decode("m", "p", {"k": 1}, request_id="abc123") == (" PLUM", None, "cmpl-abc123-0")
    assert seen["request_id"] == "abc123" and seen["kv_transfer_params"] == {"k": 1}
    line = ("WARNING [base_worker.py:3206] Declining KV pulled for request cmpl-abc123-0-9f8e7d6c: engine P rank 0 "
            "released the KV blocks before the read completed. The KV load fails.")
    assert X.match_decline(X.declines([line]), ["abc123"])["cause"] == "lease_mismatch"


class _World:
    """A stand-in prefill/decode pair shaped like the real H100 runs: a read released when its block is reallocated
    (about 5 s, after the 4 s lease) refers to released blocks but still reads the old bytes, because the new owner
    has not written yet; a read released when the block's new owner frees it (about 6 s), or held to its 14 s
    timeout, reads the new owner's bytes. The fixed build declines every stale read on the decode side, then
    recomputes (or fails the request, under kv_load_failure_policy=fail). flaky: the rewrite garbles only once."""

    BASE = " PLUM. Then the letters."

    def __init__(self, out, variant, policy="recompute", harmful=True, flaky=False, verify_unavailable=False,
                 recompute_text=None, mixed=False, local=None, decline_all=False, decline_other_id=False,
                 harm_until=None):
        self.out, self.variant, self.policy, self.harmful, self.flaky = out, variant, policy, harmful, flaky
        self.verify_unavailable, self.mixed = verify_unavailable, mixed      # fixed-build misbehaviours to catch
        self.recompute_text = recompute_text if recompute_text is not None else self.BASE
        self.local = local if local is not None else self.BASE               # X computed on D alone
        self.rewrites, self.declined = 0, 0
        self.decline_all, self.decline_other_id, self.harm_until = decline_all, decline_other_id, harm_until
        self.evs = [{"t": 0, "kind": "note", "pid": 1, "args": {"hooks": ["fake"]}, "text": "installed"}]
        self.viol, self.n, self.expired, self.lcf = [], 0, 0.0, 0.0

    def ev(self, kind, text="", **args):
        self.evs.append({"t": len(self.evs), "kind": kind, "args": args, "text": text})

    def post(self, url, body, timeout=120):
        self.n += 1
        if url.endswith(":8100/v1/completions"):
            if not body.get("kv_transfer_params"):           # the plain load on P after the lease
                return {"choices": [{"text": " ok"}]}
            prid = f"cmpl-p{self.n}-0-aa"
            for k in ("alloc", "checkpoint"):
                self.ev(k, point="read_submit")
            self.ev("publish", desc=f"{prid}@P", regions=[["P", 1], ["P", 2]], by=prid)
            self.ev("evidence", what="lease_granted", request=prid, engine="P", desc=f"{prid}@P",
                    lease_fields=self.variant == "fixed", lease_nonce=7 if self.variant == "fixed" else None)
            return {"kv_transfer_params": {"remote_engine_id": "P", "remote_block_ids": [1, 2],
                                           "remote_request_id": prid}}
        if not body.get("kv_transfer_params"):           # the local reference: X computed on D, no KV transfer
            return {"choices": [{"text": self.local}], "id": f"cmpl-l{self.n}-0"}
        did = f"cmpl-{body['request_id']}-0" if body.get("request_id") else f"cmpl-d{self.n}"   # vLLM's naming
        with open(os.path.join(self.out, "holds.json")) as f:
            holds = json.load(f)
        prid = body["kv_transfer_params"]["remote_request_id"]
        rq, fixed = f"{did}-ff", self.variant == "fixed"           # vLLM's engine-side id for this attempt
        self.ev("evidence", what="attempt", request=rq, remote_request=prid, remote_engine="P", desc=f"{prid}@P",
                blocks=2, lease_fields=fixed, lease_slot=0 if fixed else None, lease_nonce=7 if fixed else None)
        for k in ("submit", "complete", "consume"):
            self.ev(k)
        self.ev("evidence", what="transfer_done", request=rq)

        def admit(transferred, check=None):
            if check is not None:
                self.ev("evidence", what="lease_check", request=rq, result="failed" if check else "passed",
                        **({"reason": check} if check else {}))
            self.ev("evidence", what="admit", request=rq, transferred=transferred)
        if self.decline_all and self.variant == "fixed" and not holds:   # a verifier that refuses clean pulls too
            with open(os.path.join(self.out, "vllm_D.log"), "a") as f:
                f.write(f"WARNING [base_worker.py:3206] Declining KV pulled for request {did}-ff: engine P rank 0 "
                        "released the KV blocks before the read completed. The KV load fails.\n")
            admit(False, "engine P rank 0 released the KV blocks before the read completed")
            return {"choices": [{"text": self.BASE}], "id": did}
        if not holds:
            admit(True, "" if fixed else None)
            return {"choices": [{"text": self.BASE}], "id": did}
        spec, prid = holds[0], body["kv_transfer_params"]["remote_request_id"]
        by = spec["until"]["args"].get("by")
        rewritten = spec["until"]["kind"] == "free" and isinstance(by, dict)
        if rewritten:                                         # the run must name its own X, not the placeholder
            assert by == {"not": prid}, by
        cond_t = 6.0 if rewritten else 5.0 if "desc_block" in json.dumps(spec["until"]) else None
        reached = cond_t is not None and cond_t <= spec["timeout"]
        stall = cond_t if reached else float(spec["timeout"])
        self.ev("note", f"hold condition for read_submit {did}: {json.dumps(spec['until'])}")
        self.ev("released", f"released read_submit {did}-0-ff "
                            f"({'condition reached' if reached else 'hold expired'}) after {stall:.3f}s")
        if stall <= 4:                                        # read inside the lease: nothing stale
            admit(True, "" if fixed else None)
            return {"choices": [{"text": self.BASE}], "id": did}
        self.expired += 1
        self.ev("note", f"lease expired for {body['kv_transfer_params']['remote_request_id']} on P")
        self.viol.append(SimpleNamespace(rule="R1", step=len(self.evs) - 1, message=(   # index of the event
            f"stale handle submitted: read-{rq}-0 (read by decode {rq}) uses source P block 1 gen 1 was freed")))
        if self.variant == "fixed":
            self.lcf += 1
            self.declined += 1
            why = ("the lease slot of engine P rank 0 could not be read" if self.verify_unavailable
                   else "engine P rank 0 released the KV blocks before the read completed")
            whose = "cmpl-someoneelse-0" if self.decline_other_id else did
            with open(os.path.join(self.out, "vllm_D.log"), "a") as f:
                f.write(f"WARNING [base_worker.py:3206] Declining KV pulled for request {whose}-ff: {why}. "
                        "The KV load fails.\n")
            admit(False, why)
            if self.policy == "fail" or (self.mixed and self.declined % 2 == 0):
                raise urllib.error.HTTPError(url, 500, "Internal Server Error", None, io.BytesIO(b'{"error": "x"}'))
            return {"choices": [{"text": self.recompute_text}], "id": did}
        if rewritten:
            self.rewrites += 1
        admit(True)                                           # the stock build admits the stale bytes
        garbled = self.harmful and (rewritten or not reached) and not (self.flaky and rewritten and self.rewrites > 1)
        if self.harm_until is not None and self.rewrites > self.harm_until:
            garbled = False                                   # harm in discovery only, never again in the replays
        return {"choices": [{"text": "::::::::" if garbled else self.BASE}], "id": did}


class _FC:                                       # a stand-in candidate: hold the read until a block is reallocated
    source, kind = "rule", "alloc"

    def __init__(self, region):
        self.region = region
        self.spec = {"point": "read_submit", "key": None, "until": {"kind": "alloc", "args": {"region": region}}}

    def describe(self):
        return f"hold read_submit until {self.region} is reallocated"

    def hold_spec(self, timeout):
        return dict(self.spec, timeout=timeout)


CANDS = [_FC({"desc_block": 0}), _FC({"key_block": 0})]


def _patch(monkeypatch, w):
    proc = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda t=None: 0, kill=lambda: None)
    monkeypatch.setattr(X.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(X, "wait_ready", lambda *a, **k: 0.1)
    monkeypatch.setattr(X, "wait_idle", lambda timeout=15: True)
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    monkeypatch.setattr(X, "post", w.post)
    monkeypatch.setattr(X, "events", lambda d: list(w.evs))
    monkeypatch.setattr(X, "merge", lambda d, track_content=False: SimpleNamespace(events=list(w.evs),
                                                                                    violations=list(w.viol)))
    monkeypatch.setattr(X, "merge_events", lambda evs, track_content=False: SimpleNamespace(events=list(evs),
                                                                                           violations=list(w.viol)))
    monkeypatch.setattr(X, "candidates", lambda evs, symbolic=True: list(CANDS))
    monkeypatch.setattr(X, "rank", lambda cs, runs: list(cs))
    monkeypatch.setattr(X, "producer_signals", lambda logP: {
        "vllm:nixl_num_kv_expired_reqs": w.expired, "vllm:nixl_num_notifications_after_expiry": 0.0, "log_lines": 0})
    monkeypatch.setattr(X, "consumer_signals", lambda: {
        "vllm:nixl_num_lease_check_failures": w.lcf if w.variant == "fixed" else None,
        "vllm:nixl_num_failed_transfers": 0.0})


def _unpatched_run(tmp_path, monkeypatch, harmful=True, flaky=False):
    out = tmp_path / "unpatched"
    out.mkdir()
    _patch(monkeypatch, _World(str(out), "unpatched", harmful=harmful, flaky=flaky))
    code = X.main(["--out", str(out), "--variant", "unpatched"])
    return out, code, json.loads((out / "RESULT.json").read_text())


def test_the_rewritten_condition_names_x_through_a_placeholder():
    c = X.rewritten_candidate(0)
    s = c.hold_spec(14)
    assert s["until"] == {"kind": "free", "args": {"region": {"desc_block": 0}, "by": {"not": X.X_PREFILL}}}
    assert s["timeout"] == 14 and c.source == "experiment" and c.kind == "rewritten"


def test_end_to_end_unpatched_pins_the_read_right_after_the_overwrite(tmp_path, monkeypatch):
    out, code, res = _unpatched_run(tmp_path, monkeypatch)
    assert code == 1 and res["verdict"] == "harm_reproduced" and res["controls_clean"] and res["baseline_ok"]
    assert len(res["discovery"]) == 3 and res["discovery_confirmations"] == [
        {"label": X.rewritten_candidate(0).describe(), "harm": 3, "of": 3}]
    cand = json.loads((out / "candidate.json").read_text())
    assert cand["kind"] == "rewritten" and X.X_PREFILL in json.dumps(cand["spec"])     # replayable on another image
    assert cand["found_by"]["hold_outcome"] == "condition_reached" and cand["found_by"]["stall_s"] == 6.0
    assert "posting its KV read" in cand["intervention"]
    assert len(res["tried"]) == 5 and res["runs_with_harm"] == 5 and res["hold_outcomes"]["condition_reached"] == 5
    assert res["runs_trigger_demonstrated"] == 5 and res["runs_harm_after_expired_hold"] == 0
    for t in res["tried"]:                           # each run named its own X in the condition it applied
        applied = json.dumps(t["spec_applied"])
        assert X.X_PREFILL not in applied and t["x_prefill_id"] in applied
    # the hazard every time, and here the answer shows it every time too
    assert (res["held_runs"], res["held_runs_ledger_flagged_stale_read"], res["held_runs_answer_visibly_wrong"]) \
        == (8, 8, 8)
    short = res["controls"][1]                        # control B: the same hold point, a stall under the lease
    assert short["stall_s"] == 2.0 and not short["violations"] and short["outcome"]["answer"] == "correct"
    # This legacy outcome fixture deliberately does not provide recorder health, actual
    # submissions or complete ownership history. Its answer/ledger is never evaluator proof.
    assert all(t["lease_rule"] == "unknown" for t in res["tried"])
    assert [c["lease_rule"] for c in res["controls"]] == ["unknown", "unknown"]
    assert res["evaluator_validation"]["passed"] is False


def test_end_to_end_a_race_that_garbles_only_once_is_never_pinned(tmp_path, monkeypatch):
    out, code, res = _unpatched_run(tmp_path, monkeypatch, flaky=True)
    assert code == 1 and res["verdict"] == "harm_reproduced"
    first, second = res["discovery_confirmations"]
    assert (first["harm"], first["of"]) == (1, 3) and (second["harm"], second["of"]) == (3, 3)
    cand = json.loads((out / "candidate.json").read_text())
    assert "key_block" in json.dumps(cand["spec"]) and cand["found_by"]["hold_outcome"] == "expired"
    assert res["hold_outcomes"]["expired"] == 5 and res["runs_harm_after_expired_hold"] == 5
    # the reallocation race read the old bytes: flagged by the ledger, invisible in the answer
    race = [d for d in res["discovery"] if d["source"] == "rule" and "desc_block" in d["label"]]
    assert race and race[0]["outcome"]["terminal"] == "exposed_answer_unchanged"
    assert race[0]["lease_rule"] == "unknown"          # sparse fixture facts cannot establish a violation


def test_end_to_end_fixed_replays_the_saved_stall_and_recovers_or_fails_closed(tmp_path, monkeypatch):
    unp, code, _ = _unpatched_run(tmp_path, monkeypatch)
    assert code == 1
    saved = str(unp / "candidate.json")
    for policy, want_code, want in (("recompute", 0, "recovery_demonstrated"), ("fail", 0, "fail_closed_as_configured")):
        out = tmp_path / policy
        out.mkdir()
        _patch(monkeypatch, _World(str(out), "fixed", policy=policy))
        assert X.main(["--out", str(out), "--variant", "fixed", "--kv-load-failure-policy", policy,
                       "--candidate", saved]) == want_code
        res = json.loads((out / "RESULT.json").read_text())
        assert res["verdict"] == want and res["discovery"] == [] and res["pinned_candidate_in_this_baseline"]
        assert res["runs_rejected_by_fix"] == 5 and res["runs_with_harm"] == 0 and res["controls_clean"]
        assert all(t["decline_line"]["kind"] == "post_read" for t in res["tried"])
        assert all(t["decline_line"]["cause"] == "lease_mismatch" for t in res["tried"])
        # matched by the request id the experiment set in advance, even when the error body has no id
        assert all(t["decline_attribution"] == "request_id" for t in res["tried"])
        assert res["exercised_terminals"] == {("recovered" if policy == "recompute" else "failed_closed"): 5}
        assert all(X.X_PREFILL not in json.dumps(t["spec_applied"]) for t in res["tried"])
        assert not (out / "candidate.json").exists()       # only the discovering run saves a schedule
        assert all(t["lease_rule"] == "unknown" for t in res["tried"])
        assert res["evaluator_validation"]["passed"] is False
        if policy == "recompute":
            assert res["runs_recovered"] == 5 and res["runs_answer_exact"] == 5
        else:
            assert res["runs_failed_closed"] == 5 and all(t["x_error"] == "HTTP 500" for t in res["tried"])


def test_end_to_end_no_stall_that_corrupts_is_inconclusive_not_success(tmp_path, monkeypatch):
    out, code, res = _unpatched_run(tmp_path, monkeypatch, harmful=False)
    assert code == 5 and res["verdict"] == "inconclusive_no_reproducing_candidate"
    assert len(res["discovery"]) == 3 and res["tried"] == [] and not (out / "candidate.json").exists()
    assert res["held_runs_ledger_flagged_stale_read"] == 3 and res["held_runs_answer_visibly_wrong"] == 0


def _fixed_run(tmp_path, monkeypatch, name, policy="recompute", **world):
    unp, code, _ = _unpatched_run(tmp_path, monkeypatch)
    assert code == 1
    out = tmp_path / name
    out.mkdir()
    _patch(monkeypatch, _World(str(out), "fixed", policy=policy, **world))
    code = X.main(["--out", str(out), "--variant", "fixed", "--kv-load-failure-policy", policy,
                   "--candidate", str(unp / "candidate.json")])
    return code, json.loads((out / "RESULT.json").read_text())


def test_end_to_end_a_decline_whose_check_could_not_run_is_not_recovery(tmp_path, monkeypatch):
    code, res = _fixed_run(tmp_path, monkeypatch, "unverifiable", verify_unavailable=True)
    assert code == 5 and res["verdict"] == "inconclusive_lease_check_could_not_run"
    assert res["runs_recovered"] == 0 and res["runs_declined_unverifiable"] == 5
    assert res["runs_caught_stale_lease"] == 0 and res["runs_rejected_by_fix"] == 5


def test_end_to_end_recovered_plus_failed_requests_is_not_recovery(tmp_path, monkeypatch):
    code, res = _fixed_run(tmp_path, monkeypatch, "mixed", mixed=True)
    assert code == 5 and res["verdict"] == "inconclusive_mixed_outcomes"
    assert res["runs_recovered"] >= 1 and res["runs_failed_closed"] >= 1


def test_end_to_end_a_garbled_answer_after_a_decline_is_never_success(tmp_path, monkeypatch):
    code, res = _fixed_run(tmp_path, monkeypatch, "garbled", recompute_text=" PLUM :::::::::::::")
    assert code == 5 and res["verdict"] == "inconclusive_mixed_outcomes"
    assert res["runs_bad_answer_after_decline"] == 5 and res["runs_recovered"] == 0
    assert res["exercised_terminals"] == {"declined_answer_differs": 5}


def test_end_to_end_recovery_to_the_local_reference_counts_as_recovered(tmp_path, monkeypatch):
    other = " PLUM. Then the letters, computed locally."
    code, res = _fixed_run(tmp_path, monkeypatch, "localref", recompute_text=other, local=other)
    assert code == 0 and res["verdict"] == "recovery_demonstrated"
    assert res["local_x"] == other and res["runs_answer_recompute_reference"] == 5


def test_end_to_end_a_run_that_did_not_start_idle_is_inconclusive(tmp_path, monkeypatch):
    out = tmp_path / "busy"
    out.mkdir()
    _patch(monkeypatch, _World(str(out), "unpatched"))
    monkeypatch.setattr(X, "wait_idle", lambda timeout=15: False)
    assert X.main(["--out", str(out), "--variant", "unpatched"]) == 5
    res = json.loads((out / "RESULT.json").read_text())
    assert res["verdict"] == "inconclusive_unclean_start" and res["runs_unclean_start"] >= 1


def test_the_baseline_must_agree_byte_for_byte(tmp_path, monkeypatch):
    out = tmp_path / "noisy"
    out.mkdir()
    w = _World(str(out), "unpatched")
    _patch(monkeypatch, w)
    base_post, k = w.post, {"n": 0}

    def noisy(url, body, timeout=120):           # the clean references disagree: no trustworthy oracle
        r = base_post(url, body, timeout)
        if url.endswith(":8200/v1/completions") and body.get("kv_transfer_params") is not None and k["n"] < 3:
            k["n"] += 1
            r = {"choices": [{"text": w.BASE + (" x" * k["n"])}], "id": "cmpl-n"}
        return r

    monkeypatch.setattr(X, "post", noisy)
    assert X.main(["--out", str(out), "--variant", "unpatched"]) == 5
    res = json.loads((out / "RESULT.json").read_text())
    assert res["verdict"] == "inconclusive_bad_baseline" and not res["baseline_refs_agree"]


def test_end_to_end_a_fix_that_declines_clean_pulls_is_caught_at_the_baseline(tmp_path, monkeypatch):
    code, res = _fixed_run(tmp_path, monkeypatch, "declines_all", decline_all=True)
    assert code == 5 and res["verdict"] == "inconclusive_clean_pulls_declined"
    assert res["baseline_declines"] >= 3 and not res["baseline_pulls_clean"]
    assert "tried" not in res and "discovery" not in res          # stopped before any paid search


def test_end_to_end_a_decline_for_another_request_is_never_credited(tmp_path, monkeypatch):
    code, res = _fixed_run(tmp_path, monkeypatch, "otherid", decline_other_id=True)
    assert code == 5 and res["verdict"] == "inconclusive_decline_unattributed"
    assert all(t["decline_attribution"] == "unattributed" for t in res["tried"]) and res["runs_recovered"] == 0


def test_end_to_end_a_schedule_whose_harm_did_not_reproduce_is_not_saved(tmp_path, monkeypatch):
    out = tmp_path / "once"
    out.mkdir()
    _patch(monkeypatch, _World(str(out), "unpatched", harm_until=3))
    assert X.main(["--out", str(out), "--variant", "unpatched"]) == 5
    res = json.loads((out / "RESULT.json").read_text())
    assert res["verdict"] == "harm_found_in_discovery_not_reproduced_in_replays"
    assert res["candidate"] and not (out / "candidate.json").exists() and not res.get("candidate_saved")


def test_end_to_end_a_replay_released_by_its_timeout_does_not_count_as_exercised(tmp_path, monkeypatch):
    unp, code, res0 = _unpatched_run(tmp_path, monkeypatch, flaky=True)       # pins the key_block hold (expires)
    assert code == 1 and json.loads((unp / "candidate.json").read_text())["found_by"]["hold_outcome"] == "expired"
    out = tmp_path / "expired_fixed"
    out.mkdir()
    _patch(monkeypatch, _World(str(out), "fixed"))
    code = X.main(["--out", str(out), "--variant", "fixed", "--candidate", str(unp / "candidate.json")])
    res = json.loads((out / "RESULT.json").read_text())
    assert code == 5 and res["verdict"] == "inconclusive_too_few_replays_exercised" and res["runs_exercised"] == 0


def test_complete_recordings_reach_experiment_rules_without_answer_inputs():
    from test_lifetime_obligations import recording
    for options, verdict in [({}, 'satisfied'), ({'stale': True}, 'violated'),
                             ({'stale': True, 'check': 'lease_mismatch', 'rejected': True}, 'correctly_rejected')]:
        evs = recording(**options).health()
        findings = X.rules_for_request(evs, [], 'X')
        assert next(f for f in findings if f['rule'] == 'lease_valid_at_use')['verdict'] == verdict


def test_experiment_id_mapping_is_exact_and_ambiguous_match_stays_unknown():
    from test_lifetime_obligations import recording
    evs = recording().health()
    for e in evs:
        if e['args'].get('request') == 'X':
            e['args']['request'] = 'cmpl-client-0-suffix'
    assert X.rules_for_request(evs, [], 'client')[0]['verdict'] == 'satisfied'
    assert X.rules_for_request(evs, [], 'lient')[0]['verdict'] == 'unknown'
    evs.append(dict(kind='evidence', args=dict(what='attempt', request='cmpl-client-1-other')))
    assert X.rules_for_request(evs, [], 'client')[0]['verdict'] == 'unknown'


def test_evaluator_summary_does_not_hide_unmapped_outcomes_or_call_agreement_accuracy():
    def trial(rule, terminal='clean'):
        return dict(label=terminal, lease_rule=rule, outcome=dict(terminal=terminal))
    controls = [trial('satisfied'), trial('satisfied')]
    assert X.evaluator_summary([trial('violated', 'harm_garbled')], controls, 'unpatched')['passed']
    assert X.evaluator_summary([trial('correctly_rejected', 'recovered')], controls, 'fixed')['passed']
    result = X.evaluator_summary([trial('unknown', 'inconclusive')], controls, 'fixed')
    assert not result['passed']
    assert result['hooks_vs_log_and_outcome'][-1]['agrees'] is None
    assert result['hooks_vs_log_and_outcome'][-1]['expected_from_outcome'] is None
    assert not X.evaluator_summary([], controls, 'fixed')['passed']
    assert not X.evaluator_summary([trial('violated')], [], 'unpatched')['passed']


def test_notification_disposition_is_narrow_and_never_hides_raw_evidence():
    from test_lifetime_obligations import recording
    from sdcguard.lifetime.ledger import Violation
    evs = recording(reuse_after_notify=True).health()
    rules = X.rules_for_request(evs, [], 'X')
    step = next(i for i, e in enumerate(evs) if e['kind'] == 'free')
    benign = Violation('R2', step, 'completion polling overlap', regions=(('producer', 4),),
                       ops=('pull:decoder-X:0',))
    destination = Violation('R2', step, 'destination released', regions=(('decode', 8),),
                            ops=('pull:decoder-X:0',))
    other = Violation('R2', step, 'other operation', regions=(('producer', 4),), ops=('other',))
    stale = Violation('R1', step, 'stale submit', regions=(('producer', 4),), ops=('pull:decoder-X:0',))
    raw = [benign, destination, other, stale]
    actionable, dispositions = X.dispose_notification_overlaps(evs, raw, rules)
    assert actionable == [destination, other, stale]
    assert len(dispositions) == 1 and len(raw) == 4
    rules[0]['verdict'] = 'unknown'
    assert X.dispose_notification_overlaps(evs, raw, rules) == (raw, [])
