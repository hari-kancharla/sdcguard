"""Bookkeeping and packaging checks; no live engine or failure execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from sdcguard.assessment import TOPOLOGY, assess, inspect_engine, pins
from sdcguard.lifetime.obligations import Facts, lease_valid_at_use
from sdcguard.lifetime.outcomes import terminal_outcome
from sdcguard.overhead import summarize
from sdcguard.regression import export_case, historical_case, judge
from test_lifetime_obligations import recording


@pytest.mark.parametrize("field", ["seq", "clock_id", "mono", "session", "pid", "dropped"])
def test_unsequenced_lifecycle_cannot_turn_clean_recording_into_violation(field):
    events = recording().health()
    for kind, mono, args in [("free", 5, {"region": ["producer", 4], "by": "R"}),
                             ("alloc", 6, {"region": ["producer", 4], "owner": "another"})]:
        row = dict(events[0], kind=kind, mono=mono, args=args, seq=50)
        row.pop(field)
        events.append(row)
    assert lease_valid_at_use(Facts(events), [], "X").verdict == "unknown"


def terminal():
    rid = "cmpl-chosen-0-worker"
    return {"request": rid, "verdict": "correctly_rejected"}, {
        "x_request_id": "chosen", "x_finished": True, "hung": False, "x_exc": None,
        "decline_attribution": "request_id", "decline_line": {
            "req": rid, "kind": "post_read", "cause": "lease_mismatch"}, "x_error": None, "x": "answer"}


def test_lifecycle_outside_healthy_receipt_cannot_supply_a_violation():
    events = recording().health()
    for kind, mono, args in [("free", 5, {"region": ["producer", 4], "by": "R"}),
                             ("alloc", 6, {"region": ["producer", 4], "owner": "another"})]:
        events.append(dict(events[0], kind=kind, mono=mono, args=args, seq=50))
    assert lease_valid_at_use(Facts(events), [], "X").verdict == "unknown"


def test_recovery_and_refusal_are_distinct_client_outcomes():
    f, t = terminal()
    assert terminal_outcome(f, t, ["answer"] * 3)["outcome"] == "recovered"
    t.update(x=None, x_error="HTTP 500")
    assert terminal_outcome(f, t, ["answer"] * 3)["outcome"] == "refused"


def test_refusal_requires_observed_terminal_closure_not_policy():
    r = recording(rejected=True, check="lease_mismatch")
    admission = next(e for e in r.events if e["args"].get("what") == "admit")
    next(e for e in r.events if e["args"].get("what") == "load_failed")["args"]["policy"] = "fail"
    admission["args"]["closed"] = True
    assert lease_valid_at_use(Facts(r.health()), [], "X").verdict == "unknown"
    # This is a bookkeeping receipt fixture, not execution of an inference failure.
    r = recording(rejected=True, check="lease_mismatch")
    next(e for e in r.events if e["args"].get("what") == "load_failed")["args"]["policy"] = "fail"
    admission = next(e for e in r.events if e["args"].get("what") == "admit")
    admission["args"]["closed"] = True
    admission["mono"] = 25
    r.evidence("request_closed", 24, "DS", request="X", status="FINISHED_ERROR")
    # Put receipt in actual sequence order, maintaining the event envelope.
    rows = sorted(r.events, key=lambda e: e["mono"])
    counts = {}
    for row in rows:
        counts[row["pid"]] = counts.get(row["pid"], 0) + 1
        row["seq"] = counts[row["pid"]]
    r.events = rows
    assert lease_valid_at_use(Facts(r.health()), [], "X").verdict == "refused"


@pytest.mark.parametrize("change", [{"x": "answer WRONG"}, {"x_finished": False}, {"hung": True},
                                    {"decline_attribution": "unattributed"}, {"x_request_id": "other"}])
def test_uncertain_client_terminal_never_claims_recovery(change):
    f, t = terminal()
    t.update(change)
    assert terminal_outcome(f, t, ["answer"] * 3)["outcome"] == "unknown"


def test_client_output_cannot_backfill_missing_protocol_evidence():
    f, t = terminal()
    f["verdict"] = "unknown"
    assert terminal_outcome(f, t, ["answer"] * 3)["outcome"] == "unknown"


@pytest.mark.parametrize("field", ["x_error", "x_exc", "x", "decline_line"])
def test_absent_terminal_fact_is_unknown(field):
    f, t = terminal()
    t.pop(field)
    assert terminal_outcome(f, t, ["answer"] * 3)["outcome"] == "unknown"


def test_changed_engine_and_unsupported_topology_are_inconclusive(tmp_path):
    assert inspect_engine(tmp_path, "0.33.0")["fix"] == "unknown"
    assert inspect_engine(tmp_path, "0.32.0")["adapter_fits"] is False
    r = assess({"schema": 1, "engine": "vllm", "topology": dict(TOPOLOGY, tp=2)})
    assert r["status"] == "inconclusive"


def test_model_main_is_not_a_pin():
    r = assess({"schema": 1, "engine": "vllm", "topology": TOPOLOGY,
                "model": "Qwen/Qwen3-0.6B", "model_revision": "main"})
    assert "resolved Qwen3-0.6B model commit; main is not a pin" in r["missing"]


def test_export_never_overwrites_and_only_exports_saved_schedule(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    (checkout / "experiments").mkdir(parents=True)
    (checkout / "experiments/lifetime_pd_check.py").write_text("# stand-in source, never executed\n")
    import hashlib
    import sdcguard.regression as regression
    original = historical_case()
    original["reviewed_harness_sha256"] = hashlib.sha256((checkout / "experiments/lifetime_pd_check.py").read_bytes()).hexdigest()
    monkeypatch.setattr(regression, "historical_case", lambda: deepcopy(original))
    dest = tmp_path / "case"
    export_case(dest, checkout)
    assert json.loads((dest / "candidate.json").read_text())["spec"] == historical_case()["candidate"]["spec"]
    assert json.loads((dest / "case.json").read_text())["minimality"] == "unverified"
    assert "'run','--case-dir'" in (dest / "run.py").read_text()
    with pytest.raises(ValueError, match="exists"):
        export_case(dest, checkout)


def test_missing_or_incomplete_result_never_passes():
    for r in [{}, {"mode": "compare", "variant": "fixed", "tried": []}]:
        assert judge(r, "fixed")["status"] == "inconclusive"


def perf(enabled):
    return {"schema": "sdcguard.perf/1", "model": "same", "identity": {k: ("a"*40 if k == "model_revision" else "a"*64) for k in
        ("engine_files_sha256", "model_revision", "weights_sha256", "gpu_uuid", "deployment_config_sha256")},
        "checks_enabled": enabled, "complete": True, "load": {"prompts": 100, "rounds": 3,
            "concurrency": 16, "max_tokens": 24, "warmup": 0, "prompts_sha256": "a"*64},
        "rounds": [dict({k: 100 for k in ("output_tokens_per_s", "ttft_p99_ms", "itl_p99_ms", "latency_p99_ms")},
                        requests=100, completed=100, errors=0, tokens_counted_by_server=True)]*3,
        "median": {k: 100 for k in ("output_tokens_per_s", "ttft_p99_ms", "itl_p99_ms", "latency_p99_ms")}}


def test_overhead_requires_matched_identity_and_actual_tail_metrics():
    b, c = perf(False), perf(True)
    assert summarize(b, c)["status"] == "measured"
    c["identity"]["model_revision"] = "different"
    assert summarize(b, c)["status"] == "inconclusive"


@pytest.mark.parametrize("bad", [None, [], {"rounds": [None]}, {"rounds": [1]}])
def test_invalid_json_shapes_are_inconclusive(bad):
    assert judge(bad, "fixed")["status"] == "inconclusive"
    assert summarize(bad, perf(True))["status"] == "inconclusive"


def fresh_result():
    from sdcguard.regression import digest, replay_model_pin
    pin = replay_model_pin()
    result = json.loads((Path(__file__).resolve().parents[1] / "results/61229_saved/fixed_RESULT.json").read_text())
    result["model"] = "/snapshot/" + pin["revision"]
    result["model_revision"] = pin["revision"]
    result["tokenizer_revision"] = pin["revision"]
    result["evaluator_validation"] = {"passed": True}
    manifest = {"schema": 1, "complete": True, "variant": "fixed",
                "nixl_version": "1.4.1", "harness_exit_code": 0,
                "harness_sha256": historical_case()["reviewed_harness_sha256"],
                "candidate": historical_case()["candidate"], "model_path": result["model"],
                "model_pin": pin,
                "engine_identity": {"file_sha256": pins()["variants"]["fixed"], "version": "0.32.0", "adapter_fits": True},
                "result_canonical_sha256": digest(result)}
    return result, manifest


def test_fresh_result_requires_manifest_and_actual_resolved_condition():
    from sdcguard.regression import digest
    result, manifest = fresh_result()
    assert judge(result, "fixed")["status"] == "inconclusive"
    assert judge(result, "fixed", manifest)["status"] == "pass"
    result["tried"][0]["spec_applied"]["timeout"] = 15
    manifest["result_canonical_sha256"] = digest(result)
    assert judge(result, "fixed", manifest)["status"] == "inconclusive"


def test_fresh_result_cannot_hide_failed_evaluator_or_unsupported_nixl():
    from sdcguard.regression import digest
    result, manifest = fresh_result()
    result["evaluator_validation"]["passed"] = False
    manifest["result_canonical_sha256"] = digest(result)
    assert judge(result, "fixed", manifest)["status"] == "inconclusive"
    result, manifest = fresh_result()
    manifest["harness_exit_code"] = 2
    assert judge(result, "fixed", manifest)["status"] == "inconclusive"
    manifest["harness_exit_code"] = 0
    manifest["nixl_version"] = "other"
    assert judge(result, "fixed", manifest)["status"] == "inconclusive"


def test_fixed_wrong_output_is_fail_but_contradictory_terminal_is_unknown():
    from sdcguard.regression import digest
    result, manifest = fresh_result()
    result["tried"][0]["x"] = "wrong"
    manifest["result_canonical_sha256"] = digest(result)
    assert judge(result, "fixed", manifest)["status"] == "fail"
    result["tried"][0]["x_exc"] = "incomplete transport"
    manifest["result_canonical_sha256"] = digest(result)
    assert judge(result, "fixed", manifest)["status"] == "inconclusive"


def test_every_proof_stream_requires_matching_engine_identity(tmp_path, monkeypatch):
    import sdcguard.assessment as assessment
    identity = {"adapter_fits": True, "fix": "present"}
    monkeypatch.setattr(assessment, "inspect_engine", lambda *a: deepcopy(identity))
    r = recording()
    # Add recorder identity to the same existing hook note and each other stream.
    r.events[0]["args"].update(engine_identity=identity, nixl_version="1.4.1")
    for pid in ["D", "DS"]:
        r.event("note", 0.1, pid, engine_identity=identity, nixl_version="1.4.1")
    r.events.sort(key=lambda e: e["mono"])
    counts = {}
    for row in r.events:
        counts[row["pid"]] = counts.get(row["pid"], 0)+1
        row["seq"] = counts[row["pid"]]
    events = r.health()
    eventfile = tmp_path / "events.jsonl"
    eventfile.write_text(''.join(json.dumps(e)+'\n' for e in events))
    config = {"schema": 1, "engine": "vllm", "engine_version": "0.32.0", "nixl_version": "1.4.1",
              "engine_root": ".", "model": "Qwen/Qwen3-0.6B", "model_revision": "a"*40,
              "topology": TOPOLOGY, "events": "events.jsonl"}
    assert assess(config, base=tmp_path)["status"] == "fine"
    r.events[0]["args"]["engine_identity"] = {"adapter_fits": False}
    eventfile.write_text(''.join(json.dumps(e)+'\n' for e in r.events))
    assert assess(config, base=tmp_path)["status"] == "inconclusive"


def test_missing_tail_metric_is_inconclusive():
    c = perf(True)
    c["median"]["ttft_p99_ms"] = None
    assert summarize(perf(False), c)["status"] == "inconclusive"


def test_closure_hook_observes_completion_and_preserves_original_return():
    import os
    from types import SimpleNamespace
    from sdcguard.lifetime import vllm_nixl as adapter
    from sdcguard.lifetime.adapter import LocalSink, Probe
    adapter.reset()
    sink = LocalSink()
    adapter._PROBE, adapter._PROBE_PID = Probe(sink), os.getpid()
    class Scheduler:
        recompute_kv_load_failures = False
        failed_recving_kv_req_ids = set()
        def _handle_failed_recving(self, ids):
            return set(ids)
        def _free_request(self, request):
            request.status = SimpleNamespace(name="FINISHED_ERROR")
            return 42
    try:
        adapter.install_scheduler(Scheduler)
        s = Scheduler()
        request = SimpleNamespace(request_id="X")
        assert s._handle_failed_recving({"X"}) == {"X"}
        assert not any(e["args"].get("what") == "request_closed" for e in sink.events)
        assert s._free_request(request) == 42
        closed = [e for e in sink.events if e["args"].get("what") == "request_closed"]
        assert len(closed) == 1 and closed[0]["args"]["status"] == "FINISHED_ERROR"
    finally:
        adapter.reset()


def test_exported_help_does_not_import_sibling_checkout(tmp_path):
    import os
    import subprocess
    import sys
    checkout = Path(__file__).resolve().parents[1]
    sibling = tmp_path / "sibling"
    bad = sibling / "src/sdcguard"
    bad.mkdir(parents=True)
    (bad / "__init__.py").write_text("raise RuntimeError('SIBLING_CHECKOUT_IMPORTED')\n")
    case = sibling / "case"
    export_case(case, checkout)
    # Supply the trusted package path as a stand-in for a clean wheel install.
    code = (f"import sys,runpy;sys.path.insert(0,{str(checkout/'src')!r});"
            f"sys.argv=[{str(case/'existing_case.py')!r},'--help'];runpy.run_path(sys.argv[0],run_name='__main__')")
    env = dict(os.environ, SDCGUARD_PACKAGED_REPLAY="1")
    good = subprocess.run([sys.executable, "-I", "-B", "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert good.returncode == 0, good.stderr
    env.pop("SDCGUARD_PACKAGED_REPLAY")
    control = subprocess.run([sys.executable, "-I", "-B", "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert control.returncode != 0 and "SIBLING_CHECKOUT_IMPORTED" in control.stderr


def test_missing_optional_engine_package_is_inconclusive(tmp_path, monkeypatch, capsys):
    import importlib.metadata
    import sdcguard.assessment as assessment
    from sdcguard.regression import main
    checkout = Path(__file__).resolve().parents[1]
    case = tmp_path / "case"
    export_case(case, checkout)
    monkeypatch.setattr(assessment, "inspect_engine", lambda *a: {"variant": "fixed"})
    def absent(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "distribution", absent)
    code = main(["run", "--case-dir", str(case), "--variant", "fixed", "--engine-root", str(tmp_path),
                 "--model-snapshot", str(tmp_path), "--out", str(tmp_path/'output')])
    assert code == 2 and '"inconclusive"' in capsys.readouterr().out
