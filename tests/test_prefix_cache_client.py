"""Client reconciliation of native CPU observations. Owner: Hari Krishna Kancharla."""
from copy import deepcopy

import pytest

from sdcguard import checksum_client as client, evidence_io as io, prefix_cache_observer as cache
from test_checksum_client import client_case, _binding_budget, _checksum_budget, _evaluate, _on
from test_prefix_cache_observer import classes
from test_kv_checksum_observer import native, _with_observer


def prefix_case(tmp_path, native, *, local=16, producer_local=16):
    values = client_case()
    plan, inventory, captures = values
    plan["prefix_caching"] = True
    inventory["plan_sha256"] = captures["plan_sha256"] = io.digest(io.json_bytes(plan))
    capture = captures["captures"][0]
    context = {key: capture["startup"][key] for key in client.CONTEXT_KEYS}
    internal = capture["binding_snapshots"][0]["records"][0]["internal_request_id"]
    _, _, checksum, _ = _with_observer(native, local_tokens=32, local_prefix=local, request_id=internal)
    checksum["capture_context"] = dict(context, process_id=102)
    capture["checksum_snapshots"] = [checksum]
    prefill = dict(startup=deepcopy(capture["startup"]), binding_snapshots=deepcopy(capture["binding_snapshots"]))
    capture["prefill_capture"] = prefill
    producer_id = "synthetic-producer-internal"
    binding = prefill["binding_snapshots"][0]
    binding["process_id"] = 201
    binding["records"][0].update(external_request_id="cmpl-prefill-on-0", external_req_id="cmpl-prefill-on-0",
                                   internal_request_id=producer_id)
    _binding_budget(binding)
    for role, envelope, pid, request_id, local_tokens, remote_tokens in (
        ("decode", capture, 102, internal, local, 32-local),
        ("prefill", prefill, 202, producer_id, producer_local, 0),
    ):
        envelope["startup"].update(role=role, process_id=pid, prefix_caching=True)
        envelope["startup_sha256"] = io.digest(io.json_bytes(envelope["startup"]))
        Request, Scheduler, Stats, Status = classes(tmp_path / role)
        observer = cache.PrefillCacheObserver(**context, role=role)
        observer.install_cpu_source_functions(Request, Scheduler, source_root=tmp_path / role)
        request = Request()
        request.request_id, request.prompt_token_ids, request.num_prompt_tokens = request_id, list(range(32)), 32
        request.prefill_stats = Stats()
        request.prefill_stats.set(32, local_tokens, remote_tokens)
        request.num_preemptions, request.status, request._output_token_ids = 0, Status.FINISHED_LENGTH_CAPPED, [5]
        request.take_prefill_stats()
        snapshot = observer.close()
        snapshot["process_id"] = pid
        envelope["cache_snapshots"] = [snapshot]
    for row in inventory["requests"]:
        for role in ("prefill", "decode"):
            row[role]["response_body"]["usage"]["completion_tokens"] = 1
        row["prefill"]["response_body"]["kv_transfer_params"]["remote_prefill_cached_tokens"] = producer_local
        row["decode"]["request_body"]["kv_transfer_params"]["remote_prefill_cached_tokens"] = producer_local
    return values


@pytest.mark.parametrize("local,producer_local", [(0,0),(0,16),(16,0),(16,16)])
def test_prefix_cache_is_joined_to_both_client_stages(tmp_path, native, local, producer_local):
    result = _evaluate(prefix_case(tmp_path, native, local=local, producer_local=producer_local))
    row = _on(result)
    assert result["client_inventory"]["status"] == "complete"
    assert row["native_result"] == "passed"
    assert row["token_coverage"] == dict(prompt_tokens=32, transferred_tokens_checked=32-local,
        decode_cached_tokens_unchecked=local, decode_locally_computed_tokens=0,
        prefill_cached_tokens=producer_local, prefill_reused_cached_blocks=producer_local > 0)
    if local:
        assert row["check_status"] == "unverified" and row["coverage"] == "partial"
        assert row["reasons"] == ["cached prefix not covered by checksum"]
        assert result["summary"]["verified_on_requests"] == 0
    else:
        assert row["check_status"] == "passed" and row["coverage"] == "complete"
        assert row["reasons"] == []
        assert result["summary"]["verified_on_requests"] == 1
    assert result["release_decision"] == "HOLD" and result["runtime_qualified"] is False


@pytest.mark.parametrize("role", ["prefill", "decode"])
@pytest.mark.parametrize("change", ["missing_snapshot","missing_binding","wrong_request","wrong_digest",
    "late_preemption","wrong_completion","missing_terminal","duplicate_stats","wrong_counts","wrong_pid"])
def test_missing_or_contradictory_cache_facts_never_pass(tmp_path, native, role, change):
    values = prefix_case(tmp_path, native, local=0)
    envelope = values[2]["captures"][0]
    if role == "prefill": envelope = envelope["prefill_capture"]
    snapshot = envelope["cache_snapshots"][0]
    record = snapshot["records"][0]
    if change == "missing_snapshot": envelope["cache_snapshots"] = []
    elif change == "missing_binding": envelope["binding_snapshots"] = []
    elif change == "wrong_request": record["request_id"] += "-other"
    elif change == "wrong_digest": record["prompt_sha256"] = "0"*64
    elif change == "late_preemption": record["preemptions"] = 1
    elif change == "wrong_completion": record["output_tokens"] = 2
    elif change == "missing_terminal": record["status"] = "RUNNING"
    elif change == "duplicate_stats": snapshot["records"].append(dict(deepcopy(record), sequence=2))
    elif change == "wrong_counts": record["stats"]["num_external_cached_tokens"] += 1
    else: snapshot["process_id"] += 1
    snapshot["budgets"].update(retained_records=len(snapshot["records"]),
        retained_bytes=sum(len(io.json_bytes(r)) for r in snapshot["records"]))
    result = _evaluate(values)
    assert result["summary"]["declared_requests"] == 2
    assert result["summary"]["verified_on_requests"] == 0
    assert _on(result)["check_status"] != "passed" and _on(result)["reasons"]


def test_native_span_cannot_disagree_with_decode_cache_accounting(tmp_path, native):
    values = prefix_case(tmp_path, native, local=16)
    snapshot = values[2]["captures"][0]["cache_snapshots"][0]
    snapshot["records"][0]["stats"].update(num_local_cached_tokens=0, num_external_cached_tokens=32)
    snapshot["budgets"]["retained_bytes"] = sum(len(io.json_bytes(r)) for r in snapshot["records"])
    row = _on(_evaluate(values))
    assert "native_cache_to_checksum_span_unverified" in row["reasons"]
    assert row["token_coverage"]["transferred_tokens_checked"] is None
    assert row["coverage"] == "unknown"


def test_producer_descriptor_must_name_the_observed_prefill_request(tmp_path, native):
    values = prefix_case(tmp_path, native, local=0)
    row = values[1]["requests"][1]
    row["prefill"]["response_body"]["kv_transfer_params"]["remote_request_id"] = "other-producer"
    row["decode"]["request_body"]["kv_transfer_params"]["remote_request_id"] = "other-producer"
    checked = _on(_evaluate(values))
    assert "producer_cache_transfer_request_binding_unverified" in checked["reasons"]
    assert checked["check_status"] == "unverified"


@pytest.mark.parametrize("change", ["external","descriptor","absent_descriptor"])
def test_producer_counts_must_agree_with_the_native_transfer_descriptor(tmp_path, native, change):
    values = prefix_case(tmp_path, native, local=0)
    snapshot = values[2]["captures"][0]["prefill_capture"]["cache_snapshots"][0]
    if change == "external":
        snapshot["records"][0]["stats"].update(num_external_cached_tokens=1, num_computed_tokens=15, num_cached_tokens=17)
        snapshot["budgets"]["retained_bytes"] = sum(len(io.json_bytes(r)) for r in snapshot["records"])
    else:
        row = values[1]["requests"][1]
        for descriptor in (row["prefill"]["response_body"]["kv_transfer_params"], row["decode"]["request_body"]["kv_transfer_params"]):
            if change == "descriptor": descriptor["remote_prefill_cached_tokens"] = 0
            else: descriptor.pop("remote_prefill_cached_tokens")
    checked = _on(_evaluate(values))
    assert "producer_cache_descriptor_counts_unverified" in checked["reasons"]
    assert checked["check_status"] == "unverified"
    assert checked["token_coverage"]["prefill_cached_tokens"] is None


def test_stats_only_full_local_hit_does_not_claim_a_compared_suffix(tmp_path, native):
    values = prefix_case(tmp_path, native, local=16)
    capture = values[2]["captures"][0]
    snapshot = capture["cache_snapshots"][0]
    snapshot["records"][0]["stats"].update(num_local_cached_tokens=32, num_external_cached_tokens=0)
    snapshot["budgets"]["retained_bytes"] = sum(len(io.json_bytes(r)) for r in snapshot["records"])
    checksum = capture["checksum_snapshots"][0]
    checksum["records"] = []
    _checksum_budget(checksum)
    row = _on(_evaluate(values))
    assert row["coverage"] == "unknown" and row["check_status"] == "unverified"
    assert row["token_coverage"]["transferred_tokens_checked"] == 0
    assert row["token_coverage"]["decode_cached_tokens_unchecked"] == 32
    assert "cached prefix not covered by checksum" in row["reasons"]
    assert "no_transfer_checksum_comparison_recorded" in row["reasons"]
