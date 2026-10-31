"""Offline coverage controls. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import json

import pytest

from sdcguard import client_inventory as inventory


def client(name="client-1", remote="source-request-1", blocks=None):
    return {"client_request_id": name, "transfer": {
        "remote_request_id": remote, "remote_engine_id": "P", "remote_lease_slot": 0,
        "remote_lease_nonce": 19, "remote_block_ids": [[1, 2] if blocks is None else blocks]}}


def batch(*rows):
    return {"schema": inventory.SCHEMA, "workload": "declared-batch", "complete": True,
            "requests": list(rows or (client(),))}


def attempt(row=None, request="internal-decode-1"):
    transfer = (row or client())["transfer"]
    return {"kind": "evidence", "args": {"what": "attempt", "request": request,
        "remote_request": transfer["remote_request_id"], "remote_engine": transfer["remote_engine_id"],
        "lease_slot": transfer["remote_lease_slot"], "lease_nonce": transfer["remote_lease_nonce"],
        "mapping_complete": True, "lease_fields": True,
        "source_regions": [[transfer["remote_engine_id"], block] for block in transfer["remote_block_ids"][0]]}}


def request(row=None):
    row = row or client()
    params = deepcopy(row["transfer"])
    return {"complete": True, "decode_id": row["client_request_id"], "text": "not copied",
            "stages": [{"phase": "prefill", "complete": True,
                        "response": {"kv_transfer_params": deepcopy(params)}},
                       {"phase": "decode", "complete": True,
                        "body": {"kv_transfer_params": deepcopy(params), "prompt": "not copied"}}]}


def performance():
    return {"schema": "sdcguard.owner-pd-arm/1", "load": {"references": 1, "rounds": 1, "requests_per_round": 1},
        "references": [request()], "rounds": [
        {"round": 0, "planned_requests": 1, "complete": True,
         "requests": [request(client("client-2", "source-request-2"))]}]}


def test_exact_full_transfer_descriptor_reconciles_distinct_client_and_internal_ids():
    result = inventory.reconcile(batch(), [attempt()])
    assert result["status"] == "complete"
    assert result["declared_requests"] == result["matched_requests"] == result["observed_attempts"] == 1
    assert result["requests"][0]["engine_request"] == "internal-decode-1"
    assert result["requests"][0]["client_request_id"] == "client-1"
    assert result["issues"] == result["unexpected_attempts"] == []
    assert "lifecycle" in " ".join(result["limits"])


def test_wholly_absent_request_stays_in_denominator_and_prevents_complete():
    second = client("client-2", "source-request-2")
    result = inventory.reconcile(batch(client(), second), [attempt()])
    assert result["status"] == "unknown"
    assert result["declared_requests"] == 2 and result["matched_requests"] == 1
    assert result["requests"][1]["status"] == "unknown"
    assert "no recorded attempt" in result["requests"][1]["reason"]
    assert result["requests"][1]["engine_request"] is None


@pytest.mark.parametrize("field,value", [
    ("remote_request", "source-request-1-suffix"), ("remote_engine", "other"),
    ("lease_slot", 1), ("lease_nonce", 20), ("source_regions", [["P", 2], ["P", 1]])])
def test_every_descriptor_field_including_block_order_is_required(field, value):
    event = attempt()
    event["args"][field] = value
    result = inventory.reconcile(batch(), [event])
    assert result["status"] == "unknown" and result["matched_requests"] == 0


def test_prefix_related_internal_request_id_does_not_bind_different_transfer():
    event = attempt(client(remote="another-source"), request="client-1-0-suffix")
    assert inventory.reconcile(batch(), [event])["matched_requests"] == 0


@pytest.mark.parametrize("value", [False, 0, 1, None, "true"])
def test_inventory_completeness_requires_literal_true(value):
    value = dict(batch(), complete=value)
    assert inventory.reconcile(value, [attempt()])["status"] == "unknown"
    with pytest.raises(ValueError):
        inventory.validate_inventory(value)


@pytest.mark.parametrize("field,value", [
    ("remote_lease_slot", True), ("remote_lease_nonce", False),
    ("remote_lease_nonce", -1), ("remote_lease_nonce", 1 << 64),
    ("remote_block_ids", [[True]]), ("remote_block_ids", [[-1]]),
    ("remote_block_ids", [[1, 1]]), ("remote_block_ids", [[1], [2]]),
    ("remote_block_ids", [[]]), ("remote_block_ids", [1, 2]),
    ("remote_engine_id", ""), ("remote_request_id", None)])
def test_unsupported_client_descriptor_is_unknown(field, value):
    data = batch()
    data["requests"][0]["transfer"][field] = value
    assert inventory.reconcile(data, [attempt()])["status"] == "unknown"


@pytest.mark.parametrize("field,value", [
    ("mapping_complete", False), ("mapping_complete", 1), ("lease_fields", False),
    ("lease_slot", True), ("lease_nonce", -1), ("source_regions", [["other", 1]]),
    ("source_regions", [["P", 1], ["P", 1]])])
def test_unsupported_recorded_descriptor_is_unknown(field, value):
    event = attempt()
    event["args"][field] = value
    result = inventory.reconcile(batch(), [event])
    assert result["status"] == "unknown" and result["matched_requests"] == 0
    assert len(result["unexpected_attempts"]) == 1


def test_duplicate_client_ids_remain_unknown_even_when_descriptors_differ():
    other = client(remote="source-request-2")
    result = inventory.reconcile(batch(client(), other), [attempt(), attempt(other, "internal-2")])
    assert result["status"] == "unknown" and result["matched_requests"] == 0
    assert all("duplicate client request ID" in row["reason"] for row in result["requests"])


def test_duplicate_client_descriptors_are_not_counted_twice():
    other = client("client-2")
    result = inventory.reconcile(batch(client(), other), [attempt()])
    assert result["status"] == "unknown" and result["matched_requests"] == 0


def test_duplicate_recorded_attempt_is_ambiguous_even_with_same_internal_id():
    result = inventory.reconcile(batch(), [attempt(), attempt()])
    assert result["status"] == "unknown" and result["matched_requests"] == 0
    assert result["observed_attempts"] == 2


def test_same_internal_id_for_two_descriptors_is_ambiguous():
    other = client("client-2", "source-request-2")
    result = inventory.reconcile(batch(client(), other), [attempt(), attempt(other)])
    assert result["status"] == "unknown" and result["matched_requests"] == 0


def test_undeclared_extra_attempt_blocks_complete_without_erasing_matched_requests():
    extra = client("client-2", "source-request-2")
    result = inventory.reconcile(batch(), [attempt(), attempt(extra, "internal-2")])
    assert result["status"] == "unknown" and result["matched_requests"] == 1
    assert len(result["unexpected_attempts"]) == 1
    assert result["unexpected_attempts"][0]["request"] == "internal-2"


def test_malformed_record_cannot_hide_behind_other_clean_record():
    assert inventory.reconcile(batch(), [attempt(), None])["status"] == "unknown"
    assert inventory.reconcile(batch(), [attempt(), {"kind": "evidence", "args": None}])["status"] == "unknown"


def test_nonattempt_events_are_not_in_the_attempt_denominator():
    result = inventory.reconcile(batch(), [{"kind": "note", "args": {}}, attempt()])
    assert result["status"] == "complete" and result["observed_attempts"] == 1


def test_normalizer_includes_references_and_all_planned_requests_without_payloads():
    value = inventory.from_performance(performance(), workload="ordinary")
    assert value["workload"] == "ordinary" and len(value["requests"]) == 2
    assert [row["client_request_id"] for row in value["requests"]] == ["client-1", "client-2"]
    raw = json.dumps(value)
    assert "prompt" not in raw and "text" not in raw and "not copied" not in raw


@pytest.mark.parametrize("change", ["round_missing_row", "round_incomplete", "client_incomplete",
    "stage_incomplete", "stage_missing", "stage_order", "handoff_disagreement", "duplicate_client",
    "missing_load", "missing_reference", "missing_round", "wrong_reference_count", "wrong_round_id",
    "boolean_load_count"])
def test_normalizer_refuses_whole_incomplete_or_ambiguous_batch(change):
    value = performance()
    if change == "round_missing_row":
        value["rounds"][0]["planned_requests"] = 2
    elif change == "round_incomplete":
        value["rounds"][0]["complete"] = False
    elif change == "client_incomplete":
        value["references"][0]["complete"] = False
    elif change == "stage_incomplete":
        value["references"][0]["stages"][1]["complete"] = False
    elif change == "stage_missing":
        value["references"][0]["stages"].pop()
    elif change == "stage_order":
        value["references"][0]["stages"].reverse()
    elif change == "handoff_disagreement":
        value["references"][0]["stages"][1]["body"]["kv_transfer_params"]["remote_lease_nonce"] = 20
    elif change == "duplicate_client":
        value["rounds"][0]["requests"][0]["decode_id"] = "client-1"
    elif change == "missing_load":
        value.pop("load")
    elif change == "missing_reference":
        value["references"] = []
    elif change == "missing_round":
        value["rounds"] = []
    elif change == "wrong_reference_count":
        value["load"]["references"] = 2
    elif change == "wrong_round_id":
        value["rounds"][0]["round"] = 1
    elif change == "boolean_load_count":
        value["load"]["references"] = True
    with pytest.raises(ValueError):
        inventory.from_performance(value)


@pytest.mark.parametrize("raw", ['{"same": 1, "same": 2}', '{"value": NaN}', '{"value": Infinity}'])
def test_json_reader_rejects_ambiguous_or_nonfinite_input(tmp_path, raw):
    source = tmp_path / "source.json"
    source.write_text(raw)
    with pytest.raises(ValueError):
        inventory.read_json(source)


def test_reader_rejects_symlink_and_directory(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("{}")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(ValueError):
        inventory.read_json(link)
    with pytest.raises(ValueError):
        inventory.read_json(tmp_path)


def test_reader_reports_deep_json_as_invalid_input_without_a_raw_recursion_error(tmp_path):
    source = tmp_path / "nested.json"
    source.write_text("[" * 2000 + "0" + "]" * 2000)
    with pytest.raises(ValueError, match="nesting exceeds"):
        inventory.read_json(source)


def test_reader_has_the_same_explicit_container_depth_bound_on_each_python_version(tmp_path):
    source = tmp_path / "bounded.json"
    source.write_text("[" * inventory.MAX_JSON_DEPTH + "0" + "]" * inventory.MAX_JSON_DEPTH)
    assert type(inventory.read_json(source)) is list
    source.write_text("[" * (inventory.MAX_JSON_DEPTH + 1) + "0" + "]" * (inventory.MAX_JSON_DEPTH + 1))
    with pytest.raises(ValueError, match="nesting exceeds"):
        inventory.read_json(source)


def test_import_command_writes_new_file_and_never_overwrites(tmp_path, capsys):
    source, output = tmp_path / "source.json", tmp_path / "normalized.json"
    source.write_text(json.dumps(performance()))
    args = ["import-performance", str(source), "--out", str(output)]
    assert inventory.main(args) == 0
    before = output.read_bytes()
    assert inventory.main(args) == 2
    assert output.read_bytes() == before
    assert inventory.validate_inventory(json.loads(before))["complete"] is True


def test_import_command_does_not_create_output_for_incomplete_batch(tmp_path, capsys):
    source, output = tmp_path / "source.json", tmp_path / "normalized.json"
    data = performance()
    data["rounds"][0]["planned_requests"] = 2
    source.write_text(json.dumps(data))
    assert inventory.main(["import-performance", str(source), "--out", str(output)]) == 2
    assert not output.exists()


def test_bounds_are_enforced_without_large_fixtures(monkeypatch, tmp_path):
    monkeypatch.setattr(inventory, "MAX_REQUESTS", 1)
    assert inventory.reconcile(batch(client(), client("c2", "s2")), [attempt()])["status"] == "unknown"
    with pytest.raises(ValueError):
        inventory.from_performance(performance())
    monkeypatch.setattr(inventory, "MAX_EVENTS", 0)
    assert inventory.reconcile(batch(), [attempt()])["status"] == "unknown"
    monkeypatch.setattr(inventory, "MAX_FILE_BYTES", 1)
    source = tmp_path / "input"
    source.write_text("{}")
    with pytest.raises(ValueError):
        inventory.read_json(source)


def test_inputs_are_unchanged_by_validation_normalization_and_reconciliation():
    value, events, saved = batch(), [attempt()], performance()
    before = deepcopy((value, events, saved))
    inventory.validate_inventory(value)
    inventory.reconcile(value, events)
    inventory.from_performance(saved)
    assert (value, events, saved) == before
