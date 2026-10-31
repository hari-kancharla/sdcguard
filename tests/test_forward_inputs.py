"""CPU structural cases for forward metadata, never claimed as GPU traces."""
from copy import deepcopy

import pytest

from sdcguard.lifetime import forward_inputs as F


def ordinary_records():
    """A mixed ordinary batch with a historical prefix absent from the write map."""
    session = "1" * 32
    header = {"schema": 1, "kind": "header", "session": session, "seq": 0,
              "process_incarnation": session,
              "engine": {"name": "vllm", "version": "0.32.0", "source_sha256": F.source_pins(),
                         "dependencies": {"python": "3.13.9", "torch": "2.8.0", "flash_attn": "3"}},
              "profile": {**F.SUPPORTED_PROFILE, "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                          "layer_names": ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]}}
    layer = {"block_size": 4, "pool_num_blocks": 32, "block_table": [[2, 4], [3]],
             "seq_lens": [5, 2], "query_start_loc": [0, 1, 3], "write_slots": [16, 12, 13],
             "causal": True, "window": None, "layout": "LBHNC", "backend": "FLASH_ATTN",
             "backend_version": "3"}
    snapshot = {"schema": 1, "kind": "snapshot", "session": session, "seq": 1,
                "forward_id": 1, "snapshot_id": "2" * 32, "pool_incarnation": "3" * 32,
                "requests": ["R-a", "R-b"], "num_tokens": 3, "snapshot_copy_complete": True,
                "forward_device_complete": None, "observation_point": "model_entry",
                "layers": [{**deepcopy(layer), "name": name} for name in header["profile"]["layer_names"]]}
    footer = {"schema": 1, "kind": "footer", "session": session, "seq": 2,
              "snapshots": 1, "dropped": 0, "errors": 0, "healthy": True}
    return [header, snapshot, footer]


def assert_unknown(result):
    assert result["observation_status"] == "unknown"
    assert result["scope_verdict"] == "inconclusive"
    assert result["physical_read_verdict"] == result["ownership_verdict"] == "unknown"
    assert result["evidence_depth"] == "approximate"
    assert result["reasons"]


def test_valid_session_still_does_not_prove_ownership_or_physical_reads():
    records = ordinary_records()
    report = F.audit_records(records)
    assert report["observation_status"] == "observed"
    assert report["snapshot_count"] == report["valid_snapshot_count"] == 1
    assert report["candidate_block_ranges"] == 6
    assert report["query_tokens"] == 3
    assert report["physical_read_verdict"] == report["ownership_verdict"] == "unknown"
    assert report["forward_device_completion"] == "unknown"
    assert report["scope_verdict"] == "inconclusive"
    assert report["evidence_depth"] == "approximate"
    assert report["snapshots"][0]["observation_point"] == "model_entry"
    assert "candidate_plan" not in report["snapshots"][0]


def test_history_is_in_candidate_plan_even_with_only_one_new_write():
    header, snapshot, _ = ordinary_records()
    checked = F.validate_snapshot(header, snapshot)
    row = checked["candidate_plan"][0]["requests"][0]
    assert row["write_slots"] == [16]
    assert row["candidate_ranges"] == [
        {"logical_start": 0, "logical_end": 4, "physical_block": 2, "slot_start": 0, "slot_end": 4},
        {"logical_start": 4, "logical_end": 5, "physical_block": 4, "slot_start": 0, "slot_end": 1},
    ]


def test_row_order_is_preserved_instead_of_sorting_request_ids():
    header, snapshot, _ = ordinary_records()
    snapshot["requests"] = ["Z-first", "A-second"]
    result = F.validate_snapshot(header, snapshot)
    rows = result["candidate_plan"][0]["requests"]
    assert [row["request"] for row in rows] == ["Z-first", "A-second"]
    assert rows[0]["context_tokens"] == 5 and rows[1]["context_tokens"] == 2


def test_validated_plan_does_not_alias_later_input_mutation():
    records = ordinary_records()
    plan = F.validate_snapshot(records[0], records[1])
    audit = F.audit_records(records)
    records[1]["layers"][0]["block_table"][0][0] = 20
    records[1]["layers"][0]["write_slots"][0] = 0
    records[0]["profile"]["layer_names"].clear()
    assert plan["candidate_plan"][0]["requests"][0]["candidate_ranges"][0]["physical_block"] == 2
    assert plan["candidate_plan"][0]["requests"][0]["write_slots"] == [16]
    assert len(audit["recorded_profile"]["layer_names"]) == 2


@pytest.mark.parametrize("change", [
    lambda r: r[1].update(snapshot_copy_complete=False),
    lambda r: r[1].update(forward_device_complete=True),
    lambda r: r[1].update(observation_point="kernel_read"),
    lambda r: r[1].update(requests=["R-a", "R-a"]),
    lambda r: r[1].update(session="4" * 32),
    lambda r: r[1].update(snapshot_id=""),
    lambda r: r[1].update(pool_incarnation=None),
    lambda r: r[1].update(num_tokens=True),
    lambda r: r[1].update(forward_id=0),
    lambda r: r[1]["layers"].pop(),
    lambda r: r[1]["layers"][1].update(name=r[1]["layers"][0]["name"]),
    lambda r: r[1]["layers"][0].update(query_start_loc=[0, 0, 3]),
    lambda r: r[1]["layers"][0].update(query_start_loc=[0, 1, 2]),
    lambda r: r[1]["layers"][0].update(seq_lens=[5, 1]),
    lambda r: r[1]["layers"][0].update(seq_lens=[5.0, 2]),
    lambda r: r[1]["layers"][0].update(write_slots=[-1, 12, 13]),
    lambda r: r[1]["layers"][0].update(write_slots=[16, 12, 13, -1]),
    lambda r: r[1]["layers"][0].update(block_table=[[2, 4, 0], [3]]),
    lambda r: r[1]["layers"][0].update(block_table=[[2], [3]]),
    lambda r: r[1]["layers"][0].update(block_table=[[2, 32], [3]]),
    lambda r: r[1]["layers"][0].update(block_table=[[2, 2], [3]]),
    lambda r: r[1]["layers"][0].update(causal=False),
    lambda r: r[1]["layers"][0].update(window=[128, 0]),
    lambda r: r[1]["layers"][0].update(backend_version="4"),
    lambda r: r[1]["layers"][0].update(layout="NHD"),
    lambda r: r[1]["layers"][0].update(unrecorded_mask=True),
])
def test_incomplete_or_inconsistent_metadata_is_unknown(change):
    records = ordinary_records()
    change(records)
    assert_unknown(F.audit_records(records))


@pytest.mark.parametrize("profile_change", [
    {"eager": False}, {"runner": "V1"}, {"prefix_caching": True},
    {"speculative": True}, {"tp": 2}, {"cache_groups": 2}, {"dp": 2},
    {"microbatching": True}, {"kv_sharing": True}, {"lora": True},
    {"attention": "sliding_window"}, {"model_revision": "main"},
    {"cache_groups": True}, {"layer_names": []}, {"layout": "NHD"},
])
def test_unsupported_effective_profile_never_becomes_observed(profile_change):
    records = ordinary_records()
    records[0]["profile"].update(profile_change)
    assert_unknown(F.audit_records(records))


def test_version_label_alone_does_not_establish_source_support():
    records = ordinary_records()
    name = next(iter(records[0]["engine"]["source_sha256"]))
    records[0]["engine"]["source_sha256"][name] = "0" * 64
    assert_unknown(F.audit_records(records))


def test_unknown_dependency_or_missing_header_field_is_not_defaulted():
    records = ordinary_records()
    del records[0]["engine"]["dependencies"]["torch"]
    assert_unknown(F.audit_records(records))
    records = ordinary_records()
    del records[0]["profile"]["prefix_caching"]
    assert_unknown(F.audit_records(records))


def test_block_zero_or_cross_request_sharing_does_not_imply_unowned():
    records = ordinary_records()
    for layer in records[1]["layers"]:
        layer["block_table"] = [[0, 4], [0]]
        layer["write_slots"] = [16, 0, 1]
    result = F.audit_records(records)
    assert result["observation_status"] == "observed"
    assert result["ownership_verdict"] == "unknown"


@pytest.mark.parametrize("change", [
    lambda r: r.pop(),
    lambda r: r[1].update(seq=2),
    lambda r: r[2].update(snapshots=0),
    lambda r: r[2].update(snapshots=True),
    lambda r: r[2].update(healthy=False),
    lambda r: r[2].update(dropped=1),
    lambda r: r[2].update(errors=1),
    lambda r: r[1].pop("kind"),
    lambda r: r[1].update(kind="future_kind"),
    lambda r: r[1].update(kind="footer"),
])
def test_session_loss_or_accounting_gap_prevents_complete_observation(change):
    records = ordinary_records()
    change(records)
    assert_unknown(F.audit_records(records))


def test_explicit_gap_survives_a_healthy_footer():
    records = ordinary_records()
    records.insert(2, {"schema": 1, "session": records[0]["session"], "seq": 2,
                       "kind": "gap", "forward_id": 2, "reason": "snapshot copy unavailable"})
    records[-1]["seq"] = 3
    result = F.audit_records(records)
    assert_unknown(result)
    assert result["valid_snapshot_count"] == 1 and result["gap_count"] == 1


def test_repeated_snapshot_or_reordered_forward_is_not_accepted_twice():
    records = ordinary_records()
    second = deepcopy(records[1]);second["seq"] = 2
    records.insert(2, second);records[-1].update(seq=3, snapshots=2)
    assert_unknown(F.audit_records(records))
    second["snapshot_id"] = "4" * 32
    assert_unknown(F.audit_records(records))
    second["forward_id"] = 2
    assert F.audit_records(records)["observation_status"] == "observed"


def test_each_layer_must_agree_on_dense_batch_row_lengths():
    records = ordinary_records()
    other = records[1]["layers"][1]
    other["seq_lens"] = [6, 2];other["write_slots"] = [17, 12, 13]
    assert_unknown(F.audit_records(records))


def test_one_group_cannot_report_different_cache_dimensions_or_history_maps():
    records = ordinary_records()
    records[1]["layers"][1]["pool_num_blocks"] = 64
    assert_unknown(F.audit_records(records))
    records = ordinary_records()
    records[1]["layers"][1]["block_table"][0][0] = 9
    assert_unknown(F.audit_records(records))


def test_cache_reinitialization_requires_a_new_pool_incarnation():
    records = ordinary_records()
    second = deepcopy(records[1]);second.update(seq=2, forward_id=2, snapshot_id="4" * 32)
    for layer in second["layers"]:
        layer["pool_num_blocks"] = 64
    records.insert(2, second);records[-1].update(seq=3, snapshots=2)
    assert_unknown(F.audit_records(records))
    second["pool_incarnation"] = "5" * 32
    assert F.audit_records(records)["observation_status"] == "observed"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2**100, "\ud800", object()])
def test_non_json_or_unbounded_primitives_fail_closed(value):
    records = ordinary_records();records[1]["foreign"] = value
    assert_unknown(F.audit_records(records))


def test_cycle_and_nonconcrete_input_are_rejected_without_execution():
    records = ordinary_records();records[1]["foreign"] = records[1]
    assert_unknown(F.audit_records(records))
    assert_unknown(F.audit_records(iter(ordinary_records())))


@pytest.mark.parametrize("limit", ["MAX_RECORD_BYTES", "MAX_RECORD_NODES", "MAX_TOTAL_BYTES",
                                  "MAX_TOTAL_NODES", "MAX_SNAPSHOT_CELLS", "MAX_TOTAL_CELLS"])
def test_resource_budget_exhaustion_cannot_produce_truncated_success(monkeypatch, limit):
    records = ordinary_records();monkeypatch.setattr(F, limit, 1)
    assert_unknown(F.audit_records(records))


def test_legacy_admission_and_transport_events_cannot_be_upgraded():
    legacy = [{"kind": "consume", "request": "R", "regions": [["D", 1]]},
              {"kind": "complete", "op": "transfer", "marks_ready": ["D", 1]}]
    result = F.audit_records(legacy)
    assert_unknown(result)
    assert result["valid_snapshot_count"] == 0


def test_inputs_and_pin_return_values_are_not_mutated_or_shared():
    records = ordinary_records();saved = deepcopy(records)
    F.audit_records(records)
    assert records == saved
    pins = F.source_pins();pins.clear()
    assert F.source_pins()
