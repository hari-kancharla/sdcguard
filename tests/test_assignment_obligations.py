"""CPU data examples for a mapping contract, never claimed as engine recordings."""
from copy import deepcopy
import json

import pytest

from sdcguard.lifetime import assignment_obligations as A


def token(number):
    return f"{number:032x}"


def recount(records):
    for index, record in enumerate(records):
        record["seq"] = index
    if records and records[-1]["kind"] == "footer":
        records[-1].update(records=len(records),
                           samples=sum(r["kind"] == "frame_end" for r in records),
                           acquisitions=sum(r["kind"] == "acquire" for r in records))
    return records


def one(records, kind):
    return next(record for record in records if record["kind"] == kind)


def ordinary_records():
    """An explicit causal chain, positive acquisitions and two ordinary rows."""
    contract = A.adapter_contract()
    coverage = sorted(set(contract["mutation_methods"]) | set(contract["guard_methods"]))
    roles = {role: token(index) for index, role in enumerate(contract["classes"], 20)}
    layers = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    dimensions = {"capacity": 32, "block_size": 4, "scheduler_block_size": 4,
                  "hash_block_size": 4, "groups": 1, "layout": "LBHNC", "layer_names": layers}
    records = []

    def add(kind, **facts):
        records.append({"schema": 1, "session": token(1), "process": token(2),
                        "seq": len(records), "kind": kind, **facts})

    def participants(*names):
        return {"objects": {name: roles[name] for name in names},
                "classes": {name: contract["classes"][name][0] for name in names}}

    add("header", pid=1234, adapter=A.DEFAULT_ADAPTER, source_sha256=A.source_pins(),
        profile={**contract["profile"], "layer_names": layers,
                 "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"},
        dependencies={"python": "3.13.9", "torch": "2.8.0", "flash_attn": "3"}, coverage=coverage)
    add("init_begin", init=token(3), **participants("core", "executor"))
    add("configs_derived", init=token(3), worker_config=token(4), scheduler_config=token(5), dimensions=deepcopy(dimensions))
    add("worker_init", init=token(3), executor=roles["executor"], selected_config=token(4), rank=0,
        **participants("wrapper", "worker", "runner"))
    add("runner_ready", init=token(3), runner=roles["runner"], input_config=token(4),
        runner_config=token(6), worker_pool=token(7), dimensions=deepcopy(dimensions), kernel_block_size=4, ratio=1)
    add("init_return", init=token(3), scheduler_config=token(5))
    add("scheduler_ready", init=token(3), passed_config=token(5), dimensions=deepcopy(dimensions), null_block=0,
        **participants("scheduler", "coordinator", "manager", "group_manager", "pool"))
    add("mutation_begin", association=token(3), mutation=token(8),
        method="BlockPool.get_new_blocks", receiver=roles["pool"], parent=None)
    add("acquire", association=token(3), mutation=token(8),
        blocks=[{"block": block, "incarnation": 1} for block in (2, 4, 3)])
    add("mutation_end", association=token(3), mutation=token(8), success=True)
    requests = [{"request": "R-a", "incarnation": token(9), "query_tokens": 1},
                {"request": "R-b", "incarnation": token(10), "query_tokens": 2}]
    add("schedule", association=token(3), core=roles["core"], core_scheduler=roles["scheduler"],
        scheduler=roles["scheduler"], dispatch=token(11), step=1, requests=deepcopy(requests))
    add("worker_entry", association=token(3), dispatch=token(11), runner=roles["runner"], requests=deepcopy(requests))
    add("frame_begin", association=token(3), dispatch=token(11), runner=roles["runner"], frame=token(12), epoch=2, coverage=coverage[:])
    add("assignments", frame=token(12), manager=roles["manager"], pool=roles["pool"],
        rows=[{"request": "R-a", "incarnation": token(9),
               "blocks": [{"block": 2, "incarnation": 1}, {"block": 4, "incarnation": 1}]},
              {"request": "R-b", "incarnation": token(10), "blocks": [{"block": 3, "incarnation": 1}]}])
    add("model_entry", frame=token(12), runner=roles["runner"], worker_pool=token(7),
        requests=["R-a", "R-b"], seq_lens=[5, 2], query_lengths=[1, 2],
        layers=[{"name": name, "rows": [[2, 4], [3]]} for name in layers],
        semantics={"observation_point": "model_entry", "causal": True, "window": None,
                   "layout": "LBHNC", "backend": "FLASH_ATTN"}, copy_complete=True)
    add("frame_end", frame=token(12), epoch=2, coverage=coverage[:])
    add("footer", records=0, samples=1, acquisitions=1, dropped=0, errors=0, healthy=True)
    return recount(records)


def mutation_before(records, before="frame_begin", method="FullAttentionManager.cache_blocks", unsupported=False):
    """A native-call observation only, with no live engine operation."""
    index = next(i for i, r in enumerate(records) if r["kind"] == before)
    owner = one(records, "scheduler_ready")["objects"]
    role = {**A.adapter_contract()["mutation_methods"], **A.adapter_contract()["guard_methods"]}[method]
    common = {key: records[0][key] for key in ("schema", "session", "process")}
    facts = {"association": token(3), "mutation": token(40)}
    events = [{**common, "kind": "mutation_begin", **facts, "method": method, "receiver": owner[role], "parent": None}]
    if unsupported:
        events.append({**common, "kind": "unsupported_branch", **facts, "method": method, "reason": "observed unsupported mutation"})
    events.append({**common, "kind": "mutation_end", **facts, "success": True})
    records[index:index] = events
    one(records, "frame_end")["epoch"] = 4
    if before == "frame_begin":
        one(records, "frame_begin")["epoch"] = 4
    return recount(records)


def assert_unknown(report):
    assert report["verdict"] == "unknown"
    assert report["scope_verdict"] == "inconclusive"
    assert report["ownership_verdict"] == report["physical_read_verdict"] == "unknown"
    assert report["content_provenance"] == "unknown"
    assert report["runtime_isolation"] == report["authenticity"] == "not established"


def test_complete_mapping_agreement_is_only_the_named_relation():
    records = ordinary_records()
    report = A.evaluate(records)
    assert report["property"] == "allocator_to_model_entry_mapping_consistency"
    assert report["verdict"] == "satisfied" and report["scope_verdict"] == "fine"
    assert report["summary"] == {"satisfied": 2, "violated": 0, "unknown": 0,
                                 "compared_layer_positions": 6, "contradictions": 0}
    assert report["allocation_incarnations"] == 3 and report["complete_frames"] == 1
    assert report["physical_read_verdict"] == report["ownership_verdict"] == "unknown"
    assert report["content_provenance"] == "unknown"
    assert report["authenticity"] == report["runtime_isolation"] == "not established"


@pytest.mark.parametrize("actual", [0, 3, 31])
def test_positive_assignment_contradiction_retains_null_and_unacquired_actual_slots(actual):
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = actual
    report = A.evaluate(records)
    assert report["verdict"] == "violated" and report["scope_verdict"] == "not fine"
    assert report["summary"]["contradictions"] == 1
    witness = report["findings"][0]["contradictions"][0]
    assert witness["allocator_block"] == 2 and witness["model_entry_block"] == actual
    assert report["ownership_verdict"] == report["physical_read_verdict"] == "unknown"


@pytest.mark.parametrize("blocks", [None, [], [{"block": 2, "incarnation": 1}]])
def test_absent_or_short_assignment_is_unknown_never_unowned(blocks):
    records = ordinary_records()
    one(records, "assignments")["rows"][0]["blocks"] = blocks
    report = A.evaluate(records)
    assert_unknown(report)
    assert report["summary"]["contradictions"] == 0
    assert "unowned" not in json.dumps(report)


def test_known_mismatch_survives_an_unrelated_missing_assignment():
    records = ordinary_records()
    one(records, "assignments")["rows"].pop()
    one(records, "model_entry")["layers"][1]["rows"][0][1] = 31
    report = A.evaluate(records)
    assert report["verdict"] == "violated"
    assert report["summary"]["unknown"] == report["summary"]["violated"] == 1


def test_unknown_expected_acquisition_does_not_become_a_positive_contradiction():
    records = ordinary_records()
    one(records, "acquire")["blocks"] = [{"block": 3, "incarnation": 1}]
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 31
    report = A.evaluate(records)
    assert_unknown(report)
    assert report["summary"]["contradictions"] == 0


def test_map_agreement_cannot_detect_a_shared_erroneous_assignment():
    records = ordinary_records()
    one(records, "assignments")["rows"][0]["blocks"][0] = {"block": 3, "incarnation": 1}
    for layer in one(records, "model_entry")["layers"]:
        layer["rows"][0][0] = 3
    report = A.evaluate(records)
    assert report["verdict"] == "satisfied"
    assert report["ownership_verdict"] == "unknown"


@pytest.mark.parametrize("kind,field,value", [
    ("worker_init", "executor", token(60)), ("worker_init", "selected_config", token(60)),
    ("runner_ready", "input_config", token(60)), ("runner_ready", "runner", token(60)),
    ("init_return", "scheduler_config", token(60)), ("scheduler_ready", "passed_config", token(60)),
    ("schedule", "core", token(60)), ("schedule", "core_scheduler", token(60)),
    ("schedule", "scheduler", token(60)), ("worker_entry", "dispatch", token(60)),
    ("worker_entry", "runner", token(60)), ("model_entry", "worker_pool", token(60)),
    ("assignments", "pool", token(60)), ("assignments", "manager", token(60)),
])
def test_matching_values_cannot_replace_an_actual_native_object_edge(kind, field, value):
    records = ordinary_records()
    one(records, kind)[field] = value
    assert_unknown(A.evaluate(records))


@pytest.mark.parametrize("kind", ["init_begin", "configs_derived", "worker_init", "runner_ready", "init_return", "scheduler_ready"])
def test_each_initialization_edge_is_required(kind):
    records = [r for r in ordinary_records() if r["kind"] != kind]
    assert_unknown(A.evaluate(recount(records)))


@pytest.mark.parametrize("change", [
    lambda r: one(r, "worker_init").update(rank=True),
    lambda r: one(r, "configs_derived").update(scheduler_config=token(4)),
    lambda r: one(r, "runner_ready").update(runner_config=token(4)),
    lambda r: one(r, "init_begin")["classes"].update(executor="vllm.v1.executor.uniproc_executor.ExecutorWithExternalLauncher"),
    lambda r: one(r, "scheduler_ready")["classes"].update(group_manager="vllm.v1.core.single_type_kv_cache_manager.SlidingWindowManager"),
    lambda r: one(r, "runner_ready")["dimensions"].update(capacity=64),
    lambda r: one(r, "configs_derived")["dimensions"].update(ratio=2),
    lambda r: one(r, "configs_derived")["dimensions"].update(groups=True),
    lambda r: one(r, "configs_derived")["dimensions"].update(hash_block_size=8),
    lambda r: one(r, "scheduler_ready").update(null_block=32),
    lambda r: r[0]["profile"].update(prefix_caching=True),
    lambda r: r[0]["profile"].update(model_revision="main"),
    lambda r: r[0]["profile"].update(cache_groups=True),
    lambda r: r[0]["dependencies"].update(flash_attn="4"),
    lambda r: r[0]["source_sha256"].pop(next(iter(r[0]["source_sha256"]))),
    lambda r: r[0]["coverage"].pop(),
    lambda r: one(r, "frame_end")["coverage"].pop(),
    lambda r: one(r, "frame_begin").update(epoch=99),
    lambda r: one(r, "frame_end").update(epoch=True),
    lambda r: one(r, "acquire")["blocks"][0].update(incarnation=2),
    lambda r: one(r, "acquire")["blocks"][0].update(block=0),
    lambda r: one(r, "model_entry").update(copy_complete=False),
    lambda r: one(r, "model_entry")["layers"][0]["rows"][0].append(5),
    lambda r: one(r, "model_entry")["layers"][0]["rows"][0].__setitem__(0, 32),
    lambda r: one(r, "model_entry")["semantics"].update(causal=False),
    lambda r: one(r, "model_entry")["semantics"].update(window=[4, 0]),
    lambda r: one(r, "model_entry").update(requests=[["R-a"], "R-b"]),
    lambda r: one(r, "model_entry").update(query_lengths=[2, 2]),
    lambda r: one(r, "worker_entry")["requests"][0].update(incarnation=token(70)),
    lambda r: one(r, "assignments")["rows"][0].update(incarnation=token(70)),
])
def test_unsupported_or_ambiguous_evidence_is_unknown(change):
    records = ordinary_records()
    change(records)
    assert_unknown(A.evaluate(records))


def test_method_entry_that_takes_no_unsupported_branch_keeps_normal_coverage():
    records = mutation_before(ordinary_records())
    assert A.evaluate(records)["verdict"] == "satisfied"
    records = mutation_before(ordinary_records(), method="FullAttentionManager._cache_partial_tail_block")
    assert A.evaluate(records)["verdict"] == "satisfied"


def test_only_an_actual_linked_unsupported_branch_retires_supported_evidence():
    records = mutation_before(ordinary_records(), method="FullAttentionManager._cache_partial_tail_block", unsupported=True)
    assert_unknown(A.evaluate(records))
    one(records, "unsupported_branch")["mutation"] = token(99)
    report = A.evaluate(records)
    assert_unknown(report)
    assert "active native call" in report["missing"][0]


def test_completed_mutation_during_capture_makes_that_frame_unknown():
    records = mutation_before(ordinary_records(), before="model_entry")
    assert_unknown(A.evaluate(records))
    assert "overlapped" in A.evaluate(records)["findings"][0]["missing"][0]


@pytest.mark.parametrize("remove", ["acquire", "mutation_end", "worker_entry", "frame_begin", "assignments", "model_entry", "frame_end"])
def test_missing_observation_does_not_get_defaulted(remove):
    records = [r for r in ordinary_records() if r["kind"] != remove]
    assert_unknown(A.evaluate(recount(records)))


def test_omitting_an_entire_positive_frame_invalidates_the_session():
    records = ordinary_records()
    copied = deepcopy(one(records, "schedule"))
    copied.update(dispatch=token(90), step=2)
    records.insert(-1, copied)
    assert_unknown(A.evaluate(recount(records)))


def test_empty_scheduler_steps_need_no_fabricated_forward():
    records = ordinary_records()
    copied = deepcopy(one(records, "schedule"))
    copied.update(dispatch=token(90), step=2, requests=[])
    records.insert(-1, copied)
    assert A.evaluate(recount(records))["verdict"] == "satisfied"


def test_native_schedule_counter_and_output_lifetimes_cannot_be_reused():
    records = ordinary_records()
    records.insert(-1, deepcopy(one(records, "schedule")))
    assert_unknown(A.evaluate(recount(records)))
    one(records[-2:], "schedule")["dispatch"] = token(91)
    assert_unknown(A.evaluate(recount(records)))


def test_nested_native_allocation_has_an_ordered_call_chain():
    records = ordinary_records()
    outer = deepcopy(one(records, "mutation_begin"))
    outer.update(mutation=token(41), method="SingleTypeKVCacheManager.allocate_new_blocks",
                 receiver=one(records, "scheduler_ready")["objects"]["group_manager"])
    one(records, "mutation_begin")["parent"] = token(41)
    end = deepcopy(one(records, "mutation_end"))
    end["mutation"] = token(41)
    begin_index = next(i for i, r in enumerate(records) if r["kind"] == "mutation_begin")
    records.insert(begin_index, outer)
    end_index = next(i for i, r in enumerate(records) if r["kind"] == "mutation_end")
    records.insert(end_index + 1, end)
    one(records, "frame_begin")["epoch"] = one(records, "frame_end")["epoch"] = 4
    assert A.evaluate(recount(records))["verdict"] == "satisfied"
    records[end_index], records[end_index + 1] = records[end_index + 1], records[end_index]
    assert_unknown(A.evaluate(recount(records)))


def test_failed_native_call_and_open_capture_cannot_support_a_verdict():
    records = ordinary_records()
    one(records, "mutation_end")["success"] = False
    assert_unknown(A.evaluate(records))
    records = ordinary_records()
    end = one(records, "mutation_end")
    records.remove(end)
    records.insert(-1, end)
    one(records, "frame_begin")["epoch"] = one(records, "frame_end")["epoch"] = 1
    assert_unknown(A.evaluate(recount(records)))


def test_retired_cache_is_not_made_current_by_equal_numeric_blocks():
    records = ordinary_records()
    marker = {**{key: records[0][key] for key in ("schema", "session", "process")},
              "kind": "retire", "association": token(3), "reason": "cache_replacement"}
    records.insert(10, marker)
    assert_unknown(A.evaluate(recount(records)))


def two_initializations():
    first = ordinary_records()
    second = ordinary_records()[1:-1]
    retained = set(one(first, "init_begin")["objects"].values())
    retained.update(one(first, "worker_init")["objects"].values())

    def rename(value):
        if type(value) is str and len(value) == 32 and all(c in "0123456789abcdef" for c in value):
            return value if value in retained or value in (token(1), token(2)) else token(int(value, 16) + 1000)
        if type(value) is list:
            return [rename(child) for child in value]
        if type(value) is dict:
            return {key: rename(child) for key, child in value.items()}
        return value

    return recount(first[:-1] + rename(second) + first[-1:])


def test_reinitialization_requires_fresh_cache_and_allocator_lifetimes():
    records = two_initializations()
    assert A.evaluate(records)["verdict"] == "satisfied"
    runners = [r for r in records if r["kind"] == "runner_ready"]
    runners[1]["worker_pool"] = runners[0]["worker_pool"]
    assert_unknown(A.evaluate(records))
    records = two_initializations()
    schedulers = [r for r in records if r["kind"] == "scheduler_ready"]
    schedulers[1]["objects"]["pool"] = schedulers[0]["objects"]["pool"]
    assert_unknown(A.evaluate(records))


def test_stale_core_scheduler_attachment_is_rejected_after_reinitialization():
    records = two_initializations()
    schedules = [r for r in records if r["kind"] == "schedule"]
    schedules[1]["association"] = schedules[0]["association"]
    schedules[1]["scheduler"] = schedules[1]["core_scheduler"] = schedules[0]["scheduler"]
    schedules[1]["step"] = 2
    assert_unknown(A.evaluate(records))


@pytest.mark.parametrize("shared", ["worker", "runner", "both"])
def test_other_core_reinitialization_retires_every_shared_worker_or_runner_cache(shared):
    first = ordinary_records()
    combined = two_initializations()
    second_chain = deepcopy(combined[len(first) - 1:len(first) + 5])
    second_chain[0]["objects"]["core"] = token(2000)
    if shared == "worker":
        second_chain[2]["objects"]["runner"] = second_chain[3]["runner"] = token(2001)
    elif shared == "runner":
        second_chain[2]["objects"]["worker"] = token(2002)
    # The old core still points to its scheduler, but its worker cache was replaced.
    records = recount(first[:10] + second_chain + first[10:])
    report = A.evaluate(records)
    assert_unknown(report)
    assert report["summary"]["contradictions"] == 0
    assert report["summary"]["unknown"] == 1
    assert "retired" in report["findings"][0]["missing"][0]


def replacement_body(shared=()):
    first = ordinary_records()
    body = deepcopy(two_initializations()[len(first) - 1:-1])
    objects = {**one(first, "init_begin")["objects"], **one(first, "worker_init")["objects"]}
    renames = {objects[role]: token(2000 + index) for index, role in enumerate(objects) if role not in shared}

    def rename(value):
        if type(value) is str:
            return renames.get(value, value)
        if type(value) is list:
            return [rename(child) for child in value]
        if type(value) is dict:
            return {key: rename(child) for key, child in value.items()}
        return value

    return rename(body)


@pytest.mark.parametrize("shared", [("runner",), ("worker",), ("worker", "runner")])
def test_earlier_initializer_cannot_resume_after_a_conflicting_cache_initialization(shared):
    first = ordinary_records()
    initial, later = first[1:7], replacement_body(shared)
    records = first[:1] + initial[:3] + later[:6] + initial[3:] + later[6:] + first[-1:]
    report = A.evaluate(recount(records))
    assert_unknown(report)
    assert "overlapping initializations share" in report["missing"][0]
    assert report["summary"]["satisfied"] == report["summary"]["contradictions"] == 0


def test_overlapping_initializations_of_one_core_are_unknown_even_with_distinct_workers():
    first = ordinary_records()
    later = replacement_body()
    later[0]["objects"]["core"] = one(first, "init_begin")["objects"]["core"]
    records = first[:2] + later + first[2:]
    report = A.evaluate(recount(records))
    assert_unknown(report)
    assert "overlapping initializations share a native core" in report["missing"][0]


def test_interleaved_initializers_on_independent_native_participants_remain_valid():
    first = ordinary_records()
    initial, later = first[1:7], replacement_body()
    records = first[:1] + initial[:3] + later[:6] + initial[3:] + first[7:-1] + later[6:] + first[-1:]
    report = A.evaluate(recount(records))
    assert report["verdict"] == "satisfied"
    assert report["complete_frames"] == 2 and report["summary"]["satisfied"] == 4


@pytest.mark.parametrize("shared", [("runner",), ("worker",), ("worker", "runner")])
def test_sequential_cache_replacement_preserves_completed_historical_intervals(shared):
    first = ordinary_records()
    report = A.evaluate(recount(first[:-1] + replacement_body(shared) + first[-1:]))
    assert report["verdict"] == "satisfied"
    assert report["complete_frames"] == 2 and report["summary"]["satisfied"] == 4


def test_native_kernel_facts_are_later_than_the_final_config_dimensions():
    records = ordinary_records()
    keys = {"capacity", "block_size", "scheduler_block_size", "hash_block_size", "groups", "layout", "layer_names"}
    assert all(set(one(records, kind)["dimensions"]) == keys
               for kind in ("configs_derived", "runner_ready", "scheduler_ready"))
    assert one(records, "runner_ready")["kernel_block_size"] == 4
    assert one(records, "runner_ready")["ratio"] == 1
    assert A.evaluate(records)["verdict"] == "satisfied"


@pytest.mark.parametrize("kind", ["configs_derived", "runner_ready", "scheduler_ready"])
@pytest.mark.parametrize("field,value", [("kernel_block_size", 4), ("ratio", 1)])
def test_selected_kernel_facts_cannot_be_inserted_into_config_only_dimensions(kind, field, value):
    records = ordinary_records()
    one(records, kind)["dimensions"][field] = value
    assert_unknown(A.evaluate(records))


@pytest.mark.parametrize("kernel,ratio", [(None, 1), (False, 1), (0, 1), (4.0, 1), (-1, 1),
                                        (8, 1), (3, 1), (2, 1), (2, 2), (4, True), (4, 0), (4, 2)])
def test_actual_selected_kernel_and_exact_ratio_are_required(kernel, ratio):
    records = ordinary_records()
    one(records, "runner_ready").update(kernel_block_size=kernel, ratio=ratio)
    assert_unknown(A.evaluate(records))


@pytest.mark.parametrize("field", ["kernel_block_size", "ratio"])
def test_missing_runner_fact_is_not_inferred_from_a_requested_block_size(field):
    records = ordinary_records()
    one(records, "runner_ready").pop(field)
    assert_unknown(A.evaluate(records))


def second_dispatch_and_frame(records):
    start = next(i for i, r in enumerate(records) if r["kind"] == "schedule")
    copied = deepcopy(records[start:-1])
    for record in copied:
        if "dispatch" in record:
            record["dispatch"] = token(80)
        if "frame" in record:
            record["frame"] = token(81)
        if record["kind"] == "schedule":
            record["step"] = 2
    return copied


def test_earlier_frame_cannot_claim_a_superseded_direct_worker_invocation():
    records = ordinary_records()
    second = second_dispatch_and_frame(records)
    first_frame = next(i for i, r in enumerate(records) if r["kind"] == "frame_begin")
    records[first_frame:first_frame] = second[:2]
    records[-1:-1] = second[2:]
    report = A.evaluate(recount(records))
    assert_unknown(report)
    assert "current pending direct dispatch" in report["missing"][0]


def test_worker_invocation_change_during_capture_invalidates_only_that_interval():
    records = ordinary_records()
    second = second_dispatch_and_frame(records)
    before_entry = next(i for i, r in enumerate(records) if r["kind"] == "model_entry")
    records[before_entry:before_entry] = second[:2]
    records[-1:-1] = second[2:]
    report = A.evaluate(recount(records))
    assert_unknown(report)
    assert report["summary"]["unknown"] == 1 and report["summary"]["satisfied"] == 2
    assert "another direct dispatch" in report["findings"][0]["missing"][0]


def test_sequential_direct_dispatches_keep_distinct_complete_frames():
    records = ordinary_records()
    records[-1:-1] = second_dispatch_and_frame(records)
    report = A.evaluate(recount(records))
    assert report["verdict"] == "satisfied"
    assert report["complete_frames"] == 2 and report["summary"]["satisfied"] == 4


def native_identity(records, role):
    objects = {**one(records, "init_begin")["objects"],
               **one(records, "worker_init")["objects"],
               **one(records, "scheduler_ready")["objects"]}
    objects.update({key: one(records, "runner_ready")[key] for key in ("runner_config", "worker_pool")})
    objects.update({key: one(records, "configs_derived")[key] for key in ("worker_config", "scheduler_config")})
    return objects[role]


@pytest.mark.parametrize("role", ["core", "executor", "wrapper", "worker", "runner", "scheduler", "coordinator",
                                 "manager", "group_manager", "pool", "worker_pool", "worker_config",
                                 "scheduler_config", "runner_config"])
@pytest.mark.parametrize("alias_kind", ["schedule_output", "request"])
def test_native_request_and_schedule_objects_cannot_alias_another_object_role(role, alias_kind):
    records = ordinary_records()
    alias = native_identity(records, role)
    for record in records:
        if alias_kind == "schedule_output" and "dispatch" in record:
            record["dispatch"] = alias
        if alias_kind == "request":
            for row in record.get("requests", []) + record.get("rows", []):
                if type(row) is dict and row.get("request") == "R-a":
                    row["incarnation"] = alias
    report = A.evaluate(records)
    assert_unknown(report)
    assert "native role" in report["missing"][0]


def test_request_and_schedule_output_are_distinct_native_roles():
    records = ordinary_records()
    for record in records:
        for row in record.get("requests", []) + record.get("rows", []):
            if type(row) is dict and row.get("request") == "R-a":
                row["incarnation"] = one(records, "schedule")["dispatch"]
    assert_unknown(A.evaluate(records))


def test_one_native_request_object_cannot_claim_two_request_names():
    records = ordinary_records()
    for record in records:
        for row in record.get("requests", []) + record.get("rows", []):
            if type(row) is dict and row.get("request") == "R-b":
                row["incarnation"] = token(9)
    report = A.evaluate(records)
    assert_unknown(report)
    assert "changes its request identity" in report["missing"][0]


@pytest.mark.parametrize("null_block", [1, True, None])
def test_null_slot_is_bound_to_the_actual_pinned_constructor(null_block):
    records = ordinary_records()
    one(records, "scheduler_ready")["null_block"] = null_block
    for record in records:
        if record["kind"] == "acquire":
            record["blocks"][0]["block"] = 0
        if record["kind"] == "assignments":
            record["rows"][0]["blocks"][0]["block"] = 0
        if record["kind"] == "model_entry":
            for layer in record["layers"]:
                layer["rows"][0][0] = 0
    report = A.evaluate(records)
    assert_unknown(report)
    assert "pinned source constructor" in report["missing"][0]
    assert A.adapter_contract()["null_block"] == 0


def test_overlapping_observation_intervals_are_unknown():
    records = ordinary_records()
    start = next(i for i, r in enumerate(records) if r["kind"] == "schedule")
    copied = deepcopy(records[start:-1])
    for r in copied:
        if "dispatch" in r:
            r["dispatch"] = token(80)
        if "frame" in r:
            r["frame"] = token(81)
        if r["kind"] == "schedule":
            r["step"] = 2
    frame_index = next(i for i, r in enumerate(records) if r["kind"] == "frame_begin")
    records[frame_index + 1:frame_index + 1] = copied
    report = A.evaluate(recount(records))
    assert_unknown(report)
    assert report["summary"]["unknown"] == 2
    assert all("model-entry observation intervals overlap" in f["missing"] for f in report["findings"])


def test_request_row_permutation_is_not_a_mapping_change():
    records = ordinary_records()
    metadata = one(records, "model_entry")
    for field in ("requests", "seq_lens", "query_lengths"):
        metadata[field].reverse()
    for layer in metadata["layers"]:
        layer["rows"].reverse()
    one(records, "assignments")["rows"].reverse()
    one(records, "schedule")["requests"].reverse()
    one(records, "worker_entry")["requests"].reverse()
    result = A.evaluate(records)
    assert result["verdict"] == "satisfied" and result["summary"]["compared_layer_positions"] == 6


def test_consistent_opaque_identifier_renaming_preserves_the_rederived_verdict():
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 31

    def rename(value):
        if type(value) is str and len(value) == 32 and all(c in "0123456789abcdef" for c in value):
            return token(int(value, 16) + 1000)
        if type(value) is list:
            return [rename(x) for x in value]
        if type(value) is dict:
            return {key: rename(child) for key, child in value.items()}
        return value

    before, after = A.evaluate(records), A.evaluate(rename(records))
    assert before["verdict"] == after["verdict"] == "violated"
    assert before["summary"] == after["summary"]


@pytest.mark.parametrize("change", [
    lambda r: r[1].update(process=token(99)), lambda r: r[1].update(seq=99),
    lambda r: r[1].update(schema=True), lambda r: r[-1].update(healthy=False),
    lambda r: r[-1].update(records=999), lambda r: r[-1].update(samples=True),
    lambda r: r[-1].update(dropped=1), lambda r: r[-1].update(errors=1),
    lambda r: r[-1].update(process=token(99)), lambda r: r[0].update(pid=True),
])
def test_unhealthy_envelope_invalidates_even_a_retained_contradiction(change):
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0][0] = 31
    change(records)
    report = A.evaluate(records)
    assert_unknown(report)
    assert report["findings"] == []


@pytest.mark.parametrize("value", [None, (), iter(()), float("nan"), b"data", {"records": []}])
def test_nonconcrete_or_nonlist_sessions_are_unknown(value):
    assert_unknown(A.evaluate(value))


def test_legacy_saved_result_cannot_be_promoted_to_assignment_evidence():
    records = [{"event": "lease_mismatch", "request_id": "R-a"}, {"event": "recovered"}]
    assert_unknown(A.evaluate(records))


def test_inputs_and_installed_contract_results_are_not_mutated_or_shared():
    records = ordinary_records()
    saved = deepcopy(records)
    A.evaluate(records)
    assert records == saved
    contract = A.adapter_contract()
    contract["profile"].clear()
    pins = A.source_pins()
    pins.clear()
    assert A.adapter_contract()["profile"] and A.source_pins()


def test_all_input_positions_are_judged_when_diagnostic_details_are_bounded(monkeypatch):
    records = ordinary_records()
    one(records, "model_entry")["layers"][0]["rows"][0] = [31, 31]
    monkeypatch.setattr(A, "MAX_WITNESSES", 1)
    monkeypatch.setattr(A, "MAX_FINDINGS", 1)
    result = A.evaluate(records)
    assert result["verdict"] == "violated"
    assert result["summary"]["compared_layer_positions"] == 6
    assert result["summary"]["contradictions"] == 2
    assert result["summary"]["satisfied"] == 1
    assert result["findings_omitted"] == result["witnesses_omitted"] == 1
    assert result["findings_complete"] is result["witnesses_complete"] is False


@pytest.mark.parametrize("limit", ["MAX_RECORDS", "MAX_RECORD_NODES", "MAX_TOTAL_NODES", "MAX_RECORD_BYTES", "MAX_TOTAL_BYTES", "MAX_CELLS"])
def test_aggregate_bound_exhaustion_cannot_return_partial_success(monkeypatch, limit):
    records = ordinary_records()
    monkeypatch.setattr(A, limit, 1)
    result = A.evaluate(records)
    assert_unknown(result)
    assert result["findings"] == []


def test_cyclic_and_invalid_unicode_records_fail_closed():
    records = ordinary_records()
    records[0]["cycle"] = records[0]
    assert_unknown(A.evaluate(records))
    records = ordinary_records()
    records[0]["extra"] = "\ud800"
    assert_unknown(A.evaluate(records))
