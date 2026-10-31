"""Ordinary offline receipt fixtures and missing-evidence controls.

Owner: Hari Krishna Kancharla.
These are inert metadata examples, not native recordings or engine workloads.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from sdcguard import evidence_io as io
from sdcguard.lifetime import allocation_recipient_obligations as R
from sdcguard.lifetime import assignment_obligations as A
from sdcguard.lifetime import mapping_closure as C
from test_assignment_obligations import ordinary_records, token


def paired(records):
    """Construct actual raw pairs for the real settled and semantic readers."""
    for index, record in enumerate(records):
        record["seq"] = index
    footer = records[-1]
    footer.update(records=len(records), samples=sum(r["kind"] == "frame_end" for r in records),
                  acquisitions=sum(r["kind"] == "acquire" for r in records))
    raw = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False).encode("utf-8") + b"\n" for r in records)
    header = records[0]
    session, process = header["session"], header["process"]
    _, _, counts = C._header_footer(raw)
    receipt = {"format": C.FORMAT, "kind": "recorder_closure_bookkeeping", "session": session,
        "process": process, "identity": {key: deepcopy(header[key]) for key in
            ("adapter", "source_sha256", "profile", "dependencies", "process")},
        "close_succeeded": True,
        "observer": {"healthy": True, "closed": True, "sealed": True,
                     "events": len(records) - 2, "positive_dispatches": counts["positive_dispatches"]},
        "journal": {"healthy": True, "closed": True, "sealed": True, "footer_written": True,
                    "published": True, "session": session, "process": process, "process_changed": False,
                    "errors": 0, "dropped": 0, "records": len(records), "accepted_records": len(records) - 1,
                    "samples": counts["samples"], "acquisitions": counts["acquisitions"],
                    "late_calls": 0, "queued_bytes": 0, "total_bytes": len(raw), "total_nodes": counts["total_nodes"]},
        "file": {"name": session + ".jsonl", "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
        "authenticity": "not established"}
    return {session + ".jsonl": raw}, {session + ".closure.json": io.json_bytes(receipt)}


def binding(payload, receipts):
    def row(values):
        name, raw = next(iter(values.items()))
        return {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return {"journal": row(payload), "receipt": row(receipts)}


def recount(records):
    for index, record in enumerate(records):
        record["seq"] = index
    if records and records[-1]["kind"] == "footer":
        footer = records[-1]
        pools = [r for r in records if r["kind"] == "pool_return"]
        footer.update(records=len(records), outer_calls=sum(r["kind"] == "allocation_begin" for r in records),
                      phase_calls=sum(r["kind"] == "phase_begin" for r in records), pool_returns=len(pools),
                      block_facts=sum(len(r["blocks"]) for r in pools),
                      empty_pool_returns=sum(not r["blocks"] for r in pools),
                      completed_grants=sum(bool(r["blocks"]) for r in pools), **R.measure(records[:-1]))
    return records


def example(mode="ordinary"):
    native = A.adapter_contract("vllm.0.32.uniproc.eager.native.v1")
    m = ordinary_records()[:7]
    m[0].update(adapter="vllm.0.32.uniproc.eager.native.v1",
                source_sha256=A.source_pins("vllm.0.32.uniproc.eager.native.v1"),
                profile={**native["profile"], "layer_names": m[0]["profile"]["layer_names"],
                         "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"})
    graph = m[-1]["objects"]
    common = {key: m[0][key] for key in ("schema", "session", "process")}
    records = []

    def madd(kind, **values):
        m.append({**common, "seq": len(m), "kind": kind, **values})

    def add(kind, **values):
        records.append({"schema": 1, "session": token(100), "process": common["process"],
                        "seq": len(records), "kind": kind, **values})

    methods = R.adapter_contract()["methods"]
    add("header", adapter=R.DEFAULT_ADAPTER, source_sha256=R.source_pins(), profile=deepcopy(m[0]["profile"]),
        dependencies=deepcopy(m[0]["dependencies"]), mapping_session=m[0]["session"], association=token(3),
        objects=deepcopy(graph), policy=R.adapter_contract()["policy"], coverage=sorted(methods),
        callables={name: token(200 + index) for index, name in enumerate(sorted(methods))})
    external = mode == "external"
    computed = external or mode == "computed_empty"
    outer = token(101)
    arguments = {"num_new_tokens": 0 if external else 4, "num_new_computed_tokens": 0,
                 "new_computed_blocks": [[]] if mode == "computed_empty" else None,
                 "num_lookahead_tokens": 0, "num_external_computed_tokens": 4 if external else 0,
                 "delay_cache_blocks": external, "num_encoder_tokens": 0,
                 "full_sequence_must_fit": False, "reserved_blocks": 0, "has_scheduled_reqs": True,
                 "skip_zeroing_group_ids": [0] if external else [],
                 "computed_blocks_native_empty": mode != "computed_empty", "retained_hit_group_ids": []}
    request_policy = {"n": 1, "resumable": False, "streaming": False, "preemptions": 0,
                      "replay_start": 0, "shared_prefix_boundary": 0, "in_flight_tokens": 0,
                      "pooling_params_absent": True, "lora_request_absent": True}
    if mode != "unexercised":
        add("allocation_begin", call=outer, request="ordinary-request", request_incarnation=token(9),
            request_policy=request_policy,
            request_state={"num_computed_tokens": 0, "num_tokens": 16, "num_prompt_tokens": 16, "max_model_len": 32},
            arguments=arguments)
    next_call = 300

    def phase(method, parent, args, returned, pool_blocks=None):
        nonlocal next_call
        call = token(next_call)
        next_call += 1
        mutation = token(next_call)
        next_call += 1
        madd("mutation_begin", association=token(3), mutation=mutation, method=method,
             receiver=graph["group_manager"], parent=None)
        add("phase_begin", call=call, outer=outer, parent=parent, method=method,
            receiver=graph["group_manager"], request="ordinary-request", arguments=args, mapping_mutation=mutation)
        if pool_blocks is not None:
            pool_mutation = token(next_call)
            next_call += 1
            madd("mutation_begin", association=token(3), mutation=pool_mutation,
                 method="BlockPool.get_new_blocks", receiver=graph["pool"], parent=mutation)
            madd("acquire", association=token(3), mutation=pool_mutation, blocks=deepcopy(pool_blocks))
            madd("mutation_end", association=token(3), mutation=pool_mutation, success=True)
            add("pool_return", outer=outer, phase=call, mutation=pool_mutation, pool=graph["pool"],
                requested_blocks=len(pool_blocks), blocks=deepcopy(pool_blocks))
        madd("mutation_end", association=token(3), mutation=mutation, success=True)
        add("phase_end", call=call, outer=outer, outcome="returned", return_type="list" if method == R.NEW else "NoneType",
            returned=deepcopy(returned), exception_type=None)

    blocks = [{"block": 2, "incarnation": 1}]
    if computed:
        call = token(110)
        add("phase_begin", call=call, outer=outer, parent=outer, method=R.COMPUTED,
            receiver=graph["coordinator"], request="ordinary-request", mapping_mutation=None,
            arguments={"new_computed_blocks": [[]], "num_local_computed_tokens": 0,
                       "num_external_computed_tokens": arguments["num_external_computed_tokens"],
                       "skip_zeroing_group_ids": arguments["skip_zeroing_group_ids"], "already_tracked": False})
        lower_args = {"num_local_computed_tokens": 0, "num_external_computed_tokens": arguments["num_external_computed_tokens"]}
        phase(R.LOCAL, call, {**lower_args, "new_computed_blocks": []}, None)
        if external:
            phase(R.EXTERNAL, call, {**lower_args, "record_for_zeroing": False}, None, blocks)
        add("phase_end", call=call, outer=outer, outcome="returned", return_type="NoneType", returned=None, exception_type=None)
    if mode not in ("not_allocated", "unexercised"):
        call = token(111)
        add("phase_begin", call=call, outer=outer, parent=outer, method=R.ORDINARY,
            receiver=graph["coordinator"], request="ordinary-request", mapping_mutation=None,
            arguments={"num_tokens": 4, "num_tokens_main_model": 4, "num_encoder_tokens": 0})
        ordinary_blocks = [] if external else blocks
        phase(R.NEW, call, {"num_tokens": 4, "num_tokens_main_model": 4}, ordinary_blocks,
              None if external else ordinary_blocks)
        add("phase_end", call=call, outer=outer, outcome="returned", return_type="tuple",
            returned=[deepcopy(ordinary_blocks)], exception_type=None)
        add("allocation_end", call=outer, outcome="returned", return_type="KVCacheBlocks",
            returned=[deepcopy(ordinary_blocks)], exception_type=None, request_revalidated=True, graph_revalidated=True)
    elif mode == "not_allocated":
        add("allocation_end", call=outer, outcome="not_allocated", return_type="NoneType", returned=None,
            exception_type=None, request_revalidated=True, graph_revalidated=True)
    madd("footer", records=0, samples=0, acquisitions=0, errors=0, dropped=0, healthy=True)
    payload, receipts = paired(m)
    add("footer", records=0, outer_calls=0, phase_calls=0, pool_returns=0, block_facts=0,
        empty_pool_returns=0, completed_grants=0, errors=0, dropped=0, healthy=True,
        closed=True, open_calls=0, source_stable=True, coverage_stable=True, body_bytes=0, body_nodes=0,
        mapping_binding=binding(payload, receipts))
    return recount(records), payload, receipts, m


def find(records, kind, method=None):
    return next(r for r in records if r["kind"] == kind and (method is None or r.get("method") == method))


def unknown(report):
    assert report["verdict"] == "unknown" and report["scope_verdict"] == "inconclusive"
    assert report["receipts"] == [] and report["missing"]
    for key in ("ownership_verdict", "read_rights_verdict", "lease_join_verdict", "physical_read_verdict",
                "tenant_authorization", "content_provenance", "kv_load_admission"):
        assert report[key] == "unknown"


def replace_token(value, before, after):
    """Replace one inert metadata identity without executing a native method."""
    if type(value) is dict:
        return {key: replace_token(item, before, after) for key, item in value.items()}
    if type(value) is list:
        return [replace_token(item, before, after) for item in value]
    return after if value == before else value


def paired_object_tokens(records):
    result = {}
    for record in records:
        if record["kind"] in ("init_begin", "worker_init", "scheduler_ready"):
            result.update(record["objects"])
        elif record["kind"] == "configs_derived":
            result.update({role: record[role] for role in ("worker_config", "scheduler_config")})
        elif record["kind"] == "runner_ready":
            result.update({"runner_config": record["runner_config"], "worker_pool": record["worker_pool"]})
    return result


def full_mapping_context():
    """The existing ordinary metadata fixture, with its native source header."""
    recipient, _, _, initial = example()
    records = ordinary_records()
    records[0] = deepcopy(initial[0])
    payload, receipts = paired(records)
    return recipient, R._Mapping(C.validate_captured(payload, receipts))


@pytest.mark.parametrize("mode", ["ordinary", "external", "computed_empty"])
def test_completed_positive_capacity_receipt_requires_real_paired_semantic_context(mode):
    records, payload, receipts, _ = example(mode)
    assert C.validate_captured(payload, receipts)["bookkeeping"] == "locally consistent"
    report = R.evaluate(records, payload, receipts)
    assert report["verdict"] == report["status"] == "observed", report["missing"]
    assert report["metadata_completeness"] == "complete" and report["scope_verdict"] == "inconclusive"
    assert report["summary"]["completed_grants"] == len(report["receipts"]) == 1
    receipt = report["receipts"][0]
    assert receipt["blocks"] == [{"block": 2, "incarnation": 1}]
    assert receipt["request_incarnation"] == token(9)
    assert receipt["method"] == (R.EXTERNAL if mode == "external" else R.NEW)
    assert report["mapping_context"]["binding"] == binding(payload, receipts)
    assert report["read_rights_verdict"] == report["physical_read_verdict"] == "unknown"
    assert report["still_live_exclusive_grant"] == report["authenticity"] == "not established"
    if mode == "external":
        assert find(records, "allocation_end")["returned"] == [[]]
        assert find(records, "allocation_begin")["arguments"]["delay_cache_blocks"] is True
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mode", ["unexercised", "not_allocated"])
def test_zero_grants_and_normal_outer_none_never_become_a_safety_pass(mode):
    records, payload, receipts, _ = example(mode)
    report = R.evaluate(records, payload, receipts)
    unknown(report)
    assert report["status"] == "not_exercised" and report["metadata_completeness"] == "complete"
    assert report["summary"]["completed_grants"] == 0


def test_contract_is_detached_and_retains_every_native_source_pin():
    contract = R.adapter_contract()
    assert len(contract["files"]) == 36
    assert R.source_pins() == A.source_pins("vllm.0.32.uniproc.eager.native.v1")
    contract["files"].clear()
    contract["bounds"]["max_records"] = 1
    assert len(R.adapter_contract()["files"]) == 36 and R.MAX_RECORDS == 16384
    with pytest.raises(ValueError):
        R.adapter_contract("historical")


def test_actual_settled_reader_is_called_before_recipient_or_context_replay(monkeypatch):
    calls = []
    def blocked(payload, receipts):
        calls.append((payload, receipts))
        raise ValueError("actual closure reader marker")
    monkeypatch.setattr(C, "validate_captured", blocked)
    report = R.evaluate(None, {}, {})
    assert calls == [({}, {})] and report["missing"] == ["actual closure reader marker"]
    unknown(report)


@pytest.mark.parametrize("kind", ["missing_pair", "raw_tamper", "negative_close", "late_calls", "token_only"])
def test_missing_changed_negative_or_token_only_mapping_context_is_unknown(kind):
    records, payload, receipts, _ = example()
    name = next(iter(receipts))
    if kind == "missing_pair":
        receipts.clear()
    elif kind == "raw_tamper":
        payload[next(iter(payload))] += b"\n"
    elif kind in ("negative_close", "late_calls"):
        receipt = json.loads(receipts[name])
        if kind == "negative_close":
            receipt["close_succeeded"] = False
        else:
            receipt["journal"]["late_calls"] = 1
        receipts[name] = io.json_bytes(receipt)
    else:
        payload = {"session": token(1), "sha256": "a" * 64}
    unknown(R.evaluate(records, payload, receipts))


@pytest.mark.parametrize("kind", ["missing_init", "wrong_object", "wrong_ordinal", "wrong_parent"])
def test_hash_correct_mapping_envelope_is_not_a_semantic_replay_substitute(kind):
    records, _, _, m = example()
    if kind == "missing_init":
        m.remove(find(m, "configs_derived"))
    elif kind == "wrong_object":
        find(m, "scheduler_ready")["passed_config"] = token(999)
    elif kind == "wrong_ordinal":
        find(m, "acquire")["blocks"][0]["incarnation"] = 2
    else:
        [r for r in m if r["kind"] == "mutation_begin"][-1]["parent"] = None
    payload, receipts = paired(m)
    records[-1]["mapping_binding"] = binding(payload, receipts)
    unknown(R.evaluate(recount(records), payload, receipts))


@pytest.mark.parametrize("key,value", [
    ("mapping_session", token(999)), ("association", token(999)), ("process", token(999)),
    ("source_sha256", {}), ("coverage", []), ("callables", {}), ("objects", {}),
])
def test_recipient_identity_source_and_coverage_gaps_are_unknown(key, value):
    records, payload, receipts, _ = example()
    records[0][key] = value
    unknown(R.evaluate(recount(records), payload, receipts))


@pytest.mark.parametrize("role", ["core", "executor", "wrapper", "worker", "runner", "scheduler",
                                  "manager", "coordinator", "group_manager", "pool", "worker_config",
                                  "scheduler_config", "runner_config", "worker_pool"])
@pytest.mark.parametrize("namespace", ["request", "callable", "call"])
def test_recipient_namespaces_cannot_alias_any_paired_initialization_object(role, namespace):
    records, payload, receipts, mapping_records = example()
    target = paired_object_tokens(mapping_records)[role]
    if namespace == "request":
        find(records, "allocation_begin")["request_incarnation"] = target
    elif namespace == "callable":
        records[0]["callables"][sorted(records[0]["callables"])[0]] = target
    else:
        records = replace_token(records, find(records, "allocation_begin")["call"], target)
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "unknown"
    assert "namespace" in result["missing"][0]


@pytest.mark.parametrize("locator", ["initialization", "mutation", "mapping_session", "process"])
@pytest.mark.parametrize("namespace", ["request", "callable", "call"])
def test_recipient_namespaces_cannot_alias_paired_generated_or_recording_tokens(locator, namespace):
    records, payload, receipts, mapping_records = example()
    target = {"initialization": records[0]["association"],
              "mutation": find(mapping_records, "mutation_begin")["mutation"],
              "mapping_session": records[0]["mapping_session"], "process": records[0]["process"]}[locator]
    if namespace == "request":
        find(records, "allocation_begin")["request_incarnation"] = target
    elif namespace == "callable":
        records[0]["callables"][sorted(records[0]["callables"])[0]] = target
    else:
        records = replace_token(records, find(records, "allocation_begin")["call"], target)
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "unknown"


@pytest.mark.parametrize("alias", ["request_is_callable", "call_is_callable", "call_is_request",
                                   "phase_is_request", "request_is_session", "call_is_session"])
def test_request_callable_and_call_namespaces_stay_disjoint_in_both_directions(alias):
    records, payload, receipts, _ = example()
    request = find(records, "allocation_begin")
    callable_token = next(iter(records[0]["callables"].values()))
    if alias == "request_is_callable":
        request["request_incarnation"] = callable_token
    elif alias == "call_is_callable":
        records = replace_token(records, request["call"], callable_token)
    elif alias == "call_is_request":
        records = replace_token(records, request["call"], request["request_incarnation"])
    elif alias == "phase_is_request":
        records = replace_token(records, find(records, "phase_begin")["call"], request["request_incarnation"])
    elif alias == "request_is_session":
        request["request_incarnation"] = records[0]["session"]
    else:
        records = replace_token(records, request["call"], records[0]["session"])
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "unknown"


def test_complete_paired_registry_includes_requests_dispatch_and_frames():
    records, context = full_mapping_context()
    roles = context.token_roles
    assert roles[token(9)] == roles[token(10)] == "request"
    assert roles[token(11)] == "schedule_output"
    assert roles[token(12)] == "mapping_frame"
    assert roles[token(3)] == "mapping_initialization"
    assert roles[token(8)] == "mapping_mutation"
    for token_value in (token(9), token(11), token(12)):
        header = deepcopy(records[0])
        header["callables"][sorted(header["callables"])[0]] = token_value
        with pytest.raises(ValueError, match="namespace"):
            R._Recipient(header, context)
    recipient = R._Recipient(records[0], context)
    outer = deepcopy(find(records, "allocation_begin"))
    outer["request"] = "R-a"
    recipient.allocation_begin(outer)
    assert recipient.requests == {"R-a": token(9)}
    changed_name = R._Recipient(records[0], context)
    with pytest.raises(ValueError, match="name recorded by mapping"):
        changed_name.allocation_begin(find(records, "allocation_begin"))


def test_repeated_same_request_identity_and_same_native_roles_remain_supported_metadata():
    records, payload, receipts, _ = example("not_allocated")
    ordinary_call = deepcopy(records[1:-1])
    second = replace_token(ordinary_call, find(records, "allocation_begin")["call"], token(999))
    records[-1:-1] = second
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "complete"
    assert result["status"] == "not_exercised" and result["summary"]["outer_calls"] == 2


@pytest.mark.parametrize("method", ["BlockPool.unpin_blocks", "SingleTypeKVCacheManager._apply_cow"])
def test_unsupported_paired_mutation_is_unknown_even_without_acquisition(method):
    # Support-gate metadata only. No unpin or CoW body is executed.
    records, _, _, mapping_records = example()
    common = {key: mapping_records[0][key] for key in ("schema", "session", "process")}
    receiver = records[0]["objects"]["pool" if method.startswith("BlockPool.") else "group_manager"]
    mapping_records[-1:-1] = [dict(common, seq=0, kind="mutation_begin", association=token(3),
        mutation=token(950), method=method, receiver=receiver, parent=None),
        dict(common, seq=0, kind="mutation_end", association=token(3), mutation=token(950), success=True)]
    payload, receipts = paired(mapping_records)
    records[-1]["mapping_binding"] = binding(payload, receipts)
    assert C.validate_captured(payload, receipts)["bookkeeping"] == "locally consistent"
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "unknown"
    assert "support profile" in result["missing"][0]


@pytest.mark.parametrize("computed_tokens", [0, 4])
def test_source_impossible_all_zero_normal_none_is_incomplete_metadata(computed_tokens):
    # A contradictory closure record is rejected without invoking engine source.
    records, payload, receipts, _ = example("not_allocated")
    outer = find(records, "allocation_begin")
    outer["arguments"]["num_new_tokens"] = 0
    outer["request_state"]["num_computed_tokens"] = computed_tokens
    result = R.evaluate(recount(records), payload, receipts)
    unknown(result)
    assert result["metadata_completeness"] == "unknown"
    assert result["missing"] == ["all-zero source allocation cannot complete normally"]


@pytest.mark.parametrize("kind", ["missing_phase", "missing_return", "missing_end", "missing_footer", "gap"])
def test_missing_receipt_chain_is_unknown_without_foreign_or_unowned_inference(kind):
    records, payload, receipts, _ = example()
    if kind == "gap":
        records.insert(-1, {"schema": 1, "session": token(100), "process": token(2), "seq": 0,
                            "kind": "gap", "call": None, "reason": "missing local observation"})
    else:
        target = {"missing_phase": "phase_begin", "missing_return": "pool_return",
                  "missing_end": "allocation_end", "missing_footer": "footer"}[kind]
        records.remove(find(records, target))
    report = R.evaluate(recount(records), payload, receipts)
    unknown(report)
    assert "foreign" not in json.dumps(report["receipts"])


@pytest.mark.parametrize("location,key,value", [
    ("policy", "async_scheduling", 0), ("policy", "max_concurrent_batches", True),
    ("policy", "defer_block_free", True), ("request_policy", "n", True),
    ("request_policy", "pooling_params_absent", False), ("request_policy", "streaming", True),
    ("request_policy", "lora_request_absent", False), ("arguments", "num_new_tokens", True),
    ("arguments", "num_lookahead_tokens", 1), ("arguments", "retained_hit_group_ids", [0]),
    ("arguments", "computed_blocks_native_empty", False), ("arguments", "skip_zeroing_group_ids", [True]),
])
def test_strict_observed_policy_and_native_argument_premises(location, key, value):
    records, payload, receipts, _ = example()
    target = records[0][location] if location == "policy" else find(records, "allocation_begin")[location]
    target[key] = value
    unknown(R.evaluate(recount(records), payload, receipts))


@pytest.mark.parametrize("kind", ["wrong_request", "wrong_receiver", "wrong_arguments", "wrong_mutation",
                                  "wrong_pool", "wrong_count", "wrong_block", "wrong_return", "wrong_graph"])
def test_concrete_recipient_parent_pool_and_return_relations_are_required(kind):
    records, payload, receipts, _ = example()
    phase = find(records, "phase_begin", R.NEW)
    pool = find(records, "pool_return")
    if kind == "wrong_request":
        phase["request"] = "another-name"
    elif kind == "wrong_receiver":
        phase["receiver"] = token(999)
    elif kind == "wrong_arguments":
        phase["arguments"]["num_tokens"] += 1
    elif kind == "wrong_mutation":
        phase["mapping_mutation"] = pool["mutation"]
    elif kind == "wrong_pool":
        pool["pool"] = token(999)
    elif kind == "wrong_count":
        pool["requested_blocks"] = True
    elif kind == "wrong_block":
        pool["blocks"][0]["incarnation"] = 2
    elif kind == "wrong_return":
        find(records, "phase_end")["returned"] = []
    else:
        find(records, "allocation_end")["graph_revalidated"] = False
    unknown(R.evaluate(recount(records), payload, receipts))


def test_external_zeroing_policy_and_empty_outer_result_are_not_interchangeable():
    records, payload, receipts, _ = example("external")
    find(records, "phase_begin", R.EXTERNAL)["arguments"]["record_for_zeroing"] = True
    unknown(R.evaluate(recount(records), payload, receipts))
    records, payload, receipts, _ = example("external")
    find(records, "allocation_end")["returned"] = [[{"block": 2, "incarnation": 1}]]
    unknown(R.evaluate(recount(records), payload, receipts))


@pytest.mark.parametrize("key,value", [("records", True), ("block_facts", 99), ("completed_grants", 0),
                                      ("healthy", False), ("closed", False), ("open_calls", 1),
                                      ("source_stable", False), ("coverage_stable", False),
                                      ("body_bytes", 0), ("body_nodes", 0), ("mapping_binding", {})])
def test_footer_counters_raw_binding_and_closure_are_recomputed(key, value):
    records, payload, receipts, _ = example()
    records[-1][key] = value
    unknown(R.evaluate(records, payload, receipts))


def test_historical_strings_and_unbounded_or_opaque_inputs_remain_unknown():
    for records in ([], [{"kind": "allocation", "request": "saved-historical"}], iter([])):
        unknown(R.evaluate(records, {}, {}))
    records, payload, receipts, _ = example()
    records[1]["request"] = "x" * 129
    unknown(R.evaluate(recount(records), payload, receipts))
    records, payload, receipts, _ = example()
    records[1]["opaque"] = iter([])
    unknown(R.evaluate(records, payload, receipts))


def test_inputs_are_preserved_and_accounting_is_detached():
    records, payload, receipts, _ = example()
    before = deepcopy((records, payload, receipts))
    report = R.evaluate(records, payload, receipts)
    assert (records, payload, receipts) == before
    report["receipts"][0]["blocks"].clear()
    assert find(records, "pool_return")["blocks"]
    assert R.measure(records[:-1]) == {key: records[-1][key] for key in ("body_bytes", "body_nodes")}
    with pytest.raises(ValueError):
        R.measure([{"not_json": 1.0}])


def test_unknown_summary_json_preserves_null_and_complete_capture_preserves_measured_zero():
    records, payload, receipts, _ = example()
    complete = json.loads(json.dumps(R.evaluate(records, payload, receipts), allow_nan=False))
    assert complete["summary"] == {"outer_calls": 1, "phase_calls": 2, "pool_returns": 1,
                                   "block_facts": 1, "empty_pool_returns": 0, "completed_grants": 1}
    records.remove(find(records, "allocation_end"))
    incomplete = json.loads(json.dumps(R.evaluate(recount(records), payload, receipts), allow_nan=False))
    unknown(incomplete)
    assert incomplete["metadata_completeness"] == "unknown"
    assert set(incomplete["summary"]) == set(complete["summary"])
    assert all(value is None for value in incomplete["summary"].values())
    assert all(value is None for value in R.evaluate([], {}, {})["summary"].values())
