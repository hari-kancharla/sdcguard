"""CPU bookkeeping controls. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import pytest

from sdcguard import checksum_client as client, evidence_io as io, kv_checksum_observer as checksum

DATA = Path(__file__).parent / "data/checksum_client_functions"


def _binding_budget(snapshot):
    snapshot["budgets"]["retained_records"] = len(snapshot["records"])
    snapshot["budgets"]["retained_bytes"] = sum(len(io.json_bytes(row)) for row in snapshot["records"])


def _checksum_budget(snapshot):
    records = snapshot["records"]
    snapshot["budgets"]["retained_records"] = len(records)
    snapshot["budgets"]["retained_bytes"] = sum(len(json.dumps(row, sort_keys=True, allow_nan=False).encode()) for row in records)
    snapshot["budgets"]["reserved_record_bytes"] = max(snapshot["budgets"]["reserved_record_bytes"], snapshot["budgets"]["retained_bytes"])
    snapshot["budgets"]["retained_units"] = sum(len(row.get(key) or []) for row in records
                                                 for key in ("expected_units", "compared_units", "missing_units"))


def client_case(case="complete"):
    """Detached metadata assembly, never a new engine, HTTP or GPU observation."""
    assert case in ("complete", "partial-span", "empty-replicas")
    run, packet = "1" * 32, "2" * 64
    tokens = list(range(32))
    phase = dict(phase_id="c1", concurrency=1)
    arms = []
    for name, enabled in (("off", False), ("on", True)):
        arms.append(dict(arm_id=name, engine_session_id="engine-" + name, observer_enabled=enabled,
            requests=[dict(slot=0, phase_id="c1", kind="measured", concurrency=1,
                prefill_request_id="prefill-" + name, client_request_id="decode-" + name,
                prompt_tokens=len(tokens), prompt_sha256=client.prompt_digest(tokens), max_tokens=16)]))
    plan = dict(schema=client.PLAN_SCHEMA, run_id=run, packet_sha256=packet, source_revision=client.REVISION,
                model="synthetic-cpu-model", phases=[phase], arms=arms,
                evidence_origin="synthetic_cpu_bookkeeping_assembly")
    inventory = dict(schema=client.INVENTORY_SCHEMA, run_id=run, packet_sha256=packet,
                     plan_sha256=io.digest(io.json_bytes(plan)), completed=True, requests=[],
                     evidence_origin="synthetic_cpu_bookkeeping_assembly")
    descriptor = dict(do_remote_prefill=True, do_remote_decode=False,
                      remote_request_id="synthetic-producer-internal", remote_engine_id="prefill-engine",
                      remote_block_ids=[[1, 2]], remote_host="127.0.0.1", remote_port=5500,
                      tp_size=1, dcp_size=1, pp_size=1, remote_num_tokens=len(tokens),
                      remote_registration_generation=0, kv_checksums="Y3B1")
    for arm in arms:
        request = arm["requests"][0]
        def stage(name):
            rid = request["prefill_request_id"] if name == "prefill" else request["client_request_id"]
            body = dict(model=plan["model"], request_id=rid, prompt=list(tokens), n=1, stream=False,
                        add_special_tokens=False, max_tokens=1 if name == "prefill" else 16,
                        kv_transfer_params=dict(do_remote_decode=True, do_remote_prefill=False) if name == "prefill" else deepcopy(descriptor))
            response = dict(id="cmpl-" + rid, usage=dict(prompt_tokens=len(tokens)), choices=[])
            if name == "prefill":
                response["kv_transfer_params"] = deepcopy(descriptor)
            return dict(completed=True, method="POST", path="/v1/completions", http_status=200,
                        request_headers={"X-Request-Id": rid}, request_body=body, response_body=response, error=None)
        inventory["requests"].append(dict(arm_id=arm["arm_id"], slot=0,
            client_request_id=request["client_request_id"], prefill=stage("prefill"), decode=stage("decode")))
    context = dict(run_id=run, packet_sha256=packet, arm_id="on", engine_session_id="engine-on")
    external, internal = "cmpl-decode-on-0", "cmpl-decode-on-0-aabbccdd"
    binding = dict(schema=client.BINDING_SCHEMA, profile=client.BINDING_PROFILE,
        qualification="cpu_source_functions", runtime_qualified=False, release_qualifying=False,
        **context, process_id=101, source=dict(profile=client.BINDING_PROFILE, revision=client.REVISION,
        files=dict(client.SOURCE_FILES), supported=True, reason="synthetic_cpu_bookkeeping_assembly"),
        closed=True, health=dict(errors=0, dropped_records=0, late_calls=0, active_calls=0,
                                 budget_exhausted=False, complete=True),
        budgets=dict(max_records=4096, max_bytes=2 * 1024 * 1024, retained_records=0,
                     retained_bytes=0, fixed_metadata_bytes=client.BINDING_FIXED_BYTES),
        records=[dict(sequence=1, external_request_id=external, external_req_id=external,
                      internal_request_id=internal, native_return_is_none=True,
                      engine_exception=None, randomization_observed=True,
                      prompt_tokens=len(tokens), prompt_sha256=client.prompt_digest(tokens))])
    _binding_budget(binding)
    snapshot = json.loads((DATA / (case + ".json")).read_bytes())
    for row in snapshot["records"]:
        row["request_id"] = internal
    _checksum_budget(snapshot)
    snapshot["capture_context"] = dict(context, process_id=102)
    startup = dict(context, role="decode", source_revision=client.REVISION, model=plan["model"],
                   source_profile=client.source_profile(), checksum_enabled=True, request_id_randomization_enabled=True)
    capture = dict(arm_id="on", engine_session_id="engine-on", source_revision=client.REVISION,
                   capture_complete=True, startup=startup, startup_sha256=io.digest(io.json_bytes(startup)),
                   binding_snapshots=[binding], checksum_snapshots=[snapshot])
    captures = dict(schema=client.CAPTURES_SCHEMA, run_id=run, packet_sha256=packet,
                    plan_sha256=io.digest(io.json_bytes(plan)), captures=[capture],
                    evidence_origin="synthetic_cpu_bookkeeping_assembly")
    return plan, inventory, captures


def case_bytes(case="complete"):
    return tuple(io.json_bytes(value) for value in client_case(case))


def _evaluate(values):
    return client.evaluate(*(io.json_bytes(value) for value in values))


def _on(report):
    return next(row for row in report["requests"] if row["observer_enabled"])


def test_complete_client_join_is_conditional_and_cannot_promote_release():
    report = client.evaluate(*case_bytes())
    assert report["client_inventory"]["status"] == "complete"
    assert report["summary"]["declared_requests"] == 2
    assert report["summary"]["on_requests"] == report["summary"]["off_requests"] == 1
    assert report["summary"]["checksum_calls"] == report["summary"]["joined_checksum_calls"] == 1
    assert report["summary"]["verified_on_requests"] == 1
    assert report["summary"]["unexpected_binding_records"] == 0
    assert _on(report)["binding_status"] == "matched"
    assert _on(report)["coverage"] == "complete"
    assert report["runtime_qualified"] is False and report["release_decision"] == "HOLD"
    assert next(row for row in report["requests"] if not row["observer_enabled"])["reasons"] == ["observer_disabled"]


@pytest.mark.parametrize("case,reason", [("partial-span", "partial_checksum_span"),
                                         ("empty-replicas", "empty_replica_inputs")])
def test_native_pass_coverage_gaps_survive_client_join(case, reason):
    report = client.evaluate(*case_bytes(case))
    row = _on(report)
    assert row["native_result"] == "passed" and row["coverage"] == "unknown"
    assert report["summary"]["verified_on_requests"] == 0
    assert report["summary"]["engine_pass_not_covered"] == {reason: 1}
    assert reason in row["reasons"]
    assert row["checksum_calls"][0]["conditional_scope"]["compared_units"] is None


def test_missing_client_row_does_not_shrink_planned_denominator():
    values = client_case()
    values[1]["requests"] = values[1]["requests"][:1]
    report = _evaluate(values)
    assert report["summary"]["declared_requests"] == 2
    assert report["summary"]["on_requests"] == 1
    assert report["client_inventory"]["status"] == "unknown"
    assert "client_receipt_missing" in _on(report)["reasons"]
    assert report["summary"]["unexpected_checksum_calls"] == 1


@pytest.mark.parametrize("stage,key,value,reason", [
    ("decode", "completed", False, "decode_http_completion_unverified"),
    ("prefill", "http_status", 503, "prefill_http_completion_unverified"),
    ("decode", "http_status", True, "decode_http_completion_unverified"),
    ("decode", "error", "recorded client error", "decode_http_completion_unverified"),
    ("prefill", "request_body", None, "prefill_request_or_response_missing"),
])
def test_incomplete_http_receipts_remain_unverified(stage, key, value, reason):
    values = client_case()
    values[1]["requests"][1][stage][key] = value
    report = _evaluate(values)
    assert reason in _on(report)["reasons"]
    assert report["summary"]["verified_on_requests"] == 0
    assert report["client_inventory"]["status"] == "unknown"


@pytest.mark.parametrize("part", ["header", "body", "response", "tokens", "usage", "descriptor", "n"])
def test_client_contract_must_bind_actual_native_id_and_prompt(part):
    values = client_case()
    stage = values[1]["requests"][1]["decode"]
    if part == "header":
        stage["request_headers"]["X-Request-Id"] = "unbound-header"
    elif part == "body":
        stage["request_body"]["request_id"] = "unbound-body"
    elif part == "response":
        stage["response_body"]["id"] = "cmpl-unbound-response"
    elif part == "tokens":
        stage["request_body"]["prompt"][0] += 1
    elif part == "usage":
        stage["response_body"]["usage"]["prompt_tokens"] += 1
    elif part == "descriptor":
        stage["request_body"]["kv_transfer_params"]["remote_request_id"] = "unbound-source"
    else:
        stage["request_body"]["n"] = True
    report = _evaluate(values)
    assert _on(report)["client_status"] == "unknown"
    assert report["summary"]["verified_on_requests"] == 0


@pytest.mark.parametrize("target", ["inventory", "captures"])
def test_exact_plan_hash_and_run_identity_are_required(target):
    values = client_case()
    values[1 if target == "inventory" else 2]["plan_sha256"] = "0" * 64
    report = _evaluate(values)
    assert _on(report)["check_status"] == "unverified"
    assert report["summary"]["joined_checksum_calls"] == 0


@pytest.mark.parametrize("key,value", [("run_id", "3" * 32), ("packet_sha256", "4" * 64),
                                      ("arm_id", "off"), ("engine_session_id", "other-engine"),
                                      ("process_id", True)])
def test_checksum_snapshot_context_cannot_borrow_another_arm_or_run(key, value):
    values = client_case()
    values[2]["captures"][0]["checksum_snapshots"][0]["capture_context"][key] = value
    report = _evaluate(values)
    assert "checksum_capture_context_unverified" in _on(report)["reasons"]
    assert report["summary"]["verified_on_requests"] == 0
    assert _on(report)["native_result"] == "passed"


@pytest.mark.parametrize("change", ["external", "internal_missing", "source", "open", "loss", "not_randomized"])
def test_actual_binding_evidence_is_required(change):
    values = client_case()
    binding = values[2]["captures"][0]["binding_snapshots"][0]
    if change == "external":
        binding["records"][0]["external_req_id"] = "unbound-after"
    elif change == "internal_missing":
        binding["records"] = []
    elif change == "source":
        binding["source"]["files"][client.INPUT_PROCESSOR] = "0" * 64
    elif change == "open":
        binding["closed"] = False
    elif change == "loss":
        binding["health"]["dropped_records"] = 1
    else:
        binding["records"][0]["randomization_observed"] = False
    _binding_budget(binding)
    report = _evaluate(values)
    assert report["summary"]["verified_on_requests"] == 0
    assert report["summary"]["joined_checksum_calls"] == 0
    assert _on(report)["binding_status"] == "unknown"


@pytest.mark.parametrize("change", ["missing_capture", "checksum_open", "checksum_loss", "checksum_absent",
                                    "startup_missing", "startup_hash", "startup_flags", "mixed_domain"])
def test_recording_and_startup_gaps_hold_every_required_client_check(change):
    values = client_case()
    capture = values[2]["captures"][0]
    if change == "missing_capture":
        values[2]["captures"] = []
    elif change == "checksum_open":
        capture["checksum_snapshots"][0]["closed"] = False
    elif change == "checksum_loss":
        capture["checksum_snapshots"][0]["health"]["dropped_records"] = 1
    elif change == "checksum_absent":
        capture["checksum_snapshots"] = []
    elif change == "startup_missing":
        capture.pop("startup")
    elif change == "startup_hash":
        capture["startup_sha256"] = "0" * 64
    elif change == "startup_flags":
        capture["startup"]["request_id_randomization_enabled"] = False
        capture["startup_sha256"] = io.digest(io.json_bytes(capture["startup"]))
    else:
        capture["binding_snapshots"][0]["qualification"] = "source_bound_candidate"
    report = _evaluate(values)
    assert _on(report)["check_status"] == "unverified"
    assert report["summary"]["verified_on_requests"] == 0


def test_unexpected_calls_are_retained_and_not_counted_as_verified_clients():
    values = client_case()
    extra = deepcopy(values[2]["captures"][0]["checksum_snapshots"][0])
    for record in extra["records"]:
        record["request_id"] = "unplanned-ordinary-native-request"
    _checksum_budget(extra)
    extra["capture_context"]["process_id"] += 1
    values[2]["captures"][0]["checksum_snapshots"].append(extra)
    report = _evaluate(values)
    assert report["summary"]["checksum_calls"] == 2
    assert report["summary"]["unexpected_checksum_calls"] == 2
    assert len(report["unexpected_checksum_calls"]) == 2
    assert report["summary"]["declared_requests"] == 2


def test_malformed_surplus_client_rows_stay_visible_without_aliasing_boolean_slots():
    values = client_case()
    values[1]["requests"].append(dict(arm_id="on", slot=False, client_request_id="decode-on"))
    report = _evaluate(values)
    assert report["summary"]["declared_requests"] == 2
    assert report["client_inventory"]["status"] == "unknown"
    assert report["client_inventory"]["unexpected_rows"][0]["reason"] == "unplanned_or_malformed_client_row"


def test_client_receipt_order_does_not_control_native_join_order():
    values = client_case()
    values[1]["requests"].reverse()
    report = _evaluate(values)
    assert report["summary"]["verified_on_requests"] == 1
    assert [row["arm_id"] for row in report["requests"]] == ["off", "on"]
    assert _on(report)["phase_id"] == "c1" and _on(report)["kind"] == "measured"


@pytest.mark.parametrize("part", ["phase", "tokens", "source", "slot"])
def test_unsupported_plan_contract_is_refused(part):
    values = client_case()
    row = values[0]["arms"][1]["requests"][0]
    if part == "phase":
        row["phase_id"] = []
    elif part == "tokens":
        row["prompt_tokens"] = True
    elif part == "source":
        values[0]["source_revision"] = "unknown"
    else:
        row["slot"] = 2
    with pytest.raises(io.EvidenceError):
        _evaluate(values)


def native_binding(source_root):
    """Compile the original callback without executing engine imports."""
    primary = json.loads((DATA / "PRIMARY.json").read_bytes())
    for source in primary["sources"]:
        raw = source["content"].encode()
        assert io.digest(raw) == source["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == source["git_blob_sha"]
        path = source_root / source["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    filename = source_root / client.INPUT_PROCESSOR
    code = checksum._codes(compile(filename.read_bytes(), str(filename), "exec"))["InputProcessor.assign_request_id"]
    class EngineCoreRequest:
        __slots__ = ("request_id", "external_req_id", "prompt_token_ids")
        def __init__(self, request_id):
            self.request_id, self.external_req_id = request_id, None
            self.prompt_token_ids = list(range(32))
    calls = []
    def random_uuid():
        calls.append(None)
        return "aabbccdd" + "0" * 24
    namespace = dict(EngineCoreRequest=EngineCoreRequest, envs=SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False),
                     random_uuid=random_uuid, logger=SimpleNamespace(warning_once=lambda *args: None))
    function = FunctionType(code, namespace, "assign_request_id")
    processor = type("InputProcessor", (), {"assign_request_id": staticmethod(function)})
    namespace["InputProcessor"] = processor
    return SimpleNamespace(processor=processor, request=EngineCoreRequest, calls=calls, original=function,
                           source_root=source_root)


def _observer(**kwargs):
    return client.RequestBindingObserver(run_id="1" * 32, packet_sha256="2" * 64,
        arm_id="on", engine_session_id="engine-on", **kwargs)


def test_original_assignment_callback_is_called_once_and_remains_native(tmp_path):
    native = native_binding(tmp_path)
    request = native.request("cmpl-ordinary-0")
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    assert native.processor.assign_request_id(request) is None
    assert request.external_req_id == "cmpl-ordinary-0"
    assert request.request_id == "cmpl-ordinary-0-aabbccdd"
    assert len(native.calls) == 1
    snapshot = recorder.close()
    assert snapshot["health"]["complete"] is True
    assert snapshot["qualification"] == "cpu_source_functions"
    assert snapshot["records"][0]["internal_request_id"] == request.request_id
    assert snapshot["records"][0]["prompt_tokens"] == 32
    assert snapshot["records"][0]["prompt_sha256"] == client.prompt_digest(request.prompt_token_ids)
    assert native.processor.assign_request_id is native.original
    assert recorder.close() == snapshot


def test_open_binding_snapshot_cannot_supply_complete_evidence(tmp_path):
    native = native_binding(tmp_path)
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    try:
        native.processor.assign_request_id(native.request("cmpl-ordinary-0"))
        snapshot = recorder.snapshot()
        assert snapshot["closed"] is False and snapshot["health"]["complete"] is False
    finally:
        recorder.close()


def test_binding_record_budget_drops_metadata_without_changing_native_calls(tmp_path):
    native = native_binding(tmp_path)
    recorder = _observer(max_records=1).install_cpu_source_functions(native.processor, source_root=tmp_path)
    first, second = native.request("cmpl-first-0"), native.request("cmpl-second-0")
    native.processor.assign_request_id(first)
    native.processor.assign_request_id(second)
    snapshot = recorder.close()
    assert len(native.calls) == 2
    assert first.request_id == "cmpl-first-0-aabbccdd" and second.request_id == "cmpl-second-0-aabbccdd"
    assert snapshot["health"]["dropped_records"] == 1 and snapshot["health"]["budget_exhausted"] is True
    assert snapshot["health"]["complete"] is False and len(snapshot["records"]) == 1


def test_original_attribute_error_is_not_replaced_by_recording_error(tmp_path):
    native = native_binding(tmp_path)
    request = native.request("cmpl-ordinary-0")
    del request.external_req_id
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    with pytest.raises(AttributeError) as caught:
        native.processor.assign_request_id(request)
    assert caught.value.__traceback__ is not None and native.calls == []
    assert recorder.close()["health"]["complete"] is False


def test_foreign_wrapper_is_preserved_and_makes_closure_unknown(tmp_path):
    native = native_binding(tmp_path)
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    foreign = staticmethod(lambda request: None)
    native.processor.assign_request_id = foreign
    snapshot = recorder.close()
    assert vars(native.processor)["assign_request_id"] is foreign
    assert snapshot["health"]["complete"] is False and snapshot["source"]["supported"] is False


def test_binding_source_guard_checks_loaded_callback_and_full_files(tmp_path):
    native = native_binding(tmp_path)
    changed = tmp_path / client.INPUT_PROCESSOR
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(checksum.UnsupportedSource):
        _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    assert native.calls == [] and native.processor.assign_request_id is native.original


def test_binding_snapshot_is_detached_from_later_caller_edits(tmp_path):
    native = native_binding(tmp_path)
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    native.processor.assign_request_id(native.request("cmpl-ordinary-0"))
    snapshot = recorder.snapshot()
    snapshot["records"].clear()
    assert len(recorder.close()["records"]) == 1


@pytest.mark.parametrize("key,value", [
    ("do_remote_prefill", False), ("do_remote_decode", True), ("remote_engine_id", ""),
    ("remote_host", None), ("remote_port", 0), ("tp_size", True), ("pp_size", 2),
    ("remote_num_tokens", 31), ("remote_registration_generation", -1),
    ("remote_block_ids", [[False]]), ("kv_checksums", ""), ("kv_checksums", "not base64!"),
])
def test_matching_client_descriptor_copies_cannot_hide_an_unsupported_direction_or_scope(key, value):
    values = client_case()
    row = values[1]["requests"][1]
    row["prefill"]["response_body"]["kv_transfer_params"][key] = value
    row["decode"]["request_body"]["kv_transfer_params"][key] = value
    report = _evaluate(values)
    assert "client_transfer_descriptor_binding_unverified" in _on(report)["reasons"]
    assert report["summary"]["verified_on_requests"] == 0
    assert _on(report)["native_result"] == "passed"


def test_prefill_direction_is_checked_independently_of_response_descriptor():
    values = client_case()
    values[1]["requests"][1]["prefill"]["request_body"]["kv_transfer_params"]["do_remote_decode"] = False
    report = _evaluate(values)
    assert "client_prefill_direction_unverified" in _on(report)["reasons"]
    assert report["summary"]["verified_on_requests"] == 0


@pytest.mark.parametrize("key,value", [("prompt_tokens", 31), ("prompt_sha256", "0" * 64)])
def test_actual_engine_input_digest_must_match_client_prompt(key, value):
    values = client_case()
    binding = values[2]["captures"][0]["binding_snapshots"][0]
    binding["records"][0][key] = value
    _binding_budget(binding)
    report = _evaluate(values)
    assert "native_prompt_to_client_binding_unverified" in _on(report)["reasons"]
    assert report["summary"]["verified_on_requests"] == 0
    assert report["summary"]["joined_checksum_calls"] == 0


@pytest.mark.parametrize("change", ["unexpected_arm", "duplicate_envelope"])
def test_calls_inside_unexpected_or_ambiguous_envelopes_never_disappear(change):
    values = client_case()
    extra = deepcopy(values[2]["captures"][0])
    if change == "unexpected_arm":
        extra["arm_id"] = "not-a-planned-arm"
    values[2]["captures"].append(extra)
    report = _evaluate(values)
    assert report["summary"]["checksum_calls"] == 2
    assert report["summary"]["native_results"] == {"passed": 2}
    if change == "unexpected_arm":
        assert report["summary"]["joined_checksum_calls"] == 1
        assert report["summary"]["unexpected_checksum_calls"] == 1
        unplanned = [capture for capture in report["captures"] if capture["planned_arm"] is False]
        assert len(unplanned) == 1 and unplanned[0]["engine_session_id"] is None
        assert report["release_decision"] == "HOLD"
    else:
        assert report["summary"]["joined_checksum_calls"] == 0
        assert report["summary"]["unexpected_checksum_calls"] == 2
    assert report["summary"]["declared_requests"] == 2


def test_unsupported_engine_prompt_is_recorded_without_changing_native_assignment(tmp_path):
    native = native_binding(tmp_path)
    request = native.request("cmpl-ordinary-0")
    request.prompt_token_ids = None
    recorder = _observer().install_cpu_source_functions(native.processor, source_root=tmp_path)
    native.processor.assign_request_id(request)
    snapshot = recorder.close()
    assert request.request_id == "cmpl-ordinary-0-aabbccdd" and len(native.calls) == 1
    assert snapshot["records"][0]["prompt_tokens"] is None
    assert snapshot["records"][0]["prompt_sha256"] is None
    context = {key: snapshot[key] for key in client.CONTEXT_KEYS}
    _, reasons, _ = client._bindings(snapshot, context)
    assert "native_prompt_recording_unverified" in reasons


def test_binding_context_is_immutable_after_construction():
    recorder = _observer()
    with pytest.raises(TypeError):
        recorder.context["engine_session_id"] = "another-session"


@pytest.mark.parametrize("target,key", [("inventory", "requests"), ("captures", "captures"),
                                      ("capture", "binding_snapshots"), ("capture", "checksum_snapshots"),
                                      ("binding", "records")])
def test_missing_or_malformed_lists_keep_every_planned_client_visible(target, key):
    values = client_case()
    choices = dict(inventory=values[1], captures=values[2], capture=values[2]["captures"][0],
                   binding=values[2]["captures"][0]["binding_snapshots"][0])
    choices[target][key] = None
    report = _evaluate(values)
    assert report["summary"]["declared_requests"] == 2
    assert len(report["requests"]) == 2
    assert report["summary"]["verified_on_requests"] == 0
    assert report["release_decision"] == "HOLD"
