"""Pinned native functions with synthetic CPU inputs. Owner: Hari Krishna Kancharla.

These controls execute extracted functions, not full engines or serving traffic.
They establish native return and counter semantics, not runtime integration.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from sdcguard.native_checks import assess_check
from sdcguard.required_gate import decide


def _required_checksum_holds(check):
    result = decide([dict(client_request_id="synthetic-ordinary-request", checks={"checksum": check})], ["checksum"])
    assert result["decision"] == "HOLD" and result["requests"][0]["reasons"]


DATA = Path(__file__).parent / "data" / "native_check_engine_functions"
MANIFEST = json.loads((DATA / "MANIFEST.json").read_text())
FRAGMENTS = {record["id"]: record for record in MANIFEST["fragments"]}
SG_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"
VLLM_COMMIT = "29d2f4e03740acb14439ce22d24b06d8095d17d3"


def _semantic_ast(value):
    """Ignore only the empty type_params field added after Python 3.10."""
    if isinstance(value, ast.AST):
        return {
            "node": type(value).__name__,
            "fields": {
                name: _semantic_ast(field_value)
                for name, field_value in ast.iter_fields(value)
                if not (name == "type_params" and field_value == [])
            },
        }
    if isinstance(value, list):
        return [_semantic_ast(item) for item in value]
    if value is Ellipsis:
        return {"literal": "Ellipsis"}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, complex):
        return {"complex_real": value.real, "complex_imag": value.imag}
    return value


def _source(record):
    raw = (DATA / record["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    source = raw.decode()
    if record["is_method"]:
        parsed = ast.parse("class NativeFunctionContainer:\n" + source).body[0].body[0]
    else:
        parsed = ast.parse(source).body[0]
    assert record["semantic_ast_schema"] == "sdcguard.ast.ignore-empty-type-params.v1"
    normalized = json.dumps(_semantic_ast(parsed), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(normalized.encode()).hexdigest() == record["semantic_ast_sha256"]
    return source


def _load(module, fragment_id):
    record = FRAGMENTS[fragment_id]
    source = _source(record)
    if record["is_method"]:
        source = "class NativeFunctionContainer:\n" + source
    exec(compile("from __future__ import annotations\n" + source, record["file"], "exec"), vars(module))
    if record["is_method"]:
        setattr(module, record["symbol"], vars(module.NativeFunctionContainer)[record["symbol"]])


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("native_check_semantics_fragments")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.np, module.torch = np, torch
    module.dataclass, module.field = dataclass, field
    module.IntEnum = IntEnum
    module.ReplicaKey = tuple[int, int]
    # These cases use ordinary attention groups. This type only preserves the
    # upstream isinstance boundary; no Mamba comparison is claimed or executed.
    module.MambaSpec = type("UnusedMambaSpec", (), {})
    module.logger = logging.Logger("native_check_cpu_control")
    module.logger.warning_once = module.logger.warning
    for fragment_id in FRAGMENTS:
        _load(module, fragment_id)
    return module


def _bind(target, native, *names):
    for name in names:
        setattr(target, name, MethodType(getattr(native, name), target))
    return target


def test_fragments_are_pinned_unchanged_and_non_release_qualifying():
    assert MANIFEST["evidence_origin"] == "synthetic"
    assert MANIFEST["execution_scope"] == "selected_unchanged_functions_on_cpu"
    assert MANIFEST["full_engine_execution"] is False
    assert MANIFEST["cuda_execution"] is False
    assert MANIFEST["release_qualifying"] is False
    assert len(FRAGMENTS) == 24
    for record in FRAGMENTS.values():
        assert record["source_commit"] in {SG_COMMIT, VLLM_COMMIT}
        assert record["source_commit"] in record["source_url"]
        assert record["first_line"] <= record["last_line"]
        assert len(record["source_git_blob_sha"]) == 40
        assert len(record["source_sha256"]) == 64
        assert record["source_bytes"] > 0
        assert record["only_edit"] == "none; original source lines retained"
        _source(record)
    for license_record in MANIFEST["licenses"]:
        assert hashlib.sha256((DATA / license_record["file"]).read_bytes()).hexdigest() == license_record["sha256"]


def test_portable_ast_proof_keeps_body_and_literal_differences():
    original = _semantic_ast(ast.parse("def compare(x):\n    return x == 0\n"))
    changed_operator = _semantic_ast(ast.parse("def compare(x):\n    return x != 0\n"))
    changed_literal = _semantic_ast(ast.parse("def compare(x):\n    return x == 1\n"))
    changed_body = _semantic_ast(ast.parse("def compare(x):\n    return True\n"))
    assert original != changed_operator
    assert original != changed_literal
    assert original != changed_body
    assert _semantic_ast(ast.parse("[]")) != _semantic_ast(ast.parse("None"))


def test_sglang_expected_zero_admits_without_a_checksum_comparison(native):
    class SyntheticComputer:
        calls = 0

        def compute(self, *_):
            self.calls += 1
            raise AssertionError("expected-zero branch must not compute")

    class SyntheticBatch:
        reqs = [SimpleNamespace(rid="synthetic-ordinary-request", expected_kv_checksum=0)]
        filter_calls = 0

        def filter_batch(self, **_):
            self.filter_calls += 1
            raise AssertionError("expected-zero request remains admitted")

    computer, batch = SyntheticComputer(), SyntheticBatch()
    scheduler = SimpleNamespace(kv_checksum_computer=computer)
    native_result = native._verify_kv_checksums(scheduler, batch)
    assert native.is_health_check_req(batch.reqs[0]) is False
    assert native_result is None
    assert len(batch.reqs) == 1 and batch.filter_calls == 0
    assert computer.calls == 0
    result = assess_check(
        "checksum", native_result="admitted", expected_units=["transferred_kv"],
        compared_units=[], completed=False, evidence_complete=True,
        reason="expected checksum is zero; comparison skipped",
    )
    assert result["native_result"] == "admitted"
    assert result["check_status"] == "unverified"
    assert result["coverage"] == "not_covered"
    assert result["missing_units"] == ["transferred_kv"]
    _required_checksum_holds(result)


def _vllm_case(native, *, compared_groups, mismatch=False):
    blocks = native.KVChecksumBlocks(16, {0: (0, [101]), 1: (0, [201])})
    reports, sent = {}, {}
    for group_id in compared_groups:
        value = 7 + group_id
        reports[group_id] = native.KVChecksumGroup(
            0, [(1, 0)], np.array([[value + int(mismatch and group_id == 0)]], dtype=np.uint64),
        )
        sent[group_id] = (123 + group_id, 16, 0, np.array([value], dtype=np.uint64))
    record = native.KVChecksumRecord(
        pp_rank=0, tp_rank=0, checksums={"synthetic-request": reports},
        skipped_groups=frozenset({0, 1} - set(compared_groups)),
        fingerprints={0: 123, 1: 124},
    )
    pending = native._Pending(
        blocks=blocks, expected=native.KVChecksumPayload(1, 16, sent),
        reports={(0, 0): (reports, record)},
    )
    scheduler = _bind(SimpleNamespace(
        _num_workers=1, _tp_size=1,
        _groups={g: SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16)) for g in blocks.groups},
    ), native, "_verify", "_compare", "_compare_group")
    return scheduler, pending


def test_vllm_native_true_can_leave_an_actual_transfer_group_uncompared(native):
    scheduler, pending = _vllm_case(native, compared_groups={0})
    outcomes = {
        group_id: scheduler._compare_group(pending, pending.expected, group_id, group_id in pending.skipped_groups())
        for group_id in pending.blocks.groups
    }
    assert outcomes == {0: [], 1: None}
    assert scheduler._compare(pending) == ({}, [])
    assert scheduler._verify("synthetic-request", pending) is True
    result = assess_check(
        "checksum", native_result="passed",
        expected_units=[f"group:{g}" for g in pending.blocks.groups],
        compared_units=[f"group:{g}" for g, outcome in outcomes.items() if outcome is not None],
        completed=True, evidence_complete=True,
        reason="transferred group skipped by both checksum sides",
    )
    assert result["native_result"] == "passed"
    assert result["coverage"] == "partial"
    assert result["check_status"] == "unverified"
    assert result["missing_units"] == ["group:1"]
    _required_checksum_holds(result)


def test_vllm_comparing_every_transfer_group_can_pass_coverage(native):
    scheduler, pending = _vllm_case(native, compared_groups={0, 1})
    assert scheduler._verify("synthetic-request", pending) is True
    assert all(scheduler._compare_group(pending, pending.expected, g, False) == [] for g in pending.blocks.groups)
    result = assess_check(
        "checksum", native_result="passed", expected_units=["group:0", "group:1"],
        compared_units=["group:0", "group:1"], completed=True, evidence_complete=True,
    )
    assert result["coverage"] == "complete" and result["check_status"] == "passed"
    # This is a pure classification control. The fixture manifest still marks
    # every case synthetic and unable to qualify a real release.
    assert MANIFEST["release_qualifying"] is False


def test_vllm_real_comparison_rejects_different_synthetic_checksums(native):
    scheduler, pending = _vllm_case(native, compared_groups={0, 1}, mismatch=True)
    assert scheduler._compare_group(pending, pending.expected, 0, False) == [0]
    assert scheduler._verify("synthetic-request", pending) is False


def test_vllm_skipping_every_group_is_not_native_true(native):
    scheduler, pending = _vllm_case(native, compared_groups=set())
    mismatches, unverified = scheduler._compare(pending)
    assert mismatches == {}
    assert [item.reason for item in unverified] == ["no loaded group can be checksummed"]
    assert scheduler._verify("synthetic-request", pending) is False


def _empty_canary_plan(native):
    cpu = {"device": "cpu"}
    context = SimpleNamespace(
        canary_buf=torch.zeros((1, 32), dtype=torch.uint8, **cpu),
        kernel_kind=native.CanaryLaunchTag.HEAD_K_FULL,
        violation_ring=torch.zeros((1, 8), dtype=torch.int64, **cpu),
        violation_write_index=torch.zeros(1, dtype=torch.int32, **cpu),
        slot_run_counter=torch.zeros(1, dtype=torch.int64, **cpu),
        kernel_run_counter=torch.zeros(1, dtype=torch.int64, **cpu),
        real_kv_sources=(), real_kv_hash_mode=0,
    )
    plan = SimpleNamespace(
        verify_slot_indices=torch.zeros(4, dtype=torch.int64, **cpu),
        verify_num_valid=torch.zeros(1, dtype=torch.int32, **cpu),
        enable=torch.ones(1, dtype=torch.int32, **cpu),
    )
    return context, plan


def _canary_health_checker(native):
    return _bind(SimpleNamespace(
        _prev_counters_host=torch.zeros(len(native.CanaryLaunchTag), dtype=torch.int64, device="cpu"),
        _active_tags=(native.CanaryLaunchTag.HEAD_K_FULL,),
        _config=SimpleNamespace(sweep_interval=0), _is_draft_runner=False,
        _outer_step_counter_getter=lambda: 200,
    ), native, "_postprocess_on_host", "_expected_active_tags_for_health_check")


def test_sglang_original_cpu_canary_counter_rises_with_zero_active_entries(native):
    context, plan = _empty_canary_plan(native)
    assert native.use_torch_reference(context.canary_buf.device) is True
    assert context.kernel_run_counter.item() == 0
    returned = native.launch_canary_verify_kernel_torch_reference(
        context=context, plan=plan, check_verify_expected_token=True,
    )
    assert returned is None
    assert plan.verify_num_valid.item() == 0
    assert context.kernel_run_counter.item() == 1
    assert context.slot_run_counter.item() == 0
    assert context.violation_write_index.item() == 0
    health = _canary_health_checker(native)
    observed = torch.zeros_like(health._prev_counters_host)
    observed[native.CanaryLaunchTag.HEAD_K_FULL.value] = context.kernel_run_counter[0]
    assert health._postprocess_on_host(observed) is None
    result = assess_check(
        "canary", native_result="activity", expected_units=["synthetic-client-request"],
        compared_units=[], completed=True, evidence_complete=True,
        reason="empty verify plan; no request comparison entry",
    )
    assert result["native_result"] == "activity"
    assert result["check_status"] == "unverified"
    assert result["coverage"] == "not_covered"
    assert result["missing_units"] == ["synthetic-client-request"]
    _required_checksum_holds(result)


def test_sglang_native_health_control_rejects_no_counter_progress(native):
    health = _canary_health_checker(native)
    with pytest.raises(RuntimeError, match="kernel_run_counter did not increase"):
        health._postprocess_on_host(torch.zeros_like(health._prev_counters_host))
