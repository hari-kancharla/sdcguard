"""Original native callbacks with synthetic CPU metadata. Owner: Hari Krishna Kancharla.

No server, worker, device checksum or real request is run. These controls cannot
qualify a deployment. The producer payload is assembled as CPU metadata, so its
wire serialization and arrival route are not claimed.
"""
from __future__ import annotations

import ast
import base64
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from enum import IntEnum
from fractions import Fraction
from functools import cached_property
import hashlib
import inspect
import json
import logging
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from sdcguard.kv_checksum_observer import ChecksumObserver


DATA = Path(__file__).parent / "data/checksum_observer_functions"
PRIMARY = json.loads((DATA / "PRIMARY.json").read_bytes())
SOURCES = {row["path"]: row for row in PRIMARY["sources"]}
PREFIX = "vllm/distributed/kv_transfer/kv_checksum/"
SCHEDULER = PREFIX + "scheduler.py"
TYPES = PREFIX + "checksum.py"
INTERFACE = "vllm/v1/kv_cache_interface.py"
MATH = "vllm/utils/math_utils.py"


def _tree(root):
    for row in PRIMARY["sources"]:
        raw = row["content"].encode()
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["git_blob_sha"]
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def _execute(monkeypatch, root, path, module_name, names, **namespace):
    """Build unchanged definitions and bind exact whole-source function code."""
    source = SOURCES[path]["content"]
    parsed = ast.parse(source, filename=str(root / path))
    selected = [node for node in parsed.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    module = ModuleType(module_name)
    module.__file__ = str(root / path)
    module.__dict__.update(namespace)
    monkeypatch.setitem(sys.modules, module_name, module)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    ast.copy_location(future, parsed.body[0])
    code = ast.fix_missing_locations(ast.Module(body=[future, *deepcopy(selected)], type_ignores=[]))
    exec(compile(code, module.__file__, "exec", dont_inherit=True), vars(module))
    # Import declarations influence attribute-call bytecode in Python 3.13.
    # Compile the full original file without executing its imports, then use
    # those exact code objects with the explicit CPU namespace and closures.
    whole = compile(source, module.__file__, "exec", dont_inherit=True)
    originals = {}
    def collect(parent, prefix=""):
        for child in parent.co_consts:
            if isinstance(child, type(parent)):
                qualified = prefix + child.co_name
                originals[(qualified, child.co_firstlineno)] = child
                collect(child, qualified + ".")
    collect(whole)
    def bind(function, qualified):
        if not inspect.isfunction(function):
            return
        original = originals.get((qualified, function.__code__.co_firstlineno))
        if original is None:
            assert function.__code__.co_filename != module.__file__
            return  # Dataclass and enum helpers have no original source body.
        assert function.__code__.co_freevars == original.co_freevars
        function.__code__ = original
        assert function.__code__ is original
    for node in selected:
        value = vars(module)[node.name]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bind(value, node.name)
        elif isinstance(node, ast.ClassDef):
            for name, member in vars(value).items():
                qualified = node.name + "." + name
                if isinstance(member, (classmethod, staticmethod)):
                    bind(member.__func__, qualified)
                elif isinstance(member, property):
                    for accessor in (member.fget, member.fset, member.fdel):
                        bind(accessor, qualified)
                elif isinstance(member, cached_property):
                    bind(member.func, qualified)
                else:
                    bind(member, qualified)
    return module


@pytest.fixture
def native(monkeypatch, tmp_path):
    root = tmp_path / "pinned-source"
    _tree(root)
    logger = logging.Logger("checksum_observer_cpu_fixture")
    logger.warning_once = logger.warning
    math = _execute(monkeypatch, root, MATH, "vllm.utils.math_utils", {"cdiv"})
    interface = _execute(
        monkeypatch, root, INTERFACE, "vllm.v1.kv_cache_interface",
        {"KVQuantMode", "KVCacheSpec", "AttentionSpec", "FullAttentionSpec", "sliding_window_transfer_blocks"},
        dataclass=dataclass, field=field, fields=fields, replace=replace,
        IntEnum=IntEnum, Fraction=Fraction, cached_property=cached_property,
        torch=torch, cdiv=math.cdiv,
        # Only full attention executes. This preserves the original helper's
        # isinstance boundary without claiming a sliding-window implementation.
        SlidingWindowSpec=type("UnusedSlidingWindowSpec", (), {}),
    )
    types = _execute(
        monkeypatch, root, TYPES, "vllm.distributed.kv_transfer.kv_checksum.checksum",
        {"KVChecksumBlocks", "KVChecksumScheduled", "KVChecksumGroup", "KVChecksumRecord", "num_valid_tokens"},
        dataclass=dataclass, field=field, np=np, ReplicaKey=tuple[int, int],
    )
    scheduler = _execute(
        monkeypatch, root, SCHEDULER, "vllm.distributed.kv_transfer.kv_checksum.scheduler",
        {"KVChecksumPayload", "_Unverified", "_Pending", "KVChecksumScheduler",
         "_non_null_run", "_chunk_sums", "_sum_replicas"},
        dataclass=dataclass, field=field, np=np, base64=base64, logger=logger,
        KVChecksumBlocks=types.KVChecksumBlocks,
        KVChecksumGroup=types.KVChecksumGroup,
        KVChecksumRecord=types.KVChecksumRecord,
        KVChecksumScheduled=types.KVChecksumScheduled,
        num_valid_tokens=types.num_valid_tokens,
        cdiv=math.cdiv, ReplicaKey=tuple[int, int],
        KV_TRANSFER_PARAMS_KEY="kv_checksums", _PAYLOAD_VERSION=1,
        # Connector output and unused model-kind types are metadata stand-ins.
        CacheHitSource=SimpleNamespace(P2P="synthetic_p2p"),
        MambaSpec=type("UnusedMambaSpec", (), {}),
        sliding_window_transfer_blocks=interface.sliding_window_transfer_blocks,
    )
    return SimpleNamespace(root=root, interface=interface, types=types, scheduler=scheduler, math=math)


def _case(native, *, groups=1, local_tokens=32, expected_tokens=None,
          producer_start=0, producer_block_size=16, local_prefix=0,
          empty_replicas=False, skipped=(), request_id="synthetic-ordinary-request",
          fingerprint_base=123):
    """The original constructor and _blocks select the ordinary full load."""
    spec = native.interface.FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16,
        dcp_sharded=False,
    )
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(enable_kv_checksum=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                                       decode_context_parallel_size=1, prefill_context_parallel_size=1,
                                       data_parallel_size=1),
        speculative_config=None,
    )
    cache = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec) for _ in range(groups)],
        transfer_group_ids=list(range(groups)),
    )
    connector = SimpleNamespace(get_external_cache_hit_sources=lambda *_: {native.scheduler.CacheHitSource.P2P: local_tokens})
    engine = native.scheduler.KVChecksumScheduler(config, cache, connector)
    request = SimpleNamespace(
        request_id=request_id, num_computed_tokens=local_tokens,
        num_prompt_tokens=local_tokens, kv_transfer_params={"do_remote_prefill": True},
    )
    blocks = SimpleNamespace(blocks=[
        [SimpleNamespace(block_id=101 + 100 * group + pos, is_null=False)
         for pos in range((local_tokens + 15) // 16)] for group in range(groups)
    ])
    engine.on_new_request(request)
    engine.add_recv(request, blocks, local_prefix)
    pending = engine._recvs[request.request_id]
    values, producer, fingerprints = {}, {}, {}
    expected_tokens = local_tokens if expected_tokens is None else expected_tokens
    for group_id, (start, block_ids) in pending.blocks.groups.items():
        fingerprints[group_id] = fingerprint_base + group_id
        if group_id in skipped:
            continue
        if empty_replicas:
            local = np.empty((len(block_ids), 0), dtype=np.uint64)
            replica_keys = []
            sent = np.zeros((expected_tokens + producer_block_size - 1) // producer_block_size - producer_start, dtype=np.uint64)
        else:
            local = np.array([[7 + start + i + group_id] for i in range(len(block_ids))], dtype=np.uint64)
            replica_keys = [(1, 0)]
            if producer_block_size == 16:
                sent = np.array([7 + i + group_id for i in range(producer_start, (expected_tokens + 15) // 16)], dtype=np.uint64)
            else:
                assert producer_block_size == 32 and expected_tokens == 32 and producer_start == 0
                sent = np.array([15 + 2 * group_id], dtype=np.uint64)
        values[group_id] = native.types.KVChecksumGroup(start, replica_keys, local)
        producer[group_id] = (fingerprint_base + group_id, producer_block_size, producer_start, sent)
    pending.expected = native.scheduler.KVChecksumPayload(1, expected_tokens, producer)
    record = native.types.KVChecksumRecord(
        pp_rank=0, tp_rank=0, checksums={request.request_id: values},
        skipped_groups=frozenset(skipped), fingerprints=fingerprints,
    )
    output = SimpleNamespace(kv_checksums=[record], finished_recving={request.request_id})
    return SimpleNamespace(engine=engine, request=request, blocks=blocks, pending=pending,
                           output=output, local_prefix=local_prefix)


def _observer(native, **kwargs):
    observer = ChecksumObserver(**kwargs)
    observer.install_cpu_source_functions(native.scheduler.KVChecksumScheduler, source_root=native.root)
    return observer


def _verification(snapshot):
    rows = [row for row in snapshot["records"] if row["kind"] == "verification"]
    assert len(rows) == 1
    return rows[0]


def _with_observer(native, **case_options):
    observer = _observer(native)
    case = _case(native, **case_options)
    result = case.engine.update_from_output(case.output, set())
    snapshot = observer.close()
    return case, result, snapshot, _verification(snapshot)


@pytest.fixture
def closed_cpu_snapshot(native):
    """A closed result from original callbacks, always CPU-source-qualified."""
    return _with_observer(native)[2]


@pytest.fixture
def partial_span_cpu_snapshot(native):
    """The native comparison succeeds on an intersected subset of the load."""
    return _with_observer(native, producer_start=1,
                          request_id="synthetic-partial-span-request")[2]


@pytest.fixture
def empty_replicas_cpu_snapshot(native):
    """The native comparison succeeds with zero represented replica inputs."""
    return _with_observer(native, empty_replicas=True,
                          request_id="synthetic-empty-replica-request")[2]


def test_primary_bytes_and_original_ast_fixture_scope_are_explicit():
    assert PRIMARY["evidence_origin"] == "synthetic"
    assert PRIMARY["runtime_qualifying"] is False
    assert PRIMARY["full_engine_execution"] is False
    assert len(PRIMARY["sources"]) == 7
    for source in PRIMARY["sources"]:
        raw = source["content"].encode()
        assert len(raw) == source["bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == source["git_blob_sha"]


def test_original_callback_sequence_records_a_complete_cpu_comparison(native):
    case, result, snapshot, row = _with_observer(native)
    assert result == set()
    assert snapshot["closed"] is True and snapshot["health"]["complete"] is True
    assert snapshot["qualification"] == "cpu_source_functions"
    assert snapshot["release_qualifying"] is False
    assert row["native_return"] is True and row["native_result"] == "passed"
    assert row["coverage"] == "complete"
    assert row["expected_units"] == ["group:0:chunk:0", "group:0:chunk:1"]
    assert row["compared_units"] == row["expected_units"]
    assert row["missing_units"] == []
    assert row["request_id"] == case.request.request_id
    assert row["origin_update_sequence"] is not None


@pytest.mark.parametrize("fingerprint", [2**63 + 7, 2**64 - 1])
def test_valid_upper_half_uint64_fingerprint_does_not_cause_a_false_hold(native, fingerprint):
    _, result, snapshot, row = _with_observer(native, fingerprint_base=fingerprint)
    assert result == set() and row["native_return"] is True
    assert row["coverage"] == "complete"
    assert snapshot["qualification"] == "cpu_source_functions"
    assert snapshot["release_qualifying"] is False


@pytest.mark.parametrize("options,reason_fragment", [
    ({"producer_start": 1}, "scope"),
    ({"expected_tokens": 24}, "token"),
    ({"producer_block_size": 32}, "block"),
    ({"local_prefix": 16}, "prefix"),
])
def test_native_success_does_not_hide_an_unsupported_compared_interval(native, options, reason_fragment):
    _, result, _, row = _with_observer(native, **options)
    assert result == set()
    assert row["native_return"] is True
    assert row["coverage"] != "complete"
    if "local_prefix" in options:
        assert row["compared_units"] == ["group:0:chunk:1"]
        assert row["missing_units"] == ["group:0:chunk:0"]
    else:
        assert row["compared_units"] is None
        assert row["missing_units"] is None
    assert any(reason_fragment in reason for reason in row["reasons"])
    if "producer_start" in options or "expected_tokens" in options:
        assert "partial_checksum_span" in row["reasons"]
        assert row["expected_units"] == ["group:0:chunk:0", "group:0:chunk:1"]
        _required_checksum_holds(row)


def test_native_success_with_a_skipped_transfer_group_is_not_complete(native):
    _, result, _, row = _with_observer(native, groups=2, skipped={1})
    assert result == set() and row["native_return"] is True
    assert row["coverage"] != "complete"
    assert {g["group_id"]: g["native_result"] for g in row["groups"]} == {0: "passed", 1: "not_compared"}


def test_empty_replica_columns_cannot_be_complete_coverage(native):
    _, result, _, row = _with_observer(native, empty_replicas=True)
    assert result == set() and row["native_return"] is True
    assert row["coverage"] != "complete"
    assert row["compared_units"] is None
    assert row["missing_units"] is None
    assert row["expected_units"] == ["group:0:chunk:0", "group:0:chunk:1"]
    assert "empty_replica_inputs" in row["reasons"]
    _required_checksum_holds(row)


def _required_checksum_holds(row):
    from sdcguard.required_gate import decide
    check = dict(native_result="passed", expected_units=row["expected_units"], compared_units=row["compared_units"],
        completed=True, evidence_complete=row["coverage"] == "complete", applicable=True, reason=row["reasons"][0])
    result = decide([dict(client_request_id=row["request_id"], checks={"checksum": check})], ["checksum"])
    assert result["decision"] == "HOLD" and result["requests"][0]["reasons"]


def test_worker_report_without_finished_load_does_not_become_verification(native):
    observer = _observer(native)
    case = _case(native)
    case.output.finished_recving = set()
    assert case.engine.update_from_output(case.output, set()) == set()
    snapshot = observer.close()
    assert not [r for r in snapshot["records"] if r["kind"] == "verification"]
    assert any(r["kind"] == "recv_plan" for r in snapshot["records"])
    assert all(r.get("coverage") != "complete" for r in snapshot["records"])


def test_direct_native_verify_without_update_context_is_not_closed_load_evidence(native):
    observer = _observer(native)
    case = _case(native)
    record = case.output.kv_checksums[0]
    case.pending.reports[(0, 0)] = (record.checksums[case.request.request_id], record)
    assert case.engine._verify(case.request.request_id, case.pending) is True
    row = _verification(observer.close())
    assert row["native_return"] is True and row["coverage"] != "complete"
    assert row["origin_update_sequence"] is None


@pytest.mark.parametrize("target", ["helper", "callback"])
def test_loaded_code_replacement_preserves_native_success_but_loses_coverage(native, monkeypatch, target):
    observer = _observer(native)
    case = _case(native)
    if target == "helper":
        original = native.scheduler._sum_replicas
        def forwarding(*args, **kwargs):
            return original(*args, **kwargs)
        monkeypatch.setattr(native.scheduler, "_sum_replicas", forwarding)
    else:
        original = native.scheduler.KVChecksumScheduler._compare_group
        def forwarding(*args, **kwargs):
            return original(*args, **kwargs)
        monkeypatch.setattr(native.scheduler.KVChecksumScheduler, "_compare_group", forwarding)
    assert case.engine.update_from_output(case.output, set()) == set()
    snapshot = observer.close()
    assert all(r.get("coverage") != "complete" for r in snapshot["records"])
    assert snapshot["health"]["complete"] is False or snapshot["source"]["supported"] is False


def test_prefill_not_selected_for_p2p_has_no_invented_comparison(native):
    observer = _observer(native)
    case = _case(native)
    # A new independent ordinary request has no producer-to-consumer source.
    request = SimpleNamespace(request_id="synthetic-cache-tier-request", num_computed_tokens=32,
                              num_prompt_tokens=32, kv_transfer_params={})
    assert case.engine.on_new_request(request) is None
    assert case.engine.add_recv(request, case.blocks, 0) is None
    snapshot = observer.close()
    rows = [r for r in snapshot["records"] if r.get("request_id") == request.request_id]
    assert rows and all(r.get("coverage") != "complete" for r in rows)


@contextmanager
def _native_calls(native):
    watched = {}
    for obj in (native.scheduler.KVChecksumScheduler, native.scheduler._Pending):
        for name, function in vars(obj).items():
            function = inspect.unwrap(function)
            if hasattr(function, "__code__"):
                watched[function.__code__] = obj.__name__ + "." + name
    for name in ("_chunk_sums", "_sum_replicas", "_non_null_run", "num_valid_tokens",
                 "cdiv", "sliding_window_transfer_blocks"):
        watched[getattr(native.scheduler, name).__code__] = name
    counts = Counter()
    returns = {}
    arguments = {}
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == "call" and frame.f_code in watched:
            counts[watched[frame.f_code]] += 1
            arguments.setdefault(watched[frame.f_code], []).append(dict(frame.f_locals))
        if event == "return" and frame.f_code in watched:
            returns.setdefault(watched[frame.f_code], []).append(arg)
        if previous is not None:
            previous(frame, event, arg)
    sys.setprofile(profile)
    try:
        yield SimpleNamespace(counts=counts, returns=returns, arguments=arguments)
    finally:
        sys.setprofile(previous)


def test_observer_does_not_add_native_comparisons_or_change_return_identity(native):
    baseline = _case(native)
    with _native_calls(native) as native_only:
        returned = baseline.engine.update_from_output(baseline.output, set())
    observer = _observer(native)
    case = _case(native)
    skip_ids = set()
    sent_arrays = {g: values[3].copy() for g, values in case.pending.expected.groups.items()}
    local_arrays = {g: values.values.copy() for g, values in case.output.kv_checksums[0].checksums[case.request.request_id].items()}
    with _native_calls(native) as observed:
        result = case.engine.update_from_output(case.output, skip_ids)
    assert observed.counts == native_only.counts
    assert result == returned == set()
    assert result is observed.returns["KVChecksumScheduler.update_from_output"][0]
    assert returned is native_only.returns["KVChecksumScheduler.update_from_output"][0]
    assert observed.counts["KVChecksumScheduler._verify"] == 1
    assert observed.counts["KVChecksumScheduler._compare_group"] == 1
    assert observed.counts["_chunk_sums"] == 2
    entered = observed.arguments["KVChecksumScheduler.update_from_output"][0]
    assert entered["self"] is case.engine
    assert entered["kv_connector_output"] is case.output
    assert entered["skip_req_ids"] is skip_ids
    assert observed.arguments["KVChecksumScheduler._verify"][0]["pending"] is case.pending
    assert observed.arguments["KVChecksumScheduler._compare_group"][0]["expected"] is case.pending.expected
    for group_id, values in case.pending.expected.groups.items():
        assert np.array_equal(values[3], sent_arrays[group_id])
    for group_id, values in case.output.kv_checksums[0].checksums[case.request.request_id].items():
        assert np.array_equal(values.values, local_arrays[group_id])
    _verification(observer.close())


def test_original_metadata_exception_is_preserved_by_identity(native):
    class MetadataError(Exception):
        pass
    sentinel = MetadataError("synthetic metadata accessor")
    class Request:
        @property
        def kv_transfer_params(self):
            raise sentinel
    observer = _observer(native)
    case = _case(native)
    with pytest.raises(MetadataError) as caught:
        case.engine.on_new_request(Request())
    assert caught.value is sentinel
    observer.close()


def test_close_during_cpu_metadata_callback_never_reports_complete(native):
    """Coordinate only a recorder callback, with no engine or transfer running."""
    observer = _observer(native)
    case = _case(native)
    entered, release = threading.Event(), threading.Event()
    results, errors = [], []
    class Request:
        request_id = "synthetic-close-metadata"
        @property
        def kv_transfer_params(self):
            entered.set()
            # The deadline cannot force a native error. The test itself records
            # a timeout as failure and always releases and joins this thread.
            release.wait(timeout=10)
            return {}
    def call():
        try:
            results.append(case.engine.on_new_request(Request()))
        except BaseException as exc:
            errors.append(exc)
    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    try:
        assert entered.wait(timeout=5), "CPU metadata callback did not enter"
        snapshot = observer.close()
        assert snapshot["health"]["active_calls"] > 0
        assert snapshot["health"]["complete"] is False
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive(), "CPU metadata callback did not finish"
    assert results == [None] and errors == []
    assert observer.snapshot()["health"]["complete"] is False


def test_record_budget_loss_keeps_native_output_and_poisons_coverage(native):
    observer = _observer(native, max_records=1)
    case = _case(native)
    assert case.engine.update_from_output(case.output, set()) == set()
    snapshot = observer.close()
    assert snapshot["health"]["dropped_records"] > 0
    assert snapshot["health"]["complete"] is False
    assert all(r.get("coverage") != "complete" for r in snapshot["records"])


def test_unit_budget_cannot_truncate_into_complete_coverage(native):
    observer = _observer(native, max_units=1)
    case = _case(native)
    assert case.engine.update_from_output(case.output, set()) == set()
    snapshot = observer.close()
    assert all(r.get("coverage") != "complete" for r in snapshot["records"])
    assert any("budget" in str(r.get("reasons", [])) for r in snapshot["records"])


def test_global_unit_budget_cannot_reexpand_later_ordinary_requests_on_close(native):
    observer = _observer(native, max_units=4)
    for request_id in ("synthetic-first-ordinary-request", "synthetic-second-ordinary-request"):
        case = _case(native, request_id=request_id)
        assert case.engine.update_from_output(case.output, set()) == set()
    snapshot = observer.close()
    records = [row for row in snapshot["records"] if row["kind"] == "verification"]
    assert records
    assert len({row["request_id"] for row in records}) == len(records)
    assert snapshot["health"]["complete"] is False
    assert snapshot["health"]["budget_exhausted"] is True
    assert snapshot["health"]["dropped_records"] > 0 or len(records) == 2
    assert not all(row["coverage"] == "complete" for row in records)
    expanded = sum(len(row.get(name) or ()) for row in records
                   for name in ("expected_units", "compared_units", "missing_units"))
    assert expanded <= 4
    assert any("budget" in str(row.get("reasons", [])) for row in records)


def test_inflight_position_reservation_is_shared_across_recorder_contexts():
    """A scalar-list capacity control; no checksum or engine function runs."""
    observer = ChecksumObserver(max_units=3)
    first = {"position_units": 0, "position_bytes": 0}
    second = {"position_units": 0, "position_bytes": 0}
    original = [0, 1]
    try:
        copied = observer._copy_positions(first, original)
        assert copied == original and copied is not original
        assert observer._copy_positions(second, [2, 3]) is None
        snapshot = observer.snapshot()
        assert snapshot["health"]["inflight_position_units"] <= 3
        assert snapshot["health"]["budget_exhausted"] is True
        assert snapshot["health"]["complete"] is False
        assert original == [0, 1]
    finally:
        observer._release_positions(first)
        observer._release_positions(second)
    assert observer.snapshot()["health"]["inflight_position_units"] == 0
    assert observer.snapshot()["health"]["inflight_position_bytes"] == 0
    observer.close()


def test_open_capture_has_no_closed_coverage_qualification(native):
    observer = _observer(native)
    case = _case(native)
    assert case.engine.update_from_output(case.output, set()) == set()
    open_row = _verification(observer.snapshot())
    assert open_row["coverage"] != "complete"
    assert open_row["compared_units"] == open_row["expected_units"]
    assert len(open_row["compared_units"]) == 2
    assert open_row["missing_units"] == []
    assert _verification(observer.close())["coverage"] == "complete"


def test_closing_twice_preserves_the_same_closed_capture(native):
    observer = _observer(native)
    case = _case(native)
    assert case.engine.update_from_output(case.output, set()) == set()
    first = observer.close()
    second = observer.close()
    assert second == first
    assert second["health"]["complete"] is True
    assert _verification(second)["coverage"] == "complete"


def test_source_change_refuses_hook_installation_without_mutating_class(native):
    method = native.scheduler.KVChecksumScheduler.update_from_output
    path = native.root / SCHEDULER
    path.write_text(path.read_text() + "\n# CPU source mismatch control\n")
    observer = ChecksumObserver()
    with pytest.raises(ValueError, match="source"):
        observer.install_cpu_source_functions(native.scheduler.KVChecksumScheduler, source_root=native.root)
    snapshot = observer.close()
    assert snapshot["source"]["supported"] is False
    assert native.scheduler.KVChecksumScheduler.update_from_output is method


def test_uninstall_restores_original_native_methods(native):
    cls = native.scheduler.KVChecksumScheduler
    before = {n: getattr(cls, n) for n in ("on_new_request", "add_recv", "update_from_output", "_verify", "_compare", "_compare_group")}
    observer = _observer(native)
    observer.uninstall()
    assert all(getattr(cls, name) is function for name, function in before.items())
    snapshot = observer.close()
    assert snapshot["source"]["supported"] is True
    assert snapshot["health"]["complete"] is True
