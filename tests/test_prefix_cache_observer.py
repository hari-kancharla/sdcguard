"""Pinned native metadata functions on CPU only. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from types import FunctionType

import pytest

from sdcguard import evidence_io as io, prefix_cache_observer as cache
from sdcguard.kv_checksum_observer import _codes, UnsupportedSource
from test_kv_checksum_observer import native, _with_observer

DATA = Path(__file__).parent / "data/prefix_cache_functions/PRIMARY.json"
CONTEXT = dict(run_id="a"*32, packet_sha256="b"*64, arm_id="observer-on", engine_session_id="c"*32)


def classes(root):
    """Unchanged function code; stand-in containers do not qualify a runtime."""
    source = json.loads(DATA.read_bytes())
    code = {}
    for row in source["sources"]:
        raw = row["content"].encode()
        assert io.digest(raw) == row["sha256"]
        assert hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == row["git_blob_sha"]
        path = root / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        code[row["path"]] = dict(_codes(compile(raw, str(path), "exec", dont_inherit=True)))
    Request = type("Request", (), {})
    Stats = type("PrefillStats", (), {})
    Status = IntEnum("RequestStatus", {name: i+1 for i, name in enumerate(cache.STATUSES)})
    Scheduler = type("Scheduler", (), {})
    namespace = dict(Request=Request, RequestStatus=Status, PrefillStats=Stats, Scheduler=Scheduler)
    Request.take_prefill_stats = FunctionType(code[cache.REQUEST]["Request.take_prefill_stats"], namespace)
    Request.num_output_tokens = property(FunctionType(code[cache.REQUEST]["Request.num_output_tokens"], namespace))
    Stats.set = FunctionType(code[cache.STATS]["PrefillStats.set"], namespace)
    for name in ("schedule", "update_from_output"):
        setattr(Scheduler, name, FunctionType(code[cache.SCHEDULER]["Scheduler."+name], namespace))
    return Request, Scheduler, Stats, Status


def record_run(tmp_path, *, local=16, external=15, role="decode", records=100):
    Request, Scheduler, Stats, Status = classes(tmp_path)
    observer = cache.PrefillCacheObserver(**CONTEXT, role=role, max_records=records)
    observer.install_cpu_source_functions(Request, Scheduler, source_root=tmp_path)
    request = Request()
    request.request_id, request.prompt_token_ids, request.num_prompt_tokens = "native-id", list(range(31)), 31
    request.prefill_stats = Stats()
    request.prefill_stats.set(31, local, external)
    request.num_preemptions, request.status, request._output_token_ids = 0, Status.RUNNING, [5]
    return observer, request, Status


def finish(observer, request, Status, *, preemptions=0):
    request.num_preemptions = preemptions
    request._output_token_ids.extend([6, 7])
    request.status = Status.FINISHED_LENGTH_CAPPED
    assert request.take_prefill_stats() is None
    return observer.close()


def test_native_callback_returns_same_stats_and_preserves_native_clear(tmp_path):
    observer, request, status = record_run(tmp_path)
    stats = request.prefill_stats
    assert request.take_prefill_stats() is stats and request.prefill_stats is None
    snapshot = finish(observer, request, status)
    assert snapshot["health"]["complete"] is True
    assert snapshot["qualification"] == "cpu_source_functions"
    records, issues = cache.evaluate_snapshot(snapshot, CONTEXT, "decode")
    assert issues == [] and len(records) == 2
    assert records[1]["stats"] is None
    assert "num_cache_creation_tokens" not in records[0]["stats"]
    usage, issues = cache.request_usage(records, "native-id", 31, cache.prompt_digest(list(range(31))), 3)
    assert issues == []
    assert usage["num_local_cached_tokens"] == 16
    assert usage["num_external_cached_tokens"] == 15
    assert observer.close() == snapshot


@pytest.mark.parametrize("local,external,role", [(0,31,"decode"),(16,15,"decode"),(16,0,"prefill"),(0,0,"prefill")])
def test_native_cache_counts_separate_both_roles(tmp_path, local, external, role):
    observer, request, status = record_run(tmp_path, local=local, external=external, role=role)
    request.take_prefill_stats()
    snapshot = finish(observer, request, status)
    records, issues = cache.evaluate_snapshot(snapshot, CONTEXT, role)
    assert issues == []
    usage, issues = cache.request_usage(records, "native-id", 31, cache.prompt_digest(list(range(31))), 3)
    assert issues == []
    assert usage["num_local_cached_tokens"] == local
    assert usage["num_external_cached_tokens"] == external
    assert usage["num_computed_tokens"] == 31-local-external


def test_later_preemption_cannot_borrow_first_prefill_zero(tmp_path):
    observer, request, status = record_run(tmp_path)
    request.take_prefill_stats()
    snapshot = finish(observer, request, status, preemptions=1)
    records, issues = cache.evaluate_snapshot(snapshot, CONTEXT, "decode")
    assert not issues
    value, issues = cache.request_usage(records, "native-id", 31, cache.prompt_digest(list(range(31))), 3)
    assert value is None and "native_cache_preempted_request_unsupported" in issues


@pytest.mark.parametrize("change", ["missing_terminal","duplicate_stats","wrong_prompt","wrong_usage","wrong_role","negative_count","missing_sequence"])
def test_incomplete_or_contradictory_cache_observations_hold(tmp_path, change):
    observer, request, status = record_run(tmp_path)
    request.take_prefill_stats()
    snapshot = finish(observer, request, status)
    values = snapshot["records"]
    if change == "missing_terminal": values.pop()
    elif change == "duplicate_stats": values[1]["stats"] = deepcopy(values[0]["stats"])
    elif change == "wrong_prompt": values[0]["prompt_sha256"] = "0"*64
    elif change == "wrong_usage": values[-1]["output_tokens"] = 4
    elif change == "wrong_role": snapshot["role"] = "prefill"
    elif change == "negative_count": values[0]["stats"]["num_local_cached_tokens"] = -1
    else: values[1]["sequence"] = 3
    snapshot["budgets"].update(retained_records=len(values), retained_bytes=sum(len(io.json_bytes(r)) for r in values))
    records, issues = cache.evaluate_snapshot(snapshot, CONTEXT, "decode")
    if not issues:
        value, issues = cache.request_usage(records, "native-id", 31, cache.prompt_digest(list(range(31))), 3)
        assert value is None
    assert issues


def test_source_and_loaded_counter_population_are_guarded(tmp_path):
    Request, Scheduler, _, _ = classes(tmp_path)
    Scheduler.schedule = lambda self: None
    observer = cache.PrefillCacheObserver(**CONTEXT, role="decode")
    with pytest.raises(UnsupportedSource, match="loaded prefill callback differs"):
        observer.install_cpu_source_functions(Request, Scheduler, source_root=tmp_path)


def test_budget_exhaustion_keeps_native_results_and_invalidates_capture(tmp_path):
    observer, request, status = record_run(tmp_path, records=1)
    value = request.prefill_stats
    assert request.take_prefill_stats() is value
    snapshot = finish(observer, request, status)
    assert snapshot["health"]["budget_exhausted"] and not snapshot["health"]["complete"]
    assert cache.evaluate_snapshot(snapshot, CONTEXT, "decode")[1]


def test_original_exception_is_not_replaced(tmp_path):
    observer, request, status = record_run(tmp_path)
    del request.prefill_stats
    with pytest.raises(AttributeError, match="prefill_stats"):
        request.take_prefill_stats()
    assert observer.close()["health"]["complete"] is False


def test_foreign_callback_is_preserved_at_close(tmp_path):
    observer, request, status = record_run(tmp_path)
    replacement = lambda self: None
    type(request).take_prefill_stats = replacement
    assert observer.close()["health"]["complete"] is False
    assert type(request).take_prefill_stats is replacement


@pytest.mark.parametrize("tokens,prefix", [(31,16),(32,16),(63,48),(1537,1536)])
def test_real_checksum_function_pass_does_not_cover_local_prefix(native, tokens, prefix):
    _, result, snapshot, record = _with_observer(native, local_tokens=tokens, local_prefix=prefix)
    assert record["native_return"] is True
    assert record["coverage"] == "partial"
    assert record["reason"] == "cached prefix not covered by checksum"
    assert record["token_coverage"] == dict(prompt_tokens=tokens, transferred_tokens_checked=tokens-prefix,
                                            decode_cached_tokens_unchecked=prefix)
    assert len(record["expected_units"]) == (tokens+15)//16
    assert len(record["compared_units"]) == (tokens+15)//16-prefix//16
    assert len(record["missing_units"]) == prefix//16


def test_unaligned_or_unproven_local_prefix_stays_unknown(native):
    _, _, _, record = _with_observer(native, local_tokens=31, local_prefix=1)
    assert record["coverage"] == "unknown"
    assert record["token_coverage"]["transferred_tokens_checked"] is None


def test_cpu_functions_cannot_be_relabelled_as_native_modules(tmp_path):
    Request, Scheduler, Stats, _ = classes(tmp_path)
    for cls, path in ((Request, cache.REQUEST), (Scheduler, cache.SCHEDULER), (Stats, cache.STATS)):
        cls.__module__ = path[:-3].replace("/", ".")
    observer = cache.PrefillCacheObserver(**CONTEXT, role="decode")
    with pytest.raises(UnsupportedSource, match="namespace differs"):
        observer.install(Request, Scheduler, source_root=tmp_path)


@pytest.mark.parametrize("status", ["FINISHED_ERROR", "FINISHED_ABORTED", "FINISHED_IGNORED",
                                     "FINISHED_REPETITION", "PREEMPTED", "WAITING"])
def test_invalid_earlier_lifecycle_cannot_end_in_a_pass(tmp_path, status):
    observer, request, Status = record_run(tmp_path)
    request.status = Status[status]
    request.take_prefill_stats()
    records = finish(observer, request, Status)["records"]
    value, issues = cache.request_usage(records, "native-id", 31, cache.prompt_digest(list(range(31))), 3)
    assert value is None and "native_cache_terminal_request_unverified" in issues


def test_oversized_source_is_refused_before_read(tmp_path, monkeypatch):
    Request, Scheduler, _, _ = classes(tmp_path)
    path = tmp_path / cache.REQUEST
    path.write_bytes(b"x" * (1024 * 1024 + 1))
    original = Path.read_bytes
    def bounded_read(self):
        assert self != path, "oversized source was read"
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    with pytest.raises(UnsupportedSource, match="oversized"):
        cache.PrefillCacheObserver(**CONTEXT, role="decode").install_cpu_source_functions(
            Request, Scheduler, source_root=tmp_path)


def test_unbounded_counter_is_not_serialized(tmp_path, monkeypatch):
    observer, request, _ = record_run(tmp_path)
    request.prefill_stats.num_local_cached_tokens = object()
    original = cache.io.json_bytes
    def scalar_only(value):
        if "stats" in value:
            assert all(type(v) is int for v in value["stats"].values())
        return original(value)
    monkeypatch.setattr(cache.io, "json_bytes", scalar_only)
    result = request.prefill_stats
    assert request.take_prefill_stats() is result
    assert observer.close()["health"]["complete"] is False


def test_full_budget_does_not_copy_next_native_stats(tmp_path):
    observer, request, _ = record_run(tmp_path, records=1)
    request.take_prefill_stats()
    request.prefill_stats = object()
    assert request.take_prefill_stats() is not None
    snapshot = observer.close()
    assert snapshot["health"]["dropped_records"] == 1
    assert snapshot["health"]["errors"] == 0


def test_health_inspection_exception_preserves_native_call_and_uninstall(tmp_path, monkeypatch):
    observer, request, _ = record_run(tmp_path)
    original, stats = observer._original, request.prefill_stats
    def bad_metadata():
        raise OSError("metadata path cannot be inspected")
    monkeypatch.setattr(observer, "_bindings_healthy", bad_metadata)
    assert request.take_prefill_stats() is stats and request.prefill_stats is None
    snapshot = observer.close()
    assert type(request).take_prefill_stats is original
    assert snapshot["health"]["active_calls"] == 0
    assert snapshot["health"]["complete"] is False


@pytest.mark.parametrize("change", ["supported_int", "supported_float", "budget_float"])
def test_saved_cache_snapshot_rejects_numeric_type_coercion(tmp_path, change):
    observer, request, Status = record_run(tmp_path)
    request.take_prefill_stats()
    snapshot = finish(observer, request, Status)
    if change == "budget_float": snapshot["budgets"]["fixed_bytes"] = float(cache.FIXED_BYTES)
    else: snapshot["source"]["supported"] = 1 if change == "supported_int" else 1.0
    assert cache.evaluate_snapshot(snapshot, CONTEXT, "decode")[1]
