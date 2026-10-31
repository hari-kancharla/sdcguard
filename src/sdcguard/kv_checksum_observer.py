"""Passive observations of the pinned checksum scheduler.

Owner: Hari Krishna Kancharla.

This module imports no engine or device library. Installation is explicit. A
completed native comparison and its source-bound chunk scope are separate
observations. Neither is an ownership, byte-read or release certificate.
"""
from __future__ import annotations

import __future__
import functools
import hashlib
import inspect
import json
from importlib.resources import files
from pathlib import Path
import threading
import types
import sys
import weakref

SCHEMA = "sdcguard.checksum-observer.v1"
PROFILE = "vllm-checksum-62211-full-load-tp1-pp1.v1"
REVISION = "29d2f4e03740acb14439ce22d24b06d8095d17d3"
SCHEDULER = "vllm/distributed/kv_transfer/kv_checksum/scheduler.py"
TYPES = "vllm/distributed/kv_transfer/kv_checksum/checksum.py"
CACHE = "vllm/v1/kv_cache_interface.py"
MATH = "vllm/utils/math_utils.py"
METHODS = ("__init__", "on_new_request", "add_recv", "update_from_output", "_verify", "_compare", "_compare_group")
MAX_GROUPS = 16
CACHED_PREFIX_REASON = "cached prefix not covered by checksum"
STATE_ENTRY_BYTES = 8192
FIXED_METADATA_BYTES = 16384
LIMITS = [
    "Units are native checksum chunk positions, not proof every model layer or byte was checksummed.",
    "Client inventory, router forwarding, ownership, read rights and physical reads remain unknown.",
    "Named source files and loaded callbacks are checked; full immutable build and authenticity are unknown.",
    "This source profile is separate from the historical lease build and cannot qualify both checks.",
    "CPU source-function execution cannot qualify a deployment. GPU execution and overhead are unverified.",
]


class UnsupportedSource(ValueError):
    pass


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= 2**63 - 1


def _text(value):
    return type(value) is str and 0 < len(value) <= 1024


def _uint64(value):
    return type(value) is int and 0 <= value < 2**64


def _plain(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def source_pins():
    return json.loads(files("sdcguard").joinpath("case_data/checksum-observer-pins.json").read_bytes())


def source_profile():
    pins = source_pins()
    return {"profile": PROFILE, "revision": REVISION,
            "files": {name: receipt["sha256"] for name, receipt in pins["files"].items()}}


def _shape(value, expected):
    return (type(value) is list and len(value) == len(expected)
            and all(type(x) is int and x == y for x, y in zip(value, expected)))


def _replicas(value):
    return type(value) is list and len(value) == 1 and _shape(value[0], [1, 0])


def _engine_pass_gap_reasons(record, proof):
    """Name source-observed gaps without treating a native pass as full scope."""
    if (record.get("native_return") is not True or proof.get("source_supported") is not True
            or proof.get("metadata_stable") is not True or record.get("engine_exception") is not None
            or record.get("unverified_reasons") != []):
        return []
    groups, witnesses = proof.get("groups"), record.get("groups")
    if (type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict
            or type(witnesses) is not list or len(witnesses) != 1 or type(witnesses[0]) is not dict):
        return []
    group, witness = groups[0], witnesses[0]
    if (group.get("compare_called") is not True or group.get("native_group_result") != []
            or group.get("native_group_error") is not None or witness.get("native_result") != "passed"
            or witness.get("returned_positions") != [] or witness.get("reason") is not None
            or not _integer(group.get("group_id")) or type(witness.get("group_id")) is not int
            or witness["group_id"] != group["group_id"]):
        return []
    count, reports = group.get("block_count"), group.get("consumer_reports")
    reasons = []
    if _integer(count, 1) and type(reports) is list and len(reports) == 1 and type(reports[0]) is dict:
        report = reports[0]
        if (report.get("replica_keys") == [] and type(report.get("replica_keys")) is list
                and _shape(report.get("value_shape"), [count, 0])):
            reasons.append("empty_replica_inputs")
    bs, sent_bs = group.get("block_size"), group.get("sent_block_size")
    start, sent_start, sent_count = group.get("start"), group.get("sent_start"), group.get("sent_count")
    nt, et = proof.get("num_tokens"), proof.get("expected_num_tokens")
    if (group.get("spec_type") != "FullAttentionSpec" or group.get("spec_source_supported") is not True
            or not _integer(bs, 1) or type(sent_bs) is not int or bs != sent_bs
            or not _integer(start) or not _integer(sent_start) or not _integer(count, 1)
            or not _integer(sent_count, 1) or not _integer(nt, 1) or not _integer(et, 1)
            or count != (nt + bs - 1) // bs - start
            or sent_count != (et + bs - 1) // bs - sent_start
            or not _shape(group.get("sent_shape"), [sent_count])):
        return reasons
    first, end = max(start, sent_start), min(start + count, sent_start + sent_count)
    if first < end:
        partial = first > start or end < start + count
        # Only the final token chunk on either side can have differing counts.
        for position in {(nt - 1) // bs, (et - 1) // bs}:
            if first <= position < end:
                local = min(bs, max(0, nt - position * bs))
                remote = min(bs, max(0, et - position * bs))
                partial = partial or local != remote
        if partial:
            reasons.append("partial_checksum_span")
    return reasons


def classify_verification(record, *, max_units=65536):
    """Re-derive conditional chunk scope from retained primitive observations.

    This function validates scope metadata, not its authenticity or an actual
    runtime. A complete result cannot qualify a release on its own.
    """
    if not _integer(max_units, 1) or max_units > 1048576:
        raise ValueError("max_units must be between 1 and 1048576")
    if type(record) is not dict:
        record = {}
    native = record.get("native_return")
    native = native if type(native) is bool else None
    result = {"native_return": native, "native_result": "passed" if native is True else "unverified", "expected_units": None,
              "compared_units": None, "missing_units": None, "coverage": "unknown",
              "reason": "scope_metadata_unverified", "reasons": [],
              "token_coverage": {"prompt_tokens": None, "transferred_tokens_checked": None,
                                 "decode_cached_tokens_unchecked": None}}
    reasons = []
    proof = record.get("proof")
    proof = proof if type(proof) is dict else {}
    if proof.get("scope_budget_refused") is True:
        reasons.append("scope_budget_refused")
    groups = proof.get("groups")
    groups = groups if type(groups) is list else []
    unverified = record.get("unverified_reasons", [])
    if (type(unverified) is not list or len(unverified) > MAX_GROUPS
            or any(type(x) is not str or len(x) > 512 for x in unverified)):
        reasons.append("native_unverified_reason_shape_invalid")
    elif unverified:
        reasons.append("native_comparison_unverified")
    if record.get("engine_exception") is not None:
        reasons.append("native_engine_exception_present")
    path = proof.get("call_path")
    if (path != ["on_new_request", "add_recv", "update_from_output", "_verify", "_compare_group"]
            or not _integer(record.get("origin_update_sequence"), 1)
            or not _integer(proof.get("plan_sequence"), 1)):
        reasons.append("native_call_path_unverified")
    if proof.get("source_supported") is not True:
        reasons.append("source_profile_unverified")
    if proof.get("metadata_stable") is not True:
        reasons.append("scope_metadata_changed_or_unobserved")
    if (type(proof.get("parallel")) is not dict or any(
            not _integer(proof["parallel"].get(k), 1) or proof["parallel"][k] != 1
            for k in ("tp", "pp", "dp", "dcp", "pcp"))
            or proof.get("num_workers") != 1 or type(proof.get("num_workers")) is not int
            or proof.get("native_tp_size") != 1 or type(proof.get("native_tp_size")) is not int):
        reasons.append("parallel_scope_unsupported")
    if proof.get("speculative_disabled") is not True:
        reasons.append("speculative_configuration_unknown_or_enabled")
    if proof.get("producer_payload_present") is not True:
        reasons.append("producer_checksum_payload_missing_or_unverified")
    local = proof.get("num_local_computed_tokens")
    nt, et = proof.get("num_tokens"), proof.get("expected_num_tokens")
    if not _integer(local) or not _integer(nt, 1) or local >= nt:
        reasons.append("local_prefix_scope_unsupported")
    if not _integer(nt, 1) or type(et) is not int or nt != et:
        reasons.append("producer_consumer_token_scope_differs")
    if (proof.get("expected_tp_size") != 1 or type(proof.get("expected_tp_size")) is not int
            or proof.get("planned_group_count") != 1 or type(proof.get("planned_group_count")) is not int
            or proof.get("configured_group_count") != 1 or type(proof.get("configured_group_count")) is not int
            or len(groups) != 1):
        reasons.append("group_scope_unsupported")
    ranks = proof.get("worker_ranks")
    if not (type(ranks) is list and len(ranks) == 1 and _shape(ranks[0], [0, 0])
            and type(proof.get("worker_report_count")) is int and proof["worker_report_count"] == 1):
        reasons.append("worker_rank_scope_unverified")
    expected, compared = [], None
    comparison_scope_established = False
    if len(groups) == 1 and type(groups[0]) is dict:
        group = groups[0]
        gid, bs, count = group.get("group_id"), group.get("block_size"), group.get("block_count")
        if (proof.get("scope_budget_refused") is True
                or not _integer(gid) or not _integer(bs, 1) or not _integer(count, 1)
                or count > max_units // 2):
            reasons.append("chunk_scope_budget_or_shape_invalid")
        else:
            full_count = (nt + bs - 1) // bs if _integer(nt, 1) else None
            coordinate_domain = (
                group.get("spec_type") == "FullAttentionSpec" and group.get("spec_source_supported") is True
                and _integer(local) and _integer(nt, 1) and local < nt and local % bs == 0
                and type(group.get("start")) is int and group["start"] == local // bs
                and type(group.get("sent_block_size")) is int and group["sent_block_size"] == bs
                and full_count <= max_units // 2
            )
            if coordinate_domain:
                expected = ["group:" + str(gid) + ":chunk:" + str(i) for i in range(full_count)]
            else:
                reasons.append("local_prefix_coordinate_scope_unverified")
            if (group.get("spec_type") != "FullAttentionSpec" or group.get("spec_source_supported") is not True
                    or group.get("sliding_window") is not None
                    or group.get("attention_chunk_size") is not None or group.get("non_causal") is not False
                    or group.get("spec_fields_recorded") is not True):
                reasons.append("attention_group_scope_unsupported")
            if (not _shape(proof.get("configured_group_ids"), [gid])
                    or not _shape(proof.get("expected_group_ids"), [gid])):
                reasons.append("group_identity_scope_unverified")
            if (not coordinate_domain
                    or type(group.get("sent_start")) is not int or group["sent_start"] != 0
                    or type(group.get("sent_block_size")) is not int or group["sent_block_size"] != bs
                    or type(group.get("sent_count")) is not int or group["sent_count"] != full_count
                    or group.get("sent_dtype") != "uint64" or not _shape(group.get("sent_shape"), [full_count])
                    or not _integer(nt, 1) or count != full_count - group["start"]):
                reasons.append("full_load_block_scope_unverified")
            reports = group.get("consumer_reports")
            if type(reports) is not list or len(reports) != 1 or type(reports[0]) is not dict:
                reasons.append("consumer_worker_scope_unverified")
            else:
                report = reports[0]
                if (type(report.get("pp_rank")) is not int or report["pp_rank"] != 0
                        or type(report.get("tp_rank")) is not int or report["tp_rank"] != 0
                        or type(report.get("record_pp_rank")) is not int or report["record_pp_rank"] != 0
                        or type(report.get("record_tp_rank")) is not int or report["record_tp_rank"] != 0
                        or type(report.get("group_start")) is not int or report["group_start"] != group["start"]
                        or not _shape(report.get("value_shape"), [count, 1]) or report.get("value_dtype") != "uint64"
                        or not _replicas(report.get("replica_keys"))
                        or report.get("skipped") is not False):
                    reasons.append("consumer_replica_scope_unverified")
                if (not _uint64(group.get("sent_fingerprint")) or not _uint64(report.get("fingerprint"))
                        or group.get("sent_fingerprint") != report.get("fingerprint")):
                    reasons.append("group_fingerprint_scope_unverified")
            outcome = group.get("native_group_result")
            witnesses = record.get("groups")
            matching = [w for w in witnesses if type(w) is dict and type(w.get("group_id")) is int
                        and w["group_id"] == gid] if type(witnesses) is list and len(witnesses) <= MAX_GROUPS else []
            if (len(matching) != 1 or len(witnesses) != 1
                    or not _same_primitives(matching[0].get("returned_positions"), outcome)
                    or matching[0].get("reason") != group.get("native_group_error")):
                reasons.append("native_group_witness_inconsistent")
            elif ((outcome is None and matching[0].get("native_result") not in ("unverified", "not_compared"))
                  or (type(outcome) is list and matching[0].get("native_result") != ("mismatch" if outcome else "passed"))):
                reasons.append("native_group_witness_inconsistent")
            if group.get("compare_called") is not True:
                reasons.append("group_comparison_not_called")
            elif outcome is None:
                reasons.append("native_group_not_compared")
                if (native is False and proof.get("source_supported") is True
                        and proof.get("metadata_stable") is True
                        and path == ["on_new_request", "add_recv", "update_from_output", "_verify", "_compare_group"]
                        and _integer(record.get("origin_update_sequence"), 1)
                        and _integer(proof.get("plan_sequence"), 1)
                        and group.get("native_group_error") is None
                        and "native_group_witness_inconsistent" not in reasons
                        and len(matching) == 1 and matching[0].get("native_result") == "not_compared"):
                    compared, comparison_scope_established = [], True
            elif (type(outcome) is not list or len(outcome) > count
                  or not _integer(group.get("start"))
                  or any(not _integer(x) or not group["start"] <= x < group["start"] + count for x in outcome)
                  or len(set(outcome)) != len(outcome)):
                reasons.append("native_group_result_unverified")
            elif not reasons:
                compared = list(expected[group["start"]:])
                comparison_scope_established = True
            if type(outcome) is list:
                if (native is True and outcome) or (native is False and not outcome and not record.get("unverified_reasons")):
                    reasons.append("native_aggregate_group_disagrees")
                    compared, comparison_scope_established = None, False
                if native is False and outcome and "native_group_result_unverified" not in reasons and "native_group_witness_inconsistent" not in reasons:
                    result["native_result"] = "mismatch"
    if not expected:
        expected = None
    if native is None:
        reasons.append("native_verification_not_completed")
    if record.get("recording_complete") is not True:
        reasons.append("recording_budget_exhausted" if record.get("recording_issue") == "global_budget_exhausted"
                       else "recording_scope_incomplete")
    complete_transfer_scope = comparison_scope_established and expected is not None and not reasons
    if complete_transfer_scope:
        result["token_coverage"] = dict(prompt_tokens=nt, transferred_tokens_checked=nt-local,
                                        decode_cached_tokens_unchecked=local)
        if local:
            reasons.append(CACHED_PREFIX_REASON)
    missing = (sorted(set(expected) - set(compared))
               if expected is not None and comparison_scope_established else None)
    reasons = list(dict.fromkeys(_engine_pass_gap_reasons(record, proof) + reasons))
    complete = not reasons and bool(expected) and comparison_scope_established and compared == expected
    result.update(expected_units=expected, compared_units=compared if comparison_scope_established else None,
                  missing_units=missing, coverage="complete" if complete else
                  "partial" if complete_transfer_scope and local else
                  "not_covered" if expected is not None and not compared and
                  reasons == ["native_group_not_compared"] else "unknown",
                  reason="complete_native_checksum_chunk_comparison" if complete else reasons[0]
                  if reasons else "scope_metadata_unverified", reasons=reasons)
    return result


def _code_shape(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return _code_shape(value)
        if isinstance(value, tuple):
            return ("tuple", tuple(constant(x) for x in value))
        return (type(value).__name__, repr(value))
    return (code.co_code, getattr(code, "co_exceptiontable", b""), code.co_name,
            tuple(constant(x) for x in code.co_consts), code.co_names,
            code.co_varnames, code.co_argcount, code.co_posonlyargcount, code.co_kwonlyargcount,
            code.co_freevars, code.co_cellvars, code.co_flags & ~__future__.annotations.compiler_flag)


def _same_primitives(left, right):
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        return False


def _codes(code, prefix=""):
    out = {}
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            name = prefix + item.co_name
            out[name] = item
            out.update(_codes(item, name + "."))
    return out


def _fn(value):
    return value.__func__ if isinstance(value, (classmethod, staticmethod, types.MethodType)) else value


class ChecksumObserver:
    """Bounded metadata collector. Native callbacks always delegate once."""

    def __init__(self, *, max_records=4096, max_units=65536, max_bytes=8 * 1024 * 1024):
        if (not _integer(max_records, 1) or max_records > 65536 or not _integer(max_units, 1)
                or max_units > 1048576 or not _integer(max_bytes, 65536) or max_bytes > 64 * 1024 * 1024):
            raise ValueError("observer budgets exceed supported bounds")
        self.max_records, self.max_units, self.max_bytes = max_records, max_units, max_bytes
        self._lock, self._local = threading.RLock(), threading.local()
        self._records, self._plans = [], weakref.WeakKeyDictionary()
        self._configs, self._new = weakref.WeakKeyDictionary(), weakref.WeakKeyDictionary()
        self._used_units = self._used_bytes = self._sequence = self._plan_count = 0
        self._reserved_bytes = 0
        self._state_count = 0
        self._inflight_position_units = 0
        self._inflight_position_bytes = 0
        self._health = dict(errors=0, dropped_records=0, late_calls=0, active_calls=0,
                            budget_exhausted=False, complete=False)
        self._closed, self._close_active = False, False
        self._installed = False
        self._class, self._originals, self._wrappers, self._tracked = None, {}, {}, []
        self._source = dict(revision=REVISION, files={}, supported=False, reason="not_installed")
        self._qualification = "not_installed"

    def _safe(self, action, default=None):
        try:
            return action()
        except Exception:
            with self._lock:
                self._health["errors"] += 1
            return default

    def _next(self):
        with self._lock:
            self._sequence += 1
            return self._sequence

    def _reserve_state(self):
        with self._lock:
            if (self._state_count >= self.max_records
                    or FIXED_METADATA_BYTES + self._reserved_bytes + self._inflight_position_bytes
                        + (self._state_count + 1) * STATE_ENTRY_BYTES > self.max_bytes):
                self._health["dropped_records"] += 1
                self._health["budget_exhausted"] = True
                return False
            self._state_count += 1
            return True

    def _release_request(self, instance, rid):
        with self._lock:
            if rid in self._plans.get(instance, {}):
                self._plans[instance].pop(rid)
                self._state_count -= 1
                self._plan_count -= 1
            if rid in self._new.get(instance, {}):
                self._new[instance].pop(rid)
                self._state_count -= 1

    def _append(self, record):
        with self._lock:
            if self._closed:
                self._health["late_calls"] += 1
                return
            record["sequence"] = self._next()
            actual = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            reserve = len(actual)
            if record.get("kind") == "verification":
                worst = _plain(record)
                worst["recording_complete"] = False
                worst["recording_issue"] = "global_budget_exhausted"
                worst.update(classify_verification(worst, max_units=self.max_units))
                reserve = max(reserve, len(json.dumps(worst, sort_keys=True, allow_nan=False).encode()))
            units = sum(len(record.get(k) or ()) for k in ("expected_units", "compared_units", "missing_units"))
            if (len(self._records) >= self.max_records
                    or FIXED_METADATA_BYTES + self._reserved_bytes + self._inflight_position_bytes
                        + self._state_count * STATE_ENTRY_BYTES + reserve > self.max_bytes
                    or self._used_units + units > self.max_units):
                self._health["dropped_records"] += 1
                self._health["budget_exhausted"] = True
                return
            self._records.append(record)
            self._used_bytes += len(actual)
            self._reserved_bytes += reserve
            self._used_units += units

    def _copy_positions(self, active, result):
        if type(result) is not list:
            return None
        count = len(result)
        cost = count * 96
        with self._lock:
            if self._closed:
                self._health["late_calls"] += 1
                return None
            if (self._inflight_position_units + count > self.max_units
                    or FIXED_METADATA_BYTES + self._reserved_bytes + self._state_count * STATE_ENTRY_BYTES
                        + self._inflight_position_bytes + cost > self.max_bytes):
                self._health["dropped_records"] += 1
                self._health["budget_exhausted"] = True
                return None
            self._inflight_position_units += count
            self._inflight_position_bytes += cost
            active["position_units"] += count
            active["position_bytes"] += cost
        return list(result)

    def _release_positions(self, active):
        if active is not None:
            with self._lock:
                self._inflight_position_units -= active["position_units"]
                self._inflight_position_bytes -= active["position_bytes"]
                active["position_units"] = active["position_bytes"] = 0

    def _binding_ok(self):
        for owner, name, function, code in self._tracked:
            current = _fn(vars(owner).get(name))
            if owner is self._class and name in self._wrappers:
                if self._installed:
                    if current is not self._wrappers[name]:
                        return False
                    current = self._originals[name]
            if current is not function or not inspect.isfunction(current) or current.__code__ is not code:
                return False
        return True

    def _guard(self, cls, source_root, cpu):
        if type(cls) is not type or cls.__name__ != "KVChecksumScheduler":
            raise UnsupportedSource("expected the pinned checksum scheduler class")
        verify = _fn(vars(cls).get("_verify"))
        if not inspect.isfunction(verify):
            raise UnsupportedSource("native verifier is unavailable")
        actual_path = Path(verify.__code__.co_filename).resolve()
        root = Path(source_root).resolve() if source_root is not None else actual_path.parents[4]
        pins = source_pins()
        contents, compiled = {}, {}
        for name, receipt in pins["files"].items():
            path = root.joinpath(*name.split("/"))
            if not path.is_file() or path.stat().st_size > 1024 * 1024:
                raise UnsupportedSource("required source is missing or too large: " + name)
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                raise UnsupportedSource("source fingerprint differs: " + name)
            contents[name] = receipt["sha256"]
            if name in (SCHEDULER, TYPES, CACHE, MATH):
                compiled[name] = _codes(compile(raw, str(path.resolve()), "exec", dont_inherit=True))
        if actual_path != root.joinpath(*SCHEDULER.split("/")).resolve():
            raise UnsupportedSource("loaded verifier does not come from the pinned source path")
        namespace = verify.__globals__
        if namespace.get("KVChecksumScheduler") is not cls:
            raise UnsupportedSource("the scheduler class is not the loaded source binding")
        def require_loaded_path(function, path, shared_namespace=None):
            expected_path = root.joinpath(*path.split("/")).resolve()
            module_name = path[:-3].replace("/", ".")
            module = sys.modules.get(module_name)
            if (Path(function.__code__.co_filename).resolve() != expected_path
                    or Path(function.__globals__.get("__file__", "")).resolve() != expected_path
                    or module is None or vars(module) is not function.__globals__
                    or (shared_namespace is not None and function.__globals__ is not shared_namespace)):
                raise UnsupportedSource("loaded callback source namespace differs: " + path)
        checks = [(cls, name, SCHEDULER, "KVChecksumScheduler." + name) for name in METHODS]
        checks.append((cls, "_blocks", SCHEDULER, "KVChecksumScheduler._blocks"))
        pending = namespace.get("_Pending")
        if not inspect.isclass(pending):
            raise UnsupportedSource("native pending type is unavailable")
        checks.extend((pending, name, SCHEDULER, "_Pending." + name)
                      for name in ("check_complete", "skipped_groups", "fingerprint", "group_checksums"))
        for owner, name, path, key in checks:
            function = _fn(vars(owner).get(name))
            expected = compiled[path].get(key)
            if not inspect.isfunction(function) or expected is None or _code_shape(function.__code__) != _code_shape(expected):
                raise UnsupportedSource("loaded callback differs: " + key)
            require_loaded_path(function, path, namespace)
            self._tracked.append((owner, name, function, function.__code__))
        # Track helper identities and code as well as the delegated methods.
        helpers = (("_chunk_sums", SCHEDULER), ("_sum_replicas", SCHEDULER),
                   ("_non_null_run", SCHEDULER), ("num_valid_tokens", TYPES),
                   ("cdiv", MATH), ("sliding_window_transfer_blocks", CACHE))
        for name, path in helpers:
            function = namespace.get(name)
            if not inspect.isfunction(function) or _code_shape(function.__code__) != _code_shape(compiled[path][name]):
                raise UnsupportedSource("loaded helper differs: " + name)
            require_loaded_path(function, path, namespace if path == SCHEDULER else None)
        self._helper_bindings = [(namespace, name, namespace[name], namespace[name].__code__) for name, _ in helpers]
        interface = sys.modules.get("vllm.v1.kv_cache_interface")
        full_spec = getattr(interface, "FullAttentionSpec", None)
        if (not inspect.isclass(full_spec) or Path(inspect.getsourcefile(full_spec)).resolve()
                != root.joinpath(*CACHE.split("/")).resolve()):
            raise UnsupportedSource("pinned full-attention spec binding is unavailable")
        for key, expected in compiled[CACHE].items():
            if key.startswith("FullAttentionSpec.") and key.count(".") == 1:
                name = key.split(".")[1]
                function = _fn(vars(full_spec).get(name))
                if not inspect.isfunction(function) or _code_shape(function.__code__) != _code_shape(expected):
                    raise UnsupportedSource("loaded spec method differs: " + key)
                require_loaded_path(function, CACHE, vars(interface))
                self._tracked.append((full_spec, name, function, function.__code__))
        self._full_spec, self._interface = full_spec, interface
        if (type(namespace.get("_PAYLOAD_VERSION")) is not int or namespace["_PAYLOAD_VERSION"] != 1
                or namespace.get("KV_TRANSFER_PARAMS_KEY") != "kv_checksums"):
            raise UnsupportedSource("the checksum payload schema constants differ")
        self._global_bindings = [(namespace, name, namespace.get(name)) for name in
            ("_PAYLOAD_VERSION", "KV_TRANSFER_PARAMS_KEY", "np", "_Pending", "KVChecksumPayload",
             "KVChecksumBlocks", "KVChecksumGroup", "KVChecksumRecord", "MambaSpec", "_Unverified")]
        if not cpu and cls.__module__ != "vllm.distributed.kv_transfer.kv_checksum.scheduler":
            raise UnsupportedSource("runtime scheduler module is unsupported")
        self._source = dict(revision=REVISION, files=contents, supported=True,
                            reason="named_source_files_and_loaded_callbacks_match")
        self._qualification = "cpu_source_functions" if cpu else "source_bound_candidate"

    def install(self, scheduler_class, *, source_root=None):
        return self._install(scheduler_class, source_root, False)

    def install_cpu_source_functions(self, scheduler_class, *, source_root):
        """Explicit non-release seam for executing pinned original CPU functions."""
        return self._install(scheduler_class, source_root, True)

    def _install(self, cls, root, cpu):
        if self._class is not None or self._closed:
            raise ValueError("an observer can be installed once")
        self._tracked = []
        try:
            self._guard(cls, root, cpu)
        except Exception:
            self._tracked = []
            self._source["supported"], self._source["reason"] = False, "source_installation_refused"
            raise
        self._class = cls
        try:
            for name in METHODS:
                original = _fn(vars(cls)[name])
                self._originals[name] = original
                wrapped = self._wrap(name, original)
                self._wrappers[name] = wrapped
                setattr(cls, name, wrapped)
            self._installed = True
        except Exception:
            for name, wrapped in self._wrappers.items():
                if vars(cls).get(name) is wrapped:
                    setattr(cls, name, self._originals[name])
            self._installed = False
            self._class = None
            self._source.update(supported=False, reason="partial_installation_refused")
            raise
        return _plain(self._source)

    def _healthy_bindings(self):
        ok = (self._binding_ok() and getattr(self._interface, "FullAttentionSpec", None) is self._full_spec
              and all(space.get(name) is value for space, name, value in self._global_bindings)
              and all(space.get(name) is function and function.__code__ is code
                      for space, name, function, code in self._helper_bindings))
        if not ok:
            self._source["supported"], self._source["reason"] = False, "loaded_callback_changed"
        return ok and self._source["supported"]

    def _context(self):
        return getattr(self._local, "context", None)

    def _wrap(self, name, original):
        observer = self
        signature = inspect.signature(original)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            with observer._lock:
                observer._health["active_calls"] += 1
                if observer._closed:
                    observer._health["late_calls"] += 1
            bound = observer._safe(lambda: signature.bind(*args, **kwargs).arguments, {})
            instance = bound.get("self")
            state = observer._safe(lambda: observer._before(name, instance, bound))
            try:
                try:
                    result = original(*args, **kwargs)
                except BaseException as exc:
                    observer._safe(lambda: observer._after(name, instance, bound, state, None, exc))
                    raise
                else:
                    observer._safe(lambda: observer._after(name, instance, bound, state, result, None))
                    return result
            finally:
                with observer._lock:
                    observer._health["active_calls"] -= 1
        return observed

    def _before(self, name, instance, bound):
        state = {"context": self._context()}
        if name == "update_from_output":
            context = dict(instance=instance, update=self._next(), verify=None, completed=set(),
                           source_supported=self._healthy_bindings())
            self._local.context = context
            state["owned_context"] = context
        elif name == "_verify":
            context = self._context()
            request = bound.get("req_id")
            pending = bound.get("pending")
            plan = self._plans.get(instance, {}).get(request)
            causal = (context is not None and context["instance"] is instance and context.get("verify") is None
                      and plan is not None and plan["pending_ref"]() is pending)
            proof = self._metadata(instance, pending, plan)
            before_metadata = _plain(proof)
            proof["source_supported"] = self._healthy_bindings()
            proof["call_path"] = ["on_new_request", "add_recv", "update_from_output", "_verify", "_compare_group"] if causal else []
            verification = dict(kind="verification", request_id=request if _text(request) else None,
                origin_update_sequence=context["update"] if causal else None, native_return=None,
                proof=proof, groups=[], unverified_reasons=[], recording_complete=True, recording_issue=None)
            if context is not None:
                context["verify"] = dict(record=verification, pending=pending, instance=instance,
                                         position_units=0, position_bytes=0)
            state.update(verification=verification, context=context, before_metadata=before_metadata, plan=plan,
                         active=context.get("verify") if context is not None else None)
        elif name == "add_recv":
            request = bound.get("request")
            rid = getattr(request, "request_id", None)
            state["remote"] = _text(rid) and rid in getattr(instance, "_remote_prefills", ())
        return state

    def _after(self, name, instance, bound, state, result, error):
        state = state or {}
        if self._closed:
            self._health["late_calls"] += 1
            if name == "update_from_output":
                if state.get("owned_context") is not None:
                    self._release_positions(state["owned_context"].get("verify"))
                self._local.context = state.get("context")
            elif name == "_verify" and state.get("context") is not None:
                self._release_positions(state.get("active"))
                state["context"]["verify"] = None
            return
        if name == "__init__" and error is None:
            config = bound.get("vllm_config")
            parallel = getattr(config, "parallel_config", None)
            def integer_attribute(value, attr):
                item = getattr(value, attr, None)
                return item if type(item) is int and 0 <= item <= 2**63 - 1 else None
            config_record = dict(parallel={
                key: integer_attribute(parallel, attr) for key, attr in (
                    ("tp", "tensor_parallel_size"), ("pp", "pipeline_parallel_size"),
                    ("dp", "data_parallel_size"), ("dcp", "decode_context_parallel_size"),
                    ("pcp", "prefill_context_parallel_size"))},
                speculative_disabled=hasattr(config, "speculative_config") and config.speculative_config is None)
            if self._reserve_state():
                self._configs[instance] = config_record
        elif name == "on_new_request" and error is None:
            rid = getattr(bound.get("request"), "request_id", None)
            if _text(rid) and rid in getattr(instance, "_remote_prefills", ()):
                with self._lock:
                    requests = self._new.setdefault(instance, {})
                    if rid in requests:
                        self._health["errors"] += 1
                    elif self._reserve_state():
                        requests[rid] = self._next()
        elif name == "add_recv":
            self._receive(instance, bound, state, error)
        elif name == "_compare_group":
            context = self._context()
            active = context.get("verify") if context is not None else None
            if (active is not None and active["instance"] is instance
                    and active["pending"] is bound.get("pending")):
                rid = bound.get("group_id")
                if len(active["record"]["groups"]) >= MAX_GROUPS:
                    self._health["dropped_records"] += 1
                    self._health["budget_exhausted"] = True
                    return
                outcome = self._copy_positions(active, result)
                bounded = type(outcome) is list
                native = ("unverified" if error is not None or (result is not None and not bounded) else
                          "not_compared" if result is None else "mismatch" if outcome else "passed")
                group = dict(group_id=rid, native_result=native, returned_positions=outcome,
                             reason=getattr(error, "reason", type(error).__name__)[:512] if error else None)
                active["record"]["groups"].append(group)
                for proof in active["record"]["proof"]["groups"]:
                    if proof["group_id"] == rid:
                        proof.update(compare_called=True, native_group_result=outcome,
                                     native_group_error=group["reason"])
        elif name == "_compare":
            context = self._context()
            active = context.get("verify") if context is not None else None
            if active is not None and type(result) is tuple and len(result) == 2:
                errors = result[1]
                if type(errors) is list and len(errors) <= MAX_GROUPS:
                    active["record"]["unverified_reasons"] = [str(getattr(e, "reason", "unverified"))[:512] for e in errors]
        elif name == "_verify":
            record = state.get("verification")
            context = state.get("context")
            try:
                if record is not None:
                    record["native_return"] = result if type(result) is bool and error is None else None
                    record["proof"]["source_supported"] = self._healthy_bindings()
                    record["proof"]["metadata_stable"] = _same_primitives(
                        state.get("before_metadata"), self._metadata(instance, bound.get("pending"), state.get("plan")))
                    if error is not None:
                        record["engine_exception"] = type(error).__name__
                    counts = [g.get("block_count") for g in record["proof"].get("groups", [])]
                    if any(_integer(n) and (n > self.max_units // 2 or n * 96 > self.max_bytes) for n in counts):
                        record["proof"]["scope_budget_refused"] = True
                        self._health["dropped_records"] += 1
                        self._health["budget_exhausted"] = True
                    record.update(classify_verification(record, max_units=self.max_units))
                    self._append(record)
                    rid = record["request_id"]
                    if context is not None:
                        context["completed"].add(rid)
                    self._release_request(instance, rid)
            finally:
                self._release_positions(state.get("active"))
                if context is not None:
                    context["verify"] = None
        elif name == "update_from_output":
            context = state.get("owned_context")
            try:
                output = bound.get("kv_connector_output")
                finished = getattr(output, "finished_recving", None)
                if type(finished) in (set, list, tuple) and len(finished) <= self.max_records:
                    for rid in finished:
                        if _text(rid) and context is not None and rid not in context["completed"]:
                            skipped = bound.get("skip_req_ids")
                            why = ("native_update_exception" if error is not None else
                                   "load_failed_before_checksum_verification" if type(skipped) is set and rid in skipped else
                                   "native_verification_not_called")
                            self._append(dict(kind="not_verified", request_id=rid,
                                origin_update_sequence=context["update"], reason=why,
                                native_result="unverified", coverage="unknown"))
                            self._release_request(instance, rid)
                elif finished is not None:
                    self._health["dropped_records"] += 1
                    self._health["budget_exhausted"] = True
            finally:
                if context is not None:
                    self._release_positions(context.get("verify"))
                    context["verify"] = None
                self._local.context = state.get("context")

    def _receive(self, instance, bound, state, error):
        rid = getattr(bound.get("request"), "request_id", None)
        if not _text(rid):
            self._health["errors"] += 1
            return
        pending = getattr(instance, "_recvs", {}).get(rid)
        if error is not None or pending is None:
            self._append(dict(kind="recv_plan", request_id=rid, native_result="unverified", coverage="unknown",
                reason="native_add_recv_exception" if error else "not_a_single_p2p_load" if state.get("remote") else "not_scheduled_consumer"))
            return
        with self._lock:
            if not self._reserve_state():
                return
            plans = self._plans.setdefault(instance, {})
            if rid in plans:
                self._state_count -= 1
                self._health["errors"] += 1
                return
            sequence = self._next()
            plan = dict(pending_ref=weakref.ref(pending), plan_sequence=sequence,
                local_tokens=bound.get("num_local_computed_tokens"),
                request_seen=rid in self._new.get(instance, {}))
            plans[rid] = plan
            self._plan_count += 1
        load_plan = self._metadata(instance, pending, plan)
        load_plan["source_supported"] = self._healthy_bindings()
        plan["load_plan"] = load_plan
        self._append(dict(kind="recv_plan", request_id=rid, plan_sequence=sequence,
            reason="native_checksum_receive_scheduled", native_result="scheduled", coverage="unknown",
            load_plan=load_plan))

    def _metadata(self, instance, pending, plan):
        config = self._configs.get(instance, {})
        proof = dict(parallel=config.get("parallel", {}), speculative_disabled=config.get("speculative_disabled"),
            native_tp_size=getattr(instance, "_tp_size", None),
            num_workers=getattr(instance, "_num_workers", None),
            num_local_computed_tokens=plan.get("local_tokens") if plan else None,
            plan_sequence=plan.get("plan_sequence") if plan and plan.get("request_seen") else None,
            num_tokens=None, expected_num_tokens=None, expected_tp_size=None, planned_group_count=None,
            configured_group_count=None, configured_group_ids=[], expected_group_ids=[],
            worker_report_count=None, worker_ranks=[], groups=[])
        blocks = getattr(pending, "blocks", None)
        planned = getattr(blocks, "groups", None)
        expected = getattr(pending, "expected", None)
        proof.update(num_tokens=getattr(blocks, "num_tokens", None),
                     producer_payload_present=type(expected) is self._originals["_verify"].__globals__.get("KVChecksumPayload"),
                     expected_num_tokens=getattr(expected, "num_tokens", None),
                     expected_tp_size=getattr(expected, "tp_size", None),
                     planned_group_count=len(planned) if type(planned) is dict else None)
        if type(planned) is not dict or len(planned) > MAX_GROUPS:
            return proof
        np = self._originals["_verify"].__globals__.get("np")
        array_type = getattr(np, "ndarray", None)
        sent_groups, reports = getattr(expected, "groups", {}), getattr(pending, "reports", {})
        if type(sent_groups) is dict and len(sent_groups) <= MAX_GROUPS:
            proof["expected_group_ids"] = list(sent_groups)
        proof["worker_report_count"] = len(reports) if type(reports) is dict else None
        if type(reports) is dict and len(reports) <= 16:
            proof["worker_ranks"] = [list(rank) if type(rank) is tuple and len(rank) == 2 else None for rank in reports]
        if type(sent_groups) is not dict or type(reports) is not dict or len(reports) > 16:
            return proof
        groups = getattr(instance, "_groups", {})
        proof["configured_group_count"] = len(groups) if type(groups) is dict else None
        if type(groups) is dict and len(groups) <= MAX_GROUPS:
            proof["configured_group_ids"] = list(groups)
        for gid, entry in planned.items():
            if not _integer(gid) or type(entry) is not tuple or len(entry) != 2:
                continue
            start, ids = entry
            count = len(ids) if type(ids) is list else None
            spec = getattr(groups.get(gid), "kv_cache_spec", None) if type(groups) is dict else None
            group = dict(group_id=gid, spec_type=type(spec).__name__, block_size=getattr(spec, "block_size", None),
                spec_source_supported=type(spec) is self._full_spec,
                sliding_window=getattr(spec, "sliding_window", None), attention_chunk_size=getattr(spec, "attention_chunk_size", None),
                non_causal=getattr(spec, "non_causal", None), spec_fields_recorded=all(hasattr(spec, k) for k in
                    ("sliding_window", "attention_chunk_size", "non_causal")), start=start, block_count=count,
                sent_block_size=None, sent_start=None, sent_count=None, sent_dtype=None, sent_shape=None,
                sent_fingerprint=None, consumer_reports=[], compare_called=False, native_group_result=None)
            sent = sent_groups.get(gid)
            if type(sent) is tuple and len(sent) == 4:
                fingerprint, bs, pos, values = sent
                if type(values) is array_type:
                    group.update(sent_fingerprint=fingerprint, sent_block_size=bs, sent_start=pos,
                        sent_count=len(values) if values.ndim else None,
                        sent_dtype=str(values.dtype), sent_shape=list(values.shape))
            for rank, report in reports.items():
                if type(rank) is not tuple or len(rank) != 2 or type(report) is not tuple or len(report) != 2:
                    continue
                data, record = report
                check = data.get(gid) if type(data) is dict else None
                values = getattr(check, "values", None)
                keys = getattr(check, "replica_keys", None)
                if type(values) is not array_type or type(keys) is not list or len(keys) > 16:
                    continue
                keys = [list(k) if type(k) is tuple and len(k) == 2 else None for k in keys]
                skipped = getattr(record, "skipped_groups", None)
                fingerprints = getattr(record, "fingerprints", None)
                group["consumer_reports"].append(dict(pp_rank=rank[0], tp_rank=rank[1],
                    record_pp_rank=getattr(record, "pp_rank", None), record_tp_rank=getattr(record, "tp_rank", None),
                    group_start=getattr(check, "start", None), value_shape=list(values.shape),
                    value_dtype=str(values.dtype), replica_keys=keys,
                    skipped=gid in skipped if type(skipped) is frozenset else None,
                    fingerprint=fingerprints.get(gid) if type(fingerprints) is dict else None))
            proof["groups"].append(group)
        return proof

    def uninstall(self):
        with self._lock:
            if not self._installed:
                return
            self._close_active = self._close_active or self._health["active_calls"] != 0
            if self._class is not None:
                for name, original in self._originals.items():
                    if vars(self._class).get(name) is self._wrappers[name]:
                        setattr(self._class, name, original)
                    else:
                        self._source["supported"], self._source["reason"] = False, "loaded_callback_changed"
            self._installed = False

    def close(self):
        with self._lock:
            if self._closed:
                return self.snapshot()
            self._close_active = self._close_active or self._health["active_calls"] != 0
            if self._class is not None and not self._closed:
                self._safe(self._healthy_bindings)
                for instance, requests in list(self._new.items()):
                    for rid in list(requests):
                        plan = self._plans.get(instance, {}).get(rid, {})
                        self._append(dict(kind="not_verified", request_id=rid, native_result="unverified",
                            coverage="unknown", reason="native_verification_not_observed_before_close",
                            plan_sequence=plan.get("plan_sequence"), load_plan=plan.get("load_plan")))
                        self._release_request(instance, rid)
            self.uninstall()
            self._plans.clear()
            self._new.clear()
            self._configs.clear()
            self._state_count = 0
            self._closed = True
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            health = dict(self._health)
            health["inflight_position_units"] = self._inflight_position_units
            health["inflight_position_bytes"] = self._inflight_position_bytes
            health["complete"] = (self._closed and not self._close_active and not health["errors"]
                and not health["dropped_records"] and not health["late_calls"] and not health["active_calls"]
                and not self._inflight_position_units and not self._inflight_position_bytes
                and self._source["supported"])
            records = _plain(self._records)
            for record in records:
                if record.get("kind") == "verification":
                    record["recording_complete"] = health["complete"]
                    record["recording_issue"] = "global_budget_exhausted" if health["budget_exhausted"] else None
                    record.update(classify_verification(record, max_units=self.max_units))
            retained_bytes = sum(len(json.dumps(record, sort_keys=True, allow_nan=False).encode()) for record in records)
            retained_units = sum(len(record.get(key) or ()) for record in records
                                 for key in ("expected_units", "compared_units", "missing_units"))
            if (retained_units > self.max_units
                    or FIXED_METADATA_BYTES + retained_bytes + self._state_count * STATE_ENTRY_BYTES > self.max_bytes):
                health.update(complete=False, budget_exhausted=True,
                              dropped_records=health["dropped_records"] + len(records))
                records, retained_units, retained_bytes = [], 0, 0
            return _plain(dict(schema=SCHEMA, profile=PROFILE, qualification=self._qualification,
                runtime_qualified=False, release_qualifying=False, closed=self._closed,
                source=self._source, health=health, limits=LIMITS, records=records,
                budgets=dict(max_records=self.max_records, max_units=self.max_units, max_bytes=self.max_bytes,
                             retained_records=len(records), retained_units=retained_units,
                             retained_bytes=retained_bytes, reserved_record_bytes=self._reserved_bytes,
                             fixed_metadata_bytes=FIXED_METADATA_BYTES,
                             reserved_state_bytes=self._state_count * STATE_ENTRY_BYTES)))
