"""Passive native prefill accounting. Owner: Hari Krishna Kancharla.

Only CPU metadata is copied. The native callback retains its return object,
state changes and exceptions. These records do not establish cache integrity.
"""
from __future__ import annotations

from functools import wraps
import inspect
import os
from pathlib import Path
import sys
import threading
from types import MappingProxyType

from . import evidence_io as io, kv_checksum_observer as checksum
from .checksum_client import _context, _integer, _text, prompt_digest, MAX_PROMPT_TOKENS

SCHEMA = "sdcguard.native-prefill-accounting.v1"
PROFILE = "vllm-29d2f4e-native-prefill-accounting.v1"
REVISION = checksum.REVISION
REQUEST = "vllm/v1/request.py"
STATS = "vllm/v1/metrics/stats.py"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"
SOURCE_FILES = {
    REQUEST: "b17215146cad869df29115d0a1f1d2897b6200ed8d3c358e11326154f00fcf6e",
    STATS: "5c925d3c9ba8b81c52eac9cff74a2fe468d14cf710b581ab7eb5ca848450e4a6",
    SCHEDULER: "36538b72be21732194dfeeea1cd1a7566589bf5e6f87ffaf8f7cf31b9eaee27d",
}
FIELDS = ("num_prompt_tokens", "num_computed_tokens", "num_cached_tokens",
          "num_local_cached_tokens", "num_external_cached_tokens")
STATUSES = ("WAITING", "WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR", "WAITING_FOR_REMOTE_KVS",
            "WAITING_FOR_STREAMING_REQ", "RUNNING", "PREEMPTED", "FINISHED_STOPPED",
            "FINISHED_LENGTH_CAPPED", "FINISHED_ABORTED", "FINISHED_IGNORED",
            "FINISHED_ERROR", "FINISHED_REPETITION")
FINISHED_OK = ("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")
MAX_RECORDS = 16384
MAX_BYTES = 8 * 1024 * 1024
FIXED_BYTES = 8192


def source_profile():
    return dict(profile=PROFILE, revision=REVISION, files=dict(SOURCE_FILES))


class PrefillCacheObserver:
    def __init__(self, *, run_id, packet_sha256, arm_id, engine_session_id, role,
                 max_records=MAX_RECORDS, max_bytes=MAX_BYTES):
        context = dict(run_id=run_id, packet_sha256=packet_sha256, arm_id=arm_id,
                       engine_session_id=engine_session_id)
        if not _context(context) or role not in ("prefill", "decode"):
            raise ValueError("native prefill accounting requires a bounded role and run identity")
        if not _integer(max_records, 1, MAX_RECORDS) or not _integer(max_bytes, 65536, MAX_BYTES):
            raise ValueError("unsupported native prefill accounting budget")
        self.context = MappingProxyType(dict(context, role=role, process_id=os.getpid()))
        self.max_records, self.max_bytes = max_records, max_bytes
        self._lock = threading.RLock()
        self._records, self._bytes, self._sequence = [], 0, 0
        self._health = dict(errors=0, dropped_records=0, late_calls=0, active_calls=0, budget_exhausted=False)
        self._source = dict(source_profile(), supported=False)
        self._qualification = "unknown"
        self._closed = self._close_active = self._installed = False
        self._guards = []
        self._modules = []
        self._class = self._original = self._wrapper = self._stats = self._status = None

    def install(self, request_class, scheduler_class, *, source_root):
        return self._install(request_class, scheduler_class, source_root, False)

    def install_cpu_source_functions(self, request_class, scheduler_class, *, source_root):
        return self._install(request_class, scheduler_class, source_root, True)

    def _install(self, request_class, scheduler_class, source_root, cpu):
        with self._lock:
            if self._closed or self._installed:
                raise ValueError("prefill observer is already installed or closed")
            root = Path(source_root).resolve()
            compiled = {}
            for name, digest in SOURCE_FILES.items():
                path = root / name
                if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
                    raise checksum.UnsupportedSource("prefill source is absent or oversized: " + name)
                with path.open("rb") as stream:
                    raw = stream.read(1024 * 1024 + 1)
                if path.resolve() != path or io.digest(raw) != digest:
                    raise checksum.UnsupportedSource("prefill source bytes differ: " + name)
                compiled[name] = dict(checksum._codes(compile(raw, str(path), "exec", dont_inherit=True)))
            original = vars(request_class).get("take_prefill_stats")
            if not inspect.isfunction(original):
                raise checksum.UnsupportedSource("native prefill callback is not a function")
            stats = original.__globals__.get("PrefillStats")
            status = original.__globals__.get("RequestStatus")
            if not isinstance(stats, type) or not isinstance(status, type):
                raise checksum.UnsupportedSource("native prefill types are missing")
            if {k: int(v) for k, v in status.__members__.items()} != {k: i+1 for i, k in enumerate(STATUSES)}:
                raise checksum.UnsupportedSource("native request status enum differs")
            methods = [(request_class, "take_prefill_stats", REQUEST),
                       (request_class, "num_output_tokens", REQUEST),
                       (stats, "set", STATS), (scheduler_class, "schedule", SCHEDULER),
                       (scheduler_class, "update_from_output", SCHEDULER)]
            for cls, name, path in methods:
                descriptor = vars(cls).get(name)
                function = descriptor.fget if type(descriptor) is property else descriptor
                expected = compiled[path].get(cls.__name__ + "." + name)
                if (not inspect.isfunction(function) or expected is None
                        or Path(function.__code__.co_filename).resolve() != root / path
                        or checksum._code_shape(function.__code__) != checksum._code_shape(expected)
                        or function.__closure__ is not None):
                    raise checksum.UnsupportedSource("loaded prefill callback differs: " + name)
                if not cpu:
                    module_name = path[:-3].replace("/", ".")
                    module = sys.modules.get(module_name)
                    if (cls.__module__ != module_name or function.__module__ != module_name
                            or module is None or vars(module) is not function.__globals__
                            or Path(getattr(module, "__file__", "")).resolve() != root / path
                            or function.__globals__.get(cls.__name__) is not cls):
                        raise checksum.UnsupportedSource("native prefill namespace differs")
                    self._modules.append((module_name, module, function.__globals__, root / path, cls))
                self._guards.append((cls, name, descriptor, function, function.__code__,
                                     function.__defaults__, function.__kwdefaults__))
            if not cpu and original.__globals__.get("Request") is not request_class:
                raise checksum.UnsupportedSource("native request class differs")
            self._class, self._original, self._stats, self._status = request_class, original, stats, status
            self._qualification = "cpu_source_functions" if cpu else "source_bound_candidate"
            self._source["supported"] = True

            @wraps(original)
            def observed(*args, **kwargs):
                request = args[0] if len(args) == 1 and not kwargs else kwargs.get("self") if not args else None
                with self._lock:
                    self._health["active_calls"] += 1
                    if self._closed:
                        self._health["late_calls"] += 1
                    if not self._healthy():
                        self._health["errors"] += 1
                try:
                    try:
                        result = original(*args, **kwargs)
                    except BaseException:
                        with self._lock:
                            self._health["errors"] += 1
                        raise
                    with self._lock:
                        try:
                            self._record(request, result)
                        except Exception:
                            self._health["errors"] += 1
                    return result
                finally:
                    with self._lock:
                        self._health["active_calls"] -= 1
            self._wrapper = observed
            request_class.take_prefill_stats = observed
            self._installed = True
        return self

    def _healthy(self):
        try:
            return self._bindings_healthy()
        except Exception:
            self._source["supported"] = False
            return False

    def _bindings_healthy(self):
        valid = self.context["process_id"] == os.getpid()
        for name, module, namespace, path, cls in self._modules:
            module_file = namespace.get("__file__")
            valid = (valid and sys.modules.get(name) is module and vars(module) is namespace
                     and type(module_file) is str and Path(module_file).resolve() == path
                     and namespace.get(cls.__name__) is cls)
        for cls, name, descriptor, function, code, defaults, kwdefaults in self._guards:
            expected = self._wrapper if self._installed and name == "take_prefill_stats" else descriptor
            valid = valid and vars(cls).get(name) is expected and function.__code__ is code
            valid = valid and function.__defaults__ is defaults and function.__kwdefaults__ is kwdefaults
        if self._original is not None:
            valid = valid and self._original.__globals__.get("PrefillStats") is self._stats
            valid = valid and self._original.__globals__.get("RequestStatus") is self._status
        if not valid:
            self._source["supported"] = False
        return valid

    def _record(self, request, result):
        if self._closed:
            return
        self._sequence += 1
        if len(self._records) >= self.max_records or self._bytes + FIXED_BYTES >= self.max_bytes:
            self._health["dropped_records"] += 1
            self._health["budget_exhausted"] = True
            return
        if type(request) is not self._class or not _text(request.request_id):
            raise ValueError("native request identity is unavailable")
        if result is not None and type(result) is not self._stats:
            raise ValueError("native prefill result type differs")
        if type(request.status) is not self._status:
            raise ValueError("native request status type differs")
        stats = {key: getattr(result, key) for key in FIELDS} if result is not None else None
        output_tokens, preemptions = request.num_output_tokens, request.num_preemptions
        if (not _integer(output_tokens, 0, MAX_PROMPT_TOKENS)
                or not _integer(preemptions, 0, MAX_PROMPT_TOKENS)
                or stats is not None and any(not _integer(v, 0, MAX_PROMPT_TOKENS) for v in stats.values())):
            raise ValueError("native cache counters are not bounded integers")
        record = dict(sequence=self._sequence, request_id=request.request_id,
                      output_tokens=output_tokens, preemptions=preemptions,
                      status=request.status.name, stats=stats)
        if result is not None:
            if not _integer(request.num_prompt_tokens, 1, MAX_PROMPT_TOKENS):
                raise ValueError("native prompt count is unsupported")
            record.update(prompt_sha256=prompt_digest(request.prompt_token_ids),
                          prompt_tokens=request.num_prompt_tokens)
        size = len(io.json_bytes(record))
        if len(self._records) >= self.max_records or self._bytes + size + FIXED_BYTES > self.max_bytes:
            self._health["dropped_records"] += 1
            self._health["budget_exhausted"] = True
            return
        self._records.append(record)
        self._bytes += size

    def close(self):
        with self._lock:
            if not self._closed:
                self._close_active = self._health["active_calls"] != 0
                self._healthy()
                if self._installed:
                    if vars(self._class).get("take_prefill_stats") is self._wrapper:
                        self._class.take_prefill_stats = self._original
                    else:
                        self._source["supported"] = False
                self._installed = False
                self._closed = True
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            health = dict(self._health)
            health["complete"] = (self._closed and not self._close_active and self._source["supported"]
                and not any(health.values()))
            return io.parse_json(io.json_bytes(dict(schema=SCHEMA, **self.context,
                qualification=self._qualification, source=self._source, closed=self._closed,
                health=health, records=self._records,
                budgets=dict(max_records=self.max_records, max_bytes=self.max_bytes,
                    retained_records=len(self._records), retained_bytes=self._bytes, fixed_bytes=FIXED_BYTES))))


def evaluate_snapshot(snapshot, context, role):
    """Validate observations before any request join. Missing data is never zero."""
    reasons = []
    if (type(snapshot) is not dict or snapshot.get("schema") != SCHEMA
            or snapshot.get("role") != role or snapshot.get("closed") is not True
            or not _integer(snapshot.get("process_id"), 1)
            or snapshot.get("qualification") not in ("source_bound_candidate", "cpu_source_functions")
            or any(snapshot.get(k) != v for k, v in context.items())
            or type(snapshot.get("source")) is not dict or snapshot["source"].get("supported") is not True
            or snapshot.get("source") != dict(source_profile(), supported=True)):
        return [], ["native_cache_accounting_identity_unverified"]
    health, limits, records = snapshot.get("health"), snapshot.get("budgets"), snapshot.get("records")
    if (type(health) is not dict or health.get("complete") is not True
            or health.get("budget_exhausted") is not False
            or any(type(health.get(k)) is not int or health[k] != 0
                   for k in ("errors", "dropped_records", "late_calls", "active_calls"))):
        reasons.append("native_cache_accounting_incomplete")
    if type(records) is not list or len(records) > MAX_RECORDS:
        return [], reasons + ["native_cache_accounting_records_unverified"]
    if (type(limits) is not dict or not _integer(limits.get("max_records"), 1, MAX_RECORDS)
            or not _integer(limits.get("max_bytes"), 65536, MAX_BYTES)
            or type(limits.get("retained_records")) is not int or limits["retained_records"] != len(records)
            or type(limits.get("retained_bytes")) is not int
            or limits["retained_bytes"] != sum(len(io.json_bytes(r)) for r in records)
            or type(limits.get("fixed_bytes")) is not int or limits["fixed_bytes"] != FIXED_BYTES
            or len(records) > limits["max_records"]
            or limits["retained_bytes"] + FIXED_BYTES > limits["max_bytes"]):
        reasons.append("native_cache_accounting_budget_unverified")
    seq = [r.get("sequence") if type(r) is dict else None for r in records]
    if any(not _integer(s, 1) for s in seq) or seq != list(range(1, len(seq) + 1)):
        reasons.append("native_cache_accounting_sequence_unverified")
    for row in records:
        if (type(row) is not dict or not _text(row.get("request_id"))
                or not _integer(row.get("output_tokens"), 0, MAX_PROMPT_TOKENS)
                or not _integer(row.get("preemptions"), 0, MAX_PROMPT_TOKENS)
                or row.get("status") not in STATUSES or "stats" not in row):
            reasons.append("native_cache_accounting_record_unverified")
            continue
        value = row["stats"]
        if value is not None and (type(value) is not dict or set(value) != set(FIELDS)
                or any(not _integer(value[k], 0, MAX_PROMPT_TOKENS) for k in FIELDS)
                or not _integer(row.get("prompt_tokens"), 1, MAX_PROMPT_TOKENS)
                or value["num_prompt_tokens"] != row["prompt_tokens"]
                or value["num_cached_tokens"] != value["num_local_cached_tokens"] + value["num_external_cached_tokens"]
                or value["num_prompt_tokens"] != value["num_computed_tokens"] + value["num_cached_tokens"]):
            reasons.append("native_cache_accounting_counts_unverified")
    return records, list(dict.fromkeys(reasons))


def request_usage(records, internal_id, prompt_tokens, prompt_sha256, completion_tokens):
    selected = [r for r in records if type(r) is dict and r.get("request_id") == internal_id]
    initial = [r for r in selected if r.get("stats") is not None]
    if len(initial) != 1 or not selected:
        return None, ["unambiguous_native_prefill_stats_missing"]
    first, last = initial[0], selected[-1]
    reasons = []
    if (first.get("prompt_tokens") != prompt_tokens or first.get("prompt_sha256") != prompt_sha256
            or first is not selected[0]):
        reasons.append("native_cache_prompt_binding_unverified")
    counts = [r.get("output_tokens") for r in selected]
    if (any(not _integer(n) for n in counts) or counts != sorted(counts)
            or not _integer(completion_tokens, 1) or counts[-1] != completion_tokens
            or last.get("status") not in FINISHED_OK
            or any(r.get("status") != "RUNNING" for r in selected[:-1])):
        reasons.append("native_cache_terminal_request_unverified")
    if any(type(r.get("preemptions")) is not int or r["preemptions"] != 0 for r in selected):
        reasons.append("native_cache_preempted_request_unsupported")
    return first.get("stats") if not reasons else None, reasons
