"""Request-bound checksum evidence. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

from collections import Counter
from functools import wraps
from importlib.resources import files as resources
import inspect
import base64
import binascii
import json
import os
from pathlib import Path
import re
import threading
from types import MappingProxyType

from . import checksum_observer_report as checks, evidence_io as io
from . import kv_checksum_observer as checksum
from . import checksum_profiles as profiles

REVISION = "29d2f4e03740acb14439ce22d24b06d8095d17d3"
PROFILE = "vllm-checksum-62211-client-join.v1"
BINDING_PROFILE = "vllm-checksum-62211-request-id.v1"
BINDING_SCHEMA = "sdcguard.checksum-request-bindings.v1"
PLAN_SCHEMA = "sdcguard.checksum-request-plan.v1"
INVENTORY_SCHEMA = "sdcguard.checksum-client-inventory.v1"
CAPTURES_SCHEMA = "sdcguard.checksum-client-captures.v1"
REPORT_SCHEMA = "sdcguard.checksum-client-report.v1"
TITLE = "Client requests and native checksum coverage"
SOURCE_FILES = {
    "vllm/v1/engine/__init__.py": "ebbd40829b11c6c1571a6b8e62680b5610087c49f6576b9f5921887e5dae03bf",
    "vllm/v1/engine/input_processor.py": "f90797694a2b74c61f62ac02e2f6d1f708b680276007b92ee4c6168ab4bbca71",
    "vllm/entrypoints/openai/completion/serving.py": "b2556daa8eb0573bb9d05e8456da1dc7e30c76979705bb3f293230b09be41417",
    "vllm/entrypoints/serve/engine/serving.py": "e46693a37e9a8b1087c8cee8a1e2cb9a2ce3d052772e3771755ef59de0885cf4",
}
INPUT_PROCESSOR = "vllm/v1/engine/input_processor.py"
MAX_REQUESTS, MAX_ARMS, MAX_PHASES, MAX_SNAPSHOTS = 4096, 8, 16, 64
MAX_PROMPT_TOKENS = 8192
CONTEXT_KEYS = ("run_id", "packet_sha256", "arm_id", "engine_session_id")
BINDING_FIXED_BYTES = 8192
LIMITS = [
    "The plan preserves every planned client slot, including missing or incomplete HTTP receipts.",
    "Frontend assignment observations bind actual external and internal IDs; suffixes are not guessed.",
    "HTTP responses, source declarations, capture context and local hashes are not trusted execution attestation.",
    "Counts distinguish logical prefill/decode pairs from individual HTTP stages and native verification calls.",
    "Native success, conditional comparison scope and independent client completeness are separate observations.",
    "Off-arm requests have no observer coverage and cannot borrow checks from the on arm.",
    "This profile remains runtime unqualified and cannot approve a lease-plus-checksum release.",
    "Router forwarding beyond the recorded client handoff remains unknown.",
    "Ownership, read rights, physical reads and model-layer coverage remain unknown.",
]


def _text(value, maximum=1024):
    return type(value) is str and 0 < len(value) <= maximum and "\x00" not in value


def _integer(value, minimum=0, maximum=2**63 - 1):
    return type(value) is int and minimum <= value <= maximum


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _same(a, b):
    return io.json_bytes(a) == io.json_bytes(b)


def _context(value):
    return (type(value) is dict and type(value.get("run_id")) is str
            and re.fullmatch(r"[0-9a-f]{32}", value["run_id"]) is not None
            and _sha(value.get("packet_sha256"))
            and _text(value.get("arm_id"), 128) and _text(value.get("engine_session_id"), 128))


def source_profile(revision=REVISION):
    profile = profiles.get(revision)
    return {"profile": PROFILE, "revision": revision,
            "api_request_id_files": profile["binding_files"], "checksum_profile": checksum.source_profile(revision)}


def evaluator_hashes():
    result = checks.evaluator_hashes()
    result["checksum_client.py"] = io.digest(resources("sdcguard").joinpath("checksum_client.py").read_bytes())
    for name in ("prefix_cache_observer.py", "prefix_cache_coverage.py"):
        result[name] = io.digest(resources("sdcguard").joinpath(name).read_bytes())
    return result


def prompt_digest(tokens):
    if (type(tokens) is not list or not 1 <= len(tokens) <= MAX_PROMPT_TOKENS
            or any(not _integer(token, 0, 2**31 - 1) for token in tokens)):
        raise io.EvidenceError("prompt must be a bounded nonempty integer token list")
    return io.digest(io.json_bytes(tokens))


class RequestBindingObserver:
    """Observe the original request-ID assignment without selecting an ID."""

    def __init__(self, *, run_id, packet_sha256, arm_id, engine_session_id,
                 max_records=4096, max_bytes=2 * 1024 * 1024, revision=REVISION):
        self.profile = profiles.get(revision)
        self.revision = self.profile["revision"]
        self.context = dict(run_id=run_id, packet_sha256=packet_sha256,
                            arm_id=arm_id, engine_session_id=engine_session_id)
        if not _context(self.context):
            raise ValueError("a bounded run, packet, arm and engine session identity is required")
        if not _integer(max_records, 1, MAX_REQUESTS) or not _integer(max_bytes, 65536, 8 * 1024 * 1024):
            raise ValueError("unsupported request binding budget")
        self.context["process_id"] = os.getpid()
        self.context = MappingProxyType(self.context)
        self.max_records, self.max_bytes = max_records, max_bytes
        self._lock = threading.RLock()
        self._records, self._bytes, self._sequence = [], 0, 0
        self._health = dict(errors=0, dropped_records=0, late_calls=0, active_calls=0, budget_exhausted=False)
        self._source = dict(profile=BINDING_PROFILE, revision=self.revision, files=self.profile["binding_files"],
                            supported=False, reason="not_installed")
        self._qualification = "unknown"
        self._class = self._original = self._descriptor = self._wrapper = self._code = None
        self._globals = {}
        self._request_type = None
        self._closed = self._close_active = self._installed = False

    def install(self, input_processor_class, *, source_root=None):
        return self._install(input_processor_class, source_root, False)

    def install_cpu_source_functions(self, input_processor_class, *, source_root):
        return self._install(input_processor_class, source_root, True)

    def _install(self, cls, source_root, cpu):
        with self._lock:
            if self._closed or self._installed:
                raise ValueError("request binding observer is already installed or closed")
            descriptor = vars(cls).get("assign_request_id")
            if type(descriptor) is not staticmethod or not inspect.isfunction(descriptor.__func__):
                raise checksum.UnsupportedSource("assign_request_id must be the original static method")
            original = descriptor.__func__
            if original.__defaults__ is not None or original.__kwdefaults__ is not None or original.__closure__ is not None:
                raise checksum.UnsupportedSource("request-ID callback defaults or closure differ")
            actual_path = Path(original.__code__.co_filename).resolve()
            root = Path(source_root).resolve() if source_root is not None else actual_path.parents[3]
            expected_path = root / INPUT_PROCESSOR
            if actual_path != expected_path or cls.__name__ != "InputProcessor":
                raise checksum.UnsupportedSource("unsupported input processor source path")
            for name, digest in self.profile["binding_files"].items():
                target = root / name
                if target.resolve() != target or io.digest(target.read_bytes()) != digest:
                    raise checksum.UnsupportedSource("request-ID source bytes differ: " + name)
            code = compile(expected_path.read_bytes(), str(expected_path), "exec")
            compiled = dict(checksum._codes(code))
            expected_code = compiled.get("InputProcessor.assign_request_id")
            if expected_code is None or checksum._code_shape(original.__code__) != checksum._code_shape(expected_code):
                raise checksum.UnsupportedSource("loaded request-ID callback differs")
            namespace = original.__globals__
            if not cpu and (cls.__module__ != "vllm.v1.engine.input_processor"
                            or namespace.get("__name__") != cls.__module__
                            or namespace.get("InputProcessor") is not cls):
                raise checksum.UnsupportedSource("loaded input processor class is not source-bound")
            request_type = namespace.get("EngineCoreRequest")
            if not isinstance(request_type, type) or not callable(namespace.get("random_uuid")) or namespace.get("envs") is None:
                raise checksum.UnsupportedSource("request-ID callback namespace is incomplete")
            if not cpu and (request_type.__module__ != "vllm.v1.engine"
                            or Path(inspect.getfile(request_type)).resolve() != root / "vllm/v1/engine/__init__.py"):
                raise checksum.UnsupportedSource("loaded engine request type is not source-bound")
            self._class, self._original, self._descriptor = cls, original, descriptor
            self._code, self._request_type = original.__code__, request_type
            self._globals = {name: namespace[name] for name in ("EngineCoreRequest", "envs", "random_uuid")}
            self._qualification = "cpu_source_functions" if cpu else "source_bound_candidate"
            self._source.update(supported=True, reason="exact_source_and_loaded_callback")

            @wraps(original)
            def wrapped(*args, **kwargs):
                request = args[0] if len(args) == 1 and not kwargs else kwargs.get("request") if not args else None
                before = None
                with self._lock:
                    self._health["active_calls"] += 1
                    if self._closed:
                        self._health["late_calls"] += 1
                    if not self._healthy_bindings():
                        self._health["errors"] += 1
                    if type(request) is self._request_type:
                        try:
                            before = request.request_id
                        except Exception:
                            self._health["errors"] += 1
                result, error = None, None
                try:
                    result = original(*args, **kwargs)
                    return result
                except BaseException as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    with self._lock:
                        try:
                            self._record(request, before, result, error)
                        except Exception:
                            self._health["errors"] += 1
                        self._health["active_calls"] -= 1
            self._wrapper = staticmethod(wrapped)
            setattr(cls, "assign_request_id", self._wrapper)
            self._installed = True
        return self

    def _healthy_bindings(self):
        healthy = (self.context["process_id"] == os.getpid()
                   and self._original is not None and self._original.__code__ is self._code
                   and self._original.__defaults__ is None and self._original.__kwdefaults__ is None
                   and self._original.__closure__ is None
                   and all(self._original.__globals__.get(name) is value for name, value in self._globals.items()))
        if self._installed:
            healthy = healthy and vars(self._class).get("assign_request_id") is self._wrapper
        if not healthy:
            self._source.update(supported=False, reason="loaded_request_id_binding_changed")
        return healthy

    def _record(self, request, before, result, error):
        self._sequence += 1
        if (type(request) is not self._request_type or not _text(before)
                or not _text(getattr(request, "external_req_id", None))
                or not _text(getattr(request, "request_id", None))):
            self._health["errors"] += 1
            return
        tokens = getattr(request, "prompt_token_ids", None)
        try:
            digest, count = prompt_digest(tokens), len(tokens)
        except io.EvidenceError:
            digest, count = None, None
        record = dict(sequence=self._sequence, external_request_id=before,
                      external_req_id=request.external_req_id, internal_request_id=request.request_id,
                      native_return_is_none=result is None, engine_exception=error,
                      randomization_observed=request.request_id != before,
                      prompt_tokens=count, prompt_sha256=digest)
        if error is not None or result is not None or request.external_req_id != before:
            self._health["errors"] += 1
        count = len(io.json_bytes(record))
        if len(self._records) >= self.max_records or self._bytes + count + BINDING_FIXED_BYTES > self.max_bytes:
            self._health["dropped_records"] += 1
            self._health["budget_exhausted"] = True
            return
        self._records.append(record)
        self._bytes += count

    def close(self):
        with self._lock:
            if not self._closed:
                self._close_active = self._health["active_calls"] != 0
                if self._installed:
                    self._healthy_bindings()
                    if vars(self._class).get("assign_request_id") is self._wrapper:
                        setattr(self._class, "assign_request_id", self._descriptor)
                    else:
                        self._source.update(supported=False, reason="foreign_request_id_wrapper_preserved")
                    self._installed = False
                self._closed = True
            return self.snapshot()

    uninstall = close

    def snapshot(self):
        with self._lock:
            health = dict(self._health)
            health["complete"] = (self._closed and not self._close_active and self._source["supported"]
                and not any(health[k] for k in ("errors", "dropped_records", "late_calls", "active_calls", "budget_exhausted")))
            return json.loads(io.json_bytes(dict(schema=BINDING_SCHEMA, profile=BINDING_PROFILE,
                qualification=self._qualification, runtime_qualified=False, release_qualifying=False,
                **self.context, source=self._source, closed=self._closed, health=health,
                budgets=dict(max_records=self.max_records, max_bytes=self.max_bytes,
                    retained_records=len(self._records), retained_bytes=self._bytes,
                    fixed_metadata_bytes=BINDING_FIXED_BYTES), records=self._records)))


def _read(raw, schema, name):
    value = io.parse_json(raw)
    if type(value) is not dict or value.get("schema") != schema:
        raise io.EvidenceError("unsupported " + name + " schema")
    return value


def _identity(plan, value, *, plan_raw=None):
    if (value.get("run_id") != plan.get("run_id") or value.get("packet_sha256") != plan.get("packet_sha256")
            or plan_raw is not None and value.get("plan_sha256") != io.digest(plan_raw)):
        return False
    return True


def _planned(plan):
    if "gpu_blocks_override" in plan and (type(plan["gpu_blocks_override"]) is not int
            or plan["gpu_blocks_override"] != 8192):
        raise io.EvidenceError("request plan cache capacity is unsupported")
    if type(plan.get("prefix_caching", False)) is not bool:
        raise io.EvidenceError("prefix cache configuration must be a recorded boolean")
    if (type(plan.get("run_id")) is not str or re.fullmatch(r"[0-9a-f]{32}", plan["run_id"]) is None
            or not _sha(plan.get("packet_sha256")) or not profiles.supported(plan.get("source_revision"))
            or not _text(plan.get("model"))):
        raise io.EvidenceError("request plan identity, source revision or model is unsupported")
    phases, arms = plan.get("phases"), plan.get("arms")
    if not isinstance(phases, list) or not 1 <= len(phases) <= MAX_PHASES:
        raise io.EvidenceError("request plan phases exceed the supported bound")
    phase_map = {}
    for phase in phases:
        if (type(phase) is not dict or not _text(phase.get("phase_id"), 128)
                or not _integer(phase.get("concurrency"), 1, 64) or phase["phase_id"] in phase_map):
            raise io.EvidenceError("request plan phase declarations are ambiguous")
        phase_map[phase["phase_id"]] = phase["concurrency"]
    if type(arms) is not list or not 1 <= len(arms) <= MAX_ARMS:
        raise io.EvidenceError("request plan arms exceed the supported bound")
    rows, arm_map, identifiers, sessions = [], {}, set(), set()
    for arm in arms:
        if (type(arm) is not dict or not _text(arm.get("arm_id"), 128)
                or not _text(arm.get("engine_session_id"), 128) or type(arm.get("observer_enabled")) is not bool
                or arm["arm_id"] in arm_map or arm["engine_session_id"] in sessions
                or type(arm.get("requests")) is not list or not arm["requests"]):
            raise io.EvidenceError("request plan arm or session declarations are ambiguous")
        arm_map[arm["arm_id"]], sessions = arm, sessions | {arm["engine_session_id"]}
        for slot, request in enumerate(arm["requests"]):
            if len(rows) >= MAX_REQUESTS:
                raise io.EvidenceError("request plan exceeds its logical request bound")
            if (type(request) is not dict or type(request.get("slot")) is not int or request["slot"] != slot
                    or not _text(request.get("phase_id"), 128) or request["phase_id"] not in phase_map or request.get("kind") not in ("warmup", "measured")
                    or not _integer(request.get("concurrency"), 1, 64)
                    or request["concurrency"] != (1 if request["kind"] == "warmup" else phase_map[request["phase_id"]])
                    or not _text(request.get("client_request_id")) or not _text(request.get("prefill_request_id"))
                    or request["client_request_id"] == request["prefill_request_id"]
                    or request["client_request_id"] in identifiers or request["prefill_request_id"] in identifiers
                    or not _integer(request.get("prompt_tokens"), 1, MAX_PROMPT_TOKENS)
                    or not _sha(request.get("prompt_sha256")) or not _integer(request.get("max_tokens"), 1, 4096)):
                raise io.EvidenceError("request plan slot or request contract is malformed")
            identifiers.update((request["client_request_id"], request["prefill_request_id"]))
            rows.append({**request, "arm_id": arm["arm_id"], "engine_session_id": arm["engine_session_id"],
                         "observer_enabled": arm["observer_enabled"]})
    if not any(arm["observer_enabled"] for arm in arm_map.values()):
        raise io.EvidenceError("request plan must retain at least one observed arm")
    return rows, arm_map, phase_map


def _stage(stage, row, model, phase):
    reasons, external = [], None
    if type(stage) is not dict:
        return [phase + "_receipt_missing"], None
    if (stage.get("completed") is not True or "error" not in stage or stage.get("error") is not None
            or type(stage.get("http_status")) is not int or stage["http_status"] != 200
            or stage.get("method") != "POST" or stage.get("path") != "/v1/completions"):
        reasons.append(phase + "_http_completion_unverified")
    body, headers, response = stage.get("request_body"), stage.get("request_headers"), stage.get("response_body")
    if type(body) is not dict or type(headers) is not dict or type(response) is not dict:
        return reasons + [phase + "_request_or_response_missing"], None
    expected = row["prefill_request_id"] if phase == "prefill" else row["client_request_id"]
    header_values = [value for key, value in headers.items() if type(key) is str and key.lower() == "x-request-id"]
    if (header_values != [expected] or body.get("request_id") != expected
            or response.get("id") != "cmpl-" + expected):
        reasons.append(phase + "_api_request_id_binding_unverified")
    else:
        external = response["id"] + "-0"
    try:
        digest = prompt_digest(body.get("prompt"))
    except io.EvidenceError:
        digest = None
    usage = response.get("usage")
    if (digest != row["prompt_sha256"] or type(body.get("prompt")) is not list
            or len(body["prompt"]) != row["prompt_tokens"] or type(usage) is not dict
            or type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] != row["prompt_tokens"]):
        reasons.append(phase + "_prompt_contract_unverified")
    max_tokens = 1 if phase == "prefill" else row["max_tokens"]
    if (body.get("model") != model or type(body.get("n")) is not int or body["n"] != 1
            or body.get("stream") is not False or body.get("add_special_tokens") is not False
            or type(body.get("max_tokens")) is not int or body["max_tokens"] != max_tokens
            or body.get("use_beam_search", False) is not False
            or body.get("best_of", 1) != 1 or type(body.get("best_of", 1)) is not int):
        reasons.append(phase + "_single_prompt_api_contract_unverified")
    return reasons, external


def _ordinary_descriptor(value, row):
    if type(value) is not dict:
        return False
    blocks = value.get("remote_block_ids")
    if (value.get("do_remote_prefill") is not True or value.get("do_remote_decode") is not False
            or any(not _text(value.get(key)) for key in ("remote_request_id", "remote_engine_id", "remote_host"))
            or not _integer(value.get("remote_port"), 1, 65535)
            or any(type(value.get(key)) is not int or value[key] != 1 for key in ("tp_size", "dcp_size", "pp_size"))
            or type(value.get("remote_num_tokens")) is not int or value["remote_num_tokens"] != row["prompt_tokens"]
            or not _integer(value.get("remote_registration_generation"))
            or type(blocks) is not list or len(blocks) != 1 or type(blocks[0]) is not list
            or not 1 <= len(blocks[0]) <= MAX_PROMPT_TOKENS
            or any(not _integer(block, 0, 2**64 - 1) for block in blocks[0])
            or not _text(value.get("kv_checksums"), 65536)):
        return False
    try:
        decoded = base64.b64decode(value["kv_checksums"], validate=True)
    except (ValueError, binascii.Error):
        return False
    return bool(decoded)


def _client_row(raw, row, model):
    reasons = []
    if type(raw) is not dict:
        return ["client_receipt_missing"], None
    if (raw.get("arm_id") != row["arm_id"] or type(raw.get("slot")) is not int or raw["slot"] != row["slot"]
            or raw.get("client_request_id") != row["client_request_id"]):
        reasons.append("client_plan_slot_binding_unverified")
    p_reasons, _ = _stage(raw.get("prefill"), row, model, "prefill")
    d_reasons, external = _stage(raw.get("decode"), row, model, "decode")
    reasons.extend(p_reasons + d_reasons)
    prefill, decode = raw.get("prefill"), raw.get("decode")
    if type(prefill) is dict and type(decode) is dict:
        response, body = prefill.get("response_body"), decode.get("request_body")
        descriptor = response.get("kv_transfer_params") if type(response) is dict else None
        supplied = body.get("kv_transfer_params") if type(body) is dict else None
        if not _ordinary_descriptor(descriptor, row) or not _same(descriptor, supplied):
            reasons.append("client_transfer_descriptor_binding_unverified")
        prefill_body = prefill.get("request_body")
        options = prefill_body.get("kv_transfer_params") if type(prefill_body) is dict else None
        if (type(options) is not dict or options.get("do_remote_decode") is not True
                or options.get("do_remote_prefill") is not False):
            reasons.append("client_prefill_direction_unverified")
    else:
        reasons.append("client_transfer_descriptor_binding_unverified")
    return list(dict.fromkeys(reasons)), external


def _snapshot_context(snapshot, context):
    return (type(snapshot) is dict and all(snapshot.get(k) == context[k] for k in CONTEXT_KEYS)
            and _integer(snapshot.get("process_id"), 1, 2**31 - 1))


def _bindings(snapshot, context, revision=REVISION):
    reasons = []
    if type(snapshot) is not dict:
        return [], ["binding_snapshot_missing"], None
    source = snapshot.get("source")
    qualification = snapshot.get("qualification")
    qualification = qualification if type(qualification) is str else None
    if (snapshot.get("schema") != BINDING_SCHEMA or snapshot.get("profile") != BINDING_PROFILE
            or not _snapshot_context(snapshot, context) or snapshot.get("closed") is not True
            or snapshot.get("runtime_qualified") is not False or snapshot.get("release_qualifying") is not False
            or qualification not in ("source_bound_candidate", "cpu_source_functions")):
        reasons.append("binding_snapshot_identity_or_closure_unverified")
    if (type(source) is not dict or source.get("supported") is not True
            or source.get("profile") != BINDING_PROFILE or source.get("revision") != revision
            or not _same(source.get("files"), profiles.get(revision)["binding_files"])):
        reasons.append("binding_source_unverified")
    health = snapshot.get("health")
    if (type(health) is not dict or health.get("complete") is not True or health.get("budget_exhausted") is not False
            or any(type(health.get(k)) is not int or health[k] != 0 for k in
                   ("errors", "dropped_records", "late_calls", "active_calls"))):
        reasons.append("binding_recording_health_unverified")
    records, budgets = snapshot.get("records"), snapshot.get("budgets")
    if type(records) is not list:
        reasons.append("binding_record_list_unverified")
        records = []
    if len(records) > MAX_REQUESTS:
        raise io.EvidenceError("binding records exceed the supported bound")
    actual_bytes = sum(len(io.json_bytes(record)) for record in records)
    if (type(budgets) is not dict or not _integer(budgets.get("max_records"), 1, MAX_REQUESTS)
            or not _integer(budgets.get("max_bytes"), 65536, 8 * 1024 * 1024)
            or type(budgets.get("retained_records")) is not int or budgets["retained_records"] != len(records)
            or type(budgets.get("retained_bytes")) is not int or budgets["retained_bytes"] != actual_bytes
            or type(budgets.get("fixed_metadata_bytes")) is not int or budgets["fixed_metadata_bytes"] != BINDING_FIXED_BYTES
            or len(records) > budgets["max_records"] or actual_bytes + BINDING_FIXED_BYTES > budgets["max_bytes"]):
        reasons.append("binding_snapshot_budget_unverified")
    sequences = [record.get("sequence") if type(record) is dict else None for record in records]
    if (any(not _integer(seq, 1) for seq in sequences) or len(sequences) != len(set(sequences))
            or sequences != sorted(sequences)):
        reasons.append("binding_sequence_unverified")
    for record in records:
        if (type(record) is not dict or not _text(record.get("external_request_id"))
                or record.get("external_req_id") != record.get("external_request_id")
                or not _text(record.get("internal_request_id")) or record.get("randomization_observed") is not True
                or record.get("native_return_is_none") is not True or "engine_exception" not in record
                or record.get("engine_exception") is not None
                or record.get("external_request_id") == record.get("internal_request_id")):
            reasons.append("binding_native_assignment_unverified")
        if (type(record) is not dict or not _integer(record.get("prompt_tokens"), 1, MAX_PROMPT_TOKENS)
                or not _sha(record.get("prompt_sha256"))):
            reasons.append("native_prompt_recording_unverified")
    return records, list(dict.fromkeys(reasons)), qualification


def _capture(value, plan, arm):
    revision = profiles.get(plan["source_revision"])["revision"]
    context = {key: (plan[key] if key in ("run_id", "packet_sha256") else arm[key]) for key in CONTEXT_KEYS}
    reasons = []
    if type(value) is not dict:
        return {"issues": ["engine_capture_missing"], "bindings": [], "calls": [], "unverified": [],
                "qualification": "unknown"}
    if (any(value.get(k) != context[k] for k in ("arm_id", "engine_session_id"))
            or value.get("source_revision") != revision or value.get("capture_complete") is not True):
        reasons.append("engine_capture_identity_or_closure_unverified")
    startup = value.get("startup")
    if (type(startup) is not dict or not _sha(value.get("startup_sha256"))
            or io.digest(io.json_bytes(startup)) != value.get("startup_sha256")
            or any(startup.get(k) != context[k] for k in CONTEXT_KEYS)
            or startup.get("source_revision") != revision or startup.get("role") != "decode"
            or startup.get("model") != plan["model"] or not _same(startup.get("source_profile"), source_profile(revision))
            or startup.get("checksum_enabled") is not True or startup.get("request_id_randomization_enabled") is not True):
        reasons.append("engine_startup_binding_unverified")
    binding_snapshots, snapshots = value.get("binding_snapshots"), value.get("checksum_snapshots")
    if type(binding_snapshots) is not list:
        reasons.append("binding_snapshot_list_unverified")
        binding_snapshots = []
    if type(snapshots) is not list:
        reasons.append("checksum_snapshot_list_unverified")
        snapshots = []
    if len(binding_snapshots) > MAX_SNAPSHOTS or len(snapshots) > MAX_SNAPSHOTS:
        raise io.EvidenceError("capture snapshot lists exceed the supported bound")
    bindings, calls, unverified, domains, binding_active, checksum_active = [], [], [], [], 0, 0
    checksum_record_count = 0
    for index, snapshot in enumerate(binding_snapshots):
        records, issues, domain = _bindings(snapshot, context, revision)
        reasons.extend(issues)
        domains.append(domain)
        binding_active += bool(records)
        bindings.extend({**record, "binding_snapshot_index": index, "process_id": snapshot.get("process_id")}
                        for record in records if type(record) is dict)
        if len(bindings) > MAX_REQUESTS:
            raise io.EvidenceError("aggregate binding records exceed the supported bound")
    for index, snapshot in enumerate(snapshots):
        if type(snapshot) is dict and type(snapshot.get("records")) is list:
            checksum_record_count += len(snapshot["records"])
            if checksum_record_count > MAX_REQUESTS:
                raise io.EvidenceError("aggregate checksum records exceed the supported bound")
        if (type(snapshot) is not dict or not _snapshot_context(snapshot.get("capture_context"), context)):
            reasons.append("checksum_capture_context_unverified")
        try:
            report = checks.evaluate(io.json_bytes(snapshot), revision=revision)
        except (io.EvidenceError, TypeError, ValueError) as exc:
            reasons.append("checksum_snapshot_unreadable")
            unverified.append(dict(request_id=None, reason="checksum_snapshot_unreadable", detail=type(exc).__name__))
            continue
        domain = report["qualification"]
        domains.append(domain if type(domain) is str else None)
        native_issues = [issue for issue in report["hold_reasons"] if issue not in
                         ("runtime_qualification_pending", "independent_client_inventory_not_joined",
                          "joint_release_profile_not_qualified", "no_recorded_native_verification_calls")]
        reasons.extend(native_issues)
        checksum_active += bool(snapshot.get("records"))
        for row in report["verification_calls"]:
            original = snapshot["records"][row["record_index"]]
            proof = original.get("proof") if type(original) is dict else None
            calls.append({**row, "checksum_snapshot_index": index,
                          "checksum_num_tokens": proof.get("num_tokens") if type(proof) is dict else None,
                          "process_id": snapshot["capture_context"].get("process_id")
                              if type(snapshot.get("capture_context")) is dict else None})
        unverified.extend(report["unverified_observations"])
    if binding_active != 1:
        reasons.append("unambiguous_active_binding_recorder_missing")
    if checksum_active != 1:
        reasons.append("unambiguous_active_checksum_recorder_missing")
    if (not domains or any(domain not in ("source_bound_candidate", "cpu_source_functions") for domain in domains)
            or len(set(domains)) != 1):
        reasons.append("capture_qualification_domain_unverified")
    return dict(issues=list(dict.fromkeys(reasons)), bindings=bindings, calls=calls, unverified=unverified,
                qualification=domains[0] if domains and len(set(domains)) == 1 else "unknown")


def evaluate(plan_raw, inventory_raw, captures_raw):
    """Reconcile all planned logical requests without creating runtime authority."""
    plan = _read(plan_raw, PLAN_SCHEMA, "checksum request plan")
    from . import prefix_cache_coverage as cache_coverage
    inventory = _read(inventory_raw, INVENTORY_SCHEMA, "checksum client inventory")
    captures = _read(captures_raw, CAPTURES_SCHEMA, "checksum client captures")
    planned, arms, phases = _planned(plan)
    issues = []
    inventory_identity = _identity(plan, inventory, plan_raw=plan_raw)
    captures_identity = _identity(plan, captures, plan_raw=plan_raw)
    if not inventory_identity:
        issues.append("client_inventory_run_binding_unverified")
    if inventory.get("completed") is not True:
        issues.append("client_inventory_completion_unverified")
    if not captures_identity:
        issues.append("capture_manifest_run_binding_unverified")
    raw_rows, raw_captures = inventory.get("requests"), captures.get("captures")
    if type(raw_rows) is not list:
        issues.append("client_inventory_row_list_unverified")
        raw_rows = []
    if len(raw_rows) > MAX_REQUESTS:
        raise io.EvidenceError("client inventory rows exceed the supported bound")
    if type(raw_captures) is not list:
        issues.append("capture_list_unverified")
        raw_captures = []
    if len(raw_captures) > MAX_ARMS:
        raise io.EvidenceError("client capture arms exceed the supported bound")
    indexed, unexpected_rows = {}, []
    planned_keys = {(row["arm_id"], row["slot"]) for row in planned}
    for index, raw in enumerate(raw_rows):
        key = (raw.get("arm_id"), raw.get("slot")) if type(raw) is dict else None
        if (key is None or type(key[0]) is not str or type(key[1]) is not int or key not in planned_keys):
            unexpected_rows.append({"record_index": index, "reason": "unplanned_or_malformed_client_row"})
        else:
            indexed.setdefault(key, []).append(raw)
    if unexpected_rows:
        issues.append("unexpected_client_inventory_rows")
    capture_by_arm, unexpected_captures, extra_facts = {}, [], []
    def tagged(raw, arm, index):
        facts = _capture(raw, plan, arm)
        for key in ("bindings", "calls", "unverified"):
            facts[key] = [{**event, "capture_index": index} for event in facts[key]]
        return facts
    for index, raw in enumerate(raw_captures):
        arm_id = raw.get("arm_id") if type(raw) is dict else None
        if type(arm_id) is str and arm_id in arms and arms[arm_id]["observer_enabled"] is True:
            capture_by_arm.setdefault(arm_id, []).append((index, raw))
        else:
            unexpected_captures.append(dict(record_index=index, reason="capture_outside_observed_arm"))
            orphan_arm = dict(arm_id=arm_id if _text(arm_id, 128) else "unknown-capture-" + str(index),
                engine_session_id=raw.get("engine_session_id") if type(raw) is dict
                    and _text(raw.get("engine_session_id"), 128) else "unknown-session")
            facts = tagged(raw, orphan_arm, index)
            facts["issues"].append("capture_outside_observed_arm")
            extra_facts.append((orphan_arm["arm_id"], facts))
    if unexpected_captures:
        issues.append("unexpected_engine_captures")
    arm_facts, cache_facts = {}, {}
    for name, arm in arms.items():
        if arm["observer_enabled"] is False:
            arm_facts[name] = dict(issues=[], bindings=[], calls=[], unverified=[], qualification="observer_disabled")
            continue
        choices = capture_by_arm.get(name, [])
        parsed = [tagged(raw, arm, index) for index, raw in choices]
        if len(parsed) == 1:
            facts = parsed[0]
        elif not parsed:
            facts = _capture(None, plan, arm)
        else:
            facts = dict(issues=list(dict.fromkeys(["ambiguous_engine_captures"] +
                    [issue for item in parsed for issue in item["issues"]])), qualification="unknown",
                bindings=[event for item in parsed for event in item["bindings"]],
                calls=[event for item in parsed for event in item["calls"]],
                unverified=[event for item in parsed for event in item["unverified"]])
        if not captures_identity:
            facts["issues"].append("capture_manifest_run_binding_unverified")
        arm_facts[name] = facts
        if plan.get("prefix_caching") is True:
            cache_facts[name] = cache_coverage.capture_facts(choices[0][1] if len(choices) == 1 else {}, plan, arm)
    all_facts = [(name, facts, False) for name, facts in arm_facts.items()] + [
        (name, facts, True) for name, facts in extra_facts]
    rows, joined_calls, joined_bindings = [], set(), set()
    for request in planned:
        row = dict(request)
        choices = indexed.get((row["arm_id"], row["slot"]), [])
        raw = choices[0] if len(choices) == 1 else None
        reasons, external = _client_row(raw, row, plan["model"])
        if len(choices) > 1:
            reasons.append("ambiguous_client_receipts")
        if not inventory_identity:
            reasons.append("client_inventory_run_binding_unverified")
        row.update(client_status="complete" if not reasons else "unknown", external_request_id=external,
                   internal_request_id=None, binding_status="unknown", native_result="unverified",
                   coverage="unknown", check_status="unverified", checksum_calls=[], unverified_observations=[], reasons=reasons)
        row["token_coverage"] = dict(prompt_tokens=row["prompt_tokens"], transferred_tokens_checked=None,
            decode_cached_tokens_unchecked=None, decode_locally_computed_tokens=None,
            prefill_cached_tokens=None, prefill_reused_cached_blocks="unknown")
        facts = arm_facts[row["arm_id"]]
        if row["observer_enabled"] is False:
            row["binding_status"] = "observer_disabled"
            row["reasons"].append("observer_disabled")
            rows.append(row)
            continue
        row["reasons"].extend(facts["issues"])
        bindings = [b for b in facts["bindings"] if b.get("external_request_id") == external] if external else []
        if len(bindings) != 1:
            row["reasons"].append("unambiguous_native_request_binding_missing")
        else:
            binding = bindings[0]
            internal = binding.get("internal_request_id")
            if not _text(internal) or not _integer(binding.get("sequence"), 1):
                row["reasons"].append("native_request_binding_identity_unverified")
            elif sum(b.get("internal_request_id") == internal for b in facts["bindings"]) != 1:
                row["reasons"].append("native_internal_request_binding_ambiguous")
            else:
                row["internal_request_id"] = internal
                if (binding.get("prompt_sha256") != row["prompt_sha256"]
                        or type(binding.get("prompt_tokens")) is not int or binding["prompt_tokens"] != row["prompt_tokens"]):
                    row["reasons"].append("native_prompt_to_client_binding_unverified")
                if row["client_status"] == "complete" and not row["reasons"]:
                    row["binding_status"] = "matched"
                    joined_bindings.add((row["arm_id"], binding.get("capture_index"), binding["binding_snapshot_index"], binding["sequence"]))
                row["unverified_observations"] = [event for event in facts["unverified"]
                    if event.get("request_id") == internal]
                row["reasons"].extend(event["reason"] for event in row["unverified_observations"]
                    if _text(event.get("reason")))
                calls = [(index, call) for index, call in enumerate(facts["calls"]) if call.get("request_id") == internal]
                row["checksum_calls"] = [call for _, call in calls]
                if row["binding_status"] == "matched":
                    for index, _ in calls:
                        joined_calls.add((row["arm_id"], index))
                if len(calls) != 1:
                    row["reasons"].append("unambiguous_native_checksum_call_missing")
                else:
                    call = calls[0][1]
                    row["native_result"] = call["native_result"]
                    if call["coverage"] != "complete":
                        row["reasons"].append(call["engine_pass_not_covered_reason"] or call["reason"])
                    if type(call.get("checksum_num_tokens")) is not int or call["checksum_num_tokens"] != row["prompt_tokens"]:
                        row["reasons"].append("client_prompt_to_checksum_span_unverified")
                    if not row["reasons"] and row["binding_status"] == "matched" and call["coverage"] == "complete":
                        row["coverage"] = "complete"
                        row["check_status"] = "passed" if call["native_result"] == "passed" else "mismatch"
                        if row["check_status"] == "mismatch":
                            row["reasons"].append("native_checksum_mismatch")
        if plan.get("prefix_caching") is True:
            row["token_coverage"], cache_issues = cache_coverage.join(cache_facts[row["arm_id"]], raw, row)
            row["reasons"].extend(cache_issues)
            if cache_issues:
                row["coverage"] = ("partial" if row["token_coverage"]["decode_cached_tokens_unchecked"]
                    and type(row["token_coverage"]["transferred_tokens_checked"]) is int
                    and row["token_coverage"]["transferred_tokens_checked"] > 0 else "unknown")
                row["check_status"] = "unverified"
        elif len(row["checksum_calls"]) == 1 and not row["checksum_calls"][0].get("recording_issues"):
            row["token_coverage"].update(row["checksum_calls"][0]["conditional_scope"]["token_coverage"])
        if row["native_result"] == "mismatch":
            row["reasons"].append("native_checksum_mismatch")
        row["reasons"] = list(dict.fromkeys(row["reasons"]))
        if not row["checksum_calls"]:
            row["reasons"].append("no_recorded_checksum_call_for_client_request")
        rows.append(row)
    unexpected_calls, unexpected_bindings = [], []
    for arm_id, facts, outside_plan in all_facts:
        for binding in facts["bindings"]:
            sequence = binding.get("sequence")
            key = (arm_id, binding.get("capture_index"), binding["binding_snapshot_index"], sequence) if _integer(sequence, 1) else None
            if outside_plan or key is None or key not in joined_bindings:
                unexpected_bindings.append(dict(arm_id=arm_id,
                    reason="native_binding_without_verified_client_join", binding=binding))
        for index, call in enumerate(facts["calls"]):
            if outside_plan or (arm_id, index) not in joined_calls:
                unexpected_calls.append(dict(arm_id=arm_id, call_index=index,
                    reason="checksum_call_without_verified_client_join", call=call))
    if unexpected_calls:
        issues.append("unexpected_checksum_calls")
    if unexpected_bindings:
        issues.append("unexpected_native_request_bindings")
    for facts in cache_facts.values():
        issues.extend(cache_coverage.unexpected(facts))
    complete_clients = sum(row["client_status"] == "complete" for row in rows)
    client_complete = (inventory_identity and inventory.get("completed") is True and not unexpected_rows
                       and complete_clients == len(rows))
    if not client_complete:
        issues.append("client_inventory_incomplete")
    native_gaps = Counter(call["engine_pass_not_covered_reason"] for _, facts, _ in all_facts
                         for call in facts["calls"] if call["engine_pass_not_covered_reason"])
    summary = dict(declared_requests=len(rows), on_requests=sum(row["observer_enabled"] for row in rows),
        off_requests=sum(not row["observer_enabled"] for row in rows), complete_client_requests=complete_clients,
        joined_requests=sum(row["binding_status"] == "matched" for row in rows),
        checksum_calls=sum(len(facts["calls"]) for _, facts, _ in all_facts), joined_checksum_calls=len(joined_calls),
        unexpected_checksum_calls=len(unexpected_calls), unexpected_binding_records=len(unexpected_bindings),
        binding_records=sum(len(facts["bindings"]) for _, facts, _ in all_facts),
        native_results=dict(Counter(call["native_result"] for _, facts, _ in all_facts for call in facts["calls"])),
        native_pass_requests=sum(row["native_result"] == "passed" for row in rows),
        verified_on_requests=sum(row["check_status"] == "passed" for row in rows),
        on_unverified_requests=sum(row["observer_enabled"] and row["check_status"] != "passed" for row in rows),
        engine_pass_not_covered=dict(native_gaps))
    if summary["on_unverified_requests"]:
        issues.append("required_checksum_coverage_incomplete")
    return dict(schema=REPORT_SCHEMA, title=TITLE, owner="Hari Krishna Kancharla", profile=PROFILE,
        source_revision=plan["source_revision"], runtime_qualified=False, release_decision="HOLD",
        inputs_sha256={"REQUEST_PLAN.json": io.digest(plan_raw), "CLIENT_INVENTORY.json": io.digest(inventory_raw),
                       "CAPTURES.json": io.digest(captures_raw)}, source_profile=source_profile(plan["source_revision"]),
        client_inventory=dict(status="complete" if client_complete else "unknown", planned_requests=len(rows),
            recorded_rows=len(raw_rows), complete_requests=complete_clients, unexpected_rows=unexpected_rows),
        summary=summary, requests=rows,
        captures=[dict(arm_id=name,
                       engine_session_id=arms[name]["engine_session_id"] if not outside_plan else None,
                       qualification=facts["qualification"], issues=facts["issues"],
                       binding_records=len(facts["bindings"]), checksum_calls=len(facts["calls"]),
                       unverified_observations=facts["unverified"], planned_arm=not outside_plan)
                  for name, facts, outside_plan in all_facts],
        unexpected_checksum_calls=unexpected_calls, unexpected_binding_records=unexpected_bindings,
        unexpected_captures=unexpected_captures,
        hold_reasons=list(dict.fromkeys(issues + ["runtime_qualification_pending", "joint_release_profile_not_qualified"])),
        limits=LIMITS)
