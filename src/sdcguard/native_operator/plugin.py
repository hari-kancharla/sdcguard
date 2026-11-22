"""Passive ordinary startup and close records. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading
import types

_CREATOR = os.getpid()
_STATE = None


def _codes(code, prefix=""):
    result = {}
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            name = prefix + item.co_name
            result[name] = item
            result.update(_codes(item, name + "."))
    return result


def _code_shape(code):
    constants = tuple(_code_shape(x) if isinstance(x, types.CodeType) else x for x in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_freevars, code.co_cellvars,
            code.co_flags, getattr(code, "co_exceptiontable", b""))


def _guard(cls, name, root, path, pins):
    method = cls.__dict__.get(name)
    if not inspect.isfunction(method):
        raise ValueError("lifecycle callback is not the original function")
    closure = dict(zip(method.__code__.co_freevars, method.__closure__ or ()))
    if any(name != "__class__" or cell.cell_contents is not cls for name, cell in closure.items()):
        raise ValueError("lifecycle callback has an unsupported closure")
    source = root / path
    raw = source.read_bytes()
    if (source.is_symlink() or hashlib.sha256(raw).hexdigest() != pins[path]["sha256"]
            or Path(method.__code__.co_filename).resolve() != source.resolve()):
        raise ValueError("lifecycle source identity differs: " + path)
    expected = _codes(compile(raw, str(source), "exec", dont_inherit=True)).get(method.__qualname__)
    if expected is None or _code_shape(expected) != _code_shape(method.__code__):
        raise ValueError("loaded lifecycle code differs: " + name)
    namespace = vars(sys.modules[cls.__module__])
    if method.__globals__ is not namespace:
        raise ValueError("lifecycle callback namespace differs")
    return method


class State:
    def __init__(self):
        from ..checksum_profiles import directory, get, BASELINE
        from .common import checkpoint
        self.checkpoint = checkpoint
        self.revision = get(os.environ.get("SDCGUARD_OPERATOR_REVISION", BASELINE))["revision"]
        self.bundle = directory(self.revision)
        self.context = dict(run_id=os.environ["OWNER_CHECKSUM_RUN_ID"],
            packet_sha256=os.environ["OWNER_CHECKSUM_PACKET_SHA256"], arm_id=os.environ["OWNER_CHECKSUM_ARM"],
            engine_session_id=os.environ["OWNER_CHECKSUM_ENGINE_SESSION"], process_id=os.getpid())
        for key, pattern in (("run_id", r"[0-9a-f]{32}"), ("packet_sha256", r"[0-9a-f]{64}"),
                             ("engine_session_id", r"[0-9a-f]{32}"), ("arm_id", r"observer-(?:on|off)")):
            if re.fullmatch(pattern, self.context[key]) is None:
                raise ValueError("unsupported startup context: " + key)
        self.role = os.environ["OWNER_CHECKSUM_ROLE"]
        if self.role not in ("prefill", "decode"):
            raise ValueError("unsupported startup role")
        self.enabled = os.environ.get("OWNER_CHECKSUM_ENABLED") == "1"
        self.out = Path(os.environ["OWNER_CHECKSUM_DIR"])
        if not self.out.is_absolute():
            raise ValueError("startup output must be absolute")
        self.out.mkdir(parents=True, exist_ok=True)
        self.errors = []
        self.written = set()
        self.checksums = self.bindings = self.cache = None
        self.transport_events = []
        self.transport_lock = threading.RLock()
        self.transport_policy = None
        self.closes = {which: dict(lock=threading.RLock(), native_depth=0, native_failed=False,
            native_error=None, failure_reported=False, active=False, attempted=False, snapshot=None, done=False)
            for which in ("checksum", "bindings", "cache")}

    def problem(self, error):
        message = type(error).__name__ + ": " + str(error)[:512]
        if message not in self.errors and len(self.errors) < 16:
            self.errors.append(message)
        try:
            self.checkpoint(self.out / ("errors-" + str(os.getpid()) + ".json"),
                dict(context=self.context, role=self.role, errors=list(self.errors)))
        except Exception:
            pass

    def live(self):
        if os.getpid() != _CREATOR or self.context["process_id"] != os.getpid():
            self.problem(ValueError("startup recorder inherited across a process boundary"))
            return False
        return True

    def once(self, name, value):
        if name in self.written or not self.live():
            return
        try:
            self.checkpoint(self.out / (name + "-" + str(os.getpid()) + ".json"), value)
            self.written.add(name)
        except Exception as error:
            self.problem(error)

    def transport(self, event):
        from .transport import RECEIPT_SCHEMA, profile, validate_events
        if not self.live():
            raise ValueError("transport recorder crossed a process boundary")
        with self.transport_lock:
            try:
                if len(self.transport_events) >= 128:
                    raise ValueError("transport initialization event budget exceeded")
                copied = json.loads(json.dumps(event, sort_keys=True, allow_nan=False))
                self.transport_events.append(copied)
                from .cleanup import processes
                row = processes().get(os.getpid())
                if row is None:
                    raise ValueError("transport process identity is unavailable")
                identity = dict(pid=os.getpid(), **{k: row[k] for k in ("start", "group", "session")})
                value = dict(schema=RECEIPT_SCHEMA, context=dict(self.context), role=self.role,
                    process_identity=identity,
                    profile=profile(), events=list(self.transport_events),
                    status="PASS" if validate_events(self.transport_events) else "UNKNOWN")
                self.checkpoint(self.out / ("transport-" + str(os.getpid()) + ".json"), value)
            except Exception as error:
                self.problem(error)
                raise

    def startup(self, scheduler):
        if not self.live():
            return
        try:
            from .config import MODEL, MODEL_REVISION, SNAPSHOT
            from sdcguard.checksum_client import source_profile
            from vllm import envs
            cfg = scheduler.vllm_config
            parallel, cache, schedule, transfer = (cfg.parallel_config, cfg.cache_config,
                cfg.scheduler_config, cfg.kv_transfer_config)
            groups = scheduler.kv_cache_config.kv_cache_groups
            specs = [dict(type=type(group.kv_cache_spec).__name__,
                block_size=group.kv_cache_spec.block_size,
                sliding_window=getattr(group.kv_cache_spec, "sliding_window", "unrecorded"),
                attention_chunk_size=getattr(group.kv_cache_spec, "attention_chunk_size", "unrecorded"),
                non_causal=getattr(group.kv_cache_spec, "non_causal", "unrecorded")) for group in groups]
            observed = dict(**self.context, role=self.role, source_revision=self.revision,
                model=MODEL, model_revision=MODEL_REVISION, actual_model=cfg.model_config.model,
                actual_tokenizer=cfg.model_config.tokenizer, snapshot=SNAPSHOT, source_profile=source_profile(self.revision),
                checksum_enabled=transfer.enable_kv_checksum,
                request_id_randomization_enabled=envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is False,
                flashinfer_sampler_enabled=envs.VLLM_USE_FLASHINFER_SAMPLER,
                multiprocess_method=envs.VLLM_WORKER_MULTIPROC_METHOD,
                model_runner_v2=cfg.use_v2_model_runner,
                parallel=dict(tp=parallel.tensor_parallel_size, pp=parallel.pipeline_parallel_size,
                    dp=parallel.data_parallel_size, dcp=parallel.decode_context_parallel_size,
                    pcp=parallel.prefill_context_parallel_size),
                group_specs=specs, gpu_blocks=scheduler.kv_cache_config.num_blocks,
                block_size=cache.block_size, max_model_len=cfg.model_config.max_model_len,
                max_num_seqs=schedule.max_num_seqs, prefix_caching=cache.enable_prefix_caching,
                chunked_prefill=schedule.enable_chunked_prefill, async_scheduling=schedule.async_scheduling,
                dtype=str(cfg.model_config.dtype), kv_dtype=cache.cache_dtype,
                speculative_disabled=cfg.speculative_config is None,
                connector=transfer.kv_connector, transfer_role=transfer.kv_role,
                transfer_extra=transfer.kv_connector_extra_config,
                transport_profile=self.transport_policy,
                source_identity="exact pinned local source and loaded callback code; no execution attestation")
            from ..checksum_profiles import CANDIDATE
            observed["hisparse_host_num_blocks"] = getattr(scheduler.kv_cache_config, "hisparse_host_num_blocks", "unrecorded")
            observed["defer_async_loads"] = (getattr(scheduler, "defer_async_loads", "unrecorded")
                if self.revision == CANDIDATE else "not_in_source_profile")
            self.once("startup", observed)
        except Exception as error:
            self.problem(error)

    def begin_shutdown(self, which):
        if not self.live():
            return False
        record = self.closes[which]
        with record["lock"]:
            record["native_depth"] += 1
        return True

    def end_shutdown(self, which, entered, original_completed, original_error):
        if not entered or not self.live():
            return
        record = self.closes[which]
        with record["lock"]:
            record["native_depth"] -= 1
            if record["done"]:
                return
            if original_completed is not True or original_error is not None:
                if not record["native_failed"]:
                    record["native_failed"], record["native_error"] = True, original_error
            if record["native_depth"] == 0:
                self.close(which, not record["native_failed"], record["native_error"])

    def close(self, which, original_completed, original_error):
        if not self.live():
            return
        if which not in self.closes:
            self.problem(ValueError("unsupported native shutdown hook"))
            return
        record = self.closes[which]
        with record["lock"]:
            if record["done"]:
                return
            if original_completed is not True or original_error is not None:
                if not record["native_failed"]:
                    record["native_failed"], record["native_error"] = True, original_error
            if record["active"]:
                if record["native_failed"] and not record["failure_reported"]:
                    record["failure_reported"] = True
                    self.problem(RuntimeError("native " + which + " shutdown did not complete: " + str(record["native_error"])))
                return
            record["active"] = True
            try:
                if record["native_failed"] and not record["failure_reported"]:
                    record["failure_reported"] = True
                    self.problem(RuntimeError("native " + which + " shutdown did not complete: " + str(record["native_error"])))
                if record["native_depth"]:
                    return
                observer = {"checksum": self.checksums, "bindings": self.bindings, "cache": self.cache}[which]
                name = which + "-disabled" if observer is None else which
                if observer is not None and not record["attempted"]:
                    record["attempted"] = True
                    try:
                        snapshot = observer.close()
                        if type(snapshot) is not dict:
                            raise ValueError("native close observer returned an unsupported snapshot")
                        if which == "checksum":
                            snapshot["capture_context"] = dict(self.context)
                        record["snapshot"] = snapshot
                    except Exception as error:
                        self.problem(error)
                if observer is not None and record["snapshot"] is None:
                    return
                value = dict(context=dict(self.context), role=self.role,
                    original_shutdown_completed=not record["native_failed"],
                    original_shutdown_error=record["native_error"], recorder_errors=list(self.errors))
                if observer is None:
                    value["observer_enabled"] = False
                else:
                    value.update(schema="sdcguard.owner-checksum-close.v1", snapshot=record["snapshot"])
                self.once(name, value)
                record["done"] = name in self.written
            finally:
                record["active"] = False


def register():
    global _STATE
    if "OWNER_CHECKSUM_RUN_ID" not in os.environ:
        return
    if _STATE is not None:
        if not _STATE.live():
            return
        return
    state = State()
    _STATE = state
    try:
        from .transport import install_policy
        state.transport_policy = install_policy(state.transport)
        from .config import ENGINE_ROOT
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.async_llm import AsyncLLM
        root = Path(ENGINE_ROOT)
        pins = json.loads((state.bundle / "artifacts/ENGINE_BUILD_LOCK.json").read_bytes())["sources"]
        original_init = _guard(Scheduler, "__init__", root, "vllm/v1/core/sched/scheduler.py", pins)
        original_scheduler_close = _guard(Scheduler, "shutdown", root, "vllm/v1/core/sched/scheduler.py", pins)
        original_api_close = _guard(AsyncLLM, "shutdown", root, "vllm/v1/engine/async_llm.py", pins)
        if state.enabled:
            from sdcguard.kv_checksum_observer import ChecksumObserver
            from sdcguard.checksum_client import RequestBindingObserver
            from vllm.distributed.kv_transfer.kv_checksum.scheduler import KVChecksumScheduler
            from vllm.v1.engine.input_processor import InputProcessor
            if state.role == "decode":
                state.checksums = ChecksumObserver(revision=state.revision)
                state.checksums.install(KVChecksumScheduler, source_root=root)
            context = {k: v for k, v in state.context.items() if k != "process_id"}
            state.bindings = RequestBindingObserver(**context, revision=state.revision)
            state.bindings.install(InputProcessor, source_root=root)
            from sdcguard.prefix_cache_observer import PrefillCacheObserver
            from vllm.v1.request import Request
            state.cache = PrefillCacheObserver(**context, role=state.role, revision=state.revision)
            state.cache.install(Request, Scheduler, source_root=root)

        @functools.wraps(original_init)
        def initialized(self, *args, **kwargs):
            result = original_init(self, *args, **kwargs)
            state.startup(self)
            return result

        def closing(original, *kinds):
            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                entered = {}
                for which in kinds:
                    try:
                        entered[which] = state.begin_shutdown(which)
                    except Exception as raised:
                        state.problem(raised)
                completed, error = False, None
                try:
                    result = original(self, *args, **kwargs)
                    completed = True
                    return result
                except BaseException as raised:
                    error = type(raised).__name__
                    raise
                finally:
                    for which in kinds:
                        try:
                            state.end_shutdown(which, entered.get(which, False), completed, error)
                        except Exception as raised:
                            state.problem(raised)
            return wrapped

        Scheduler.__init__ = initialized
        Scheduler.shutdown = closing(original_scheduler_close, "checksum", "cache")
        AsyncLLM.shutdown = closing(original_api_close, "bindings")
        state.once("registered", dict(context=state.context, role=state.role,
            observer_enabled=state.enabled, native_shutdown_hook=True, creator_pid=_CREATOR,
            transport_profile=state.transport_policy))
    except Exception as error:
        state.problem(error)
