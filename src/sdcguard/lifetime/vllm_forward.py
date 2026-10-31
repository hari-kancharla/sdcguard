"""Opt-in observations of eager vLLM forward inputs.

This records metadata at a Python model boundary. It never observes physical
attention reads, establishes ownership, or starts an inference engine. Device
copies can synchronize and affect timing. Original engine exceptions propagate.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import queue
import threading
import uuid
import weakref

from . import forward_inputs as contract
from .. import config_inputs

_INSTALLATION = None
_QUEUE_BYTES = 8 * 1024 * 1024
_QUEUE_RECORDS = 32


class ObservationGap(ValueError):
    """A bounded diagnostic authored by the observer, without engine payloads."""


def _reason(exc):
    return str(exc)[:512] if isinstance(exc, ObservationGap) else type(exc).__name__


def _uuid():
    return uuid.uuid4().hex


def _open_directory(value, *, create=False):
    """Anchor every component before creating an exclusive recording file."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ObservationGap("secure recorder paths are unavailable on this platform")
    path = Path(value).absolute()
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part in ("", ".", ".."):
                raise ObservationGap("recording path traversal is unsupported")
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return path, fd
    except BaseException:
        os.close(fd)
        raise


def _directory(value):
    path, fd = _open_directory(value, create=True)
    os.close(fd)
    return path


class JsonlRecorder:
    """Bounded process-local writer. A missing or unhealthy footer is a gap."""

    def __init__(self, directory, engine, profile, *, queue_bytes=_QUEUE_BYTES):
        self.pid = os.getpid()
        self.session = _uuid()
        self.header = dict(schema=1, kind="header", session=self.session, seq=0,
                           process_incarnation=self.session, engine=engine, profile=profile)
        path, parent = _open_directory(directory, create=True)
        self.path = path / (self.session + ".jsonl")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            fd = os.open(self.path.name, flags, 0o600, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            self._file = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=_QUEUE_RECORDS)
        self._queue_limit = queue_bytes
        self._queued_bytes = 0
        self._total_bytes = 0
        self._total_nodes = 0
        self._total_cells = 0
        self._seq = 1
        self._snapshots = 0
        self._dropped = 0
        self._errors = 0
        self._closing = False
        self._failed = False
        self._closed = False
        try:
            raw, nodes = self._encode(self.header)
            self._file.write(raw)
            self._file.flush()
            self._total_bytes = len(raw)
            self._total_nodes = nodes
        except Exception:
            self._file.close()
            raise
        self._thread = threading.Thread(target=self._write_loop, daemon=True,
                                        name="forward-input-writer")
        try:
            self._thread.start()
        except BaseException:
            self._file.close()
            raise

    @staticmethod
    def _encode(record):
        raw, nodes = contract._bounded(record)
        if len(raw) + 1 > contract.MAX_RECORD_BYTES:
            raise ObservationGap("forward record exceeds byte limit")
        return raw + b"\n", nodes

    def emit(self, kind, *, tensor_cells=0, **facts):
        if os.getpid() != self.pid:
            return False
        try:
            with self._lock:
                if self._closing or self._failed:
                    self._dropped += 1
                    return False
                record = dict(schema=1, session=self.session, seq=self._seq, kind=kind, **facts)
                raw, nodes = self._encode(record)
                if (self._seq >= contract.MAX_RECORDS - 1
                        or self._queued_bytes + len(raw) > self._queue_limit
                        or self._total_bytes + len(raw) > contract.MAX_TOTAL_BYTES - 4096
                        or self._total_nodes + nodes > contract.MAX_TOTAL_NODES - 128
                        or self._total_cells + tensor_cells > contract.MAX_TOTAL_CELLS):
                    self._dropped += 1
                    return False
                try:
                    self._queue.put_nowait((raw, kind))
                except queue.Full:
                    self._dropped += 1
                    return False
                self._seq += 1
                self._queued_bytes += len(raw)
                self._total_bytes += len(raw)
                self._total_nodes += nodes
                self._total_cells += tensor_cells
                return True
        except Exception:
            with self._lock:
                self._errors += 1
                self._dropped += 1
            return False

    def gap(self, reason, forward_id=None):
        if os.getpid() != self.pid:
            return
        with self._lock:
            self._errors += 1
        self.emit("gap", reason=str(reason)[:512], forward_id=forward_id)

    def _write_loop(self):
        try:
            while True:
                try:
                    raw, kind = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._closing:
                        break
                    continue
                try:
                    if not self._failed:
                        self._file.write(raw)
                        if kind == "snapshot":
                            self._snapshots += 1
                except Exception:
                    with self._lock:
                        self._failed = True
                        self._errors += 1
                finally:
                    with self._lock:
                        self._queued_bytes -= len(raw)
                    self._queue.task_done()
            with self._lock:
                footer = dict(schema=1, session=self.session, seq=self._seq, kind="footer",
                              snapshots=self._snapshots, dropped=self._dropped, errors=self._errors,
                              healthy=not (self._failed or self._errors or self._dropped))
            if not self._failed:
                self._file.write(self._encode(footer)[0])
                self._file.flush()
        except Exception:
            self._failed = True
        finally:
            self._file.close()
            self._closed = True

    def close(self, timeout=1.0):
        if os.getpid() != self.pid:
            return False
        with self._lock:
            self._closing = True
        self._thread.join(timeout=max(0.0, min(float(timeout), 5.0)))
        return self._closed


def inspect_source(root, version):
    """Inspect selected local source bytes, not remote or binary authenticity."""
    root = Path(root).absolute()
    got = {}
    for name, expected in contract.source_pins().items():
        raw = config_inputs._read_regular(root, Path(name).parts, 2 * 1024 * 1024)
        got[name] = hashlib.sha256(raw).hexdigest()
        if got[name] != expected:
            raise ObservationGap("engine source pin mismatch: " + name)
    if version != "0.32.0":
        raise ObservationGap("unsupported engine version")
    return got


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ObservationGap("unsupported " + label)


def _positive(value, maximum, label):
    if type(value) is not int or not 0 < value <= maximum:
        raise ObservationGap("invalid " + label)
    return value


def _no_value(obj, names):
    for name in names:
        if getattr(obj, name) is not None:
            raise ObservationGap("unsupported " + name)


def _profile(runner, types, *, model_binding=None):
    """Read effective runtime state. Missing fields are unsupported, never defaults."""
    model = runner.model_config
    parallel = runner.parallel_config
    cache = runner.kv_cache_config
    _equal(runner.vllm_config.use_v2_model_runner, True, "model runner")
    from .. import model_reference
    reference = model_reference.observe(model, model_binding)
    _equal(model.enforce_eager, True, "eager execution")
    _equal(model.architectures, ["Qwen3ForCausalLM"], "architecture")
    if type(runner.model) is not types["model"]:
        raise ObservationGap("unsupported effective model class")
    if type(runner.model_state) is not types["model_state"]:
        raise ObservationGap("unsupported model-state preparation")
    if runner.compilation_config.mode is not types["compile_none"]:
        raise ObservationGap("compiled model execution is unsupported")
    _equal(runner.cache_config.enable_prefix_caching, False, "prefix caching")
    _equal(runner.cache_config.cache_dtype, "auto", "cache dtype")
    _equal(parallel.enable_dbo, False, "dual batch overlap")
    _equal(runner.is_encoder_decoder, False, "encoder decoder")
    _equal(runner.supports_mm_inputs, False, "multimodal input")
    _equal(cache.kv_cache_layout, "LBHNC", "cache layout")
    _no_value(runner, ("speculative_config", "lora_config", "ubatch_runner", "pcp_manager"))
    _no_value(model, ("rswa_window",))
    fields = {"tp": "tensor_parallel_size", "pp": "pipeline_parallel_size",
              "dcp": "decode_context_parallel_size", "pcp": "prefill_context_parallel_size",
              "dp": "data_parallel_size"}
    for key, attr in fields.items():
        _equal(getattr(parallel, attr), 1, attr)
    if type(cache.kv_cache_groups) is not list or len(cache.kv_cache_groups) != 1:
        raise ObservationGap("unsupported cache groups")
    group = cache.kv_cache_groups[0]
    spec = group.kv_cache_spec
    if type(spec) is not types["spec"]:
        raise ObservationGap("unsupported cache specification")
    _equal(group.is_eagle_group, False, "draft cache")
    _equal(group.host_resident, False, "host cache")
    _equal(spec.non_causal, False, "noncausal attention")
    _equal(spec.tokens_per_state, 1, "compressed cache")
    _no_value(spec, ("sliding_window", "attention_chunk_size", "page_size_padded",
                     "num_head_slots", "state_content_bytes"))
    if spec.kv_quant_mode != types["quant_none"]:
        raise ObservationGap("unsupported cache quantization")
    if type(group.layer_names) is not list:
        raise ObservationGap("layer names are not a concrete list")
    _positive(len(group.layer_names), contract.MAX_LAYERS, "layer count")
    names = list(group.layer_names)
    _equal(model.hf_text_config.num_hidden_layers, len(names), "complete layer set")
    if len(set(names)) != len(names):
        raise ObservationGap("duplicate attention layer")
    _positive(spec.block_size, contract.MAX_CONTEXT, "block size")
    _positive(cache.num_blocks, 2**31 - 1, "pool capacity")
    _equal(runner.kernel_block_sizes, [spec.block_size], "kernel block size")
    layers = runner.compilation_config.static_forward_context
    if set(layers) != set(names):
        raise ObservationGap("incomplete static attention layer set")
    for name in names:
        _layer(layers[name], spec, cache.num_blocks, types)
    profile = dict(contract.SUPPORTED_PROFILE)
    profile.update(model=model.model, model_revision=model.revision, layer_names=names)
    if reference is not None:
        profile["model_reference"] = reference
    return profile


def _layer(layer, spec, num_blocks, types):
    if type(layer) is not types["attention"] or type(layer.impl) is not types["impl"]:
        raise ObservationGap("unsupported attention implementation")
    impl = layer.impl
    _equal(layer.attn_backend.get_name(), "FLASH_ATTN", "backend")
    _equal(layer.attn_type, types["decoder"], "attention type")
    _equal(impl.attn_type, types["decoder"], "implementation attention type")
    _equal(impl.dcp_world_size, 1, "implementation context parallelism")
    _equal(impl.vllm_flash_attn_version, 3, "FlashAttention selector")
    _equal(impl.sliding_window, (-1, -1), "attention window")
    _equal(layer.mm_prefix_clamp_sliding_window, False, "prefix mask")
    _equal(layer.kv_cache_dtype, "auto", "layer cache dtype")
    _no_value(layer, ("kv_sharing_target_layer_name", "sliding_window"))
    _no_value(impl, ("kv_sharing_target_layer_name", "alibi_slopes", "sinks"))
    tensor = layer.kv_cache
    torch = types["torch"]
    if type(tensor) is not torch.Tensor or tensor.device.type != "cuda":
        raise ObservationGap("cache is not a device tensor")
    expected = (num_blocks, spec.num_kv_heads, spec.block_size, spec.head_size + spec.head_size_v)
    if tuple(tensor.shape) != expected or not tensor.is_contiguous() or tensor.dtype != spec.dtype:
        raise ObservationGap("cache storage layout differs from the declared pool")
    if tensor.dtype not in (torch.float16, torch.bfloat16):
        raise ObservationGap("unsupported cache storage dtype")
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))


def _copy_integer(tensor, shape, types, device):
    torch = types["torch"]
    if (type(tensor) is not torch.Tensor or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype not in (torch.int32, torch.int64) or tensor.device != device):
        raise ObservationGap("unsupported metadata tensor")
    # Blocking host copy orders after preceding work on the current stream.
    # copy=True also prevents aliases in CPU stand-in tests.
    return tensor.detach().to(device="cpu", copy=True, non_blocking=False).tolist()


def _vector(tensor, count, maximum, types, device, copies):
    torch = types["torch"]
    if (type(tensor) is not torch.Tensor or tensor.ndim != 1
            or not count <= tensor.shape[0] <= maximum or tensor.device != device
            or tensor.dtype not in (torch.int32, torch.int64)):
        raise ObservationGap("unsupported metadata vector")
    key = (id(tensor), count)
    if key not in copies:
        copies[key] = (tensor, _copy_integer(tensor[:count], (count,), types, device))
    original, values = copies[key]
    if original is not tensor:
        raise ObservationGap("metadata identity changed during snapshot")
    return values


def _table(tensor, rows, width, types, device, copies):
    key = (id(tensor), rows, width)
    if key not in copies:
        copies[key] = (tensor, _copy_integer(tensor[:rows, :width], (rows, width), types, device))
    original, values = copies[key]
    if original is not tensor:
        raise ObservationGap("metadata identity changed during snapshot")
    return values


class _State:
    def __init__(self, runner, installation):
        self.owner = installation
        self.pid = os.getpid()
        self.runner = weakref.ref(runner)
        self.model = weakref.ref(runner.model)
        self.pool = _uuid()
        self.counter = 0
        self.busy = threading.Lock()
        self.closing = False
        self.model_binding = installation.model_binding_for(runner)
        self.profile = _profile(runner, installation.types, model_binding=self.model_binding)
        self.writer = JsonlRecorder(installation.directory, installation.engine, self.profile)
        try:
            valid = contract.validate_header(self.writer.header)
            if valid["observation_status"] != "observed":
                raise ObservationGap("unsupported effective header")
            self.pool_object = runner.kv_cache_config
            self.tables_object = runner.block_tables
            self.layer_objects = dict(runner.compilation_config.static_forward_context)
            spec = self.pool_object.kv_cache_groups[0].kv_cache_spec
            self.cache_identity = {name: _layer(layer, spec, self.pool_object.num_blocks, installation.types)
                                   for name, layer in self.layer_objects.items()}
            if runner.model._forward_pre_hooks:
                raise ObservationGap("other model pre-hooks are not supported")
            callback = self.pre_hook
            self.handle = runner.model.register_forward_pre_hook(callback)
            if installation.bridge is not None:
                self.hook_callback = callback
        except Exception:
            self.writer.gap("recorder initialization incomplete")
            self.writer.close()
            raise

    def error(self, reason, forward_id=None):
        try:
            self.writer.gap(reason, forward_id)
        except Exception:
            pass

    def close(self):
        self.closing = True
        try:
            if self.busy.locked():
                self.error("recorder closed during runner invocation")
            self.handle.remove()
        except Exception:
            self.error("model pre-hook removal failed")
        finally:
            # Hook cleanup must not prevent a bounded attempt to finish the
            # recording. A footer says nothing about device completion.
            try:
                closed = self.writer.close()
            except Exception:
                self.error("recorder closure failed")
                closed = False
        return closed

    def pre_hook(self, module, args):
        if os.getpid() != self.pid or self.closing:
            return None
        active = getattr(self.owner.local, "active", None)
        if active is None or active.get("state") is not self:
            return None
        try:
            if active["seen"]:
                raise ObservationGap("multiple model calls in one invocation")
            active["seen"] = True
            if self.owner.bridge is not None:
                self.owner.bridge.begin(active.get("assignment"), self, active)
            snapshot = self.snapshot(active)
            if self.owner.bridge is not None:
                self.owner.bridge.copied(active.get("assignment"), self, snapshot)
        except Exception as exc:
            if self.owner.bridge is not None:
                self.owner.bridge.abandon(active.get("assignment"), self.runner(), _reason(exc))
            self.error(_reason(exc), active["forward_id"])
            return None
        try:
            valid = contract.validate_snapshot(self.writer.header, snapshot)
            if valid["observation_status"] != "observed":
                raise ObservationGap("snapshot does not satisfy the metadata contract")
            self.writer.emit("snapshot", tensor_cells=valid["tensor_cells"], **{k: v for k, v in snapshot.items()
                                           if k not in ("kind", "seq", "schema", "session")})
        except Exception as exc:
            self.error(_reason(exc), active["forward_id"])
        return None

    def snapshot(self, active):
        if active.get("poisoned", False):
            raise ObservationGap("batch preparation observation is incomplete")
        runner = self.runner()
        if runner is None or self.model() is not runner.model:
            raise ObservationGap("model identity changed")
        if self.pool_object is not runner.kv_cache_config or self.tables_object is not runner.block_tables:
            raise ObservationGap("cache pool changed without recorder initialization")
        if _profile(runner, self.owner.types, model_binding=self.model_binding) != self.profile:
            raise ObservationGap("effective profile changed")
        if len(runner.model._forward_pre_hooks) != 1:
            raise ObservationGap("model pre-hook coverage changed")
        ctx = self.owner.get_context()
        if ctx.cudagraph_runtime_mode is not self.owner.types["graph_none"]:
            raise ObservationGap("graph execution is unsupported")
        _no_value(ctx, ("ubatch_slices", "dp_metadata"))
        if type(ctx.attn_metadata) is not dict or type(ctx.slot_mapping) is not dict:
            raise ObservationGap("unsupported forward metadata container")
        names = self.profile["layer_names"]
        if set(ctx.attn_metadata) != set(names) or set(ctx.slot_mapping) != set(names):
            raise ObservationGap("forward layer coverage differs")
        pending = active.get("pending")
        if pending is None:
            raise ObservationGap("forward has no associated prepared batch")
        requests, count = pending
        n = len(requests)
        torch = self.owner.types["torch"]
        if torch.cuda.current_stream(runner.device) != active["stream"]:
            raise ObservationGap("forward stream changed")
        # Padding of real query positions would change the row interpretation.
        if ctx.is_padding is not None:
            pad = ctx.is_padding
            if (type(pad) is not torch.Tensor or pad.dtype != torch.bool
                    or pad.device != runner.device or pad.ndim != 1 or pad.shape[0] < count
                    or pad.shape[0] > contract.MAX_TOKENS):
                raise ObservationGap("unsupported padding mask")
            if any(pad[:count].detach().to(device="cpu", copy=True, non_blocking=False).tolist()):
                raise ObservationGap("padding inside the real query range")
        layers = []
        cells = 0
        # The single-group builder often shares tensor objects across layers.
        # Reuse only copies of the identical object and identical slice in this
        # invocation. Equal values or assumed group membership are insufficient.
        copies = {}
        spec = self.pool_object.kv_cache_groups[0].kv_cache_spec
        for name in names:
            layer = ctx.no_compile_layers[name]
            if layer is not self.layer_objects[name]:
                raise ObservationGap("effective attention layer changed")
            identity = _layer(layer, spec, self.pool_object.num_blocks, self.owner.types)
            if identity != self.cache_identity[name]:
                raise ObservationGap("physical cache object changed")
            meta = ctx.attn_metadata[name]
            if type(meta) is not self.owner.types["metadata"]:
                raise ObservationGap("unsupported metadata class")
            _equal(meta.num_actual_tokens, count, "actual metadata query count")
            _equal(meta.causal, True, "causal mask")
            _no_value(meta, ("mm_prefix_query_range_tensor", "rswa_prefix_lens", "rswa_window",
                             "rswa_window_tensor", "dcp_context_kv_lens"))
            # The ordinary non-DCP metadata builder sets this to integer zero.
            _equal(meta.max_dcp_context_kv_len, 0, "DCP context length")
            table = meta.block_table
            if (type(table) is not torch.Tensor or table.ndim != 2 or table.shape[0] < n
                    or table.shape[0] > contract.MAX_REQUESTS or table.shape[1] > contract.MAX_BLOCKS_PER_ROW):
                raise ObservationGap("block table bounds unavailable")
            seq = _vector(meta.seq_lens, n, contract.MAX_REQUESTS, self.owner.types, runner.device, copies)
            lengths = []
            for value in seq:
                _positive(value, contract.MAX_CONTEXT, "sequence length")
                lengths.append((value + spec.block_size - 1) // spec.block_size)
            width = max(lengths)
            cells += n * width + n + n + 1 + count
            if cells > contract.MAX_SNAPSHOT_CELLS:
                raise ObservationGap("snapshot cell limit exceeded")
            if width > table.shape[1]:
                raise ObservationGap("block table does not cover the sequence")
            rows = _table(table, n, width, self.owner.types, runner.device, copies)
            starts = _vector(meta.query_start_loc, n + 1, contract.MAX_REQUESTS + 1,
                             self.owner.types, runner.device, copies)
            _positive(meta.max_query_len, contract.MAX_TOKENS, "maximum query length")
            _positive(meta.max_seq_len, contract.MAX_CONTEXT, "maximum sequence length")
            if meta.max_query_len < max(b - a for a, b in zip(starts, starts[1:])) or meta.max_seq_len < max(seq):
                raise ObservationGap("metadata length bounds contradict the actual rows")
            slots = _vector(ctx.slot_mapping[name], count, contract.MAX_TOKENS,
                            self.owner.types, runner.device, copies)
            layers.append(dict(name=name, block_size=spec.block_size, pool_num_blocks=self.pool_object.num_blocks,
                               block_table=[row[:size] for row, size in zip(rows, lengths)],
                               seq_lens=seq, query_start_loc=starts, write_slots=slots, causal=True,
                               window=None, layout="LBHNC", backend="FLASH_ATTN", backend_version="3"))
        return dict(schema=1, session=self.writer.session, seq=1, kind="snapshot",
                    forward_id=active["forward_id"], snapshot_id=_uuid(), pool_incarnation=self.pool,
                    requests=requests, num_tokens=count, snapshot_copy_complete=True,
                    forward_device_complete=None, observation_point="model_entry", layers=layers)


class Installation:
    """Own only the wrappers and hooks installed by this observer."""

    def __init__(self, runner_class, directory, engine, types, get_context, *, assignment_factory=None):
        self.runner_class = runner_class
        self.directory = directory
        self.engine = engine
        self.types = types
        self.get_context = get_context
        self.pid = os.getpid()
        self.states = weakref.WeakKeyDictionary()
        self.model_bindings = weakref.WeakKeyDictionary()
        self.unsupported = weakref.WeakKeyDictionary()
        self.local = threading.local()
        self.wrappers = {}
        self.assignment_factory = assignment_factory
        self.bridge = None
        self.bridge_error = None
        if assignment_factory is not None:
            try:
                from .forward_bridge import ForwardBridge
                self.bridge = ForwardBridge(self, assignment_factory)
            except Exception as exc:
                self.bridge_error = "assignment setup: " + type(exc).__name__

    def bridge_coverage(self, state):
        """Check the existing hook and wrappers without importing an engine."""
        try:
            model = state.model()
            global_hooks = self.types["global_hook_registry"]._global_forward_pre_hooks
            return (self.pid == os.getpid() and not state.closing and model is not None
                    and len(global_hooks) == 0 and len(model._forward_pre_hooks) == 1
                    and next(iter(model._forward_pre_hooks.values())) is state.hook_callback
                    and len(self.wrappers) == 4 and all(
                        getattr(self.runner_class, name, None) is wrapper
                        for name, (_, wrapper) in self.wrappers.items()))
        except Exception:
            return False

    def model_binding_for(self, runner):
        """Own setup file proof for this actual runner in this process only."""
        from .. import model_reference
        self._pid()
        if runner not in self.model_bindings:
            self.model_bindings[runner] = model_reference.bind(runner.model_config)
        binding = self.model_bindings[runner]
        model_reference.observe(runner.model_config, binding)
        return binding

    def _pid(self):
        if self.pid == os.getpid():
            return
        # Do not touch inherited writer threads, locks, files or queues.
        for state in list(self.states.values()):
            try:
                state.handle.remove()
            except Exception:
                pass
        self.pid = os.getpid()
        self.states = weakref.WeakKeyDictionary()
        self.model_bindings = weakref.WeakKeyDictionary()
        self.unsupported = weakref.WeakKeyDictionary()
        self.local = threading.local()
        # No factory call or native association can be reconstructed in a child.
        # Leave inherited observer queues, files and locks completely alone.
        self.bridge = None
        if self.assignment_factory is not None:
            self.bridge_error = "assignment initialization was not observed in this process"

    def _end_pool(self, runner):
        state = self.states.pop(runner, None)
        if state is not None:
            state.close()

    def _start_pool(self, runner, *, reset=False):
        try:
            self._pid()
            if not reset and runner in self.unsupported:
                return
            self.unsupported.pop(runner, None)
            self._end_pool(runner)
            self.states[runner] = _State(runner, self)
        except Exception as exc:
            self.unsupported[runner] = True
            _diagnostic(self.directory, self.engine, _reason(exc))

    def attach(self):
        originals = {name: getattr(self.runner_class, name) for name in
                     ("initialize_kv_cache", "prepare_attn", "execute_model", "shutdown")}
        if any(getattr(fn, "_sdcguard_forward", False) for fn in originals.values()):
            raise ObservationGap("forward wrappers already present")
        # Validate introspection before the first mutation.
        execute_signature = inspect.signature(originals["execute_model"])
        initialize_signature = inspect.signature(originals["initialize_kv_cache"])
        for name in ("initialize_kv_cache", "prepare_attn", "execute_model", "shutdown"):
            original = originals[name]
            if name == "initialize_kv_cache":
                def make(fn):
                    @functools.wraps(fn)
                    def initialize(runner, *args, **kwargs):
                        if self.bridge is not None:
                            try:
                                self._pid()
                                if self.bridge is not None:
                                    self.bridge.retire(runner, "cache initialization began")
                            except Exception as exc:
                                if self.bridge is not None:
                                    self.bridge.health.error("initialization setup: " + _reason(exc))
                        try:
                            result = fn(runner, *args, **kwargs)
                        except BaseException:
                            try:
                                self._pid()
                                self._end_pool(runner)
                            except Exception:
                                pass
                            raise
                        try:
                            bound = initialize_signature.bind(runner, *args, **kwargs)
                            bound.apply_defaults()
                            if bound.arguments.get("is_profiling", False):
                                self._pid()
                                self._end_pool(runner)
                            else:
                                self._start_pool(runner, reset=True)
                                if self.bridge is not None and runner in self.states:
                                    self.bridge.initialized(runner, bound, self.states[runner])
                        except Exception:
                            pass
                        return result
                    return initialize
            elif name == "prepare_attn":
                def make(fn):
                    @functools.wraps(fn)
                    def prepare(runner, input_batch, *args, **kwargs):
                        result = fn(runner, input_batch, *args, **kwargs)
                        active = None
                        try:
                            active = getattr(self.local, "active", None)
                            if active is not None and active.get("runner") is runner:
                                if active["poisoned"]:
                                    raise ObservationGap("batch preparation observation is incomplete")
                                if active["pending"] is not None:
                                    raise ObservationGap("multiple batch preparations")
                                n = _positive(input_batch.num_reqs, contract.MAX_REQUESTS, "request count")
                                count = _positive(input_batch.num_tokens, contract.MAX_TOKENS, "query tokens")
                                _equal(input_batch.num_draft_tokens, 0, "draft tokens")
                                if type(input_batch.req_ids) is not list or len(input_batch.req_ids) != n:
                                    raise ObservationGap("request row identity unavailable")
                                requests = list(input_batch.req_ids)
                                if any(type(r) is not str for r in requests):
                                    raise ObservationGap("request row identity unavailable")
                                active["pending"] = (requests, count)
                                if self.bridge is not None:
                                    self.bridge.prepared(active.get("assignment"), runner, requests, count)
                        except Exception as exc:
                            if active is not None:
                                active["poisoned"] = True
                                active["pending"] = None
                                active["state"].error(_reason(exc), active["forward_id"])
                        return result
                    return prepare
            elif name == "shutdown":
                def make(fn):
                    @functools.wraps(fn)
                    def shutdown(runner, *args, **kwargs):
                        failed = False
                        exception_type = None
                        if self.bridge is not None:
                            self.bridge.retire(runner, "native shutdown")
                        try:
                            return fn(runner, *args, **kwargs)
                        except BaseException as exc:
                            failed = True
                            exception_type = type(exc).__name__
                            raise
                        finally:
                            try:
                                self._pid()
                                # Only explicit cache initialization may start
                                # another recording after native shutdown.
                                self.unsupported[runner] = True
                                state = self.states.pop(runner, None)
                                if state is not None:
                                    if failed:
                                        state.error("native runner shutdown raised")
                                    state.close()
                            except BaseException:
                                # Preserve the original return or exception,
                                # including when observer cleanup fails.
                                pass
                            if self.bridge is not None:
                                self.bridge.runner_closed(runner, exception_type)
                    return shutdown
            else:
                def make(fn):
                    signature = execute_signature
                    @functools.wraps(fn)
                    def execute(runner, *args, **kwargs):
                        active = None
                        previous = None
                        state = None
                        occupied = False
                        assignment = None
                        returned = False
                        exception_type = None
                        try:
                            self._pid()
                            bound = signature.bind(runner, *args, **kwargs)
                            bound.apply_defaults()
                            scheduled = bound.arguments["scheduler_output"].total_num_scheduled_tokens
                            real = not bound.arguments.get("dummy_run", False) and not bound.arguments.get("is_profile", False)
                            previous = getattr(self.local, "active", None)
                            self.local.active = None
                            if self.bridge is not None and real and scheduled > 0:
                                assignment = self.bridge.enter(runner, bound.arguments["scheduler_output"], None)
                            if real and scheduled > 0:
                                state = self.states.get(runner)
                                # Only a real scheduled invocation can lazily
                                # establish a child process recording. Startup
                                # profiling often precedes cache initialization.
                                if state is None:
                                    self._start_pool(runner)
                                    state = self.states.get(runner)
                            if state is not None and real and scheduled > 0:
                                occupied = state.busy.acquire(blocking=False)
                                if not occupied:
                                    raise ObservationGap("concurrent runner invocation")
                                if state.closing:
                                    raise ObservationGap("runner recording is closing")
                                state.counter += 1
                                if previous is not None:
                                    raise ObservationGap("nested runner invocation")
                                active = dict(state=state, runner=runner, forward_id=state.counter,
                                              pending=None, seen=False, poisoned=False,
                                              stream=self.types["torch"].cuda.current_stream(runner.device))
                                self.local.active = active
                            if active is not None and self.bridge is not None:
                                active["assignment"] = assignment
                        except Exception as exc:
                            if self.bridge is not None:
                                self.bridge.health.error("forward invocation: " + _reason(exc))
                            if state is not None:
                                state.error(_reason(exc))
                            elif self.pid == os.getpid():
                                # A failed observer setup must not leave earlier
                                # observations looking complete for this process.
                                for existing in list(self.states.values()):
                                    existing.error(_reason(exc))
                        try:
                            result = fn(runner, *args, **kwargs)
                            returned = True
                            return result
                        except BaseException as exc:
                            exception_type = type(exc).__name__
                            raise
                        finally:
                            if self.bridge is not None:
                                self.bridge.leave(assignment, runner=runner, returned=returned, exception_type=exception_type,
                                                  poisoned=active is None or active["poisoned"])
                            if active is not None and not active["seen"]:
                                active["state"].error("scheduled invocation had no observed eager model call", active["forward_id"])
                            if occupied:
                                state.busy.release()
                            self.local.active = previous
                    return execute
            wrapper = make(original)
            wrapper._sdcguard_forward = True
            try:
                setattr(self.runner_class, name, wrapper)
                self.wrappers[name] = (original, wrapper)
            except Exception:
                self.detach()
                raise
        return self

    def detach(self):
        self._pid()
        changed = any(getattr(self.runner_class, name, None) is not wrapper
                      for name, (_, wrapper) in self.wrappers.items())
        for state in list(self.states.values()):
            try:
                if changed:
                    state.error("recorder wrapper coverage changed")
                state.close()
            except Exception:
                pass
        self.states.clear()
        self.model_bindings.clear()
        if self.bridge is not None:
            self.bridge.close()
        for name, (original, wrapper) in self.wrappers.items():
            if getattr(self.runner_class, name, None) is wrapper:
                setattr(self.runner_class, name, original)
        self.wrappers.clear()


def _diagnostic(directory, engine, reason):
    try:
        writer = JsonlRecorder(directory, engine, {})
        writer.gap(reason)
        writer.close()
    except Exception:
        pass


def install(*, assignment_factory=None):
    """Install only when explicitly enabled and the selected engine bytes match."""
    global _INSTALLATION
    directory = os.environ.get("SDCGUARD_FORWARD_DIR")
    if not directory:
        return {"status": "disabled", "hooks": []}
    if _INSTALLATION is not None:
        if assignment_factory is not None and assignment_factory is not _INSTALLATION.assignment_factory:
            return {"status": "unknown", "reason": "assignment registration requires a fresh installation",
                    "hooks": []}
        result = {"status": "installed", "hooks": sorted(_INSTALLATION.wrappers)}
        if assignment_factory is not None:
            reason = _INSTALLATION.bridge_error
            if _INSTALLATION.pid != os.getpid():
                reason = "assignment initialization was not observed in this process"
            result["assignment_bridge"] = {"status": "unknown" if reason else "registered",
                                           "reason": reason,
                                           "native_assignment_adapter": "not supplied by the bridge"}
        return result
    engine = {"name": "vllm", "version": None, "source_sha256": {}, "dependencies": {}}
    candidate = None
    try:
        directory = _directory(directory)
        vllm = importlib.import_module("vllm")
        engine["version"] = vllm.__version__
        root = Path(vllm.__file__).absolute().parent.parent
        engine["source_sha256"] = inspect_source(root, engine["version"])
        torch = importlib.import_module("torch")
        runner = importlib.import_module("vllm.v1.worker.gpu.model_runner")
        backend = importlib.import_module("vllm.v1.attention.backends.flash_attn")
        attention = importlib.import_module("vllm.model_executor.layers.attention.attention")
        model = importlib.import_module("vllm.model_executor.models.qwen3")
        model_state = importlib.import_module("vllm.v1.worker.gpu.model_states.default")
        context = importlib.import_module("vllm.forward_context")
        compilation = importlib.import_module("vllm.config.compilation")
        specs = importlib.import_module("vllm.v1.kv_cache_interface")
        engine["dependencies"] = {"python": platform.python_version(), "torch": str(torch.__version__), "flash_attn": "3"}
        types = dict(torch=torch, model=model.Qwen3ForCausalLM, attention=attention.Attention,
                     model_state=model_state.DefaultModelState, compile_none=compilation.CompilationMode.NONE,
                     impl=backend.FlashAttentionImpl, metadata=backend.FlashAttentionMetadata,
                     spec=specs.FullAttentionSpec, quant_none=specs.KVQuantMode.NONE,
                     decoder=backend.AttentionType.DECODER, graph_none=compilation.CUDAGraphMode.NONE)
        if assignment_factory is not None:
            types["global_hook_registry"] = torch.nn.modules.module
        candidate = Installation(runner.GPUModelRunner, directory, engine, types, context.get_forward_context,
                                 assignment_factory=assignment_factory)
        candidate.attach()
        _INSTALLATION = candidate
        atexit.register(uninstall)
        result = {"status": "installed", "hooks": sorted(candidate.wrappers)}
        if assignment_factory is not None:
            result["assignment_bridge"] = {"status": "unknown" if candidate.bridge_error else "registered",
                                           "reason": candidate.bridge_error,
                                           "native_assignment_adapter": "not supplied by the bridge"}
        return result
    except Exception as exc:
        if candidate is not None:
            candidate.detach()
        _diagnostic(directory, engine, _reason(exc))
        return {"status": "unknown", "reason": _reason(exc), "hooks": []}


def uninstall():
    global _INSTALLATION
    installation, _INSTALLATION = _INSTALLATION, None
    if installation is not None:
        installation.detach()
