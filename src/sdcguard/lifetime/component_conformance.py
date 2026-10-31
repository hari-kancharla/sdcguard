"""Run only pinned, ordinary vendor component functions with controlled CPU mocks.

This is source_fragment_cpu conformance, not a deployed-engine or GPU test.
The two positive paths are the existing reviewed ordinary allocation/READ/DONE
and matching-generation ACK/deferral/release paths. There is one request per
component, no retry/reused ID/stale ACK, no introduced failure ordering, and no
network, server, subprocess, GPU allocation, holds file or traffic generator.
Dependencies representing transport, memory, descriptors and collectives are
CPU mocks. A satisfied result cannot attest physical_write_safety, which stays
unknown. The existing source-lease evaluator is unchanged.

Every packaged excerpt and license is verified before any excerpt is executed.
The manifest itself is anchored by a trusted SHA256 in this module; expected
methods, exact filenames and source pins are also checked. Extra files, path
traversal, changed source/fragments or unpinned manifests are refused. No user
path or external code is accepted by the command.
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque, namedtuple
from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import logging
import os
import sys
from types import SimpleNamespace as NS
import textwrap
import time

from .adapter import Probe
from .generation_obligations import SCOPE, evaluate_generation_obligations
from .generation_observations import SOURCE_PINS, sglang_component_observations, vllm_probe_observations
from . import vllm_nixl as observer


class ComponentConformanceError(ValueError):
    """A packaged CPU component or its trusted provenance could not be verified."""


MANIFEST_SHA256 = '73ee21080c814525b070184e9e2d3f8ff04d41f3a4b46daf0c179330365aabeb'
EXPECTED_METHODS = frozenset([
    ('sglang', 'CommonKVManager', '_maybe_ack_drained_abort'),
    ('sglang', 'CommonKVManager', '_prefill_unique_rank'),
    ('sglang', 'CommonKVManager', '_send_abort_acks'),
    ('sglang', 'CommonKVManager', 'clear_deferred_abort_state'),
    ('sglang', 'CommonKVManager', 'is_abort_release_safe'),
    ('sglang', 'CommonKVManager', 'note_abort_ack'),
    ('sglang', 'CommonKVManager', 'register_deferred_abort_room'),
    ('sglang', 'CommonKVManager', 'register_deferred_ack_target'),
    ('sglang', 'DecodeTransferQueue', '_defer_release'),
    ('sglang', 'DecodeTransferQueue', '_do_release'),
    ('sglang', 'DecodeTransferQueue', '_is_host_destination'),
    ('sglang', 'DecodeTransferQueue', '_observe_deferred_kv_release'),
    ('sglang', 'DecodeTransferQueue', '_release_request'),
    ('sglang', 'DecodeTransferQueue', 'resolve_deferred_releases'),
    ('sglang', 'MooncakeKVManager', '_handle_deferred_abort_notification'),
    ('vllm', 'BlockPool', 'free_blocks'),
    ('vllm', 'BlockPool', 'get_new_blocks'),
    ('vllm', 'KVCacheManager', 'allocate_slots'),
    ('vllm', 'KVCacheManager', 'create_kv_cache_blocks'),
    ('vllm', 'KVCacheManager', 'free'),
    ('vllm', 'KVCacheManager', 'get_block_ids'),
    ('vllm', 'KVCacheManager', 'get_blocks'),
    ('vllm', 'NixlBaseConnectorWorker', '_pop_done_transfers'),
    ('vllm', 'NixlPullConnectorWorker', '_read_blocks'),
    ('vllm', 'NixlPullConnectorWorker', '_read_blocks_for_req'),
])


def _data_root():
    return files("sdcguard").joinpath("case_data").joinpath("engine_generation")


def _verified_bundle():
    root = _data_root()
    try:
        manifest_raw = root.joinpath("MANIFEST.json").read_bytes()
        if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
            raise ComponentConformanceError("packaged component manifest is not trusted")
        manifest = json.loads(manifest_raw)
        if (manifest.get("schema") != "sdcguard.engine-generation-fragments.v1"
                or manifest.get("scope") != SCOPE
                or manifest.get("engines") != {engine: pin["commit"] for engine, pin in SOURCE_PINS.items()}):
            raise ComponentConformanceError("packaged source contract does not match supported pins")
        fragments = manifest.get("fragments")
        licenses = manifest.get("licenses")
        if not isinstance(fragments, list) or not isinstance(licenses, list):
            raise ComponentConformanceError("packaged fragment/license inventory missing")
        seen = set()
        expected_files = {"MANIFEST.json"}
        content = {}
        for item in fragments:
            identity = item["engine"], item["class"], item["method"]
            if identity not in EXPECTED_METHODS or identity in seen:
                raise ComponentConformanceError("unknown or duplicate packaged method")
            seen.add(identity)
            filename = "_".join(identity) + ".py.txt"
            if item["file"] != filename or "/" in filename or "\\" in filename or ".." in filename:
                raise ComponentConformanceError("packaged method path is not allowed")
            pin = SOURCE_PINS[item["engine"]]
            if (item["commit"] != pin["commit"]
                    or pin["sha256"].get(item["source_path"]) != item["source_sha256"]):
                raise ComponentConformanceError("packaged method source identity is unsupported")
            raw = root.joinpath(filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ComponentConformanceError("packaged method hash does not match")
            parsed = ast.parse(textwrap.dedent(raw.decode("utf-8")))
            if (len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.FunctionDef)
                    or parsed.body[0].name != item["method"] or parsed.body[0].decorator_list
                    or any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(parsed))):
                raise ComponentConformanceError("packaged excerpt is not the reviewed import-free method")
            expected_files.add(filename)
            content[filename] = raw
        if seen != EXPECTED_METHODS:
            raise ComponentConformanceError("packaged method inventory is incomplete")
        if len(licenses) != 2 or {item["engine"] for item in licenses} != set(SOURCE_PINS):
            raise ComponentConformanceError("packaged upstream licenses are incomplete")
        for item in licenses:
            filename = item["engine"] + "-LICENSE.txt"
            if item["file"] != filename:
                raise ComponentConformanceError("packaged license path is not allowed")
            raw = root.joinpath(filename).read_bytes()
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            if hashlib.sha256(raw).hexdigest() != item["sha256"] or blob != item["git_blob_sha"]:
                raise ComponentConformanceError("packaged upstream license hash does not match")
            expected_files.add(filename)
        if {entry.name for entry in root.iterdir()} != expected_files:
            raise ComponentConformanceError("packaged component directory has missing or extra files")
        return {"manifest": manifest, "fragments": content}
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError, SyntaxError) as exc:
        raise ComponentConformanceError("packaged component data is unavailable or malformed") from exc


def _methods(engine, classname, namespace):
    bundle = _verified_bundle()
    methods = {}
    for item in bundle["manifest"]["fragments"]:
        if item["engine"] != engine or item["class"] != classname:
            continue
        raw = bundle["fragments"][item["file"]]
        code = "from __future__ import annotations\n" + textwrap.dedent(raw.decode())
        exec(compile(code, item["source_url"], "exec"), namespace)
        methods[item["method"]] = namespace[item["method"]]
    if not methods:
        raise ComponentConformanceError("unknown component class")
    return methods


class _Sink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


def _probe():
    probe = Probe(_Sink())
    # One observer process uses its own actual monotonic clock. No cross-host
    # clock equality is invented, including when this unit runs on macOS.
    probe.clock_id = f"component-monotonic-process:{os.getpid()}:{probe.session}"
    probe._emit("component_start")
    return probe


def _metadata(engine):
    return {"scope": "source_fragment_cpu", "source": deepcopy(SOURCE_PINS[engine]),
            "complete_from_start": True, "closed": True}


def _sglang_receipt():
    probe = _probe()
    logger = logging.getLogger("generation-component")
    namespace = {"time": time, "logger": logger, "DeferredAbortAckState": NS,
                 "torch": NS(distributed=NS(ReduceOp=NS(MIN="min"))),
                 "all_reduce_flags": lambda *, flags, group, op: flags,
                 "KVPoll": NS(Failed="Failed")}
    common = type("CommonKVManager", (), _methods("sglang", "CommonKVManager", namespace))
    sender_type = type("MooncakeKVManager", (common,), _methods("sglang", "MooncakeKVManager", namespace))
    released_kv = []
    namespace["release_kv_cache"] = lambda req, tree, *, checkpoint: released_kv.append((req, checkpoint))
    queue_type = type("DecodeTransferQueue", (), _methods("sglang", "DecodeTransferQueue", namespace))
    mgr = common()
    mgr._deferred_abort_ack_tracker = {}
    mgr._deferred_abort_generation = 0
    mgr.enable_deferred_decode_kv_release = True
    room, idx = 200, 7
    q = queue_type()
    q._deferred_releases = []
    q.deferred_kv_release_timeout = 30.0
    q.enable_host_receive = q.enable_staging = False
    q.gloo_group = None
    q.tree_cache = object()
    freed = []
    allocator = NS()
    pool = f"cpu-metadata-allocator:{id(allocator)}"
    generation = mgr.register_deferred_abort_room(room)
    identity = {"room": room, "generation": generation, "metadata_index": idx, "pool": pool}

    def free(slot):
        assert slot == idx
        freed.append(slot)
        probe._emit("metadata_free", **identity)

    allocator.free = free
    q.req_to_metadata_buffer_idx_allocator = allocator
    cleared = []
    q.metadata_buffers = NS(clear_slot=cleared.append)
    q.scheduler = NS(enable_hisparse=False, metrics_reporter=NS(enable_metrics=True),
                     metrics_collector=NS(observe_decode_deferred_kv_release=lambda **facts:
                         probe._emit("release_outcome", **identity, outcome=facts["outcome"])))
    req = NS(bootstrap_room=room, rid=str(room))
    receiver = NS(kv_mgr=mgr, abort_notified=True, bootstrap_infos=[{"rank": 0}, {"rank": 1}], clear=lambda: None)
    dreq = NS(req=req, metadata_buffer_index=idx, kv_receiver=receiver, host_staged=False)
    q._defer_release(dreq)
    actual_entry = q._deferred_releases[0]
    assert actual_entry[0] is dreq and actual_entry[3:] == (idx, 2)
    probe._emit("hold", **identity, operation=dreq.req.rid,
                required_peers=[entry["rank"] for entry in receiver.bootstrap_infos])
    q.resolve_deferred_releases()
    assert len(q._deferred_releases) == 1 and freed == []
    probe._emit("observation", held=len(q._deferred_releases))
    target_type = namedtuple("AckTarget", "ip port generation")
    for rank in (0, 1):
        sender = sender_type()
        sender._staging_outstanding = {}  # ordinary already-drained sender, no write injection
        sender._deferred_ack_targets = {}
        sender._deferred_ack_poisoned_rooms = set()
        sender.request_status = {room: "Transferring"}
        sender.attn_tp_rank, sender.pp_size, sender.attn_cp_size = rank, 1, 1
        sender.pp_rank = sender.attn_cp_rank = 0
        sender.update_status = lambda r, status, sender=sender: sender.request_status.__setitem__(r, status)

        def deliver(r, target, sender=sender):
            peer = sender._prefill_unique_rank()
            mgr.note_abort_ack(r, peer, target.generation)
            state = mgr._deferred_abort_ack_tracker[r]
            probe._emit("drain_ack", **identity, peer=peer,
                        outstanding=sender._staging_outstanding.get(r, 0),
                        admission_failed=sender.request_status[r] == namespace["KVPoll"].Failed,
                        ack_accepted=state.generation == target.generation and peer in state.prefill_ranks)

        sender._send_abort_ack = deliver  # controlled CPU wire delivery, no network
        target = target_type("127.0.0.1", 12345, generation)
        notification = NS(room=room, deferred_ack_target=lambda: target)
        sender._handle_deferred_abort_notification(notification, room_active=True)
        q.resolve_deferred_releases()
        if rank == 0:
            assert len(q._deferred_releases) == 1 and freed == []
    assert q._deferred_releases == [] and freed == [idx] and cleared == [idx]
    assert released_kv == [(req, False)] and dreq.kv_receiver is None
    assert room not in mgr._deferred_abort_ack_tracker
    probe._emit("component_end")
    metadata = _metadata("sglang")
    metadata["metadata_pool"] = pool
    return deepcopy(probe.sink.events), metadata


def _vllm_receipt():
    probe = _probe()
    namespace = {"time": time, "logger": logging.getLogger("generation-component"),
                 "RequestStatus": NS(WAITING="WAITING", PREEMPTED="PREEMPTED")}

    class Blocks:
        def __init__(self, blocks):
            self.blocks = blocks

        def get_block_ids(self):
            return tuple([block.block_id for block in group] for group in self.blocks)

    namespace["KVCacheBlocks"] = Blocks
    manager_type = type("KVCacheManager", (), _methods("vllm", "KVCacheManager", namespace))
    pool_type = type("BlockPool", (), _methods("vllm", "BlockPool", namespace))
    pool = pool_type()
    block = NS(block_id=7, ref_cnt=0, pool=pool, is_null=False, block_hash=None)
    queue = deque([block])
    pool.free_block_queue = NS(popleft_n=lambda n: [queue.popleft() for _ in range(n)],
                              prepend_n=lambda blocks: queue.extendleft(reversed(blocks)),
                              append_n=queue.extend)
    pool.get_num_free_blocks = lambda: len(queue)
    pool._reuse_watchers, pool.enable_caching, pool.metrics_collector = {}, False, None
    held = {}

    def allocate(rid, *unused):
        held[rid] = pool.get_new_blocks(1)
        return (held[rid],)

    coordinator = NS(remove_skipped_blocks=lambda *a, **k: None,
                     get_num_blocks_to_allocate=lambda **k: 1,
                     allocate_new_blocks=allocate, get_blocks=lambda rid: (held.get(rid, []),),
                     free=lambda rid: pool.free_blocks(held.pop(rid)))
    manager = manager_type()
    manager.engine_id, manager.coordinator, manager.block_pool = "D", coordinator, pool
    manager.empty_kv_cache_blocks, manager.retained_hit_group_ids = Blocks(([],)), ()
    manager.max_model_len, manager.watermark_blocks = 32, 0
    base_type = type("NixlBaseConnectorWorker", (), _methods("vllm", "NixlBaseConnectorWorker", namespace))

    class ReadSpec:
        def __init__(self, remote_rank, local_block_ids, remote_block_ids, block_ids_by_region=False):
            self.remote_rank, self.local_block_ids, self.remote_block_ids = remote_rank, local_block_ids, remote_block_ids
            self.block_ids_by_region = block_ids_by_region

    namespace["ReadSpec"] = ReadSpec
    worker_type = type("NixlPullConnectorWorker", (base_type,), _methods("vllm", "NixlPullConnectorWorker", namespace))
    worker = worker_type()
    worker.engine_id = "D"
    worker._recving_transfers = defaultdict(list)
    worker._invalid_remote_engines, worker._engine_last_active = set(), {}
    worker._bidirectional_kv_xfer_enabled = worker._has_mamba = worker._mixed_mem_types = worker.use_mla = False
    worker.dcp_size = worker.pcp_size = worker.world_size = worker.pp_size = worker._physical_blocks_per_logical_kv_block = 1
    worker.tp_rank, worker.block_size, worker.num_regions = 0, 16, 1
    worker._remote_agents = {"P": {(0, 0): "prefill-agent"}}
    remote_info = NS(remote_block_size=16, remote_tp_size=1, remote_dcp_size=1, remote_physical_blocks_per_logical=1)
    worker.transfer_topo = NS(get_engine_info=lambda _: remote_info, tp_ratio=lambda _: 1, block_size_ratio=lambda _: 1)
    worker.tp_mappings = {"P": NS(all_source_ranks=[0], source_ranks_per_group=[{0}], local_consumers=1)}
    worker.dst_region_group_ids, worker.dst_region_num_blocks, worker.region_group_ids = {"P": []}, {}, []
    worker.dst_uses_region_group_mapping, worker._uses_region_group_mapping = {"P": False}, False
    worker._logical_to_kernel_block_ids = lambda ids, scale: ids
    worker._apply_prefix_caching = lambda **k: (k["decode_block_ids"], k["prefill_block_ids"])
    worker._compute_desc_ids = lambda **k: [b for group in k["block_ids"] for b in group]
    worker.dst_num_blocks = {"P": 1, "D": 1}
    worker.src_xfer_handles_by_block_size, worker.dst_xfer_side_handles = {16: 10}, {"P": {0: 20}}
    worker.kv_cache_config = NS(transfer_groups=(0,))
    states, transfers, released = iter(("PROC", "DONE")), [], []

    def prepare(kind, local_handle, local_ids, remote_handle, remote_ids, **unused):
        assert (kind, local_handle, local_ids, remote_handle, remote_ids) == ("READ", 10, [7], 20, [0])
        return 101

    def poll(handle):
        state = next(states)
        owners = [rid for rid, handles in worker._recving_transfers.items() if handle in handles]
        assert len(owners) == 1
        rid = owners[0]
        probe.evidence("destination_transport_poll", request=rid, handle=handle,
                       state=state, op_ids=observer._ops(rid))
        return state

    worker.nixl_wrapper = NS(make_prepped_xfer=prepare, transfer=transfers.append,
                            check_xfer_state=poll, get_xfer_telemetry=lambda h: {"handle": h},
                            release_xfer_handle=released.append)
    telemetry = []
    worker.xfer_stats = NS(record_transfer=telemetry.append)
    observer.reset()
    observer._PROBE = probe
    try:
        hooks = observer.install_manager(manager_type)
        hooks += observer.install_connector(WorkerBase=base_type, PullWorker=worker_type)
        probe._emit("note", "hook coverage", hooks=hooks)
        request = NS(request_id="one-request", num_computed_tokens=0, num_prompt_tokens=16,
                     num_tokens=16, num_in_flight_tokens=0, status="WAITING")
        result = manager.allocate_slots(request, 16, delay_cache_blocks=True)
        assert result.get_block_ids() == ([7],) and block.ref_cnt == 1
        meta = NS(remote=NS(engine_id="P", request_id="one-request", block_ids=([0],)),
                  local_block_ids=([7],), local_physical_block_ids=([7],), awaiting_kvs=True)
        worker._read_blocks_for_req(request.request_id, meta)
        assert transfers == [101]
        probe.evidence("destination_transfer_handles", request=request.request_id,
                       handles=list(worker._recving_transfers[request.request_id]),
                       op_ids=observer._ops(request.request_id))
        assert worker._pop_done_transfers(worker._recving_transfers) == (set(), set())
        assert released == [] and worker._recving_transfers[request.request_id] == [101]
        assert worker._pop_done_transfers(worker._recving_transfers) == ({request.request_id}, set())
        assert released == [101] and telemetry == [{"handle": 101}]
        manager.free(request)
        assert block.ref_cnt == 0 and list(queue) == [block]
        probe.evidence("recording_health", healthy=probe._dropped == 0)
        probe._emit("component_end")
        events = deepcopy(probe.sink.events)
    finally:
        observer.reset()
    metadata = _metadata("vllm")
    metadata.update(manager_engine=manager.engine_id, worker_engine=worker.engine_id,
                    pool=f"cpu-block-pool:{id(pool)}", single_cache_group=True)
    return events, metadata


def _evaluate(engine, events, metadata):
    translate = sglang_component_observations if engine == "sglang" else vllm_probe_observations
    return evaluate_generation_obligations(translate(events, metadata))


def run_component_conformance(engine="all"):
    """Return inspectable CPU receipts and scoped assessments; do no I/O writes."""
    if engine not in ("vllm", "sglang", "all"):
        raise ComponentConformanceError("unsupported component engine")
    if not __debug__:
        raise ComponentConformanceError("CPU component conformance requires Python assertions enabled")
    if any(name in sys.modules for name in ("vllm", "sglang")):
        raise ComponentConformanceError("run CPU conformance in a fresh process without serving engines loaded")
    if observer._PROBE is not None or any(observer._STATE.values()):
        raise ComponentConformanceError("run CPU conformance with a fresh inactive lifetime observer")
    _verified_bundle()
    targets = ("vllm", "sglang") if engine == "all" else (engine,)
    results = {}
    for target in targets:
        events, metadata = _vllm_receipt() if target == "vllm" else _sglang_receipt()
        assessment = _evaluate(target, events, metadata)
        results[target] = {"source": deepcopy(SOURCE_PINS[target]), "assessment": assessment.as_dict(),
                           "observation_count": len(events),
                           "receipt": {"metadata": metadata, "observations": events}}
    return {"schema": "sdcguard.component-conformance.v1", "scope": SCOPE,
            "physical_write_safety": "unknown", "evidence_origin": "pinned_original_functions_with_cpu_mocks",
            "status": "satisfied" if all(r["assessment"]["status"] == "satisfied" for r in results.values()) else "unknown",
            "engines": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run pinned ordinary component functions with CPU mocks; no deployed-engine/GPU validation")
    parser.add_argument("--engine", choices=("vllm", "sglang", "all"), default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_component_conformance(args.engine)
    except Exception as exc:
        result = {"schema": "sdcguard.component-conformance.v1", "scope": SCOPE,
                  "physical_write_safety": "unknown", "status": "unknown",
                  "reason": f"{type(exc).__name__}: {exc}",
                  "expected_sources": deepcopy(SOURCE_PINS)}
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print("Scope: source_fragment_cpu (original functions with controlled CPU mocks)")
        print("Physical GPU/write safety: unknown; deployed-engine validation: not performed")
        for engine, item in result.get("engines", {}).items():
            print(f"{engine}: {item['assessment']['status']} at {item['source']['commit']}")
        if "reason" in result:
            print("Inconclusive: " + result["reason"])
    return 0 if result["status"] == "satisfied" else 2


if __name__ == "__main__":
    raise SystemExit(main())
