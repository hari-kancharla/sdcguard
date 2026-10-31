"""Passive translators for pinned CPU component observations, never a forcer.

The SGLang contract binds a metadata-slot protocol hold to the actual armed
abort wire generation. A fence requires the sender's observed zero outstanding
counter, Failed admission state, and acceptance by the decode manager for that
same generation/rank. This certifies manager bookkeeping; the counter and mocked
transport do not attest physical GPU writes or an allocator incarnation.
The slot is a real auxiliary-write destination in send_aux, which computes its
remote address from dst_aux_index. Its observed release is not relabelled as a
physical KV-page release.

The vLLM contract uses existing alloc/free, attempt, submit, submission-result,
transfer-done and complete receipts. A READ writes its LOCAL destination. Its
terminal completion is a destination writer fence, not source retirement. An
allocation generation is inferred only from complete ordered alloc/free
receipts, one observed manager/worker engine and one pool/cache group. A NIXL
handle release alone is insufficient. The actual registered handle set and
DONE poll results must also be passively observed; done-request receipts can
include failures after cleanup. Cross-process resource mapping is not yet
supported here: such recordings stay UNKNOWN rather than guessing a pool ID.

Both adapters retain every observation envelope for the generic completeness
check. They accept only source-pinned fragment CPU receipts with explicit fresh
start/close boundaries. These are conformance witnesses with controlled mocks,
not complete upstream test suites, a deployment adapter, a live-engine run, or
cryptographic source attestation. No function installs hooks or runs an engine.
"""

from __future__ import annotations

import json

from .generation_obligations import SCHEMA, SCOPE


SOURCE_PINS = {
    "sglang": {
        "commit": "41fb719cf25966e6155f3e3d91f1157bc1a15123",
        "sha256": {
            "python/sglang/srt/disaggregation/common/conn.py": "83a0521f91d0a32118d69caa0693aa95ddd292d95d484fd2cc02fec65990c8c2",
            "python/sglang/srt/disaggregation/mooncake/conn.py": "509407543cf53e54f44a6511a82cbe0e4d894967f50447c00381c95a45c8654e",
            "python/sglang/srt/disaggregation/decode.py": "4441377cab1c4a3c5026f1213e2fd42eef157573bcded00ca5a107503b22e9ea",
        },
    },
    "vllm": {
        "commit": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "sha256": {
            "vllm/v1/core/kv_cache_manager.py": "d08315437c6b07040d1b7e12d7bd7d6e966c976a286f05a9788098cfdd9754c7",
            "vllm/v1/core/block_pool.py": "6d1e0d89b43e3b2ae396906b9c2f0551dbe192b60d05f8c1ca2d93a91bbbeea3",
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py": "6985fbd91aed81dc982abaa009d93475a89d0893deb1ff48fce406d62509563a",
            "vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py": "5da22389ba154696ba19a936fdaa89305a4dd981ef4c19d5c40322b050443f9b",
        },
    },
}


def _nonempty(value):
    return isinstance(value, str) and bool(value)


def _index(value):
    return type(value) is int and value >= 0


def _stream(event):
    return f"{event.get('session')}:{event.get('pid')}"


def _base(engine, observations, metadata):
    result = {"schema": SCHEMA, "scope": SCOPE, "issues": [], "events": [],
              "coverage": {"complete_from_start": False, "closed": False, "streams": []}}
    if not isinstance(metadata, dict) or not isinstance(observations, list):
        raise ValueError("component metadata or observations missing")
    if (metadata.get("scope") != SCOPE or metadata.get("source") != SOURCE_PINS[engine]
            or metadata.get("complete_from_start") is not True or metadata.get("closed") is not True):
        raise ValueError("source pin or complete component coverage not established")
    if not observations or any(not isinstance(e, dict) for e in observations):
        raise ValueError("no valid component observations")
    if (observations[0].get("kind") != "component_start"
            or observations[-1].get("kind") != "component_end"):
        raise ValueError("fresh component start and close boundaries missing")
    stream_id = _stream(observations[0])
    if (not _nonempty(observations[0].get("session"))
            or type(observations[0].get("pid")) is not int
            or not _nonempty(observations[0].get("clock_id"))):
        raise ValueError("component stream identity or comparable clock missing")
    if any(_stream(e) != stream_id for e in observations):
        raise ValueError("cross-process manager/worker pool mapping is not established")
    result["coverage"] = {"complete_from_start": True, "closed": True, "streams": [
        {"id": stream_id, "clock_id": observations[0]["clock_id"], "first_seq": 1,
         "last_seq": observations[-1].get("seq"), "closed": True, "dropped": 0}]}
    result["events"] = [{"stream": stream_id, "clock_id": e.get("clock_id"),
                         "seq": e.get("seq"), "mono": e.get("mono"),
                         "dropped": e.get("dropped"), "facts": []} for e in observations]
    return result


def _unknown(message):
    return {"schema": SCHEMA, "scope": SCOPE, "issues": [message], "events": [],
            "coverage": {"complete_from_start": False, "closed": False, "streams": []}}


def sglang_component_observations(observations, metadata):
    """Translate an ordinary matching-generation, drained CPU hold/release.

    No stale ACK, room reuse, timeout, retry, physical KV writes or second
    request is needed. The observer must record the real allocator identity,
    slot, room, armed generation and actual required rank set at deferral.
    """
    try:
        result = _base("sglang", observations, metadata)
        pool = metadata.get("metadata_pool")
        if not _nonempty(pool):
            raise ValueError("observed metadata pool identity missing")
        holds = {}
        for raw, event in zip(observations, result["events"]):
            args, kind = raw.get("args", {}), raw.get("kind")
            if not isinstance(args, dict):
                raise ValueError("malformed component facts")
            if kind == "hold":
                room, generation, idx = args.get("room"), args.get("generation"), args.get("metadata_index")
                peers = args.get("required_peers")
                if (not _index(room) or not _index(generation) or not _index(idx)
                        or not _nonempty(args.get("operation")) or room in holds
                        or not isinstance(peers, list) or not peers
                        or any(not _index(p) for p in peers) or len(set(peers)) != len(peers)
                        or args.get("pool") != pool):
                    raise ValueError("room/hold generation, writer set, or slot binding unavailable")
                key = {"operation": args["operation"], "resource": f"{pool}/metadata/{idx}", "generation": generation}
                holds[room] = {"key": key, "idx": idx, "peers": peers}
                event["facts"].append({"kind": "bind", "key": key,
                                       "writers": [str(p) for p in peers], "generation_basis": "protocol_hold"})
            elif kind in ("drain_ack", "metadata_free", "release_outcome"):
                hold = holds.get(args.get("room"))
                if (not _index(args.get("room")) or not _index(args.get("generation"))
                        or not _index(args.get("metadata_index"))
                        or hold is None or args.get("generation") != hold["key"]["generation"]
                        or args.get("metadata_index") != hold["idx"] or args.get("pool") != pool):
                    raise ValueError("ACK/release is not attributed to the observed destination hold")
                if kind == "drain_ack":
                    if (type(args.get("outstanding")) is not int or args["outstanding"] != 0
                            or args.get("admission_failed") is not True or args.get("ack_accepted") is not True
                            or args.get("peer") not in hold["peers"]):
                        raise ValueError("same-generation accepted ACK and sender drain not observed")
                    event["facts"].append({"kind": "fence", "key": hold["key"],
                                           "writer": str(args["peer"]), "terminal": True,
                                           "semantics": "matching_drain_ack"})
                elif kind == "metadata_free":
                    if "free_event" in hold:
                        raise ValueError("duplicate destination release receipt")
                    hold["free_event"] = event
                else:
                    if "free_event" not in hold or "outcome" in hold:
                        raise ValueError("release outcome lacks an observed metadata-slot release")
                    hold["outcome"] = args.get("outcome")
                    hold["free_event"]["facts"].append({"kind": "release", "key": hold["key"],
                        "cause": "observed_release" if args.get("outcome") == "drained" else "timeout_or_error"})
            elif kind not in ("component_start", "component_end", "observation"):
                raise ValueError("unsupported component observation")
        if any("outcome" not in h for h in holds.values()):
            result["issues"].append("release outcome was not observed")
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return _unknown(str(exc))


def _region(value, engine):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or value[0] != engine or not _index(value[1])):
        raise ValueError("destination engine/pool/block correspondence unavailable")
    return tuple(value)


def vllm_probe_observations(observations, metadata):
    """Translate existing Probe receipts without changing or replaying hooks.

    Current bounded support is a fresh, single-process CPU component containing
    one manager and worker, one cache group, complete alloc/free coverage and
    unique request/operation IDs. Real multiprocess recordings need a passive
    startup/pool-to-worker mapping and stream-close receipt before this adapter
    can establish the corresponding proof. Transfer handle release is ignored.
    Every mapped operation must have its own live allocation binding, submit,
    terminal transport evidence, destination completion and release. One proven
    operation cannot stand in for an unrecorded part of a successful request.
    """
    try:
        result = _base("vllm", observations, metadata)
        engine, pool = metadata.get("manager_engine"), metadata.get("pool")
        if (not _nonempty(engine) or metadata.get("worker_engine") != engine or not _nonempty(pool)
                or metadata.get("single_cache_group") is not True):
            raise ValueError("observed manager/worker engine and single-pool identity unavailable")
        hooks = set()
        healthy = False
        allocated, generations, attempts, operations = {}, {}, {}, {}
        attempt_allocations = {}
        submission_results, done_ops, done_requests = set(), set(), set()
        transport_handles, transport_states = {}, {}
        for raw, event in zip(observations, result["events"]):
            args, kind = raw.get("args", {}), raw.get("kind")
            if not isinstance(args, dict):
                raise ValueError("malformed probe arguments")
            if kind in ("alloc", "free"):
                region = _region(args.get("region"), engine)
                if kind == "alloc":
                    owner = args.get("owner")
                    if region in allocated or not _nonempty(owner):
                        raise ValueError("destination allocation incarnation is ambiguous")
                    generations[region] = generations.get(region, 0) + 1
                    allocated[region] = (owner, generations[region])
                else:
                    if region not in allocated or args.get("by") != allocated[region][0]:
                        raise ValueError("destination free lacks its allocation/owner")
                    for operation in operations.values():
                        if operation["region"] == region and operation["generation"] == allocated[region][1]:
                            event["facts"].append({"kind": "release", "key": operation["key"], "cause": "observed_release"})
                            operation["released"] = True
                    del allocated[region]
            elif kind == "note":
                if raw.get("text") == "hook coverage":
                    names = args.get("hooks")
                    if not isinstance(names, list) or any(not _nonempty(n) for n in names):
                        raise ValueError("malformed hook coverage")
                    hooks.update(names)
            elif kind == "evidence":
                what = args.get("what")
                if what == "recording_health":
                    if args.get("healthy") is not True:
                        raise ValueError("recording health was not established")
                    healthy = True
                elif what == "attempt":
                    rid, ops, dst = args.get("request"), args.get("op_ids"), args.get("destination_regions")
                    if (not _nonempty(rid) or rid in attempts or args.get("mapping_complete") is not True
                            or not isinstance(ops, list) or not ops or any(not _nonempty(o) for o in ops)
                            or len(set(ops)) != len(ops) or not isinstance(dst, list) or len(dst) != len(ops)
                            or type(args.get("blocks")) is not int or args["blocks"] != len(ops)):
                        raise ValueError("unique complete request/operation/destination mapping unavailable")
                    regions = [_region(d, engine) for d in dst]
                    if len(set(regions)) != len(regions):
                        raise ValueError("duplicate destination mapping")
                    if set(ops) & {op for mapping in attempts.values() for op in mapping}:
                        raise ValueError("operation identity appears in more than one request mapping")
                    if any(region not in allocated or allocated[region][0] != rid for region in regions):
                        raise ValueError("mapped destination operation lacks an observed allocation/owner")
                    attempts[rid] = dict(zip(ops, regions))
                    attempt_allocations[rid] = {op: allocated[region] for op, region in attempts[rid].items()}
                elif what == "read_submission_result":
                    rid, ops = args.get("request"), args.get("op_ids")
                    if (rid not in attempts or ops != list(attempts[rid]) or args.get("submitted") is not True
                            or rid in submission_results):
                        raise ValueError("actual transport submission not established")
                    submission_results.add(rid)
                elif what == "destination_transfer_handles":
                    rid, handles = args.get("request"), args.get("handles")
                    if (rid not in submission_results or rid in transport_handles
                            or args.get("op_ids") != list(attempts[rid])
                            or not isinstance(handles, list) or not handles
                            or any(not _index(h) for h in handles) or len(set(handles)) != len(handles)):
                        raise ValueError("complete registered destination transfer handle set unavailable")
                    if any(set(handles) & set(other) for other in transport_handles.values()):
                        raise ValueError("transfer handles are not uniquely attributed")
                    transport_handles[rid] = handles
                elif what == "destination_transport_poll":
                    rid, handle = args.get("request"), args.get("handle")
                    if (rid not in transport_handles or handle not in transport_handles[rid]
                            or args.get("op_ids") != list(attempts[rid])
                            or args.get("state") not in ("PROC", "DONE")):
                        raise ValueError("actual terminal transport status lacks exact operation/handle attribution")
                    if transport_states.get((rid, handle)) == "DONE":
                        raise ValueError("transport status follows an already completed handle")
                    transport_states[rid, handle] = args["state"]
                elif what == "transfer_done":
                    rid, ops = args.get("request"), args.get("op_ids")
                    if (rid not in submission_results or ops != list(attempts[rid])
                            or rid in done_requests
                            or rid not in transport_handles
                            or any(transport_states.get((rid, h)) != "DONE" for h in transport_handles[rid])):
                        raise ValueError("terminal transfer does not match its submitted operations")
                    done_ops.update(ops)
                    done_requests.add(rid)
            elif kind == "submit":
                op, dst = args.get("op"), _region(args.get("dst"), engine)
                matches = [(rid, mapping) for rid, mapping in attempts.items() if op in mapping]
                if (len(matches) != 1 or op in operations or args.get("kind") != "read"
                        or matches[0][1][op] != dst or dst not in allocated
                        or matches[0][0] != allocated[dst][0]
                        or allocated[dst] != attempt_allocations[matches[0][0]][op]):
                    raise ValueError("submitted writer lacks exact live destination allocation attribution")
                rid = matches[0][0]
                key = {"operation": op, "resource": json.dumps([pool, *dst], separators=(",", ":")),
                       "generation": allocated[dst][1]}
                operations[op] = {"key": key, "region": dst, "generation": allocated[dst][1], "request": rid,
                                  "completed": False, "released": False}
                event["facts"].append({"kind": "bind", "key": key, "writers": ["transport"],
                                       "generation_basis": "observed_allocation"})
            elif kind == "complete":
                op = args.get("op")
                if (op not in operations or op not in done_ops
                        or operations[op]["completed"]
                        or _region(args.get("marks_ready"), engine) != operations[op]["region"]
                        or allocated.get(operations[op]["region"]) != (operations[op]["request"], operations[op]["generation"])):
                    raise ValueError("destination completion lacks a matching observed terminal transfer")
                operations[op]["completed"] = True
                event["facts"].append({"kind": "fence", "key": operations[op]["key"], "writer": "transport",
                                       "terminal": True, "semantics": "transport_done"})
            elif kind in ("share", "cancel", "stopped", "recording_error"):
                raise ValueError("unsupported sharing/cancellation or incomplete recording")
            elif kind not in ("component_start", "component_end", "publish", "accept", "checkpoint"):
                raise ValueError("unsupported probe observation")
        required = {"allocate_slots", "free", "_read_blocks_for_req", "_read_blocks", "_pop_done_transfers"}
        if not required <= {name.rsplit(".", 1)[-1] for name in hooks} or not healthy:
            result["issues"].append("required allocation/transfer hooks or final health receipt missing")
        if allocated:
            result["issues"].append("destination pool still has live allocations at component close")
        expected_ops = {op for mapping in attempts.values() for op in mapping}
        if (set(attempts) != submission_results or set(attempts) != set(transport_handles)
                or set(attempts) != done_requests or expected_ops != set(operations)
                or expected_ops != done_ops):
            result["issues"].append("request mapping has unaccounted destination operations or partial lifecycle coverage")
        if any(not operation["completed"] or not operation["released"] for operation in operations.values()):
            result["issues"].append("every mapped destination operation needs its own completion and release")
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return _unknown(str(exc))
