"""Owner-only ordinary engine lifecycle. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile

from .config import (SNAPSHOT, BLOCK_SIZE, MAX_MODEL_LEN, MAX_SEQS, MIN_GPU_BLOCKS,
                          JOB_SECONDS, STARTUP_SECONDS, PORTS, REVISION, environment, server_command)
from .common import checkpoint, digest, encoded, regular_bytes


def owned_transport(role_processes, stop):
    """Recheck the retained role and connected peer before sending a request."""
    from .client import post
    from .supervisor import poll_engine
    from .listeners import owned, connected_owned

    def guard(role, sock):
        process = role_processes.get(role)
        until = time.monotonic() + 2
        while not stop.is_set() and process is not None and poll_engine(process) is None:
            if owned(PORTS[role], process) if sock is None else connected_owned(sock, process):
                return
            # The TCP handshake may finish before the server accepts its socket.
            if sock is None or time.monotonic() >= until:
                break
            time.sleep(.01)
        stop.set()
        raise ValueError("owned " + role + " server or connection is no longer available")

    def send(role, body, headers, *, timeout):
        try:
            return post(role, body, headers, timeout=timeout, guard=guard)
        except Exception:
            stop.set()
            raise
    return send


def startup_valid(record, *, plan, arm, role):
    from sdcguard.checksum_client import source_profile
    from ..checksum_profiles import startup_supported
    if (type(record) is not dict or any(record.get(key) != value for key, value in
            dict(run_id=plan["run_id"], packet_sha256=plan["packet_sha256"], arm_id=arm["arm_id"],
                 engine_session_id=arm["engine_session_id"], role=role, source_revision=plan["source_revision"]).items())
            or record.get("source_profile") != source_profile(plan["source_revision"])
            or not startup_supported(record, plan["source_revision"])
            or type(record.get("process_id")) is not int or record["process_id"] <= 0
            or record.get("checksum_enabled") is not True
            or record.get("request_id_randomization_enabled") is not True
            or record.get("flashinfer_sampler_enabled") is not False
            or record.get("model_runner_v2") is not True or record.get("multiprocess_method") != "spawn"
            or record.get("actual_model") != SNAPSHOT or record.get("actual_tokenizer") != SNAPSHOT
            or record.get("prefix_caching") is not True or record.get("chunked_prefill") is not True
            or record.get("async_scheduling") is not False or record.get("speculative_disabled") is not True
            or record.get("dtype") not in ("torch.bfloat16", "bfloat16") or record.get("kv_dtype") != "auto"
            or record.get("connector") != "NixlConnector"
            or record.get("transfer_role") != ("kv_producer" if role == "prefill" else "kv_consumer")):
        return False
    from .transport import profile
    if record.get("transport_profile") != profile():
        return False
    parallel = record.get("parallel")
    if type(parallel) is not dict or any(type(parallel.get(k)) is not int or parallel[k] != 1
                                        for k in ("tp", "pp", "dp", "dcp", "pcp")):
        return False
    if (any(type(record.get(k)) is not int for k in ("gpu_blocks", "block_size", "max_model_len", "max_num_seqs"))
            or record["gpu_blocks"] < MIN_GPU_BLOCKS or record["block_size"] != BLOCK_SIZE
            or ("gpu_blocks_override" in plan and (type(plan["gpu_blocks_override"]) is not int
                or plan["gpu_blocks_override"] != 8192 or record["gpu_blocks"] != plan["gpu_blocks_override"]))
            or record["max_model_len"] != MAX_MODEL_LEN or record["max_num_seqs"] != MAX_SEQS):
        return False
    specs = record.get("group_specs")
    return (type(specs) is list and len(specs) == 1 and specs[0] == dict(type="FullAttentionSpec",
        block_size=BLOCK_SIZE, sliding_window=None, attention_chunk_size=None, non_causal=False))



def transport_capture(directory, *, plan, arm, role, process, startup_process_id, membership=None):
    """Require observed native acceptance, separately from plugin installation."""
    from .transport import validate_receipt
    from .cleanup import processes
    receipts, errors = [], []
    if membership is None:
        rows = {pid: dict(pid=pid, **row) for pid, row in processes().items()}
    else:
        rows = {row["pid"]: row for row in membership}
    anchor = getattr(process, "_owner_process_identity", {})
    paths = sorted(Path(directory).glob("transport-*.json"))
    if not paths or len(paths) > 64:
        errors.append("native transport receipt count is missing or unsupported")
    for path in paths[:64]:
        try:
            value = json.loads(regular_bytes(path, 512 * 1024))
            if (not validate_receipt(value, plan["run_id"], plan["packet_sha256"], arm["arm_id"],
                    arm["engine_session_id"], role)
                    or path.name != "transport-" + str(value["context"]["process_id"]) + ".json"):
                raise ValueError("native transport acceptance or context differs")
            identity = value["process_identity"]
            if type(startup_process_id) is not int or identity["pid"] != startup_process_id:
                raise ValueError("native UCX constructor is not the supported uni scheduler process")
            row = rows.get(identity["pid"], {})
            if (anchor.get("pid") != process.pid or type(anchor.get("start")) is not int
                    or any(row.get(k) != v for k, v in identity.items())
                    or identity["group"] != process.pid or identity["session"] != process.pid):
                raise ValueError("native transport receipt is not bound to this role's observed session")
            receipts.append(value)
        except Exception as error:
            errors.append(path.name + ": " + str(error)[:512])
    return dict(complete=not errors, role=role, receipts=receipts, errors=errors)


def wait_ready(process, directory, *, plan, arm, role, deadline, stop_event=None):
    from .supervisor import poll_engine
    until = min(deadline, time.monotonic() + STARTUP_SECONDS)
    directory = Path(directory)
    while time.monotonic() < until:
        if stop_event is not None and stop_event.is_set():
            raise ValueError("the ordinary owner run reached its deadline or retained-log bound")
        code = poll_engine(process)
        if code is not None:
            raise ValueError("ordinary " + role + " server exited before readiness with status " + str(code))
        errors = list(directory.glob("errors-*.json"))
        if errors:
            raise ValueError("passive startup recorder reported an error: " + errors[0].name)
        records = list(directory.glob("startup-*.json"))
        if len(records) > 1:
            raise ValueError("multiple scheduler startup records are unsupported")
        if records:
            record = json.loads(regular_bytes(records[0], 512 * 1024))
            if not startup_valid(record, plan=plan, arm=arm, role=role):
                raise ValueError("actual ordinary " + role + " configuration is unsupported")
            from .listeners import owned as owned_listener
            if not owned_listener(PORTS[role], process):
                time.sleep(.25)
                continue
            connection = http.client.HTTPConnection("127.0.0.1", PORTS[role], timeout=2)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                response.read(4096)
                if response.status == 200:
                    observed = transport_capture(directory, plan=plan, arm=arm, role=role, process=process,
                        startup_process_id=record["process_id"])
                    if not observed["complete"]:
                        raise ValueError("native transport evidence is incomplete: " + str(observed["errors"]))
                    return dict(record, native_transport=observed)
            except (OSError, http.client.HTTPException):
                pass
            finally:
                connection.close()
        time.sleep(.25)
    raise TimeoutError("ordinary " + role + " startup deadline reached")


def closed_capture(directory, *, plan, arm, startup, role):
    directory = Path(directory)
    expected_context = dict(run_id=plan["run_id"], packet_sha256=plan["packet_sha256"],
                            arm_id=arm["arm_id"], engine_session_id=arm["engine_session_id"])
    snapshots, bindings, cache, errors = [], [], [], []
    for pattern, selected in (("checksum-*.json", snapshots), ("bindings-*.json", bindings), ("cache-*.json", cache)):
        for path in sorted(directory.glob(pattern)):
            if "-disabled-" in path.name:
                continue
            value = json.loads(regular_bytes(path, 32 * 1024 * 1024))
            if (value.get("schema") != "sdcguard.owner-checksum-close.v1" or value.get("role") != role
                    or value.get("original_shutdown_completed") is not True
                    or value.get("original_shutdown_error") is not None or value.get("recorder_errors") != []
                    or type(value.get("context")) is not dict
                    or any(value["context"].get(k) != v for k, v in expected_context.items())):
                errors.append("native close receipt is incomplete: " + path.name)
            if type(value.get("snapshot")) is dict:
                selected.append(value["snapshot"])
    if list(directory.glob("errors-*.json")):
        errors.append("passive recorder error file present")
    if role == "decode" and not snapshots:
        errors.append("native scheduler checksum close receipt missing")
    if not cache:
        errors.append("native prefill accounting close receipt missing")
    if not bindings:
        errors.append("native API request-binding close receipt missing")
    return dict(arm_id=arm["arm_id"], engine_session_id=arm["engine_session_id"], source_revision=plan["source_revision"],
        capture_complete=not errors, startup=startup, startup_sha256=digest(encoded(startup)),
        checksum_snapshots=snapshots, binding_snapshots=bindings, cache_snapshots=cache, capture_errors=errors)


def _archive(root):
    files = [p for p in sorted(Path(root).rglob("*")) if p.is_file()]
    if len(files) > 1024:
        raise ValueError("owner result file count exceeds its bound")
    stream, total = io.BytesIO(), 0
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            raw = regular_bytes(path, 32 * 1024 * 1024)
            total += len(raw)
            if total > 128 * 1024 * 1024:
                raise ValueError("owner result bytes exceed their bound")
            archive.writestr(path.relative_to(root).as_posix(), raw)
    return stream.getvalue()


def execute(evidence_root, *, plan_raw, payloads_raw, preflight, commit):
    """No caller except the owner's explicit GPU function may execute this body."""
    from .preflight import model_cache
    from .installed import product_identity, validate, profile
    from .identity import installed_identity, validate_pinned_source
    from .client import ClientCapture
    from .cleanup import subreaper, baseline, register, cleanup
    from .supervisor import launch, start
    from .config import BUNDLE, ENGINE_ROOT
    packet_sha256 = preflight["packet_sha256"]
    if len(plan_raw) + len(payloads_raw) > 8 * 1024 * 1024:
        raise ValueError("operator input exceeds its bound")
    plan, payloads = json.loads(plan_raw), json.loads(payloads_raw)
    from ..checksum_profiles import directory
    bundle = directory(plan["source_revision"])
    from .workload import validate_payloads
    validate_payloads(plan, payloads)
    if encoded(plan) != plan_raw or plan.get("packet_sha256") != packet_sha256 or not validate(preflight, plan, plan_raw):
        raise ValueError("the independent client plan bytes differ")
    out = Path(evidence_root) / plan["run_id"]
    out.mkdir(exist_ok=False)
    status = dict(owner="Hari Krishna Kancharla", run_id=plan["run_id"], packet_sha256=packet_sha256,
        complete=False, stage="gpu_preflight", failure=None, runtime_qualified=False,
        scope="ordinary same-host checksum observation and client reconciliation only")
    checkpoint(out / "STARTED.json", status)
    commit()
    deadline = time.monotonic() + JOB_SECONDS - 150
    captures, owned, streams, cleanups, transports = [], [], [], [], []
    client = None
    stop_watch = threading.Event()
    try:
        from .python_runtime import require_interpreter
        python_identity = require_interpreter()
        checkpoint(out / "PYTHON_IDENTITY.json", python_identity)
        if (product_identity() != preflight["product"]
                or installed_identity(bundle) != preflight["installed_engine"]
                or validate_pinned_source(Path(ENGINE_ROOT), profile("ENGINE_BUILD_LOCK.json", plan["source_revision"])) != preflight["pinned_source_guard"]
                or python_identity != preflight["installed_engine"]["python_identity"]
                or model_cache(bundle) != preflight["model_cache"]):
            raise ValueError("installed runtime changed after CPU preflight")
        checkpoint(out / "BUILD_WITNESS.json", dict(witness=preflight["build_witness"], sha256=preflight["build_witness_sha256"]))
        import torch
        if (torch.cuda.device_count() != 1 or torch.version.cuda is None or not torch.version.cuda.startswith("13.")
                or "H100" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (9, 0)):
            raise ValueError("the allocated GPU or CUDA runtime differs from the supported H100 profile")
        import nixl._api as nixl_api
        if "nixl_cu13" not in str(nixl_api.__file__):
            raise ValueError("NIXL did not select the reviewed CUDA13 backend")
        checkpoint(out / "GPU_PROFILE.json", dict(name=torch.cuda.get_device_name(0),
            capability=list(torch.cuda.get_device_capability(0)), cuda=torch.version.cuda,
            nixl_api_file=str(nixl_api.__file__), source_revision=plan["source_revision"]))
        subreaper()
        before = baseline()
        client = ClientCapture(plan, payloads, out / "client")

        def watch():
            while not stop_watch.wait(.5):
                try:
                    size = sum(p.stat().st_size for p in out.rglob("*.log") if p.is_file())
                    if size > 32 * 1024 * 1024 or time.monotonic() >= deadline:
                        client.stop.set()
                        return
                except OSError:
                    client.stop.set()
                    return
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        for arm in plan["arms"]:
            status["stage"] = "startup:" + arm["arm_id"]
            arm_dir = out / arm["arm_id"]
            arm_dir.mkdir()
            registry = arm_dir / "PROCESS_GROUPS.json"
            starts, role_processes = {}, {}
            owned, streams = [], []
            try:
                from .listeners import require_available
                require_available()
                for role in ("prefill", "decode"):
                    directory = arm_dir / role
                    directory.mkdir()
                    stream = (arm_dir / (role + ".log")).open("xb")
                    streams.append(stream)
                    env = environment(os.environ, out=directory, run_id=plan["run_id"],
                        packet_sha256=packet_sha256, arm_id=arm["arm_id"],
                        engine_session_id=arm["engine_session_id"], role=role, enabled=arm["observer_enabled"],
                        revision=plan["source_revision"])
                    capacity = dict(gpu_blocks=plan["gpu_blocks_override"]) if "gpu_blocks_override" in plan else {}
                    process = launch(server_command(role, **capacity), env=env, stdout=stream,
                                     directory=directory, label=role)
                    owned.append(process)
                    role_processes[role] = process
                    try:
                        register(registry, process, role)
                    except BaseException:
                        from .supervisor import cancel_unstarted
                        cancellation = cancel_unstarted(process)
                        checkpoint(directory / "REGISTRATION_CANCELLED.json", cancellation)
                        owned.remove(process)
                        raise
                    start(process, timeout=min(10, max(.01, deadline-time.monotonic())))
                    starts[role] = wait_ready(process, directory, plan=plan, arm=arm, role=role,
                                             deadline=deadline, stop_event=client.stop)
                status["stage"] = "ordinary_traffic:" + arm["arm_id"]
                client.transport = owned_transport(role_processes, client.stop)
                client.run_arm(arm["arm_id"], deadline)
            finally:
                cleanup_record = cleanup(leaders=owned, scope=os.getpid(), baseline=before,
                                         registry=registry, grace=45, kill_wait=5)
                cleanups.append(cleanup_record)
                checkpoint(arm_dir / "CLEANUP.json", cleanup_record)
                transport_record = dict(arm_id=arm["arm_id"], roles=[transport_capture(arm_dir / role,
                    plan=plan, arm=arm, role=role, process=role_processes[role],
                    startup_process_id=starts.get(role, {}).get("process_id"),
                    membership=cleanup_record.get("process_membership", [])) for role in role_processes])
                transport_record["complete"] = len(transport_record["roles"]) == 2 and all(row["complete"] for row in transport_record["roles"])
                transports.append(transport_record)
                checkpoint(arm_dir / "TRANSPORT.json", transport_record)
                for stream in streams:
                    stream.close()
                streams = []
                owned = []
                if arm["observer_enabled"]:
                    capture = closed_capture(arm_dir / "decode", plan=plan, arm=arm, startup=starts.get("decode", {}), role="decode")
                    capture["prefill_capture"] = closed_capture(arm_dir / "prefill", plan=plan, arm=arm, startup=starts.get("prefill", {}), role="prefill")
                    captures.append(capture)
            commit()
            if (cleanup_record.get("complete") is not True or transport_record["complete"] is not True
                    or client.stop.is_set()):
                raise ValueError("ordinary arm did not complete; no replacement traffic will be sent")
        status["stage"] = "assessment"
    except BaseException as error:
        status["failure"] = dict(stage=status["stage"], kind=type(error).__name__, detail=str(error)[:1024])
    finally:
        stop_watch.set()
        for stream in streams:
            stream.close()
        if client is not None:
            inventory = client.finish()
        else:
            from .client import initial_inventory
            inventory = initial_inventory(plan, plan_raw)
            (out / "client").mkdir(exist_ok=True)
            checkpoint(out / "client/REQUEST_PLAN.json", plan)
            checkpoint(out / "client/CLIENT_INVENTORY.json", inventory)
        manifest = dict(schema="sdcguard.checksum-client-captures.v1", run_id=plan["run_id"],
            packet_sha256=packet_sha256, plan_sha256=digest(plan_raw), captures=captures)
        checkpoint(out / "CAPTURES.json", manifest)
        try:
            from sdcguard.checksum_client import evaluate
            assessment = evaluate(plan_raw, encoded(inventory), encoded(manifest))
            checkpoint(out / "CHECKSUM_JOIN.json", assessment)
            status["coverage_summary"] = assessment["summary"]
        except Exception as error:
            status["assessment_error"] = dict(kind=type(error).__name__, detail=str(error)[:512])
        status.update(complete=status["failure"] is None and "assessment_error" not in status and inventory["completed"] is True
                      and bool(cleanups) and all(row.get("complete") is True for row in cleanups)
                      and len(transports) == len(plan["arms"]) and all(row["complete"] for row in transports),
                      client_inventory_complete=inventory["completed"],
                      native_transport_complete=len(transports) == len(plan["arms"]) and all(row["complete"] for row in transports),
                      cleanup_complete=bool(cleanups)
                      and all(row.get("complete") is True for row in cleanups), stage="complete")
        checkpoint(out / "FINISHED.json", status)
        commit()
    archive = None
    if status["cleanup_complete"]:
        try:
            archive = _archive(out)
        except Exception as error:
            status.update(complete=False, archive_error=dict(kind=type(error).__name__, detail=str(error)[:512]))
            checkpoint(out / "FINISHED.json", status)
            commit()
    else:
        status["archive_withheld"] = "owned process cleanup is incomplete; raw evidence remains on the volume"
        checkpoint(out / "FINISHED.json", status)
        commit()
    return dict(report=status, archive=archive, archive_sha256=digest(archive) if archive is not None else None,
                packet_sha256=packet_sha256, run_id=plan["run_id"], build_witness_sha256=preflight["build_witness_sha256"])
