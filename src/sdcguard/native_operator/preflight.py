"""CPU-only installed-profile prerequisites. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import uuid

from .config import SNAPSHOT, BUNDLE, ENGINE_ROOT
from .workload import MODEL, MODEL_REVISION, REVISION, ARMS, build_plan
from .common import checkpoint, digest, encoded, regular_bytes

def model_cache(bundle, snapshot=SNAPSHOT):
    snapshot = Path(snapshot)
    expected = json.loads(regular_bytes(Path(bundle) / "artifacts/MODEL_PIN.json", 128 * 1024))
    if (expected["model"] != MODEL or expected["revision"] != MODEL_REVISION
            or snapshot.name != MODEL_REVISION or not snapshot.is_dir()):
        raise ValueError("the exact ordinary model snapshot is unavailable")
    cache_root = snapshot.parents[2].resolve(strict=True)
    observed = {}
    for name, expected_hash in expected["files"].items():
        original = snapshot / name
        path = original.resolve(strict=True)
        if cache_root not in path.parents:
            raise ValueError("a model file resolves outside its existing cache")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 * 1024 ** 3:
                raise ValueError("model cache member size is unsupported")
            hasher, size = hashlib.sha256(), 0
            while chunk := os.read(descriptor, 1024 * 1024):
                size += len(chunk)
                hasher.update(chunk)
            after = os.fstat(descriptor)
            if (hasher.hexdigest() != expected_hash or size != before.st_size
                    or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
                    != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)):
                raise ValueError("the pinned model cache bytes differ: " + name)
            observed[name] = dict(bytes=size, sha256=hasher.hexdigest())
        finally:
            os.close(descriptor)
    return dict(model=MODEL, revision=MODEL_REVISION, snapshot=str(snapshot), files=observed,
                scope="local cached bytes before execution; no loaded-weight or device attestation")


def execute(*, run_id=None, expected_identity=None):
    """Check the local CPU runtime only. Never starts a server or requests a GPU."""
    from . import installed, python_runtime, cleanup_preflight, transport
    from .identity import installed_identity, validate_pinned_source
    run_id = run_id or uuid.uuid4().hex
    product = installed.product_identity()
    packet = product["sha256"]
    value = dict(schema=installed.SCHEMA, owner="Hari Krishna Kancharla", run_id=run_id,
        packet_sha256=packet, source_revision=REVISION, status="UNKNOWN", failure=None,
        cpu_only=True, gpu_submitted=False, runtime_qualified=False, product=product)
    stage = "installed_engine"
    plan = payloads = None
    try:
        value["installed_engine"] = installed_identity(BUNDLE)
        value["pinned_source_guard"] = validate_pinned_source(Path(ENGINE_ROOT), installed.profile("ENGINE_BUILD_LOCK.json"))
        stage = "python_runtime"
        value[stage] = python_runtime.probe(run_id, packet)
        if not python_runtime.validate(value[stage], run_id, packet):
            raise ValueError("Python interpreter, headers or dependencies did not pass")
        stage = "cleanup_capability"
        value[stage] = cleanup_preflight.probe(run_id, packet)
        if not cleanup_preflight.validate(value[stage], run_id, packet):
            raise ValueError("strict process cleanup preflight did not pass")
        stage = "transport_readiness"
        value[stage] = transport.probe(run_id, packet)
        if not transport.validate(value[stage], run_id, packet):
            raise ValueError("same-host native transport preflight did not pass")
        stage = "model_cache"
        value[stage] = model_cache(BUNDLE)
        stage = "tokenizer"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False)
        sessions = {arm: uuid.uuid4().hex for arm, _ in ARMS}
        plan, payloads = build_plan(tokenizer.encode, run_id=run_id,
                                    packet_sha256=packet, engine_sessions=sessions)
        value["plan_sha256"] = digest(encoded(plan))
        if not any(row["prompt_tokens"] % 16 for arm in plan["arms"] for row in arm["requests"]):
            raise ValueError("ordinary tokenizer workload lacks a partial final block")
        if expected_identity is not None and product != expected_identity:
            raise ValueError("installed product differs from the previously checked runtime")
        value["status"] = "PASS"
        value["build_witness"] = installed.witness(value)
        value["build_witness_sha256"] = digest(encoded(value["build_witness"]))
        if not installed.validate(value, plan, encoded(plan)):
            raise ValueError("installed runtime receipt did not satisfy its offline contract")
    except Exception as error:
        value.update(status="UNKNOWN", failure=dict(stage=stage, kind=type(error).__name__, detail=str(error)[:1024]))
    return value, plan, payloads

