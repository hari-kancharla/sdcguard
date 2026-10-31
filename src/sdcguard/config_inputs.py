"""Bounded, read-only static deployment inputs and saved startup evidence.

Bindings establish consistency of operator-supplied bytes only. They do not
authenticate a deployment or attest that an engine ran this configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


SCHEMA = "sdcguard.config-safety.v1"
MAX_CONFIG_BYTES = 1024 * 1024
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_JSON_NODES = 100_000
BINDING_SCOPE = "operator-supplied consistency only; no authenticated attestation"
_ROOT_KEYS = {"schema", "deployment_id", "advertised_context_tokens", "model", "workers"}
_MODEL_KEYS = {"id", "revision", "architecture"}
_WORKER_KEYS = {"id", "role", "engine", "version", "build_commit", "components", "connector",
                "parameters", "environment", "layout", "startup_log"}
_LOG_KEYS = {"path", "sha256", "config_sha256", "worker_id"}


def _no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _json_values(value, depth=0, count=None, *, max_nodes=MAX_JSON_NODES):
    """Bound nesting and reject escaped surrogates and overflowed JSON numbers."""
    if count is None:
        if type(max_nodes) is not int or max_nodes <= 0:
            raise ValueError("JSON node budget must be a positive integer")
        count = [0]
    count[0] += 1
    if depth > 64 or count[0] > max_nodes:
        raise ValueError("JSON structure exceeds supported bounds")
    if isinstance(value, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("JSON contains an unpaired Unicode surrogate")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            _json_values(key, depth + 1, count, max_nodes=max_nodes)
            _json_values(item, depth + 1, count, max_nodes=max_nodes)
    elif isinstance(value, list):
        for item in value:
            _json_values(item, depth + 1, count, max_nodes=max_nodes)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON contains a non-finite number")
    elif value is not None and type(value) not in (bool, int):
        raise ValueError("value is not supported JSON")


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory(path):
    """Open a directory without following any path-component symlink."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("secure no-symlink file reads are unavailable on this platform")
    absolute = Path(path).absolute()
    fd = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in absolute.parts[1:]:
            if part in ("", ".", ".."):
                raise ValueError("directory traversal is not supported")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_regular(base, parts, limit):
    """Read at most limit bytes through stable, no-follow directory descriptors."""
    directory = _directory(base)
    fd = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("input must be a regular file")
        if before.st_size > limit:
            raise ValueError(f"input exceeds {limit}-byte limit")
        chunks = []
        total = 0
        while True:
            block = os.read(fd, min(64 * 1024, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > limit:
                raise ValueError(f"input exceeds {limit}-byte limit")
        after = os.fstat(fd)
        named = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        if (not stat.S_ISREG(named.st_mode) or _signature(before) != _signature(after)
                or _signature(after) != _signature(named) or total != after.st_size):
            raise ValueError("input changed while it was read")
        return b"".join(chunks)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(directory)


def _object(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(sorted(unknown))}")


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _hex(value, length, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{" + str(length) + r"}", value) is None:
        raise ValueError(f"{name} must contain {length} hexadecimal characters")


def _log_metadata(value):
    _object(value, _LOG_KEYS, "startup_log")
    if not {"path", "sha256", "config_sha256"}.issubset(value):
        raise ValueError("startup_log requires path, sha256 and config_sha256")
    _string(value["path"], "startup_log.path")
    _hex(value["sha256"], 64, "startup_log.sha256")
    _hex(value["config_sha256"], 64, "startup_log.config_sha256")
    if "worker_id" in value:
        _string(value["worker_id"], "startup_log.worker_id")


def _validate(document):
    _object(document, _ROOT_KEYS, "deployment")
    if document.get("schema") != SCHEMA:
        raise ValueError(f"deployment schema must be {SCHEMA}")
    if "deployment_id" in document:
        _string(document["deployment_id"], "deployment_id")
    if "advertised_context_tokens" in document:
        length = document["advertised_context_tokens"]
        if type(length) is not int or length <= 0:
            raise ValueError("advertised_context_tokens must be a positive integer")
    if "model" in document:
        model = document["model"]
        _object(model, _MODEL_KEYS, "model")
        for key in ("id", "architecture"):
            if key in model:
                _string(model[key], f"model.{key}")
        if "revision" in model:
            _hex(model["revision"], 40, "model.revision")
    workers = document.get("workers")
    if not isinstance(workers, list) or not 1 <= len(workers) <= 32:
        raise ValueError("workers must be an array with 1 to 32 workers")
    ids = set()
    for worker in workers:
        _object(worker, _WORKER_KEYS, "worker")
        _string(worker.get("id"), "worker.id")
        if worker["id"] in ids:
            raise ValueError(f"duplicate worker id: {worker['id']}")
        ids.add(worker["id"])
        if "role" in worker and worker["role"] not in ("prefill", "decode"):
            raise ValueError("worker.role must be prefill or decode")
        for key in ("engine", "version"):
            if key in worker:
                _string(worker[key], f"worker.{key}")
        if "build_commit" in worker:
            _hex(worker["build_commit"], 40, "worker.build_commit")
        for key in ("components", "connector", "parameters", "environment", "layout"):
            if key in worker and not isinstance(worker[key], dict):
                raise ValueError(f"worker.{key} must be a JSON object")
        if "startup_log" in worker:
            _log_metadata(worker["startup_log"])


def binding_digest(worker, deployment):
    """Bind worker settings to the deployment's model and advertised context.

    startup_log is excluded to avoid a circular digest. A deployment is required:
    a worker-only digest cannot notice a changed root model or context contract.
    """
    if not isinstance(worker, dict) or not isinstance(deployment, dict):
        raise ValueError("worker binding requires worker and deployment JSON objects")
    supplied = {"worker": {key: value for key, value in worker.items() if key != "startup_log"},
                "model": deployment.get("model", {}),
                "advertised_context_tokens": deployment.get("advertised_context_tokens")}
    _json_values(supplied)
    try:
        raw = json.dumps(supplied, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("worker binding is not supported JSON") from exc
    return hashlib.sha256(raw).hexdigest()


def read_startup(worker, base, deployment):
    """Return frozen saved log bytes as text, or an explicit missing/bad evidence error.

    Paths are relative to base, cannot traverse or follow symlinks, and are not
    interpreted as URLs. Binding is consistency only, never authentication.
    """
    supplied_path = None
    try:
        if not isinstance(worker, dict):
            raise ValueError("worker must be a JSON object")
        metadata = worker.get("startup_log")
        if metadata is None:
            raise ValueError("startup log not supplied")
        _log_metadata(metadata)
        supplied_path = metadata["path"]
        relative = Path(supplied_path)
        parts = supplied_path.split("/")
        if (relative.is_absolute() or "\\" in supplied_path or "\x00" in supplied_path
                or any(part in ("", ".", "..") for part in parts)):
            raise ValueError("startup log path must be relative without traversal")
        if "worker_id" in metadata and metadata["worker_id"] != worker.get("id"):
            raise ValueError("startup log worker_id does not match worker")
        if metadata["config_sha256"].lower() != binding_digest(worker, deployment):
            raise ValueError("startup log config_sha256 does not match worker binding")
        raw = _read_regular(base, parts, MAX_LOG_BYTES)
        sha256 = hashlib.sha256(raw).hexdigest()
        if sha256 != metadata["sha256"].lower():
            raise ValueError("startup log sha256 does not match file bytes")
        text = raw.decode("utf-8")
        return {"text": text, "sha256": sha256, "path": str(Path(base).absolute() / relative),
                "binding_scope": BINDING_SCOPE}
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        return {"error": str(exc), "path": supplied_path, "binding_scope": BINDING_SCOPE}


def load_deployment(path):
    """Load strict static JSON plus bound saved logs, preserving missing facts.

    Missing model/build/version facts remain absent so the evaluator can report
    unknown. Structural input errors raise ValueError. Log errors are retained
    per worker and must not become successful capacity evidence.
    """
    try:
        source = Path(path).absolute()
        raw = _read_regular(source.parent, [source.name], MAX_CONFIG_BYTES)
        text = raw.decode("utf-8")
        document = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_invalid_constant)
        _json_values(document)
        _validate(document)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"invalid deployment input: {exc}") from exc
    provenance = {"source_path": str(source), "raw_config_sha256": hashlib.sha256(raw).hexdigest(),
                  "logs": {worker["id"]: read_startup(worker, source.parent, document)
                           for worker in document["workers"]},
                  "binding_scope": BINDING_SCOPE}
    return document, provenance
