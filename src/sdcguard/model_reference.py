"""Bind one observed local model reference to stable pinned files.

Owner: Hari Krishna Kancharla.
The producer hashes local files before engine startup. Recorded receipts are
local consistency evidence, not proof of loaded weights or deployment identity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import weakref


SCHEMA = "sdcguard.model-reference.v1"
OBSERVED_SCHEMA = "sdcguard.observed-model-reference.v1"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PIN_SHA256 = "773a13fe7f55622feb7322e4431b44c1ed34f1b5b06f825db95114a66ae30981"
PIN_RESOURCE = "case_data/61229/MODEL_PIN.json"
ENV_FILE = "SDCGUARD_MODEL_REFERENCE_FILE"
ENV_RUN = "SDCGUARD_MODEL_REFERENCE_RUN_ID"
ENV_PACKET = "SDCGUARD_MODEL_REFERENCE_PACKET_SHA256"
MAX_BYTES = 64 * 1024
MAX_PATH = 4096
MAX_FILE_BYTES = 2 * 1024**3
MAX_TOTAL_BYTES = 2 * 1024**3
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_SIGNED = re.compile(r"(?:0|-?[1-9][0-9]{0,18})\Z")
_STAT_KEYS = {"device", "inode", "mode", "size", "mtime_ns", "ctime_ns"}
_KEYS = {"schema", "model", "revision", "run_id", "packet_sha256", "configured_snapshot",
         "resolved_snapshot", "model_pin_sha256", "files", "basis", "loaded_weights", "authenticity"}
_FILE_KEYS = {"sha256", "bytes", "configured_path", "resolved_path", "entry", "target"}


class ModelReferenceError(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise ModelReferenceError(reason)


def _path(value):
    _require(type(value) is str and 0 < len(value) <= MAX_PATH
             and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in value)
             and "\\" not in value and value.startswith("/") and "//" not in value,
             "model reference path is not an exact bounded absolute path")
    p = PurePosixPath(value)
    _require(str(p) == value and all(part not in (".", "..") for part in value.split("/")[1:]),
             "model reference path contains an alias or traversal")
    return value


def _signature(value):
    return dict(device=format(value.st_dev, "016x"), inode=format(value.st_ino, "016x"),
                mode=value.st_mode, size=value.st_size,
                mtime_ns=str(value.st_mtime_ns), ctime_ns=str(value.st_ctime_ns))


def _validate_stat(value, *, target=False):
    _require(type(value) is dict and value.keys() == _STAT_KEYS,
             "model reference file stat receipt is unsupported")
    _require(all(type(value[k]) is str and _HEX16.fullmatch(value[k]) for k in ("device", "inode"))
             and type(value["mode"]) is int and 0 <= value["mode"] < 2**32
             and type(value["size"]) is int and 0 <= value["size"] <= MAX_FILE_BYTES,
             "model reference file stat identity or size is unsupported")
    _require(all(type(value[k]) is str and _SIGNED.fullmatch(value[k])
                 and -(2**63) <= int(value[k]) < 2**63 for k in ("mtime_ns", "ctime_ns")),
             "model reference file time receipt is unsupported")
    _require(stat.S_ISREG(value["mode"]) if target else
             stat.S_ISREG(value["mode"]) or stat.S_ISLNK(value["mode"]),
             "model reference file kind is unsupported")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "model reference JSON has duplicate keys")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ModelReferenceError("nonfinite model reference JSON is unsupported")


def _decode(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES, "model reference JSON exceeds its byte bound")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ModelReferenceError("model reference JSON is invalid: " + str(error)) from None
    pending = [(result, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        _require(nodes <= 8192 and depth <= 16, "model reference JSON exceeds its structure bound")
        if type(value) is dict:
            pending.extend((child, depth + 1) for child in value.values())
        elif type(value) is list:
            pending.extend((child, depth + 1) for child in value)
    return result


def _pin():
    raw = files("sdcguard").joinpath(PIN_RESOURCE).read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == PIN_SHA256, "installed model pin bytes differ")
    value = _decode(raw)
    _require(type(value) is dict and value.keys() == {"model", "revision", "files"}
             and value["model"] == MODEL and value["revision"] == REVISION
             and type(value["files"]) is dict and len(value["files"]) == 10,
             "installed model pin is unsupported")
    return value


def _run(run_id, packet_sha256):
    _require(type(run_id) is str and _HEX32.fullmatch(run_id)
             and type(packet_sha256) is str and _HEX64.fullmatch(packet_sha256),
             "model reference run or packet identity is unsupported")


def validate_capsule(value):
    """Validate recorded facts only, without reading the historical model files."""
    _require(type(value) is dict and value.keys() == _KEYS and value.get("schema") == SCHEMA,
             "model reference capsule fields or version differ")
    pin = _pin()
    _run(value["run_id"], value["packet_sha256"])
    _require(all(type(value[key]) is str for key in ("model", "revision", "model_pin_sha256", "basis",
                                                     "loaded_weights", "authenticity"))
             and value["model"] == MODEL and value["revision"] == REVISION
             and value["model_pin_sha256"] == PIN_SHA256,
             "model reference is not bound to the installed model pin")
    root, resolved = _path(value["configured_snapshot"]), _path(value["resolved_snapshot"])
    _require(PurePosixPath(root).name == REVISION and PurePosixPath(resolved).name == REVISION,
             "model reference is not an explicit immutable snapshot")
    _require(value["basis"] == "stable local file hashes before engine startup"
             and value["loaded_weights"] == "not observed" and value["authenticity"] == "not established",
             "model reference scope differs")
    entries = value["files"]
    _require(type(entries) is dict and entries.keys() == pin["files"].keys(),
             "model reference file inventory differs from the installed pin")
    total, targets, inodes = 0, {}, {}
    for name, digest in pin["files"].items():
        entry = entries[name]
        _require(type(entry) is dict and entry.keys() == _FILE_KEYS and entry["sha256"] == digest,
                 "model reference file digest or fields differ")
        _require(_path(entry["configured_path"]) == root + "/" + name,
                 "model reference member is outside its configured snapshot")
        _path(entry["resolved_path"])
        _validate_stat(entry["entry"])
        _validate_stat(entry["target"], target=True)
        if stat.S_ISREG(entry["entry"]["mode"]):
            _require(entry["entry"] == entry["target"]
                     and entry["resolved_path"] == resolved + "/" + name,
                     "regular model reference leaf has contradictory target facts")
        else:
            _require((entry["entry"]["device"], entry["entry"]["inode"])
                     != (entry["target"]["device"], entry["target"]["inode"]),
                     "model reference symlink and regular target claim one inode")
        _require(type(entry["bytes"]) is int and 0 < entry["bytes"] <= MAX_FILE_BYTES
                 and entry["target"]["size"] == entry["bytes"], "model reference member size differs")
        claim = entry["sha256"], entry["bytes"], tuple(sorted(entry["target"].items()))
        for identities, identity in ((targets, entry["resolved_path"]),
                                     (inodes, (entry["target"]["device"], entry["target"]["inode"]))):
            _require(identity not in identities or identities[identity] == claim,
                     "one model reference target has conflicting member receipts")
            identities[identity] = claim
        total += entry["bytes"]
        _require(total <= MAX_TOTAL_BYTES, "model reference aggregate exceeds its bound")
    raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    _require(len(raw) <= MAX_BYTES, "model reference capsule exceeds its byte bound")
    return json.loads(raw)


def capture(snapshot, *, run_id, packet_sha256):
    """Hash the ten pinned local files once in the actual engine pod.

    The caller writes this capsule before starting the engine. It must not use
    file stat receipts from a different pod or mount as this setup evidence.
    """
    _run(run_id, packet_sha256)
    configured = _path(str(Path(snapshot).absolute()))
    resolved = _path(str(Path(configured).resolve(strict=True)))
    pin = _pin()
    value = dict(schema=SCHEMA, model=MODEL, revision=REVISION, run_id=run_id,
                 packet_sha256=packet_sha256, configured_snapshot=configured, resolved_snapshot=resolved,
                 model_pin_sha256=PIN_SHA256, files={}, basis="stable local file hashes before engine startup",
                 loaded_weights="not observed", authenticity="not established")
    total = 0
    for name, expected in pin["files"].items():
        path = Path(configured) / name
        entry_stat = _signature(path.lstat())
        target = _path(str(path.resolve(strict=True)))
        descriptor = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
        try:
            before = os.fstat(descriptor)
            _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_FILE_BYTES,
                     "model reference member is not a bounded regular file")
            total += before.st_size
            _require(total <= MAX_TOTAL_BYTES, "model reference aggregate exceeds its bound")
            digest, count = hashlib.sha256(), 0
            while True:
                block = os.read(descriptor, min(1024 * 1024, MAX_FILE_BYTES + 1 - count))
                if not block:
                    break
                count += len(block)
                _require(count <= before.st_size, "model reference member changed while hashing")
                digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        _require(count == before.st_size and digest.hexdigest() == expected
                 and _signature(before) == _signature(after) == _signature(Path(target).stat())
                 and _signature(path.lstat()) == entry_stat and str(path.resolve(strict=True)) == target,
                 "model reference member bytes or identity differ from the pin")
        value["files"][name] = dict(sha256=digest.hexdigest(), bytes=count, configured_path=str(path),
                                    resolved_path=target, entry=entry_stat, target=_signature(after))
    _require(str(Path(configured).resolve(strict=True)) == resolved, "model snapshot changed during capture")
    return validate_capsule(value)


def _current_files(value):
    _require(str(Path(value["configured_snapshot"]).resolve(strict=True)) == value["resolved_snapshot"],
             "model snapshot path differs from its file receipt")
    for entry in value["files"].values():
        path, target = Path(entry["configured_path"]), Path(entry["resolved_path"])
        _require(_signature(path.lstat()) == entry["entry"] and str(path.resolve(strict=True)) == str(target)
                 and _signature(target.stat()) == entry["target"],
                 "model reference member no longer matches its setup file receipt")


def validate_observation(model, revision, value):
    """Validate a recorded local-path profile. This function performs no IO to weights."""
    _require(type(value) is dict and value.keys() == {"schema", "reference", "tokenizer", "tokenizer_revision"}
             and value["schema"] == OBSERVED_SCHEMA, "observed model reference fields or version differ")
    reference = validate_capsule(value["reference"])
    roots = {reference["configured_snapshot"], reference["resolved_snapshot"]}
    _require(_path(model) in roots and _path(value["tokenizer"]) in roots
             and revision == reference["revision"] and type(revision) is str
             and value["tokenizer_revision"] == reference["revision"]
             and type(value["tokenizer_revision"]) is str,
             "observed model or tokenizer reference is not bound to the pinned snapshot")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _read_capsule(path):
    path = Path(_path(path))
    _require(str(path.resolve(strict=True)) == str(path), "model reference capsule path is an alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES,
                 "model reference capsule is not a bounded regular file")
        chunks, count = [], 0
        while True:
            chunk = os.read(descriptor, MAX_BYTES + 1 - count)
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            _require(count <= MAX_BYTES, "model reference capsule grew beyond its byte bound")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(len(raw) == before.st_size and _signature(before) == _signature(after)
             == _signature(path.stat(follow_symlinks=False)), "model reference capsule changed during setup")
    return validate_capsule(_decode(raw)), _signature(after)


def _observed(config):
    return tuple(getattr(config, name, None) for name in ("model", "revision", "tokenizer", "tokenizer_revision"))


@dataclass(frozen=True, init=False, eq=False)
class ModelBinding:
    """One process/config association. File consistency is checked at setup only.

    Hot checks retain the exact config fields, environment and capsule inode.
    They do not rehash model files or attest their continuously loaded contents.
    """
    def __init__(self, model_config):
        try:
            config_reference = weakref.ref(model_config)
        except TypeError:
            raise ModelReferenceError("local model config is not weak-referenceable") from None
        environment = tuple(os.environ.get(name) for name in (ENV_FILE, ENV_RUN, ENV_PACKET))
        path, run, packet = environment
        _require(type(path) is str, "local model reference capsule is required")
        reference, signature = _read_capsule(path)
        _require(reference["run_id"] == run and reference["packet_sha256"] == packet,
                 "model reference capsule belongs to another run or packet")
        _current_files(reference)
        model, revision, tokenizer, tokenizer_revision = _observed(model_config)
        value = dict(schema=OBSERVED_SCHEMA, reference=reference,
                     tokenizer=tokenizer, tokenizer_revision=tokenizer_revision)
        value = validate_observation(model, revision, value)
        for name, field in dict(_pid=os.getpid(), _config=config_reference, _environment=environment,
                                _fields=_observed(model_config), _path=path,
                                _signature=tuple(sorted(signature.items())),
                                _raw=json.dumps(value, sort_keys=True, allow_nan=False)).items():
            object.__setattr__(self, name, field)

    def observe(self, model_config):
        actual = self._config()
        _require(os.getpid() == self._pid and actual is not None and model_config is actual,
                 "model reference binding belongs to another process or config")
        observed = _observed(model_config)
        _require(tuple(os.environ.get(name) for name in (ENV_FILE, ENV_RUN, ENV_PACKET)) == self._environment
                 and all(type(value) is str for value in observed) and observed == self._fields
                 and tuple(sorted(_signature(Path(self._path).stat(follow_symlinks=False)).items())) == self._signature,
                 "observed model reference or capsule identity changed")
        return json.loads(self._raw)


def bind(model_config):
    return None if type(getattr(model_config, "model", None)) is str and model_config.model == MODEL else ModelBinding(model_config)


def observe(model_config, binding):
    if binding is None:
        _require(type(getattr(model_config, "model", None)) is str and model_config.model == MODEL,
                 "unsupported model without a local reference binding")
        return None
    _require(type(binding) is ModelBinding, "model reference binding type is unsupported")
    return binding.observe(model_config)
