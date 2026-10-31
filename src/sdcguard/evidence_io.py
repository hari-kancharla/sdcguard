"""Bounded local packet IO. Hashes establish consistency, not authenticity.

Owner: Hari Krishna Kancharla.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from . import config_inputs

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 512
MAX_JSON_NODES = 1_000_000


class EvidenceError(ValueError):
    pass


def json_bytes(value):
    config_inputs._json_values(value, max_nodes=MAX_JSON_NODES)
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_json(raw, *, max_nodes=MAX_JSON_NODES):
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE_BYTES:
        raise EvidenceError("JSON input exceeds the bounded regular-file contract")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=config_inputs._no_duplicates,
                           parse_constant=config_inputs._invalid_constant)
        config_inputs._json_values(value, max_nodes=max_nodes)
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise EvidenceError("malformed strict JSON: " + str(exc)) from exc


def relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(part in ("", ".", "..") or ":" in part
                                           for part in name.split("/"))):
        raise EvidenceError("unsafe relative evidence path")
    return name.split("/")


def read_file(root, name):
    try:
        return config_inputs._read_regular(Path(root), relative(name), MAX_FILE_BYTES)
    except (OSError, ValueError) as exc:
        raise EvidenceError("cannot read captured file " + str(name) + ": " + str(exc)) from exc


def names(root, *, directories=False):
    """List only regular members through no-follow directory descriptors."""
    found = []
    directory_paths = []
    visited = 0

    def visit(directory, prefix):
        nonlocal visited
        visited += 1
        if visited > MAX_FILES or prefix.count("/") > 32:
            raise EvidenceError("packet directory structure exceeds the supported bound")
        for name in sorted(os.listdir(directory)):
            relative(name)
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            path = prefix + name
            if stat.S_ISDIR(info.st_mode):
                directory_paths.append(path)
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                try:
                    visit(child, path + "/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                found.append(path)
                if len(found) > MAX_FILES:
                    raise EvidenceError("packet file count exceeds the supported bound")
            else:
                raise EvidenceError("packet contains a link or non-regular member")

    try:
        directory = config_inputs._directory(Path(root).absolute())
        try:
            visit(directory, "")
        finally:
            os.close(directory)
    except (OSError, ValueError) as exc:
        raise EvidenceError("cannot inventory regular packet: " + str(exc)) from exc
    return sorted(directory_paths if directories else found)


def snapshot(root):
    inventory = names(root)
    result = {}
    size = 0
    for name in inventory:
        data = read_file(root, name)
        size += len(data)
        if size > MAX_TOTAL_BYTES:
            raise EvidenceError("packet byte count exceeds the supported bound")
        result[name] = data
    if names(root) != inventory:
        raise EvidenceError("packet membership changed during capture")
    # Compare complete bytes, not just metadata or a possibly stale listing.
    for name, raw in result.items():
        if read_file(root, name) != raw:
            raise EvidenceError("packet bytes changed during capture")
    return result


def create_directory(path):
    """Create one new private destination; its parent must already exist."""
    target = Path(path).absolute()
    if any(part in (".", "..") for part in target.parts):
        raise EvidenceError("destination traversal is unsupported")
    parent = config_inputs._directory(target.parent)
    try:
        os.mkdir(target.name, mode=0o700, dir_fd=parent)
    finally:
        os.close(parent)
    return target


def make_directory(root, name):
    parts = relative(name)
    directory = config_inputs._directory(Path(root).absolute())
    try:
        for part in parts:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
    finally:
        os.close(directory)


def write_file(root, name, raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_FILE_BYTES:
        raise EvidenceError("output file exceeds the supported byte bound")
    parts = relative(name)
    directory = config_inputs._directory(Path(root).absolute())
    try:
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=directory)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory)
    finally:
        os.close(directory)


def write_snapshot(root, files, prefix=""):
    if len(files) > MAX_FILES or sum(len(raw) for raw in files.values()) > MAX_TOTAL_BYTES:
        raise EvidenceError("captured input exceeds packet bounds")
    for name, raw in sorted(files.items()):
        relative(name)
        write_file(root, prefix + name, raw)


def file_receipts(files):
    return {name: {"sha256": digest(raw), "bytes": len(raw)} for name, raw in sorted(files.items())}
