"""Stream bounded native packet inventories before copying external inputs.

Owner: Hari Krishna Kancharla.
Repeated byte checks establish local consistency, not an atomic capture or
authenticated runtime identity. Older packet IO remains a separate contract.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat

from . import config_inputs
from . import evidence_io as io

MAX_DIRECTORY_DEPTH = 32


def _bound(value, maximum, name):
    value = maximum if value is None else value
    if type(value) is not int or not 0 <= value <= maximum:
        raise io.EvidenceError("unsupported native packet " + name + " allowance")
    return value


def inventory(root, *, max_files=None, max_directories=None, max_depth=None):
    """Bound entries before retaining names, sorting or opening subdirectories."""
    files = []
    directories = []
    entries = 0
    file_limit = _bound(max_files, io.MAX_FILES, "file")
    directory_limit = _bound(max_directories, io.MAX_FILES - 1, "directory")
    depth_limit = _bound(max_depth, MAX_DIRECTORY_DEPTH, "depth")

    def visit(directory, prefix, depth):
        nonlocal entries
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                if entries > file_limit + directory_limit:
                    raise io.EvidenceError("native packet entry count exceeds its bound")
                name = entry.name
                if len(io.relative(name)) != 1:
                    raise io.EvidenceError("native packet entry is not a single name")
                path = prefix + name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if depth >= depth_limit or len(directories) >= directory_limit:
                        raise io.EvidenceError("native packet directory structure exceeds its bound")
                    directories.append(path)
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=directory)
                    try:
                        opened = os.fstat(child)
                        if (info.st_dev, info.st_ino) != (opened.st_dev, opened.st_ino):
                            raise io.EvidenceError("native packet directory changed during inventory")
                        visit(child, path + "/", depth + 1)
                    finally:
                        os.close(child)
                elif stat.S_ISREG(info.st_mode):
                    if len(files) >= file_limit:
                        raise io.EvidenceError("native packet file count exceeds its bound")
                    files.append(path)
                else:
                    raise io.EvidenceError("native packet contains a link or non-regular member")

    try:
        directory = config_inputs._directory(Path(root).absolute())
        try:
            visit(directory, "", 0)
        finally:
            os.close(directory)
    except (OSError, ValueError) as exc:
        raise io.EvidenceError("cannot inventory native packet: " + str(exc)) from exc
    return sorted(files), sorted(directories)


def _read(root, name, limit):
    try:
        return config_inputs._read_regular(Path(root).absolute(), io.relative(name), limit)
    except (OSError, ValueError) as exc:
        raise io.EvidenceError("cannot read native packet member " + name + ": " + str(exc)) from exc


def recheck_snapshot(root, snapshot, directories):
    """Recheck names and exact bytes without retaining a second full packet."""
    expected = (sorted(snapshot), directories)
    if inventory(root) != expected:
        raise io.EvidenceError("native packet membership changed during capture")
    for name, raw in snapshot.items():
        if _read(root, name, len(raw)) != raw:
            raise io.EvidenceError("native packet bytes changed during capture")
    if inventory(root) != expected:
        raise io.EvidenceError("native packet membership changed during byte recheck")


def capture_snapshot(root, *, max_files=None, max_bytes=None, max_directories=None, max_depth=None):
    """Capture supported regular files with count and cumulative byte limits."""
    files, directories = inventory(root, max_files=max_files,
                                   max_directories=max_directories, max_depth=max_depth)
    snapshot = {}
    size = 0
    byte_limit = _bound(max_bytes, io.MAX_TOTAL_BYTES, "byte")
    for name in files:
        raw = _read(root, name, min(io.MAX_FILE_BYTES, byte_limit - size))
        size += len(raw)
        snapshot[name] = raw
    recheck_snapshot(root, snapshot, directories)
    return snapshot, directories
