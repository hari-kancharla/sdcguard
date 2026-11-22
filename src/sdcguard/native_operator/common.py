"""Local packet records. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import re


def file_receipt(value):
    return (type(value) is dict and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and value["bytes"] >= 0
            and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None)


def file_inventory(value):
    return (type(value) is dict and bool(value)
            and all(type(name) is str and name and file_receipt(row) for name, row in value.items()))


def isolated_module(name, *arguments):
    if name not in ("supervisor", "transport"):
        raise ValueError("unsupported operator child module")
    bootstrap = ("import runpy,sys; root=sys.argv.pop(1); module=sys.argv.pop(1); "
                 "sys.path.insert(0,root); runpy.run_module(module,run_name='__main__')")
    return [sys.executable, "-I", "-B", "-u", "-c", bootstrap,
            str(Path(__file__).resolve().parents[2]), "sdcguard.native_operator."+name, *arguments]


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_digest(value):
    return digest(encoded(value))


def checkpoint(path, value):
    """Replace one owned checkpoint atomically; failures remain visible."""
    path = Path(path)
    raw = encoded(value)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("checkpoint exceeds its retained byte bound")
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def regular_bytes(path, maximum):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("a retained input cannot be a symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= maximum:
            raise ValueError("a retained input must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        if (len(raw) != before.st_size or len(raw) > maximum
                or (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
                != (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size)):
            raise ValueError("a retained input changed while being read")
        return raw
    finally:
        os.close(descriptor)
