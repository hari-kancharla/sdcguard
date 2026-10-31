"""Check complete installed extraction bytes. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import hashlib
import importlib.util
import marshal
from pathlib import Path, PurePosixPath
import types


def _shape(code):
    return (code.co_code, tuple(_shape(x) if isinstance(x, types.CodeType) else x for x in code.co_consts),
        code.co_names, code.co_varnames, code.co_argcount, code.co_posonlyargcount,
        code.co_kwonlyargcount, code.co_freevars, code.co_cellvars, code.co_flags,
        getattr(code, "co_exceptiontable", b""))


def receipt(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("an installed extraction member is missing or nonregular: " + str(path))
    before = path.stat()
    if not 0 <= before.st_size <= 1024 ** 3:
        raise ValueError("an installed extraction member exceeds its byte bound")
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            hasher.update(chunk)
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != before.st_size):
        raise ValueError("an installed extraction member changed during read")
    return dict(bytes=size, sha256=hasher.hexdigest())


def verify(root, expected_members, tracked, *, package="vllm"):
    root = Path(root)
    if type(expected_members) is not dict or not expected_members:
        raise ValueError("complete expected extraction inventory is required")
    checked = {}
    for name, expected in expected_members.items():
        path = PurePosixPath(name)
        if path.is_absolute() or path.as_posix() != name or any(x in (".", "..") for x in path.parts):
            raise ValueError("an extraction member path is not canonical")
        actual = receipt(root / name)
        if actual != expected:
            raise ValueError("installed extraction payload differs: " + name)
        checked[name] = actual
    generated, bytecode = {}, {}
    for path in (root / package).rglob("*"):
        if path.is_symlink():
            raise ValueError("an engine execution input is a symlink")
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        if name in expected_members or name in tracked:
            continue
        if name == "vllm/_version.py":
            generated[name] = receipt(path)
            continue
        if path.suffix == ".pyc" and "__pycache__" in path.parts:
            raw = path.read_bytes()
            if len(raw) > 8 * 1024 * 1024 or raw[:4] != importlib.util.MAGIC_NUMBER:
                raise ValueError("engine bytecode format is unsupported")
            source = Path(importlib.util.source_from_cache(str(path)))
            source_name = source.relative_to(root).as_posix()
            if source_name not in expected_members and source_name not in tracked and source_name != "vllm/_version.py":
                raise ValueError("engine bytecode has no bound source")
            code = marshal.loads(raw[16:])
            if not isinstance(code, types.CodeType) or _shape(code) != _shape(compile(
                    source.read_bytes(), str(source), "exec", dont_inherit=True)):
                raise ValueError("engine bytecode differs from its bound source")
            bytecode[name] = receipt(path)
            continue
        raise ValueError("unexpected untracked engine execution input: " + name)
    return dict(extracted_members=checked, generated_metadata=generated, bytecode=bytecode,
        scope="complete installed extraction bytes and separately bound generated metadata; no selected archive claim")
