"""Read pinned installed engine identity. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import ENGINE_ROOT, REVISION
from .common import checkpoint, digest, regular_bytes

CRITICAL_PATHS = {"csrc", "cmake", "CMakeLists.txt", "pyproject.toml", "vllm/_custom_ops.py", "rust", "setup.py"}
def git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True, timeout=60,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def validate_pinned_source(root, lock):
    target, base = lock["target"]["revision"], lock["base"]["revision"]
    if target != REVISION or git(root, "rev-parse", "HEAD") != target:
        raise ValueError("pinned source HEAD differs from the reviewed target")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("pinned tracked source is dirty before installation")
    if git(root, "rev-parse", base + "^{commit}") != base:
        raise ValueError("pinned source base object differs")
    ancestry = git(root, "merge-base", "HEAD", base)
    if ancestry != base:
        raise ValueError("the pinned base is not an ancestor of the target")
    identities = lock["base"]["critical_identities"]
    expected = identities[target]
    if set(expected) != CRITICAL_PATHS or identities.get(base) != expected:
        raise ValueError("all seven pinned critical source identities must agree")
    observed = {}
    for name in sorted(CRITICAL_PATHS):
        head_id = git(root, "rev-parse", "HEAD:" + name)
        base_id = git(root, "rev-parse", base + ":" + name)
        if head_id != expected[name] or base_id != expected[name]:
            raise ValueError("pinned compiled input identity differs: " + name)
        observed[name] = dict(expected=expected[name], head=head_id, base=base_id)
    return dict(head=target, tracked_status="", pinned_base=base, ancestry=ancestry, critical_identities=observed,
        policy="exact pinned source compatibility replaces the upstream automatic wheel-selection guard")


def installed_identity(bundle, *, engine_root=Path(ENGINE_ROOT)):
    lock = json.loads(regular_bytes(Path(bundle) / "artifacts/ENGINE_BUILD_LOCK.json", 2 * 1024 * 1024))
    if git(engine_root, "rev-parse", "HEAD") != lock["target"]["revision"]:
        raise ValueError("installed engine source revision differs")
    if git(engine_root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("installed tracked engine source is dirty")
    source_files = {}
    for name, expected in lock["sources"].items():
        path = engine_root / name
        raw = regular_bytes(path, 4 * 1024 * 1024)
        if digest(raw) != expected["sha256"]:
            raise ValueError("installed engine source differs: " + name)
        source_files[name] = dict(bytes=len(raw), sha256=digest(raw))
    inventory = json.loads(regular_bytes(Path(bundle) / "artifacts/PRECOMPILED_PAYLOAD.json", 2 * 1024 * 1024))
    if (inventory.get("wheel_sha256") != lock["base"]["wheel"]["sha256"]
            or inventory.get("member_count") != len(inventory.get("members", {}))
            or not set(lock["base"]["binary_members"]) <= set(inventory["members"])
            or any(inventory["members"][name] != value for name, value in lock["base"]["binary_members"].items())):
        raise ValueError("complete official extraction inventory differs from the audited base wheel")
    tracked = {name for name in git(engine_root, "ls-files", "-z", "vllm").split("\x00") if name}
    from .payload import verify as verify_payload
    payload = verify_payload(engine_root, inventory["members"], tracked)
    binaries = {name: payload["extracted_members"][name] for name in lock["base"]["binary_members"]}
    native_names = {p.relative_to(engine_root).as_posix() for p in (engine_root / "vllm").rglob("*.so")}
    native_names.add("vllm/vllm-rs")
    if native_names != set(binaries):
        raise ValueError("installed compiled member inventory differs from the audited wheel")
    versions = {name: importlib.metadata.version(name) for name in
                ("vllm", "torch", "triton", "nixl", "nixl-cu12", "nixl-cu13", "uv")}
    if (versions["torch"].split("+")[0] != lock["environment"]["torch"]
            or any(versions[name] != lock["environment"]["nixl"] for name in ("nixl", "nixl-cu12", "nixl-cu13"))):
        raise ValueError("installed engine dependencies differ from the reviewed profile")
    distribution = importlib.metadata.distribution("vllm")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("url") != engine_root.as_uri() or direct.get("dir_info", {}).get("editable") is not True:
        raise ValueError("the engine is not the source-supported editable installation")
    from .python_runtime import require_interpreter
    python_identity = require_interpreter()
    return dict(schema="sdcguard.owner-checksum-installed.v1", source_revision=REVISION,
        source_files=source_files, compiled_members=binaries, versions=versions,
        precompiled_payload=payload, direct_url=direct, runtime_qualified=False,
        python_identity=python_identity,
        compatibility_guard="exact pinned HEAD/base plus seven critical source identities; upstream automatic guard replaced")

