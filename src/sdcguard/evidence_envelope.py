"""Bounded versioned envelope storage. Owner: Hari Krishna Kancharla."""
from pathlib import Path
from tempfile import TemporaryDirectory

from . import evidence_io as io
from . import evidence_json_budget as json_budget
from . import evidence_native_io as bounded


def validate_snapshot(files, directories):
    if (type(files) is not dict or len(files) > io.MAX_FILES
            or type(directories) is not list or len(directories) >= io.MAX_FILES
            or any(type(name) is not str for name in directories)
            or directories != sorted(set(directories))):
        raise io.EvidenceError("unsupported envelope inventory")
    required = set()
    total = 0
    for name in directories:
        parts = io.relative(name)
        if len(parts) > bounded.MAX_DIRECTORY_DEPTH or name in files:
            raise io.EvidenceError("unsupported envelope directory shape")
        required.update("/".join(parts[:i]) for i in range(1, len(parts)))
    for name, raw in files.items():
        parts = io.relative(name)
        if (len(parts) > bounded.MAX_DIRECTORY_DEPTH + 1
                or type(raw) is not bytes or len(raw) > io.MAX_FILE_BYTES):
            raise io.EvidenceError("unsupported envelope member")
        required.update("/".join(parts[:i]) for i in range(1, len(parts)))
        total += len(raw)
        if total > io.MAX_TOTAL_BYTES:
            raise io.EvidenceError("envelope exceeds its aggregate byte bound")
    if not required <= set(directories):
        raise io.EvidenceError("envelope parents are missing")
    json_budget.count_members(files)


def identity(files, directories):
    validate_snapshot(files, directories)
    return io.digest(io.json_bytes({"files": io.file_receipts(files), "directories": directories}))


def reject_overlap(output, sources):
    target = Path(output).absolute()
    for source in sources:
        root = Path(source).absolute()
        if target == root or root in target.parents or target in root.parents:
            raise io.EvidenceError("output must be outside every external input packet tree")


def materialize(root, files, directories):
    validate_snapshot(files, directories)
    io.write_snapshot(root, files)
    for name in directories:
        io.make_directory(root, name)


def evaluate_private(files, directories, function):
    validate_snapshot(files, directories)
    with TemporaryDirectory(prefix="sdcguard-envelope-") as temporary:
        root = Path(temporary).resolve()
        materialize(root, files, directories)
        return function(root)


def directories_for(files, extra=()):
    result = set(extra)
    for name in list(files) + list(extra):
        parts = io.relative(name)
        result.update("/".join(parts[:i]) for i in range(1, len(parts)))
    return sorted(result)


def complete(output, saved, directories, *, schema, profile):
    if "manifest.json" in saved:
        raise io.EvidenceError("completed envelope inputs cannot include its outer manifest")
    directories = directories_for(saved, directories)
    manifest = {"schema": schema, "profile": profile, "complete": True,
                "integrity": "local SHA256 consistency only; no authenticity claim",
                "files": io.file_receipts(saved), "directories": directories}
    files = {**saved, "manifest.json": io.json_bytes(manifest)}
    validate_snapshot(files, directories)
    root = io.create_directory(output)
    materialize(root, saved, directories)
    bounded.recheck_snapshot(root, saved, directories)
    io.write_file(root, "manifest.json", files["manifest.json"])
    bounded.recheck_snapshot(root, files, directories)
    return str(root)


def check_manifest(files, directories, *, schema):
    validate_snapshot(files, directories)
    manifest = io.parse_json(files.get("manifest.json", b""))
    if (type(manifest) is not dict
            or set(manifest) != {"schema", "profile", "complete", "integrity", "files", "directories"}
            or manifest["schema"] != schema or manifest["complete"] is not True
            or manifest["integrity"] != "local SHA256 consistency only; no authenticity claim"
            or type(manifest["files"]) is not dict):
        raise io.EvidenceError("unsupported or incomplete envelope manifest")
    for name, value in manifest["files"].items():
        io.relative(name)
        if (type(value) is not dict or set(value) != {"bytes", "sha256"}
                or type(value["bytes"]) is not int or value["bytes"] < 0
                or type(value["sha256"]) is not str or len(value["sha256"]) != 64):
            raise io.EvidenceError("malformed envelope file receipt")
    retained = {name: raw for name, raw in files.items() if name != "manifest.json"}
    if manifest["files"] != io.file_receipts(retained) or manifest["directories"] != directories:
        raise io.EvidenceError("envelope receipts disagree with retained members")
    return manifest["profile"]


def subset(files, directories, prefix):
    members = {name[len(prefix):]: raw for name, raw in files.items() if name.startswith(prefix)}
    child_directories = [name[len(prefix):] for name in directories if name.startswith(prefix)]
    validate_snapshot(members, child_directories)
    return members, child_directories
