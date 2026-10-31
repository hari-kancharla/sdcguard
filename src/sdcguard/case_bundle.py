"""Portable, pinned files for the saved public case; verification never executes them."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import os
import stat
from pathlib import Path
import re

SCHEMA = "sdcguard.case-bundle/1"
SCOPE = "Pinned historical #61229 artifacts and owner-run wrapper; no GPU execution or minimality proof"
RUNNER = '''"""Owner-run replay; no search arguments are accepted."""
from pathlib import Path
import sys
from sdcguard.regression import main
raise SystemExit(main(['run','--case-dir',str(Path(__file__).resolve().parent),*sys.argv[1:]]))
'''
README = '''# Saved #61229 regression packet

Open RESULT.html for the saved result. BUNDLE_MANIFEST.json names every packet
file. Verify this packet, including after moving it, with:

    sdcguard regression verify /path/to/packet

Verification checks local file integrity against the installed package's pinned
case. It neither executes the runner nor authenticates a remote deployment.

The GPU replay is owner-run. Install this same reviewed package in the engine
environment, then run:

    python -I -B /path/to/packet/run.py --variant fixed --engine-root /path/to/site-packages --model-snapshot /path/to/COMMIT --out /new/output

Use --variant unpatched for the preserved stock failure criterion. The packet supplies MODEL_PIN.json with the recovered immutable commit and all
ten file SHA256 values; the wrapper checks the actual snapshot against it. Use
--model-pin /path/to/pin.json only for an external copy of that same exact pin.
The original cache can remain read-only. The installed vLLM/NIXL and console interpreter must
match the pinned engine identity. No source checkout is needed.

Judge a new run with:

    sdcguard regression judge /new/output/RESULT.json --variant fixed --manifest /new/output/RUN_MANIFEST.json

The wrapper only replays the saved condition and workload. It adds no search
arguments, retry case or reused request identity. See CRITERIA.md and SUPPORT.md.

Minimality is NOT established. Original artifacts recorded revision=main. A
shared-cache snapshot and its ten files were subsequently verified against the
immutable published revision; see MODEL_PROVENANCE.json. This corroborates the
historical association without replacing the missing original run manifest. The saved output does not validate revised hooks, prove
actual attention consumption, measure recorder overhead or certify a server.
'''
CRITERIA = '''# Explicit regression criteria

Engine: vLLM 0.32.0 at the packaged complete relevant file hashes. Transport:
NIXL 1.4.1. Model: Qwen/Qwen3-0.6B; fresh runs require a complete locally pinned
commit-named snapshot matching MODEL_PIN.json. Original artifacts recorded main;
MODEL_PROVENANCE.json gives the corroborative shared-cache recovery evidence.

Both arms must retain both clean controls, three agreed nonempty clean references
and all five replays. Every replay must start clean, reach the exact saved hold
condition with the attributed producer, and finish without a hang or unexplained
exception. No alternate failure ordering, retries or reused IDs are permitted.

Stock criterion: all five replays reproduce the saved harm_garbled failure class.
Fixed recompute criterion: all five return the exact clean reference after a
request-attributed, normalized post_read lease_mismatch rejection. Rejection,
client recovery and client refusal are different observations. A configured
policy alone does not prove completion. A differing nonempty fixed answer after
complete evidence fails. Missing, contradictory or unbound facts are inconclusive.

Fresh judgments also require the expected harness exit, successful revised
evaluator validation and a matching complete exported-run integrity manifest.
An exact preserved historical JSON can satisfy its historical experiment
criterion only; it does not establish fresh reproduction or model identity.

Minimality, revised-hook GPU behavior and matched observer overhead remain open.
SOURCE FIX PRESENT, CHECK OBSERVED EXECUTING and CLIENT OUTCOME are separate facts.
'''


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _case():
    from .regression import historical_case
    return historical_case()


def _resources(case, checkout=None):
    base = files("sdcguard").joinpath("case_data")
    pin = json.loads(base.joinpath("61229_resources.json").read_text())
    required = {"existing_case.py", "SUPPORT.md", "OVERLAP.md", "RESULT.html", "MODEL_PIN.json", "MODEL_PROVENANCE.json", "PRODUCT_REVIEW.md", "LIFETIME_RULES.md", "OWNER_ACTIONS.md", "deployment.example.json", *case["artifacts"]}
    if not isinstance(pin, dict) or set(pin) != required:
        raise ValueError("complete packaged case resource pin required")
    result = {}
    for name, expected in pin.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None):
            raise ValueError("invalid packaged case resource pin")
        data = base.joinpath("61229", name).read_bytes()
        if _sha(data) != expected:
            raise ValueError("packaged case resource differs from its pin: " + name)
        result[name] = data
    if checkout is not None:
        result["existing_case.py"] = (Path(checkout) / "experiments/lifetime_pd_check.py").read_bytes()
    if _sha(result["existing_case.py"]) != case["reviewed_harness_sha256"]:
        raise ValueError("harness does not match the reviewed source pin")
    if _sha(result["MODEL_PIN.json"]) != case["replay_model_pin_sha256"]:
        raise ValueError("replay model pin differs from its reviewed digest")
    for name, identity in case["artifacts"].items():
        if _sha(result[name]) != identity["sha256"]:
            raise ValueError("saved artifact differs from the historical pin: " + name)
    return result


def payloads(checkout=None):
    case = _case()
    return {**_resources(case, checkout), "case.json": _json(case),
            "candidate.json": _json(case["candidate"]), "run.py": RUNNER.encode(),
            "README.md": README.encode(), "CRITERIA.md": CRITERIA.encode()}


def export_bundle(destination: Path, checkout=None):
    """Validate every input before reserving a new output; never overwrite files."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError("export destination exists")
    data = payloads(checkout)
    manifest = {"schema": SCHEMA, "scope": SCOPE,
                "file_sha256": {name: _sha(value) for name, value in sorted(data.items())}}
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in data.items():
        with (destination / name).open("xb") as stream:
            stream.write(value)
    # A partial export cannot verify: the complete manifest is written last.
    with (destination / "BUNDLE_MANIFEST.json").open("xb") as stream:
        stream.write(_json(manifest))
    return manifest


def _signature(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _read_descriptor(fd, limit):
    data = bytearray()
    while True:
        block = os.read(fd, min(65536, limit + 1 - len(data)))
        if not block:
            return bytes(data)
        data.extend(block)
        if len(data) > limit:
            raise ValueError("packet file grew beyond its pinned size")


def verify_bundle(directory: Path):
    """Verify a coherent inert snapshot against package pins; never execute it."""
    result = {"status": "inconclusive", "scope": SCOPE, "missing_or_changed": [],
              "gpu_executed": False, "minimality": "unverified",
              "historical_model_revision": "unresolved",
              "trust": "local integrity against installed package pins; not a signature or remote attestation",
              "verification_scope": "captured file identity/content checked during verification; later changes require re-verification"}
    root_fd, descriptors = None, {}
    try:
        directory = Path(directory).resolve(strict=True)
        expected = payloads()
        names = {*expected, "BUNDLE_MANIFEST.json"}
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise ValueError("safe directory-descriptor verification is unavailable on this platform")
        root_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        root_before = os.fstat(root_fd)
        if set(os.listdir(root_fd)) != names:
            raise ValueError("complete packet file set required; unexpected or missing files")
        snapshots = {}
        for name in sorted(names):
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
            descriptors[name] = fd
            before = os.fstat(fd)
            limit = 256 * 1024 if name == "BUNDLE_MANIFEST.json" else len(expected[name])
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("regular bounded packet file required: " + name)
            snapshots[name] = _signature(before)
            data = _read_descriptor(fd, limit)
            if _signature(os.fstat(fd)) != snapshots[name]:
                raise ValueError("packet file changed during reading: " + name)
            if name == "BUNDLE_MANIFEST.json":
                manifest = json.loads(data)
            elif data != expected[name]:
                raise ValueError("changed packet file: " + name)
        pinned = {name: _sha(data) for name, data in sorted(expected.items())}
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "scope", "file_sha256"}
                or manifest["schema"] != SCHEMA or manifest["scope"] != SCOPE
                or manifest["file_sha256"] != pinned):
            raise ValueError("packet manifest does not match the installed package's pinned case")
        # Final checks cover earlier files changed while a later file was read,
        # pathname replacement, and directory membership mutation.
        for name, fd in descriptors.items():
            if (_signature(os.fstat(fd)) != snapshots[name]
                    or _signature(os.stat(name, dir_fd=root_fd, follow_symlinks=False)) != snapshots[name]):
                raise ValueError("packet changed during verification: " + name)
        if (set(os.listdir(root_fd)) != names or _signature(os.fstat(root_fd)) != _signature(root_before)
                or _signature(os.stat(directory, follow_symlinks=False)) != _signature(root_before)):
            raise ValueError("packet directory changed during verification")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["missing_or_changed"].append(str(exc))
        return result
    finally:
        for fd in descriptors.values():
            os.close(fd)
        if root_fd is not None:
            os.close(root_fd)
    result["status"] = "verified"
    result["replay_model_revision"] = _case()["replay_model_revision"]
    result["replay_snapshot_integrity"] = "ten cached files matched immutable upstream identities"
    result["verified_files"] = len(expected)
    return result
