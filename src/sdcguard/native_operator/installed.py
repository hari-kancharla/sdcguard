"""Pinned installed-runtime receipt. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .common import digest, encoded, regular_bytes, file_inventory
from .config import BUNDLE, ENGINE_ROOT
from .workload import REVISION, MODEL, MODEL_REVISION
from .. import checksum_profiles as profiles

SCHEMA = "sdcguard.installed-checksum-preflight.v1"
WITNESS = "sdcguard.installed-checksum-witness.v1"
VERSION = "0.33.0rc1.dev205+g29d2f4e03.precompiled"


def profile(name, revision=REVISION):
    return json.loads(regular_bytes(profiles.directory(revision) / "artifacts" / name, 2 * 1024 * 1024))


def product_identity():
    """Validate installed wheel bytes and return their reproducible inventory."""
    distribution = importlib.metadata.distribution("sdcguard")
    selected = [ep for ep in importlib.metadata.entry_points(group="vllm.general_plugins")
                if ep.name == "sdcguard_coverage"]
    entrypoint = "sdcguard.native_operator.plugin:register"
    if (len(selected) != 1 or selected[0].value != entrypoint or selected[0].dist is None
            or selected[0].dist.metadata["Name"] != "sdcguard"
            or not any(ep.group == "vllm.general_plugins" and ep.name == "sdcguard_coverage"
                       and ep.value == entrypoint for ep in distribution.entry_points)):
        raise ValueError("the installed native coverage entry point is missing or ambiguous")
    root = Path(distribution.locate_file("sdcguard")).resolve(strict=True)
    if Path(__file__).resolve().parent.parent != root:
        raise ValueError("the running product is not the installed distribution")
    files = {}
    for entry in distribution.files or ():
        name = str(entry)
        if not name.startswith("sdcguard/") or name.endswith(".pyc"):
            continue
        expected = entry.hash
        if expected is None or expected.mode != "sha256":
            raise ValueError("installed product RECORD lacks a SHA256: " + name)
        raw = regular_bytes(Path(distribution.locate_file(entry)), 8 * 1024 * 1024)
        actual = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        if actual != expected.value or len(raw) != entry.size:
            raise ValueError("installed product differs from its RECORD: " + name)
        files[name] = dict(bytes=len(raw), sha256=digest(raw))
    if not files or len(files) > 1024:
        raise ValueError("installed product inventory is missing or oversized")
    actual_files = {"sdcguard/" + p.relative_to(root).as_posix() for p in root.rglob("*")
                    if p.is_file() and p.suffix != ".pyc"}
    if actual_files != set(files):
        raise ValueError("unlisted product execution or data files are present")
    from .payload import verify
    verify(root.parent, files, set(), package="sdcguard")
    return dict(version=distribution.version, files=files, sha256=digest(encoded(files)),
                entrypoint=entrypoint, distribution="sdcguard")


def witness(value):
    return dict(schema=WITNESS, packet_sha256=value["packet_sha256"], run_id=value["run_id"],
        plan_sha256=value["plan_sha256"], installed_identity_sha256=digest(encoded(value["installed_engine"])),
        product_identity_sha256=digest(encoded(value["product"])),
        preflight_sha256=digest(encoded({k:v for k,v in value.items() if k not in ("build_witness", "build_witness_sha256")})),
        runtime_qualified=False, scope="observed installed bytes and CPU prerequisites; no execution attestation")


def observer_files():
    root = Path(__file__).resolve().parent.parent
    paths = list((root / "native_operator").rglob("*.py"))
    paths += list((root / "native_operator/profile_data").rglob("*.json"))
    paths += [root/name for name in ("kv_checksum_observer.py", "checksum_client.py", "prefix_cache_observer.py",
                                      "prefix_cache_coverage.py", "checksum_observer_report.py", "checksum_profiles.py",
                                      "case_data/checksum-builds.json", "case_data/checksum-legacy-recorder.json")]
    return {"sdcguard/"+p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,
            sha256=digest(regular_bytes(p, 8*1024*1024))) for p in paths}


def recorder_matches(product, revision):
    inventories = [observer_files()]
    if revision == profiles.BASELINE:
        root = Path(__file__).resolve().parent.parent
        inventories.append(json.loads(regular_bytes(root / "case_data/checksum-legacy-recorder.json", 256*1024)))
    return any(all(encoded(product["files"].get(name)) == encoded(row) for name,row in inventory.items())
               for inventory in inventories)


def validate(value, plan, plan_raw):
    """Offline validation uses pinned values, never just a reported PASS."""
    from .. import evidence_io as io
    from . import python_runtime, cleanup_preflight, transport
    try:
        revision = plan["source_revision"]
        selected = profiles.get(revision)
        if (type(value) is not dict or value.get("schema") != SCHEMA or value.get("status") != "PASS"
                or value.get("failure") is not None or value.get("cpu_only") is not True
                or value.get("gpu_submitted") is not False or value.get("runtime_qualified") is not False
                or value.get("run_id") != plan["run_id"] or value.get("packet_sha256") != plan["packet_sha256"]
                or value.get("plan_sha256") != digest(plan_raw) or value.get("source_revision") != revision):
            return False
        product, engine = value["product"], value["installed_engine"]
        lock, payload, model = (profile(n, revision) for n in ("ENGINE_BUILD_LOCK.json", "PRECOMPILED_PAYLOAD.json", "MODEL_PIN.json"))
        if (product.get("distribution") != "sdcguard"
                or product.get("entrypoint") != "sdcguard.native_operator.plugin:register"
                or not file_inventory(product.get("files")) or not 1 <= len(product["files"]) <= 1024
                or not recorder_matches(product, revision)
                or product.get("sha256") != digest(encoded(product["files"]))
                or product["sha256"] != plan["packet_sha256"]
                or engine.get("schema") != "sdcguard.owner-checksum-installed.v1"
                or engine.get("source_revision") != revision or engine.get("runtime_qualified") is not False
                or engine["versions"]["vllm"] != selected["version"]
                or engine["versions"]["torch"].split("+")[0] != lock["environment"]["torch"]
                or any(engine["versions"][n] != lock["environment"]["nixl"] for n in ("nixl", "nixl-cu12", "nixl-cu13"))
                or not all(file_inventory(rows) for rows in (engine["compiled_members"], engine["source_files"],
                           engine["precompiled_payload"]["extracted_members"]))
                or encoded(engine["compiled_members"]) != encoded(lock["base"]["binary_members"])
                or encoded(engine["precompiled_payload"]["extracted_members"]) != encoded(payload["members"])
                or set(engine["source_files"]) != set(lock["sources"])
                or any(engine["source_files"][n]["sha256"] != v["sha256"] for n,v in lock["sources"].items())
                or encoded(engine["direct_url"]) != encoded({"dir_info": {"editable": True}, "url": Path(ENGINE_ROOT).as_uri()})):
            return False
        source = value["pinned_source_guard"]
        if (source.get("head") != revision or source.get("tracked_status") != ""
                or source.get("pinned_base") != lock["base"]["revision"] or source.get("ancestry") != lock["base"]["revision"]
                or set(source["critical_identities"]) != set(lock["base"]["critical_identities"][revision])
                or any(row != dict(expected=expected, head=expected, base=expected) for name,expected in
                       lock["base"]["critical_identities"][revision].items() for row in [source["critical_identities"][name]])):
            return False
        identity = value["model_cache"]
        if (identity.get("model") != MODEL or identity.get("revision") != MODEL_REVISION
                or type(identity.get("snapshot")) is not str or not Path(identity["snapshot"]).is_absolute()
                or Path(identity["snapshot"]).name != MODEL_REVISION
                or not file_inventory(identity.get("files"))
                or set(identity["files"]) != set(model["files"])
                or any(identity["files"][name]["sha256"] != expected for name,expected in model["files"].items())):
            return False
        rid, pid = plan["run_id"], plan["packet_sha256"]
        if (not python_runtime.validate(value["python_runtime"], rid, pid)
                or encoded(value["python_runtime"]["identity"]) != encoded(engine["python_identity"])
                or not cleanup_preflight.validate(value["cleanup_capability"], rid, pid)
                or not transport.validate(value["transport_readiness"], rid, pid)):
            return False
        expected = witness(value)
        return (io.json_bytes(value["build_witness"]) == io.json_bytes(expected)
                and value.get("build_witness_sha256") == digest(encoded(expected)))
    except (ValueError, KeyError, TypeError, AttributeError, OSError):
        return False
