"""Offline local integrity receipts for the scoped observer-overhead workflow.

Reading files and distribution metadata does not attest a running server. GPU
identity and the checks-on/off state remain explicit operator declarations.
"""
from __future__ import annotations

from importlib.metadata import distributions
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from . import __version__
from . import assessment
from .recording_plan import validate_deployment


SCHEMA = "sdcguard.performance-identity/1"
MEASUREMENT_SCHEMA = "sdcguard.performance-measurement/1"
IDENTITY_FIELDS = ("engine_files_sha256", "model_revision", "weights_sha256", "gpu_uuid",
                   "deployment_config_sha256")
_TOOL_FILES = ("perf.py", "overhead.py", "performance_identity.py", "assessment.py", "recording_plan.py",
               "lifetime/adapter.py", "lifetime/vllm_nixl.py", "lifetime/obligations.py",
               "lifetime/outcomes.py")
_BASIS = {
    "engine": "complete relevant local source files and local distribution metadata",
    "model": "complete existing local snapshot integrity pin; not proof of live loaded weights",
    "deployment": "local configuration declaration; not runtime topology attestation",
    "gpu_uuid": "operator-declared; no GPU or remote server was queried during preparation",
    "checks_enabled": "operator-declared at measurement; this tool does not enable or disable checks",
    "remote_attestation": False,
}


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path):
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=_object, parse_constant=invalid)


def _sha(path: Path) -> str:
    # Hash incrementally; model snapshots can contain multi-gigabyte files.
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _installed_version(root: Path, package: str) -> str:
    versions = [dist.version for dist in distributions(path=[str(root)])
                if str(dist.metadata.get("Name", "")).lower().replace("-", "_") == package]
    if len(versions) != 1:
        raise ValueError(f"one readable local {package} distribution is required in engine_root")
    return versions[0]


def _snapshot(snapshot: Path, model: str, revision: str, pin_path: Path) -> tuple[dict, str]:
    pin = read_json(pin_path)
    if (not isinstance(pin, dict) or pin.get("model") != model or pin.get("revision") != revision
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None or snapshot.name != revision
            or not isinstance(pin.get("files"), dict) or not pin["files"]):
        raise ValueError("matching commit-named model snapshot and complete existing integrity pin required")
    for name, digest in pin["files"].items():
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (relative is None or not name or relative.is_absolute() or ".." in relative.parts
                or str(relative) != name or name == "sdcguard-model-pin.json"
                or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError("model pin requires normalized relative file names and SHA256 values")
    metadata_files = {pin_path, snapshot / "sdcguard-model-pin.json"}
    actual = {str(path.relative_to(snapshot)): _sha(path) for path in sorted(snapshot.rglob("*"))
              if path.is_file() and path not in metadata_files}
    if actual != pin["files"]:
        raise ValueError("model snapshot files do not match the complete local integrity pin")
    return pin, _sha(pin_path)


def _tool_identity() -> dict:
    root = Path(__file__).resolve().parent
    return {"version": __version__, "files_sha256": {name: _sha(root / name) for name in _TOOL_FILES}}


def prepare_identity(config_path: Path, snapshot: Path, gpu_uuid: str, *, model_pin: Path | None = None) -> dict:
    """Inspect local inputs only. No engine import, requests, traffic or GPU allocation."""
    if (not isinstance(gpu_uuid, str)
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid) is None):
        raise ValueError("a physical GPU UUID in GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx form is required")
    gpu_uuid = "GPU-" + gpu_uuid[4:].lower()
    config_path, snapshot = Path(config_path).resolve(), Path(snapshot).resolve()
    config = read_json(config_path)
    # This validates the deployment declaration without activating recording or
    # interpreting the measurement process's unrelated inherited environment.
    applicability = validate_deployment(config, base=config_path.parent, environ={})
    if applicability["status"] != "supported":
        raise ValueError("; ".join(applicability["missing"]))
    engine = applicability["engine"]
    engine_root = Path(applicability["engine_root"])
    engine_version = _installed_version(engine_root, "vllm")
    nixl_version = _installed_version(engine_root, "nixl")
    if engine_version != config["engine_version"] or engine_version != "0.32.0":
        raise ValueError("local vLLM distribution version does not match the supported source/config pin")
    if nixl_version != config["nixl_version"] or nixl_version != "1.4.1":
        raise ValueError("local NIXL distribution version does not match the supported 1.4.1 pin")
    pin_path = Path(model_pin).resolve() if model_pin is not None else snapshot / "sdcguard-model-pin.json"
    pin, pin_digest = _snapshot(snapshot, config["model"], config["model_revision"], pin_path)
    value = {
        "schema": SCHEMA,
        "scope": "Observer-check overhead on one unchanged supported engine source variant",
        "model": config["model"], "engine": engine, "nixl_version": nixl_version,
        "identity": {"engine_files_sha256": canonical_digest(engine["file_sha256"]),
                     "model_revision": pin["revision"], "weights_sha256": canonical_digest(pin["files"]),
                     "gpu_uuid": gpu_uuid, "deployment_config_sha256": canonical_digest(config)},
        "local_inputs": {"deployment_config": str(config_path), "model_snapshot": str(snapshot),
                         "model_pin": str(pin_path)},
        "local_inspection": {"deployment_file_sha256": _sha(config_path),
                             "model_pin_file_sha256": pin_digest, "model_files_sha256": pin["files"],
                             "engine_distribution_version": engine_version},
        "tool": _tool_identity(), "basis": dict(_BASIS),
    }
    value["manifest_sha256"] = canonical_digest(value)
    return value


def write_identity(config_path: Path, snapshot: Path, gpu_uuid: str, output: Path, *,
                   model_pin: Path | None = None) -> dict:
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("identity output already exists; no file was overwritten")
    value = prepare_identity(config_path, snapshot, gpu_uuid, model_pin=model_pin)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def check_identity_document(value: dict) -> None:
    """Check a portable document's internal binding, without reading its old local paths."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("supported prepared performance identity required")
    if value.get("manifest_sha256") != canonical_digest({k: v for k, v in value.items()
                                                         if k != "manifest_sha256"}):
        raise ValueError("prepared performance identity content digest differs")
    if value.get("basis") != _BASIS:
        raise ValueError("explicit local-inspection and operator-declaration basis required")
    identity = value.get("identity")
    engine = value.get("engine")
    pin = assessment.pins()
    if (not isinstance(identity, dict) or set(identity) != set(IDENTITY_FIELDS)
            or not isinstance(engine, dict) or engine.get("version") != pin["version"]
            or engine.get("adapter_fits") is not True or engine.get("variant") not in pin["variants"]
            or engine.get("file_sha256") != pin["variants"][engine["variant"]]
            or engine.get("fix") != ("present" if engine["variant"] == "fixed" else "absent")
            or engine.get("inspected_source_only") is not True
            or identity["engine_files_sha256"] != canonical_digest(engine["file_sha256"])
            or value.get("model") != "Qwen/Qwen3-0.6B" or value.get("nixl_version") != "1.4.1"):
        raise ValueError("prepared identity does not contain the exact supported engine/model pins")
    inspection, inputs, tool = (value.get(name) for name in ("local_inspection", "local_inputs", "tool"))
    if (not isinstance(inspection, dict) or not isinstance(inspection.get("model_files_sha256"), dict)
            or not inspection["model_files_sha256"] or not isinstance(inputs, dict)
            or set(inputs) != {"deployment_config", "model_snapshot", "model_pin"}
            or any(not isinstance(path, str) or not Path(path).is_absolute() for path in inputs.values())
            or not isinstance(tool, dict) or not isinstance(tool.get("files_sha256"), dict)
            or set(tool["files_sha256"]) != set(_TOOL_FILES)
            or not isinstance(tool.get("version"), str) or not tool["version"]
            or inspection.get("engine_distribution_version") != engine["version"]):
        raise ValueError("complete local input and tool inspection receipt required")
    digests = [inspection.get("deployment_file_sha256"), inspection.get("model_pin_file_sha256"),
               *inspection["model_files_sha256"].values(), *tool["files_sha256"].values()]
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in digests):
        raise ValueError("valid local file SHA256 inspection values required")
    if identity["weights_sha256"] != canonical_digest(inspection["model_files_sha256"]):
        raise ValueError("snapshot fingerprint differs from the recorded file map")
    for name in IDENTITY_FIELDS:
        pattern = (r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}" if name == "gpu_uuid"
                   else r"[0-9a-f]{40}" if name == "model_revision" else r"[0-9a-f]{64}")
        if not isinstance(identity[name], str) or re.fullmatch(pattern, identity[name]) is None:
            raise ValueError(f"valid prepared {name} required")


def validate_identity(value: dict) -> dict:
    """Recompute local pins before/after measurement, outside the timed rounds."""
    check_identity_document(value)
    inputs = value["local_inputs"]
    actual = prepare_identity(Path(inputs["deployment_config"]), Path(inputs["model_snapshot"]),
                              value["identity"]["gpu_uuid"], model_pin=Path(inputs["model_pin"]))
    for name in ("identity", "engine", "nixl_version", "model", "local_inputs", "local_inspection", "tool"):
        if actual[name] != value[name]:
            raise ValueError(f"prepared {name} no longer matches local inputs")
    return actual


def measurement_digest(result: dict) -> str:
    return canonical_digest({k: v for k, v in result.items() if k != "measurement_receipt"})


def check_measurement_receipt(result: dict) -> None:
    """Validate captured receipt structure/bindings, not remote truth or old local paths."""
    receipt = result.get("measurement_receipt")
    if (not isinstance(receipt, dict) or receipt.get("schema") != MEASUREMENT_SCHEMA
            or receipt.get("status") != "complete" or receipt.get("missing") != []
            or receipt.get("interrupted") is not False
            or receipt.get("local_validation") != {"before": True, "after": True}
            or receipt.get("checks_state_basis") != _BASIS["checks_enabled"]):
        raise ValueError("complete pre/post local-validation measurement receipt required")
    prepared = receipt.get("prepared_identity")
    check_identity_document(prepared)
    if (result.get("identity") != prepared["identity"]
            or type(result.get("checks_enabled")) is not bool
            or receipt.get("checks_enabled") is not result["checks_enabled"]
            or receipt.get("measurement_sha256") != measurement_digest(result)):
        raise ValueError("measurement values or checks state differ from their receipt")
    # A digest of aggregates cannot prove stream completion. Bound receipts
    # require the per-request successful terminal and strict usage observations.
    rows = result.get("rounds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("raw rounds with request terminal evidence required")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("request_results"), list):
            raise ValueError("per-request successful SSE terminal evidence required")
        requests = row["request_results"]
        if (row.get("interrupted") is not False or type(row.get("requests")) is not int
                or row["requests"] <= 0 or len(requests) != row["requests"]
                or row.get("started") != row["requests"] or row.get("unstarted") != 0
                or row.get("completed") != row["requests"] or row.get("errors") != 0):
            raise ValueError("all recorded requests must complete without interruption")
        for index, request in enumerate(requests):
            if (not isinstance(request, dict) or request.get("request_index") != index
                    or request.get("complete") is not True or request.get("terminal_observed") is not True
                    or request.get("usage_valid") is not True or request.get("http_status") != 200
                    or request.get("error") is not None
                    or type(request.get("tokens")) is not int or request["tokens"] < 0):
                raise ValueError("successful HTTP/SSE terminal and nonnegative integer token usage required")
        tokens = sum(request["tokens"] for request in requests)
        if (type(row.get("output_tokens")) is not int or row["output_tokens"] != tokens
                or row.get("tokens_counted_by_server") is not True):
            raise ValueError("raw token counts must agree with request observations")
    metadata = receipt.get("server_metadata")
    allowed_models = {prepared["model"], prepared["local_inputs"]["model_snapshot"]}
    if (not isinstance(metadata, dict) or metadata.get("before") != metadata.get("after")
            or not isinstance(metadata.get("before"), dict)
            or metadata["before"].get("model") not in allowed_models
            or metadata["before"].get("version") != prepared["engine"]["version"]
            or result.get("model") != metadata["before"]["model"]
            or result.get("server_version") != metadata["before"]["version"]):
        raise ValueError("stable compatible server model/version self-reports required")
