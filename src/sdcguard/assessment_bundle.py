"""Portable, locally verifiable evidence for the scoped recording assessment.

Creation copies only explicitly supplied config/recordings and the nine pinned
engine paths. Verification is offline: hashes are local integrity, not a
signature, remote attestation, or a guarantee about later serving traffic.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys

from . import assessment

MANIFEST = "manifest.json"
FORMAT = "sdcguard-scoped-assessment"
INTEGRITY = "local SHA256 consistency only; no signature or authenticity claim"
CAPTURE_BASIS = "captured local bytes and two-pass source stability checks"
CONFIG_KEYS = {"schema", "engine", "engine_version", "nixl_version", "engine_root", "model",
               "model_revision", "topology", "events", "ready", "client_inventory"}
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024


class CaptureError(ValueError):
    pass


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(x in ("", ".", "..") for x in name.split("/"))
            or any(":" in x for x in name.split("/"))):
        raise CaptureError("unsafe relative packet path")
    path = PurePosixPath(name)
    if path.is_absolute():
        raise CaptureError("unsafe relative packet path")
    return path


def _signature(s):
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _open_relative(root, name, flags, mode=0o600):
    """Walk directory descriptors so a replaced link cannot escape the root."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise CaptureError("secure packet file access is unsupported on this platform")
    parts = _relative(name).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(parts[-1], flags | os.O_NOFOLLOW, mode, dir_fd=directory)
    finally:
        os.close(directory)


def _mkdir_packet_directory(root, name):
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise CaptureError("secure packet file access is unsupported on this platform")
    parts = _relative(name).parts
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
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


def _read_regular(path: Path, *, root: Path | None = None):
    """Reject links and detect replacement or writing during this read."""
    if root is not None:
        try:
            parts = path.relative_to(root).parts
        except ValueError as exc:
            raise CaptureError("source path escaped its declared root") from exc
        cursor = root
        for part in parts:
            cursor /= part
            if cursor.is_symlink():
                raise CaptureError("source symlink is not captured")
    if path.is_symlink():
        raise CaptureError("source symlink is not captured")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise CaptureError("source is not a regular file")
    base = root if root is not None else path.parent
    with os.fdopen(_open_relative(base, path.relative_to(base).as_posix(), os.O_RDONLY), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise CaptureError("source is not a supported bounded regular file")
        data = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(stream.fileno())
    if (len(data) > MAX_FILE_BYTES or _signature(before) != _signature(after)
            or _signature(path.stat()) != _signature(after)):
        raise CaptureError("source changed while being captured")
    return data, _signature(after)


def _event_names(path: Path):
    if path.is_symlink():
        raise CaptureError("recording symlink is not captured")
    return sorted(x.name for x in path.iterdir() if x.name.endswith(".jsonl"))


def _write_packet(root, name, data, role, inventory):
    parts = _relative(name).parts
    if len(parts) > 1:
        _mkdir_packet_directory(root, "/".join(parts[:-1]))
    with os.fdopen(_open_relative(root, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    inventory[name] = {"sha256": _digest(data), "bytes": len(data), "role": role}


def _safe_config(config):
    """Do not copy unrelated options or credential-bearing metrics URLs."""
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise CaptureError("unsupported configuration fields; original config was not copied")
    for key in ("engine", "engine_version", "nixl_version", "engine_root", "model", "events", "client_inventory"):
        if key in config and not isinstance(config[key], str):
            raise CaptureError("unsupported configuration value; original config was not copied")
    if "schema" in config and type(config["schema"]) is not int:
        raise CaptureError("unsupported configuration schema; original config was not copied")
    topology = config.get("topology")
    if topology is not None and not isinstance(topology, dict):
        raise CaptureError("unsupported topology value; original config was not copied")
    if isinstance(topology, dict) and set(topology) - set(assessment.TOPOLOGY):
        raise CaptureError("unsupported topology fields; original config was not copied")
    if isinstance(topology, dict) and any(type(value) is not type(assessment.TOPOLOGY[key])
                                         for key, value in topology.items()):
        raise CaptureError("unsupported topology value; original config was not copied")
    revision = config.get("model_revision")
    if revision not in (None, "main") and (not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise CaptureError("unresolved or unsupported model revision; original config was not copied")
    ready = config.get("ready")
    if ready is not None and not isinstance(ready, dict):
        raise CaptureError("unsupported readiness value; original config was not copied")
    if isinstance(ready, dict) and set(ready) - {"metrics_urls"}:
        raise CaptureError("unsupported readiness fields; original config was not copied")
    if isinstance(ready, dict) and "metrics_urls" in ready and not isinstance(ready["metrics_urls"], list):
        raise CaptureError("unsupported metrics URL list; original config was not copied")
    if isinstance(ready, dict) and isinstance(ready.get("metrics_urls"), list):
        from urllib.parse import urlsplit
        for value in ready["metrics_urls"]:
            if not isinstance(value, str):
                raise CaptureError("unsupported metrics URL; original config was not copied")
            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
                raise CaptureError("credential-bearing or parameterized metrics URL was not copied")


def _portable_config(config):
    value = deepcopy(config)
    if isinstance(config.get("engine_root"), str):
        value["engine_root"] = "evidence/engine"
    if isinstance(config.get("events"), str):
        value["events"] = "evidence/events"
    if isinstance(config.get("client_inventory"), str):
        value["client_inventory"] = "evidence/client-inventory.json"
    return value


def _failure_report(errors):
    return assessment.report_sections({"schema": 1, "scope": assessment.SCOPE,
                                      "status": "inconclusive", "missing": list(errors),
                                      "findings": []}, {})


def render_html(report):
    """Static, escaped, self-contained report: no scripts or remote assets."""
    esc = html.escape
    rows = []
    titles = ["applicability", "fix_presence", "check_execution", "lifecycle", "client_outcomes",
              "regression", "overhead", "missing_evidence"]
    if report.get("schema") == 2:
        titles.insert(2, "request_coverage")
    for title in titles:
        section = report.get("sections", {}).get(title, {})
        value = section.get("status", "see evidence")
        rows.append("<tr><th scope='row'>" + esc(title.replace("_", " ")) + "</th><td>"
                    + esc(str(value)) + "</td><td><pre>"
                    + esc(json.dumps(section, indent=2, sort_keys=True)) + "</pre></td></tr>")
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Scoped #61229 assessment</title><style>"
            "body{font:16px/1.5 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#17202a}"
            "table{width:100%;border-collapse:collapse}th,td{border:1px solid #aaa;padding:12px;text-align:left;vertical-align:top}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}a{color:#124e96}</style>"
            "<main><h1>Within scope: " + esc(report["status"].upper()) + "</h1><p>"
            + esc(report["scope"]) + "</p><p>Local evidence integrity is separate from the verdict. "
            "No authenticity signature or remote deployment attestation is provided. "
            "Unprovided client, regression and performance evidence remains unknown.</p>"
            "<p><a href='report.json'>JSON report</a> · <a href='manifest.json'>Evidence manifest</a></p>"
            "<h2>Declared configuration</h2><p>Live model weights are not independently verified.</p><pre>"
            + esc(json.dumps(report.get("configuration", {}), indent=2, sort_keys=True)) + "</pre>"
            "<h2>Missing lifecycle evidence</h2><pre>"
            + esc(json.dumps(report.get("missing", []), indent=2)) + "</pre>"
            "<h2>Evidence and limits</h2><table><thead><tr><th>Aspect</th><th>Status</th><th>Evidence</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></main></html>\n").encode()


def create_bundle(config_path: str | Path, output: str | Path, *, run_ready=True):
    """Capture first, assess the captured bytes, then check source stability.

    An existing destination is always refused. Partial files are retained on an
    unexpected write failure. A completed manifest is the final write.
    """
    output = Path(output).absolute()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    config_path = Path(config_path).absolute()
    inventory, sources, absent, errors, source_roots = {}, [], [], [], []
    directory = None
    config, portable, report = None, None, None
    total = 0

    def capture(source, destination, role, *, root=None):
        nonlocal total
        data, signature = _read_regular(source, root=root)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise CaptureError("capture exceeds supported total byte limit")
        _write_packet(output, destination, data, role, inventory)
        sources.append((source, signature, _digest(data), root))
        return data

    try:
        raw, signature = _read_regular(config_path)
        config = json.loads(raw)
        _safe_config(config)
        _write_packet(output, "evidence/config-original.json", raw, "original_config", inventory)
        sources.append((config_path, signature, _digest(raw), None))
        total += len(raw)
        portable = _portable_config(config)
        _write_packet(output, "config.json", _json_bytes(portable), "portable_config", inventory)
        _write_packet(output, "metadata/engine-pins.json", _json_bytes(assessment.pins()), "adapter_pins", inventory)

        engine = config.get("engine_root")
        if isinstance(engine, str):
            requested_root = config_path.parent / engine
            engine_root = requested_root.resolve()
            source_roots.append((requested_root, engine_root))
            for relative in assessment.pins()["variants"]["fixed"]:
                _relative(relative)
                source = engine_root / relative
                if not source.exists() and not source.is_symlink():
                    absent.append(source)
                    if assessment.pins()["variants"]["stock"].get(relative) is not None:
                        errors.append("a required inspected engine file was unavailable")
                    continue
                capture(source, "evidence/engine/" + relative, "engine_source", root=engine_root)
        events = config.get("events")
        if isinstance(events, str):
            events_path = (config_path.parent / events).absolute()
            source_roots.append((events_path, events_path.resolve()))
            if events_path.is_dir():
                directory = (events_path, _event_names(events_path))
                paths = [events_path / name for name in directory[1]]
            else:
                paths = [events_path]
            for i, path in enumerate(paths):
                capture(path, f"evidence/events/{i:06d}.jsonl", "recording", root=path.parent)
        client_inventory = config.get("client_inventory")
        if isinstance(client_inventory, str):
            path = (config_path.parent / client_inventory).absolute()
            source_roots.append((path, path.resolve()))
            capture(path, "evidence/client-inventory.json", "client_inventory", root=path.parent)
        # The ordinary reader sees only immutable captured copies, never a second
        # read of a live recording after a fine result has been computed.
        _mkdir_packet_directory(output, "evidence/events")
        report = assessment.assess(portable, base=output.resolve(), run_ready=run_ready)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append("input capture could not be completed: "
                      + (str(exc) if isinstance(exc, CaptureError) else type(exc).__name__))

    # Recheck complete bytes and membership, including files absent on stock.
    for source, signature, digest, root in sources:
        try:
            data, now = _read_regular(source, root=root)
            if now != signature or _digest(data) != digest:
                errors.append("a source changed during assessment capture")
        except (OSError, ValueError):
            errors.append("a source changed or became unavailable during assessment capture")
    if any(path.exists() or path.is_symlink() for path in absent):
        errors.append("an absent engine source appeared during assessment capture")
    for source, resolved in source_roots:
        try:
            if source.resolve() != resolved:
                errors.append("a configured source path changed during assessment capture")
        except OSError:
            errors.append("a configured source path became unavailable during assessment capture")
    if directory is not None:
        try:
            if _event_names(directory[0]) != directory[1]:
                errors.append("recording directory membership changed during assessment capture")
        except (OSError, ValueError):
            errors.append("recording directory became unavailable during assessment capture")
    # Do not let a changed packet copy claim the verdict computed from other bytes.
    for name, entry in inventory.items():
        try:
            data, _ = _read_regular(output / name, root=output)
            if _digest(data) != entry["sha256"]:
                errors.append("a captured packet file changed during assessment")
        except (OSError, ValueError):
            errors.append("a captured packet file became unavailable during assessment")
    errors = list(dict.fromkeys(errors))
    if report is None:
        report = _failure_report(errors)
    elif errors:
        report["status"] = "inconclusive"
        report["missing"] = list(dict.fromkeys(report["missing"] + errors))
        assessment.report_sections(report, portable)
    report["capture"] = {"complete": not errors, "errors": errors,
                         "basis": CAPTURE_BASIS,
                         "remote_attestation": False, "authenticity": "not established"}
    if "readiness" in report:
        _write_packet(output, "evidence/readiness.json", _json_bytes(report["readiness"]), "historical_readiness", inventory)
    _write_packet(output, "report.json", _json_bytes(report), "assessment_report", inventory)
    _write_packet(output, "report.html", render_html(report), "readable_report", inventory)
    manifest = {"schema": 1, "format": FORMAT, "scope": report["scope"], "files": inventory,
                "capture_complete": not errors, "capture_errors": errors,
                "integrity": INTEGRITY,
                "remote_attestation": False}
    # Completion marker: no writes follow this manifest.
    with os.fdopen(_open_relative(output, MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL), "wb") as stream:
        stream.write(_json_bytes(manifest))
        stream.flush()
        os.fsync(stream.fileno())
    return report


def _packet_files(root):
    found = set()
    for current, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            p = Path(current) / name
            if p.is_symlink():
                raise CaptureError("packet contains a symlink")
        for name in names:
            p = Path(current) / name
            if not stat.S_ISREG(p.stat().st_mode):
                raise CaptureError("packet contains a non-regular file")
            found.add(p.relative_to(root).as_posix())
    return found


def verify_bundle(directory: str | Path):
    """Verify packet integrity and recompute core local facts without network IO."""
    result = {"integrity": "failed", "scope_verdict": "inconclusive", "errors": [],
              "authenticity": "not established", "remote_attestation": False,
              "readiness": "not refreshed; any saved readiness is historical evidence"}
    root = Path(directory).absolute()
    try:
        if root.is_symlink() or not root.is_dir():
            raise CaptureError("packet must be a regular local directory")
        raw_manifest, _ = _read_regular(root / MANIFEST, root=root)
        manifest = json.loads(raw_manifest)
        if (not isinstance(manifest, dict) or set(manifest) != {"schema", "format", "scope", "files",
                "capture_complete", "capture_errors", "integrity", "remote_attestation"}
                or type(manifest.get("schema")) is not int
                or manifest["schema"] != 1 or manifest.get("format") != FORMAT
                or manifest.get("scope") not in (assessment.SCOPE, assessment.WORKLOAD_SCOPE) or manifest.get("integrity") != INTEGRITY
                or manifest.get("remote_attestation") is not False
                or type(manifest.get("capture_complete")) is not bool
                or not isinstance(manifest.get("capture_errors"), list)
                or any(not isinstance(error, str) for error in manifest["capture_errors"])
                or not isinstance(manifest.get("files"), dict)):
            raise CaptureError("unsupported packet manifest")
        actual = _packet_files(root)
        expected = set(manifest["files"]) | {MANIFEST}
        for name, entry in manifest["files"].items():
            _relative(name)
            if (not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "role"}
                    or type(entry["bytes"]) is not int or entry["bytes"] < 0
                    or not isinstance(entry["role"], str)):
                raise CaptureError("invalid packet file receipt")
            data, _ = _read_regular(root / name, root=root)
            if len(data) != entry["bytes"] or _digest(data) != entry["sha256"]:
                raise CaptureError("packet file hash or length mismatch")
        if actual != expected:
            raise CaptureError("packet files do not match manifest")
        report = json.loads(_read_regular(root / "report.json", root=root)[0])
        if not isinstance(report, dict):
            raise CaptureError("invalid assessment report")
        if manifest.get("capture_complete") is not True:
            result.update(integrity="incomplete", errors=list(manifest.get("capture_errors", [])))
            return result
        if manifest["capture_errors"] or report.get("capture") != {
                "complete": True, "errors": [], "basis": CAPTURE_BASIS,
                "remote_attestation": False, "authenticity": "not established"}:
            raise CaptureError("report capture or integrity claims do not match the manifest")
        config = json.loads(_read_regular(root / "evidence/config-original.json", root=root)[0])
        _safe_config(config)
        portable = json.loads(_read_regular(root / "config.json", root=root)[0])
        if portable != _portable_config(config):
            raise CaptureError("portable configuration does not match original captured config")
        expected_scope = assessment.WORKLOAD_SCOPE if config.get("schema") == 2 else assessment.SCOPE
        if manifest["scope"] != expected_scope:
            raise CaptureError("manifest scope does not match the selected assessment contract")
        if json.loads(_read_regular(root / "metadata/engine-pins.json", root=root)[0]) != assessment.pins():
            raise CaptureError("installed adapter pins differ; core facts cannot be verified")
        # Readiness is a historical receipt, never an HTTP call during verification.
        core_config = dict(portable)
        core_config.pop("ready", None)
        core = assessment.assess(core_config, base=root.resolve(), run_ready=False)
        for key in ("schema", "scope", "engine", "findings", "model_binding", "configuration",
                    "excluded", "integrity_scope", "request_coverage"):
            if _json_bytes(core.get(key)) != _json_bytes(report.get(key)):
                raise CaptureError("captured evidence does not reproduce the reported core facts")
        if not set(core.get("missing", [])).issubset(set(report.get("missing", []))):
            raise CaptureError("reported missing evidence omits a required captured-fact gap")
        expected_sections = assessment.report_sections(deepcopy(report), portable)["sections"]
        if report.get("sections") != expected_sections:
            raise CaptureError("report sections do not match the captured facts and supplied evidence")
        if _read_regular(root / "report.html", root=root)[0] != render_html(report):
            raise CaptureError("readable report does not match the JSON report")
        if report.get("status") not in ("fine", "not fine", "inconclusive"):
            raise CaptureError("invalid scope verdict")
        if core["status"] == "inconclusive" and report["status"] != "inconclusive":
            raise CaptureError("incomplete local evidence was reported conclusively")
        if report["status"] != "inconclusive" and core["status"] != report["status"]:
            raise CaptureError("scope verdict contradicts captured evidence")
        if "ready" in config and report["status"] != "inconclusive":
            ready = json.loads(_read_regular(root / "evidence/readiness.json", root=root)[0])
            if ready != report.get("readiness") or ready.get("exit_code") != 0:
                raise CaptureError("reported readiness does not match saved successful receipt")
        for name, entry in manifest["files"].items():
            data, _ = _read_regular(root / name, root=root)
            if len(data) != entry["bytes"] or _digest(data) != entry["sha256"]:
                raise CaptureError("packet changed during offline verification")
        result.update(integrity="verified", scope_verdict=report["status"],
                      files_verified=len(manifest["files"]), core_facts_recomputed=True)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result["errors"].append(str(exc))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify", help="verify a saved packet offline; never refresh readiness")
    verify.add_argument("directory")
    a = p.parse_args(argv)
    result = verify_bundle(a.directory)
    print(json.dumps(result, indent=2))
    return 0 if result["integrity"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
