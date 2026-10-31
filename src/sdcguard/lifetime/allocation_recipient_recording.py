"""Explicit finite recipient export and offline local consistency checking.

Owner: Hari Krishna Kancharla.
Collection is bounded host metadata in memory. A crash before export supplies no
capture. Export requires actual mapping raw/settled-receipt bytes and starts no
engine, traffic, SDK, device operation or current owner-packet integration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .. import evidence_io as io
from .. import config_inputs
from . import allocation_recipient_obligations as contract
from . import mapping_closure


FORMAT = "sdcguard.allocation-recipient-packet.v1"
MANIFEST = "RECIPIENT_MANIFEST.json"


def _require(value, reason):
    if not value:
        raise io.EvidenceError(reason)


def _identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _line(record):
    raw, _ = contract._bounded(record)
    return raw + b"\n"


def _completed(records):
    begins = {r["call"]: r for r in records if r["kind"] == "phase_begin"}
    ends = {r["call"]: r["outcome"] for r in records if r["kind"] in ("phase_end", "allocation_end")}
    count = 0
    for record in records:
        if record["kind"] != "pool_return" or not record["blocks"] or ends.get(record["outer"]) != "returned":
            continue
        phase = record["phase"]
        eligible, visited = True, set()
        while phase != record["outer"]:
            if phase in visited or phase not in begins or ends.get(phase) != "returned":
                eligible = False
                break
            visited.add(phase);phase = begins[phase]["parent"]
        count += eligible
    return count


def _stamp(observer):
    return dict(status=observer.status, records=hashlib.sha256(b"".join(_line(r) for r in observer.records)).hexdigest())


def prepare(observer, mapping_payload, mapping_receipts):
    """Freeze facts and exact paired bytes; this never publishes or runs traffic.

    Missing or negative settlement returns an unknown report with the partial
    recipient facts retained. It never substitutes a healthy raw footer.
    """
    from .allocation_recipient_observer import AllocationRecipientObserver
    _require(type(observer) is AllocationRecipientObserver, "recipient observer type is unsupported")
    payload = dict(mapping_payload) if type(mapping_payload) is dict else {}
    receipts = dict(mapping_receipts) if type(mapping_receipts) is dict else {}
    binding = None
    pair_error = None
    try:
        captured = mapping_closure.validate_captured(payload, receipts)
        _require(len(captured["payload"]) == 1, "recipient requires exactly one settled mapping session")
        name, raw = next(iter(captured["payload"].items()))
        receipt_name, receipt_raw = next(iter(captured["receipts"].items()))
        binding = dict(journal=dict(name=name, **_identity(raw)), receipt=dict(name=receipt_name, **_identity(receipt_raw)))
    except (ValueError, OSError, TypeError, KeyError) as exc:
        pair_error = str(exc)[:256]
    with observer._lock:
        records = deepcopy(observer.records)
        stamp = _stamp(observer)
        if records:
            measured = contract.measure(records)
            healthy = (observer._closed and not observer._errors and not observer._dropped and not observer._open
                       and observer._source_stable and observer._coverage_stable and binding is not None)
            footer = dict(schema=1, session=observer.session, process=records[0]["process"], seq=len(records), kind="footer",
                records=len(records) + 1, outer_calls=observer._outer_calls, phase_calls=observer._phase_calls,
                pool_returns=observer._pool_returns, block_facts=observer._block_facts,
                empty_pool_returns=observer._empty_returns, completed_grants=_completed(records),
                errors=observer._errors + int(pair_error is not None), dropped=observer._dropped, healthy=healthy,
                closed=observer._closed, open_calls=len(observer._open), source_stable=observer._source_stable,
                coverage_stable=observer._coverage_stable, mapping_binding=binding, **measured)
            contract._bounded(footer)
            records.append(footer)
    report = contract.evaluate(records, payload, receipts)
    if pair_error is not None:
        report["missing"] = [pair_error] + report.get("missing", [])
    raw = b"".join(_line(record) for record in records)
    _require(len(raw) <= contract.MAX_TOTAL_BYTES, "recipient physical journal exceeds its separate byte bound")
    return dict(records=records, raw=raw, payload=payload, receipts=receipts, report=report, stamp=stamp)


def write_capture(observer, directory, mapping_payload, mapping_receipts):
    """Write a new portable local packet. Existing directories are refused.

    Partial results remain explicitly incomplete. Source facts are held in RAM
    until this explicit export; this is not a durable live recorder or a complete
    operator workflow. Manifest publication supplies local consistency only.
    Admission reserves the only export attempt, including failed attempts. No
    retry or second destination is admitted for the same observer.
    """
    from .allocation_recipient_observer import AllocationRecipientObserver
    _require(type(observer) is AllocationRecipientObserver, "recipient observer type is unsupported")
    with observer._lock:
        _require(observer._export_attempted is False, "recipient export attempt was already reserved")
        observer._export_attempted = True
        observer._export_in_progress = True
    try:
        return _write_reserved(observer, directory, mapping_payload, mapping_receipts)
    finally:
        with observer._lock:
            observer._export_in_progress = False


def _write_reserved(observer, directory, mapping_payload, mapping_receipts):
    frozen = prepare(observer, mapping_payload, mapping_receipts)
    complete = (frozen["records"] and frozen["records"][-1].get("healthy") is True
                and frozen["report"]["metadata_completeness"] == "complete")
    name = "recipient.jsonl" if complete else "recipient.jsonl.incomplete"
    inputs = {name: frozen["raw"]}
    for collection in (frozen["payload"], frozen["receipts"]):
        for relative, raw in collection.items():
            _require(type(relative) is str and "/" not in relative and "\\" not in relative
                     and type(raw) is bytes, "recipient paired input name or bytes are unsupported")
            io.relative(relative)
            target = "mapping/" + relative
            _require(target not in inputs, "recipient paired input names overlap")
            inputs[target] = raw
    _require(len(inputs) <= 3 and sum(map(len, inputs.values())) <= contract.MAX_TOTAL_BYTES + 64 * 1024 * 1024,
             "recipient packet exceeds separate capture bounds")
    root = io.create_directory(directory)
    # No cleanup removes an unfinished export after a failed write.
    io.write_snapshot(root, inputs)
    with observer._lock:
        _require(_stamp(observer) == frozen["stamp"], "recipient facts changed during export")
        manifest = dict(schema=FORMAT, owner="Hari Krishna Kancharla", adapter=contract.DEFAULT_ADAPTER,
                        recipient_file=name, files=io.file_receipts(inputs), authenticity="not established",
                        scope="positive local capacity-recipient observations; permissions and runtime validation not established")
        io.write_file(root, MANIFEST, io.json_bytes(manifest))
        observer._exported = True
    checked = verify(root)
    if complete:
        _require(checked["report"] == frozen["report"], "recipient exported report differs from frozen facts")
    return dict(path=str(Path(root)), **checked)


def _inventory(root, expected):
    """Visit only the exact named root and optional mapping child directory.

    Names are consumed from scandir one at a time. An unexpected name is refused
    before retaining it, reading its contents or opening an unrelated directory.
    At most four files and one child directory can belong to this format.
    """
    allowed_files = set(expected) | {MANIFEST}
    allowed_directories = {"mapping"} if any(name.startswith("mapping/") for name in expected) else set()
    file_limit = min(io.MAX_FILES, len(allowed_files))
    directory_limit = min(io.MAX_FILES - 1, len(allowed_directories))
    _require(file_limit >= 0 and directory_limit >= 0, "recipient inventory bounds are invalid")
    found_files, found_directories = set(), set()
    entries = 0

    def visit(directory, prefix):
        nonlocal entries
        with os.scandir(directory) as stream:
            for entry in stream:
                entries += 1
                name = entry.name
                _require(len(io.relative(name)) == 1, "recipient packet entry name is unsupported")
                path = prefix + name
                _require(path in allowed_files or path in allowed_directories,
                         "recipient packet contains an unlisted file or unlisted directory")
                _require(entries <= file_limit + directory_limit, "recipient packet entry count exceeds its bound")
                info = entry.stat(follow_symlinks=False)
                if path in allowed_directories:
                    _require(stat.S_ISDIR(info.st_mode) and len(found_directories) < directory_limit,
                             "recipient declared directory is missing or exceeds its bound")
                    found_directories.add(path)
                    child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    try:
                        opened = os.fstat(child)
                        _require((info.st_dev, info.st_ino) == (opened.st_dev, opened.st_ino),
                                 "recipient directory changed during inventory")
                        visit(child, path + "/")
                    finally:
                        os.close(child)
                else:
                    _require(stat.S_ISREG(info.st_mode) and len(found_files) < file_limit,
                             "recipient packet member is not a bounded regular file")
                    found_files.add(path)

    directory = config_inputs._directory(Path(root).absolute())
    try:
        visit(directory, "")
    finally:
        os.close(directory)
    _require(found_files == allowed_files and found_directories == allowed_directories,
             "recipient packet exact inventory differs")
    return sorted(found_files), sorted(found_directories)


def verify(directory):
    """Recompute exact packet integrity and causal facts without engine access."""
    root = Path(directory).absolute()
    manifest_raw = config_inputs._read_regular(root, (MANIFEST,), 64 * 1024)
    manifest = io.parse_json(manifest_raw, max_nodes=4096)
    _require(type(manifest) is dict and set(manifest) == {"schema", "owner", "adapter", "recipient_file", "files", "authenticity", "scope"}
             and manifest["schema"] == FORMAT and manifest["owner"] == "Hari Krishna Kancharla"
             and manifest["adapter"] == contract.DEFAULT_ADAPTER and manifest["authenticity"] == "not established"
             and manifest["scope"] == "positive local capacity-recipient observations; permissions and runtime validation not established",
             "recipient manifest fields are unsupported")
    name = manifest["recipient_file"]
    expected = manifest["files"]
    _require(name in ("recipient.jsonl", "recipient.jsonl.incomplete") and type(expected) is dict
             and 1 <= len(expected) <= 3 and name in expected,
             "recipient packet exact inventory differs")
    for path, row in expected.items():
        _require(type(path) is str and (path == name or re.fullmatch(
            r"mapping/[0-9a-f]{32}\.(?:jsonl|closure\.json)", path) is not None),
            "recipient paired file path is unsupported")
        _require(type(row) is dict and set(row) == {"bytes", "sha256"}
                 and type(row["bytes"]) is int and 0 <= row["bytes"] <= io.MAX_FILE_BYTES
                 and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None,
                 "recipient file identity fields are malformed")
    inventory = _inventory(root, expected)
    _require(sum(row["bytes"] for row in expected.values()) <= contract.MAX_TOTAL_BYTES + 64 * 1024 * 1024,
             "recipient packet aggregate exceeds its separate bounds")
    _require(expected[name]["bytes"] <= contract.MAX_TOTAL_BYTES, "recipient journal exceeds separate byte bounds")
    files = {}
    remaining = contract.MAX_TOTAL_BYTES + 64 * 1024 * 1024
    for path, row in sorted(expected.items()):
        raw = config_inputs._read_regular(root, io.relative(path), min(row["bytes"], remaining))
        files[path] = raw
        remaining -= len(raw)
    _require(expected == io.file_receipts(files), "recipient packet bytes differ from manifest")
    _require(config_inputs._read_regular(root, (MANIFEST,), len(manifest_raw)) == manifest_raw,
             "recipient manifest changed during capture")
    payload, receipts = {}, {}
    for path, raw in files.items():
        if path == name:
            continue
        parts = io.relative(path)
        _require(len(parts) == 2 and parts[0] == "mapping", "recipient paired file path is unsupported")
        if parts[1].endswith(".jsonl"):
            payload[parts[1]] = raw
        else:
            _require(parts[1].endswith(".closure.json"), "recipient paired file kind is unsupported")
            receipts[parts[1]] = raw
    raw = files[name]
    _require(len(raw) <= contract.MAX_TOTAL_BYTES, "recipient journal exceeds separate byte bounds")
    parse_error = None
    try:
        _require(raw.endswith(b"\n") and 2 <= len(raw.splitlines()) <= contract.MAX_RECORDS,
                 "recipient journal is empty, unclosed or exceeds record bounds")
        records = [io.parse_json(line, max_nodes=contract.MAX_RECORD_NODES) for line in raw.splitlines()]
    except (ValueError, TypeError, RecursionError) as exc:
        records, parse_error = [], str(exc)[:256]
    report = contract.evaluate(records, payload, receipts)
    if parse_error is not None:
        report["missing"] = [parse_error] + report.get("missing", [])
    # The report was computed only from the frozen bytes above.
    _require(_inventory(root, expected) == inventory, "recipient packet changed during capture")
    _require(config_inputs._read_regular(root, (MANIFEST,), len(manifest_raw)) == manifest_raw
             and all(config_inputs._read_regular(root, io.relative(path), len(value)) == value
                     for path, value in files.items()), "recipient packet bytes changed during verification")
    _require(_inventory(root, expected) == inventory, "recipient packet changed during byte recheck")
    return dict(schema=FORMAT, integrity="locally consistent", authenticity="not established",
                manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), report=report)
