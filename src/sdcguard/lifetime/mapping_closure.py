"""One post-close local bookkeeping receipt and strict owner capture.

Owner: Hari Krishna Kancharla.
This is not a native event, signature, ownership proof or runtime attestation.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import uuid

from .. import config_inputs
from .. import evidence_io as io
from . import assignment_obligations as contract


FORMAT = "sdcguard.mapping-closure.v1"
MAX_RECEIPT_BYTES = 64 * 1024
MAX_SESSIONS = 32
_SESSION = re.compile(r"[0-9a-f]{32}\Z")
_SUFFIX = ".closure.json"
_KEYS = {"format", "kind", "session", "process", "identity", "close_succeeded",
         "observer", "journal", "file", "authenticity"}
_BODY_KINDS = {"init_begin", "configs_derived", "worker_init", "runner_ready", "init_return",
               "scheduler_ready", "retire", "mutation_begin", "mutation_end", "acquire",
               "unsupported_branch", "schedule", "worker_entry", "frame_begin", "assignments",
               "model_entry", "frame_end", "gap"}


def _require(condition, reason):
    if not condition:
        raise io.EvidenceError(reason)


def _receipt_name(session):
    _require(type(session) is str and _SESSION.fullmatch(session), "closure session is unsupported")
    return session + _SUFFIX


def _read(root, name, limit):
    return config_inputs._read_regular(Path(root), io.relative(name), limit)


def _header_footer(raw):
    _require(type(raw) is bytes and raw and raw.endswith(b"\n")
             and len(raw) <= contract.MAX_TOTAL_BYTES, "closure journal is empty or unfinished")
    lines = raw.splitlines()
    _require(2 <= len(lines) <= contract.MAX_RECORDS, "closure journal envelope exceeds bounds")
    records, nodes, positive = [], 0, 0
    for index, line in enumerate(lines):
        _require(0 < len(line) <= contract.MAX_RECORD_BYTES, "closure journal record exceeds bounds")
        record = io.parse_json(line, max_nodes=contract.MAX_RECORD_NODES)
        _require(type(record) is dict and type(record.get("schema")) is int and record["schema"] == 1
                 and type(record.get("seq")) is int and record["seq"] == index
                 and contract._id(record.get("session")) and contract._id(record.get("process")),
                 "closure journal record envelope is unsupported")
        _, count = contract._bounded(record)
        nodes += count
        _require(nodes <= contract.MAX_TOTAL_NODES, "closure journal aggregate node bound exceeded")
        if index:
            _require(record["session"] == records[0]["session"] and record["process"] == records[0]["process"],
                     "closure journal record identity differs")
        kind = record.get("kind")
        expected = {"header"} if index == 0 else {"footer"} if index == len(lines) - 1 else _BODY_KINDS
        _require(type(kind) is str and kind in expected, "closure journal record kind is unsupported")
        if kind == "schedule":
            requests = record.get("requests")
            _require(type(requests) is list and len(requests) <= contract.MAX_REQUESTS,
                     "closure cannot count an unavailable scheduled request set")
            positive += bool(requests)
        records.append(record)
    return records[0], records[-1], dict(
        records=len(records), events=len(records) - 2, accepted_records=len(records) - 1,
        positive_dispatches=positive, samples=sum(r["kind"] == "frame_end" for r in records),
        acquisitions=sum(r["kind"] == "acquire" for r in records), total_nodes=nodes)


def _validate(receipt, raw, name):
    _require(type(receipt) is dict and set(receipt) == _KEYS and receipt["format"] == FORMAT
             and receipt["kind"] == "recorder_closure_bookkeeping"
             and receipt["authenticity"] == "not established", "unsupported closure receipt")
    session = receipt["session"]
    _receipt_name(session)
    _require(name == session + ".jsonl" and contract._id(receipt["process"]),
             "closure filename or process differs")
    identity, observed, journal, file = (receipt[k] for k in ("identity", "observer", "journal", "file"))
    _require(type(identity) is dict and set(identity) == {"adapter", "source_sha256", "profile", "dependencies", "process"}
             and identity["process"] == receipt["process"]
             and identity["adapter"] == "vllm.0.32.uniproc.eager.native.v1"
             and type(observed) is dict and set(observed) == {"healthy", "closed", "sealed", "events", "positive_dispatches"}
             and type(journal) is dict and type(file) is dict and set(file) == {"name", "sha256", "bytes"},
             "closure bookkeeping shape differs")
    _require(receipt["close_succeeded"] is True
             and all(observed.get(k) is True for k in ("healthy", "closed", "sealed"))
             and all(journal.get(k) is True for k in ("healthy", "closed", "sealed", "footer_written", "published"))
             and type(observed["events"]) is int and observed["events"] >= 0
             and type(observed["positive_dispatches"]) is int and observed["positive_dispatches"] >= 0
             and journal.get("session") == session and journal.get("process") == receipt["process"]
             and journal.get("process_changed") is False
             and type(journal.get("errors")) is int and journal["errors"] == 0
             and type(journal.get("dropped")) is int and journal["dropped"] == 0,
             "closure did not retain a successful healthy settled interval")
    for key in ("records", "accepted_records", "samples", "acquisitions", "late_calls", "queued_bytes",
                "total_bytes", "total_nodes"):
        _require(type(journal.get(key)) is int and journal[key] >= 0,
                 "closure journal counter is unsupported")
    _require(file["name"] == name and type(file["bytes"]) is int and file["bytes"] == len(raw)
             and type(file["sha256"]) is str and file["sha256"] == hashlib.sha256(raw).hexdigest(),
             "closure journal bytes differ")
    header, footer, counts = _header_footer(raw)
    _require(type(header) is dict and header.get("kind") == "header" and header.get("session") == session
             and header.get("process") == receipt["process"]
             and all(header.get(k) == identity[k] for k in ("adapter", "source_sha256", "profile", "dependencies")),
             "closure journal identity differs")
    contract._Trace(header)
    _require(observed["events"] == counts["events"]
             and observed["positive_dispatches"] == counts["positive_dispatches"]
             and journal["accepted_records"] == counts["accepted_records"]
             and journal["queued_bytes"] == 0 and journal["total_nodes"] == counts["total_nodes"],
             "closure settled bookkeeping differs from the recorded body")
    for key in ("records", "samples", "acquisitions"):
        _require(type(footer.get(key)) is int and footer[key] == counts[key] == journal[key],
                 "closure footer count differs from actual recorded event kinds")
    _require(type(footer) is dict and footer.get("kind") == "footer" and footer.get("session") == session
             and footer.get("process") == receipt["process"] and footer.get("healthy") is True
             and type(footer.get("errors")) is int and footer["errors"] == 0
             and type(footer.get("dropped")) is int and footer["dropped"] == 0
             and journal.get("total_bytes") == len(raw), "closure footer accounting differs")


def _publish(root, name, raw):
    """Publish an exclusive complete inode. Cleanup cannot undo the receipt."""
    _require(len(raw) <= MAX_RECEIPT_BYTES, "closure receipt exceeds byte bound")
    parent = config_inputs._directory(Path(root).absolute())
    temporary = "." + name + "." + uuid.uuid4().hex + ".tmp"
    fd = None
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        offset = 0
        while offset < len(raw):
            count = os.write(fd, raw[offset:])
            _require(type(count) is int and count > 0, "closure receipt write is incomplete")
            offset += count
        os.fsync(fd)
        own = os.fstat(fd)
        named = os.stat(temporary, dir_fd=parent, follow_symlinks=False)
        _require(stat.S_ISREG(named.st_mode) and (own.st_dev, own.st_ino, own.st_size)
                 == (named.st_dev, named.st_ino, named.st_size) and own.st_size == len(raw),
                 "closure receipt inode changed before publication")
        os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        # There is no fallible acceptance operation after the exclusive link.
        return True
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(temporary, dir_fd=parent)
        except OSError:
            pass
        try:
            os.close(parent)
        except OSError:
            pass


def write_receipt(facts):
    """Write once after close. Publication does not change that close result."""
    _require(type(facts) is dict and set(facts) == {"close_succeeded", "identity", "observer"},
             "closure facts differ")
    contract._bounded(facts)
    observed, identity = facts["observer"], facts["identity"]
    _require(type(observed) is dict and type(identity) is dict and type(facts["close_succeeded"]) is bool,
             "closure facts are unavailable")
    journal, artifacts = observed.get("journal"), observed.get("artifacts")
    _require(type(journal) is dict and type(artifacts) is dict and type(artifacts.get("path")) is str,
             "no journal established a closure identity")
    session = journal.get("session")
    name = _receipt_name(session)
    path = Path(artifacts["path"])
    _require(path.name == session + ".jsonl" and path.is_absolute(), "closure journal path differs")
    receipt = dict(format=FORMAT, kind="recorder_closure_bookkeeping", session=session,
                   process=identity.get("process"), identity=identity,
                   close_succeeded=facts["close_succeeded"],
                   observer={k:observed[k] for k in ("healthy", "closed", "sealed", "events", "positive_dispatches")},
                   journal=journal, file=dict(name=path.name, sha256=None, bytes=None),
                   authenticity="not established")
    if facts["close_succeeded"]:
        raw = _read(path.parent, path.name, contract.MAX_TOTAL_BYTES)
        receipt["file"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        _validate(receipt, raw, path.name)
        _require(_read(path.parent, path.name, len(raw)) == raw, "journal changed before closure publication")
    published = _publish(path.parent, name, io.json_bytes(receipt))
    if facts["close_succeeded"]:
        captured = capture_closed(path.parent)
        _require(path.name in captured["payload"], "published closure was not captured")
    return published


def validate_captured(payload, receipts):
    """Recheck byte pairs without the original recording or engine access."""
    _require(type(payload) is dict and type(receipts) is dict,
             "closure inputs must be byte dictionaries")
    payload, receipts = dict(payload), dict(receipts)
    _require(0 < len(payload) <= MAX_SESSIONS and len(receipts) == len(payload),
             "closure input count is empty or exceeds bound")
    _require(all(type(name) is str and "/" not in name and name.endswith(".jsonl")
                 and type(raw) is bytes for name, raw in payload.items())
             and all(type(name) is str and "/" not in name and type(raw) is bytes
                     for name, raw in receipts.items()), "closure input names or bytes are unsupported")
    _require(set(receipts) == {name[:-6] + _SUFFIX for name in payload},
             "closure input contains missing or unexpected pairs")
    _require(sum(map(len, payload.values())) + sum(map(len, receipts.values()))
             <= contract.MAX_TOTAL_BYTES, "closure capture exceeds byte bound")
    for name, raw in sorted(payload.items()):
        receipt_raw = receipts[name[:-6] + _SUFFIX]
        _require(len(receipt_raw) <= MAX_RECEIPT_BYTES, "closure receipt exceeds byte bound")
        receipt = io.parse_json(receipt_raw, max_nodes=8192)
        _validate(receipt, raw, name)
    return dict(payload=payload, receipts=receipts, bookkeeping="locally consistent",
                authenticity="not established", client_attribution="unknown")


def capture_closed(directory):
    """Capture exact pairs or raise. A caller must report unknown on rejection."""
    names = io.names(directory)
    _require(names and len(names) <= MAX_SESSIONS * 2, "closure input count is empty or exceeds bound")
    journals = [name for name in names if name.endswith(".jsonl")]
    expected = {name for journal in journals for name in (journal, journal[:-6] + _SUFFIX)}
    _require(journals and set(names) == expected and all("/" not in name for name in names),
             "closure input contains missing, unfinished or unexpected members")
    payload, receipts, remaining = {}, {}, contract.MAX_TOTAL_BYTES
    for name in journals:
        receipt_name = name[:-6] + _SUFFIX
        raw = _read(directory, name, remaining)
        receipt_raw = _read(directory, receipt_name, MAX_RECEIPT_BYTES)
        remaining -= len(raw) + len(receipt_raw)
        _require(remaining >= 0, "closure capture exceeds byte bound")
        payload[name], receipts[receipt_name] = raw, receipt_raw
    captured = validate_captured(payload, receipts)
    _require(io.names(directory) == names, "closure inventory changed during capture")
    for name, raw in {**payload, **receipts}.items():
        _require(_read(directory, name, len(raw)) == raw, "closure bytes changed during capture")
    return captured
