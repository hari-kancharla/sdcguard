"""Exact local closure pairs and failure controls, never an engine run.

Owner: Hari Krishna Kancharla.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sdcguard import assignment_report
from sdcguard import evidence_io as io
from sdcguard.lifetime import mapping_activation as A
from sdcguard.lifetime import mapping_closure as E
from test_assignment_init import primary
from test_mapping_activation import active, builder, isolated_activation
from test_mapping_observer import observer_environment, run


def finished(value):
    run(value)
    value.runner.shutdown()
    sink, = value.sinks
    receipt = sink.path.with_name(sink.session + ".closure.json")
    return sink, receipt


def test_real_recorder_closure_exports_exact_bytes_and_separate_bookkeeping(active):
    sink, path = finished(active)
    captured = E.capture_closed(path.parent)
    assert captured["payload"] == {sink.path.name:sink.path.read_bytes()}
    assert captured["receipts"] == {path.name:path.read_bytes()}
    assert captured["client_attribution"] == "unknown" and captured["authenticity"] == "not established"
    receipt = json.loads(path.read_bytes())
    assert receipt["kind"] == "recorder_closure_bookkeeping"
    assert receipt["file"]["sha256"] == hashlib.sha256(sink.path.read_bytes()).hexdigest()
    assert receipt["session"] == sink.session and receipt["process"] == sink.process
    assert receipt["identity"]["profile"] == json.loads(sink.path.read_bytes().splitlines()[0])["profile"]
    assert A.status()["observer"]["export_ready"] is True
    assert assignment_report.inspect_payload(captured["payload"])["verdict"] == "satisfied"


def test_real_recorder_complete_sixteen_request_burst(primary, tmp_path, monkeypatch, record_property):
    value = observer_environment(primary, tmp_path, journal=True, installation_builder=builder(monkeypatch),
                                 request_count=16, capacity=64)
    try:
        output = run(value)
        assert output.total_num_scheduled_tokens == 16
        assert value.runner.shutdown() is value.runner.shutdown_result
        status = value.mapping.status
        record_property("ordinary_sixteen_request_burst", json.dumps(status, sort_keys=True))
        assert status["sealed"] and status["export_ready"], json.dumps(status, sort_keys=True)
        captured = E.capture_closed(value.journal_directory)
        result = assignment_report.inspect_payload(captured["payload"])
        assert result["verdict"] == "satisfied"
        records = assignment_report.parse_records(next(iter(captured["payload"].values())))
        assert len(next(r for r in records if r["kind"] == "schedule")["requests"]) == 16
        assert len(next(r for r in records if r["kind"] == "assignments")["rows"]) == 16
        assert sum(r["kind"] == "acquire" for r in records) == 16
        assert sum(r["kind"] == "frame_end" for r in records) == 1
        assert records[-1]["records"] == len(records) == 95
        assert records[-1]["samples"] == 1 and records[-1]["acquisitions"] == 16
        assert records[-1]["errors"] == records[-1]["dropped"] == 0 and records[-1]["healthy"] is True
    finally:
        value.installation.detach()


def test_duplicate_close_does_not_retry_or_replace_its_receipt(active):
    calls = []
    callback = active.mapping._closure_callback
    def counted(facts):
        calls.append(facts)
        return callback(facts)
    active.mapping._closure_callback = counted
    sink, path = finished(active)
    original = path.read_bytes()
    assert active.mapping.close() is True and A.uninstall()["observer"]["sealed"]
    assert len(calls) == 1 and path.read_bytes() == original
    with pytest.raises(FileExistsError):
        E.write_receipt(calls[0])
    assert path.read_bytes() == original and E.capture_closed(path.parent)["payload"]


def test_failed_close_is_recorded_once_and_cannot_be_rewritten_as_healthy(active):
    run(active)
    sink = active.sinks[0]
    close = sink.close
    def negative(timeout=1.0):
        assert close(timeout=timeout) is True
        return False
    sink.close = negative
    active.runner.shutdown()
    path = sink.path.with_name(sink.session + ".closure.json")
    raw = path.read_bytes()
    assert json.loads(raw)["close_succeeded"] is False
    assert A.status()["observer"]["export_ready"] is False
    with pytest.raises(io.EvidenceError):
        E.capture_closed(path.parent)
    sink.close = close
    assert active.mapping.close() is False and path.read_bytes() == raw


@pytest.mark.parametrize("change", [
    lambda r:r.update(close_succeeded=False),
    lambda r:r.update(session="a" * 32),
    lambda r:r.update(process="b" * 32),
    lambda r:r["observer"].update(healthy=False),
    lambda r:r["journal"].update(closed=False),
    lambda r:r["journal"].update(healthy=False),
    lambda r:r["journal"].update(records=True),
    lambda r:r["journal"].update(samples=True),
    lambda r:r["file"].update(sha256="0" * 64),
    lambda r:r["file"].update(bytes=True),
    lambda r:r["identity"]["profile"].update(model_revision="d" * 40),
    lambda r:r["identity"]["source_sha256"].clear(),
    lambda r:r.update(unexpected="field"),
])
def test_stale_malformed_or_mismatched_receipt_never_exports(active, change):
    sink, path = finished(active)
    receipt = json.loads(path.read_bytes())
    change(receipt)
    path.write_bytes(io.json_bytes(receipt))
    with pytest.raises((ValueError, OSError)):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("kind", ["missing", "duplicate_key", "unfinished", "oversized", "symlink"])
def test_absent_or_nonregular_receipt_is_unknown(active, tmp_path, kind):
    sink, path = finished(active)
    original = path.read_bytes()
    if kind == "missing":
        path.unlink()
    elif kind == "duplicate_key":
        path.write_bytes(original.replace(b'{', b'{"format":"duplicate",', 1))
    elif kind == "unfinished":
        path.write_bytes(original[:len(original)//2])
    elif kind == "oversized":
        path.write_bytes(b" " * (E.MAX_RECEIPT_BYTES + 1))
    else:
        target = tmp_path / "receipt-copy"; target.write_bytes(original)
        path.unlink(); path.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        E.capture_closed(sink.path.parent)


@pytest.mark.parametrize("kind", ["changed_bytes", "symlink", "unfinished_member", "foreign_receipt"])
def test_replaced_journal_or_ambiguous_inventory_never_exports(active, tmp_path, kind):
    sink, path = finished(active)
    if kind == "changed_bytes":
        sink.path.write_bytes(sink.path.read_bytes().replace(b'"epoch":0', b'"epoch":9', 1) + b"\n")
    elif kind == "symlink":
        copy = tmp_path / "journal-copy"; copy.write_bytes(sink.path.read_bytes())
        sink.path.unlink(); sink.path.symlink_to(copy)
    elif kind == "unfinished_member":
        sink.incomplete_path.write_bytes(b"unfinished\n")
    else:
        (path.parent / ("e" * 32 + ".closure.json")).write_bytes(path.read_bytes())
    with pytest.raises((ValueError, OSError)):
        E.capture_closed(path.parent)


def test_partial_receipt_publication_does_not_change_actual_interval_close(active, monkeypatch):
    def fail_link(*args, **kwargs):
        raise OSError("host publication unavailable")
    # Journal publication uses the same operating-system function, so retain
    # it until the closure callback has actually begun.
    callback = active.mapping._closure_callback
    def blocked(facts):
        with monkeypatch.context() as changed:
            changed.setattr(E.os, "link", fail_link)
            return callback(facts)
    active.mapping._closure_callback = blocked
    sink, path = finished(active)
    assert active.mapping.status["sealed"] and active.mapping.status["healthy"]
    assert active.mapping.status["export_ready"] is False and not path.exists()
    assert assignment_report.inspect_payload({sink.path.name:sink.path.read_bytes()})["verdict"] == "satisfied"
    with pytest.raises(io.EvidenceError):
        E.capture_closed(sink.path.parent)


def test_unpublished_temporary_bytes_are_never_an_export_receipt(active, monkeypatch):
    publish = E._publish
    def partial(root, name, raw):
        (Path(root) / ("." + name + ".partial.tmp")).write_bytes(raw)
        raise OSError("host final publication unavailable")
    monkeypatch.setattr(E, "_publish", partial)
    sink, path = finished(active)
    assert not path.exists() and active.mapping.status["sealed"]
    assert active.mapping.status["export_ready"] is False
    with pytest.raises(io.EvidenceError):
        E.capture_closed(path.parent)


def test_receipt_publication_cannot_overwrite_an_existing_owned_name(active):
    run(active)
    sink = active.sinks[0]
    path = sink.path.with_name(sink.session + ".closure.json")
    path.write_bytes(b"existing incomplete receipt\n")
    active.runner.shutdown()
    assert path.read_bytes() == b"existing incomplete receipt\n"
    assert active.mapping.status["sealed"] and active.mapping.status["export_ready"] is False
    with pytest.raises((ValueError, OSError)):
        E.capture_closed(path.parent)


def rewrite_pair(sink, path, records, receipt):
    """Update byte binding only, leaving tested bookkeeping claims intact."""
    raw = b"".join(json.dumps(record, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
                   for record in records)
    assert assignment_report.parse_records(raw) == records
    receipt["file"].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    receipt["journal"].update(total_bytes=len(raw),
                            total_nodes=sum(E.contract._bounded(record)[1] for record in records))
    sink.path.write_bytes(raw)
    path.write_bytes(io.json_bytes(receipt))


@pytest.mark.parametrize("counter", ["records", "samples", "acquisitions"])
def test_boolean_footer_counters_are_rejected_even_when_equal_to_an_integer(active, counter):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    records[-1][counter] = True
    rewrite_pair(sink, path, records, receipt)
    with pytest.raises(io.EvidenceError, match="footer count differs"):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("section,counter,value", [
    ("observer", "events", 0),
    ("observer", "positive_dispatches", 0),
    ("journal", "accepted_records", 0),
    ("journal", "queued_bytes", 1),
    ("journal", "total_nodes", 0),
])
def test_settled_claims_must_match_complete_raw_body(active, section, counter, value):
    sink, path = finished(active)
    receipt = json.loads(path.read_bytes())
    receipt[section][counter] = value
    path.write_bytes(io.json_bytes(receipt))
    assert assignment_report.inspect_payload({sink.path.name:sink.path.read_bytes()})["verdict"] == "satisfied"
    with pytest.raises(io.EvidenceError, match="settled bookkeeping differs"):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("counter", ["samples", "acquisitions"])
def test_matching_wrong_footer_and_journal_claims_do_not_replace_event_counts(active, counter):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    records[-1][counter] += 1
    receipt["journal"][counter] = records[-1][counter]
    rewrite_pair(sink, path, records, receipt)
    with pytest.raises(io.EvidenceError, match="footer count differs"):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("kind", ["frame_end", "acquire"])
def test_actual_event_kinds_are_counted_from_the_bound_body(active, kind):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    next(record for record in records if record["kind"] == kind)["kind"] = "gap"
    rewrite_pair(sink, path, records, receipt)
    with pytest.raises(io.EvidenceError, match="footer count differs"):
        E.capture_closed(path.parent)


def test_positive_dispatch_count_comes_from_recorded_nonempty_schedules(active):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    next(record for record in records if record["kind"] == "schedule")["requests"] = []
    rewrite_pair(sink, path, records, receipt)
    with pytest.raises(io.EvidenceError, match="settled bookkeeping differs"):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("field,value", [("seq", True), ("schema", True), ("session", "a" * 32),
                                         ("process", "b" * 32), ("kind", "unsupported")])
def test_every_body_envelope_is_checked_before_export(active, field, value):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    records[1][field] = value
    rewrite_pair(sink, path, records, receipt)
    reason = ("record identity differs" if field in ("session", "process") else
              "record kind is unsupported" if field == "kind" else "record envelope is unsupported")
    with pytest.raises(io.EvidenceError, match=reason):
        E.capture_closed(path.parent)


@pytest.mark.parametrize("kind", ["violated", "unknown"])
def test_complete_bookkeeping_does_not_require_mapping_agreement(active, kind):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    receipt = json.loads(path.read_bytes())
    if kind == "violated":
        next(r for r in records if r["kind"] == "model_entry")["layers"][0]["rows"][0][0] = 0
    else:
        next(r for r in records if r["kind"] == "assignments")["rows"][0]["blocks"] = None
    rewrite_pair(sink, path, records, receipt)
    captured = E.capture_closed(path.parent)
    assert captured["bookkeeping"] == "locally consistent"
    assert assignment_report.inspect_payload(captured["payload"])["verdict"] == kind


@pytest.mark.parametrize("limit", ["MAX_RECORD_NODES", "MAX_TOTAL_NODES"])
def test_raw_record_bounds_are_applied_before_bookkeeping_acceptance(active, monkeypatch, limit):
    sink, path = finished(active)
    records = assignment_report.parse_records(sink.path.read_bytes())
    nodes = [E.contract._bounded(record)[1] for record in records]
    threshold = max(nodes) - 1 if limit == "MAX_RECORD_NODES" else sum(nodes) - 1
    monkeypatch.setattr(E.contract, limit, threshold)
    with pytest.raises(ValueError):
        E.capture_closed(path.parent)
