"""Finite journal CPU controls. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import json
import errno
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from sdcguard import assignment_report as R
from sdcguard.lifetime import assignment_recording as J
from sdcguard.lifetime import assignment_obligations as A
from test_assignment_obligations import ordinary_records, one, recount, token


def arguments(records=None):
    header = (records or ordinary_records())[0]
    return {name: deepcopy(header[name]) for name in ("adapter", "source_sha256", "profile", "coverage", "dependencies")} | {
        "process_incarnation": header["process"]}


def load(recorder):
    path = recorder.path if recorder.status.get("published") else recorder.incomplete_path
    records = []
    for line in path.read_bytes().splitlines():
        try:
            records.append(json.loads(line))
        except (ValueError, UnicodeError):
            break
    return records


def populate(recorder, records):
    for record in records[1:-1]:
        assert recorder.emit(record["kind"], **{k: v for k, v in record.items() if k not in J._RESERVED})


class FileProxy:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def write(self, raw):
        return self.wrapped.write(raw)

    def flush(self):
        return self.wrapped.flush()

    def close(self):
        return self.wrapped.close()

    @property
    def closed(self):
        return self.wrapped.closed


class GateFile(FileProxy):
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, raw):
        self.entered.set()
        assert self.release.wait(3.0)
        return super().write(raw)


@pytest.mark.parametrize("contradiction", [False, True])
def test_complete_journal_is_evaluated_and_footer_counts_exactly(tmp_path, contradiction):
    records = ordinary_records()
    if contradiction:
        next(r for r in records if r["kind"] == "model_entry")["layers"][0]["rows"][0][0] = 0
    journal = J.AssignmentRecorder(tmp_path / "recording", **arguments(records))
    assert not journal.path.exists() and journal.incomplete_path.exists()
    populate(journal, records)
    assert journal.close() is True
    assert journal.path.exists() and not journal.incomplete_path.exists()
    saved = load(journal)
    assert [r["seq"] for r in saved] == list(range(len(saved)))
    footer = saved[-1]
    assert footer["records"] == len(saved) == len(records)
    assert footer["samples"] == sum(r["kind"] == "frame_end" for r in saved) == 1
    assert footer["acquisitions"] == sum(r["kind"] == "acquire" for r in saved) == 1
    assert footer["dropped"] == footer["errors"] == 0 and footer["healthy"] is True
    result = A.evaluate(saved)
    assert result["verdict"] == ("violated" if contradiction else "satisfied")
    assert result["ownership_verdict"] == result["physical_read_verdict"] == "unknown"
    assert stat.S_IMODE(journal.path.stat().st_mode) == 0o600
    assert journal.close() is True


def test_caller_header_and_fact_mutation_cannot_change_stored_records(tmp_path):
    records = ordinary_records()
    kwargs = arguments(records)
    journal = J.AssignmentRecorder(tmp_path, **kwargs)
    kwargs["profile"]["layer_names"].append("changed")
    kwargs["source_sha256"].clear()
    populate(journal, records)
    for record in records:
        record.clear()
    assert journal.close()
    assert A.evaluate(load(journal))["verdict"] == "satisfied"
    for name in ("session", "process", "status", "path", "incomplete_path"):
        with pytest.raises(AttributeError):
            setattr(journal, name, "changed")
    snapshot = journal.status
    snapshot["errors"] = 99
    assert journal.status["errors"] == 0


def test_minimal_one_request_one_layer_trace_round_trips_through_evaluator(tmp_path):
    records = ordinary_records()
    layer = records[0]["profile"]["layer_names"][0]
    records[0]["profile"]["layer_names"] = [layer]
    for record in records:
        if "dimensions" in record:
            record["dimensions"]["layer_names"] = [layer]
        if record["kind"] == "acquire":
            record["blocks"] = record["blocks"][:1]
        elif record["kind"] in ("schedule", "worker_entry"):
            record["requests"] = record["requests"][:1]
        elif record["kind"] == "assignments":
            record["rows"] = record["rows"][:1]
            record["rows"][0]["blocks"] = record["rows"][0]["blocks"][:1]
        elif record["kind"] == "model_entry":
            record.update(requests=["R-a"], seq_lens=[1], query_lengths=[1],
                          layers=[{"name": layer, "rows": [[2]]}])
    assert A.evaluate(records)["verdict"] == "satisfied"
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    populate(journal, records)
    assert journal.close()
    result = A.evaluate(load(journal))
    assert result["verdict"] == "satisfied" and result["summary"]["satisfied"] == 1
    assert result["allocation_incarnations"] == 1


@pytest.mark.parametrize("field,value", [
    ("adapter", "unsupported"), ("process_incarnation", "main"),
    ("dependencies", {"python": "3.13", "torch": "2.8", "flash_attn": "4"}),
    ("coverage", []), ("source_sha256", {}), ("profile", {}),
])
def test_invalid_header_creates_no_directory_or_file(tmp_path, field, value):
    directory = tmp_path / "never-created"
    kwargs = arguments()
    kwargs[field] = value
    with pytest.raises(ValueError):
        J.AssignmentRecorder(directory, **kwargs)
    assert not directory.exists()


@pytest.mark.parametrize("key", ["schema", "session", "process", "seq", "kind"])
def test_reserved_envelope_keys_are_not_admitted(tmp_path, key):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    assert journal.emit("gap", **{key: "override"}) is False
    assert journal.close() is False
    saved = load(journal)
    assert saved[-1]["dropped"] == saved[-1]["errors"] == 1
    assert A.evaluate(saved)["verdict"] == "unknown"


@pytest.mark.parametrize("kind", ["header", "footer", "unsupported", True])
def test_session_markers_and_unsupported_kinds_are_rejected(tmp_path, kind):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    assert journal.emit(kind) is False
    assert journal.close() is False
    assert A.evaluate(load(journal))["verdict"] == "unknown"


@pytest.mark.parametrize("value", [1.0, float("nan"), (1,), b"bytes", object(), 2**64, {False: "key"}])
def test_non_json_or_out_of_range_facts_make_evidence_unknown(tmp_path, value):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    assert journal.emit("gap", reason=value) is False
    assert journal.close() is False
    assert A.evaluate(load(journal))["verdict"] == "unknown"
    assert load(journal)[-1]["errors"] > 0


def test_oversized_and_deep_facts_are_bounded_before_admission(tmp_path):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    assert not journal.emit("gap", reason="x" * (A.MAX_RECORD_BYTES + 1))
    assert not journal.emit("gap", reason=[None] * (A.MAX_RECORD_NODES + 1))
    value = "nested"
    for _ in range(34):
        value = [value]
    assert not journal.emit("gap", reason=value)
    assert journal.close() is False
    assert load(journal)[-1]["dropped"] == 3


@pytest.mark.parametrize("budget", ["bytes", "nodes", "records"])
def test_aggregate_budgets_reserve_space_for_unhealthy_footer(tmp_path, monkeypatch, budget):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    if budget == "bytes":
        monkeypatch.setattr(A, "MAX_TOTAL_BYTES", journal.status["total_bytes"] + J._FOOTER_BYTES + 1)
    elif budget == "nodes":
        monkeypatch.setattr(A, "MAX_TOTAL_NODES", journal.status["total_nodes"] + J._FOOTER_NODES + 1)
    else:
        monkeypatch.setattr(A, "MAX_RECORDS", 2)
    assert journal.emit("gap", reason="bounded observer gap") is False
    assert journal.close() is False
    saved = load(journal)
    assert saved[-1]["kind"] == "footer" and saved[-1]["dropped"] == 1
    assert saved[-1]["records"] == len(saved) == 2
    assert A.evaluate(saved)["verdict"] == "unknown"


def test_queue_rejection_is_recorded_without_sequence_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "_QUEUE_RECORDS", 1)
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    gate = GateFile(journal._file)
    journal._file = gate
    assert journal.emit("gap", reason="first")
    assert gate.entered.wait(1.0)
    assert journal.emit("gap", reason="second")
    assert journal.emit("gap", reason="not admitted") is False
    gate.release.set()
    assert journal.close() is False
    saved = load(journal)
    assert [r["seq"] for r in saved] == list(range(len(saved)))
    assert saved[-1]["dropped"] == 1 and saved[-1]["records"] == len(saved)
    assert A.evaluate(saved)["verdict"] == "unknown"


def test_default_waiting_capacity_overflow_remains_unknown_for_complete_cpu_data(tmp_path):
    assert J._QUEUE_RECORDS == 1024
    records = ordinary_records()
    schedule = one(records, "schedule")
    empty = []
    for number in range(1, 1027):
        row = deepcopy(schedule)
        row.update(step=number, dispatch=token(2000 + number), requests=[])
        empty.append(row)
    schedule["step"] = len(empty) + 1
    index = records.index(schedule)
    records[index:index] = empty
    recount(records)
    assert A.evaluate(records)["verdict"] == "satisfied"
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    gate = GateFile(journal._file)
    journal._file = gate
    accepted = []
    try:
        first = records[1]
        assert journal.emit(first["kind"], **{k:v for k,v in first.items() if k not in J._RESERVED})
        assert gate.entered.wait(1.0)
        for record in records[2:-1]:
            accepted.append(journal.emit(record["kind"], **{k:v for k,v in record.items() if k not in J._RESERVED}))
        assert sum(accepted) == 1024 and accepted.count(False) > 0
    finally:
        gate.release.set()
    assert journal.close() is False
    saved = load(journal)
    assert saved[-1]["dropped"] == accepted.count(False) and saved[-1]["healthy"] is False
    assert A.evaluate(saved)["verdict"] == "unknown"


@pytest.mark.parametrize("short", [False, True])
def test_disk_or_short_write_retains_explicit_unfinished_artifact(tmp_path, short):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    class FailedFile(FileProxy):
        def write(self, raw):
            if short:
                self.wrapped.write(raw[:3])
                return 3
            raise OSError("ordinary journal disk failure")
    journal._file = FailedFile(journal._file)
    assert journal.emit("gap", reason="data")
    assert journal.close() is False
    assert not journal.path.exists() and journal.incomplete_path.exists()
    assert journal.status["errors"] > 0 and not journal.status["published"]
    assert A.evaluate(load(journal))["verdict"] == "unknown"


@pytest.mark.parametrize("failure", ["flush", "close"])
def test_failure_after_footer_bytes_never_publishes_a_healthy_final_file(tmp_path, failure):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    class LateFailure(FileProxy):
        accepted_footer = False
        def write(self, raw):
            result = super().write(raw)
            if json.loads(raw)["kind"] == "footer":
                self.accepted_footer = True
            return result
        def flush(self):
            if failure == "flush" and self.accepted_footer:
                raise OSError("ordinary final flush failure")
            return super().flush()
        def close(self):
            result = super().close()
            if failure == "close" and self.accepted_footer:
                raise OSError("ordinary final close failure")
            return result
    proxy = LateFailure(journal._file)
    journal._file = proxy
    populate(journal, records)
    assert journal.close() is False
    assert proxy.accepted_footer
    assert not journal.path.exists() and journal.incomplete_path.exists()
    assert journal.status["closed"] and not journal.status["published"]
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_shared_health_failure_is_in_the_footer(tmp_path):
    health = SimpleNamespace(snapshot=lambda: {"healthy": False, "errors": 7})
    journal = J.AssignmentRecorder(tmp_path, health=health, **arguments())
    assert journal.close() is False
    assert load(journal)[-1]["errors"] == 7
    assert load(journal)[-1]["healthy"] is False
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_gap_marks_accounting_unhealthy_even_with_otherwise_complete_trace(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    populate(journal, records)
    journal.gap("observer could not capture one required fact")
    assert journal.close() is False
    assert load(journal)[-1]["errors"] == 1
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_timeout_is_bounded_and_open_closing_rejections_are_not_lost(tmp_path):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    gate = GateFile(journal._file)
    journal._file = gate
    assert journal.emit("gap", reason="pending")
    assert gate.entered.wait(1.0)
    started = time.monotonic()
    assert journal.close(timeout=0.01) is False
    assert time.monotonic() - started < 1.0
    assert journal.emit("gap", reason="rejected during closure") is False
    assert journal.status["dropped"] > 0
    gate.release.set()
    assert journal.close() is False
    assert not journal.path.exists() and journal.incomplete_path.exists()
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_calls_after_sealing_are_outside_the_immutable_interval(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    populate(journal, records)
    assert journal.close()
    before = journal.path.read_bytes()
    assert journal.emit("gap", reason="outside the sealed interval") is False
    journal.gap("also outside")
    assert journal.status["late_calls"] == 2
    assert journal.path.read_bytes() == before
    assert A.evaluate(load(journal))["verdict"] == "satisfied"


def test_timeout_during_publication_cannot_leave_a_healthy_named_file(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path, **arguments(records))
    original = journal._publish
    entered, release = threading.Event(), threading.Event()
    def delayed_publication():
        entered.set()
        assert release.wait(3.0)
        original()
    journal._publish = delayed_publication
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(timeout=0.05)))
    closer.start()
    assert entered.wait(1.0)
    closer.join(1.0)
    assert not closer.is_alive() and result == [False]
    release.set()
    assert journal.close() is False
    assert not journal.path.exists() and journal.incomplete_path.exists()
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_changed_pid_never_touches_inherited_locks_or_health(tmp_path, monkeypatch):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    original_lock, original_health = journal._lock, journal._health
    class Trap:
        def __enter__(self):
            raise AssertionError("inherited lock used")
        def snapshot(self):
            raise AssertionError("inherited health used")
    with monkeypatch.context() as changed:
        changed.setattr(J.os, "getpid", lambda: journal._pid + 1)
        journal._lock = journal._health = Trap()
        try:
            assert journal.emit("gap", reason="child") is False
            journal.gap("child")
            assert journal.close() is False
            assert journal.status["process_changed"] is True
        finally:
            journal._lock, journal._health = original_lock, original_health
    assert journal.close()


def test_large_finite_integer_timeout_is_clamped_without_float_overflow(tmp_path):
    journal = J.AssignmentRecorder(tmp_path, **arguments())
    assert journal.close(timeout=10**1000) is True
    assert journal.status["published"] and journal.status["closed"]


def test_secure_paths_and_exclusive_final_name_preserve_existing_files(tmp_path, monkeypatch):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        J.AssignmentRecorder(link / "not-created", **arguments())
    assert not (real / "not-created").exists()
    fixed = "f" * 32
    monkeypatch.setattr(J.uuid, "uuid4", lambda: SimpleNamespace(hex=fixed))
    existing = real / (fixed + ".jsonl")
    existing.write_text("preserve")
    with pytest.raises(FileExistsError):
        J.AssignmentRecorder(real, **arguments())
    assert existing.read_text() == "preserve"
    assert not list(real.glob("*.incomplete"))


@pytest.mark.parametrize("replacement", ["file", "symlink"])
def test_publication_never_overwrites_a_destination_created_after_setup(tmp_path, replacement):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "recording", **arguments(records))
    preserved = tmp_path / "preserved"
    preserved.write_bytes(b"preserve existing data")
    if replacement == "file":
        journal.path.write_bytes(preserved.read_bytes())
    else:
        journal.path.symlink_to(preserved)
    populate(journal, records)
    assert journal.close() is False
    assert preserved.read_bytes() == journal.path.read_bytes() == b"preserve existing data"
    assert journal.status["published"] is False and journal.incomplete_path.exists()
    assert A.evaluate(load(journal))["verdict"] == "unknown"


def test_import_has_no_engine_device_or_network_modules():
    code = ('import sys; from sdcguard.lifetime import assignment_recording; '
            'assert not any(n in sys.modules for n in ("torch","vllm","modal","requests","httpx"))')
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def check_recovered_artifacts(journal, tmp_path, owned_paths):
    """Evaluate the recovered bytes, then capture and verify a report packet."""
    for index, path in enumerate(owned_paths):
        raw = path.read_bytes()
        saved = R.parse_records(raw)
        assert saved and all(record["kind"] != "footer" for record in saved)
        assert A.evaluate(saved)["verdict"] == "unknown"
        capture = tmp_path / ("recovered-" + str(index))
        capture.mkdir()
        (capture / "session.jsonl").write_bytes(raw)
        packet = tmp_path / ("packet-" + str(index))
        result = R.create_packet(capture, packet)
        assert result["verdict"] == "unknown" and result["scope_verdict"] == "inconclusive"
        verified = R.verify_packet(packet)
        assert verified["integrity"] == "verified" and verified["verdict"] == "unknown"
    status = journal.status
    assert status["healthy"] is False and status["errors"] > 0
    assert status["footer_written"] is False and status["closed"] is True
    assert status["records"] == status["accepted_records"]
    # The original capture must also fail closed if it contains an unfinished
    # artifact or an unrelated file. It must never produce a fine packet.
    try:
        result = R.create_packet(journal.path.parent, tmp_path / "original-packet")
    except R.io.EvidenceError:
        pass
    else:
        assert result["verdict"] == "unknown" and result["scope_verdict"] == "inconclusive"


@pytest.mark.parametrize("failure", ["rollback_link", "rollback_unlink", "publish_cleanup"])
def test_publication_recovery_failures_invalidate_owned_footer_before_names(tmp_path, monkeypatch, failure):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    original_publish, original_link, original_unlink = journal._publish, J.os.link, J.os.unlink
    entered, release = threading.Event(), threading.Event()
    preserved = tmp_path / "unrelated.txt"
    preserved.write_bytes(b"unrelated accepted data")

    def link(source, target, *args, **kwargs):
        if failure == "rollback_link" and str(source) == journal.path.name:
            raise OSError(errno.EIO, "controlled rollback link failure")
        return original_link(source, target, *args, **kwargs)

    def unlink(name, *args, **kwargs):
        if (failure == "rollback_unlink" and str(name) == journal.path.name
                or failure == "publish_cleanup" and str(name) in
                (journal.path.name, journal.incomplete_path.name)):
            raise OSError(errno.EIO, "controlled cleanup failure")
        return original_unlink(name, *args, **kwargs)

    def publish():
        entered.set()
        assert release.wait(3.0)
        original_publish()

    monkeypatch.setattr(J.os, "link", link)
    monkeypatch.setattr(J.os, "unlink", unlink)
    journal._publish = publish
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(0.05)))
    closer.start()
    assert entered.wait(1.0)
    closer.join(1.0)
    assert not closer.is_alive() and result == [False]
    release.set()
    journal._thread.join(2.0)
    assert not journal._thread.is_alive()
    assert journal.close() is False
    assert preserved.read_bytes() == b"unrelated accepted data"
    paths = [path for path in (journal.path, journal.incomplete_path) if path.exists()]
    assert journal.path in paths
    assert journal.incomplete_path.exists() is (failure != "rollback_link")
    check_recovered_artifacts(journal, tmp_path, paths)


@pytest.mark.parametrize("foreign_name", ["final", "incomplete"])
def test_timeout_recovery_never_truncates_or_removes_a_foreign_replacement(tmp_path, foreign_name):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    original = journal._publish
    entered, release = threading.Event(), threading.Event()
    preserved = b"foreign file must remain byte exact"
    retained = tmp_path / "retained-owned.jsonl"
    foreign = journal.path if foreign_name == "final" else journal.incomplete_path

    def publish():
        entered.set()
        assert release.wait(3.0)
        original()
        if foreign_name == "final":
            journal.path.rename(retained)
        foreign.write_bytes(preserved)

    journal._publish = publish
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(0.05)))
    closer.start()
    assert entered.wait(1.0)
    closer.join(1.0)
    assert not closer.is_alive() and result == [False]
    release.set()
    journal._thread.join(2.0)
    assert not journal._thread.is_alive()
    assert journal.close() is False and foreign.read_bytes() == preserved
    owned = retained if foreign_name == "final" else journal.path
    check_recovered_artifacts(journal, tmp_path, [owned])
    assert foreign.read_bytes() == preserved


def test_failed_recovery_descriptor_close_uses_only_verified_owned_paths(tmp_path, monkeypatch):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    original_close = J.os.close
    recovery_fd = journal._recovery_fd
    failed = []

    def close(fd):
        result = original_close(fd)
        if fd == recovery_fd and not failed:
            failed.append(True)
            raise OSError(errno.EIO, "controlled recovery descriptor close failure")
        return result

    monkeypatch.setattr(J.os, "close", close)
    populate(journal, records)
    assert journal.close() is False and failed
    assert not journal.path.exists() and journal.incomplete_path.exists()
    check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])


def test_failed_close_never_retries_an_unrelated_reused_descriptor(tmp_path, monkeypatch):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    original_close = J.os.close
    recovery_fd = journal._recovery_fd
    preserved = tmp_path / "preserved.txt"
    preserved.write_bytes(b"unrelated descriptor contents")
    reused = []

    def close(fd):
        result = original_close(fd)
        if fd == recovery_fd and not reused:
            replacement = os.open(preserved, os.O_RDWR)
            assert replacement == recovery_fd
            reused.append(replacement)
            raise OSError(errno.EIO, "close released its descriptor before failure")
        return result

    monkeypatch.setattr(J.os, "close", close)
    populate(journal, records)
    try:
        assert journal.close() is False and reused
        assert os.fstat(reused[0]).st_ino == preserved.stat().st_ino
        assert preserved.read_bytes() == b"unrelated descriptor contents"
        check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])
    finally:
        if reused:
            original_close(reused[0])


@pytest.mark.parametrize("edge", ["data_close", "recovery_close", "footer_recovery"])
def test_close_timeout_never_waits_for_filesystem_or_recovery_edge(tmp_path, monkeypatch, edge):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    entered, release, returned = threading.Event(), threading.Event(), threading.Event()
    original_close, original_recover = J.os.close, journal._discard_footer
    recovery_fd = journal._recovery_fd

    class CloseGate(FileProxy):
        def close(self):
            if edge == "data_close":
                entered.set()
                assert release.wait(3.0)
            result = super().close()
            if edge == "footer_recovery":
                raise OSError(errno.EIO, "controlled data close failure")
            return result

    def close(fd):
        if edge == "recovery_close" and fd == recovery_fd:
            entered.set()
            assert release.wait(3.0)
        return original_close(fd)

    def recover():
        if edge == "footer_recovery":
            entered.set()
            assert release.wait(3.0)
        return original_recover()

    journal._file = CloseGate(journal._file)
    journal._discard_footer = recover
    monkeypatch.setattr(J.os, "close", close)
    populate(journal, records)
    result, elapsed = [], []

    def stop():
        started = time.monotonic()
        result.append(journal.close(0.03))
        elapsed.append(time.monotonic() - started)
        returned.set()

    closer = threading.Thread(target=stop)
    closer.start()
    try:
        assert entered.wait(1.0)
        # The filesystem remains blocked throughout the entire caller wait.
        assert returned.wait(0.15), "close exceeded its budget waiting for storage"
        assert result == [False] and elapsed[0] < 0.15
        assert journal.status["healthy"] is False and not journal.status["closed"]
        # Bounded control cannot synchronously rewrite bytes while storage is
        # blocked. An unfinished footer is not accepted from raw bytes alone.
        current_path = journal.path if journal.path.exists() else journal.incomplete_path
        pending = R.parse_records(current_path.read_bytes())
        assert pending[-1]["kind"] == "footer"
        assert A.evaluate(pending)["verdict"] == "satisfied"
        if journal.incomplete_path.exists():
            with pytest.raises(R.io.EvidenceError):
                R.create_packet(journal.path.parent, tmp_path / "pending-packet")
        else:
            # A prematurely captured final name also lacks the close/status
            # boundary. The offline parser cannot infer that missing fact.
            assert R.inspect_payload({journal.path.name: current_path.read_bytes()})["verdict"] == "satisfied"
    finally:
        release.set()
        closer.join(1.0)
        journal._thread.join(2.0)
    assert not closer.is_alive() and not journal._thread.is_alive()
    assert journal.close() is False
    owned = [path for path in (journal.path, journal.incomplete_path) if path.exists()]
    check_recovered_artifacts(journal, tmp_path, owned)


def test_close_does_not_acquire_the_record_state_lock(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    populate(journal, records)
    journal._queue.join()
    result = []
    journal._lock.acquire()
    try:
        closer = threading.Thread(target=lambda: result.append(journal.close(0.03)))
        closer.start()
        closer.join(0.15)
        assert not closer.is_alive() and result == [False]
    finally:
        journal._lock.release()
    journal._thread.join(2.0)
    assert not journal._thread.is_alive()
    check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])


def test_closing_rejection_after_footer_write_is_not_hidden_as_a_late_call(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    entered, release = threading.Event(), threading.Event()

    class CloseGate(FileProxy):
        def close(self):
            entered.set()
            assert release.wait(3.0)
            return super().close()

    journal._file = CloseGate(journal._file)
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(1.0)))
    closer.start()
    try:
        assert entered.wait(1.0)
        assert journal.emit("gap", reason="closing interval rejection") is False
        assert journal.status["dropped"] == 1 and journal.status["late_calls"] == 0
    finally:
        release.set()
        closer.join(2.0)
        journal._thread.join(2.0)
    assert result == [False] and not journal._thread.is_alive()
    check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])


def test_filesystem_functions_run_outside_record_and_terminal_locks(tmp_path, monkeypatch):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    operations = []
    for name in ("stat", "open", "link", "unlink", "close", "ftruncate"):
        original = getattr(J.os, name)
        def guard(*args, _name=name, _original=original, **kwargs):
            assert not journal._lock.locked(), _name + " held record state lock"
            assert not journal._terminal_lock.locked(), _name + " held terminal lock"
            operations.append(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(J.os, name, guard)
    populate(journal, records)
    assert journal.close() is True
    assert {"stat", "link", "unlink", "close"} <= set(operations)
    assert A.evaluate(load(journal))["verdict"] == "satisfied"


def test_shared_health_changed_during_data_close_is_observed_before_commit(tmp_path):
    records = ordinary_records()
    state = {"healthy": True, "errors": 0}
    journal = J.AssignmentRecorder(tmp_path / "capture", health=SimpleNamespace(snapshot=lambda: dict(state)),
                                   **arguments(records))
    entered, release = threading.Event(), threading.Event()

    class CloseGate(FileProxy):
        def close(self):
            entered.set()
            assert release.wait(3.0)
            return super().close()

    journal._file = CloseGate(journal._file)
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(1.0)))
    closer.start()
    try:
        assert entered.wait(1.0)
        state.update(healthy=False, errors=2)
    finally:
        release.set()
        closer.join(2.0)
        journal._thread.join(2.0)
    assert result == [False] and not journal._thread.is_alive()
    check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])


def test_last_closing_rejection_is_compared_atomically_with_commit(tmp_path):
    records = ordinary_records()
    journal = J.AssignmentRecorder(tmp_path / "capture", **arguments(records))
    original = journal._check_final_accounting
    entered, release = threading.Event(), threading.Event()
    calls = []

    def final_accounting():
        original()
        calls.append(True)
        if len(calls) == 2:
            entered.set()
            assert release.wait(3.0)

    journal._check_final_accounting = final_accounting
    populate(journal, records)
    result = []
    closer = threading.Thread(target=lambda: result.append(journal.close(1.0)))
    closer.start()
    try:
        assert entered.wait(1.0)
        assert journal.emit("gap", reason="last closing rejection") is False
    finally:
        release.set()
        closer.join(2.0)
        journal._thread.join(2.0)
    assert result == [False] and not journal._thread.is_alive()
    assert journal.status["dropped"] == 1
    check_recovered_artifacts(journal, tmp_path, [journal.incomplete_path])
