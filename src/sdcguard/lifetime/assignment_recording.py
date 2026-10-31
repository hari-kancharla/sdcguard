"""Bounded JSONL storage for one finite assignment observation interval.

Owner: Hari Krishna Kancharla.
This module starts no engine and establishes no native association or verdict.
Producers and shared health must be quiescent before close. Late calls are outside
the sealed file; callers must inspect close/status before accepting a packet.
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading
import uuid

from . import assignment_obligations as contract
from .vllm_forward import _open_directory


_QUEUE_RECORDS = 1024
_QUEUE_BYTES = 8 * 1024 * 1024
_FOOTER_BYTES = 4096
_FOOTER_NODES = 128
_RESERVED = {"schema", "session", "process", "seq", "kind"}
_KINDS = {"init_begin", "configs_derived", "worker_init", "runner_ready", "init_return",
          "scheduler_ready", "retire", "mutation_begin", "mutation_end", "acquire",
          "unsupported_branch", "schedule", "worker_entry", "frame_begin", "assignments",
          "model_entry", "frame_end", "gap"}


class RecordingError(ValueError):
    pass


def _require(value, reason):
    if not value:
        raise RecordingError(reason)


def _encode(record):
    """Copy ordinary JSON within bounds before serializing the detached tree."""
    nodes, size = 0, 0

    def copy_value(value, depth):
        nonlocal nodes, size
        nodes += 1
        _require(nodes <= contract.MAX_RECORD_NODES and depth <= 32, "record node or depth limit")
        kind = type(value)
        if kind is dict:
            length = len(value)
            _require(length <= contract.MAX_RECORD_NODES, "record object limit")
            size += 2 + max(0, length - 1) + length
            result = {}
            for key, child in value.items():
                _require(type(key) is str, "record keys must be strings")
                result[copy_value(key, depth + 1)] = copy_value(child, depth + 1)
            _require(len(result) == length == len(value), "record changed during capture")
        elif kind is list:
            length = len(value)
            _require(length <= contract.MAX_RECORD_NODES, "record list limit")
            size += 2 + max(0, length - 1)
            result = [copy_value(value[index], depth + 1) for index in range(length)]
            _require(len(value) == length, "record changed during capture")
        elif kind is str:
            _require(len(value) <= contract.MAX_RECORD_BYTES, "record string limit")
            size += len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            result = value
        elif kind is int:
            _require(value.bit_length() <= 63, "record integer limit")
            size += len(str(value))
            result = value
        else:
            _require(value is None or kind is bool, "record contains unsupported JSON values")
            size += 4 if value is None or value is True else 5
            result = value
        _require(size + 1 <= contract.MAX_RECORD_BYTES, "record byte limit")
        return result

    copied = copy_value(record, 0)
    raw = json.dumps(copied, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    _require(len(raw) <= contract.MAX_RECORD_BYTES, "record byte limit")
    return raw, nodes, copied


class AssignmentRecorder:
    """Nonblocking admission to one bounded process-local writer."""

    def __init__(self, directory, *, adapter, source_sha256, profile, coverage, dependencies,
                 process_incarnation, health=None):
        self._pid = os.getpid()
        self._session = uuid.uuid4().hex
        self._process = process_incarnation
        _require(contract._id(self._process), "process incarnation must be UUID hex")
        _require(health is None or callable(getattr(health, "snapshot", None)), "shared health is unsupported")
        header = dict(schema=1, session=self._session, process=self._process, seq=0, kind="header",
                      pid=self._pid, adapter=adapter, source_sha256=source_sha256, profile=profile,
                      coverage=coverage, dependencies=dependencies)
        raw, nodes, copied = _encode(header)
        contract._Trace(copied)
        _require(len(raw) <= contract.MAX_TOTAL_BYTES - _FOOTER_BYTES
                 and nodes <= contract.MAX_TOTAL_NODES - _FOOTER_NODES
                 and contract.MAX_RECORDS >= 2, "header leaves no footer reserve")
        self._health = health
        self._lock = threading.Lock()
        self._terminal_lock = threading.Lock()
        self._terminal = None
        self._close_failed = False
        self._queue = queue.Queue(maxsize=_QUEUE_RECORDS)
        self._queued_bytes = 0
        self._total_bytes, self._total_nodes = len(raw), nodes
        self._next_seq = 1
        self._written = 1
        self._samples = self._acquisitions = 0
        self._dropped = self._errors = self._shared_errors = self._late_calls = 0
        self._closing = self._sealed = self._closed = self._failed = False
        self._footer_written = False
        self._published = False
        self._footer_start = None
        self._footer_counters = None
        self._body_bytes = len(raw)
        path, parent = _open_directory(directory, create=True)
        self._path = path / (self._session + ".jsonl")
        self._incomplete_path = path / (self._session + ".jsonl.incomplete")
        self._parent = parent
        try:
            try:
                os.stat(self._path.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError("assignment journal destination exists")
            fd = os.open(self._incomplete_path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
        except BaseException:
            os.close(parent)
            raise
        self._fd = fd
        self._recovery_fd = None
        info = os.fstat(fd)
        self._identity = (info.st_dev, info.st_ino)
        try:
            os.fchmod(fd, 0o600)
            self._file = os.fdopen(fd, "wb", buffering=0)
        except BaseException:
            os.close(fd)
            os.close(parent)
            raise
        try:
            self._write(raw)
            self._file.flush()
            # Keep the owned inode writable until publication and rollback settle.
            self._recovery_fd = os.dup(fd)
        except BaseException:
            self._file.close()
            os.close(parent)
            raise
        self._thread = threading.Thread(target=self._write_loop, daemon=True, name="assignment-journal-writer")
        try:
            self._thread.start()
        except BaseException:
            self._file.close()
            os.close(self._recovery_fd)
            os.close(parent)
            raise

    @property
    def session(self):
        return self._session

    @property
    def process(self):
        return self._process

    @property
    def path(self):
        return self._path

    @property
    def incomplete_path(self):
        return self._incomplete_path

    def _current(self):
        return os.getpid() == self._pid

    def _is_failed(self):
        return self._failed or self._close_failed

    def _error_count(self):
        return self._errors + self._shared_errors + int(self._close_failed)

    def _counters(self):
        return (self._written, self._samples, self._acquisitions, self._dropped, self._error_count())

    def _shared_health(self):
        if self._health is None:
            return
        try:
            snapshot = self._health.snapshot()
            valid = (type(snapshot) is dict and type(snapshot.get("healthy")) is bool
                     and type(snapshot.get("errors")) is int and 0 <= snapshot["errors"] < 2**63)
            errors = snapshot["errors"] if valid else 1
            if not valid or snapshot["healthy"] is not True or errors:
                with self._lock:
                    self._shared_errors = max(self._shared_errors, errors, 1)
        except Exception:
            with self._lock:
                self._shared_errors = max(self._shared_errors, 1)

    @property
    def status(self):
        if not self._current():
            return dict(healthy=False, closed=False, sealed=False, process_changed=True,
                        errors=None, dropped=None, late_calls=None)
        if not self._sealed:
            self._shared_health()
        with self._lock:
            errors = self._error_count()
            return dict(session=self._session, process=self._process, records=self._written,
                        accepted_records=self._next_seq, samples=self._samples, acquisitions=self._acquisitions,
                        dropped=self._dropped, errors=errors, late_calls=self._late_calls,
                        queued_bytes=self._queued_bytes, total_bytes=self._total_bytes,
                        total_nodes=self._total_nodes, closing=self._closing, sealed=self._sealed,
                        closed=self._closed, footer_written=self._footer_written,
                        published=self._published, incomplete_path=str(self._incomplete_path),
                        healthy=not (errors or self._dropped or self._is_failed()), process_changed=False)

    def emit(self, kind, /, **facts):
        if not self._current():
            return False
        with self._lock:
            if self._sealed:
                self._late_calls += 1
                return False
            if self._closing or self._is_failed():
                self._dropped += 1
                return False
            try:
                _require(type(kind) is str and kind in _KINDS and not facts.keys() & _RESERVED,
                         "record kind or envelope override is unsupported")
                record = dict(schema=1, session=self._session, process=self._process,
                              seq=self._next_seq, kind=kind, **facts)
                raw, nodes, _ = _encode(record)
                if (self._next_seq >= contract.MAX_RECORDS - 1
                        or self._queued_bytes + len(raw) > _QUEUE_BYTES
                        or self._total_bytes + len(raw) > contract.MAX_TOTAL_BYTES - _FOOTER_BYTES
                        or self._total_nodes + nodes > contract.MAX_TOTAL_NODES - _FOOTER_NODES):
                    self._dropped += 1
                    return False
                self._queue.put_nowait((raw, kind))
                self._next_seq += 1
                self._queued_bytes += len(raw)
                self._total_bytes += len(raw)
                self._total_nodes += nodes
                return True
            except queue.Full:
                self._dropped += 1
                return False
            except Exception:
                self._errors += 1
                self._dropped += 1
                return False

    def gap(self, reason):
        if not self._current():
            return
        with self._lock:
            if self._sealed:
                self._late_calls += 1
                return
            self._errors += 1
        self.emit("gap", reason=reason)

    def _write(self, raw):
        if self._file.write(raw) != len(raw):
            raise OSError("short assignment journal write")

    def _owned_name(self, name):
        try:
            info = os.stat(name, dir_fd=self._parent, follow_symlinks=False)
            return (info.st_dev, info.st_ino) == self._identity
        except OSError:
            return False

    def _discard_footer(self):
        """Invalidate only the retained inode, independently of name cleanup."""
        if self._footer_start is None:
            return True
        # A retained descriptor survives removal or replacement of either name.
        # The original descriptor is only a fallback and must pass the same check.
        for fd in (self._recovery_fd, self._fd):
            if fd is None:
                continue
            try:
                info = os.fstat(fd)
                if (info.st_dev, info.st_ino) == self._identity:
                    os.ftruncate(fd, self._footer_start)
                    return True
            except OSError:
                pass
        for name in (self._incomplete_path.name, self._path.name):
            try:
                fd = os.open(name, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=self._parent)
                try:
                    info = os.fstat(fd)
                    if (info.st_dev, info.st_ino) == self._identity:
                        os.ftruncate(fd, self._footer_start)
                        return True
                finally:
                    os.close(fd)
            except OSError:
                pass
        return False

    def _publish(self):
        _require(self._owned_name(self._incomplete_path.name), "unfinished journal identity changed")
        os.link(self._incomplete_path.name, self._path.name, src_dir_fd=self._parent,
                dst_dir_fd=self._parent, follow_symlinks=False)
        # Retain this state even if removal of the unfinished link fails.
        self._published = True
        _require(self._owned_name(self._incomplete_path.name), "unfinished journal identity changed")
        os.unlink(self._incomplete_path.name, dir_fd=self._parent)

    def _unpublish(self):
        # Footer invalidation must happen before any fallible naming operation.
        # All hard links share these bytes, including a final name left by EIO.
        self._discard_footer()
        try:
            if not self._owned_name(self._path.name):
                self._published = False
                return
            if not self._owned_name(self._incomplete_path.name):
                os.link(self._path.name, self._incomplete_path.name, src_dir_fd=self._parent,
                        dst_dir_fd=self._parent, follow_symlinks=False)
            _require(self._owned_name(self._incomplete_path.name), "unfinished journal identity changed")
            _require(self._owned_name(self._path.name), "published journal identity changed")
            os.unlink(self._path.name, dir_fd=self._parent)
            self._published = False
        finally:
            # Name cleanup failure must not skip either recovery descriptor/path.
            self._discard_footer()

    def _finish(self):
        self._shared_health()
        with self._lock:
            if self._is_failed():
                return
            counters = self._counters()
            footer = dict(schema=1, session=self._session, process=self._process, seq=self._next_seq,
                          kind="footer", records=self._written + 1, samples=self._samples,
                          acquisitions=self._acquisitions, dropped=self._dropped,
                          errors=counters[-1], healthy=not (self._dropped or counters[-1]))
            raw, nodes, _ = _encode(footer)
            _require(len(raw) <= _FOOTER_BYTES and nodes <= _FOOTER_NODES
                     and self._total_bytes + len(raw) <= contract.MAX_TOTAL_BYTES
                     and self._total_nodes + nodes <= contract.MAX_TOTAL_NODES,
                     "footer reserve was exceeded")
            self._footer_start = self._body_bytes
            self._footer_counters = counters
        self._file.flush()
        self._write(raw)
        self._file.flush()
        self._shared_health()
        with self._lock:
            _require(not self._is_failed() and counters == self._counters(),
                     "footer accounting changed during closure")
            self._total_bytes += len(raw)
            self._total_nodes += nodes

    def _write_loop(self):
        try:
            while True:
                try:
                    raw, kind = self._queue.get(timeout=0.02)
                except queue.Empty:
                    if self._closing:
                        break
                    continue
                try:
                    if not self._is_failed():
                        self._write(raw)
                        with self._lock:
                            self._written += 1
                            self._body_bytes += len(raw)
                            self._samples += kind == "frame_end"
                            self._acquisitions += kind == "acquire"
                    else:
                        with self._lock:
                            self._dropped += 1
                except Exception:
                    with self._lock:
                        self._failed = True
                        self._errors += 1
                        self._dropped += 1
                finally:
                    with self._lock:
                        self._queued_bytes -= len(raw)
                    self._queue.task_done()
            self._finish()
        except BaseException:
            with self._lock:
                self._failed = True
                self._errors += 1
            self._discard_footer()
        finally:
            try:
                self._file.close()
                _require(self._file.closed, "journal file did not close")
            except BaseException:
                with self._lock:
                    self._failed = True
                    self._errors += 1
            self._check_final_accounting()
            if not self._is_failed():
                try:
                    self._publish()
                except BaseException:
                    with self._lock:
                        self._failed = True
                        self._errors += 1
            if self._is_failed():
                self._recover()
            # No state lock spans a descriptor close or any other filesystem
            # operation. The close caller only waits on the join budget.
            recovery_fd, self._recovery_fd = self._recovery_fd, None
            try:
                # A failed close may already have released the number.
                # Never retry it against an unrelated reused descriptor.
                os.close(recovery_fd)
            except BaseException:
                with self._lock:
                    self._failed = True
                    self._errors += 1
            self._check_final_accounting()
            with self._lock:
                # Admission rejections can arrive after the last health sample.
                # Compare again in the same critical section as the commit.
                if (not self._is_failed() and self._footer_counters is not None
                        and self._footer_counters != self._counters()):
                    self._failed = True
                    self._errors += 1
                with self._terminal_lock:
                    if self._terminal is None and not self._is_failed():
                        self._closed = self._sealed = True
                        self._footer_written = self._published and self._footer_start is not None
                        self._written += self._footer_written
                        result = self._footer_written and not (self._dropped or self._error_count())
                        self._terminal = ("completed", bool(result))
                    committed = self._terminal is not None and self._terminal[0] == "completed"
            if not committed:
                # A timeout before publication still invalidates the footer.
                # Recovery does not depend on there being a final filename.
                self._recover()
                with self._lock:
                    self._closed = self._sealed = True
                    self._footer_written = False
                with self._terminal_lock:
                    if self._terminal is None:
                        self._terminal = ("failed", False)
            try:
                os.close(self._parent)
            except OSError:
                pass

    def _check_final_accounting(self):
        self._shared_health()
        with self._lock:
            if (not self._is_failed() and self._footer_counters is not None
                    and self._footer_counters != self._counters()):
                self._failed = True
                self._errors += 1

    def _recover(self):
        # These fallible operations run only in the writer, outside state locks.
        self._discard_footer()
        if self._published:
            try:
                self._unpublish()
            except BaseException:
                with self._lock:
                    self._errors += 1
        self._discard_footer()

    def close(self, timeout=1.0):
        if not self._current():
            return False
        valid_timeout = type(timeout) is int or type(timeout) is float and math.isfinite(timeout)
        seconds = max(0.0, min(timeout, 5.0)) if valid_timeout else 0.0
        # Admission observes this process-local flag while holding its own lock.
        # The closer never waits for that lock, caller serialization or disk I/O.
        self._closing = True
        with self._terminal_lock:
            if self._terminal is not None and (self._closed or self._terminal[0] == "completed"):
                return self._terminal[1]
        try:
            self._thread.join(timeout=seconds)
        except Exception:
            pass
        # Only short terminal arbitration uses this separate lock. It contains
        # no filesystem work, health callback, serialization or state-lock wait.
        with self._terminal_lock:
            if self._terminal is None:
                self._close_failed = True
                self._terminal = ("timed_out", False)
            return self._terminal[1]
