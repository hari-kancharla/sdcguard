"""Keep a bound session anchor until cleanup. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from .common import checkpoint, digest, encoded, regular_bytes, isolated_module

SCHEMA = "sdcguard.owner-session-anchor.v1"


def _identity(pid):
    from .cleanup import processes
    row = processes().get(pid)
    if row is None:
        raise ValueError("session process identity is unavailable")
    return dict(pid=pid, **row)


def launch(command, *, env, stdout, directory, label):
    """Return the waiting anchor so its caller retains it before registration."""
    from .cleanup import require_pidfds
    require_pidfds()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    spec = dict(schema=SCHEMA, owner="Hari Krishna Kancharla", label=label,
        creator=_identity(os.getpid()), command=list(command))
    raw = encoded(spec)
    checkpoint(directory / "SUPERVISOR_SPEC.json", spec)
    process = subprocess.Popen(isolated_module("supervisor", "--anchor", str(directory)),
        env=env, stdin=subprocess.PIPE, stdout=stdout, cwd=str(directory),
        stderr=subprocess.STDOUT, start_new_session=True)
    process._owner_supervision = dict(directory=str(directory), spec_sha256=digest(raw),
        label=label, child=None, released=False, start_requested=False)
    return process


def _read(process, name):
    meta = process._owner_supervision
    path = Path(meta["directory"]) / name
    if not path.exists():
        return None
    value = json.loads(regular_bytes(path, 128 * 1024))
    identity = getattr(process, "_owner_process_identity", None)
    if (type(value) is not dict or value.get("schema") != SCHEMA
            or value.get("spec_sha256") != meta["spec_sha256"]
            or type(identity) is not dict or value.get("anchor", {}).get("pid") != process.pid
            or value["anchor"].get("start") != identity.get("start")):
        raise ValueError("session-anchor receipt identity differs")
    return value


def _control(process, command):
    if command not in ("start", "release", "cancel") or process.stdin is None or process.stdin.closed:
        raise ValueError("session-anchor control channel is unavailable")
    process.stdin.write((command + "\n").encode())
    process.stdin.flush()


def start(process, *, timeout=10):
    """Start the native command only after the parent retained the anchor pidfd."""
    if type(getattr(process, "_owner_pidfd", None)) is not int:
        raise ValueError("anchor must have a retained parent pidfd before native startup")
    until = time.monotonic() + timeout
    started = False
    while time.monotonic() < until:
        error = _read(process, "SUPERVISOR_ERROR.json")
        if error:
            raise ValueError("session anchor failed: " + str(error.get("failure"))[:512])
        ready = _read(process, "ANCHOR_READY.json")
        if ready and not started:
            process._owner_supervision["start_requested"] = True
            _control(process, "start")
            started = True
        child = _read(process, "ENGINE_STARTED.json")
        if child:
            row = child.get("child")
            if (type(row) is not dict or type(row.get("pid")) is not int or row["pid"] <= 1
                    or type(row.get("start")) is not int or row["start"] <= 0
                    or row.get("ppid") != process.pid or row.get("group") != process.pid
                    or row.get("session") != process.pid):
                raise ValueError("engine child is not bound to its session anchor")
            process._owner_supervision["child"] = row
            return child
        if process.poll() is not None:
            raise ValueError("session anchor exited before native startup was recorded")
        time.sleep(.02)
    raise TimeoutError("session-anchor startup receipt deadline reached")


def poll_engine(process):
    """Read actual bound child status. Do not reap the persistent anchor."""
    if not hasattr(process, "_owner_supervision"):
        return process.poll()
    error = _read(process, "SUPERVISOR_ERROR.json")
    if error:
        raise ValueError("session anchor failed: " + str(error.get("failure"))[:512])
    exited = _read(process, "ENGINE_EXIT.json")
    if exited:
        if (exited.get("child") != process._owner_supervision.get("child")
                or type(exited.get("returncode")) is not int):
            raise ValueError("engine exit is not bound to its recorded child")
        return exited["returncode"]
    if process.poll() is not None:
        raise ValueError("session anchor exited without a bound engine status")
    return None


def release(process):
    if not process._owner_supervision["released"]:
        _control(process, "release")
        process._owner_supervision["released"] = True


def closed(process):
    value = _read(process, "ANCHOR_CLOSED.json")
    exited = _read(process, "ENGINE_EXIT.json")
    return (type(value) is dict and value.get("complete") is True and value.get("errors") == []
        and type(exited) is dict and type(exited.get("returncode")) is int
        and exited.get("child") == process._owner_supervision.get("child")
        and value.get("child_exit") == exited)


def cancel_unstarted(process):
    """Cancel a waiting direct child without signalling a numeric process ID."""
    meta = process._owner_supervision
    if meta.get("start_requested") is not False or meta.get("child") is not None:
        raise ValueError("cannot use unstarted cancellation after a native start request")
    try:
        _control(process, "cancel")
    except BrokenPipeError:
        pass
    code = process.wait(timeout=10)
    if process.stdin is not None:
        process.stdin.close()
    descriptor = getattr(process, "_owner_pidfd", None)
    if type(descriptor) is int:
        os.close(descriptor)
        process._owner_pidfd = None
    return dict(complete=True, native_start_requested=False, pid=process.pid,
                returncode=code, basis="retained direct-child wait before any native start command")


def anchor(directory):
    """Own one ordinary child, forward TERM and remain until parent release."""
    from .cleanup import require_pidfds, subreaper, pidfd_state
    directory = Path(directory)
    raw = regular_bytes(directory / "SUPERVISOR_SPEC.json", 128 * 1024)
    spec = json.loads(raw)
    identity = _identity(os.getpid())
    base = dict(schema=SCHEMA, owner="Hari Krishna Kancharla", anchor=identity, spec_sha256=digest(raw))
    child = None
    descriptor = None
    stopped = False
    requested = False
    released = False
    errors = []
    exit_receipt = None
    pending = b""
    until = time.monotonic() + 2100
    def term(*_):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, term)
    signal.signal(signal.SIGINT, term)
    try:
        require_pidfds()
        subreaper()
        if (spec.get("schema") != SCHEMA or spec.get("owner") != "Hari Krishna Kancharla"
                or spec.get("creator", {}).get("pid") != os.getppid()
                or _identity(os.getppid())["start"] != spec["creator"].get("start")
                or identity["group"] != identity["pid"] or identity["session"] != identity["pid"]
                or type(spec.get("command")) is not list or not spec["command"]
                or any(type(item) is not str for item in spec["command"])):
            raise ValueError("session anchor creator, command or session identity differs")
        checkpoint(directory / "ANCHOR_READY.json", base)
        poller = select.poll()
        poller.register(sys.stdin.fileno(), select.POLLIN | select.POLLHUP)
        while time.monotonic() < until:
            if child is not None and exit_receipt is None:
                code = child.poll()
                if code is not None:
                    exit_receipt = dict(base, child=child_identity, returncode=code,
                        basis="status from the anchor's retained direct-child Popen")
                    checkpoint(directory / "ENGINE_EXIT.json", exit_receipt)
            if requested and child is not None and exit_receipt is None and not stopped:
                if pidfd_state(descriptor)["status"] == "live_or_unobserved":
                    signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
                stopped = True
            if released and child is not None and exit_receipt is not None:
                checkpoint(directory / "ANCHOR_CLOSED.json", dict(base, complete=True, errors=errors,
                    child_exit=exit_receipt, basis="parent released anchor after observing its descendants stopped"))
                return 0
            for _, flags in poller.poll(20):
                part = os.read(sys.stdin.fileno(), 256)
                if not part:
                    raise ValueError("session anchor control channel closed before release")
                pending += part
                if len(pending) > 1024:
                    raise ValueError("session anchor control input exceeded its bound")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line == b"start" and child is None:
                        child = subprocess.Popen(spec["command"], stdin=subprocess.DEVNULL)
                        child_identity = _identity(child.pid)
                        descriptor = os.pidfd_open(child.pid, 0)
                        if _identity(child.pid)["start"] != child_identity["start"]:
                            raise ValueError("native child changed while retaining its handle")
                        checkpoint(directory / "ENGINE_STARTED.json", dict(base, child=child_identity))
                    elif line == b"cancel" and child is None:
                        checkpoint(directory / "ANCHOR_CANCELLED.json", dict(base, complete=True, native_started=False))
                        return 0
                    elif line == b"release":
                        released = True
                    else:
                        raise ValueError("unexpected session anchor control command")
        raise TimeoutError("session anchor maximum lifetime reached")
    except BaseException as error:
        errors.append(dict(kind=type(error).__name__, detail=str(error)[:1024]))
        checkpoint(directory / "SUPERVISOR_ERROR.json", dict(base, failure=errors[-1], complete=False))
        if child is None:
            return 2
        # The owner still holds the anchor and member handles. Do not signal an
        # unobserved PID or report a child exit that was not actually waited.
        while time.monotonic() < until:
            time.sleep(.05)
        return 2
    finally:
        if descriptor is not None:
            os.close(descriptor)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--anchor":
        raise SystemExit("one owned session-anchor directory is required")
    raise SystemExit(anchor(sys.argv[2]))
