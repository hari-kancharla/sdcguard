"""Ordinary stdlib process supervision controls. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

from copy import deepcopy
import errno
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(BUNDLE))

from sdcguard.native_operator import cleanup as owner_cleanup
from sdcguard.native_operator import supervisor as owner_supervisor

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd lifecycle controls")


def _optional_test_handle(row, process_table):
    """Retain a fallback handle if still present; never establish exit proof."""
    current = process_table().get(row["pid"])
    if current is None or current["start"] != row["start"]:
        return None
    try:
        descriptor = os.pidfd_open(row["pid"], 0)
    except ProcessLookupError as error:
        if error.errno != errno.ESRCH or row["pid"] in process_table():
            raise
        return None
    try:
        later = process_table().get(row["pid"])
    except BaseException:
        os.close(descriptor)
        raise
    if later is None or later["start"] != row["start"]:
        os.close(descriptor)
        return None
    return descriptor


# This isolated owner uses only ordinary Python children. Its direct child may
# exit normally before cleanup. No engine, server, model or accelerator is used.
OWNER_HARNESS = "import errno, os\n" + inspect.getsource(_optional_test_handle) + r'''
import json, pathlib, signal, subprocess, sys, time, traceback
sys.path.insert(0, sys.argv[1])
from sdcguard.native_operator import cleanup
from sdcguard.native_operator import supervisor
from sdcguard.native_operator.common import checkpoint

out = pathlib.Path(sys.argv[2])
case = sys.argv[3]
out.mkdir(exist_ok=True)
references, retained, identities = [], {}, []
streams = []
result = {"case":case,"complete":False,"scope":"ordinary stdlib children only"}

def remember(row):
    key = (row["pid"], row["start"])
    if key in retained:
        return
    descriptor = _optional_test_handle(row, cleanup.processes)
    if descriptor is None:
        return
    retained[key] = descriptor
    identities.append(dict(pid=key[0],start=key[1],group=row["group"],session=row["session"]))
    checkpoint(out / "RETAINED_IDENTITIES.json", identities)

def fallback():
    facts = []
    for key, descriptor in reversed(list(retained.items())):
        try:
            state = cleanup.pidfd_state(descriptor)
            action = dict(pid=key[0],start=key[1],before=state)
            if state["status"] != "exit_observed":
                signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
                action["signal"] = int(signal.SIGKILL)
            facts.append(action)
        except ProcessLookupError:
            facts.append(dict(pid=key[0],start=key[1],already_absent=True))
        except BaseException as error:
            facts.append(dict(pid=key[0],start=key[1],failure=repr(error)))
    until = time.monotonic() + 3
    while time.monotonic() < until:
        live = []
        for key, descriptor in retained.items():
            try:
                if cleanup.pidfd_state(descriptor)["status"] != "exit_observed":
                    live.append(key)
                else:
                    try:
                        os.waitid(os.P_PIDFD, descriptor, os.WEXITED | os.WNOHANG)
                    except (ChildProcessError, ProcessLookupError):
                        pass
            except BaseException:
                live.append(key)
        if not live:
            break
        time.sleep(.02)
    checkpoint(out / "FALLBACK.json", dict(actions=facts,live_incarnations=live))

try:
    cleanup.require_pidfds()
    cleanup.subreaper()
    before = cleanup.baseline()
    registry = out / "PROCESS_GROUPS.json"
    codes = [0, 7] if case == "multiple" else [7 if case == "nonzero" else 0]
    for index, code in enumerate(codes):
        directory = out / ("anchor-" + str(index))
        child_file = out / ("orphan-" + str(index) + ".json")
        if case == "orphan":
            native = """
import json, pathlib, subprocess, sys
child = subprocess.Popen([sys.executable,"-u","-c","import time; time.sleep(40)"])
text = pathlib.Path("/proc/"+str(child.pid)+"/stat").read_text()
fields = text[text.rfind(")")+2:].split()
pathlib.Path(sys.argv[1]).write_text(json.dumps(dict(pid=child.pid,start=int(fields[19]),
    group=int(fields[2]),session=int(fields[3]))))
"""
            command = [sys.executable,"-u","-c",native,str(child_file)]
        else:
            command = [sys.executable,"-u","-c","import sys;sys.exit("+str(code)+")"]
        stream = (out / ("anchor-"+str(index)+".log")).open("xb")
        streams.append(stream)
        process = supervisor.launch(command,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),
            stdout=stream,directory=directory,label="ordinary-"+str(index))
        references.append(process)
        row = cleanup.processes()[process.pid]
        remember(dict(pid=process.pid,**row))
        cleanup.register(registry,process,"ordinary-"+str(index))
        started = supervisor.start(process,timeout=5)
        remember(started["child"])
        if case == "orphan":
            until = time.monotonic()+5
            while not child_file.exists() and time.monotonic()<until:
                time.sleep(.01)
            assert child_file.is_file(), "ordinary orphan identity was not retained"
            remember(json.loads(child_file.read_text()))
        until = time.monotonic()+5
        status = None
        while status is None and time.monotonic()<until:
            status = supervisor.poll_engine(process)
            time.sleep(.01)
        assert status == code, (status,code)
        assert process.returncode is None, "native status was confused with anchor status"
        live = cleanup.processes().get(process.pid)
        assert live is not None and live["start"] == row["start"] and live["state"] not in ("Z","X")
        if case == "orphan":
            orphan = json.loads(child_file.read_text())
            adopted = cleanup.processes().get(orphan["pid"])
            assert adopted is not None and adopted["start"] == orphan["start"]
            assert adopted["ppid"] == process.pid and adopted["state"] not in ("Z","X")
            result["orphan_before_cleanup"] = dict(pid=orphan["pid"],**adopted)
    assert all(cleanup.processes().get(p.pid,{}).get("state") not in (None,"Z","X") for p in references)
    # Numeric PID/group signalling is forbidden even in this detached control.
    def unsafe_signal(*args,**kwargs):
        raise AssertionError("numeric process signalling was reached")
    os.kill = unsafe_signal
    os.killpg = unsafe_signal
    stopped = cleanup.cleanup(leaders=references,scope=os.getpid(),baseline=before,
        registry=registry,grace=.1,kill_wait=3)
    checkpoint(out / "CLEANUP.json",stopped)
    assert stopped["complete"] is True, stopped
    assert stopped["remaining_pids"] == []
    assert stopped["missing_terminal_evidence"] == []
    assert stopped["roots_without_exact_exit_status"] == []
    assert len(stopped["supervisor_receipts"]) == len(references)
    assert all(row["complete"] is True for row in stopped["supervisor_receipts"])
    assert all(supervisor.closed(process) for process in references)
    exits = [json.loads((pathlib.Path(p._owner_supervision["directory"])/"ENGINE_EXIT.json").read_text())
             for p in references]
    assert [row["returncode"] for row in exits] == codes
    assert all(process.returncode == 0 for process in references)
    final = cleanup.processes()
    assert not [row for row in identities if row["pid"] in final and final[row["pid"]]["start"]==row["start"]]
    result.update(complete=True,cleanup=stopped,child_exit_codes=codes,
        anchor_exit_codes=[p.returncode for p in references],retained_identities=identities)
except BaseException as error:
    result["failure"] = dict(kind=type(error).__name__,detail=str(error),
                             traceback=traceback.format_exc())
finally:
    fallback()
    for descriptor in retained.values():
        os.close(descriptor)
    for stream in streams:
        stream.close()
    checkpoint(out / "RESULT.json",result)
sys.exit(0 if result["complete"] else 1)
'''


def _stop_retained_failure_handles(directory, owner_pidfd):
    """Retain exact fallback handles; never signal an inferred process number."""
    opened, facts = [], []
    path = Path(directory) / "RETAINED_IDENTITIES.json"
    rows = json.loads(path.read_text()) if path.is_file() else []
    for row in rows:
        descriptor = _optional_test_handle(row, owner_cleanup.processes)
        if descriptor is None:
            continue
        opened.append((row, descriptor))
    # Descendants first, owner last. Every signal has its own retained pidfd.
    for row, descriptor in reversed(opened):
        try:
            if owner_cleanup.pidfd_state(descriptor)["status"] != "exit_observed":
                signal.pidfd_send_signal(descriptor, signal.SIGKILL, None, 0)
            facts.append(dict(pid=row["pid"], start=row["start"], signal_via="retained_pidfd"))
        except ProcessLookupError:
            pass
    if owner_cleanup.pidfd_state(owner_pidfd)["status"] != "exit_observed":
        signal.pidfd_send_signal(owner_pidfd, signal.SIGKILL, None, 0)
    until = time.monotonic() + 3
    while time.monotonic() < until:
        if all(owner_cleanup.pidfd_state(fd)["status"] == "exit_observed" for _, fd in opened):
            break
        time.sleep(.02)
    for row, descriptor in opened:
        facts.append(dict(pid=row["pid"], start=row["start"], after=owner_cleanup.pidfd_state(descriptor)))
        os.close(descriptor)
    (Path(directory) / "OUTER_FALLBACK.json").write_text(json.dumps(facts, indent=2) + "\n")


def test_optional_fallback_handle_accepts_only_confirmed_disappearance(monkeypatch):
    row = dict(pid=123456, start=20)
    snapshots = iter([{row["pid"]: dict(start=20)}, {}])

    def reaped_after_snapshot(pid, flags):
        assert (pid, flags) == (row["pid"], 0)
        raise ProcessLookupError(errno.ESRCH, "ordinary child was reaped")

    monkeypatch.setattr(os, "pidfd_open", reaped_after_snapshot)
    assert _optional_test_handle(row, lambda: next(snapshots)) is None
    assert list(snapshots) == []


@pytest.mark.parametrize("fresh_start,error_number", [(20, errno.ESRCH), (21, errno.ESRCH),
                                                    (None, errno.EPERM)])
def test_optional_fallback_handle_does_not_hide_ambiguous_or_unrelated_errors(
        monkeypatch, fresh_start, error_number):
    row = dict(pid=123456, start=20)
    fresh = {} if fresh_start is None else {row["pid"]: dict(start=fresh_start)}
    snapshots = iter([{row["pid"]: dict(start=20)}, fresh])
    failure = ProcessLookupError(error_number, "synthetic handle error")

    def failed_open(pid, flags):
        raise failure

    monkeypatch.setattr(os, "pidfd_open", failed_open)
    with pytest.raises(ProcessLookupError) as caught:
        _optional_test_handle(row, lambda: next(snapshots))
    assert caught.value is failure


@pytest.mark.parametrize("case,expected", [("zero", [0]), ("nonzero", [7]), ("orphan", [0]), ("multiple", [0, 7])])
def test_real_ordinary_early_child_exit_keeps_anchor_until_complete_cleanup(tmp_path, case, expected):
    owner_cleanup.require_pidfds()
    directory = tmp_path / case
    directory.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (directory / "OWNER.log").open("xb") as stream:
        process = subprocess.Popen([sys.executable, "-I", "-u", "-c", OWNER_HARNESS,
            str(BUNDLE), str(directory), case], stdout=stream, stderr=subprocess.STDOUT, env=env)
        descriptor = os.pidfd_open(process.pid, 0)
        try:
            try:
                code = process.wait(timeout=18)
            except subprocess.TimeoutExpired:
                _stop_retained_failure_handles(directory, descriptor)
                process.wait(timeout=5)
                pytest.fail("ordinary lifecycle owner exceeded its bound; retained diagnostics: " + str(directory))
            if code != 0:
                _stop_retained_failure_handles(directory, descriptor)
            result_path = directory / "RESULT.json"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {}
            assert code == 0 and result.get("complete") is True, (result, str(directory))
            assert result["child_exit_codes"] == expected
            assert result["anchor_exit_codes"] == [0] * len(expected)
            if case == "orphan":
                orphan = result["orphan_before_cleanup"]
                assert orphan["pid"] in result["cleanup"]["observed_pids"]
                assert any(row["pid"] == orphan["pid"] for row in result["cleanup"]["terminal_handles"])
            assert result["cleanup"]["complete"] is True
            assert result["cleanup"]["errors"] == []
            assert json.loads((directory / "FALLBACK.json").read_text())["live_incarnations"] == []
        finally:
            if process.returncode is None:
                _stop_retained_failure_handles(directory, descriptor)
                process.wait(timeout=5)
            os.close(descriptor)


def _row(ppid, group, start):
    return dict(state="S", ppid=ppid, group=group, session=group, start=start,
                cmdline=["ordinary-control"], cmdline_status="observed")


def test_failed_registration_cancels_only_the_unstarted_owned_anchor(tmp_path):
    owner_cleanup.require_pidfds()
    directory=tmp_path/'anchor';marker=tmp_path/'must-not-start'
    registry=tmp_path/'registry.json'
    registry.with_suffix('.pending').write_text('synthetic existing pending write')
    with (tmp_path/'anchor.log').open('wb') as stream:
        process=owner_supervisor.launch([sys.executable,'-c','from pathlib import Path;Path('+repr(str(marker))+').touch()'],
            env=dict(os.environ),stdout=stream,directory=directory,label='ordinary-cancellation')
        try:
            with pytest.raises(FileExistsError):owner_cleanup.register(registry,process,'ordinary-cancellation')
            result=owner_supervisor.cancel_unstarted(process)
            assert result['complete'] and result['native_start_requested'] is False
            assert result['returncode']==0 and process.returncode==0
            assert not marker.exists() and not (directory/'ENGINE_STARTED.json').exists()
            assert json.loads((directory/'ANCHOR_CANCELLED.json').read_text())['native_started'] is False
        finally:
            if process.returncode is None:owner_supervisor.cancel_unstarted(process)


def test_isolated_anchor_ignores_a_shadow_package_in_cwd(tmp_path,monkeypatch):
    shadow=tmp_path/'sdcguard';shadow.mkdir()
    (shadow/'__init__.py').write_text('raise RuntimeError("shadow package loaded")')
    monkeypatch.chdir(tmp_path)
    with (tmp_path/'anchor.log').open('wb') as stream:
        process=owner_supervisor.launch([sys.executable,'-c','pass'],env=dict(os.environ,PYTHONPATH=str(tmp_path)),
            stdout=stream,directory=tmp_path/'anchor',label='ordinary-isolation')
        result=owner_supervisor.cancel_unstarted(process)
    assert result['returncode']==0


def _inert_cleanup(tmp_path, monkeypatch, *, replacement=False, replace_after_open=False,
                   unknown_member=False, denied=None):
    """Only fake process-table rows and harmless /dev/null descriptors are used."""
    own, group = os.getpid(), os.getpgrp()
    leader = own + 100000
    table = {own: _row(os.getppid(), group, 11), leader: _row(own, leader, 21)}
    if replacement:
        table[leader]["start"] = 22
    if unknown_member:
        table[leader + 1] = _row(own, leader + 1, 31)
    registry = tmp_path / "PROCESS_GROUPS.json"
    registry.write_text(json.dumps(dict(schema=2, scope=dict(pid=own, start=11), groups=[
        dict(pid=leader, start=21, group=leader, session=leader, label="ordinary")])) + "\n")
    descriptors, sends = {}, []
    real_open = os.open

    def open_handle(pid, flags=0):
        if denied == "open":
            raise PermissionError(errno.EPERM, "synthetic pidfd open denied")
        descriptor = real_open(os.devnull, os.O_RDONLY)
        descriptors[descriptor] = pid
        if replace_after_open:
            table[pid]["start"] = 22
        return descriptor

    def send_handle(descriptor, selected, info=None, flags=0):
        sends.append((descriptors[descriptor], int(selected)))
        if denied == "signal":
            raise PermissionError(errno.EPERM, "synthetic pidfd signal denied")

    def forbidden(*args, **kwargs):
        pytest.fail("numeric process signal fallback reached")

    monkeypatch.setattr(owner_cleanup, "require_pidfds", lambda: None)
    monkeypatch.setattr(owner_cleanup, "processes", lambda: deepcopy(table))
    monkeypatch.setattr(owner_cleanup.os, "pidfd_open", open_handle)
    monkeypatch.setattr(owner_cleanup.os, "kill", forbidden)
    monkeypatch.setattr(owner_cleanup.os, "killpg", forbidden)
    monkeypatch.setattr(owner_cleanup.signal, "pidfd_send_signal", send_handle)
    monkeypatch.setattr(owner_cleanup.os, "waitid", lambda *args: None)
    monkeypatch.setattr(owner_cleanup, "pidfd_state", lambda descriptor, timeout=0:
        dict(status="live_or_unobserved", flags=0, events=[], fully_reaped=False))
    result = owner_cleanup.cleanup(scope=own, baseline={group}, registry=registry, grace=0, kill_wait=0)
    return result, sends, leader


@pytest.mark.parametrize("denied", ["open", "signal"])
def test_denied_exact_handle_or_signal_holds_cleanup_without_numeric_fallback(tmp_path, monkeypatch, denied):
    result, sends, _ = _inert_cleanup(tmp_path, monkeypatch, denied=denied)
    assert result["complete"] is False
    assert any("PermissionError" in error for error in result["errors"])
    if denied == "open":
        assert sends == []
    else:
        assert sends and all(row["outcome"] == "failed" for row in result["signal_actions"])


@pytest.mark.parametrize("during_open", [False, True])
def test_pid_replacement_is_unproven_and_never_receives_a_signal(tmp_path, monkeypatch, during_open):
    result, sends, _ = _inert_cleanup(tmp_path, monkeypatch,
        replacement=not during_open, replace_after_open=during_open)
    assert result["complete"] is False and sends == []
    assert any("replaced" in error or "changed while opening" in error for error in result["errors"])


def test_unproven_descendant_holds_cleanup_and_is_not_signalled(tmp_path, monkeypatch):
    result, sends, leader = _inert_cleanup(tmp_path, monkeypatch, unknown_member=True)
    assert result["complete"] is False
    assert any("unproven process membership" in error for error in result["errors"])
    assert sends and all(pid == leader for pid, _ in sends)


def test_registration_denial_prevents_native_start_and_does_not_write_registry(tmp_path, monkeypatch):
    own, leader = os.getpid(), os.getpid() + 100000
    table = {own: _row(os.getppid(), os.getpgrp(), 11), leader: _row(own, leader, 21)}
    process = SimpleNamespace(pid=leader, _owner_pidfd=None)
    monkeypatch.setattr(owner_cleanup, "processes", lambda: deepcopy(table))
    def denied(*args):
        raise PermissionError(errno.EPERM, "synthetic registration denied")
    monkeypatch.setattr(owner_cleanup.os, "pidfd_open", denied)
    registry = tmp_path / "PROCESS_GROUPS.json"
    with pytest.raises(PermissionError):
        owner_cleanup.register(registry, process, "ordinary")
    assert not registry.exists() and process._owner_pidfd is None
    with pytest.raises(ValueError, match="retained parent pidfd"):
        owner_supervisor.start(process, timeout=.01)

