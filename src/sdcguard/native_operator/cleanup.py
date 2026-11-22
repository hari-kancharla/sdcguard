"""Stop recorded Linux process incarnations. Owner: Hari Krishna Kancharla."""
from __future__ import annotations

import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import select
import time


def subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'owner process supervision could not be enabled')


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text(); fields = text[text.rfind(')') + 2:].split()
            result[int(path.parent.name)] = dict(state=fields[0], ppid=int(fields[1]),
                group=int(fields[2]), session=int(fields[3]), start=int(fields[19]))
            row = result[int(path.parent.name)]
            try:
                with (path.parent / 'cmdline').open('rb') as stream:raw=stream.read(4097)
                row.update(cmdline=[part.decode('utf-8','replace') for part in raw[:4096].split(b'\0') if part],
                    cmdline_status='truncated' if len(raw)>4096 else 'observed')
            except OSError as error:
                row.update(cmdline=None,cmdline_status='unobservable',cmdline_errno=error.errno)
        except (FileNotFoundError, ProcessLookupError):
            pass
    if not result:
        raise OSError('no Linux process table available for cleanup verification')
    return result


def descendants(table, root):
    found = {root}
    while True:
        more = {pid for pid, row in table.items() if row['ppid'] in found}
        if more <= found:
            return found - {root}
        found |= more


def require_pidfds():
    """Refuse an unsupported platform before an owner process can be launched."""
    if not (callable(getattr(os, 'pidfd_open', None))
            and callable(getattr(signal, 'pidfd_send_signal', None))
            and callable(getattr(os, 'waitid', None)) and hasattr(os, 'P_PIDFD')
            and callable(getattr(select, 'poll', None))
            and all(type(getattr(select,key,None)) is int and getattr(select,key)>0 for key in ('POLLIN','POLLHUP'))):
        raise OSError('exact process-incarnation cleanup APIs are unavailable')
    descriptor = os.pidfd_open(os.getpid(), 0)
    os.close(descriptor)


def pidfd_state(descriptor, timeout=0):
    """Read only this retained pidfd. Readiness does not reveal exit status."""
    if type(descriptor) is not int or descriptor < 0 or not 0 <= timeout <= 1000:
        raise ValueError('bounded exact process polling arguments required')
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    rows = poller.poll(timeout)
    if not rows:
        return dict(status='live_or_unobserved', flags=0, events=[], fully_reaped=False)
    if len(rows) != 1 or rows[0][0] != descriptor:
        raise ValueError('exact process poll returned an unmatched handle')
    flags = rows[0][1]
    if (type(flags) is not int or flags <= 0 or flags & ~(select.POLLIN | select.POLLHUP)):
        raise OSError(errno.EINVAL,'exact process poll returned unsupported or invalid events: ' + str(flags))
    return dict(status='exit_observed', flags=flags,
        events=[name for name in ('POLLIN','POLLHUP') if flags & getattr(select,name)],
        fully_reaped=bool(flags & select.POLLHUP))


def baseline(scope=None):
    require_pidfds()
    scope = os.getpid() if scope is None else scope
    if scope != os.getpid():
        raise ValueError('cleanup scope must be the current owner process')
    table = processes()
    return {os.getpgrp()} | {table[pid]['group'] for pid in descendants(table, scope)}


def register(path, process, label):
    """Retain the live leader identity, not just a reusable process number."""
    table = processes(); creator = os.getpid(); row = table.get(process.pid)
    if (row is None or row['ppid'] != creator or row['group'] != process.pid
            or row['session'] != process.pid or table.get(creator) is None):
        raise ValueError('new-session leader identity is unavailable')
    scope = dict(pid=creator, start=table[creator]['start'])
    path = Path(path)
    prior = json.loads(path.read_text()) if path.is_file() else dict(schema=2, scope=scope, groups=[])
    if prior.keys() != {'schema', 'scope', 'groups'} or prior['schema'] != 2 or prior['scope'] != scope:
        raise ValueError('process registry creator differs')
    if any(item['pid'] == process.pid for item in prior['groups']):
        raise ValueError('a process leader is already registered')
    descriptor = os.pidfd_open(process.pid, 0)
    later = processes().get(process.pid)
    if later is None or later['start'] != row['start']:
        os.close(descriptor)
        raise ValueError('session leader changed while retaining its early handle')
    process._owner_process_identity = dict(pid=process.pid, start=row['start'], scope=scope)
    process._owner_pidfd = descriptor
    prior['groups'].append(dict(pid=process.pid, start=row['start'], group=row['group'],
                               session=row['session'], label=label))
    temporary = path.with_suffix('.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(prior, indent=2) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def cleanup(process=None, *, leaders=(), scope=None, baseline=(), registry=None, grace=20, kill_wait=8):
    """Only pidfds receive signals. Ambiguous membership withholds completion."""
    own = os.getpid(); protected = set(baseline) | {os.getpgrp()}
    errors, roots, handles, observed, reaped = [], {}, {}, {}, {}
    terminal, sent, actions, wait_errors = {}, set(), [], []
    members, poll_events, last_poll = {}, [], {}
    vanished_before_pidfd = {}
    leader_waited = process is None or type(getattr(process, 'returncode', None)) is int
    remaining = None
    scope = own if scope is None else scope
    references = list(leaders) + ([process] if process is not None else [])
    bound = {item.pid:item for item in references}
    released = {}

    def problem(text):
        if text not in errors: errors.append(text)

    try:
        require_pidfds()
        initial = processes()
        if scope != own or own not in initial:
            raise ValueError('cleanup scope identity is unavailable')
        scope_identity = dict(pid=own, start=initial[own]['start'])
        if registry is None or not Path(registry).is_file():
            raise ValueError('incarnation-bound process registry is missing')
        value = json.loads(Path(registry).read_text())
        if (type(value) is not dict or value.keys() != {'schema', 'scope', 'groups'}
                or value['schema'] != 2 or value['scope'] != scope_identity
                or type(value['groups']) is not list or not value['groups'] or len(value['groups']) > 64):
            raise ValueError('process registry identity or bounds differ')
        for row in value['groups']:
            if (type(row) is not dict or row.keys() != {'pid', 'start', 'group', 'session', 'label'}
                    or any(type(row[key]) is not int or row[key] <= 0 for key in ('pid', 'start', 'group', 'session'))
                    or row['pid'] <= 1 or row['group'] != row['pid'] or row['session'] != row['pid']
                    or row['group'] in protected or row['pid'] in roots
                    or type(row['label']) is not str or not 0 < len(row['label']) <= 128):
                raise ValueError('registered leader identity is invalid')
            roots[row['pid']] = row
        if process is not None and process.pid not in roots:
            raise ValueError('requested leader is not registered')
        if len(bound)!=len(references) or not set(bound)<=set(roots):
            raise ValueError('retained leader objects are duplicate or unregistered')
    except (Exception, KeyboardInterrupt) as error:
        return dict(complete=False, leader_waited=leader_waited, groups=[], observed_pids=[],
                    remaining_pids=None, errors=[type(error).__name__ + ': ' + str(error)],
                    basis='no signal sent; process-incarnation cleanup preflight failed')

    def bound_parent_status(pid,start):
        ref=bound.get(pid)
        return (ref is not None and type(ref.returncode) is int
            and getattr(ref,'_owner_process_identity',None)==dict(pid=pid,start=start,scope=scope_identity))

    for pid,row in roots.items():
        key=(pid,row['start']);ref=bound.get(pid)
        if ref is not None and type(getattr(ref,'_owner_pidfd',None)) is int:
            if getattr(ref,'_owner_process_identity',None)!=dict(pid=pid,start=row['start'],scope=scope_identity):
                problem('retained early leader handle identity differs: '+str(pid))
            else:
                handles[key]=ref._owner_pidfd;observed[pid]=row['start']
                members[key]=dict(pid=pid,start=row['start'],initial_ppid=own,current_ppid=own,
                    group=row['group'],session=row['session'],basis='registered early parent pidfd')
        if bound_parent_status(pid,row['start']):
            fact=dict(pid=pid,start=row['start'],basis='bound_parent_process_wait',returncode=ref.returncode)
            terminal[key]=fact;reaped[pid]=dict(pid=pid,start=row['start'],returncode=ref.returncode)
            leader_waited=True

    def inspect():
        nonlocal remaining
        try:
            table = processes()
            if own not in table or table[own]['start'] != scope_identity['start']:
                raise ValueError('owner scope incarnation changed')
            active = {pid for pid, start in observed.items()
                      if pid in table and table[pid]['start'] == start}
            for pid, record in roots.items():
                if pid in table:
                    if table[pid]['start'] != record['start']:
                        problem('registered process number was replaced: ' + str(pid))
                    elif table[pid]['group'] != record['group'] or table[pid]['session'] != record['session']:
                        problem('registered leader session changed: ' + str(pid))
                    else:
                        active.add(pid)
            # A live original session anchor or a proven parent identifies new members.
            while True:
                sessions = {(table[pid]['group'], table[pid]['session']) for pid in active}
                added = {pid for pid, row in table.items() if pid != own and row['group'] not in protected
                         and (row['ppid'] in active or (row['group'], row['session']) in sessions)}
                if added <= active: break
                active |= added
            for pid in active:
                row = table[pid]; key = (pid, row['start'])
                if key in members:
                    members[key]['current_ppid']=row['ppid']
                    members[key]['cmdline']=row.get('cmdline')
                    members[key]['cmdline_status']=row.get('cmdline_status','unobservable')
                if key in handles: continue
                if len(handles)>=4096:raise ValueError('owned process handle bound exceeded')
                try:
                    descriptor = os.pidfd_open(pid, 0)
                except ProcessLookupError as error:
                    if pid in roots or error.errno != errno.ESRCH:
                        raise
                    fresh = processes()
                    if pid in fresh:
                        problem('process present after failed exact handle open: ' + str(pid))
                    else:
                        vanished_before_pidfd[key] = dict(pid=pid, start=row['start'],
                            initial_ppid=row['ppid'], group=row['group'], session=row['session'],
                            basis='ESRCH opening child handle and absent from fresh process table',
                            exit_status='unknown', reap_state='unknown', signal_sent=False)
                    continue
                later = processes().get(pid)
                if later is None or later['start'] != row['start']:
                    os.close(descriptor)
                    problem('process changed while opening exact handle: ' + str(pid))
                    continue
                handles[key] = descriptor; observed[pid] = row['start']
                members[key]=dict(pid=pid,start=row['start'],initial_ppid=row['ppid'],current_ppid=row['ppid'],
                    group=row['group'],session=row['session'],cmdline=row.get('cmdline'),
                    cmdline_status=row.get('cmdline_status','unobservable'))
            ambiguous = {pid for pid in descendants(table, own) if table[pid]['group'] not in protected and pid not in active}
            ambiguous |= {pid for pid, row in table.items() if row['group'] in roots and pid not in active}
            if ambiguous:
                problem('unproven process membership: ' + ','.join(map(str, sorted(ambiguous))))
            remaining = {pid: table[pid] for pid in active}
            return remaining
        except (Exception, KeyboardInterrupt) as error:
            problem('process inspection: ' + type(error).__name__ + ': ' + str(error))
            remaining = None
            return None

    def exited(key,descriptor):
        pid,start=key
        try:
            state=pidfd_state(descriptor)
            marker=(state['status'],state['flags'])
            if last_poll.get(key,('',''))[0]=='exit_observed' and state['status']!='exit_observed':
                raise ValueError('non-monotonic exit readiness on the same retained pidfd')
            if last_poll.get(key)!=marker:
                poll_events.append(dict(pid=pid,start=start,**state));last_poll[key]=marker
            if state['status']=='exit_observed':
                current=processes().get(pid)
                if current is not None and current['start']==start and current['state'] not in ('Z','X'):
                    raise ValueError('positive pidfd exit readiness disagrees with a live original task')
                previous=terminal.get(key)
                if previous is None or previous['basis']=='same_pidfd_exit_readiness':
                    terminal[key]=dict(pid=pid,start=start,basis='same_pidfd_exit_readiness',
                        returncode=None,exit_status='unknown',reap_state='observed' if state['fully_reaped'] else 'unknown',poll=state)
                return True
            return key in terminal
        except (Exception,KeyboardInterrupt) as error:
            problem('exact process poll pid='+str(pid)+' start='+str(start)+': '+type(error).__name__+': '+str(error))
            return None

    def send(selected):
        for key, descriptor in list(handles.items()):
            pid, start = key
            if key in terminal or (key, selected) in sent:
                continue
            if exited(key,descriptor) is not False:
                continue
            # Let each registered manager stop its own children during grace.
            if selected == signal.SIGTERM and (pid not in roots or roots[pid]['start'] != start):
                continue
            if selected==signal.SIGKILL and hasattr(bound.get(pid),'_owner_supervision'):
                # Keep the causal anchor until every observed non-anchor task
                # has stopped. A later bounded pass may kill an unresponsive
                # released anchor, which cannot produce a successful receipt.
                if pid not in released or any(k[0] not in roots and k not in terminal for k in handles):
                    continue
                if time.monotonic()<until-.1:continue
            sent.add((key, selected))
            event = dict(pid=pid, start=start, signal=int(selected), outcome='unknown')
            try:
                signal.pidfd_send_signal(descriptor, selected, None, 0)
                event['outcome'] = 'sent'
            except ProcessLookupError as error:
                event.update(outcome='already_absent', errno=error.errno)
            except (Exception, KeyboardInterrupt) as error:
                event.update(outcome='failed', errno=getattr(error, 'errno', None))
                problem('exact process signal pid=' + str(pid) + ' start=' + str(start) + ': '
                        + type(error).__name__ + ': ' + str(error))
            actions.append(event)

    def reap():
        nonlocal leader_waited
        for (pid, start), descriptor in list(handles.items()):
            key = (pid, start)
            fact=terminal.get(key)
            if fact is not None and fact['basis']!='same_pidfd_exit_readiness':
                continue
            positive=exited(key,descriptor)
            if positive is None:continue
            present=(remaining is not None and pid in remaining and remaining[pid]['start']==start)
            # Exit observation stops signals, but an adopted zombie still needs
            # an exact child wait. Only original absence retires a non-child wait.
            if positive and pid not in roots and (not present or members[key]['current_ppid']!=own):
                continue
            try:
                result = os.waitid(os.P_PIDFD, descriptor, os.WEXITED | os.WNOHANG)
                if result is None:
                    continue
                if (result.si_pid != pid or result.si_code not in
                        (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED)):
                    raise ValueError('exact wait returned an unmatched or nonterminal event')
                fact = dict(pid=pid, start=start, basis='exact_pidfd_wait',
                    returncode=result.si_status if result.si_code == os.CLD_EXITED else -result.si_status)
                terminal[key] = fact
                if pid in roots and start == roots[pid]['start']:
                    reaped[pid] = dict(pid=pid, start=start, returncode=fact['returncode'])
                if pid in bound and start == roots[pid]['start']:
                    bound[pid].returncode = fact['returncode']
                    leader_waited = True
            except ChildProcessError as error:
                if bound_parent_status(pid,start):
                    terminal[key] = dict(pid=pid, start=start, basis='bound_parent_process_wait',
                                         returncode=bound[pid].returncode)
                    leader_waited = True
                    reaped[pid]=dict(pid=pid,start=start,returncode=bound[pid].returncode)
                else:
                    # A manager can reap its own child. Absence is checked later;
                    # this result is not a terminal or successful reaping fact.
                    if not any(row['pid'] == pid and row['start'] == start for row in wait_errors):
                        wait_errors.append(dict(pid=pid, start=start, errno=error.errno,
                                                reason='not_a_waitable_child'))
            except (Exception, KeyboardInterrupt) as error:
                event = dict(pid=pid, start=start, errno=getattr(error, 'errno', None),
                             reason=type(error).__name__ + ': ' + str(error))
                if event not in wait_errors:
                    wait_errors.append(event)
                stopped=False
                if getattr(error,'errno',None)==errno.ESRCH and pid not in roots and exited(key,descriptor) is True:
                    fresh=inspect()
                    if fresh is not None:
                        absent=pid not in fresh or fresh[pid]['start']!=start
                        zombie=not absent and fresh[pid]['state'] in ('Z','X')
                        stopped=absent or zombie
                        if stopped:event['classification']=('exit_status_unavailable_after_positive_exit' if absent
                            else 'exit_observed_zombie_pending_parent_reap')
                if not stopped:
                    problem('exact process reap pid=' + str(pid) + ' start=' + str(start) + ': '
                            + type(error).__name__ + ': ' + str(error))

    def release_anchors():
        if remaining is None or any(pid not in roots and (pid,row['start']) not in terminal
                                    for pid,row in remaining.items()):return
        from .supervisor import release
        for pid,ref in bound.items():
            key=(pid,roots[pid]['start'])
            if hasattr(ref,'_owner_supervision') and key not in terminal and pid not in released:
                released[pid]=time.monotonic()
                try:
                    release(ref)
                    actions.append(dict(pid=pid,start=key[1],control='release_anchor',outcome='sent'))
                except (Exception,KeyboardInterrupt) as error:
                    actions.append(dict(pid=pid,start=key[1],control='release_anchor',outcome='failed'))
                    problem('session anchor release: '+type(error).__name__+': '+str(error))

    try:
        for selected, seconds in ((signal.SIGTERM, grace), (signal.SIGKILL, kill_wait)):
            until = time.monotonic() + seconds
            while True:
                current = inspect()
                if current is None: break
                if not current: break
                send(selected); reap(); release_anchors()
                if time.monotonic() >= until: break
                time.sleep(.05)
        reap(); inspect()
    finally:
        for descriptor in set(handles.values()): os.close(descriptor)
        for ref in bound.values():
            if getattr(ref,'_owner_pidfd',None) in handles.values():ref._owner_pidfd=None
            if hasattr(ref,'_owner_supervision') and ref.stdin is not None and not ref.stdin.closed:ref.stdin.close()
    supervisor_receipts=[]
    from .supervisor import closed,poll_engine
    for pid,ref in bound.items():
        if hasattr(ref,'_owner_supervision'):
            try:good=closed(ref)
            except Exception as error:
                good=False;problem('session anchor receipt: '+type(error).__name__+': '+str(error))
            child_status=None
            if good:
                try:child_status=poll_engine(ref)
                except Exception as error:
                    good=False;problem('bound engine exit receipt: '+type(error).__name__+': '+str(error))
            supervisor_receipts.append(dict(pid=pid,start=roots[pid]['start'],complete=good,
                directory=ref._owner_supervision['directory'],engine_returncode=child_status,
                anchor_returncode=ref.returncode))
            if not good:problem('session anchor did not record a complete child-status close: '+str(pid))
    missing=[dict(pid=pid,start=start) for pid,start in handles if (pid,start) not in terminal]
    roots_unreaped=[dict(pid=pid,start=row['start']) for pid,row in roots.items() if pid not in reaped]
    return dict(complete=leader_waited and remaining == {} and not errors and not missing and not roots_unreaped,
        leader_waited=leader_waited, groups=sorted({row['group'] for row in roots.values()}),
        observed_pids=sorted(observed), remaining_pids=sorted(remaining) if isinstance(remaining, dict) else None,
        reaped_leaders=list(reaped.values()), terminal_handles=list(terminal.values()),
        signal_actions=actions, wait_errors=wait_errors, errors=errors,
        process_membership=list(members.values()),pidfd_observations=poll_events,
        vanished_before_pidfd=list(vanished_before_pidfd.values()),
        missing_terminal_evidence=missing,roots_without_exact_exit_status=roots_unreaped,
        supervisor_receipts=supervisor_receipts,
        basis='owned processes stopped; same-handle exit readiness is separate from exact child reaping; descendant exit status and reaping can remain unknown')
