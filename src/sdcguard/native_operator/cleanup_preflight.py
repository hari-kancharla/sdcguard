"""Bounded ordinary process lifecycle capability check. Owner: Hari Krishna Kancharla."""
import errno
import contextlib
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import time
from .cleanup import require_pidfds, pidfd_state, processes

SCHEMA='sdcguard.owner-cleanup-capability.v1'
SCOPE='bounded ordinary stdlib process lifecycle and session supervision only; no engine, model, server, request or GPU operation'
MANAGER='''import json,os,sys
r,w=os.pipe()
child=os.fork()
if child==0:
 os.close(w);os.read(r,1);os.close(r);os._exit(0)
os.close(r)
print(json.dumps({'child':child}),flush=True)
try:
 if sys.stdin.readline().strip()!='release':raise RuntimeError('release handshake missing')
 os.write(w,b'1');os.close(w);w=None
 print('released',flush=True)
 if sys.stdin.readline().strip()!='reap':raise RuntimeError('reap handshake missing')
 found,status=os.waitpid(child,0)
 if found!=child or status!=0:raise RuntimeError('ordinary child exit differs')
 print('reaped',flush=True)
 if sys.stdin.readline().strip()!='stop':raise RuntimeError('stop handshake missing')
finally:
 if w is not None:os.close(w)
 try:os.waitpid(child,0)
 except ChildProcessError:pass
'''


def _line(stream,deadline):
    raw=b'';poller=select.poll();poller.register(stream.fileno(),select.POLLIN)
    while time.monotonic()<deadline:
        if not poller.poll(25):continue
        part=os.read(stream.fileno(),256)
        if not part:raise ValueError('lifecycle handshake ended before completion')
        raw+=part
        if len(raw)>256:raise ValueError('lifecycle handshake exceeds bound')
        if raw.endswith(b'\n'):return raw.decode().strip()
    raise TimeoutError('lifecycle handshake deadline reached')


def _state(fd,deadline):
    while time.monotonic()<deadline:
        value=pidfd_state(fd,25)
        if value['status']=='exit_observed':return value
    raise TimeoutError('required same-handle process exit readiness was not observed')


def _pidfd_probe(run_id,packet_sha256):
    import re
    if re.fullmatch('[0-9a-f]{32}',run_id) is None or re.fullmatch('[0-9a-f]{64}',packet_sha256) is None:
        raise ValueError('lifecycle probe invocation identity differs')
    result=dict(schema=SCHEMA,owner='Hari Krishna Kancharla',run_id=run_id,packet_sha256=packet_sha256,
        status='UNKNOWN',events=[],failure=None,cleanup_complete=False,cleanup_failures=[],
        scope=SCOPE,
        manager_source_sha256=hashlib.sha256(MANAGER.encode()).hexdigest())
    manager=None;manager_fd=child_fd=None;child=None;stderr=None;step='require_pidfds';deadline=time.monotonic()+8
    try:
        require_pidfds();step='start_owned_manager'
        stderr=tempfile.TemporaryFile()
        manager=subprocess.Popen([sys.executable,'-I','-S','-u','-c',MANAGER],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=stderr,start_new_session=True)
        manager_before=processes().get(manager.pid)
        if (manager_before is None or manager_before['ppid']!=os.getpid()
            or manager_before['group']!=manager.pid or manager_before['session']!=manager.pid):
            raise ValueError('owned manager identity is unavailable')
        manager_fd=os.pidfd_open(manager.pid,0)
        manager_after=processes().get(manager.pid)
        if manager_after is None or manager_after['start']!=manager_before['start']:
            raise ValueError('manager changed while opening its exact handle')
        result['manager_identity']=dict(pid=manager.pid,start=manager_before['start'])
        step='owned_manager_signal_zero'
        signal.pidfd_send_signal(manager_fd,0,None,0)
        result['events'].append(dict(step=step,signal=0,outcome='accepted'))
        step='open_live_nonchild'
        child=json.loads(_line(manager.stdout,deadline))['child']
        if type(child) is not int or child<=1:raise ValueError('lifecycle child identity is invalid')
        before=processes().get(child)
        if before is None or before['ppid']!=manager.pid:raise ValueError('manager-owned child identity unavailable')
        child_fd=os.pidfd_open(child,0)
        after=processes().get(child)
        if after is None or after['start']!=before['start']:raise ValueError('child changed while opening its exact handle')
        if pidfd_state(child_fd)['status']!='live_or_unobserved':raise ValueError('ordinary child was not observed live')
        result['events'].append(dict(step=step,pid=child,start=before['start']))
        step='nonchild_wait_refused'
        try:os.waitid(os.P_PIDFD,child_fd,os.WEXITED|os.WNOHANG)
        except ChildProcessError as error:
            if error.errno!=errno.ECHILD:raise
            result['events'].append(dict(step=step,errno=error.errno))
        else:raise ValueError('nonchild wait was not refused as documented')
        step='nonchild_exit_readiness'
        manager.stdin.write(b'release\n');manager.stdin.flush()
        if _line(manager.stdout,deadline)!='released':raise ValueError('release acknowledgement differs')
        result['events'].append(dict(step=step,**_state(child_fd,deadline)))
        step='manager_reaps_nonchild'
        manager.stdin.write(b'reap\n');manager.stdin.flush()
        if _line(manager.stdout,deadline)!='reaped':raise ValueError('manager reaping acknowledgement differs')
        result['events'].append(dict(step=step,**_state(child_fd,deadline)))
        later=processes().get(child)
        if later is not None and later['start']==before['start']:raise ValueError('original child still appears after reaping')
        step='direct_child_exact_wait'
        manager.stdin.write(b'stop\n');manager.stdin.flush();manager.stdin.close()
        result['events'].append(dict(step='manager_exit_readiness',**_state(manager_fd,deadline)))
        waited=os.waitid(os.P_PIDFD,manager_fd,os.WEXITED|os.WNOHANG)
        if waited is None or waited.si_pid!=manager.pid or waited.si_code!=os.CLD_EXITED or waited.si_status!=0:
            raise ValueError('ordinary manager exact exit status differs')
        manager.returncode=0
        result['events'].append(dict(step=step,returncode=0))
        result['events'].append(dict(step='direct_child_after_wait',**_state(manager_fd,deadline)))
        result.update(status='PASS',cleanup_complete=True)
    except (Exception,KeyboardInterrupt) as error:
        result['failure']=dict(step=step,kind=type(error).__name__,errno=getattr(error,'errno',None),detail=str(error)[:256])
    finally:
        if manager is not None:
            if manager.stdin and not manager.stdin.closed:manager.stdin.close()
            if manager.returncode is None:
                try:manager.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    for fd in (child_fd,manager_fd):
                        if fd is not None:
                            try:signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
                            except ProcessLookupError:pass
                            except OSError as error:result['cleanup_failures'].append(dict(step='probe_kill',kind=type(error).__name__,errno=error.errno,detail=str(error)[:128]))
                    try:manager.wait(timeout=1)
                    except subprocess.TimeoutExpired:result['cleanup_failures'].append(dict(step='probe_manager_wait',kind='TimeoutExpired',errno=None,detail='owned manager termination remains unobserved'))
            if manager.stdout:manager.stdout.close()
        for fd in (child_fd,manager_fd):
            if fd is not None:os.close(fd)
        if manager is not None:
            result['manager_returncode']=manager.returncode
        if stderr is not None:
            size=stderr.seek(0,os.SEEK_END);stderr.seek(max(0,size-1024))
            result['manager_stderr_tail']=stderr.read(1024).decode('utf-8',errors='replace');stderr.close()
    return result


def _validate_pidfd(value,run_id,packet_sha256):
    required=['owned_manager_signal_zero','open_live_nonchild','nonchild_wait_refused','nonchild_exit_readiness','manager_reaps_nonchild',
        'manager_exit_readiness','direct_child_exact_wait','direct_child_after_wait']
    keys={'schema','owner','run_id','packet_sha256','status','events','failure','cleanup_complete',
        'cleanup_failures','scope','manager_source_sha256','manager_identity','manager_returncode','manager_stderr_tail'}
    basic=(type(value) is dict and value.keys()==keys and value.get('schema')==SCHEMA and value.get('status')=='PASS'
        and value.get('owner')=='Hari Krishna Kancharla'
        and value.get('run_id')==run_id and value.get('packet_sha256')==packet_sha256
        and value.get('failure') is None and value.get('cleanup_complete') is True
        and value.get('cleanup_failures')==[] and type(value.get('cleanup_failures')) is list
        and type(value.get('manager_returncode')) is int and value['manager_returncode']==0
        and value.get('scope')==SCOPE and type(value.get('manager_stderr_tail')) is str and len(value['manager_stderr_tail'])<=1024
        and type(value.get('events')) is list and all(type(row) is dict for row in value['events'])
        and [row.get('step') for row in value['events']]==required
        and value.get('manager_source_sha256')==hashlib.sha256(MANAGER.encode()).hexdigest())
    if not basic:return False
    rows=value['events'];zero=rows[0];opened=rows[1]
    manager=value['manager_identity']
    if not (type(manager) is dict and manager.keys()=={'pid','start'}
            and type(manager['pid']) is int and 1<manager['pid']<2**31 and type(manager['start']) is int and 0<manager['start']<2**64
            and opened.keys()=={'step','pid','start'} and type(opened.get('pid')) is int and 1<opened['pid']<2**31
            and manager['pid']!=opened['pid'] and type(opened.get('start')) is int and 0<opened['start']<2**64
            and zero==dict(step='owned_manager_signal_zero',signal=0,outcome='accepted')
            and rows[2].keys()=={'step','errno'} and type(rows[2].get('errno')) is int and rows[2]['errno']==errno.ECHILD
            and rows[6].keys()=={'step','returncode'} and type(rows[6].get('returncode')) is int and rows[6]['returncode']==0):return False
    for index in (3,4,5,7):
        row=rows[index];flags=row.get('flags')
        if (row.keys()!={'step','status','flags','events','fully_reaped'} or row.get('status')!='exit_observed' or type(flags) is not int or flags<=0
            or flags & ~(select.POLLIN|select.POLLHUP) or type(row.get('fully_reaped')) is not bool
            or row['fully_reaped']!=bool(flags & select.POLLHUP)
            or row.get('events')!=[name for name in ('POLLIN','POLLHUP') if flags & getattr(select,name)]):return False
    return True


def describe(value):
    if type(value) is not dict:return 'malformed lifecycle capability result'
    text=str(value.get('failure'))[:384]
    if value.get('manager_returncode') is not None:text+='; manager_returncode='+str(value['manager_returncode'])[:16]
    if value.get('manager_stderr_tail'):text+='; manager_stderr='+str(value['manager_stderr_tail'])[-384:]
    return text


SUPERVISION_CHILD='''import os,time
child=os.fork()
if child==0:
 time.sleep(20)
 os._exit(0)
os._exit(0)
'''


def supervision_probe(run_id,packet_sha256):
    """Use the production anchor and cleanup on ordinary CPU children."""
    from pathlib import Path
    from .cleanup import subreaper,baseline,register,cleanup,processes
    from .supervisor import launch,start,poll_engine,closed
    from .common import regular_bytes
    result=dict(schema='sdcguard.owner-supervision-capability.v1',owner='Hari Krishna Kancharla',
        run_id=run_id,packet_sha256=packet_sha256,status='UNKNOWN',failure=None,cleanup=None,
        source_sha256=hashlib.sha256(SUPERVISION_CHILD.encode()).hexdigest(),
        child_returncode=None,child_exit=None,anchor_closed=None,anchor_live_after_child_exit=False,
        scope='ordinary stdlib parent exits and leaves one child under the same production session anchor')
    process=None;record=None
    with contextlib.nullcontext(tempfile.mkdtemp(prefix='sdcguard-supervision-probe-')) as temporary:
        directory=Path(temporary);registry=directory/'PROCESS_GROUPS.json'
        result['directory']=str(directory)
        subreaper();before=baseline()
        with (directory/'ordinary.log').open('xb') as log:
            try:
                process=launch([sys.executable,'-u','-c',SUPERVISION_CHILD],env=dict(os.environ),stdout=log,
                    directory=directory/'anchor',label='ordinary-cpu-probe')
                register(registry,process,'ordinary-cpu-probe');start(process,timeout=5)
                until=time.monotonic()+5
                while time.monotonic()<until:
                    code=poll_engine(process)
                    if code is not None:break
                    time.sleep(.02)
                else:raise TimeoutError('ordinary direct child did not exit within its bound')
                result['child_returncode']=code
                row=processes().get(process.pid)
                result['anchor_live_after_child_exit']=(row is not None and row['start']==process._owner_process_identity['start']
                    and row['state'] not in ('Z','X'))
                if code!=0 or not result['anchor_live_after_child_exit']:
                    raise ValueError('ordinary child exit or persistent anchor observation differs')
            except Exception as error:
                result['failure']=dict(kind=type(error).__name__,detail=str(error)[:512])
            finally:
                if process is not None:
                    record=cleanup(leaders=[process],scope=os.getpid(),baseline=before,registry=registry,
                        grace=.15,kill_wait=3)
                    result['cleanup']=record
                    for field,name in (('child_exit','ENGINE_EXIT.json'),('anchor_closed','ANCHOR_CLOSED.json')):
                        path=directory/'anchor'/name
                        if path.exists():result[field]=json.loads(regular_bytes(path,128*1024))
                if result['failure'] is None and record is not None and record.get('complete') is True and closed(process):
                    result['status']='PASS'
                else:result['status']='FAIL'
    return result


def validate_supervision(value,run_id,packet_sha256):
    if not (type(value) is dict and value.get('schema')=='sdcguard.owner-supervision-capability.v1'
        and value.get('owner')=='Hari Krishna Kancharla' and value.get('run_id')==run_id
        and value.get('packet_sha256')==packet_sha256 and value.get('status')=='PASS' and value.get('failure') is None
        and value.get('source_sha256')==hashlib.sha256(SUPERVISION_CHILD.encode()).hexdigest()
        and type(value.get('child_returncode')) is int and value['child_returncode']==0
        and value.get('anchor_live_after_child_exit') is True):return False
    cleanup=value.get('cleanup');child=value.get('child_exit');anchor=value.get('anchor_closed')
    return (type(cleanup) is dict and cleanup.get('complete') is True and cleanup.get('errors')==[]
        and cleanup.get('remaining_pids')==[] and cleanup.get('roots_without_exact_exit_status')==[]
        and bool(cleanup.get('supervisor_receipts')) and all(x.get('complete') is True for x in cleanup['supervisor_receipts'])
        and type(child) is dict and type(child.get('returncode')) is int and child['returncode']==0
        and type(anchor) is dict and anchor.get('complete') is True and anchor.get('errors')==[]
        and anchor.get('child_exit')==child and anchor.get('anchor')==child.get('anchor')
        and anchor.get('spec_sha256')==child.get('spec_sha256'))


def probe(run_id,packet_sha256):
    result=_pidfd_probe(run_id,packet_sha256)
    result['supervision']=(supervision_probe(run_id,packet_sha256) if result.get('status')=='PASS'
        else dict(status='NOT_RUN',reason='native pidfd capability did not pass'))
    if not validate_supervision(result['supervision'],run_id,packet_sha256):
        result['status']='UNKNOWN';result['cleanup_complete']=False
    return result


def validate(value,run_id,packet_sha256):
    if type(value) is not dict or 'supervision' not in value:return False
    original={key:item for key,item in value.items() if key!='supervision'}
    return _validate_pidfd(original,run_id,packet_sha256) and validate_supervision(value['supervision'],run_id,packet_sha256)
