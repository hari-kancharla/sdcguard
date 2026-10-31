"""Installed flow controls with no engine or GPU. Owner: Hari Krishna Kancharla."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sdcguard import cli, evidence_io as io, required_gate
from sdcguard.native_operator import command, config, installed, listeners, preflight, runtime, workload
from sdcguard.native_operator.common import encoded, digest


def plan():
    return workload.build_plan(lambda text, **kw:list(text.encode()),
        run_id="a"*32, packet_sha256="b"*64, engine_sessions={"observer-on":"c"*32})


def test_plan_has_one_complete_denominator_and_shared_prefix():
    value, payloads = plan()
    assert value["prefix_caching"] is True
    assert len(value["arms"]) == 1 and value["arms"][0]["observer_enabled"] is True
    rows = value["arms"][0]["requests"]
    assert len(rows) == len(payloads) == 153
    assert {r["concurrency"] for r in rows} == {1,4,16}
    assert value["chat_template"] is False
    assert len({r["cache_group"] for r in rows}) == 9
    assert all(r["shared_prefix_tokens"] >= 64 for r in rows)
    assert any(r["prompt_tokens"] % 16 for r in rows)
    assert all(r["prompt_tokens"] == len(payloads[r["client_request_id"]]) for r in rows)
    assert len({r["prefill_request_id"] for r in rows} | set(payloads)) == 306


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_operator_uses_explicit_profile_and_only_the_native_plugin(tmp_path, role):
    env = config.environment(dict(PYTHONPATH="/untrusted", VLLM_PLUGINS="other", SDCGUARD="1",
        OWNER_CHECKSUM_RUN_ID="wrong", UCX_TLS="tcp", NIXL_FOO="other"), out=tmp_path,
        run_id="a"*32, packet_sha256="b"*64, arm_id="observer-on", engine_session_id="c"*32, role=role, enabled=True)
    assert env["VLLM_PLUGINS"] == "sdcguard_coverage" and env["OWNER_CHECKSUM_ENABLED"] == "1"
    assert "PYTHONPATH" not in env and "UCX_TLS" not in env and "NIXL_FOO" not in env
    args = config.server_command(role)
    assert "--enable-prefix-caching" in args and "--no-enable-prefix-caching" not in args
    assert "--enforce-eager" in args and "--no-async-scheduling" in args
    transfer = json.loads(args[args.index("--kv-transfer-config")+1])
    assert transfer["enable_kv_checksum"] is True


def test_unsupported_combined_profile_never_enters_preflight(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(preflight,"execute",lambda:pytest.fail("preflight must not run"))
    assert command.main(["--out",str(tmp_path/"out"),"--require","checksum","--require","lease"]) == 2
    assert "unsupported" in capsys.readouterr().out
    assert not (tmp_path/"out").exists()


def test_failed_cpu_preflight_never_starts_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight,"execute",lambda:(dict(status="UNKNOWN",failure="synthetic unavailable profile"),None,None))
    monkeypatch.setattr(runtime,"execute",lambda *a,**k:pytest.fail("runtime must not run"))
    assert cli.main(["operator","run","--out",str(tmp_path/"out"),"--require","checksum"]) == 2
    assert (tmp_path/"out/CPU_PREFLIGHT.json").is_file()


def test_cpu_only_command_has_no_server_path(tmp_path, monkeypatch):
    value,payloads=plan()
    monkeypatch.setattr(preflight,"execute",lambda:(dict(status="PASS"),value,payloads))
    monkeypatch.setattr(runtime,"execute",lambda *a,**k:pytest.fail("runtime must not run"))
    assert cli.main(["operator","run","--out",str(tmp_path/"out"),"--require","checksum","--preflight-only"]) == 0
    assert (tmp_path/"out/REQUEST_PLAN.json").is_file()
    assert not (tmp_path/"out/RESULT.json").exists()


def test_command_saves_raw_inputs_then_runs_the_existing_gate(tmp_path, monkeypatch):
    data=Path(__file__).parent/'data/checksum_owner_870c1aac'
    observed=json.loads((data/'RESULT.json').read_bytes())
    archive=(data/'ordinary-evidence.zip').read_bytes()
    receipt=json.loads((data/'CPU_PREFLIGHT.json').read_bytes())
    value,payloads=plan()
    monkeypatch.setattr(preflight,"execute",lambda:(receipt,value,payloads))
    monkeypatch.setattr(runtime,"execute",lambda *a,**k:dict(deepcopy(observed),archive=archive))
    out=tmp_path/'out'
    assert cli.main(['operator','run','--out',str(out),'--require','checksum']) == 2
    # All 180 saved requests stay in scope; the command cannot hide the unobserved arm.
    gate=json.loads((out/'GATE.json').read_bytes())
    assert gate['selected_requests']==180 and gate['passed_requests']==90
    assert gate['excluded_requests']==0 and gate['decision']=='HOLD'
    assert (out/'ordinary-evidence.zip').read_bytes()==archive


def test_owned_listener_rejects_an_unrelated_server(tmp_path):
    (tmp_path/'net').mkdir()
    (tmp_path/'net/tcp').write_text('header\n 0: 0100007F:1FA4 00000000:0000 0A 0 0 0 0 0 99\n')
    process=SimpleNamespace(pid=12,_owner_process_identity=dict(pid=12,start=40))
    rows={12:dict(start=40,group=12,session=12),13:dict(start=41,group=12,session=12),99:dict(start=42,group=99,session=99)}
    (tmp_path/'99/fd').mkdir(parents=True);(tmp_path/'99/fd/1').symlink_to('socket:[99]')
    assert not listeners.owned(8100,process,proc=tmp_path,table=rows)
    (tmp_path/'13/fd').mkdir(parents=True);(tmp_path/'13/fd/1').symlink_to('socket:[99]')
    assert listeners.owned(8100,process,proc=tmp_path,table=rows)
    rows[12]['start']=50
    assert not listeners.owned(8100,process,proc=tmp_path,table=rows)


@pytest.mark.parametrize('role',['prefill','decode'])
def test_closed_capture_requires_cache_and_binding_on_both_roles(tmp_path,role):
    value,_=plan();arm=value['arms'][0]
    context=dict(run_id=value['run_id'],packet_sha256=value['packet_sha256'],arm_id=arm['arm_id'],engine_session_id=arm['engine_session_id'],process_id=123)
    for kind in (['bindings','cache','checksum'] if role=='decode' else ['bindings','cache']):
        (tmp_path/(kind+'-123.json')).write_bytes(encoded(dict(schema='sdcguard.owner-checksum-close.v1',role=role,
            context=context,original_shutdown_completed=True,original_shutdown_error=None,recorder_errors=[],snapshot={'test':'CPU control'})))
    result=runtime.closed_capture(tmp_path,plan=value,arm=arm,startup={},role=role)
    assert result['capture_complete'] is True and len(result['cache_snapshots'])==1
    (tmp_path/'cache-123.json').unlink()
    assert runtime.closed_capture(tmp_path,plan=value,arm=arm,startup={},role=role)['capture_complete'] is False


def installed_receipt():
    """Offline schema assembly only, never a new installed-runtime observation."""
    source=json.loads((Path(__file__).parent/'data/checksum_owner_870c1aac/CPU_PREFLIGHT.json').read_bytes())
    files=installed.observer_files()
    product=dict(version='0.4.0',files=files,sha256=digest(encoded(files)),distribution='sdcguard',entrypoint='sdcguard.native_operator.plugin:register')
    value,payloads=plan();value['packet_sha256']=product['sha256']
    result={k:deepcopy(source[k]) for k in ('installed_engine','pinned_source_guard','model_cache','python_runtime','cleanup_capability','transport_readiness')}
    result.update(schema=installed.SCHEMA,run_id=value['run_id'],packet_sha256=value['packet_sha256'],plan_sha256=digest(encoded(value)),
        source_revision=workload.REVISION,product=product,status='PASS',failure=None,cpu_only=True,gpu_submitted=False,runtime_qualified=False)
    for name in ('python_runtime','cleanup_capability','transport_readiness'):
        result[name].update(run_id=value['run_id'],packet_sha256=value['packet_sha256'])
    result['cleanup_capability']['supervision'].update(run_id=value['run_id'],packet_sha256=value['packet_sha256'])
    from sdcguard.native_operator import transport
    result['transport_readiness']['schema']=transport.SCHEMA
    result['transport_readiness']['cases'].pop('baseline')
    for name,expected in installed.profile('ENGINE_BUILD_LOCK.json')['sources'].items():
        result['installed_engine']['source_files'].setdefault(name,dict(bytes=1,sha256=expected['sha256']))
    result['build_witness']=installed.witness(result)
    result['build_witness_sha256']=digest(encoded(result['build_witness']))
    return result,value


def test_installed_receipt_uses_pinned_values():
    receipt,value=installed_receipt()
    assert installed.validate(receipt,value,encoded(value))


@pytest.mark.parametrize('change',['source','compiled','model','product','entrypoint','interpreter','cleanup','transport','schema','version','plan','witness'])
def test_bad_installed_identity_never_passes(change):
    receipt,value=installed_receipt()
    if change=='source':next(iter(receipt['installed_engine']['source_files'].values()))['sha256']='0'*64
    elif change=='compiled':receipt['installed_engine']['compiled_members'].pop(next(iter(receipt['installed_engine']['compiled_members'])))
    elif change=='model':next(iter(receipt['model_cache']['files'].values()))['sha256']='0'*64
    elif change=='product':receipt['product']['files'].pop('sdcguard/native_operator/plugin.py')
    elif change=='entrypoint':receipt['product']['entrypoint']='unrelated:register'
    elif change=='interpreter':receipt['python_runtime']['identity']['version']=[3,12,True]
    elif change=='cleanup':receipt['cleanup_capability']['status']='UNKNOWN'
    elif change=='transport':receipt['transport_readiness']['status']='UNKNOWN'
    elif change=='schema':receipt['schema']='sdcguard.owner-checksum-cpu-preflight.v1'
    elif change=='version':receipt['installed_engine']['versions']['vllm']='0.99.0'
    elif change=='plan':receipt['plan_sha256']='0'*64
    else:receipt['build_witness_sha256']='0'*64
    if change != 'witness':
        if change == 'product':
            receipt['product']['sha256']=digest(encoded(receipt['product']['files']))
            value['packet_sha256']=receipt['packet_sha256']=receipt['product']['sha256']
            receipt['plan_sha256']=digest(encoded(value))
            for key in ('python_runtime','cleanup_capability','transport_readiness'):
                receipt[key]['packet_sha256']=value['packet_sha256']
            receipt['cleanup_capability']['supervision']['packet_sha256']=value['packet_sha256']
        receipt['build_witness']=installed.witness(receipt)
        receipt['build_witness_sha256']=digest(encoded(receipt['build_witness']))
    assert not installed.validate(receipt,value,encoded(value))


@pytest.mark.parametrize('field', ['editable','compiled','payload','source','model','product',
    'compile_status','load_status','pip_status','distro_status','header_version','header_size','command_size','availability'])
def test_malformed_typed_receipts_hold_after_witness_rehashed(field):
    receipt,value=installed_receipt()
    engine=receipt['installed_engine']; python=receipt['python_runtime']
    if field=='editable':engine['direct_url']['dir_info']['editable']=1
    elif field in ('compiled','payload','source','model','product'):
        rows={'compiled':engine['compiled_members'],'payload':engine['precompiled_payload']['extracted_members'],
              'source':engine['source_files'],'model':receipt['model_cache']['files'],'product':receipt['product']['files']}[field]
        row=next(iter(rows.values()));row['bytes']=float(row['bytes'])
    elif field.endswith('_status'):
        row={'compile_status':python['header_probe']['compile'],'load_status':python['header_probe']['load'],
             'pip_status':python['pip_check'],'distro_status':python['distro_packages']}[field]
        row['returncode']=False
    elif field=='header_version':python['header_probe']['observation']['compiled'][0]=3.0
    elif field=='header_size':python['header_probe']['extension']['bytes']=True
    elif field=='command_size':python['pip_check']['output_bytes']=float(python['pip_check']['output_bytes'])
    else:python['identity']['cleanup_api_availability']['os.pidfd_open']=1
    if field=='product':
        receipt['product']['sha256']=digest(encoded(receipt['product']['files']))
        value['packet_sha256']=receipt['packet_sha256']=receipt['product']['sha256']
        receipt['plan_sha256']=digest(encoded(value))
        for key in ('python_runtime','cleanup_capability','transport_readiness'):
            receipt[key]['packet_sha256']=value['packet_sha256']
        receipt['cleanup_capability']['supervision']['packet_sha256']=value['packet_sha256']
    receipt['build_witness']=installed.witness(receipt)
    receipt['build_witness_sha256']=digest(encoded(receipt['build_witness']))
    assert not installed.validate(receipt,value,encoded(value))


def test_connected_peer_must_belong_to_retained_role(tmp_path):
    (tmp_path/'net').mkdir()
    (tmp_path/'net/tcp').write_text('header\n 0: 0100007F:1FA4 0100007F:A000 01 0 0 0 0 0 99\n')
    process=SimpleNamespace(pid=12,_owner_process_identity=dict(pid=12,start=40))
    rows={12:dict(start=40,group=12,session=12),13:dict(start=41,group=12,session=12)}
    sock=SimpleNamespace(getsockname=lambda:('127.0.0.1',40960),getpeername=lambda:('127.0.0.1',8100))
    assert not listeners.connected_owned(sock,process,proc=tmp_path,table=rows)
    (tmp_path/'13/fd').mkdir(parents=True);(tmp_path/'13/fd/1').symlink_to('socket:[99]')
    assert listeners.connected_owned(sock,process,proc=tmp_path,table=rows)
    rows[13]['session']=99
    assert not listeners.connected_owned(sock,process,proc=tmp_path,table=rows)


@pytest.mark.parametrize('loss',['listener','exit','stop'])
def test_per_stage_ownership_loss_stops_new_sends(monkeypatch,loss):
    import threading
    from sdcguard.native_operator import client,supervisor
    stop=threading.Event();sends=[]
    monkeypatch.setattr(supervisor,'poll_engine',lambda p:0 if loss=='exit' else None)
    monkeypatch.setattr(listeners,'owned',lambda *a:loss!='listener')
    def transport(role,body,headers,*,timeout,guard):
        guard(role,None);sends.append(role)
    monkeypatch.setattr(client,'post',transport)
    if loss=='stop':stop.set()
    send=runtime.owned_transport({'decode':object()},stop)
    with pytest.raises(ValueError,match='owned decode'):
        send('decode',{}, {},timeout=1)
    assert stop.is_set() and sends==[]


def test_post_does_not_send_body_until_connected_peer_is_owned(monkeypatch):
    from sdcguard.native_operator import client
    actions=[]
    class Connection:
        sock=object()
        def __init__(self,*a,**k):pass
        def connect(self):actions.append('connect')
        def request(self,*a,**k):actions.append('body')
        def close(self):actions.append('close')
    monkeypatch.setattr(client.http.client,'HTTPConnection',Connection)
    def guard(role,sock):
        if sock is not None:raise ValueError('peer changed')
    with pytest.raises(ValueError,match='peer changed'):
        client.post('decode',{}, {},timeout=1,guard=guard)
    assert actions==['connect','close']


@pytest.mark.parametrize('change',[None,'missing','extra','token','length','boolean','budget','prefix'])
def test_payloads_are_bound_to_independent_plan_before_runtime(change):
    value,payloads=plan();key=next(iter(payloads))
    if change=='missing':payloads.pop(key)
    elif change=='extra':payloads['unplanned']=[]
    elif change=='token':payloads[key][0]+=1
    elif change=='length':payloads[key].append(3)
    elif change=='boolean':payloads[key][0]=True
    elif change=='budget':value['arms'][0]['requests'][0]['max_tokens']=10000
    elif change=='prefix':value['prefix_caching']=False
    if change is None:workload.validate_payloads(value,payloads)
    else:
        with pytest.raises(ValueError):workload.validate_payloads(value,payloads)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux socket ownership observation')
def test_owned_transport_accepts_an_ordinary_stdlib_peer_then_stops_after_exit(monkeypatch):
    import signal
    import subprocess
    import threading
    from sdcguard.native_operator import cleanup,client
    script='''import http.server,json
class Handler(http.server.BaseHTTPRequestHandler):
 def do_POST(self):
  self.rfile.read(int(self.headers['Content-Length']))
  raw=b'{"ordinary_cpu_control":true}'
  self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
 def log_message(self,*args):pass
server=http.server.HTTPServer(('127.0.0.1',0),Handler)
print(server.server_port,flush=True)
server.handle_request();server.server_close()
'''
    process=subprocess.Popen([sys.executable,'-I','-B','-u','-c',script],stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,text=True,start_new_session=True)
    descriptor=os.pidfd_open(process.pid,0)
    try:
        port=int(process.stdout.readline())
        process._owner_process_identity=dict(pid=process.pid,**cleanup.processes()[process.pid])
        monkeypatch.setitem(runtime.PORTS,'decode',port);monkeypatch.setitem(client.PORTS,'decode',port)
        stop=threading.Event();send=runtime.owned_transport({'decode':process},stop)
        receipt=send('decode',{}, {},timeout=3)
        assert receipt['http_status']==200 and receipt['response_body']=={'ordinary_cpu_control':True}
        process.wait(timeout=5)
        with pytest.raises(ValueError,match='owned decode'):send('decode',{}, {},timeout=1)
        assert stop.is_set()
    finally:
        if process.poll() is None:signal.pidfd_send_signal(descriptor,signal.SIGTERM,None,0)
        process.wait(timeout=5);os.close(descriptor);process.stdout.close();process.stderr.close()
