"""Explicit build selection and synthetic evidence controls.

Owner: Hari Krishna Kancharla.
Edited evidence below is a CPU contract fixture, never a new engine run.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from sdcguard import checksum_profiles as profiles, checksum_client as client
from sdcguard import checksum_observer_report as checks, prefix_cache_observer as cache
from sdcguard import required_gate as gate, transfer_coverage as transfer, transfer_upgrade as upgrade
from sdcguard import evidence_io as io
from sdcguard.native_operator import installed, command, config, preflight, runtime, workload
from test_kv_checksum_observer import native, _observer, _case, _verification
from test_prefix_cache_observer import classes, CONTEXT
from test_native_operator import installed_receipt

SAVED = Path(__file__).parent / 'data/checksum_owner_3a2ad847'
NAMES = ('RESULT.json', 'ordinary-evidence.zip', 'CPU_PREFLIGHT.json')


@pytest.fixture(scope='module')
def saved():
    return tuple((SAVED / n).read_bytes() for n in NAMES)


def rewrite(value, mapping):
    if type(value) is dict:
        return {k: rewrite(v, mapping) for k,v in value.items()}
    if type(value) is list:
        return [rewrite(v, mapping) for v in value]
    return mapping.get(value, value) if type(value) is str else value


def synthetic_side(saved, revision, suffix, gpu_blocks=None):
    """Assemble a labeled test input; no output is written or called measured."""
    raw_files = gate.read_archive(saved[1])
    plan = json.loads(raw_files['client/REQUEST_PLAN.json'])
    receipt = json.loads(saved[2])
    product = deepcopy(receipt['product'])
    product['files'] = installed.observer_files()
    product['sha256'] = io.digest(io.json_bytes(product['files']))
    mapping = {plan['run_id']: suffix*32, plan['packet_sha256']: product['sha256'],
               plan['arms'][0]['engine_session_id']: (suffix+'e')*16}
    documents = {n: rewrite(json.loads(raw), mapping) for n,raw in raw_files.items() if n.endswith('.json')}
    receipt = rewrite(receipt, mapping)
    result = rewrite(json.loads(saved[0]), mapping)
    plan = documents['client/REQUEST_PLAN.json']
    plan['source_revision'] = revision
    if gpu_blocks is not None:plan['gpu_blocks_override']=gpu_blocks
    plan_hash = io.digest(io.json_bytes(plan))
    documents['client/CLIENT_INVENTORY.json']['plan_sha256'] = plan_hash
    documents['CAPTURES.json']['plan_sha256'] = plan_hash
    for capture in documents['CAPTURES.json']['captures']:
        for role, part in (('decode',capture), ('prefill',capture['prefill_capture'])):
            part['source_revision'] = revision
            startup = part['startup']
            startup.update(source_revision=revision, source_profile=client.source_profile(revision),
                hisparse_host_num_blocks=None,
                defer_async_loads=False if revision == profiles.CANDIDATE else 'not_in_source_profile')
            if gpu_blocks is not None:startup['gpu_blocks']=gpu_blocks
            part['startup_sha256'] = io.digest(io.json_bytes(startup))
            name = capture['arm_id']+'/'+role+'/startup-'+str(startup['process_id'])+'.json'
            documents[name] = {k:deepcopy(v) for k,v in startup.items() if k != 'native_transport'}
            for kind in ('checksum','bindings','cache'):
                key = 'binding_snapshots' if kind == 'bindings' else kind+'_snapshots'
                for snapshot in part.get(key, []):
                    snapshot['source']['revision'] = revision
                    if kind == 'cache':
                        snapshot['source'] = dict(cache.source_profile(revision),supported=True)
                    pid = snapshot['capture_context']['process_id'] if kind == 'checksum' else snapshot['process_id']
                    name = capture['arm_id']+'/'+role+'/'+kind+'-'+str(pid)+'.json'
                    documents[name]['snapshot'] = deepcopy(snapshot)
    selected = profiles.get(revision)
    lock = installed.profile('ENGINE_BUILD_LOCK.json', revision)
    payload = installed.profile('PRECOMPILED_PAYLOAD.json', revision)
    receipt.update(source_revision=revision, product=product, plan_sha256=plan_hash)
    engine = receipt['installed_engine']
    engine.update(source_revision=revision, compiled_members=deepcopy(lock['base']['binary_members']))
    engine['versions']['vllm'] = selected['version']
    engine['source_files'] = {n:dict(bytes=1,sha256=row['sha256']) for n,row in lock['sources'].items()}
    engine['precompiled_payload']['extracted_members'] = deepcopy(payload['members'])
    ids = lock['base']['critical_identities'][revision]
    receipt['pinned_source_guard'].update(head=revision,pinned_base=lock['base']['revision'],ancestry=lock['base']['revision'],
        critical_identities={n:dict(expected=h,head=h,base=h) for n,h in ids.items()})
    receipt['build_witness'] = installed.witness(receipt)
    receipt['build_witness_sha256'] = io.digest(io.json_bytes(receipt['build_witness']))
    documents['BUILD_WITNESS.json'] = dict(witness=receipt['build_witness'],sha256=receipt['build_witness_sha256'])
    result['build_witness_sha256'] = receipt['build_witness_sha256']
    stream = BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for name,raw in raw_files.items():
            if name in documents:
                raw = io.json_bytes(documents[name])
            elif name.endswith('.jsonl'):
                raw = b''.join(json.dumps(rewrite(json.loads(line),mapping),sort_keys=True,separators=(',',':')).encode()+b'\n' for line in raw.splitlines())
            output.writestr(name,raw)
    result['archive_sha256'] = io.digest(stream.getvalue())
    return io.json_bytes(result),stream.getvalue(),io.json_bytes(receipt)


@pytest.fixture(scope='module')
def synthetic_pair(saved):
    return synthetic_side(saved,profiles.BASELINE,'a',8192),synthetic_side(saved,profiles.CANDIDATE,'b',8192)


def test_original_saved_record_still_has_exact_coverage_after_profile_addition(saved):
    value = transfer.evaluate(*saved)
    assert value['transfer_decision'] == 'PASS'
    assert value['full_prompt_gate']['decision'] == 'HOLD'
    assert value['full_prompt_gate']['passed_requests'] == 9
    assert value['summary']['verified_requests'] == 153
    assert value['summary']['compared_transfer_tokens'] == 9822
    assert value['summary']['decode_cached_tokens_outside_transfer'] == 99968


def test_second_profile_is_not_a_short_sha_or_moving_branch():
    for revision in ('main',profiles.CANDIDATE[:7],profiles.CANDIDATE.upper(),'f'*40,None,True):
        with pytest.raises(ValueError): profiles.get(revision)
    selected=profiles.get(profiles.CANDIDATE);selected['prefix_files'].clear()
    assert profiles.get(profiles.CANDIDATE)['prefix_files']


@pytest.mark.parametrize('options,coverage,native_result', [({},'complete','passed'),({'local_prefix':16},'partial','passed'),
    ({'producer_start':1},'unknown','passed'),({'empty_replicas':True},'unknown','passed'),
    ({'skipped':frozenset({0})},'unknown','unverified')])
def test_second_profile_delegates_real_checksum_functions_and_keeps_scope(native,options,coverage,native_result):
    observer = _observer(native,revision=profiles.CANDIDATE)
    case = _case(native,**options)
    result = case.engine.update_from_output(case.output,set())
    snapshot=observer.close();row=_verification(snapshot)
    assert snapshot['source']['revision'] == profiles.CANDIDATE
    assert snapshot['qualification'] == 'cpu_source_functions' and snapshot['release_qualifying'] is False
    assert row['native_result'] == native_result
    assert result == ({'synthetic-ordinary-request'} if native_result == 'unverified' else set())
    if coverage == 'complete': assert row['coverage'] == 'complete'
    else: assert row['coverage'] != 'complete'
    assert 'declared_source_identity_unverified' not in checks.evaluate(io.json_bytes(snapshot))['hold_reasons']
    assert 'declared_source_identity_unverified' in checks.evaluate(io.json_bytes(snapshot),revision=profiles.BASELINE)['hold_reasons']


def test_new_core_scheduler_source_binds_native_prefill_callback(tmp_path):
    primary=Path(__file__).parent/'data/prefix_cache_functions/PRIMARY_1a0c6954.json'
    Request,Scheduler,Stats,Status=classes(tmp_path,primary)
    observer=cache.PrefillCacheObserver(**CONTEXT,role='decode',revision=profiles.CANDIDATE)
    observer.install_cpu_source_functions(Request,Scheduler,source_root=tmp_path)
    request=Request();request.request_id='cpu-scope';request.prompt_token_ids=list(range(31));request.num_prompt_tokens=31
    request.prefill_stats=Stats();request.prefill_stats.set(31,16,15)
    request.num_preemptions=0;request.status=Status.RUNNING;request._output_token_ids=[5]
    expected=request.prefill_stats
    assert request.take_prefill_stats() is expected and request.prefill_stats is None
    request.status=Status.FINISHED_LENGTH_CAPPED
    assert request.take_prefill_stats() is None
    snapshot=observer.close()
    assert cache.evaluate_snapshot(snapshot,CONTEXT,'decode',profiles.CANDIDATE)[1] == []
    assert cache.evaluate_snapshot(snapshot,CONTEXT,'decode',profiles.BASELINE)[1]


@pytest.mark.parametrize('field,value', [('hisparse_host_num_blocks','missing'),('hisparse_host_num_blocks',1),
    ('hisparse_host_num_blocks',False),('defer_async_loads','missing'),('defer_async_loads',True),
    ('defer_async_loads',0),('defer_async_loads',None)])
def test_new_startup_never_infers_disabled_features(field,value):
    startup=dict(hisparse_host_num_blocks=None,defer_async_loads=False)
    assert profiles.startup_supported(startup,profiles.CANDIDATE)
    if value == 'missing':startup.pop(field)
    else:startup[field]=value
    assert not profiles.startup_supported(startup,profiles.CANDIDATE)


def test_legacy_recorder_inventory_cannot_qualify_candidate_or_edited_source(saved):
    product=json.loads(saved[2])['product']
    assert installed.recorder_matches(product,profiles.BASELINE)
    assert not installed.recorder_matches(product,profiles.CANDIDATE)
    product['files']['sdcguard/native_operator/plugin.py']['sha256']='0'*64
    assert not installed.recorder_matches(product,profiles.BASELINE)


def test_two_synthetic_build_contracts_reuse_the_same_rules(synthetic_pair):
    before,after=synthetic_pair
    for raw in synthetic_pair:
        answer=transfer.evaluate(*raw)
        assert answer['transfer_decision'] == 'PASS',answer['hold_reasons']
        assert answer['full_prompt_gate']['decision'] == 'HOLD'
        assert answer['full_prompt_gate']['passed_requests'] == 9
    answer=upgrade.evaluate(before,after)
    assert answer['transfer_preservation_decision'] == 'PASS',answer['hold_reasons']
    assert answer['upgrade_decision'] == 'HOLD'
    assert answer['summary']['preserved'] == 153 and answer['summary']['unverifiable'] == 0
    assert upgrade.evaluate(after,before)['transfer_preservation_decision'] == 'HOLD'


def test_candidate_cannot_use_self_consistent_legacy_preflight(synthetic_pair):
    result_raw,archive_raw,receipt_raw=synthetic_pair[1]
    receipt=json.loads(receipt_raw);result=json.loads(result_raw)
    receipt.update(schema='sdcguard.owner-checksum-cpu-preflight.v1',
        build_result_sha256='1'*64,build_archive_sha256='2'*64)
    receipt['installed_engine']['schema']='sdcguard.owner-checksum-installed.v1'
    witness=dict(schema='sdcguard.owner-checksum-build-witness.v1',
        packet_sha256=receipt['packet_sha256'],
        installed_identity_sha256=io.digest(io.json_bytes(receipt['installed_engine'])),
        build_result_sha256=receipt['build_result_sha256'],build_archive_sha256=receipt['build_archive_sha256'],
        pinned_source_guard_sha256=io.digest(io.json_bytes(receipt['pinned_source_guard'])),
        python_identity_sha256=io.digest(io.json_bytes(receipt['python_runtime']['identity'])))
    receipt['build_witness']=witness
    receipt['build_witness_sha256']=io.digest(io.json_bytes(witness))
    result['build_witness_sha256']=receipt['build_witness_sha256']
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for name,raw in gate.read_archive(archive_raw).items():
            if name == 'BUILD_WITNESS.json':raw=io.json_bytes(dict(witness=witness,sha256=receipt['build_witness_sha256']))
            output.writestr(name,raw)
    result['archive_sha256']=io.digest(stream.getvalue())
    answer=transfer.evaluate(io.json_bytes(result),stream.getvalue(),io.json_bytes(receipt))
    assert answer['transfer_decision'] == 'HOLD'
    assert 'candidate_requires_installed_runtime_preflight' in answer['hold_reasons']


def test_offline_profile_selection_is_order_thread_and_environment_independent(synthetic_pair,monkeypatch):
    monkeypatch.setenv('SDCGUARD_OPERATOR_REVISION','untrusted-moving-branch')
    a,b=synthetic_pair
    first=transfer.evaluate(*a)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda raw:transfer.evaluate(*raw),(a,b,a)))
    assert results[0] == results[2] == first
    assert results[1]['engine']['source_revision'] == profiles.CANDIDATE
    assert all(r['transfer_decision'] == 'PASS' for r in results)


@pytest.mark.parametrize('field', ['source_revision','version','compiled','source','critical','product'])
def test_mixed_build_receipts_hold_even_with_recomputed_local_witness(synthetic_pair,field):
    raw=list(synthetic_pair[1]);value=json.loads(raw[2]);engine=value['installed_engine']
    if field == 'source_revision':value[field]=profiles.BASELINE
    elif field == 'version':engine['versions']['vllm']=profiles.get(profiles.BASELINE)['version']
    elif field == 'compiled':engine['compiled_members']=installed.profile('ENGINE_BUILD_LOCK.json')['base']['binary_members']
    elif field == 'source':next(iter(engine['source_files'].values()))['sha256']='0'*64
    elif field == 'critical':next(iter(value['pinned_source_guard']['critical_identities'].values()))['head']='0'*40
    else:value['product']['files']['sdcguard/checksum_profiles.py']['sha256']='0'*64
    value['build_witness']=installed.witness(value)
    value['build_witness_sha256']=io.digest(io.json_bytes(value['build_witness']))
    files=gate.read_archive(raw[1]);plan_raw=files['client/REQUEST_PLAN.json']
    assert not installed.validate(value,json.loads(plan_raw),plan_raw)
    stream=BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as output:
        for name,payload in files.items():
            if name == 'BUILD_WITNESS.json':
                payload=io.json_bytes(dict(witness=value['build_witness'],sha256=value['build_witness_sha256']))
            output.writestr(name,payload)
    result=json.loads(raw[0]);result['build_witness_sha256']=value['build_witness_sha256']
    result['archive_sha256']=io.digest(stream.getvalue())
    raw[0],raw[1]=io.json_bytes(result),stream.getvalue()
    raw[2]=io.json_bytes(value)
    answer=transfer.evaluate(*raw)
    assert answer['transfer_decision'] == 'HOLD'
    assert 'installed_runtime_preflight_unverified' in answer['hold_reasons']


def test_operator_passes_explicit_revision_to_cpu_preflight_and_children(tmp_path,monkeypatch):
    def fake(*,revision):
        assert revision == profiles.CANDIDATE
        return dict(status='UNKNOWN',failure='CPU control'),None,None
    monkeypatch.setattr(preflight,'execute',fake)
    monkeypatch.setattr(runtime,'execute',lambda *a,**k:pytest.fail('runtime must not start'))
    assert command.main(['--out',str(tmp_path/'out'),'--require','checksum','--engine-revision',profiles.CANDIDATE,'--preflight-only']) == 2
    env=config.environment({'SDCGUARD_OPERATOR_REVISION':profiles.BASELINE},out=tmp_path,
        run_id='a'*32,packet_sha256='b'*64,arm_id='observer-on',engine_session_id='c'*32,role='decode',enabled=True,
        revision=profiles.CANDIDATE)
    assert env['SDCGUARD_OPERATOR_REVISION'] == profiles.CANDIDATE
    assert env['PYTHONDONTWRITEBYTECODE'] == '1'


@pytest.mark.parametrize('role',['prefill','decode'])
def test_paired_capacity_is_an_explicit_native_argument(role):
    assert '--num-gpu-blocks-override' not in config.server_command(role)
    command=config.server_command(role,gpu_blocks=8192)
    assert command[command.index('--num-gpu-blocks-override')+1]=='8192'
    assert '--enable-prefix-caching' in command


@pytest.mark.parametrize('value',[True,0,-1,8191,8192.0,'8192',None])
def test_paired_plan_rejects_non_exact_capacity(saved,value):
    plan=json.loads(gate.read_archive(saved[1])['client/REQUEST_PLAN.json'])
    plan['gpu_blocks_override']=value
    with pytest.raises(io.EvidenceError,match='capacity'):client._planned(plan)
    if value is not None:
        with pytest.raises(ValueError,match='8192'):config.server_command('decode',gpu_blocks=value)


@pytest.mark.parametrize('role',['prefill','decode'])
def test_observed_capacity_must_equal_declared_capacity(synthetic_pair,role):
    files=gate.read_archive(synthetic_pair[1][1]);plan=json.loads(files['client/REQUEST_PLAN.json']);arm=plan['arms'][0]
    name=next(name for name in files if name.startswith('observer-on/'+role+'/startup-'))
    startup=json.loads(files[name])
    assert gate._startup(startup,plan,arm,role)
    # Both admission and offline checks bind the observed native pool count.
    startup['actual_model']=startup['actual_tokenizer']=config.SNAPSHOT
    assert runtime.startup_valid(startup,plan=plan,arm=arm,role=role)
    for value in (None,True,8191,8193):
        wrong=dict(startup,gpu_blocks=value)
        assert not gate._startup(wrong,plan,arm,role)
        assert not runtime.startup_valid(wrong,plan=plan,arm=arm,role=role)


def test_cli_threads_capacity_into_the_cpu_plan(tmp_path,monkeypatch):
    def fake(*,revision,gpu_blocks):
        assert revision==profiles.CANDIDATE and gpu_blocks==8192
        return dict(status='UNKNOWN',failure='CPU control'),None,None
    monkeypatch.setattr(preflight,'execute',fake)
    monkeypatch.setattr(runtime,'execute',lambda *a,**k:pytest.fail('must remain CPU-only'))
    assert command.main(['--out',str(tmp_path/'out'),'--require','checksum','--engine-revision',profiles.CANDIDATE,
        '--gpu-blocks','8192','--preflight-only'])==2


def test_cpu_plan_records_capacity_without_assuming_cache_hits():
    import hashlib
    def encode(text,**kwargs):
        return [int.from_bytes(hashlib.sha256(word.encode()).digest()[:4],'big')%1000000 for word in text.split()]
    args=dict(run_id='a'*32,packet_sha256='b'*64,engine_sessions={'observer-on':'c'*32},gpu_blocks=8192)
    plan,payloads=workload.build_plan(encode,**args)
    assert plan['gpu_blocks_override']==8192
    workload.validate_payloads(plan,payloads)
    assert all('cache_hit' not in row for row in plan['arms'][0]['requests'])


def test_cpu_plan_refuses_capacity_shortfall_without_assuming_reuse():
    with pytest.raises(ValueError,match='no-sharing cache capacity'):
        workload.build_plan(lambda text,**kwargs:[ord(c) for c in text],run_id='a'*32,
            packet_sha256='b'*64,engine_sessions={'observer-on':'c'*32},gpu_blocks=8192)
