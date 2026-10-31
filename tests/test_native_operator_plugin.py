"""Plugin wiring with explicit CPU stand-ins. Owner: Hari Krishna Kancharla."""
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sdcguard.native_operator import plugin, config, transport


def module(monkeypatch, name, **members):
    parent, _, child=name.rpartition('.')
    if parent and parent not in sys.modules:module(monkeypatch,parent)
    value=ModuleType(name);value.__path__=[]
    for k,v in members.items():setattr(value,k,v)
    monkeypatch.setitem(sys.modules,name,value)
    if parent:monkeypatch.setattr(sys.modules[parent],child,value,raising=False)


def wiring(monkeypatch,tmp_path,role='decode'):
    monkeypatch.setattr(plugin,'_STATE',None)
    monkeypatch.setattr(plugin,'_CREATOR',os.getpid())
    (tmp_path/'artifacts').mkdir()
    (tmp_path/'artifacts/ENGINE_BUILD_LOCK.json').write_text('{"sources":{}}')
    monkeypatch.setattr(config,'BUNDLE',str(tmp_path))
    monkeypatch.setattr(config,'ENGINE_ROOT',str(tmp_path))
    for k,v in config.environment({},out=tmp_path/'capture',run_id='a'*32,packet_sha256='b'*64,
        arm_id='observer-on',engine_session_id='c'*32,role=role,enabled=True).items():
        if k.startswith('OWNER_'):monkeypatch.setenv(k,v)
    events=[]
    class Scheduler:
        def __init__(self):pass
        def shutdown(self,value):
            events.append('native-start')
            if getattr(self,'callback',None):self.callback()
            if getattr(self,'error',None):raise self.error
            events.append('native-return')
            return value
    class AsyncLLM:
        shutdown=Scheduler.shutdown
    class Request:pass
    class InputProcessor:pass
    class KVChecksumScheduler:pass
    class Observer:
        kind='checksum'
        def __init__(self,**context):self.context=context;self.closes=0
        def install(self,*args,**kwargs):events.append(('install',self.kind));return self
        def close(self):
            self.closes+=1;events.append(('close',self.kind))
            return dict(qualification='inert_cpu_wiring_only',closed=True,records=[])
    class Binding(Observer):kind='bindings'
    class Cache(Observer):kind='cache'
    module(monkeypatch,'vllm.v1.core.sched.scheduler',Scheduler=Scheduler)
    module(monkeypatch,'vllm.v1.engine.async_llm',AsyncLLM=AsyncLLM)
    module(monkeypatch,'vllm.v1.request',Request=Request)
    module(monkeypatch,'vllm.v1.engine.input_processor',InputProcessor=InputProcessor)
    module(monkeypatch,'vllm.distributed.kv_transfer.kv_checksum.scheduler',KVChecksumScheduler=KVChecksumScheduler)
    module(monkeypatch,'sdcguard.kv_checksum_observer',ChecksumObserver=Observer)
    module(monkeypatch,'sdcguard.checksum_client',RequestBindingObserver=Binding)
    module(monkeypatch,'sdcguard.prefix_cache_observer',PrefillCacheObserver=Cache)
    monkeypatch.setattr(plugin,'_guard',lambda cls,name,*args:vars(cls)[name])
    monkeypatch.setattr(transport,'install_policy',lambda callback:transport.profile())
    monkeypatch.setattr(plugin.State,'startup',lambda *args:None)
    plugin.register()
    assert plugin._STATE.errors==[]
    return SimpleNamespace(state=plugin._STATE,events=events,Scheduler=Scheduler,AsyncLLM=AsyncLLM,out=tmp_path/'capture')


@pytest.mark.parametrize('role',['prefill','decode'])
def test_cache_and_bindings_are_installed_and_closed_in_both_roles(monkeypatch,tmp_path,role):
    c=wiring(monkeypatch,tmp_path,role)
    kinds=['bindings','cache'] if role=='prefill' else ['checksum','bindings','cache']
    assert {v[1] for v in c.events if type(v) is tuple and v[0]=='install'}==set(kinds)
    scheduler,api=c.Scheduler(),c.AsyncLLM()
    token=object()
    assert scheduler.shutdown(token) is token and api.shutdown(token) is token
    for kind in kinds:
        record=json.loads((c.out/(kind+'-'+str(os.getpid())+'.json')).read_bytes())
        assert record['original_shutdown_completed'] is True and record['recorder_errors']==[]
        assert record['snapshot']['qualification']=='inert_cpu_wiring_only'
    assert c.events.index('native-return')<c.events.index(('close','cache'))
    assert c.state.cache.context['role']==role


def test_repeated_and_recursive_shutdown_closes_cache_once(monkeypatch,tmp_path):
    c=wiring(monkeypatch,tmp_path);scheduler=c.Scheduler()
    def nested():
        scheduler.callback=None
        assert scheduler.shutdown(4)==4
        assert c.state.cache.closes==0 and c.state.checksums.closes==0
    scheduler.callback=nested
    assert scheduler.shutdown(3)==3
    assert scheduler.shutdown(2)==2
    assert c.state.cache.closes==c.state.checksums.closes==1


def test_native_shutdown_error_is_preserved_in_cache_receipt(monkeypatch,tmp_path):
    c=wiring(monkeypatch,tmp_path);scheduler=c.Scheduler();failure=RuntimeError('inert CPU shutdown error')
    scheduler.error=failure
    with pytest.raises(RuntimeError) as caught:scheduler.shutdown(3)
    assert caught.value is failure
    for kind in ('cache','checksum'):
        record=json.loads((c.out/(kind+'-'+str(os.getpid())+'.json')).read_bytes())
        assert record['original_shutdown_completed'] is False and record['recorder_errors']


def test_plugin_import_and_registration_are_inert_without_run_context(monkeypatch):
    monkeypatch.delenv('OWNER_CHECKSUM_RUN_ID',raising=False)
    monkeypatch.setattr(plugin,'_STATE',None)
    plugin.register()
    assert plugin._STATE is None
