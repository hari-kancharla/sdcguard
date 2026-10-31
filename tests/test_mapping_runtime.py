"""Host-only source and observed-header controls. Owner: Hari Krishna Kancharla."""
from dataclasses import FrozenInstanceError
from enum import Enum
import hashlib
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace as NS
import weakref

import pytest

from sdcguard.lifetime import mapping_runtime as R
from sdcguard.lifetime import assignment_init as I
from sdcguard.lifetime import assignment_obligations as A
from test_vllm_forward import environment


class Fields:
    def __init__(self, **values): self.__dict__.update(values)


def header_environment(tmp_path):
    """Reusable actual-field host fixture. It is not native engine evidence."""
    env = environment(tmp_path)
    env.installation.detach()
    runner, types = env.runner, dict(env.types)
    class Backend:
        @staticmethod
        def get_name(): return "FLASH_ATTN"
    class AttentionGroup(Fields): pass
    class TorchVersion(str): pass
    types.update(backend=Backend, attention_group=AttentionGroup)
    types["torch"].__version__ = TorchVersion("2.8.0+cpu")
    runner.parallel_config.ubatch_size = 0
    runner.parallel_config.worker_extension_cls = ""
    runner.compilation_config.cudagraph_mode = types["graph_none"]
    runner.vllm_config.speculative_config = None
    runner.vllm_config.lora_config = None
    runner.vllm_config.kernel_config = NS(enable_jit_warmup=False)
    for layer in runner.compilation_config.static_forward_context.values():
        layer.attn_backend = Backend
    group = runner.kv_cache_config.kv_cache_groups[0]
    runner.attn_groups = [[AttentionGroup(
        backend=Backend, kv_cache_group_id=0, kernel_block_size=None, kernel_block_stride=0,
        kernel_block_table=None, kernel_block_offsets=None, kv_cache_spec=group.kv_cache_spec,
        layer_names=list(group.layer_names))]]
    core, executor, wrapper = Fields(), Fields(), Fields()
    worker = Fields(model_runner=runner, use_v2_model_runner=True)
    scheduler, manager, coordinator, manager_group, pool = [Fields() for _ in range(5)]
    core.model_executor, core.scheduler = executor, scheduler
    executor.driver_worker, wrapper.worker = wrapper, worker
    scheduler.kv_cache_manager = manager
    manager.coordinator, manager.block_pool = coordinator, pool
    coordinator.block_pool, coordinator.single_type_managers = pool, (manager_group,)
    manager_group.block_pool = pool
    native = I.NativeBindings(
        core_module=ModuleType("host.core"), manager_module=ModuleType("host.manager"),
        coordinator_module=ModuleType("host.coordinator"), core=Fields, core_proc=Fields,
        executor=Fields, uniproc=Fields, wrapper=Fields, worker=Fields, runner=type(runner),
        scheduler=Fields, async_scheduler=Fields, manager=Fields, coordinator=Fields,
        group_manager=Fields, pool=Fields)
    participants = dict(core=core, executor=executor, wrapper=wrapper, worker=worker, runner=runner,
                        scheduler=scheduler, manager=manager, coordinator=coordinator,
                        group_manager=manager_group, pool=pool)
    bindings = R.MappingBindings(native, Fields, Fields, Fields, types)
    registration = NS(runner_class=type(runner), engine=MappingProxyType(dict(
        name="vllm", version="0.32.0", source_sha256=env.engine["source_sha256"],
        dependencies={"python":"untrusted", "torch":"untrusted", "flash_attn":"untrusted"})))
    return NS(runner=runner, worker=worker, types=types, bindings=bindings,
              registration=registration, participant_refs=MappingProxyType({k:weakref.ref(v) for k,v in participants.items()}),
              participants=participants, env=env)


def test_header_is_rederived_from_native_fields_and_contains_only_host_values(tmp_path):
    s = header_environment(tmp_path)
    profile, dependencies = R.observed_header(s.registration, s.bindings, s.participant_refs)
    assert profile == {**A.adapter_contract(R.NATIVE_ADAPTER)["profile"],
                       "model_revision": "c"*40, "layer_names": list(s.runner.kv_cache_config.kv_cache_groups[0].layer_names)}
    assert dependencies == dict(python=platform.python_version(), torch="2.8.0+cpu", flash_attn="3")
    assert all(layer.kv_cache.copies == 0 for layer in s.runner.compilation_config.static_forward_context.values())
    profile["layer_names"].clear()
    assert s.runner.kv_cache_config.kv_cache_groups[0].layer_names
    json.dumps(dependencies)


def test_forward_gate_constant_filled_return_is_never_used(tmp_path, monkeypatch):
    from sdcguard.lifetime import forward_inputs
    s = header_environment(tmp_path)
    monkeypatch.setattr(forward_inputs, "SUPPORTED_PROFILE", {"model":"invented", "tp":99})
    actual, _ = R.observed_header(s.registration, s.bindings, s.participant_refs)
    assert actual["model"] == s.runner.model_config.model and actual["tp"] == 1


@pytest.mark.parametrize("revision", ["main", "c1899de2", "x"*40, None, True])
def test_missing_or_mutable_revision_is_not_filled(tmp_path, revision):
    s = header_environment(tmp_path)
    s.runner.model_config.revision = revision
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, s.participant_refs)


@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_actual_jit_warmup_must_be_the_supported_false_value(tmp_path, value):
    s = header_environment(tmp_path)
    s.runner.vllm_config.kernel_config.enable_jit_warmup = value
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, s.participant_refs)


def test_missing_jit_warmup_is_not_filled_from_requested_configuration(tmp_path):
    s = header_environment(tmp_path)
    del s.runner.vllm_config.kernel_config.enable_jit_warmup
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, s.participant_refs)


@pytest.mark.parametrize("change", [
    lambda s:setattr(s.runner.parallel_config,"ubatch_size",2),
    lambda s:setattr(s.runner.parallel_config,"worker_extension_cls","extension"),
    lambda s:setattr(s.runner.compilation_config,"cudagraph_mode",object()),
    lambda s:setattr(s.runner.attn_groups[0][0],"kernel_block_stride",2),
    lambda s:setattr(s.runner.attn_groups[0][0],"layer_names",[]),
    lambda s:setattr(s.runner.model_config.hf_text_config,"num_hidden_layers",True),
    lambda s:setattr(next(iter(s.runner.compilation_config.static_forward_context.values())).impl,
                     "vllm_flash_attn_version",4),
    lambda s:delattr(s.runner.model_config,"revision"),
])
def test_unsupported_or_missing_actual_profile_is_unavailable(tmp_path, change):
    s = header_environment(tmp_path)
    change(s)
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, s.participant_refs)


def test_header_requires_exact_live_runner_and_immutable_weak_participants(tmp_path):
    s = header_environment(tmp_path)
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, dict(s.participant_refs))
    wrong = Fields()
    refs = dict(s.participant_refs, runner=weakref.ref(wrong))
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, MappingProxyType(refs))
    refs = dict(s.participant_refs, runner=lambda:s.runner)
    with pytest.raises(R.MappingRuntimeGap):
        R.observed_header(s.registration, s.bindings, MappingProxyType(refs))


def test_bindings_are_frozen_and_type_inventory_is_not_shared(tmp_path):
    s = header_environment(tmp_path)
    with pytest.raises(FrozenInstanceError): s.bindings.request = Fields
    with pytest.raises(TypeError): s.bindings.profile_types["torch"] = object()
    s.types.clear()
    assert s.bindings.profile_types["torch"].__version__ == "2.8.0+cpu"


def source_fixture(tmp_path, monkeypatch):
    files = {"vllm/v1/request.py":b"request source\n", "vllm/v1/engine/core.py":b"core source\n"}
    pins = {k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    monkeypatch.setattr(A,"source_pins",lambda adapter=None:dict(pins))
    for name, raw in files.items():
        target = tmp_path/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(raw)
    return pins


def test_source_validation_reads_actual_bytes_and_requires_request(tmp_path, monkeypatch):
    expected = source_fixture(tmp_path, monkeypatch)
    assert R.validate_sources(tmp_path) == expected
    (tmp_path/"vllm/v1/request.py").unlink()
    with pytest.raises(R.MappingRuntimeGap): R.validate_sources(tmp_path)


def test_source_mismatch_and_symlink_are_rejected(tmp_path, monkeypatch):
    source_fixture(tmp_path, monkeypatch)
    target = tmp_path/"vllm/v1/request.py"
    target.write_bytes(b"changed\n")
    with pytest.raises(R.MappingRuntimeGap): R.validate_sources(tmp_path)
    target.unlink()
    target.symlink_to(tmp_path/"vllm/v1/engine/core.py")
    with pytest.raises(R.MappingRuntimeGap): R.validate_sources(tmp_path)


def loaded_fixture(tmp_path, monkeypatch):
    names = {
        "vllm.v1.engine.core":("EngineCore","EngineCoreProc"),
        "vllm.v1.executor.abstract":("Executor",),
        "vllm.v1.executor.uniproc_executor":("UniProcExecutor",),
        "vllm.v1.worker.worker_base":("WorkerWrapperBase",),
        "vllm.v1.worker.gpu_worker":("Worker",),
        "vllm.v1.worker.gpu.model_runner":("GPUModelRunner",),
        "vllm.v1.core.sched.scheduler":("Scheduler",),
        "vllm.v1.core.sched.async_scheduler":("AsyncScheduler",),
        "vllm.v1.core.kv_cache_manager":("KVCacheManager",),
        "vllm.v1.core.kv_cache_coordinator":("KVCacheCoordinatorNoPrefixCache",),
        "vllm.v1.core.single_type_kv_cache_manager":("FullAttentionManager","SingleTypeKVCacheManager"),
        "vllm.v1.core.block_pool":("BlockPool",),"vllm.v1.request":("Request",),
        "vllm.v1.core.sched.output":("SchedulerOutput",),
        "vllm.v1.kv_cache_interface":("FullAttentionSpec",),
        "vllm.v1.worker.utils":("AttentionGroup",),
        "vllm.model_executor.models.qwen3":("Qwen3ForCausalLM",),
        "vllm.model_executor.layers.attention.attention":("Attention",),
        "vllm.v1.worker.gpu.model_states.default":("DefaultModelState",),
        "vllm.v1.attention.backend":(),
        "vllm.v1.attention.backends.flash_attn":("FlashAttentionImpl","FlashAttentionMetadata","FlashAttentionBackend"),
    }
    modules, pins = {}, {}
    for name, classes in names.items():
        relative = name.replace(".","/")+".py"
        path = tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        raw = ("# host source identity: "+name+"\n").encode()
        path.write_bytes(raw)
        pins[relative] = hashlib.sha256(raw).hexdigest()
        module = modules[name] = ModuleType(name)
        module.__file__ = str(path)
        module.__spec__ = ModuleSpec(name,loader=None,origin=str(path))
        for cls in classes: setattr(module,cls,type(cls,(),{"__module__":name}))
    runner = modules["vllm.v1.worker.gpu.model_runner"]
    runner.CompilationMode = Enum("CompilationMode",{"NONE":0},module="vllm.config.compilation")
    runner.CUDAGraphMode = Enum("CUDAGraphMode",{"NONE":0},module="vllm.config.compilation")
    modules["vllm.v1.kv_cache_interface"].KVQuantMode = Enum("KVQuantMode",{"NONE":0})
    modules["vllm.v1.attention.backend"].AttentionType = Enum("AttentionType",{"DECODER":"decoder"})
    package = modules["vllm"] = ModuleType("vllm")
    package.__file__, package.__version__ = str(tmp_path/"vllm/__init__.py"),"0.32.0"
    torch = modules["torch"] = ModuleType("torch")
    torch.__version__ = "2.8.0+cpu"
    monkeypatch.setattr(A,"source_pins",lambda adapter=None:dict(pins))
    original_import = R.importlib.import_module
    monkeypatch.setattr(R.importlib,"import_module",
                        lambda name,*args,**kwargs:modules[name] if name in modules
                        else original_import(name,*args,**kwargs))
    registration = NS(runner_class=runner.GPUModelRunner,
                      engine=dict(name="vllm",version="0.32.0",source_sha256=dict(pins)))
    return NS(modules=modules, registration=registration, pins=pins)


def test_explicit_loading_resolves_canonical_classes_without_wrapping(tmp_path,monkeypatch):
    s = loaded_fixture(tmp_path,monkeypatch)
    result = R.load_bindings(tmp_path,"0.32.0",s.registration)
    assert result.initialization.runner is s.registration.runner_class
    assert result.request is s.modules["vllm.v1.request"].Request
    assert result.scheduler_output is s.modules["vllm.v1.core.sched.output"].SchedulerOutput
    assert result.manager_base is s.modules["vllm.v1.core.single_type_kv_cache_manager"].SingleTypeKVCacheManager
    assert not any(hasattr(cls,"_initialization_observer") for cls in (result.request,result.scheduler_output))


@pytest.mark.parametrize("change,reason", [
    (lambda s:setattr(s.registration,"runner_class",Fields), "registered runner"),
    (lambda s:setattr(s.modules["vllm.v1.request"],"__file__","/other/request.py"), "module path"),
    (lambda s:setattr(s.modules["vllm.v1.request"].Request,"__module__","other"), "class differs"),
    (lambda s:setattr(s.modules["vllm"],"__version__","0.33.0"), "loaded engine version"),
    (lambda s:s.registration.engine.update(version="0.33.0"), "registration engine"),
])
def test_loading_rejects_wrong_loaded_provenance_or_registration(tmp_path,monkeypatch,change,reason):
    s = loaded_fixture(tmp_path,monkeypatch)
    change(s)
    with pytest.raises(R.MappingRuntimeGap,match=reason): R.load_bindings(tmp_path,"0.32.0",s.registration)


def test_module_import_is_stdlib_only():
    code = '''import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','vllm','modal','requests','httpx'} or fullname in {
   'sdcguard.lifetime.assignment_init','sdcguard.lifetime.assignment_obligations','sdcguard.lifetime.vllm_forward'}:
   raise AssertionError('unexpected eager import: '+fullname)
sys.meta_path.insert(0,Guard())
from sdcguard.lifetime import mapping_runtime
assert mapping_runtime.NATIVE_ADAPTER == 'vllm.0.32.uniproc.eager.native.v1'
'''
    result = subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
