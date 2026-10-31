"""Opt-in plugin controls on host objects, not an engine run.

Owner: Hari Krishna Kancharla.
"""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from types import SimpleNamespace as NS
import zipfile

import pytest

from sdcguard.lifetime import assignment_obligations as C
from sdcguard.lifetime import mapping_activation as A
from sdcguard.lifetime import mapping_observer as M
from sdcguard.lifetime import vllm_forward as V
from test_assignment_init import primary
from test_mapping_observer import observer_environment, run, Mode


@pytest.fixture(autouse=True)
def isolated_activation(monkeypatch):
    for name in ("SDCGUARD_ASSIGNMENT_DIR", "SDCGUARD_FORWARD_DIR", "SDCGUARD_LIFETIME_DIR", "SDCGUARD_REQUIRE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SDCGUARD", "0")
    monkeypatch.setattr(A, "_ACTIVATION", None)
    monkeypatch.setattr(A, "_INSTALLING", False)
    monkeypatch.setattr(V, "_INSTALLATION", None)
    yield
    if V._INSTALLATION is not None and hasattr(V._INSTALLATION, "detach"):
        V.uninstall()


def builder(monkeypatch):
    def install(value, runner, forward, engine, types, context, factory, directory, source_root):
        monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(forward))
        monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(directory))
        modules = {
            "vllm": NS(__version__="0.32.0", __file__=str(source_root / "vllm/__init__.py")),
            "vllm.v1.worker.gpu.model_runner": NS(GPUModelRunner=runner),
            "vllm.v1.attention.backends.flash_attn": NS(
                FlashAttentionImpl=types["impl"], FlashAttentionMetadata=types["metadata"],
                AttentionType=NS(DECODER=types["decoder"])),
            "vllm.model_executor.layers.attention.attention": NS(Attention=types["attention"]),
            "vllm.model_executor.models.qwen3": NS(Qwen3ForCausalLM=types["model"]),
            "vllm.v1.worker.gpu.model_states.default": NS(DefaultModelState=types["model_state"]),
            "vllm.forward_context": NS(get_forward_context=context),
            "vllm.config.compilation": NS(CompilationMode=Mode, CUDAGraphMode=Mode),
            "vllm.v1.kv_cache_interface": NS(FullAttentionSpec=types["spec"], KVQuantMode=Mode),
            "torch": types["torch"],
        }
        types["torch"].nn = NS(modules=NS(module=types["global_hook_registry"]))
        original_import = V.importlib.import_module
        monkeypatch.setattr(V.importlib, "import_module", lambda name, *a, **k:
                            modules[name] if name in modules else original_import(name, *a, **k))
        roots = []
        def selected(path, root, **kwargs):
            assert Path(path) == directory and Path(root) == source_root
            roots.append(root)
            return lambda registration, health:factory(registration, health, **kwargs)
        monkeypatch.setattr(M, "mapping_factory", selected)
        result = A.install()
        assert result["status"] == "registered" and result["verdict"] == "unknown"
        assert roots == [source_root]
        assert V._INSTALLATION is not None
        return V._INSTALLATION
    return install


@pytest.fixture
def active(primary, tmp_path, monkeypatch):
    value = observer_environment(primary, tmp_path, installation_builder=builder(monkeypatch), journal=True)
    try:
        yield value
    finally:
        value.installation.detach()


def test_disabled_module_and_install_import_no_engine_or_device():
    code = """
import sys, importlib.abc, os
class Denied(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('vllm','torch'):
   raise AssertionError('runtime import while disabled')
sys.meta_path.insert(0,Denied())
os.environ.pop('SDCGUARD_ASSIGNMENT_DIR',None)
from sdcguard.lifetime import mapping_activation as a
assert a.install()['status']=='disabled'
assert a.status()['status']=='disabled'
assert a.uninstall()['status']=='disabled'
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


@pytest.mark.parametrize("kind", ["missing_forward", "same_directory", "symlink_directory"])
def test_invalid_explicit_directories_stay_unknown_without_engine_import(tmp_path, monkeypatch, kind):
    assignment = tmp_path / "assignment"
    monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(assignment))
    if kind == "same_directory":
        monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(assignment))
    elif kind == "symlink_directory":
        destination = tmp_path / "destination"; destination.mkdir()
        assignment.symlink_to(destination, target_is_directory=True)
        monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(tmp_path / "forward"))
    monkeypatch.setattr(V.importlib, "import_module", lambda *a: pytest.fail("unexpected engine import"))
    result = A.install()
    assert result["status"] == "unknown" and result["verdict"] == "unknown"
    assert V._INSTALLATION is None


def test_exact_source_mismatch_has_no_hooks_or_optimistic_registration(primary, tmp_path, monkeypatch):
    source_root = tmp_path / "wrong-source"
    shutil.copytree(primary[0], source_root)
    (source_root / "vllm/v1/worker/gpu/model_runner.py").write_bytes(b"different source\n")
    monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(tmp_path / "assignment"))
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(tmp_path / "forward"))
    package = NS(__version__="0.32.0", __file__=str(source_root / "vllm/__init__.py"))
    def imported(name):
        assert name == "vllm"
        return package
    monkeypatch.setattr(V.importlib, "import_module", imported)
    assert A.install()["status"] == "unknown"
    assert A.status()["status"] == "unknown" and V._INSTALLATION is None


def test_existing_forward_only_installation_is_preserved_and_never_retrofitted(tmp_path, monkeypatch):
    original = NS(pid=os.getpid(), assignment_factory=None)
    monkeypatch.setattr(V, "_INSTALLATION", original)
    monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(tmp_path / "assignment"))
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(tmp_path / "forward"))
    monkeypatch.setattr(V, "install", lambda **kwargs: pytest.fail("must not replace existing hooks"))
    assert A.install()["status"] == "unknown" and V._INSTALLATION is original


def test_partial_owned_registration_is_cleaned_up_without_removing_a_foreign_installation(tmp_path, monkeypatch):
    monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(tmp_path / "assignment"))
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", str(tmp_path / "forward"))
    class Incomplete:
        pass
    removed = []
    def install(*, assignment_factory):
        owned = Incomplete()
        owned.pid, owned.assignment_factory = os.getpid(), assignment_factory
        owned.bridge, owned.bridge_error = None, "assignment setup is unsupported"
        V._INSTALLATION = owned
        return {"status":"installed"}
    def cleanup():
        removed.append(V._INSTALLATION)
        V._INSTALLATION = None
    monkeypatch.setattr(V, "install", install)
    monkeypatch.setattr(V, "uninstall", cleanup)
    result = A.install()
    assert result["status"] == "unknown" and len(removed) == 1 and V._INSTALLATION is None


def test_repeated_plugin_registration_and_primary_loader_keep_one_shared_hook(active):
    from sdcguard import vllm_plugin
    original = dict(active.installation.wrappers)
    calls = []
    with zipfile.ZipFile(Path(__file__).parent / "fixtures/assignment_init/primary.zip") as archive:
        tree = ast.parse(archive.read("vllm/plugins/__init__.py"))
    method = deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_general_plugins"))
    def plugin():
        calls.append(None)
        vllm_plugin.register()
    namespace = dict(plugins_loaded=False, DEFAULT_PLUGINS_GROUP="vllm.general_plugins",
                     load_plugins_by_group=lambda **kwargs:{"sdcguard":plugin})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "<retained primary plugin loader>", "exec"), namespace)
    namespace["load_general_plugins"]()
    namespace["load_general_plugins"]()
    vllm_plugin.register()
    assert calls == [None] and active.installation.wrappers == original
    assert len(active.runner.model._forward_pre_hooks) == 1
    assert A.status()["healthy"] and A.status()["verdict"] == "unknown"
    run(active)
    active.runner.shutdown()
    status = A.status()
    assert status["status"] == "closed" and status["healthy"]
    assert status["observer"]["sealed"]
    assert Path(status["observer"]["artifacts"]["path"]).is_file()
    assert C.evaluate([json.loads(line) for line in active.sinks[0].path.read_text().splitlines()])["verdict"] == "satisfied"
    assert len(list((Path(status["directories"]["forward"])).glob("*.jsonl"))) == 1


def test_changed_directory_does_not_start_a_second_recording(active, tmp_path, monkeypatch):
    selected = tmp_path / "another-assignment"
    monkeypatch.setenv("SDCGUARD_ASSIGNMENT_DIR", str(selected))
    assert A.install()["status"] == "unknown" and not selected.exists()
    run(active)
    active.runner.shutdown()
    assert not active.mapping.status["sealed"]


def test_inherited_activation_never_calls_parent_observer_or_cleanup(active, monkeypatch):
    original = V._INSTALLATION
    with monkeypatch.context() as changed:
        changed.setattr(A.os, "getpid", lambda:A._PID + 100)
        assert A.install()["status"] == A.status()["status"] == A.uninstall()["status"] == "unknown"
        assert V._INSTALLATION is original
    run(active)
    active.runner.shutdown()
    assert active.mapping.status["sealed"]


def test_uninstall_closes_owned_bridge_and_preserves_native_return(active):
    run(active)
    assert active.runner.shutdown() is active.runner.shutdown_result
    result = A.uninstall()
    assert result["observer"]["sealed"] and result["healthy"]
    assert V._INSTALLATION is None and not active.installation.wrappers
    assert A.uninstall()["observer"]["sealed"]


def test_replaced_installation_is_not_removed_by_activation_cleanup(active, monkeypatch):
    foreign = NS(pid=os.getpid(), assignment_factory=None)
    monkeypatch.setattr(V, "_INSTALLATION", foreign)
    assert A.uninstall()["status"] == "unknown" and V._INSTALLATION is foreign


def test_raw_healthy_footer_cannot_override_failed_caller_close(active):
    run(active)
    sink = active.sinks[0]
    close = sink.close
    def failed_acknowledgement(timeout=1.0):
        assert close(timeout=timeout) is True
        return False
    sink.close = failed_acknowledgement
    active.runner.shutdown()
    records = [json.loads(line) for line in sink.path.read_text().splitlines()]
    assert C.evaluate(records)["verdict"] == "satisfied" and sink.status["published"]
    status = A.status()
    assert status["observer"]["sealed"] is False and status["healthy"] is False
    assert status["verdict"] == "unknown"
