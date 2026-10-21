"""What the vLLM plugin's register() does to the process it runs in: it runs in every vLLM process (the front end,
the engine core, each worker), before any of them has a GPU of its own.

vLLM's compile cache (VLLM_CACHE_ROOT/torch_compile_cache: AOT-compiled graphs, Inductor and Triton caches) is keyed
by vLLM's settings only, and a graph traced with sdcguard has its checks built in, so sdcguard's graphs need a
folder of their own for each sdcguard setting they depend on. And register() must not initialize CUDA: vLLM picks
fork or spawn for its processes by whether CUDA is initialized."""
import os

import torch

from _vllm_standins import install_fake_vllm, plugin_env


def test_a_user_set_vllm_cache_root_is_still_kept_apart_from_unchecked_graphs(monkeypatch, tmp_path):
    """register() used to move VLLM_CACHE_ROOT only when the user hadn't set it. vLLM's AOT compile (on by default
    with torch >= 2.10) then loaded a graph compiled without sdcguard from the user's folder: no check ran while
    the plugin reported every layer checked (a GPU-only consequence; the folder is the cause)."""
    install_fake_vllm(monkeypatch.setitem)
    plugin = plugin_env(monkeypatch, tmp_path)
    user_root = str(tmp_path / "persistent_vllm_cache")
    monkeypatch.setenv("VLLM_CACHE_ROOT", user_root)           # e.g. a mounted volume in a container
    plugin.register()
    root = os.environ["VLLM_CACHE_ROOT"]
    assert os.path.dirname(root) == user_root and os.path.basename(root).startswith("sdcguard-")


def _root_after_register(monkeypatch, tmp_path, **env):
    from sdcguard import vllm_plugin
    install_fake_vllm(monkeypatch.setitem)
    plugin_env(monkeypatch, tmp_path, **env)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    vllm_plugin.register()
    return os.environ["VLLM_CACHE_ROOT"]


def test_the_sdcguard_cache_folder_depends_on_the_settings_baked_into_the_graph(monkeypatch, tmp_path):
    """The traced graph bakes in which layers have checks (SDCGUARD_SHARE, SDCGUARD_OFFSET), each check's slot and
    z (constants in its call) and inline vs custom op (SDCGUARD_OPAQUE). With one folder for all of them, a restart
    with another share reloaded stale slots: alerts named after the wrong layer, layers counted checked that the
    graph never checks. Settings that don't change the graph keep the folder (and its compiled graphs)."""
    base = _root_after_register(monkeypatch, tmp_path)
    assert base.startswith(str(tmp_path / ".cache" / "vllm" / "sdcguard-"))
    for env in ({"SDCGUARD_SHARE": "0.25"}, {"SDCGUARD_SHARE": "0.25", "SDCGUARD_OFFSET": "2"},
                {"SDCGUARD_Z": "8"}, {"SDCGUARD_OPAQUE": "1"}):
        assert _root_after_register(monkeypatch, tmp_path, **env) != base, env
    assert _root_after_register(monkeypatch, tmp_path, SDCGUARD_SHARE="1/4") == \
        _root_after_register(monkeypatch, tmp_path, SDCGUARD_SHARE="25%")
    assert _root_after_register(monkeypatch, tmp_path, SDCGUARD_POLL_EVERY="7") == base


def test_a_child_process_keeps_its_parents_cache_folder(monkeypatch, tmp_path):
    """vLLM's engine core and workers inherit the front end's environment (Ray copies the VLLM_ variables) and run
    register() again: the folder must stay the same, not become sdcguard-<key>/sdcguard-<key>."""
    first = _root_after_register(monkeypatch, tmp_path)
    install_fake_vllm(monkeypatch.setitem)
    plugin = plugin_env(monkeypatch, tmp_path)
    monkeypatch.setenv("VLLM_CACHE_ROOT", first)                # what the child inherits
    plugin.register()
    assert os.environ["VLLM_CACHE_ROOT"] == first
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_SHARE="0.5")
    install_fake_vllm(monkeypatch.setitem)
    monkeypatch.setenv("VLLM_CACHE_ROOT", first)                # another setting: a sibling folder
    plugin.register()
    assert os.path.dirname(os.environ["VLLM_CACHE_ROOT"]) == os.path.dirname(first) != first


def test_register_does_not_initialize_cuda_in_the_vllm_frontend_process(monkeypatch, tmp_path):
    """vLLM loads general plugins in its front-end process too (EngineArgs, argument parsing), before it starts the
    engine core and workers. register() called known_issues.log_once() -> torch.cuda.current_device(), which
    initializes CUDA; vLLM's get_mp_context() then sees torch.cuda.is_initialized() and forces spawn ("CUDA is
    initialized"), and an offline LLM(...) script without `if __name__ == "__main__":`, fine without SDCGUARD=1,
    fails with it. In each worker that ran before init_device() selected the worker's GPU. Mocked here."""
    from sdcguard import known_issues
    install_fake_vllm(monkeypatch.setitem)
    plugin = plugin_env(monkeypatch, tmp_path)
    monkeypatch.delitem(known_issues._LOGGED, "done", raising=False)
    inits, probes = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: probes.append(1) or True)
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: inits.append(1))
    m = torch.backends.cuda.matmul
    saved = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    try:
        plugin.register()
    finally:
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved
    assert inits == [], "register() initialized CUDA in a process that only loads plugins"
    assert probes == [], "register() asked for the GPU (torch.cuda.is_available() can initialize the driver)"


def test_the_known_issue_report_reads_the_gpu_only_once_cuda_is_in_use(monkeypatch):
    """known_issues.environment() (used by log_once) must not be the one to initialize CUDA either; a caller that
    names a device, or the `sdcguard issues` command (its own process), may."""
    from sdcguard import known_issues
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: calls.append("current_device") or 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: calls.append("properties"))
    env = known_issues.environment()
    assert calls == [] and env["gpu"] is None
    known_issues.environment("cuda:0")
    assert calls == ["properties"]
