"""experiments/machete_diag.py without vLLM or a GPU: each variant of the Machete patch calls what it says it calls,
and the comparison counts the completions that match the run without sdcguard."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments" / "machete_diag.py"


@pytest.fixture
def diag():
    spec = importlib.util.spec_from_file_location("machete_diag_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_vllm(monkeypatch, log):
    """A MacheteLinearKernel whose apply_weights is shaped like sdcguard's patched one (its vLLM original in _orig),
    and a plugin whose register() does nothing."""
    def vllm_apply(self, layer, x, bias=None):
        log.append(("vllm", bias is not None))
        return Out(log)

    def shipped(self, layer, x, bias=None, _orig=vllm_apply, _rule="machete"):
        log.append(("sdcguard", bias is not None))
        return Out(log)

    K = type("MacheteLinearKernel", (), {"apply_weights": shipped})
    mod = types.ModuleType("machete")
    mod.MacheteLinearKernel = K
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.linear",
                 "vllm.model_executor.kernels.linear.mixed_precision"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.model_executor.kernels.linear.mixed_precision"].machete = mod
    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.linear.mixed_precision.machete", mod)
    from sdcguard import _quant, vllm_plugin
    monkeypatch.setattr(vllm_plugin, "register", lambda: None)

    def rounded(y, b):
        log.append(("add_bias", None))
        return y
    monkeypatch.setattr(_quant, "add_bias_op", lambda: rounded)
    return K


class Out:
    """A stand-in output that records how the bias is added to it."""
    def __init__(self, log):
        self.log = log

    def add_(self, b):
        self.log.append(("add_", None))
        return self

    def __add__(self, b):
        self.log.append(("+", None))
        return self


@pytest.mark.parametrize("variant, with_bias, expect", [
    ("wrap_only", True, [("vllm", True)]),
    ("wrap_only", False, [("vllm", False)]),
    ("split_inplace", True, [("vllm", False), ("add_", None)]),
    ("split_oop", True, [("vllm", False), ("+", None)]),
    ("split_oop", False, [("vllm", False)]),
    ("split_rounded", True, [("vllm", False), ("add_bias", None)]),
    ("full_rounded", True, [("sdcguard", False), ("add_bias", None)]),
    ("full_rounded", False, [("sdcguard", False)]),
    ("check_no_bias", False, [("sdcguard", False)]),
    ("check_no_bias", True, [("vllm", False), ("+", None)]),
    ("check_bias", True, [("sdcguard", True)]),
    ("check_bias", False, [("vllm", False)]),
    ("full", True, [("sdcguard", True)]),
])
def test_each_variant_runs_what_it_says(diag, monkeypatch, variant, with_bias, expect):
    log = []
    K = _fake_vllm(monkeypatch, log)
    diag._install(variant)
    layer = types.SimpleNamespace(_sdcguard_args=("args",))
    K.apply_weights(K(), layer, "x", "bias" if with_bias else None)
    assert log == expect


@pytest.mark.parametrize("variant", ["split_inplace", "split_oop", "split_rounded", "check_no_bias", "check_bias",
                                     "full_rounded"])
def test_a_layer_sdcguard_does_not_check_always_runs_vllms_own(diag, monkeypatch, variant):
    log = []
    K = _fake_vllm(monkeypatch, log)
    diag._install(variant)
    K.apply_weights(K(), types.SimpleNamespace(), "x", "bias")
    assert log == [("vllm", True)]


def test_the_comparison_counts_matching_completions(diag, tmp_path):
    base = {"model": "m", "outputs": ["a", "b", "c"], "light_outputs": ["a", "b"]}
    (tmp_path / "baseline.json").write_text(json.dumps(base))
    (tmp_path / "full.json").write_text(json.dumps({"outputs": ["a", "x", "c"], "light_outputs": ["a", "b"],
                                                    "checked_layers": 112}))
    (tmp_path / "wrap_only.json").write_text(json.dumps({"outputs": ["a", "b", "c"], "light_outputs": ["a", "b"]}))
    s = diag.compare(str(tmp_path))
    assert s["variants"]["full"] == {"same": "2/3", "light_same": "2/2", "all_same": False, "checked_layers": 112}
    assert s["variants"]["wrap_only"]["all_same"] is True
    assert s["variants"]["split_oop"] == {"error": "no result"}
    assert json.loads((tmp_path / "DIAG.json").read_text()) == s
