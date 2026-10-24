"""The vLLM plugin's FP8 quantization check with stand-ins for vLLM's FP8 path (tests/_fp8_standins.py): off by
default; with SDCGUARD_FP8=1 the scaled-mm kernels' apply_weights runs the plugin's copy only when it is vLLM's
code; layers registered after load; clean steps quiet; a per-tensor scale taken over padding rows that hold an
earlier step's values named, with the request it concerns; a pre-quantized input passed through; an FP8 MoE layer's
expert inputs checked through vLLM's moe_kernel_quantize_input."""
import logging
import types

import torch

import _fp8_standins as fs
from _vllm_standins import install_fake_vllm, plugin_env


class Model(torch.nn.Module):
    def __init__(self, per_token=True):
        super().__init__()
        self.layers = torch.nn.ModuleList([fs.Fp8Linear(64, 64, f"model.layers.{i}.mlp.down_proj",
                                                        per_token=per_token, seed=i) for i in range(2)])

    def forward(self, x):
        for lay in self.layers:
            x = lay(x)
        return x


def _setup(monkeypatch, tmp_path, model, **env):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    smk, utils, tm = fs.install(monkeypatch.setitem)

    def init(self, factory):
        self.factory = factory
        self.input_batch = types.SimpleNamespace(req_ids=[])

    def load_model(self):
        self.model = self.factory()

    def execute_model(self, so):
        self.input_batch.req_ids = list(so.num_scheduled_tokens)
        return self.model(so.x)

    monkeypatch.setattr(GM, "__init__", init, raising=False)
    monkeypatch.setattr(GM, "load_model", load_model, raising=False)
    monkeypatch.setattr(GM, "execute_model", execute_model, raising=False)
    vp = plugin_env(monkeypatch, tmp_path, **{"SDCGUARD_FP8": "1", **env})
    for k, v in (("fp8_classes", []), ("fp8_registered", False), ("fp8_why", None)):
        monkeypatch.setitem(vp._STATE, k, v)
    vp.register()
    runner = GM(model)
    runner.load_model()
    return vp, runner, smk, tm


def _so(x, counts):
    return types.SimpleNamespace(x=x, num_scheduled_tokens=dict(counts),
                                 total_num_scheduled_tokens=sum(counts.values()))


def _x(rows, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, 64, generator=g) * scale).to(torch.bfloat16)


def test_fp8_checks_are_off_unless_sdcguard_fp8_is_1(monkeypatch, tmp_path):
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Model, SDCGUARD_FP8="0")
    assert vp._STATE["fp8_classes"] == [] and not getattr(smk.FP8ScaledMMLinearKernel, "_sdcguard_fp8", False)
    runner.execute_model(_so(_x(4), {"a": 4}))
    g = vp.guard()
    assert g is None or g.fp8 is None


def test_clean_steps_are_quiet_and_every_layer_is_checked(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Model)
    assert "FP8ScaledMMLinearKernel" in vp._STATE["fp8_classes"]
    assert any("checking the FP8 quantization of 2 layers" in r.getMessage() for r in caplog.records)
    for step in range(4):
        out = runner.execute_model(_so(_x(5, step), {"a": 3, "b": 2}))
        assert out.shape == (5, 64)
    assert vp.poll(wait=True) == []
    cov = vp.guard().coverage_report()["fp8_quantization"]
    assert cov["layers_checked"] == 2 and cov["calls"] == 8 and cov["recipes"] == ["dynamic_token"]


def test_the_layer_summary_says_how_many_fp8_layers_are_checked(monkeypatch, tmp_path, caplog):
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Model)
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        runner.execute_model(_so(_x(4), {"a": 4}))
    lines = [r.getMessage() for r in caplog.records if "linear layers" in r.getMessage()]
    assert lines and "FP8 input quantization checked on 2 layers (SDCGUARD_FP8=1)" in lines[0], lines
    assert "is (SDCGUARD_FP8=1)" not in lines[0]


def test_a_missing_fp8_kernel_module_is_said_out_loud(monkeypatch, tmp_path, caplog):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    fs.install(monkeypatch.setitem)
    import sys
    monkeypatch.delitem(sys.modules, "vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel")
    vp = plugin_env(monkeypatch, tmp_path, SDCGUARD_FP8="1")
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        vp.register()
    assert "FP8ScaledMMLinearKernel" not in vp._STATE["fp8_classes"]
    assert "not found" in (vp._STATE.get("fp8_why") or "")
    assert any("not checking the FP8 quantization of linear layers" in r.getMessage() for r in caplog.records)


def test_a_per_tensor_scale_from_padding_rows_is_named_with_its_request(monkeypatch, tmp_path):
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, lambda: Model(per_token=False))
    x = _x(8)
    x[5:] = x[5:] * 10                    # rows 5..7: CUDA-graph padding left by an earlier, larger request
    runner.execute_model(_so(x, {"req-a": 3, "req-b": 2}))
    ev = vp.poll(wait=True)
    assert ev and all(e.detector == "serve_fp8" for e in ev)
    d = ev[0].detail
    assert d["where"] == "model.layers.0.mlp.down_proj" and d["finding"] == "scale_from_outside_rows"
    assert d["directly_implicated"] and d["directly_implicated"][0] in ("req-a", "req-b")
    assert any("communication between GPUs" in u for u in d["unknown_impact"])
    # padding with smaller values than the real rows: quiet
    x = _x(8)
    x[5:] = x[5:] * 0.1
    runner.execute_model(_so(x, {"req-a": 3, "req-b": 2}))
    assert [e for e in vp.poll(wait=True) if e.detail["where"].endswith("0.mlp.down_proj")] == []


def test_runs_outside_model_steps_are_not_judged(monkeypatch, tmp_path):
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, lambda: Model(per_token=False))
    runner.execute_model(_so(_x(4), {"a": 4}))
    x = _x(8)
    x[2:] *= 50
    runner.model(x)                       # a warm-up / capture run between steps
    assert vp.poll(wait=True) == []


def test_another_apply_weights_is_not_replaced(monkeypatch, tmp_path, caplog):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    smk, utils, tm = fs.install(monkeypatch.setitem)

    def other(self, layer, x, bias=None):
        return self.apply_scaled_mm(A=x, B=layer.weight, out_dtype=x.dtype, As=None, Bs=None, bias=bias,
                                    output_shape=[*x.shape[:-1], layer.weight.shape[1]])

    monkeypatch.setattr(smk.FP8ScaledMMLinearKernel, "apply_weights", other)
    vp = plugin_env(monkeypatch, tmp_path, SDCGUARD_FP8="1")
    for k, v in (("fp8_classes", []), ("fp8_registered", False), ("fp8_why", None)):
        monkeypatch.setitem(vp._STATE, k, v)
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        vp.register()
    assert smk.FP8ScaledMMLinearKernel.apply_weights is other
    assert "unknown version" in (vp._STATE.get("fp8_why") or "")


def test_a_prequantized_input_passes_through(monkeypatch, tmp_path):
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Model)
    lay = runner.model.layers[0]
    x = _x(3)
    q, s = fs.quant_per_token(x)
    qa = fs.QuantizedActivation(q, s, x.shape, x.dtype)
    out = lay(qa)
    assert out.shape == (3, 64)


def test_an_fp8_moe_layers_expert_inputs_are_checked(monkeypatch, tmp_path):
    class MoEModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            import sys
            cls = sys.modules["vllm.model_executor.layers.fused_moe.routed_experts"].RoutedExperts
            self.experts = cls(sys.modules["vllm.model_executor.layers.fused_moe.experts.triton_moe"])

        def forward(self, x):
            w = torch.full((x.shape[0], 2), 0.5)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]])[:x.shape[0]]
            return self.experts.forward_modular(x=x, topk_weights=w, topk_ids=ids.to(torch.int32))

    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, MoEModel)
    assert any("moe_kernel_quantize_input" in c for c in vp._STATE["fp8_classes"])
    runner.execute_model(_so(_x(6), {"a": 6}))
    assert vp.poll(wait=True) == []
    x = _x(8)
    x[:6] *= 0.5
    x[6:] *= 20
    runner.execute_model(_so(x, {"a": 4, "b": 2}))
    ev = vp.poll(wait=True)
    assert ev and ev[0].detail["where"] == "experts (expert inputs)"
    assert ev[0].detail["finding"] == "scale_from_outside_rows"


def test_a_model_with_more_than_64_fp8_layers_checks_its_first_moe_layer(monkeypatch, tmp_path):
    """granite-3.1-1b-a400m's layout: each of 24 blocks has an FP8 qkv_proj, an FP8 o_proj and an FP8 MoE layer, 72
    FP8 layers in all, so the check's state grows past 64 while they register. The MoE layers registered before the
    growth kept the old state (the reviewer's finding before the second fp8_moe run): their checks counted nothing
    and raised nothing. Here block 0's MoE layer, registered first, must still report."""
    import sys

    class Block(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            cls = sys.modules["vllm.model_executor.layers.fused_moe.routed_experts"].RoutedExperts
            self.qkv_proj = fs.Fp8Linear(64, 64, f"model.layers.{i}.self_attn.qkv_proj", seed=i)
            self.o_proj = fs.Fp8Linear(64, 64, f"model.layers.{i}.self_attn.o_proj", seed=100 + i)
            self.experts = cls(sys.modules["vllm.model_executor.layers.fused_moe.experts.triton_moe"], seed=i)

    class Granite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(i) for i in range(24)])

        def forward(self, x):
            w = torch.full((x.shape[0], 2), 0.5)
            ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]])[:x.shape[0]]
            return self.layers[0].experts.forward_modular(x=x, topk_weights=w, topk_ids=ids.to(torch.int32))

    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Granite)
    fc = vp.guard().fp8
    assert len(fc.layers) == 72 and vp._STATE["fp8"]["checked"] == 72
    x = _x(8)
    x[:6] *= 0.5
    x[6:] *= 20                            # rows 6 and 7: padding holding an earlier request's larger values
    runner.execute_model(_so(x, {"a": 4, "b": 2}))
    ev = vp.poll(wait=True)
    assert ev and ev[0].detail["where"] == "layers.0.experts (expert inputs)", [e.detail["where"] for e in ev]
    assert ev[0].detail["finding"] == "scale_from_outside_rows"


def test_the_plugins_apply_weights_gives_vllms_output_bit_for_bit(monkeypatch, tmp_path):
    for per_token in (True, False):
        vp, runner, smk, tm = _setup(monkeypatch, tmp_path, lambda: Model(per_token=per_token))
        lay = runner.model.layers[0]
        kern = lay.quant_method.fp8_linear
        x = _x(7, 3)
        mine = kern.apply_weights(lay, x)                         # the plugin's copy, with the check
        theirs = fs.FP8ScaledMMLinearKernel.apply_weights(kern, lay, x)   # vLLM's code
        assert type(kern).apply_weights is not fs.FP8ScaledMMLinearKernel.apply_weights
        assert torch.equal(mine, theirs)


def test_a_lazy_module_that_raises_on_attribute_access_doesnt_stop_the_patch(monkeypatch, tmp_path):
    """In a real server other packages' lazy modules are loaded too (transformers' imports a submodule on attribute
    access, which can raise for a missing optional dependency): the patch reads only vLLM's modules, through their
    __dict__."""
    import sys

    class Lazy(types.ModuleType):
        def __getattr__(self, name):
            raise ModuleNotFoundError("No module named 'torchvision'")

    monkeypatch.setitem(sys.modules, "lazy_pkg_for_test", Lazy("lazy_pkg_for_test"))
    vp, runner, smk, tm = _setup(monkeypatch, tmp_path, Model)
    assert "FP8ScaledMMLinearKernel" in vp._STATE["fp8_classes"]
    assert any("moe_kernel_quantize_input" in c for c in vp._STATE["fp8_classes"])
