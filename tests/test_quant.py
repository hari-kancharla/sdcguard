"""Weight-only quantized layers in vLLM (Marlin and Machete, GPTQ / AWQ / compressed-tensors INT4 and INT8).

vLLM isn't installed here. The stand-in kernels below compute what vLLM 0.31's CUDA kernels compute (fp32
accumulation of x times weights dequantized into bf16, one rounding of the output; the bias added in bf16 after) and
then scramble their weights, as the real ones repack them, so nothing can read the matrix back from the layer. The
reference weights are computed independently, in float64, from the formulas in _quant's docstring."""
import sys
import types

import pytest
import torch
from torch import nn

from _vllm_standins import install_fake_vllm, load, plugin_env

BF = torch.bfloat16


class WT:
    """vLLM's ScalarType, as far as the plugin reads it."""

    def __init__(self, name, bits, bias):
        self.name, self.size_bits, self.bias = name, bits, bias

    def __str__(self):
        return self.name


U4B8, U8B128, U4 = WT("uint4b8", 4, 8), WT("uint8b128", 8, 128), WT("uint4", 4, 0)


def pack(codes, bits, axis):
    """int codes -> int32 words, 32 // bits per word along axis, lowest bits first (GPTQ's order)."""
    per = 32 // bits
    c = codes.movedim(axis, -1).to(torch.int64)
    out = torch.zeros(*c.shape[:-1], c.shape[-1] // per, dtype=torch.int64)
    for i in range(per):
        out |= (c[..., i::per] & ((1 << bits) - 1)) << (bits * i)
    out = torch.where(out >= 2 ** 31, out - 2 ** 32, out).to(torch.int32)
    return out.movedim(-1, axis).contiguous()


def param(t, **meta):
    p = nn.Parameter(t, requires_grad=False)
    for k, v in meta.items():
        setattr(p, k, v)
    return p


def reference(codes, scales, zeros, c, group, rule):
    """W (n x k, bf16) in float64 arithmetic: every product here is exact, so .to(bf16) is the one rounding."""
    k = codes.shape[0]
    g = torch.arange(k) // group
    s = scales.double()[g]
    if rule == "machete" and zeros is not None:
        zs = (-(scales.double() * zeros.double())).to(BF).double()
        w = (codes.double() * s).to(BF).double() + zs[g]
    else:
        w = (codes.double() - (c if zeros is None else zeros.double()[g])) * s
    return w.to(BF).t().contiguous()


class Cfg:
    """vLLM's MPLinearLayerConfig."""

    def __init__(self, k, n, wt, group, zp, act=BF, out=None):
        self.partition_weight_shape, self.full_weight_shape = (k, n), (k, n)
        self.weight_type, self.group_size, self.zero_points, self.act_type, self.out_type = wt, group, zp, act, out


class _Kernel:
    RULE = None
    FAULT = None                                   # (row, col, amount): a wrong output, for the alert tests

    def __init__(self, c, wq="qweight", ws="scales", wz=None):
        self.config, self.w_q_name, self.w_s_name, self.w_zp_name = c, wq, ws, wz

    def _read(self, layer):
        c = self.config
        q = layer.qweight
        codes = torch.zeros(q.shape[0] * 32 // c.weight_type.size_bits, q.shape[1], dtype=torch.int64)
        per, bits = 32 // c.weight_type.size_bits, c.weight_type.size_bits
        for i in range(per):
            codes[i::per] = (q.data.to(torch.int64) >> (bits * i)) & ((1 << bits) - 1)
        s = layer.scales.data if getattr(layer.scales, "output_dim", 1) == 1 else layer.scales.data.t()
        z = None
        if c.zero_points:
            zp = getattr(layer, self.w_zp_name).data            # (n / per, groups), packed along the outputs
            z = torch.zeros(zp.shape[0] * per, zp.shape[1], dtype=torch.int64)
            for i in range(per):
                z[i::per] = (zp.to(torch.int64) >> (bits * i)) & ((1 << bits) - 1)
            z = z.t()
        group = codes.shape[0] if c.group_size == -1 else c.group_size
        return reference(codes, s, z, c.weight_type.bias, group, self.RULE)

    def process_weights_after_loading(self, layer):
        W = self._read(layer)
        self._perm = torch.randperm(W.numel(), generator=torch.Generator().manual_seed(1))
        self._inv = torch.argsort(self._perm)
        self._shape = W.shape
        # the repacked weights: the matrix's bits in an order only this kernel knows
        layer.qweight = param(W.reshape(-1)[self._perm].contiguous().view(torch.int16))
        layer.scales = param(layer.scales.data.flip(0).contiguous())
        if self.RULE == "marlin":
            if self.w_zp_name is None:
                self.w_zp_name = "w_zp"
                layer.w_zp = param(torch.empty(0, dtype=torch.int32))
            if getattr(layer, "bias", None) is not None:          # Marlin permutes within groups of 32 outputs
                n = layer.bias.shape[0]
                self._bperm = torch.arange(n).view(-1, 32).flip(1).reshape(-1) if n % 32 == 0 else torch.arange(n)
                layer.bias.data = layer.bias.data[self._bperm].contiguous()

    def _w(self, layer):
        return layer.qweight.data.view(BF)[self._inv].view(self._shape)

    def apply_weights(self, layer, x, bias=None):
        W = self._w(layer)
        y = (x.reshape(-1, x.shape[-1]).float() @ W.float().t()).to(x.dtype)
        if type(self).FAULT is not None:
            r, col, amount = type(self).FAULT
            if r < y.shape[0]:
                y[r, col] += amount
        if bias is not None:
            b = bias
            if self.RULE == "marlin":
                b = torch.empty_like(bias)
                b[self._bperm] = bias
            y = y + b
        return y.reshape(*x.shape[:-1], -1)


def install_kernels(monkeypatch):
    """vllm.model_executor.kernels.linear.mixed_precision.{marlin,machete} with fresh kernel classes."""
    M, Runner = install_fake_vllm(monkeypatch.setitem)
    base = "vllm.model_executor.kernels.linear.mixed_precision"
    for n in ("vllm.model_executor.kernels", "vllm.model_executor.kernels.linear", base):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    classes = {}
    for mod, cls, rule in (("marlin", "MarlinLinearKernel", "marlin"), ("machete", "MacheteLinearKernel", "machete")):
        m = types.ModuleType(f"{base}.{mod}")
        K = type(cls, (_Kernel,), {"RULE": rule})
        setattr(m, cls, K)
        monkeypatch.setitem(sys.modules, f"{base}.{mod}", m)
        classes[rule] = K
    return M, Runner, classes


class AutoGPTQLinearMethod:
    def __init__(self, kernel):
        self.kernel = kernel

    def process_weights_after_loading(self, layer):
        self.kernel.process_weights_after_loading(layer)

    def apply(self, layer, x, bias=None):
        return self.kernel.apply_weights(layer, x, bias)


class QLin(nn.Module):
    """A quantized vLLM linear layer in the standard layout: qweight (k / per, n) packed along the inputs, scales
    (groups, n), and for zero-point types qzeros (n / per, groups) packed along the outputs (AWQ after vLLM's
    conversion, compressed-tensors' weight_zero_point)."""

    def __init__(self, K, k=256, n=192, wt=U4B8, group=64, bias=False, prefix="model.layers.0.qkv_proj",
                 scales_out_first=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        bits, zp = wt.size_bits, wt.bias == 0
        self.codes = torch.randint(0, 1 << bits, (k, n), generator=g)
        groups = 1 if group == -1 else k // group
        self.s = (torch.rand(groups, n, generator=g) * 0.02 + 0.001).to(BF)
        self.z = torch.randint(0, 1 << bits, (groups, n), generator=g) if zp else None
        self.qweight = param(pack(self.codes, bits, 0), input_dim=0, output_dim=1, packed_dim=0)
        if scales_out_first:                       # compressed-tensors: weight_scale (n, groups)
            self.scales = param(self.s.t().contiguous(), input_dim=1, output_dim=0)
        else:
            self.scales = param(self.s.clone(), input_dim=0, output_dim=1)
        if zp:
            self.qzeros = param(pack(self.z.t().contiguous(), bits, 0), output_dim=0, packed_dim=0)
        self.bias = param(torch.randn(n, generator=g).to(BF)) if bias else None
        self.prefix = prefix
        c = Cfg(k, n, wt, group, zp)
        self.quant_method = AutoGPTQLinearMethod(K(c, "qweight", "scales", "qzeros" if zp else None))

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


def x_in(k=256, m=5, seed=3):
    return torch.randn(m, k, generator=torch.Generator().manual_seed(seed)).to(BF)


# ---------------------------------------------------------------- the weights each kernel multiplies by
@pytest.mark.parametrize("rule", ["marlin", "machete"])
@pytest.mark.parametrize("wt,group", [(U4B8, 64), (U4B8, 128), (U8B128, 32), (U4, 128), (U4, 64)])
def test_effective_weights_follow_each_kernels_rule(rule, wt, group):
    from sdcguard import _quant
    K = type("K", (_Kernel,), {"RULE": rule})
    lay = QLin(K, wt=wt, group=group)
    qw = _quant.from_layer(lay, "qweight", "scales", "qzeros" if wt.bias == 0 else None, wt, group, rule, BF)
    want = reference(lay.codes, lay.s, lay.z, wt.bias, group, rule)
    assert torch.equal(qw.weight(), want)
    assert torch.equal(qw.weight(40, 70), want[40:110])                # any range of outputs


def test_marlin_and_machete_dequantize_zero_points_differently():
    """Marlin: rnd((q - z) * s). Machete: rnd(rnd(q * s) - rnd(s * z)). Reading AWQ weights by the wrong kernel's
    rule gives other bf16 values, which is why the rule follows the kernel and probe() confirms it."""
    lay = QLin(type("K", (_Kernel,), {"RULE": "marlin"}), wt=U4, group=64)
    a = reference(lay.codes, lay.s, lay.z, 0, 64, "marlin")
    b = reference(lay.codes, lay.s, lay.z, 0, 64, "machete")
    assert (a != b).sum() > 0


def test_compressed_tensors_scales_stored_outputs_first_are_read_the_same():
    from sdcguard import _quant
    K = type("K", (_Kernel,), {"RULE": "marlin"})
    a, b = QLin(K, seed=4), QLin(K, seed=4, scales_out_first=True)
    qa = _quant.from_layer(a, "qweight", "scales", None, U4B8, 64, "marlin", BF)
    qb = _quant.from_layer(b, "qweight", "scales", None, U4B8, 64, "marlin", BF)
    assert torch.equal(qa.weight(), qb.weight())


def test_probe_finds_a_wrong_reading_and_accepts_the_right_one():
    from sdcguard import _quant
    K = type("K", (_Kernel,), {"RULE": "machete"})
    lay = QLin(K, wt=U4, group=64)
    right = _quant.from_layer(lay, "qweight", "scales", "qzeros", U4, 64, "machete", BF).weight()
    wrong = _quant.from_layer(lay, "qweight", "scales", "qzeros", U4, 64, "marlin", BF).weight()
    kern = lay.quant_method.kernel
    kern.process_weights_after_loading(lay)
    rows = _quant.probe_rows(256, 64)
    assert _quant.probe(lambda x: kern.apply_weights(lay, x, None), right, rows)["mismatches"] == 0
    bad = _quant.probe(lambda x: kern.apply_weights(lay, x, None), wrong, rows)
    assert bad["mismatches"] > 0 and bad["first"] is not None


def test_probe_rows_cover_every_group_when_sampling():
    from sdcguard import _quant
    rows = _quant.probe_rows(8192, 128, limit=512)
    assert len(rows) == 512 and {r // 128 for r in rows} == set(range(64)) and 0 in rows and 8191 in rows
    assert _quant.probe_rows(300, 64) == list(range(300))


# ---------------------------------------------------------------- the plugin
def _model(classes, **kw):
    return load(nn.ModuleList([QLin(classes[kw.pop("rule", "marlin")], **kw)]))


@pytest.mark.parametrize("rule", ["marlin", "machete"])
@pytest.mark.parametrize("wt,group,bias", [(U4B8, 64, True), (U4, 128, True), (U8B128, 64, False)])
def test_a_checked_quantized_layer_gives_the_same_output_and_no_alarm(monkeypatch, tmp_path, rule, wt, group, bias):
    """The checked call runs the kernel without the bias, checks that output and adds the bias in bf16: the same
    output, bit for bit, as the kernel adding it itself."""
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _model(classes, rule=rule, wt=wt, group=group, bias=bias)
    lay = model[0]
    g = plugin.guard()
    assert g is not None and [la["name"] for la in g.layers] == ["model.layers.0.qkv_proj"]
    x = x_in()
    y = lay(x)
    plain = type(lay.quant_method.kernel).__mro__[1].apply_weights(lay.quant_method.kernel, lay, x, lay.bias)
    assert torch.equal(y, plain)
    assert plugin.poll(wait=True) == []
    assert g.counts()["model.layers.0.qkv_proj"]["checks"] >= 1
    r = plugin.layer_report(model)
    assert r["checked"] == 1 and r["unchecked_by_method"] == {}


@pytest.mark.parametrize("rule", ["marlin", "machete"])
def test_a_wrong_output_of_a_quantized_layer_is_caught_and_rechecked(monkeypatch, tmp_path, rule):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_EVIDENCE_DIR=str(tmp_path / "ev"))
    plugin.register()
    model = _model(classes, rule=rule, wt=U4 if rule == "machete" else U4B8, group=64, bias=True)
    classes[rule].FAULT = (2, 77, 3.0)
    try:
        model[0](x_in())
    finally:
        classes[rule].FAULT = None
    ev = plugin.poll(wait=True)
    assert len(ev) == 1
    d = ev[0].detail
    assert d["where"] == "model.layers.0.qkv_proj" and d["row"] == 2 and d["col_block"] == 0
    assert d["weights_changed_since_load"] is False
    assert d["recheck"]["verdict"] == "output_wrong", d["recheck"]
    assert "copy kept at load" in d["recheck"]["by"]
    assert d["evidence"] and str(tmp_path / "ev") in d["evidence"]


def test_changed_quantized_weights_are_reported_as_changed(monkeypatch, tmp_path):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=64)
    lay = model[0]
    with torch.no_grad():
        lay.qweight.data[123] ^= 0x4000                     # a flipped bit in the repacked weights
    lay(x_in())
    g = plugin.guard()
    assert g.verify_weights() == ["model.layers.0.qkv_proj"]
    ev = plugin.poll(wait=True)
    if ev:                                                  # the flip changed an output enough to alarm
        assert ev[0].detail["weights_changed_since_load"] is True
        assert ev[0].detail["recheck"]["verdict"] == "weights_changed"


def test_a_kernel_that_disagrees_with_the_reading_is_not_checked(monkeypatch, tmp_path, caplog):
    """A kernel whose arithmetic isn't what _quant assumes (here: Machete's zero-point rule read as Marlin's) is
    caught by the probe before serving: the layer runs unchecked and is counted as a quantized layer left over."""
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    classes["marlin"].RULE = "machete"                      # the stand-in now computes Machete's way
    model = _model(classes, rule="marlin", wt=U4, group=64)
    assert plugin.guard() is None or not plugin.guard().layers
    r = plugin.layer_report(model)
    assert r["checked"] == 0 and r["quantized_or_moe"] == 1
    assert any("didn't match" in why for why in r["unchecked_by_method"])
    assert "differed from the weights read before its repack" in caplog.text
    y = model[0](x_in())                                     # serves, unchecked
    assert y.shape == (5, 192)


@pytest.mark.parametrize("case,expect", [
    ("channelwise", "one scale per output on Marlin"),
    ("atomic", "VLLM_MARLIN_USE_ATOMIC_ADD=1"),
    ("fp32", "float32 activations"),
])
def test_unsupported_quantized_layers_are_listed_with_why(monkeypatch, tmp_path, case, expect):
    M, Runner, classes = install_kernels(monkeypatch)
    extra = {"SDCGUARD_REQUIRE": "1"}
    if case == "atomic":
        extra["VLLM_MARLIN_USE_ATOMIC_ADD"] = "1"
    plugin = plugin_env(monkeypatch, tmp_path, **extra)
    plugin.register()
    lay = QLin(classes["marlin"], group=-1 if case == "channelwise" else 64)
    if case == "fp32":
        lay.quant_method.kernel.config.act_type = torch.float32
    model = load(nn.ModuleList([lay]))
    r = plugin.layer_report(model)
    assert r["checked"] == 0 and r["quantized_or_moe"] == 1
    assert any(expect in why for why in r["unchecked_by_method"]), r
    with pytest.raises(RuntimeError, match="SDCGUARD_REQUIRE"):           # a quantized layer left over
        plugin._require(r, "msg")


def test_machete_checks_one_scale_per_output(monkeypatch, tmp_path):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=-1, bias=True)
    assert len(plugin.guard().layers) == 1
    model[0](x_in())
    assert plugin.poll(wait=True) == []


def test_without_the_kept_copy_an_alert_says_it_cant_recheck(monkeypatch, tmp_path):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_QUANT_RECHECK="0")
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=64)
    assert plugin.guard().layers[0]["source"].weights is None
    classes["machete"].FAULT = (0, 5, 4.0)
    try:
        model[0](x_in())
    finally:
        classes["machete"].FAULT = None
    ev = plugin.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["recheck"]["verdict"] == "unknown"
    assert "SDCGUARD_QUANT_RECHECK=0" in ev[0].detail["recheck"]["why"]


def test_refresh_turns_a_quantized_layer_off_and_says_why(monkeypatch, tmp_path):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=64)
    assert plugin.refresh() == 0
    g = plugin.guard()
    assert "repacked" in g.unsupported["model.layers.0.qkv_proj"]
    assert model[0]._sdcguard_args is None
    model[0](x_in())                                         # serves, unchecked


def test_share_selects_among_quantized_and_unquantized_layers_alike(monkeypatch, tmp_path):
    """SDCGUARD_SHARE numbers the eligible layers in load order, quantized or not, so replicas with offsets 0 to R-1
    still check every layer between them exactly once."""
    from _vllm_standins import Lin
    from sdcguard.serve import select_layer
    picked = set()
    for offset in range(2):
        M, Runner, classes = install_kernels(monkeypatch)
        plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_SHARE="0.5", SDCGUARD_OFFSET=str(offset))
        plugin.register()
        layers = [QLin(classes["machete"], prefix=f"model.layers.{i}.proj", seed=i) if i % 2 == 0 else
                  Lin(M, 256, 256, f"model.layers.{i}.proj") for i in range(8)]
        model = load(nn.ModuleList(layers))
        want = [i for i in range(8) if select_layer(i, 0.5, offset)]
        assert [la["name"] for la in plugin.guard().layers] == [f"model.layers.{i}.proj" for i in want]
        r = plugin.layer_report(model)
        assert r["checked"] == len(want) and r["unchecked_by_method"] == {"not selected by SDCGUARD_SHARE": 8 - len(want)}
        assert not picked & set(want)
        picked |= set(want)
    assert picked == set(range(8))


def test_sdcguard_quant_0_leaves_quantized_layers_alone(monkeypatch, tmp_path):
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_QUANT="0")
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=64)
    r = plugin.layer_report(model)
    assert r["checked"] == 0 and r["unchecked_by_method"] == {"AutoGPTQLinearMethod": 1}


def test_marlins_bias_is_added_by_one_opaque_call_and_machetes_by_add_(monkeypatch, tmp_path):
    """Marlin adds the bias inside its kernel, rounded; the plugin's own addition must not be fused by torch.compile
    into the next operations (which would keep it in fp32), so it is one custom op. Machete adds it with add_ after
    its GEMM, which the plugin does the same way."""
    from sdcguard import _quant
    op = _quant.add_bias_op()
    y, b = torch.randn(3, 5).to(BF), torch.randn(5).to(BF)
    assert torch.equal(op(y, b), y + b)
    assert op is torch.ops.sdcguard.add_bias
    calls = []
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    monkeypatch.setattr(_quant, "add_bias_op", lambda: (lambda y, b: calls.append(1) or y + b))
    plugin.register()
    marlin = _model(classes, rule="marlin", wt=U4B8, group=64, bias=True)
    machete = _model(classes, rule="machete", wt=U4B8, group=64, bias=True)
    marlin[0](x_in())
    assert calls == [1]
    machete[0](x_in())
    assert calls == [1]


def test_the_vllm_experiment_refuses_pairs_run_on_different_kernels():
    """A baseline on Machete against a guarded run on Marlin would compare two kernels, not sdcguard."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "experiments" / "vllm_check.py"
    spec = importlib.util.spec_from_file_location("vllm_check_kernels_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    b = {"model": "m", "vllm": "0.31.0", "enforce_eager": False, "first_outputs": ["x"], "output_tokens_per_s": 1.0,
         "disabled_kernels": None}
    g = {**b, "disabled_kernels": "MacheteLinearKernel"}
    assert "baseline and guarded ran with different VLLM_DISABLED_KERNELS" in module._comparison_failures(b, g)
