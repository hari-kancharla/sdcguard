"""Weight-only quantized layers in vLLM (Marlin and Machete, GPTQ / AWQ / compressed-tensors INT4 and INT8).

vLLM isn't installed here. The stand-in kernels below compute what vLLM 0.31's CUDA kernels compute (fp32
accumulation of x times weights dequantized into bf16, one rounding of the output; the bias added in bf16 after) and
then scramble their weights, as the real ones repack them, so nothing can read the matrix back from the layer. The
reference weights are computed independently, in float64, from the formulas in _quant's docstring."""
import sys
import types
from typing import Optional

import pytest
import torch
from torch import nn

import _vllm_custom_ops
import _vllm_machete
from _vllm_standins import install_fake_vllm, load, plugin_env

BF = torch.bfloat16


class WT:
    """vLLM's ScalarType, as far as the plugin and vLLM's Machete code read it."""

    def __init__(self, name, bits, bias, id):
        self.name, self.size_bits, self.bias, self.id = name, bits, bias, id

    def __str__(self):
        return self.name


U4B8, U8B128, U4 = WT("uint4b8", 4, 8, 101), WT("uint8b128", 8, 128, 102), WT("uint4", 4, 0, 103)

# The stand-in kernels by their repacked weights' address: what the stand-in machete_mm (vLLM's custom op) computes
_BY_QWEIGHT = {}


@torch.library.custom_op("sdcguard_test::machete_mm", mutates_args=())
def _machete_mm(a: torch.Tensor, b_q: torch.Tensor, b_type_id: int, out_type: Optional[torch.dtype],
                b_group_scales: Optional[torch.Tensor], b_group_zeros: Optional[torch.Tensor],
                b_group_size: Optional[int], b_channel_scales: Optional[torch.Tensor],
                a_token_scales: Optional[torch.Tensor], schedule: Optional[str]) -> torch.Tensor:
    """torch.ops._C.machete_mm as vLLM 0.31 declares it: one opaque call, here the stand-in kernel's GEMM."""
    return _BY_QWEIGHT[b_q.data_ptr()]._gemm(b_q, a)


@_machete_mm.register_fake
def _(a, b_q, b_type_id, out_type, b_group_scales, b_group_zeros, b_group_size, b_channel_scales, a_token_scales,
      schedule):
    return torch.empty((a.size(0), b_q.size(1)), device=a.device, dtype=a.dtype)       # vLLM's machete_mm_fake


# what vLLM's wrapper (vllm._custom_ops.machete_mm) reaches as torch.ops._C.machete_mm
TORCH_SHIM = types.SimpleNamespace(ops=types.SimpleNamespace(_C=types.SimpleNamespace(
    machete_mm=torch.ops.sdcguard_test.machete_mm)))


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


def reference(codes, scales, zeros, c, group, rule, fused=True):
    """W (n x k, bf16) in float64 arithmetic: every product here is exact, so .to(bf16) is the one rounding.
    Machete with zero points: q * s + zs rounded once (fused, what it computed on an H100) or q * s rounded first."""
    k = codes.shape[0]
    g = torch.arange(k) // group
    s = scales.double()[g]
    if rule == "machete" and zeros is not None:
        zs = (-(scales.double() * zeros.double())).to(BF).double()
        qs = codes.double() * s
        w = (qs if fused else qs.to(BF).double()) + zs[g]
    else:
        w = (codes.double() - (c if zeros is None else zeros.double()[g])) * s
    return w.to(BF).t().contiguous()


class Cfg:
    """vLLM's MPLinearLayerConfig."""

    def __init__(self, k, n, wt, group, zp, act=BF, out=None):
        self.partition_weight_shape, self.full_weight_shape = (k, n), (k, n)
        self.weight_type, self.group_size, self.zero_points, self.act_type, self.out_type = wt, group, zp, act, out


class _Kernel:
    """A stand-in for vLLM's MPLinearKernel subclasses. Its apply_weights is its own (Marlin's here); the Machete
    stand-in runs vLLM's MacheteLinearKernel.apply_weights instead (_vllm_machete), whose machete_mm call reaches
    _gemm through the stand-in op. _gemm's result is what both compute."""
    RULE = None
    FAULT = None                                   # (row, col, amount): a wrong output, for the alert tests
    FUSED = True                                   # Machete's zero-point rounding (bool, or a function of the batch)

    def __init__(self, c, wq="qweight", ws="scales", wz=None):
        self.config, self.w_q_name, self.w_s_name, self.w_zp_name = c, wq, ws, wz

    _get_weight_params = _vllm_machete._get_weight_params

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
        f = type(self).FUSED
        if callable(f):
            return {fused: reference(codes, s, z, c.weight_type.bias, group, self.RULE, fused)
                    for fused in (True, False)}
        return reference(codes, s, z, c.weight_type.bias, group, self.RULE, f)

    def process_weights_after_loading(self, layer):
        W = self._read(layer)
        if isinstance(W, dict):                    # the rounding depends on the batch size: keep both
            self._by_fused = {k: v.clone() for k, v in W.items()}
            W = W[True]
        self._perm = torch.randperm(W.numel(), generator=torch.Generator().manual_seed(1))
        self._inv = torch.argsort(self._perm)
        self._shape = W.shape
        # the repacked weights: the matrix's bits in an order only this kernel knows, (k, n) as Machete's are
        layer.qweight = param(W.reshape(-1)[self._perm].contiguous().view(torch.int16).reshape(W.shape[1], -1))
        _BY_QWEIGHT[layer.qweight.data_ptr()] = self
        layer.scales = param(layer.scales.data.flip(0).contiguous())
        if self.RULE == "marlin":
            if self.w_zp_name is None:
                self.w_zp_name = "w_zp"
                layer.w_zp = param(torch.empty(0, dtype=torch.int32))
            if getattr(layer, "bias", None) is not None:          # Marlin permutes within groups of 32 outputs
                n = layer.bias.shape[0]
                self._bperm = torch.arange(n).view(-1, 32).flip(1).reshape(-1) if n % 32 == 0 else torch.arange(n)
                layer.bias.data = layer.bias.data[self._bperm].contiguous()

    def _w(self, qweight):
        return qweight.detach().reshape(-1).view(BF)[self._inv].view(self._shape)

    def _gemm(self, qweight, x2):
        """x2 (m, k) times the kernel's weights: fp32 accumulation, one rounding (and FAULT, if set)."""
        W = self._w(qweight)
        f = type(self).FUSED
        if callable(f):
            W = self._by_fused[bool(f(x2.shape[0]))]
        y = (x2.float() @ W.float().t()).to(x2.dtype)
        if type(self).FAULT is not None:
            r, col, amount = type(self).FAULT
            if r < y.shape[0]:
                y[r, col] += amount
        return y

    def apply_weights(self, layer, x, bias=None):
        y = self._gemm(layer.qweight, x.reshape(-1, x.shape[-1]))
        if bias is not None:
            b = bias
            if self.RULE == "marlin":
                b = torch.empty_like(bias)
                b[self._bperm] = bias
            if self.RULE == "machete":
                y.add_(b)                          # vLLM's Machete: output.add_(bias) after the GEMM
            else:
                y = y + b
        return y.reshape(*x.shape[:-1], -1)


def install_kernels(monkeypatch):
    """vllm.model_executor.kernels.linear.mixed_precision.{marlin,machete} with fresh kernel classes, and
    vllm._custom_ops with vLLM's machete_mm wrapper. The Machete class runs vLLM's own apply_weights."""
    M, Runner = install_fake_vllm(monkeypatch.setitem)
    base = "vllm.model_executor.kernels.linear.mixed_precision"
    for n in ("vllm.model_executor.kernels", "vllm.model_executor.kernels.linear", base):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    monkeypatch.setattr(_vllm_custom_ops, "torch", TORCH_SHIM)
    monkeypatch.setattr(_vllm_machete, "ops", _vllm_custom_ops)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", _vllm_custom_ops)
    classes = {}
    for mod, cls, rule in (("marlin", "MarlinLinearKernel", "marlin"), ("machete", "MacheteLinearKernel", "machete")):
        m = types.ModuleType(f"{base}.{mod}")
        K = type(cls, (_Kernel,), {"RULE": rule})
        if rule == "machete":
            K.apply_weights = _vllm_machete.apply_weights
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
    """Marlin: rnd((q - z) * s). Machete: rnd(q * s - rnd(s * z)). Reading AWQ weights by the wrong kernel's
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


def test_machete_zero_points_round_once_and_the_two_readings_differ():
    """On an H100 (October 22, 2026) Machete's AWQ weights matched neither the two-rounding reading of its source
    (about 43% of values differed); its multiply and add are fused into one rounding by the compiler. The two
    readings differ in a large share of values, so probe() tells them apart, and either can be read."""
    from sdcguard import _quant
    lay = QLin(type("K", (_Kernel,), {"RULE": "machete"}), wt=U4, group=64)
    qw = _quant.from_layer(lay, "qweight", "scales", "qzeros", U4, 64, "machete", BF)
    once, twice = qw.weight(), qw.other_rounding().weight()
    assert torch.equal(once, reference(lay.codes, lay.s, lay.z, 0, 64, "machete", True))
    assert torch.equal(twice, reference(lay.codes, lay.s, lay.z, 0, 64, "machete", False))
    assert (once != twice).float().mean() > 0.1
    assert qw.rounding_varies and qw.fused and not qw.other_rounding().fused
    assert "one rounding" in qw.describe() and "rounded multiply" in qw.other_rounding().describe()
    for rule, wt, zp in (("marlin", U4, "qzeros"), ("machete", U4B8, None)):
        other = QLin(type("K", (_Kernel,), {"RULE": rule}), wt=wt, group=64)
        assert not _quant.from_layer(other, "qweight", "scales", zp, wt, 64, rule, BF).rounding_varies


def test_probe_runs_the_kernel_at_a_batch_size_in_every_range():
    """Machete picks its tile shape (a different compiled kernel) by batch size: 1-16, 17-32, 33-64, 65-128,
    129-256 and more. probe() feeds its unit vectors in batches of each, so a kernel is confirmed as served."""
    from sdcguard import _quant
    w = torch.randn(8, 1100).to(BF)
    seen = []

    def apply(x):
        seen.append(x.shape[0])
        return (x.float() @ w.float().t()).to(BF)
    r = _quant.probe(apply, w, _quant.probe_rows(1100, 128))
    assert r["mismatches"] == 0 and r["rows"] == 1024 and sum(seen) == 1024
    assert seen[:6] == [16, 32, 64, 128, 256, 512] and r["batches"] == seen and r["batches_with_mismatches"] == []
    bad = _quant.probe(lambda x: apply(x) if x.shape[0] != 128 else apply(x) * 2, w, list(range(1100)))
    assert bad["mismatches"] > 0 and bad["batches_with_mismatches"] == [128]


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
        lay.qweight.data.view(-1)[123] ^= 0x4000            # a flipped bit in the repacked weights
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


def test_a_machete_kernel_that_rounds_its_zero_points_twice_is_still_checked(monkeypatch, tmp_path):
    """The rounding is the compiler's choice: a Machete build that rounds after the multiply too is read that way
    once the probe confirms it, and the copy kept for rechecks reads it the same way."""
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    classes["machete"].FUSED = False
    model = _model(classes, rule="machete", wt=U4, group=64, bias=True)
    lay = model[0]
    g = plugin.guard()
    assert [la["name"] for la in g.layers] == ["model.layers.0.qkv_proj"]
    src = g.layers[0]["source"]
    assert "after a rounded multiply" in src.label
    assert torch.equal(src.rows(0, 192), reference(lay.codes, lay.s, lay.z, 0, 64, "machete", False))
    x = x_in()
    plain = type(lay.quant_method.kernel).__mro__[1].apply_weights(lay.quant_method.kernel, lay, x, lay.bias)
    assert torch.equal(lay(x), plain)
    assert plugin.poll(wait=True) == []


def test_a_machete_kernel_whose_rounding_changes_with_the_batch_size_is_not_checked(monkeypatch, tmp_path, caplog):
    """Neither reading holds at every batch size, so the layer isn't checked rather than checked against the wrong
    weights at some batch sizes (which would be false alarms)."""
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    classes["machete"].FUSED = lambda m: m <= 64
    model = _model(classes, rule="machete", wt=U4, group=64)
    assert plugin.guard() is None or not plugin.guard().layers
    r = plugin.layer_report(model)
    assert r["checked"] == 0 and any("didn't match" in why for why in r["unchecked_by_method"])
    assert "differed from the weights read before its repack" in caplog.text
    assert model[0](x_in()).shape == (5, 192)                # serves, unchecked


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


def test_marlins_bias_is_added_by_one_opaque_call_and_machetes_by_a_plain_add(monkeypatch, tmp_path):
    """Marlin adds the bias inside its kernel, rounded; the plugin's own addition must not be fused by torch.compile
    into the next operations (which would keep it in fp32), so it is one custom op. Machete adds it after its GEMM,
    which torch.compile does fuse; the plugin runs vLLM's own add_ for it (test_checked_machete_layers_compile_to_
    vllms_own_values)."""
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


# ---------------------------------------------------------------- Machete under torch.compile
def _rope(t, cos, sin):
    a, b = t.chunk(2, -1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def _diag():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "experiments" / "machete_diag.py"
    spec = importlib.util.spec_from_file_location("machete_diag_for_test_quant", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _machete_block(monkeypatch, tmp_path, opaque="1"):
    """Two checked Machete layers in what Inductor fuses around vLLM's: qkv_proj with a bias, its split and the
    rotary embedding of q and k; then o_proj without a bias, the residual add and an RMSNorm. Returns the plugin,
    the layers, and block(apply) where apply(kernel, layer, x, bias) is the layer's apply_weights."""
    import torch._inductor.config as ic
    M, Runner, classes = install_kernels(monkeypatch)
    plugin = plugin_env(monkeypatch, tmp_path, SDCGUARD_OPAQUE=opaque)
    plugin.register()
    monkeypatch.setattr(ic, "enable_auto_functionalized_v2", False)        # vLLM's Inductor setting
    monkeypatch.setattr(ic, "fx_graph_cache", False)
    K = classes["machete"]
    qkv, o = load(nn.ModuleList([QLin(K, k=256, n=192, wt=U4B8, group=64, bias=True,
                                      prefix="model.layers.0.self_attn.qkv_proj"),
                                 QLin(K, k=64, n=256, wt=U4, group=64, prefix="model.layers.0.self_attn.o_proj",
                                      seed=1)]))
    assert [la["name"] for la in plugin.guard().layers] == [qkv.prefix, o.prefix]
    g = torch.Generator().manual_seed(5)
    cos, sin = torch.randn(64, 32, generator=g).to(BF), torch.randn(64, 32, generator=g).to(BF)
    w_norm = (torch.rand(256, generator=g) + 0.5).to(BF)

    def block(apply):
        def f(x):
            q, k, v = apply(qkv.quant_method.kernel, qkv, x, qkv.bias).split([64, 64, 64], -1)
            q, k = _rope(q, cos, sin), _rope(k, cos, sin)
            h = x + apply(o.quant_method.kernel, o, v, None)
            hf = h.float()
            n = (hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * w_norm
            return q, k, n, h
        return f
    return plugin, (qkv, o), block


def _patched(kernel, layer, x, bias):
    return type(kernel).apply_weights(kernel, layer, x, bias)           # the plugin's Machete apply_weights


@pytest.mark.parametrize("opaque", ["1", "0"])
def test_checked_machete_layers_compile_to_vllms_own_values(monkeypatch, tmp_path, opaque):
    """Compiled with vLLM's Inductor setting, the checked layers give vLLM's own values bit for bit (vLLM's
    apply_weights, _vllm_machete, compiled the same way), with SDCGUARD_OPAQUE either way (Machete always uses the
    one op), and every check runs."""
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path, opaque)
    x = x_in(m=64)
    torch._dynamo.reset()
    try:
        want = torch.compile(block(_vllm_machete.apply_weights), fullgraph=True)(x)
        got = torch.compile(block(_patched), fullgraph=True)(x)
    finally:
        torch._dynamo.reset()
    for name, a, b in zip(("q", "k", "norm", "residual"), want, got):
        assert torch.equal(a, b), f"{name}: {int((a != b).sum())} of {a.numel()} values differ"
    c = plugin.guard().counts()
    assert c[qkv.prefix]["checks"] >= 1 and c[o.prefix]["checks"] >= 1
    assert plugin.poll(wait=True) == []


def test_checked_machete_layers_compile_to_vllms_own_kernels(monkeypatch, tmp_path):
    """Inductor generates the same kernels for the checked layers as for vLLM's own code, and calls them and the
    opaque ops in the same order: sdcguard::machete_checked where vLLM has machete_mm, nothing added (no copies of
    the check's buffers, no check op of its own). experiments/machete_diag.py compares the same on a GPU."""
    from torch._inductor.utils import run_and_get_code
    diag = _diag()
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path)
    x = x_in(m=64)
    torch._dynamo.reset()
    try:
        _, want = run_and_get_code(torch.compile(block(_vllm_machete.apply_weights), fullgraph=True), x)
        torch._dynamo.reset()
        _, got = run_and_get_code(torch.compile(block(_patched), fullgraph=True), x)
    finally:
        torch._dynamo.reset()
    assert len(want) == len(got) == 1
    w, g = diag.kernels_and_calls(want[0]), diag.kernels_and_calls(got[0])
    assert w["kernels"] and g["kernels"] == w["kernels"]
    assert g["calls"] == w["calls"] and g["calls"].count("X:MACHETE") == 2
    assert "sdcguard.machete_checked" in got[0] and "sdcguard_test.machete_mm" in want[0]
    assert "check_static" not in got[0]


def test_the_kernel_comparison_sees_a_check_op_of_its_own(monkeypatch, tmp_path):
    """The comparison above can fail: the plugin's earlier design (Machete, then the check as an op of its own, then
    y + bias) compiles to another call sequence. (On the CPU with PyTorch 2.14.1 it also generated another kernel for
    the bias and rotary embedding, storing the bias sum rounded to bf16 where vLLM's kernel keeps it in fp32, and
    other q and k values; on an H100 that design changed 2 to 3 of 64 completions.)"""
    from torch._inductor.utils import run_and_get_code
    from sdcguard.serve import _op
    diag = _diag()
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path)
    op = _op()

    def separate(kernel, layer, x, bias):
        y = _vllm_machete.apply_weights(kernel, layer, x, None)
        op(x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1]), *layer._sdcguard_args[0])
        return y + bias if bias is not None else y
    x = x_in(m=64)
    torch._dynamo.reset()
    try:
        _, want = run_and_get_code(torch.compile(block(_vllm_machete.apply_weights), fullgraph=True), x)
        torch._dynamo.reset()
        _, got = run_and_get_code(torch.compile(block(separate), fullgraph=True), x)
    finally:
        torch._dynamo.reset()
    w, g = diag.kernels_and_calls(want[0]), diag.kernels_and_calls(got[0])
    assert g["calls"] != w["calls"] and g["calls"].count("X:sdcguard.check_static") == 2
    cmp = diag.compare_code({"kernels": w["kernels"], "sequences": [w["calls"]], "with_code": 1},
                            {"kernels": g["kernels"], "sequences": [g["calls"]], "with_code": 1})
    assert cmp["captured"] and not cmp["same"]


def test_a_wrong_machete_output_is_caught_under_torch_compile(monkeypatch, tmp_path):
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path)
    K = type(qkv.quant_method.kernel)
    x = x_in(m=64)
    torch._dynamo.reset()
    try:
        f = torch.compile(block(_patched), fullgraph=True)
        f(x)
        assert plugin.poll(wait=True) == []
        K.FAULT = (7, 100, 3.0)
        f(x)
    finally:
        K.FAULT = None
        torch._dynamo.reset()
    ev = plugin.poll(wait=True)
    # FAULT is the stand-in Machete class's: both layers' GEMMs computed row 7, output 100 wrong (each check judges
    # its own layer's output against its own input, so qkv's wrong output doesn't make o_proj alarm by itself)
    assert sorted(e.detail["where"] for e in ev) == sorted([qkv.prefix, o.prefix])
    for e in ev:
        assert e.detail["row"] == 7 and e.detail["recheck"]["verdict"] == "output_wrong", e.detail


def test_the_machete_op_takes_the_check_ops_arguments_without_the_buffers_it_writes(monkeypatch, tmp_path):
    """serve.OP_BUFFERS names where op_args has the tensors the check writes, which the op looks up instead
    (ServeGuard.buffers); what's left is what it takes after Machete's own arguments, in its schema's order."""
    from sdcguard import serve
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path)
    g = plugin.guard()
    names = [a.name for a in plugin._machete_op().default._schema.arguments][6:]
    check = [a.name for a in torch.ops.sdcguard.check_static.default._schema.arguments][2:]
    assert [check[i] for i in serve.OP_BUFFERS] == ["state", "cap_meta", "cap_data"]
    assert names == [n for n in check if n not in ("state", "cap_meta", "cap_data")]
    for slot in range(2):
        args = g.op_args(slot, False)
        assert len(args) == len(check)
        assert all(args[i] is b for i, b in zip(serve.OP_BUFFERS, g.buffers(slot)))
        rest = plugin._without_buffers(args, serve.OP_BUFFERS)
        assert len(rest) == len(names) and all(a is b for a, b in zip(rest, [a for i, a in enumerate(args)
                                                                             if i not in serve.OP_BUFFERS]))


@torch.library.custom_op("sdcguard_test::tagged", mutates_args=(), tags=(torch.Tag.needs_fixed_stride_order,))
def _tagged(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def test_the_machete_op_gets_machete_mms_layout_tags(monkeypatch, tmp_path):
    """Inductor lays out a custom op's inputs by its layout tag, or by one default for every op without one: the
    plugin's op takes machete_mm's (here the stand-in's: none), so the two are laid out alike either way."""
    from sdcguard import vllm_plugin
    assert vllm_plugin._layout_tags(torch.ops.sdcguard_test.tagged) == (torch.Tag.needs_fixed_stride_order,)
    assert vllm_plugin._layout_tags(torch.ops.sdcguard_test.machete_mm) == ()
    plugin, (qkv, o), block = _machete_block(monkeypatch, tmp_path)
    assert plugin._MACHETE["tags"] == () and plugin._layout_tags(plugin._MACHETE["op"]) == ()


def test_the_copied_machete_code_is_the_code_its_fingerprints_name():
    """_vllm_machete and _vllm_custom_ops are vLLM 0.31.0's code verbatim; the fingerprints in _quant were computed
    from vLLM 0.31.0, 0.32.0 and main's source files (identical there), on Pythons 3.10 to 3.13."""
    from sdcguard import _quant
    assert _quant.code_fingerprint(_vllm_machete.apply_weights) in _quant.MACHETE_APPLY_KNOWN
    assert _quant.code_fingerprint(_vllm_custom_ops.machete_mm) in _quant.MACHETE_MM_KNOWN


def test_code_fingerprints_ignore_formatting_and_catch_changes(tmp_path, monkeypatch):
    import importlib
    from sdcguard import _quant
    variants = {
        "same": "def f(self, x, bias=None):\n    y = g(x)  # comment\n    if bias is not None:\n        y.add_(bias)"
                "\n    return y\n",
        "formatted": 'def f(\n    self,\n    x: int,\n    bias: "T | None" = None,\n) -> int:\n    """Doc."""\n'
                     "    y = g(\n        x,\n    )\n\n    if bias is not None:\n        y.add_(bias)\n    return y\n",
        "changed": "def f(self, x, bias=None):\n    y = g(x)\n    if bias is not None:\n        y = y + bias\n"
                   "    return y\n",
        "default": "def f(self, x, bias=0):\n    y = g(x)\n    if bias is not None:\n        y.add_(bias)\n"
                   "    return y\n",
    }
    monkeypatch.syspath_prepend(str(tmp_path))
    fp = {}
    for name, src in variants.items():
        (tmp_path / f"fp_{name}.py").write_text(src)
        fp[name] = _quant.code_fingerprint(importlib.import_module(f"fp_{name}").f)
    assert fp["same"] == fp["formatted"] and fp["same"] is not None
    assert len({fp["same"], fp["changed"], fp["default"]}) == 3
    assert _quant.code_fingerprint(len) is None                          # no Python source


def test_machete_code_other_than_vllms_known_one_is_not_checked(monkeypatch, tmp_path, caplog):
    """A vLLM whose Machete apply_weights isn't the code the plugin copies: its layers run vLLM's own code,
    unchecked, and the start-up count says why."""
    M, Runner, classes = install_kernels(monkeypatch)
    monkeypatch.setattr(classes["machete"], "apply_weights", _Kernel.apply_weights)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _model(classes, rule="machete", wt=U4B8, group=64, bias=True)
    assert plugin.guard() is None or not plugin.guard().layers
    r = plugin.layer_report(model)
    assert r["checked"] == 0 and r["quantized_or_moe"] == 1
    assert any("other than vLLM 0.31.0's and 0.32.0's" in why for why in r["unchecked_by_method"]), r
    assert "isn't the code sdcguard's check copies" in caplog.text
    lay, x = model[0], x_in()
    assert torch.equal(lay(x), _Kernel.apply_weights(lay.quant_method.kernel, lay, x, lay.bias))
    assert plugin.layer_report(load(nn.ModuleList([QLin(classes["marlin"], wt=U4B8, group=64)])))["checked"] == 1


def test_the_vllm_experiment_flips_nothing_when_no_quantized_layer_is_checked(monkeypatch, tmp_path):
    """On October 22 Machete's AWQ layers were all left unchecked, and the experiment then stopped trying to flip a
    bf16 weight the quantized layer doesn't have; now it records that nothing was flipped."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "experiments" / "vllm_check.py"
    spec = importlib.util.spec_from_file_location("vllm_check_flip_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plugin = plugin_env(monkeypatch, tmp_path)
    assert plugin.guard() is None

    class Model:
        def named_modules(self):
            return [("model.layers.10.mlp.down_proj", nn.Module())]       # quantized: qweight, no weight
    assert module._flip(Model()) is None
