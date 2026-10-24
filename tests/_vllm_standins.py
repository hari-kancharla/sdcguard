"""Stand-ins for the vLLM classes the plugin touches (vLLM isn't installed for the tests), as test_serve.py has
them, plus the layer kinds a served model holds besides plain linear layers. Each mirrors the vLLM class it names,
checked against vLLM main 8e2beaa and v0.32.0 (Oct 2026)."""
import sys
import types

import torch
import torch.nn as nn


def install_fake_vllm(setitem):
    """vllm.model_executor.layers.linear.UnquantizedLinearMethod and the V1 GPUModelRunner."""
    linear = types.ModuleType("vllm.model_executor.layers.linear")

    class UnquantizedLinearMethod:
        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    linear.UnquantizedLinearMethod = UnquantizedLinearMethod
    runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def __init__(self, model):
            self.model = model

        def execute_model(self, x):
            return self.model(x)

    runner.GPUModelRunner = GPUModelRunner
    for n in ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.v1", "vllm.v1.worker"]:
        setitem(sys.modules, n, types.ModuleType(n))
    setitem(sys.modules, "vllm.model_executor.layers.linear", linear)
    setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", runner)
    sys.modules["vllm.model_executor.layers"].linear = linear
    sys.modules["vllm.v1.worker"].gpu_model_runner = runner
    return UnquantizedLinearMethod, GPUModelRunner


def plugin_env(monkeypatch, tmp_path, **extra):
    """The plugin on, on the CPU, reading results only when a test asks; nothing left from an earlier test."""
    from sdcguard import vllm_plugin
    monkeypatch.setenv("SDCGUARD", "1")
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")
    monkeypatch.setenv("SDCGUARD_POLL_EVERY", "100000")
    monkeypatch.setenv("SDCGUARD_POLL_SECONDS", "1e9")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    for k in ("SDCGUARD_REQUIRE", "SDCGUARD_SHARE", "SDCGUARD_OFFSET", "SDCGUARD_WEBHOOK", "SDCGUARD_METRICS_PORT",
              "SDCGUARD_SHARED_CACHE", "SDCGUARD_OPAQUE", "SDCGUARD_Z", "SDCGUARD_NORMS", "SDCGUARD_NORM_Z"):
        monkeypatch.delenv(k, raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    for k, v in (("patched", False), ("guard", None), ("reported", False), ("layers", None), ("steps", 0),
                 ("rechecked", None), ("z", None), ("hooked", False), ("step_log", None), ("norm_classes", []),
                 ("norms_stale", False), ("norms_registered", False)):
        monkeypatch.setitem(vllm_plugin._STATE, k, v)
    return vllm_plugin


class Lin(nn.Module):
    """vLLM's LinearBase: forward calls quant_method.apply (ReplicatedLinear / Column / RowParallelLinear)."""

    def __init__(self, M, k, n, prefix, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k).to(torch.bfloat16) * 0.05, requires_grad=False)
        self.bias = nn.Parameter(torch.randn(n).to(torch.bfloat16), requires_grad=False) if bias else None
        self.prefix, self.quant_method = prefix, M()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


class UnquantizedEmbeddingMethod:
    """vllm/model_executor/layers/vocab_parallel_embedding.py: the LM head's (and embedding's) method."""

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class ParallelLMHead(nn.Module):
    """vLLM's ParallelLMHead(VocabParallelEmbedding): weight + UnquantizedEmbeddingMethod; the logits processor
    calls quant_method.apply. Llama and Qwen2 build one even with tied embeddings."""

    def __init__(self, vocab, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab, hidden).to(torch.bfloat16), requires_grad=False)
        self.quant_method = UnquantizedEmbeddingMethod()


class Decoder(nn.Module):
    def __init__(self, M, n_layers=4, width=256, lm_head=True, extra=None):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([Lin(M, width, width, f"model.layers.{i}.proj") for i in range(n_layers)])
        self.extra = nn.ModuleList(extra or [])
        if lm_head:
            self.lm_head = ParallelLMHead(1024, width)

    def forward(self, x):
        for lay in self.layers:
            x = torch.relu(lay(x))
        for e in self.extra:
            e(x)
        return x


def load(model):
    """vLLM's model_loader.utils.process_weights_after_loading: every module with a quant_method (also what a
    weight reload calls again)."""
    for mod in model.modules():
        qm = getattr(mod, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(mod)
    return model


def x_in():
    return torch.randn(3, 256).to(torch.bfloat16)


class UnquantizedFusedMoEMethod:
    def process_weights_after_loading(self, layer):
        pass


class RoutedExperts(nn.Module):
    """vLLM's fused-MoE expert container (vllm/model_executor/layers/fused_moe/routed_experts.py, FusedMoE in
    older releases): quant_method plus w13_weight [E, 2I, H] and w2_weight [E, H, I]; no `weight` attribute."""

    def __init__(self, experts=8, hidden=256, inter=512):
        super().__init__()
        self.quant_method = UnquantizedFusedMoEMethod()
        self.w13_weight = nn.Parameter(torch.randn(experts, 2 * inter, hidden).to(torch.bfloat16),
                                       requires_grad=False)
        self.w2_weight = nn.Parameter(torch.randn(experts, hidden, inter).to(torch.bfloat16), requires_grad=False)

    def forward(self, x):
        return x


class ConvAsLinear(nn.Module):
    """vLLM's Mamba/ShortConv conv1d: a ColumnParallelLinear (UnquantizedLinearMethod) whose weight is unsqueezed
    to [dim, 1, width] in __init__ and used by causal_conv1d kernels, never through apply."""

    def __init__(self, M):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(256, 1, 4).to(torch.bfloat16), requires_grad=False)
        self.prefix, self.quant_method = "model.layers.0.mixer.conv1d", M()

    def forward(self, x):
        return x


class GateLinear(nn.Module):
    """vLLM's MoE router GateLinear(ReplicatedLinear): UnquantizedLinearMethod and a bf16 weight. With out_dtype
    float32 its forward on NVIDIA GPUs is torch.mm(x, W.T, out_dtype=fp32) or ll_bf16_gemm, never
    quant_method.apply (fused_moe/router/gate_linear.py, main and v0.32.0). flags=False stands for a forward that
    bypasses apply without saying so."""

    def __init__(self, M, hidden=256, experts=16, flags=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(experts, hidden).to(torch.bfloat16) * 0.05, requires_grad=False)
        self.prefix, self.quant_method = "model.layers.0.mlp.gate", M()
        if flags:
            self.allow_cublas_router_gemm, self.allow_ll_bf16_gemm = True, False

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight).float()       # stands in for torch.mm(..., out_dtype=fp32)


def install_fake_layernorm(setitem, custom_op_enabled=True):
    """vllm.model_executor.layers.layernorm's RMSNorm and GemmaRMSNorm as vLLM 0.31/0.32 have them: CustomOp binds
    the forward when the layer is built (forward_cuda, which calls forward_native, or forward_native itself), and
    forward_native is ir.ops.rms_norm / fused_add_rms_norm's native version (fp32 math, the normalized row rounded
    before the multiply by w; with a residual, the fp32 sum normalized and the rounded sum returned)."""
    ln = types.ModuleType("vllm.model_executor.layers.layernorm")
    ln.__name__ = "vllm.model_executor.layers.layernorm"

    def rms_norm(x, weight, eps):
        orig = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig)

    def fused_add_rms_norm(x, res, weight, eps):
        orig = x.dtype
        x = x.to(torch.float32) + res.to(torch.float32)
        res = x.to(orig)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        if weight is not None:
            x = x.to(weight.dtype) * weight
        return x.to(orig), res

    class CustomOp(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_cuda if custom_op_enabled else self.forward_native

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

    class RMSNorm(CustomOp):
        def __init__(self, hidden_size, eps=1e-6, var_hidden_size=None, has_weight=True, dtype=None):
            super().__init__()
            self.hidden_size, self.variance_epsilon = hidden_size, eps
            self.variance_size_override = None if var_hidden_size == hidden_size else var_hidden_size
            self.has_weight = has_weight
            self.weight = torch.ones(hidden_size, dtype=dtype or torch.get_default_dtype())
            if has_weight:
                self.weight = nn.Parameter(self.weight, requires_grad=False)

        def forward_native(self, x, residual=None):
            w = self.weight.data if self.has_weight else None
            if residual is None:
                return rms_norm(x, w, self.variance_epsilon)
            return fused_add_rms_norm(x, residual, w, self.variance_epsilon)

        def forward_cuda(self, x, residual=None):
            return self.forward_native(x, residual)

    class GemmaRMSNorm(CustomOp):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
            self.variance_epsilon = eps

        def forward_native(self, x, residual=None):
            weight = self.weight.float() + 1.0
            if residual is None:
                return rms_norm(x, weight, self.variance_epsilon)
            return fused_add_rms_norm(x, residual, weight, self.variance_epsilon)

        def forward_cuda(self, x, residual=None):
            return self.forward_native(x, residual)

    class LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return torch.nn.functional.layer_norm(x, x.shape[-1:], self.weight)

    for c in (CustomOp, RMSNorm, GemmaRMSNorm, LayerNorm):
        c.__module__ = ln.__name__
    ln.RMSNorm, ln.GemmaRMSNorm, ln.LayerNorm, ln.CustomOp = RMSNorm, GemmaRMSNorm, LayerNorm, CustomOp
    setitem(sys.modules, "vllm.model_executor.layers.layernorm", ln)
    sys.modules["vllm.model_executor.layers"].layernorm = ln
    return ln


class NormDecoder(nn.Module):
    """A decoder the way vLLM's Llama/Qwen2 use norms: input_layernorm(x) for the first layer and
    input_layernorm(x, residual) after it, post_attention_layernorm(x, residual), and the final norm(x, residual);
    plus one LayerNorm the check doesn't know. Built after the plugin is registered, as vLLM builds its model."""

    def __init__(self, ln, M, n_layers=3, width=256, dtype=torch.bfloat16):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([Lin(M, width, width, f"model.layers.{i}.proj") for i in range(n_layers)])
        self.inorm = nn.ModuleList([ln.RMSNorm(width, dtype=dtype) for _ in range(n_layers)])
        self.pnorm = nn.ModuleList([ln.RMSNorm(width, dtype=dtype) for _ in range(n_layers)])
        self.norm = ln.RMSNorm(width, dtype=dtype)
        self.vision_norm = ln.LayerNorm(width)
        with torch.no_grad():
            for n in [*self.inorm, *self.pnorm, self.norm]:
                n.weight.copy_((1 + 0.2 * torch.randn(width)).to(dtype))

    def forward(self, x):
        residual = None
        for i, lay in enumerate(self.layers):
            if residual is None:
                residual, x = x, self.inorm[i](x)
            else:
                x, residual = self.inorm[i](x, residual)
            x = lay(x)
            x, residual = self.pnorm[i](x, residual)
            x = torch.relu(x)
        x, _ = self.norm(x, residual)
        return x
