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
              "SDCGUARD_SHARED_CACHE", "SDCGUARD_OPAQUE", "SDCGUARD_Z", "SDCGUARD_NORMS", "SDCGUARD_NORM_Z",
              "SDCGUARD_MOE", "SDCGUARD_FP8"):
        monkeypatch.delenv(k, raising=False)
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    for k, v in (("patched", False), ("guard", None), ("reported", False), ("layers", None), ("steps", 0),
                 ("rechecked", None), ("z", None), ("hooked", False), ("step_log", None), ("norm_classes", []),
                 ("norms_stale", False), ("norms_registered", False), ("moe_classes", []), ("moe_registered", False),
                 ("stack", None), ("fp8_classes", []), ("fp8_registered", False), ("route_classes", []),
                 ("fp8", None), ("fp8_why", None)):
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


def moe_kernel_like(x, w13, w2, ids, g, act=0, dtype=torch.bfloat16, blk=64, expert_map=None):
    """vLLM's Triton MoE kernel's rounding (TritonExperts, 0.31/0.32), with another order of fp32 additions (blocked
    partial sums): the first matmul rounded, silu rounded then times the up projection rounded, the second matmul
    times the routing weight rounded, the experts summed in fp32 and rounded once."""
    from sdcguard.serve_moe import _act
    m, h = x.shape
    out = torch.zeros(m, h)
    for t in range(m):
        acc = torch.zeros(h)
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            if expert_map is not None and e >= 0:
                e = int(expert_map[e])
            if e < 0:
                continue
            w, xf = w13[e].float(), x[t].float()
            a = sum(w[:, i:i + blk] @ xf[i:i + blk] for i in range(0, h, blk)).to(dtype).float()
            half = a.shape[0] // 2
            hv = (_act(a[:half], act).to(dtype).float() * a[half:]).to(dtype).float()
            w2e = w2[e].float()
            c = sum(w2e[:, i:i + blk] @ hv[i:i + blk] for i in range(0, half, blk))
            acc += (c * float(g[t, k])).to(dtype).float()
        out[t] = acc
    return out.to(dtype)


def install_fake_moe(setitem):
    """vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts as vLLM 0.32 has it, as far as the plugin
    uses it: w13_weight [E, 2I, H] and w2_weight [E, H, I], quant_method (UnquantizedFusedMoEMethod with
    unquantized_backend TRITON), activation, top_k, apply_router_weight_on_input, expert_map, and forward_modular
    (called by vLLM's MoE runner with the routing already chosen; here it computes as the Triton kernel rounds)."""
    import enum
    for n in ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe"]:
        if n not in sys.modules:
            setitem(sys.modules, n, types.ModuleType(n))
    mod = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")

    class UnquantizedMoeBackend(enum.Enum):
        TRITON = "TRITON"
        FLASHINFER_CUTLASS = "FlashInfer CUTLASS"

    class MoEActivation(enum.Enum):
        SILU = "silu"
        GELU = "gelu"
        SWIGLUOAI = "swigluoai"

    class UnquantizedFusedMoEMethod:
        def __init__(self, backend):
            self.unquantized_backend = backend

    class RoutedExperts(nn.Module):
        def __init__(self, experts=8, hidden=128, inter=64, top_k=2, backend="TRITON", act="SILU",
                     dtype=torch.bfloat16, seed=0):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.w13_weight = nn.Parameter((torch.randn(experts, 2 * inter, hidden, generator=g) * 0.05).to(dtype),
                                           requires_grad=False)
            self.w2_weight = nn.Parameter((torch.randn(experts, hidden, inter, generator=g) * 0.05).to(dtype),
                                          requires_grad=False)
            self.quant_method = UnquantizedFusedMoEMethod(UnquantizedMoeBackend[backend])
            self.activation = MoEActivation[act]
            self.top_k, self.apply_router_weight_on_input, self.expert_map = top_k, False, None
            self.dtype = dtype
            self.fault = None

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
            from sdcguard.serve_moe import ACT
            out = moe_kernel_like(x, self.w13_weight, self.w2_weight, topk_ids, topk_weights,
                                  ACT[self.activation.value], self.dtype)
            if self.fault is not None:
                out = self.fault(out)
            return out

        def forward(self, *a, **k):
            raise AssertionError("Call forward_modular instead.")

    for c in (RoutedExperts, UnquantizedFusedMoEMethod):
        c.__module__ = mod.__name__
    mod.RoutedExperts, mod.UnquantizedFusedMoEMethod = RoutedExperts, UnquantizedFusedMoEMethod
    mod.UnquantizedMoeBackend, mod.MoEActivation = UnquantizedMoeBackend, MoEActivation
    setitem(sys.modules, mod.__name__, mod)
    sys.modules["vllm.model_executor.layers.fused_moe"].routed_experts = mod
    return mod


class MoEDecoder(nn.Module):
    """A decoder with a fused MoE layer per block, routed as vLLM's runner does (softmax, top-k, renormalized),
    then forward_modular."""

    def __init__(self, moe_mod, M, n_layers=3, hidden=128, experts=8, top_k=2):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([Lin(M, hidden, hidden, f"model.layers.{i}.proj") for i in range(n_layers)])
        self.gates = [torch.randn(experts, hidden) * 0.1 for _ in range(n_layers)]
        self.experts = nn.ModuleList([moe_mod.RoutedExperts(experts, hidden, 64, top_k, seed=i)
                                      for i in range(n_layers)])

    def forward(self, x):
        for lay, gate, ex in zip(self.layers, self.gates, self.experts):
            x = lay(x)
            p = torch.softmax(x.float() @ gate.T, -1)
            w, ids = torch.topk(p, ex.top_k, -1)
            w = w / w.sum(-1, keepdim=True)
            x = ex.forward_modular(x=x, topk_weights=w, topk_ids=ids.to(torch.int32))
        return x


def vllm_grouped_topk(gating_output, topk, renormalize, num_expert_group=0, topk_group=0, scoring_func="softmax",
                      routed_scaling_factor=1.0, e_score_correction_bias=None):
    """vLLM 0.32.0's grouped_topk, its PyTorch path (fused_moe/router/grouped_topk_router.py), transcribed."""
    scores = torch.softmax(gating_output, dim=-1) if scoring_func == "softmax" else gating_output.sigmoid()
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    else:
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(num_token, num_expert_group,
                                                 scores.size(-1) // num_expert_group).reshape(num_token, -1)
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def vllm_topk(gating_output, topk, renormalize, scoring_func="softmax"):
    """vLLM's fused top-k (topk_softmax / topk_sigmoid kernels) in PyTorch: scores in fp32, top-k, renormalized."""
    s = torch.softmax(gating_output.float(), -1) if scoring_func == "softmax" else gating_output.float().sigmoid()
    w, ids = torch.topk(s, topk, -1)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return w, ids.to(torch.int32)


def install_fake_router(setitem):
    """vllm.model_executor.layers.fused_moe.router.fused_moe_router.FusedMoERouter and two of its routers as vLLM 0.32
    has them, as far as the plugin uses them: select_experts calling _select_experts (vLLM's MoE runner calls
    select_experts with keywords), and the routers' attributes (top_k, global_num_experts, renormalize,
    scoring_func, e_score_correction_bias, num_expert_group, topk_group, routed_scaling_factor, eplb_state)."""
    for n in ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fused_moe",
              "vllm.model_executor.layers.fused_moe.router"]:
        if n not in sys.modules:
            setitem(sys.modules, n, types.ModuleType(n))
    mod = types.ModuleType("vllm.model_executor.layers.fused_moe.router.fused_moe_router")

    class FusedMoERouter:
        def __init__(self, eplb_state=None):
            self.eplb_state = eplb_state
            self.fault = None

        def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            raise NotImplementedError

        def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            return self._select_experts(hidden_states, router_logits, topk_indices_dtype=topk_indices_dtype,
                                        input_ids=input_ids)

    class FusedTopKRouter(FusedMoERouter):
        def __init__(self, top_k, global_num_experts, renormalize=True, scoring_func="softmax"):
            super().__init__()
            self.top_k, self.global_num_experts = top_k, global_num_experts
            self.renormalize, self.scoring_func = renormalize, scoring_func

        def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            w, ids = vllm_topk(router_logits, self.top_k, self.renormalize, self.scoring_func)
            return self.fault(w, ids) if self.fault is not None else (w, ids)

    class GroupedTopKRouter(FusedMoERouter):
        def __init__(self, top_k, global_num_experts, num_expert_group, topk_group, renormalize=True,
                     scoring_func="sigmoid", routed_scaling_factor=1.0, e_score_correction_bias=None):
            super().__init__()
            self.top_k, self.global_num_experts = top_k, global_num_experts
            self.num_expert_group, self.topk_group = num_expert_group, topk_group
            self.renormalize, self.scoring_func = renormalize, scoring_func
            self.routed_scaling_factor, self.e_score_correction_bias = routed_scaling_factor, e_score_correction_bias
            self.num_fused_shared_experts = 0

        def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
            w, ids = vllm_grouped_topk(router_logits, self.top_k, self.renormalize, self.num_expert_group,
                                       self.topk_group, self.scoring_func, self.routed_scaling_factor,
                                       self.e_score_correction_bias)
            return self.fault(w, ids) if self.fault is not None else (w, ids)

    for c in (FusedMoERouter, FusedTopKRouter, GroupedTopKRouter):
        c.__module__ = mod.__name__
        setattr(mod, c.__name__, c)
    setitem(sys.modules, mod.__name__, mod)
    sys.modules["vllm.model_executor.layers.fused_moe.router"].fused_moe_router = mod
    return mod


class MoERunnerLike(nn.Module):
    """vLLM's MoERunner as far as the plugin sees it: a module holding `router` and `routed_experts`."""

    def __init__(self, router, routed_experts):
        super().__init__()
        self.router = router
        self.routed_experts = routed_experts


class RoutedMoEDecoder(nn.Module):
    """A decoder whose MoE blocks route through vLLM-like routers, called as vLLM's runner calls them (fp32 router
    logits), then forward_modular."""

    def __init__(self, moe_mod, router_mod, M, n_layers=3, hidden=128, experts=16, top_k=4, grouped=False):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([Lin(M, hidden, hidden, f"model.layers.{i}.proj") for i in range(n_layers)])
        self.gates = [torch.randn(experts, hidden) * 0.1 for _ in range(n_layers)]
        bias = torch.linspace(-0.05, 0.05, experts)
        mk = (lambda: router_mod.GroupedTopKRouter(top_k, experts, 4, 2, True, "sigmoid", 2.5, bias)) if grouped \
            else (lambda: router_mod.FusedTopKRouter(top_k, experts))
        self.runners = nn.ModuleList([MoERunnerLike(mk(), moe_mod.RoutedExperts(experts, hidden, 64, top_k, seed=i))
                                      for i in range(n_layers)])

    def forward(self, x):
        for lay, gate, run in zip(self.layers, self.gates, self.runners):
            x = lay(x)
            w, ids = run.router.select_experts(hidden_states=x, router_logits=x.float() @ gate.T,
                                               topk_indices_dtype=None, input_ids=None)
            x = run.routed_experts.forward_modular(x=x, topk_weights=w, topk_ids=ids)
        return x
