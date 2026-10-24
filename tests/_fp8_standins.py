"""Stand-ins for vLLM's FP8 W8A8 path (vLLM isn't installed for the tests): the scaled-mm kernel base class with
vLLM 0.31/0.32's apply_weights (the code the plugin's copy must match; Apache-2.0, from vLLM), its input
quantization (per-token by division, per-tensor over every row it's given, as vLLM's CUDA kernels), a CUTLASS-like
GEMM, and the MoE input quantization (vllm.model_executor.layers.fused_moe.utils.moe_kernel_quantize_input) as a
Triton MoE experts module imports it."""
import sys
import types

import torch

FP8_MAX = 448.0


class QuantizedActivation:
    def __init__(self, data, scale, orig_shape, orig_dtype, quant_key=None):
        self.data, self.scale, self.orig_shape, self.orig_dtype, self.quant_key = (data, scale, orig_shape,
                                                                                    orig_dtype, quant_key)


def as_quantized_activation(x, expected_key):
    if not isinstance(x, QuantizedActivation):
        return None
    return x


def quant_per_token(x):
    amax = x.float().abs().amax(-1, keepdim=True)
    s = (amax / FP8_MAX).clamp(min=1.0 / (FP8_MAX * 512.0))
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def quant_per_tensor(x):
    s = (x.float().abs().amax() / FP8_MAX).reshape(1)
    return (x.float() * (1.0 / s)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


class QuantFP8:
    def __init__(self, static=False, per_token=True):
        self.static, self.use_per_token_if_dynamic, self.is_group_quant = static, per_token, False
        self.group_shape = types.SimpleNamespace(name="PER_TOKEN" if per_token else "PER_TENSOR")

    def __call__(self, x, scale=None, scale_ub=None):
        if self.static:
            return (x.float() * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), scale
        return quant_per_token(x) if self.use_per_token_if_dynamic else quant_per_tensor(x)


class FP8ScaledMMLinearKernel:
    def __init__(self, per_token=True, static=False):
        self.quant_fp8 = QuantFP8(static=static, per_token=per_token)
        self.fp8_dtype = torch.float8_e4m3fn
        self.config = types.SimpleNamespace(out_dtype=None)
        self.layer_param_names = ("weight", "weight_scale", "input_scale", "input_scale_ub")

    def input_quant_key(self):
        return None

    def _get_layer_params(self, layer):
        w, w_s, x_s, x_s_ub = self.layer_param_names
        return (getattr(layer, w), getattr(layer, w_s), getattr(layer, x_s, None), getattr(layer, x_s_ub, None))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: "torch.Tensor | QuantizedActivation",
        bias: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        fp8_dtype = self.fp8_dtype
        maybe_out_dtype = self.config.out_dtype
        w, w_s, x_s, x_s_ub = self._get_layer_params(layer)

        qa = as_quantized_activation(x, self.input_quant_key())
        if qa is not None:
            x_data, x_s = qa.data, qa.scale
            orig_shape, orig_dtype = qa.orig_shape, qa.orig_dtype
            assert x_data.dtype == fp8_dtype
        else:
            assert isinstance(x, torch.Tensor)
            x_data = x
            orig_shape, orig_dtype = x.shape, x.dtype

        x_2d = x_data.view(-1, x_data.shape[-1])
        output_shape = [*orig_shape[:-1], w.shape[1]]
        out_dtype = orig_dtype if maybe_out_dtype is None else maybe_out_dtype

        x_2d_q = x_2d
        if qa is None:
            x_2d_q, x_s = self.quant_fp8(x_2d, x_s, x_s_ub)
        return self.apply_scaled_mm(
            A=x_2d_q,
            B=w,
            out_dtype=out_dtype,
            As=x_s,
            Bs=w_s,
            bias=bias,
            output_shape=output_shape,
        )


class CutlassFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):
        y = (A.float() * As.float()) @ (B.float() * Bs.float())
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype).view(*output_shape)


class Fp8LinearMethod:
    def __init__(self, kernel):
        self.fp8_linear = kernel

    def apply(self, layer, x, bias=None):
        return self.fp8_linear.apply_weights(layer, x, bias)


class Fp8Linear(torch.nn.Module):
    """vLLM's linear layer with Fp8LinearMethod: weight [K, N] in FP8 (as the scaled-mm kernels hold it), per-channel
    weight scales."""

    def __init__(self, k, n, prefix, per_token=True, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(k, n, generator=g) * 0.05
        ws = (w.abs().amax(0) / FP8_MAX).reshape(1, n)
        self.weight = torch.nn.Parameter((w / ws).to(torch.float8_e4m3fn), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(ws.float(), requires_grad=False)
        self.input_scale = None
        self.prefix = prefix
        smk = sys.modules["vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel"]
        self.quant_method = Fp8LinearMethod(smk.CutlassFP8ScaledMMLinearKernel(per_token=per_token))

    def forward(self, x):
        return self.quant_method.apply(self, x)


def moe_kernel_quantize_input(A, A_scale, quant_dtype, per_act_token_quant, block_shape=None, is_scale_swizzled=True,
                              quantization_emulation=False, mx_alignment=0):
    if quant_dtype != torch.float8_e4m3fn:
        return A, A_scale
    return quant_per_token(A) if per_act_token_quant else quant_per_tensor(A)


def install(setitem):
    """The FP8 modules under vllm.* (on top of tests/_vllm_standins.install_fake_vllm)."""
    smk = types.ModuleType("vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel")
    # fresh classes each time: the plugin patches them, and one test's patch mustn't reach the next
    smk.ScaledMMLinearKernel = type("ScaledMMLinearKernel", (), {})
    base = type("FP8ScaledMMLinearKernel", (FP8ScaledMMLinearKernel, smk.ScaledMMLinearKernel),
                {"apply_weights": FP8ScaledMMLinearKernel.apply_weights})
    smk.FP8ScaledMMLinearKernel = base
    smk.CutlassFP8ScaledMMLinearKernel = type("CutlassFP8ScaledMMLinearKernel", (base,),
                                              {"apply_scaled_mm": CutlassFP8ScaledMMLinearKernel.apply_scaled_mm})
    smk.as_quantized_activation = as_quantized_activation
    utils = types.ModuleType("vllm.model_executor.layers.fused_moe.utils")
    utils.moe_kernel_quantize_input = moe_kernel_quantize_input
    tm = types.ModuleType("vllm.model_executor.layers.fused_moe.experts.triton_moe")
    tm.moe_kernel_quantize_input = moe_kernel_quantize_input
    for n in ["vllm.model_executor.kernels", "vllm.model_executor.kernels.linear",
              "vllm.model_executor.kernels.linear.scaled_mm", "vllm.model_executor.layers.fused_moe",
              "vllm.model_executor.layers.fused_moe.experts"]:
        if n not in sys.modules:
            setitem(sys.modules, n, types.ModuleType(n))
    setitem(sys.modules, smk.__name__, smk)
    setitem(sys.modules, utils.__name__, utils)
    setitem(sys.modules, tm.__name__, tm)
    re_ = types.ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")
    re_.RoutedExperts = type("RoutedExperts", (RoutedExperts,), {"forward_modular": RoutedExperts.forward_modular})
    setitem(sys.modules, re_.__name__, re_)
    # as vLLM's scaled_mm/__init__.py: the package exports the kernel classes, and under the name of the module
    # ScaledMMLinearKernel.py, the ScaledMMLinearKernel class (so `from ...scaled_mm import ScaledMMLinearKernel`
    # gives the class, not the module)
    pkg = sys.modules["vllm.model_executor.kernels.linear.scaled_mm"]
    setitem(pkg.__dict__, "ScaledMMLinearKernel", smk.ScaledMMLinearKernel)
    setitem(pkg.__dict__, "FP8ScaledMMLinearKernel", smk.FP8ScaledMMLinearKernel)
    setitem(pkg.__dict__, "CutlassFP8ScaledMMLinearKernel", smk.CutlassFP8ScaledMMLinearKernel)
    sys.modules["vllm.model_executor.layers.fused_moe"].utils = utils
    return smk, utils, tm


class RoutedExperts(torch.nn.Module):
    """A fused MoE layer with FP8 experts and dynamic per-tensor input scales (vLLM's fp8_per_tensor): its experts
    quantize their input, then the activation's output, through the triton_moe module's moe_kernel_quantize_input."""

    def __init__(self, tm, hidden=64, inter=32, experts=4, top_k=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.tm = tm
        self.w13_weight = torch.nn.Parameter((torch.randn(experts, 2 * inter, hidden, generator=g) * 0.05)
                                             .to(torch.float8_e4m3fn), requires_grad=False)
        self.w2_weight = torch.nn.Parameter((torch.randn(experts, hidden, inter, generator=g) * 0.05)
                                            .to(torch.float8_e4m3fn), requires_grad=False)
        self.top_k = top_k
        # vLLM's FusedMoEQuantConfig sits on the method as moe_quant_config (fused_moe_method_base.py)
        self.quant_method = types.SimpleNamespace(moe_quant_config=types.SimpleNamespace(
            use_fp8_w8a8=True, block_shape=None, per_act_token_quant=False, a1_scale=None))

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        xq, s1 = self.tm.moe_kernel_quantize_input(x, None, torch.float8_e4m3fn, False, None)
        h = torch.zeros(x.shape[0], self.top_k, self.w2_weight.shape[2])
        for k in range(self.top_k):
            e = topk_ids[:, k].long()
            a = torch.bmm(self.w13_weight.float()[e], (xq.float() * s1)[:, :, None]).squeeze(-1)
            half = a.shape[1] // 2
            h[:, k] = torch.nn.functional.silu(a[:, :half]) * a[:, half:]
        h2 = h.reshape(-1, h.shape[-1]).to(torch.bfloat16)
        hq, s2 = self.tm.moe_kernel_quantize_input(h2, None, torch.float8_e4m3fn, False, None)
        out = torch.zeros(x.shape[0], self.w2_weight.shape[1])
        hd = (hq.float() * s2).reshape(x.shape[0], self.top_k, -1)
        for k in range(self.top_k):
            e = topk_ids[:, k].long()
            out += topk_weights[:, k:k + 1].float() * torch.bmm(self.w2_weight.float()[e],
                                                                 hd[:, k, :, None]).squeeze(-1)
        return out.to(x.dtype)

