"""vLLM 0.31.0's MacheteLinearKernel code that sdcguard's plugin copies, verbatim from
vllm/model_executor/kernels/linear/mixed_precision/machete.py and MPLinearKernel.py (vLLM, Apache-2.0,
https://github.com/vllm-project/vllm; the same in 0.32.0 and on main, October 23, 2026). The stand-in Machete kernel
in test_quant.py runs this code, with `ops` bound to the stand-in vllm._custom_ops, so the plugin is tested against
the code it was written for."""
import torch

ops = None                     # vllm._custom_ops; the tests bind it to _vllm_custom_ops


def apply_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    c = self.config
    w_q, w_s, w_zp = self._get_weight_params(layer)

    x_2d = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

    if c.zero_points:
        assert w_zp is not None
    else:
        w_zp = None

    output = ops.machete_mm(
        a=x_2d,
        b_q=w_q,
        b_type=c.weight_type,
        b_group_zeros=w_zp,
        b_group_scales=w_s,
        b_group_size=c.group_size,
    )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


def _get_weight_params(
    self, layer: torch.nn.Module
) -> tuple[
    torch.Tensor,  # w_q
    torch.Tensor,  # w_s
    torch.Tensor | None,  # w_zp
]:
    return (
        getattr(layer, self.w_q_name),
        getattr(layer, self.w_s_name),
        getattr(layer, self.w_zp_name or "", None),
    )
