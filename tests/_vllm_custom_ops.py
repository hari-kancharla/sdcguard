"""vLLM 0.31.0's vllm._custom_ops.machete_mm, verbatim (vLLM, Apache-2.0, https://github.com/vllm-project/vllm; the
same in 0.32.0 and on main, October 23, 2026). The tests bind this module's `torch` to a stand-in whose
torch.ops._C.machete_mm is a test op, so the call the plugin copies runs as vLLM makes it."""
import torch

ScalarType = object            # vllm.scalar_type.ScalarType (only an annotation here)


def machete_mm(
    a: torch.Tensor,
    # b_q Should be the tensor returned by machete_prepack_B
    b_q: torch.Tensor,
    b_type: ScalarType,
    out_type: torch.dtype | None = None,
    b_group_scales: torch.Tensor | None = None,
    b_group_zeros: torch.Tensor | None = None,
    b_group_size: int | None = None,
    b_channel_scales: torch.Tensor | None = None,
    a_token_scales: torch.Tensor | None = None,
    schedule: str | None = None,
) -> torch.Tensor:
    return torch.ops._C.machete_mm(
        a,
        b_q,
        b_type.id,
        out_type,
        b_group_scales,
        b_group_zeros,
        b_group_size,
        b_channel_scales,
        a_token_scales,
        schedule,
    )
