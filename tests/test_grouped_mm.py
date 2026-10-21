"""MoE grouped GEMMs (torch._grouped_mm): checked per group in the default mode, a wrong expert output is caught,
and FP8/fused-activation GEMMs the guard can't check are counted instead of looking covered."""
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sdcguard import SDCGuard

def _runs_on_cpu() -> bool:
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:                                   # PyTorch 2.8 has the op, but no CPU kernel for it
        torch._grouped_mm(torch.zeros(4, 16, dtype=torch.bfloat16), torch.zeros(2, 16, 16, dtype=torch.bfloat16),
                          offs=torch.tensor([2, 4], dtype=torch.int32))
        return True
    except (NotImplementedError, RuntimeError):
        return False


if not _runs_on_cpu():
    pytest.skip("this PyTorch can't run torch._grouped_mm on the CPU", allow_module_level=True)


def _operands():
    torch.manual_seed(0)
    a = torch.randn(96, 64, dtype=torch.bfloat16)
    b = torch.randn(4, 64, 128, dtype=torch.bfloat16)
    offs = torch.tensor([10, 40, 40, 96], dtype=torch.int32)      # one expert gets no tokens
    return a, b, offs


def test_every_expert_is_checked_and_clean_runs_dont_alarm():
    a, b, offs = _operands()
    g = SDCGuard()
    with g.at_step(0):
        torch._grouped_mm(a, b, offs=offs)
    assert g.stats["checked"] == 3 and g.stats["alerts"] == 0          # the empty group isn't a check


@pytest.mark.parametrize("layout", ["2d_3d", "3d_2d", "3d_3d"])
def test_a_wrong_value_in_one_expert_is_caught(layout):
    torch.manual_seed(0)
    if layout == "2d_3d":
        a, b, offs = _operands()
        call = lambda: torch._grouped_mm(a, b, offs=offs)           # noqa: E731
    elif layout == "3d_2d":
        a = torch.randn(3, 32, 64, dtype=torch.bfloat16)
        b = torch.randn(64, 96, dtype=torch.bfloat16)
        offs = torch.tensor([32, 64, 96], dtype=torch.int32)
        call = lambda: torch._grouped_mm(a, b, offs=offs)           # noqa: E731
    else:
        a = torch.randn(3, 32, 64, dtype=torch.bfloat16)
        b = torch.randn(3, 64, 96, dtype=torch.bfloat16)
        call = lambda: torch._grouped_mm(a, b)                      # noqa: E731
    from torch.utils._python_dispatch import TorchDispatchMode

    class Faulty(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if "grouped_mm" in str(func):
                out.view(-1)[out.numel() // 2] += 16.0
            return out
    g = SDCGuard()
    with Faulty():
        with g.at_step(0):
            call()
    assert g.stats["alerts"] == 1, g.stats


def test_expert_gradients_are_checked_too():
    a, b, offs = _operands()
    a.requires_grad_(True)
    b.requires_grad_(True)
    g = SDCGuard()
    with g.at_step(0):
        torch._grouped_mm(a, b, offs=offs).float().pow(2).sum().backward()
    assert g.stats["checked"] > 3 and g.stats["alerts"] == 0, (g.stats, g.ops_seen)


def test_unchecked_gemms_are_counted():
    torch.manual_seed(0)
    a = torch.randn(32, 64)
    b = torch.randn(64, 48)
    bias = torch.randn(48)
    g = SDCGuard()
    with g.at_step(0):
        torch.ops.aten._addmm_activation(bias, a, b, use_gelu=True)
    assert g.stats["matmuls_unchecked"] == 1 and g.unchecked_ops == {"aten::_addmm_activation": 1}
    m = torch.nn.Linear(64, 64).bfloat16()
    g2 = SDCGuard().protect(m)
    a2, b2, offs = _operands()
    cov = g2.coverage(lambda: (m(torch.randn(8, 64, dtype=torch.bfloat16)), torch._grouped_mm(a2, b2, offs=offs)))
    assert cov["unchecked_multiply_adds_by_op"]["aten::_grouped_mm"] == 96 * 64 * 128
    assert cov["share_eligible_in_fast_mode"] < 0.1                 # the experts hold most of the work


def test_a_bias_is_shared_by_every_group():
    """PyTorch 2.14's grouped_mm docs: the bias is "not jagged and must be broadcastable to the result shape of
    each group". The adapter used to pass bias[i] (one number) for group i, so a right output could fail."""
    from sdcguard.checksum import check_matmul
    from sdcguard.guard import _grouped_checks
    a = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16)
    b = torch.stack([torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bfloat16), torch.ones(2, 3, dtype=torch.bfloat16)])
    bias = torch.tensor([1, 30, -20], dtype=torch.bfloat16)
    out = torch.cat([a[:2] @ b[0] + bias, a[2:] @ b[1] + bias])
    checks = _grouped_checks((a, b, torch.tensor([2, 4], dtype=torch.int32), bias), {}, out)
    assert len(checks) == 2
    for aa, bb, cc, bi, alpha, beta in checks:
        assert bi is bias and check_matmul(aa, bb, cc, bias=bi, alpha=alpha, beta=beta).ok
    # a bias that can't broadcast to a group's output isn't guessed at: that group isn't checked
    assert _grouped_checks((a, b, torch.tensor([2, 4], dtype=torch.int32), torch.zeros(4, 3)), {}, out) == []


class _CorruptEmptyExpertGrad(TorchDispatchMode):
    """Faulty hardware: writes 1000 into the weight gradient of expert 1, which got no tokens."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if "grouped_mm" in str(func) and out.dim() == 3 and args[0].dim() == 2 and args[1].dim() == 2:
            out[1, 3, 7] = 1000.0
        return out


def test_the_weight_gradient_of_an_expert_with_no_tokens_is_checked():
    """In the 2-D x 2-D layout (expert weight gradients, grouped along k) an expert that got no tokens still has
    an (m, n) output, and it must be all zeros. It is checked as a product with k = 0, so a wrong value written
    there is caught (it was skipped like an empty group of tokens, and the wrong gradient passed)."""
    torch.manual_seed(0)
    a = torch.randn(96, 64, dtype=torch.bfloat16, requires_grad=True)          # tokens x hidden
    b = torch.randn(3, 64, 128, dtype=torch.bfloat16, requires_grad=True)      # experts
    offs = torch.tensor([40, 40, 96], dtype=torch.int32)                       # expert 1: no tokens
    g = SDCGuard()
    with g.at_step(0):
        torch._grouped_mm(a, b, offs=offs).float().pow(2).sum().backward()
    assert g.stats["alerts"] == 0 and torch.count_nonzero(b.grad[1]) == 0     # a clean zero gradient passes
    checked = g.stats["checked"]
    a.grad = b.grad = None
    g2 = SDCGuard()
    with _CorruptEmptyExpertGrad():
        with g2.at_step(0):
            torch._grouped_mm(a, b, offs=offs).float().pow(2).sum().backward()
    assert b.grad[1, 3, 7] == 1000.0                                           # the fault is in the gradient
    assert g2.stats["checked"] == checked and g2.stats["alerts"] == 1, g2.stats
    assert g2.log.events[0].detail["shape_B"][0] == 0                          # the k = 0 product


def test_rows_past_the_last_group_are_never_checked():
    """offs[-1] may stop short of the rows of a 2-D x 3-D grouped matmul: torch._grouped_mm leaves the output rows
    past it unwritten (whatever memory held), so checking them would raise false alarms."""
    a, b, _ = _operands()
    offs = torch.tensor([10, 40, 40, 80], dtype=torch.int32)                  # rows 80..95 belong to no group

    class Garbage(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if "grouped_mm" in str(func):
                out[80:] = float("nan")
            return out
    g = SDCGuard()
    with Garbage():
        with g.at_step(0):
            torch._grouped_mm(a, b, offs=offs)
    assert g.stats["checked"] == 3 and g.stats["alerts"] == 0 and g.stats["unjudged"] == 0, g.stats
