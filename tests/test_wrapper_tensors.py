"""Tensors a GPU kernel can't be handed. Tensor parallel gives the check AsyncCollectiveTensors (DTensor's to_local()
and its sharding rules do, while a collective is still running): wrappers with no memory of their own, whose
data_ptr() is 0. Handed to a Triton kernel, one is read at address 0, an illegal memory access that ended the
two-H100 tensor-parallel run on October 21, 2026 (Xid 31 at 0x0 and 0x2000 on both GPUs). The check now waits for
such a tensor first (_group.ready), and every kernel entry point refuses any tensor without memory of its own
(_group.own_memory), whose check then runs in PyTorch through the subclass's own dispatch."""
import pytest
import torch
import torch.utils._pytree as pytree

from sdcguard import _fingerprint, _group
from sdcguard.recompute import _fingerprint_torch, fingerprint

ACT = pytest.importorskip("torch.distributed._functional_collectives").AsyncCollectiveTensor


class _Wrapper(torch.Tensor):
    """Another wrapper subclass (as quantized and other tensor subclasses are): no memory of its own."""

    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=elem.dtype, device=elem.device)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            return t.elem if isinstance(t, _Wrapper) else t
        return func(*pytree.tree_map(unwrap, args), **pytree.tree_map(unwrap, kwargs or {}))


class _Rewrap(_Wrapper):
    """A wrapper that wraps what it computes again, as DTensor does: a view of it is a wrapper too."""

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return pytree.tree_map(lambda t: _Rewrap(t) if isinstance(t, torch.Tensor) else t, out)


def _addressed(*ts) -> bool:
    """What a kernel needs, checked here directly rather than by the code under test: a real address."""
    return all(t is None or t.numel() == 0 or t.data_ptr() != 0 for t in ts)


def test_wrappers_have_no_address_and_are_refused():
    e = torch.randn(8, 16, dtype=torch.bfloat16)
    assert ACT(e).data_ptr() == 0 and _Wrapper(e).data_ptr() == 0         # what a kernel would have been handed
    assert _group.own_memory(e, torch.nn.Parameter(e.clone()), None, torch.empty(0))
    assert not _group.own_memory(ACT(e)) and not _group.own_memory(e, _Wrapper(e))


def test_an_async_collective_tensor_is_waited_for_into_the_tensor_itself():
    e = torch.randn(8, 16, dtype=torch.bfloat16)
    r = _group.ready(ACT(e))
    assert type(r) is torch.Tensor and r.data_ptr() == e.data_ptr() and torch.equal(r, e)
    assert _group.ready(e) is e and _group.ready(None) is None
    w = _Wrapper(e)
    assert _group.ready(w) is w                  # other subclasses are left alone (and refused by own_memory)


def test_the_grouped_check_refuses_operands_without_memory():
    A, B = torch.randn(32, 64, dtype=torch.bfloat16), torch.randn(64, 128, dtype=torch.bfloat16)
    C, bias = A @ B, torch.randn(128, dtype=torch.bfloat16)
    assert _group.supports(A, B, C, bias, 1.0)
    for args in ((ACT(A), B, C, bias), (A, ACT(B), C, bias), (A, B, _Wrapper(C), bias), (A, B, C, ACT(bias))):
        assert not _group.supports(*args, 1.0)
    assert _group.supports(A, B, C, ACT(bias), 0.0)              # a bias that isn't added isn't handed over


def test_the_check_waits_for_a_local_shard_before_its_gpu_kernel(monkeypatch):
    """What reaches the grouped GPU check's gate (with the GPU check made available) has memory of its own, when
    the matmul's operands come in as AsyncCollectiveTensors; the check itself runs and is clean."""
    from sdcguard import SDCGuard
    from sdcguard import checksum as _cs
    seen = []

    def spy(A, B, C, bias, beta, block=128):
        seen.append(_addressed(A, B, C, bias))
        return False                              # then the plain check runs, as it does on a CPU
    monkeypatch.setattr(_cs, "USE_FAST", True)
    monkeypatch.setattr(_group, "available", lambda C: True)
    monkeypatch.setattr(_group, "supports", spy)
    A, B = torch.randn(32, 64, dtype=torch.bfloat16), torch.randn(64, 128, dtype=torch.bfloat16)
    g = SDCGuard()
    g._queue("aten::mm", "layer", None, ACT(A), ACT(B), ACT(A @ B), None, 1.0, 0.0)
    g.flush()
    assert seen == [True] and g.stats["checked"] == 1 and g.stats["alerts"] == 0


def test_the_fused_kernel_is_never_handed_a_wrapper(monkeypatch):
    from sdcguard import SDCGuard
    from sdcguard import fused as _fused
    handed = []

    def kernel(A, B, bias, **_):
        handed.append(_addressed(A, B, bias))
        raise RuntimeError("stop here")           # (the guard then turns the fused check off: enough for this test)
    monkeypatch.setattr(_fused, "available", lambda x: True)
    monkeypatch.setattr(_fused, "fused_mm_checked", kernel)
    A, B = torch.randn(32, 64, dtype=torch.bfloat16), torch.randn(64, 128, dtype=torch.bfloat16)
    g = SDCGuard(precise="fused")
    monkeypatch.setattr(g, "_active", True, raising=False)
    assert g._fused_mm("layer", "forward", _Wrapper(A), B, None) is None and handed == []   # never handed over
    with pytest.warns(RuntimeWarning, match="fused precise check failed"):
        assert g._fused_mm("layer", "forward", ACT(A), ACT(B), None) is None
    assert handed == [True]                       # waited for: the kernel got the tensors themselves


def test_the_fingerprint_kernel_is_never_handed_a_wrapper(monkeypatch):
    """A view of an AsyncCollectiveTensor as int16 words is still one (address 0): the fingerprint waits first."""
    e = torch.randn(4097, dtype=torch.bfloat16)
    handed = []

    def kernel(bits):
        handed.append(_addressed(bits))
        return _fingerprint_torch(bits)
    monkeypatch.setattr(_fingerprint, "usable", lambda device, reference: True)
    monkeypatch.setattr(_fingerprint, "fingerprint_bits", kernel)
    assert type(ACT(e).contiguous().view(-1).view(torch.int16)).__name__ == "AsyncCollectiveTensor"
    assert torch.equal(fingerprint(ACT(e)), fingerprint(e)) and handed == [True, True]
    assert torch.equal(fingerprint(_Wrapper(e)), fingerprint(e)) and handed and all(handed)


def test_the_fingerprint_kernels_status_names_cuda_and_cuda0_as_one_device(monkeypatch):
    """The GPU run's comparison asked for the kernel's status under "cuda" and found none: the kernel had recorded
    it under "cuda:0", where fingerprint() ran (October 21, 2026). Both names are one device, as for the other
    kernels."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setitem(_fingerprint.STATUS, "cuda:0", (True, ""))
    assert _fingerprint.status("cuda") == (True, "") and _fingerprint.status(torch.device("cuda", 0)) == (True, "")


def test_the_fingerprint_of_a_tensor_that_stays_a_wrapper_takes_the_pytorch_sums(monkeypatch):
    """A wrapper whose views are wrappers too reaches the int16 words still a wrapper (address 0): the kernel is
    never handed it, and the PyTorch sums give the plain tensor's fingerprint."""
    e = torch.randn(4097, dtype=torch.bfloat16)
    handed = []
    monkeypatch.setattr(_fingerprint, "usable", lambda device, reference: True)
    monkeypatch.setattr(_fingerprint, "fingerprint_bits", lambda bits: handed.append(bits) or _fingerprint_torch(bits))
    w = _Rewrap(e)
    assert isinstance(w.detach().contiguous().view(-1).view(torch.int16), _Rewrap)
    assert torch.equal(fingerprint(w), fingerprint(e)) and len(handed) == 1 and _addressed(handed[0])

