"""protect() under torch.compile, fullgraph=True included (_compiled.py): checked steps run every check through
one custom operator, unchecked steps run the plain compiled graph, gradients match eager, and a wrong output
inside a compiled checked step is caught."""
import contextlib
import warnings

import pytest
import torch

from sdcguard import SDCGuard


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 64)).bfloat16()


def _x():
    torch.manual_seed(1)
    return torch.randn(16, 64, dtype=torch.bfloat16)


def _grads(m):
    return [p.grad.clone() for p in m.parameters()]


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_fullgraph_compile_checks_on_checked_steps_only(backend):
    torch._dynamo.reset()
    ref = _model()
    ref(_x()).float().sum().backward()
    m = _model()
    g = SDCGuard(every=2).protect(m)
    f = torch.compile(m, backend=backend, fullgraph=True)
    counts = []
    try:
        for step in range(4):
            m.zero_grad()
            with g.on_step(step):
                f(_x()).float().sum().backward()
            counts.append(g.stats["checked"])
    except Exception as e:  # noqa: BLE001
        if backend == "inductor" and "compiler" in str(e).lower():
            pytest.skip(f"no working C++ compiler for Inductor here: {e}")
        raise
    assert counts == [5, 5, 10, 10]                  # 2 forward, 2 dW, 1 dX per checked step; none otherwise
    assert g.stats["alerts"] == 0
    if backend == "aot_eager":
        assert all(torch.equal(a, b) for a, b in zip(_grads(m), _grads(ref)))
    else:
        assert all(torch.allclose(a.float(), b.float(), rtol=0.05, atol=0.05) for a, b in zip(_grads(m), _grads(ref)))


def test_a_wrong_output_inside_a_compiled_step_is_caught(monkeypatch):
    torch._dynamo.reset()
    m = _model()
    g = SDCGuard(every=1).protect(m)
    f = torch.compile(m, backend="aot_eager", fullgraph=True)
    with g.on_step(0):
        f(_x()).float().sum().backward()             # compiled clean first
    assert g.stats["alerts"] == 0
    real = torch.nn.functional.linear

    def faulty(x, w, b=None):                        # the GPU computes one output wrong
        y = real(x, w, b)
        y.view(-1)[5] += 8.0
        return y
    monkeypatch.setattr(torch.nn.functional, "linear", faulty)
    with g.on_step(1):
        f(_x()).float().sum().backward()
    assert g.stats["alerts"] >= 1 and g.log.events[0].step == 1


def test_a_backward_after_the_step_is_counted_not_checked():
    m = _model()
    g = SDCGuard(every=1).protect(m)
    with g.on_step(0):
        loss = m(_x()).float().sum()
    checked = g.stats["checked"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loss.backward()
    assert g.stats["checked"] == checked and g.stats["backward_outside_step"] == 2
    assert any("backward ran after" in str(x.message) for x in w)
    ref = _model()
    ref(_x()).float().sum().backward()
    assert all(torch.equal(a, b) for a, b in zip(_grads(m), _grads(ref)))


def test_a_checked_step_with_no_checks_says_so():
    m = _model()
    g = SDCGuard(every=1).protect(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with g.on_step(0):
            pass                                     # the forward ran elsewhere
    assert g.stats["checked_steps_without_checks"] == 1
    assert any("ran no checks" in str(x.message) for x in w)


def test_tensor_parallel_layers_inside_a_compiled_graph_are_counted_not_passed(monkeypatch):
    """Tensor parallel inside torch.compile isn't checked yet. Stood in for here by making every weight look like
    a DTensor to the compiled path: each checked step counts the layers it skipped and warns once, and the graph
    isn't recompiled every step."""
    from sdcguard import guard as guard_mod
    torch._dynamo.reset()
    m = _model()
    g = SDCGuard(every=1).protect(m)
    monkeypatch.setattr(guard_mod, "_is_dtensor", lambda t: isinstance(t, torch.nn.Parameter))
    from torch._dynamo.utils import counters
    counters.clear()
    f = torch.compile(m, backend="aot_eager", fullgraph=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for step in range(3):
            with g.on_step(step):
                f(_x()).float().sum().backward()
    assert g.stats["skipped_dtensor_compiled"] == 6 and g.stats["checked"] == 0      # 2 layers x 3 steps
    assert sum("ran inside torch.compile and weren't checked" in str(x.message) for x in w) == 1
    assert counters["stats"]["unique_graphs"] <= 2                                 # compiled once, not per step


def test_the_custom_operators_pass_pytorchs_opcheck():
    """PyTorch's own test of a custom operator's schema, fake (meta) implementation, autograd registration and
    AOT dispatch, outside and inside a checked step (where the operator also queues its checks)."""
    from sdcguard import _compiled  # noqa: F401  (registers the operators)
    torch.manual_seed(0)
    x = torch.randn(4, 8, 16, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(32, 16, dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(32, dtype=torch.bfloat16, requires_grad=True)
    gy = torch.randn(4, 8, 32, dtype=torch.bfloat16)
    g = SDCGuard()
    for active in (False, True):
        ctx = g._fast_step() if active else contextlib.nullcontext()
        with ctx:
            for bias in (b, None):
                r = torch.library.opcheck(torch.ops.sdcguard.checked_linear.default, (x, w, bias, g._op_id, "lin"))
                assert set(r.values()) == {"SUCCESS"}, r
            for need in ((True, True, True), (False, True, False)):
                r = torch.library.opcheck(torch.ops.sdcguard.checked_linear_backward.default,
                                          (gy, x.detach(), w.detach(), g._op_id, "lin", *need))
                assert set(r.values()) == {"SUCCESS"}, r
    assert g.stats["checked"] > 0 and g.stats["alerts"] == 0


def test_the_checked_operators_stay_outside_cuda_graphs():
    """Under mode="reduce-overhead" a custom operator inside a captured CUDA graph never runs its Python on replay,
    so its checks would silently stop: both operators are tagged cudagraph_unsafe, which makes Inductor run them
    outside the graph (its graph partitioning), every step."""
    import pytest
    from sdcguard import _compiled  # noqa: F401  (registers the operators)
    tag = getattr(torch.Tag, "cudagraph_unsafe", None)
    if tag is None:
        pytest.skip("this PyTorch has no cudagraph_unsafe tag")
    assert tag in torch.ops.sdcguard.checked_linear.default.tags
    assert tag in torch.ops.sdcguard.checked_linear_backward.default.tags
