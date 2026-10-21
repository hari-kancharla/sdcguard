"""Activation checkpointing (torch.utils.checkpoint) with checked layers. Its recompute runs the forward again
inside the backward pass, possibly after the checked step ended (or with a forward from before it): a protected
layer's recompute makes the choice its forward made, so a non-reentrant checkpoint sees the same saved tensors
(anything else is a CheckpointError) and a backward after the step is counted, not run unchecked in silence. In
the default mode a recomputed forward is reported as forward work, with its layer's name."""
import gc
import warnings

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.checkpoint import checkpoint

from sdcguard import SDCGuard


def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 16)).bfloat16()


def _x():
    torch.manual_seed(1)
    return torch.randn(4, 8, 16, dtype=torch.bfloat16, requires_grad=True)


def _reference_grads(reentrant):
    m, x = _mlp(), _x()
    checkpoint(m, x, use_reentrant=reentrant).float().pow(2).sum().backward()
    return [p.grad for p in m.parameters()] + [x.grad]


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("case", ["backward_after_step", "forward_before_step", "both_inside", "both_outside"])
def test_checkpointed_layers_across_the_step_boundary(case, reentrant):
    """Whichever side of the step boundary the forward and the backward fall on, training computes the same
    gradients as without the guard. A backward after the step ended is counted in stats["backward_outside_step"]
    (as it is without checkpointing); a forward and backward both inside are checked twice in the forward (the
    forward and its recompute) and once in the backward, as before."""
    m, x = _mlp(), _x()
    g = SDCGuard().protect(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case == "backward_after_step":
            with g.at_step(0):
                loss = checkpoint(m, x, use_reentrant=reentrant).float().pow(2).sum()
            loss.backward()
        elif case == "forward_before_step":
            loss = checkpoint(m, x, use_reentrant=reentrant).float().pow(2).sum()
            with g.at_step(0):
                loss.backward()
        elif case == "both_inside":
            with g.at_step(0):
                checkpoint(m, x, use_reentrant=reentrant).float().pow(2).sum().backward()
        else:
            checkpoint(m, x, use_reentrant=reentrant).float().pow(2).sum().backward()
    assert all(torch.equal(a, b) for a, b in zip([p.grad for p in m.parameters()] + [x.grad],
                                                 _reference_grads(reentrant)))
    st = g.stats
    assert st["alerts"] == 0 and not g._pending
    if case == "backward_after_step":
        assert st["checked"] == 2 and st.get("backward_outside_step") == 2, st      # the two forward matmuls
    elif case == "both_inside":
        assert st["checked"] == 8 and not st.get("backward_outside_step"), st
    elif case == "forward_before_step":
        # a reentrant recompute builds its own graph, so inside a checked step it's checked (forward and
        # backward); a non-reentrant one must save what its unchecked forward saved
        assert st["checked"] == (6 if reentrant else 0), st
    else:
        assert st["checked"] == 0 and not st.get("backward_outside_step"), st


@pytest.mark.parametrize("reentrant", [False, True])
def test_remembered_checkpoint_calls_go_with_their_graph(reentrant):
    """Each checkpointed call whose forward ran checked is remembered for its recompute, through a weak
    reference: once its graph is freed nothing of it is left, and unchecked steps are back on the plain path."""
    m = _mlp()
    g = SDCGuard().protect(m)
    for step in range(3):
        with g.at_step(step):
            loss = checkpoint(m, _x(), use_reentrant=reentrant).float().pow(2).sum()
            loss.backward()
    del loss
    gc.collect()
    assert g._checked_calls == {}
    assert g.stats["checked"] == 24 and g.stats["alerts"] == 0


class _RecomputeFault(TorchDispatchMode):
    """Faulty hardware in the recompute only: the addmm that runs inside a backward node is off by 50."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if "addmm" in str(func) and torch._C._current_autograd_node() is not None:
            out.view(-1)[5] += 50.0
        return out


@pytest.mark.parametrize("watched", [True, False])
def test_a_recomputed_forward_is_named_as_forward_work(watched):
    """Default mode, non-reentrant checkpointing (PyTorch's recommended kind): the recompute runs inside a
    backward node, but its matmuls are the forward's, so an alert says pass "forward" and names the layer
    (watch()), or "forward" without it, instead of the backward node ("AddmmBackward0")."""
    m = _mlp()
    g = SDCGuard()
    if watched:
        g.watch(m)
    x = _x()
    with _RecomputeFault():
        with g.at_step(0):
            checkpoint(m, x, use_reentrant=False).float().sum().backward()
    assert g.log.events
    for e in g.log.events:
        assert e.detail["pass"] == "forward", e.detail
        assert e.detail["where"] in (("0", "2") if watched else ("forward",)), e.detail


class _CheckpointedBlock(torch.nn.Module):
    def __init__(self, reentrant):
        super().__init__()
        self.inner, self.reentrant = _mlp(), reentrant

    def forward(self, x):
        return checkpoint(self.inner, x, use_reentrant=self.reentrant)


@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("case", ["backward_after_step", "forward_before_step"])
def test_a_checkpoint_inside_a_compiled_graph_across_the_step_boundary(monkeypatch, case, reentrant):
    """The same under torch.compile(fullgraph=True), with the checkpoint inside the compiled region (as when each
    block is wrapped, then compiled): the compiler recomputes it in its own backward graph. Gradients match
    eager, and a backward after the step is counted. The guard's first checked step in a fresh process is
    one of them (its DTensor lookup must not happen inside the compiled checkpoint, which refuses side
    effects)."""
    from sdcguard import guard as guard_mod
    monkeypatch.setattr(guard_mod, "_DTENSOR", {})               # as in a fresh process
    torch._dynamo.reset()
    m, x = _CheckpointedBlock(reentrant), _x()
    g = SDCGuard().protect(m)
    f = torch.compile(m, backend="aot_eager", fullgraph=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case == "backward_after_step":
            with g.at_step(0):
                loss = f(x).float().pow(2).sum()
            loss.backward()
        else:
            loss = f(x).float().pow(2).sum()
            with g.at_step(0):
                loss.backward()
    assert all(torch.equal(a, b) for a, b in zip([p.grad for p in m.inner.parameters()] + [x.grad],
                                                 _reference_grads(reentrant)))
    st = g.stats
    assert st["alerts"] == 0 and not g._pending
    if case == "backward_after_step":
        assert st["checked"] == 2 and st.get("backward_outside_step") == 2, st
    else:
        assert st["checked"] == 0, st
