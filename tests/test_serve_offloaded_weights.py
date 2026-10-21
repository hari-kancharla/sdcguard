"""Layers whose weights the serving guard can't read back as their own. vLLM's CPU offloaders re-point a layer's
weight after it was loaded (and registered): the prefetch offloader at a static GPU buffer that layers N, N + its
capacity, ... share and that holds each layer's weights only while it runs (vllm/model_executor/offloader/
prefetch.py), the UVA one at a view of CPU memory. The checks still hold (each matmul runs with the weights it was
given), but what the guard reads there later may belong to another layer. So an alert must not say whether the
weights changed, verify_weights() must not compare them, and refresh() must not compute checksums from them: it
turns the layer's check off and lists it in unsupported instead."""
import torch
import torch.nn as nn

from sdcguard.serve import ServeGuard


def test_layers_whose_weights_share_one_buffer_are_not_read_back_as_their_own():
    """Two registered layers whose weights are one static buffer, each registered while the buffer held its own
    weights. A weight two layers hold (tied: one tensor) is still compared."""
    torch.manual_seed(0)
    a, b = nn.Linear(64, 128, bias=False).bfloat16(), nn.Linear(64, 128, bias=False).bfloat16()
    tied = nn.Linear(64, 128, bias=False).bfloat16()
    tied2 = nn.Linear(64, 128, bias=False).bfloat16()
    tied2.weight = tied.weight
    wa, wb = a.weight.detach().clone(), b.weight.detach().clone()
    buf = torch.empty_like(wa)
    a.weight.data, b.weight.data = buf, buf              # the offloader's static buffer
    g = ServeGuard(exact_reductions=False)
    x = torch.randn(4, 64).bfloat16()
    with torch.no_grad():
        buf.copy_(wa)
        g.add_layer(a, "a")
        g.check(0, x, a(x))
        buf.copy_(wb)
        g.add_layer(b, "b")
        g.check(1, x, b(x))
    g.add_layer(tied, "tied")
    g.add_layer(tied2, "tied2")
    assert g.poll(wait=True) == []                       # each layer ran with its own weights: clean
    assert g.verify_weights() == []                      # the buffer holds b's weights; a's aren't "changed"
    with torch.no_grad():
        g.check(0, x, a(x))                              # a fault: a ran with b's weights
    ev = g.poll(wait=True)
    assert [e.detail["where"] for e in ev] == ["a"]
    assert ev[0].detail["weights_changed_since_load"] is None
    assert "share memory with another checked layer's weights" in ev[0].detail["reading"]
    with torch.no_grad():
        tied.weight.view(torch.int16)[0, 0] ^= 0x4000
    assert g.verify_weights() == ["tied", "tied2"]       # tied weights are still compared
    with torch.no_grad():
        tied.weight.view(torch.int16)[0, 0] ^= 0x4000
    assert g.refresh() == 2                              # tied and tied2; a and b can't be
    assert set(g.unsupported) == {"a", "b"} and "share memory" in g.unsupported["a"]
    assert not g.refresh_layer(0, weight=b.weight)       # passed in, but it's another layer's memory too


def test_a_weight_moved_to_other_memory_after_registration_is_not_read_back_as_its_own():
    """vLLM registers a layer while its weight is a temporary GPU copy (model_loader.utils.device_loading_context)
    and the offloader points it at its buffer afterwards. The layer sharing that buffer may be one the guard doesn't
    check (SDCGUARD_SHARE < 1), so sharing alone doesn't show it: a weight that now lives in other memory than at
    registration, with the same shape and dtype, is treated the same way. refresh_layer() with the weight passed in
    (as vLLM's weight reload calls it) still takes it."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 128, bias=False).bfloat16()
    g = ServeGuard(exact_reductions=False).attach(nn.ModuleDict({"l": lin}))
    try:
        x = torch.randn(4, 64).bfloat16()
        with torch.no_grad():
            lin.weight.data = lin.weight.data.clone()    # the offloader's buffer, holding this layer's weights now
            lin(x)
        assert g.poll(wait=True) == [] and g.verify_weights() == []
        with torch.no_grad():
            lin.weight[0, 0] += 1.0                      # the buffer holds other weights when the guard reads it
            lin(x)
        ev = g.poll(wait=True)
        assert len(ev) == 1 and ev[0].detail["weights_changed_since_load"] is None
        assert "moved to other memory" in ev[0].detail["reading"]
        assert g.verify_weights() == []                  # not compared
        assert g.refresh() == 0 and "moved to other memory" in g.unsupported["l"]
        with torch.no_grad():
            lin(x)                                       # its check is off
        assert g.poll(wait=True) == [] and g.counts()["l"]["checks"] == 2
        assert g.refresh_layer(0, weight=lin.weight) and g.unsupported == {}
        with torch.no_grad():
            lin(x)
        assert g.poll(wait=True) == [] and g.counts()["l"]["checks"] == 3
    finally:
        g.detach()
