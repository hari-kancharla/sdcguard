"""Fast mode (SDCGuard.protect): only the linear layers' matmuls go through Python."""
import copy

import pytest
import torch
import torch.nn as nn

from sdcguard import FaultInjector, SDCGuard
from sdcguard.inject import fault_model


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(256, 512)
        self.b = nn.Linear(512, 300, bias=False)      # a remainder block (300 = 2 x 128 + 44)
        self.ln = nn.LayerNorm(300)

    def forward(self, x):
        return self.ln(self.b(torch.relu(self.a(x))))


def _run(model, guard, x, inj=None, steps=3, autocast=True):
    grads, losses = [], []
    for s in range(steps):
        if inj is not None:
            inj.step = s
        with (inj if inj is not None else torch.enable_grad()):
            with guard.at_step(s):
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                    loss = model(x).float().pow(2).mean()
                loss.backward()
        grads.append({k: p.grad.clone() for k, p in model.named_parameters()})
        losses.append(loss.item())
        model.zero_grad(set_to_none=True)
    return grads, losses


def test_same_checks_and_bit_identical_gradients():
    torch.manual_seed(0)
    base, x = Net(), torch.randn(3, 20, 256)
    slow = SDCGuard()
    gd, ld = _run(copy.deepcopy(base), slow, x)
    m = copy.deepcopy(base)
    fast = SDCGuard().protect(m)
    gf, lf = _run(m, fast, x)
    assert ld == lf
    for a, b in zip(gd, gf):
        assert all(torch.equal(a[k], b[k]) for k in a)
    # per step: a (forward, dW; its input needs no gradient) + b (forward, dX, dW)
    assert fast.stats["checked"] == 3 * 5 == slow.stats["checked"]
    assert fast.stats["alerts"] == 0


def test_unchecked_steps_run_the_layer_unchanged():
    torch.manual_seed(1)
    base, x = Net(), torch.randn(8, 256)
    m = copy.deepcopy(base)
    g = SDCGuard().protect(m)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.equal(m(x), base(x))
    assert g.stats["matmuls_seen"] == 0


def test_catches_injected_faults_and_names_the_layer():
    torch.manual_seed(2)
    m, x = Net(), torch.randn(64, 256)
    g = SDCGuard().protect(m)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=4).arm_on([m.a])
    g.on_check_start = lambda _op, C: inj.match(C)
    g.on_check = lambda _op, tok, res: inj.resolve(tok, res.ok, res.max_z)
    _run(m, g, x, inj=inj, steps=2)
    inj.remove_hooks()
    caught = [e for e in inj.events if e.get("caught")]
    assert len(inj.events) >= 2 and caught
    assert g.log.events[0].detail["where"] == "a"


def test_fp32_layers_are_left_alone():
    torch.manual_seed(3)
    m, x = Net(), torch.randn(8, 256)
    g = SDCGuard().protect(m)
    _run(m, g, x, autocast=False, steps=1)
    assert g.stats["matmuls_seen"] == 0


def test_unprotect_restores_forward():
    """protect() gives each layer a subclass whose forward runs the check (sdcguard._swap); unprotect() puts
    the layer's own class back and leaves nothing on the instance."""
    import torch.nn as nn
    m = Net()
    g = SDCGuard().protect(m)
    assert type(m.a) is not nn.Linear and isinstance(m.a, nn.Linear) and type(m.a).forward is not nn.Linear.forward
    assert "forward" not in m.a.__dict__ and repr(m.a).startswith("Linear(")
    g.unprotect()
    assert type(m.a) is nn.Linear and "forward" not in m.a.__dict__
    assert not [k for k in m.a.__dict__ if k.startswith("_sdcguard")]


def test_coverage_counts_matmuls_outside_linear_layers():
    class WithBmm(Net):
        def forward(self, x):
            h = torch.relu(self.a(x))
            h = torch.bmm(h, h.transpose(1, 2)) @ h          # matmuls outside nn.Linear
            return self.ln(self.b(h))

    m, x = WithBmm(), torch.randn(2, 16, 256)
    g = SDCGuard().protect(m)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cov = g.coverage(lambda: m(x))
    assert 0.0 < cov["share_checked_in_fast_mode"] < 1.0


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="no torch.compile")
def test_works_under_torch_compile():
    torch.manual_seed(5)
    m, x = Net(), torch.randn(16, 256)
    g = SDCGuard().protect(m)
    cm = torch.compile(m)
    try:
        for s in range(2):                      # unchecked, then checked
            ctx = g.at_step(s) if s == 1 else torch.enable_grad()
            with ctx:
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    loss = cm(x).float().pow(2).mean()
                loss.backward()
    except Exception as e:  # noqa: BLE001  (as in test_compile.py)
        if "compiler" in str(e).lower():
            pytest.skip(f"no working C++ compiler for Inductor here: {e}")
        raise
    assert g.stats["checked"] == 5 and g.stats["alerts"] == 0          # a: forward, dW; b: forward, dX, dW


def test_layer_used_twice_in_a_step_matches_autocast():
    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = nn.Linear(128, 128)

        def forward(self, x):
            return self.c(torch.relu(self.c(x)))

    torch.manual_seed(7)
    base, x = Twice(), torch.randn(16, 128)
    g_plain, _ = _run(copy.deepcopy(base), SDCGuard(), x, steps=1)
    m = copy.deepcopy(base)
    g_fast, _ = _run(m, SDCGuard().protect(m), x, steps=1)
    assert all(torch.equal(g_plain[0][k], g_fast[0][k]) for k in g_plain[0])


def test_non_contiguous_input_with_bias_no_false_alarm():
    torch.manual_seed(8)
    lin = nn.Linear(256, 300)
    with torch.no_grad():
        lin.bias.copy_(torch.randn(300) * 50)        # big bias, the double-rounding worst case
    x = torch.randn(4, 300, 256).transpose(0, 1)     # non-contiguous: at::linear adds the bias separately
    g = SDCGuard().protect(lin)
    with g.at_step(0):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lin(x).float().sum().backward()
    assert g.stats["alerts"] == 0 and g.stats["checked"] == 2


def test_reentrant_checkpointing_keeps_gradients():
    from torch.utils.checkpoint import checkpoint

    class Net2(Net):
        def forward(self, x):
            h = checkpoint(lambda t: torch.relu(self.a(t)), x, use_reentrant=True)
            return self.ln(self.b(h))

    torch.manual_seed(9)
    base, x = Net2(), torch.randn(8, 256, requires_grad=True)
    g_plain, _ = _run(copy.deepcopy(base), SDCGuard(), x, steps=1)
    m = copy.deepcopy(base)
    g_fast, _ = _run(m, SDCGuard().protect(m), x, steps=1)
    assert set(g_fast[0]) == set(g_plain[0])
    assert all(g_fast[0][k] is not None for k in g_fast[0])
    assert all(torch.equal(g_plain[0][k], g_fast[0][k]) for k in g_plain[0])


def test_no_grad_forward_earlier_in_the_checked_step():
    torch.manual_seed(10)
    m, x = Net(), torch.randn(8, 256)
    g = SDCGuard().protect(m)
    with g.at_step(0):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                m(x)                                  # e.g. an eval or logging forward
            loss = m(x).float().pow(2).mean()
        loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert g.stats["alerts"] == 0


def test_a_loss_spike_checks_every_following_step():
    from sdcguard import SDCGuard
    g = SDCGuard(every=100)
    rng = __import__("random").Random(0)
    for step in range(1, 60):
        assert not g.observe(step, loss=2.0 + 0.01 * rng.random(), grad_norm=1.0 + 0.02 * rng.random())
    assert not g.should_check(61)
    assert g.observe(60, loss=9.0) and g.anomalies[-1]["step"] == 60
    assert all(g.should_check(s) for s in range(61, 111)) and not g.should_check(111)
    assert g.observe(70, grad_norm=float("nan"))                     # not finite: checked too, window extended
    assert g.should_check(120) and g.stats["anomaly_windows"] == 2
    assert not g.observe(71, loss=1.0)                                # a drop isn't a spike


def test_every_checked_step_without_checks_is_counted():
    """A checked step whose protected layers ran no checked matmul is warned about once, but every such step is
    counted in stats["checked_steps_without_checks"] (as backward_outside_step and the other counts are), so the
    count says how many checked steps checked nothing."""
    import warnings
    m = nn.Linear(8, 8)                              # float32 without autocast: nothing is checked
    g = SDCGuard(every=1).protect(m)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for s in range(5):
            with g.on_step(s):
                m(torch.randn(2, 8)).sum().backward()
    assert g.stats["checked"] == 0 and g.stats["checked_steps_without_checks"] == 5, g.stats
    assert sum("ran no checks" in str(x.message) for x in w) == 1
