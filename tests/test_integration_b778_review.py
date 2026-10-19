"""Behavioral regressions from independent review of the complete b778d27 source snapshot."""
import copy

import pytest
import torch
from torch import nn

from sdcguard import SDCGuard
from sdcguard.serve import ServeGuard


def test_protect_preserves_float64_outputs_and_gradients_inside_autocast():
    torch.manual_seed(927)
    plain = nn.Linear(16, 32).double()
    protected = copy.deepcopy(plain)
    x = torch.randn(4, 16, dtype=torch.float64)
    g = SDCGuard().protect(protected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = plain(x)
        expected.square().sum().backward()
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = protected(x)
        actual.square().sum().backward()
    assert actual.dtype == torch.float64 and torch.equal(actual, expected)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(plain.parameters(), protected.parameters()))
    assert g.stats["checked"] == 0
    g.unprotect()


@pytest.mark.parametrize("autocast,weight_dtype", [(False, torch.float32), (True, torch.float64)])
def test_protect_preserves_pytorch_mixed_dtype_errors(autocast, weight_dtype):
    layer = nn.Linear(16, 32).to(weight_dtype)
    x = torch.randn(4, 16, dtype=torch.float16)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), pytest.raises(RuntimeError):
        layer(x)
    g = SDCGuard().protect(layer)
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), pytest.raises(RuntimeError):
        layer(x)
    g.unprotect()


@pytest.mark.parametrize("attach_autocast,run_autocast", [(False, True), (True, False), (True, True)])
def test_serving_autocast_cannot_lower_checksum_precision(attach_autocast, run_autocast):
    # Legitimate bias cancellation exposes a low-precision checksum. b778d27 alarmed at z46.3,
    # z42.0 and z9.45 for these three cases, even though every output matched plain PyTorch.
    torch.manual_seed(32)
    layer = nn.Linear(256, 128).bfloat16()
    x = torch.ones(4, 256, dtype=torch.bfloat16)
    with torch.no_grad():
        layer.bias.copy_(-layer.weight.float().sum(1).bfloat16())
        expected = layer(x)
    baseline = ServeGuard().attach(layer)
    with torch.no_grad():
        layer(x)
    expected_z = baseline.worst_z()["Linear"]
    baseline.detach()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=attach_autocast):
        g = ServeGuard().attach(layer)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=run_autocast):
        actual = layer(x)
    assert torch.equal(actual, expected)
    assert g.worst_z()["Linear"] == pytest.approx(expected_z, abs=1e-6)
    assert g.poll(wait=True) == []
    # The fix must preserve positive detection under that same ambient autocast region.
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=run_autocast):
        layer.weight[0, 0].add_(100)
        layer(x)
    assert g.poll(wait=True)
    g.detach()


def test_refresh_resumes_checking_after_unsupported_dtype_without_shape_change():
    torch.manual_seed(927)
    layer = nn.Linear(64, 128).bfloat16()
    x = torch.randn(4, 64).bfloat16()
    g = ServeGuard().attach(layer)
    layer.float()
    g.refresh()
    assert "Linear" in g.unsupported
    layer.bfloat16()
    g.refresh()
    assert not g.unsupported
    with torch.no_grad():
        layer(x)
    assert g.worst_z()["Linear"] > 0
    assert g.poll(wait=True) == []
    with torch.no_grad():
        layer.weight[0, 0].add_(100)
        layer(x)
    events = g.poll(wait=True)
    assert len(events) == 1 and events[0].detail["where"] == "Linear"
    g.detach()


def test_vllm_weight_reload_refreshes_engine_bindings(monkeypatch, tmp_path):
    from test_serve import _fake_vllm, _plugin_env, _plugin_model

    method, runner = _fake_vllm(monkeypatch)
    plugin = _plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = _plugin_model(method, n_layers=1)
    layer = model[0]
    g = plugin.guard()
    x = torch.randn(4, 256).bfloat16()
    # A replacement with a new output shape must replace cached plugin arguments too. The original
    # plugin kept the old checksum tensors and crashed while checking this otherwise valid output.
    with torch.no_grad():
        layer.weight = nn.Parameter(torch.randn(40, 256).bfloat16() / 16)
        layer.bias = nn.Parameter(torch.randn(40).bfloat16())
    layer.quant_method.process_weights_after_loading(layer)
    with torch.no_grad():
        assert runner(model).execute_model(x).shape == (4, 40)
    assert plugin.poll(wait=True) == []
    # Temporarily unsupported weights disable the plugin binding, then restore it when bf16 returns.
    layer.float()
    plugin.refresh()
    assert layer._sdcguard_args is None
    with torch.no_grad():
        runner(model).execute_model(x.float())
    layer.bfloat16()
    plugin.refresh()
    assert layer._sdcguard_args is not None
    with torch.no_grad():
        runner(model).execute_model(x)
    assert plugin.poll(wait=True) == []
    with torch.no_grad():
        layer.weight[0, 0].add_(100)
        runner(model).execute_model(x)
    assert plugin.poll(wait=True)
    g.detach()


@pytest.mark.parametrize("k,n", [(0, 128), (64, 0)])
@pytest.mark.parametrize("precise", [False, True])
def test_zero_channel_linears_preserve_forward_and_backward(k, n, precise):
    torch.manual_seed(927)
    plain = nn.Linear(k, n)
    protected = copy.deepcopy(plain)
    x = torch.randn(2, k, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    g = SDCGuard(precise=precise).protect(protected)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = plain(x)
        expected.sum().backward()
    with g.at_step(0), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = protected(x2)
        actual.sum().backward()
    assert torch.equal(actual, expected) and torch.equal(x.grad, x2.grad)
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(plain.parameters(), protected.parameters()))
    assert g.stats["alerts"] == 0
    g.unprotect()


def test_coverage_does_not_count_unchecked_float32_work_as_protected():
    torch.manual_seed(927)
    model = nn.Linear(16, 32)
    x = torch.randn(4, 16)
    g = SDCGuard().protect(model)
    plain = g.coverage(lambda: model(x))
    assert plain["in_protected_linear_layers"] > 0
    assert plain["share_checked_in_fast_mode"] == 0
    assert plain["coverage_kind"] == "potential" and plain["guard_stats"]["checked"] == 0
    with g.at_step(0):
        model(x)
    assert g.stats["checked"] == 0
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lowp = g.coverage(lambda: model(x))
    assert lowp["share_checked_in_fast_mode"] == 1
    assert lowp["share_eligible_in_fast_mode"] == 1 and lowp["guard_stats"]["checked"] == 0
    with g.at_step(1), torch.autocast("cpu", dtype=torch.bfloat16):
        model(x)
    assert g.stats["checked"] == 1
    g.unprotect()
