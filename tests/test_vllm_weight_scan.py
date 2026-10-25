"""The vLLM plugin's weight scan (SDCGUARD_WEIGHT_SCAN=1) with stand-ins for vLLM (tests/_vllm_standins.py): off by
default; on, the checked layers and the ones SDCGUARD_SHARE leaves out are both compared with their weights at load,
a changed bit in either is named, a reload takes new fingerprints instead of calling the reload corruption, and the
summary, the stack and the alerts say what's scanned."""
import logging
import types

import torch

from _vllm_standins import Decoder, install_fake_vllm, load, plugin_env, x_in


def _setup(monkeypatch, tmp_path, **env):
    M, Runner = install_fake_vllm(monkeypatch.setitem)
    plugin = plugin_env(monkeypatch, tmp_path, **env)
    plugin.register()
    return plugin, M, Runner


def _flip(t, index, bit):
    with torch.no_grad():
        t.view(-1).view(torch.int16)[index] ^= (1 << bit)


_R = {"checked": 0, "total": 0, "weight_share_checked": 0.0, "unchecked_by_method": {}, "quantized_or_moe": 0,
      "never_checked": []}


def test_off_by_default(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path)
    model = load(Decoder(M, n_layers=4, lm_head=False))
    Runner(model).execute_model(x_in())
    assert plugin.guard().wscan is None
    plugin._publish_stack(Runner(model))
    assert "weights=off" in plugin._STATE["stack"]["checks"]
    _flip(model.layers[0].weight, 0, 0)
    assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]


def test_checked_and_share_excluded_layers_are_scanned_and_a_flip_in_either_is_named(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_SHARE="0.5")
    model = load(Decoder(M, n_layers=4, lm_head=False))
    g = plugin.guard()
    checked = {lay["name"] for lay in g.layers}
    assert len(checked) == 2 and g.wscan is not None
    runner = Runner(model)
    runner.execute_model(x_in())
    assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    cov = g.coverage_report()["weights"]
    assert cov["entries"] == 4 and cov["by_kind"] == {"checked": 2, "unchecked": 2}
    left_out = next(lay for lay in model.layers if lay.prefix not in checked)
    inside = next(lay for lay in model.layers if lay.prefix in checked)
    _flip(left_out.weight, 11, 0)
    _flip(inside.weight, 5, 3)
    found = {}
    for _ in range(3):
        runner.execute_model(x_in())
        found.update({e.detail["where"]: e.detail["kind"] for e in plugin.poll(wait=True)
                      if e.detector == "serve_weights"})
    assert found == {left_out.prefix: "unchecked", inside.prefix: "checked"}
    msg = plugin._summary(dict(_R))
    assert "weights compared with their load-time fingerprint on 4 layers in turn" in msg
    assert "SDCGUARD_WEIGHT_SCAN=1" in msg
    plugin._publish_stack(runner)
    assert "weights=on" in plugin._STATE["stack"]["checks"]
    ev = types.SimpleNamespace(detector="serve", detail={})
    plugin._attribute(ev)
    assert "the scanned layers' weights against their load-time fingerprints" in " ".join(ev.detail["unknown_impact"])


def test_a_reload_takes_new_fingerprints(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_SHARE="0.5")
    model = load(Decoder(M, n_layers=4, lm_head=False))
    runner = Runner(model)
    runner.execute_model(x_in())
    plugin.poll(wait=True)
    with torch.no_grad():                         # an RL weight sync: every weight changes in place, then reloads
        for lay in model.layers:
            lay.weight.mul_(1.5)
    load(model)
    for _ in range(3):
        runner.execute_model(x_in())
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    assert plugin.guard().coverage_report()["weights"]["entries"] == 4      # refreshed, not added again


def test_the_megabytes_a_read_compares(monkeypatch, tmp_path, caplog):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_WEIGHT_SCAN_MB="8")
    assert plugin._weight_scan_mb() == 8.0
    for bad in ("abc", "0.5"):
        monkeypatch.setenv("SDCGUARD_WEIGHT_SCAN_MB", bad)
        with caplog.at_level(logging.WARNING, logger="sdcguard"):
            assert plugin._weight_scan_mb() == 64.0
    assert sum("SDCGUARD_WEIGHT_SCAN_MB" in r.getMessage() for r in caplog.records) == 2
    for v, on in (("1", True), ("true", True), ("0", False), ("", False), ("off", False)):
        monkeypatch.setenv("SDCGUARD_WEIGHT_SCAN", v)
        assert plugin._weight_scan_on() is on, v


def test_a_layerwise_reload_keeps_left_out_layers_scanned_and_alerts_are_logged_and_attributed(monkeypatch, tmp_path,
                                                                                               caplog):
    """vLLM 0.32's reload runs process_weights_after_loading on temporary tensors and copies them into the original
    parameters afterwards: the scan takes the reference again at the next turn instead of losing the layer."""
    import torch.nn as nn
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_SHARE="0.5")
    model = load(Decoder(M, n_layers=4, lm_head=False))
    g = plugin.guard()
    runner = Runner(model)
    runner.execute_model(x_in())
    plugin.poll(wait=True)
    left_out = next(lay for lay in model.layers if lay.prefix not in {x["name"] for x in g.layers})
    orig = left_out.weight
    left_out.weight = nn.Parameter(orig.detach().clone() * 1.25, requires_grad=False)      # the temporary
    left_out.quant_method.process_weights_after_loading(left_out)
    with torch.no_grad():
        orig.copy_(left_out.weight)
    left_out.weight = orig
    for _ in range(2):
        runner.execute_model(x_in())
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    assert left_out.prefix not in g.coverage_report()["weights"]["not_scanned"]
    _flip(orig, 3, 6)
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        found = []
        for _ in range(3):
            runner.execute_model(x_in())
            found += [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["where"] for e in found] == [left_out.prefix]
    assert any(f"the weights of {left_out.prefix}" in r.getMessage() for r in caplog.records)
    d = found[0].detail
    assert "potentially_exposed" in d and d.get("read_window_steps") is not None
