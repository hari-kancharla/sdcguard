"""The vLLM plugin's weight scan (SDCGUARD_WEIGHT_SCAN=1) with stand-ins for vLLM (tests/_vllm_standins.py): off by
default; on, the checked layers and the ones SDCGUARD_SHARE leaves out are both compared with their weights at load,
a changed bit in either is named, a reload takes new fingerprints instead of calling the reload corruption, and the
summary, the stack and the alerts say what's scanned."""
import logging
import types

import pytest
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


class _Holder(torch.nn.Module):
    def __init__(self, **params):
        super().__init__()
        for k, v in params.items():
            setattr(self, k, torch.nn.Parameter(v, requires_grad=False))


def _whole_model(M):
    model = load(Decoder(M, n_layers=2, lm_head=True))
    torch.manual_seed(5)
    model.embed = _Holder(weight=torch.randn(1024, 256).to(torch.bfloat16))
    model.lm_head.weight = model.embed.weight                                   # tied: one tensor, two modules
    model.norm = _Holder(weight=torch.ones(256, dtype=torch.bfloat16))
    model.experts = _Holder(w13_weight=torch.randn(4, 256, 1280).to(torch.bfloat16) * 0.02,   # 2.6 MB: 3 slices
                            w2_weight=torch.randn(4, 640, 64).to(torch.bfloat16) * 0.02)
    model.attn = _Holder(k_scale=torch.tensor(1.0))
    return model


def test_the_models_other_parameters_are_scanned_too(monkeypatch, tmp_path):
    """Embeddings and the LM head (tied: scanned once), norm weights, MoE experts' weights (sliced to the read budget),
    each named when a bit changes or two values are swapped; attention scales vLLM may set at first use aren't
    scanned, and a reload takes their new values as the reference."""
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_WEIGHT_SCAN_MB="1")
    model = _whole_model(M)
    runner = Runner(model)
    info = plugin._register_scan_rest(runner)
    g = plugin.guard()
    names = [e["name"] for e in g.wscan.entries if e["kind"] == "other"]
    step = (1 << 20) // 2
    assert names == ["lm_head", "norm", f"experts.w13_weight[0:{step}]", f"experts.w13_weight[{step}:{2 * step}]",
                     f"experts.w13_weight[{2 * step}:{4 * 256 * 1280}]", "experts.w2_weight"], names
    assert info == {"other_entries": 6, "not_scanned": 1}
    assert plugin._register_scan_rest(runner) is None                           # once
    cov = g.coverage_report()["weights"]
    assert cov["by_kind"] == {"checked": 2, "other": 6}
    assert "k_scale" in cov["not_scanned"]["attn.k_scale"] or "attention scale" in cov["not_scanned"]["attn.k_scale"]
    runner.execute_model(x_in())
    for _ in range(12):
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    _flip(model.embed.weight, 300, 2)
    _flip(model.norm.weight, 7, 0)
    _flip(model.experts.w13_weight, step + 11, 9)
    w2 = model.experts.w2_weight.view(-1).view(torch.int16)
    i = int((w2 != w2[0]).nonzero()[0])
    with torch.no_grad():
        a, b = int(w2[0]), int(w2[i])
        w2[0], w2[i] = b, a                                                     # a swap
    found = set()
    for _ in range(12):
        runner.execute_model(x_in())
        found |= {e.detail["where"] for e in plugin.poll(wait=True) if e.detector == "serve_weights"}
    assert found == {"lm_head", "norm", f"experts.w13_weight[{step}:{2 * step}]", "experts.w2_weight"}, found
    with torch.no_grad():                                                       # an intended update, then a reload
        model.norm.weight.mul_(2)
    plugin._rearm()
    for _ in range(12):
        runner.execute_model(x_in())
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    _flip(model.norm.weight, 3, 1)
    found = set()
    for _ in range(12):
        runner.execute_model(x_in())
        found |= {e.detail["where"] for e in plugin.poll(wait=True) if e.detector == "serve_weights"}
    assert found == {"norm"}


def test_the_verification_age_settings(monkeypatch, tmp_path, caplog):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_WEIGHT_SCAN_AGE="30",
                               SDCGUARD_WEIGHT_SCAN_MAX_MB="256")
    assert plugin._weight_scan_age() == 30.0 and plugin._weight_scan_max_mb() == 256.0
    model = load(Decoder(M, n_layers=2, lm_head=False))
    ws = plugin.guard().wscan
    assert ws.target_age == 30.0 and ws.max_budget == 256 << 20
    for bad in ("abc", "0.5"):
        monkeypatch.setenv("SDCGUARD_WEIGHT_SCAN_AGE", bad)
        with caplog.at_level(logging.WARNING, logger="sdcguard"):
            assert plugin._weight_scan_age() is None
    monkeypatch.delenv("SDCGUARD_WEIGHT_SCAN_AGE")
    assert plugin._weight_scan_age() is None and plugin._weight_scan_max_mb() == 256.0
    assert sum("SDCGUARD_WEIGHT_SCAN_AGE" in r.getMessage() for r in caplog.records) == 2


def test_parameters_sharing_one_buffer_arent_scanned(monkeypatch, tmp_path):
    """vLLM's prefetch offloader points layers N, N + its step, ... at one GPU buffer (distinct Parameters, one
    address) and copies each layer's weights in before it runs: none of them is scanned, each is listed, and the
    copies raise nothing. A layer the checks found unreadable as its own isn't taken in either."""
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1")
    model = load(Decoder(M, n_layers=2, lm_head=False))
    buf = torch.randn(256).to(torch.bfloat16)
    model.norm_layer0 = _Holder(weight=torch.empty(0))
    model.norm_layer0.weight = torch.nn.Parameter(buf, requires_grad=False)
    model.norm_layer2 = _Holder(weight=torch.empty(0))
    model.norm_layer2.weight = torch.nn.Parameter(buf.view_as(buf), requires_grad=False)
    model.unreadable = _Holder(weight=torch.randn(8).to(torch.bfloat16))
    g = plugin.guard()
    g.wscan.not_scanned["unreadable"] = "its weights can't be read back as its own: a test"
    runner = Runner(model)
    plugin._register_scan_rest(runner)
    ns = g.wscan.not_scanned
    assert "shares its memory" in ns["norm_layer0.weight"] and "shares its memory" in ns["norm_layer2.weight"]
    assert "can't be read back" in ns["unreadable"]
    assert not [e for e in g.wscan.entries if e["kind"] == "other"]
    runner.execute_model(x_in())
    for _ in range(3):
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]
    with torch.no_grad():
        buf.copy_(torch.randn(256).to(torch.bfloat16))                  # the offloader copies layer 2's weights in
    for _ in range(3):
        runner.execute_model(x_in())
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]


def test_with_the_expert_load_balancer_on_experts_weights_arent_scanned(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1")
    model = load(Decoder(M, n_layers=2, lm_head=False))

    class RoutedExperts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.randn(2, 8, 4).to(torch.bfloat16), requires_grad=False)

    model.moe = RoutedExperts()
    runner = Runner(model)
    runner.parallel_config = types.SimpleNamespace(enable_eplb=True)
    plugin._register_scan_rest(runner)
    ns = plugin.guard().coverage_report()["weights"]["not_scanned"]
    assert "--enable-eplb" in ns["moe.w13_weight"]
    assert not [e for e in plugin.guard().wscan.entries if e["kind"] == "other"]


def test_the_experiments_rpcs_change_a_parameter_and_name_its_scan_entry(monkeypatch, tmp_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    import sdcguard_test_ext as ext
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN="1", SDCGUARD_WEIGHT_SCAN_MB="1")
    model = _whole_model(M)
    runner = Runner(model)
    plugin._register_scan_rest(runner)
    fi = ext.FaultInjection()
    fi.get_model = lambda: model
    cov = fi.sdcguard_test_scan_uncovered()
    assert cov["uncovered"] == ["attn.k_scale"] and "attn.k_scale" in cov["not_scanned"], cov
    runner.execute_model(x_in())
    plugin.poll(wait=True)
    step = (1 << 20) // 2
    r = fi.sdcguard_test_change_param("experts.", "w13_weight", 0.5, "flip", 15)
    assert r["name"] == "experts.w13_weight" and r["index"] == 4 * 256 * 1280 // 2 and \
        r["entry"] == f"experts.w13_weight[{step}:{2 * step}]" and r["before"][0] != r["after"][0]
    back = fi.sdcguard_test_change_param("experts.", "w13_weight", 0.5, "flip", 15, r["index"])
    assert back["after"] == r["before"]
    s = fi.sdcguard_test_change_param("norm", "weight", 0.0, "swap")       # all ones: no two neighbours differ
    assert "error" in s
    s = fi.sdcguard_test_change_param("embed", "weight", 0.25, "swap")
    assert s["entry"] == "lm_head" and s["after"] == s["before"][::-1] and s["before"][0] != s["before"][1]
    found = set()
    for _ in range(12):
        runner.execute_model(x_in())
        found |= {e.detail["where"] for e in plugin.poll(wait=True) if e.detector == "serve_weights"}
    assert found == {"lm_head"}, found                                    # (the flip was undone before any read)
    u = fi.sdcguard_test_change_param("embed", "weight", 0.25, "swap", 0, s["index"])
    assert u["after"] == s["before"]
    assert fi.sdcguard_test_scan_refresh_entry("lm_head") == {"refreshed": 1}
    assert fi.sdcguard_test_scan_refresh_entry(f"experts.w13_weight[{step}:{2 * step}]") == {"refreshed": 1}
    for _ in range(12):
        runner.execute_model(x_in())
        assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_weights"]


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p



@pytest.mark.parametrize("scan,want", [("1", "VERIFIED"), ("0", "NO CHANGE SEEN")])
def test_a_decision_says_the_weights_were_verified_only_when_the_scan_ran(monkeypatch, tmp_path, scan, want):
    """From the metrics page a worker exports to `sdcguard decide`: with the weight scan, every entry compared and
    matched, and how long ago the least recently verified one was; without it, the checksum checks' silence, which
    verifies no weight."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard import decide as dc
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_WEIGHT_SCAN=scan,
                               SDCGUARD_METRICS_PORT=str(_free_port()))
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model = load(Decoder(M, n_layers=4, lm_head=False))
    runner = Runner(model)
    for _ in range(3):
        runner.execute_model(x_in())
        plugin.poll(wait=True)
    text = generate_latest(plugin.guard().metrics.registry).decode()
    cov = dc.plugin_coverage([text])
    w = dc.decide(None, alerts=[], plugin=cov)["weights"]
    assert w["verdict"] == want, w
    if scan == "1":
        assert cov["workers"][0]["weight_scan_layers"] == 4 and cov["workers"][0]["weight_scan_never_compared"] == 0
        assert w["oldest_verification_age_seconds"] is not None
    else:
        assert cov["workers"][0]["weight_scan_layers"] is None and "SDCGUARD_WEIGHT_SCAN" in w["why"]
