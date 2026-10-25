"""The vLLM plugin's spot checks (SDCGUARD_SPOT=1) with stand-ins for vLLM (tests/_vllm_standins.py): off by default;
on, the layers the checksum check takes are registered after load, one layer a step recomputes a few outputs, a
served output a little wrong is named with its row and column, and the stack says spot checks are on."""
import torch

from _vllm_standins import Decoder, install_fake_vllm, load, plugin_env, x_in


def _setup(monkeypatch, tmp_path, **env):
    M, Runner = install_fake_vllm(monkeypatch.setitem)
    plugin = plugin_env(monkeypatch, tmp_path, **env)
    plugin.register()
    return plugin, M, Runner


def test_off_by_default(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path)
    model = load(Decoder(M, n_layers=3, lm_head=False))
    runner = Runner(model)
    assert plugin._register_spot(runner) is None and plugin.guard().spot is None
    plugin._publish_stack(runner)
    assert "spot=off" in plugin._STATE["stack"]["checks"]


def test_a_slightly_wrong_output_is_named(monkeypatch, tmp_path):
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_SPOT="1")
    model = load(Decoder(M, n_layers=3, lm_head=False))
    runner = Runner(model)
    info = plugin._register_spot(runner)
    assert info == {"checked": 3, "not_checked": {}}
    g = plugin.guard()
    for _ in range(6):
        runner.execute_model(x_in())
    assert not [e for e in plugin.poll(wait=True) if e.detector == "serve_spot"]
    assert g.spot.stats["active_calls"] == 6 and max(g.spot.worst_seen.values()) <= 1.0
    target = model.layers[1].weight
    real_linear = torch.nn.functional.linear

    def linear(x, w, b=None):
        y = real_linear(x, w, b)
        if w is target:                                 # every output of layer 1 3% too large: small for the
            y = (y.float() * 1.03).to(y.dtype)          # checksum check, over the spot check's line
        return y
    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    found = []
    for _ in range(6):
        runner.execute_model(x_in())
        found += [e for e in plugin.poll(wait=True) if e.detector == "serve_spot"]
    assert found and {e.detail["where"] for e in found} == {"model.layers.1.proj"}
    assert all(e.detail["recheck"]["verdict"] == "output_wrong" for e in found if e.detail.get("captured"))
    plugin._publish_stack(runner)
    assert "spot=on" in plugin._STATE["stack"]["checks"]
    assert g.coverage_report()["spot"]["layers"] == 3


def test_the_compile_cache_key_follows_the_spot_fp8_moe_and_attention_settings(monkeypatch, tmp_path):
    """vLLM's AOT compile cache is keyed by vLLM's own settings only: a graph traced without the spot op (or the FP8
    op) would be loaded by a server that has them on, which then checks nothing in its compiled layers."""
    plugin = plugin_env(monkeypatch, tmp_path)
    keys = {}
    for name, env in (("default", {}), ("spot", {"SDCGUARD_SPOT": "1"}), ("fp8", {"SDCGUARD_FP8": "1"}),
                      ("moe", {"SDCGUARD_MOE": "1"}), ("noattn", {"SDCGUARD_ATTN": "0"})):
        for k in ("SDCGUARD_SPOT", "SDCGUARD_FP8", "SDCGUARD_MOE", "SDCGUARD_ATTN"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        keys[name] = plugin._graph_key(1.0, 0, True)
    assert len(set(keys.values())) == 5, keys
    for k in ("SDCGUARD_FP8", "SDCGUARD_MOE", "SDCGUARD_ATTN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SDCGUARD_SPOT", "on")                    # the same setting spelled another way
    assert plugin._graph_key(1.0, 0, True) == keys["spot"]
    from sdcguard import serve_spot
    monkeypatch.setattr(serve_spot, "SPOT_SCHEMA", serve_spot.SPOT_SCHEMA + 1)   # other op arguments: another key
    assert plugin._graph_key(1.0, 0, True) != keys["spot"]


def test_a_spot_alert_is_logged_as_a_spot_alert(monkeypatch, tmp_path, caplog):
    import logging
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_SPOT="1")
    model = load(Decoder(M, n_layers=3, lm_head=False))
    runner = Runner(model)
    plugin._register_spot(runner)
    target = model.layers[0].weight
    real_linear = torch.nn.functional.linear

    def linear(x, w, b=None):
        y = real_linear(x, w, b)
        return (y.float() * 1.05).to(y.dtype) if w is target else y
    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        for _ in range(4):
            runner.execute_model(x_in())
            plugin.poll(wait=True)
    lines = [r.getMessage() for r in caplog.records if "spot check of" in r.getMessage()]
    assert lines and "model.layers.0.proj" in lines[0] and "line 2.0" in lines[0] and "output_wrong" in lines[0]
    assert not [r for r in caplog.records if "checksum" in r.getMessage() and "z=" in r.getMessage()
                and "None checksum" in r.getMessage()]


def test_spot_is_off_in_the_stack_when_every_layer_is_left_out(monkeypatch, tmp_path):
    from sdcguard import serve_spot
    plugin, M, Runner = _setup(monkeypatch, tmp_path, SDCGUARD_SPOT="1")
    monkeypatch.setattr(serve_spot, "bound_holds", lambda d, t: "reductions in bf16")
    model = load(Decoder(M, n_layers=3, lm_head=False))
    runner = Runner(model)
    info = plugin._register_spot(runner)
    assert info["checked"] == 0 and len(info["not_checked"]) == 3
    plugin._publish_stack(runner)
    assert "spot=off" in plugin._STATE["stack"]["checks"]
