"""The new opt-in must not activate other engine observers or checks."""
from sdcguard import vllm_plugin
from sdcguard.lifetime import vllm_forward, vllm_nixl


def test_forward_opt_in_is_separate_from_existing_activation(monkeypatch):
    calls = []
    monkeypatch.setenv("SDCGUARD", "0")
    monkeypatch.delenv("SDCGUARD_REQUIRE", raising=False)
    monkeypatch.delenv("SDCGUARD_LIFETIME_DIR", raising=False)
    monkeypatch.delenv("SDCGUARD_FORWARD_DIR", raising=False)
    monkeypatch.setattr(vllm_forward, "install", lambda: calls.append("forward"))
    monkeypatch.setattr(vllm_nixl, "install", lambda: calls.append("lifetime"))
    vllm_plugin.register()
    assert calls == []
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", "/unused/in/this/unit/test")
    vllm_plugin.register()
    assert calls == ["forward"]


def test_ordinary_install_error_does_not_turn_into_engine_error(monkeypatch):
    monkeypatch.setenv("SDCGUARD", "0")
    monkeypatch.delenv("SDCGUARD_REQUIRE", raising=False)
    monkeypatch.delenv("SDCGUARD_LIFETIME_DIR", raising=False)
    monkeypatch.setenv("SDCGUARD_FORWARD_DIR", "/unused/in/this/unit/test")

    def fail():
        raise ValueError("unsupported observer input")

    monkeypatch.setattr(vllm_forward, "install", fail)
    assert vllm_plugin.register() is None
