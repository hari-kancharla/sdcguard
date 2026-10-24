"""What a server runs (sdcguard.stack), published by the plugin, recorded by `sdcguard qualify` and checked by
`sdcguard ready --qualified`: the fields read from vLLM's config and model, the fingerprint, the metrics label round
trip, the comparison, and the qualified verdicts."""
import json
import os
import socket
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from sdcguard import qualify, ready, stack

HERE = os.path.dirname(os.path.abspath(__file__))


class _Backend:
    @staticmethod
    def get_name():
        return "FLASH_ATTN"


class _Attn(torch.nn.Module):
    attn_backend = _Backend


class _HFConfig:
    def __init__(self, **over):
        self.d = {"hidden_size": 1536, "rms_norm_eps": 1e-6, "_name_or_path": "/some/local/path", **over}

    def to_dict(self):
        return dict(self.d)


def _runner(attn=_Backend, mode="VLLM_COMPILE", quant=None, hf=None, overrides=None, gen="auto"):
    cfg = NS(model_config=NS(model="Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.bfloat16, quantization=quant,
                             max_model_len=2048, enforce_eager=False, hf_config=hf or _HFConfig(),
                             hf_overrides=overrides or {}, generation_config=gen, override_generation_config={},
                             revision="main"),
             parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
             compilation_config=NS(mode=f"CompilationMode.{mode}", cudagraph_mode="CUDAGraphMode.FULL_AND_PIECEWISE"),
             cache_config=NS(cache_dtype="auto"), kernel_config=NS(ir_op_priority=NS(rms_norm=["native"])))
    model = torch.nn.Sequential(_Attn())
    model[0].attn_backend = attn
    return NS(vllm_config=cfg), model


def test_the_stack_is_read_from_vllms_config_and_model():
    r, m = _runner()
    s = stack.collect(r, m, {"share": "1"})
    assert s["model"] == "Qwen/Qwen2.5-1.5B-Instruct" and s["dtype"] == "bfloat16" and s["attention"] == "FLASH_ATTN"
    assert s["compilation"] == "VLLM_COMPILE" and s["cudagraph"] == "FULL_AND_PIECEWISE" and s["rms_norm"] == "native"
    assert s["checks"] == "share=1" and "quantization" not in s and len(s["fingerprint"]) == 16
    assert stack.collect(r, m, {"share": "0.125"})["fingerprint"] == s["fingerprint"]     # checks aren't the stack

    class Triton:
        @staticmethod
        def get_name():
            return "TRITON_ATTN"
    for other in (_runner(attn=Triton), _runner(mode="NONE"), _runner(quant="fp8")):
        assert stack.collect(*other)["fingerprint"] != s["fingerprint"]


def test_the_metrics_label_round_trip():
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    r, m = _runner()
    s = stack.collect(r, m, {"share": "1"})
    s["env"] = 'VLLM_ATTENTION_BACKEND=FLASH_ATTN,X="quoted"'
    met = Metrics()
    met.stack(s)
    met.stack(s)                                          # a second call replaces the first
    text = generate_latest(met.registry).decode()
    back = stack.from_metrics(text)
    assert text.count("sdcguard_stack_info{") == 1
    assert back["fingerprint"] == s["fingerprint"] and back["env"] == s["env"] and back["model"] == s["model"]
    assert stack.compare(s, back)["same"] is True


def test_compare_names_what_differs():
    a = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "fingerprint": "x"}
    b = {**a, "attention": "TRITON_ATTN", "fingerprint": "y"}
    c = stack.compare(a, b)
    assert c["same"] is False and c["differences"] == ["attention: 'FLASH_ATTN' qualified, 'TRITON_ATTN' now"]
    assert stack.compare(a, None)["same"] is None and stack.compare({}, b)["same"] is None
    assert stack.compare(a, {**a, "fingerprint": "z"})["same"] is False      # a field the record doesn't show


def _decision(tmp_path, decision="ACCEPT", deployed=True):
    st = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0"}
    st["fingerprint"] = stack.fingerprint(st)
    d = {"decision": decision, "scope": {"stacks": {"candidate": {"vllm": "0.32.0",
                                                                    **({"deployed": st} if deployed else {})}}}}
    p = tmp_path / f"decision_{decision}_{deployed}.json"
    p.write_text(json.dumps(d))
    return str(p), st


def _page(st):
    return "sdcguard_stack_info{" + ",".join(f'{k}="{v}"' for k, v in st.items()) + "} 1.0\n"


def test_ready_qualified_verdicts(tmp_path):
    path, st = _decision(tmp_path)
    assert ready.qualified(path, [_page(st), _page(st)])["verdict"] is None
    other = {**st, "attention": "TRITON_ATTN"}
    other["fingerprint"] = stack.fingerprint(other)
    q = ready.qualified(path, [_page(st), _page(other)])
    assert q["verdict"] == "NOT THE QUALIFIED STACK" and q["differences"][0].startswith("worker 1: attention")
    assert ready.qualified(path, [_page(st), "no stack here"])["verdict"] == "INCONCLUSIVE"
    assert ready.qualified(_decision(tmp_path, "REJECT")[0], [_page(st)])["verdict"] == "NOT QUALIFIED"
    assert ready.qualified(_decision(tmp_path, deployed=False)[0], [_page(st)])["verdict"] == "INCONCLUSIVE"
    assert ready.qualified(str(tmp_path / "missing.json"), [_page(st)])["verdict"] == "INCONCLUSIVE"


def _free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_qualify_records_the_deployed_stack_and_ready_holds_a_server_to_it(tmp_path):
    out = str(tmp_path / "q")
    serve = f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}"
    code = qualify.main([
        "--out", out, "--baseline", "fake/m", "--candidate", "same=fake/m",
        "--candidate", "triton=fake/m --attention-backend TRITON_ATTN", "--no-isolation", "--no-perf",
        "--serve-cmd", serve, "--port", str(_free()), "--metrics-port", str(_free()), "--tokens", "8",
        "--startup-timeout", "30"])
    assert code in (0, 1)
    runs = json.load(open(os.path.join(out, "QUALIFY.json")))["runs"]
    assert runs["same"]["deployed_stack"]["fingerprint"] == runs["baseline"]["deployed_stack"]["fingerprint"]
    assert runs["triton"]["deployed_stack"]["attention"] == "TRITON_ATTN"
    dec = os.path.join(out, "decision_same.json")
    assert json.load(open(dec))["scope"]["stacks"]["candidate"]["deployed"]["attention"] == "FLASH_ATTN"
    st_same = runs["same"]["deployed_stack"]
    st_triton = runs["triton"]["deployed_stack"]
    assert ready.qualified(dec, [_page(st_same)])["verdict"] is None
    q = ready.qualified(dec, [_page(st_triton)])
    assert q["verdict"] == "NOT THE QUALIFIED STACK" and any("attention" in d for d in q["differences"])


def test_the_plugin_publishes_the_stack_whenever_its_guard_is_made(monkeypatch, tmp_path):
    """The stack is read at load_model. The guard (and its metrics page) is usually made while the weights load, but
    not always (no layer of the model checked yet): the stack reaches the page either way."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from _vllm_standins import Decoder, install_fake_vllm, load, plugin_env, x_in
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    monkeypatch.setattr(GM, "__init__", lambda self, factory: setattr(self, "factory", factory), raising=False)
    monkeypatch.setattr(GM, "load_model", lambda self: setattr(self, "model", load(self.factory())), raising=False)
    vp = plugin_env(monkeypatch, tmp_path, SDCGUARD_METRICS_PORT=str(_free()))
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    vp.register()
    runner = GM(lambda: Decoder(UL))
    runner.load_model()
    runner.execute_model(x_in())
    st = vp._STATE["stack"]
    back = stack.from_metrics(generate_latest(vp.guard().metrics.registry).decode())
    assert back and back["fingerprint"] == st["fingerprint"]
    assert "norms=off" in back["checks"] and "moe=off" in back["checks"]
    # a guard made after the stack was read
    monkeypatch.setitem(vp._STATE, "guard", None)
    monkeypatch.setenv("SDCGUARD_METRICS_PORT", str(_free()))
    back = stack.from_metrics(generate_latest(vp._guard(None).metrics.registry).decode())
    assert back and back["fingerprint"] == st["fingerprint"]


def test_a_deployment_beyond_what_was_tested_is_named(tmp_path):
    st = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "max_model_len": "32768"}
    st["fingerprint"] = stack.fingerprint(st)
    d = {"decision": "ACCEPT", "scope": {"stacks": {"candidate": {"deployed": st}},
                                         "tested": {"max_prompt_tokens": 600, "max_context_tokens": 760,
                                                    "prefix_cache_hit_tested": True}}}
    p = tmp_path / "d.json"
    p.write_text(json.dumps(d))
    hist = "\n".join(f'vllm:request_prompt_tokens_bucket{{engine="0",le="{le}",model_name="m"}} {n}'
                     for le, n in (("128.0", 10), ("512.0", 30), ("1024.0", 38), ("+Inf", 40)))
    q = ready.qualified(str(p), [_page(st)], hist)
    assert q["verdict"] is None
    out = q["outside_tested"]
    assert any("up to 32768 tokens" in x and "760" in x for x in out)
    assert any("at least 2 of 40 requests" in x and "1024" in x for x in out)
    assert ready.outside_tested({"max_context_tokens": 40000}, st, None) == []


def test_qualify_records_what_each_candidate_was_tested_on(tmp_path):
    out = str(tmp_path / "q")
    serve = f"{sys.executable} {os.path.join(HERE, '_fake_vllm.py')}"
    qualify.main(["--out", out, "--baseline", "fake/m", "--candidate", "same=fake/m", "--no-isolation", "--no-perf",
                  "--serve-cmd", serve, "--port", str(_free()), "--metrics-port", str(_free()), "--tokens", "8",
                  "--startup-timeout", "30"])
    t = json.load(open(os.path.join(out, "decision_same.json")))["scope"]["tested"]
    assert t["probes"] == ["quality", "consistency"] and t["prefix_cache_hit_tested"] is True
    assert t["decode_past_128_tested"] is True and t["max_context_tokens"] > 160 and t["max_concurrency"] is None
    rows = json.load(open(os.path.join(out, "QUALIFY.json")))["decisions"]
    assert rows["same"]["consistency"] == "CONSISTENT"


def test_a_changed_model_config_is_another_stack_and_ready_names_it():
    """The first qualify GPU run (Oct 24) gave the server started with --hf-overrides '{"rms_norm_eps": 1.0}' the same
    fingerprint as the trusted one, so `ready --qualified` would have held a misconfigured server to be the qualified
    stack. The effective config and the overrides are part of the stack now."""
    good = stack.collect(*_runner())
    bad = stack.collect(*_runner(hf=_HFConfig(rms_norm_eps=1.0), overrides={"rms_norm_eps": 1.0}))
    assert good["fingerprint"] != bad["fingerprint"] and bad["hf_overrides"] == "rms_norm_eps=1.0"
    c = stack.compare(good, bad)
    assert c["same"] is False and any(d.startswith("model_config:") for d in c["differences"])
    assert any("rms_norm_eps=1.0" in d for d in c["differences"])
    # where the config was read from isn't the stack; the generation config is (it sets default sampling)
    moved = stack.collect(*_runner(hf=_HFConfig(_name_or_path="/elsewhere")))
    assert moved["fingerprint"] == good["fingerprint"] and good["generation_config"] == "auto"
    assert stack.collect(*_runner(gen="vllm"))["fingerprint"] != good["fingerprint"]
