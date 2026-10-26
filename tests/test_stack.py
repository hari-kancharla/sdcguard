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

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.arange(8.0).to(torch.bfloat16), requires_grad=False)   # weights to digest


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
    a = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "weights_digest": "0" * 16, "fingerprint": "x"}
    b = {**a, "attention": "TRITON_ATTN", "fingerprint": "y"}
    c = stack.compare(a, b)
    assert c["same"] is False and c["differences"] == ["attention: 'FLASH_ATTN' qualified, 'TRITON_ATTN' now"]
    assert stack.compare(a, None)["same"] is None and stack.compare({}, b)["same"] is None
    assert stack.compare(a, {**a, "fingerprint": "z"})["same"] is False      # a field the record doesn't show


def _decision(tmp_path, decision="ACCEPT", deployed=True):
    st = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "weights_digest": "0123456789abcdef",
          "tensor_parallel": "2"}
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
    # a two-worker qualification can't be met by reading one worker (the other was never seen)
    one = ready.qualified(path, [_page(st)])
    assert one["verdict"] == "INCONCLUSIVE" and "1 workers read, the qualification ran on 2" in one["why"]


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
    # every worker's stack is recorded, and other weights under the same name and config are another stack
    cand = json.load(open(dec))["scope"]["stacks"]["candidate"]
    assert cand["deployed_workers"][0]["weights_digest"] == st_same["weights_digest"]
    q = ready.qualified(dec, [_page({**st_same, "weights_digest": "fedcba9876543210"})])
    assert q["verdict"] == "NOT THE QUALIFIED STACK" and q["differences"][0].startswith("worker 0: weights_digest")


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
    st = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "max_model_len": "32768",
          "weights_digest": "0123456789abcdef"}
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


# ---------------------------------------------------------------- the model's weights and revision are the stack too
class _Rev(str):
    """huggingface_hub's ResolvedRevision, as vLLM 0.32 keeps model_config.revision: the revision asked for as a str
    ("main"), the commit it resolved to as .resolved."""

    def __new__(cls, asked, resolved):
        s = super().__new__(cls, asked)
        s.resolved = resolved
        return s


def test_changed_weights_or_another_commit_are_another_stack():
    """An external review changed one model parameter, keeping its name and the config, and the fingerprint stayed
    the same; so it did with another resolved commit (the revision stays "main", and the config's hash drops
    _commit_hash)."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(16, 8).to(torch.bfloat16)
    r, _ = _runner()
    m = torch.nn.Sequential(_Attn(), lin)
    a = stack.collect(r, m)
    assert len(a["weights_digest"]) == 16 and stack.collect(r, m) == a       # the same weights: the same stack
    with torch.no_grad():
        lin.weight.view(-1).view(torch.int16)[5] ^= 1                         # one bit of one weight
    b = stack.collect(r, m)
    assert b["weights_digest"] != a["weights_digest"] and b["fingerprint"] != a["fingerprint"]
    c = stack.compare(a, b)
    assert c["same"] is False and any(d.startswith("weights_digest:") for d in c["differences"])
    r1, _ = _runner()
    r1.vllm_config.model_config.revision = _Rev("main", "a" * 40)
    r2, _ = _runner()
    r2.vllm_config.model_config.revision = _Rev("main", "b" * 40)
    s1, s2 = stack.collect(r1, m), stack.collect(r2, m)
    assert s1["revision"] == s2["revision"] == "main" and s1["model_commit"] == "a" * 40
    assert s1["fingerprint"] != s2["fingerprint"]
    assert any(d.startswith("model_commit:") for d in stack.compare(s1, s2)["differences"])
    hf = _HFConfig()
    hf._commit_hash = "c" * 40                                                # an older transformers keeps it here
    assert stack.collect(*_runner(hf=hf))["model_commit"] == "c" * 40
    assert "model_commit" not in stack.collect(*_runner())                    # a local checkpoint: none to read


def test_lora_is_recorded_and_its_adapters_are_said_not_covered():
    r, m = _runner()
    assert stack.collect(r, m)["lora"] == "off"
    r.vllm_config.lora_config = NS(max_loras=4, max_lora_rank=16, default_mm_loras=None)
    s = stack.collect(r, m)
    assert s["lora"].startswith("on") and "max_loras=4" in s["lora"] and "not in the weights digest" in s["lora"]
    assert s["fingerprint"] != stack.collect(*_runner())["fingerprint"]


def _decision_file(tmp_path, workers, name="d.json"):
    d = {"decision": "ACCEPT", "scope": {"stacks": {"candidate": {"deployed": workers[0],
                                                                  "deployed_workers": workers}}}}
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def _st(**kv):
    s = {"model": "m", "attention": "FLASH_ATTN", "vllm": "0.32.0", "lora": "off", **kv}
    s["fingerprint"] = stack.fingerprint(s)
    return s


def test_ready_holds_each_worker_to_its_own_weights(tmp_path):
    """With tensor parallelism each worker holds its own shard, so each has its own digest: qualify records every
    worker's stack and ready compares worker by worker."""
    w0, w1 = _st(tensor_parallel="2", weights_digest="0" * 16), _st(tensor_parallel="2", weights_digest="1" * 16)
    path = _decision_file(tmp_path, [w0, w1])
    assert ready.qualified(path, [_page(w0), _page(w1)])["verdict"] is None
    changed = _st(tensor_parallel="2", weights_digest="2" * 16)
    q = ready.qualified(path, [_page(w0), _page(changed)])
    assert q["verdict"] == "NOT THE QUALIFIED STACK" and q["differences"][0].startswith("worker 1: weights_digest")


def test_ready_needs_a_record_for_every_worker_of_the_layout(tmp_path):
    """A decision whose worker list has a gap ([worker 0, null] for tensor parallelism 2), or fewer records than the
    stack's parallelism makes, can't stand for the missing worker: INCONCLUSIVE whatever is read, never a match."""
    w0, w1 = _st(tensor_parallel="2", weights_digest="0" * 16), _st(tensor_parallel="2", weights_digest="1" * 16)
    gap = _decision_file(tmp_path, [w0, None], "gap.json")
    for pages in ([_page(w0)], [_page(w0), _page(w1)]):
        q = ready.qualified(gap, pages)
        assert q["verdict"] == "INCONCLUSIVE" and "worker 1" in q["why"], q
    short = _decision_file(tmp_path, [w0], "short.json")
    q = ready.qualified(short, [_page(w0)])
    assert q["verdict"] == "INCONCLUSIVE" and "parallelism make 2" in q["why"], q
    assert ready.qualified(_decision_file(tmp_path, [w0, w1], "full.json"), [_page(w0), _page(w1)])["verdict"] is None


def test_ready_refuses_a_decision_without_the_weights_identity_the_server_has(tmp_path):
    """A decision made before the weights' digest was recorded can't be shown to be about these weights: INCONCLUSIVE,
    never a silent match; and the same when the server doesn't publish the digest the decision records, or neither
    records one."""
    old = _st()
    live = _st(weights_digest="0" * 16, model_commit="a" * 40)
    q = ready.qualified(_decision_file(tmp_path, [old]), [_page(live)])
    assert q["verdict"] == "INCONCLUSIVE" and "weights" in q["why"] and "older sdcguard" in q["why"]
    q = ready.qualified(_decision_file(tmp_path, [live], "new.json"), [_page(old)])
    assert q["verdict"] == "INCONCLUSIVE" and "doesn't publish" in q["why"]
    q = ready.qualified(_decision_file(tmp_path, [old], "neither.json"), [_page(old)])
    assert q["verdict"] == "INCONCLUSIVE" and "neither" in q["why"]
    # a real difference elsewhere still says what differs
    q = ready.qualified(_decision_file(tmp_path, [old], "other.json"), [_page({**live, "attention": "TRITON_ATTN"})])
    assert q["verdict"] == "NOT THE QUALIFIED STACK"


def test_the_plugin_publishes_the_weights_digest_at_load_with_the_scan_on_or_off(monkeypatch, tmp_path):
    """One digest of every parameter, taken once when the model has loaded (the scan's fingerprint, on the CPU
    stand-in its PyTorch version): the same with the weight scan on or off, another when a weight differs, and left
    out, without breaking the load, when it can't be read."""
    from _vllm_standins import Decoder, install_fake_vllm, load, plugin_env

    def start(factory, **env):
        UL, GM = install_fake_vllm(monkeypatch.setitem)
        monkeypatch.setattr(GM, "__init__", lambda self, f: setattr(self, "factory", f), raising=False)
        monkeypatch.setattr(GM, "load_model", lambda self: setattr(self, "model", load(self.factory(UL))),
                            raising=False)
        vp = plugin_env(monkeypatch, tmp_path, **env)
        vp.register()
        runner = GM(factory)
        runner.load_model()
        return vp, runner

    def flipped(UL):
        d = Decoder(UL)
        with torch.no_grad():
            d.layers[2].weight.view(-1).view(torch.int16)[7] ^= 1 << 3
        return d

    vp, _ = start(Decoder)
    off = vp._STATE["stack"]["weights_digest"]
    vp, _ = start(Decoder, SDCGUARD_WEIGHT_SCAN="1")
    assert vp._STATE["stack"]["weights_digest"] == off and len(off) == 16
    vp, _ = start(flipped)
    assert vp._STATE["stack"]["weights_digest"] != off

    from sdcguard import serve_weights

    def broken(model):
        raise RuntimeError("unreadable")
    monkeypatch.setattr(serve_weights, "model_digest", broken)
    vp, runner = start(Decoder)
    assert runner.model is not None and vp._STATE["stack"] and "weights_digest" not in vp._STATE["stack"]


def test_a_weight_update_withdraws_the_digest_until_the_new_one_is_taken(monkeypatch, tmp_path):
    """After weights change in place and refresh() (or a reload) the published stack names no weights digest at
    once, so `ready --qualified` can't match the old identity with new weights; the next step outside a capture
    publishes the new weights' digest (and the old one again if the weights came back)."""
    from _vllm_standins import Decoder, install_fake_vllm, load, plugin_env
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    monkeypatch.setattr(GM, "__init__", lambda self, f: setattr(self, "factory", f), raising=False)
    monkeypatch.setattr(GM, "load_model", lambda self: setattr(self, "model", load(self.factory(UL))), raising=False)
    vp = plugin_env(monkeypatch, tmp_path)
    vp.register()
    runner = GM(Decoder)
    runner.load_model()
    first = dict(vp._STATE["stack"])
    w = runner.model.layers[2].weight
    with torch.no_grad():
        w.view(-1).view(torch.int16)[7] ^= 1 << 3
    vp.refresh()
    assert "weights_digest" not in vp._STATE["stack"] and vp._STATE["stack"]["fingerprint"] != first["fingerprint"]
    vp._take_new_weights(vp._STATE["guard"])
    new = vp._STATE["stack"]
    assert new["weights_digest"] not in (None, first["weights_digest"]) and new["fingerprint"] != first["fingerprint"]
    with torch.no_grad():
        w.view(-1).view(torch.int16)[7] ^= 1 << 3                # back as loaded
    vp.refresh()
    vp._take_new_weights(vp._STATE["guard"])
    assert vp._STATE["stack"]["weights_digest"] == first["weights_digest"]
    assert vp._STATE["stack"]["fingerprint"] == first["fingerprint"]
