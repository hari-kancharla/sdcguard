"""The output check of the upgrade gate (gate_model): its metrics, its verdicts and its alignment of positions, with a
stand-in for vLLM whose next-token distributions depend on the previous token (so scoring a position against the
wrong recorded distribution shows up)."""
import json
import math
import sys
import types

import pytest
import torch

from sdcguard import gate_model as gm

V = 50


class _LP:
    def __init__(self, lp):
        self.logprob = lp


def _table(noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(V, V, generator=g) * 3
    if noise:
        logits = logits + noise * torch.randn(V, V, generator=torch.Generator().manual_seed(seed + 1))
    return torch.log_softmax(logits.double(), dim=1)


def _top(row, also=None):
    idx = torch.topk(row, gm.TOP).indices.tolist()
    d = {i: _LP(float(row[i])) for i in idx}
    if also is not None:
        d[also] = _LP(float(row[also]))
    return d


def install_fake_vllm(monkeypatch, table):
    vllm = types.ModuleType("vllm")
    inputs = types.ModuleType("vllm.inputs")

    class SamplingParams:
        def __init__(self, temperature=0.0, max_tokens=16, logprobs=None, prompt_logprobs=None, ignore_eos=False):
            self.max_tokens, self.logprobs, self.prompt_logprobs = max_tokens, logprobs, prompt_logprobs

    class Tok:
        chat_template = None

    class LLM:
        def __init__(self, model, **kw):
            self.model, self.kw = model, kw

        def get_tokenizer(self):
            return Tok()

        def generate(self, prompts, sp, use_tqdm=False):
            outs = []
            for p in prompts:
                ids = list(p["prompt_token_ids"]) if isinstance(p, dict) else [ord(c) % V for c in p[:24]]
                comp, lps = [], []
                prev = ids[-1]
                for _ in range(sp.max_tokens):
                    row = table[prev]
                    nxt = int(torch.argmax(row))
                    comp.append(nxt)
                    if sp.logprobs:
                        lps.append(_top(row, nxt))
                    prev = nxt
                plp = None
                if sp.prompt_logprobs:
                    plp = [None] + [_top(table[ids[i - 1]], ids[i]) for i in range(1, len(ids))]
                outs.append(types.SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=plp, outputs=[
                    types.SimpleNamespace(token_ids=comp, logprobs=lps)]))
            return outs

    def TokensPrompt(prompt_token_ids):
        return {"prompt_token_ids": prompt_token_ids}

    vllm.LLM, vllm.SamplingParams, vllm.__version__ = LLM, SamplingParams, "0.31.0-fake"
    inputs.TokensPrompt = TokensPrompt
    vllm.inputs = inputs
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def test_identical_distributions_agree_exactly():
    row = _table()[3]
    m = gm.position_metrics(_top(row), _top(row, 7), int(torch.argmax(row)))
    assert m["top1"] and m["kl"] == pytest.approx(0.0, abs=1e-12) and m["dlogprob"] == 0.0


def test_kl_grows_with_the_difference_and_a_flipped_top_token_is_counted():
    a, b = _table()[3], _table(noise=0.5)[3]
    tok = int(torch.argmax(a))
    m = gm.position_metrics(_top(a), _top(b, tok), tok)
    assert m["kl"] > 1e-3
    assert m["top1"] == (int(torch.argmax(b)) == tok)


def test_verdicts_follow_the_thresholds_and_the_noise_floor():
    base = {"positions": 1000, "prompts": 32, "top1_agreement": 1.0, "max_prompt_mean_kl": 1e-4}
    assert gm.verdict({**base, "mean_kl": 5e-4})["verdict"] == "PASS"
    assert gm.verdict({**base, "mean_kl": 3e-3})["verdict"] == "WARN"
    assert gm.verdict({**base, "mean_kl": 2e-2, "max_prompt_mean_kl": 0.2})["verdict"] == "FAIL"
    assert gm.verdict({**base, "mean_kl": 5e-4, "top1_agreement": 0.95})["verdict"] == "FAIL"
    # a noisy stack (same-stack KL of 6e-4) raises the PASS line to 1.8e-3
    assert gm.verdict({**base, "mean_kl": 1.5e-3}, {"positions": 1000, "mean_kl": 6e-4})["verdict"] == "PASS"
    assert gm.verdict({"positions": 0})["verdict"] == "INCONCLUSIVE"


def test_same_stack_passes_and_a_changed_model_fails_end_to_end(monkeypatch, tmp_path):
    """The candidate scores prompt + recorded completion; position p_len + t must be compared with the recorded
    step t. Any off-by-one would compare distributions conditioned on different tokens and fail the same stack."""
    install_fake_vllm(monkeypatch, _table())
    ref = str(tmp_path / "ref.json")
    r = gm.make_ref("fake/model", ref, tokens=24)
    assert r["prompts"] == len(gm.PROMPTS) and r["positions"] == 24 * len(gm.PROMPTS)
    same = gm.check(ref, str(tmp_path / "same.json"))
    assert same["verdict"]["verdict"] == "PASS", same
    assert same["summary"]["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    assert same["summary"]["top1_agreement"] == 1.0
    install_fake_vllm(monkeypatch, _table(noise=1.0))
    broken = gm.check(ref, str(tmp_path / "broken.json"), noise_path=str(tmp_path / "same.json"))
    assert broken["verdict"]["verdict"] == "FAIL", broken["summary"]
    saved = json.load(open(tmp_path / "broken.json"))
    assert saved["schema"] == gm.CHECK_SCHEMA and saved["noise"]["mean_kl"] == same["summary"]["mean_kl"]


def test_a_small_change_warns_rather_than_fails(monkeypatch, tmp_path):
    install_fake_vllm(monkeypatch, _table())
    ref = str(tmp_path / "ref.json")
    gm.make_ref("fake/model", ref, tokens=24)
    install_fake_vllm(monkeypatch, _table(noise=0.06))
    run = gm.check(ref, str(tmp_path / "small.json"))
    s = run["summary"]
    assert PASS_OR_WARN(run["verdict"]["verdict"]) and 0 < s["mean_kl"] < gm.WARN_KL, s


def PASS_OR_WARN(v):
    return v in ("PASS", "WARN")


def test_the_command_line(monkeypatch, tmp_path, capsys):
    install_fake_vllm(monkeypatch, _table())
    from sdcguard import cli
    ref = str(tmp_path / "ref.json")
    assert cli.main(["gate", "model-ref", "--model", "fake/model", "--out", ref, "--tokens", "8"]) == 0
    assert cli.main(["gate", "model-check", ref, "--out", str(tmp_path / "c.json")]) == 0
    assert '"PASS"' in capsys.readouterr().out
    assert math.isfinite(json.load(open(tmp_path / "c.json"))["summary"]["mean_kl"])
