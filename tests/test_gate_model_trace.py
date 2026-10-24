"""gate model-check through vLLM's Trace Replay, with a stand-in engine whose decode path can differ from its
prefill path (a fault only decode shows)."""
import sys
import types

import torch

from sdcguard import gate_model as gm
from test_gate_model import _table, _top, install_fake_vllm

V = 50


def install_trace_vllm(monkeypatch, table, decode_table=None, drop=False):
    install_fake_vllm(monkeypatch, table)
    vllm = sys.modules["vllm"]
    base = vllm.LLM
    dec = decode_table if decode_table is not None else table

    class SamplingParams:
        def __init__(self, temperature=0.0, max_tokens=16, logprobs=None, prompt_logprobs=None, ignore_eos=False,
                     trace_decode_token_ids=None):
            self.max_tokens, self.logprobs, self.prompt_logprobs = max_tokens, logprobs, prompt_logprobs
            self.trace = trace_decode_token_ids

    class LLM(base):
        def generate(self, prompts, sps, use_tqdm=False):
            if not isinstance(sps, list):
                return base.generate(self, prompts, sps, use_tqdm)
            assert self.kw.get("enable_trace_replay") is True
            outs = []
            for p, sp in zip(prompts, sps):
                ids = list(p["prompt_token_ids"])
                prev, lps, toks = ids[-1], [], []
                for t in sp.trace:
                    lps.append(_top(dec[prev], t))
                    toks.append(t)
                    prev = t
                if drop:
                    toks = toks[:-1]
                outs.append(types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=toks, logprobs=lps)]))
            return outs
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams


def test_trace_replay_scores_the_decode_path(monkeypatch, tmp_path):
    t = _table()
    install_trace_vllm(monkeypatch, t)
    ref = str(tmp_path / "ref.json")
    gm.make_ref("fake/model", ref, tokens=16)
    same = gm.check_trace(ref, str(tmp_path / "same.json"))
    assert same["verdict"]["verdict"] == "PASS" and same["summary"]["mean_kl"] < 1e-12
    assert same["stack"]["via"] == "trace"
    # a fault only the decode path shows: the prefill check passes it, the trace check doesn't
    bad = t.clone()
    bad[:, 0] += 6.0
    bad = torch.log_softmax(bad, dim=1)
    install_trace_vllm(monkeypatch, t, decode_table=bad)
    pre = gm.check(ref, str(tmp_path / "prefill.json"))
    dec = gm.check_trace(ref, str(tmp_path / "decode.json"))
    assert pre["verdict"]["verdict"] == "PASS" and dec["verdict"]["verdict"] == "FAIL"


def test_a_trace_that_wasnt_what_ran_is_not_scored(monkeypatch, tmp_path):
    install_trace_vllm(monkeypatch, _table())
    ref = str(tmp_path / "ref.json")
    gm.make_ref("fake/model", ref, tokens=8)
    install_trace_vllm(monkeypatch, _table(), drop=True)
    r = gm.check_trace(ref, str(tmp_path / "x.json"))
    assert r["verdict"]["verdict"] == "INCONCLUSIVE"
