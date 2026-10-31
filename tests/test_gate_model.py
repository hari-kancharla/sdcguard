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
    cov = {"expected_positions": 1000, "missing_positions": 0, "missing_prompts": 0, "extra_results": 0}
    base = {"positions": 1000, "prompts": 32, "top1_agreement": 1.0, "max_prompt_mean_kl": 1e-4, **cov}
    assert gm.verdict({**base, "mean_kl": 5e-4})["verdict"] == "PASS"
    assert gm.verdict({**base, "mean_kl": 3e-3})["verdict"] == "WARN"
    assert gm.verdict({**base, "mean_kl": 2e-2, "max_prompt_mean_kl": 0.2})["verdict"] == "FAIL"
    assert gm.verdict({**base, "mean_kl": 5e-4, "top1_agreement": 0.95})["verdict"] == "FAIL"
    # a noisy stack (same-stack KL of 6e-4) raises the PASS line to 1.8e-3
    assert gm.verdict({**base, "mean_kl": 1.5e-3}, {"positions": 1000, "mean_kl": 6e-4, **cov})["verdict"] == "PASS"
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


# ---------------------------------------------------------------- a verdict only on complete evidence
def _ref(prompts=2, positions=2):
    """A reference of `prompts` prompts with `positions` recorded positions each (two-token prompts)."""
    rows = _table()
    recs = []
    for p in range(prompts):
        prompt, comp = [p, p + 1], []
        prev = prompt[-1]
        for _ in range(positions):
            prev = int(torch.argmax(rows[prev]))
            comp.append(prev)
        lps, prev = [], prompt[-1]
        for tok in comp:
            lps.append({str(k): v.logprob for k, v in _top(rows[prev], tok).items()})
            prev = tok
        recs.append({"prompt_ids": prompt, "completion_ids": comp, "logprobs": lps})
    return {"records": recs}


def _cand(ref, keep=None):
    """The same stack's prompt_logprobs for each record; keep: how many completion positions to return."""
    rows, out = _table(), []
    for r in ref["records"]:
        ids = r["prompt_ids"] + r["completion_ids"]
        plp = [None] + [_top(rows[ids[i - 1]], ids[i]) for i in range(1, len(ids))]
        out.append(plp if keep is None else plp[:len(r["prompt_ids"]) + keep])
    return out


def test_a_candidate_missing_prompts_or_positions_is_inconclusive_not_pass():
    """An outside review's case: a reference of two prompts and four positions, a candidate with one position. It
    used to PASS on the one position it had."""
    ref = _ref(prompts=2, positions=2)
    full = gm.compare(ref, _cand(ref))["summary"]
    assert full["positions"] == full["expected_positions"] == 4 and full["missing_positions"] == 0
    assert gm.verdict(full, min_positions=1)["verdict"] == "PASS"
    assert "a verdict needs at least" in gm.verdict(full)["why"]       # four positions: too few to judge
    one = gm.compare(ref, _cand(ref, keep=1)[:1])["summary"]
    assert one["positions"] == 1 and one["missing_positions"] == 3 and one["missing_prompts"] == 1
    v = gm.verdict(one)
    assert v["verdict"] == "INCONCLUSIVE" and "scored 1 of 4" in v["why"]
    gaps = _cand(ref)
    gaps[1][len(ref["records"][1]["prompt_ids"])] = None          # one position without a distribution
    assert gm.verdict(gm.compare(ref, gaps)["summary"])["verdict"] == "INCONCLUSIVE"
    extra = gm.compare(ref, _cand(ref) + [None])["summary"]
    assert extra["extra_results"] == 1 and gm.verdict(extra)["verdict"] == "INCONCLUSIVE"


def test_a_summary_without_its_coverage_counts_or_an_incomplete_noise_run_is_inconclusive():
    ref = _ref()
    s = gm.compare(ref, _cand(ref))["summary"]
    bare = {k: v for k, v in s.items() if k not in ("expected_positions", "missing_positions")}
    assert gm.verdict(bare)["verdict"] == "INCONCLUSIVE"
    part = gm.compare(ref, _cand(ref, keep=1))["summary"]
    v = gm.verdict(s, part, min_positions=1)
    assert v["verdict"] == "INCONCLUSIVE" and "--noise run can't set the line" in v["why"]
    assert gm.verdict(s, {"positions": 0}, min_positions=1)["verdict"] == "INCONCLUSIVE"
    # a same-stack KL that puts the pass line at the warn line: a result within that noise can't be told from it
    loud = {**s, "mean_kl": gm.WARN_KL / 3}
    v = gm.verdict(s, loud, min_positions=1)
    assert v["verdict"] == "INCONCLUSIVE" and "too noisy for a pass" in v["why"]


def test_a_noise_run_of_another_reference_or_model_is_inconclusive(monkeypatch, tmp_path):
    install_fake_vllm(monkeypatch, _table())
    ref, other = str(tmp_path / "ref.json"), str(tmp_path / "other.json")
    gm.make_ref("fake/model", ref, tokens=8)
    gm.make_ref("fake/model", other, tokens=6)
    gm.check(other, str(tmp_path / "noise_other.json"))
    run = gm.check(ref, str(tmp_path / "c.json"), noise_path=str(tmp_path / "noise_other.json"))
    assert run["verdict"]["verdict"] == "INCONCLUSIVE" and "another reference" in run["verdict"]["why"]
    gm.check(ref, str(tmp_path / "noise_model.json"), model="fake/quantized")
    run = gm.check(ref, str(tmp_path / "c2.json"), noise_path=str(tmp_path / "noise_model.json"))
    assert run["verdict"]["verdict"] == "INCONCLUSIVE" and "another model" in run["verdict"]["why"]
    gm.check(ref, str(tmp_path / "same.json"), extra={"max_num_seqs": 4})     # other batching is what noise is
    run = gm.check(ref, str(tmp_path / "c3.json"), noise_path=str(tmp_path / "same.json"))
    assert run["verdict"]["verdict"] == "PASS" and run["ref_digest"] == gm.ref_digest(json.load(open(ref)))


def test_a_stack_that_drops_a_prompt_is_inconclusive_and_exits_2(monkeypatch, tmp_path, capsys):
    install_fake_vllm(monkeypatch, _table())
    from sdcguard import cli
    ref = str(tmp_path / "ref.json")
    assert cli.main(["gate", "model-ref", "--model", "fake/model", "--out", ref, "--tokens", "8"]) == 0
    real = sys.modules["vllm"].LLM.generate

    def drops_one(self, prompts, sp, use_tqdm=False):
        outs = real(self, prompts, sp, use_tqdm)
        outs[3].prompt_logprobs = None
        return outs
    monkeypatch.setattr(sys.modules["vllm"].LLM, "generate", drops_one)
    assert cli.main(["gate", "model-check", ref, "--out", str(tmp_path / "c.json")]) == 2
    assert '"INCONCLUSIVE"' in capsys.readouterr().out


def test_nan_or_inf_from_the_new_stack_never_passes():
    """An outside review's case: NaN log-probabilities scored KL 0 (max(0.0, nan) is 0.0) and top-1 agreement 100%
    (max with a NaN key returns the first key), so a stack returning NaN passed; a -inf crashed the comparison."""
    ref = _ref(prompts=2, positions=3)
    cand = _cand(ref)
    for plp in cand:
        for pos in plp[1:]:
            for v in pos.values():
                v.logprob = float("nan")
    s = gm.compare(ref, cand)["summary"]
    assert s["nonfinite_positions"] == s["positions"] == 6
    v = gm.verdict(s)
    assert v["verdict"] == "FAIL" and "weren't a distribution" in v["why"]
    one = _cand(ref)
    p_len = len(ref["records"][0]["prompt_ids"])
    first = next(iter(one[0][p_len].values()))
    first.logprob = float("nan")                                 # one NaN in one list
    assert gm.verdict(gm.compare(ref, one)["summary"])["verdict"] == "FAIL"
    ninf = _cand(ref)
    tok = ref["records"][0]["completion_ids"][0]
    ninf[0][p_len][tok].logprob = float("-inf")                  # the recorded token given probability 0
    m = gm.compare(ref, ninf)
    assert m["per_prompt"][0][0]["kl"] == float("inf") and gm.verdict(m["summary"])["verdict"] == "FAIL"


def test_a_reference_with_nan_is_not_scored_so_the_check_is_inconclusive():
    ref = _ref(prompts=2, positions=2)
    ref["records"][1]["logprobs"][0] = {k: float("nan") for k in ref["records"][1]["logprobs"][0]}
    s = gm.compare(ref, _cand(ref))["summary"]
    assert s["missing_positions"] == 1 and gm.verdict(s)["verdict"] == "INCONCLUSIVE"


def test_a_noise_run_with_nan_or_inf_cant_set_the_line():
    ref = _ref()
    s = gm.compare(ref, _cand(ref))["summary"]
    for bad in (float("nan"), float("inf")):
        v = gm.verdict(s, {**s, "mean_kl": bad}, min_positions=1)
        assert v["verdict"] == "INCONCLUSIVE", v


# ---------------------------------------------------------------- the reference on the candidate's path; calibrated lines
def test_the_reference_is_scored_on_the_prefill_path_so_a_repeat_measures_noise_not_the_path_gap(monkeypatch, tmp_path):
    """An outside review's case: the reference was the decode path's and the candidate the prefill path's, so the
    same stack started again scored its own decode-vs-prefill gap (1.8e-3 on an L4, over the 1e-3 pass line). The
    stand-in here gives the decode path a slightly different distribution than the prefill path."""
    table, decode_table = _table(), _table(noise=0.15, seed=0)
    install_fake_vllm(monkeypatch, table)
    real = sys.modules["vllm"].LLM.generate

    def decode_differs(self, prompts, sp, use_tqdm=False):
        outs = real(self, prompts, sp, use_tqdm)
        if sp.logprobs:                                    # the decode path: its own (slightly different) numbers
            for o in outs:
                prev = o.prompt_token_ids[-1]
                lps = []
                for tok in o.outputs[0].token_ids:
                    lps.append(_top(decode_table[prev], tok))
                    prev = tok
                o.outputs[0].logprobs = lps
        return outs
    monkeypatch.setattr(sys.modules["vllm"].LLM, "generate", decode_differs)
    ref = str(tmp_path / "ref.json")
    gm.make_ref("fake/model", ref, tokens=16)
    saved = json.load(open(ref))
    assert saved["logprobs_path"] == "prefill" and "decode_logprobs" in saved["records"][0]
    same = gm.check(ref, str(tmp_path / "same.json"))
    assert same["summary"]["same_path"] and same["summary"]["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    assert same["verdict"]["verdict"] == "PASS"
    old = {**saved, "records": [{**r, "logprobs": r["decode_logprobs"]} for r in saved["records"]]}
    del old["logprobs_path"]                               # a reference from before: decode path only
    s = gm.compare(old, [o.prompt_logprobs for o in sys.modules["vllm"].LLM(model="fake/model").generate(
        [{"prompt_token_ids": r["prompt_ids"] + r["completion_ids"]} for r in old["records"]],
        sys.modules["vllm"].SamplingParams(max_tokens=1, prompt_logprobs=gm.TOP))])["summary"]
    assert s["ref_path"] == "decode" and s["candidate_path"] == "prefill" and not s["same_path"]
    assert s["mean_kl"] > 0                                # the path gap the old reference measured


def _summary(kl, top1, worst=1e-3, n=2048, prompts=32):
    cov = {"expected_positions": n, "missing_positions": 0, "missing_prompts": 0, "extra_results": 0}
    return {"positions": n, "prompts": prompts, "mean_kl": kl, "top1_agreement": top1, "max_prompt_mean_kl": worst,
            "max_kl": 1.0, **cov}


def test_a_noisy_repeat_no_longer_hides_a_clearly_broken_candidate():
    """An outside review's case: with the noise run at KL 2.4e-3 every verdict was INCONCLUSIVE, even KL 5e-2 with
    90% top-1 agreement. Now a result past the warn lines and the noise fails; one inside the noise stays unjudged."""
    noise = _summary(2.4e-3, 0.995)
    assert gm.verdict(_summary(5e-2, 0.90), noise)["verdict"] == "FAIL"
    v = gm.verdict(_summary(2.0e-3, 0.995), noise)
    assert v["verdict"] == "INCONCLUSIVE" and "too noisy" in v["why"]


def test_the_top1_line_moves_with_the_repeat_and_needs_a_real_excess():
    """The 99% line was fixed: at 2,048 positions a stack whose own repeat disagreed on 1% of positions failed it
    by chance about a third of the time. With a repeat, the candidate fails only when it disagrees significantly more
    than the repeat did."""
    noise = _summary(3e-4, 0.988)
    assert gm.verdict(_summary(4e-4, 0.986), noise)["verdict"] == "PASS"       # as noisy as its own repeat
    assert gm.verdict(_summary(4e-4, 0.986))["verdict"] == "WARN"              # without one: the fixed 99% line
    assert gm.verdict(_summary(4e-4, 0.95), noise)["verdict"] == "FAIL"        # far more disagreement than the repeat
    assert gm.verdict(_summary(4e-4, float("nan")), noise)["verdict"] == "FAIL"
    assert gm._excess_z(0.05, 2048, 0.012, 2048) > gm.Z_EXCESS > gm._excess_z(0.014, 2048, 0.012, 2048)


def test_the_worst_prompt_line_moves_with_the_repeat():
    noise = _summary(3e-4, 0.995, worst=0.04)
    assert gm.verdict(_summary(4e-4, 0.995, worst=0.06), noise)["verdict"] == "PASS"   # 0.06 < 3 x 0.04
    assert gm.verdict(_summary(4e-4, 0.995, worst=0.06))["verdict"] == "WARN"


def test_too_few_positions_get_no_verdict_but_an_impossible_token_still_fails():
    small = _summary(5e-2, 0.5, n=24)
    v = gm.verdict(small)
    assert v["verdict"] == "INCONCLUSIVE" and "a verdict needs at least" in v["why"]
    assert gm.verdict({**small, "max_kl": float("inf")})["verdict"] == "FAIL"
