"""The attention check (serve_attn.py): its bound against a model of how vLLM's backends round, which outputs it
samples, what it catches and locates, the KV-write check, CUDA-graph padding, and the checks' flow (CPU, the PyTorch
check; tests/_serve_attn_compare.py runs the Triton kernels in Triton's interpreter against it)."""
import os
import re
import subprocess
import sys
import types

import pytest
import torch

from _attn_emul import VARIANTS, emulate
from sdcguard import serve_attn as sa
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def keys_of(c, r, hk, pos, lo=0):
    """Keys and values of request r's positions lo..pos for KV head hk, through the block table."""
    N = c["kv"].shape[2]
    D = c["q"].shape[-1]
    rows = torch.stack([c["kv"][int(c["bt"][r, t // N]), hk, t % N] for t in range(lo, pos + 1)])
    return rows[:, :D], rows[:, D:]


def run_steps(c, steps, fn=sa.attn_check_torch, **kw):
    """The state after each of `steps`, each from fresh buffers."""
    out = []
    for step in steps:
        b = sa.buffers("cpu", 1, c["q"].shape[-1], 512, c["q"].dtype)
        sa.run_case(fn, c, b, step=step, **kw)
        out.append(b)
    return out


def samples_of(c, step, idx=0, nl=1):
    qsl, sls = c["qsl"].tolist(), c["seq_lens"].tolist()
    T, Hq = c["q"].shape[0], c["q"].shape[1]
    return [sa.sample_python(step, idx, nl, j, qsl, sls, T, Hq) for j in range(sa.SAMPLES)]


# ------------------------------------------------------------------------------------------------- the bound
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_outputs_of_backends_that_round_like_vllms_score_at_most_one(dtype):
    """FlashAttention- and Triton-like rounding (P rounded to the dtype, fp32 accumulation, tiles, split-KV, partials
    rounded as in FlashAttention's cascade path, the sum of P before or after rounding) on sharp and diffuse
    attention, values of one sign, cancelling, large and tiny, up to 3000 keys and fp16 P values in its subnormal
    range: every output within the bound (score at most 1; the line is 4)."""
    g = torch.Generator().manual_seed(1)
    worst = 0.0
    for D in (64, 128):
        for L in (1, 9, 300, 3000):
            for qs in (0.02, 1.0, 5.0):
                for vm in ("pos", "cancel", "big", "tiny"):
                    K = torch.randn(L, D, generator=g)
                    q = torch.randn(D, generator=g) * qs
                    V = {"pos": torch.rand(L, D, generator=g) + 0.5, "cancel": torch.randn(L, D, generator=g),
                         "big": torch.randn(L, D, generator=g) * 300 + 1000,
                         "tiny": torch.randn(L, D, generator=g) * 1e-3}[vm]
                    K, V, q = K.to(dtype), V.to(dtype), q.to(dtype)
                    for var in VARIANTS[:3] if L == 3000 else VARIANTS:
                        o = emulate(q, K, V, D ** -0.5, dtype, **var)
                        worst = max(worst, sa.recheck(q.float(), o.float(), K, V, D ** -0.5, dtype)["worst_score"])
    # scores spread over many orders of magnitude: most P values far below the dtype's smallest normal
    K = (torch.randn(4000, 64, generator=g) * torch.linspace(0.1, 4, 4000)[:, None]).to(dtype)
    V = torch.randn(4000, 64, generator=g).to(dtype)
    q = (torch.randn(64, generator=g) * 3).to(dtype)
    for var in VARIANTS:
        o = emulate(q, K, V, 0.125, dtype, **var)
        worst = max(worst, sa.recheck(q.float(), o.float(), K, V, 0.125, dtype)["worst_score"])
    assert 0.0 < worst <= 1.0, worst


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_the_gpu_checks_score_is_the_float64_recheck_score(dtype):
    """The check (fp32 recompute) and its float64 recheck agree on the score of every sampled output, and the state
    keeps the worst one with its row, head, position and step."""
    c = sa.make_case("cpu", dtype, D=64, Hq=4, Hkv=2, lens=((300, 1), (40, 12), (9, 9), (0, 0)), NB=40, seed=3)
    c["o"] = c["o"].float().mul(1 + 2 ** -9).to(dtype)       # off by a little everywhere: scores well above 0
    for step, b in zip(range(10), run_steps(c, range(10))):
        st = b["state"][0].tolist()
        best = None
        for s in samples_of(c, step):
            K, V = keys_of(c, s["request"], s["head"] // 2, s["position"])
            z = sa.recheck(c["q"][s["row"], s["head"]].float(), c["o"][s["row"], s["head"]].float(), K, V,
                           c["scale"], dtype)["worst_score"]
            if best is None or z > best[0]:
                best = (z, s)
        z = sa._f32(st[sa.C_ZT] >> 32)
        assert abs(z - best[0]) <= 1e-3 * best[0] + 1e-4, (step, z, best)
        assert st[sa.C_ZT] & sa.M32 == best[1]["row"] and st[sa.C_ZH] & sa.M32 == best[1]["head"]
        assert st[sa.C_ZP] & sa.M32 == best[1]["position"] and st[sa.C_ZS] & sa.M32 == step
        assert st[sa.C_SAMPLES] == sa.SAMPLES and st[sa.C_CALLS] == 1
        assert st[sa.C_KEYS] == sum(s["position"] + 1 for s in samples_of(c, step))


# ----------------------------------------------------------------------------------------------- the samples
def test_samples_are_tokens_and_newest_tokens_of_requests():
    """Even samples draw a token of the step, odd ones the newest token of a request (a decode's only token): over
    steps every request's newest token and every head is drawn, and padding never is."""
    c = sa.make_case("cpu", torch.bfloat16, lens=((50, 1), (20, 1), (64, 30), (5, 5), (0, 0), (0, 0)), NB=24,
                     padded_rows=7)
    qsl = c["qsl"].tolist()
    newest = {qsl[r + 1] - 1 for r in range(4)}
    seen_rows, seen_heads, odd_rows = set(), set(), set()
    for step in range(200):
        for j, s in enumerate(samples_of(c, step)):
            assert s is not None and 0 <= s["row"] < qsl[-1] and s["request"] < 4
            seen_rows.add(s["row"])
            seen_heads.add(s["head"])
            if j % 2:
                odd_rows.add(s["row"])
    assert odd_rows == newest and seen_heads == set(range(4)) and len(seen_rows) > 30


def test_each_layer_checks_every_token_and_head_of_a_steady_batch_in_turn():
    """One number picks a sample's token (or request) and head together, from the layer's own turn count: with 28
    layers taking turns and 28 heads, a steady batch of 8 decodes has every (token, head) pair of every layer
    checked within 8 x 28 of its turns, by even samples and by odd ones. (Token and head drawn apart from the step
    number would lock each layer into a few pairs: 28 turns share every factor of 28 heads.)"""
    qsl, sls = list(range(9)), [100 + r for r in range(8)]
    for idx in (0, 5, 27):
        for j in (0, 1):
            seen = set()
            for k in range(8 * 28):
                s = sa.sample_python(k * 28 + idx, idx, 28, j, qsl, sls, 8, 28)
                seen.add((s["row"], s["head"]))
            assert len(seen) == 8 * 28, (idx, j, len(seen))


def test_one_layer_a_step_and_only_in_model_steps():
    c = sa.make_case("cpu", torch.bfloat16)
    for step in range(9):
        for idx in range(3):
            (b,) = run_steps(c, [step], nl=3, idx=idx, slot=0)
            assert int(b["state"][0, sa.C_CALLS]) == (step % 3 == idx)
    (b,) = run_steps(c, [4], active=False)
    assert int(b["state"].abs().sum()) == 0


def test_cuda_graph_padding_is_never_judged():
    """A replayed graph's padding: requests with no query and seq_len 0, rows past the real tokens with garbage
    queries and outputs. None of it is sampled; with no real token at all every sample is counted as without one."""
    c = sa.make_case("cpu", torch.bfloat16, lens=((30, 1), (12, 4), (0, 0), (0, 0)), padded_rows=5)
    n = int(c["qsl"][-1])
    c["o"][n:] = float("nan")
    c["q"][n:] = 1e4
    for b in run_steps(c, range(20)):
        st = b["state"][0].tolist()
        assert st[sa.C_SAMPLES] == sa.SAMPLES and st[sa.C_SKIP] == 0 and sa._f32(st[sa.C_ZT] >> 32) <= 1.0
    empty = dict(c, qsl=torch.zeros_like(c["qsl"]), seq_lens=torch.zeros_like(c["seq_lens"]))
    for b in run_steps(empty, range(3)):
        st = b["state"][0].tolist()
        assert st[sa.C_SKIP] == sa.SAMPLES and st[sa.C_SAMPLES] == 0 and st[sa.C_ZT] == 0


# ------------------------------------------------------------------------------------------------ what it catches
def _first_step_sampling(c, row, head, start=0):
    for step in range(start, start + 5000):
        if any(s and s["row"] == row and s["head"] == head for s in samples_of(c, step)):
            return step
    raise AssertionError("never sampled")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_wrong_output_is_caught_located_and_kept_when_sampled(dtype):
    c = sa.make_case("cpu", dtype, lens=((90, 1), (33, 7), (0, 0)), NB=20, seed=5)
    row, head, dim = 4, 3, 17
    bits = c["o"][row, head, dim].view(torch.int16) ^ (1 << 13)          # an exponent bit
    c["o"][row, head, dim] = bits.view(dtype)
    hit = _first_step_sampling(c, row, head)
    for step in range(hit - 3, hit + 1):
        (b,) = run_steps(c, [step])
        st = b["state"][0].tolist()
        z = sa._f32(st[sa.C_ZT] >> 32)
        sampled = any(s and s["row"] == row and s["head"] == head for s in samples_of(c, step))
        assert (z > sa.ATTN_Z) == sampled, (step, z)
        if sampled:
            assert (st[sa.C_ZT] & sa.M32, st[sa.C_ZH] & sa.M32, st[sa.C_ZD] & sa.M32) == (row, head, dim)
            m = b["capm"][0].tolist()
            assert m[sa.CM_CLAIM] == 1 and (m[sa.CM_ROW], m[sa.CM_HEAD], m[sa.CM_DIM]) == (row, head, dim)
            assert torch.equal(b["capo"][0, :64], c["o"][row, head].float())
            assert torch.equal(b["capq"][0, :64], c["q"][row, head].float())


def _backend_with(c, fn):
    """Outputs of a backend that attends what fn(request, position) returns (keys, values) instead."""
    o = c["o"].clone()
    D = c["q"].shape[-1]
    Hq, Hkv = c["q"].shape[1], c["kv"].shape[1]
    qsl, sls = c["qsl"].tolist(), c["seq_lens"].tolist()
    for r in range(len(sls)):
        ql = qsl[r + 1] - qsl[r]
        for jj in range(ql):
            i, pos = qsl[r] + jj, sls[r] - ql + jj
            for h in range(Hq):
                K, V = fn(r, pos, h // (Hq // Hkv))
                s = (K.double() @ c["q"][i, h].double()) * c["scale"]
                o[i, h] = (torch.softmax(s, 0) @ V.double()).to(o.dtype)
    return o


def test_attention_over_other_keys_than_the_metadata_names_is_caught():
    """A kernel that reads the wrong request's blocks, or one key short with attention on the newest key (as a
    stale length or split schedule would): caught at every sampled output it changed by more than the bound."""
    c = sa.make_case("cpu", torch.bfloat16, D=64, Hq=2, Hkv=1, lens=((40, 1), (40, 1), (0, 0)), NB=12, seed=7)
    other = dict(c, o=_backend_with(c, lambda r, pos, hk: keys_of(c, 1 - r, hk, pos)))
    caught = [sa._f32(int(b["state"][0, sa.C_ZT]) >> 32) > sa.ATTN_Z for b in run_steps(other, range(30))]
    assert all(caught)
    # sharp attention on the newest key, which the wrong kernel leaves out
    sharp = dict(c, q=c["q"].clone(), kv=c["kv"].clone())
    for r in range(2):
        pos = int(c["seq_lens"][r]) - 1
        blk, off = int(c["bt"][r, pos // 16]), pos % 16
        sharp["kv"][blk, 0, off, :64] = ((sharp["q"][r, 0].float() + sharp["q"][r, 1].float()) * 4).to(torch.bfloat16)
    sharp["o"] = _backend_with(sharp, lambda r, pos, hk: keys_of(sharp, r, hk, pos))
    short = dict(sharp, o=_backend_with(sharp, lambda r, pos, hk: keys_of(sharp, r, hk, pos - 1)))
    assert all(sa._f32(int(b["state"][0, sa.C_ZT]) >> 32) <= 1.0 for b in run_steps(sharp, range(10)))
    zs = [sa._f32(int(b["state"][0, sa.C_ZT]) >> 32) for b in run_steps(short, range(10))]
    assert all(z > sa.ATTN_Z for z in zs), zs


def test_non_finite_outputs_alarm_and_non_finite_keys_are_counted_not_judged():
    c = sa.make_case("cpu", torch.bfloat16, lens=((20, 1), (0, 0)), padded_rows=0)
    nan_out = dict(c, o=c["o"].clone())
    nan_out["o"][0] = float("nan")
    assert all(sa._f32(int(b["state"][0, sa.C_ZT]) >> 32) == float("inf") for b in run_steps(nan_out, range(4)))
    nan_kv = dict(c, kv=c["kv"].clone())
    nan_kv["kv"][int(c["bt"][0, 0])] = float("nan")
    for b in run_steps(nan_kv, range(4), has_write=False):
        st = b["state"][0].tolist()
        assert st[sa.C_UNJ] == sa.SAMPLES and st[sa.C_SAMPLES] == 0 and st[sa.C_ZT] == 0


def test_block_ids_outside_the_cache_are_counted_and_never_read():
    c = sa.make_case("cpu", torch.bfloat16, lens=((40, 1), (0, 0)), padded_rows=0)
    for bad in (-5, 10 ** 6):
        cb = dict(c, bt=c["bt"].clone())
        cb["bt"][0, 1] = bad
        for b in run_steps(cb, range(3), has_write=False):
            st = b["state"][0].tolist()
            assert st[sa.C_BADBLK] == sa.SAMPLES and st[sa.C_SAMPLES] == 0


# -------------------------------------------------------------------------------------------- the KV writes
def test_the_sampled_tokens_kv_writes_are_checked():
    c = sa.make_case("cpu", torch.bfloat16, lens=((40, 1), (17, 5), (0, 0)), NB=12)
    for b in run_steps(c, range(6)):
        st = b["state"][0].tolist()
        assert st[sa.C_WN] == sa.SAMPLES and st[sa.C_WBAD] == 0 and st[sa.C_WSLOT] == 0
    # the cache doesn't hold what the layer computed (a write that didn't happen, or was corrupted)
    wrong = dict(c, key=c["key"].clone(), val=c["val"].clone())
    wrong["val"][:, 1, 9] = wrong["val"][:, 1, 9].float() + 1
    for step, b in zip(range(6), run_steps(wrong, range(6))):
        st = b["state"][0].tolist()
        assert st[sa.C_WBAD] == sa.SAMPLES and st[sa.C_WSLOT] == 0
        rows = {s["row"] for s in samples_of(c, step)}
        assert st[sa.C_WLAST] >> 32 == step and (st[sa.C_WLAST] & sa.M32) in rows
        w = st[sa.C_WWHERE] & 0x0FFFFFFF
        assert (st[sa.C_WWHERE] >> 28) & 0xF == sa.W_CONTENT and (w >> 12, (w >> 1) & 0x7FF, w & 1) == (1, 9, 1)
    # written where its block doesn't say (the content is there, at the wrong slot)
    moved = dict(c, slots=c["slots"].clone(), kv=c["kv"].clone())
    spare = [b for b in range(12) if b not in c["bt"].flatten().tolist()][0]
    for i in range(int(c["qsl"][-1])):
        moved["slots"][i] = spare * 16 + i
        moved["kv"][spare, :, i, :64] = c["key"][i]
        moved["kv"][spare, :, i, 64:] = c["val"][i]
    for b in run_steps(moved, range(4)):
        st = b["state"][0].tolist()
        assert st[sa.C_WSLOT] == sa.SAMPLES and st[sa.C_WBAD] == 0 and (st[sa.C_WWHERE] >> 28) & 0xF == sa.W_SLOT
    # padding slots (-1) and layers that share another's cache aren't checked
    pad = dict(c, slots=torch.full_like(c["slots"], -1))
    assert all(int(b["state"][0, sa.C_WN]) == 0 for b in run_steps(pad, range(3)))
    assert all(int(b["state"][0, sa.C_WN]) == 0 for b in run_steps(wrong, range(3), has_write=False))


# ----------------------------------------------------------------------------------------- recheck and flow
def test_the_recheck_confirms_a_wrong_output_and_clears_a_right_one():
    c = sa.make_case("cpu", torch.bfloat16, lens=((60, 1), (0, 0)), padded_rows=0)
    K, V = keys_of(c, 0, 0, 59)
    q, o = c["q"][0, 0].float(), c["o"][0, 0].float()
    assert sa.recheck(q, o, K, V, c["scale"], torch.bfloat16)["verdict"] == "output_agrees"
    o2 = o.clone()
    o2[3] += 0.5
    r = sa.recheck(q, o2, K, V, c["scale"], torch.bfloat16)
    assert r["verdict"] == "output_wrong" and r["outputs_over_line"] == 1 and r["keys"] == 60


class Attention(torch.nn.Module):
    def __init__(self, impl, name):
        super().__init__()
        self.impl, self.layer_name, self.head_size_v = impl, name, impl.head_size


class FlashAttentionImpl:
    def __init__(self, heads=4, kv_heads=2, head_size=64, **kw):
        self.num_heads, self.num_kv_heads, self.head_size = heads, kv_heads, head_size
        self.scale = head_size ** -0.5
        self.attn_type, self.alibi_slopes, self.sinks, self.logits_soft_cap = "decoder", None, None, 0
        self.sliding_window, self.kv_cache_dtype, self.kv_sharing_target_layer_name = (-1, -1), "auto", None
        for k, v in kw.items():
            setattr(self, k, v)


class Meta:
    def __init__(self, c, **kw):
        self.query_start_loc, self.seq_lens, self.block_table = c["qsl"], c["seq_lens"], c["bt"]
        self.slot_mapping, self.num_actual_tokens, self.causal = c["slots"], c["q"].shape[0], True
        for k, v in kw.items():
            setattr(self, k, v)


def test_unsupported_layers_and_calls_are_listed_not_checked():
    ok = FlashAttentionImpl()
    assert sa.why_not(Attention(ok, "a"), ok) is None
    for kw, why in ((dict(alibi_slopes=[1.0]), "ALiBi"), (dict(sinks=torch.zeros(4)), "sinks"),
                    (dict(logits_soft_cap=30.0), "soft cap"), (dict(sliding_window=(4095, 0)), "sliding window"),
                    (dict(kv_cache_dtype="fp8"), "fp8"), (dict(attn_type="encoder_only"), "encoder"),
                    (dict(head_size=512), "head size"), (dict(dcp_world_size=2), "context parallel")):
        impl = FlashAttentionImpl(**kw)
        assert why in sa.why_not(Attention(impl, "a"), impl), kw
    other = type("FlashInferImpl", (), dict(vars(ok)))()
    assert "FlashInferImpl" in sa.why_not(Attention(ok, "a"), other)
    for v in (2, 3):                                               # FlashAttention 4 (Blackwell) isn't checked
        impl = FlashAttentionImpl(vllm_flash_attn_version=v)
        assert sa.why_not(Attention(impl, "a"), impl) is None
    impl = FlashAttentionImpl(vllm_flash_attn_version=4)
    assert "FlashAttention 4" in sa.why_not(Attention(impl, "a"), impl)
    c = sa.make_case("cpu", torch.bfloat16)
    args = (ok, c["q"], c["o"], c["kv"])
    assert sa.call_issue(*args, Meta(c), None) is None
    assert "causal" in sa.call_issue(*args, Meta(c, causal=torch.ones(3, dtype=torch.bool)), None)
    assert "prefix" in sa.call_issue(*args, Meta(c, mm_prefix_range_tensor=torch.zeros(1)), None)
    assert "quantization" in sa.call_issue(*args, Meta(c), torch.ones(1))
    assert "KV cache" in sa.call_issue(ok, c["q"], c["o"], c["kv"].float(), Meta(c), None)
    R = sa.MAX_REQUESTS                     # the kernels' search over query_start_loc covers this many requests
    many = dict(query_start_loc=torch.zeros(R + 2, dtype=torch.int32), seq_lens=torch.zeros(R + 1, dtype=torch.int32),
                block_table=torch.zeros(R + 1, 1, dtype=torch.int32))
    assert "more than 65,535 requests" in sa.call_issue(*args, Meta(c, **many), None)
    fits = {k: v[1:] for k, v in many.items()}
    assert sa.call_issue(*args, Meta(c, **fits), None) is None
    gpu, cpu = types.SimpleNamespace(type="cuda"), torch.device("cpu")   # the PyTorch path's limit on GPUs
    kv16 = torch.zeros(1, 2, 16, 128)
    wide = Meta(c, block_table=torch.zeros(2, sa.TORCH_MAX_KEYS // 16 + 1, dtype=torch.int32))
    assert "8,192 keys" in sa.torch_path_issue(gpu, wide, kv16)
    assert sa.torch_path_issue(cpu, wide, kv16) is None
    assert sa.torch_path_issue(gpu, Meta(c, block_table=torch.zeros(2, sa.TORCH_MAX_KEYS // 16, dtype=torch.int32)),
                               kv16) is None
    with pytest.raises(ValueError):
        sa.AttnChecks(None, line=1.0)


def test_precompiling_a_shape_that_fails_turns_the_kernels_off_before_any_capture(monkeypatch):
    key = (sa._dev_key(torch.device("cpu")), torch.bfloat16)
    monkeypatch.setitem(sa.STATUS, key, (True, ""))
    ac = sa.AttnChecks(None)
    launches = []

    def launch(*a):
        launches.append((a[-2], a[-1], int(a[11][1])))            # (has_write, keep_kv, active)
    monkeypatch.setattr(sa, "attn_check_triton", launch)
    ac._precompile(torch.device("cpu"), torch.bfloat16, 64, 4, 2)
    assert sa.STATUS[key] == (True, "") and sorted(launches) == [(False, False, 0), (False, True, 0),
                                                                  (True, False, 0), (True, True, 0)]

    def counts(*a):                                                 # a launch that wasn't inactive
        a[9][0, sa.C_CALLS] += 1
    monkeypatch.setattr(sa, "attn_check_triton", counts)
    ac._precompile(torch.device("cpu"), torch.bfloat16, 64, 4, 2)
    assert sa.STATUS[key][0] is False and "head size 64" in sa.STATUS[key][1] and "counted" in sa.STATUS[key][1]
    monkeypatch.setitem(sa.STATUS, key, (True, ""))

    def fails(*a):
        raise RuntimeError("PassManager::run failed\nmore")
    monkeypatch.setattr(sa, "attn_check_triton", fails)
    ac._precompile(torch.device("cpu"), torch.bfloat16, 128, 2, 1)
    assert sa.STATUS[key] == (False, "compiling for head size 128: RuntimeError: PassManager::run failed")


def test_attention_metrics(monkeypatch):
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    c = sa.make_case("cpu", torch.bfloat16, lens=((48, 1), (20, 6), (0, 0)), NB=16)
    impls = [FlashAttentionImpl() for _ in range(2)]
    layers = [Attention(im, f"model.layers.{i}.self_attn.attn") for i, im in enumerate(impls)]
    met = Metrics()
    g = ServeGuard(metrics=met)
    ac = g.attn_checks()
    for lay in layers:
        ac.add(lay, lay.impl, lay.layer_name, "cpu")
    ac.finalize(max_len=128)
    wrong = c["o"].clone()
    wrong[:, :, 7] = wrong[:, :, 7].float() * 2 + 3
    for step in range(1, 7):
        g.mark_step(step)
        for lay in layers:
            sa.check_call(lay._sdcguard_attn, lay.impl, c["q"], c["key"], c["val"], c["kv"], Meta(c),
                          wrong if lay is layers[1] else c["o"])
        g.end_step()
    g.poll(wait=True)
    text = generate_latest(met.registry).decode()
    assert "sdcguard_attn_layers_checked 2.0" in text and "sdcguard_attn_calls_checked_total 6.0" in text
    assert f"sdcguard_attn_kv_writes_checked_total {6 * sa.SAMPLES}.0" in text
    assert 'sdcguard_alerts_total{UUID="",layer="model.layers.1.self_attn.attn"} 1.0' in text
    worst = lambda t: float(re.search(r"^sdcguard_attn_worst_score (\S+)$", t, re.M).group(1))  # noqa: E731
    assert worst(text) > sa.ATTN_Z
    g.mark_step(7)                                                  # a clean read: the counters only grow
    sa.check_call(layers[1]._sdcguard_attn, impls[1], c["q"], c["key"], c["val"], c["kv"], Meta(c), c["o"])
    g.end_step()
    g.poll(wait=True)
    text = generate_latest(met.registry).decode()
    assert "sdcguard_attn_calls_checked_total 7.0" in text and worst(text) <= 1.0


def test_a_step_past_2_to_the_31_is_named_exactly():
    c = sa.make_case("cpu", torch.bfloat16, lens=((48, 1), (20, 6), (0, 0)), NB=16)
    impl = FlashAttentionImpl()
    lay = Attention(impl, "model.layers.0.self_attn.attn")
    g = ServeGuard()
    ac = g.attn_checks()
    ac.add(lay, impl, lay.layer_name, "cpu")
    ac.finalize(max_len=128)
    wrong = c["o"].clone()
    wrong[:, :, 7] = wrong[:, :, 7].float() * 2 + 3
    key = c["key"].clone()
    key[:, :, 3] += 1                                                 # every write wrong too
    step = (1 << 31) + 5
    g.mark_step(step)
    sa.check_call(lay._sdcguard_attn, impl, c["q"], key, c["val"], c["kv"], Meta(c), wrong)
    g.end_step()
    ev = {e.detail["finding"]: e.detail for e in g.poll(wait=True) if e.detector == "serve_attn"}
    assert ev["output_wrong_or_misread"]["captured"]["step"] == step
    assert ev["kv_write_wrong"]["latest"]["step"] == step


def test_serveguard_checks_attention_layers_reads_alerts_and_detaches():
    c = sa.make_case("cpu", torch.bfloat16, lens=((48, 1), (20, 6), (0, 0)), NB=16)
    impls = [FlashAttentionImpl() for _ in range(2)]
    layers = [Attention(im, f"model.layers.{i}.self_attn.attn") for i, im in enumerate(impls)]
    g = ServeGuard()
    ac = g.attn_checks()
    for lay in layers:
        assert ac.add(lay, lay.impl, lay.layer_name, "cpu") is not None
    bad = FlashAttentionImpl(sinks=torch.zeros(4))
    assert ac.add(Attention(bad, "x"), bad, "model.layers.9.self_attn.attn", "cpu") is None
    assert ac.finalize(max_len=128) == 2
    wrong = c["o"].clone()
    wrong[:, :, 7] = wrong[:, :, 7].float() * 2 + 3                 # every output wrong at dimension 7
    for step in range(1, 7):
        g.mark_step(step)
        for lay in layers:
            sa.check_call(lay._sdcguard_attn, lay.impl, c["q"], c["key"], c["val"], c["kv"], Meta(c),
                          wrong if lay is layers[1] else c["o"])
        g.end_step()
    ev = [e for e in g.poll(wait=True) if e.detector == "serve_attn"]
    assert len(ev) == 1 and ev[0].detail["where"] == "model.layers.1.self_attn.attn"
    d = ev[0].detail
    assert d["dim"] == 7 and d["max_z"] > sa.ATTN_Z and d["captured"]["step"] in (1, 3, 5)
    assert d["recheck"]["verdict"] == "not_rechecked"           # the PyTorch check keeps no keys
    cov = g.coverage_report()["attention"]
    assert cov["layers"] == 2 and cov["calls_checked"] == 6 and cov["layers_checked"] == 2
    assert cov["not_attached"] == {"model.layers.9.self_attn.attn": "attention sinks"}
    assert ac.stats["samples"] == 6 * sa.SAMPLES and ac.stats["writes_checked"] == 6 * sa.SAMPLES
    # outside a model step (warm-up, capture runs): nothing is checked
    for lay in layers:
        sa.check_call(lay._sdcguard_attn, lay.impl, c["q"], c["key"], c["val"], c["kv"], Meta(c), wrong)
    assert not g.poll(wait=True) and ac.counts() == {lay.layer_name: 3 for lay in layers}
    # a call it can't check is counted by why
    g.mark_step(8)
    sa.check_call(layers[0]._sdcguard_attn, impls[0], c["q"], c["key"], c["val"], c["kv"],
                  Meta(c, causal=torch.ones(3, dtype=torch.bool)), c["o"])
    assert ac.not_checked_calls == {"attention that isn't causal for every request": 1}
    g.detach()
    assert not hasattr(layers[0], "_sdcguard_attn") and g.attn is None


def test_the_triton_kernels_match_the_pytorch_check():
    """Through Triton's interpreter on the CPU: the first-use comparison passes, the kernels' results equal the
    PyTorch check's over many calls, and an alarm keeps the keys and values its sample attended, whose float64
    recheck confirms the wrong output."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), HERE,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_attn_compare.py")], env=env, capture_output=True,
                       text=True, timeout=3000)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout
