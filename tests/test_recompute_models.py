"""The recompute check on real checkpointed models: Hugging Face GPT-2 and Llama (gradient_checkpointing_enable),
torch's checkpoint_wrapper, reentrant and non-reentrant, dropout with and without preserved RNG, early stop,
coverage, and unchanged gradients. CPU, small random-weight configs (no downloads)."""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint as tuc

from sdcguard import SDCGuard
from sdcguard.recompute import RecomputeCheck

transformers = pytest.importorskip("transformers")


def gpt2(reentrant=False, dropout=0.1, rng=True, layers=3):
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=layers, n_embd=64, n_head=4, vocab_size=128, n_positions=64,
                                  resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)
    m = transformers.GPT2LMHeadModel(cfg)
    kw = {"use_reentrant": reentrant}
    if not rng:
        kw["preserve_rng_state"] = False
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kw)
    return m.train()


def llama(layers=2):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                                   num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                                   max_position_embeddings=64)
    m = transformers.LlamaForCausalLM(cfg)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return m.train()


def step(m, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, 128, (2, 32))
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    return out.loss.detach()


def grads(m):
    g = [p.grad.clone() for p in m.parameters() if p.grad is not None]
    m.zero_grad(set_to_none=True)
    return g


@pytest.mark.parametrize("reentrant", [False, True])
def test_gpt2_checkpointing_turned_on_before_the_check_is_covered(reentrant):
    m = gpt2(reentrant)                                  # gradient_checkpointing_enable() ran first
    with RecomputeCheck() as rc:
        step(m)
        assert rc.flush() == [] and rc.stats["recomputes_compared"] == 0      # the gap without the model
    with RecomputeCheck().install(m) as rc:
        for s in range(2):
            step(m, s)
            assert rc.flush() == []
    assert rc.stats["recomputes_compared"] == 2 * 3 and rc.stats["mismatches"] == 0     # dropout on, RNG kept
    # uninstalled: the model's partials are back to torch's checkpoint, and training still works
    assert m.transformer.h[0]._gradient_checkpointing_func.func is tuc.checkpoint
    assert torch.isfinite(step(m))


def test_gradients_are_bit_identical_with_and_without_the_check():
    m = gpt2()
    step(m, 3)
    plain = grads(m)
    with RecomputeCheck().install(m) as rc:
        step(m, 3)
        rc.flush()
    for a, b in zip(plain, grads(m)):
        assert torch.equal(a, b)


def _transient(module, calls=(1,), delta=1e-3):
    """A fault in one forward call of `module` (counting the recompute as a call), like a GPU computing one block
    wrong once."""
    n = {"i": 0}

    def hook(mod, inp, out):
        # the same operations on every call: a fault changes values, not the graph (checkpointing also requires
        # the recompute to save the same tensors as the forward)
        n["i"] += 1
        with torch.no_grad():
            corr = torch.zeros_like(out)
            if n["i"] in calls:
                corr += delta * (out.abs() > out.abs().median()).to(out.dtype)
        return out + corr
    return module.register_forward_hook(hook)


@pytest.mark.parametrize("which", ["forward", "recompute"])
def test_a_transient_fault_in_a_real_block_is_caught_and_named(which):
    m = gpt2()
    h = _transient(m.transformer.h[1].mlp.c_fc, calls=(1,) if which == "forward" else (2,))
    seen = []
    try:
        with RecomputeCheck(on_alert=seen.append).install(m) as rc:
            step(m)
            rc.flush()
    finally:
        h.remove()
    assert len(seen) == 1 and seen[0].detail["block_call"] == 2 and "GPT2Block" in seen[0].detail["where"]


def test_dropout_without_preserved_rng_is_skipped_not_a_false_alarm():
    m = gpt2(rng=False)
    with RecomputeCheck().install(m) as rc:
        step(m)
        assert rc.flush() == []
    assert rc.stats["skipped_rng_not_preserved"] == 3 and rc.stats["recomputes_compared"] == 0


@pytest.mark.parametrize("rng", [True, False])
def test_why_unpreserved_rng_is_skipped(rng):
    """Plain torch checkpointing, no check: with preserved RNG the recompute repeats the forward exactly; without
    it, dropout draws new masks and the two runs differ on healthy hardware (so comparing them would be wrong)."""
    m = gpt2(rng=rng)
    runs = []
    h = m.transformer.h[0].register_forward_hook(lambda mod, inp, out: runs.append(out[0].detach().clone()))
    try:
        with tuc.set_checkpoint_early_stop(False):       # so the recompute runs the whole block
            step(m)
    finally:
        h.remove()
    assert len(runs) == 2 and torch.equal(runs[0], runs[1]) is rng


def test_llama_layers_and_coverage():
    m = llama()
    rc = RecomputeCheck()
    cov = rc.coverage(m, lambda: step(m))
    assert cov["blocks_compared"] == 2 and 0.2 < cov["parameter_share_inside_compared_blocks"] < 1.0
    assert "Embedding" in cov["module_types_with_parameters_outside"]
    assert rc.stats["mismatches"] == 0 and not rc._installed


def test_checkpoint_wrapper_applied_before_the_check():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
    torch.manual_seed(0)
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(32, 4, 64, dropout=0.1, batch_first=True), 3,
                                enable_nested_tensor=False)
    apply_activation_checkpointing(enc, check_fn=lambda mod: isinstance(mod, nn.TransformerEncoderLayer))
    enc.train()
    with RecomputeCheck().install(enc) as rc:
        enc(torch.randn(2, 10, 32, requires_grad=True)).sum().backward()
        assert rc.flush() == []
    assert rc.stats["recomputes_compared"] == 3


def test_explicit_early_stop_is_overridden_so_outputs_exist_to_compare():
    def two(x):
        a = torch.sin(x) * 2
        return a, a.exp()
    x = torch.randn(16, requires_grad=True)
    with RecomputeCheck() as rc:
        a, b = tuc.checkpoint(two, x, use_reentrant=False, early_stop=True)
        (a.sum() + b.sum()).backward()
        rc.flush()
    assert rc.stats["recomputes_compared"] == 1 and rc.stats["mismatches"] == 0


def test_a_reference_captured_while_installed_works_after_uninstall():
    rc = RecomputeCheck().install()
    captured = tuc.checkpoint                             # e.g. gradient_checkpointing_enable() ran meanwhile
    rc.uninstall()
    x = torch.randn(4, requires_grad=True)
    y = captured(torch.sin, x, use_reentrant=False)
    assert y is not None and torch.allclose(y, torch.sin(x))


def test_sdcguard_reaches_a_model_whose_checkpointing_was_on_first():
    m = gpt2()
    g = SDCGuard(recompute=True).watch(m)
    try:
        with g.at_step(0):
            step(m)
    finally:
        g.unwatch()
        g._recompute.uninstall()
    assert g.stats["recompute_compared"] == 3 and g.stats["recompute_mismatches"] == 0


def test_sdcguard_warns_when_a_checked_step_compared_nothing():
    m = gpt2()
    g = SDCGuard(recompute=True)                          # model not given to protect() or watch()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with g.at_step(0):
            step(m)
    g._recompute.uninstall()
    assert g.stats["recompute_steps_without_blocks"] == 1
    assert any("no checkpointed block was compared" in str(x.message) for x in w)
