"""Hugging Face's Conv1D (GPT-2 and relatives store linear weights as (in, out)) is checked like nn.Linear."""
import contextlib

import pytest
import torch

transformers = pytest.importorskip("transformers")

from sdcguard import FaultInjector, SDCGuard  # noqa: E402

CFG = dict(n_layer=2, n_embd=64, n_head=4, vocab_size=128, n_positions=32, resid_pdrop=0.0, embd_pdrop=0.0,
           attn_pdrop=0.0)


def _step(guarded, inject=False, exact_reductions=False):
    torch.manual_seed(0)
    m = transformers.GPT2LMHeadModel(transformers.GPT2Config(**CFG)).train()
    g = SDCGuard(every=1, exact_reductions=exact_reductions).protect(m) if guarded else None
    ids = torch.randint(0, 128, (4, 32), generator=torch.Generator().manual_seed(1))
    inj = FaultInjector(rate=1.0, seed=2).arm_on([m.transformer.h[0].mlp.c_fc]) if inject else None
    with (g.at_step(0) if g else contextlib.nullcontext()), (inj if inj else contextlib.nullcontext()):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
    return m, g, [p.grad.clone() for p in m.parameters()]


def test_gpt2_conv1d_layers_are_protected_and_gradients_unchanged():
    m, g, grads = _step(True)
    kinds = sorted({type(k).__name__ for k in g._protected})
    assert kinds == ["Conv1D", "Linear"] and len(g._protected) == 9          # 4 Conv1D per block + LM head
    assert g.stats["checked"] == 27 and g.stats["alerts"] == 0 and g.stats["unjudged"] == 0
    _, _, plain = _step(False)
    assert all(torch.equal(a, b) for a, b in zip(grads, plain))


def test_a_fault_in_a_conv1d_layer_is_caught_and_named():
    _, g, _ = _step(True, inject=True)
    assert g.stats["alerts"] >= 1
    assert g.log.events[0].detail["where"] == "transformer.h.0.mlp.c_fc"


def test_unprotect_restores_conv1d():
    m, g, _ = _step(True)
    g.unprotect()
    layer = m.transformer.h[0].mlp.c_fc
    assert type(layer) is transformers.pytorch_utils.Conv1D
    assert not [k for k in layer.__dict__ if k.startswith("_sdcguard")]
