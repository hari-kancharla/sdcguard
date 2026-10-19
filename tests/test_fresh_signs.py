"""Training checks take a new sign pattern on every checked step.

Two wrong outputs of a block cancel in both checksums when they trade places where the pattern has the
same sign. With one pattern for the whole run, a fault that does that on every step would never show.
With a new pattern each checked step it shows on about half of them.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import SDCGuard, ServeGuard
from sdcguard import checksum as cs


def _partner():
    """An output that shares output 0's sign in the epoch-0 pattern (128-wide blocks)."""
    s0 = cs._signs(128, "cpu", 0)
    return next(j for j in range(1, 128) if s0[j] == s0[0])


def _differs(epoch, j):
    s = cs._signs(128, "cpu", epoch)
    return bool(s[j] != s[0])


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))


def _train(guard, model, steps=8):
    flagged = set()
    guard.on_alert = lambda ev: flagged.add(ev.step)
    for step in range(steps):
        x = torch.randn(32, 256, generator=torch.Generator().manual_seed(step))
        with guard.on_step(step), torch.autocast("cpu", dtype=torch.bfloat16):
            model(x).float().pow(2).mean().backward()
        model.zero_grad()
    return flagged


def test_patterns_differ_by_epoch_and_epoch_zero_is_unchanged():
    w = 128
    seed = (cs._SIGN_SEED + w) % (2 ** 63)
    g = torch.Generator().manual_seed(seed)
    assert torch.equal(cs._signs(w, "cpu"), (torch.randint(0, 2, (w,), generator=g) * 2 - 1).float())
    pats = [cs._signs(w, "cpu", e) for e in range(4)]
    for a in range(4):
        for b in range(a + 1, 4):
            assert not torch.equal(pats[a], pats[b])


def test_the_cache_keeps_epoch_zero_and_the_current_epoch_only():
    for e in range(1, 40):
        cs._signs(64, "cpu", e)
        cs._pair_weights(64, torch.bfloat16, "cpu", e)
    assert {k[-1] for k in cs._SIGNS} <= {0, 39}
    assert {k[-1] for k in cs._PAIR_W} <= {0, 39}


@pytest.mark.parametrize("fresh", [False, True])
def test_precise_mode_swapped_accumulator_outputs(fresh):
    """A layer whose fp32 result has outputs 0 and j swapped in every row, every step."""
    j = _partner()

    def swap(layer, pass_, c32):
        if layer == "0" and pass_ == "forward":
            c32[:, [0, j]] = c32[:, [j, 0]]

    model = _model()
    g = SDCGuard(every=1, precise=True, fresh_signs=fresh, on_accumulator=swap).protect(model)
    flagged = _train(g, model)
    if not fresh:
        assert flagged == set()               # the blind spot this option closes
    else:
        expected = {step for step in range(8) if _differs(step + 1, j)}     # step s runs epoch s + 1
        assert expected and flagged == expected


@pytest.mark.parametrize("fresh", [False, True])
def test_bf16_check_swapped_outputs(fresh, monkeypatch):
    """The ordinary bf16 checks: a matmul that swaps two outputs of every row, every time."""
    j = _partner()
    real = F.linear

    def bad_linear(x, w, b=None):
        y = real(x, w, b)
        if w.shape[0] == 512:
            y = y.clone()
            y[..., [0, j]] = y[..., [j, 0]]
        return y

    model = _model()
    g = SDCGuard(every=1, fresh_signs=fresh).protect(model)
    monkeypatch.setattr(torch.nn.functional, "linear", bad_linear)
    flagged = _train(g, model)
    if not fresh:
        assert flagged == set()
    else:
        assert flagged == {step for step in range(8) if _differs(step + 1, j)}


def test_clean_training_stays_clean_with_fresh_patterns():
    model = _model()
    g = SDCGuard(every=1).protect(model)
    flagged = _train(g, model, steps=6)
    assert flagged == set() and g.stats["alerts"] == 0 and g._epoch == 6


def test_serving_keeps_its_pattern_while_training_moves_on():
    """Serving computes its weight checksums once, with epoch 0; training's new patterns mustn't touch it."""
    torch.manual_seed(1)
    lin = nn.Linear(256, 384).to(torch.bfloat16)
    sg = ServeGuard()
    slot = sg.add_layer(lin, "lin")
    model = _model()
    g = SDCGuard(every=1).protect(model)
    _train(g, model, steps=5)
    x = torch.randn(16, 256).to(torch.bfloat16)
    y = lin(x).detach()
    sg.check(slot, x, y)
    z, _flat, _noise = sg.read_slot(slot)
    assert z < 6.0
    y[3, 10] += 40 * y.float().abs().mean().to(torch.bfloat16)
    sg.check(slot, x, y)
    assert sg.read_slot(slot)[0] > 6.0


def test_the_pattern_caches_survive_many_threads():
    """Four threads moving through epochs at once used to hit "dictionary changed size during iteration" while
    pruning (outside review, Oct 19, 2026)."""
    import concurrent.futures
    import sys
    from sdcguard import _group

    def worker(t):
        for i in range(1500):
            e = t * 100000 + i + 1
            cs._signs((i % 128) + 1, "cpu", e)
            cs._pair_weights((i % 128) + 1, torch.bfloat16, "cpu", e)
            if i % 50 == 0:
                _group._sign_table(torch.device("cpu"), e)
        return True

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            assert all(pool.map(worker, range(4)))
    finally:
        sys.setswitchinterval(old)
