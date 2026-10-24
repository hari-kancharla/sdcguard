"""The canary's exact fingerprints (tol=0): raw bits, position-sensitive, with dtype and shape, so a swap or two
changes that cancel in the sums are seen; the sums of tol > 0 can't see them."""
import torch

from sdcguard.canary import Canary, _shape_code, exact_row


def _canary(tol=0.0):
    c = Canary.__new__(Canary)
    c.tol = tol
    return c


def _sums(t):
    t = t.double()
    return torch.stack([t.sum(), t.abs().sum()])


def test_a_swap_and_offsetting_changes_disagree_exactly_but_not_in_the_sums():
    torch.manual_seed(0)
    a = torch.randn(64, 128).to(torch.bfloat16)
    swapped = a.clone().view(-1)
    swapped[[3, 900]] = swapped[[900, 3]]
    swapped = swapped.view(64, 128)
    offset = a.clone()
    offset[0, 0], offset[0, 1] = 1.0, -1.0
    a[0, 0], a[0, 1] = -1.0, 1.0                    # same values, the other way round: same sum and |sum|
    swapped[0, 0], swapped[0, 1] = -1.0, 1.0
    exact, loose = _canary(), _canary(1e-9)
    for b in (swapped, offset):
        assert not torch.equal(a, b)
        assert loose._agree(_sums(a), _sums(b))                     # the sums can't tell them apart
        assert not exact._agree(exact_row(a), exact_row(b))         # the exact fingerprint can
    assert exact._agree(exact_row(a), exact_row(a.clone()))


def test_one_bit_dtype_and_shape_change_the_exact_row():
    a = torch.randn(8, 16).to(torch.bfloat16)
    b = a.clone()
    b.view(torch.int16)[2, 5] ^= 1                  # the lowest mantissa bit of one value
    rows = [exact_row(a), exact_row(b), exact_row(a.view(torch.int16)), exact_row(a.reshape(16, 8))]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            assert not torch.equal(rows[i], rows[j]), (i, j)
    assert _shape_code((16, 8)) != _shape_code((8, 16)) and _shape_code(()) != _shape_code((1,))


def test_a_model_whose_layer_swaps_two_outputs_is_seen_by_the_exact_canary():
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8))
    batch = torch.randn(4, 16)
    clean = Canary(model).fingerprints(batch)

    def swap(_m, _i, out):
        out = out.clone()
        out[0, [1, 2]] = out[0, [2, 1]]
        return out
    h = model[2].register_forward_hook(swap)
    try:
        bad = Canary(model).fingerprints(batch)
    finally:
        h.remove()
    assert clean.shape == bad.shape == (3, 6)
    assert torch.equal(clean[:2], bad[:2]) and not torch.equal(clean[2], bad[2])


def test_a_missing_layer_call_pads_with_a_value_no_row_has():
    from sdcguard.canary import _PAD
    row = exact_row(torch.zeros(3))
    pad = torch.full((6,), _PAD, dtype=torch.int64)
    assert not _canary()._agree(row, pad) and _canary()._agree(pad, pad.clone())
