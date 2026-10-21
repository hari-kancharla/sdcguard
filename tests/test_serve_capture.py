"""Original-event capture (serve.CAPTURE_SLOTS): when a serving check alarms, the check itself keeps the row of the
layer's input and the block of its output it judged, and the alert carries a float64 recheck of them. The review
that asked for this listed what it must show; each test here is one of those, on the CPU (the PyTorch check; the
Triton check's capture is compared with it bit for bit in _serve_fast._probe and tests/_serve_fast_compare.py)."""
import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard import serve
from sdcguard.serve import CAPTURE_SLOTS, ServeGuard


def _model(widths=(64, 256, 64), bias=True, seed=0):
    torch.manual_seed(seed)
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers += [nn.Linear(a, b, bias=bias), nn.ReLU()]
    return nn.Sequential(*layers[:-1]).to(torch.bfloat16)


class _Faults:
    """Wrong values in chosen calls of F.linear, the way a faulty GPU would produce them: (call number, function
    of the output) pairs; calls are counted over all layers."""

    def __init__(self, monkeypatch, plan):
        self.n, self.plan, real = 0, dict(plan), F.linear

        def linear(*a, **k):
            y = real(*a, **k)
            self.n += 1
            f = self.plan.get(self.n)
            return f(y) if f is not None else y
        monkeypatch.setattr(F, "linear", linear)


def _bump(row, col, by):
    def f(y):
        y = y.clone()
        y.view(-1, y.shape[-1])[row, col] += by
        return y
    return f


def test_a_one_time_fault_keeps_the_numbers_that_alarmed(monkeypatch, tmp_path):
    """A wrong value made once, then clean passes that reuse the memory: the alert carries the original wrong
    output and its input row (not a recomputation, which would be clean), the recheck says the output was wrong,
    and the evidence file holds exactly those numbers."""
    m = _model()
    g = ServeGuard(evidence_dir=str(tmp_path / "ev")).attach(m)
    xs = [torch.randn(8, 64).to(torch.bfloat16) for _ in range(5)]
    _Faults(monkeypatch, {3: _bump(5, 77, 40.0)})          # the second pass's first layer, row 5
    seen = {}
    with torch.no_grad():
        for i, x in enumerate(xs):
            y = m[0](x)
            if i == 1:
                seen["x"], seen["y"] = x.clone(), y.clone()
            m[2](F.relu(y))
    (ev,) = g.poll(wait=True)
    d = ev.detail
    assert d["where"] == "0" and d["captured"] == {"row": 5, "col_block": 0, "with_bias": True, "z": d["captured"]["z"]}
    assert d["captured"]["z"] > 6 and d["recheck"]["verdict"] == "output_wrong"
    assert d["recheck"]["cpu"]["outputs_outside_rounding_envelope"] == 1 and "wrong" in d["reading"]
    payload = torch.load(d["evidence"] + ".pt")
    assert torch.equal(payload["A"][0], seen["x"][5]) and torch.equal(payload["C"][0], seen["y"][5, :128])
    assert torch.equal(payload["B"], m[0].weight.detach()[:128].t())
    info = json.load(open(d["evidence"] + ".json"))
    assert info["captured_at_check"] and info["layer_row"] == 5 and info["recheck"]["verdict"] == "output_wrong"
    assert not info.get("captured")                       # (that key marks a second-pass recapture)
    assert info["cpu_recheck"]["output_mismatch_confirmed"] is True     # the saved slice, judged again
    assert g.stats["captured"] == 1 and g.evidence == [d["evidence"]]
    assert g.poll(wait=True) == []


def test_a_fault_in_the_checks_own_side_is_not_blamed_on_the_output():
    """The check's expected side wrong (its block sums of W, as a fault in its own arithmetic would make it) while
    the output is right: the alarm still fires, and the recheck says the output agrees with float64."""
    m = _model()
    g = ServeGuard().attach(m)
    cs = g.layers[0]["cs"]
    with torch.no_grad():
        cs.parts[:, 0] += 0.5                             # the plain checksum of block 0
        m[0](torch.randn(4, 64).to(torch.bfloat16))
    (ev,) = g.poll(wait=True)
    r = ev.detail["recheck"]
    assert r["verdict"] == "output_agrees", r
    assert r["cpu"]["outputs_outside_rounding_envelope"] == 0 and r["cpu"]["max_z"] < 6
    assert "nothing here says this output was wrong" in ev.detail["reading"]


def test_weights_changed_since_load_are_said_not_rechecked_as_the_cause():
    m = _model()
    g = ServeGuard().attach(m)
    with torch.no_grad():
        m[0].weight[3, 7] += 4.0                          # memory corruption, or an update without refresh()
        m[0](torch.randn(4, 64).to(torch.bfloat16))
    (ev,) = g.poll(wait=True)
    assert ev.detail["weights_changed_since_load"] is True
    assert ev.detail["recheck"]["verdict"] == "weights_changed"


def test_errors_that_add_up_are_told_from_one_wrong_value(monkeypatch):
    """Every output of a block off by up to 80% of what worst-case fp32 accumulation allows, in random directions
    (a kernel that accumulates sloppily): each value is inside the rounding envelope, so none is a confirmed wrong
    value, but the block's checksums still disagree far beyond their noise. Neither the check's own side nor one
    wrong value: output_off, a statistical mismatch."""
    from sdcguard.screen import exact_block
    torch.manual_seed(1)
    k = 4096
    m = nn.Sequential(nn.Linear(k, 128, bias=False)).to(torch.bfloat16)
    g = ServeGuard().attach(m)
    x = torch.randn(4, k).to(torch.bfloat16)
    W = m[0].weight.detach()
    exact = x.double() @ W.double().t()
    tol = 2 * (2.0 ** -8 * exact.abs() + k * 2.0 ** -23 * (x.double().abs() @ W.double().abs().t()))
    sign = torch.where(torch.rand(exact.shape, generator=torch.Generator().manual_seed(2)) < 0.5, -1.0, 1.0)
    sloppy = (exact + 0.8 * tol * sign.double()).to(torch.bfloat16)
    for r in range(4):                                     # the setup itself: inside the envelope everywhere
        assert exact_block(x[r:r + 1], W.t(), sloppy[r:r + 1], None, 1.0, 1.0)["outputs_outside_rounding_envelope"] == 0
    _Faults(monkeypatch, {1: lambda y: sloppy})
    with torch.no_grad():
        m(x)
    (ev,) = g.poll(wait=True)
    r = ev.detail["recheck"]
    assert r["verdict"] == "output_off", r
    assert r["cpu"]["outputs_outside_rounding_envelope"] == 0 and r["cpu"]["max_z"] > 6


def test_outputs_all_rounded_the_same_wrong_way_pass_by_design(monkeypatch):
    """The plain checksum allows equal-looking rounding in every output of a block (identical columns round alike:
    test_identical_columns_do_not_false_alarm), so a block whose outputs are all truncated toward zero, each within
    one unit, is not flagged. Documented here so the limit is known, not discovered."""
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(64, 128, bias=False)).to(torch.bfloat16)
    with torch.no_grad():
        m[0].weight.copy_(m[0].weight.abs())
    g = ServeGuard().attach(m)
    x = torch.rand(4, 64).to(torch.bfloat16) + 0.1        # positive outputs: all truncated downward

    def truncate(y):
        exact = (x.double() @ m[0].weight.detach().double().t()).float()
        return (exact.view(torch.int32) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)
    _Faults(monkeypatch, {1: truncate})
    with torch.no_grad():
        m(x)
    assert g.poll(wait=True) == []


def test_one_row_per_layer_per_read_and_a_full_buffer_is_said(monkeypatch):
    """Each layer keeps at most one row between reads; past CAPTURE_SLOTS rows a device keeps no more, and those
    alerts say so (and the next read has room again)."""
    widths = (64,) * (CAPTURE_SLOTS + 3)
    m = _model(widths)
    g = ServeGuard().attach(m)
    n = len(widths) - 1
    _Faults(monkeypatch, {i: _bump(1, 3, 50.0) for i in range(1, 2 * n + 1)})    # every layer, two passes
    with torch.no_grad():
        for _ in range(2):
            m(torch.randn(4, 64).to(torch.bfloat16))
    evs = g.poll(wait=True)
    assert len(evs) == n
    kept = [e for e in evs if e.detail["captured"]]
    assert len(kept) == CAPTURE_SLOTS and g.stats["captured"] == CAPTURE_SLOTS
    full = [e for e in evs if not e.detail["captured"]]
    assert all("buffer was full" in e.detail["capture_missing"] for e in full)
    assert g.stats["captures_dropped"] == n - CAPTURE_SLOTS        # (one row per layer: the second pass kept none)
    _Faults(monkeypatch, {1: _bump(1, 3, 50.0)})
    with torch.no_grad():
        m(torch.randn(4, 64).to(torch.bfloat16))
    (ev,) = g.poll(wait=True)
    assert ev.detail["captured"] and ev.detail["recheck"]["verdict"] == "output_wrong"


def test_each_read_keeps_its_own_rows(monkeypatch):
    """A layer that alarms in two read windows gets its row captured in each, matched to the right alert."""
    m = _model()
    g = ServeGuard().attach(m)
    xs = [torch.randn(4, 64).to(torch.bfloat16) for _ in range(2)]
    _Faults(monkeypatch, {1: _bump(0, 9, 30.0), 3: _bump(2, 140, 30.0)})
    out = []
    with torch.no_grad():
        for x in xs:
            m(x)
            out += g.poll(wait=True)
    assert [(e.detail["captured"]["row"], e.detail["captured"]["col_block"]) for e in out] == [(0, 0), (2, 1)]
    assert all(e.detail["recheck"]["verdict"] == "output_wrong" for e in out)


def test_capture_works_after_the_buffer_grows_for_a_wider_layer(monkeypatch):
    """A layer whose input is longer than the capture buffer's rows reallocates it, and layers attached before get
    the new buffer: their rows are still kept, whole."""
    m1, m2 = _model(), nn.Sequential(nn.Linear(9000, 8)).to(torch.bfloat16)
    g = ServeGuard().attach(m1)
    before = g._cap[torch.device("cpu")][1]
    g.attach(m2)
    after = g._cap[torch.device("cpu")][1]
    assert after.shape[1] >= g._cap_w + 9000 > before.shape[1]
    x = torch.randn(4, 64).to(torch.bfloat16)
    _Faults(monkeypatch, {1: _bump(3, 2, 30.0)})
    with torch.no_grad():
        m1[0](x)
    (ev,) = g.poll(wait=True)
    assert ev.detail["where"] == "0" and ev.detail["recheck"]["verdict"] == "output_wrong"
    x2 = torch.randn(2, 9000).to(torch.bfloat16)
    _Faults(monkeypatch, {1: _bump(1, 5, 30.0)})
    with torch.no_grad():
        m2(x2)
    (ev,) = g.poll(wait=True)
    assert ev.detail["captured"]["row"] == 1 and ev.detail["recheck"]["verdict"] == "output_wrong"


def test_capture_inside_a_compiled_graph(monkeypatch):
    """Under torch.compile the PyTorch check, capture included, is traced into the graph (no host sync, no
    data-dependent branch). The fault is a tensor the graph reads, set for one call only."""
    m = _model()
    g = ServeGuard().attach(m)
    fault = torch.zeros(8, 256, dtype=torch.bfloat16)
    real = F.linear

    def linear(x, w, b=None):
        y = real(x, w, b)
        return y + fault if y.shape == fault.shape else y
    monkeypatch.setattr(F, "linear", linear)
    cm = torch.compile(m, backend="aot_eager", fullgraph=True)
    xs = [torch.randn(8, 64).to(torch.bfloat16) for _ in range(3)]
    with torch.no_grad():
        cm(xs[0])
        fault[6, 200] = 25.0
        cm(xs[1])
        fault.zero_()
        cm(xs[2])
    (ev,) = g.poll(wait=True)
    assert ev.detail["captured"]["row"] == 6 and ev.detail["captured"]["col_block"] == 1
    assert ev.detail["recheck"]["verdict"] == "output_wrong"


def test_the_check_op_keeps_its_arguments_in_one_place():
    """op_args (what the vLLM plugin and the guard's own forward pass to the check) carries the capture buffers;
    the custom op's schema has them too (its version is part of the vLLM compile cache key)."""
    m = _model()
    g = ServeGuard().attach(m)
    args = g.op_args(0, with_bias=True)
    cm, cd, cw, tag = args[-4:]
    assert cm is g._cap[torch.device("cpu")][0] and cd is g._cap[torch.device("cpu")][1]
    assert cw == g._cap_w and tag == 3 and g.op_args(0, with_bias=False)[-1] == 2
    assert serve.OP_SCHEMA == 2
    op = serve._op()
    if op is not serve.check_static_any:
        assert "cap_meta" in str(torch.ops.sdcguard.check_static.default._schema)


def _changed_and_put_back(model, x, how):
    """Layer 0's weight changed, the forward run with it, the weight put back before the read: the output is right
    for the weight the matmul used, and the weight now matches load time again."""
    w = model[0].weight
    original = w.detach().clone()
    with torch.no_grad():
        if how == "in place":
            w[3, 7] += 8
        else:
            w.data[3, 7] += 8                       # (through .data: PyTorch doesn't count it as a write)
        used = w.detach().clone()
        y = model[0](x)
        if how == "in place":
            w.copy_(original)
        else:
            w.data.copy_(original)
    assert torch.equal(y, F.linear(x, used, model[0].bias))
    return y


def test_a_weight_changed_and_put_back_before_the_read_is_not_called_a_wrong_output(tmp_path):
    """An outside review's case (October 21, 2026): the recheck used the weight as it is at the read, the one put
    back, and called the right output wrong (output_wrong). An in-place write since load now makes the verdict
    unknown, the alert says why, and so does the casefile of the evidence it saved."""
    from sdcguard import casefile
    m = _model()
    g = ServeGuard(evidence_dir=str(tmp_path)).attach(m)
    _changed_and_put_back(m, torch.randn(4, 64).to(torch.bfloat16), "in place")
    (ev,) = g.poll(wait=True)
    d = ev.detail
    assert d["weights_changed_since_load"] is False and d["weights_rewritten_since_load"] is True
    assert d["captured"] is not None and d["recheck"]["verdict"] == "unknown"
    assert "written in place since load" in d["recheck"]["why"] and d["reading"] == d["recheck"]["why"]
    inc = casefile.load_incident(d["evidence"] + ".pt")
    assert inc["weights_rewritten_since_load"] is True
    a = casefile.assess(inc, [])
    assert any("written in place since load" in n for n in a["next"])
    # the CPU's judgment, against the weight put back, isn't taken: the right output isn't shown wrong, and no
    # hypothesis rests on it
    assert a["observations"]["recorded output vs CPU"] == "unknown"
    assert all(e["status"] != casefile.SUPPORTED for e in a["explanations"].values())
    assert "may not be the ones the matmul used" in a["explanations"]["check"]["because"][0]


def test_a_write_through_data_isnt_counted_as_documented():
    """The limit _recheck states: PyTorch doesn't count a write through .data, so a weight changed and put back
    that way still reads as unchanged, and the right output is called wrong for the weights the recheck has."""
    m = _model()
    g = ServeGuard().attach(m)
    _changed_and_put_back(m, torch.randn(4, 64).to(torch.bfloat16), ".data")
    (ev,) = g.poll(wait=True)
    assert ev.detail["weights_rewritten_since_load"] is False
    rc = ev.detail["recheck"]
    assert rc["verdict"] == "output_wrong" and "wrong for these weights" in rc["why"]


def test_refresh_after_an_update_starts_the_write_count_again(monkeypatch):
    """An intended update followed by refresh(): later alerts are judged normally again."""
    m = _model()
    g = ServeGuard().attach(m)
    with torch.no_grad():
        m[0].weight.mul_(1.01)
    g.refresh()
    _Faults(monkeypatch, {1: _bump(2, 75, 40.0)})
    with torch.no_grad():
        m[0](torch.randn(4, 64).to(torch.bfloat16))
    (ev,) = g.poll(wait=True)
    assert ev.detail["weights_rewritten_since_load"] is False and ev.detail["recheck"]["verdict"] == "output_wrong"
    assert ev.detail["reading"].endswith("PyTorch counted no in-place write to them since)")


def test_weights_made_in_inference_mode_attach_and_are_judged(monkeypatch):
    """Inference tensors keep no write count (reading one raises): a model built in torch.inference_mode() attaches,
    its alerts are judged, and whether its weights were written isn't known, so the alert doesn't say there was
    no write."""
    with torch.inference_mode():
        m = _model()
    assert m[0].weight.is_inference()
    g = ServeGuard().attach(m)
    _Faults(monkeypatch, {1: _bump(2, 75, 40.0)})
    with torch.inference_mode():
        m[0](torch.randn(4, 64).to(torch.bfloat16))
    (ev,) = g.poll(wait=True)
    assert ev.detail["weights_rewritten_since_load"] is None and ev.detail["recheck"]["verdict"] == "output_wrong"
    assert "no in-place write" not in ev.detail["reading"] and "load-time fingerprint" in ev.detail["reading"]
