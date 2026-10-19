"""Serving counts: how many checks each layer executed and how many checksums couldn't be judged. A worst score
of 0 can't tell a clean layer from one never checked (outside review, Oct 19); these counts can."""
import json
import math

import pytest
import torch

from sdcguard import _serve_fast
from sdcguard.serve import STATE_COLS, LayerChecksums, ServeGuard, check_static_impl, decode_counts


def _layer(m=8, k=64, n=300, dt=torch.float16, seed=0, bias=True):
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dt)
    b = torch.randn(n, generator=g).to(dt) if bias else None
    x = torch.randn(m, k, generator=g).to(dt)
    return x, W, b


def _check(x, y, cs, cols=STATE_COLS):
    st = torch.zeros(1, cols, dtype=torch.int64)
    check_static_impl(x, y, *cs.tensors(True), st, 0, 6.0, 1e-20)
    return st


def test_a_clean_check_counts_one_execution_and_nothing_unjudged():
    x, W, b = _layer()
    cs = LayerChecksums(W, b, 128)
    y = torch.nn.functional.linear(x, W, b)
    st = _check(x, y, cs)
    assert decode_counts(st) == [(1, 0)]
    for _ in range(2):
        check_static_impl(x, y, *cs.tensors(True), st, 0, 6.0, 1e-20)
    assert decode_counts(st) == [(3, 0)]
    assert _serve_fast.decode(*st[0].tolist())[0] > 0             # the score is still there (4 columns decode)


def test_an_empty_batch_is_not_counted_as_a_check():
    x, W, b = _layer(m=0)
    cs = LayerChecksums(W, b, 128)
    st = _check(x, torch.nn.functional.linear(x, W, b), cs)
    assert decode_counts(st) == [(0, 0)]


def test_an_infinite_input_makes_its_rows_checksums_unjudged():
    x, W, b = _layer(m=4, n=300)
    cs = LayerChecksums(W, b, 128)
    x[1, 0] = float("inf")
    y = torch.nn.functional.linear(x, W, b)
    st = _check(x, y, cs)
    ex, un = decode_counts(st)[0]
    assert ex == 1 and un == 2 * cs.nb          # both checksums of every block of row 1, nothing else
    assert _serve_fast.decode(*st[0].tolist())[0] <= 6.0           # scored 0 there: no alert, but not "clean"


def test_an_overflow_the_inputs_explain_is_unjudged_and_a_nan_they_dont_is_judged():
    x, W, b = _layer(m=4, n=256)
    y = torch.nn.functional.linear(x * 300.0, W * 300.0, b)
    assert not bool(torch.isfinite(y).all())
    cs = LayerChecksums(W * 300.0, b, 128)
    st = _check(x * 300.0, y, cs)
    ex, un = decode_counts(st)[0]
    assert ex == 1 and un > 0
    # a NaN where small inputs can't produce one: corruption, judged (score inf), nothing unjudged
    x2, W2, b2 = _layer(m=4, n=256, seed=1)
    cs2 = LayerChecksums(W2, b2, 128)
    y2 = torch.nn.functional.linear(x2, W2, b2)
    y2[2, 7] = float("nan")
    st2 = _check(x2, y2, cs2)
    assert decode_counts(st2) == [(1, 0)] and math.isinf(_serve_fast.decode(*st2[0].tolist())[0])


def test_a_two_column_state_still_works_without_counts():
    x, W, b = _layer()
    cs = LayerChecksums(W, b, 128)
    y = torch.nn.functional.linear(x, W, b)
    old, new = _check(x, y, cs, cols=2), _check(x, y, cs)
    assert decode_counts(old) == [(0, 0)]
    assert torch.equal(old[0], new[0, :2])


def test_the_guard_reports_executed_checks_and_layers_never_checked():
    torch.manual_seed(0)

    class Branchy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(32, 64)
            self.b = torch.nn.Linear(64, 16)
            self.unused = torch.nn.Linear(16, 16)        # attached, never run

        def forward(self, x):
            return self.b(torch.relu(self.a(x)))

    model = Branchy().to(torch.bfloat16)
    g = ServeGuard().attach(model)
    try:
        for _ in range(3):
            model(torch.randn(5, 32, dtype=torch.bfloat16))
        before = g.counts()                              # live state, nothing polled yet
        assert before["a"] == {"checks": 3, "unjudged": 0} and before["unused"]["checks"] == 0
        assert g.poll(wait=True) == []
        model(torch.randn(2, 32, dtype=torch.bfloat16))
        cov = g.coverage_report()
        assert cov["layers"] == 3 and cov["layers_checked"] == 2 and cov["never_checked"] == ["unused"]
        assert cov["checks_executed"] == 8 and cov["unjudged_checksums"] == 0
        assert g.stats["checks_executed"] == 6          # (what polls have read back so far)
        g.poll(wait=True)
        assert g.stats["checks_executed"] == 8 and g.counts()["b"]["checks"] == 4
        # a zero score alone can't tell "unused" from a clean layer; the counts can
        assert g.worst_z()["unused"] == 0.0
    finally:
        g.detach()


def test_an_fp32_layer_is_listed_as_not_attached():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8).to(torch.bfloat16), torch.nn.Linear(8, 8))
    g = ServeGuard().attach(model)
    try:
        assert g.stats["layers"] == 1 and "float32" in g.coverage_report()["not_attached"]["1"]
    finally:
        g.detach()


def test_the_guard_counts_unjudged_checksums_from_an_infinite_input():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 200)).to(torch.bfloat16)
    g = ServeGuard().attach(model)
    try:
        x = torch.randn(3, 16, dtype=torch.bfloat16)
        x[0, 3] = float("inf")
        model(x)
        assert g.poll(wait=True) == []                   # can't be judged: no alert...
        c = g.counts()["0"]
        assert c["checks"] == 1 and c["unjudged"] > 0    # ...and not counted as clean either
        assert g.coverage_report()["layers_with_unjudged"]
    finally:
        g.detach()


def test_prometheus_counters_follow_the_counts(tmp_path):
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)).to(torch.bfloat16)
    met = Metrics()
    g = ServeGuard(metrics=met).attach(model)
    try:
        model(torch.randn(2, 8, dtype=torch.bfloat16))
        g.poll(wait=True)
        model(torch.randn(2, 8, dtype=torch.bfloat16))
        g.poll(wait=True)
        text = generate_latest(met.registry).decode()
        assert "sdcguard_checks_executed_total 4.0" in text and "sdcguard_unjudged_checksums_total 0.0" in text
    finally:
        g.detach()


def test_the_screen_reports_measured_serving_checks(tmp_path):
    from sdcguard.screen import screen_one
    s = screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path), max_steps=2)
    assert s["result"] == "PASS"
    assert s["serving_layers_never_checked"] == [] and s["unjudged_serving_checksums"] == 0
    assert s["linear_layer_checks_while_serving"] == s["serving_passes"] * s["serving_linear_layers"]
    saved = json.load(open(f"{s['out_dir']}/summary.json"))
    assert saved["linear_layer_checks_while_serving"] == s["linear_layer_checks_while_serving"]


def test_the_screen_is_incomplete_when_a_serving_layer_never_ran(tmp_path, monkeypatch):
    from sdcguard import screen
    real = ServeGuard.coverage_report

    def gap(self):
        r = real(self)
        return {**r, "layers_checked": r["layers_checked"] - 1, "never_checked": [self.layers[0]["name"]]}

    monkeypatch.setattr(ServeGuard, "coverage_report", gap)
    s = screen.screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path), max_steps=1)
    assert s["result"] == "INCOMPLETE" and s["serving_layers_never_checked"]


def test_the_screen_is_incomplete_when_serving_checksums_went_unjudged(tmp_path, monkeypatch):
    from sdcguard import screen
    real = ServeGuard.coverage_report
    monkeypatch.setattr(ServeGuard, "coverage_report", lambda self: {**real(self), "unjudged_checksums": 5})
    s = screen.screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path), max_steps=1)
    assert s["result"] == "INCOMPLETE" and s["unjudged_serving_checksums"] == 5
