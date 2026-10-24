"""The RMSNorm check (serve_norm.py) on the CPU: correct norms of every variant vLLM computes stay under the bound,
wrong outputs and changed weights are caught as the bound says (and small ones missed as it says), a kept row is
rechecked and freed, ServeGuard takes the norms along, and under torch.compile the check is one opaque op."""
import os
import subprocess
import sys

import pytest
import torch

from sdcguard import serve_norm as sn
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


class Norm(torch.nn.Module):
    """A norm computed the ways vLLM computes it. rounding: "native" (the normalized row rounded, then the product:
    two roundings, the fused sum normalized in fp32), "kernel" (its CUDA kernel: one rounding, the rounded sum
    normalized), "gemma" (x * (1 + w) in fp32, one rounding)."""

    def __init__(self, h, dtype=torch.bfloat16, rounding="native", seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.rounding = rounding
        w = (0.2 * torch.randn(h, generator=g)) if rounding == "gemma" else (1 + 0.3 * torch.randn(h, generator=g))
        self.weight = torch.nn.Parameter(w.to(dtype), requires_grad=False)
        self.variance_epsilon = 1e-6

    def compute(self, x, residual=None):
        orig = x.dtype
        s = x.float() if residual is None else x.float() + residual.float()
        res = None if residual is None else s.to(orig)
        if self.rounding == "kernel" and res is not None:
            s = res.float()
        n = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        if self.rounding == "native":
            y = (n.to(orig) * self.weight).to(orig)
        elif self.rounding == "kernel":
            y = (n * self.weight.float()).to(orig)
        else:
            y = (n * (self.weight.float() + 1.0)).to(orig)
        return y if residual is None else (y, res)

    def forward(self, x, residual=None):
        out = self.compute(x, residual)
        a = getattr(self, "_sdcguard_norm", None)
        if a is not None:
            sn.check_call(a, x, residual, out)
        return out


def _checks(mods, guard=None):
    nc = sn.NormChecks(guard)
    for i, m in enumerate(mods):
        nc.add(m, f"model.layers.{i}.norm", "gemma_rmsnorm" if m.rounding == "gemma" else "rmsnorm")
    nc.finalize(dtypes=())
    return nc


def _rows(m, h, dtype, scale=4.0, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, h, generator=g) * scale).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rounding", ["native", "kernel", "gemma"])
def test_correct_norms_stay_within_four_roundings(dtype, rounding):
    mods = [Norm(h, dtype, rounding, seed=h) for h in (64, 896, 1536, 4097)]
    nc = _checks(mods)
    for step in range(6):
        nc.mark_step(step)
        for m in mods:
            h = m.weight.shape[0]
            m(_rows(37, h, dtype, seed=step))
            m(_rows(37, h, dtype, seed=100 + step), _rows(37, h, dtype, 0.5, seed=200 + step))
        assert nc.poll(wait=True) == []
    worst = max(nc.worst_seen.values())
    assert 0.3 < worst <= 4.05, worst
    assert nc.stats["calls_checked"] == 6 * 4 * 2 and nc.stats["rows_checked"] == 6 * 4 * 2 * 37
    assert nc.stats["unjudged_outputs"] == 0 and nc.stats["alerts"] == 0


def test_a_wrong_output_is_caught_located_kept_and_rechecked():
    m = Norm(1536)
    nc = _checks([m])
    nc.mark_step(41)
    x = _rows(9, 1536, torch.bfloat16)
    y = m.compute(x)
    y[5, 700] = (y[5, 700].float() * 1.05).to(torch.bfloat16) + 0.0   # 5%: past 10 roundoffs, caught for sure
    sn.check_call(m._sdcguard_norm, x, None, y)
    ev = nc.poll(wait=True)
    assert len(ev) == 1
    d = ev[0].detail
    assert ev[0].detector == "serve_norm" and d["where"] == "model.layers.0.norm" and d["kind"] == "rmsnorm"
    assert (d["row"], d["col"]) == (5, 700) and d["max_z"] > 6
    assert d["captured"]["row"] == 5 and d["captured"]["step"] == 41
    assert d["recheck"]["verdict"] == "output_wrong" and d["recheck"]["outputs_over_line"] == 1
    assert d["weights_changed_since_load"] is False and d["read_window_steps"] == [41, 41]
    assert abs(d["relative_error"] - 0.05) < 0.01
    # the kept row was freed: the next wrong output is kept again
    nc.mark_step(42)
    y2 = m.compute(x)
    y2[2, 3] = -y2[2, 3]
    sn.check_call(m._sdcguard_norm, x, None, y2)
    d2 = nc.poll(wait=True)[0].detail
    assert d2["captured"]["row"] == 2 and d2["captured"]["step"] == 42 and d2["last_clean_read_steps"] is None


@pytest.mark.parametrize("dtype,small,large", [(torch.bfloat16, 0.009, 0.04), (torch.float16, 0.0012, 0.005)])
def test_what_is_caught_follows_the_bound(dtype, small, large):
    """A relative error under 2 roundoffs is never caught (score at most 2 + 4 < 6); one over 10 always is."""
    m = Norm(1024, dtype)
    nc = _checks([m])
    x = _rows(64, 1024, dtype)
    for rel, expect in ((small, False), (large, True), (-large, True)):
        y = m.compute(x)
        y2 = (y.float() * (1 + rel)).to(dtype)          # every output off by about rel (after rounding)
        sn.check_call(m._sdcguard_norm, x, None, y2)
        ev = nc.poll(wait=True)
        assert bool(ev) == expect, (rel, nc.worst_seen)
        if ev:
            assert ev[0].detail["recheck"]["verdict"] == "output_wrong"


def test_a_weight_changed_in_memory_reads_as_weights_changed_until_refresh():
    m = Norm(512, rounding="kernel")
    nc = _checks([m])
    x = _rows(4, 512, torch.bfloat16)
    with torch.no_grad():
        m.weight.data.view(torch.int16)[77] ^= (1 << 14)        # the top exponent bit: a huge weight
    m(x)
    ev = nc.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["col"] == 77
    assert ev[0].detail["recheck"]["verdict"] == "weights_changed"
    assert ev[0].detail["weights_changed_since_load"] is True
    assert nc.refresh() == 1
    m(x)
    assert nc.poll(wait=True) == []


def test_gemma_norms_compare_with_one_plus_their_weights():
    m = Norm(768, rounding="gemma")
    nc = _checks([m])
    x = _rows(5, 768, torch.bfloat16)
    m(x)
    assert nc.poll(wait=True) == []
    with torch.no_grad():
        m.weight.data[10] += 0.5                                # 1 + w changes by a lot
    m(x)
    ev = nc.poll(wait=True)
    assert ev and ev[0].detail["col"] == 10 and ev[0].detail["recheck"]["verdict"] == "weights_changed"


def test_non_finite_inputs_and_fp16_overflow_are_counted_not_alerted():
    m = Norm(256, torch.float16, rounding="kernel")
    nc = _checks([m])
    x = _rows(3, 256, torch.float16)
    x[1, 4] = float("nan")                                       # the input was already wrong upstream
    m(x)
    assert nc.poll(wait=True) == [] and nc.stats["unjudged_outputs"] == 256
    with torch.no_grad():
        m.weight.data[9] = 60000.0                               # the true output overflows fp16: inf is right
    m(_rows(3, 256, torch.float16, scale=1.0))
    ev = nc.poll(wait=True)                                      # (unannounced, that's a changed weight)
    assert ev and ev[0].detail["recheck"]["verdict"] == "weights_changed"
    nc.refresh()                                                 # announced
    before = nc.stats["unjudged_outputs"]
    m(_rows(3, 256, torch.float16, scale=1.0))
    assert nc.poll(wait=True) == []
    assert nc.stats["unjudged_outputs"] > before


def test_rows_near_fp32_overflow_are_not_judged():
    """A row whose sum of squares is near fp32's largest value: summation orders can disagree on whether it
    overflowed (one side 0, the other the norm), so it's counted unjudged, never alerted."""
    m = Norm(256)
    nc = _checks([m])
    x = torch.full((2, 256), 3e18, dtype=torch.bfloat16)        # sum of squares about 2.3e39
    y = torch.zeros(2, 256, dtype=torch.bfloat16)                # what an overflowed sum gives
    sn.check_call(m._sdcguard_norm, x, None, y)
    assert nc.poll(wait=True) == [] and nc.stats["unjudged_outputs"] == 512


def test_a_non_finite_output_from_a_finite_input_is_caught():
    m = Norm(256)
    nc = _checks([m])
    x = _rows(3, 256, torch.bfloat16)
    y = m.compute(x)
    y[0, 0] = float("nan")
    sn.check_call(m._sdcguard_norm, x, None, y)
    ev = nc.poll(wait=True)
    assert ev and ev[0].detail["max_z"] == float("inf") and ev[0].detail["recheck"]["verdict"] == "output_wrong"


def test_only_one_row_is_kept_until_its_alert_is_read():
    m = Norm(128)
    nc = _checks([m])
    x = _rows(6, 128, torch.bfloat16)
    for step, row in ((1, 4), (2, 1)):
        nc.mark_step(step)
        y = m.compute(x)
        y[row, 0] = y[row, 0] * 3 + 1
        sn.check_call(m._sdcguard_norm, x, None, y)
    ev = nc.poll(wait=True)                                      # one read: both calls, the first one's row kept
    assert len(ev) == 1 and ev[0].detail["captured"] == {"row": 4, "step": 1,
                                                         "score": ev[0].detail["captured"]["score"]}
    assert ev[0].detail["read_window_steps"] == [1, 2]


def test_unknown_norms_and_overrides_are_listed_not_checked():
    class Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(8))
            self.variance_size_override = 4

    nc = sn.NormChecks(None)
    assert nc.add(Odd(), "a", "rmsnorm") is None and "variance-size" in nc.unsupported["a"]
    assert nc.add(torch.nn.LayerNorm(8), "b") is None and "LayerNorm" in nc.unsupported["b"]
    with pytest.raises(ValueError):
        sn.NormChecks(None, line=4.0)


def test_serveguard_takes_the_norms_along(tmp_path):
    seen = []
    g = ServeGuard(log_path=str(tmp_path / "alerts.jsonl"), on_alert=seen.append)
    g.attribute = lambda ev: ev.detail.__setitem__("attributed", True)
    m = Norm(256)
    nc = g.norm_checks()
    nc.add(m, "model.norm", "rmsnorm")
    nc.finalize(dtypes=())
    x = _rows(2, 256, torch.bfloat16)
    g.mark_step(7)
    m(x)
    assert g.poll(wait=True) == []
    assert g.coverage_report()["norms"]["norms_checked"] == 1
    with torch.no_grad():
        m.weight.data[3] *= 2
    g.mark_step(8)
    m(x)
    ev = g.poll(wait=True)
    assert len(ev) == 1 and seen == ev and ev[0].detail["attributed"] is True
    assert ev[0].detail["captured"]["step"] == 8
    assert "serve_norm" in (tmp_path / "alerts.jsonl").read_text()
    assert g.refresh() == 0                                     # no linear layers; the norm takes its new weight
    m(x)
    assert g.poll(wait=True) == []
    g.detach()
    assert not hasattr(m, "_sdcguard_norm") and g.norms is None


def test_metrics_count_norm_checks_and_alerts():
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    met = Metrics()
    g = ServeGuard(metrics=met)
    m = Norm(128)
    nc = g.norm_checks()
    nc.add(m, "model.norm", "rmsnorm")
    nc.finalize(dtypes=())
    x = _rows(2, 128, torch.bfloat16)
    m(x)
    m(x)
    g.poll(wait=True)
    with torch.no_grad():
        m.weight.data[0] = -m.weight.data[0]
    m(x)
    g.poll(wait=True)
    text = generate_latest(met.registry).decode()
    assert "sdcguard_norms_checked 1.0" in text and "sdcguard_norm_checks_executed_total 3.0" in text
    assert 'sdcguard_alerts_total{UUID=""' in text and "sdcguard_all_alerts_total 1.0" in text


def test_strided_and_empty_inputs():
    """A per-head norm (vLLM's q_norm/k_norm) gets a strided view; an empty batch checks nothing."""
    m = Norm(64)
    nc = _checks([m])
    qkv = _rows(5, 3 * 4 * 64, torch.bfloat16)
    q = qkv[:, :4 * 64].view(5, 4, 64)                          # non-contiguous rows
    m(q)
    m(torch.zeros(0, 64, dtype=torch.bfloat16))
    assert nc.poll(wait=True) == [] and nc.stats["rows_checked"] == 20 and nc.stats["calls_checked"] == 1


def test_under_torch_compile_the_check_is_one_opaque_op():
    """The recompute must not be traced into the graph (the compiler would merge it with the norm's own mean and
    rsqrt): the graph holds sdcguard::norm_check as one call, and the results match eager mode."""
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    m = Norm(256, rounding="native")
    nc = _checks([m])
    x = _rows(4, 256, torch.bfloat16)
    r = _rows(4, 256, torch.bfloat16, 0.5, seed=3)
    torch._dynamo.reset()
    f = torch.compile(lambda a, b: m(a, b), backend=backend, fullgraph=True)
    f(x, r)
    calls = [n for gm in graphs for n in gm.graph.nodes if n.op == "call_function" and "norm_check" in str(n.target)]
    assert len(calls) == 1, [str(n.target) for gm in graphs for n in gm.graph.nodes]
    assert not any("rsqrt" in str(n.target) and n.args and "norm_check" in str(n.args) for gm in graphs
                   for n in gm.graph.nodes)
    assert nc.poll(wait=True) == [] and nc.stats["calls_checked"] == 1
    with torch.no_grad():
        m.weight.data[11] *= 4
    f(x, r)
    ev = nc.poll(wait=True)
    assert ev and ev[0].detail["col"] == 11
    torch._dynamo.reset()


def test_the_triton_kernel_matches_the_pytorch_check():
    """Through Triton's interpreter on the CPU: the first-use comparison passes, and the kernel's results equal the
    PyTorch check's over shapes that use row blocks, several tiles, masks, strided rows and wrong outputs."""
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_norm_compare.py")], env=env, capture_output=True,
                       text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout
