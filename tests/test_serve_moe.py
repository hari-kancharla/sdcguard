"""The fused-MoE check (serve_moe.py) on the CPU: the recompute against a kernel that rounds as vLLM's Triton MoE
kernel does but adds in another order (clean scores far under the line), gross faults caught, small ones not
claimed, layers taking turns by step, the kept row rechecked, a changed expert weight read as such, the layers the
check doesn't take listed, and the Triton kernels against the PyTorch check through Triton's interpreter."""
import os
import subprocess
import sys

import pytest
import torch

from _vllm_standins import install_fake_moe, moe_kernel_like
from sdcguard import serve_moe as sm
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def _layer(E=8, H=256, I=128, K=4, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    w13 = (torch.randn(E, 2 * I, H, generator=g) * 0.04).to(dtype)
    w2 = (torch.randn(E, H, I, generator=g) * 0.05).to(dtype)
    return w13, w2


def _route(M, E, K, seed=1):
    g = torch.Generator().manual_seed(seed)
    w, ids = torch.topk(torch.softmax(torch.randn(M, E, generator=g), -1), K, -1)
    return w / w.sum(-1, keepdim=True), ids.to(torch.int32)


def _z(y, x, ids, w, w13, w2, dtype, act=0, emap=None):
    r, cmax, sq, mx = sm.recompute(x, ids, w, w13, w2, emap, act, False, dtype)
    return sm.scores(y.float(), r, cmax, sq, mx, dtype)[0]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("act", [0, 1])
def test_a_correct_layer_scores_far_under_the_line(dtype, act):
    w13, w2 = _layer(dtype=dtype)
    x = torch.randn(16, 256).to(dtype)
    w, ids = _route(16, 8, 4)
    y = moe_kernel_like(x, w13, w2, ids, w, act, dtype)
    z = _z(y, x, ids, w, w13, w2, dtype, act)
    assert float(z.max()) < 1.0, float(z.max())


def test_an_expert_map_drops_experts_off_this_gpu():
    w13, w2 = _layer(E=5)
    emap = torch.tensor([0, -1, 1, 2, -1, 3, 4, -1], dtype=torch.int32)    # 8 global experts, 5 here
    x = torch.randn(10, 256).to(torch.bfloat16)
    w, ids = _route(10, 8, 3)
    y = moe_kernel_like(x, w13, w2, ids, w, 0, torch.bfloat16, expert_map=emap)
    assert float(_z(y, x, ids, w, w13, w2, torch.bfloat16, 0, emap).max()) < 1.0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gross_faults_are_caught(dtype):
    w13, w2 = _layer(dtype=dtype)
    x = torch.randn(12, 256).to(dtype)
    w, ids = _route(12, 8, 4)
    y = moe_kernel_like(x, w13, w2, ids, w, 0, dtype)
    wrong_expert = ids.clone()
    wrong_expert[5, 0] = (wrong_expert[5, 0] + 1) % 8
    cases = {
        "a wrong expert": moe_kernel_like(x, w13, w2, wrong_expert, w, 0, dtype)[5],
        "another row's output": y[6],
        "the output scaled by 1.5": (y[5].float() * 1.5).to(dtype),
        "a NaN": torch.where(torch.arange(256) == 9, torch.tensor(float("nan"), dtype=dtype), y[5]),
        "an exponent bit flipped": (y[5].view(torch.int16) ^ torch.where(torch.arange(256) == 3, 1 << 13, 0)
                                    .to(torch.int16)).view(dtype),
    }
    for what, row in cases.items():
        yb = y.clone()
        yb[5] = row
        z = _z(yb, x, ids, w, w13, w2, dtype)
        assert float(z[5].max()) > 6, (what, float(z[5].max()))
        assert float(torch.cat([z[:5], z[6:]]).max()) < 1.0


def test_small_errors_in_bf16_are_not_claimed():
    """In bf16 the bound is a few roundoffs of the experts' outputs: a 5% error in one output isn't seen."""
    w13, w2 = _layer()
    x = torch.randn(8, 256).to(torch.bfloat16)
    w, ids = _route(8, 8, 4)
    y = moe_kernel_like(x, w13, w2, ids, w, 0, torch.bfloat16)
    yb = y.clone()
    yb[2, 40] = (yb[2, 40].float() * 1.05).to(torch.bfloat16)
    assert float(_z(yb, x, ids, w, w13, w2, torch.bfloat16)[2, 40]) < 6


def _fake_layers(n, monkeypatch, **kw):
    mod = install_fake_moe(monkeypatch.setitem)
    return mod, [mod.RoutedExperts(seed=i, **kw) for i in range(n)]


def _route_for(layer, x, seed):
    w, ids = _route(x.shape[0], layer.w13_weight.shape[0], layer.top_k, seed)
    return w, ids


def test_layers_take_turns_and_an_alert_keeps_and_rechecks_its_row(monkeypatch, tmp_path):
    mod, layers = _fake_layers(3, monkeypatch)
    g = ServeGuard(log_path=str(tmp_path / "alerts.jsonl"))
    mc = g.moe_checks()
    for i, lay in enumerate(layers):
        assert mc.add(lay, f"model.layers.{i}.mlp.experts") == i
    mc.finalize()
    x = torch.randn(7, 128).to(torch.bfloat16)
    for step in range(6):
        g.mark_step(step)
        for i, lay in enumerate(layers):
            w, ids = _route_for(lay, x, i)
            out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
            sm.check_call(lay._sdcguard_moe, x, w, ids, out)
        assert g.poll(wait=True) == []
    assert mc.stats["active_calls"] == 6 and mc.stats["rows_checked"] == 6 * sm.SAMPLES
    assert all(v == 2 for v in mc.checks_seen.values()), mc.checks_seen
    assert max(mc.worst_seen.values()) < 1.0
    # layer 1 starts returning wrong rows: caught at the next step it is active (step % 3 == 1)
    layers[1].fault = lambda out: torch.flip(out, [0])
    hits = []
    for step in range(6, 9):
        g.mark_step(step)
        for i, lay in enumerate(layers):
            w, ids = _route_for(lay, x, i)
            out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
            sm.check_call(lay._sdcguard_moe, x, w, ids, out)
        hits += g.poll(wait=True)
    assert len(hits) == 1, [e.detail["where"] for e in hits]
    d = hits[0].detail
    assert hits[0].detector == "serve_moe" and d["where"] == "model.layers.1.mlp.experts"
    assert d["captured"]["step"] == 7 and d["recheck"]["verdict"] == "output_wrong"
    assert d["weights_changed_since_load"] is False and d["max_z"] > 6
    assert "serve_moe" in (tmp_path / "alerts.jsonl").read_text()
    assert g.coverage_report()["moe"]["layers_checked"] == 3


def test_a_changed_expert_weight_reads_as_weights_changed_until_refresh(monkeypatch):
    mod, (lay,) = _fake_layers(1, monkeypatch)
    g = ServeGuard()
    mc = g.moe_checks()
    mc.add(lay, "experts")
    mc.finalize()
    x = torch.randn(4, 128).to(torch.bfloat16)
    w, ids = _route_for(lay, x, 0)
    out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
    with torch.no_grad():
        for e in ids.reshape(-1).tolist():
            lay.w2_weight.data[e] *= 3                    # the layer's weights changed after its output was made
    g.mark_step(0)
    sm.check_call(lay._sdcguard_moe, x, w, ids, out)
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["weights_changed_since_load"] is True
    assert mc.refresh() == 1
    out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
    sm.check_call(lay._sdcguard_moe, x, w, ids, out)
    assert g.poll(wait=True) == []


def test_layers_the_check_doesnt_take_are_listed(monkeypatch):
    mod = install_fake_moe(monkeypatch.setitem)
    mc = sm.MoEChecks(None)
    assert mc.add(mod.RoutedExperts(backend="FLASHINFER_CUTLASS"), "a") is None
    assert "FLASHINFER_CUTLASS" in mc.unsupported["a"]
    assert mc.add(mod.RoutedExperts(act="SWIGLUOAI"), "b") is None and "activation" in mc.unsupported["b"]
    biased = mod.RoutedExperts()
    biased.w13_bias = torch.zeros(8, 128)
    assert mc.add(biased, "c") is None and "biases" in mc.unsupported["c"]
    assert mc.add(mod.RoutedExperts(dtype=torch.float32), "d") is None


def test_the_sampled_rows_cover_the_batch_over_steps():
    seen = set()
    for step in range(64):
        seen |= set(sm.sample_rows(torch.tensor(step), 13, "cpu").tolist())
    assert seen == set(range(13))
    assert sm.sample_rows(torch.tensor(5), 1, "cpu").tolist() == [0] * sm.SAMPLES
    # layers taking turns in a number of steps that shares factors with the batch: each layer still takes every row
    # in turn (rows drawn from the step number itself would give layer 5 of 24 only 2 of 16 rows here)
    for idx in (0, 5, 23):
        seen = set()
        for k in range(16):
            seen |= set(sm.sample_rows(torch.tensor(k * 24 + idx), 16, "cpu", 24).tolist())
        assert seen == set(range(16)), (idx, seen)


def test_the_triton_kernels_match_the_pytorch_check():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), HERE,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_moe_compare.py")], env=env, capture_output=True,
                       text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout


def test_metrics_count_moe_checks_and_alerts(monkeypatch):
    pytest.importorskip("prometheus_client")
    from prometheus_client import generate_latest

    from sdcguard.alerts import Metrics
    mod, (lay,) = _fake_layers(1, monkeypatch)
    met = Metrics()
    g = ServeGuard(metrics=met)
    mc = g.moe_checks()
    mc.add(lay, "experts")
    mc.finalize()
    x = torch.randn(4, 128).to(torch.bfloat16)
    w, ids = _route_for(lay, x, 0)
    for step in range(3):
        g.mark_step(step)
        out = lay.forward_modular(x=x, topk_weights=w, topk_ids=ids)
        if step == 2:
            out = out * 2
        sm.check_call(lay._sdcguard_moe, x, w, ids, out)
    g.poll(wait=True)
    text = generate_latest(met.registry).decode()
    assert "sdcguard_moe_layers_checked 1.0" in text and "sdcguard_moe_checks_executed_total 3.0" in text
    assert 'sdcguard_alerts_total{UUID="",layer="experts"} 1.0' in text
