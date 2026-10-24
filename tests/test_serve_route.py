"""The MoE routing check (serve_route.py): quiet on vLLM 0.32's own routing code (its grouped_topk transcribed in
tests/_vllm_standins.py) and on routing computed as vLLM's bf16 grouped top-k kernel rounds (scores and the bias sum
in bf16, ties broken either way); wrong experts and wrong weights named with their step and row; routers taking
turns; padding and capture runs not judged; policies it doesn't know listed; its Triton kernel against its PyTorch
version; and through the vLLM plugin with stand-ins for vLLM's routers and MoE runner, where the experts' own check
agrees with a wrong routing and this one doesn't."""
import logging
import os
import subprocess
import sys
import types

import pytest
import torch

from _vllm_standins import (RoutedMoEDecoder, install_fake_moe, install_fake_router, install_fake_vllm, load,
                            plugin_env, vllm_grouped_topk, vllm_topk)
from sdcguard import serve_route as sr
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def _router(name, **kw):
    return type(name, (), dict(kw, _select_experts=lambda *a, **k: None))()


def _setup(router, name="model.layers.0.mlp.experts.router"):
    g = ServeGuard()
    rc = g.route_checks()
    a = rc.add(router, name, torch.device("cpu"))
    rc.finalize()
    return g, rc, router._sdcguard_route if a is not None else None


def _logits(m=16, e=64, seed=0, spread=2.0):
    return torch.randn(m, e, generator=torch.Generator().manual_seed(seed)) * spread


BIAS = torch.linspace(-0.05, 0.05, 64)
CASES = {
    "softmax top-k, renormalized": (dict(top_k=4, scoring_func="softmax", renormalize=True),
                                    "FusedTopKRouter", lambda lg: vllm_topk(lg, 4, True)),
    "sigmoid top-k": (dict(top_k=6, scoring_func="sigmoid", renormalize=False), "FusedTopKRouter",
                      lambda lg: vllm_topk(lg, 6, False, "sigmoid")),
    "DeepSeek grouped, biased, scaled": (
        dict(top_k=8, scoring_func="sigmoid", renormalize=True, num_expert_group=8, topk_group=4,
             routed_scaling_factor=2.5, global_num_experts=64, e_score_correction_bias=BIAS),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 8, True, 8, 4, "sigmoid", 2.5, BIAS)),
    "grouped softmax, no bias": (
        dict(top_k=6, scoring_func="softmax", renormalize=False, num_expert_group=4, topk_group=2,
             routed_scaling_factor=1.0, global_num_experts=64),
        "GroupedTopKRouter", lambda lg: vllm_grouped_topk(lg, 6, False, 4, 2, "softmax")),
    "biased top-k (FusedTopKBiasRouter)": (
        dict(top_k=4, scoring_func="sigmoid", renormalize=True, routed_scaling_factor=1.5,
             e_score_correction_bias=BIAS, _hash_indices_table=None, bias_vl=None, num_fused_shared_experts=0),
        "FusedTopKBiasRouter", lambda lg: _biased_topk(lg, 4, 1.5)),
}


def _biased_topk(lg, k, scale):
    s = lg.float().sigmoid()
    ids = (s + BIAS).topk(k, -1).indices
    w = s.gather(1, ids)
    return w / w.sum(-1, keepdim=True) * scale, ids.to(torch.int32)


def _step(rc, a, step, lg, w, ids, rows=None):
    rc.mark_step(step, rows=lg.shape[0] if rows is None else rows)
    sr.check_call(a, lg, w, ids)
    rc.end_step()


@pytest.mark.parametrize("case", list(CASES))
def test_vllms_routing_is_quiet_and_wrong_routing_is_named(case):
    kw, cls, vllm = CASES[case]
    g, rc, a = _setup(_router(cls, **kw))
    for step in range(6):
        lg = _logits(seed=step)
        _step(rc, a, step, lg, *vllm(lg))
    assert g.poll(wait=True) == [] and rc.stats["rows"] == 12 and rc.stats["calls"] == 6
    lg = _logits(seed=9)
    w, ids = vllm(lg)
    bad = ids.clone()
    bad[:, -1] = lg.argmin(-1).to(bad.dtype)             # every row: its last expert swapped for its worst one
    _step(rc, a, 9, lg, w, bad)
    ev = g.poll(wait=True)
    assert len(ev) == 1 and ev[0].detector == "serve_route"
    d = ev[0].detail
    assert d["finding"] == "wrong_experts" and d["captured"]["step"] == 9 and d["rows_with_wrong_experts"] == 2
    assert d["where"] == "model.layers.0.mlp.experts.router"
    assert d["recheck"]["verdict"] == "output_wrong" and d["recheck"]["captured"]["step"] == 9
    assert d["recheck"]["experts"] == bad[d["recheck"]["captured"]["row"]].tolist()
    _step(rc, a, 10, lg, w * 1.1, ids)                   # the right experts with weights 10% off
    ev = g.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["finding"] == "wrong_weights"
    assert ev[0].detail["worst_weight_difference"] > 0.005


def _random_ties_topk(vals, k, gen):
    """Top-k of vals with ties broken at random (vLLM's kernels may pick equal scores either way)."""
    order = torch.rand(vals.shape, generator=gen).argsort(-1)
    idx = vals.gather(-1, order).argsort(dim=-1, descending=True, stable=True)[..., :k]
    return order.gather(-1, idx)


def _bf16_kernel(lg, k, groups, topk_group, bias, scale, renorm, gen, scoring="sigmoid"):
    """Routing as vLLM's grouped top-k computes it with 16-bit logits: for sigmoid, its CUDA kernel (sigmoid in fp32
    cast to the logits' dtype, the bias cast and added in that dtype, group scores summed in it); for softmax, its
    PyTorch path (softmax, weights and renormalization in the logits' dtype); ties broken at random."""
    T = lg.dtype
    s = torch.sigmoid(lg.float()).to(T) if scoring == "sigmoid" else torch.softmax(lg, -1)
    ch = s + bias.to(T) if bias is not None else s
    r, e = ch.shape
    gv = ch.view(r, groups, e // groups)
    if bias is not None:
        t2 = gv.topk(2, -1).values
        gs = t2[..., 0] + t2[..., 1]
    else:
        gs = gv.amax(-1)
    gi = _random_ties_topk(gs.float(), topk_group, gen)
    keep = torch.zeros(r, groups, dtype=torch.bool).scatter_(1, gi, True)
    cand = ch.float().masked_fill(~keep.unsqueeze(-1).expand(r, groups, e // groups).reshape(r, e), float("-inf"))
    ids = _random_ties_topk(cand, k, gen)
    if scoring == "sigmoid":
        w = s.float().gather(1, ids)
        if renorm:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * scale
    else:
        w = s.gather(1, ids)
        if renorm:
            w = w / w.sum(-1, keepdim=True)
        w = (w * scale if scale != 1.0 else w).float()
    return w, ids.to(torch.int32)


@pytest.mark.parametrize("dtype,scoring,spread", [(torch.bfloat16, "sigmoid", 0.3), (torch.bfloat16, "sigmoid", 3.0),
                                                  (torch.bfloat16, "softmax", 1.0), (torch.float16, "sigmoid", 0.3),
                                                  (torch.float16, "softmax", 2.0)])
def test_routing_rounded_as_vllms_16_bit_kernels_round_is_never_flagged(dtype, scoring, spread):
    gen = torch.Generator().manual_seed(1)
    E, k, groups, tg = 64, 6, 8, 3
    bias = torch.randn(E, generator=gen) * 0.02 if scoring == "sigmoid" else None
    scale = 2.5 if scoring == "sigmoid" else 1.0
    r = _router("GroupedTopKRouter", top_k=k, scoring_func=scoring, renormalize=True, num_expert_group=groups,
                topk_group=tg, routed_scaling_factor=scale, global_num_experts=E, e_score_correction_bias=bias)
    g, rc, a = _setup(r)
    for step in range(600):
        lg = (torch.randn(2, E, generator=gen) * spread).to(dtype)
        _step(rc, a, step, lg, *_bf16_kernel(lg, k, groups, tg, bias, scale, True, gen, scoring))
    assert g.poll(wait=True) == [] and rc.stats["rows"] == 1200
    lg = (torch.randn(2, E, generator=gen) * spread).to(dtype)
    w, ids = _bf16_kernel(lg, k, groups, tg, bias, scale, True, gen, scoring)
    bad = ids.clone()
    bad[:, 0] = lg.float().argmin(-1).to(bad.dtype)
    _step(rc, a, 600, lg, w, bad)
    assert [e.detail["finding"] for e in g.poll(wait=True)] == ["wrong_experts"]


def test_the_recheck_clears_a_row_that_fits_the_policy_and_frees_its_slot():
    g, rc, a = _setup(_router("FusedTopKRouter", top_k=2, scoring_func="softmax", renormalize=True))
    lg = _logits(4, 16)
    w, ids = vllm_topk(lg, 2, True)
    meta, clg, cid, cw = rc._caps[torch.device("cpu")]
    meta[0] = torch.tensor([1, 3, 7, sr.KIND_SET, 16, 0])    # as if the GPU check had kept row 3 as wrong
    clg[0, :16], cid[0, :2], cw[0, :2] = lg[3], ids[3].long(), w[3]
    out = rc._recheck(torch.device("cpu"), rc.layers[0])
    assert out["verdict"] == "output_agrees" and out["captured"] == {"row": 3, "step": 7, "kind": "experts"}
    assert int(meta[0, 0]) == 0
    assert rc._recheck(torch.device("cpu"), rc.layers[0])["verdict"] == "not_rechecked"


def test_a_routing_alert_confirmed_by_its_recheck_is_a_fault_for_decide():
    from sdcguard import decide
    ev = {"detector": "serve_route", "detail": {"where": "layers.3.router", "captured": {"step": 9},
                                                "recheck": {"verdict": "output_wrong"}}}
    a = decide.arithmetic(alerts=[ev], plugin={"checked": True, "min_weights_checked": 1.0})
    assert a["verdict"] == "FAULT" and "layers.3.router at step 9" in a["faults"][0]


def test_tied_scores_may_be_picked_either_way():
    g, rc, a = _setup(_router("FusedTopKRouter", top_k=2, scoring_func="softmax", renormalize=True))
    lg = torch.zeros(4, 8)
    lg[:, 0] = 3.0                                          # expert 0 first, then seven experts tied
    s = torch.softmax(lg, -1)
    for alt in (1, 5):
        ids = torch.tensor([[0, alt]]).repeat(4, 1).to(torch.int32)
        w = s.gather(1, ids.long())
        _step(rc, a, alt, lg, w / w.sum(-1, keepdim=True), ids)
    assert g.poll(wait=True) == []


def test_routers_take_turns():
    g = ServeGuard()
    rc = g.route_checks()
    rs = [_router("FusedTopKRouter", top_k=2, scoring_func="softmax", renormalize=True) for _ in range(3)]
    for i, r in enumerate(rs):
        rc.add(r, f"r{i}", torch.device("cpu"))
    rc.finalize()
    for step in range(6):
        rc.mark_step(step, rows=8)
        for r in rs:
            lg = _logits(8, 16, seed=step)
            sr.check_call(r._sdcguard_route, lg, *vllm_topk(lg, 2, True))
        rc.end_step()
    assert g.poll(wait=True) == [] and rc.seen == {"r0": 2, "r1": 2, "r2": 2}
    assert g.coverage_report()["moe_routing"]["routers_checked"] == 3


def test_padding_and_runs_outside_steps_arent_judged():
    g, rc, a = _setup(_router("FusedTopKRouter", top_k=2, scoring_func="softmax", renormalize=True))
    lg = _logits(8, 16)
    w, ids = vllm_topk(lg, 2, True)
    ids[4:] = 0                                             # padding rows: whatever the kernel left
    _step(rc, a, 1, lg, w, ids, rows=4)
    sr.check_call(a, lg, w, ids * 0)                        # a capture run between steps
    marked = ids.clone()
    marked[:] = -1                                          # rows vLLM marked as padding
    _step(rc, a, 2, lg, w, marked)
    assert g.poll(wait=True) == [] and rc.stats["calls"] == 2 and rc.stats["rows"] == 2


def test_policies_it_doesnt_know_are_listed():
    g = ServeGuard()
    rc = g.route_checks()
    cpu = torch.device("cpu")
    assert rc.add(_router("CustomRoutingRouter", top_k=2), "a", cpu) is None
    assert "CustomRoutingRouter" in rc.unsupported["a"]
    assert rc.add(_router("FusedTopKRouter", top_k=2, scoring_func="softmax", renormalize=True,
                          eplb_state=object()), "b", cpu) is None
    assert "load balancing" in rc.unsupported["b"]
    assert rc.add(_router("FusedTopKBiasRouter", top_k=2, scoring_func="sigmoid", renormalize=True,
                          _hash_indices_table=torch.zeros(3)), "c", cpu) is None
    assert rc.add(_router("FusedTopKBiasRouter", top_k=2, scoring_func="sqrtsoftplus"), "d", cpu) is None
    assert rc.add(_router("FusedTopKRouter", top_k=64, scoring_func="softmax"), "e", cpu) is None
    assert rc.layers == [] and g.coverage_report().get("moe_routing") is None


def test_a_grouped_router_without_a_valid_grouping_is_judged_as_vllm_routes_it():
    r = _router("GroupedTopKRouter", top_k=2, scoring_func="sigmoid", renormalize=True, num_expert_group=3,
                topk_group=1, routed_scaling_factor=2.0, global_num_experts=16)    # 16 experts, 3 groups
    pol, why = sr.policy_of(r)
    assert why is None and pol["groups"] == 0 and pol["scoring"] == sr.SCORING["softmax"] and pol["scale"] == 1.0


def test_the_triton_kernel_matches_the_pytorch_check():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), HERE,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_route_compare.py")], env=env, capture_output=True,
                       text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout


# ------------------------------------------------------------------------------------------- through the plugin
def _plugin(monkeypatch, tmp_path, grouped=False, **env):
    UL, GM = install_fake_vllm(monkeypatch.setitem)
    moe = install_fake_moe(monkeypatch.setitem)
    rt = install_fake_router(monkeypatch.setitem)

    def init(self, factory):
        self.factory = factory
        self.input_batch = types.SimpleNamespace(req_ids=[])

    def load_model(self):
        self.model = load(self.factory())

    def execute_model(self, so):
        self.input_batch.req_ids = list(so.num_scheduled_tokens)
        return self.model(so.x)

    monkeypatch.setattr(GM, "__init__", init, raising=False)
    monkeypatch.setattr(GM, "load_model", load_model, raising=False)
    monkeypatch.setattr(GM, "execute_model", execute_model, raising=False)
    vp = plugin_env(monkeypatch, tmp_path, **{"SDCGUARD_MOE": "1", **env})
    vp.register()
    runner = GM(lambda: RoutedMoEDecoder(moe, rt, UL, grouped=grouped))
    return vp, runner, rt


def _so(seed, rows=6):
    x = (torch.randn(rows, 128, generator=torch.Generator().manual_seed(seed)) * 2).to(torch.bfloat16)
    return types.SimpleNamespace(x=x, num_scheduled_tokens={"req-a": rows - 2, "req-b": 2},
                                 total_num_scheduled_tokens=rows)


@pytest.mark.parametrize("grouped", [False, True])
def test_a_wrong_router_is_caught_through_the_plugin_while_the_experts_check_agrees(monkeypatch, tmp_path, caplog,
                                                                                    grouped):
    vp, runner, rt = _plugin(monkeypatch, tmp_path, grouped=grouped)
    assert vp._STATE["route_classes"] == ["FusedMoERouter"]
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        runner.load_model()
    assert any("checking the routing of 3 of 3 MoE routers" in r.getMessage() for r in caplog.records)
    g = vp.guard()
    for s in range(6):
        runner.execute_model(_so(s))
    assert vp.poll(wait=True) == []
    cov = g.coverage_report()["moe_routing"]
    assert cov["routers"] == 3 and cov["routers_checked"] == 3 and cov["calls"] == 6
    # layer 1's router starts giving each row the experts of the row before it (weights still its own)
    runner.model.runners[1].router.fault = lambda w, ids: (w, torch.roll(ids, 1, 0))
    ev = []
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        for s in range(6, 9):
            runner.execute_model(_so(s))
            ev += vp.poll(wait=True)
    assert [e.detector for e in ev] == ["serve_route"], [(e.detector, e.detail.get("where")) for e in ev]
    assert ev[0].detail["where"] == "runners.1.router" and ev[0].detail["finding"] == "wrong_experts"
    assert any("MoE routing of runners.1.router" in r.getMessage() for r in caplog.records)
    # its weights, off by half, with the right experts
    runner.model.runners[1].router.fault = lambda w, ids: (w * 1.5, ids)
    ev = []
    for s in range(9, 12):
        runner.execute_model(_so(s))
        ev += vp.poll(wait=True)
    assert [(e.detector, e.detail["finding"]) for e in ev] == [("serve_route", "wrong_weights")]


def test_routing_checks_are_off_without_sdcguard_moe(monkeypatch, tmp_path):
    vp, runner, rt = _plugin(monkeypatch, tmp_path, SDCGUARD_MOE="0")
    assert vp._STATE["route_classes"] == [] and getattr(rt.FusedMoERouter, "_sdcguard_route_orig", None) is None
    runner.load_model()
    runner.execute_model(_so(0))
    assert vp.guard().route is None


def test_a_failing_routing_check_turns_that_router_off_without_breaking_serving(monkeypatch, tmp_path, caplog):
    vp, runner, rt = _plugin(monkeypatch, tmp_path)
    runner.load_model()
    monkeypatch.setattr(sr, "route_check_torch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        out = runner.execute_model(_so(0))
    assert out.shape == (6, 128)
    assert all(getattr(r.router, "_sdcguard_route", None) is None for r in runner.model.runners)
    assert any("routing check failed" in r.getMessage() for r in caplog.records)
