"""ServeGuard (inference) and the vLLM plugin (against a stand-in for vLLM's classes)."""
import contextlib
import math
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

from sdcguard import FaultInjector
from sdcguard.checksum import check_matmul
from sdcguard.inject import fault_model
from sdcguard.serve import ServeGuard


@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k,n,bias", [(64, 256, 512, False), (7, 300, 1000, True), (1, 64, 40, True)])
@pytest.mark.parametrize("corrupt", [None, "zero", "nan", "big"])
def test_same_answer_as_the_training_check(dt, m, k, n, bias, corrupt):
    torch.manual_seed(m + k + n)
    lin = nn.Linear(k, n, bias=bias).to(dt)
    x = torch.randn(m, k).to(dt)
    y = lin(x).detach()
    if corrupt == "zero":
        y[3 % m, 5:9] = 0
    elif corrupt == "nan":
        y[1 % m, n - 1] = float("nan")
    elif corrupt == "big":
        y[2 % m, n // 2] += 40 * y.float().abs().mean().to(dt)
    g = ServeGuard()
    slot = g.add_layer(lin, "lin")
    g.check(slot, x, y)
    z, flat, _noise = g.read_slot(slot)
    ref = check_matmul(x, lin.weight.t(), y, lin.bias)
    if math.isinf(ref.max_z):
        assert math.isinf(z)
    else:
        # the two compute the weight-side sums in a different fp32 order: clean scores (small
        # differences divided by the noise) can differ in the third digit
        tol = 1e-3 if corrupt is not None else 2e-2
        assert z == pytest.approx(ref.max_z, rel=tol, abs=tol)
    if corrupt is not None:
        cs = g.layers[slot]["cs"]
        assert divmod(int(flat) // 2, cs.nb) == (ref.row, ref.col_block)
        assert z > 6


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 384), nn.GELU(),
                         nn.Linear(384, 300)).to(torch.bfloat16)


def test_clean_inference_no_alerts_and_same_outputs():
    m = _model()
    x = torch.randn(9, 256).to(torch.bfloat16)
    ref = m(x)
    g = ServeGuard().attach(m)
    assert g.stats["layers"] == 3
    with torch.no_grad():
        assert torch.equal(m(x), ref)
    assert g.poll(wait=True) == []


def test_catches_compute_faults():
    m = _model()
    g = ServeGuard().attach(m)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=2).arm_on([m[2]])
    with torch.no_grad(), inj:
        m(torch.randn(32, 256).to(torch.bfloat16))
    inj.remove_hooks()
    ev = g.poll(wait=True)
    assert any(e.detail["where"] == "2" for e in ev)


def test_catches_a_weight_bit_flip_and_verify_weights_finds_it():
    m = _model()
    g = ServeGuard().attach(m)
    assert g.verify_weights() == []
    with torch.no_grad():
        m[4].weight.view(torch.int16)[130, 7] ^= (1 << 14)        # high exponent bit of one weight
        m(torch.randn(5, 256).to(torch.bfloat16))
    ev = g.poll(wait=True)
    assert [e.detail["where"] for e in ev] == ["4"]
    assert ev[0].detail["col_block"] == 1                         # output 130 is in block 1
    assert g.verify_weights() == ["4"]
    g.refresh()
    assert g.verify_weights() == []


def test_poll_resets_and_detach():
    m = _model()
    g = ServeGuard().attach(m)
    with torch.no_grad():
        m[0].weight.view(torch.int16)[0, 0] ^= (1 << 14)
        m(torch.randn(3, 256).to(torch.bfloat16))
    assert len(g.poll(wait=True)) >= 1
    assert g.poll(wait=True) == []                                # reset after reading
    g.detach()
    assert "forward" not in m[0].__dict__


def test_compile_one_graph_and_still_checks():
    m = _model()
    g = ServeGuard().attach(m)
    import torch._dynamo
    expl = torch._dynamo.explain(m)(torch.randn(8, 256).to(torch.bfloat16))
    assert expl.graph_count == 1 and expl.graph_break_count == 0
    cm = torch.compile(m, fullgraph=True, dynamic=True)
    try:
        with torch.no_grad():
            cm(torch.randn(8, 256).to(torch.bfloat16))
            cm(torch.randn(3, 256).to(torch.bfloat16))
    except Exception as e:  # noqa: BLE001  (as in test_compile.py)
        if "compiler" in str(e).lower():
            pytest.skip(f"no working C++ compiler for Inductor here: {e}")
        raise
    assert g.poll(wait=True) == []
    with torch.no_grad():
        m[2].weight.view(torch.int16)[9, 9] ^= (1 << 14)
        cm(torch.randn(5, 256).to(torch.bfloat16))
    assert [e.detail["where"] for e in g.poll(wait=True)] == ["2"]


# ---------------------------------------------------------------- the vLLM plugin, with stand-in classes
def _fake_vllm(monkeypatch):
    linear = types.ModuleType("vllm.model_executor.layers.linear")

    class UnquantizedLinearMethod:
        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)

    linear.UnquantizedLinearMethod = UnquantizedLinearMethod
    runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        def __init__(self, model):
            self.model = model

        def execute_model(self, x):
            return self.model(x)

    runner.GPUModelRunner = GPUModelRunner
    names = ["vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.v1", "vllm.v1.worker"]
    for n in names:
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", runner)
    sys.modules["vllm.model_executor.layers"].linear = linear
    sys.modules["vllm.v1.worker"].gpu_model_runner = runner
    return UnquantizedLinearMethod, GPUModelRunner


def test_vllm_plugin_off_by_default(monkeypatch):
    from sdcguard import vllm_plugin
    M, _ = _fake_vllm(monkeypatch)
    monkeypatch.delenv("SDCGUARD", raising=False)
    monkeypatch.setitem(vllm_plugin._STATE, "patched", False)
    orig = M.apply
    vllm_plugin.register()
    assert M.apply is orig


def _plugin_model(M, n_layers=12):
    class Lin(nn.Module):
        def __init__(self, k, n, prefix, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(n, k).to(torch.bfloat16) * 0.05)
            self.bias = nn.Parameter(torch.randn(n).to(torch.bfloat16)) if bias else None
            self.prefix, self.quant_method, self.skip_bias = prefix, M(), False

        def forward(self, x):
            return self.quant_method.apply(self, x, None if self.skip_bias else self.bias)

    torch.manual_seed(0)
    mods = []
    for i in range(n_layers):
        mods += [Lin(256, 256, f"layers.{i}.proj", bias=(i % 2 == 0)), nn.ReLU()]
    model = nn.Sequential(*mods)
    for mod in model.modules():
        if hasattr(mod, "quant_method"):
            mod.quant_method.process_weights_after_loading(mod)
    return model


def _plugin_env(monkeypatch, tmp_path, **extra):
    from sdcguard import vllm_plugin
    monkeypatch.setenv("SDCGUARD", "1")
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")
    monkeypatch.setenv("SDCGUARD_POLL_EVERY", "100000")       # these tests read the results themselves
    monkeypatch.setenv("SDCGUARD_POLL_SECONDS", "1e9")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    for k, v in extra.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setitem(vllm_plugin._STATE, "patched", False)
    monkeypatch.setitem(vllm_plugin._STATE, "guard", None)
    return vllm_plugin


def test_vllm_plugin_checks_every_layer(monkeypatch, tmp_path):
    M, Runner = _fake_vllm(monkeypatch)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path)
    vllm_plugin.register()
    vllm_plugin.register()                         # vLLM may call it more than once
    model = _plugin_model(M)
    g = vllm_plugin.guard()
    assert g.stats["layers"] == 12                 # every layer, not a sample
    r = Runner(model)
    x = torch.randn(9, 256).to(torch.bfloat16)
    r.execute_model(x)
    assert all(z > 0 for z in g.worst_z().values())          # positive control: every layer was checked
    assert vllm_plugin.poll(wait=True) == []
    for mod in model:
        if hasattr(mod, "skip_bias"):
            mod.skip_bias = True                   # tensor-parallel ranks > 0 don't add the bias
    r.execute_model(x)
    assert all(z > 0 for z in g.worst_z().values())
    assert vllm_plugin.poll(wait=True) == []
    with torch.no_grad():                          # layer 0: its input is x itself, so no column is all zeros
        model[0].weight.view(torch.int16)[5, 9] ^= (1 << 14)
    r.execute_model(x)
    ev = vllm_plugin.poll(wait=True)
    assert ev[0].detail["where"] == "layers.0.proj"
    assert ev[0].detail["weights_changed_since_load"] is True
    # an intended weight update followed by refresh(): quiet again, same tensors
    assert vllm_plugin.refresh() == 12
    r.execute_model(x)
    assert vllm_plugin.poll(wait=True) == []


def test_vllm_plugin_share(monkeypatch, tmp_path):
    M, _ = _fake_vllm(monkeypatch)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path, SDCGUARD_SHARE="0.25", SDCGUARD_OFFSET="1")
    vllm_plugin.register()
    _plugin_model(M)
    from sdcguard.serve import select_layer
    names = [lay["name"] for lay in vllm_plugin.guard().layers]
    expected = [f"layers.{i}.proj" for i in range(12) if select_layer(i, 0.25, 1)]
    assert names == expected and 2 <= len(names) <= 4


def test_share_spreads_over_layer_kinds():
    """A share must not pick the same projection of every block (every 4th layer would)."""
    from sdcguard.serve import select_layer
    for share in (0.5, 0.25, 0.125):
        picks = [i for i in range(112) if select_layer(i, share)]
        assert abs(len(picks) - 112 * share) <= 2
        kinds = {i % 4 for i in picks}
        assert kinds == {0, 1, 2, 3}
        assert max(picks) > 80 and min(picks) < 10                    # shallow and deep layers


def test_screen_tiny_run(tmp_path):
    from sdcguard.screen import replay, screen_one
    s = screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path), max_steps=3)
    assert s["result"] == "PASS" and s["matmuls_checked"] > 0
    f = screen_one(torch.device("cpu"), minutes=5, size="tiny", out_root=str(tmp_path / "f"), max_steps=6,
                   simulate_faults=0.05)
    assert f["result"] == "FAIL" and f["evidence"]
    r = replay(f["evidence"][0] + ".pt", times=3, device="cpu")
    assert r["recorded_output_check"]["ok"] is False
    assert r["recomputations_that_failed_the_check"] == 0          # the fault was simulated, not in this CPU
    assert os.path.exists(f["evidence"][0] + ".json")


def test_refresh_after_an_intended_weight_update_keeps_tensors_in_place():
    m = _model()
    g = ServeGuard().attach(m)
    cs = g.layers[0]["cs"]
    ptrs = [t.data_ptr() for t in cs.tensors()]
    with torch.no_grad():
        m[0].weight.mul_(1.5)                          # an intended update (e.g. RL weight sync)
        m(torch.randn(4, 256).to(torch.bfloat16))
    ev = g.poll(wait=True)
    assert ev and ev[0].detail["weights_changed_since_load"] is True
    g.refresh()
    assert [t.data_ptr() for t in g.layers[0]["cs"].tensors()] == ptrs     # same memory: CUDA graphs stay valid
    with torch.no_grad():
        m(torch.randn(4, 256).to(torch.bfloat16))
    assert g.poll(wait=True) == []


def test_alert_says_whether_weights_changed():
    m = _model()
    g = ServeGuard().attach(m)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=2).arm_on([m[0]])
    with torch.no_grad(), inj:
        m(torch.randn(32, 256).to(torch.bfloat16))
    inj.remove_hooks()
    ev = g.poll(wait=True)
    assert ev and all(e.detail["weights_changed_since_load"] is False for e in ev)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an NVIDIA GPU")


@contextlib.contextmanager
def _fp32_partial_sums():
    """Layer outputs made without a guard: keep bf16/fp16 partial sums in fp32, as ServeGuard does while it serves,
    so these GPU tests don't depend on what an earlier test left set (vllm_plugin._exact_reductions sets it for the
    rest of a process)."""
    m = torch.backends.cuda.matmul
    saved = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
    m.allow_bf16_reduced_precision_reduction = m.allow_fp16_reduced_precision_reduction = False
    try:
        yield
    finally:
        m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction = saved


@cuda
@_fp32_partial_sums()
def test_gpu_fast_serving_check_agrees_and_is_used():
    """On a GPU: the Triton check passes its first-use comparison, is used, and gives the PyTorch
    check's answers on clean and corrupted layers."""
    from sdcguard import _serve_fast
    from sdcguard.serve import STATE_COLS, LayerChecksums, check_static_impl
    dev = torch.device("cuda")
    for dt in (torch.bfloat16, torch.float16):
        assert _serve_fast.verified(dev, dt), _serve_fast.WHY
        for m, k, n in ((64, 2048, 5632), (3, 5632, 2048), (4096, 512, 300), (1, 64, 40)):
            torch.manual_seed(m + n)
            W = (torch.randn(n, k, device=dev) / k ** 0.5).to(dt)
            x = torch.randn(m, k, device=dev).to(dt)
            y = x @ W.t()
            cs = LayerChecksums(W, None, 128)
            for corrupt in (False, True, "inf_input"):
                yy, xx = y.clone(), x
                if corrupt == "inf_input":       # its row can't be judged: counted as unjudged, scored 0
                    xx = x.clone()
                    xx[m // 2, 0] = float("inf")
                    yy = xx @ W.t()
                elif corrupt:
                    yy[m // 2, n // 3] += 30.0
                ref = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                fast = torch.zeros(1, STATE_COLS, dtype=torch.int64, device=dev)
                check_static_impl(xx, yy, *cs.tensors(True), ref, 0, 6.0, 1e-20)
                _serve_fast.check(xx, yy, *cs.tensors(True), fast, 0, 6.0, 1e-20)
                (rz, rf, _), (fz, ff, _) = (_serve_fast.decode(*ref[0].tolist()), _serve_fast.decode(*fast[0].tolist()))
                rc, fc = ref[0, 2:].tolist(), fast[0, 2:].tolist()
                assert rc == fc, (dt, m, k, n, corrupt, rc, fc)          # same counts, exactly
                if corrupt == "inf_input":
                    assert rz <= 6 and fz <= 6 and rc == [1, 2 * cs.nb]
                elif corrupt:
                    assert rz > 6 and fz > 6 and rf // 2 == ff // 2 and rc == [1, 0]
                else:
                    assert rz <= 6 and fz <= 6 and abs(fz - rz) <= 0.05 + 0.02 * rz and rc == [1, 0]


@cuda
@pytest.mark.parametrize("fast", ["1", "0"])
def test_gpu_compiled_cuda_graphs_catch_a_flip(fast, monkeypatch):
    """torch.compile(mode="reduce-overhead") (CUDA graphs) with the Triton check inside (fast=1) or the
    PyTorch check traced into the graph (fast=0): quiet when clean, then a flipped weight bit is caught
    in a replayed graph."""
    monkeypatch.setenv("SDCGUARD_SERVE_FAST", fast)
    torch._dynamo.reset()
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512)).cuda().to(torch.bfloat16)
    g = ServeGuard().attach(m)
    assert all(g.uses_fast(s) == (fast == "1") for s in range(len(g.layers)))
    cm = torch.compile(m, mode="reduce-overhead", dynamic=False)
    x = torch.randn(16, 512, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        for _ in range(4):
            cm(x)
    assert g.poll(wait=True) == []
    assert all(z > 0 for z in g.worst_seen.values())
    cov = g.coverage_report()           # the counts are written inside the replayed graphs too
    assert cov["never_checked"] == [] and cov["unjudged_checksums"] == 0, cov
    assert all(v["checks"] >= 4 for v in g.counts().values()), g.counts()
    # The graph path itself, not an eager fallback: the next calls replay a CUDA graph, and each replay runs every
    # layer's check exactly once (outside review, Oct 20: a compiled call that silently ran eagerly passed before)
    hook = getattr(torch.cuda.graphs, "register_graph_replay_start_hook", None)
    if hook is None:                    # say so instead of passing without the replay proof (outside check, Oct 20)
        g.detach()
        torch._dynamo.reset()
        pytest.skip(f"torch {torch.__version__} has no graph-replay hook, so CUDA graph replay can't be proven here")
    replays = []
    handle = hook(lambda graph: replays.append(1))
    try:
        before = {k: v["checks"] for k, v in g.counts().items()}
        with torch.no_grad():
            for _ in range(3):
                cm(x)
        torch.cuda.synchronize()
        after = {k: v["checks"] for k, v in g.counts().items()}
        clean_replays = len(replays)
        # the flip is caught in calls that replay a graph too (the hook stays registered through it)
        with torch.no_grad():
            m[2].weight.view(torch.int16)[7, 9] ^= (1 << 14)
            for _ in range(2):
                cm(x)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    assert clean_replays >= 3, f"the compiled model ran without replaying a CUDA graph ({clean_replays} replays)"
    assert all(after[k] - before[k] == 3 for k in after), (before, after)
    assert len(replays) > clean_replays, "the calls after the flip replayed no CUDA graph"
    print(f"CUDA graph replays: {clean_replays} for 3 clean calls, {len(replays) - clean_replays} for 2 flipped calls")
    ev = g.poll(wait=True)
    assert {e.detail["where"] for e in ev} == {"2"}
    g.detach()
    torch._dynamo.reset()


def test_checksums_stay_fp32_under_a_bf16_default_dtype():
    """vLLM calls process_weights_after_loading inside set_default_torch_dtype(bf16)."""
    from sdcguard.serve import LayerChecksums
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        W = torch.randn(300, 64).to(torch.bfloat16)
        cs = LayerChecksums(W, torch.randn(300).to(torch.bfloat16), 128)
        assert all(t.dtype == torch.float32 for t in cs.tensors(True) + cs.tensors(False))
        g = ServeGuard()
        lin = nn.Linear(64, 300).to(torch.bfloat16)
        slot = g.add_layer(lin, "lin")
        x = torch.randn(5, 64).to(torch.bfloat16)
        g.check(slot, x, lin(x))
        assert g.read_slot(slot)[0] < 6
    finally:
        torch.set_default_dtype(old)


@cuda
@_fp32_partial_sums()
def test_gpu_no_kernel_compile_while_serving():
    """After add_layer() (which compiles the Triton check for the layer's shape), batches of any size
    must not compile anything (vLLM's JIT monitor flagged compiles during inference before this)."""
    from sdcguard import _serve_fast
    torch.manual_seed(0)
    lin = nn.Linear(2048, 5632, bias=False).cuda().to(torch.bfloat16)
    g = ServeGuard()
    slot = g.add_layer(lin, "lin")
    assert g.uses_fast(slot)

    def compiled():
        caches = getattr(_serve_fast._check, "device_caches", None)
        return sum(len(v[0]) for v in caches.values()) if caches else None
    before = compiled()
    for m in (1, 7, 64, 333, 1100, 2500, 5000, 9000):
        x = torch.randn(m, 2048, device="cuda").to(torch.bfloat16)
        g.check(slot, x, lin(x))
    assert g.read_slot(slot)[0] < 6
    assert before is None or compiled() == before          # (add_layer compiled the variant with counts)
    assert g.counts()["lin"] == {"checks": 8, "unjudged": 0}


# ---------------------------------------------------------------- what the plugin leaves out, and its plumbing
def test_vllm_plugin_reports_quantized_layers_it_cant_check(monkeypatch, tmp_path, caplog):
    import logging
    M, Runner = _fake_vllm(monkeypatch)
    linear = sys.modules["vllm.model_executor.layers.linear"]

    class Fp8LinearMethod:
        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)
    linear.Fp8LinearMethod = Fp8LinearMethod
    vllm_plugin = _plugin_env(monkeypatch, tmp_path)
    monkeypatch.setitem(vllm_plugin._STATE, "reported", False)
    vllm_plugin.register()
    model = _plugin_model(Fp8LinearMethod, n_layers=4)          # every layer quantized
    with caplog.at_level(logging.INFO, logger="sdcguard"):
        Runner(model).execute_model(torch.randn(3, 256).to(torch.bfloat16))
    r = vllm_plugin._STATE["layers"]
    assert {k: r[k] for k in ("checked", "total", "share_checked", "unchecked_by_method", "cant_check")} == {
        "checked": 0, "total": 4, "share_checked": 0.0, "unchecked_by_method": {"Fp8LinearMethod": 4}, "cant_check": 4}
    assert any("Nothing in this model is being checked" in r.message for r in caplog.records)
    # SDCGUARD_REQUIRE=1 refuses to serve it
    monkeypatch.setenv("SDCGUARD_REQUIRE", "1")
    monkeypatch.setitem(vllm_plugin._STATE, "reported", False)
    with pytest.raises(RuntimeError, match="SDCGUARD_REQUIRE"):
        Runner(model).execute_model(torch.randn(3, 256).to(torch.bfloat16))


def test_vllm_plugin_mixed_model_counts_both(monkeypatch, tmp_path):
    M, Runner = _fake_vllm(monkeypatch)

    class AWQLinearMethod:                     # a quantized method: vLLM's are their own classes
        def process_weights_after_loading(self, layer):
            pass

        def apply(self, layer, x, bias=None):
            return torch.nn.functional.linear(x, layer.weight, bias)
    vllm_plugin = _plugin_env(monkeypatch, tmp_path)
    monkeypatch.setitem(vllm_plugin._STATE, "reported", False)
    vllm_plugin.register()
    model = nn.Sequential(_plugin_model(M, n_layers=3), _plugin_model(AWQLinearMethod, n_layers=1))
    Runner(model).execute_model(torch.randn(3, 256).to(torch.bfloat16))
    r = vllm_plugin._STATE["layers"]
    assert r["checked"] == 3 and r["unchecked_by_method"] == {"AWQLinearMethod": 1}
    assert vllm_plugin.guard().stats["layers_unchecked"] == 1


@pytest.mark.parametrize("text,share", [("0.125", 0.125), ("1/8", 0.125), ("12.5%", 0.125), ("junk", 1.0),
                                        ("2", 1.0)])
def test_vllm_share_formats(text, share):
    from sdcguard.vllm_plugin import _share
    assert _share(text) == share


def test_vllm_metrics_port_follows_local_rank(monkeypatch):
    from sdcguard.vllm_plugin import _metrics_port
    monkeypatch.setenv("SDCGUARD_METRICS_PORT", "9400")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert _metrics_port() == 9403
    monkeypatch.delenv("LOCAL_RANK")
    assert _metrics_port() == 9400


def test_a_taken_metrics_port_keeps_the_webhook(monkeypatch, tmp_path):
    pytest.importorskip("prometheus_client")
    import socket
    s = socket.socket()
    s.bind(("0.0.0.0", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        M, Runner = _fake_vllm(monkeypatch)
        vllm_plugin = _plugin_env(monkeypatch, tmp_path, SDCGUARD_METRICS_PORT=str(port),
                                  SDCGUARD_WEBHOOK="https://hooks.example.invalid/T000")
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        vllm_plugin.register()
        _plugin_model(M, n_layers=1)
        g = vllm_plugin.guard()
        assert g.webhook is not None and g.metrics is None
    finally:
        s.close()


def test_attaching_inside_inference_mode_keeps_alerts():
    from sdcguard.serve import ServeGuard
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).bfloat16()
    with torch.inference_mode():
        g = ServeGuard().attach(model)
    with torch.no_grad():
        model[0].weight.view(torch.int16)[0, 0] ^= 0x4000
    with torch.inference_mode():
        model(torch.randn(8, 64).bfloat16())
    assert len(g.poll(wait=True)) == 1 and g.stats["alerts"] == 1


def test_a_webhook_that_cant_start_a_thread_tries_again_next_time(monkeypatch):
    import threading
    from sdcguard.alerts import Webhook

    class Event:
        detail = {"where": "layer"}
        identity = {"gpu_uuid": "gpu-1"}

        def to_json(self):
            return '{"message": "fault"}'
    w = Webhook("https://example.invalid", fmt="json")
    real = threading.Thread.start
    monkeypatch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(OSError("no threads")))
    w(Event())                                    # never raises into training or serving
    assert not w._sending and w.failed == 1
    monkeypatch.setattr(threading.Thread, "start", real)
    started = []
    monkeypatch.setattr(Webhook, "_post", lambda self, body, key=None: started.append(key))
    w(Event())
    assert started == [("layer", "gpu-1")] or ("layer", "gpu-1") in w._sending
