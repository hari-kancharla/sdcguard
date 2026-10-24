"""Which requests an alert concerns: the step a captured row came from (ServeGuard.mark_step), the reads' step
windows, the last clean read of a layer, and the vLLM plugin's map of a step's rows to its requests. On the CPU,
with the PyTorch check and stand-ins for vLLM's runner and scheduler output."""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from sdcguard.serve import ServeGuard
from _vllm_standins import install_fake_vllm, load, plugin_env, Lin


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 64)).to(torch.bfloat16)


def _faulty_linear(monkeypatch, plan):
    """F.linear with y[row, col] += 40 on the calls listed in plan (call number -> row)."""
    real, n = F.linear, [0]

    def linear(*a, **k):
        y = real(*a, **k)
        n[0] += 1
        if n[0] in plan:
            y = y.clone()
            y.view(-1, y.shape[-1])[plan[n[0]], 77] += 40.0
        return y
    monkeypatch.setattr(F, "linear", linear)


def _steps(m, g, first, last, x):
    with torch.no_grad():
        for s in range(first, last + 1):
            g.mark_step(s)
            m(x)


def test_a_captured_row_carries_its_step_and_reads_carry_their_window(monkeypatch):
    m = _model()
    g = ServeGuard().attach(m)
    x = torch.randn(8, 64).to(torch.bfloat16)
    _faulty_linear(monkeypatch, {7: 5})            # 2 linear calls per step: call 7 is step 4's first layer
    _steps(m, g, 1, 6, x)
    (ev,) = g.poll(wait=True)
    d = ev.detail
    assert d["captured"]["step"] == 4 and d["captured"]["row"] == 5 and d["read_window_steps"] == [1, 6]
    assert d["first_alert_for_layer"] is True and d["last_clean_read_steps"] is None
    _steps(m, g, 7, 9, x)                          # a clean read of the layer
    assert g.poll(wait=True) == []
    _faulty_linear(monkeypatch, {3: 2})            # (counting starts again) step 11's first layer, row 2
    _steps(m, g, 10, 12, x)
    (ev,) = g.poll(wait=True)
    d = ev.detail
    assert d["captured"]["step"] == 11 and d["captured"]["row"] == 2 and d["read_window_steps"] == [10, 12]
    assert d["last_clean_read_steps"] == [7, 9] and d["first_alert_for_layer"] is False


def test_without_marked_steps_nothing_changes():
    m = _model()
    g = ServeGuard().attach(m)
    with torch.no_grad():
        m(torch.randn(4, 64).to(torch.bfloat16))
    assert g.poll(wait=True) == [] and g._marked_step is None


class _Sched:
    def __init__(self, counts):
        self.num_scheduled_tokens = counts


def _runner_module(monkeypatch, plan, padded=False):
    """A V2-like GPUModelRunner: execute_model(scheduler_output) runs the model on one row per scheduled token and
    keeps the step's batch (req_ids in batch order) on execute_model_state, as vLLM 0.32's does."""
    import sys
    mod = types.ModuleType("vllm.v1.worker.gpu.model_runner")

    class GPUModelRunner:
        def __init__(self, model):
            self.model, self.execute_model_state = model, None

        def execute_model(self, scheduler_output, dummy_run=False, **kwargs):
            ids = sorted(scheduler_output.num_scheduled_tokens)          # vLLM sorts the batch
            self.execute_model_state = types.SimpleNamespace(input_batch=types.SimpleNamespace(req_ids=ids))
            if padded:
                self.execute_model_state.input_batch.num_tokens_after_padding = \
                    sum(scheduler_output.num_scheduled_tokens.values()) + 2
            rows = sum(scheduler_output.num_scheduled_tokens.values())
            return self.model(torch.randn(rows + 2, 256).to(torch.bfloat16))    # + 2 rows of graph padding

        def _dummy_run(self, num_tokens, **kwargs):     # V2: a batch of made-up requests through execute_model
            return self.execute_model(_Sched({f"_dummy_req_{i}": 1 for i in range(num_tokens)}), dummy_run=True)
    mod.GPUModelRunner = GPUModelRunner
    for n in ("vllm.v1.worker.gpu",):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", mod)
    return GPUModelRunner


def test_the_plugin_names_the_request_whose_row_alarmed(monkeypatch, tmp_path):
    M, _ = install_fake_vllm(monkeypatch.setitem)
    Runner = _runner_module(monkeypatch, None)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(nn.Sequential(Lin(M, 256, 256, "model.layers.0.proj")))
    r = Runner(model)
    reqs = {"chat-a-1a2b3c4d": 3, "chat-b-5e6f7a8b": 1, "chat-c-9c0d1e2f": 4}   # rows: a 0-2, b 3, c 4-7
    _faulty_linear(monkeypatch, {3: 3})            # step 3, row 3: request b
    for _ in range(4):
        r.execute_model(_Sched(reqs))
    (ev,) = plugin.poll(wait=True)
    d = ev.detail
    assert d["captured"]["step"] == 3 and d["directly_implicated"] == ["chat-b"] and d["attribution"].startswith("exact")
    assert d["potentially_exposed"]["read_window"]["requests"] == ["chat-a", "chat-b", "chat-c"]
    assert any("prefix-cache" in u for u in d["unknown_impact"])
    # what a replay at the real shape needs: the layer's matmul and the step's rows (this stand-in has no padding)
    assert d["matmul"]["out_features"] == 256 and d["matmul"]["in_features"] == 256
    assert d["step_rows"] == {"tokens": 8, "with_padding": None}
    _faulty_linear(monkeypatch, {1: 9})            # a padding row
    r.execute_model(_Sched(reqs))
    (ev,) = plugin.poll(wait=True)
    assert ev.detail["directly_implicated"] == [] and "padding" in ev.detail["attribution"]


def test_the_read_window_and_the_requests_since_the_last_clean_read(monkeypatch, tmp_path):
    M, _ = install_fake_vllm(monkeypatch.setitem)
    Runner = _runner_module(monkeypatch, None)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(nn.Sequential(Lin(M, 256, 256, "model.layers.0.proj")))
    r = Runner(model)
    r.execute_model(_Sched({"a-00000000": 2}))
    assert plugin.poll(wait=True) == []            # a clean read: step 1
    g = plugin.guard()
    _faulty_linear(monkeypatch, {1: 0})
    r.execute_model(_Sched({"b-11111111": 2}))
    r.execute_model(_Sched({"c-22222222": 2}))
    caps = g._take_captures(g.layers[0]["device"])            # lose the captured row: the window is named instead
    assert caps and caps[0]["step"] == 2
    (ev,) = plugin.poll(wait=True)
    d = ev.detail
    assert d["directly_implicated"] is None and "only the read's window" in d["attribution"]
    assert d["potentially_exposed"]["read_window"] == {"steps": [2, 3], "requests": ["b", "c"], "complete": True}
    since = d["potentially_exposed"]["since_last_clean_read"]
    assert since["steps"] == [2, 3] and since["count"] == 2 and since["complete"]


def test_attribution_is_unavailable_rather_than_guessed(monkeypatch):
    from sdcguard import vllm_plugin as p
    monkeypatch.setitem(p._STATE, "step_log", None)
    ev = types.SimpleNamespace(detail={"captured": {"step": 1, "row": 0}})
    p._attribute(ev)
    assert ev.detail["directly_implicated"] is None and "unavailable" in ev.detail["attribution"]
    runner = types.SimpleNamespace(execute_model_state=types.SimpleNamespace(
        input_batch=types.SimpleNamespace(req_ids=["x-00000000"])))
    assert p._batch_rows(runner, (_Sched({"x-00000000": 2, "y-11111111": 1}),), {}) is None   # order unknown
    assert p._batch_rows(runner, (object(),), {}) is None
    split = types.SimpleNamespace(parallel_config=types.SimpleNamespace(enable_dbo=True))
    p._record_batch(split, 5, (_Sched({"x-00000000": 1}),), {})
    ev = types.SimpleNamespace(detail={"captured": {"step": 5, "row": 0}})
    p._attribute(ev)
    assert ev.detail["directly_implicated"] is None and "microbatches" in ev.detail["attribution"]
    assert p._external("chat-1-deadbeef") == "chat-1" and p._external("plain") == "plain"


def test_the_step_rows_with_padding_come_from_vllms_batch(monkeypatch, tmp_path):
    M, _ = install_fake_vllm(monkeypatch.setitem)
    Runner = _runner_module(monkeypatch, None, padded=True)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(nn.Sequential(Lin(M, 256, 256, "model.layers.0.proj")))
    r = Runner(model)
    _faulty_linear(monkeypatch, {1: 1})
    r.execute_model(_Sched({"q-00000000": 3}))
    (ev,) = plugin.poll(wait=True)
    assert ev.detail["step_rows"] == {"tokens": 3, "with_padding": 5} and ev.detail["directly_implicated"] == ["q"]


def test_a_row_captured_in_a_dummy_batch_doesnt_name_the_last_steps_requests(monkeypatch, tmp_path):
    """vLLM's dummy batches (no request's tokens) aren't model steps; the linear checks still check them, and a row
    they capture there carries the last step's number, so its alert names no request as the one it concerns."""
    M, _ = install_fake_vllm(monkeypatch.setitem)
    Runner = _runner_module(monkeypatch, None)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(nn.Sequential(Lin(M, 256, 256, "model.layers.0.proj")))
    r = Runner(model)
    reqs = {"chat-a-1a2b3c4d": 3}
    r.execute_model(_Sched(reqs))
    r.execute_model(_Sched(reqs))
    assert plugin.poll(wait=True) == []
    _faulty_linear(monkeypatch, {1: 1})            # the dummy batch's linear call, row 1
    r._dummy_run(4)
    assert plugin._STATE["steps"] == 2 and plugin._STATE["dummy_runs"] == 1
    assert list(plugin._STATE["dummy_after"]) == [2] and [e[0] for e in plugin._STATE["step_log"]] == [1, 2]
    r.execute_model(_Sched(reqs))
    (ev,) = plugin.poll(wait=True)
    d = ev.detail
    assert d["captured"]["step"] == 2 and d["directly_implicated"] is None
    assert "dummy batch" in d["attribution"] and "may be from it" in d["attribution"]
    # the checks that judge model steps only (attention, routing, FP8) never run in a dummy batch: exact as before
    ev2 = types.SimpleNamespace(detector="serve_attn", detail={"captured": {"step": 2, "row": 1}})
    plugin._attribute(ev2)
    assert ev2.detail["directly_implicated"] == ["chat-a"] and ev2.detail["attribution"].startswith("exact")


def test_the_v1_runners_dummy_batches_are_seen_too(monkeypatch, tmp_path):
    """The V1 runner makes its dummy batches (and CUDA-graph captures) in _dummy_run without execute_model."""
    M, GM = install_fake_vllm(monkeypatch.setitem)
    monkeypatch.setattr(GM, "_dummy_run", lambda self, n: self.model(torch.randn(n, 256).to(torch.bfloat16)),
                        raising=False)
    plugin = plugin_env(monkeypatch, tmp_path)
    plugin.register()
    model = load(nn.Sequential(Lin(M, 256, 256, "model.layers.0.proj")))
    r = GM(model)
    r.execute_model(torch.randn(4, 256).to(torch.bfloat16))
    r._dummy_run(8)
    r._dummy_run(8)
    assert plugin._STATE["steps"] == 1 and plugin._STATE["dummy_runs"] == 2 and list(plugin._STATE["dummy_after"]) == [1]


def test_new_weights_are_taken_before_a_dummy_batch_but_never_inside_a_capture(monkeypatch, tmp_path):
    plugin = plugin_env(monkeypatch, tmp_path)
    refreshed = []
    norms = types.SimpleNamespace(refresh=lambda: refreshed.append("norms"))
    monkeypatch.setitem(plugin._STATE, "guard", types.SimpleNamespace(norms=norms, moe=None))
    monkeypatch.setitem(plugin._STATE, "norms_stale", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    plugin._dummy_seen()                            # inside a capture: later
    assert refreshed == [] and plugin._STATE["norms_stale"] is True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    plugin._dummy_seen()
    assert refreshed == ["norms"] and plugin._STATE["norms_stale"] is False
    plugin._dummy_seen()
    assert refreshed == ["norms"] and plugin._STATE["dummy_runs"] == 3
