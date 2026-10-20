"""Independent serving lifecycle and release-gate regressions for source 12e23f5.

These tests use CPU tensors. The asynchronous-poll test substitutes event/stream objects to inspect which
device owns the completion fence; it does not claim to execute multiple GPUs.
"""
from __future__ import annotations

import torch

from sdcguard.serve import COL_CHECKS, COL_UNJUDGED, STATE_COLS, ServeGuard


def test_detach_does_not_give_a_new_model_the_old_models_counts():
    guard = ServeGuard()
    first = torch.nn.Sequential(torch.nn.Linear(4, 8)).bfloat16()
    guard.attach(first)
    with torch.no_grad():
        first(torch.ones(2, 4, dtype=torch.bfloat16))
    guard.poll(wait=True)
    assert guard.counts()["0"]["checks"] == 1
    guard.detach()
    second = torch.nn.Sequential(torch.nn.Linear(4, 8)).bfloat16()
    guard.attach(second)
    try:
        assert guard.counts()["0"] == {"checks": 0, "unjudged": 0}
        assert guard.coverage_report()["never_checked"] == ["0"]
    finally:
        guard.detach()


def test_detach_collects_unread_counts_before_releasing_the_layers():
    layer = torch.nn.Linear(4, 8).bfloat16()
    guard = ServeGuard().attach(layer)
    with torch.no_grad():
        layer(torch.ones(2, 4, dtype=torch.bfloat16))
    guard.detach()
    assert guard.stats["checks_executed"] == 1
    assert guard.stats["layers"] == 0


def test_detach_invalidates_engine_op_arguments():
    layer = torch.nn.Linear(4, 8).bfloat16()
    guard = ServeGuard(exact_reductions=False)
    slot = guard.add_layer(layer, "engine_layer")
    layer._sdcguard_args = (guard.op_args(slot, False), guard.op_args(slot, True))
    layer._sdcguard_trace = True
    guard.detach()
    assert getattr(layer, "_sdcguard_args", None) is None
    assert not getattr(layer, "_sdcguard_trace", False)


def test_async_poll_records_completion_on_the_states_device_stream(monkeypatch):
    guard = ServeGuard(exact_reductions=False)
    dev = torch.device("cuda:1")
    guard._state[dev] = torch.zeros((1, STATE_COLS), dtype=torch.int64)
    guard.layers.append({"name": "layer", "device": dev})
    stream = object()
    calls = []

    class Event:
        def record(self, used_stream=None):
            calls.append(used_stream)

    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream if device == dev else None)
    guard.poll(wait=False)
    assert calls == [stream], "An event on the current GPU cannot fence a copy from a different GPU."


def test_pending_counts_are_not_consumed_or_counted_twice_by_snapshot_then_poll():
    guard = ServeGuard(exact_reductions=False)
    layer = torch.nn.Linear(4, 8).bfloat16()
    guard.add_layer(layer, "layer")
    dev = layer.weight.device
    host = torch.zeros((1, STATE_COLS), dtype=torch.int64)
    host[0, COL_CHECKS:COL_UNJUDGED + 1] = torch.tensor([2, 3])
    guard._state[dev][0, COL_CHECKS:COL_UNJUDGED + 1] = torch.tensor([4, 5])

    class Event:
        def synchronize(self):
            pass

    guard._host[dev] = (host, Event())
    try:
        expected = {"layer": {"checks": 6, "unjudged": 8}}
        assert guard.counts() == expected
        assert guard.counts() == expected
        guard.poll(wait=True)
        assert guard.counts() == expected
        guard.poll(wait=True)
        assert guard.counts() == expected
    finally:
        guard.detach()


def test_device_refresh_keeps_unread_counts_from_the_old_state():
    guard = ServeGuard(exact_reductions=False)
    layer = torch.nn.Linear(4, 8).bfloat16()
    guard.add_layer(layer, "layer")
    dev = layer.weight.device
    # CPU tensors stand in for old and new device buffers, so no second device is required to exercise
    # bookkeeping during a migration. The code takes this branch whenever the weight's device changes.
    old_dev = torch.device("cpu:1")
    guard.layers[0]["device"] = old_dev
    guard._state[old_dev] = guard._state.pop(dev)
    guard._state[old_dev][0, COL_CHECKS:COL_UNJUDGED + 1] = torch.tensor([2, 3])
    pending = torch.zeros((1, STATE_COLS), dtype=torch.int64)
    pending[0, COL_CHECKS:COL_UNJUDGED + 1] = torch.tensor([4, 5])

    class Event:
        def synchronize(self):
            pass

    guard._host[old_dev] = (pending, Event())
    try:
        guard.refresh_layer(0)
        assert guard.counts() == {"layer": {"checks": 6, "unjudged": 8}}
        guard.poll(wait=True)
        assert guard.counts() == {"layer": {"checks": 6, "unjudged": 8}}
    finally:
        guard.detach()


def test_shape_refresh_reports_pending_alert_with_its_original_layout():
    guard = ServeGuard(exact_reductions=False)
    layer = torch.nn.Linear(4, 300, bias=False).bfloat16()
    slot = guard.add_layer(layer, "layer")
    x = torch.ones(4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        y = layer(x)
        y[2, 140] += 1000
        guard.check(slot, x, y)
        layer.weight = torch.nn.Parameter(torch.ones(64, 4, dtype=torch.bfloat16))
    try:
        guard.refresh_layer(slot)
        guard.poll(wait=True)
        events = guard.log.events
        assert len(events) == 1
        assert (events[0].detail["row"], events[0].detail["col_block"]) == (2, 1)
    finally:
        guard.detach()
