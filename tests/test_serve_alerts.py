"""What a serving alert says about its symptom: whether the output held Inf/NaN or finite values too large for its
inputs. It goes to the JSONL log, the webhook and the person reading them, so it may not claim more than the check
saw."""
import pytest
import torch
import torch.nn as nn

from sdcguard.alerts import describe
from sdcguard.serve import ServeGuard


def _alert(value):
    torch.manual_seed(0)
    lin = nn.Linear(64, 256).bfloat16()
    g = ServeGuard(exact_reductions=False)
    slot = g.add_layer(lin, "proj")
    x = torch.randn(4, 64).bfloat16()
    with torch.no_grad():
        y = lin(x)
    y[1, 7] = value
    g.check(slot, x, y)
    ev = g.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["max_z"] == float("inf")
    return ev[0]


def test_a_finite_but_impossibly_large_output_is_not_reported_as_inf_or_nan():
    """detail["nonfinite"] was `not isfinite(score)`: a finite output too large for its inputs (the check's
    `imposs` rule, or values whose squares overflow fp32) scores inf too, so the event said nonfinite=True and
    describe() "produced Inf/NaN the inputs can't explain" although every output value was finite. The check now
    keeps NaN as the noise of a checksum whose output sum is Inf/NaN, which the report reads with the score (no
    extra read of the GPU): 1e15 and 1e30 (finite in bf16) are finite values too large, Inf and NaN are Inf/NaN."""
    for value in (1e15, 1e30):
        ev = _alert(value)
        assert ev.detail["nonfinite"] is False and "Inf/NaN" not in describe(ev), (ev.detail, describe(ev))
        assert "finite values far larger than its inputs allow" in describe(ev)
    for value in (float("inf"), float("nan")):
        ev = _alert(value)
        assert ev.detail["nonfinite"] is True
        assert "produced Inf/NaN the inputs can't explain" in describe(ev), describe(ev)


def test_alert_metrics_carry_the_gpu_uuid_as_dcgm_exporter_labels_it():
    """sdcguard_alerts put the GPU's UUID in a label called gpu, which in dcgm-exporter's series is NVIDIA's device
    index: a join with the GPU's DCGM metrics on gpu matched nothing (or the wrong GPU). The UUID goes in UUID."""
    pytest.importorskip("prometheus_client")
    from prometheus_client import CollectorRegistry, generate_latest
    from sdcguard.alerts import Metrics
    from sdcguard.events import SDCEvent

    class Guard:
        layers, stats = [object()], {}

        def uses_fast(self, slot):
            return False
    r = CollectorRegistry()
    m = Metrics(registry=r)
    uuid = "GPU-1b2c3d4e-0000-1111-2222-333344445555"
    m.observe(Guard(), [("model.layers.0.mlp", 9.0)],
              [SDCEvent(detector="serve_checksum", step=None, detail={"where": "model.layers.0.mlp"},
                        identity={"gpu_uuid": uuid})])
    lines = [ln for ln in generate_latest(r).decode().splitlines() if ln.startswith("sdcguard_alerts_total")]
    assert lines == [f'sdcguard_alerts_total{{UUID="{uuid}",layer="model.layers.0.mlp"}} 1.0'], lines
