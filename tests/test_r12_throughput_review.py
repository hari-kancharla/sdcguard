"""Stored measurements and presentation must preserve a slow, positive result."""
from types import SimpleNamespace

import pytest
import torch

from sdcguard import check


def test_slow_matmul_keeps_measured_throughput(monkeypatch):
    # Fixed timings make the regression independent of the machine's speed.
    monkeypatch.setattr(check, "_mat_size", lambda _: 128)
    monkeypatch.setattr(check, "_elapsed", lambda *_: 0.5)
    monkeypatch.setattr(check, "time", SimpleNamespace(time=lambda: 0.0))
    result = check.matmul_speed(torch.device("cpu"), torch.float32, seconds=0)
    expected = 2 * 128**3 / 0.5 / 1e12
    assert expected < 0.005
    assert result["windows"] == 3
    for field in ("tflops", "first_tflops", "last_tflops"):
        assert result[field] == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("value", [8.388608e-6, 0.004, 0.04])
def test_small_positive_measurement_does_not_display_as_zero(value):
    displayed = float(check._n(value).replace(",", ""))
    assert displayed > 0
    assert displayed == pytest.approx(value, rel=0.005, abs=0)
