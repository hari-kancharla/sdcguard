"""The recompute check's one-pass fingerprint kernel (sdcguard/_fingerprint.py).

The kernel cases run tests/_fingerprint_compare.py in a separate process under Triton's CPU interpreter
(TRITON_INTERPRET=1 has to be set before Triton is imported); on a GPU the same script runs on the hardware.
"""
import os
import subprocess
import sys

import pytest
import torch

from sdcguard import _fingerprint
from sdcguard.recompute import _fingerprint_torch, fingerprint

HERE = os.path.dirname(os.path.abspath(__file__))


def test_kernel_matches_the_pytorch_sums_in_the_interpreter():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_fingerprint_compare.py"), "cpu"], env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0 and "ALL AGREE" in p.stdout, p.stdout[-4000:] + p.stderr[-4000:]


@pytest.mark.skipif(os.environ.get("TRITON_INTERPRET") == "1", reason="checks the case without the interpreter")
def test_plain_cpu_uses_the_pytorch_sums():
    t = torch.randn(1000, dtype=torch.bfloat16)
    assert not _fingerprint.usable(t.device, _fingerprint_torch)
    assert torch.equal(fingerprint(t), _fingerprint_torch(t.view(torch.int16)))


def test_a_disagreeing_kernel_is_never_used(monkeypatch):
    """The probe comparison has to pass before the kernel is used on a device."""
    monkeypatch.setattr(_fingerprint, "_launch", lambda bits: torch.zeros(3, dtype=torch.int64))
    ok, why = _fingerprint._probe_ok(torch.device("cpu"), _fingerprint_torch)
    assert not ok and "disagreed" in why


def test_switches(monkeypatch):
    monkeypatch.setenv("SDCGUARD_RECOMPUTE_FAST", "0")
    assert not _fingerprint.enabled()
    monkeypatch.setenv("SDCGUARD_RECOMPUTE_FAST", "1")
    monkeypatch.setenv("SDCGUARD_FAST", "0")
    assert not _fingerprint.enabled()
