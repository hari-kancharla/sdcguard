"""The Triton check must give the same answers as the plain PyTorch check.

Runs tests/_fast_compare.py in a separate process with Triton's CPU interpreter
(TRITON_INTERPRET=1 has to be set before Triton is imported). On a GPU machine the self-test
runs the same comparison on the real GPU.
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("native", ["0", "1"])
def test_triton_kernels_match_plain_check(native):
    """native=1: the bf16 tensor-core math of NVIDIA GPUs (sm80+), with its products done in fp32
    because Triton's CPU interpreter gets bf16 dots wrong."""
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_TEST_NATIVE=native,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_fast_compare.py"), "cpu"], env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0 and "ALL AGREE" in p.stdout, p.stdout + p.stderr


def test_triton_serving_check_matches_plain_serving_check():
    """sdcguard._serve_fast (one Triton kernel per layer) vs serve.check_static_impl, on clean and
    corrupted layers of many shapes, bf16 and fp16, both block-sum layouts, plus a guard end to end."""
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_fast_compare.py")], env=env,
                       capture_output=True, text=True, timeout=1500)
    assert p.returncode == 0 and "ALL AGREE" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]
