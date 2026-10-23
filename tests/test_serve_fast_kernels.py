"""The serving check's two Triton designs (_serve_fast: v1, cuBLAS then _check; v3, cuBLAS then _sums and _score)
against the plain PyTorch check, in Triton's CPU interpreter, and how check() picks between them. tests/test_fast.py
runs the full comparison set (tests/_serve_fast_compare.py) as served (v1 below V3_FROM_ROWS rows, v3 from there
on), and test_v3_runs_the_full_comparison_set with v3 on every case."""
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("triton")

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = r'''
import json, sys, torch
from sdcguard import _serve_fast
from sdcguard.checksum import _split3
from sdcguard.serve import STATE_COLS, LayerChecksums, check_static_impl
g = torch.Generator().manual_seed(5)
rows = []
# (m, k, n, bias): one row; a partial last block; a long sum over k; many rows
for m, k, n, bias in [(1, 1024, 300, True), (5, 64, 129, False), (3, 2048, 1000, True), (70, 96, 260, False)]:
    for dt in (torch.bfloat16, torch.float16):
        W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dt)
        b = torch.randn(n, generator=g).to(dt) if bias else None
        x = torch.randn(m, k, generator=g).to(dt)
        y = torch.nn.functional.linear(x, W, b)
        cs = LayerChecksums(W, b, 128)
        cs.parts = _split3(cs.parts / cs.scale.view(1, -1), dt).contiguous()    # the GPU's three parts
        for corrupt in (None, "big", "nan"):
            yy = y.clone()
            r, c = (m * 2) // 3, (n * 3) // 5
            if corrupt == "big":
                yy[r, c] += 40.0
            elif corrupt == "nan":
                yy[r, c] = float("nan")
            ref = torch.zeros(1, STATE_COLS, dtype=torch.int64)
            fast = torch.zeros(1, STATE_COLS, dtype=torch.int64)
            check_static_impl(x, yy, *cs.tensors(True), ref, 0, 6.0, 1e-20)
            _serve_fast.check(x, yy, *cs.tensors(True), fast, 0, 6.0, 1e-20)
            rows.append([m, k, n, str(dt), corrupt, list(_serve_fast.decode(*ref[0].tolist())),
                         list(_serve_fast.decode(*fast[0].tolist())), ref[0, 2:].tolist(), fast[0, 2:].tolist()])
print(json.dumps(rows))
'''


@pytest.mark.parametrize("kernel", ["v1", "v3"])
def test_each_serving_kernel_gives_the_plain_checks_answers(kernel):
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_SIGN_SEED="1",
               SDCGUARD_SERVE_KERNEL=kernel,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stderr[-3000:]
    rows = json.loads(p.stdout.strip().splitlines()[-1])
    assert len(rows) == 24
    for m, k, n, dt, corrupt, (rz, rf, _), (fz, ff, _), rc, fc in rows:
        where = (m, k, n, dt, corrupt, rz, fz, rf, ff)
        assert rc == fc == [1, 0], where
        if corrupt is None:
            assert rz <= 6.0 and fz <= 6.0 and abs(fz - rz) <= 0.05 + 0.02 * rz, where
        else:
            assert rz > 6.0 and fz > 6.0 and rf // 2 == ff // 2, where


@pytest.mark.parametrize("kernel", ["v3"])
def test_a_forced_design_runs_the_full_comparison_set(kernel):
    """tests/_serve_fast_compare.py with v3 on every case: clean, large and small changes, NaN, Inf, finite outputs
    too large for the inputs, infinite inputs, fp16 overflow, partial blocks, bias, the capture, and a guard end to
    end (the default run in tests/test_fast.py takes v1 below V3_FROM_ROWS rows)."""
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_SIGN_SEED="1",
               SDCGUARD_SERVE_KERNEL=kernel,
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_fast_compare.py")], env=env, capture_output=True,
                       text=True, timeout=2400)
    assert p.returncode == 0 and "ALL AGREE" in p.stdout, p.stdout[-3000:] + p.stderr[-3000:]


def test_the_kernel_is_picked_by_rows_unless_forced(monkeypatch):
    from sdcguard import _serve_fast
    monkeypatch.delenv("SDCGUARD_SERVE_KERNEL", raising=False)
    n = _serve_fast.V3_FROM_ROWS
    assert n == 96                       # (the October 22 timing runs, docs/METHOD.md)
    assert _serve_fast.mode() == "auto" and _serve_fast.kernels() == ("v1", "v3")
    assert [_serve_fast.kernel_for(m) for m in (1, n - 1, n, 16384)] == ["v1", "v1", "v3", "v3"]
    assert _serve_fast.tile_heights("v1") == (1, 2, 4, 8, 16) and _serve_fast.tile_heights("v3") == (None,)
    for forced in ("v1", "v3"):
        monkeypatch.setenv("SDCGUARD_SERVE_KERNEL", forced)
        assert _serve_fast.kernels() == (forced,)
        assert {_serve_fast.kernel_for(m) for m in (1, n, 16384)} == {forced}
    for other in ("v5", "something else"):          # (v5 was removed: anything else is the default)
        monkeypatch.setenv("SDCGUARD_SERVE_KERNEL", other)
        assert _serve_fast.mode() == "auto"


def test_check_runs_the_design_for_its_batch_size(monkeypatch):
    """check() calls v1 below V3_FROM_ROWS rows and v3 from there on (kern= forces one)."""
    from sdcguard import _serve_fast
    monkeypatch.delenv("SDCGUARD_SERVE_KERNEL", raising=False)
    seen = []
    monkeypatch.setattr(_serve_fast, "check_v1", lambda x, y, *a, **k: seen.append(("v1", y.shape[0])))
    monkeypatch.setattr(_serve_fast, "check_v3", lambda x, y, *a, **k: seen.append(("v3", y.shape[0])))
    import torch
    n = _serve_fast.V3_FROM_ROWS
    for m in (1, n - 1, n, n + 1):
        _serve_fast.check(torch.zeros(m, 4), torch.zeros(m, 8))
    _serve_fast.check(torch.zeros(1, 4), torch.zeros(1, 8), kern="v3")
    assert seen == [("v1", 1), ("v1", n - 1), ("v3", n), ("v3", n + 1), ("v3", 1)]

