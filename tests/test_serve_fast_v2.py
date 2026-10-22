"""The serving check's two Triton designs (_serve_fast: v1, cuBLAS then _check; v2, _emm then _pairs) against the
plain PyTorch check, in Triton's CPU interpreter, and v2's tile and split choices. tests/test_fast.py runs the full
comparison set (tests/_serve_fast_compare.py) with the default kernel."""
import json
import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("triton")

HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPT = r'''
import json, sys, torch
from sdcguard import _serve_fast
from sdcguard.checksum import _split3
from sdcguard.serve import STATE_COLS, LayerChecksums, check_static_impl
g = torch.Generator().manual_seed(5)
rows = []
# (m, k, n, bias): one row; a partial last block; a long sum over k (v2 splits it at few rows); many rows
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


@pytest.mark.parametrize("kernel", ["v1", "v2"])
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


def test_v2_tiles_and_splits_follow_the_batch_size():
    """Tiles of 16, 64 and 128 rows; the sum over k split only while the output tiles alone leave the GPU (here
    the interpreter's stand-in of 64 programs) under half full, in parts of at least 4 steps."""
    from sdcguard import _serve_fast
    cpu = torch.device("cpu")
    assert [_serve_fast.config_v2(m, 3584, 1776, cpu)[0] for m in (1, 16, 17, 1024, 1025, 9000)] == \
        [16, 16, 64, 64, 128, 128]
    for m in (1, 16, 64, 256, 1024, 4096):
        bm, bn, warps, splits, ks = _serve_fast.config_v2(m, 3584, 1776, cpu)
        steps = -(-3584 // _serve_fast.EMM_BK)
        assert ks % _serve_fast.EMM_BK == 0 and splits * ks >= 3584 and (splits - 1) * ks < 3584
        tiles = -(-m // bm) * -(-1776 // bn)
        if tiles >= 32:
            assert splits == 1
        else:
            assert splits > 1 and ks // _serve_fast.EMM_BK >= 4 and splits * tiles <= 32 + tiles
    assert _serve_fast.config_v2(1, 64, 216, cpu)[3] == 1          # a short sum isn't split
    assert _serve_fast.tile_heights() in ((16, 64, 128), (1, 2, 4, 8, 16))
