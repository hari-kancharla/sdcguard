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


@pytest.mark.parametrize("kernel", ["v1", "v3", "v5"])
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


@pytest.mark.parametrize("kernel", ["v3", "v5"])
def test_a_forced_design_runs_the_full_comparison_set(kernel):
    """tests/_serve_fast_compare.py with v3 or v5 on every case: clean, large and small changes, NaN, Inf, finite
    outputs too large for the inputs, infinite inputs, fp16 overflow, partial blocks, bias, the capture, and a guard
    end to end."""
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
    assert _serve_fast.V5_BELOW_ROWS == 0            # (not used until a GPU timing run says where it is cheaper)
    assert _serve_fast.mode() == "auto" and _serve_fast.kernels() == ("v1", "v3")
    assert [_serve_fast.kernel_for(m) for m in (1, n - 1, n, 16384)] == ["v1", "v1", "v3", "v3"]
    assert _serve_fast.tile_heights("v1") == (1, 2, 4, 8, 16) and _serve_fast.tile_heights("v3") == (None,)
    assert _serve_fast.tile_heights("v5") == (16, 64)
    monkeypatch.setattr(_serve_fast, "V5_BELOW_ROWS", 129)
    assert _serve_fast.kernels() == ("v5", "v1", "v3")
    assert [_serve_fast.kernel_for(m) for m in (1, 128, 129, n)] == ["v5", "v5", "v1", "v3"]
    # only on the GPU generations v5 was timed and checked on (an H100 is 9; an A100 8, a B200 10)
    import torch
    cuda = torch.device("cuda", 0)
    for major, used in ((9, True), (8, False), (10, False)):
        monkeypatch.setattr(_serve_fast._group, "_major", lambda d, major=major: major)
        assert (_serve_fast.kernel_for(1, cuda) == "v5") == used and ("v5" in _serve_fast.kernels(cuda)) == used
    monkeypatch.setattr(_serve_fast, "V5_BELOW_ROWS", 0)
    for forced in ("v1", "v3", "v5"):
        monkeypatch.setenv("SDCGUARD_SERVE_KERNEL", forced)
        # (forced v5 can still run v1 or v3 when it has no counters: those are checked and compiled too)
        assert _serve_fast.kernels() == ((forced,) if forced != "v5" else ("v5", "v1", "v3"))
        assert {_serve_fast.kernel_for(m) for m in (1, n, 16384)} == {forced}
    monkeypatch.setenv("SDCGUARD_SERVE_KERNEL", "something else")
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


V5_SCRIPT = r'''
import json, torch
from sdcguard import _serve_fast as sf
from sdcguard.checksum import _split3
from sdcguard.serve import STATE_COLS, LayerChecksums, check_static_impl
g = torch.Generator().manual_seed(9)
rows = []
# (m, k, n, bias, y strided): several splits of k with a partial last tile (nb 3); one split (k <= BK); full 128-wide
# blocks and whole tiles (n 2048: nb 16, 2 tiles); nb not a multiple of 4 (n 1100: nb 9); a partial last block; more
# than one tile of rows for both heights (m 130); y a view with a longer row stride
for m, k, n, bias, strided in [(2, 1000, 300, True, False), (9, 64, 129, False, False), (20, 300, 2048, False, False),
                               (7, 260, 1100, True, True), (130, 200, 129, True, False), (1, 3000, 64, False, False)]:
    dt = torch.bfloat16 if m % 2 else torch.float16
    W = (torch.randn(n, k, generator=g) / k ** 0.5).to(dt)
    b = torch.randn(n, generator=g).to(dt) if bias else None
    x = torch.randn(m, k, generator=g).to(dt)
    y = torch.nn.functional.linear(x, W, b)
    if strided:
        wide = torch.zeros(m, n + 40, dtype=dt)
        wide[:, :n] = y
        y = wide[:, :n]
    cs = LayerChecksums(W, b, 128)
    cs.parts = _split3(cs.parts / cs.scale.view(1, -1), dt).contiguous()        # the GPU's three parts
    for bm in (16, 64):
        cfg = sf.config_v5(m, k, cs.nb, x.device, bm)
        for corrupt in (None, "big", "nan"):
            yy = y.clone() if not strided else y
            if strided:
                keep = wide.clone()
            r, c = (m * 2) // 3, (n * 3) // 5
            if corrupt == "big":
                yy[r, c] += 40.0
            elif corrupt == "nan":
                yy[r, c] = float("nan")
            ref = torch.zeros(1, STATE_COLS, dtype=torch.int64)
            out = []
            check_static_impl(x, yy, *cs.tensors(True), ref, 0, 6.0, 1e-20)
            for _ in range(2):              # twice: the same state bit for bit (deterministic)
                fast = torch.zeros(1, STATE_COLS, dtype=torch.int64)
                sf.check(x, yy, *cs.tensors(True), fast, 0, 6.0, 1e-20, kern="v5", _bm=bm)
                out.append(fast[0].tolist())
            counters_zero = sf.counters_clean() and len(sf._CNT) > 0
            rows.append([m, k, n, bm, cfg[2], corrupt, list(sf.decode(*ref[0].tolist())), list(sf.decode(*out[0])),
                         ref[0, 2:].tolist(), out[0][2:], out[0] == out[1], counters_zero])
            if strided:
                wide.copy_(keep)
print(json.dumps({"rows": rows, "fallbacks": sf.V5_FALLBACKS[0]}))
'''


def test_v5_with_each_tile_height_gives_the_plain_checks_answers():
    """v5 (_fused) with 16- and 64-row tiles on shapes that take each of its paths: k split several ways and not at
    all, partial and whole tiles of blocks, nb not a multiple of 4, a partial last block, several tiles of rows, y a
    strided view; the same state twice (deterministic), and every tile's counter back at zero after each launch."""
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_SIGN_SEED="1",
               PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-c", V5_SCRIPT], env=env, capture_output=True, text=True, timeout=1200)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["fallbacks"] == 0 and len(res["rows"]) == 36
    splits = {(r[0], r[1], r[3]): r[4] for r in res["rows"]}
    assert splits[(2, 1000, 16)] > 1 and splits[(9, 64, 16)] == 1          # both the counted and the uncounted path
    for m, k, n, bm, ns, corrupt, (rz, rf, _), (fz, ff, _), rc, fc, same, zero in res["rows"]:
        where = (m, k, n, bm, ns, corrupt, rz, fz, rf, ff)
        assert rc == fc == [1, 0] and same and zero, where
        if corrupt is None:
            assert rz <= 6.0 and fz <= 6.0 and abs(fz - rz) <= 0.05 + 0.02 * rz, where
        else:
            assert rz > 6.0 and fz > 6.0 and rf // 2 == ff // 2, where


def test_v5_counters_one_set_per_stream_and_layer_and_none_made_during_a_capture(monkeypatch):
    """Each layer has its own counters and two streams never share a set (past V5_SETS streams v5 steps aside); a
    layer's buffer grows (at least twofold, sets 32 counters apart) when nothing is being captured, and during a capture
    it is never made or grown: check_v5 then runs v1 or v3 instead and counts it. The load-time reservation covers
    every batch size v5 serves."""
    import torch
    from sdcguard import _serve_fast as sf
    dev = torch.device("cpu")
    monkeypatch.setattr(sf, "_CNT", {})
    monkeypatch.setattr(sf, "_CNT_SET", {})
    monkeypatch.setattr(sf, "_CNT_KEEP", [])
    stream = [0]
    capturing = [False]
    monkeypatch.setattr(sf, "_stream_key", lambda d: stream[0])
    monkeypatch.setattr(sf, "_capturing", lambda d: capturing[0])
    a = sf._counter_slice(dev, 1, 10)
    stream[0] = 7
    b = sf._counter_slice(dev, 1, 10)
    assert a is not None and b is not None and a.numel() == b.numel() == 32
    assert a.data_ptr() != b.data_ptr() and int(a.sum()) == int(b.sum()) == 0
    other = sf._counter_slice(dev, 2, 10)                                  # another layer: other memory
    assert other is not None and not (a.data_ptr() <= other.data_ptr() < a.data_ptr() + 4 * sf.V5_SETS * 32)
    stream[0] = 0
    assert sf._counter_slice(dev, 1, 10).data_ptr() == a.data_ptr()       # the same stream keeps its set
    for s in range(100, 100 + sf.V5_SETS - 2):
        stream[0] = s
        assert sf._counter_slice(dev, 1, 10) is not None
    stream[0] = 999
    assert sf._counter_slice(dev, 1, 10) is None                           # every set taken
    stream[0] = 0
    capturing[0] = True
    kept = len(sf._CNT_KEEP)
    assert sf._counter_slice(dev, 1, 40) is None and len(sf._CNT_KEEP) == kept     # not grown while capturing
    capturing[0] = False
    big = sf._counter_slice(dev, 1, 40)
    assert big is not None and big.numel() == 64 and len(sf._CNT_KEEP) == kept + 1  # twofold; the old one kept
    capturing[0] = True
    assert sf._counter_slice(dev, 1, 64) is not None                       # big enough: no allocation needed
    assert sf.counters_clean()
    big[3] = 1
    assert not sf.counters_clean()
    big[3] = 0
    # the reservation covers every batch size v5 serves, with the tile heights config_v5 picks
    for below, t64 in ((0, 17), (129, 17), (257, 300), (4097, 65)):
        monkeypatch.setattr(sf, "V5_BELOW_ROWS", below)
        monkeypatch.setattr(sf, "V5_T64_FROM_ROWS", t64)
        need = max(-(-m // sf.config_v5(m, 64, 296, dev)[0]) * 37 for m in range(1, max(256, below)))
        assert sf._v5_tiles_max(296) == need
    # check_v5 steps aside to v1 when it can't have counters
    calls = []
    monkeypatch.setattr(sf, "check_v1", lambda *a, **k: calls.append("v1"))
    monkeypatch.setattr(sf, "_counter_slice", lambda d, layer, t: None)
    from sdcguard.serve import STATE_COLS, LayerChecksums
    W = torch.randn(300, 1000).bfloat16()
    cs = LayerChecksums(W, None, 128)
    before = sf.V5_FALLBACKS[0]
    st = torch.zeros(1, STATE_COLS, dtype=torch.int64)
    sf.check_v5(torch.randn(2, 1000).bfloat16(), torch.randn(2, 300).bfloat16(), *cs.tensors(True), st, 0, 6.0, 1e-20)
    assert calls == ["v1"] and sf.V5_FALLBACKS[0] == before + 1


def test_v5_config_splits_k_for_few_rows_and_compiles_once_per_tile_height():
    """Nothing in config_v5 depends on the number of rows within a tile height (warm() compiles each variant once);
    with few rows the sum over k is split so the tiles make about two programs per SM."""
    import torch
    from sdcguard import _serve_fast as sf
    dev = torch.device("cpu")
    for k, nb in ((3584, 36), (18944, 28), (3584, 296), (64, 9)):
        for bm in (16, 64):
            cfgs = {sf.config_v5(m, k, nb, dev, bm) for m in (1, 7, 16, 17, 64, 255, 4096)}
            assert len(cfgs) == 1, (k, nb, bm, cfgs)
            bm_, bk, ns, kc = cfgs.pop()
            assert bm_ == bm and kc % bk == 0 and (ns - 1) * kc < k <= ns * kc and ns <= (32 if bm == 16 else 8)
    assert sf.config_v5(1, 3584, 36, dev)[0] == 16 and sf.config_v5(sf.V5_T64_FROM_ROWS, 3584, 36, dev)[0] == 64
