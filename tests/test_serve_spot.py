"""Sampled higher-precision verification (serve_spot.SpotChecks) on a CPU: layers take turns, a correct layer is
quiet (also over large layers and hard data), an output off by 2% (bf16) is named with its row and column and
rechecked in float64, the step's real rows bound the rows sampled, columns follow the golden-ratio order, the stated
floor holds, inactive layers are skipped on the host, a failed launch and an unusable kernel are handled, the bound's
assumptions are checked, each read keeps one output, the compiled call counts, and the Triton kernel gives the
PyTorch version's answers through Triton's interpreter."""
import os
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from sdcguard import serve_spot as sp
from sdcguard.serve import ServeGuard

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("SDCGUARD_ALLOW_CPU", "1")


def _layers(n=3, k=256, out=192, dtype=torch.bfloat16, bias=True):
    torch.manual_seed(0)
    return [nn.Linear(k, out, bias=bias).to(dtype) for _ in range(n)]


def _checks(layers, guard=None):
    g = guard or ServeGuard()
    sc = sp.SpotChecks(g)
    for i, lay in enumerate(layers):
        assert sc.add(lay, f"layer.{i}") == i
    sc.finalize()
    return g, sc


def _run(sc, layers, x, step, rows=None, fault=None):
    sc.mark_step(step, rows)
    for i, lay in enumerate(layers):
        y = lay(x)
        if fault is not None and fault[0] == i:
            y = fault[1](y)
        sp.check_call(lay._sdcguard_spot, x, y, lay.bias)
    sc.end_step()


def _off_at(r, c, by=1.02):
    def off(y):
        y = y.clone()
        # 2% of the value, rounded again: always more than 3 roundings off (bf16: u = 2^-8)
        y[r, c] = (y[r, c].float() * by).to(y.dtype) if abs(float(y[r, c])) > 0.05 else y[r, c] + 0.05
        return y
    return off


def test_a_correct_model_is_quiet_and_layers_take_turns():
    layers = _layers()
    g, sc = _checks(layers)
    x = torch.randn(10, 256).to(torch.bfloat16)
    for step in range(6):
        _run(sc, layers, x, step, 10)
    assert sc.poll(wait=True) == []
    assert sc.stats["active_calls"] == 6 and sc.checks_seen == {"layer.0": 2, "layer.1": 2, "layer.2": 2}
    assert sc.stats["outputs_judged"] == 6 * 10 * sp.SPOT_COLS
    assert max(sc.worst_seen.values()) <= 1.0


def test_an_output_off_by_two_percent_is_named_and_rechecked():
    layers = _layers()
    g, sc = _checks(layers)
    x = torch.randn(10, 256).to(torch.bfloat16)
    step = 4                                            # layer 1's turn (4 % 3)
    rows, rmask, cols = sp.sampled(torch.tensor(step), 3, 10, 192)
    r, c = int(rows[3]), int(cols[5])
    _run(sc, layers, x, step, 10, fault=(1, _off_at(r, c)))
    ev = sc.poll(wait=True)
    assert len(ev) == 1 and ev[0].detector == "serve_spot" and ev[0].detail["where"] == "layer.1"
    d = ev[0].detail
    assert (d["row"], d["col"]) == (r, c) and d["captured"]["row"] == r and d["captured"]["col"] == c
    assert d["recheck"]["verdict"] == "output_wrong" and d["max_z"] > sp.SPOT_Z
    # an off output in a column not sampled this turn isn't seen (each turn looks at SPOT_COLS columns)
    other = next(j for j in range(192) if j not in [int(v) for v in cols])
    _run(sc, layers, x, step, 10, fault=(1, lambda y: y.index_fill(1, torch.tensor([other]), 7.0)))
    assert sc.poll(wait=True) == []


def test_each_read_keeps_one_output_and_the_next_read_keeps_another():
    layers = _layers()
    g, sc = _checks(layers)
    x = torch.randn(10, 256).to(torch.bfloat16)
    hits = {}
    for step in (4, 5):                                 # layers 1 and 2 in the same read
        rows, _, cols = sp.sampled(torch.tensor(step), 3, 10, 192)
        hits[step % 3] = (int(rows[0]), int(cols[0]))
        _run(sc, layers, x, step, 10, fault=(step % 3, _off_at(*hits[step % 3], by=1.05)))
    ev = {e.detail["where"]: e.detail for e in sc.poll(wait=True)}
    assert set(ev) == {"layer.1", "layer.2"}
    assert ev["layer.1"]["recheck"]["verdict"] == "output_wrong"            # the first over the line is kept
    assert ev["layer.2"]["captured"] is None and "another layer" in ev["layer.2"]["recheck"]["why"]
    assert sc.stats["alerts_without_kept_output"] == 1
    rows, _, cols = sp.sampled(torch.tensor(8), 3, 10, 192)
    _run(sc, layers, x, 8, 10, fault=(2, _off_at(int(rows[0]), int(cols[0]), by=1.05)))
    ev = sc.poll(wait=True)                             # the read cleared the kept output: layer 2's is kept now
    assert len(ev) == 1 and ev[0].detail["captured"]["col"] == int(cols[0])
    assert ev[0].detail["recheck"]["verdict"] == "output_wrong"


def test_the_real_rows_bound_the_rows_sampled_and_columns_turn():
    layers = _layers(n=1)
    g, sc = _checks(layers)
    x = torch.randn(40, 256).to(torch.bfloat16)
    # rows past the 5 real ones are garbage (padding): never sampled
    _run(sc, layers, x, 0, 5, fault=(0, lambda y: torch.cat([y[:5], torch.full_like(y[5:], 1e4)])))
    assert sc.poll(wait=True) == [] and sc.stats["outputs_judged"] == 5 * sp.SPOT_COLS
    rows = set()
    for t in range(100):
        r, rm, _ = sp.sampled(torch.tensor(t), 1, 100, 192)
        rows |= {int(v) for v in r[rm]}
    assert rows == set(range(100))                       # every real row in turn


@pytest.mark.parametrize("n", [192, 1000, 37, 3, 1536, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 9216,
                               11008, 12288, 13824, 14336, 17920, 18944, 24576, 27648, 28672, 151936])
def test_columns_are_all_covered_and_bands_are_sampled_evenly(n):
    """Every column within N / SPOT_COLS turns when SPOT_COLS divides N (else within N); a band of w adjacent
    columns, wherever it starts, is sampled about once every N / (w SPOT_COLS) turns and never more than 2.5 times
    that apart, over common layer widths (moving the offset by one would leave gaps of up to N / SPOT_COLS turns;
    the nearest coprime step alone left gaps of 9.9 times that at N = 4,096)."""
    p = sp.col_step(n)
    assert 1 <= p < max(n, 2) and (n <= 2 or torch.gcd(torch.tensor(p), torch.tensor(n)) == 1)
    within = n // sp.SPOT_COLS if n % sp.SPOT_COLS == 0 else n
    if within <= 4096:
        seen = set()
        for t in range(within):
            seen |= {int(c) for c in sp.sampled(torch.tensor(t), 1, 4, n)[2]}
        assert seen == set(range(n))
    if n % sp.SPOT_COLS or n < 1000:
        return
    s = n // sp.SPOT_COLS
    t = torch.arange(s)
    q = ((t % n) * p % n) % s                           # the columns at turn t are those = q mod s (period s)
    assert sorted(q.tolist()) == list(range(s))          # (so every column within s turns, at any width)
    pos = torch.empty(s, dtype=torch.int64)
    pos[q] = t
    for w in (16, 64, 128):
        hits = pos[(torch.arange(s)[:, None] + torch.arange(w)[None, :]) % s].sort(1).values   # each band's turns
        gaps = torch.cat([hits.diff(1), hits[:, :1] + s - hits[:, -1:]], 1)
        assert int(gaps.max()) <= 2.5 * s / w, (n, w, int(gaps.max()), s / w)


def test_fp16_without_a_bias_is_judged_on_fp16s_rounding():
    layers = _layers(n=2, dtype=torch.float16, bias=False)
    g, sc = _checks(layers)
    x = torch.randn(3, 256).to(torch.float16)
    _run(sc, layers, x, 1, 3, fault=(1, lambda y: y * (1 + 2 ** -7)))   # 0.8%: about 16 fp16 roundings
    ev = sc.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["where"] == "layer.1"
    assert ev[0].detail["check_config"]["unit_roundoff"] == 2 ** -11
    assert sp.describe(nn.Linear(4, 4))[1] == "weights in torch.float32"


def _data(kind, m, n, k, g):
    x = torch.randn(m, k, generator=g)
    w = torch.randn(n, k, generator=g) / k ** 0.5
    if kind == "outliers":                              # a few input channels 100 times larger
        x[:, torch.randperm(k, generator=g)[:max(1, k // 256)]] *= 100.0
    elif kind == "relu":                                # non-negative inputs, weights with a non-zero mean
        x, w = x.clamp(min=0), w + 0.02
    elif kind == "cancel_halves":                       # each output's two halves cancel exactly
        x[:, k // 2:] = x[:, :k // 2]
        w[:, k // 2:] = -w[:, :k // 2]
    elif kind == "cancel_blocks":                       # blocks of 512 cancel in pairs
        for i in range(0, k - 1023, 1024):
            x[:, i + 512:i + 1024] = x[:, i:i + 512]
            w[:, i + 512:i + 1024] = -w[:, i:i + 512]
    return x, w


def test_correct_outputs_score_under_1_over_large_layers_and_hard_data():
    """PyTorch's own matmul over K of 1,536, 4,096 and 8,960, Gaussian inputs, outliers, ReLU inputs and inputs and
    weights built to cancel, with and without a bias, in bf16 and fp16: every output (not just samples) scores at
    most 1 against a float64 recompute (docs/METHOD.md quotes the largest)."""
    g = torch.Generator().manual_seed(11)
    worst = 0.0
    for k in (1536, 4096, 8960):
        for kind in ("gaussian", "outliers", "relu", "cancel_halves", "cancel_blocks"):
            x, w = _data(kind, 16, 96, k, g)
            for dt in (torch.bfloat16, torch.float16):
                for with_bias in (False, True):
                    b = (torch.randn(96, generator=g) * 0.5).to(dt) if with_bias else None
                    xd, wd = x.to(dt), w.to(dt)
                    y = torch.nn.functional.linear(xd, wd, b)
                    z = sp.score_all(xd, wd, b, y)
                    worst = max(worst, float(z.max()))
                    assert float(z.max()) <= 1.0, (k, kind, dt, with_bias, float(z.max()))
    print(f"worst clean score {worst:.3f}")


def test_a_fault_over_the_stated_floor_is_always_caught_and_the_stated_floors():
    g = torch.Generator().manual_seed(3)
    for dt in (torch.bfloat16, torch.float16):
        for with_bias in (False, True):
            x, w = _data("gaussian", 8, 64, 1536, g)
            b = torch.randn(64, generator=g) * 0.05 if with_bias else None
            xd, wd = x.double(), w.double()
            acc = xd @ wd.t()
            ref = acc + (b.double() if b is not None else 0.0)
            norm = xd.norm(dim=1)[:, None] * wd.norm(dim=1)[None, :]
            for sign in (1.0, -1.0):
                e = torch.tensor([[sp.min_caught(dt, 1536, float(norm[i, j]), float(ref[i, j]),
                                                 float(b[j]) if b is not None else None) * 1.001 * sign
                                   for j in range(64)] for i in range(8)], dtype=torch.float64)
                y = (ref + e).to(dt)            # rounded once, with no accumulation error (the floor allows for one)
                u, spc, _ = sp.dtype_consts(dt)
                den = u * ref.abs() + (u * acc.abs() if b is not None else 0.0) + \
                    sp.acc_term(1536, norm, ref.abs()) + spc
                z = (y.double() - ref).abs() / den
                assert float(z.min()) > sp.SPOT_Z, (dt, with_bias, float(z.min()))
    # the floors the module and docs/METHOD.md state: median Gaussian outputs, as a share of |r|
    stated = {(torch.bfloat16, 1536, None): 0.013, (torch.float16, 1536, None): 0.0027,
              (torch.bfloat16, 8960, None): 0.019, (torch.float16, 8960, None): 0.0087,
              (torch.bfloat16, 1536, 0.0): 0.025}                 # (a layer with a small bias)
    for (dt, k, bias), share in stated.items():
        x, w = _data("gaussian", 32, 128, k, g)
        acc = x.double() @ w.double().t()
        norm = x.double().norm(dim=1)[:, None] * w.double().norm(dim=1)[None, :]
        rel = torch.tensor([sp.min_caught(dt, k, float(norm[i, j]), float(acc[i, j]), bias) / abs(float(acc[i, j]))
                            for i in range(32) for j in range(128)])
        assert abs(float(rel.median()) - share) <= 0.1 * share, (dt, k, bias, float(rel.median()), share)


def test_extreme_scales_neither_hide_a_wrong_output_nor_pass_an_unbounded_one_as_clean():
    """Inputs 2^100 times larger and weights 2^100 times smaller give the same products, but their float32 squares
    overflow and underflow: a wrong output must still be caught, a right one stay quiet; and where the bound itself
    can't be computed (a cancelling row whose norm overflows float32) the outputs are unjudged, not clean."""
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(6, 256, generator=g) * 2.0 ** 100).to(torch.bfloat16)
    w = (torch.randn(24, 256, generator=g) * 0.05 * 2.0 ** -100).to(torch.bfloat16)
    y = torch.nn.functional.linear(x, w)
    for wrong in (False, True):
        yy = y.clone()
        rows, _, cols = sp.sampled(torch.tensor(4), 3, 6, 24)
        if wrong:
            yy[int(rows[2]), int(cols[5])] *= 2.0              # 100% off
        st = torch.zeros(2, sp.COLS, dtype=torch.int64)
        caps = sp.new_caps("cpu", 256)
        sb = torch.full((1,), sp.pack_step(4), dtype=torch.int64)
        sp.spot_check_torch(x, yy, w, None, sb, st, 1, 3, 1, *caps, sp.SPOT_Z)
        z = sp._f32(int(st[1, sp.C_ROW]) >> 32)
        assert int(st[1, sp.C_JUDGED]) == 6 * sp.SPOT_COLS and int(st[1, sp.C_UNJ]) == 0
        if wrong:
            assert z > 100 and int(caps[0][0]) == 1 and int(st[1, sp.C_COL]) & sp.M32 == int(cols[5])
        else:
            assert z <= 1.0 and int(caps[0][0]) == 0
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    x[:, :2] = 1e20
    w = torch.zeros(16, 64, dtype=torch.bfloat16)
    w[:, 0], w[:, 1] = 1e20, -1e20
    y = torch.full((2, 16), 5.0, dtype=torch.bfloat16)         # (the exact outputs are 0)
    st = torch.zeros(2, sp.COLS, dtype=torch.int64)
    caps = sp.new_caps("cpu", 64)
    sp.spot_check_torch(x, y, w, None, torch.full((1,), sp.pack_step(4), dtype=torch.int64), st, 1, 3, 1, *caps,
                        sp.SPOT_Z)
    assert int(st[1, sp.C_JUDGED]) == 0 and int(st[1, sp.C_UNJ]) == 2 * sp.SPOT_COLS and int(st[1, 0]) == 0


def test_inactive_layers_are_skipped_on_the_host_outside_capture(monkeypatch):
    layers = _layers()
    g, sc = _checks(layers)
    calls = []
    real = sp.spot_check_torch
    monkeypatch.setattr(sp, "spot_check_torch", lambda *a: (calls.append(a[6]), real(*a))[1])
    x = torch.randn(10, 256).to(torch.bfloat16)
    for step in range(6):
        _run(sc, layers, x, step, 10)
    # every layer's first call runs (its kernel variant compiled before any capture), then only the active one
    assert calls == [0, 1, 2, 1, 2, 0, 1, 2]
    assert sc.poll(wait=True) == [] and sc.stats["active_calls"] == 6


def test_a_failed_launch_falls_back_to_the_pytorch_check(monkeypatch):
    layers = _layers()
    g, sc = _checks(layers)
    key = (sp._dev_key(torch.device("cpu")), torch.bfloat16)
    monkeypatch.setitem(sp.STATUS, key, (True, ""))

    def broken(*a):
        raise RuntimeError("no such kernel")
    monkeypatch.setattr(sp, "spot_check_triton", broken)
    x = torch.randn(10, 256).to(torch.bfloat16)
    rows, _, cols = sp.sampled(torch.tensor(4), 3, 10, 192)
    _run(sc, layers, x, 4, 10, fault=(1, _off_at(int(rows[0]), int(cols[0]))))
    ev = sc.poll(wait=True)
    assert len(ev) == 1 and ev[0].detail["recheck"]["verdict"] == "output_wrong"
    assert sp.STATUS[key][0] is False and "a launch failed (RuntimeError: no such kernel)" in sp.STATUS[key][1]


def test_on_a_gpu_layers_arent_spot_checked_where_the_kernel_or_the_bound_fails(monkeypatch):
    sc = sp.SpotChecks(ServeGuard())
    gpu = torch.device("cuda", 0)
    monkeypatch.setattr(sp, "bound_holds", lambda d, t: None)
    monkeypatch.setattr(sp, "verify_kernel", lambda d, t: (False, "AssertionError: x"))
    assert "Triton kernel isn't usable here (AssertionError: x)" in sc._why_not(gpu, torch.bfloat16)
    assert sc._why_not(torch.device("cpu"), torch.bfloat16) is None        # the CPU (tests) runs the PyTorch version
    monkeypatch.setenv("SDCGUARD_SPOT_TORCH", "1")
    assert sc._why_not(gpu, torch.bfloat16) is None
    monkeypatch.setattr(sp, "bound_holds", lambda d, t: "reductions in bf16")
    assert sc._why_not(gpu, torch.bfloat16) == "the bound doesn't hold: reductions in bf16"
    # finalize lists those layers as not spot-checked, but every layer keeps its arguments (4 slots, 4 turns), so the
    # traced graph doesn't depend on what finalize found, and the left-out layers' calls return at once
    layers = _layers(n=2) + _layers(n=2, dtype=torch.float16)
    sc = sp.SpotChecks(ServeGuard())
    for i, lay in enumerate(layers):
        sc.add(lay, f"layer.{i}")
    monkeypatch.setattr(sc, "_why_not", lambda d, t: "no" if t == torch.float16 else None)
    assert sc.finalize() == 2 and sc.unsupported == {"layer.2": "no", "layer.3": "no"}
    assert [lay["slot"] for lay in sc.layers] == [0, 1] and sc.stats["layers"] == 2
    assert [lay._sdcguard_spot[3:6] for lay in layers] == [(0, 4, 0), (1, 4, 1), (2, 4, 2), (3, 4, 3)]
    x = torch.randn(4, 256).to(torch.float16)
    sc.mark_step(2, 4)                                    # layer 2's turn
    y = layers[2](x)
    y[0, :] += 100.0
    sp.check_call(layers[2]._sdcguard_spot, x, y, layers[2].bias)
    assert sc.poll(wait=True) == [] and int(sc._state[torch.device("cpu")].abs().sum()) == 0
    sc.detach()
    assert not any(hasattr(lay, "_sdcguard_spot") for lay in layers)


def test_inside_cuda_graph_capture_every_layer_launches(monkeypatch):
    """Outside capture an inactive layer's call is skipped on the host; inside it, the call must reach the GPU (the
    graph replays it on every later step, when it may be that layer's turn)."""
    layers = _layers()
    g, sc = _checks(layers)
    calls = []
    real = sp.spot_check_torch
    monkeypatch.setattr(sp, "spot_check_torch", lambda *a: (calls.append(a[6]), real(*a))[1])
    x = torch.randn(10, 256).to(torch.bfloat16)
    _run(sc, layers, x, 0, 10)                            # (each layer's first call runs)
    calls.clear()
    _run(sc, layers, x, 1, 10)
    assert calls == [1]
    monkeypatch.setattr(sp, "_capturing", lambda t: True)
    calls.clear()
    _run(sc, layers, x, 1, 10)
    assert calls == [0, 1, 2]


def test_the_bound_needs_fp32_reductions(monkeypatch):
    m = torch.backends.cuda.matmul
    gpu = torch.device("cuda", 0)
    monkeypatch.setattr(m, "allow_bf16_reduced_precision_reduction", False)
    monkeypatch.setattr(m, "allow_fp16_reduced_precision_reduction", False)
    assert sp.bound_holds(gpu, torch.bfloat16) is None and sp.bound_holds(gpu, torch.float16) is None
    monkeypatch.setattr(m, "allow_bf16_reduced_precision_reduction", True)
    assert "bf16" in sp.bound_holds(gpu, torch.bfloat16) and sp.bound_holds(gpu, torch.float16) is None
    assert sp.bound_holds(torch.device("cpu"), torch.bfloat16) is None        # (no cuBLAS on the CPU)
    monkeypatch.setattr(m, "allow_fp16_reduced_precision_reduction", True)
    assert "fp16" in sp.bound_holds(gpu, torch.float16)
    monkeypatch.setattr(m, "allow_fp16_reduced_precision_reduction", False)
    monkeypatch.setattr(m, "allow_bf16_reduced_precision_reduction", False)
    if hasattr(m, "allow_fp16_accumulation"):              # (PyTorch 2.7+)
        monkeypatch.setattr(m, "allow_fp16_accumulation", True)
        assert "fp16 accumulation" in sp.bound_holds(gpu, torch.float16) and sp.bound_holds(gpu, torch.bfloat16) is None


def test_the_line_is_at_least_1_5():
    with pytest.raises(ValueError):
        sp.SpotChecks(ServeGuard(), line=1.0)


def test_refresh_before_finalize_does_nothing():
    sc = sp.SpotChecks(ServeGuard())
    sc.add(_layers(n=1)[0], "layer.0")
    sc.refresh()
    assert not sc.finalized


def test_the_compiled_call_counts_and_finds_a_wrong_output():
    """check_call's compiled branch: torch.compile (Inductor, dynamic shapes) calls sdcguard::spot_check, which
    updates the state in place (no copy of the mutated buffers)."""
    layers = _layers()
    g, sc = _checks(layers)

    def forward(x, fault_scale):
        for i, lay in enumerate(layers):
            y = lay(x)
            if i == 1:
                y = y * fault_scale
            sp.check_call(lay._sdcguard_spot, x, y, lay.bias)
        return x

    fn = torch.compile(forward, fullgraph=True, dynamic=True)
    one = torch.ones((), dtype=torch.bfloat16)
    for step, m in enumerate((10, 12, 10, 7, 10, 9)):
        sc.mark_step(step, m)
        fn(torch.randn(m, 256).to(torch.bfloat16), one)
        sc.end_step()
    assert sc.poll(wait=True) == [] and sc.stats["active_calls"] == 6 and max(sc.worst_seen.values()) <= 1.0
    sc.mark_step(7, 10)                                  # layer 1's turn, every output 3% too large
    fn(torch.randn(10, 256).to(torch.bfloat16), torch.tensor(1.03, dtype=torch.bfloat16))
    sc.end_step()
    ev = sc.poll(wait=True)
    assert [e.detail["where"] for e in ev] == ["layer.1"] and ev[0].detail["recheck"]["verdict"] == "output_wrong"


def test_the_triton_kernel_matches_the_pytorch_check():
    pytest.importorskip("triton")
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1", SDCGUARD_ALLOW_CPU="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(HERE), "src"), HERE,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(HERE, "_serve_spot_compare.py")], env=env, capture_output=True,
                       text=True, timeout=1200)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout


def test_the_guard_takes_spot_checks_along():
    layers = _layers(n=2)
    g = ServeGuard()
    sc = g.spot_checks()
    for i, lay in enumerate(layers):
        sc.add(lay, f"layer.{i}")
    sc.finalize()
    x = torch.randn(4, 256).to(torch.bfloat16)
    g.mark_step(1, 4)
    y = layers[1](x)
    y[0, int(sp.sampled(torch.tensor(1), 2, 4, 192)[2][0])] += 5.0
    sp.check_call(layers[1]._sdcguard_spot, x, y, layers[1].bias)
    g.end_step()
    ev = g.poll(wait=True)
    assert [e.detector for e in ev] == ["serve_spot"] and g.coverage_report()["spot"]["layers"] == 2
    g.detach()
    assert g.spot is None and not hasattr(layers[0], "_sdcguard_spot")


@pytest.mark.parametrize("k,bias", [(1536, True), (1536, False), (8960, False)])
def test_a_whole_layer_2_percent_off_puts_most_sampled_outputs_over_the_line(k, bias):
    """Every output of a layer 2% too large (rounded again, as the trials inject it): at least half of its outputs
    score over the line, the qkv-like layer with a bias (its floor is over 2%: 2.5% at K = 1,536) included, so one
    turn of even 4 rows (32 outputs) catches it all but surely."""
    g = torch.Generator().manual_seed(4)
    x = torch.randn(32, k, generator=g).to(torch.bfloat16)
    w = (torch.randn(256, k, generator=g) / k ** 0.5).to(torch.bfloat16)
    b = (torch.randn(256, generator=g) * 0.5).to(torch.bfloat16) if bias else None
    y = torch.nn.functional.linear(x, w, b)
    share = float((sp.score_all(x, w, b, (y.float() * 1.02).to(y.dtype)) > sp.SPOT_Z).double().mean())
    assert share >= 0.5, share
    print(f"K={k} bias={bias}: {share:.2f} of outputs over the line")


def test_detach_then_refresh_and_a_second_guard():
    layers = _layers(n=2)
    g, sc = _checks(layers)
    sc.detach()
    sc.refresh()                                          # nothing left to point at: does nothing
    assert not any(hasattr(lay, "_sdcguard_spot") for lay in layers)
    g2, sc2 = _checks(layers)                             # another guard can take them after the first let go
    sc3 = sp.SpotChecks(ServeGuard())
    assert sc3.add(layers[0], "again") is None and "another ServeGuard" in sc3.unsupported["again"]
