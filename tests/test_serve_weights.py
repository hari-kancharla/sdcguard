"""Scheduled weight integrity (serve_weights.WeightScan) on a CPU: a weight bit changed after load is named within a
pass over the model and only once, a clean model raises nothing, the byte budget sets how many reads a pass takes,
an intended update (refresh) isn't called corruption, tensors that moved aren't compared, quantized layers' own
tensors and layers outside the arithmetic checks are scanned too, and the fingerprint sees every single bit flip."""
import os

import pytest
import torch
import torch.nn as nn

from sdcguard.serve import ServeGuard
from sdcguard import serve_weights as sw
from sdcguard.serve_weights import BLOCK, fingerprint


def _model(widths=(64, 96, 64, 32)):
    torch.manual_seed(0)
    layers = [nn.Linear(a, b) for a, b in zip(widths, widths[1:])]
    return nn.Sequential(*layers).to(torch.bfloat16)


def _flip(t: torch.Tensor, index: int, bit: int) -> None:
    with torch.no_grad():
        t.view(-1).view(torch.int16)[index] ^= (1 << bit)


def _scan(model, budget_mb=64.0):
    g = ServeGuard().attach(model)
    s = g.weight_scan(budget_mb=budget_mb)
    return g, s


def test_a_clean_model_raises_nothing_and_every_layer_is_compared():
    m = _model()
    g, s = _scan(m)
    for _ in range(3):
        assert g.poll(wait=True) == []
    cov = g.coverage_report()["weights"]
    assert cov["entries"] == 3 and cov["by_kind"] == {"checked": 3} and cov["never_compared"] == []
    assert cov["changed"] == [] and s.stats["passes"] >= 1


def test_a_single_bit_flip_is_named_within_a_pass_and_only_once():
    m = _model()
    g, s = _scan(m, budget_mb=1e-6)              # one entry a read: a pass is 3 reads
    assert g.poll(wait=True) == []
    _flip(m[1].weight, 17, 0)                     # the lowest mantissa bit: no output check would see it
    seen = []
    for _ in range(3):
        seen += [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert len(seen) == 1
    d = seen[0].detail
    assert d["where"] == g.layers[1]["name"] and d["finding"] == "weights_changed" and d["kind"] == "checked"
    assert d["recheck"]["verdict"] == "weights_changed" and d["weights_changed_since_load"] is True
    for _ in range(6):                            # alerted once, not at every pass
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert g.coverage_report()["weights"]["changed"] == [g.layers[1]["name"]]
    assert g.log.events and g.log.events[-1].detector == "serve_weights"


def test_the_budget_sets_how_many_reads_a_pass_takes():
    m = _model()
    g, s = _scan(m, budget_mb=1e-6)
    for _ in range(3):
        g.poll(wait=True)
    assert s.stats["passes"] == 1 and s.stats["entries_compared"] == 3
    g2, s2 = _scan(_model(), budget_mb=64)        # everything in one read
    g2.poll(wait=True)
    assert s2.stats["passes"] == 1 and s2.stats["entries_compared"] == 3


def test_a_bias_change_is_named_too():
    m = _model()
    g, s = _scan(m)
    g.poll(wait=True)
    with torch.no_grad():
        m[2].bias.view(torch.int16)[0] ^= 1
    ev = [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["where"] for e in ev] == [g.layers[2]["name"]]


def test_an_intended_update_after_refresh_isnt_corruption():
    m = _model()
    g, s = _scan(m)
    g.poll(wait=True)
    with torch.no_grad():
        m[0].weight.mul_(2)
    g.refresh()
    for _ in range(3):
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    _flip(m[0].weight, 3, 5)                      # a change after the refresh is still caught
    assert [e.detail["where"] for e in g.poll(wait=True) if e.detector == "serve_weights"] == [g.layers[0]["name"]]


def test_refresh_layer_takes_a_new_fingerprint_for_that_layer_only():
    m = _model()
    g, s = _scan(m)
    g.poll(wait=True)
    with torch.no_grad():
        m[1].weight.add_(1)
        m[2].weight.add_(1)
    g.refresh_layer(1)
    ev = [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["where"] for e in ev] == [g.layers[2]["name"]]


def test_tensors_that_moved_arent_compared():
    m = _model()
    g, s = _scan(m)
    g.poll(wait=True)
    m[1].weight = nn.Parameter(m[1].weight.detach().clone() * 3)   # other memory: not the fingerprinted tensor
    assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert g.layers[1]["name"] in g.coverage_report()["weights"]["not_scanned"]


def test_layers_outside_the_checks_and_quantized_tensors_are_scanned():
    m = _model()
    g, s = _scan(m)
    extra = nn.Linear(16, 16).to(torch.bfloat16)
    packed = {"q": torch.randint(-2 ** 31, 2 ** 31 - 1, (8, 4), dtype=torch.int32),
              "scales": torch.rand(8, dtype=torch.float16)}
    assert g.scan_weights_of(extra, "model.lm_extra") is not None
    assert g.scan_weights_of(None, "model.packed", lambda: (packed["q"], packed["scales"])) is not None
    g.poll(wait=True)
    assert g.coverage_report()["weights"]["by_kind"] == {"checked": 3, "unchecked": 2}
    _flip(extra.weight, 0, 14)
    with torch.no_grad():
        packed["scales"].view(torch.int16)[3] ^= 1 << 2
    names = sorted(e.detail["where"] for e in g.poll(wait=True) if e.detector == "serve_weights")
    assert names == ["model.lm_extra", "model.packed"]


def test_without_weight_scan_nothing_is_scanned():
    m = _model()
    g = ServeGuard().attach(m)
    assert g.scan_weights_of(nn.Linear(4, 4), "x") is None
    _flip(m[0].weight, 0, 0)
    assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert "weights" not in g.coverage_report()


def test_the_fingerprint_sees_every_single_bit_flip_and_is_exact():
    torch.manual_seed(1)
    for dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int32, torch.uint8):
        t = (torch.randn(3 * BLOCK + 77) * 10).to(dtype) if dtype.is_floating_point else \
            torch.randint(0, 100, (3 * BLOCK + 77,)).to(dtype)
        ref = fingerprint(t)
        assert torch.equal(ref, fingerprint(t.clone()))
        bits = t.element_size() * 8
        raw = t.view(torch.int16 if bits == 16 else torch.int32 if bits == 32 else torch.uint8)
        for index in (0, BLOCK - 1, BLOCK, 3 * BLOCK + 76):
            for bit in range(bits):
                c = raw.clone()
                c[index] ^= (1 << bit) if bit < bits - 1 else (-(1 << bit) if bits > 8 else (1 << bit))
                assert not torch.equal(fingerprint(c), ref), (dtype, index, bit)
    assert torch.equal(fingerprint(None), torch.zeros(2, dtype=torch.int64))


def test_cancelling_changes_and_swaps_in_one_block_are_seen():
    """The old fingerprint (block sums) missed these: two opposite changes in one block, and two values swapped."""
    torch.manual_seed(3)
    t = (torch.randn(2 * BLOCK) * 5).to(torch.bfloat16).view(torch.int16)
    ref = fingerprint(t)
    for i, j in ((5, 6), (0, BLOCK - 1), (17, BLOCK + 17), (BLOCK + 1, 2 * BLOCK - 1)):
        c = t.clone()
        c[i] += 1
        c[j] -= 1                                       # the plain sum is unchanged
        assert not torch.equal(fingerprint(c), ref), (i, j)
        if t[i] != t[j]:
            s = t.clone()
            s[i], s[j] = t[j], t[i]                     # a swap: the plain sum is unchanged
            assert not torch.equal(fingerprint(s), ref), (i, j)
    s = t.clone()
    s[100:356] = t[1000:1256]                           # a region copied over another
    assert not torch.equal(fingerprint(s), ref)
    s = t.clone()
    s[:BLOCK] = t[BLOCK:]                               # two blocks exchanged
    s[BLOCK:] = t[:BLOCK]
    assert not torch.equal(fingerprint(s), ref)


def _reference(values) -> list:
    """The fingerprint written out in Python integers, independently of the torch version: values as signed
    (16, 32 bits) or unsigned (8 bits) integers, sums modulo 2^64, keys from MurmurHash3's fmix64."""
    mask = (1 << 64) - 1

    def mix(z):
        z = (z ^ (z >> 33)) * 0xFF51AFD7ED558CCD & mask
        z = (z ^ (z >> 33)) * 0xC4CEB9FE1A85EC53 & mask
        return z ^ (z >> 33)

    a = b = 0
    for i, x in enumerate(values):
        a = (a + x) & mask
        b = (b + x * mix(i ^ 0x2545F4914F6CDD1D)) & mask
    signed = [v - (1 << 64) if v >= (1 << 63) else v for v in (a, b)]
    return signed


def test_the_fingerprint_is_the_documented_one():
    torch.manual_seed(4)
    for dtype, view, bits in ((torch.bfloat16, torch.int16, 16), (torch.float32, torch.int32, 32),
                              (torch.uint8, torch.uint8, 8)):
        t = (torch.randn(97) * 1e4).to(dtype) if dtype.is_floating_point else torch.randint(0, 256, (97,)).to(dtype)
        raw = t.view(view)
        if bits > 8:
            raw[0] = -(1 << (bits - 1))                 # the most negative value
            raw[1] = (1 << (bits - 1)) - 1              # the largest
        else:
            raw[0], raw[1] = 0, 255
        assert fingerprint(t).tolist() == _reference([int(v) for v in raw.tolist()]), dtype
    t = torch.randn(33, dtype=torch.float64)
    assert fingerprint(t).tolist() == _reference([int(v) for v in t.view(torch.int64).view(torch.int32).tolist()])


def test_eight_byte_tensors_are_fingerprinted_as_pairs_of_32_bit_values():
    for dtype in (torch.float64, torch.int64):
        t = (torch.randn(BLOCK + 9) * 7).to(dtype)
        ref = fingerprint(t)
        raw = t.view(torch.int64)
        for index in (0, BLOCK + 8):
            for bit in (0, 31, 32, 62):
                c = raw.clone()
                c[index] ^= (1 << bit)
                assert not torch.equal(fingerprint(c.view(dtype)), ref), (dtype, index, bit)


def test_the_fingerprint_needs_no_copy_of_the_whole_tensor():
    """It sums in chunks: the same result as one pass over the whole tensor, chunk boundaries included."""
    from sdcguard import serve_weights as sw
    t = torch.randint(-2 ** 15, 2 ** 15 - 1, (2 * sw.CHUNK + 5 * BLOCK + 3,), dtype=torch.int16)
    v = t.to(torch.int64)
    expect = torch.stack([v.sum(), (v * sw.keys(0, v.numel(), v.device)).sum()])
    assert torch.equal(fingerprint(t), expect)
    k = sw.keys(0, 1 << 16, torch.device("cpu"))
    assert torch.unique(k).numel() == k.numel()          # no two positions share a key (the mix is a bijection)


def test_a_reload_through_temporary_tensors_keeps_the_layer_compared():
    """vLLM 0.32's layerwise reload processes a layer's weights in temporary tensors, then copies them into the
    originals: the reference taken from the temporaries is provisional and is taken again at the next turn."""
    m = _model()
    g, s = _scan(m)
    g.poll(wait=True)
    orig = m[1].weight
    m[1].weight = nn.Parameter(orig.detach().clone() * 2, requires_grad=False)    # the temporary
    g.refresh_layer(1)
    with torch.no_grad():
        orig.copy_(m[1].weight)
    m[1].weight = orig                                                           # copied back, restored
    for _ in range(2):
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert s.stats["references_retaken"] == 1 and g.layers[1]["name"] not in s.not_scanned
    _flip(orig, 9, 1)
    assert [e.detail["where"] for e in g.poll(wait=True) if e.detector == "serve_weights"] == [g.layers[1]["name"]]


def test_layers_sharing_one_buffer_arent_compared_and_raise_nothing():
    """vLLM's prefetch offloader points several layers at one GPU buffer and copies each one's weights in before it
    runs: those layers aren't compared (listed), and a refresh doesn't make them a false alarm."""
    m = _model((64, 64, 64, 64))
    g, s = _scan(m)
    buf = torch.zeros_like(m[0].weight)
    a, b = nn.Linear(64, 64).to(torch.bfloat16), nn.Linear(64, 64).to(torch.bfloat16)
    a.weight = nn.Parameter(buf, requires_grad=False)
    b.weight = nn.Parameter(buf.view_as(buf), requires_grad=False)                # another tensor, the same memory
    g.scan_weights_of(a, "offloaded.a")
    g.scan_weights_of(b, "offloaded.b")
    g.refresh()
    for k in range(3):
        with torch.no_grad():
            buf.copy_(torch.randn_like(buf) * (k + 1))                              # the offloader copies a layer in
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    ns = g.coverage_report()["weights"]["not_scanned"]
    assert "offloaded.a" in ns and "offloaded.b" in ns and "shares its memory" in ns["offloaded.a"]


def test_two_layers_moved_onto_one_buffer_before_their_first_comparison_are_both_left_out():
    """The first to move is taken again where it now is (it was never compared); when the second turns up in the same
    memory, both are left out, not just the second: what is there belongs to whichever ran last."""
    m = _model((64, 64, 64, 64))
    g, s = _scan(m)
    a, b = nn.Linear(64, 64).to(torch.bfloat16), nn.Linear(64, 64).to(torch.bfloat16)
    g.scan_weights_of(a, "offloaded.a")
    g.scan_weights_of(b, "offloaded.b")
    buf = torch.zeros_like(a.weight)
    a.weight = nn.Parameter(buf, requires_grad=False)
    b.weight = nn.Parameter(buf.view_as(buf), requires_grad=False)
    for k in range(4):
        with torch.no_grad():
            buf.copy_(torch.randn_like(buf) * (k + 1))
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"], k
    ns = g.coverage_report()["weights"]["not_scanned"]
    assert "shares its memory" in ns["offloaded.a"] and "shares its memory" in ns["offloaded.b"]


def test_an_alert_names_the_steps_since_its_last_clean_comparison():
    m = _model()
    g, s = _scan(m)
    g.mark_step(10)
    g.poll(wait=True)
    _flip(m[2].weight, 0, 4)
    g.mark_step(20)
    ev = [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert len(ev) == 1
    assert ev[0].detail["read_window_steps"] == [11, 20] and ev[0].detail["last_clean_read_steps"] == [10, 10]


def test_paused_compares_nothing_until_resumed():
    m = _model()
    g, s = _scan(m)
    s.paused = True
    _flip(m[0].weight, 0, 0)
    assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"] and s.stats["entries_compared"] == 0
    s.paused = False
    assert [e.detail["where"] for e in g.poll(wait=True) if e.detector == "serve_weights"] == [g.layers[0]["name"]]


def test_a_comparison_read_back_later_is_judged_when_it_arrives(monkeypatch):
    """The CUDA path: the flags are copied without waiting, and judged at a later read once their event says so."""
    from sdcguard.serve_weights import WeightScan

    class Ev:
        def __init__(self):
            self.done = False

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True
    made = []

    def copy(flags, dev):
        made.append(Ev())
        return flags.clone(), made[-1]
    monkeypatch.setattr(WeightScan, "_async", staticmethod(lambda dev: True))
    monkeypatch.setattr(WeightScan, "_copy", staticmethod(copy))
    m = _model()
    g, s = _scan(m)
    _flip(m[1].weight, 2, 2)
    assert not [e for e in g.poll() if e.detector == "serve_weights"]           # started, not back yet
    assert made and not made[0].done
    made[0].done = True
    ev = [e for e in g.poll() if e.detector == "serve_weights"]                 # back: judged
    assert [e.detail["where"] for e in ev] == [g.layers[1]["name"]]
    _flip(m[2].weight, 0, 0)
    g.poll()
    assert [e.detail["where"] for e in g.poll(wait=True) if e.detector == "serve_weights"] == [g.layers[2]["name"]]


def test_scan_alerts_are_counted_in_the_metrics():
    from prometheus_client import CollectorRegistry
    from sdcguard.alerts import Metrics
    reg = CollectorRegistry()
    m = _model()
    g, s = _scan(m)
    g.metrics = Metrics(registry=reg)
    g.poll(wait=True)
    _flip(m[0].weight, 1, 1)
    g.poll(wait=True)
    assert reg.get_sample_value("sdcguard_all_alerts_total") == 1
    assert reg.get_sample_value("sdcguard_weight_scan_layers") == 3
    assert reg.get_sample_value("sdcguard_weight_scan_passes_total") >= 1


def test_the_fingerprint_kernel_matches_the_pytorch_version_and_names_a_flip():
    pytest.importorskip("triton")
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ, TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1",
               PYTHONPATH=os.pathsep.join([os.path.join(os.path.dirname(here), "src"), here,
                                           os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, os.path.join(here, "_serve_weights_compare.py")], env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    assert "ALL MATCH" in p.stdout


def test_without_the_kernel_the_pytorch_version_is_used(monkeypatch):
    from sdcguard import serve_weights as sw
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.delenv("SDCGUARD_FAST_CPU_TEST", raising=False)
    monkeypatch.setattr(sw, "STATUS", {})
    t = torch.randn(5000).to(torch.bfloat16)
    assert not sw.kernel_usable(torch.device("cpu"))
    assert torch.equal(sw._fingerprints([t], torch.device("cpu"))[0], sw.fingerprint(t))
    monkeypatch.setattr(sw, "HAVE_TRITON", True)            # (the setting's message, with or without Triton installed)
    monkeypatch.setenv("SDCGUARD_WEIGHT_SCAN_FAST", "0")
    assert "turned off" in sw._why_not(torch.device("cpu"))
    monkeypatch.setattr(sw, "HAVE_TRITON", False)
    assert sw._why_not(torch.device("cpu")) == "Triton isn't installed"


def test_a_kernel_that_fails_its_first_use_comparison_or_a_launch_falls_back(monkeypatch):
    """A kernel giving another answer at first use isn't used (and says why); one whose later launch raises is
    turned off on that device and element size, and the tensor's fingerprint comes from the PyTorch version."""
    from sdcguard import serve_weights as sw
    monkeypatch.setattr(sw, "STATUS", {})
    monkeypatch.setattr(sw, "_why_not", lambda device: None)
    calls = []

    def wrong(t, row):
        calls.append(1)
        row += 1                                             # another answer

    monkeypatch.setattr(sw, "_launch", wrong)
    t = torch.randn(3000).to(torch.bfloat16)
    ok, why = sw.verify_kernel(torch.device("cpu"), 2)
    assert not ok and "first-use comparison failed" in why and calls
    assert torch.equal(sw._fingerprints([t], torch.device("cpu"))[0], sw.fingerprint(t))

    def boom(t, row):
        raise RuntimeError("launch failed for this variant")

    monkeypatch.setattr(sw, "_launch", boom)
    sw.STATUS[sw._key(torch.device("cpu"), 4)] = (True, "")  # passed its first use
    f = torch.randn(3000)
    assert torch.equal(sw._fingerprints([f], torch.device("cpu"))[0], sw.fingerprint(f))
    ok, why = sw.STATUS[sw._key(torch.device("cpu"), 4)]
    assert not ok and "a launch failed" in why and "launch failed for this variant" in why
    m = _model()
    g, s = _scan(m)
    kern = g.coverage_report()["weights"]["kernel"]
    assert sorted(kern) == ["cpu 2-byte", "cpu 4-byte"], kern
    assert kern["cpu 2-byte"].startswith("PyTorch (its first-use comparison failed: AssertionError")
    assert kern["cpu 4-byte"].startswith("PyTorch (a launch failed: RuntimeError")


def test_a_verification_age_target_sets_the_bytes_a_read_compares(monkeypatch, caplog):
    """With a target, each read compares enough that a pass takes at most 2/3 of it at the measured time between
    reads (never less than the budget, never more than max_mb), the oldest verification age stays under the target,
    and a target max_mb can't meet is reported (target_met False, one warning)."""
    import logging
    from sdcguard import serve_weights as sw
    clock = [100.0]
    monkeypatch.setattr(sw.time, "monotonic", lambda: clock[0])
    m = _model((64, 96, 96, 96, 64, 32))
    g = ServeGuard().attach(m)
    s = g.weight_scan(budget_mb=1 / 1024, target_age=3.0)                  # a 1 KB budget
    total = s.stats["bytes"]
    for _ in range(12):
        clock[0] += 0.5                                                     # a read every 0.5 s
        assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    want = total // 4 + 1                                                   # 2 s for a pass: 4 reads
    assert s.stats["budget_bytes_now"] == max(1024, want) and s.stats["target_met"] is True
    assert s.oldest_age() <= 3.0 and s.coverage()["oldest_verification_age_seconds"] <= 3.0
    assert s.coverage()["oldest_within_target"] is True                    # measured, not just scheduled
    s2 = sw.WeightScan(g, budget_mb=1 / 1024, target_age=3.0, max_mb=2 / 1024)   # at most 2 KB a read
    for e in s.entries:
        s2.add(e["name"], e["get"], e["kind"], e["module"])
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        for _ in range(6):
            clock[0] += 0.5
            s2.poll(wait=True)
    assert s2.stats["target_met"] is False and s2.stats["budget_bytes_now"] == 2048
    assert sum("can't compare every weight within" in r.getMessage() for r in caplog.records) == 1
    with caplog.at_level(logging.WARNING, logger="sdcguard"):             # met again, then unmet: no second warning
        s2.max_budget = 1 << 30
        clock[0] += 0.5
        s2.poll(wait=True)
        assert s2.stats["target_met"] is True
        s2.max_budget = 2048
        clock[0] += 0.5
        s2.poll(wait=True)
    assert sum("can't compare every weight within" in r.getMessage() for r in caplog.records) == 1
    clock[0] += 600                                                         # an idle server: the gap counts as 4 s
    assert s2.coverage()["oldest_within_target"] is False                   # 10 minutes unread: the schedule can't
    s2.poll(wait=True)                                                      # make an age it doesn't measure true
    assert s2._interval <= 0.8 * 0.5 + 0.2 * 4.0 + 1e-9
    s3 = sw.WeightScan(g, budget_mb=1 / 1024)                               # no target: the budget alone
    s3.add("x", s.entries[0]["get"], "checked")
    clock[0] += 0.5
    s3.poll(wait=True)
    assert s3.stats["budget_bytes_now"] == 1024 and s3.stats["target_met"] is None
    assert s3.coverage()["oldest_within_target"] is None


def _repairing(model, budget_mb=64.0):
    g = ServeGuard().attach(model)
    s = g.weight_scan(budget_mb=budget_mb, repair=True)
    return g, s


def test_repair_writes_a_changed_layer_back_and_verifies_it():
    """With repair on, a weight bit changed after load is alerted, then written back in place from the copy kept at
    load at the end of the same read and compared again: the layer holds its load-time values (the same tensor), the
    alert stands, a record says it was restored (not an alert: not counted as one), and a change that comes back is
    alerted and restored again."""
    from prometheus_client import CollectorRegistry
    from sdcguard.alerts import Metrics
    m = _model()
    g, s = _repairing(m)
    reg = CollectorRegistry()
    g.metrics = Metrics(registry=reg)
    clean = m[1].weight.detach().clone()
    ptr = m[1].weight.data_ptr()
    assert s.stats["host_copy_bytes"] == sum(p.numel() * p.element_size() for p in m.parameters())
    assert g.poll(wait=True) == []
    _flip(m[1].weight, 17, 9)
    alerts = [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert [a.detail["finding"] for a in alerts] == ["weights_changed"]   # (the repair is a record, not an alert)
    assert torch.equal(m[1].weight, clean) and m[1].weight.data_ptr() == ptr
    rec = g.log.events[-1]
    assert rec.detector == "serve_weights" and rec.detail["finding"] == "weights_restored"
    assert rec.detail["recheck"]["verdict"] == "weights_restored" and rec.detail["repairs"] == 1
    assert rec.detail["values_changed"] == 1 and rec.detail["values"] == m[1].weight.numel() + m[1].bias.numel()
    assert s.coverage()["repair"]["not_compared_since_repair"] == [g.layers[1]["name"]]
    assert g.poll(wait=True) == []
    cov = g.coverage_report()["weights"]
    assert cov["changed"] == [] and cov["repair"]["repaired"] == 1 and cov["repair"]["entries_repaired"] == \
        [g.layers[1]["name"]]
    assert reg.get_sample_value("sdcguard_all_alerts_total") == 1
    assert reg.get_sample_value("sdcguard_confirmed_alerts_total") == 1
    assert reg.get_sample_value("sdcguard_weight_repairs_total", {"result": "restored"}) == 1
    assert cov["repair"]["not_compared_since_repair"] == []           # (the read after the repair compared it)
    for _ in range(3):
        assert g.poll(wait=True) == []                              # compared again from then on: clean
    _flip(m[1].weight, 3, 14)                                       # it comes back
    assert [e.detail["finding"] for e in g.poll(wait=True)] == ["weights_changed"]
    g.poll(wait=True)
    assert torch.equal(m[1].weight, clean) and s.stats["repaired"] == 2
    assert reg.get_sample_value("sdcguard_weight_repairs_total", {"result": "restored"}) == 2


def test_a_repair_that_cant_restore_the_reference_leaves_the_layer_changed():
    """A clean copy that is itself wrong restores something that still doesn't match the reference: the layer stays
    changed and a record says the repair failed."""
    m = _model()
    g, s = _repairing(m)
    g.poll(wait=True)
    e = next(x for x in s.entries if x["name"] == g.layers[0]["name"])
    _flip(e["clean"][0], 5, 3)                                      # the copy kept at load is bad
    _flip(m[0].weight, 11, 2)
    g.poll(wait=True)
    g.poll(wait=True)
    rec = g.log.events[-1]
    assert rec.detail["finding"] == "weights_restore_failed" and rec.detail["recheck"]["verdict"] == "restore_failed"
    assert g.coverage_report()["weights"]["changed"] == [g.layers[0]["name"]] and s.stats["repair_failed"] == 1
    assert [x for x in g.poll(wait=True) if x.detector == "serve_weights"] == []   # still alerted once only


def test_without_repair_no_copy_is_kept_and_nothing_is_written_back():
    m = _model()
    g, s = _scan(m)
    assert s.stats["host_copy_bytes"] == 0 and all("clean" not in e for e in s.entries)
    g.poll(wait=True)
    _flip(m[2].weight, 0, 8)
    g.poll(wait=True)
    g.poll(wait=True)
    assert g.coverage_report()["weights"]["changed"] == [g.layers[2]["name"]] and s.stats["repaired"] == 0


def test_repair_writes_into_inference_mode_weights_too():
    """vLLM may build its weights under inference mode: the write-back happens in it for such tensors."""
    with torch.inference_mode():
        m = _model()
    assert m[0].weight.is_inference()
    clean = m[0].weight.clone()
    g, s = _repairing(m)
    g.poll(wait=True)
    with torch.inference_mode():
        m[0].weight.view(-1).view(torch.int16)[4] ^= 1 << 10
    g.poll(wait=True)
    g.poll(wait=True)
    assert torch.equal(m[0].weight, clean) and s.stats["repaired"] == 1


def test_repair_refuses_a_change_of_more_values_than_a_fault_makes():
    """An update made without refresh() changes nearly every value: writing the old weights over it would be a silent
    error of the repair's own making. It is refused (the layer keeps the new values), the record says how many
    values changed, the alert stands, and it isn't tried again at later reads."""
    m = _model()
    g, s = _repairing(m)
    g.poll(wait=True)
    with torch.no_grad():
        m[1].weight.mul_(1.5)
    updated = m[1].weight.detach().clone()
    found = [e for e in g.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["finding"] for e in found] == ["weights_changed"]
    rec = g.log.events[-1]
    assert rec.detail["finding"] == "weights_restore_failed" and rec.detail["values_changed"] > sw.REPAIR_MAX_VALUES
    assert rec.detail["values_changed_at_least"] is True
    assert "at least 6144 of 6208" in rec.detail["recheck"]["why"]         # (counting stops past the limit)
    assert "more than a fault in memory would change" in rec.detail["recheck"]["why"]
    assert "refresh()" in rec.detail["recheck"]["why"] and "take the server out of service" in rec.detail["reading"]
    assert torch.equal(m[1].weight, updated) and s.stats["repair_failed"] == 1
    n = len(g.log.events)
    for _ in range(3):
        g.poll(wait=True)
    assert len(g.log.events) == n and s.stats["repair_failed"] == 1
    assert g.coverage_report()["weights"]["changed"] == [g.layers[1]["name"]]
    s2 = sw.WeightScan(g, repair=True, repair_max_values=0)         # a limit of 0: nothing is written back
    s2.poll(wait=True)                                              # (takes in the guard's layers as they are now)
    _flip(m[0].weight, 3, 4)
    flipped = m[0].weight.detach().clone()
    s2.poll(wait=True)
    assert s2.stats["repaired"] == 0 and s2.stats["repair_failed"] == 1 and torch.equal(m[0].weight, flipped)


def test_the_repair_record_names_the_steps_and_requests_that_used_the_changed_weights():
    """The record covers the steps from the alert's window (after the last clean comparison) to the repair, asks the
    guard's attribution for the requests in them, and the next clean comparison starts after the repair."""
    m = _model()
    g, s = _repairing(m)
    named = []
    g.attribute = lambda ev: named.append(ev.detail.get("finding"))
    g.mark_step(1)
    g.poll(wait=True)
    _flip(m[2].weight, 1, 13)
    for step in (2, 3):
        g.mark_step(step)
    g.poll(wait=True)
    rec = g.log.events[-1]
    assert rec.detail["finding"] == "weights_restored" and rec.detail["repaired_at_step"] == 3
    assert rec.detail["read_window_steps"] == [2, 3] and rec.detail["last_clean_read_steps"] == [1, 1]
    assert named == ["weights_changed", "weights_restored"]
    e = next(x for x in s.entries if x["name"] == g.layers[2]["name"])
    assert e["last_clean_step"] == 3


def test_a_comparison_in_flight_before_a_repair_doesnt_alert_again_after_it(monkeypatch):
    """Two comparisons of a changed entry in flight: the first alerts and the entry is restored; the second, which
    saw the changed weights before the repair, comes back afterwards and is dropped, not alerted."""
    from sdcguard.serve_weights import WeightScan

    class Ev:
        def __init__(self):
            self.done = False

        def query(self):
            return self.done

        def synchronize(self):
            self.done = True
    made = []

    def copy(flags, dev):
        made.append(Ev())
        return flags.clone(), made[-1]
    monkeypatch.setattr(WeightScan, "_async", staticmethod(lambda dev: True))
    monkeypatch.setattr(WeightScan, "_copy", staticmethod(copy))
    m = _model()
    g, s = _repairing(m)
    clean = m[0].weight.detach().clone()
    _flip(m[0].weight, 5, 12)
    g.poll()                                     # comparison 1 started
    g.poll()                                     # comparison 2 started, 1 not back
    assert len(made) == 2
    made[0].done = True
    found = [e for e in g.poll() if e.detector == "serve_weights"]
    assert [e.detail["finding"] for e in found] == ["weights_changed"] and torch.equal(m[0].weight, clean)
    for ev in made:
        ev.done = True
    for _ in range(3):
        assert not [e for e in g.poll() if e.detector == "serve_weights"]
    assert s.stats["changed"] == 1 and s.stats["repaired"] == 1


def test_a_refresh_from_the_alert_handler_cancels_the_repair():
    """An operator whose on_alert refreshes (the change was an update they meant): nothing is written back."""
    m = _model()
    g, s = _repairing(m)
    g.poll(wait=True)
    g.on_alert = lambda ev: s.refresh(name=ev.detail["where"]) if ev.detail.get("finding") == "weights_changed" \
        else None
    _flip(m[1].weight, 2, 3)
    changed = m[1].weight.detach().clone()
    g.poll(wait=True)
    assert torch.equal(m[1].weight, changed) and s.stats["repaired"] == 0 and s.stats["repair_failed"] == 0
    assert not [e for e in g.log.events if e.detail.get("finding") in ("weights_restored", "weights_restore_failed")]
    assert not [e for e in g.poll(wait=True) if e.detector == "serve_weights"]       # the new values are the reference


def test_repair_turns_itself_off_when_the_copies_dont_fit_in_host_memory(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(sw, "_host_room", lambda: 4000)
    monkeypatch.setattr(sw, "_local_gpus", lambda: 2)
    m = _model()
    with caplog.at_level(logging.WARNING, logger="sdcguard"):
        g, s = _repairing(m)
    assert s.repair is False and "allowance of" in s.repair_off and s.stats["host_copy_bytes"] == 0
    assert all(not e.get("clean") for e in s.entries)
    assert sum("weight repair is off" in r.getMessage() for r in caplog.records) == 1
    cov = s.coverage()["repair"]
    assert cov["on"] is False and cov["off_because"] == s.repair_off and cov["host_allowance_bytes"] == 1000
    g.poll(wait=True)
    _flip(m[0].weight, 0, 9)
    assert [e.detail["finding"] for e in g.poll(wait=True) if e.detector == "serve_weights"] == ["weights_changed"]
    assert s.stats["repaired"] == 0 and not [e for e in g.log.events if e.detail.get("finding") == "weights_restored"]


def test_a_refresh_writes_into_the_copies_it_already_keeps_and_an_excluded_entry_lets_its_copy_go():
    m = _model()
    g, s = _repairing(m)
    e = next(x for x in s.entries if x["name"] == g.layers[0]["name"])
    before = [id(c) for c in e["clean"]]
    total = s.stats["host_copy_bytes"]
    with torch.no_grad():
        m[0].weight.add_(1.0)
    s.refresh(name=e["name"])
    assert [id(c) for c in e["clean"]] == before and torch.equal(e["clean"][0], m[0].weight.detach())
    assert s.stats["host_copy_bytes"] == total
    s._exclude(e, "a test")
    assert e["clean"] is None and s.stats["host_copy_bytes"] == total - sum(
        p.numel() * p.element_size() for p in m[0].parameters())
    e["alerted"] = True
    assert s._restore(e)[0] is False and "no longer compares" in s._restore(e)[1]


def test_with_no_clean_comparison_yet_the_window_starts_at_the_reference():
    """A change found at an entry's first comparison may have happened at any step since its reference was taken:
    since load (step 0 on) when no step was marked then, or since the step of the last refresh."""
    m = _model()
    g, s = _repairing(m)
    for step in range(1, 6):
        g.mark_step(step)                              # steps run, no read yet
    _flip(m[1].weight, 9, 12)
    g.mark_step(6)
    g.poll(wait=True)
    (alert,) = [e for e in g.log.events if e.detail.get("finding") == "weights_changed"]
    assert alert.detail["read_window_steps"] == [0, 6] and alert.detail["window_since"] == "load"
    rec = g.log.events[-1]
    assert rec.detail["finding"] == "weights_restored" and rec.detail["read_window_steps"] == [0, 6]
    s.refresh(name=g.layers[2]["name"])                # a refresh after step 6, then a change in that entry
    g.mark_step(7)
    g.mark_step(8)
    _flip(m[2].weight, 3, 12)
    g.poll(wait=True)
    alert = [e for e in g.log.events if e.detail.get("finding") == "weights_changed"][-1]
    assert alert.detail["where"] == g.layers[2]["name"] and alert.detail["read_window_steps"] == [7, 8]


def test_without_a_reading_of_free_host_memory_repair_stays_off(monkeypatch):
    monkeypatch.setattr(sw, "_host_room", lambda: None)
    m = _model()
    g, s = _repairing(m)
    assert s.repair is False and "couldn't be read" in s.repair_off and s.stats["host_copy_bytes"] == 0
    monkeypatch.setattr(sw, "_host_room", lambda: -5000)             # (a cgroup just over its limit)
    s2 = sw.WeightScan(g, repair=True)
    s2.add("x", lambda: (m[0].weight,), "checked", m[0])
    assert s2.repair is False and s2._host_allowance == 0


def test_a_restore_that_fails_after_writing_still_tells_the_guard(monkeypatch):
    """The write-back happened, then the check of it raised: the guard's checks must still know the weights were
    written after that step (outputs from before it can't be judged with them), and the record says it failed."""
    m = _model()
    g, s = _repairing(m)
    g.mark_step(1)
    g.poll(wait=True)
    counted = []
    real_count = sw.WeightScan._count_changed

    def count(self, pairs):
        counted.append(1)
        return real_count(self, pairs)
    real_fp = sw._fingerprints

    def fp(tensors, dev):
        if counted:                                    # the check after the write-back
            raise RuntimeError("the fingerprint failed")
        return real_fp(tensors, dev)
    monkeypatch.setattr(sw.WeightScan, "_count_changed", count)
    monkeypatch.setattr(sw, "_fingerprints", fp)
    _flip(m[0].weight, 4, 13)
    g.mark_step(2)
    g.poll(wait=True)
    rec = g.log.events[-1]
    assert rec.detail["finding"] == "weights_restore_failed" and "the fingerprint failed" in rec.detail["recheck"]["why"]
    assert g.layers[0]["restored"]["step"] == 2 and s.stats["repair_failed"] == 1
