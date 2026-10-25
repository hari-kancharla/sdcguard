"""Scheduled weight integrity (serve_weights.WeightScan) on a CPU: a weight bit changed after load is named within a
pass over the model and only once, a clean model raises nothing, the byte budget sets how many reads a pass takes,
an intended update (refresh) isn't called corruption, tensors that moved aren't compared, quantized layers' own
tensors and layers outside the arithmetic checks are scanned too, and the fingerprint sees every single bit flip."""
import torch
import torch.nn as nn

from sdcguard.serve import ServeGuard
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


def test_a_flip_and_its_cancelling_twin_in_another_block_is_still_seen():
    t = torch.zeros(2 * BLOCK, dtype=torch.int16)
    ref = fingerprint(t)
    c = t.clone()
    c[5] = 1
    c[BLOCK + 5] = -1                             # the plain sum is unchanged; the block-weighted one isn't
    assert not torch.equal(fingerprint(c), ref)


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
    t = torch.randint(-2 ** 15, 2 ** 15 - 1, (3 * sw.CHUNK + 5 * BLOCK + 3,), dtype=torch.int16)
    v = t.to(torch.int64)
    full = t.numel() - t.numel() % BLOCK
    sums = v[:full].view(-1, BLOCK).sum(1)
    w = torch.arange(sums.numel()) % sw.POSITION_MOD + 1
    tail = v[full:].sum()
    expect = torch.stack([sums.sum() + tail, (sums * w).sum() + tail * ((full // BLOCK) % sw.POSITION_MOD + 1)])
    assert torch.equal(fingerprint(t), expect)


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
