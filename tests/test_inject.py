import random
from collections import Counter

import torch
import torch.nn as nn

from sdcguard import FaultInjector, FaultModel, corrupt_


def test_fault_mix_matches_nvidia_proportions():
    rng = random.Random(0)
    kinds = Counter()
    for _ in range(3000):
        t = torch.randn(4, 256).bfloat16()
        info = corrupt_(t, rng)
        kinds[info["kind"]] += 1
    n = sum(kinds.values())
    assert abs(kinds["zero"] / n - 0.5068) < 0.04
    assert abs(kinds["bitflip"] / n - 0.4831) < 0.04
    assert kinds["special"] / n < 0.03


def test_corruption_stays_inside_one_warp_chunk():
    rng = random.Random(1)
    for _ in range(200):
        t = torch.randn(8, 512).bfloat16()
        before = t.clone()
        info = corrupt_(t, rng, FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0))
        changed = (t != before).nonzero()
        if changed.numel():
            assert set(changed[:, 0].tolist()) == {info["row"]}
            cols = changed[:, 1]
            assert cols.min() >= info["col"] and cols.max() < info["col"] + 32


def test_tile_events_hit_one_tile_with_expected_share():
    rng = random.Random(3)
    fm = FaultModel(p_tile=1.0, tile=(64, 64), tile_frac=(0.2, 0.75), tile_p_zero=1.0)
    for _ in range(20):
        t = torch.randn(256, 256).bfloat16() + 10      # no natural zeros
        info = corrupt_(t, rng, fm)
        zeros = (t == 0).nonzero()
        assert info["kind"] == "tile_zero"
        assert zeros[:, 0].min() >= info["row"] and zeros[:, 0].max() < info["row"] + 64
        assert zeros[:, 1].min() >= info["col"] and zeros[:, 1].max() < info["col"] + 64
        share = zeros.shape[0] / (64 * 64)
        assert 0.15 < share < 0.8


def test_tile_bitflips_change_values():
    rng = random.Random(4)
    fm = FaultModel(p_tile=1.0, tile=(32, 32), tile_p_zero=0.0)
    t = torch.randn(64, 64).bfloat16()
    before = t.clone()
    info = corrupt_(t, rng, fm)
    assert info["kind"] == "tile_bitflip"
    assert int((t != before).sum()) > 0


def test_arm_on_only_hits_target_module():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
    x = torch.randn(16, 64)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        clean0 = model[0](x).clone()
        inj = FaultInjector(rate=1.0, seed=2).arm_on([model[1]])
        with inj:
            out0 = model[0](x)
        inj.remove_hooks()
    assert torch.equal(clean0, out0)        # untouched module stays clean
    assert len(inj.events) == 0


def test_rel_change_is_recorded_and_sensible():
    from sdcguard.inject import fault_model
    rng = random.Random(5)
    zero = [corrupt_(torch.randn(8, 512).bfloat16() + 1, rng, fault_model("zero")) for _ in range(50)]
    assert all(e["rel_change"] > 0.01 for e in zero)          # zeroing values of ~1 is visible
    mant = [corrupt_(torch.randn(8, 512).bfloat16(), rng, fault_model("mantissa")) for _ in range(50)]
    assert all(e["rel_change"] < 0.2 for e in mant)           # mantissa flips stay small


def test_mantissa_only_flips_never_touch_sign_or_exponent():
    from sdcguard.inject import fault_model
    rng = random.Random(6)
    fm = fault_model("mantissa")
    for _ in range(200):
        t = torch.randn(4, 256).bfloat16()
        before = t.clone()
        corrupt_(t, rng, fm)
        diff = (t.view(torch.int16) ^ before.view(torch.int16)).to(torch.int32) & 0xFFFF
        assert int((diff & 0xFF80).sum()) == 0                # bits 7..15 untouched in bf16


def test_named_fault_models():
    from sdcguard.inject import fault_model
    assert fault_model("zero", "warp").p_zero == 1.0
    assert fault_model("nvidia", "tile").p_tile == 1.0
    assert fault_model("nvidia", "warp", specials=True).p_special > 0
    assert fault_model("nvidia", "warp").p_special == 0


def test_match_accepts_a_reshaped_view_of_the_corrupted_output():
    # Apple GPUs run nn.Linear as aten::linear on a 3D input; the detector checks a 2D view of
    # the output, so matching a fault to its check must not depend on the shape.
    inj = FaultInjector(rate=1.0)
    out = torch.randn(4, 16, 64).bfloat16()
    inj.events.append({"kind": "zero", "checked": False, "caught": False})
    inj.last_out, inj.pending = out, 0
    assert inj.match(out[:2].reshape(-1, 64)) is None           # different size: not this fault
    assert inj.match(out.reshape(-1, 64)) == 0
    assert inj.events[0]["checked"] and inj.pending is None
