"""Run by tests/test_serve_attn.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the attention check's Triton
kernels against its PyTorch version on the CPU, and the alarm path that keeps the attended keys and values. Prints
ALL MATCH when every case agrees. (The interpreter runs every program in Python: the key range is split among 4
programs per sample here, as on GPUs among 32, except in one case that keeps 32.)"""
import os

import torch

from sdcguard import serve_attn as sa
from sdcguard.serve import ServeGuard

COUNTS = [sa.C_CALLS, sa.C_SAMPLES, sa.C_SKIP, sa.C_UNJ, sa.C_BADBLK, sa.C_KEYS, sa.C_WN, sa.C_WBAD, sa.C_WSLOT]


def both(c, step, **kw):
    out = []
    for fn in (sa.attn_check_torch, sa.attn_check_triton):
        b = sa.buffers("cpu", 2, c["q"].shape[-1], 512, c["q"].dtype)
        sa.run_case(fn, c, b, step=step, slot=1, **kw)
        assert b["state"][0].abs().sum() == 0 and b["capm"][0].abs().sum() == 0      # only its own row
        out.append(b)
    return out


def same(a, b, what):
    sa_, sb = a["state"][1].tolist(), b["state"][1].tolist()
    assert [sa_[k] for k in COUNTS] == [sb[k] for k in COUNTS], (what, sa_, sb)
    za, zb = sa._f32(sa_[sa.C_ZT] >> 32), sa._f32(sb[sa.C_ZT] >> 32)
    assert abs(za - zb) <= 1e-3 * max(za, 1e-3) or za == zb, (what, za, zb)
    if za > sa.ATTN_Z:
        for col in (sa.C_ZT, sa.C_ZH, sa.C_ZD, sa.C_ZS, sa.C_ZP):
            assert sa_[col] & sa.M32 == sb[col] & sa.M32, (what, col)
        assert int(a["capm"][1, 0]) == 1 and int(b["capm"][1, 0]) == 1
    for col in (sa.C_WLAST, sa.C_WWHERE):
        assert sa_[col] == sb[col], (what, col)
    return za


def main():
    os.environ["SDCGUARD_ALLOW_CPU"] = "1"
    for dt in (torch.bfloat16, torch.float16):
        sa.NSPLIT = 4
        ok, why = sa.verify_kernel(torch.device("cpu"), dt)
        assert ok, why
    n = 0
    for dt in (torch.bfloat16, torch.float16):
        for D, Hq, Hkv, lens in ((64, 4, 4, ((5, 5), (0, 0))), (128, 8, 2, ((130, 1), (33, 1), (70, 20), (0, 0))),
                                 (96, 2, 1, ((17, 3), (1, 1), (0, 0), (0, 0)))):
            c = sa.make_case("cpu", dt, D=D, Hq=Hq, Hkv=Hkv, lens=lens, NB=24, seed=D)
            cases = {"clean": c}
            w = dict(c, o=c["o"].clone())
            w["o"][:, :, D - 1] = w["o"][:, :, D - 1].float() * -2
            cases["wrong last dim"] = w
            k = dict(c, key=c["key"].clone())
            k["key"][:, Hkv - 1, 0] = k["key"][:, Hkv - 1, 0].float() + 1
            cases["wrong write"] = k
            for name, cc in cases.items():
                for step in range(3):
                    a, b = both(cc, step)
                    z = same(a, b, (dt, D, name, step))
                    assert (z <= 1.0) == (name != "wrong last dim") or name == "wrong write", (name, z)
                    n += 1
            for step in range(2):                          # no write check; a window of 9 keys
                same(*both(c, step, has_write=False, left=8), (dt, D, "window", step))
                n += 1
    # the split as on GPUs: 32 programs per sample, most with few keys or none
    sa.NSPLIT = 32
    c = sa.make_case("cpu", torch.bfloat16, D=64, Hq=2, Hkv=1, lens=((300, 1), (3, 3), (0, 0)), NB=24, seed=9)
    for step in range(2):
        z = same(*both(c, step), ("32 programs", step))
        assert z <= 1.0
        n += 1
    sa.NSPLIT = 4
    # an alarm keeps its sample's keys and values; the alert's float64 recheck confirms the wrong output
    sa.STATUS.clear()
    c = sa.make_case("cpu", torch.bfloat16, D=64, Hq=4, Hkv=2, lens=((90, 1), (0, 0)), NB=12, seed=2)
    c["o"][0, :, 11] = c["o"][0, :, 11].float() + 1

    class Attention(torch.nn.Module):
        def __init__(self, impl):
            super().__init__()
            self.impl, self.layer_name, self.head_size_v = impl, "model.layers.0.self_attn.attn", 64

    class FlashAttentionImpl:
        num_heads, num_kv_heads, head_size, scale = 4, 2, 64, 64 ** -0.5
        attn_type, alibi_slopes, sinks, logits_soft_cap, sliding_window = "decoder", None, None, 0, (-1, -1)
        kv_cache_dtype, kv_sharing_target_layer_name = "auto", None

    class Meta:
        query_start_loc, seq_lens, block_table, slot_mapping = c["qsl"], c["seq_lens"], c["bt"], c["slots"]
        num_actual_tokens, causal = c["q"].shape[0], True

    impl = FlashAttentionImpl()
    lay = Attention(impl)
    g = ServeGuard()
    ac = g.attn_checks()
    ac.add(lay, impl, lay.layer_name, "cpu")
    ac.finalize(max_len=200)
    assert ac.kernel_status() and all(v[0] for v in ac.kernel_status().values()), ac.kernel_status()
    g.mark_step(1)
    sa.check_call(lay._sdcguard_attn, impl, c["q"], c["key"], c["val"], c["kv"], Meta(), c["o"])
    g.end_step()
    ev = g.poll(wait=True)
    assert len(ev) == 1, [e.detail for e in ev]
    d = ev[0].detail
    assert d["dim"] == 11 and d["row"] == 0 and d["captured"]["keys_kept"] and d["captured"]["position"] == 89
    assert d["place"].startswith("the kept sample's worst output") and d["recheck"]["dim"] == 11
    assert d["recheck"]["verdict"] == "output_wrong" and d["recheck"]["keys"] == 90, d["recheck"]
    hk = d["captured"]["kv_head"]
    kept = ac.bufs[torch.device("cpu")]["capkv"]
    assert int(ac.bufs[torch.device("cpu")]["capkvm"][0]) == 0          # freed by the alert
    rows = torch.stack([c["kv"][int(c["bt"][0, t // 16]), hk, t % 16] for t in range(90)])
    assert torch.equal(kept[:90, 0, :64], rows[:, :64]) and torch.equal(kept[:90, 1, :64], rows[:, 64:])
    n += 1
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
