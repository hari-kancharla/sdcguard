"""Run by tests/test_serve_weights.py with TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1: the weight scan's fingerprint
kernel against the PyTorch version on the CPU, then a scan that uses it naming a flipped bit. Prints ALL MATCH."""
import torch
import torch.nn as nn

from sdcguard import serve_weights as sw
from sdcguard.serve import ServeGuard


def main():
    dev = torch.device("cpu")
    launched = []
    real_launch = sw._launch

    def counting(t, row):
        launched.append(t.numel())
        real_launch(t, row)

    sw._launch = counting
    for es in (1, 2, 4, 8):
        ok, why = sw.verify_kernel(dev, es)
        assert ok, (es, why)
    g = torch.Generator().manual_seed(1)
    n = 0
    for dt in (torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.uint8, torch.int32):
        for size in ((5,), (4096,), (4097,), (3, 5000), (64, 300)):
            if dt in (torch.uint8, torch.int32):
                t = torch.randint(0, 200, size, generator=g).to(dt)
            else:
                t = (torch.randn(size, generator=g) * 10).to(dt)
            before = len(launched)
            got = sw._fingerprints([t, None, t.t() if t.dim() == 2 else t], dev)
            assert len(launched) == before + 2, (dt, size)          # the kernel made both rows, not the fallback
            want = torch.stack([sw.fingerprint(t), torch.zeros(2, dtype=torch.int64),
                                sw.fingerprint(t.t() if t.dim() == 2 else t)])
            assert torch.equal(got, want), (dt, size, got.tolist(), want.tolist())
            n += 1
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 96), nn.Linear(96, 64)).to(torch.bfloat16)
    gd = ServeGuard().attach(m)
    scan = gd.weight_scan()
    assert sw.STATUS[sw._key(dev, 2)] == (True, "")
    assert not [e for e in gd.poll(wait=True) if e.detector == "serve_weights"]
    with torch.no_grad():
        m[1].weight.view(-1).view(torch.int16)[77] ^= 1                # the lowest bit of one weight
    before = len(launched)
    hits = []
    for _ in range(4):
        hits += [e for e in gd.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["where"] for e in hits] == [gd.layers[1]["name"]], [e.detail for e in hits]
    assert len(launched) > before                               # the scan's comparisons used the kernel
    # a swap of two values in one block, which the earlier block-sum fingerprint couldn't see
    w = m[0].weight.view(-1).view(torch.int16)
    a, b = int(w[3]), int(w[4])
    assert a != b
    with torch.no_grad():
        w[3], w[4] = b, a
    hits = []
    for _ in range(4):
        hits += [e for e in gd.poll(wait=True) if e.detector == "serve_weights"]
    assert [e.detail["where"] for e in hits] == [gd.layers[0]["name"]], [e.detail for e in hits]
    print(f"ALL MATCH ({n} cases)")


if __name__ == "__main__":
    main()
