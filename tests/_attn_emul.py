"""A model of how vLLM's attention backends round, to test the attention check's bound against (tests only).

FlashAttention (2, 3) and vLLM's Triton attention: scores in fp32, an online softmax over tiles of keys (a running
maximum, rescaling by exp(m_old - m_new)), P rounded to the KV dtype before the P V product, which accumulates in fp32,
the sum of P in fp32 (from P before rounding; `l_from_rounded` takes it after), the output divided by it and rounded
once. Split-KV: each split's output and log-sum-exp, combined in fp32 (`round_partials` rounds the partial outputs to
the dtype first, as a cascade path would; vLLM 0.32 has none, so it is a harsher case than the real ones)."""
import torch


def emulate(q, K, V, scale, dtype, tile=64, splits=1, round_partials=False, l_from_rounded=False):
    """o for one query row q [D] over keys K, values V [L, D] (all already in dtype's values), in dtype."""
    q, K, V = q.float(), K.float(), V.float()
    n = K.shape[0]
    edges = [round(i * n / splits) for i in range(splits + 1)]
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        m = torch.tensor(float("-inf"))
        l = torch.tensor(0.0)
        acc = torch.zeros(V.shape[1])
        for t0 in range(a, b, tile):
            t1 = min(b, t0 + tile)
            s = (K[t0:t1] @ q) * scale
            mn = torch.maximum(m, s.max())
            alpha = torch.exp(m - mn)
            p = torch.exp(s - mn)
            pr = p.to(dtype).float()
            l = l * alpha + (pr.sum() if l_from_rounded else p.sum())
            acc = acc * alpha + pr @ V[t0:t1]
            m = mn
        o = acc / l
        if round_partials:
            o = o.to(dtype).float()
        parts.append((o, m + torch.log(l)))
    lse = torch.stack([p[1] for p in parts])
    M = lse.max()
    w = torch.exp(lse - M)
    o = sum(wi * p[0] for wi, p in zip(w, parts)) / w.sum()
    return o.to(dtype)


VARIANTS = [dict(tile=64), dict(tile=128, splits=3), dict(tile=32, splits=2, round_partials=True),
            dict(tile=64, l_from_rounded=True), dict(tile=16, splits=5, round_partials=True, l_from_rounded=True)]
