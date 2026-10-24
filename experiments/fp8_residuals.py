"""How far vLLM's FP8 GEMM on Hopper lands from exact arithmetic on the same FP8 operands, and what a checksum check
of it would see on clean runs (docs/METHOD.md, "FP8 arithmetic on Hopper: what a check would see", written before
the run). Measured before building an FP8 check: the noise model has to fit what the kernel really does.

Qwen2.5-1.5B-Instruct in bf16 (transformers) reads 16 prompts; the inputs of 4 linear layers (q/k/v, o, gate/up,
down) in blocks 0, 9, 18 and 27 are kept, with the layers' weights. Each layer is quantized as vLLM's
`--quantization fp8` does on Hopper: the weight to FP8 E4M3 with one scale per tensor, the activations per token,
dynamically (vLLM's scaled_fp8_quant). Then three products of the same FP8 operands and scales:
  fast   vLLM's cutlass_scaled_mm (the call `--quantization fp8` makes; its Hopper kernels use FP8 fast accumulation)
  slow   torch._scaled_mm with use_fast_accum=False (cuBLASLt FP8 with full-precision accumulation), when available
  exact  As * Bs * (A @ B) in float64 (FP8 values are exact in float64)
Per layer and product: each output's error against exact (relative to the largest |exact| in its row), its sign
bias, and each 128-column block's sum against the exact block sum in units of what bf16 rounding of the outputs
alone explains (2^-8 * sqrt(sum of y^2 / 12), the size of the term the current check's noise model is built on):
the score a checksum check would see on clean FP8 arithmetic, before any extra term.

    python experiments/fp8_residuals.py --out runs/fp8_residuals
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BLOCKS = (0, 9, 18, 27)
KINDS = ("self_attn.q_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj")
PROMPTS = ["Explain how a hash map handles collisions.", "Write a short poem about the sea at night.",
           "What is 37 times 43? Show the steps.", "Summarize the causes of the French Revolution.",
           "Translate into Spanish: the train leaves at nine.", "List five uses of a binary search tree.",
           "Describe photosynthesis to a ten year old.", "Give three tips for a job interview.",
           "Why is the sky blue?", "Write a Python function that reverses a linked list.",
           "What are the main differences between TCP and UDP?", "Explain the rules of chess briefly.",
           "How does a refrigerator keep food cold?", "Plan a three day trip to Rome.",
           "What is the derivative of x^3 sin x?", "Describe a sunrise over the mountains."]


def capture(device):
    """{layer name: (inputs [tokens, K] bf16, weight [N, K] bf16)} from the bf16 model on the prompts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(device).eval()
    want = {f"model.layers.{b}.{k}" for b in BLOCKS for k in KINDS}
    seen: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if name in want:
            def hook(m, args, _n=name):
                seen.setdefault(_n, []).append(args[0].detach().reshape(-1, args[0].shape[-1]))
            hooks.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").to(device)
            model(**ids)
    for h in hooks:
        h.remove()
    mods = dict(model.named_modules())
    return {n: (torch.cat(xs), mods[n].weight.detach()) for n, xs in seen.items()}


def _stats(y, exact, bs: int = 128) -> dict:
    """y: [M, N] product as returned (bf16); exact: [M, N] float64. Errors relative to each row's largest |exact|."""
    import torch
    yd = y.double()
    err = yd - exact
    scale = exact.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    rel = (err / scale).abs().flatten()
    m, n = y.shape
    nb = n // bs
    ys = yd[:, :nb * bs].reshape(m, nb, bs)
    es = exact[:, :nb * bs].reshape(m, nb, bs)
    resid = (ys.sum(-1) - es.sum(-1)).abs()
    noise = (2.0 ** -8) * torch.sqrt((ys ** 2).sum(-1) / 12.0).clamp_min(1e-30)
    z = (resid / noise).flatten()
    q = lambda t, p: float(torch.quantile(t.float(), p)) if t.numel() else None  # noqa: E731
    return {"outputs": int(rel.numel()), "rel_err_max": float(rel.max()), "rel_err_p99": q(rel, 0.99),
            "rel_err_median": q(rel, 0.5), "signed_err_over_scale_mean": float((err / scale).mean()),
            "block_z_max": float(z.max()), "block_z_p999": q(z, 0.999), "block_z_p99": q(z, 0.99),
            "block_z_median": q(z, 0.5), "blocks": int(z.numel())}


def measure(device) -> dict:
    import torch
    from vllm import _custom_ops as ops
    out = {"model": MODEL, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device), "layers": {}}
    for name, (x, w) in capture(device).items():
        x = x.to(device)
        w = w.to(device)
        wq, ws = ops.scaled_fp8_quant(w)                                    # per tensor, static scale from max
        xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)     # per token, dynamic
        b = wq.t()                                                          # [K, N], column major as vLLM passes it
        fast = ops.cutlass_scaled_mm(xq, b, xs, ws, torch.bfloat16)
        exact = (xs.double() * ws.double()) * (xq.double() @ b.double())
        r = {"tokens": int(x.shape[0]), "K": int(x.shape[1]), "N": int(w.shape[0]),
             "fast": _stats(fast, exact)}
        try:
            slow = torch._scaled_mm(xq, b, scale_a=xs.float(), scale_b=ws.float().view(1, 1).expand(1, w.shape[0])
                                    .contiguous(), out_dtype=torch.bfloat16, use_fast_accum=False)
            r["slow"] = _stats(slow, exact)
            fa = torch._scaled_mm(xq, b, scale_a=xs.float(), scale_b=ws.float().view(1, 1).expand(1, w.shape[0])
                                  .contiguous(), out_dtype=torch.bfloat16, use_fast_accum=True)
            r["torch_fast"] = _stats(fa, exact)
        except Exception as e:  # noqa: BLE001
            r["slow"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        out["layers"][name] = r
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: v.get(kk) for kk in
                                                                        ("rel_err_max", "block_z_p99", "block_z_max")})
                                for k, v in r.items()}), flush=True)
    for kind in ("fast", "slow", "torch_fast"):
        zs = [r[kind]["block_z_max"] for r in out["layers"].values() if isinstance(r.get(kind), dict)
              and "block_z_max" in r[kind]]
        out[f"{kind}_block_z_max_over_layers"] = max(zs) if zs else None
        p99 = [r[kind]["block_z_p99"] for r in out["layers"].values() if isinstance(r.get(kind), dict)
               and "block_z_p99" in r[kind]]
        out[f"{kind}_block_z_p99_worst_layer"] = max(p99) if p99 else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import torch
    res = measure(torch.device("cuda"))
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "layers"}, indent=1))
    return 0 if res.get("fast_block_z_max_over_layers") is not None and not math.isnan(
        res["fast_block_z_max_over_layers"]) else 1


if __name__ == "__main__":
    sys.exit(main())
