"""The data an FP8 matmul check's noise model is fitted to: what vLLM's FP8 GEMM does to block sums on real layers
(docs/METHOD.md, "Calibrating an FP8 matmul check", written before the run).

vLLM 0.32's FP8 linear layers multiply with CUTLASS: on Hopper every configuration uses FP8 fast accumulation (the
products of a whole output are summed in the tensor core's accumulator, 13 fractional bits kept and the rest
truncated at every 32-product step), on Ada most do (16-product steps). A block-checksum check of such a GEMM compares,
for a row i and a block b of 128 output columns, the sum of the outputs O with the exact sum E of the same FP8
products, and needs to allow for
  S   the outputs' own bf16 rounding: sqrt(sum ulp(y)^2 / 12) over the block
  A   the truncation's random part: u * sqrt(L * Q), Q = sum over the block's products of p^2, u = 2^-13, L = K
  B   its bias where products add coherently: u * L * Y, Y = sum of |y| over the block
so that |O - E| <= z * sqrt(S^2 + alpha^2 A^2 + beta^2 B^2) on clean arithmetic. This records r = O - E, S, A, B
and E for up to ROWS rows of every layer it reads, so alpha, beta and z can be fitted afterwards (and checked on
prompts not used for the fit), and prints the scores at starting values.

Layers: Qwen2.5-1.5B-Instruct (every linear layer of every block) and, on a GPU with the memory, Qwen2.5-7B-Instruct
(every linear layer of blocks 0, 7, 14, 21 and 27), on 64 prompts (questions, code, other languages, numbers) and 4
long documents (about 1,500 tokens each). Each layer is quantized as vLLM quantizes it: the weight to FP8 E4M3 per
tensor, the activations per token (vLLM's online FP8 on CUTLASS) and, separately, per tensor (a checkpoint with
per-tensor activation scales); the products come from vLLM's cutlass_scaled_mm (what vLLM runs) and, for comparison,
torch._scaled_mm without fast accumulation (cuBLASLt's promoted accumulation).

    python experiments/fp8mm_calibration.py --out runs/fp8mm_calibration [--models 1.5b,7b]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

U = 2.0 ** -13
BS = 128
ROWS = 256                       # rows kept per layer and recipe (sampled with a fixed seed)
EMU_ROWS = 16                    # of them, the rows the accumulation emulator recomputes
MODELS = {"1.5b": ("Qwen/Qwen2.5-1.5B-Instruct", None), "7b": ("Qwen/Qwen2.5-7B-Instruct", (0, 7, 14, 21, 27))}
KINDS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
         "mlp.up_proj", "mlp.down_proj")
PROMPTS = [
    "Explain how a hash map handles collisions.", "Write a short poem about the sea at night.",
    "What is 37 times 43? Show the steps.", "Summarize the causes of the French Revolution.",
    "Translate into Spanish: the train leaves at nine.", "List five uses of a binary search tree.",
    "Describe photosynthesis to a ten year old.", "Give three tips for a job interview.", "Why is the sky blue?",
    "Write a Python function that reverses a linked list.", "What are the main differences between TCP and UDP?",
    "Explain the rules of chess briefly.", "How does a refrigerator keep food cold?", "Plan a three day trip to Rome.",
    "What is the derivative of x^3 sin x?", "Describe a sunrise over the mountains.",
    "def quicksort(a):\n    if len(a) <= 1:\n        return a\n    pivot = a[len(a) // 2]\n",
    "SELECT name, COUNT(*) FROM orders JOIN users ON orders.user_id = users.id GROUP BY name HAVING",
    "#include <stdio.h>\nint main(void) {\n    for (int i = 0; i < 10; i++) {\n",
    "Écris une lettre courte pour remercier un collègue.", "Erkläre, wie ein Elektromotor funktioniert.",
    "日本の四季について短く説明してください。", "请用三句话介绍长城。", "Объясни, что такое машинное обучение.",
    "मुझे दिल्ली के बारे में तीन रोचक तथ्य बताओ।", "Explique o que é inflação em termos simples.",
    "3.14159 2.71828 1.41421 1.73205 0.57721 1.61803 2.50290 4.66920", "0x7f 0xff 0x00 0x1a 0x2b 0x3c 0x4d",
    "The quick brown fox jumps over the lazy dog. " * 6, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Prove that the square root of 2 is irrational.", "What causes the seasons on Earth?",
    "Give me a recipe for pancakes.", "How do vaccines train the immune system?",
    "Write a haiku about autumn leaves.", "What is the capital of Australia and why was it chosen?",
    "Explain gradient descent with an analogy.", "List the planets in order from the sun.",
    "Write a limerick about a cat.", "How does public key cryptography work?", "What is a black hole?",
    "Compare Python and Rust for systems programming.", "Explain the Monty Hall problem.",
    "What is the difference between weather and climate?", "Write a short story opening in a lighthouse.",
    "How do I center a div in CSS?", "Explain what a mutex is.", "What is 1234567 * 7654321?",
    "Describe the water cycle.", "How do airplanes stay in the air?", "Write a SQL query to find duplicates.",
    "What is entropy in thermodynamics?", "Explain recursion to a beginner.", "What were the Apollo missions?",
    "How does a transistor work?", "Give five synonyms for happy.", "What is the Pythagorean theorem?",
    "Explain supply and demand.", "How do bees make honey?", "Write a regex that matches an email address.",
    "What is the speed of light in a vacuum?", "Describe the plot of Hamlet in two sentences.",
    "How does GPS determine location?", "What is a prime number? List the first ten.",
]
DOC = ("The harbor town keeps a ledger of every ship that docks, with its cargo, its crew and the weather on arrival. "
       "Each entry is signed by the harbor master, and the ledger has been kept without a gap for two hundred years. ")


def long_docs():
    import random
    rng = random.Random(0)
    words = DOC.split()
    out = []
    for d in range(4):
        rng.shuffle(words)
        out.append(" ".join(words * 40) + f"\n\nQuestion {d}: what does the ledger record?")
    return out


def capture(model_id: str, blocks, device):
    """{layer name: (inputs [tokens, K] bf16, weight [N, K] bf16)}, with the prompts each row came from."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
    n_blocks = model.config.num_hidden_layers
    use = range(n_blocks) if blocks is None else blocks
    want = {f"model.layers.{b}.{k}" for b in use for k in KINDS}
    seen: dict = {}
    hooks = []
    for name, mod in model.named_modules():
        if name in want:
            def hook(m, args, _n=name):
                seen.setdefault(_n, []).append(args[0].detach().reshape(-1, args[0].shape[-1]).cpu())
            hooks.append(mod.register_forward_pre_hook(hook))
    prompt_of = []
    texts = PROMPTS + long_docs()
    with torch.no_grad():
        for pi, p in enumerate(texts):
            ids = tok(p, return_tensors="pt").to(device)
            model(**ids)
            prompt_of += [pi] * int(ids["input_ids"].shape[1])
    for h in hooks:
        h.remove()
    mods = dict(model.named_modules())
    layers = {n: (torch.cat(xs), mods[n].weight.detach().cpu()) for n, xs in seen.items()}
    del model
    torch.cuda.empty_cache()
    return layers, prompt_of


def block_terms(y, xq, xs, wq_t, ws, bs: int = BS):
    """Per row and block: r = O - E, S, A, B, E (float64). y [M, N] bf16 output; xq [M, K] FP8; xs [M, 1] or [1];
    wq_t [K, N] FP8 (column major, as vLLM passes it); ws [1] or [N]."""
    import torch
    M, N = y.shape
    K = xq.shape[1]
    nb = -(-N // bs)
    pad = nb * bs - N
    xd = xq.double()
    wd = wq_t.double() * ws.double().reshape(1, -1)                      # s_w[j] * w_q[k, j], exact enough
    sx = xs.double().reshape(-1, 1) if xs.numel() > 1 else xs.double().reshape(1, 1)
    c = torch.nn.functional.pad(wd, (0, pad)).reshape(K, nb, bs).sum(-1)       # [K, nb] block sums
    v = torch.nn.functional.pad(wd * wd, (0, pad)).reshape(K, nb, bs).sum(-1)  # [K, nb] sums of squares
    E = sx * (xd @ c)
    Q = sx * sx * ((xd * xd) @ v)
    yd = torch.nn.functional.pad(y.double(), (0, pad)).reshape(M, nb, bs)
    O = yd.sum(-1)
    ay = yd.abs()
    ulp = torch.where(ay > 0, torch.pow(2.0, torch.floor(torch.log2(ay.clamp_min(1e-300))) - 7), torch.zeros_like(ay))
    S = torch.sqrt((ulp * ulp).sum(-1) / 12.0)
    A = U * torch.sqrt(K * Q)
    B = U * K * ay.sum(-1)
    return O - E, S, A, B, E


def _raw_exp(x):
    import torch
    ax = x.abs()
    e = torch.floor(torch.log2(ax.clamp_min(1e-300)))
    e = torch.where(ax < 2.0 ** -6, torch.full_like(e, -6.0), e)           # E4M3 subnormals
    return torch.where(ax == 0, torch.full_like(e, -1e9), e)


def emulate_fast_accum(xq, wq_t, g: int = 32, F: int = 13, chunk: int = 2048):
    """The tensor core's FP8 fast accumulation as published (DeepSeek-V3 3.5.2; Khattak & Mikaitis 2026; MMA-Sim):
    for each output, the products of g consecutive k at a time, together with the running accumulator, aligned to the
    largest of them, F fractional bits kept and the rest truncated toward zero, summed exactly, normalized and
    truncated to F fractional bits again, kept as FP32; the whole of K in one chain, k in increasing order. g = 32 for
    Hopper's wgmma, 16 for Ada's mma. xq [M, K], wq_t [K, N]: FP8 tensors. Returns the FP32 accumulators [M, N] as
    float64. In column chunks (the work holds [M, g, chunk] float64 values)."""
    import torch
    a = xq.double()
    M, K = a.shape
    N = wq_t.shape[1]
    out = torch.empty(M, N, dtype=torch.float64, device=a.device)
    ea = _raw_exp(a)
    for n0 in range(0, N, chunk):
        b = wq_t[:, n0:n0 + chunk].double()
        eb = _raw_exp(b)
        c = torch.zeros(M, b.shape[1], dtype=torch.float64, device=a.device)
        for k0 in range(0, K, g):
            p = a[:, k0:k0 + g, None] * b[None, k0:k0 + g, :]
            ep = torch.where(p == 0, torch.full_like(p, -1e9), ea[:, k0:k0 + g, None] + eb[None, k0:k0 + g, :])
            ec = torch.where(c == 0, torch.full_like(c, -1e9), torch.floor(torch.log2(c.abs().clamp_min(1e-300))))
            emax = torch.maximum(ep.max(1).values, ec)
            ulp = torch.where(emax < -1e8, torch.ones_like(emax), torch.pow(2.0, emax - F))
            sm = (torch.trunc(p / ulp[:, None, :]) * ulp[:, None, :]).sum(1) + torch.trunc(c / ulp) * ulp
            es = torch.floor(torch.log2(sm.abs().clamp_min(1e-300)))
            u2 = torch.pow(2.0, es - F)
            sm = torch.where(sm == 0, sm, torch.trunc(sm / u2) * u2)
            c = sm.to(torch.float32).to(torch.float64)
        out[:, n0:n0 + chunk] = c
    return out


def emulate_output(acc, xs, ws):
    """vLLM's CUTLASS epilogue for these scales: bf16(a_scale * (b_scale * acc)), each product rounded in FP32."""
    import torch
    t = (ws.float().reshape(1, -1) * acc.float())
    return (xs.float().reshape(-1, 1) * t).to(torch.bfloat16) if xs.numel() > 1 else \
        (xs.float().reshape(1, 1) * t).to(torch.bfloat16)


def agreement(y, emu) -> dict:
    """How the GPU's outputs compare with the emulation, bit for bit and in bf16 steps."""
    import torch
    yi = y.view(torch.int16).to(torch.int32)
    ei = emu.view(torch.int16).to(torch.int32)
    order = lambda t: torch.where(t < 0, -(t & 0x7FFF), t)  # noqa: E731  (bf16 bits as ordered integers)
    d = (order(yi) - order(ei)).abs()
    return {"outputs": int(d.numel()), "bit_identical": float((d == 0).float().mean()),
            "within_1_step": float((d <= 1).float().mean()), "max_steps": int(d.max())}


def score(r, S, A, B, alpha, beta):
    import torch
    d = torch.sqrt(S * S + (alpha * A) ** 2 + (beta * B) ** 2).clamp_min(1e-300)
    return r.abs() / d


def take_rows(t, rows):
    """t[rows] for an FP8 tensor too (indexing isn't implemented for FP8 dtypes everywhere: PyTorch 2.8 on the CPU),
    through its bytes."""
    import torch
    if t.element_size() == 1 and t.dtype not in (torch.uint8, torch.int8):
        return t.view(torch.uint8)[rows].view(t.dtype)
    return t[rows]


def measure(models, device, out: str) -> dict:
    import numpy as np
    import torch
    from vllm import _custom_ops as ops
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(device), "models": {}}
    gen = torch.Generator().manual_seed(0)
    for tag in models:
        model_id, blocks = MODELS[tag]
        layers, prompt_of = capture(model_id, blocks, device)
        prompt_of = torch.tensor(prompt_of)
        mres = {"model": model_id, "layers": {}}
        for name, (x, w) in layers.items():
            x, w = x.to(device), w.to(device)
            wq, ws = ops.scaled_fp8_quant(w)                                 # weight per tensor
            b = wq.t()
            rows = torch.randperm(x.shape[0], generator=gen)[:ROWS].sort().values
            lres = {"K": int(x.shape[1]), "N": int(w.shape[0]), "tokens": int(x.shape[0]), "rows": int(rows.numel())}
            arrays = {"prompt": prompt_of[rows].numpy().astype(np.int16)}
            for recipe in ("per_token", "per_tensor"):
                xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=recipe == "per_token")
                for kern in ("cutlass", "torch_promoted"):
                    try:
                        if kern == "cutlass":
                            y = ops.cutlass_scaled_mm(xq, b, xs, ws, torch.bfloat16)
                        elif recipe == "per_token":          # row-wise scales on both sides (as cuBLASLt asks)
                            y = torch._scaled_mm(xq, b, scale_a=xs.float(),
                                                 scale_b=ws.float().view(1, 1).expand(1, w.shape[0]).contiguous(),
                                                 out_dtype=torch.bfloat16, use_fast_accum=False)
                        else:
                            y = torch._scaled_mm(xq, b, scale_a=xs.float().reshape(()), scale_b=ws.float().reshape(()),
                                                 out_dtype=torch.bfloat16, use_fast_accum=False)
                    except Exception as e:  # noqa: BLE001
                        lres[f"{recipe}_{kern}"] = {"error": f"{type(e).__name__}: {e}"[:300]}
                        continue
                    xs_rows = xs[rows] if xs.numel() > 1 else xs
                    r, S, A, B, E = (t.cpu() for t in block_terms(y[rows], take_rows(xq, rows), xs_rows, b, ws))
                    key = f"{recipe}_{kern}"
                    # r in float32; the scales it is judged against need only a few digits (float16 keeps 3)
                    arrays[f"{key}_r"] = r.numpy().astype(np.float32)
                    for nm, t in (("S", S), ("A", A), ("B", B), ("E", E)):
                        arrays[f"{key}_{nm}"] = t.numpy().astype(np.float32 if nm == "E" else np.float16)
                    st = {"bf16_units_max": float((r.abs() / S.clamp_min(1e-300)).max()),
                          "bf16_units_p999": float(torch.quantile((r.abs() / S.clamp_min(1e-300)).flatten().float(),
                                                                  0.999))}
                    for a_, b_ in ((0.5, 0.0), (0.7, 0.0), (0.5, 0.01), (0.5, 0.05)):
                        st[f"score_max_alpha{a_}_beta{b_}"] = float(score(r, S, A, B, a_, b_).max())
                    st["bias_corr"] = float(torch.corrcoef(torch.stack([r.flatten(), E.flatten()]))[0, 1])
                    if kern == "cutlass":                # does the published model reproduce the GPU's outputs?
                        er = rows[:EMU_ROWS]
                        g = 32 if torch.cuda.get_device_capability(device)[0] >= 9 else 16
                        acc = emulate_fast_accum(take_rows(xq, er), b, g=g)
                        emu = emulate_output(acc, xs[er] if xs.numel() > 1 else xs, ws)
                        st["emulation"] = {"g": g, **agreement(y[er], emu)}
                    lres[key] = st
            np.savez_compressed(os.path.join(out, f"{tag}__{name}.npz"), **arrays)
            mres["layers"][name] = lres
            print(tag, name, json.dumps({k: (v.get("bf16_units_max"), v.get("score_max_alpha0.5_beta0.0"))
                                         if isinstance(v, dict) else v for k, v in lres.items()}), flush=True)
            del x, w, wq, b
        res["models"][tag] = mres
        del layers
        torch.cuda.empty_cache()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="1.5b,7b")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import torch
    models = [m for m in a.models.split(",") if m]
    free = torch.cuda.mem_get_info()[1] / 2 ** 30
    if free < 40 and "7b" in models:                  # the 7B model needs about 16 GiB plus the captured inputs
        models.remove("7b")
    res = measure(models, torch.device("cuda"), a.out)
    with open(os.path.join(a.out, "SUMMARY.json"), "w") as f:
        json.dump(res, f, indent=1)
    worst = [v.get("bf16_units_max") for m in res["models"].values() for lr in m["layers"].values()
             for v in lr.values() if isinstance(v, dict) and "bf16_units_max" in v]
    print(json.dumps({"layers": sum(len(m["layers"]) for m in res["models"].values()),
                      "worst_bf16_units": max(worst) if worst else None}))
    return 0 if worst and all(not math.isnan(w) for w in worst) else 1


if __name__ == "__main__":
    sys.exit(main())
