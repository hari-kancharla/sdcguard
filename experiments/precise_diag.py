"""Why does precise mode raise false alarms on some GPUs? Every fp32-accumulator check of a few clean training
steps, scored and taken apart.

Precise mode (SDCGuard(precise=True)) checks each linear layer's fp32 matmul result before it's rounded. On
Oct 19, 2026 it raised 70 to 78 false alarms in 2,970 clean checks on an A100, H100 and B200 (worst score 17.9
to 20.3), 3 on an L4 and none on a T4, while the same noise model stays far from the alarm on synthetic
matmuls and on a CPU emulation of the tensor cores. This runs the same model and data and, for every check,
records the layer, the pass, the shapes and the score. For the checks that score high (and a sample of the
rest) it computes the exact answer in float64 on the GPU and reports:
  - which checksum set the score off (the plain sum or the signed one)
  - how large the per-output errors are, in units of the model's own rounding unit times the output
  - how coherent the errors in the worst block are: |sum of errors| / sqrt(sum of squared errors), about 1 when
    they're independent and up to sqrt(128) = 11.3 when they all point the same way
  - how often an error points toward zero (what truncation does)
  - whether the same matmul gives the same bits when run again, and which cuBLAS kernel ran it
  - the same score for the matmul done with plain fp32 FMAs (TF32 off), as a control

    python experiments/precise_diag.py --device cuda --out results/precise_diag
    python experiments/precise_diag.py                    # CPU, tiny model: a smoke test
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import SDCGuard, modes  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402
from sdcguard import guard as gmod  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

CONFIG = {"cuda": dict(vocab=32000, layers=8, width=1024, heads=16, seq=512, batch=8),
          "cpu": dict(vocab=512, layers=2, width=128, heads=4, seq=64, batch=4)}
BLOCK = 128


def kernel_names(fn) -> list:
    """CUDA kernels one call of fn launches (empty on the CPU)."""
    if not torch.cuda.is_available():
        return []
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        return sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})[:6]
    except Exception as e:  # noqa: BLE001
        return [f"(profiler failed: {type(e).__name__})"]


def model_terms(A, B, C, nfma):
    """The precise check's noise per (row, block), split into its parts (checksum._acc_zscores)."""
    u = cs.ACC_UNIT
    m, k = A.shape
    n = C.shape[1]
    nblk = n // BLOCK
    Cb = C[:, :nblk * BLOCK].double().view(m, nblk, BLOCK)
    Bb = B[:, :nblk * BLOCK].double().view(k, nblk, BLOCK)
    cnorm = Cb.norm(dim=-1)                                              # (m, nblk)
    a_norm = A.double().norm(dim=1)[:, None]                             # (m, 1)
    b_frob = Bb.pow(2).sum(dim=(0, 2)).sqrt()[None, :]                   # (1, nblk)
    steps = k / nfma if nfma else 0.0
    per_out = cs.ACC_C2 * math.sqrt(k) + cs.ACC_C3 * steps + math.sqrt(BLOCK)
    return {"output_terms": cnorm * per_out * u, "product_term": a_norm * b_frob * cs.ACC_C1 * math.sqrt(k) * u,
            "cnorm": cnorm, "per_out": per_out, "nblk": nblk}


def dissect(A, B, C, bias, nfma, signs) -> dict:
    """Exact float64 answer on the same device, and what the errors look like."""
    u = cs.ACC_UNIT
    ref = A.double() @ B.double()
    if bias is not None:
        ref = ref + bias.double()
    t = model_terms(A, B, C, nfma)
    nblk = t["nblk"]
    m = A.shape[0]
    e = (C.double() - ref)[:, :nblk * BLOCK].view(m, nblk, BLOCK)
    c = C.double()[:, :nblk * BLOCK].view(m, nblk, BLOCK)
    s = signs.double().view(1, 1, BLOCK)
    noise = t["output_terms"] + t["product_term"]
    z_plain = e.sum(-1).abs() / noise
    z_signed = (e * s).sum(-1).abs() / noise
    zp, zs = float(z_plain.max()), float(z_signed.max())
    worst = z_plain if zp >= zs else z_signed
    idx = int(worst.argmax())
    r, b = divmod(idx, nblk)
    eb, cb = e[r, b], c[r, b]
    nz = eb != 0
    rel = (eb.abs() / (u * cb.abs().clamp_min(1e-30)))
    arow = A[r].double()
    bblk = B[:, b * BLOCK:(b + 1) * BLOCK].double()
    prods = arow[:, None] * bblk                                         # (k, 128) products of the worst block
    return {
        "worst_block_inputs": {"a_row_positive": round(float((arow > 0).float().mean()), 3),
                               "b_block_positive": round(float((bblk > 0).float().mean()), 3),
                               "products_positive": round(float((prods > 0).float().mean()), 3),
                               "outputs_positive": round(float((cb > 0).float().mean()), 3),
                               "sum_of_abs_products_over_abs_output": round(float(prods.abs().sum(0).median()
                                                                                / cb.abs().median().clamp_min(1e-30)), 1)},
        "z_plain": round(zp, 2), "z_signed": round(zs, 2),
        "worst_block": [r, b],
        "worst_block_coherence": round(float(eb.sum().abs() / eb.norm().clamp_min(1e-300)), 2),
        "worst_block_toward_zero": round(float(((eb * cb) < 0)[nz].float().mean()) if bool(nz.any()) else 0.0, 3),
        "worst_block_err_in_u_times_output": {"median": round(float(rel.median()), 1), "max": round(float(rel.max()), 1)},
        "worst_block_noise_parts": {"output_terms": float(t["output_terms"][r, b]),
                                    "product_term": float(t["product_term"][r, b]),
                                    "plain_sum_error": float(eb.sum()), "block_norm": float(t["cnorm"][r, b])},
        "all_blocks_coherence_median": round(float((e.sum(-1).abs() / e.norm(dim=-1).clamp_min(1e-300)).median()), 2),
        "max_err_in_u_times_output_all": round(float((e.abs() / (u * c.abs().clamp_min(1e-30))).flatten()
                                                     .kthvalue(max(1, int(0.999 * e.numel()))).values), 1),
    }


def run(dev: torch.device, steps: int, seed: int = 0, say=print) -> dict:
    cfg = CONFIG["cuda" if dev.type == "cuda" else "cpu"]
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    torch.manual_seed(seed)
    model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(seed + 1)
    rng = random.Random(seed)
    g = SDCGuard(every=1, precise=True)
    records, detailed = [], []
    state = {"step": 0}

    def grab(self, name, pass_, A, B, C32, bias):
        nfma = self._tensor_core_steps(C32, A.dtype)
        st = cs.matmul_zscores_acc(A, B, C32, bias, nfma=nfma, epoch=self._epoch)
        z = float(cs.resolve([st])[0].max_z) if st is not None else None
        kind = name.split(".")[-1]
        rec = {"step": state["step"], "layer": name, "kind": kind, "pass": pass_, "m_k_n": [A.shape[0], A.shape[1],
               C32.shape[1]], "z": None if z is None else round(z, 3), "nfma": nfma}
        records.append(rec)
        if z is not None and (z > 3.0 or rng.random() < 0.02) and len(detailed) < 60:
            d = dict(rec)
            d.update(dissect(A, B, C32, bias, nfma, cs._signs(BLOCK, C32.device, self._epoch)))
            again = gmod._acc_mm(A, B)
            if bias is not None:
                again = again.add_(bias.float())
            d["same_bits_when_rerun"] = bool(torch.equal(again, C32))
            d["kernels"] = kernel_names(lambda: gmod._acc_mm(A, B))
            with torch.autocast(dev.type, enabled=False):
                mm = torch.backends.cuda.matmul
                saved = mm.allow_tf32
                mm.allow_tf32 = False
                try:
                    alt = A.float() @ B.float()
                finally:
                    mm.allow_tf32 = saved
            if bias is not None:
                alt = alt + bias.float()
            st2 = cs.matmul_zscores_acc(A, B, alt, bias, nfma=None, epoch=self._epoch)
            d["z_with_plain_fp32_fma"] = round(float(cs.resolve([st2])[0].max_z), 3)
            detailed.append(d)
        return orig(self, name, pass_, A, B, C32, bias)

    orig = SDCGuard._accumulator
    SDCGuard._accumulator = grab
    try:
        g.protect(model)
        for n in range(steps):
            state["step"] = n
            idx = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), generator=gen).to(dev)
            with g.on_step(n):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            say(f"step {n}: {len(records)} checks so far, {g.stats['alerts']} alarms")
        g.unprotect()
    finally:
        SDCGuard._accumulator = orig

    by = collections.defaultdict(lambda: {"checks": 0, "over_6": 0, "worst": 0.0, "m_k_n": None, "nfma": None})
    for r in records:
        key = f"{r['kind']} {r['pass']}"
        b = by[key]
        b["checks"] += 1
        b["m_k_n"], b["nfma"] = r["m_k_n"], r["nfma"]
        if r["z"] is not None:
            b["over_6"] += r["z"] > 6
            b["worst"] = max(b["worst"], r["z"])
    out = {"config": cfg, "dtype": str(low).replace("torch.", ""), "steps": steps, "checks": len(records),
           "alarms": g.stats["alerts"], "by_layer_and_pass": dict(sorted(by.items(), key=lambda kv: -kv[1]["worst"])),
           "detailed": sorted(detailed, key=lambda d: -(d["z"] or 0))}
    if dev.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(dev)
        out["cc"] = list(torch.cuda.get_device_capability(dev))
        out["torch"] = torch.__version__
        try:
            s = modes.screen(dev, 5.0)
            out["math_modes"] = {m: {k: v for k, v in r.items() if k in ("matmuls", "wrong", "worst_z", "skipped", "error")}
                                 for m, r in s["modes"].items()}
        except Exception as e:  # noqa: BLE001
            out["math_modes"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = run(torch.device(a.device), a.steps)
    res["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in res.items() if k != "detailed"}, indent=1, default=str)[:4000])
    for d in res["detailed"][:5]:
        print(json.dumps(d, default=str))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = (res.get("gpu") or "cpu").replace(" ", "_")
        with open(os.path.join(a.out, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
