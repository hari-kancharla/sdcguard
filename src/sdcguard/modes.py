"""Matmuls in six math modes, checked: BF16, FP16, TF32, FP32, FP64 and INT8 (FP8 and FP4 not yet).

A defective GPU can compute some modes correctly and not others. In SDCHunter (OSDI '26) all four
GEMM-defective GPUs computed BF16 and FP16 correctly and failed in FP32, FP64, TF32 or INT8 (which ones
depended on the GPU), and some were wrong only for inputs in certain value ranges. So each mode gets its own
check, over several input distributions and scales.

Each check runs one GEMM in one mode on the GPU and compares every row-block sum of the result (plain, and
with random signs, as in checksum.py) with a float64 reference computed on the CPU from the same inputs.
The inputs are chosen so that every product is exact in the accumulator (bf16 and fp16 values,
TF32-representable fp32 values, small integers) and the results are kept in fp32 (fp64, int32), so the only
error left is the accumulation's own rounding:

    INT8            int32 accumulation: has to match exactly
    FP64            sequential IEEE FMAs: u * (|a_row| |B_block| + sqrt(k) |C_block|), u = 2^-53
    FP32            IEEE FMAs:            the same with u = 2^-24
    BF16 FP16 TF32  tensor cores add N products per step and truncate the sum to fp32 (Khattak and
                    Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004): the truncation
                    always rounds toward zero, so its error grows with the number of steps k / N, and the
                    model adds u * (k / N) |C_block|

A result is wrong when it's more than 6 of those units away from the reference. Inputs are zero-mean: with
all-positive inputs the tensor cores' truncation errors would line up and need a looser model.

Compared with checking a bf16 result (checksum.py), the fp32 result keeps 16 more bits, so the check sees
much smaller errors; each mode's card entry says how small (the error, relative to a typical output, that
would set it off). AEGIS (OSDI '26) found its production incidents with checksums over fp32 accumulators
and not with ones over bf16 outputs.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Optional

import torch

ALARM = 6.0
BLOCK = 128
C1, C2, C3 = 1.0, 1.0, 1.0
UNIT = {"fp32": 2.0 ** -24, "fp64": 2.0 ** -53}
MODES = ("bf16", "fp16", "tf32", "fp32", "fp64", "int8")
SHAPE = {"cuda": (2048, 1024, 2048), "cpu": (256, 256, 256)}     # (m, k, n)
# what each check runs: inputs, how the GPU computes, the result's type, what it's compared with
PROFILE = {
    "bf16": ("bf16", "tensor cores, fp32 accumulation", "fp32", "float64 on the CPU"),
    "fp16": ("fp16", "tensor cores, fp32 accumulation", "fp32", "float64 on the CPU"),
    "tf32": ("fp32 (TF32-exact values)", "tensor cores (TF32), fp32 accumulation", "fp32", "float64 on the CPU"),
    "fp32": ("fp32", "fp32 FMA (TF32 off)", "fp32", "float64 on the CPU"),
    "fp64": ("fp64", "fp64 FMA", "fp64", "float64 on the CPU"),
    "int8": ("int8", "integer tensor cores, int32 accumulation", "int32", "exact integers on the CPU"),
}
# (distribution, scale): zero-mean, spread over magnitudes
VARIANTS = (("normal", 1.0), ("uniform", 2.0 ** -8), ("sparse", 2.0 ** 6), ("normal", 2.0 ** 8),
            ("laplace", 2.0 ** -4), ("uniform", 1.0))


def n_fma(mode: str, cc: Optional[tuple]) -> Optional[int]:
    """Products the tensor cores add per truncation step (Khattak and Mikaitis, Table 3). Older or unknown
    GPUs get the smallest value seen on any generation, which assumes the most steps."""
    if mode not in ("bf16", "fp16", "tf32") or cc is None:
        return None
    major = cc[0]
    if major in (9, 10, 12):                     # Hopper, Blackwell
        return 8 if mode == "tf32" else 16
    if major == 8:                               # Ampere, Ada
        return 4 if mode == "tf32" else 8
    return 4


def _tf32_exact(x: torch.Tensor) -> torch.Tensor:
    """fp32 values with the low 13 mantissa bits cleared: exactly representable in TF32."""
    return (x.view(torch.int32) & -8192).view(torch.float32)


def _sample(shape, dist: str, scale: float, gen: torch.Generator) -> torch.Tensor:
    if dist == "normal":
        x = torch.randn(shape, generator=gen)
    elif dist == "uniform":
        x = torch.rand(shape, generator=gen) * 2 - 1
    elif dist == "laplace":
        u = torch.rand(shape, generator=gen).clamp(1e-7, 1 - 1e-7) - 0.5
        x = (-torch.sign(u) * torch.log1p(-2 * u.abs())).clamp(-20, 20)
    elif dist == "sparse":
        x = torch.randn(shape, generator=gen) * (torch.rand(shape, generator=gen) < 0.1)
    else:
        raise ValueError(dist)
    return x * scale


def _inputs(mode: str, shape: tuple, dist: str, scale: float, gen: torch.Generator):
    """(A, B) on the CPU in the mode's input type, and their exact float64 copies."""
    m, k, n = shape
    if mode == "int8":
        hi = {"normal": 127, "uniform": 3, "sparse": 127, "laplace": 15}[dist]
        a = torch.randint(-hi, hi + 1, (m, k), generator=gen, dtype=torch.int32)
        b = torch.randint(-hi, hi + 1, (k, n), generator=gen, dtype=torch.int32)
        if dist == "sparse":
            a = a * (torch.rand(m, k, generator=gen) < 0.1)
        a, b = a.to(torch.int8), b.to(torch.int8)
        return a, b, a.double(), b.double()
    a, b = _sample((m, k), dist, scale, gen), _sample((k, n), dist, scale, gen)
    if mode == "bf16":
        a, b = a.bfloat16(), b.bfloat16()
    elif mode == "fp16":
        a, b = a.clamp(-6e4, 6e4).half(), b.clamp(-6e4, 6e4).half()
    elif mode == "tf32":
        a, b = _tf32_exact(a), _tf32_exact(b)
    elif mode == "fp64":
        a, b = a.double(), b.double()
    return a, b, a.double(), b.double()


def gemm(mode: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One GEMM in `mode`, result in fp32 (fp64 for FP64, int32 for INT8)."""
    if a.device.type != "cuda":                   # CPU (tests and demos): the same math, no tensor cores
        if mode == "int8":
            return (a.long() @ b.long()).int()
        return a.double() @ b.double() if mode == "fp64" else a.float() @ b.float()
    if mode in ("bf16", "fp16"):
        return torch.ops.aten.mm.dtype(a, b, torch.float32)
    if mode == "int8":
        return torch._int_mm(a, b)
    if mode == "fp64":
        return a @ b
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    try:
        return a @ b
    finally:
        torch.set_float32_matmul_precision(prev)


def available(mode: str, dev: torch.device) -> Optional[str]:
    """None if the mode can run on dev, else why not (shown on the card as not checked)."""
    if dev.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(dev)
    if mode in ("tf32", "bf16") and major < 8:
        return "needs compute capability 8.0 or newer"
    try:
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "int8": torch.int8,
              "fp64": torch.float64}.get(mode, torch.float32)
        x = torch.ones(32, 32, device=dev).to(dt)
        gemm(mode, x, x)
        torch.cuda.synchronize(dev)
    except (RuntimeError, NotImplementedError, AttributeError, TypeError) as e:
        return f"not supported by this GPU or PyTorch ({type(e).__name__})"
    return None


def _signs(width: int) -> torch.Tensor:
    from .checksum import _signs as cs_signs
    return cs_signs(width, "cpu").double()


def judge(mode: str, a64: torch.Tensor, b64: torch.Tensor, c: torch.Tensor, nfma: Optional[int] = None) -> dict:
    """Compare a GEMM result c (fp32, fp64 or int32, on any device) with the float64 reference from the exact
    inputs a64 @ b64 (CPU). Returns per-block scores and the summary check_once reports."""
    m, n = c.shape
    k = a64.shape[1]
    nblk = n // BLOCK
    s = _signs(BLOCK).to(c.device)
    if mode == "int8":
        cb = c.long().view(m, nblk, BLOCK)
        actual = torch.stack([cb.sum(-1), (cb * s.long()).sum(-1)], dim=-1).cpu()
    else:
        cb = c.double().view(m, nblk, BLOCK)
        actual = torch.stack([cb.sum(-1), (cb * s).sum(-1)], dim=-1).cpu()          # (m, nblk, 2)
        c_norm = cb.norm(dim=-1).cpu()                                               # (m, nblk)
    del cb
    bb = b64.view(k, nblk, BLOCK)
    expected = torch.stack([a64 @ bb.sum(-1), a64 @ (bb * _signs(BLOCK)).sum(-1)], dim=-1)
    if mode == "int8":
        bad = (actual != expected.round().long()).any(dim=-1)
        return {"wrong_blocks": int(bad.sum()), "blocks": m * nblk, "worst_z": None, "smallest_caught": None,
                "z": torch.where(bad, float("inf"), 0.0)}
    diff = (expected - actual.double()).abs()
    u = UNIT["fp64" if mode == "fp64" else "fp32"]
    a_norm = a64.norm(dim=1)[:, None]                                                # (m, 1)
    b_frob = bb.pow(2).sum(dim=(0, 2)).sqrt()[None, :]                               # (1, nblk)
    steps = k / nfma if nfma else 0.0
    noise = u * (C1 * a_norm * b_frob + (C2 * math.sqrt(k) + C3 * steps) * c_norm)
    if mode == "fp64":
        noise = noise * 2                     # the reference's own float64 rounding is of the same size
    noise = noise + 1e-300
    z = (diff / noise.unsqueeze(-1)).amax(dim=-1)
    z = torch.where(torch.isfinite(actual).all(dim=-1), z, torch.full_like(z, float("inf")))
    # the smallest error in one output that this check would catch, relative to a typical output there
    typical = c_norm / math.sqrt(BLOCK)
    caught = (ALARM * noise / typical.clamp_min(1e-300))[typical > 0]
    return {"wrong_blocks": int((z > ALARM).sum()), "blocks": m * nblk, "worst_z": float(z.max()),
            "smallest_caught": float(caught.median()) if caught.numel() else None, "z": z}


def check_once(mode: str, dev: torch.device, shape: tuple, dist: str, scale: float, gen: torch.Generator,
               fault: Optional[float] = None, run: Optional[Callable] = None, nfma: Optional[int] = None) -> dict:
    """One GEMM in `mode` on dev, checked against the CPU reference.

    fault (tests, experiments): add an error to one output, this many times the typical (root-mean-square)
    output of its block; for INT8 any fault adds 1. run: compute the GEMM some other way (tests emulate
    tensor-core rounding with it). nfma: tensor-core products per step (default: from the GPU)."""
    m, k, n = shape
    if n % BLOCK:
        raise ValueError(f"n must be a multiple of {BLOCK}")
    a, b, a64, b64 = _inputs(mode, shape, dist, scale, gen)
    c = (run or gemm)(mode, a.to(dev), b.to(dev))
    if fault is not None:
        i, j = m // 3, n // 2
        if mode == "int8":
            c[i, j] += 1
        else:
            blk = c[i, (j // BLOCK) * BLOCK:(j // BLOCK + 1) * BLOCK].double()
            c[i, j] += fault * float(blk.norm()) / math.sqrt(BLOCK)
    if nfma is None and dev.type == "cuda":
        nfma = n_fma(mode, torch.cuda.get_device_capability(dev))
    r = judge(mode, a64, b64, c, nfma)
    r.pop("z")
    return r


def screen(dev: torch.device, seconds: float, shape: Optional[tuple] = None, seed: int = 0, say=None) -> dict:
    """Every mode the GPU and PyTorch support, cycling through the input variants until `seconds` is used
    (each mode at least one full cycle, at most four). Modes that can't run are listed with the reason."""
    say = say or (lambda s: None)
    shape = tuple(shape or SHAPE["cuda" if dev.type == "cuda" else "cpu"])
    gen = torch.Generator().manual_seed(1000 + seed)
    out = {"shape": list(shape), "modes": {}}
    runnable = []
    for m in MODES:
        why = available(m, dev)
        if why:
            out["modes"][m] = {"skipped": why}
        else:
            out["modes"][m] = {"matmuls": 0, "wrong": 0, "blocks_checked": 0, "worst_z": 0.0,
                               "smallest_error_caught": None}
            runnable.append(m)
    t_end = time.time() + seconds
    rnd = 0
    while runnable and rnd < 4 * len(VARIANTS) and (rnd < len(VARIANTS) or time.time() < t_end):
        dist, scale = VARIANTS[rnd % len(VARIANTS)]
        for m in list(runnable):
            rec = out["modes"][m]
            try:
                r = check_once(m, dev, shape, dist, scale, gen)
            except Exception as e:  # noqa: BLE001  (an unsupported shape mustn't stop the whole check)
                say(f"math mode {m} stopped: {type(e).__name__}: {e}")
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                runnable.remove(m)
                continue
            rec["matmuls"] += 1
            rec["blocks_checked"] += r["blocks"]
            rec["wrong"] += bool(r["wrong_blocks"])
            if r["worst_z"] is not None and math.isfinite(r["worst_z"]):
                rec["worst_z"] = round(max(rec["worst_z"], r["worst_z"]), 3)
            if r["smallest_caught"] is not None:     # the worst variant's value: true for all of them
                prev = rec["smallest_error_caught"]
                rec["smallest_error_caught"] = max(prev or 0.0, r["smallest_caught"])
        rnd += 1
    for m in runnable:
        rec = out["modes"][m]
        if m == "int8":
            rec["worst_z"] = None
            rec["smallest_error_caught"] = "exact"
        elif rec["smallest_error_caught"] is not None:
            rec["smallest_error_caught"] = float(f"{rec['smallest_error_caught']:.2g}")
    out["wrong"] = sum(r.get("wrong", 0) for r in out["modes"].values())
    out["matmuls"] = sum(r.get("matmuls", 0) for r in out["modes"].values())
    out["ran"] = [m for m in MODES if out["modes"][m].get("matmuls") and "error" not in out["modes"][m]]
    out["not_checked"] = {m: r.get("skipped") or r.get("error") for m, r in out["modes"].items() if m not in out["ran"]}
    return out
