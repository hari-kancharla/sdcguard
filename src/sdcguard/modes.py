"""Matmuls in seven math modes, checked: BF16, FP16, TF32, FP32, FP64, INT8 and FP8 (E4M3; FP4 not yet).

A defective GPU can compute some modes correctly and not others. In SDCHunter (OSDI '26) all four
GEMM-defective GPUs computed BF16 and FP16 correctly and failed in FP32, FP64, TF32 or INT8 (which ones
depended on the GPU), and some were wrong only for inputs in certain value ranges. So each mode gets its own
check, over several input distributions and scales.

Each check runs one GEMM in one mode on the GPU and compares every row-block sum of the result (plain, and
with random signs, as in checksum.py) with a float64 reference computed on the CPU from the same inputs.
The inputs are chosen so that every product is exact in the accumulator (bf16, fp16 and fp8 values,
TF32-representable fp32 values, small integers) and the results are kept in fp32 (fp64, int32), so the only
error left is the accumulation's own rounding:

    INT8            int32 accumulation: has to match exactly
    FP64            sequential IEEE FMAs: u * (|a_row| |B_block| + sqrt(k) |C_block|), u = 2^-53
    FP32            IEEE FMAs:            the same with u = 2^-24
    BF16 FP16 TF32  tensor cores add N products per step and truncate the sum to fp32 (Khattak and
                    Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004): the truncation
                    always rounds toward zero, so its error grows with the number of steps k / N, and the
                    model adds u * (k / N) |C_block|
    FP8             the same model, but on Ada and Hopper the tensor cores keep only 13 bits below the
                    largest term of each step (Blackwell keeps 25), with N = 16 (Ada) or 32 (Hopper,
                    Blackwell), from the same paper. u = 2^-14 there (2^-24 on Blackwell): on an emulation of
                    that model, clean results then stay as far from the alarm as the other modes' do (worst
                    score under 0.5). Inputs avoid fp8 subnormals, and k is 256 instead of 1024 to keep the
                    coarser rounding small. The GPU calibration run is this model's first real test

A result is wrong when it's more than 6 of those units away from the reference. Inputs are zero-mean: with
all-positive inputs the tensor cores' truncation errors would line up and need a looser model.

Compared with checking a bf16 result (checksum.py), the fp32 result keeps 16 more bits, so the check sees
much smaller errors; each mode's card entry says how small (the error, relative to a typical output, that
would set it off). AEGIS (OSDI '26) found its production incidents with checksums over fp32 accumulators
and not with ones over bf16 outputs.
"""
from __future__ import annotations

import math
import re
import time
from typing import Callable, Optional

import torch

ALARM = 6.0
BLOCK = 128
C1, C2, C3 = 1.0, 1.0, 1.0
UNIT = {"fp32": 2.0 ** -24, "fp64": 2.0 ** -53, "fp8": 2.0 ** -14}
MODES = ("bf16", "fp16", "tf32", "fp32", "fp64", "int8", "fp8")
FP8_K = 256                 # FP8 checks use this inner size at most (see above)
FP8_MAX, FP8_MIN_NORMAL = 448.0, 2.0 ** -6
SHAPE = {"cuda": (2048, 1024, 2048), "cpu": (256, 256, 256)}     # (m, k, n)
# what each check runs: inputs, how the GPU computes, the result's type, what it's compared with
PROFILE = {
    "bf16": ("bf16", "tensor cores, fp32 accumulation", "fp32", "float64 on the CPU"),
    "fp16": ("fp16", "tensor cores, fp32 accumulation", "fp32", "float64 on the CPU"),
    "tf32": ("fp32 (TF32-exact values)", "tensor cores (TF32), fp32 accumulation", "fp32", "float64 on the CPU"),
    "fp32": ("fp32", "fp32 FMA (TF32 off)", "fp32", "float64 on the CPU"),
    "fp64": ("fp64", "fp64 FMA", "fp64", "float64 on the CPU"),
    "int8": ("int8", "integer tensor cores, int32 accumulation", "int32", "exact integers on the CPU"),
    "fp8": ("fp8 E4M3 (no subnormals)", "tensor cores (FP8), fp32 result", "fp32", "float64 on the CPU"),
}
# (distribution, scale): zero-mean, spread over magnitudes
VARIANTS = (("normal", 1.0), ("uniform", 2.0 ** -8), ("sparse", 2.0 ** 6), ("normal", 2.0 ** 8),
            ("laplace", 2.0 ** -4), ("uniform", 1.0))


def n_fma(mode: str, cc: Optional[tuple]) -> Optional[int]:
    """Products the tensor cores add per truncation step (Khattak and Mikaitis, Table 3). Older or unknown
    GPUs get the smallest value seen on any generation, which assumes the most steps."""
    if mode not in ("bf16", "fp16", "tf32", "fp8") or cc is None:
        return None
    major = cc[0]
    if mode == "fp8":
        return 32 if major in (9, 10, 12) else 16
    if major in (9, 10, 12):                     # Hopper, Blackwell
        return 8 if mode == "tf32" else 16
    if major == 8:                               # Ampere, Ada
        return 4 if mode == "tf32" else 8
    return 4


def acc_unit(mode: str, cc: Optional[tuple]) -> float:
    """The accumulation's rounding unit u in the noise model. FP8 on a GPU: 2^-24 on Blackwell, which keeps 25
    bits below each step's largest term like the other modes; 2^-14 on Ada, Hopper and anything unknown, which
    keep 13. Without a GPU (cc None) every mode is computed with fp32 FMAs."""
    if mode == "fp64":
        return UNIT["fp64"]
    if mode == "fp8" and cc is not None and cc[0] not in (10, 12):
        return UNIT["fp8"]
    return UNIT["fp32"]


def _fp8_exact(x: torch.Tensor) -> torch.Tensor:
    """fp8 E4M3 values with no subnormals (whether a GPU keeps fp8 subnormals isn't something to bet a verdict
    on): magnitudes below the smallest normal become 0, the rest are clamped to the largest finite value."""
    x = x.clamp(-FP8_MAX, FP8_MAX)
    return torch.where(x.abs() < FP8_MIN_NORMAL, torch.zeros_like(x), x).to(torch.float8_e4m3fn)


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
    if mode == "fp8":            # fp8's range is narrow (2^-6 to 448): real FP8 matmuls scale into it
        s = min(max(scale, 2.0 ** -4), 2.0 ** 4)
        a, b = _fp8_exact(_sample((m, k), dist, s, gen)), _fp8_exact(_sample((k, n), dist, s, gen))
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
    if mode == "fp8":            # per-tensor scales of 1: the GPU multiplies the fp8 values as they are
        one = torch.ones((), device=a.device, dtype=torch.float32)
        return torch._scaled_mm(a, b.t().contiguous().t(), scale_a=one, scale_b=one, out_dtype=torch.float32,
                                use_fast_accum=False)
    if mode == "fp64":
        return a @ b
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high" if mode == "tf32" else "highest")
    try:
        return a @ b
    finally:
        torch.set_float32_matmul_precision(prev)


# Error messages that mean "this GPU or this PyTorch build doesn't offer that", as opposed to a failure.
_UNSUPPORTED = re.compile(r"not supported|NOT_SUPPORTED|not implemented for|only supported|is not available|"
                          r"isn't supported|requires .*(?:compute capability|sm_?\d+|SM ?\d+)", re.IGNORECASE)


def expected(mode: str, cc: tuple) -> bool:
    """Does every GPU of compute capability cc run this mode (with the PyTorch sdcguard supports)? A mode the GPU
    should run that fails its probe is an error, never "unsupported"."""
    need = {"bf16": (8, 0), "tf32": (8, 0), "fp8": (8, 9), "int8": (7, 5)}.get(mode, (0, 0))
    return tuple(cc) >= need


class ProbeError(RuntimeError):
    """A mode's capability probe failed in a way that isn't a missing feature."""


def available(mode: str, dev: torch.device) -> Optional[str]:
    """None if the mode can run on dev, else why not (shown on the card as unsupported). Raises ProbeError when
    the probe fails for another reason, or when a mode this GPU should support fails at all: the caller counts
    that as an error (an incomplete check), not as a missing feature."""
    if dev.type != "cuda":
        return None
    major, minor = torch.cuda.get_device_capability(dev)
    if mode in ("tf32", "bf16") and major < 8:
        return "needs compute capability 8.0 or newer"
    if mode == "fp8" and (major, minor) < (8, 9):
        return "needs compute capability 8.9 or newer"
    try:
        dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "int8": torch.int8, "fp8": torch.float8_e4m3fn,
              "fp64": torch.float64}.get(mode, torch.float32)
        x = torch.ones(32, 32, device=dev).to(dt)
        gemm(mode, x, x)
        torch.cuda.synchronize(dev)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:160]}"
        if not expected(mode, (major, minor)) and isinstance(e, (RuntimeError, NotImplementedError, AttributeError,
                                                                 TypeError)) and _UNSUPPORTED.search(str(e)):
            return f"not supported by this GPU or PyTorch ({msg[:100]})"
        raise ProbeError(f"capability probe failed: {msg}") from e
    return None


PROJECTIONS = 3        # signed checksums per block, on top of the plain sum


def draw_signs(gen: torch.Generator, n: int = PROJECTIONS) -> torch.Tensor:
    """n fresh random +-1 patterns (float64, CPU) for one matmul's signed checksums. Fresh for every matmul:
    an error that one pattern can't see (two outputs of a block swapped where the pattern has the same sign,
    say) usually shows in another: each pattern misses a given swapped pair with probability 1/2, all three
    together with 1/8, drawn anew for every matmul, so a persistent fault shows on most matmuls."""
    return (torch.randint(0, 2, (n, BLOCK), generator=gen) * 2 - 1).double()


def judge(mode: str, a64: torch.Tensor, b64: torch.Tensor, c: torch.Tensor, nfma: Optional[int] = None,
          signs: Optional[torch.Tensor] = None, unit: Optional[float] = None) -> dict:
    """Compare a GEMM result c (fp32, fp64 or int32, on any device) with the float64 reference from the exact
    inputs a64 @ b64 (CPU): the plain sum of every 128-column block and one signed sum per row of `signs`
    (default: checksum.py's per-process pattern). Returns per-block scores and the summary check_once reports.

    typical_caught: the size of an error in one output that this check typically catches, relative to a typical
    output of its block (the median over blocks of the alarm line; some blocks need more, cancelling errors
    can need more, and an error the checksums can't see at all needs any size). unit: the accumulation's
    rounding unit (default: fp32's, or fp64's for FP64; see acc_unit)."""
    m, n = c.shape
    k = a64.shape[1]
    nblk = n // BLOCK
    if signs is None:
        from .checksum import _signs as cs_signs
        signs = cs_signs(BLOCK, "cpu").double().view(1, BLOCK)
    w = torch.cat([torch.ones(1, BLOCK, dtype=torch.float64), signs.double()], dim=0)      # (P+1, BLOCK)
    wd = w.to(c.device)
    if mode == "int8":
        # integer sums in float64: exact here (every sum stays far below 2^53), and float64 matmuls run on every
        # device, while CUDA has no int64 one ("baddbmm_cuda not implemented for 'Long'")
        cb = c.double().view(m, nblk, BLOCK)
        actual = torch.einsum("mbw,pw->mbp", cb, wd).cpu()
    else:
        cb = c.double().view(m, nblk, BLOCK)
        actual = torch.einsum("mbw,pw->mbp", cb, wd).cpu()                       # (m, nblk, P+1)
        # block norms scaled by the block's largest value, so one huge wrong output can't overflow them (an
        # FP64 exponent flip gives values near 1e155, whose square overflows float64)
        mx = cb.abs().amax(dim=-1)
        c_norm = (mx * (cb / mx.clamp_min(1e-300).unsqueeze(-1)).norm(dim=-1)).cpu()   # (m, nblk)
        # and an output larger than |a_row| |b_col| can't come from these inputs at all (Cauchy-Schwarz; the
        # factor 2 leaves room for rounding)
        a_rows = a64.norm(dim=1).to(c.device)
        b_cols = b64.norm(dim=0).to(c.device).view(nblk, BLOCK)
        impossible = (cb.abs() > 2 * a_rows.view(m, 1, 1) * b_cols.unsqueeze(0) + 1e-300).any(dim=-1).cpu()
    del cb
    bb = b64.view(k, nblk, BLOCK)
    expected = a64 @ torch.einsum("kbw,pw->kbp", bb, w).reshape(k, nblk * w.shape[0])
    expected = expected.view(m, nblk, w.shape[0])
    if mode == "int8":
        bad = (actual != expected).any(dim=-1)
        return {"wrong_blocks": int(bad.sum()), "blocks": m * nblk, "worst_z": None, "typical_caught": None,
                "z": torch.where(bad, float("inf"), 0.0)}
    diff = (expected - actual.double()).abs()
    u = unit if unit is not None else UNIT["fp64" if mode == "fp64" else "fp32"]
    a_norm = a64.norm(dim=1)[:, None]                                                # (m, 1)
    b_frob = bb.pow(2).sum(dim=(0, 2)).sqrt()[None, :]                               # (1, nblk)
    steps = k / nfma if nfma else 0.0
    noise = u * (C1 * a_norm * b_frob + (C2 * math.sqrt(k) + C3 * steps) * c_norm)
    if mode == "fp64":
        noise = noise * 2                     # the reference's own float64 rounding is of the same size
    noise = noise + 1e-300
    z = (diff / noise.unsqueeze(-1)).amax(dim=-1)
    # wrong, not merely unjudged: non-finite results, outputs these inputs can't produce, and blocks whose
    # tolerance itself overflowed (only an absurdly large output does that)
    bad = ~torch.isfinite(actual).all(dim=-1) | impossible | ~torch.isfinite(noise) | torch.isnan(z)
    z = torch.where(bad, torch.full_like(z, float("inf")), z)
    typical = c_norm / math.sqrt(BLOCK)
    caught = (ALARM * noise / typical.clamp_min(1e-300))[(typical > 0) & ~bad]
    return {"wrong_blocks": int((z > ALARM).sum()), "blocks": m * nblk, "worst_z": float(z.max()),
            "typical_caught": float(caught.median()) if caught.numel() else None, "z": z}


def check_once(mode: str, dev: torch.device, shape: tuple, dist: str, scale: float, gen: torch.Generator,
               fault: Optional[float] = None, run: Optional[Callable] = None, nfma: Optional[int] = None,
               keep: bool = False, unit: Optional[float] = None) -> dict:
    """One GEMM in `mode` on dev, checked against the CPU reference with fresh sign patterns.

    fault (tests, experiments): add an error to one output, this many times the typical (root-mean-square)
    output of its block; for INT8 any fault adds 1. run: compute the GEMM some other way (tests emulate
    tensor-core rounding with it). nfma, unit: tensor-core products per step and the rounding unit (default:
    from the GPU). keep: on a wrong result, return the operands too (for evidence that `sdcguard replay` can
    recompute). FP8 checks use an inner size of at most FP8_K."""
    m, k, n = shape
    if mode == "fp8":
        k = min(k, FP8_K)
        shape = (m, k, n)
    if n % BLOCK:
        raise ValueError(f"n must be a multiple of {BLOCK}")
    a, b, a64, b64 = _inputs(mode, shape, dist, scale, gen)
    signs = draw_signs(gen)
    c = (run or gemm)(mode, a.to(dev), b.to(dev))
    if fault is not None:
        i, j = m // 3, n // 2
        if mode == "int8":
            c[i, j] += 1
        else:
            blk = c[i, (j // BLOCK) * BLOCK:(j // BLOCK + 1) * BLOCK].double()
            c[i, j] += fault * float(blk.norm()) / math.sqrt(BLOCK)
    cc = torch.cuda.get_device_capability(dev) if dev.type == "cuda" else None
    if nfma is None:
        nfma = n_fma(mode, cc)
    if unit is None:
        unit = acc_unit(mode, cc)
    r = judge(mode, a64, b64, c, nfma, signs, unit)
    r.pop("z")
    if keep and r["wrong_blocks"]:
        r["operands"] = {"mode": mode, "A": a, "B": b, "C": c.detach().cpu(), "signs": signs, "shape": list(shape),
                         "dist": dist, "scale": scale, "nfma": nfma, "unit": unit}
    return r


def save_evidence(out_dir: str, n: int, failure: dict, gpu: dict, meta: dict) -> str:
    """Write a math-mode failure as math_mode_evidence_<n>.pt (inputs, result, sign patterns) and .json."""
    import json
    import os
    path = os.path.join(out_dir, f"math_mode_evidence_{n}")
    torch.save({"kind": "math_mode", **failure}, path + ".pt")
    info = {"what": f"a matmul in {failure['mode']} whose result disagreed with the CPU's float64 answer",
            "mode": failure["mode"], "profile": dict(zip(("inputs", "compute", "result", "reference"),
                                                         PROFILE[failure["mode"]])),
            "shape_mkn": failure["shape"], "input_distribution": f"{failure['dist']} x{failure['scale']:g}",
            "wrong_blocks": failure.get("wrong_blocks"), "worst_z": failure.get("worst_z"), "alarm_line": ALARM,
            "gpu": gpu, **meta, "replay": f"sdcguard replay {path}.pt"}
    with open(path + ".json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    return path


def replay(ev: dict, times: int, dev: torch.device, nfma: Optional[int] = None) -> dict:
    """Recompute a saved math-mode GEMM `times` times on dev and compare each result with the CPU reference."""
    mode = ev["mode"]
    a, b, signs = ev["A"], ev["B"], ev["signs"]
    a64, b64 = a.double(), b.double()
    cc = torch.cuda.get_device_capability(dev) if dev.type == "cuda" else None
    if nfma is None:
        nfma = n_fma(mode, cc)
    unit = acc_unit(mode, cc)
    rec = judge(mode, a64, b64, ev["C"], ev.get("nfma"), signs, ev.get("unit"))
    bad = 0
    worst = 0.0
    for _ in range(times):
        r = judge(mode, a64, b64, gemm(mode, a.to(dev), b.to(dev)), nfma, signs, unit)
        bad += bool(r["wrong_blocks"])
        if r["worst_z"] is not None:
            worst = max(worst, r["worst_z"] if math.isfinite(r["worst_z"]) else float("inf"))
    return {"mode": mode, "recorded_wrong_blocks": rec["wrong_blocks"], "recomputations": times,
            "recomputations_that_failed_the_check": bad, "worst_z_of_recomputations": None if mode == "int8" else worst}


def screen(dev: torch.device, seconds: float, shape: Optional[tuple] = None, seed: int = 0, say=None,
           keep_failures: int = 3) -> dict:
    """Every mode the GPU and PyTorch support, cycling through the input variants until `seconds` is used
    (each mode at least one full cycle, at most four). Modes the GPU or PyTorch doesn't support are listed as
    unsupported; a supported mode that errors is listed as an error (an incomplete check, not a pass). The
    operands of up to keep_failures wrong results come back in out["failures"] (not JSON: save them with
    save_evidence and pop them)."""
    say = say or (lambda s: None)
    shape = tuple(shape or SHAPE["cuda" if dev.type == "cuda" else "cpu"])
    gen = torch.Generator().manual_seed(1000 + seed)
    out = {"shape": list(shape), "modes": {}}
    runnable = []
    for m in MODES:
        try:
            why = available(m, dev)
        except ProbeError as e:
            say(f"math mode {m}: {e}")
            out["modes"][m] = {"matmuls": 0, "wrong": 0, "error": str(e)[:160]}
            continue
        if why:
            out["modes"][m] = {"skipped": why}
        else:
            out["modes"][m] = {"matmuls": 0, "wrong": 0, "blocks_checked": 0, "worst_z": 0.0,
                               "typical_error_caught": None}
            runnable.append(m)
    failures = []
    t_end = time.time() + seconds
    rnd = 0
    while runnable and rnd < 4 * len(VARIANTS) and (rnd < len(VARIANTS) or time.time() < t_end):
        dist, scale = VARIANTS[rnd % len(VARIANTS)]
        for m in list(runnable):
            rec = out["modes"][m]
            try:
                r = check_once(m, dev, shape, dist, scale, gen, keep=len(failures) < keep_failures)
            except Exception as e:  # noqa: BLE001  (an unsupported shape mustn't stop the whole check)
                say(f"math mode {m} stopped: {type(e).__name__}: {e}")
                rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                runnable.remove(m)
                continue
            rec["matmuls"] += 1
            rec["blocks_checked"] += r["blocks"]
            rec["wrong"] += bool(r["wrong_blocks"])
            if "operands" in r:
                failures.append({**r.pop("operands"), "wrong_blocks": r["wrong_blocks"], "worst_z": r["worst_z"]})
            if r["worst_z"] is not None and math.isfinite(r["worst_z"]):
                rec["worst_z"] = round(max(rec["worst_z"], r["worst_z"]), 3)
            if r["typical_caught"] is not None:      # the least sensitive input distribution's value
                prev = rec["typical_error_caught"]
                rec["typical_error_caught"] = max(prev or 0.0, r["typical_caught"])
        rnd += 1
    for m in runnable:
        rec = out["modes"][m]
        if m == "int8":
            rec["worst_z"] = None
            rec["typical_error_caught"] = "exact sums"       # integer checksums must match exactly
        elif rec["typical_error_caught"] is not None:
            rec["typical_error_caught"] = float(f"{rec['typical_error_caught']:.2g}")
    out["wrong"] = sum(r.get("wrong", 0) for r in out["modes"].values())
    out["matmuls"] = sum(r.get("matmuls", 0) for r in out["modes"].values())
    out["ran"] = [m for m in MODES if out["modes"][m].get("matmuls") and "error" not in out["modes"][m]]
    out["not_checked"] = {m: r.get("skipped") or r.get("error") for m, r in out["modes"].items() if m not in out["ran"]}
    out["errors"] = {m: r["error"] for m, r in out["modes"].items() if "error" in r}
    out["unsupported"] = {m: r["skipped"] for m, r in out["modes"].items() if "skipped" in r}
    out["failures"] = failures
    return out
