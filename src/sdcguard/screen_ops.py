"""Operator tests for screening GPUs: computations whose exact answer is known, compared bit for bit, and matmuls of
realistic data checked against a stated bound and against themselves.

    python -m sdcguard.screen_ops --minutes 5                 # this machine's first GPU (or --device cuda:3)
    python -m sdcguard.screen_ops --minutes 5 --out screen_ops_out --simulate gemm_bf16   # a demo of a failure

`sdcguard screen` runs a checked transformer, which is what GPUs are bought for, but checks bf16 and fp16 tensor-core
matmuls only, within a noise model. Defective GPUs in the field also compute wrong FP64 and FP32 results on their
CUDA cores (in ByteDance's study, OSDI '26, those units failed more often than the bf16 tensor cores), and wrong values
in memory. These tests judge each result exactly, with no noise model, except the realistic-data matmuls, which use
the spot check's stated bound:

  gemm_<p>      matmuls in FP64, FP32 (TF32 off), TF32, BF16, FP16, FP8 (e4m3, torch._scaled_mm, with and without
                fast accumulation, where the GPU has it) and INT8 (torch._int_mm), on integer-valued inputs small enough that every product and
                every partial sum is exact in that arithmetic in any order of additions, so the result is the exact
                integer (rounded once to the output dtype for BF16 and FP16, with PyTorch's reduced-precision split-K
                reductions turned off). Each input set is also run scaled by powers of two (exact) over much of the
                exponent range, and transposed (another kernel); the reference is computed once per set on the CPU in
                float64. Each precision has SETS input sets of different shapes and inner sizes (other cuBLAS kernels
                and tile edges), two of them each round in turn.
  gemm_bf16_real, matmuls of random normal inputs (the data a served model's layers see: every exponent and
  gemm_fp16_real  rounding, which integer-valued inputs never reach) in three shapes, a round each, alternately
                transposed. REAL_SAMPLES outputs of each are compared with a float64 reference computed on the CPU
                and scored against the spot check's bound (output rounding plus 6 times the fp32 accumulation's error
                model; a correct output scores at most 1, over 2 is wrong: a median sampled output moved by more than
                about 1.3% to 1.9% in bf16 or 0.3% to 0.9% in fp16, by K, is always found, serve_spot.min_caught), and
                every output is compared bit for bit with the same matmul run again (a value that changes between two
                identical runs, however little). The report keeps the worst score.
  fma_fp32,     x * y + z elementwise on the CUDA cores over tens of millions of elements, against the same in int64
  fma_fp64      on the integer units.
  sm_consistency  exp, sin, tanh, rsqrt and log of the same values laid out in two different orders, so other SMs and
                lanes compute each one: the results must be identical (these functions are deterministic per element;
                their exact values aren't known, so this only finds a unit that disagrees with the others).
  memory        a share of free memory filled with an address-dependent pattern and its complement, read back after
                every chunk is written: values that changed.

Every test runs in rounds until the time is up (memory at the start and the end). A single wrong value is a FAIL; the
report says which test, round, input set, scaling, element, the value and the expected one, so it can be rerun. The GPU's
identity and telemetry (UUID, serial, PCI, driver, VBIOS, ECC mode and counters, remapped rows, clocks, temperature,
power) are read from nvidia-smi before and after. Exit code: 0 = no wrong value, 1 = wrong values found, 2 = could not
run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zlib
from typing import Optional

import torch

from . import __version__
from .events import device_identity

SMI_FIELDS = ("name", "uuid", "serial", "pci.bus_id", "driver_version", "vbios_version", "ecc.mode.current",
              "ecc.errors.corrected.volatile.total", "ecc.errors.uncorrected.volatile.total", "clocks.sm",
              "clocks.mem", "clocks.max.sm", "temperature.gpu", "power.draw", "power.limit", "pstate",
              "clocks_throttle_reasons.active")
# integer ranges [-r, r] and inner sizes K that keep every product and partial sum exact (see each line)
GEMM = {
    "fp64": dict(dtype=torch.float64, r=2 ** 19, k=1024, tf32=False),   # |sum| <= 2^38 * 2^10 = 2^48 < 2^53
    "fp32": dict(dtype=torch.float32, r=32, k=2048, tf32=False),        # 2^10 * 2^11 = 2^21 < 2^24
    "tf32": dict(dtype=torch.float32, r=32, k=2048, tf32=True),         # inputs < 2^11 exact in TF32; 2^21 < 2^24
    "bf16": dict(dtype=torch.bfloat16, r=8, k=2048, tf32=False),        # inputs exact; 2^6 * 2^11 = 2^17 < 2^24
    "fp16": dict(dtype=torch.float16, r=4, k=2048, tf32=False),         # 2^4 * 2^11 = 2^15 < 65504
    "fp8": dict(dtype=getattr(torch, "float8_e4m3fn", None), r=3, k=512, tf32=False),   # 9 * 2^9 < 2^13
    "int8": dict(dtype=torch.int8, r=127, k=2048, tf32=False),          # 127^2 * 2^11 < 2^31 (int32 output)
}
SIZE = (1024, 1024)            # M, N of the first input set of each precision
SETS = 4                       # input sets per precision (each with its own CPU reference), two run each round
# the other sets' M, N and their K as a share of the precision's largest exact K: other shapes and lengths reach other
# cuBLAS kernels and tile edges (M, N, K multiples of 16 where FP8 and INT8 need them)
SHAPES = ((None, 1.0), ((32, 4096), 1.0), ((512, 1536), 0.75), ((272, 2992), 0.5))
# M, N, K of the realistic-data matmuls, one a round in turn: a decode step's few rows, a prefill block, a long K
REAL = ((16, 4096, 4096), (512, 1536, 2048), (256, 3072, 8960))
REAL_SAMPLES = 256             # outputs of each realistic-data matmul recomputed in float64 on the CPU
REAL_LINE = 2.0                # a sampled output further from its reference than this many bounds is wrong
REAL_DTYPE = {"gemm_bf16_real": torch.bfloat16, "gemm_fp16_real": torch.float16}
SCALES = {"fp64": (-600, 600), "fp32": (-50, 50), "tf32": (-50, 50), "bf16": (-50, 50), "fp16": (0, 0),
          "fp8": (0, 0), "int8": (0, 0)}       # exponents added to the result (half on each input), kept within normal range
FMA_N = 1 << 24
SM_N = 1 << 22
MAX_REPORTED = 8


def _smi(index) -> dict:
    """nvidia-smi's identity and telemetry for one GPU (its UUID or index), as text values ('[N/A]' kept)."""
    if index is None:
        return {}
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=" + ",".join(SMI_FIELDS),
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        line = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if line:
            out = dict(zip(SMI_FIELDS, [v.strip() for v in line[-1].split(",")]))
    except Exception:  # noqa: BLE001  (telemetry is best effort)
        return out
    try:
        r = subprocess.run(["nvidia-smi", f"--id={index}", "--query-remapped-rows=remapped_rows.correctable,"
                            "remapped_rows.uncorrectable,remapped_rows.pending,remapped_rows.failure",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        line = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if r.returncode == 0 and line:
            out["remapped_rows"] = line[-1].strip()
    except Exception:  # noqa: BLE001
        pass
    return out


class _Found:
    """Wrong values of one test: a count and the first few, with where to rerun them."""

    def __init__(self, name: str):
        self.name, self.wrong, self.checked, self.rounds, self.first = name, 0, 0, 0, []
        self.worst: Optional[float] = None               # the largest score, for tests judged against a bound
        self.cases: set = set()                          # (input set or shape, transposed) the test ran

    def _case(self, where: dict) -> None:
        k = where.get("set", tuple(where["shape"]) if "shape" in where and "set" not in where else None)
        if k is not None and "transposed" in where:
            self.cases.add((k, bool(where["transposed"])))

    def add(self, got: torch.Tensor, want: torch.Tensor, where: dict) -> None:
        self._case(where)
        g, w = got.reshape(-1), want.reshape(-1)
        bad = g.view(_int_view(g.dtype)) != w.view(_int_view(w.dtype)) if g.dtype == w.dtype else g != w
        n = int(bad.sum())
        self.checked += g.numel()
        if n:
            self.wrong += n
            for i in torch.nonzero(bad).flatten()[:max(0, MAX_REPORTED - len(self.first))].tolist():
                self.first.append({**where, "element": int(i), "got": _num(g[i]), "expected": _num(w[i])})

    def add_scores(self, z: torch.Tensor, where: dict, at: dict, line: float = REAL_LINE) -> None:
        """Sampled values judged against a bound: z [S] is each one's distance from its reference in units of the
        bound, and over the line is wrong (NaN too). at: tensors [S] reported with a wrong value (its row, column,
        value and reference)."""
        self._case(where)
        self.checked += z.numel()
        if z.numel():
            top = float(torch.nan_to_num(z.float(), nan=float("inf")).max())
            self.worst = top if self.worst is None else max(self.worst, top)
        bad = ~(z <= line)
        n = int(bad.sum())
        if n:
            self.wrong += n
            for i in torch.nonzero(bad).flatten()[:max(0, MAX_REPORTED - len(self.first))].tolist():
                self.first.append({**where, **{k: (_num(v[i]) if v.is_floating_point() else int(v[i]))
                                               for k, v in at.items()}, "score": _num(z[i])})

    def result(self) -> dict:
        out = {"test": self.name, "rounds": self.rounds, "values_checked": self.checked, "wrong": self.wrong,
               "first_wrong": self.first}
        if self.cases:
            out["cases"] = sorted([list(k) if isinstance(k, tuple) else k, t] for k, t in self.cases)
        if self.worst is not None:
            out["worst_score"] = round(self.worst, 4) if self.worst != float("inf") else repr(self.worst)
        return out


def _int_view(dtype) -> torch.dtype:
    return {8: torch.int64, 4: torch.int32, 2: torch.int16, 1: torch.uint8}[torch.empty(0, dtype=dtype).element_size()]


def _num(v: torch.Tensor):
    x = float(v.float() if v.dtype not in (torch.float64,) else v)
    return x if x == x and abs(x) != float("inf") else repr(x)


def _ints(gen: torch.Generator, shape, r: int) -> torch.Tensor:
    return torch.randint(-r, r + 1, shape, generator=gen, dtype=torch.int64).double()


def gemm_supported(name: str, device: torch.device) -> Optional[str]:
    """Why a matmul precision can't run on this device, or None."""
    p = GEMM[name]
    if p["dtype"] is None:
        return "this PyTorch has no float8_e4m3fn"
    if device.type != "cuda":
        return None if name in ("fp64", "fp32", "bf16", "fp16", "int8") else f"{name} needs an NVIDIA GPU"
    cc = torch.cuda.get_device_capability(device)
    if name == "tf32" and cc[0] < 8:
        return "TF32 needs Ampere or newer"
    if name == "bf16" and cc[0] < 8:
        return "BF16 tensor cores need Ampere or newer"
    if name == "fp8" and (cc[0], cc[1]) < (8, 9):
        return "FP8 tensor cores need Ada or Hopper or newer"
    if name == "int8" and cc[0] < 7:
        return "INT8 tensor cores need Turing or newer"
    return None


class GemmSet:
    """One input set of a precision: integer-valued A [M, K] and B [K, N], their exact product (float64, on the CPU)
    and its value in the output dtype."""

    def __init__(self, name: str, seed: int, k: int, device: torch.device, size=SIZE):
        p = GEMM[name]
        gen = torch.Generator().manual_seed(seed)
        m, n = size
        a, b = _ints(gen, (m, k), p["r"]), _ints(gen, (k, n), p["r"])
        self.name, self.seed, self.device = name, seed, device
        self.exact = a @ b                                    # float64: exact (|sum| < 2^53)
        if name == "int8":
            self.want = self.exact.to(torch.int64).to(torch.int32).to(device)
            self.name, self.seed, self.device = name, seed, device
            self.a, self.b = a.to(torch.int8).to(device), b.to(torch.int8).to(device)
            return
        out = torch.float32 if name == "fp8" else p["dtype"]
        # FP64: the exact value; the others: exact in float32 (< 2^24), then one rounding to the output dtype
        self.want = (self.exact if out == torch.float64 else self.exact.to(torch.float32).to(out)).to(device)
        self.a, self.b = a.to(device), b.to(device)

    def run(self, ea: int, eb: int, transposed: bool) -> tuple:
        """(result, expected) with A scaled by 2^ea and B by 2^eb; transposed computes (B^T A^T)^T."""
        p = GEMM[self.name]
        dt = p["dtype"]
        if self.name == "int8":                          # INT8 tensor cores, int32 accumulation and output
            if transposed:
                return torch._int_mm(self.b.t().contiguous(), self.a.t().contiguous()).t(), self.want
            return torch._int_mm(self.a, self.b), self.want
        a = (self.a * 2.0 ** ea).to(dt) if self.name != "fp8" else self.a.float().to(dt)
        b = (self.b * 2.0 ** eb).to(dt) if self.name != "fp8" else self.b.float().to(dt)
        if self.name == "fp8":
            one = torch.ones((), device=self.device, dtype=torch.float32)
            fast = transposed                                  # both accumulation paths, in turn
            y = torch._scaled_mm(a, b.t().contiguous().t(), scale_a=one, scale_b=one, out_dtype=torch.float32,
                                 use_fast_accum=fast)
            return y, self.want
        if transposed:
            y = (b.t() @ a.t()).t()
        else:
            y = a @ b
        want = self.want
        if ea + eb:
            want = (want.double() * 2.0 ** (ea + eb)).to(want.dtype)       # exact: a power of two, in range
        return y, want


class _Matmul:
    """TF32 set, PyTorch's reduced-precision split-K reductions and fp16 accumulation (PyTorch 2.7+) off, for an exact
    test, then every setting put back as it was (the reductions' split-K part too, where PyTorch has it)."""

    RED = ("allow_bf16_reduced_precision_reduction", "allow_fp16_reduced_precision_reduction")

    def __init__(self, tf32: bool):
        self.tf32 = tf32

    @staticmethod
    def _get(name):
        m = torch.backends.cuda.matmul
        v = getattr(m, name)
        try:
            return (v, getattr(m, name + "_split_k"))           # (PyTorch with the split-K part: set as a pair)
        except AttributeError:
            return v

    def __enter__(self):
        m = torch.backends.cuda.matmul
        self.saved = {"allow_tf32": m.allow_tf32, **{n: self._get(n) for n in self.RED}}
        try:
            self.saved["allow_fp16_accumulation"] = m.allow_fp16_accumulation
        except AttributeError:
            pass
        m.allow_tf32 = self.tf32
        for n in self.RED:
            v = self.saved[n]
            setattr(m, n, (False, v[1]) if isinstance(v, tuple) else False)
        if "allow_fp16_accumulation" in self.saved:
            m.allow_fp16_accumulation = False
        return self

    def __exit__(self, *a):
        m = torch.backends.cuda.matmul
        for n, v in self.saved.items():
            setattr(m, n, v)
        return False


def _scales(name: str, rnd: int, s: int) -> tuple:
    lo, hi = SCALES[name]
    if lo == hi:
        return 0, 0
    span = hi - lo
    e = lo + ((rnd * 7919 + s * 104729) % (span + 1))
    return e // 2, e - e // 2


def real_samples(seed: int, shape: tuple) -> tuple:
    """(rows, cols) of the outputs a realistic-data matmul recomputes, REAL_SAMPLES distinct ones drawn from its
    seed, so a report's seed and shape say which outputs were sampled."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    m, n = shape[0], shape[1]
    idx = torch.randperm(m * n, generator=g)[:REAL_SAMPLES]
    return idx // n, idx % n


def real_round(dtype, shape: tuple, seed: int, device, transposed: bool = False, wrong: Optional[str] = None) -> dict:
    """A matmul of random normal inputs in bf16 or fp16 (drawn on the device): the data a served model's layers see,
    which integer-valued inputs don't reach (every exponent and rounding, and the accumulation's own error). It is
    checked two ways: REAL_SAMPLES of its outputs against a reference computed in float64 on the CPU from the sampled
    inputs as the device held them, each scored as the spot check scores them (serve_spot: |y - r| over the output's
    rounding plus 6 times the fp32 accumulation's error model, so a correct output scores at most 1 and over REAL_LINE
    is wrong); and every output against the same matmul run again, bit for bit (cuBLAS gives the same bits for the
    same call on the same GPU, so a value that changed between the runs is a fault the reference can't see unless it
    is sampled). transposed computes
    (B^T A^T)^T, another kernel. wrong (a demo of a failure): "both" moves one sampled output in both runs, as
    a persistent fault would; "second" does it in the second run only, as an intermittent one would. (3% of its value
    plus 0.01, away from zero: well over the line for any value.)
    Returns the scores, the sampled rows, columns, values and references, and the two runs."""
    from .serve_norm import dtype_consts
    from .serve_spot import acc_term
    m, n, k = shape
    rows, cols = real_samples(seed, shape)
    gd = torch.Generator(device=device).manual_seed(seed + 1)       # (drawn where they are used: fast on a GPU)
    ad = torch.randn(m, k, generator=gd, device=device).to(dtype)
    bd = (torch.randn(k, n, generator=gd, device=device) * k ** -0.5).to(dtype)
    mm = (lambda: (bd.t() @ ad.t()).t()) if transposed else (lambda: ad @ bd)
    y, again = mm(), mm()
    if wrong:
        r0, c0 = int(rows[0]), int(cols[0])
        y0 = y[r0, c0].float()
        v = (y0 + torch.copysign(0.03 * y0.abs() + 0.01, y0)).to(dtype)
        if wrong == "both":
            y = y.clone()
            y[r0, c0] = v
        again = again.clone()
        again[r0, c0] = v
    rd, cd = rows.to(device), cols.to(device)
    got = y[rd, cd].cpu().double()
    # the sampled inputs as the GPU held them ([S, K] each, contiguous: about 18 MB in float64 at K = 8,960)
    xa = ad.index_select(0, rd).cpu().double()
    wb = bd.index_select(1, cd).t().contiguous().cpu().double()
    ref = (xa * wb).sum(1)                                 # products exact; the sum's own error is negligible
    norm = xa.norm(dim=1) * wb.norm(dim=1)
    u, sp, _ = dtype_consts(dtype)
    den = u * ref.abs() + acc_term(k, norm, ref.abs()).double() + sp
    z = (got - ref).abs() / den
    return {"scores": z, "rows": rows, "cols": cols, "got": got, "ref": ref, "y": y, "again": again}


def fma_round(dtype, n: int, seed: int, device) -> tuple:
    """(x * y + z on the floating-point units, the same in int64 on the integer units) for integer-valued x, y, z
    whose result is exact in dtype."""
    g = torch.Generator(device=device).manual_seed(seed)     # (drawn where they are used: seeds repeat per kind)
    if dtype == torch.float32:
        rx, rz = 2 ** 11, 2 ** 22                     # |x y + z| <= 2^22 + 2^22 = 2^23 < 2^24
    else:
        rx, rz = 2 ** 25, 2 ** 51                     # |x y + z| <= 2^50 + 2^51 < 2^53
    x = torch.randint(-rx, rx + 1, (n,), generator=g, dtype=torch.int64, device=device)
    y = torch.randint(-rx, rx + 1, (n,), generator=g, dtype=torch.int64, device=device)
    z = torch.randint(-rz, rz + 1, (n,), generator=g, dtype=torch.int64, device=device)
    got = torch.addcmul(z.to(dtype), x.to(dtype), y.to(dtype))
    return got, (x * y + z).to(dtype)


SM_FUNCS = {"exp": torch.exp, "sin": torch.sin, "tanh": torch.tanh, "rsqrt": torch.rsqrt, "log": torch.log}


def sm_round(n: int, seed: int, device) -> list:
    """For each function: (f(x) laid out in a random order, then put back; f(x) in order)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(n, generator=g) * 20 - 10).to(device)
    perm = torch.randperm(n, generator=g).to(device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=device)
    out = []
    for fname, f in SM_FUNCS.items():
        xx = x.abs() + 1e-3 if fname in ("rsqrt", "log") else x
        out.append((fname, f(xx[perm])[inv], f(xx)))
    return out


def _pattern(n: int, base: int, seed: int, device, invert: bool) -> torch.Tensor:
    i = torch.arange(base, base + n, device=device, dtype=torch.int64)
    h = (i * 2654435761 + seed * 40503) & 0xFFFFFFFF
    h = (h ^ (h >> 15)) * 2246822519 & 0xFFFFFFFF
    h = h ^ (h >> 13)
    if invert:
        h = h ^ 0xFFFFFFFF
    return (h - ((h >> 31) << 32)).to(torch.int32)               # as int32, wrapping


def memory_test(device, share: float, seed: int, found: _Found, chunk_mb: int = 256, limit_gb: float = 32.0,
                inject: Optional[int] = None) -> dict:
    """Fill `share` of free memory (at most limit_gb) with a pattern in chunks, then read every chunk back; the same
    with the complement. inject: for tests, change one value of chunk 0 before it is read."""
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
    else:
        free = 1 << 30
    total = int(min(free * share, limit_gb * 2 ** 30))
    per = max(1 << 14, min(chunk_mb * 2 ** 20, total) // 4)          # int32 values a chunk
    n_chunks = max(1, total // (per * 4))
    chunks = []
    try:
        for _ in range(n_chunks):
            chunks.append(torch.empty(per, dtype=torch.int32, device=device))
    except torch.cuda.OutOfMemoryError:
        pass
    for invert in (False, True):
        sub = 1 << 22
        for c, t in enumerate(chunks):
            for o in range(0, per, sub):
                t[o:o + sub] = _pattern(min(sub, per - o), c * per + o, seed, device, invert)
        if inject is not None and chunks and not invert:
            chunks[0][inject] ^= 1 << 7
        for c, t in enumerate(chunks):
            for o in range(0, per, sub):
                found.add(t[o:o + sub], _pattern(min(sub, per - o), c * per + o, seed, device, invert),
                          {"chunk": c, "offset": o, "complement": invert})
    found.rounds += 1
    return {"bytes": len(chunks) * per * 4}


def coverage_rounds(real=REAL) -> int:
    """Rounds for every matmul test to run each of its SETS input sets (two a round) and each realistic-data shape,
    plain and transposed (a shape a round, transposed in every other pass over them)."""
    return max(-(-SETS // 2), 2 * len(real), 1)


def run(device: torch.device, minutes: float, out_dir: Optional[str] = None, seed: int = 0, mem_share: float = 0.5,
        simulate: Optional[str] = None, tests=None, size=SIZE, fma_n: int = FMA_N, sm_n: int = SM_N,
        mem_limit_gb: float = 32.0, shapes=SHAPES, real=REAL, min_rounds: int = 1) -> dict:
    """Run the tests on one device for about `minutes` of testing, timed from when the inputs and their references
    are ready, and with minutes > 0 for at least as many rounds as it takes every test to run each of its input sets
    and shapes, plain and transposed (coverage_rounds), and at least min_rounds. simulate: a test name whose first
    result is made wrong in one value (a demo of what a failure looks like, never a GPU fault); a realistic-data
    test's name with ".repeat" makes the value wrong in its second run only, which the repeat finds. shapes: each
    input set's (M, N) (None: size) and its K as a share of the precision's largest; real: the realistic-data
    matmuls' (M, N, K)."""
    t0 = time.time()
    ident = device_identity(device)
    smi_id = ident.get("gpu_uuid") or (str(device.index) if device.type == "cuda" else None)
    report: dict = {"sdcguard": __version__, "torch": torch.__version__, "device": str(device), "gpu": ident,
                    "telemetry_before": _smi(smi_id) if device.type == "cuda" else {}, "seed": seed,
                    "minutes_asked": minutes, "simulated": simulate, "tests": {}, "skipped": {}}
    sim_mode = "second" if simulate and simulate.endswith(".repeat") else "both"
    simulate = simulate[:-len(".repeat")] if simulate and simulate.endswith(".repeat") else simulate
    names = [f"gemm_{p}" for p in GEMM] + ["gemm_bf16_real", "gemm_fp16_real", "fma_fp32", "fma_fp64",
                                           "sm_consistency", "memory"]
    names = [n for n in names if tests is None or n in tests]
    found = {n: _Found(n) for n in names}
    sets: dict = {}
    for n in names:
        if n.startswith("gemm_") and not n.endswith("_real"):
            why = gemm_supported(n[5:], device)
            if why:
                report["skipped"][n] = why
                del found[n]
                continue
            p = GEMM[n[5:]]
            sets[n] = []
            for s in range(SETS):
                shape, kshare = shapes[s % len(shapes)]
                k = max(16, int(p["k"] * kshare) // 16 * 16)
                sets[n].append(GemmSet(n[5:], seed * 1000 + 17 * s + zlib.crc32(n.encode()) % 997, k, device,
                                       size if shape is None else shape))
        elif n.endswith("_real"):
            why = gemm_supported(n[5:-5], device)
            if device.type != "cuda" and simulate != n and tests is None:
                why = "realistic-data matmuls are for GPUs (on the CPU, ask for them with tests)"
            if why:
                report["skipped"][n] = why
                del found[n]
    error = None
    if minutes > 0:
        min_rounds = max(min_rounds, coverage_rounds(real))
    report["prepare_minutes"] = round((time.time() - t0) / 60, 2)
    deadline = time.time() + 60.0 * minutes          # testing time: the inputs and their references are ready
    rnd = 0
    try:
        if "memory" in found:
            memory_test(device, mem_share, seed, found["memory"], limit_gb=mem_limit_gb,
                        inject=12345 if simulate == "memory" else None)
        while True:
            for n, ss in sets.items():
                p = GEMM[n[5:]]
                with _Matmul(p["tf32"]):
                    for s in ((2 * rnd) % len(ss), (2 * rnd + 1) % len(ss)) if len(ss) > 2 else range(len(ss)):
                        gs = ss[s]
                        ea, eb = _scales(n[5:], rnd, s)
                        for tr in (False, True):
                            y, want = gs.run(ea, eb, tr)
                            if simulate == n and rnd == 0 and s == 0 and not tr:
                                y = y.clone()
                                y.view(-1)[777] = y.view(-1)[777] * 2 + 1
                            found[n].add(y, want, {"round": rnd, "set": s, "seed": gs.seed, "scale_a": ea,
                                                   "scale_b": eb, "transposed": tr, "shape": list(y.shape)})
                found[n].rounds += 1
            for n, dt in REAL_DTYPE.items():
                if n in found:
                    shape, tr = real[rnd % len(real)], (rnd // len(real)) % 2 == 1
                    rs = seed * 13 + rnd
                    with _Matmul(False):
                        rr = real_round(dt, shape, rs, device, tr, wrong=sim_mode if simulate == n and rnd == 0 else None)
                    where = {"round": rnd, "seed": rs, "shape": list(shape), "transposed": tr}
                    found[n].add_scores(rr["scores"], {**where, "check": "reference"},
                                        {"row": rr["rows"], "col": rr["cols"], "got": rr["got"], "expected": rr["ref"]})
                    found[n].add(rr["again"], rr["y"], {**where, "check": "repeat"})
                    found[n].rounds += 1
            for n, dt in (("fma_fp32", torch.float32), ("fma_fp64", torch.float64)):
                if n in found:
                    got, want = fma_round(dt, fma_n, seed * 7 + rnd, device)
                    if simulate == n and rnd == 0:
                        got[5] += 1
                    found[n].add(got, want, {"round": rnd, "seed": seed * 7 + rnd})
                    found[n].rounds += 1
            if "sm_consistency" in found:
                for fname, a, b in sm_round(sm_n, seed * 11 + rnd, device):
                    if simulate == "sm_consistency" and rnd == 0 and fname == "exp":
                        a = a.clone()
                        a[3] = a[3] * 2
                    found["sm_consistency"].add(a, b, {"round": rnd, "function": fname, "seed": seed * 11 + rnd})
                found["sm_consistency"].rounds += 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            rnd += 1
            if time.time() >= deadline and rnd >= min_rounds:
                break
        if "memory" in found:
            memory_test(device, mem_share, seed + 1, found["memory"], limit_gb=mem_limit_gb)
    except Exception as e:  # noqa: BLE001  (a failing GPU can raise after computing wrong values: both are reported)
        error = f"{type(e).__name__}: {str(e).splitlines()[0][:300] if str(e) else ''}"
    report["tests"] = {n: f.result() for n, f in found.items()}
    wrong = sum(f.wrong for f in found.values())
    checked = sum(f.checked for f in found.values())
    report.update(rounds=rnd, minutes=round((time.time() - t0) / 60, 2), values_checked=checked, wrong=wrong,
                  telemetry_after=_smi(smi_id) if device.type == "cuda" else {}, error=error,
                  result="FAIL" if wrong else ("INCOMPLETE" if error or not checked else "PASS"))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = (ident.get("gpu_uuid") or str(device)).replace(":", "_").replace("/", "_")
        path = os.path.join(out_dir, f"screen_ops_{tag}.json")
        with open(path + ".tmp", "w") as f:
            json.dump(report, f, indent=1, default=str)
        os.replace(path + ".tmp", path)
        report["path"] = path
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sdcguard.screen_ops", description=__doc__.split("\n\n")[0])
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--device", default=None, help="e.g. cuda:0 (default: the first GPU, or the CPU)")
    ap.add_argument("--out", default="screen_ops_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--memory-share", type=float, default=0.5, help="share of free GPU memory the memory test uses")
    ap.add_argument("--tests", default=None, help="comma-separated subset of the tests")
    ap.add_argument("--simulate", default=None, help="demo only: make one value of this test's first result wrong "
                                                    "(gemm_bf16_real.repeat: in its second run only)")
    a = ap.parse_args(argv)
    try:
        dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        r = run(dev, a.minutes, a.out, a.seed, a.memory_share, a.simulate,
                a.tests.split(",") if a.tests else None)
    except Exception as e:  # noqa: BLE001
        print(f"sdcguard screen_ops: could not run: {type(e).__name__}: {e}")
        return 2
    g = r["gpu"]
    if r["simulated"]:
        print(f"  (SIMULATED: one value of {r['simulated']} was made wrong in software: a demo, not a GPU fault)")
    print(f"sdcguard screen_ops: {r['result']}  {g.get('gpu_name', r['device'])} {g.get('gpu_uuid') or ''} "
          f"PCI {g.get('pci_bus_id')}: {r['values_checked']:,} values checked in {r['minutes']} min, "
          f"{r['wrong']} wrong")
    for n, t in r["tests"].items():
        worst = f"  worst score {t['worst_score']} (line {REAL_LINE})" if "worst_score" in t else ""
        print(f"  {n:16s} {t['rounds']:5d} rounds {t['values_checked']:>16,} values  {t['wrong']} wrong{worst}")
        for w in t["first_wrong"][:3]:
            print(f"      {w}")
    for n, why in r["skipped"].items():
        print(f"  {n:16s} skipped: {why}")
    if r.get("error"):
        print(f"  stopped by an error: {r['error']} (what was found before it is reported)")
    if r.get("path"):
        print(f"  report: {r['path']}")
    return {"FAIL": 1, "INCOMPLETE": 2}.get(r["result"], 0)


if __name__ == "__main__":
    sys.exit(main())
