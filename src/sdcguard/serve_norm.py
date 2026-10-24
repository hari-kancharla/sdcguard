"""Checks of RMSNorm in serving: every output of a norm layer compared with the norm recomputed in fp32 from the input
it read and the weights it had at registration.

RMSNorm is y = x / sqrt(mean(x^2) + eps) * w over each row (GemmaRMSNorm: * (1 + w)). vLLM computes it in fp32 and
rounds to bf16/fp16 once (its CUDA kernel) or twice (its native PyTorch version, which rounds the normalized row
before multiplying by w; torch.compile runs that one). With the residual add fused in, the norm's input is the sum the
layer returns, which vLLM's CUDA kernel normalizes as rounded and the native version before rounding. So a correct
output is within 4 roundings of the value recomputed here from what the layer returned: |y - ref| <= 4 u |ref| (u: the
output dtype's unit roundoff, 2^-8 for bf16, 2^-11 for fp16), plus fp32 rounding in the mean, far below that. The
score of an output is |y - ref| / (u |ref| + s) (s: the dtype's subnormal spacing, so tiny values don't count): at
most 4 on a correct norm to first order, 4.03 counting the products of the rounding errors (bf16; 4.003 for fp16).
The alarm line is 6. An output off by more than about 10 u (3.9% in bf16) is caught for sure, one off by more than 2 u
when the roundings happen to agree.

The weights compared against are copied at registration, so a norm weight changed in memory shows up as wrong
outputs (refresh() takes the current weights after an intended update).

Under torch.compile the check must not be traced into vLLM's graph: Inductor would merge the recompute with the norm's
own computation (the same mean, the same rsqrt) and the check would compare the norm with itself. It runs as one
opaque custom op, sdcguard::norm_check: a Triton kernel on NVIDIA GPUs (one launch per norm call, which CUDA graphs
capture), the PyTorch version elsewhere and where the kernel didn't pass its first-use comparison.

What a read reports, per norm layer: the worst score with its row, column, output, expected value and model step, as
int64 keys with the score's bits on top (one atomic maximum per key; an exact tie between two rows can mix their row
and column), the calls and rows checked, and outputs that couldn't be judged: a non-finite expected value or one
beyond the dtype's range (the input itself was already wrong, or the true result overflows), and rows whose sum of
squares is within a few times of fp32's largest value (where two summation orders can disagree on overflowing; no
healthy model comes near it). The first row over the
line in a layer is kept (its input and output, row and step) until its alert has read it; the alert rechecks it in
float64 on the CPU, and the vLLM plugin names the request that held it.

Not covered: the residual add fused into the norm (the sum is taken as the layer returned it), RMSNorm with a
variance-size override, vLLM's batch-invariant mode (its own kernel), LayerNorm, RMSNormGated and norms fused into
other kernels.
"""
from __future__ import annotations

import os
import struct
import threading
from typing import Optional

import torch

from .events import SDCEvent, device_identity

NORM_Z = 6.0
MIN_LINE = 4.5           # a line at or below 4.03 roundings could alarm on correct norms
NORM_SCHEMA = 1          # the op's arguments (the vLLM plugin keys its compile cache on it)
COLS = 8                 # state columns, see C_*
C_ROW, C_COL, C_OUT, C_REF, C_STEP, C_CALLS, C_ROWS, C_UNJ = range(COLS)
CAP_COLS = 4             # a kept row's record: claimed, row, step (low 32 bits), score bits
M32 = 0xFFFFFFFF
STEP_MASK = M32
SS_MAX = 1e38            # rows with a larger sum of squares (fp32's largest is 3.4e38) aren't judged

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

STATUS: dict = {}        # (device key, dtype) -> (ok, why): the kernel's first-use comparison
FALLBACKS = [0]
OUT_DTYPES: dict = {}    # (state address, slot) -> the dtype of the outputs a norm's check last judged


def dtype_consts(dtype: torch.dtype) -> tuple:
    """(unit roundoff, subnormal spacing, largest finite value) of a norm's output dtype. bf16's spacing is a small
    normal fp32 value instead of its own (2^-133, which GPUs flush to 0 in fp32), so 0 / spacing stays 0. Outputs in
    fp32 (not expected in serving) are judged on bf16's roundoff: a lenient line."""
    if dtype == torch.float16:
        return 2.0 ** -11, 2.0 ** -24, 65504.0
    if dtype == torch.bfloat16:
        return 2.0 ** -8, 1e-37, float(torch.finfo(torch.bfloat16).max)
    return 2.0 ** -8, 1e-37, float(torch.finfo(torch.float32).max)


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def _rows2d(t: torch.Tensor, h: int) -> torch.Tensor:
    t2 = t.reshape(-1, h)
    return t2 if t2.shape[0] == 0 or t2.stride(-1) == 1 else t2.contiguous()


# ---------------------------------------------------------------------------------------------------- PyTorch check
def norm_check_torch(inp: torch.Tensor, out: torch.Tensor, w_ref: torch.Tensor, eps: float, state: torch.Tensor,
                     slot: int, stepbuf: torch.Tensor, cap_meta: torch.Tensor, cap_in: torch.Tensor,
                     cap_out: torch.Tensor, line: float) -> None:
    """The check in PyTorch operations (no host sync, so it runs inside CUDA graphs). inp, out: [..., h]."""
    u, spacing, fmax = dtype_consts(out.dtype)
    h = inp.shape[-1]
    x = inp.reshape(-1, h).float()
    y = out.reshape(-1, h).float()
    m = x.shape[0]
    if m == 0:
        return
    ss = x.square().sum(-1, keepdim=True)
    r = torch.rsqrt(ss / h + eps)
    ref = x * r * w_ref
    aref = ref.abs()
    z = (y - ref).abs() / (u * aref + spacing)
    # judged: an expected value that is finite and that the dtype can hold, in a row whose sum of squares is far
    # from fp32's overflow (near it, two summation orders can disagree on whether it overflowed)
    judged = (aref <= fmax) & (ss < SS_MAX)
    z = torch.where(judged, torch.nan_to_num(z, nan=float("inf"), posinf=float("inf")), torch.zeros_like(z))
    zmax, arg = z.reshape(-1).max(0)
    row, col = arg // h, arg % h
    yv, rv = y.reshape(-1)[arg], ref.reshape(-1)[arg]
    bits = lambda v: v.reshape(1).view(torch.int32).to(torch.int64).reshape(()) & M32  # noqa: E731
    zb = bits(zmax) << 32
    raw = stepbuf.reshape(-1)[0]
    keys = torch.stack([zb | row, zb | col, zb | bits(yv), zb | bits(rv), zb | (raw & M32)])
    srow = state[slot]
    srow[:5] = torch.maximum(srow[:5], keys)
    srow[C_CALLS] += 1
    srow[C_ROWS] += m
    srow[C_UNJ] += (~judged).sum()
    take = (zmax > line) & (cap_meta[slot, 0] == 0)
    cap_in[slot, :h] = torch.where(take, x[row], cap_in[slot, :h])
    cap_out[slot, :h] = torch.where(take, y[row], cap_out[slot, :h])
    rec = torch.stack([torch.ones_like(row), row, raw & M32, zb >> 32]).to(cap_meta.dtype)
    cap_meta[slot] = torch.where(take, rec, cap_meta[slot])


# ----------------------------------------------------------------------------------------------------- Triton check
if HAVE_TRITON:
    @triton.jit
    def _norm_check_kernel(inp, out, w, state, capm, capi, capo, stepbuf,
                           M, H, s_in, s_out, capw, slot, eps, line, u, spacing, fmax,
                           BR: tl.constexpr, TILE: tl.constexpr, NT: tl.constexpr, NCOLS: tl.constexpr,
                           SSMAX: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BR + tl.arange(0, BR)
        rmask = rows < M
        rows64 = rows.to(tl.int64)
        cols0 = tl.arange(0, TILE)
        acc = tl.zeros([BR, TILE], dtype=tl.float32)
        for t in tl.static_range(NT):
            c = t * TILE + cols0
            mk = rmask[:, None] & (c < H)[None, :]
            x = tl.load(inp + rows64[:, None] * s_in + c[None, :], mask=mk, other=0.0).to(tl.float32)
            acc += x * x
        ssum = tl.sum(acc, 1)
        r = 1.0 / tl.sqrt(ssum / H + eps)
        rowok = ssum < SSMAX
        zbest = tl.zeros([BR], dtype=tl.float32)
        cbest = tl.zeros([BR], dtype=tl.int32)
        ybest = tl.zeros([BR], dtype=tl.float32)
        rbest = tl.zeros([BR], dtype=tl.float32)
        unj = tl.zeros([BR], dtype=tl.int32)
        for t in tl.static_range(NT):
            c = t * TILE + cols0
            cm = c < H
            mk = rmask[:, None] & cm[None, :]
            x = tl.load(inp + rows64[:, None] * s_in + c[None, :], mask=mk, other=0.0).to(tl.float32)
            y = tl.load(out + rows64[:, None] * s_out + c[None, :], mask=mk, other=0.0).to(tl.float32)
            wv = tl.load(w + c, mask=cm, other=0.0)
            ref = x * r[:, None] * wv[None, :]
            aref = tl.abs(ref)
            z = tl.abs(y - ref) / (u * aref + spacing)
            z = tl.where(z == z, z, float("inf"))
            ok = mk & (aref <= fmax) & rowok[:, None]
            unj += tl.sum(tl.where(mk, 1, 0) - tl.where(ok, 1, 0), 1)
            z = tl.where(ok, z, 0.0)
            tz = tl.max(z, 1)
            ta = tl.argmax(z, 1)
            sel = cols0[None, :] == ta[:, None]
            ty = tl.sum(tl.where(sel, y, 0.0), 1)
            tr = tl.sum(tl.where(sel, ref, 0.0), 1)
            better = tz > zbest
            zbest = tl.where(better, tz, zbest)
            cbest = tl.where(better, ta + t * TILE, cbest)
            ybest = tl.where(better, ty, ybest)
            rbest = tl.where(better, tr, rbest)
        pz = tl.max(zbest, 0)
        pi = tl.argmax(zbest, 0)
        selr = tl.arange(0, BR) == pi
        prow = (pid * BR + pi).to(tl.int64)
        pcol = tl.sum(tl.where(selr, cbest, 0), 0).to(tl.int64)
        py = tl.sum(tl.where(selr, ybest, 0.0), 0)
        pr = tl.sum(tl.where(selr, rbest, 0.0), 0)
        nunj = tl.sum(unj, 0)
        base = state + slot * NCOLS
        zbits = pz.to(tl.int32, bitcast=True)
        zb = zbits.to(tl.int64) << 32
        raw = tl.load(stepbuf)
        if pz > 0.0:
            k0 = zb | prow
            cur = tl.load(base)
            if k0 > cur:                       # (most rows: read, not one atomic per row on one address)
                tl.atomic_max(base + 0, k0)
                tl.atomic_max(base + 1, zb | pcol)
                tl.atomic_max(base + 2, zb | (py.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                tl.atomic_max(base + 3, zb | (pr.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF))
                tl.atomic_max(base + 4, zb | (raw & 0xFFFFFFFF))
        if pid == 0:
            tl.atomic_add(base + 5, 1)
            tl.atomic_add(base + 6, M)
        if nunj > 0:
            tl.atomic_add(base + 7, nunj.to(tl.int64))
        if pz > line:
            cbase = capm + slot * 4
            old = tl.atomic_cas(cbase, 0, 1)
            if old == 0:                       # the first row over the line since the layer's last alert was read
                tl.store(cbase + 1, prow.to(tl.int32))
                tl.store(cbase + 2, raw.to(tl.int32))
                tl.store(cbase + 3, zbits)
                for t in tl.static_range(NT):
                    c = t * TILE + cols0
                    cm = c < H
                    xr = tl.load(inp + prow * s_in + c, mask=cm, other=0.0).to(tl.float32)
                    yr = tl.load(out + prow * s_out + c, mask=cm, other=0.0).to(tl.float32)
                    tl.store(capi + slot * capw + c, xr, mask=cm)
                    tl.store(capo + slot * capw + c, yr, mask=cm)


def _pow2_ceil(x: int) -> int:
    return 1 << (max(1, int(x)) - 1).bit_length()


def kernel_config(h: int) -> tuple:
    """(rows per program, tile width, tiles per row, warps): at most 2048 values of a tile per program at once
    (4096 for a row wider than 2048), 16 per thread."""
    tile = min(4096, _pow2_ceil(h))
    nt = -(-h // tile)
    br = max(1, min(16, 2048 // tile))
    return br, tile, nt, max(2, min(8, br * tile // 512))


def norm_check_triton(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line) -> None:
    u, spacing, fmax = dtype_consts(out.dtype)
    h = inp.shape[-1]
    x2, y2 = _rows2d(inp, h), _rows2d(out, h)
    m = x2.shape[0]
    if m == 0:
        return
    br, tile, nt, warps = kernel_config(h)
    _norm_check_kernel[(-(-m // br),)](x2, y2, w_ref, state, cap_meta, cap_in, cap_out, stepbuf,
                                       m, h, x2.stride(0), y2.stride(0), cap_in.shape[1], slot, float(eps),
                                       float(line), float(u), float(spacing), float(fmax),
                                       BR=br, TILE=tile, NT=nt, NCOLS=COLS, SSMAX=SS_MAX, num_warps=warps)


def _dev_key(device: torch.device) -> str:
    from ._group import _dev_key as k
    return k(device)


def _cpu_test() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1"


def kernel_usable(device: torch.device, dtype: torch.dtype) -> bool:
    if not HAVE_TRITON or os.environ.get("SDCGUARD_FAST", "1") == "0" or \
            os.environ.get("SDCGUARD_NORM_FAST", "1") == "0":
        return False
    if dtype not in (torch.bfloat16, torch.float16):
        return False
    return device.type == "cuda" or (device.type == "cpu" and _cpu_test())


def verify_kernel(device: torch.device, dtype: torch.dtype) -> tuple:
    """The Triton check against the PyTorch one before its first use on a device and dtype, on norms of a few
    shapes: correct (both quiet, the same counts), one output made wrong (both alarm on its row and column and keep
    the same row), and a row with a NaN input (both count it unjudged). (ok, why)."""
    key = (_dev_key(device), dtype)
    if key in STATUS:
        return STATUS[key]
    if not kernel_usable(device, dtype):
        STATUS[key] = (False, "the Triton kernel isn't available here")
        return STATUS[key]
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        for m, h in ((1, 64), (5, 1536), (33, 128), (3, 5000)):
            xs = (torch.randn(m, h, generator=g) * 3).to(dtype)
            w = (1 + 0.1 * torch.randn(h, generator=g)).to(dtype)
            xf = xs.float()
            y = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(dtype)
            bad_row, bad_col = m // 2, (7 * h) // 11
            yb = y.clone()
            yb[bad_row, bad_col] = yb[bad_row, bad_col].float() * 1.5 + 1
            xn = xs.clone()
            xn[0, 0] = float("nan")
            for case, (xx, yy) in enumerate(((xs, y), (xs, yb), (xn, y))):
                res = []
                for fn in (norm_check_torch, norm_check_triton):
                    st = torch.zeros(1, COLS, dtype=torch.int64, device=device)
                    cm = torch.zeros(1, CAP_COLS, dtype=torch.int32, device=device)
                    ci = torch.zeros(1, h, dtype=torch.float32, device=device)
                    co = torch.zeros_like(ci)
                    sb = torch.full((1,), 7, dtype=torch.int64, device=device)
                    fn(xx.to(device), yy.to(device), w.float().to(device), 1e-6, st, 0, sb, cm, ci, co, NORM_Z)
                    res.append((st.cpu(), cm.cpu(), ci.cpu(), co.cpu()))
                (s0, c0, i0, o0), (s1, c1, i1, o1) = res
                z0, z1 = _f32(int(s0[0, C_ROW]) >> 32), _f32(int(s1[0, C_ROW]) >> 32)
                if s0[0, C_CALLS:].tolist() != s1[0, C_CALLS:].tolist():
                    raise AssertionError(f"counts differ ({s0[0, C_CALLS:].tolist()} vs {s1[0, C_CALLS:].tolist()})")
                if case == 1:
                    for s in (s0, s1):
                        if int(s[0, C_ROW]) & M32 != bad_row or int(s[0, C_COL]) & M32 != bad_col:
                            raise AssertionError("a wrong output wasn't located")
                    if not (z0 > NORM_Z and z1 > NORM_Z) or abs(z0 - z1) > 1e-3 * max(1.0, z0):
                        raise AssertionError(f"scores of a wrong output differ ({z0} vs {z1})")
                    if int(c0[0, 0]) != 1 or int(c1[0, 0]) != 1 or int(c1[0, 1]) != bad_row or \
                            int(c1[0, 2]) != 7 or not torch.equal(i0[0], i1[0]) or not torch.equal(o0[0], o1[0]):
                        raise AssertionError("the kept row differs")
                elif not (z0 <= NORM_Z and z1 <= NORM_Z) or abs(z0 - z1) > 0.05:
                    raise AssertionError(f"scores of a correct norm differ or alarm ({z0} vs {z1})")
        STATUS[key] = (True, "")
    except Exception as e:  # noqa: BLE001
        STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
    return STATUS[key]


def norm_check_any(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line) -> None:
    """The Triton check where it passed its first-use comparison, else the PyTorch one."""
    OUT_DTYPES[(state.data_ptr(), slot)] = out.dtype     # (in Python: at capture time under CUDA graphs)
    with torch.no_grad(), torch.autocast(device_type=inp.device.type, enabled=False):
        key = (_dev_key(out.device), out.dtype)
        if STATUS.get(key, (False,))[0]:
            try:
                norm_check_triton(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line)
                return
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as e:  # noqa: BLE001  (compile/launch problem: the PyTorch check from now on)
                STATUS[key] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}")
                FALLBACKS[0] += 1
        norm_check_torch(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line)


_lib_lock = threading.Lock()
_OP = [None]


def norm_op():
    """sdcguard::norm_check, registered once: torch.compile sees one opaque call (nothing of it is merged with the
    norm it checks), CUDA graphs capture its kernel."""
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::norm_check",
                                         mutates_args=("state", "cap_meta", "cap_in", "cap_out"))
                def norm_check(inp: torch.Tensor, out: torch.Tensor, w_ref: torch.Tensor, eps: float,
                               state: torch.Tensor, slot: int, stepbuf: torch.Tensor, cap_meta: torch.Tensor,
                               cap_in: torch.Tensor, cap_out: torch.Tensor, line: float) -> None:
                    norm_check_any(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line)

                @norm_check.register_fake
                def _(inp, out, w_ref, eps, state, slot, stepbuf, cap_meta, cap_in, cap_out, line):
                    return None

                _OP[0] = torch.ops.sdcguard.norm_check
            except (AttributeError, RuntimeError):      # very old PyTorch, or already registered
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "norm_check", None) or norm_check_any
        return _OP[0]


def check_call(args: tuple, x: torch.Tensor, residual: Optional[torch.Tensor], result) -> None:
    """Check one call of a registered norm, from what its forward was called with and what it returned (the output,
    or (output, sum) with a residual): the opaque op under torch.compile, the check itself in eager mode."""
    if residual is None:
        inp, y = x, result
    elif isinstance(result, (tuple, list)) and len(result) == 2:
        y, inp = result
    else:
        return
    if not (isinstance(y, torch.Tensor) and isinstance(inp, torch.Tensor)) or y.shape[-1] != inp.shape[-1]:
        return
    if torch.compiler.is_compiling():
        # the op registered at finalize() (norm_op() itself takes a lock, which can't be traced)
        (_OP[0] if _OP[0] is not None else norm_op())(inp, y, *args)
    else:
        norm_check_any(inp, y, *args)


# ------------------------------------------------------------------------------------------------------ the recheck
def recheck(x_row: torch.Tensor, y_row: torch.Tensor, w_ref: torch.Tensor, w_now: Optional[torch.Tensor], eps: float,
            dtype: torch.dtype, line: float = NORM_Z) -> dict:
    """The kept row recomputed in float64 on the CPU, with the load-time weights and, when they differ, with the
    weights the layer holds now."""
    u, sp, fm = dtype_consts(dtype)
    x = x_row.double().cpu()
    y = y_row.double().cpu()

    def score(w):
        ref = x * torch.rsqrt(x.square().mean() + eps) * w.double().cpu()
        z = (y - ref).abs() / (u * ref.abs() + sp)
        ok = (ref.abs() <= fm) & (x.square().sum() < SS_MAX)
        z = torch.where(ok, torch.nan_to_num(z, nan=float("inf"), posinf=float("inf")), torch.zeros_like(z))
        return (float(z.max()) if z.numel() else 0.0), int((z > line).sum())

    worst, bad = score(w_ref)
    out = {"worst_score": worst, "outputs_over_line": bad, "outputs": int(x.numel())}
    changed = w_now is not None and not torch.equal(w_now.float().cpu(), w_ref.float().cpu())
    if bad and changed:
        wn, bn = score(w_now)
        if not bn:
            return {**out, "verdict": "weights_changed", "worst_score_current_weights": wn,
                    "why": "the kept output disagrees with the norm of its kept input under the weights registered "
                           "at load, and agrees with it under the weights the layer holds now: its weights changed in "
                           "memory (call refresh() after an intended update)"}
    if bad:
        return {**out, "verdict": "output_wrong",
                "why": f"{bad} of {x.numel()} outputs of the kept row are more than {line:g} roundoffs from the norm "
                       "recomputed in float64 from its kept input" +
                       (" (under the load-time weights and under the current ones)" if changed else "")}
    return {**out, "verdict": "output_agrees",
            "why": "the float64 recomputation of the kept row agrees with its output: the GPU's check disagreed, or "
                   "the kept input and output aren't from the same call"}


# -------------------------------------------------------------------------------------------------- the norm layers
KINDS = {"RMSNorm": "rmsnorm", "GemmaRMSNorm": "gemma_rmsnorm"}


def kind_of(mod) -> Optional[str]:
    """"rmsnorm" or "gemma_rmsnorm" for vLLM's two RMSNorm classes and subclasses that keep their forward, else
    None."""
    for cls in type(mod).__mro__:
        if cls.__name__ in KINDS and cls.__module__.endswith("layernorm"):
            own = [c for c in type(mod).__mro__[:type(mod).__mro__.index(cls)] if "forward_native" in vars(c)]
            return None if own else KINDS[cls.__name__]
    return None


def load_time_weight(mod, kind: str) -> Optional[torch.Tensor]:
    """The weights the norm multiplies by, in fp32 (GemmaRMSNorm: 1 + w; a norm without a weight: ones)."""
    w = getattr(mod, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 1:
        return None
    w = w.detach()
    if kind == "gemma_rmsnorm":
        return w.float() + 1.0
    if not getattr(mod, "has_weight", True):
        return torch.ones(w.shape, dtype=torch.float32, device=w.device)
    return w.float().clone()


class NormChecks:
    """The norm layers one ServeGuard also checks (ServeGuard.norm_checks()). add() each layer, then finalize() once,
    before the model is compiled or its CUDA graphs captured: it allocates the buffers the checks write (fixed
    addresses) and gives each layer its arguments (layer._sdcguard_norm)."""

    def __init__(self, guard, line: float = NORM_Z):
        line = float(line)
        if not MIN_LINE <= line < float("inf"):
            raise ValueError(f"the norm check's line must be at least {MIN_LINE} roundoffs (a correct norm can be "
                             f"4.03 off), not {line!r}")
        self.guard, self.line = guard, line
        self.layers: list = []                 # slot -> {"name", "module", "kind", "device", "w_ref", "eps", "h"}
        self.unsupported: dict = {}            # name -> why it isn't checked
        self._state: dict = {}                 # device -> (slots, COLS) int64
        self._cap: dict = {}                   # device -> (cap_meta int32 (slots, 4), cap_in, cap_out fp32 (slots, w))
        self._step: dict = {}                  # device -> int64 [1]: the marked step
        self._host: dict = {}                  # device -> (pinned state, pinned cap_meta, CUDA event, window)
        self._marked: Optional[int] = None
        self._read_from: dict = {}             # device -> first step marked since its last read
        self._last_clean: dict = {}
        self.worst_seen: dict = {}             # norm name -> worst score read back so far
        self.checks_seen: dict = {}            # norm name -> calls checked, read back so far
        self.finalized = False
        self.stats = {"norms": 0, "calls_checked": 0, "rows_checked": 0, "unjudged_outputs": 0, "alerts": 0,
                      "reads": 0}

    # ------------------------------------------------------------------------------------------------- setup
    def why_not(self, mod, kind: Optional[str]) -> Optional[str]:
        if kind is None:
            return f"{type(mod).__name__}: not an RMSNorm the check knows"
        if getattr(mod, "variance_size_override", None) is not None:
            return "its variance is taken over part of the row (variance-size override)"
        w = getattr(mod, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 1:
            return "no weight vector"
        if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return f"weight dtype {w.dtype}"
        if not (w.is_cuda or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
            return "not on a GPU"
        if getattr(mod, "_sdcguard_norm", None) is not None:
            return "already checked by another ServeGuard"
        return None

    def add(self, mod, name: str, kind: Optional[str] = None) -> Optional[int]:
        if self.finalized:
            raise RuntimeError("norm layers are added before finalize()")
        kind = kind or kind_of(mod)
        why = self.why_not(mod, kind)
        if why:
            self.unsupported[name] = why
            return None
        with torch.inference_mode(False):            # ordinary tensors (vLLM loads and runs in inference mode)
            w_ref = load_time_weight(mod, kind)
        self.layers.append({"name": name, "module": mod, "kind": kind, "device": w_ref.device, "w_ref": w_ref,
                            "eps": float(getattr(mod, "variance_epsilon", 1e-6)), "h": int(w_ref.shape[0])})
        self.stats["norms"] += 1
        return len(self.layers) - 1

    def finalize(self, dtypes=(torch.bfloat16, torch.float16)) -> int:
        """Allocate the buffers, compare the kernel with the PyTorch check on each device for the activation dtypes
        given, and give each layer its arguments. Returns the layers checked."""
        if self.finalized:
            return len(self.layers)
        devs: dict = {}
        for lay in self.layers:
            devs.setdefault(lay["device"], []).append(lay)
        n = len(self.layers)
        for dev, lays in devs.items():
            width = max(lay["h"] for lay in lays)
            with torch.inference_mode(False):
                self._state[dev] = torch.zeros(n, COLS, dtype=torch.int64, device=dev)
                self._cap[dev] = (torch.zeros(n, CAP_COLS, dtype=torch.int32, device=dev),
                                  torch.zeros(n, width, dtype=torch.float32, device=dev),
                                  torch.zeros(n, width, dtype=torch.float32, device=dev))
                self._step[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
                for dt in dtypes:
                    verify_kernel(dev, dt)
        norm_op()                              # registered before any graph is traced
        for slot, lay in enumerate(self.layers):
            lay["module"]._sdcguard_norm = self.args(slot)
        self.finalized = True
        return n

    def args(self, slot: int) -> tuple:
        """What the check takes after its input and output, for norm `slot` (fixed for the life of the buffers)."""
        lay = self.layers[slot]
        dev = lay["device"]
        cm, ci, co = self._cap[dev]
        return (lay["w_ref"], lay["eps"], self._state[dev], slot, self._step[dev], cm, ci, co, self.line)

    def kernel_status(self) -> dict:
        """(device, dtype) -> (Triton kernel in use, why not) for the devices these norms are on."""
        devs = {_dev_key(lay["device"]) for lay in self.layers}
        return {f"{d} {str(dt).replace('torch.', '')}": v for (d, dt), v in STATUS.items() if d in devs}

    def refresh(self) -> int:
        """Take each layer's current weights as the ones to compare with (after an intended update; queued on the GPU
        like the model's work, into the same memory, so captured CUDA graphs keep working)."""
        n = 0
        with torch.inference_mode(False):
            for lay in self.layers:
                w = load_time_weight(lay["module"], lay["kind"])
                if w is not None and w.shape == lay["w_ref"].shape:
                    lay["w_ref"].copy_(w)
                    n += 1
        return n

    def detach(self) -> None:
        try:
            if self.layers and self.finalized:
                self.poll(wait=True)
        finally:
            for lay in self.layers:
                lay["module"].__dict__.pop("_sdcguard_norm", None)
            self.layers.clear()
            for d in (self._state, self._cap, self._step, self._host, self._read_from):
                d.clear()
            self.finalized = False

    # ------------------------------------------------------------------------------------------------- steps
    def mark_step(self, step: int) -> None:
        """The model step the work queued next belongs to (outside any CUDA graph, like ServeGuard.mark_step)."""
        step = int(step) & STEP_MASK
        self._marked = step
        for dev, sb in self._step.items():
            sb.fill_(step)
            self._read_from.setdefault(dev, step)

    def _window(self, dev) -> Optional[list]:
        if self._marked is None:
            return None
        a = self._read_from.pop(dev, None)
        return [a if a is not None else self._marked, self._marked]

    # ------------------------------------------------------------------------------------------------- reads
    def poll(self, wait: bool = False) -> list:
        """Read the results back. Without wait it never waits for the GPU: it starts a copy and reports the previous
        one once it has arrived."""
        events = []
        for dev, st in self._state.items():
            cm = self._cap[dev][0]
            if dev.type != "cuda" or wait:
                pend = self._host.pop(dev, None)
                if pend is not None:
                    pend[2].synchronize()
                    events += self._report(dev, pend[0], pend[1], pend[3])
                host, hcap = st.to("cpu", copy=True), cm.to("cpu", copy=True)
                st.zero_()
                events += self._report(dev, host, hcap, self._window(dev))
                continue
            pend = self._host.get(dev)
            if pend is not None and pend[2].query():
                events += self._report(dev, pend[0], pend[1], pend[3])
                pend = None
                self._host.pop(dev, None)
            if pend is None:
                hs = torch.empty(st.shape, dtype=st.dtype, pin_memory=True)
                hc = torch.empty(cm.shape, dtype=cm.dtype, pin_memory=True)
                hs.copy_(st, non_blocking=True)
                hc.copy_(cm, non_blocking=True)
                st.zero_()                     # same stream: after the copy has read it
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (hs, hc, ev, self._window(dev))
        return events

    def counts(self) -> dict:
        """Calls checked per norm layer: read back so far plus what hasn't been read (waits for the GPU)."""
        live: dict = {}
        for dev, st in self._state.items():
            rows = st.to("cpu")
            pend = self._host.get(dev)
            if pend is not None:
                pend[2].synchronize()
                rows = rows + pend[0]
            for slot, lay in enumerate(self.layers):
                if lay["device"] == dev:
                    live[lay["name"]] = int(rows[slot, C_CALLS])
        return {lay["name"]: self.checks_seen.get(lay["name"], 0) + live.get(lay["name"], 0) for lay in self.layers}

    def coverage(self) -> dict:
        c = self.counts()
        return {"norms": len(self.layers), "norms_checked": sum(v > 0 for v in c.values()),
                "never_checked": sorted(n for n, v in c.items() if v == 0), "calls_checked": sum(c.values()),
                "kernel": self.kernel_status(), "not_attached": dict(self.unsupported)}

    def _report(self, dev, host: torch.Tensor, hcap: torch.Tensor, window) -> list:
        g = self.guard
        out = []
        worst = 0.0
        self.stats["reads"] += 1
        for slot, lay in enumerate(self.layers):
            if lay["device"] != dev or slot >= host.shape[0]:
                continue
            keys = [int(v) for v in host[slot].tolist()]
            calls = keys[C_CALLS]
            self.stats["calls_checked"] += calls
            self.stats["rows_checked"] += keys[C_ROWS]
            self.stats["unjudged_outputs"] += keys[C_UNJ]
            self.checks_seen[lay["name"]] = self.checks_seen.get(lay["name"], 0) + calls
            z = _f32(keys[C_ROW] >> 32)
            worst = max(worst, z)
            self.worst_seen[lay["name"]] = max(self.worst_seen.get(lay["name"], 0.0), z)
            if not (z > self.line):
                if calls and window is not None:
                    self._last_clean[lay["name"]] = list(window)
                continue
            out.append(self._alert(dev, slot, keys, [int(v) for v in hcap[slot].tolist()], window, z))
        if g is not None and getattr(g, "metrics", None) is not None and hasattr(g.metrics, "observe_norms"):
            try:
                g.metrics.observe_norms(self, out, worst)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
        return out

    def _alert(self, dev, slot: int, keys: list, cap: list, window, z: float) -> SDCEvent:
        g = self.guard
        lay = self.layers[slot]
        mod = lay["module"]
        dtype = OUT_DTYPES.get((self._state[dev].data_ptr(), slot)) or \
            (mod.weight.dtype if mod.weight.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16)
        row, col = keys[C_ROW] & M32, keys[C_COL] & M32
        yv, rv = _f32(keys[C_OUT]), _f32(keys[C_REF])
        captured, rc = None, None
        w_now = None
        try:
            w_now = load_time_weight(mod, lay["kind"])
        except Exception:  # noqa: BLE001  (diagnosis is best effort)
            pass
        if cap[0]:
            # the kept row: read it now (waits for the GPU: alerts only), then free the layer's capture
            cm, ci, co = self._cap[dev]
            h = lay["h"]
            x_row, y_row = ci[slot, :h].to("cpu"), co[slot, :h].to("cpu")
            cm[slot].zero_()
            captured = {"row": cap[1] & M32, "step": (cap[2] & M32) if self._marked is not None else None,
                        "score": _f32(cap[3])}
            rc = recheck(x_row, y_row, lay["w_ref"], w_now, lay["eps"], dtype, self.line)
        changed = None
        if w_now is not None:
            changed = not torch.equal(w_now.cpu(), lay["w_ref"].cpu())
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        key = str(dev)
        if key not in ident:
            ident[key] = device_identity(dev)
        detail = {"where": lay["name"], "kind": lay["kind"], "max_z": z, "row": row, "col": col,
                  "output": yv, "expected": rv,
                  "relative_error": (abs(yv - rv) / abs(rv)) if rv and rv == rv else None,
                  "captured": captured,
                  "recheck": rc or {"verdict": "not_rechecked",
                                    "why": "no row was kept for this alert (the layer's kept row was taken by an "
                                           "earlier alert that hasn't been read yet)"},
                  "weights_changed_since_load": changed,
                  "read_window_steps": window, "last_clean_read_steps": self._last_clean.get(lay["name"]),
                  "check_config": {"line": self.line, "unit_roundoff": dtype_consts(dtype)[0],
                                   "reference": "fp32 recompute from the norm's input and its load-time weights"},
                  "reading": "a norm output differs from the norm recomputed from its input and its load-time "
                             "weights by more than a correct implementation's rounding allows"}
        ev = SDCEvent(detector="serve_norm", step=None, detail=detail, identity=dict(ident[key]))
        self.stats["alerts"] += 1
        if g is not None:
            if getattr(g, "attribute", None) is not None:
                try:
                    g.attribute(ev)
                except Exception:  # noqa: BLE001  (attribution only)
                    g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
            g.log.add(ev)
            if getattr(g, "on_alert", None):
                try:
                    g.on_alert(ev)
                except Exception:  # noqa: BLE001
                    pass
        return ev
