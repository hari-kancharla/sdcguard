"""Checks of FP8 quantization in serving: were a layer's FP8 inputs and their scale made from this step's own rows,
by the recipe the layer runs?

vLLM's FP8 layers quantize their bf16/fp16 input to FP8 E4M3 right before the GEMM, with a scale that is either
fixed (static) or computed from the input (dynamic): one per row (per-token: max |x| of the row / 448, at least
1/(448*512)) or one for the whole input (per-tensor: max |x| over every row / 448). The input of a step is padded
to a CUDA graph's batch size, and the padding rows hold whatever an earlier step left there. A per-tensor scale
taken over them then depends on earlier requests: a request's answer changes with the request before it (vLLM issue
#62124, FP8 per-tensor MoE with CUDA graphs). The GEMM's own arithmetic can be right on such operands, so a
checksum of the GEMM doesn't see this; nor does a comparison with a bf16 stack, which differs anyway.

At each call this check recomputes, from the layer's input over this step's real rows only (the plugin publishes
their number on the GPU before each step and clears it after, so warm-up and graph-capture runs aren't judged):
  - the scale: per-tensor dynamic, max |x| over the real rows / 448 against the scale the GEMM got (a scale that
    matches the maximum over all rows, padding included, is reported as taken from rows outside the batch); per-token
    dynamic, the scale of SAMPLES real rows (each row's own, so padding can't enter);
  - the FP8 values of SAMPLES real rows, quantized with the scale the GEMM got, as vLLM's quantization kernels do
    (x / scale for per-token, x * (1 / scale) for per-tensor and static, clamped to +-448, rounded to nearest even),
    against the FP8 values the GEMM got. vLLM builds those kernels with fast math, so a value one FP8 step off is
    counted (fp8_one_step) but isn't an alarm; two or more steps off is.
Static scales are weights: the check compares them with their value at load (a changed one is reported).

What it doesn't see: the GEMM's arithmetic (that is a separate check); quantization recipes other than these three
(block or group scales: their layers are listed as not checked); rows not sampled, for the values.
"""
from __future__ import annotations

import os
import struct
import threading
from typing import Optional

import torch

from .events import SDCEvent, device_identity
from .serve_norm import pick

FP8_MAX = 448.0
MIN_SCALE = 1.0 / (448.0 * 512.0)
SAMPLES = 2
SCALE_TOL = 2.0 ** -8          # relative: one bf16 rounding (a compiled graph may quantize the norm's fp32 output
                               # without rounding it to bf16 first, moving the maximum by up to that)
RECIPES = {"static": 0, "dynamic_tensor": 1, "dynamic_token": 2}
ROW_A, ROW_B = 7919, 104729
M32 = 0xFFFFFFFF
COLS = 8
# calls judged, calls with a wrong scale, of them taken from outside rows, sampled rows with a value 2+ steps off,
# values one step off, last bad event (step << 32 | kind << 24 | row), worst scale ratio bits, static scale changed
C_CALLS, C_SCALE, C_OUTSIDE, C_VALUES, C_ONE, C_LAST, C_RATIO, C_STATIC = range(COLS)
KIND_SCALE, KIND_OUTSIDE, KIND_VALUES, KIND_STATIC = 1, 2, 3, 4


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & M32))[0]


def _bits(v: torch.Tensor) -> torch.Tensor:
    return v.float().reshape(1).view(torch.int32).to(torch.int64).reshape(()) & M32


def fp8_order(codes: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 codes (uint8) as integers in the order of the values they encode (so a difference of 1 is one
    representable step; +0 and -0 are both 0)."""
    c = codes.to(torch.int32)
    mag = c & 0x7F
    return torch.where((c & 0x80) != 0, -mag, mag)


def quantize(x_rows: torch.Tensor, scale: torch.Tensor, divide: bool) -> torch.Tensor:
    """vLLM's FP8 quantization of some rows with their scale ([rows, 1] or a scalar), as uint8 codes."""
    xf = x_rows.float()
    s = scale.float()
    v = xf / s if divide else xf * (1.0 / s)
    v = v.clamp(-FP8_MAX, FP8_MAX)
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def sample_rows(step: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    j = torch.arange(SAMPLES, device=real.device, dtype=torch.int64)
    return (step * ROW_A + j * ROW_B) % real.clamp(min=1)


def quant_check_torch(x2: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, recipe: int, scale_ub: Optional[torch.Tensor],
                      static_ref: Optional[torch.Tensor], rows_buf: torch.Tensor, stepbuf: torch.Tensor,
                      state: torch.Tensor, slot: int, mult: int = 1) -> None:
    """The check in PyTorch operations, without a host sync (safe inside CUDA graphs). x2: [M, K] the layer's input;
    xq: [M', K] (M' >= M) the FP8 input the GEMM got; xs: its scale(s). rows_buf[0]: this step's real rows (0 outside
    model steps: nothing is judged); mult: input rows per token (an MoE layer's second quantization has top_k)."""
    m = x2.shape[0]
    if m == 0:
        return
    real = (rows_buf.reshape(-1)[0] * mult).clamp(min=0, max=m)
    armed = real > 0
    step = stepbuf.reshape(-1)[0] & M32
    rows = sample_rows(step, real)
    xr = x2[rows]
    xa = x2.abs().float()
    zero = torch.zeros((), dtype=torch.int64, device=x2.device)
    one = torch.ones((), dtype=torch.int64, device=x2.device)
    scale_bad, outside, static_bad = torch.zeros((), dtype=torch.bool, device=x2.device), None, None
    ratio = torch.ones((), device=x2.device)
    if recipe == RECIPES["dynamic_tensor"]:
        ks = xs.reshape(-1)[0].float()
        in_batch = torch.arange(m, device=x2.device)[:, None] < real
        ref = torch.where(in_batch, xa, torch.zeros_like(xa)).amax() / FP8_MAX
        everything = xa.amax() / FP8_MAX
        scale_bad = (ks - ref).abs() > SCALE_TOL * ref.abs() + 1e-38
        outside = scale_bad & ((ks - everything).abs() <= SCALE_TOL * everything.abs() + 1e-38) & (everything > ref)
        ratio = ks / ref.clamp(min=1e-38)
        use = ks
        divide = False
    elif recipe == RECIPES["dynamic_token"]:
        amax = xa[rows].amax(-1, keepdim=True)
        if scale_ub is not None:
            amax = torch.minimum(amax, scale_ub.float().reshape(1, 1))
        ref = (amax / FP8_MAX).clamp(min=MIN_SCALE)
        ks = xs.reshape(-1)[rows].float().reshape(-1, 1)
        diff = (ks - ref).abs() > SCALE_TOL * ref.abs() + 1e-38
        scale_bad = diff.any()
        ratio = pick((ks / ref.clamp(min=1e-38)).reshape(-1), diff.reshape(-1).to(torch.int64).argmax())
        use = ks
        divide = True
    else:
        use = xs.reshape(-1)[0].float()
        divide = False
        if static_ref is not None:
            static_bad = (use != static_ref.float().reshape(-1)[0])
    q_ref = quantize(xr, use, divide)
    q_k = xq.view(torch.uint8)[rows]                 # indexed as bytes: PyTorch can't index FP8 on every backend
    d = (fp8_order(q_ref) - fp8_order(q_k)).abs()
    nbad = (d >= 2).sum(-1)                                   # per sampled row
    vbad_row = nbad > 0
    n_one = (d == 1).sum()
    srow = state[slot]
    a64 = armed.to(torch.int64)
    srow[C_CALLS] += a64
    sb = (scale_bad & armed)
    srow[C_SCALE] += sb.to(torch.int64)
    if outside is not None:
        srow[C_OUTSIDE] += (outside & armed).to(torch.int64)
    vb = vbad_row & armed
    srow[C_VALUES] += vb.sum()
    srow[C_ONE] += n_one * a64
    if static_bad is not None:
        srow[C_STATIC] = torch.maximum(srow[C_STATIC], (static_bad & armed).to(torch.int64))
    kind = torch.where(outside & armed if outside is not None else torch.zeros_like(armed), KIND_OUTSIDE * one,
                       torch.where(sb, KIND_SCALE * one,
                                   torch.where(vb.any(), KIND_VALUES * one,
                                               torch.where(static_bad & armed if static_bad is not None else
                                                           torch.zeros_like(armed), KIND_STATIC * one, zero))))
    bad_row = pick(rows, vb.to(torch.int64).argmax()) if rows.numel() else zero
    rec = (step << 32) | (kind << 24) | (bad_row & 0xFFFFFF)
    srow[C_LAST] = torch.where(kind > 0, torch.maximum(srow[C_LAST], rec), srow[C_LAST])
    rb = _bits(ratio.abs().clamp(max=3e38)) if ratio.numel() == 1 else zero
    srow[C_RATIO] = torch.where(sb, torch.maximum(srow[C_RATIO], rb), srow[C_RATIO])


_lib_lock = threading.Lock()
_OP = [None]


def quant_op():
    """sdcguard::fp8_quant_check, registered once: one opaque op under torch.compile, so the check's operations
    aren't traced into vLLM's graph."""
    with _lib_lock:
        if _OP[0] is None:
            try:
                @torch.library.custom_op("sdcguard::fp8_quant_check", mutates_args=("state",))
                def fp8_quant_check(x2: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, recipe: int,
                                    scale_ub: Optional[torch.Tensor], static_ref: Optional[torch.Tensor],
                                    rows_buf: torch.Tensor, stepbuf: torch.Tensor, state: torch.Tensor,
                                    slot: int, mult: int) -> None:
                    with torch.no_grad():
                        quant_check_torch(x2, xq, xs, recipe, scale_ub, static_ref, rows_buf, stepbuf, state, slot,
                                          mult)

                @fp8_quant_check.register_fake
                def _(*a):
                    return None

                _OP[0] = torch.ops.sdcguard.fp8_quant_check
            except (AttributeError, RuntimeError):
                _OP[0] = getattr(getattr(torch.ops, "sdcguard", None), "fp8_quant_check", None) or quant_check_torch
        return _OP[0]


def check_call(args: tuple, x2: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor,
               scale_ub: Optional[torch.Tensor] = None, mult: int = 1) -> None:
    """One quantization of a registered layer: its input, the FP8 input the GEMM got and its scale."""
    recipe, static_ref, rows_buf, stepbuf, state, slot = args
    if not (isinstance(xq, torch.Tensor) and isinstance(xs, torch.Tensor)) or xq.dim() != 2 or x2.dim() != 2 or \
            xq.shape[1] != x2.shape[1] or xq.shape[0] < x2.shape[0]:
        return
    if torch.compiler.is_compiling():
        (_OP[0] if _OP[0] is not None else quant_op())(x2, xq, xs, recipe, scale_ub, static_ref, rows_buf, stepbuf,
                                                       state, slot, int(mult))
    else:
        with torch.no_grad():
            quant_check_torch(x2, xq, xs, recipe, scale_ub, static_ref, rows_buf, stepbuf, state, slot, int(mult))


def _per_tensor(gs) -> bool:
    """Whether a QuantFP8's group shape is per tensor: vLLM 0.32's GroupShape is a NamedTuple (row, col), per tensor
    when both are -1 (GroupShape.is_per_tensor); an older or stand-in one may only have a name."""
    f = getattr(gs, "is_per_tensor", None)
    if callable(f):
        try:
            return bool(f())
        except Exception:  # noqa: BLE001
            return False
    if isinstance(gs, tuple) and len(gs) == 2:
        return tuple(gs) == (-1, -1)
    return "TENSOR" in str(getattr(gs, "name", gs) or "").upper()


def recipe_of(quant_fp8) -> tuple:
    """(recipe, why not): vLLM's QuantFP8 of a layer's kernel."""
    if quant_fp8 is None:
        return None, "no FP8 input quantization found"
    if getattr(quant_fp8, "is_group_quant", False):
        return None, "FP8 with group (block) activation scales"
    gs = getattr(quant_fp8, "group_shape", None)
    if getattr(quant_fp8, "static", False):
        if not _per_tensor(gs):
            return None, f"static FP8 activation scales per {getattr(gs, 'name', gs) or 'group'}"
        return RECIPES["static"], None
    if getattr(quant_fp8, "use_per_token_if_dynamic", False):
        return RECIPES["dynamic_token"], None
    return RECIPES["dynamic_tensor"], None


class Fp8Checks:
    """The FP8 layers one ServeGuard checks the quantization of (ServeGuard.fp8_checks()): add() each (at weight
    processing), then mark_step(step, rows) before each model step and end_step() after it; poll() with the guard's
    reads."""

    def __init__(self, guard):
        self.guard = guard
        self.layers: list = []
        self.unsupported: dict = {}
        self._state, self._rows, self._step, self._host = {}, {}, {}, {}
        self.stats = {"layers": 0, "calls": 0, "scale_wrong": 0, "scale_from_outside_rows": 0, "values_wrong": 0,
                      "fp8_one_step": 0, "alerts": 0}
        self.seen: dict = {}
        self._lock = threading.Lock()

    def _dev(self, dev):
        if dev not in self._state:
            with torch.inference_mode(False):
                self._state[dev] = torch.zeros(64, COLS, dtype=torch.int64, device=dev)
                self._rows[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
                self._step[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
        return self._state[dev]

    def add(self, layer, name: str, quant_fp8, static_scale: Optional[torch.Tensor], device) -> Optional[tuple]:
        recipe, why = recipe_of(quant_fp8)
        if why:
            self.unsupported[name] = why
            return None
        if not (device.type == "cuda" or os.environ.get("SDCGUARD_ALLOW_CPU") == "1"):
            self.unsupported[name] = "not on a GPU"
            return None
        with self._lock:
            st = self._dev(device)
            slot = sum(1 for lay in self.layers if lay["device"] == device)
            if slot >= st.shape[0]:
                with torch.inference_mode(False):
                    grown = torch.zeros(st.shape[0] * 2, COLS, dtype=torch.int64, device=device)
                    grown[:st.shape[0]] = st
                self._state[device] = st = grown
                for lay in self.layers:               # their args hold the old state: point them at the new one
                    if lay["device"] == device:
                        lay["layer"]._sdcguard_fp8 = self.args(lay)
            ref = None
            if recipe == RECIPES["static"] and isinstance(static_scale, torch.Tensor):
                with torch.inference_mode(False):
                    ref = static_scale.detach().float().reshape(-1)[:1].clone()
            lay = {"name": name, "layer": layer, "device": device, "slot": slot, "recipe": recipe, "static_ref": ref}
            self.layers.append(lay)
            self.stats["layers"] += 1
            quant_op()
            a = self.args(lay)
            layer._sdcguard_fp8 = a
            return a

    def args(self, lay) -> tuple:
        dev = lay["device"]
        return (lay["recipe"], lay["static_ref"], self._rows[dev], self._step[dev], self._state[dev], lay["slot"])

    def mark_step(self, step: int, rows: Optional[int] = None) -> None:
        for dev in self._state:
            self._step[dev].fill_(int(step) & M32)
            self._rows[dev].fill_(int(rows) if rows is not None else 0)

    def end_step(self) -> None:
        for dev in self._state:
            self._rows[dev].zero_()

    def detach(self) -> None:
        try:
            self.poll(wait=True)
        finally:
            for lay in self.layers:
                lay["layer"].__dict__.pop("_sdcguard_fp8", None)
                lay["layer"].__dict__.pop("_sdcguard_fp8_moe", None)     # (set by the vLLM plugin on MoE layers)
            self.layers.clear()
            for d in (self._state, self._rows, self._step, self._host):
                d.clear()

    def coverage(self) -> dict:
        return {"layers": len(self.layers), "calls": self.stats["calls"],
                "layers_checked": sum(self.seen.get(lay["name"], 0) > 0 for lay in self.layers),
                "never_checked": sorted(lay["name"] for lay in self.layers if not self.seen.get(lay["name"])),
                "not_attached": dict(self.unsupported), "samples_per_call": SAMPLES,
                "recipes": sorted({k for lay in self.layers for k, v in RECIPES.items() if v == lay["recipe"]})}

    def poll(self, wait: bool = False) -> list:
        events = []
        for dev, st in list(self._state.items()):
            if dev.type != "cuda" or wait:
                pend = self._host.pop(dev, None)
                if pend is not None:
                    pend[1].synchronize()
                    events += self._report(dev, pend[0])
                host = st.to("cpu", copy=True)
                st.zero_()
                events += self._report(dev, host)
                continue
            pend = self._host.get(dev)
            if pend is not None and pend[1].query():
                events += self._report(dev, pend[0])
                self._host.pop(dev, None)
                pend = None
            if pend is None:
                hs = torch.empty(st.shape, dtype=st.dtype, pin_memory=True)
                hs.copy_(st, non_blocking=True)
                st.zero_()
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(dev))
                self._host[dev] = (hs, ev)
        return events

    def _report(self, dev, host) -> list:
        out = []
        for lay in self.layers:
            if lay["device"] != dev or lay["slot"] >= host.shape[0]:
                continue
            k = [int(v) for v in host[lay["slot"]].tolist()]
            self.stats["calls"] += k[C_CALLS]
            self.seen[lay["name"]] = self.seen.get(lay["name"], 0) + k[C_CALLS]
            self.stats["scale_wrong"] += k[C_SCALE]
            self.stats["scale_from_outside_rows"] += k[C_OUTSIDE]
            self.stats["values_wrong"] += k[C_VALUES]
            self.stats["fp8_one_step"] += k[C_ONE]
            if k[C_SCALE] or k[C_VALUES] or k[C_STATIC]:
                out.append(self._alert(dev, lay, k))
        g = self.guard
        if g is not None and getattr(g, "metrics", None) is not None and hasattr(g.metrics, "observe_fp8"):
            try:
                g.metrics.observe_fp8(self, out)
            except Exception:  # noqa: BLE001  (metrics never break serving)
                pass
        return out

    def _alert(self, dev, lay, k) -> SDCEvent:
        g = self.guard
        last = k[C_LAST]
        step, kind, row = (last >> 32) & M32, (last >> 24) & 0xFF, last & 0xFFFFFF
        what = {KIND_OUTSIDE: "scale_from_outside_rows", KIND_SCALE: "scale_wrong", KIND_VALUES: "values_wrong",
                KIND_STATIC: "static_scale_changed"}.get(kind, "unknown")
        reading = {
            "scale_from_outside_rows": "the per-tensor FP8 scale the GEMM used is the maximum over every row of its "
                                       "input, padding included, not over this step's real rows: the output depends "
                                       "on data that isn't this batch's (rows an earlier request left)",
            "scale_wrong": "the FP8 scale the GEMM used differs from the scale its recipe gives for this step's rows",
            "values_wrong": "FP8 values the GEMM got are two or more FP8 steps from the layer's input quantized with "
                            "that scale",
            "static_scale_changed": "the layer's static FP8 input scale changed since load",
        }.get(what, "")
        ident = g._identity if g is not None and hasattr(g, "_identity") else {}
        if str(dev) not in ident:
            ident[str(dev)] = device_identity(dev)
        detail = {"where": lay["name"], "kind": "fp8_quantization", "finding": what, "calls_judged": k[C_CALLS],
                  "calls_with_wrong_scale": k[C_SCALE], "calls_scale_from_outside_rows": k[C_OUTSIDE],
                  "sampled_rows_with_wrong_values": k[C_VALUES], "values_one_fp8_step_off": k[C_ONE],
                  "scale_ratio": _f32(k[C_RATIO]) if k[C_RATIO] else None,
                  "captured": {"step": step, "row": row} if kind in (KIND_VALUES, KIND_SCALE, KIND_OUTSIDE) else None,
                  "recipe": next(n for n, v in RECIPES.items() if v == lay["recipe"]),
                  "check_config": {"samples_per_call": SAMPLES, "scale_tolerance": SCALE_TOL,
                                   "values": "2 or more FP8 steps off is an alarm; 1 step is counted (fast math)"},
                  "reading": reading}
        ev = SDCEvent(detector="serve_fp8", step=None, detail=detail, identity=dict(ident[str(dev)]))
        self.stats["alerts"] += 1
        if g is not None:
            if getattr(g, "attribute", None) is not None:
                try:
                    g.attribute(ev)
                except Exception:  # noqa: BLE001
                    g.stats["attribution_errors"] = g.stats.get("attribution_errors", 0) + 1
            g.log.add(ev)
            if getattr(g, "on_alert", None):
                try:
                    g.on_alert(ev)
                except Exception:  # noqa: BLE001
                    pass
        return ev
