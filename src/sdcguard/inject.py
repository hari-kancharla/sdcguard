"""Fault injection that mimics published GPU silent-data-corruption patterns.

Patterns follow NVIDIA's study "The Anatomy of Silent Data Corruption" (arXiv
2605.04213, May 2026), a gate-level stuck-at fault-injection study of a synthesized
data-center GPU model (simulation, not field data):
  * ~51% of corrupted values are zeroed ("nullified")
  * ~48% are other wrong-but-valid values (bit flips, often several bits per value;
    flip rates drop from the lowest to the highest bit, exponent bits are hit least)
  * ~1% become NaN or Inf
  * corrupted positions repeat periodically, aligned to the 32-thread warp; modelled here
    as chunks of up to 32 values with a period of 1/2/4/8/16
  * control-logic faults often corrupt 20-75% of a GPU core's (SM's) output; modelled
    here as part of a 128x128 output tile
ByteDance ("SDCs in the Wild", OSDI '26) adds: faults are persistent and can get
worse over time, and some only trigger for certain value ranges.

This is a software model of a faulty GPU. For instruction-level injection on real
NVIDIA GPUs, see TU Berlin's NVBit harness (github.com/aaltenbernd/llm-sdc-training).

Usage (order matters: the injector is the "hardware", the guard sits on top):
    inj = FaultInjector(rate=0.05)
    inj.arm_on([blk.attn.q for blk in model.blocks] + [blk.attn.k for blk in model.blocks])
    with inj:            # faulty hardware
        with guard:      # detector, entered last so it sees the corrupted output
            loss = model(x); loss.backward()
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten
_INT_VIEW = {2: torch.int16, 4: torch.int32}
_MANTISSA_BITS = {torch.bfloat16: 7, torch.float16: 10, torch.float32: 23}


def _op_name(func) -> str:
    try:
        return func._schema.name
    except AttributeError:
        return str(func)


# Forward matmul ops whose outputs a faulty GPU would corrupt (Apple GPUs use _mps_linear).
TARGET_OPS = {"aten::mm", "aten::bmm", "aten::addmm", "aten::_mps_linear", "aten::linear"}


@dataclass
class FaultModel:
    p_zero: float = 0.5068
    p_bitflip: float = 0.4831
    p_special: float = 0.0101           # NaN / Inf
    chunk: int = 32                     # warp width
    periods: tuple = (1, 2, 4, 8, 16)   # 1 = every value in the chunk
    bit_decay: float = 0.8              # flip weight of bit i ~ bit_decay**i (low bits more likely)
    bits_per_flip: tuple = ((1, 0.4), (2, 0.3), (3, 0.2), (4, 0.1))
    mantissa_only: bool = False         # only flip mantissa bits: values stay plausible
    # "Tile" events model control-logic faults, which NVIDIA found often corrupt 20-75% of an
    # SM's output, mostly by zeroing. p_tile = share of events that hit a whole output tile.
    p_tile: float = 0.0
    tile: tuple = (128, 128)
    tile_frac: tuple = (0.2, 0.75)
    tile_p_zero: float = 0.9
    ref_block: int = 128                # size of the reference block for "how big was this fault"


def fault_model(mix: str = "nvidia", scale: str = "warp", specials: bool = False) -> FaultModel:
    """Named fault models used in the experiments.

    mix:   nvidia   NVIDIA's mix (zeroed / bit flips / NaN-Inf), NaN-Inf only if specials=True
           zero     only zeroing (the most common corruption, 51%)
           mantissa only bit flips that stay in the mantissa (values stay plausible)
    scale: warp (a few values per event), tile (20-75% of a 128x128 output tile), mixed (30% tile)
    """
    p_tile = {"warp": 0.0, "tile": 1.0, "mixed": 0.3}[scale]
    if mix == "nvidia":
        if specials:
            return FaultModel(p_tile=p_tile)
        tot = 0.5068 + 0.4831
        return FaultModel(p_zero=0.5068 / tot, p_bitflip=0.4831 / tot, p_special=0.0, p_tile=p_tile)
    if mix == "zero":
        return FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0, p_tile=p_tile, tile_p_zero=1.0)
    if mix == "mantissa":
        return FaultModel(p_zero=0.0, p_bitflip=1.0, p_special=0.0, p_tile=p_tile, tile_p_zero=0.0,
                          mantissa_only=True)
    raise ValueError(f"unknown fault mix {mix!r}")


def _bit_range(dtype: torch.dtype, fm: FaultModel) -> int:
    nbits = torch.finfo(dtype).bits
    return _MANTISSA_BITS.get(dtype, nbits) if fm.mantissa_only else nbits


def _flip_bits_vectorized(vals: torch.Tensor, gen: torch.Generator, fm: FaultModel) -> torch.Tensor:
    """Flip one random bit (low bits more likely) in every value. For large tile events.
    Bit twiddling happens on the CPU so it works the same on every backend."""
    nbits = vals.element_size() * 8
    hi = _bit_range(vals.dtype, fm)
    itype = _INT_VIEW[vals.element_size()]
    ints = vals.detach().to("cpu").contiguous().view(itype).clone()
    weights = torch.tensor([fm.bit_decay**i for i in range(hi)])
    bits = torch.multinomial(weights, ints.numel(), replacement=True, generator=gen)
    masks = torch.bitwise_left_shift(torch.ones_like(bits), bits)
    masks = torch.where(bits == nbits - 1, torch.full_like(bits, -(1 << (nbits - 1))), masks)
    ints ^= masks.to(itype).view(ints.shape)
    return ints.view(vals.dtype).to(vals.device)


def _rel_change(old: torch.Tensor, new: torch.Tensor, ref: torch.Tensor) -> float:
    """||new - old|| / ||reference block||, per row; returns the worst row (inf if non-finite)."""
    d = (new.float() - old.float())
    if d.dim() == 1:
        d, ref = d.unsqueeze(0), ref.unsqueeze(0)
    num = torch.linalg.vector_norm(d, dim=-1)
    den = torch.linalg.vector_norm(ref.float(), dim=-1) + 1e-30
    v = float((num / den).max())
    return v if math.isfinite(v) else float("inf")


def _corrupt_tile_(flat: torch.Tensor, rng: random.Random, fm: FaultModel) -> dict:
    rows, cols = flat.shape
    th, tw = min(fm.tile[0], rows), min(fm.tile[1], cols)
    r0 = rng.randrange(max(1, rows // th)) * th
    c0 = rng.randrange(max(1, cols // tw)) * tw
    frac = rng.uniform(*fm.tile_frac)
    gen = torch.Generator().manual_seed(rng.randrange(2**31))
    mask = (torch.rand(th, tw, generator=gen) < frac).to(flat.device)
    region = flat[r0:r0 + th, c0:c0 + tw]
    old = region.clone()
    if not bool(mask.any()):
        mask[0, 0] = True
    if rng.random() < fm.tile_p_zero:
        kind = "tile_zero"
        region.masked_fill_(mask, 0)
    else:
        kind = "tile_bitflip"
        flipped = _flip_bits_vectorized(region[mask], gen, fm)
        region[mask] = flipped
    return {"kind": kind, "row": r0, "col": c0, "n": int(mask.sum()), "frac": round(frac, 3),
            "rel_change": _rel_change(old, region, old)}


def _flip_bits(vals: torch.Tensor, rng: random.Random, fm: FaultModel) -> torch.Tensor:
    nbits = vals.element_size() * 8
    hi = _bit_range(vals.dtype, fm)
    ints = vals.detach().to("cpu").contiguous().view(_INT_VIEW[vals.element_size()]).clone()
    weights = [fm.bit_decay**i for i in range(hi)]
    counts, probs = zip(*fm.bits_per_flip)
    for j in range(ints.numel()):
        for bit in rng.choices(range(hi), weights=weights, k=rng.choices(counts, probs)[0]):
            mask = (1 << bit) if bit < nbits - 1 else -(1 << (nbits - 1))
            ints[j] ^= mask
    return ints.view(vals.dtype).to(vals.device)


def corrupt_(t: torch.Tensor, rng: random.Random, fm: FaultModel = FaultModel(),
             band: Optional[tuple] = None) -> Optional[dict]:
    """Corrupt `t` in place: usually one warp-aligned chunk of one row, or (with probability
    fm.p_tile) a large share of one output tile. Returns what was done, including
    rel_change: how much the affected output block changed, relative to its size."""
    if t.numel() == 0 or t.element_size() not in _INT_VIEW or not t.is_contiguous():
        return None
    flat = t.view(-1, t.shape[-1]) if t.dim() > 1 else t.view(1, -1)
    if fm.p_tile > 0 and rng.random() < fm.p_tile:
        return _corrupt_tile_(flat, rng, fm)
    rows, cols = flat.shape
    r = rng.randrange(rows)
    c0 = rng.randrange(max(1, cols // fm.chunk)) * fm.chunk
    period = rng.choice(fm.periods)
    idx = torch.arange(c0, min(c0 + fm.chunk, cols), period, device=t.device)
    if band is not None:  # data-dependent fault: only values inside a magnitude band
        v = flat[r, idx].float().abs()
        idx = idx[(v >= band[0]) & (v <= band[1])]
    if idx.numel() == 0:
        return None
    b0 = (c0 // fm.ref_block) * fm.ref_block
    ref = flat[r, b0:b0 + fm.ref_block].clone()
    old = flat[r, idx].clone()
    u = rng.random()
    if u < fm.p_zero:
        kind = "zero"
        flat[r, idx] = 0
    elif u < fm.p_zero + fm.p_bitflip:
        kind = "bitflip"
        flat[r, idx] = _flip_bits(flat[r, idx].contiguous(), rng, fm)
    else:
        kind = "special"
        flat[r, idx] = rng.choice([float("nan"), float("inf"), float("-inf")])
    return {"kind": kind, "row": r, "col": c0, "period": period, "n": int(idx.numel()),
            "rel_change": _rel_change(old, flat[r, idx], ref)}


class FaultInjector(TorchDispatchMode):
    """Corrupts outputs of matmuls with probability `rate` per call while armed.

    start_step: the "GPU goes bad" moment; no faults before it (set .step every step).
    """

    def __init__(self, rate: float = 0.01, fault_model: FaultModel = FaultModel(),
                 dtypes: Iterable[torch.dtype] = (torch.bfloat16, torch.float16),
                 band: Optional[tuple] = None, escalate: float = 1.0, seed: int = 1234,
                 start_step: Optional[int] = None):
        super().__init__()
        self.rate = rate
        self.fm = fault_model
        self.dtypes = set(dtypes)       # skip fp32 so detector arithmetic isn't hit
        self.band = band
        self.escalate = escalate        # rate *= escalate after every call (ByteDance: faults worsen)
        self.start_step = start_step
        self.armed = True
        self.eligible = 0               # matmul outputs that could have been corrupted (for sanity checks)
        self.events: list[dict] = []
        self.step: Optional[int] = None
        self.last_out: Optional[torch.Tensor] = None
        self.pending: Optional[int] = None
        self._rng = random.Random(seed)
        self._hooks = []

    def arm_on(self, modules: Iterable[torch.nn.Module]) -> "FaultInjector":
        """Only corrupt matmuls that run inside these modules' forward passes."""
        self.armed = False

        def pre(_m, _inp):
            self.armed = True

        def post(_m, _inp, _out):
            self.armed = False

        for m in modules:
            self._hooks.append(m.register_forward_pre_hook(pre))
            self._hooks.append(m.register_forward_hook(post))
        return self

    # A detector reports its checks here, so experiments can tell which faults were caught.
    def match(self, C: torch.Tensor) -> Optional[int]:
        """If C is the output this injector just corrupted, return that fault's index.

        C may be a reshaped view of that output (Apple GPUs check linear layers on a 2D view
        of a 3D output), so compare the memory it starts at and its size, not its shape.
        """
        if (self.pending is not None and self.last_out is not None
                and C.data_ptr() == self.last_out.data_ptr() and C.numel() == self.last_out.numel()):
            idx = self.pending
            self.events[idx]["checked"] = True
            self.pending, self.last_out = None, None
            return idx
        return None

    def resolve(self, idx: Optional[int], ok: bool, z: Optional[float] = None) -> None:
        if idx is not None:
            self.events[idx]["caught"] = not ok
            self.events[idx]["z"] = z if z is None or math.isfinite(z) else str(z)

    def record_check(self, C: torch.Tensor, ok: bool) -> None:
        """Older one-call form of match() + resolve()."""
        self.resolve(self.match(C), ok)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.armed = True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if (self.armed and isinstance(out, torch.Tensor) and out.dtype in self.dtypes
                and _op_name(func) in TARGET_OPS
                and (self.start_step is None or (self.step is not None and self.step >= self.start_step))):
            self.eligible += 1
            if self._rng.random() < self.rate:
                with torch.no_grad():
                    info = corrupt_(out, self._rng, self.fm, self.band)
                if info:
                    info.update(step=self.step, op=_op_name(func), checked=False, caught=False)
                    self.events.append(info)
                    # Keep a reference so the memory can't be reused by another tensor; a detector
                    # matches its check to this fault by data pointer.
                    self.last_out = out
                    self.pending = len(self.events) - 1
            self.rate = min(1.0, self.rate * self.escalate)
        return out
