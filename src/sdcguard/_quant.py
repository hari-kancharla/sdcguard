"""Weight-only quantized linear layers (INT4 and INT8 with group scales) as vLLM's Marlin and Machete kernels run
them: the weights each kernel actually multiplies by, read from the layer before the kernel repacks them.

The check itself doesn't change. It needs the effective weight matrix W (n outputs x k inputs) the GEMM multiplies x
by, and for these kernels that matrix is exactly representable in the activation dtype (bf16 or fp16), because
each kernel dequantizes into that dtype before its fp32 tensor-core accumulation (vLLM 0.31's CUDA source):

  Marlin, grouped scales     W = rnd((q - c) * s)            c = 8 (uint4b8), 128 (uint8b128) or the group's zero
                                                              point (uint4); q - c is exact, then one multiply
  Machete, no zero points    W = rnd((q - c) * s)            the same
  Machete, zero points       W = rnd(q * s + zs)             zs = -rnd(s * z), computed once at load by vLLM;
                                                              q * s + zs rounded once (a fused multiply-add)

where rnd rounds to the activation dtype (round to nearest even) and s is the scale of the weight's group (k //
group_size). Marlin with one scale per output (group_size -1) applies it after accumulating instead, which adds a
second rounding of every output: those layers aren't checked yet.

Machete's source (machete_mainloop.cuh, transform_A_kblock) multiplies by the scale and then adds the zero term in
two separate steps, which reads as two roundings, rnd(rnd(q * s) + zs). But on Hopper both steps are bf16x2/f16x2
mul and add instructions without a rounding modifier (cuda_bf16.hpp), which PTX lets the compiler fuse into one
fused multiply-add, and Machete's own comment there names hfma.bf16_v2 as the goal. On an H100 (October 22, 2026)
the two-rounding reading differed from Machete's output in 43% to 51% of the values of every AWQ layer it was
probed on, in the range rounding once instead of twice gives for codes near their zero point (a share, not a
proof). The rule is a compiler decision, not something the source fixes, so a QuantWeights for Machete with
zero points can give either reading (fused), and
the plugin keeps whichever one probe() confirms bit for bit, at every batch size Machete picks a different kernel
for; when neither matches, the layer isn't checked.

After the kernel repacks its weights (Marlin's tile layout, Machete's CUTLASS prepacking) they can't be read back
as a matrix, so:
  - the checksums are computed from W built here, before the repack;
  - probe() runs the kernel itself on unit vectors after the repack: row i of the result is exactly row i of W^T
    (one product by 1.0 per output, added to zeros in fp32, then rounded: exact), so any difference means this
    module's reading of the weights is wrong for this kernel and the layer isn't checked;
  - QuantWeights keeps a CPU copy of the packed codes, scales and zero points (the size of the quantized layer)
    so an alert's recheck can rebuild the 128 rows it needs;
  - changes to the weights are detected from the kernel's own tensors (fingerprint()).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional, Sequence

import torch

MARLIN, MACHETE = "marlin", "machete"

_lock = threading.Lock()
_ADD = [None]


def add_bias_op():
    """sdcguard::add_bias: y + b in y's dtype, rounded once, as one opaque call. Marlin adds a layer's bias inside its
    kernel, rounded to the activation dtype; when the plugin adds it after the check instead, torch.compile must not
    fuse that addition into the next operations and keep its result in fp32 (Inductor computes a fused chain in fp32
    and rounds only what it stores), or the output would differ from Marlin's."""
    with _lock:
        if _ADD[0] is None:
            try:
                @torch.library.custom_op("sdcguard::add_bias", mutates_args=())
                def add_bias(y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                    return y + b

                @add_bias.register_fake
                def _(y, b):
                    return torch.empty_like(y)

                _ADD[0] = torch.ops.sdcguard.add_bias
            except (AttributeError, RuntimeError):      # very old PyTorch, or already registered
                _ADD[0] = getattr(getattr(torch.ops, "sdcguard", None), "add_bias", None) or (lambda y, b: y + b)
        return _ADD[0]


_SKIP_FIELDS = frozenset(("ctx", "type_comment", "kind", "type_params", "annotation", "returns"))


def _ast_shape(node):
    """A nested tuple of an AST's node types and values, without what doesn't change what the code does (load or
    store context, type comments and annotations, a string constant's kind) and without positions. The same on
    Pythons 3.10 to 3.13 for the functions fingerprinted below (checked on each)."""
    import ast
    if isinstance(node, ast.AST):
        return (type(node).__name__,
                tuple((f, _ast_shape(getattr(node, f, None))) for f in node._fields if f not in _SKIP_FIELDS))
    if isinstance(node, list):
        return tuple(_ast_shape(x) for x in node)
    return node


def code_fingerprint(fn) -> Optional[str]:
    """16 hex digits that change when a function's code changes, but not with its formatting, comments, docstring
    or type annotations: its arguments and body, parsed. None when its source can't be read."""
    import ast
    import hashlib
    import inspect
    import textwrap
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    f = tree.body[0]
    body = f.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return hashlib.sha256(repr((_ast_shape(f.args), _ast_shape(body))).encode()).hexdigest()[:16]


# The plugin replaces Machete's apply_weights with a copy of vLLM's that calls one opaque op for Machete's GEMM and
# the check together (machete_apply_weights in vllm_plugin._patch_quant, and vllm_plugin._machete_op), and that op
# calls torch.ops._C.machete_mm exactly as vLLM's Python wrapper does. Those copies are only right for the vLLM code
# they were copied from: these are the fingerprints of MacheteLinearKernel.apply_weights and
# vllm._custom_ops.machete_mm in vLLM 0.31.0, unchanged in 0.32.0 and on main (October 23, 2026). Any other code:
# Machete's layers aren't checked, and say why.
MACHETE_APPLY_KNOWN = frozenset({"0b57e96e0254ff1f"})
MACHETE_MM_KNOWN = frozenset({"a0b7b9fc3a58d060"})


def _dims(p: torch.Tensor) -> tuple:
    """(input dim, output dim) of a 2-D vLLM parameter, from its metadata (one may be missing: then it's the other
    axis, as vLLM's permute_param_layout_ takes it)."""
    i, o = getattr(p, "input_dim", None), getattr(p, "output_dim", None)
    if p.dim() != 2 or (i is None and o is None):
        raise ValueError(f"expected a 2-D parameter with input_dim or output_dim, got shape {tuple(p.shape)}")
    if i is None:
        i = 1 - o
    if o is None:
        o = 1 - i
    if {i, o} != {0, 1}:
        raise ValueError(f"input_dim {i} and output_dim {o} of a 2-D parameter")
    return i, o


def _in_out(p: torch.Tensor) -> tuple:
    """The parameter's data as (input, output), and which of those two axes holds packed values (None: none)."""
    i, o = _dims(p)
    t = p.data if (i, o) == (0, 1) else p.data.t()
    pd = getattr(p, "packed_dim", None)
    return t, (None if pd is None else (0 if pd == i else 1))


def unpack(t: torch.Tensor, bits: int, axis: int) -> torch.Tensor:
    """int32 words holding 32 // bits values each (the lowest bits first) along `axis`, as one int32 per value.
    The same order as vLLM's unpack_quantized_values_into_int32."""
    if t.dtype != torch.int32:
        raise ValueError(f"packed values must be int32, got {t.dtype}")
    per = 32 // bits
    mask = (1 << bits) - 1
    moved = t.movedim(axis, -1)
    out = torch.empty(*moved.shape[:-1], moved.shape[-1] * per, dtype=torch.int32, device=t.device)
    for i in range(per):
        out[..., i::per] = (moved >> (bits * i)) & mask
    return out.movedim(-1, axis)


class QuantWeights:
    """One weight-only quantized layer: its codes (packed, input x output), scales and zero points (group x output),
    the code offset, the group size and the dequantization rule of the kernel that runs it. weight() gives W
    (n x k, the activation dtype) for any range of outputs, computed in chunks."""

    def __init__(self, packed: torch.Tensor, bits: int, scales: torch.Tensor, zeros: Optional[torch.Tensor],
                 code_bias: int, group_size: int, rule: str, dtype: torch.dtype, fused: bool = True):
        if rule not in (MARLIN, MACHETE):
            raise ValueError(f"unknown dequantization rule {rule!r}")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"activations must be bf16 or fp16, got {dtype}")
        self.packed, self.bits, self.scales, self.zeros = packed, bits, scales.to(dtype), zeros
        self.code_bias, self.rule, self.dtype = int(code_bias), rule, dtype
        # Machete with zero points: q * s + zs rounded once (True) or rounded after the multiply too (False)
        self.fused = bool(fused)
        self.k = packed.shape[0] * (32 // bits)
        self.n = packed.shape[1]
        self.group = self.k if group_size in (-1, 0) or group_size >= self.k else int(group_size)
        groups = -(-self.k // self.group)
        if tuple(self.scales.shape) != (groups, self.n):
            raise ValueError(f"scales of shape {tuple(self.scales.shape)} for {groups} groups of {self.n} outputs")
        if zeros is not None and tuple(zeros.shape) != (groups, self.n):
            raise ValueError(f"zero points of shape {tuple(zeros.shape)} for {groups} groups of {self.n} outputs")

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales, self.zeros) if t is not None)

    def to(self, device) -> "QuantWeights":
        q = QuantWeights.__new__(QuantWeights)
        q.__dict__.update(self.__dict__)
        q.packed, q.scales = self.packed.to(device), self.scales.to(device)
        q.zeros = None if self.zeros is None else self.zeros.to(device)
        return q

    @property
    def rounding_varies(self) -> bool:
        """True when the kernel's rounding isn't fixed by its source (Machete with zero points): see the module
        docstring."""
        return self.rule == MACHETE and self.zeros is not None

    def other_rounding(self) -> "QuantWeights":
        """The same weights read with the other Machete zero-point rounding (the tensors are shared)."""
        q = QuantWeights.__new__(QuantWeights)
        q.__dict__.update(self.__dict__)
        q.fused = not self.fused
        return q

    def describe(self) -> str:
        if not self.rounding_varies:
            return ""
        return "zero points added in one rounding" if self.fused else "zero points added after a rounded multiply"

    def weight(self, start: int = 0, width: Optional[int] = None, chunk: int = 1024) -> torch.Tensor:
        """W[start:start + width] (outputs x inputs) on the device the copy is on, in the activation dtype."""
        width = self.n - start if width is None else width
        out = torch.empty(width, self.k, dtype=self.dtype, device=self.packed.device)
        g = torch.arange(self.k, device=self.packed.device) // self.group          # group of each input
        for a in range(start, start + width, chunk):
            b = min(a + chunk, start + width)
            q = unpack(self.packed[:, a:b].contiguous(), self.bits, 0)              # (k, cols) codes
            s = self.scales[:, a:b].float()[g]                                      # (k, cols)
            if self.rule == MACHETE and self.zeros is not None:
                # zs = -rnd(s * z): vLLM computes it at load, in the activation dtype. q * s (at most 8 + 11
                # significant bits) plus zs is exact in fp32, so the one rounding below is the fused multiply-add's
                zs = -(self.scales[:, a:b] * self.zeros[:, a:b].to(self.dtype))
                qs = q.float() * s
                w = (qs if self.fused else qs.to(self.dtype).float()) + zs.float()[g]
            else:
                c = self.code_bias if self.zeros is None else self.zeros[:, a:b][g]
                w = (q - c).float() * s                                             # exact in fp32
            out[a - start:b - start] = w.to(self.dtype).t()
        return out


def from_layer(layer: torch.nn.Module, w_q_name: str, w_s_name: str, w_zp_name: Optional[str], weight_type,
               group_size: int, rule: str, dtype: torch.dtype) -> QuantWeights:
    """Read a layer's quantized weights in vLLM's standard layout (what Marlin's and Machete's
    process_weights_after_loading start from), without changing the layer. weight_type: vLLM's ScalarType
    (size_bits, bias)."""
    bits = int(weight_type.size_bits)
    if bits not in (4, 8):
        raise ValueError(f"{bits}-bit weights")
    q, qa = _in_out(getattr(layer, w_q_name))
    if qa != 0:
        raise ValueError("the codes aren't packed along the input dimension")
    s, _ = _in_out(getattr(layer, w_s_name))
    zeros = None
    if w_zp_name is not None and isinstance(getattr(layer, w_zp_name, None), torch.Tensor):
        z, za = _in_out(getattr(layer, w_zp_name))
        if za is None:
            raise ValueError("the zero points aren't packed")
        zeros = unpack(z.contiguous(), bits, za)
    return QuantWeights(q.contiguous(), bits, s.contiguous(), zeros, int(weight_type.bias), group_size, rule, dtype)


def probe_rows(k: int, group: int, limit: int = 1024, seed: int = 0) -> list:
    """Which unit vectors probe() feeds the kernel: every input when there are at most `limit`, else one input of
    every group (a different position in each), the first and last input, and random ones up to `limit`."""
    if k <= limit:
        return list(range(k))
    gsz = k if group in (-1, 0) or group >= k else group
    rows = {0, k - 1}
    for gi, a in enumerate(range(0, k, gsz)):
        rows.add(min(k - 1, a + (gi * 37) % gsz))
    gen = torch.Generator().manual_seed(seed)
    extra = torch.randperm(k, generator=gen).tolist()
    for r in extra:
        if len(rows) >= limit:
            break
        rows.add(r)
    return sorted(rows)


# Batch sizes probe() runs the kernel at, in turn: one in each of the ranges Machete picks a different tile shape
# (and so a different compiled kernel) for, 1-16, 17-32, 33-64, 65-128, 129-256 and over 256 (vLLM 0.31's
# generate.py heuristic); Marlin also changes its blocking with the batch size.
PROBE_BATCHES = (16, 32, 64, 128, 256, 512)


@torch.no_grad()
def probe(apply: Callable[[torch.Tensor], torch.Tensor], w: torch.Tensor, rows: Sequence[int],
          batches: Sequence[int] = PROBE_BATCHES) -> dict:
    """Run the kernel (apply: x (m x k) -> y (m x n), no bias) on unit vectors e_i for i in rows and compare each
    result with row i of W^T (w: n x k, on the kernel's device). Every value must be equal (-0 and +0 count as
    equal). The rows go through in batches of the sizes in `batches`, in turn, so a kernel that computes differently
    at different batch sizes is seen at each (all of them when there are at least 497 rows). Returns {"rows": how
    many, "values": compared, "mismatches": unequal, "first": (input, output) of the first unequal value or None,
    "batches": the batch sizes run}."""
    n, k = w.shape
    bad, first, seen, sizes, bad_at = 0, None, 0, [], []
    a = 0
    while a < len(rows):
        m = int(batches[len(sizes) % len(batches)])
        idx = torch.tensor(list(rows[a:a + m]), device=w.device, dtype=torch.long)
        a += m
        sizes.append(len(idx))
        x = torch.zeros(len(idx), k, dtype=w.dtype, device=w.device)
        x[torch.arange(len(idx), device=w.device), idx] = 1
        y = apply(x).reshape(len(idx), -1)
        if y.shape[1] != n:
            raise ValueError(f"the kernel returned {y.shape[1]} outputs, the weights have {n}")
        ref = w[:, idx].t()
        diff = (y != ref) & ~(torch.isnan(y) & torch.isnan(ref))
        nd = int(diff.sum())
        if nd and first is None:
            r, c = (int(v) for v in diff.nonzero()[0])
            first = (int(idx[r]), c)
        if nd:
            bad_at.append(len(idx))
        bad += nd
        seen += y.numel()
    return {"rows": len(rows), "values": seen, "mismatches": bad, "first": first, "batches": sizes,
            "batches_with_mismatches": sorted(set(bad_at))}


def fingerprint(tensors: Sequence[Optional[torch.Tensor]]) -> torch.Tensor:
    """Two sums per tensor of its raw bytes (plain, and weighted by position): any bit flip changes at least one.
    (k x 2 int64 on the CPU; a missing tensor gives zeros.)"""
    out = torch.zeros(len(tensors), 2, dtype=torch.int64)
    for j, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        b = t.detach().contiguous().view(-1).view(torch.uint8)
        tot = torch.zeros(2, dtype=torch.int64, device=b.device)
        step = 1 << 24
        for i in range(0, b.numel(), step):
            v = b[i:i + step].to(torch.int64)
            pos = torch.arange(i, i + v.numel(), device=b.device, dtype=torch.int64) % 65521 + 1
            tot += torch.stack([v.sum(), (v * pos).sum()])
        out[j] = tot.cpu()
    return out


class QuantSource:
    """What ServeGuard keeps for a quantized layer instead of reading its weight: the names of the tensors the kernel
    reads (for in-place write counts and the fingerprint), their fingerprint at registration, and the CPU copy of the
    weights (None when it isn't kept, SDCGUARD_QUANT_RECHECK=0: alerts then can't be rechecked)."""

    def __init__(self, layer: torch.nn.Module, names: Sequence[str], weights: Optional[QuantWeights], label: str):
        self.names = tuple(names)
        self.weights = weights
        self.label = label
        self.fp = fingerprint(self.live(layer))

    def live(self, layer: torch.nn.Module) -> tuple:
        return tuple(getattr(layer, nm, None) for nm in self.names)

    def changed(self, layer: torch.nn.Module) -> bool:
        return not torch.equal(fingerprint(self.live(layer)), self.fp)

    def rows(self, start: int, width: int) -> torch.Tensor:
        """W[start:start + width] on the CPU (outputs x inputs, the activation dtype)."""
        if self.weights is None:
            raise RuntimeError("the quantized weights weren't kept (SDCGUARD_QUANT_RECHECK=0)")
        return self.weights.weight(start, width)
