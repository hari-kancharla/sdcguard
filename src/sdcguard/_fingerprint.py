"""One-pass GPU fingerprint for the recompute check (recompute.fingerprint).

The PyTorch version launches about a dozen kernels per million words of a tensor (positions, their two weights,
products and sums). On an H100 the earlier PyTorch fingerprint, about ten kernels per tensor, made the recompute
check cost 16% to 20% of a training step with every step checked (docs/METHOD.md, "The recompute check on GPUs").
This kernel reads the tensor once, computes each word's weights from its position in registers, and adds the same
three int64 sums with integer atomics. int64 addition and multiplication wrap around the same way here as in
PyTorch, and wrapping addition doesn't depend on the order, so the result is bit-for-bit the PyTorch one, and a
block's forward run and its recompute are compared the same way whichever path computed each.

Before first use on a device, the kernel has to match the PyTorch version on probe tensors (int16 and int8
views, sizes that aren't a multiple of the kernel's block); otherwise the PyTorch version runs and the reason is
kept in STATUS. SDCGUARD_RECOMPUTE_FAST=0 (or SDCGUARD_FAST=0) turns the kernel off.
"""
from __future__ import annotations

import os
import threading
import warnings

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

BLOCK = 4096
STATUS: dict = {}          # str(device) -> (ok, why)
_LOCK = threading.Lock()


def enabled() -> bool:
    return (HAVE_TRITON and os.environ.get("SDCGUARD_FAST", "1") != "0"
            and os.environ.get("SDCGUARD_RECOMPUTE_FAST", "1") != "0")


def _cpu_test() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1" and os.environ.get("SDCGUARD_FAST_CPU_TEST") == "1"


if HAVE_TRITON:
    @triton.jit
    def _lsr(z, s: tl.constexpr):
        """Logical right shift of int64 values, as recompute._lsr."""
        return (z >> s) & ((1 << (64 - s)) - 1)

    @triton.jit
    def _fingerprint_kernel(X, OUT, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        mask = offs < n
        b = tl.load(X + offs, mask=mask, other=0).to(tl.int64)
        # recompute._fp_weights, with its constants (_FP_SEED, _FP_M1, _FP_M2) written out
        z = offs + -7046029254386353131
        z = (z ^ _lsr(z, 30)) * -4658895280553007687
        z = (z ^ _lsr(z, 27)) * -7723592293110705685
        z = z ^ _lsr(z, 31)
        w1 = z | 1
        w2 = (z ^ _lsr(z, 32)) | 1
        tl.atomic_add(OUT + 0, tl.sum(b, axis=0))
        tl.atomic_add(OUT + 1, tl.sum(b * w1, axis=0))
        tl.atomic_add(OUT + 2, tl.sum(b * w2, axis=0))


def _launch(bits: torch.Tensor) -> torch.Tensor:
    """The three sums of a flat int16 or int8 tensor, in one pass."""
    n = bits.numel()
    out = torch.zeros(3, dtype=torch.int64, device=bits.device)
    if n == 0:
        return out
    grid = (triton.cdiv(n, BLOCK),)
    _fingerprint_kernel[grid](bits, out, n, BLOCK=BLOCK, num_warps=4)
    return out


def _probe_ok(device: torch.device, reference) -> tuple:
    """The kernel must give exactly the PyTorch sums before it is used on this device."""
    try:
        g = torch.Generator(device="cpu").manual_seed(251)
        cases = [torch.randint(-32768, 32768, (3 * BLOCK + 19,), generator=g, dtype=torch.int32).to(torch.int16),
                 torch.randint(-128, 128, (BLOCK + 3,), generator=g, dtype=torch.int32).to(torch.int8),
                 torch.randint(-32768, 32768, (250,), generator=g, dtype=torch.int32).to(torch.int16),
                 torch.full((2259,), 32767, dtype=torch.int16), torch.full((BLOCK,), -32768, dtype=torch.int16)]
        for c in cases:
            c = c.to(device)
            fast, slow = _launch(c), reference(c)
            if not torch.equal(fast.cpu(), slow.cpu()):
                return False, f"fingerprint kernel disagreed with the PyTorch sums ({fast.tolist()} vs {slow.tolist()})"
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def usable(device: torch.device, reference) -> bool:
    """Whether the kernel runs for tensors on `device`. The first call per device runs the probe comparison."""
    if not enabled():
        return False
    if device.type == "cuda":
        from ._group import arch_ok
        if not arch_ok(device):
            return False
    elif not (device.type == "cpu" and _cpu_test()):
        return False
    from ._group import _dev_key
    key = _dev_key(device)                 # "cuda" and "cuda:0" are one device, as for the other kernels
    st = STATUS.get(key)
    if st is None:
        with _LOCK:
            st = STATUS.get(key)
            if st is None:
                st = _probe_ok(device, reference)
                STATUS[key] = st
                if not st[0]:
                    warnings.warn(f"sdcguard: the recompute fingerprint kernel can't run on {key} ({st[1]}); "
                                  "using the PyTorch version, which gives the same result more slowly", stacklevel=3)
    return st[0]


def status(device):
    """(ok, why) of the probe comparison on this device, or None if it hasn't run."""
    from ._group import _dev_key
    return STATUS.get(_dev_key(torch.device(device)))


def fingerprint_bits(bits: torch.Tensor) -> torch.Tensor:
    """Fast path for recompute.fingerprint: `bits` is the flat int16 or int8 view of a contiguous tensor."""
    return _launch(bits)
