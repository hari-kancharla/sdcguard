"""Device self-test: proves the detector really sees and checks this machine's matmuls.

Run:  python -m sdcguard.selftest

On the current device (NVIDIA CUDA, Apple MPS, or CPU) it checks that:
  1. the detector sees matrix multiplies in the forward AND backward pass
  2. clean training steps raise zero alerts
  3. injected faults are actually injected, and caught
On NVIDIA GPUs it also checks that a checked step never makes the CPU wait for the GPU
(PyTorch's sync debugger set to "error"), and that matmul partial sums are kept in fp32 while
the guard is on (and the user's setting comes back afterwards).
It prints PASS/FAIL for each and exits non-zero if anything fails. Run it on every new machine
before trusting any experiment: a detector that silently checks nothing is worse than none.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from .guard import SDCGuard, op_name
from .inject import FaultInjector, FaultModel


class _OpCensus(TorchDispatchMode):
    """Records every operator name that runs, to spot matmul ops a new backend might use."""

    def __init__(self):
        super().__init__()
        self.names: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        n = op_name(func)
        self.names[n] = self.names.get(n, 0) + 1
        return func(*args, **(kwargs or {}))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype(device: torch.device) -> torch.dtype:
    for dt in (torch.bfloat16, torch.float16):
        try:
            with torch.autocast(device.type, dtype=dt):
                out = nn.functional.linear(torch.randn(4, 8, device=device), torch.randn(8, 8, device=device))
            if out.dtype == dt:
                return dt
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"no bf16/fp16 autocast on {device}")


def _step(model, x, device, dtype):
    with torch.autocast(device.type, dtype=dtype):
        loss = model(x).float().pow(2).mean()
    loss.backward()
    model.zero_grad(set_to_none=True)


def _sync_free_checked_step(model, x, device, dtype):
    """True if a whole checked forward+backward runs without a single CPU-GPU sync."""
    def run(guard):
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.autocast(device.type, dtype=dtype):
                loss = model(x).float().pow(2).mean()
            loss.backward()
            return None
        except RuntimeError as e:
            return str(e).splitlines()[0][:160]
        finally:
            torch.cuda.set_sync_debug_mode(0)
            model.zero_grad(set_to_none=True)

    # last layer with 1000 outputs = 7 full blocks of 128 + a remainder of 104: covers the
    # multi-span path too (GPT-2's 50,257-word output layer has a remainder block)
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(x.shape[-1], 1024), nn.GELU(), nn.Linear(1024, 1000)).to(device)
    warm = SDCGuard(sample_rate=1.0)              # first use caches small constants on the GPU
    with warm:                                    # and compiles the kernels the test below uses
        _step(model, x, device, dtype)
    if run(None) is not None:
        return None, "skipped: the test model itself syncs"
    g = SDCGuard(sample_rate=1.0)
    with g:
        err = run(g)
    if err is None and g.stats["checked"] == 0:
        return False, "nothing was checked"
    return err is None, err


def _triton_agrees(device, dtype):
    """The Triton GPU check and the plain check must agree on a clean and a corrupted matmul."""
    from . import checksum as cs
    g = torch.Generator().manual_seed(5)
    A = torch.randn(256, 512, generator=g).to(device=device, dtype=dtype)
    B = torch.randn(512, 1000, generator=g).to(device=device, dtype=dtype)
    C = A @ B
    bad = C.clone()
    bad[100, 300:332:2] = 0
    out = []
    try:
        for fast in (True, False):
            cs.USE_FAST = fast
            out.append((cs.check_matmul(A, B, C), cs.check_matmul(A, B, bad)))
    finally:
        cs.USE_FAST = True
    (fc, fb), (pc, pb) = out
    ok = (fc.ok and pc.ok and abs(fc.max_z - pc.max_z) <= 0.05 + 0.02 * pc.max_z
          and not fb.ok and not pb.ok and (fb.row, fb.col_block) == (pb.row, pb.col_block))
    return ok, (f"clean z {fc.max_z:.3f} vs {pc.max_z:.3f}; corrupted at row/block "
                f"{fb.row}/{fb.col_block} vs {pb.row}/{pb.col_block}")


def _fp16_overflow_ok(device):
    """fp16 training with too high a loss scale overflows inside the matmuls on purpose
    (GradScaler then skips the step). That must not look like a hardware fault."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64)).to(device).half()
    x = torch.randn(32, 64, device=device).half()
    g = SDCGuard(sample_rate=1.0)
    with g:
        loss = m(x).float().pow(2).mean()
        (loss * 2.0**28).backward()
    overflowed = not bool(torch.isfinite(m[0].weight.grad).all())
    return overflowed and g.stats["checked"] > 0 and g.stats["alerts"] == 0


def main() -> int:
    device = _device()
    dtype = _dtype(device)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256)).to(device)
    x = torch.randn(16, 64, 256, device=device)       # (batch, tokens, features), like a transformer

    fwd = SDCGuard(sample_rate=1.0)
    with fwd, torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        model(x)

    clean = SDCGuard(sample_rate=1.0)
    for _ in range(5):
        with clean:
            _step(model, x, device, dtype)

    inj = FaultInjector(rate=1.0, seed=1,
                        fault_model=FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0)).arm_on([model[0]])
    caught = SDCGuard(sample_rate=1.0)
    for _ in range(5):
        with inj:
            with caught:
                _step(model, x, device, dtype)
    inj.remove_hooks()

    # Tiny but correct values (like collapsed-attention gradients, ~1e-24) must not look corrupted.
    from .checksum import check_matmul
    g = torch.Generator().manual_seed(0)
    W = (torch.randn(768, 768, generator=g) * 0.02).to(device=device, dtype=dtype)
    tiny_ok = True
    for scale in (1e-18, 1e-22, 1e-24, 1e-26):
        dY = (torch.randn(1024, 768, generator=g) * scale).to(device=device, dtype=dtype)
        res = check_matmul(dY, W, dY @ W)
        tiny_ok &= res is not None and res.ok

    checks = [
        ("detector sees matmuls in the forward pass", fwd.stats["matmuls_seen"] > 0),
        ("detector sees extra matmuls in the backward pass",
         clean.stats["matmuls_seen"] > 5 * fwd.stats["matmuls_seen"]),
        ("detector actually checks them (dtype supported)", clean.stats["checked"] > 0),
        ("zero false alarms on clean steps", clean.stats["alerts"] == 0),
        ("faults were injected", len(inj.events) > 0),
        ("injected faults were caught", caught.stats["alerts"] > 0),
        ("no false alarms on tiny values (1e-18 to 1e-26, like collapsed attention gradients)", tiny_ok),
    ]
    try:
        checks.append(("no false alarm when fp16 loss scaling overflows (Inf the inputs explain)",
                       _fp16_overflow_ok(device)))
    except (RuntimeError, NotImplementedError) as e:   # a device without fp16 support
        print(f"  (fp16 overflow check skipped: {str(e).splitlines()[0][:120]})")

    tf32_note = None
    if device.type == "cuda":
        # Training scripts often turn on TF32 for fp32 matmuls; the check must not depend on it.
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            tf32 = SDCGuard(sample_rate=1.0)
            for _ in range(3):
                with tf32:
                    _step(model, x, device, dtype)
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            checks.append(("zero false alarms with TF32 turned on", tf32.stats["alerts"] == 0))
        except (AttributeError, RuntimeError) as e:
            tf32_note = f"TF32 check skipped ({e})"
        from . import checksum as cs
        fast = cs._LOWP_GEMM_OK.get(dtype, True) and cs._LOWP_FALLBACKS[0] == 0
        # Speed only: without it the checks use plain reductions, which are just as correct.
        fast_note = ("plain-check fast GEMM (fp32 output): in use" if fast else
                     "plain-check fast GEMM (fp32 output): not available with this PyTorch/GPU, "
                     "using the slower fallback (results are the same)")
        from . import _group
        if _group.enabled():
            # Triton is installed: its kernels must pass their first-use comparison here, or every
            # check falls back to the (correct but slower) plain check
            used = _group.available(torch.empty(1, 1, device=device, dtype=dtype))
            first = _group.status(device, dtype) or (False, "comparison did not run")
            checks.append(("Triton GPU check passed its first-use comparison with the plain check"
                           + ("" if used else f" ({first[1]})"), used))
            if used:
                agree, why = _triton_agrees(device, dtype)
                checks.append(("Triton GPU check gives the same answers as the plain check"
                               + ("" if agree else f" ({why})"), agree))
                fast_note += f"; Triton GPU check: in use ({why})"
            checks.append(("Triton GPU check ran with no fallbacks"
                           + ("" if _group.FALLBACKS[0] == 0 else f" ({'; '.join(_group.WHY[-3:])})"),
                           _group.FALLBACKS[0] == 0))
        else:
            fast_note += ("; Triton GPU check: not available (Triton not installed or SDCGUARD_FAST=0), "
                          "using the plain check (same results, slower)")
        from . import _serve_fast
        if _serve_fast.usable(device, dtype):
            # the serving check's Triton kernel (ServeGuard, the vLLM plugin): same rule
            sv = _serve_fast.verified(device, dtype)
            st = _serve_fast.STATUS.get((_group._dev_key(device), dtype), (False, "comparison did not run"))
            checks.append(("serving Triton check passed its first-use comparison with the PyTorch serving check"
                           + ("" if sv else f" ({st[1]})"), sv))

        try:
            ok, why = _sync_free_checked_step(model, x, device, dtype)
        except Exception as e:  # noqa: BLE001  (report, don't crash the self-test)
            ok, why = False, f"{type(e).__name__}: {e}"[:160]
        if ok is None:
            tf32_note = (tf32_note + "; " if tf32_note else "") + f"no-stall check {why}"
        else:
            checks.append(("checked step never makes the CPU wait for the GPU" + ("" if ok else f" ({why})"), ok))

        try:
            m = torch.backends.cuda.matmul
            before = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
            with SDCGuard(sample_rate=1.0):
                inside = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
            after = (m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction)
            flags_ok, flags_note = inside == (False, False) and after == before, f"{before} -> {inside} -> {after}"
        except Exception as e:  # noqa: BLE001
            flags_ok, flags_note = False, f"{type(e).__name__}: {e}"[:160]
        checks.append(("matmul partial sums kept in fp32 while checking, setting restored after"
                       + ("" if flags_ok else f" ({flags_note})"), flags_ok))

    census = _OpCensus()
    with census:
        _step(model, x, device, dtype)
    matmul_like = {k: v for k, v in census.names.items()
                   if any(t in k for t in ("mm", "linear", "matmul", "gemm"))}

    print(f"sdcguard self-test on {device} ({dtype}), torch {torch.__version__}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  matmul ops seen: {clean.ops_seen}")
    print(f"  all matmul-like ops in one step: {matmul_like}")
    print(f"  clean: {clean.stats['checked']} checks, {clean.stats['alerts']} alerts, "
          f"worst clean z {clean.max_z_seen:.2f} (alert threshold {clean.z})")
    print(f"  faulty: {len(inj.events)} faults injected, {caught.stats['alerts']} alerts")
    if tf32_note:
        print(f"  {tf32_note}")
    if device.type == "cuda":
        print(f"  {fast_note}")
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed
    print("ALL PASS" if ok else "SELF-TEST FAILED: do not trust experiments on this machine yet")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
