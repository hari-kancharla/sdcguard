"""sdcguard check: is this GPU computing correctly, and at full speed? One command, about 5 minutes.

    pip install sdcguard
    sdcguard check                 # every GPU on this machine, 5 minutes
    sdcguard check --share         # also print a link that posts the result card to the public board

What it does, per GPU (all GPUs at once, so a machine is tested the way it's used):

  memory      fills most of the free GPU memory with patterns, reads it back, counts mismatches
  speed       big matmuls in bf16 (fp16 on GPUs without bf16): sustained TFLOPS, and how much the speed
              drops while the GPU heats up; memory bandwidth
  stress      big matmuls at full power with every output checked against its checksums
  math modes  matmuls in BF16, FP16, TF32, FP32, FP64 and INT8 (whichever the GPU supports), each result kept
              in full precision and compared with a float64 answer computed on the CPU
  workload    a GPT-style model training and serving with every matrix multiply checked (sdcguard screen)
  repeatable  the same fixed model and input run three times (start, hot, end); a healthy GPU gives
              bit-identical outputs every time. The output fingerprint can also be compared with other
              GPUs of the same model running the same software (here, and on the public board)
  health      temperature, power, clocks, throttling, PCIe link, ECC and memory-repair counters

Verdict: PASS (no failure detected under the completed checks), FAIL (a failure criterion was met;
the failed matmuls of the workload, stress and math-mode checks are saved for `sdcguard replay`), SUSPECT
(no checksum failed, but outputs that should be bit-identical weren't), INCOMPLETE (nothing wrong was found,
but part of the check couldn't run or couldn't judge its results). Exit code 0 PASS, 1 FAIL or SUSPECT,
2 INCOMPLETE or couldn't run.

Files, per GPU, in sdcguard_check/<GPU UUID>/: check.json (everything), result.json (shareable: no
serial number, hostname or full UUID), card.svg / card.png / card.md (the result card), plus the
screen's summary.json and any evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from typing import Optional

from . import verdict
from .board import SCHEMA, digest
FP_VERSION = 2
FP_CONFIG = {"gpu": (6, 1024, 16, 256, 2), "cpu": (2, 128, 4, 64, 2)}   # layers, width, heads, seq, batch
BOARD_REPO = os.environ.get("SDCGUARD_BOARD_REPO", "hari-kancharla/sdcguard")
BOARD_URL = os.environ.get("SDCGUARD_BOARD_URL", f"https://{BOARD_REPO.split('/')[0]}.github.io/{BOARD_REPO.split('/')[-1]}/board/")
REFERENCE_URL = os.environ.get("SDCGUARD_REFERENCE_URL",
                               f"https://raw.githubusercontent.com/{BOARD_REPO}/main/docs/board/reference.json")


# --- small helpers ------------------------------------------------------------------------------------------

def _torch():
    import torch
    return torch


def _sync(dev) -> None:
    if dev.type == "cuda":
        _torch().cuda.synchronize(dev)


def _elapsed(dev, fn) -> float:
    """Seconds fn() takes on dev (CUDA events on GPUs)."""
    torch = _torch()
    if dev.type == "cuda":
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        return a.elapsed_time(b) / 1e3
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def _median(v):
    v = sorted(x for x in v if x is not None)
    return v[len(v) // 2] if v else None


def _gpu_dtype(dev):
    torch = _torch()
    if dev.type != "cuda":
        return torch.bfloat16
    with torch.cuda.device(dev):
        try:
            ok = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:
            ok = torch.cuda.get_device_capability()[0] >= 8
    return torch.bfloat16 if ok else torch.float16


def _mat_size(dev) -> int:
    torch = _torch()
    if dev.type != "cuda":
        return 384
    gb = torch.cuda.get_device_properties(dev).total_memory / 2**30
    return 8192 if gb >= 12 else 4096


# --- the tests ----------------------------------------------------------------------------------------------

def memory_test(dev, seconds: float, share: float = 0.8) -> dict:
    """Write patterns into most of the free memory and read them back. Counts words that came back wrong."""
    torch = _torch()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(dev)
        budget = int(free * share) - (512 << 20)
    else:
        budget, total = 64 << 20, None
    chunk = 64 << 20 if dev.type == "cuda" else 8 << 20            # bytes per tensor
    n = chunk // 4
    if budget < 3 * chunk:
        return {"skipped": "not enough free memory"}
    idx = torch.arange(n, dtype=torch.int32, device=dev)
    scratch = torch.empty(n, dtype=torch.int32, device=dev)
    blocks = []
    try:
        while (len(blocks) + 2) * chunk < budget:
            blocks.append(torch.empty(n, dtype=torch.int32, device=dev))
    except torch.cuda.OutOfMemoryError:
        if blocks:
            blocks.pop()
        torch.cuda.empty_cache()
    if not blocks:
        return {"skipped": "no memory block could be allocated"}

    def pattern(out, j, p):
        torch.mul(idx, -1640531535, out=out)                        # Knuth's multiplicative hash, wraps
        out.add_(j * 40503 + p * 977 + 12345)
        if p % 2:
            out.bitwise_not_()
        return out

    t_end, passes, bad, words = time.time() + seconds, 0, 0, 0
    error = None
    try:
        while True:
            for j, b in enumerate(blocks):
                pattern(b, j, passes)
            _sync(dev)
            for j, b in enumerate(blocks):
                bad += int(torch.ne(b, pattern(scratch, j, passes)).sum())
                words += n
            passes += 1
            if time.time() >= t_end and passes >= 2:
                break
    except Exception as e:  # noqa: BLE001
        from .screen import error_text
        error = error_text(e)
    gib = len(blocks) * chunk / 2**30
    blocks.clear()
    if dev.type == "cuda" and error is None:
        try:
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            from .screen import error_text
            error = error_text(e)
    return {"gib_tested": round(gib, 2), "of_gib": round(total / 2**30, 1) if total else None, "passes": passes,
            "words_checked": words, "wrong_words": bad, "error": error}


def matmul_speed(dev, dtype, seconds: float) -> dict:
    """Sustained matmul throughput: many big matmuls, timed in windows of about half a second."""
    torch = _torch()
    n = _mat_size(dev)
    g = torch.Generator(device="cpu").manual_seed(11)
    A = torch.randn(n, n, generator=g).to(dev, dtype)
    B = torch.randn(n, n, generator=g).to(dev, dtype)
    C = torch.empty(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        torch.mm(A, B, out=C)
    _sync(dev)
    one = _elapsed(dev, lambda: torch.mm(A, B, out=C))
    k = max(1, int(0.5 / max(one, 1e-6)))

    def burst():
        for _ in range(k):
            torch.mm(A, B, out=C)

    rates, t_end = [], time.time() + seconds
    while time.time() < t_end or len(rates) < 3:
        rates.append(2 * n ** 3 * k / _elapsed(dev, burst) / 1e12)
    head, tail = _median(rates[:3]), _median(rates[-3:])
    # Keep measurements in the record. Decimal-place rounding can turn a slow
    # positive CPU result into zero, removing it from cards and certificates.
    return {"tflops": _median(rates), "first_tflops": head, "last_tflops": tail,
            "drop_pct": round(max(0.0, (head - tail) / head * 100), 1) if head else None, "size": n,
            "dtype": {"bfloat16": "bf16", "float16": "fp16"}.get(str(dtype).replace("torch.", ""), str(dtype)),
            "windows": len(rates)}


def bandwidth(dev, seconds: float) -> dict:
    """Device memory copy bandwidth (reads plus writes)."""
    torch = _torch()
    if dev.type == "cuda":
        gb = torch.cuda.get_device_properties(dev).total_memory / 2**30
        size = (1 << 30) if gb >= 24 else (256 << 20)
    else:
        size = 64 << 20
    src = torch.empty(size, dtype=torch.uint8, device=dev).fill_(7)
    dst = torch.empty_like(src)
    dst.copy_(src)
    _sync(dev)
    one = _elapsed(dev, lambda: dst.copy_(src))
    k = max(1, int(0.25 / max(one, 1e-6)))

    def burst():
        for _ in range(k):
            dst.copy_(src)

    rates, t_end = [], time.time() + seconds
    while time.time() < t_end or len(rates) < 3:
        rates.append(2 * size * k / _elapsed(dev, burst) / 1e9)
    return {"gbs": round(_median(rates), 1), "buffer_mib": size >> 20}


def stress_check(dev, dtype, seconds: float) -> dict:
    """Big matmuls back to back at full power, every output checked against its checksums."""
    torch = _torch()
    from .checksum import check_matmul
    from .guard import _exact_reductions_off, _exact_reductions_on
    n = _mat_size(dev)
    g = torch.Generator(device="cpu").manual_seed(12)
    B = torch.randn(n, n, generator=g).to(dev, dtype)
    restore = _exact_reductions_on()
    checked = wrong = unjudged = 0
    worst = 0.0
    failures = []                                   # (result, operands) of the first wrong ones, for evidence

    def one():
        nonlocal checked, wrong, unjudged, worst
        A = torch.randn(n, n, device=dev, dtype=torch.float32).to(dtype)    # new data every time
        C = A @ B
        r = check_matmul(A, B, C)
        checked += 1
        if r is not None:
            if not r.ok:
                wrong += 1
                if len(failures) < 2:
                    failures.append((r, (A.detach().cpu(), B.detach().cpu(), C.detach().cpu(), None, 1.0, 1.0,
                                         "aten::mm", "stress check")))
            elif not r.judged:
                unjudged += 1
            elif math.isfinite(r.max_z):
                worst = max(worst, r.max_z)
        else:
            unjudged += 1

    error = None
    try:
        one()                                   # the first check compiles the GPU kernels: not timed
        _sync(dev)
        t_end = time.time() + seconds
        while time.time() < t_end or checked < 3:
            one()
    except Exception as e:  # noqa: BLE001
        # Preserve completed checks and CPU evidence when this same phase later
        # crashes. The caller stops subsequent GPU work and reports these counts.
        from .screen import error_text
        error = error_text(e)
    finally:
        _exact_reductions_off(restore)
    return {"matmuls_checked": checked, "wrong": wrong, "unjudged": unjudged, "worst_clean_score": round(worst, 3),
            "size": n, "failures": failures, "error": error}


def software() -> dict:
    torch = _torch()
    import importlib.metadata as md
    cublas = None
    names = ("nvidia-cublas", "nvidia-cublas-cu13", "nvidia-cublas-cu12")
    if (torch.version.cuda or "").startswith("12"):
        names = ("nvidia-cublas-cu12", "nvidia-cublas")
    for name in names:
        try:
            cublas = md.version(name)
            break
        except md.PackageNotFoundError:
            pass
    from . import __version__
    return {"sdcguard": __version__, "torch": torch.__version__, "cuda": torch.version.cuda, "cublas": cublas,
            "matmul": gemm_settings()}


# The PyTorch settings that change which GEMM kernel runs or how it accumulates. Each is read on its own: a
# setting this PyTorch doesn't have is left out (unknown), never guessed. The *_split_k halves exist from
# PyTorch 2.10; fp32_precision is PyTorch 2.9's new TF32 API (reading allow_tf32 after it was used raises).
_GEMM_FLAGS = ("fp32_precision", "allow_tf32", "allow_bf16_reduced_precision_reduction",
               "allow_bf16_reduced_precision_reduction_split_k", "allow_fp16_reduced_precision_reduction",
               "allow_fp16_reduced_precision_reduction_split_k", "allow_fp16_accumulation",
               "prefer_cublaslt_grouped_gemm")
# environment variables that pick or change GEMM kernels: cuBLAS and cuBLASLt switches (FP32 emulation, TF32
# override, workspace, logging aside) and PyTorch's BLAS backend choice. They are read when the library starts,
# so a replay can't set them for itself: they are compared, and the casefile says when they differ
_GEMM_ENV = ("CUBLAS_", "CUBLASLT_", "NVIDIA_TF32", "TORCH_BLAS_", "TORCH_CUBLAS", "TORCH_ALLOW_TF32",
             "DISABLE_ADDMM_CUDA_LT", "PYTORCH_TUNABLEOP")


def gemm_settings() -> dict:
    """What decides which units a matmul really runs on and how it sums, as set right now: TF32 (both APIs), the
    bf16/fp16 reduced-precision and split-K reductions, fp16 accumulation, the preferred BLAS library, the
    grouped-GEMM backend, and the environment variables that switch cuBLAS kernels (CUDA 12.9 and later can
    emulate FP32/FP64 on lower-precision tensor cores through them). Recorded with evidence when the checked
    matmul ran, and applied by `sdcguard replay` so a recomputation runs the same way."""
    torch = _torch()
    m = torch.backends.cuda.matmul
    mm: dict = {}
    for name in _GEMM_FLAGS:
        try:
            v = getattr(m, name)
        except Exception:  # noqa: BLE001  (not in this PyTorch, or unreadable after the other TF32 API was used)
            continue
        mm[name] = str(v) if name == "fp32_precision" else bool(v)
    if "allow_tf32" not in mm and "fp32_precision" in mm:
        mm["allow_tf32"] = mm["fp32_precision"] == "tf32"
    try:   # (reading resolves PyTorch's "default" to the backend it stands for, as the first GEMM would)
        mm["blas_library"] = str(getattr(torch.backends.cuda.preferred_blas_library(), "name", "")).lower() or None
    except Exception:  # noqa: BLE001
        pass
    try:   # TunableOp picks each GEMM's implementation by timing them: a replay needs the same choice
        if torch.cuda.is_available():
            mm["tunableop"] = bool(torch.cuda.tunable.is_enabled())
    except Exception:  # noqa: BLE001
        pass
    # these settings go on the shareable card: short switch values are kept (NVIDIA_TF32_OVERRIDE=0), anything
    # else (a log file path, a hostname) is recorded only as set
    env = {k: (v if re.fullmatch(r"[A-Za-z0-9_.,:+-]{1,24}", v) else "<set>") for k, v in sorted(os.environ.items())
           if k.startswith(_GEMM_ENV) and k != "CUBLAS_WORKSPACE_CONFIG"}
    if env:
        mm["env"] = env
    return mm


def tf32_of(mm: Optional[dict]) -> Optional[bool]:
    """Whether fp32 matmuls run on TF32 under these recorded settings (the new API decides once it was used)."""
    if not isinstance(mm, dict):
        return None
    p = mm.get("fp32_precision")
    if p not in (None, "none"):
        return p == "tf32"
    return bool(mm["allow_tf32"]) if "allow_tf32" in mm else None


def gemm_effective(mm: Optional[dict]) -> Optional[dict]:
    """The recorded GEMM settings as what they do (TF32 on or off, each reduction mode, ...), so two records
    written through different APIs compare equal when the matmuls ran the same way. None when nothing was
    recorded; a setting one PyTorch didn't have is left out."""
    if not isinstance(mm, dict) or not mm:
        return None
    out = {"tf32": tf32_of(mm)}
    for kind in ("bf16", "fp16"):
        key = f"allow_{kind}_reduced_precision_reduction"
        if key in mm:
            out[f"{kind}_reduced_precision_reduction"] = bool(mm[key])
            if not mm[key]:
                out[f"{kind}_split_k"] = bool(mm.get(key + "_split_k", True))
    for key in ("allow_fp16_accumulation", "prefer_cublaslt_grouped_gemm", "blas_library", "tunableop"):
        if key in mm:
            out[key] = mm[key]
    out["env"] = dict(mm.get("env") or {})
    return out


@contextlib.contextmanager
def gemm_settings_applied(recorded: Optional[dict]):
    """Run with the GEMM settings an evidence file recorded, as far as a process can set them for itself (the
    flags; not the environment variables, which cuBLAS reads when it starts). Yields what was applied; puts every
    setting back afterwards. Settings this PyTorch doesn't have are skipped."""
    torch = _torch()
    m = torch.backends.cuda.matmul
    undo, applied = [], {}

    def put(name, value, setter, getter):
        try:
            before = getter()
            if before == value:
                applied[name] = value
                return
            setter(value)
            undo.append((setter, before))
            applied[name] = value
        except Exception:  # noqa: BLE001
            pass

    rec = recorded if isinstance(recorded, dict) else {}
    try:
        want = tf32_of(rec)
        if want is not None and tf32_of(gemm_settings()) != want:
            # either API changes both readings (the legacy flag sets fp32_precision): put back both as they were
            before = {k: v for k, v in gemm_settings().items() if k in ("fp32_precision", "allow_tf32")}

            def restore_tf32(before=before):
                if "allow_tf32" in before:
                    try:
                        m.allow_tf32 = before["allow_tf32"]
                    except Exception:  # noqa: BLE001
                        pass
                if "fp32_precision" in before:
                    m.fp32_precision = before["fp32_precision"]
            try:
                if "fp32_precision" in before:
                    m.fp32_precision = "tf32" if want else "ieee"
                else:
                    m.allow_tf32 = want
                undo.append((lambda _v: restore_tf32(), None))
                applied["tf32"] = want
            except Exception:  # noqa: BLE001
                restore_tf32()
        elif want is not None:
            applied["tf32"] = want
        for kind in ("bf16", "fp16"):
            key = f"allow_{kind}_reduced_precision_reduction"
            if key not in rec:
                continue
            want = (bool(rec[key]), bool(rec.get(key + "_split_k", True)))

            def get(key=key):
                return (bool(getattr(m, key)), bool(getattr(m, key + "_split_k", True)))

            def set_(v, key=key):
                try:
                    setattr(m, key, v)                    # PyTorch 2.10+: (reduced precision, split-K)
                except TypeError:
                    if not v[1]:
                        raise
                    setattr(m, key, v[0])
            put(key, want, set_, get)
        for key in ("allow_fp16_accumulation", "prefer_cublaslt_grouped_gemm"):
            if key in rec:
                put(key, bool(rec[key]), lambda v, key=key: setattr(m, key, v), lambda key=key: bool(getattr(m, key)))
        if rec.get("blas_library"):
            lib = torch.backends.cuda.preferred_blas_library
            put("blas_library", rec["blas_library"], lambda v: lib(v),
                lambda: str(getattr(lib(), "name", "")).lower())
        yield applied
    finally:
        for setter, before in reversed(undo):
            try:
                setter(before)
            except Exception:  # noqa: BLE001
                pass


class Fingerprinter:
    """A fixed model (seeded weights) and a fixed input. Its outputs, hashed, are the GPU's fingerprint:
    the same GPU model running the same software should produce bit-identical outputs, run after run and
    GPU after GPU. Covers every kind of operation the model runs (embeddings, layer norm, attention,
    softmax, GELU, matmuls), not just matmuls."""

    def __init__(self, dev, dtype, seed: int = 0):
        torch = _torch()
        from .screen import ScreenModel
        L, d, h, seq, batch = FP_CONFIG["gpu" if dev.type == "cuda" else "cpu"]
        self.sdpa_shape = (batch, h, seq, d // h)
        vocab = 32000 if dev.type == "cuda" else 512
        torch.manual_seed(1000 + seed)
        m = ScreenModel(vocab, L, d, h, seq)
        hw = hashlib.sha256()
        for k, v in m.state_dict().items():
            hw.update(k.encode())
            hw.update(v.detach().contiguous().numpy().tobytes())
        self.weights = hw.hexdigest()[:16]
        self.model = m.to(dev, dtype).eval()
        gen = torch.Generator(device="cpu").manual_seed(2000 + seed)
        self.x = torch.randint(0, vocab, (batch, seq), generator=gen).to(dev)
        # fused attention too (FlashAttention or memory-efficient kernels on GPUs), which the model above
        # writes out as separate matmuls
        self.qkv = [torch.randn(*self.sdpa_shape, generator=gen).to(dev, dtype) for _ in range(3)]
        self.dev = dev
        self.config = {"layers": L, "width": d, "heads": h, "seq": seq, "batch": batch, "vocab": vocab,
                       "dtype": str(dtype).replace("torch.", ""), "seed": seed, "version": FP_VERSION,
                       "exact_reductions": True}
        self.hw = {}
        if dev.type == "cuda":
            props = torch.cuda.get_device_properties(dev)
            self.hw = {"sms": props.multi_processor_count, "cudnn": torch.backends.cudnn.version(),
                       "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}

    def key(self, gpu_name: Optional[str], sw: dict) -> str:
        """Runs with the same key are expected to give the same fingerprint."""
        k = {"gpu": gpu_name, "config": self.config, "weights": self.weights, "hw": self.hw,
             "torch": sw.get("torch"), "cuda": sw.get("cuda"), "cublas": sw.get("cublas")}
        return hashlib.sha256(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]

    def run(self) -> dict:
        torch = _torch()
        outs = []
        hooks = [b.register_forward_hook(lambda _m, _i, o: outs.append(o)) for b in self.model.blocks]
        was = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        # the same cuBLAS settings on every run, whatever other code (the serving check) set in between
        mm = torch.backends.cuda.matmul
        saved = (mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction)
        mm.allow_bf16_reduced_precision_reduction = False
        mm.allow_fp16_reduced_precision_reduction = False
        try:
            with torch.no_grad():
                logits = self.model(self.x)
                att = torch.nn.functional.scaled_dot_product_attention(*self.qkv, is_causal=True)
            _sync(self.dev)

            def h(t, n):
                return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:n]

            lg, sd = h(logits, 16), h(att, 16)
            return {"output": hashlib.sha256((lg + sd).encode()).hexdigest()[:16], "logits": lg, "sdpa": sd,
                    "layers": [h(o, 8) for o in outs]}
        finally:
            for k in hooks:
                k.remove()
            mm.allow_bf16_reduced_precision_reduction, mm.allow_fp16_reduced_precision_reduction = saved
            torch.use_deterministic_algorithms(was, warn_only=warn)


def fetch_reference(timeout: float = 4.0) -> Optional[dict]:
    """The public board's reference data (fingerprint counts, per-model medians). A GET request only;
    nothing about this machine is sent. None when offline."""
    try:
        with urllib.request.urlopen(REFERENCE_URL, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:  # noqa: BLE001
        return None


# --- one GPU ------------------------------------------------------------------------------------------------

def _notes(static: dict, before: dict, after: dict, live: dict, speed: dict, spec: dict) -> list:
    from .gpuinfo import BAD_THROTTLE, _num
    notes = []          # (uncorrectable ECC errors and row-remapping failures: hardware_errors(), which fail the check)
    if (after.get("remapped_rows.pending") or "").lower() in ("yes", "1", "true") or \
            (after.get("retired_pages.pending") or "").lower() in ("yes", "1", "true"):
        notes.append(["warn", "a memory repair is pending (it takes effect after a GPU reset)"])
    share = live.get("throttle_share_busy") or {}
    slow = {k: v for k, v in share.items() if k in BAD_THROTTLE and v >= 0.05}
    if slow:
        what = ", ".join(f"{k.replace('_', ' ')} {round(v * 100)}% of the time" for k, v in slow.items())
        notes.append(["warn", f"the GPU slowed itself down under load: {what}"])
    elif share.get("sw_power_cap", 0) >= 0.2:
        notes.append(["info", "clocks held back by the power limit at full load (normal for big matmuls)"])
    wmax, wnow = _num(static.get("pcie.link.width.max")), live.get("pcie_width_busy")
    if wmax and wnow and wnow < wmax:
        notes.append(["warn", f"PCIe link running at x{wnow} under load; the card supports x{int(wmax)}"])
    if speed.get("drop_pct") is not None and speed["drop_pct"] >= 10:
        notes.append(["warn", f"matmul speed dropped {speed['drop_pct']}% from the first to the last seconds of the test"])
    if (static.get("ecc.mode.current") or "").lower() == "disabled":
        notes.append(["info", "ECC is turned off on this GPU"])
    return notes


def check_one(device, minutes: float = 5.0, size: Optional[str] = None, out_root: str = "sdcguard_check",
              seed: int = 0, simulate_faults: float = 0.0, provider: Optional[str] = None,
              reference: Optional[dict] = None, key_path: Optional[str] = None, issuer: str = "",
              progress=None, tag: Optional[str] = None) -> dict:
    torch = _torch()
    from . import gpuinfo
    from .screen import SIZES, screen_one
    say = progress or (lambda s: None)
    dev = torch.device(device)
    if dev.type == "cuda":                        # CUDA events and Triton launch on the current device
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)
    t_start = time.time()
    total = 60.0 * minutes
    budget = {"memory": 0.04, "speed": 0.08, "bandwidth": 0.02, "stress": 0.06, "modes": 0.05}
    secs = {k: max(0.2, v * total) for k, v in budget.items()}
    idx = None
    if dev.type == "cuda":                        # nvidia-smi by UUID: its index order isn't CUDA's
        from .events import device_identity
        ident = device_identity(dev)
        idx = ident.get("gpu_uuid") or ident.get("gpu_index")
    static = gpuinfo.query(idx, gpuinfo.STATIC)
    before = gpuinfo.health(idx)
    sampler = gpuinfo.Sampler(idx).start()
    dtype = _gpu_dtype(dev)
    name = torch.cuda.get_device_properties(dev).name if dev.type == "cuda" else "CPU"
    if size is None:
        gb = torch.cuda.get_device_properties(dev).total_memory / 2**30 if dev.type == "cuda" else 0
        size = "1b" if gb >= 70 else ("small" if gb >= 12 else "tiny")
    if size not in SIZES:
        raise ValueError(f"--size must be one of {', '.join(SIZES)}")
    sw = software()
    from .modes import MODES, screen as modes_screen
    from .screen import _save_evidence, device_dir, error_text
    from .modes import save_evidence as save_mode_evidence
    out_dir, ident = device_dir(dev, out_root, tag)
    run_id = uuid.uuid4().hex
    if tag is not None:          # (a child of a multi-GPU check: tell the parent where this GPU's results go now,
        with open(os.path.join(out_root, f".child_{tag}.json"), "w") as f:     # in case this process dies)
            json.dump({"out_dir": out_dir, "run_id": run_id}, f)
    _started(out_dir, {"result": "INCOMPLETE", "running": True, "run_id": run_id, "device": str(dev), "gpu": ident,
                       "simulated_faults": simulate_faults > 0,
                       "notes": [["bad", "this check hadn't finished (still running, or stopped before it could "
                                         "report)"]]})
    ev_meta = {"sdcguard": sw.get("sdcguard"), "torch": sw.get("torch"), "seed": seed, "run_id": run_id}
    evidence: list = []
    stopped: dict = {}

    def phase(label: str, fn):
        """Run one part of the check. A failing GPU often computes wrong results and then raises a CUDA error
        (which usually ends the CUDA context): after the first error nothing more runs on the GPU, but what
        was found so far is kept and reported, and the result is never PASS."""
        if stopped:
            return None
        try:
            value = fn()
            if isinstance(value, dict) and value.get("error"):
                stopped.update(phase=label, error=value["error"])
                say(f"{name}: {label} stopped with an error ({stopped['error']}); reporting what was found so far")
            return value
        except Exception as e:  # noqa: BLE001
            stopped.update(phase=label, error=error_text(e))
            say(f"{name}: {label} stopped with an error ({stopped['error']}); reporting what was found so far")
            return None

    say(f"{name}: fingerprint run 1 of 3")
    fp = phase("fingerprint", lambda: Fingerprinter(dev, dtype, seed))
    prints = []
    if fp is not None:
        p0 = phase("fingerprint", fp.run)
        if p0 is not None:
            prints.append(p0)
    say(f"{name}: memory patterns ({secs['memory']:.0f} s)")
    mem = phase("memory test", lambda: memory_test(dev, secs["memory"])) or {"skipped": "not run"}
    say(f"{name}: matmul speed ({secs['speed']:.0f} s) and memory bandwidth")
    speed = phase("speed", lambda: matmul_speed(dev, dtype, secs["speed"])) or {}
    bw = phase("bandwidth", lambda: bandwidth(dev, secs["bandwidth"])) or {}
    say(f"{name}: checked matmuls at full power ({secs['stress']:.0f} s)")
    stress = phase("stress", lambda: stress_check(dev, dtype, secs["stress"])) or \
        {"matmuls_checked": 0, "wrong": 0, "unjudged": 0, "worst_clean_score": 0.0, "not_run": True}
    # evidence is saved as soon as each part finishes, so a later crash can't lose it
    for res, ops in stress.pop("failures", []):
        evidence.append(_save_evidence(out_dir, len(evidence), res, ops, ident, {**ev_meta, "phase": "stress"}))
    say(f"{name}: matmuls in {len(MODES)} math modes against a float64 reference ({secs['modes']:.0f} s)")
    math_modes = phase("math modes", lambda: modes_screen(dev, secs["modes"], seed=seed)) or \
        {"matmuls": 0, "wrong": 0, "modes": {}, "ran": [], "not_checked": {}, "errors": {}, "not_run": True}
    for i, fail in enumerate(math_modes.pop("failures", [])):
        evidence.append(save_mode_evidence(out_dir, i, fail, ident, {**ev_meta, "phase": "math modes"}))
    library: dict = {"not_run": "no CUDA GPU"}
    if dev.type == "cuda" and not stopped:
        say(f"{name}: every cuBLASLt algorithm for {len(LIBRARY_SHAPES)} matmul shapes against a float64 reference")
        library = phase("library", lambda: library_sweep(dev, dtype)) or {"not_run": "stopped"}
    if fp is not None and prints:
        p1 = phase("fingerprint", fp.run)
        if p1 is not None:
            prints.append(p1)
    left = max(0.05, (total - (time.time() - t_start) - 3) / 60)
    summary: dict = {}
    if not stopped:
        say(f"{name}: training and serving workload, every matmul checked ({left:.1f} min)")
        # screen_one reports a workload that stops on an error itself (with what it found before the error)
        summary = phase("workload", lambda: screen_one(dev, left, size, out_root, seed,
                                                       simulate_faults=simulate_faults, tag=tag, run_id=run_id)) or {}
        if summary.get("error"):
            stopped.update(phase="workload", error=summary["error"])
    if fp is not None and len(prints) == 2:
        p2 = phase("fingerprint", fp.run)
        if p2 is not None:
            prints.append(p2)
    try:
        sampler.stop()
        live = sampler.summary()
    except Exception:  # noqa: BLE001
        live = {}
    after = gpuinfo.health(idx)
    spec = gpuinfo.datasheet(name)
    out_dir = summary.get("out_dir") or out_dir
    os.makedirs(out_dir, exist_ok=True)

    wrong = {"workload_training": summary.get("corrupted_results_training") or 0,
             "workload_serving": summary.get("corrupted_results_serving") or 0,
             "stress": stress["wrong"], "math_modes": math_modes["wrong"], "memory_words": mem.get("wrong_words") or 0}
    # repeatability needs at least two fingerprint runs: with fewer it wasn't checked (None)
    repeatable = all(p == prints[0] for p in prints) if len(prints) >= 2 else None
    worst = max(summary.get("worst_clean_z") or 0.0, stress["worst_clean_score"])
    checked = (summary.get("matmuls_checked") or 0) + (summary.get("linear_layer_checks_while_serving") or 0) + \
        stress["matmuls_checked"] + math_modes["matmuls"]
    evidence[:0] = list(summary.get("evidence") or [])
    unj_train = summary.get("unjudged_training_checks") or 0
    skipped_train = summary.get("skipped_training_checks") or 0
    serve_nf = summary.get("serving_passes_with_nonfinite_output") or 0
    mem_ran = bool(mem.get("gib_tested")) and bool(mem.get("words_checked"))
    hw_errors = hardware_errors(before, after)
    incomplete = (bool(stopped) or not summary or summary.get("result") == "INCOMPLETE"
                  or verdict.from_counts(0, stress["matmuls_checked"], stress["unjudged"]) == "INCOMPLETE"
                  or bool(math_modes.get("errors")) or not math_modes.get("ran") and not math_modes.get("unsupported")
                  or not mem_ran or repeatable is None)
    # the workload's own verdict counts too (it judges its counts the same way, so they normally agree)
    workload_fail = summary.get("result") == "FAIL"
    result = ("FAIL" if any(wrong.values()) or hw_errors or workload_fail else "SUSPECT" if repeatable is False else
              "INCOMPLETE" if incomplete else "PASS")
    notes = _notes(static, before, after, live, speed, spec)
    if workload_fail and not (wrong["workload_training"] or wrong["workload_serving"]):
        notes.insert(0, ["bad", "the workload found corrupted results it couldn't count by phase (see summary.json)"])
    for m, err in (math_modes.get("errors") or {}).items():
        notes.insert(0, ["bad", f"the {m} math-mode check couldn't run ({err[:80]}): that part wasn't checked"])
    notes[:0] = verdict.unjudged_note(unj_train, summary.get("matmuls_checked") or 0, "workload training")
    notes[:0] = verdict.unjudged_note(stress["unjudged"], stress["matmuls_checked"], "stress")
    if skipped_train:
        notes.insert(0, ["bad", f"{skipped_train:,} workload training checks were skipped (not enough GPU memory "
                                "for the check): those results weren't checked"])
    if serve_nf:
        notes.insert(0, ["bad", f"{serve_nf} serving passes gave output that wasn't finite (the serving check can't "
                                "judge those)"])
    serve_unj = summary.get("unjudged_serving_checksums") or 0
    if serve_unj:
        notes.insert(0, ["bad", f"{serve_unj:,} serving checksums couldn't be judged (outputs or sums too large or not "
                                "finite, in a way the inputs can explain): those results weren't checked"])
    never = summary.get("serving_layers_never_checked") or []
    if never:
        notes.insert(0, ["bad", f"{len(never)} serving layers never ran their check ({', '.join(never[:3])}"
                                f"{' ...' if len(never) > 3 else ''}): those layers weren't checked"])
    if not mem_ran and not stopped:
        why = mem.get("skipped") or "no memory could be allocated for it"
        notes.insert(0, ["bad", f"the memory pattern test didn't run ({why}): GPU memory wasn't tested"])
    if repeatable is False:
        first = next(i for i, p in enumerate(prints) if p != prints[0])
        lay = next((j for j, (a, b) in enumerate(zip(prints[0]["layers"], prints[first]["layers"])) if a != b), None)
        where = (f" (first difference after layer {lay})" if lay is not None else
                 " (in fused attention)" if prints[0].get("sdpa") != prints[first].get("sdpa") else "")
        notes.insert(0, ["bad", "the same model on the same input gave different outputs on this GPU" + where])
    for text in hw_errors:
        notes.insert(0, ["bad", text])
    known: dict = {}
    if dev.type == "cuda":
        try:
            from . import known_issues
            known = known_issues.report(dev)
        except Exception as e:  # noqa: BLE001
            known = {"error": error_text(e)}
    for m in (known.get("matches") or []):
        if m["status"] == "applies":
            notes.insert(0, ["bad", f"NVIDIA cuBLAS known issue {m['id']} applies here: {m['what']}"
                                    + (f" (workaround: {m['workaround']})" if m.get("workaround") else "")])
    maybe = [m["id"] for m in (known.get("matches") or []) if m["status"] == "may apply"]
    if maybe:
        notes.append(["info", f"{len(maybe)} listed cuBLAS wrong-result issue(s) may apply, depending on what your "
                              f"jobs run ({', '.join(maybe)}): `sdcguard issues` says when"])
    if library.get("wrong"):
        w0 = library["wrong"][0]
        notes.insert(0, ["bad", f"{len(library['wrong'])} cuBLASLt algorithm(s) computed wrong results here (first: "
                                f"algorithm {w0.get('algo_id')}, {w0.get('cluster_ctas')} CTA cluster, shape "
                                f"{w0.get('shape')}): a library or configuration defect (NVIDIA known issue 6862950 "
                                "under MPS looked like this). Frameworks run an algorithm only when cuBLAS picks it "
                                "for a shape; `sdcguard blas` shows every one"])
    health_read = any(k.startswith(("ecc.", "remapped_rows", "retired_pages")) and v is not None
                      and str(v).strip() not in ("", "[N/A]", "N/A", "[Not Supported]")
                      for k, v in {**(before or {}), **(after or {})}.items())
    if dev.type == "cuda" and not health_read:
        notes.insert(0, ["warn", "the driver's ECC and row-remapping counters couldn't be read (no nvidia-smi here, "
                                 "a MIG instance, or a GPU without ECC): memory errors the driver saw aren't part "
                                 "of this result"])
    if stopped:
        notes.insert(0, ["bad", f"the check stopped in the {stopped['phase']} part with an error ({stopped['error']}); "
                                "the parts after it didn't run"])
    fp_key = fp.key(name, sw) if fp is not None and prints else None
    prov = gpuinfo.detect_provider()
    if provider:
        prov = {"name": provider, "detected_from": "--provider"}
    full = {
        "schema": SCHEMA.replace("check/", "check-full/"), "result": result, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "minutes": round((time.time() - t_start) / 60, 2), "device": str(dev), "run_id": run_id,
        "gpu": {**(summary.get("gpu") or ident), **{k: static.get(k) for k in static}}, "software": sw, "provider": prov,
        "wrong": wrong, "hardware_errors": hw_errors, "checked_results": checked, "repeatable": repeatable,
        "fingerprints": prints, "fingerprint_key": fp_key, "fingerprint_weights": fp.weights if fp else None,
        "fingerprint_config": fp.config if fp else None,
        "worst_clean_score": round(worst, 3), "alarm_line": 6.0,
        "memory": mem, "speed": speed, "bandwidth": bw, "stress": stress, "math_modes": math_modes,
        "library": library, "known_issues": known, "datasheet": spec,
        "health_before": before, "health_after": after, "health_counters_read": health_read, "live": live,
        "notes": notes, "board": None,
        "workload": {k: summary.get(k) for k in ("size", "seed", "training_steps", "serving_passes", "matmuls_checked",
                                                  "linear_layer_checks_while_serving", "approx_matmul_tflop",
                                                  "evidence", "dtype", "unjudged_training_checks",
                                                  "skipped_training_checks", "serving_passes_with_nonfinite_output",
                                                  "serving_linear_layers", "serving_layers_never_checked",
                                                  "unjudged_serving_checksums", "error")},
        "evidence": evidence, "unjudged": unj_train + stress["unjudged"] + skipped_train,
        "stopped": stopped or None,
        "simulated_faults": bool(summary.get("SIMULATED_FAULTS")) or simulate_faults > 0, "out_dir": out_dir,
    }
    if reference:
        try:
            apply_board(full, reference)
        except Exception as e:  # noqa: BLE001  (a malformed reference file mustn't lose the result)
            full["notes"].append(["info", f"couldn't compare with the public board's data ({error_text(e)})"])
    write_result(full, key_path, issuer)
    return full


LIBRARY_SHAPES = [(4096, 4096, 512), (1000, 2048, 16)]      # (out, in, tokens): a projection and an odd decode


def library_sweep(dev, dtype) -> dict:
    """Every cuBLASLt algorithm the heuristic offers for a few shapes, each checked against float64 (blas.py).
    Reported with notes, not in the verdict yet: a wrong algorithm is a library or configuration defect, and the
    sweep's own limits are still being measured on GPUs."""
    from . import blas
    ok, why = blas.available()
    if not ok:
        return {"not_run": why}
    torch = _torch()
    dt = dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    tested, wrong, problems = 0, [], []
    for m, k, n in LIBRARY_SHAPES:
        try:
            r = blas.sweep(m, k, n, dtype=dt, device=dev, max_algos=16)
        except Exception as e:  # noqa: BLE001  (the rest of the check goes on)
            problems.append(f"{m}x{k}x{n}: {type(e).__name__}: {str(e)[:100]}")
            continue
        if r.get("error"):
            problems.append(f"{m}x{k}x{n}: {r['error']}")
            continue
        tested += r.get("algorithms_tested", 0)
        wrong += [{**w, "shape": f"{m}x{k}x{n}"} for w in r.get("wrong_algorithms", [])]
        # an algorithm whose call returned an error status wasn't tested: never PASS, never a wrong result
        problems += [f"{m}x{k}x{n}: algorithm {x.get('algo_id')} didn't run ({x.get('why')})"
                     for x in r.get("algorithms_not_run", [])]
    verdict = "FAIL" if wrong else ("PASS" if tested and not problems else "COULD NOT TEST")
    return {"verdict": verdict, "algorithms_tested": tested, "wrong": wrong, "problems": problems,
            "dtype": str(dt).replace("torch.", "")}


def hardware_errors(before: dict, after: dict) -> list:
    """Memory faults the driver itself reported during the check. NVIDIA's guidance treats an uncorrectable ECC
    error as an event that needs action and a row-remapping failure as grounds to replace the GPU, so either
    one fails the check even when no numerical mismatch was detected."""
    from .gpuinfo import _num
    out = []
    eu0, eu1 = _num(before.get("ecc.errors.uncorrected.volatile.total")), _num(after.get("ecc.errors.uncorrected.volatile.total"))
    if eu0 is not None and eu1 is not None and eu1 > eu0:
        out.append(f"the driver reported {int(eu1 - eu0)} new uncorrectable ECC memory errors during the check")
    if (after.get("remapped_rows.failure") or "").lower() in ("yes", "1", "true"):
        out.append("the driver reports a memory row-remapping failure (errors it could not repair)")
    return out


CERTIFICATE = {"always": False, "valid_days": 30.0}   # set by --certificate and --valid-days (sdcguard certify)


def write_result(full: dict, key_path: Optional[str] = None, issuer: str = "") -> dict:
    """check.json, result.json, the cards and (with a key, an issuer or --certificate) the certificate, for one
    GPU."""
    out_dir = full["out_dir"]
    pub = public(full)
    with open(os.path.join(out_dir, "check.json"), "w") as f:
        json.dump(full, f, indent=2, default=str)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(pub, f, indent=2)
    write_cards(pub, out_dir)
    if not (key_path is not None or issuer or CERTIFICATE["always"]):
        # an earlier run's certificate doesn't describe these files: left here, it would judge them
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(out_dir, "certificate.json"))
    else:
        from . import certify
        try:
            certify.write(_summary_of(full), out_dir, key_path=key_path, issuer=issuer,
                          valid_days=CERTIFICATE["valid_days"], check=pub, full=full)
        except Exception as e:  # noqa: BLE001  (the result stands; the missing certificate is reported)
            from .screen import error_text
            full["certificate_error"] = error_text(e)
            full["notes"].insert(0, ["warn", f"the certificate couldn't be written ({full['certificate_error']})"])
            return write_result(full)
    return pub


def _started(out_dir: str, record: dict) -> None:
    """What a GPU's folder says while its check runs: INCOMPLETE, under this run's id. A check that is killed (a
    deadline, the OOM killer, a hung driver call) leaves this, never the previous run's check.json or certificate
    for a report to read. Its own result replaces it when it finishes."""
    from .screen import _write_atomic
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(out_dir, "certificate.json"))
    rec = {"schema": SCHEMA.replace("check/", "check-full/"),
           "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wrong": {}, "evidence": [], "out_dir": out_dir,
           **record}

    def dump(t):
        with open(t, "w") as f:
            json.dump(rec, f, indent=2, default=str)
    _write_atomic(os.path.join(out_dir, "check.json"), dump)


def _summary_of(full: dict) -> dict:
    """The workload's summary.json for a GPU (what certify.write reads the GPU's identity from), or what
    check.json knows when the workload didn't get to write one."""
    try:
        with open(os.path.join(full["out_dir"], "summary.json")) as f:
            summary = json.load(f)
        if full.get("run_id") and summary.get("run_id") != full["run_id"]:
            raise ValueError("summary belongs to a previous run")
        return summary
    except (OSError, ValueError):
        return {"gpu": full.get("gpu") or {}, "device": full.get("device"), "result": full["result"]}


def apply_board(full: dict, reference: dict) -> None:
    """Compare a result with the public board's reference data (fingerprints of the identical setup,
    medians for the GPU model). The board is built from cards anyone can post, so it only adds notes: it never
    changes the result (a few forged cards could otherwise turn honest PASS results into SUSPECT). When at
    least 3 distinct GPUs, shared by at least 2 different people, all agree on one other fingerprint, the note
    says so plainly."""
    if not isinstance(reference, dict) or not full.get("fingerprints") or not full.get("fingerprint_key"):
        return
    b = _compare(reference, full["fingerprint_key"], full["fingerprints"][0]["output"],
                 (full.get("gpu") or {}).get("gpu_name"))
    full["board"] = b
    if b.get("disagrees_with_majority"):
        full["notes"].insert(0, ["bad", f"this GPU's output fingerprint differs from {b['other_gpus']} other GPUs "
                                        "of the identical setup on the public board (posted results, not "
                                        "verified: rerun, or compare with a GPU you trust)"])
    elif b.get("same_fingerprint") == 0 and b.get("other_gpus"):
        full["notes"].insert(0, ["warn", f"this GPU's output fingerprint differs from {b['other_gpus']} other "
                                         "GPUs of the identical setup on the public board"])


def _counts(v) -> tuple:
    """(distinct GPUs, distinct sources) for one fingerprint in reference.json."""
    if isinstance(v, dict):
        return int(v.get("gpus") or 0), int(v.get("sources") or 0)
    try:
        return int(v), 1
    except (TypeError, ValueError):
        return 0, 0


def _compare(reference: dict, key: str, output: str, name: str) -> dict:
    fps = (reference.get("fingerprints") or {}).get(key) or {}
    same = _counts(fps.get(output))[0] if output in fps else 0
    others = {h: _counts(v) for h, v in fps.items() if h != output}
    other_gpus = sum(g for g, _ in others.values())
    model = (reference.get("models") or {}).get(name) or {}
    lone = len(others) == 1 and next(iter(others.values()))
    return {"same_setup_runs": same + other_gpus, "same_fingerprint": same, "other_gpus": other_gpus,
            "disagrees_with_majority": bool(same == 0 and lone and lone[0] >= 3 and lone[1] >= 2),
            "model_runs": model.get("runs"), "model_median_tflops": model.get("median_tflops"),
            "model_median_bandwidth_gbs": model.get("median_bandwidth_gbs")}


# --- what gets shared ---------------------------------------------------------------------------------------

def public(full: dict) -> dict:
    """The shareable result: what was checked and found, the GPU model and software, no serial number,
    hostname, PCI slot or full UUID (a short hash of the UUID lets the board count distinct GPUs)."""
    g = full.get("gpu") or {}
    uuid = g.get("gpu_uuid") or g.get("uuid") or ""
    spec = full.get("datasheet") or {}
    sp, bw, live = full.get("speed") or {}, full.get("bandwidth") or {}, full.get("live") or {}

    def pct(x, peak):
        return round(100 * x / peak, 1) if x and peak else None

    mem_total = g.get("memory.total")
    pub = {
        "schema": SCHEMA, "result": full["result"], "date": full["time"][:10], "minutes": full["minutes"],
        "gpu": {"name": g.get("gpu_name") or g.get("name") or full.get("device"),
                "memory_gib": round(float(mem_total) / 1024, 1) if mem_total else None,
                "compute_cap": g.get("compute_cap"), "driver": g.get("driver") or g.get("driver_version"),
                "id_hash": hashlib.sha256(uuid.encode()).hexdigest()[:12] if uuid else None},
        "software": full.get("software"), "provider": _provider(full.get("provider")),
        "checked": {"results": full["checked_results"], "unjudged": full.get("unjudged", 0),
                    "serving_nonfinite": (full.get("workload") or {}).get("serving_passes_with_nonfinite_output") or 0,
                    "workload_tflop": (full.get("workload") or {}).get("approx_matmul_tflop"),
                    "memory_gib": (full.get("memory") or {}).get("gib_tested"),
                    "memory_passes": (full.get("memory") or {}).get("passes"),
                    "memory_skipped": None if (full.get("memory") or {}).get("words_checked") else
                    ((full.get("memory") or {}).get("skipped") or "no memory could be allocated for it")},
        "wrong": full["wrong"],
        "hardware_errors": list(full.get("hardware_errors") or []),
        "stopped": full.get("stopped"),
        "headroom": {"worst_clean_score": full["worst_clean_score"], "alarm_line": full["alarm_line"]},
        "repeatable": full["repeatable"],
        "fingerprint": {"key": full.get("fingerprint_key"),
                        "output": (full.get("fingerprints") or [{}])[0].get("output"),
                        "runs": len(full.get("fingerprints") or [])},
        "speed": {"matmul_tflops": sp.get("tflops"), "dtype": sp.get("dtype"),
                  "pct_of_datasheet": pct(sp.get("tflops"), spec.get("matmul_tflops")),
                  "drop_pct": sp.get("drop_pct"), "bandwidth_gbs": bw.get("gbs"),
                  "bandwidth_pct_of_datasheet": pct(bw.get("gbs"), (spec.get("memory_tbs") or 0) * 1000)},
        "health": {"max_temp_c": live.get("max_temp_c"), "max_power_w": live.get("max_power_w"),
                   "power_limit_w": _f(g.get("power.limit")), "sm_clock_mhz": live.get("median_sm_clock_mhz_busy"),
                   "max_sm_clock_mhz": _f(g.get("clocks.max.sm")), "throttle": live.get("throttle_share_busy"),
                   "pcie": _pcie(g, live), "ecc": g.get("ecc.mode.current")},
        "notes": full["notes"], "board": full.get("board"),
        "simulated_faults": full.get("simulated_faults", False),
    }
    mm = full.get("math_modes")
    if mm:
        pub["math_modes"] = {"ran": list(mm.get("ran") or []), "matmuls": mm.get("matmuls") or 0,
                             "wrong": mm.get("wrong") or 0,
                             "not_checked": {k: str(v) for k, v in (mm.get("not_checked") or {}).items()},
                             "errors": {k: str(v) for k, v in (mm.get("errors") or {}).items()},
                             "typical_error_caught": {k: r["typical_error_caught"] for k, r in mm["modes"].items()
                                                      if r.get("typical_error_caught") is not None
                                                      and k in (mm.get("ran") or [])}}
    pub["digest"] = digest(pub)
    return pub


def _provider(p):
    from .gpuinfo import cloud_name
    p = dict(p or {})
    if "underlying_cloud" in p:
        p["underlying_cloud"] = cloud_name(p["underlying_cloud"])
    return p


def refresh(out_dir: str) -> dict:
    """Rewrite one GPU's result.json and cards from its check.json (after a change in how cards are made)."""
    with open(os.path.join(out_dir, "check.json")) as f:
        full = json.load(f)
    pub = public(full)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(pub, f, indent=2)
    stale = os.path.join(out_dir, "card.png")
    if "png" not in write_cards(pub, out_dir) and os.path.exists(stale):
        os.remove(stale)
    return pub


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pcie(g, live):
    gen, width = live.get("pcie_gen_busy"), live.get("pcie_width_busy")
    if not gen and not width:
        return None
    gmax, wmax = g.get("pcie.link.gen.max"), g.get("pcie.link.width.max")
    full = str(gen) == str(gmax) and str(width) == str(wmax)
    mx = f" (card supports Gen{gmax} x{wmax})" if gmax and not full else ""
    return f"Gen{gen} x{width}{mx}"


VERDICT = {"PASS": "no failure detected in completed checks", "FAIL": "a numerical, memory or hardware-error criterion failed",
           "SUSPECT": "outputs that should be identical were not",
           "INCOMPLETE": "part of the check couldn't run or couldn't judge its results"}


def headline(pub: dict) -> str:
    """The one-line meaning of the result. A FAIL with no wrong result is a hardware error the driver reported."""
    if pub["result"] == "FAIL" and not sum((pub.get("wrong") or {}).values()) and pub.get("hardware_errors"):
        return "the driver reported a hardware error"
    return VERDICT[pub["result"]]


def _n(x) -> str:
    if x is None:
        return "?"
    if isinstance(x, float) and not x.is_integer():
        if abs(x) < 1:
            return f"{x:.3g}"
        return f"{x:,.1f}"
    return f"{int(x):,}"


def _pct(x: float) -> str:
    v = 100 * x
    return f"{v:.2g}%" if v < 10 else f"{v:.0f}%"


def _modes_row(mm: dict) -> Optional[str]:
    """Always shown when the math modes were attempted, including when none of them ran."""
    if not mm:
        return None
    ran = mm.get("ran") or []
    errors = mm.get("errors") or {}
    unsupported = [k for k in (mm.get("not_checked") or {}) if k not in errors]
    if ran:
        names = f"all {len(ran)} modes" if not unsupported and not errors else " ".join(ran)
        text = f"{names} vs float64: {_n(mm['matmuls'])} matmuls, " + (
            f"{mm['wrong']} WRONG" if mm.get("wrong") else "0 wrong")
        sens = {k: v for k, v in (mm.get("typical_error_caught") or {}).items() if isinstance(v, (int, float))}
        rel = [v for k, v in sens.items() if k != "fp8"]
        if rel:
            text += f", typical sensitivity {_pct(max(rel))}"
        if "fp8" in sens:            # FP8 tensor cores round much more coarsely: shown on its own
            text += f" (fp8 {_pct(sens['fp8'])})" if rel else f", typical sensitivity {_pct(sens['fp8'])} (fp8)"
    else:
        text = "none ran"
    if errors:
        text += "; ERROR in " + " ".join(errors)
    if unsupported:
        text += "; unsupported: " + " ".join(unsupported)
    return text


def lines(pub: dict) -> list:
    """The card as (label, text) rows; the terminal, Markdown, SVG and PNG cards all use these. Problems come
    right after what was checked, so they're always on the card image (it shows the first CARD_ROWS)."""
    w, c, s, h = pub["wrong"], pub["checked"], pub["speed"], pub["health"]
    nwrong = sum(w.values())
    hr = pub["headroom"]
    wrong_mm = nwrong - (w.get("memory_words") or 0)
    tflop = c.get("workload_tflop")
    rows = [("checked", f"{_n(c['results'])} matmul results" + (f" ({tflop:,.0f} TFLOP of work)" if tflop else "")
             + (f", {wrong_mm} WRONG (" + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in w.items()
                                                    if v and k != "memory_words") + ")" if wrong_mm else ", 0 wrong"))]
    modes = _modes_row(pub.get("math_modes") or {})
    if modes:
        rows.append(("math modes", modes))
    if c.get("memory_gib"):
        mw = w.get("memory_words") or 0
        rows.append(("memory test", f"{_n(c['memory_gib'])} GiB written and read back x{c['memory_passes']}, "
                     + (f"{mw} WORDS WRONG" if mw else "0 errors")))
    elif "memory_skipped" in c:          # (cards from before this field existed have no row)
        rows.append(("memory test", f"NOT RUN ({c.get('memory_skipped') or 'unknown reason'})"))
    order = {"bad": 0, "warn": 1}
    notes = sorted(pub.get("notes") or [], key=lambda n: order.get(n[0], 2))
    rows += [(level, text) for level, text in notes if level in order]
    fpo = (pub.get("fingerprint") or {}).get("output")
    if pub["repeatable"] is None:
        rows.append(("repeatable", "not checked (the check stopped before the second fingerprint run)"))
    else:
        runs = (pub.get("fingerprint") or {}).get("runs")
        recorded = f"{runs} recorded runs" if isinstance(runs, int) and not isinstance(runs, bool) else "recorded runs"
        rows.append(("repeatable", (f"{recorded} bit-identical" if pub["repeatable"] else
                                    "NO: outputs changed between runs") + (f", fingerprint {fpo[:12]}" if fpo else "")))
    if s.get("matmul_tflops"):
        rows.append(("speed", f"{s['dtype']} matmul {_n(s['matmul_tflops'])} TFLOPS"
                     + (f" ({s['pct_of_datasheet']:.0f}% of datasheet peak)" if s.get("pct_of_datasheet") else "")
                     + (f", dropped {s['drop_pct']}% while hot" if (s.get("drop_pct") or 0) >= 3 else ", steady")))
        rows.append(("bandwidth", f"{_n(s['bandwidth_gbs'])} GB/s"
                     + (f" ({s['bandwidth_pct_of_datasheet']:.0f}% of datasheet)" if s.get("bandwidth_pct_of_datasheet") else "")))
    hp = []
    if h.get("max_temp_c") is not None:
        hp.append(f"{h['max_temp_c']:.0f} C max")
    if h.get("max_power_w") is not None:
        hp.append(f"{h['max_power_w']:.0f} W max" + (f" (limit {h['power_limit_w']:.0f} W)" if h.get("power_limit_w") else ""))
    if h.get("pcie"):
        hp.append("PCIe " + h["pcie"])
    if h.get("ecc"):
        hp.append("ECC " + {"enabled": "on", "disabled": "off"}.get(h["ecc"].lower(), h["ecc"].lower()))
    if hp:
        rows.append(("health", ", ".join(hp)))
    rows.append(("headroom", f"worst clean score {hr['worst_clean_score']:.2f} (alarm at {hr['alarm_line']:g})"
                 + (f": {hr['alarm_line'] / hr['worst_clean_score']:.1f}x margin" if hr["worst_clean_score"] else "")))
    b = pub.get("board") or {}
    if b.get("same_setup_runs"):
        rows.append(("board", f"fingerprint matches {b['same_fingerprint']} of {b['same_setup_runs']} runs of this exact "
                              "setup on the public board"))
    if b.get("model_median_tflops") and s.get("matmul_tflops"):
        rows.append(("vs others", f"median {pub['gpu']['name']} on the board: {_n(b['model_median_tflops'])} TFLOPS "
                                  f"({b.get('model_runs')} runs)"))
    rows += [("note", text) for level, text in notes if level not in order]
    return rows


def _where(pub) -> str:
    p = pub.get("provider") or {}
    bits = [x for x in (p.get("name"), p.get("underlying_cloud")) if x]
    return " / ".join(bits)


def _subtitle(pub) -> str:
    g, sw = pub["gpu"], pub.get("software") or {}
    bits = [_where(pub)] if _where(pub) else []
    if g.get("memory_gib"):
        bits.append(f"{g['memory_gib']:g} GiB")
    if g.get("driver"):
        bits.append(f"driver {g['driver']}")
    bits.append(f"torch {sw.get('torch')}")
    return "  ·  ".join(bits)


def terminal(pub: dict, out_dir: str = "") -> str:
    g = pub["gpu"]
    head = f"sdcguard check  {g['name']}" + (f"  {_where(pub)}" if _where(pub) else "") + f"  {pub['date']}"
    out = [head, f"  {pub['result']}  {headline(pub)}"]
    if pub.get("simulated_faults"):
        out.append("  (SIMULATED faults were injected in software: a demo, not a real GPU fault)")
    out += [f"  {k:<11} {v}" for k, v in lines(pub)]
    if out_dir:
        out.append(f"  {'files':<11} {out_dir}/ (result.json, card.svg, card.md)")
    nxt = next_steps(pub, out_dir)
    if nxt:
        out.append(f"  {'next':<11} {nxt}")
    return "\n".join(out)


def next_steps(pub: dict, out_dir: str = "") -> str:
    """What to do about a result that isn't a PASS (the terminal output; not on the shared card)."""
    where = out_dir or "the output folder"
    if pub["result"] == "FAIL":
        if pub.get("simulated_faults"):
            return "(a demo: the faults were simulated in software) try `sdcguard replay` on an evidence .pt file"
        return (f"stop scheduling work on this GPU. To show the failure follows the board, run `sdcguard replay "
                f"<evidence>.pt` from {where} on this GPU and on a healthy one; `sdcguard report {where}` bundles "
                f"the evidence for an RMA")
    if pub["result"] == "SUSPECT":
        return "run `sdcguard check` again; if outputs differ again, take the card to your provider"
    if pub["result"] == "INCOMPLETE":
        return "the notes above say what didn't run or couldn't be judged; fix that (often free GPU memory) and rerun"
    return ""


def _md(x) -> str:
    """Text from a card (which anyone can edit) made inert in Markdown: no HTML, links, emphasis or tables."""
    s = str(x)
    for ch in "\\`*_[]()|#!~":
        s = s.replace(ch, "\\" + ch)
    return html.escape(s, quote=False).replace("\n", " ")


def markdown(pub: dict) -> str:
    g, sw = pub["gpu"], pub.get("software") or {}
    where = f" ({_md(_where(pub))})" if _where(pub) else ""
    rows = "\n".join(f"| {_md(k)} | {_md(v)} |" for k, v in lines(pub))
    sim = "\n\n**SIMULATED faults (demo), not a real result.**" if pub.get("simulated_faults") else ""
    tick = lambda v: str(v).replace("`", "'").replace("\n", " ")       # noqa: E731  (inside a code span)
    return (f"**sdcguard check: {_md(pub['result'])}** on {_md(g['name'])}{where}, {_md(pub['date'])}{sim}\n\n"
            f"| | |\n|---|---|\n{rows}\n\n"
            f"`sdcguard {tick(sw.get('sdcguard'))} · torch {tick(sw.get('torch'))} · digest {tick(pub['digest'])}`  \n"
            f"Check yours: `pip install sdcguard && sdcguard check`\n")


_COLORS = {"PASS": "#22c55e", "FAIL": "#ef4444", "SUSPECT": "#f59e0b", "INCOMPLETE": "#94a3b8"}
CARD_ROWS = 9


def svg(pub: dict) -> str:
    e = html.escape
    g = pub["gpu"]
    col = _COLORS[pub["result"]]
    pw = 150 if len(pub["result"]) <= 7 else 236          # the verdict pill fits its word
    rows = lines(pub)[:CARD_ROWS]
    body = []
    y = 288
    for k, v in rows:
        v = v if len(v) <= 90 else v[:87] + "..."
        body.append(f'<text x="64" y="{y}" class="k">{e(k)}</text><text x="210" y="{y}" class="v">{e(v)}</text>')
        y += 32
    sim = ('<text x="1136" y="120" text-anchor="end" class="sim">SIMULATED FAULTS (DEMO)</text>'
           if pub.get("simulated_faults") else "")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
<style>
text{{font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;fill:#e7e5e4}}
.t{{font-size:22px;fill:#a8a29e}}.g{{font-size:40px;font-weight:700}}.w{{font-size:22px;fill:#a8a29e}}
.p{{font-size:30px;font-weight:800;fill:#0c0a09}}.vd{{font-size:24px}}.k{{font-size:21px;fill:#a8a29e}}
.v{{font-size:21px}}.f{{font-size:19px;fill:#78716c}}.sim{{font-size:20px;font-weight:700;fill:#f59e0b}}
</style>
<rect width="1200" height="630" rx="28" fill="#0c0a09"/><rect x="1" y="1" width="1198" height="628" rx="28" fill="none" stroke="#292524" stroke-width="2"/>
<text x="64" y="72" class="t">sdcguard check</text><text x="1136" y="72" text-anchor="end" class="t">{e(pub['date'])}</text>
<text x="64" y="132" class="g">{e(str(g['name']))}</text>{sim}
<text x="64" y="168" class="w">{e(_subtitle(pub))}</text>
<rect x="64" y="190" width="{pw}" height="48" rx="24" fill="{col}"/><text x="{64 + pw // 2}" y="225" text-anchor="middle" class="p">{e(pub['result'])}</text>
<text x="{86 + pw}" y="223" class="vd">{e(headline(pub))}</text>
{''.join(body)}
<text x="64" y="596" class="f">digest {e(pub['digest'])}  ·  pip install sdcguard  ·  sdcguard check</text>
</svg>
"""


def png(pub: dict, path: str) -> bool:
    """The same card as a PNG (for sites that don't take SVG), when matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except Exception:  # noqa: BLE001
        return False
    with matplotlib.rc_context({"text.parse_math": False, "text.usetex": False}):   # card text is never math
        return _png(pub, path, plt, FancyBboxPatch)


def _png(pub: dict, path: str, plt, FancyBboxPatch) -> bool:
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    fig.patch.set_facecolor("#0c0a09")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(630, 0)
    ax.axis("off")
    fg, mute = "#e7e5e4", "#a8a29e"
    ax.text(64, 72, "sdcguard check", color=mute, fontsize=16, va="baseline")
    ax.text(1136, 72, pub["date"], color=mute, fontsize=16, ha="right", va="baseline")
    ax.text(64, 132, str(pub["gpu"]["name"]), color=fg, fontsize=28, fontweight="bold", va="baseline")
    ax.text(64, 168, _subtitle(pub), color=mute, fontsize=15, va="baseline")
    if pub.get("simulated_faults"):
        ax.text(1136, 120, "SIMULATED FAULTS (DEMO)", color="#f59e0b", fontsize=14, ha="right", fontweight="bold")
    pw = 150 if len(pub["result"]) <= 7 else 236
    ax.add_patch(FancyBboxPatch((64, 190), pw, 48, boxstyle="round,pad=0,rounding_size=24",
                                fc=_COLORS[pub["result"]], ec="none"))
    ax.text(64 + pw / 2, 222, pub["result"], color="#0c0a09", fontsize=21, fontweight="bold", ha="center",
            va="baseline")
    ax.text(86 + pw, 222, headline(pub), color=fg, fontsize=17, va="baseline")
    y = 288
    for k, v in lines(pub)[:CARD_ROWS]:
        v = v if len(v) <= 90 else v[:87] + "..."
        ax.text(64, y, k, color=mute, fontsize=14, va="baseline")
        ax.text(210, y, v, color=fg, fontsize=14, va="baseline")
        y += 32
    ax.text(64, 596, f"digest {pub['digest']}  ·  pip install sdcguard  ·  sdcguard check", color="#78716c",
            fontsize=13, va="baseline")
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def write_cards(pub: dict, out_dir: str) -> dict:
    paths = {"svg": os.path.join(out_dir, "card.svg"), "md": os.path.join(out_dir, "card.md"),
             "png": os.path.join(out_dir, "card.png")}
    with open(paths["svg"], "w") as f:
        f.write(svg(pub))
    with open(paths["md"], "w") as f:
        f.write(markdown(pub))
    if not png(pub, paths["png"]):
        paths.pop("png")
    return paths


def share_url(pub: dict) -> str:
    """A link that opens a new issue on the board's GitHub repo with this result filled in. Nothing is sent
    until you press Submit there."""
    compact = json.dumps(pub, separators=(",", ":"), default=str)
    title = f"[result] {pub['result']}: {pub['gpu']['name']}" + (f" ({_where(pub)})" if _where(pub) else "")
    q = urllib.parse.urlencode({"template": "result.yml", "title": title, "result": compact})
    return f"https://github.com/{BOARD_REPO}/issues/new?{q}"


# --- many GPUs ----------------------------------------------------------------------------------------------

def cross_compare(results: list) -> None:
    """GPUs of the same model on one machine run the same software, so their fingerprints must match.
    With three or more, a GPU that differs from all the others is the odd one out. Updates results."""
    groups: dict = {}
    for r in results:
        if r.get("fingerprint_key") and r.get("fingerprints"):      # (a check that stopped early may have none)
            groups.setdefault(r["fingerprint_key"], []).append(r)
    for rs in groups.values():
        if len(rs) < 2:
            continue
        counts: dict = {}
        for r in rs:
            counts[r["fingerprints"][0]["output"]] = counts.get(r["fingerprints"][0]["output"], 0) + 1
        if len(counts) == 1:
            for r in rs:
                r["cross_gpu"] = f"identical output on all {len(rs)} GPUs of this model here"
            continue
        top, n_top = max(counts.items(), key=lambda kv: kv[1])
        for r in rs:
            mine = r["fingerprints"][0]["output"]
            if len(rs) >= 3 and n_top > len(rs) / 2 and mine != top:
                r["cross_gpu"] = f"differs from the other {n_top} GPUs of this model here (odd one out)"
                r["notes"].insert(0, ["bad", r["cross_gpu"]])
                if r["result"] in ("PASS", "INCOMPLETE"):          # FAIL > SUSPECT > INCOMPLETE > PASS
                    r["result"] = "SUSPECT"
            elif len(rs) >= 3 and n_top > len(rs) / 2:
                r["cross_gpu"] = f"matches {n_top - 1} other GPUs of this model here"
            else:
                # the same deterministic workload, the same GPU model and software, different outputs: at least
                # one of them computed it wrong, and nothing here says which, so none of them passes
                r["cross_gpu"] = "GPUs of this model here disagree and there's no clear majority"
                r["notes"].insert(0, ["bad", r["cross_gpu"] + "; at least one computed the workload wrong. Rerun "
                                             "to see which one changes, or compare with the board"])
                if r["result"] in ("PASS", "INCOMPLETE"):
                    r["result"] = "SUSPECT"


def _children(args) -> tuple:
    """One process per GPU, all at once. Returns (results, problems): a GPU whose check crashed or hung is
    a problem, never silently missing (a GPU that raises CUDA errors may be the bad one)."""
    torch = _torch()
    n = torch.cuda.device_count()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [v.strip() for v in vis.split(",") if v.strip()][:n] if vis else [str(i) for i in range(n)]
    os.makedirs(args.out, exist_ok=True)
    procs, logs = [], []
    for i, gid in enumerate(ids):
        marker = os.path.join(args.out, f".child_{i}.json")
        if os.path.exists(marker):
            os.remove(marker)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gid)
        child_dev = os.environ.get("SDCGUARD_CHILD_DEVICE", "cuda:0")      # (tests run the children on the CPU)
        cmd = [sys.executable, "-m", "sdcguard.check", "--device", child_dev, "--minutes", str(args.minutes), "--out",
               args.out, "--seed", str(args.seed), "--simulate-faults", str(args.simulate_faults), "--child", str(i),
               "--offline"]
        if args.size:
            cmd += ["--size", args.size]
        if args.provider:
            cmd += ["--provider", args.provider]
        logs.append(open(os.path.join(args.out, f"child_{i}.log"), "w"))
        procs.append(subprocess.Popen(cmd, env=env, stdout=logs[-1], stderr=subprocess.STDOUT))
    print(f"sdcguard check: {len(ids)} GPUs in parallel, about {args.minutes:g} minutes", flush=True,
          file=sys.stderr if getattr(args, "json", False) else sys.stdout)
    limit = time.time() + worker_limit(args.minutes)
    codes = []
    for p in procs:
        try:
            codes.append(p.wait(timeout=max(1.0, limit - time.time())))
        except subprocess.TimeoutExpired:
            p.kill()
            codes.append("timeout")
    for f in logs:
        f.close()
    found, problems, failed = [], [], 0
    for i, code in enumerate(codes):
        log = os.path.join(args.out, f"child_{i}.log")
        out_dir = None
        run_id = None
        try:
            with open(os.path.join(args.out, f".child_{i}.json")) as f:
                marker = json.load(f)
                out_dir, run_id = marker["out_dir"], marker.get("run_id")
            with open(os.path.join(out_dir, "check.json")) as f:
                r = json.load(f)
            if not isinstance(r, dict) or "result" not in r:
                raise ValueError("check.json without a result")
            if not run_id or r.get("run_id") != run_id:
                raise ValueError("check.json belongs to a previous run")
            if r.get("running"):
                raise ValueError("the check didn't finish")
            if code != 0:
                r.setdefault("notes", []).insert(0, ["bad", f"the GPU worker exited unexpectedly ({code}); "
                                                      "the saved result may be incomplete"])
                if r["result"] == "PASS":
                    r["result"] = "INCOMPLETE"
            found.append(r)
            continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # the child process died (or hung) before writing its result: whatever it saved still counts
        bad, died = 0, None
        if out_dir and run_id:
            died = _died_worker(out_dir, run_id, code, log)
            bad = sum(died["wrong"].values())
        if bad:
            failed += 1
            problems.append(f"GPU {ids[i]}: FAIL: it computed wrong results and then its check died (exit {code}); "
                            f"evidence in {out_dir}; see {log}")
        else:
            problems.append(f"GPU {ids[i]}: could not be checked (exit {code}); see {log}")
    return found, problems, failed


def worker_limit(minutes: float) -> float:
    """How long one GPU's worker may run: its budget plus room for loading and the slowest phase. Two minutes
    less than the deadline `sdcguard nhc` and `sdcguard certify` give the whole check (cli._deadline), so this
    process always gets to stop a hung worker and report what the others found."""
    return 60.0 * float(minutes) * 1.5 + 480.0


def saved_failures(out_dir: str, run_id: str) -> tuple:
    """What a run that stopped before it reported had saved: (failed results by phase, evidence paths), from the
    evidence files tagged with its run id. Old evidence can remain beside this run after a rerun: only an incident
    explicitly tagged with this run's identifier counts."""
    counts = {"workload_training": 0, "workload_serving": 0, "stress": 0, "math_modes": 0}
    paths = []
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        names = []
    for name in names:
        if not name.startswith(("evidence_", "math_mode_evidence_")) or not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(out_dir, name)) as f:
                ev = json.load(f)
            if not run_id or ev.get("run_id") != run_id:
                continue
        except (OSError, ValueError, AttributeError):
            continue
        paths.append(os.path.join(out_dir, name[:-len(".json")]))
        phase = {"stress": "stress", "math modes": "math_modes", "serving": "workload_serving"}.get(ev.get("phase"),
                                                                                                 "workload_training")
        counts[phase] += 1
    return counts, paths


def _died_worker(out_dir: str, run_id: str, code, log: str) -> dict:
    """check.json for a GPU whose worker died (or hung and was stopped) before it reported: FAIL when it had
    already computed wrong results (its summary's counts, and evidence tagged with this run), else INCOMPLETE.
    Written to its folder, so `sdcguard nhc` and the fleet report see the same result this check prints."""
    from .screen import _write_atomic
    wrong = {"workload_training": 0, "workload_serving": 0, "stress": 0, "math_modes": 0, "memory_words": 0}
    gpu, simulated = {}, False
    for name in ("check.json", "summary.json"):           # (its own placeholder and workload summary, this run's)
        try:
            with open(os.path.join(out_dir, name)) as f:
                rec = json.load(f)
            if not isinstance(rec, dict) or rec.get("run_id") != run_id:
                continue
            gpu = rec.get("gpu") if isinstance(rec.get("gpu"), dict) else gpu
            simulated = simulated or bool(rec.get("simulated_faults") or rec.get("SIMULATED_FAULTS"))
            if name == "summary.json":
                wrong["workload_training"] = int(rec.get("corrupted_results_training") or 0)
                wrong["workload_serving"] = int(rec.get("corrupted_results_serving") or 0)
        except (OSError, ValueError, TypeError):
            pass
    seen, evidence = saved_failures(out_dir, run_id)
    for k, n in seen.items():        # each saved failure is a wrong result, so the counts are at least these
        wrong[k] = max(wrong[k], n)
    failed = any(wrong.values())
    why = (f"the GPU worker died (exit {code}) before it reported" if code != "timeout" else
           "the GPU worker didn't finish in time and was stopped")
    record = {"schema": SCHEMA.replace("check/", "check-full/"), "result": "FAIL" if failed else "INCOMPLETE",
              "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": run_id, "gpu": gpu,
              "wrong": wrong, "evidence": evidence, "out_dir": out_dir, "simulated_faults": simulated,
              "stopped": {"phase": "worker", "error": why},
              "notes": [["bad", why + (": it had already computed wrong results (at least the counts here; "
                                       "evidence saved)" if failed else ": not checked") + f"; see {log}"]]}

    def dump(t):
        with open(t, "w") as f:
            json.dump(record, f, indent=2, default=str)
    try:
        _write_atomic(os.path.join(out_dir, "check.json"), dump)
    except OSError:
        pass
    return record


def main(argv=None) -> int:
    want = ":4096:8"                              # deterministic cuBLAS, same workspace as everyone else's run
    json_out = "--json" in (sys.argv[1:] if argv is None else argv)
    say = (lambda *x, **k: print(*x, file=sys.stderr, **k)) if json_out else print   # --json: stdout is JSON only
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", want) != want:
        say(f"sdcguard check: using CUBLAS_WORKSPACE_CONFIG={want} (was {os.environ['CUBLAS_WORKSPACE_CONFIG']}) "
            "so fingerprints compare with other runs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = want
    p = argparse.ArgumentParser(prog="sdcguard check", description=__doc__.split("\n\n")[0])
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--device", default=None, help="e.g. cuda:0 (default: every visible GPU)")
    p.add_argument("--size", default=None, choices=["tiny", "small", "1b", "3b"],
                   help="workload model size (default: by GPU memory)")
    p.add_argument("--out", default="sdcguard_check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provider", default=None, help="who you rent this GPU from, if not detected (shown on the card)")
    p.add_argument("--share", action="store_true", help="print a link that posts the result card to the public board")
    p.add_argument("--offline", action="store_true", help="don't fetch the public board's reference data")
    p.add_argument("--key", default=None, help="also write a signed certificate (key from `sdcguard keygen`)")
    p.add_argument("--issuer", default="")
    p.add_argument("--json", action="store_true", help="print the shareable result as JSON")
    p.add_argument("--certificate", action="store_true", help="write certificate.json per GPU (signed with --key)")
    p.add_argument("--valid-days", type=float, default=30.0, help="how long a certificate is valid")
    p.add_argument("--simulate-faults", type=float, default=0.0, help="demo only: corrupt this share of matmul outputs")
    p.add_argument("--child", default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    CERTIFICATE.update(always=bool(a.certificate), valid_days=float(a.valid_days))
    torch = _torch()
    if a.key and a.child is None:              # a key that can't sign is found now, not after the whole check
        try:
            from . import certify
            certify.load_private(a.key)
        except Exception as e:  # noqa: BLE001
            say(f"sdcguard check: can't sign with --key {a.key}: {type(e).__name__}: {e}")
            return 2
    ref = None if (a.offline or a.child is not None) else fetch_reference()
    problems: list = []
    died_failed = 0
    multi = a.device is None and a.child is None and torch.cuda.is_available() and torch.cuda.device_count() > 1
    if multi:
        results, problems, died_failed = _children(a)
        if not results:
            say("sdcguard check: could not run\n  " + "\n  ".join(problems))
            return 1 if died_failed else 2
        cross_compare(results)
        for r in results:
            if ref and r.get("board") is None:
                try:
                    apply_board(r, ref)
                except Exception as e:  # noqa: BLE001
                    r["notes"].append(["info", f"couldn't compare with the public board's data ({type(e).__name__})"])
            write_result(r, a.key, a.issuer)
    else:
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if a.child is None and not torch.cuda.is_available():
            say("sdcguard check: no CUDA GPU found, checking the CPU instead (a demo of what the check does)")
        try:
            results = [check_one(dev, a.minutes, a.size, a.out, a.seed, a.simulate_faults, a.provider, ref,
                                 None if a.child is not None else a.key, "" if a.child is not None else a.issuer,
                                 progress=None if a.child is not None else (lambda s: say("  " + s, flush=True)),
                                 tag=a.child)]
        except Exception as e:  # noqa: BLE001
            say(f"sdcguard check: could not run on {dev}: {type(e).__name__}: {e}")
            return 2
        if a.child is not None:
            return 0
    pubs = []
    for r in results:
        pub = public(r)
        pubs.append(pub)
        if a.json:
            continue                                # printed below as one document
        else:
            print(terminal(pub, r["out_dir"]))
            if r.get("cross_gpu"):
                print(f"  {'this node':<11} {r['cross_gpu']}")
            print()
        if a.share:
            say(f"Post this card to the public board ({BOARD_URL}):\n{share_url(pub)}\n")
    if a.json:                                     # one GPU: its card; several: one array of cards
        print(json.dumps(pubs[0] if len(pubs) == 1 and not problems else
                         pubs + [{"result": "COULD NOT RUN", "problem": pr} for pr in problems], indent=2))
    for pr in problems:
        say(f"sdcguard check: {pr}")
    if (len(pubs) > 1 or problems) and not a.json:
        c = {k: sum(x["result"] == k for x in pubs) for k in ("PASS", "FAIL", "SUSPECT", "INCOMPLETE")}
        print(f"sdcguard check: {len(pubs) + len(problems)} GPUs, {c['PASS']} PASS, {c['FAIL'] + died_failed} FAIL, "
              f"{c['SUSPECT']} SUSPECT, {c['INCOMPLETE']} INCOMPLETE"
              + (f", {len(problems) - died_failed} could not be checked" if len(problems) > died_failed else ""))
    if not a.share and not a.json:
        print("Share it: sdcguard check --share prints a link that posts your card to the public board.")
    if died_failed or any(x["result"] in ("FAIL", "SUSPECT") for x in pubs):
        return 1
    if any(r.get("certificate_error") for r in results):
        return 2          # a certificate was asked for and couldn't be written
    return 2 if problems or any(x["result"] == "INCOMPLETE" for x in pubs) else 0


if __name__ == "__main__":
    sys.exit(main())
