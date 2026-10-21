"""Known wrong-result issues in NVIDIA's cuBLAS, matched against this machine.

NVIDIA's CUDA release notes list cuBLAS issues that make matmuls return wrong numbers without an error: some
only on one GPU generation, some only in one mode (MPS below 100% of the SMs, concurrent kernels) or one kind of
matmul (grouped GEMM, NVFP4, split-K). A health check on an idle GPU can't see most of them (DCGM and gpu-burn
passed while issue 6862950 corrupted matmuls under MPS). This reads the cuBLAS version actually loaded, the GPU's
compute capability and whether the process is an MPS client below 100%, and says which listed issues apply:

  applies     the version, the GPU and the mode all match (for MPS: this process sees fewer SMs than the GPU
              has, as an MPS client below 100% does): matmuls here can come out wrong
  may apply   the version and the GPU match; whether it bites depends on what the job runs (named in `when`)
  not active  the version and the GPU match, but the mode it needs isn't on here

    sdcguard issues            # this machine
    sdcguard issues --json

The table follows NVIDIA's release notes from CUDA 12.6 Update 2 to 13.4 Update 2 and the cuBLAS patch notes (read
Oct 20, 2026). A version outside a listed range isn't proof of safety, and an issue NVIDIA hasn't listed can't be
matched. `sdcguard blas` tests the actual algorithms for a shape.
"""
from __future__ import annotations

import json
import os
from typing import Optional

NOTES = "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html"
PATCHES = "https://docs.nvidia.com/cuda/cublas-patch-release-notes/"

# Versions are cuBLAS (major, minor, patch), as cublasLtGetVersion() reports them. NVIDIA fixes an issue on several
# release branches separately (a standalone cuBLAS patch, or a toolkit update), so a fix is per branch, keyed by
# (major, minor):
#   introduced  first affected version as NVIDIA states it (None: NVIDIA doesn't say)
#   fixed       every version from this one on is fixed (None: not fixed in any toolkit yet)
#   branches    {(major, minor): first fixed version on that branch}
# Sources: the CUDA release notes from 12.6 Update 2 to 13.4 Update 2 (each release's component table gives its
# cuBLAS version) and the cuBLAS patch notes, read Oct 20, 2026.
_BR = {(12, 8): (12, 8, 5), (12, 9): (12, 9, 2), (13, 1): (13, 1, 1), (13, 2): (13, 2, 2)}
ISSUES = [
    {"id": "6862950", "what": "GEMM kernels can produce incorrect results on Hopper GPUs for FP16, BF16 and FP8 when "
                              "the process runs as an MPS client with CUDA_MPS_ACTIVE_THREAD_PERCENTAGE below 100 "
                              "(cuBLASLt algorithm 66, clusters of 4 or more CTAs)",
     "introduced": (12, 6, 3), "fixed": None, "branches": {}, "cc": ["9.0"], "needs": "mps_partial",
     "workaround": "use cluster configurations with 1 or 2 CTAs in affected processes (NVIDIA), or run at 100% "
                   "of the SMs", "when": "the process runs under MPS with less than 100% of the SMs"},
    {"id": "6580689", "what": "cublasLtMatmul() can return incorrect output with split-K when SPLITK_NUM doesn't "
                              "divide K and K / SPLITK_NUM is a multiple of the stage size (algorithm 66)",
     "introduced": (12, 6, 3), "fixed": (13, 8, 0), "branches": {}, "cc": ["9.0", "10.*", "11.*"],
     "when": "cuBLASLt picks a split-K algorithm for one of the job's shapes"},
    {"id": "6681084", "what": "cuBLASLt grouped GEMM with CUBLAS_POINTER_MODE_HOST can give incorrect results on "
                              "Hopper when the number of waves is larger than 2",
     "introduced": (13, 4, 0), "fixed": (13, 8, 0), "branches": {(13, 4): (13, 4, 2), (13, 6): (13, 6, 2)},
     "cc": ["9.0"], "workaround": "pass alpha and beta with CUBLAS_POINTER_MODE_DEVICE",
     "when": "the job runs cuBLASLt grouped GEMMs (MoE)"},
    {"id": "6681084 (NVFP4)", "what": "cuBLASLt grouped GEMM with NVFP4 inputs can give incorrect results on B300 "
                                      "and Rubin (more than 3 waves, CUBLASLT_MATMUL_STAGES_768xAUTO)",
     "introduced": (13, 7, 0), "fixed": (13, 8, 0), "branches": {}, "cc": ["10.3", "10.7"],
     "when": "the job runs NVFP4 grouped GEMMs"},
    {"id": "6798616", "what": "cuBLASLt grouped GEMM operations can lead to incorrect results on Blackwell and Rubin "
                              "GPUs",
     "introduced": (13, 7, 0), "fixed": None, "branches": {(13, 8): (13, 8, 1)}, "cc": ["10.*", "11.*", "12.*"],
     "workaround": "cuBLAS patch 13.8.1", "when": "the job runs cuBLASLt grouped GEMMs (MoE)"},
    {"id": "CUB-10573", "what": "cublasLtMatmul() can produce incorrect results on B300 and Rubin with NVFP4 inputs "
                                "(algorithm 66, CUBLASLT_MATMUL_STAGES_768xAUTO)",
     "introduced": (13, 5, 1), "fixed": (13, 8, 0), "branches": {(13, 6): (13, 6, 2)}, "cc": ["10.3", "10.7"],
     "when": "the job runs NVFP4 matmuls"},
    {"id": "CUB-10445", "what": "GEMV-like operations (gemv, or cublasLtMatmul with M or N = 1) may produce incorrect "
                                "results on compute capability 12.0 and 12.1 with a transposed matrix, a "
                                "non-accumulation dimension above 3,145,680 and beta != 0",
     "introduced": (13, 6, 0), "fixed": (13, 7, 0), "branches": {(13, 6): (13, 6, 1)}, "cc": ["12.0", "12.1"],
     "when": "the job runs very large matrix-vector products with beta != 0"},
    {"id": "6059292", "what": "cublasLtMatmul() ignores the tensor-wide scale for NVFP4 matmuls with NVFP4 output "
                              "(algorithm 66), giving incorrect results",
     "introduced": (13, 4, 0), "fixed": (13, 4, 1), "branches": {}, "cc": ["10.*", "11.*"],
     "when": "the job runs NVFP4 matmuls with NVFP4 output"},
    {"id": "5807900", "what": "cublasLtMatmul() may produce incorrect results when run concurrently with another "
                              "kernel that uses Tensor Memory (algorithm 66)",
     "introduced": (12, 8, 0), "fixed": (13, 3, 0), "branches": dict(_BR), "cc": ["10.*", "11.*"],
     "when": "matmuls run concurrently with other Tensor Memory kernels (several streams, overlapped "
             "communication)"},
    {"id": "CUB-9572", "what": "cublasLtMatmul could give incorrect results or invalid memory accesses when the data "
                               "type size times a leading dimension exceeds the signed 32-bit range (algorithm 66)",
     "introduced": (12, 6, 3), "fixed": (13, 3, 0), "branches": dict(_BR), "cc": ["9.0", "10.*", "11.0"],
     "when": "matrices with very large leading dimensions"},
    {"id": "CUB-9529", "what": "the cuBLASLt grouped GEMM API ignores groups with k = 0, which can lead to incorrect "
                               "results",
     "introduced": (13, 2, 0), "fixed": (13, 3, 0), "branches": {}, "cc": ["10.*", "11.0"],
     "workaround": "initialize D with beta*C for all groups, then compute D += A*B",
     "when": "grouped GEMMs with empty groups along k"},
    {"id": "CUB-9633", "what": "on Blackwell, FP64 fixed-point emulation kernels may produce incorrect results or "
                               "corrupt data when run concurrently with third-party kernels that allocate Tensor "
                               "Memory",
     "introduced": (13, 1, 0), "introduced_note": "NVIDIA gives no first version; FP64 emulation arrived in "
                                                  "cuBLAS 13.1.0", "fixed": None, "branches": {},
     "cc": ["10.*", "11.*", "12.*"],
     "when": "FP64 emulation is on and other Tensor Memory kernels run at the same time"},
    {"id": "CUB-10066", "what": "cublasLtMatmul() with INT8 inputs and scale type CUDA_R_32I allowed non-default "
                                "epilogues on compute capability 10.x and returned incorrect results",
     "introduced": None, "fixed": (13, 6, 0), "branches": {}, "cc": ["10.*"],
     "when": "INT8 matmuls with fused epilogues"},
    {"id": "5845724", "what": "the cuBLASLt matmul API could give incorrect results with C broadcasting (LDC = 0)",
     "introduced": None, "fixed": (13, 3, 0), "branches": {}, "cc": None, "when": "matmuls that broadcast C"},
    {"id": "CUB-9577", "what": "cublasLtMatmul could give incorrect NVFP4 results on B300 and GB300 when m is not a "
                               "multiple of 64",
     "introduced": None, "fixed": (13, 3, 0), "branches": {}, "cc": ["10.3"], "when": "NVFP4 matmuls"},
    {"id": "6207926", "what": "cublasZgemv() with CUBLAS_OP_C and incy > 1 gave incorrect results with FP64 "
                              "fixed-point emulation on (CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH, eager strategy)",
     "introduced": (13, 1, 0), "introduced_note": "NVIDIA gives no first version; FP64 emulation arrived in "
                                                  "cuBLAS 13.1.0", "fixed": (13, 7, 0), "branches": {}, "cc": None,
     "when": "complex matrix-vector products (Zgemv, conjugate transpose) run with FP64 emulation"},
    {"id": "6155165", "what": "cublasXt<t>spmm() could give incorrect results with m greater than 46340",
     "introduced": None, "fixed": (13, 6, 0), "branches": {}, "cc": None, "when": "the job calls cublasXt spmm"},
    {"id": "CUB-9942", "what": "cublasLtMatmul() could run an unsupported type combination (an FP32-like compute "
                               "type, FP32 C and D, non-FP32 A and B), reading A and B as FP32 matrices",
     "introduced": None, "fixed": (13, 6, 0), "branches": {}, "cc": None,
     "when": "matmuls with FP32 C and D and A and B of another type under an FP32-like compute type"},
    {"id": "5720478", "what": "complex FP64 GEMMs with fixed-point emulation (cublasZgemm, cublasZtrsm, cublasGemmEx, "
                              "cublasLtMatmul) could give incorrect results when 2*m*n + m exceeded UINT_MAX (an "
                              "integer overflow)",
     "introduced": (13, 1, 0), "introduced_note": "NVIDIA gives no first version; FP64 emulation arrived in "
                                                  "cuBLAS 13.1.0", "fixed": (13, 3, 0), "branches": {}, "cc": None,
     "when": "very large complex FP64 matmuls (m*n above 2^31) run with FP64 emulation"},
    {"id": "5452663", "what": "some cublasZtrmm() kernels gave incorrect results when m = 1 and side is "
                              "CUBLAS_SIDE_RIGHT on Ada and Blackwell GeForce-class GPUs",
     "introduced": None, "fixed": None, "branches": {**_BR, (13, 0): (13, 0, 0)}, "cc": ["8.9", "12.0"],
     "note": "NVIDIA lists the fix in CUDA 13.0 and in the cuBLAS patches 12.8.5, 12.9.2, 13.1.1 and 13.2.2 (for "
             "12.8 Update 1, 12.9 Update 1, 13.0 Update 2 and 13.1 Update 1), and for no toolkit after them",
     "when": "the job runs ztrmm (complex FP64 triangular multiply) with m = 1 from the right"},
    {"id": "CUB-8874", "what": "cublasLtMatmul() with INT8 inputs, INT32 accumulation and INT32 output could give "
                               "incorrect results when A and B used regular ordering (CUBLASLT_ORDER_COL or ROW)",
     "introduced": None, "fixed": (13, 0, 2), "branches": {}, "cc": None,
     "when": "the job runs INT8 matmuls with INT32 output"},
    {"id": "CUB-8846", "what": "some cublasSsyrkx() kernels gave incorrect results when beta = 0 on Blackwell GPUs",
     "introduced": None, "fixed": (13, 0, 2), "branches": {}, "cc": ["10.*", "11.*", "12.*"],
     "when": "the job calls cublasSsyrkx with beta = 0"},
    {"id": "5104822", "what": "cublasLtMatmul() with m or n equal to 1 and the CUBLASLT_EPILOGUE_BIAS epilogue could "
                              "give incorrect results",
     "introduced": None, "fixed": (12, 8, 4), "branches": {}, "cc": None,
     "when": "matrix-vector-shaped matmuls (m or n = 1) with a fused bias"},
    {"id": "4910924", "what": "cublasLtMatmul() with the CUBLASLT_EPILOGUE_BGRADA or BGRADB epilogue accepted an "
                              "output in CUBLASLT_ORDER_ROW layout and computed the bias gradients incorrectly",
     "introduced": None, "fixed": (12, 8, 3), "branches": {}, "cc": None,
     "when": "fused bias-gradient epilogues with a row-major output"},
    {"id": "4941052", "what": "FP8 cublasLtMatmul() on Hopper could compute the maximum of the absolute values of the "
                              "output (CUBLASLT_MATMUL_DESC_AMAX_D_POINTER) incorrectly (algorithm 66; also CUB-7595)",
     "introduced": None, "fixed": (12, 6, 4), "branches": {}, "cc": ["9.0"],
     "when": "FP8 matmuls that compute the output's amax (FP8 training's scaling)"},
    {"id": "CUB-7596", "what": "FP8 cublasLtMatmul() on Hopper could ignore "
                               "CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER (algorithm 66), leaving that amax "
                               "unwritten",
     "introduced": None, "fixed": (12, 6, 4), "branches": {}, "cc": ["9.0"],
     "when": "FP8 matmuls with an epilogue that computes the auxiliary output's amax"},
    {"id": "4751576", "what": "cublasLtMatmul() with INT8 inputs, INT32 accumulation, INT8 output and FP32 scaling "
                              "factors could give numerically inaccurate results with a split-K reduction",
     "introduced": None, "fixed": (12, 6, 3), "branches": {}, "cc": None,
     "workaround": "set the reduction scheme to none and split-K to 1 (cublasLtMatmulAlgoConfigSetAttribute)",
     "when": "INT8 matmuls with INT8 output and FP32 scales that cuBLASLt runs with split-K"},
]

# SMs on the full GPU, to notice an MPS client limited below 100% when the variable isn't in this process. Matched
# by the longest name a GPU's name contains ("H800 PCIe" before "H800", "H200" before "H20"). The H800 PCIe has the
# H100 PCIe's 114 SMs (14,592 CUDA cores, 456 Tensor Cores in its board makers' datasheets).
FULL_SMS = {"H100 80GB HBM3": 132, "H100 NVL": 132, "H100 PCIe": 114, "H200": 132, "H200 NVL": 132, "GH200": 132,
            "H800": 132, "H800 PCIe": 114, "H20": 78}


def _vt(v) -> Optional[tuple]:
    """A cuBLAS version string ('13.1.1', '13.1.1.3') as (major, minor, patch)."""
    if v is None:
        return None
    if isinstance(v, tuple):
        return tuple(v[:3])
    parts = []
    for p in str(v).split(".")[:3]:
        digits = "".join(ch for ch in p if ch.isdigit())
        if not digits:
            return None
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def affected(it: dict, ver: tuple) -> Optional[bool]:
    """Whether a cuBLAS version is in an issue's affected range (None: NVIDIA doesn't give a first version and
    this one is before the fix, so it can't be ruled out)."""
    lo, hi = it.get("introduced"), it.get("fixed")
    if hi is not None and ver >= hi:
        return False
    fix = (it.get("branches") or {}).get(ver[:2])
    if fix is not None and ver >= fix:
        return False
    if lo is not None:
        return ver >= lo
    return None


def _show(v: tuple) -> str:
    return ".".join(map(str, v))


def _cc_match(pattern: Optional[list], cc: Optional[str]) -> Optional[bool]:
    if pattern is None:
        return True
    if cc is None:
        return None
    return any(p == cc or (p.endswith(".*") and cc.split(".")[0] == p[:-2]) for p in pattern)


def mps_state(gpu: Optional[str], sms: Optional[int], share: Optional[float], visible: str = "") -> tuple:
    """(the GPU's full SM count or None, MPS state). What decides it is what this process can use: an MPS client
    below 100% sees fewer SMs than the GPU has (an H200 at 40% reports 52 of 132). The variable alone doesn't
    prove the process is an MPS client."""
    full = next((FULL_SMS[k] for k in sorted(FULL_SMS, key=len, reverse=True) if gpu and k in gpu), None)
    if "MIG-" in (visible or "") or "MIG" in (gpu or ""):
        return full, "no (a MIG instance)"
    if full is not None and sms is not None and sms < full:
        return full, f"partial ({sms} of {full} SMs visible)"
    if share is not None and share < 100:
        if full is not None and sms == full:
            return full, (f"configured (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is {share:g}, but this process sees all "
                          f"{sms} SMs)")
        return full, (f"configured (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is {share:g}; the SMs left to this process "
                      "couldn't be confirmed)")
    if full is not None and sms == full:
        return full, "full"
    if sms is not None:
        return full, "unknown (this GPU's full SM count isn't in the table)"
    return full, None


def environment(device=None) -> dict:
    """What the matching needs: the loaded cuBLAS version, the GPU, and MPS. The GPU is read only when this process
    already uses CUDA or the caller names a device: reading it initializes CUDA, which in vLLM's front-end process
    makes vLLM start its processes with spawn, and in a worker that hasn't selected its GPU yet reads GPU 0.
    (`sdcguard issues` is its own process and initializes CUDA first.)"""
    env: dict = {"cublas": None, "cublas_source": None, "gpu": None, "cc": None, "sms": None, "sms_full": None,
                 "mps_share": None, "mps": None}
    try:
        import torch
        if (device is not None or torch.cuda.is_initialized()) and torch.cuda.is_available():
            dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
            p = torch.cuda.get_device_properties(dev)
            env.update(gpu=p.name, cc=f"{p.major}.{p.minor}", sms=p.multi_processor_count)
    except Exception:  # noqa: BLE001
        pass
    try:
        from .blas import _library
        lib = _library()
        if lib is not None:
            v = int(lib.cublasLtGetVersion())
            env.update(cublas=f"{v // 10000}.{(v // 100) % 100}.{v % 100}", cublas_source="loaded cuBLASLt")
    except Exception:  # noqa: BLE001
        pass
    if env["cublas"] is None:
        try:
            from .check import software
            v = software().get("cublas")
            if v:
                env.update(cublas=v, cublas_source="installed nvidia-cublas package")
        except Exception:  # noqa: BLE001
            pass
    share = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    try:
        env["mps_share"] = float(share) if share is not None else None
    except ValueError:
        env["mps_share"] = None
    env["sms_full"], env["mps"] = mps_state(env["gpu"], env["sms"], env["mps_share"],
                                            os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    return env


def match(env: Optional[dict] = None, issues: Optional[list] = None) -> list:
    env = env if env is not None else environment()
    ver = _vt(env.get("cublas"))
    out = []
    if ver is None:
        return out
    for it in issues if issues is not None else ISSUES:
        hit = affected(it, ver)
        if hit is False:
            continue
        gpu = _cc_match(it.get("cc"), env.get("cc"))
        if gpu is False:
            continue
        lo, hi = it.get("introduced"), it.get("fixed")
        rng = (f"from {_show(lo)}" if lo else "NVIDIA gives no first version") + (
            f", fixed in {_show(hi)}" if hi else ", not fixed in a CUDA toolkit yet")
        fixes = it.get("branches") or {}
        if fixes:
            rng += "; patched on " + ", ".join(f"{a}.{b}.x in {_show(v)}" for (a, b), v in sorted(fixes.items()))
        notes = [n for n in (it.get("introduced_note"), it.get("note")) if n]
        why = [f"cuBLAS {env.get('cublas')} ({rng})" + "".join(": " + n for n in notes)]
        if it.get("cc"):
            why.append(f"compute capability {env.get('cc')}" if gpu else "the GPU couldn't be read")
        status = "may apply"
        if it.get("needs") == "mps_partial":
            mode = env.get("mps") or ""
            if mode.startswith("partial"):
                status = "applies" if gpu and hit else "may apply"
                why.append(f"MPS {mode}" + (f", share {env['mps_share']:g}%" if env.get("mps_share") is not None
                                            else ""))
            elif mode == "full" or mode.startswith("no "):
                status = "not active"
                why.append("this process can use every SM: not an MPS client below 100%")
            else:
                why.append(f"MPS {mode}" if mode else "whether this process is an MPS client below 100% couldn't "
                                                       "be told (no GPU visible)")
        out.append({"id": it["id"], "status": status, "what": it["what"], "when": it.get("when"),
                    "workaround": it.get("workaround"), "why": why, "source": NOTES})
    order = {"applies": 0, "may apply": 1, "not active": 2}
    return sorted(out, key=lambda m: order[m["status"]])


def report(device=None) -> dict:
    env = environment(device)
    return {"environment": env, "matches": match(env),
            "table_from": f"{NOTES} and {PATCHES} (read Oct 20, 2026)"}


def log_once(device=None, logger=None) -> list:
    """Log the issues that apply to this machine, once per process (used by the training and serving guards)."""
    if _LOGGED.get("done"):
        return []
    _LOGGED["done"] = True
    try:
        hits = [m for m in match(environment(device)) if m["status"] == "applies"]
    except Exception:  # noqa: BLE001
        return []
    import logging
    log = logger or logging.getLogger("sdcguard")
    for m in hits:
        log.warning("sdcguard: NVIDIA cuBLAS known issue %s applies to this machine: %s. %s", m["id"], m["what"],
                    m.get("workaround") or "")
    return hits


_LOGGED: dict = {}


def main(argv=None) -> int:
    """sdcguard issues: which of NVIDIA's listed cuBLAS wrong-result issues apply here. Exit 1 when one applies,
    0 otherwise (2 when the cuBLAS version couldn't be read)."""
    import argparse
    p = argparse.ArgumentParser(prog="sdcguard issues", description=main.__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    r = report(a.device)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        e = r["environment"]
        print(f"cuBLAS {e['cublas'] or 'not found'} ({e['cublas_source'] or 'no source'}), "
              f"{e['gpu'] or 'no CUDA GPU'}" + (f" (compute capability {e['cc']}, {e['sms']} SMs)" if e["cc"] else "")
              + (f", MPS {e['mps']}" if e["mps"] else ""))
        if not r["matches"]:
            print("No listed cuBLAS wrong-result issue matches this machine."
                  if e["cublas"] else "Couldn't read the cuBLAS version, so nothing could be matched.")
        for m in r["matches"]:
            print(f"\n{m['status'].upper()}: {m['id']}: {m['what']}")
            print("  " + "; ".join(m["why"]))
            if m["status"] == "may apply" and m.get("when"):
                print(f"  only if {m['when']}")
            if m.get("workaround"):
                print(f"  workaround: {m['workaround']}")
        print(f"\nFrom NVIDIA's release notes ({NOTES}). `sdcguard blas` tests the algorithms themselves.")
    if not r["environment"]["cublas"]:
        return 2
    return 1 if any(m["status"] == "applies" for m in r["matches"]) else 0
