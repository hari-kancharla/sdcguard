"""Every cuBLASLt kernel a matmul could run, each checked against float64.

PyTorch runs a matmul with whichever cuBLASLt algorithm the library's heuristic picks for that shape, data type,
GPU and setting (for example the share of the GPU an MPS client gets). A library defect can sit in one of the
other algorithms and show up only when the heuristic switches to it: cuBLAS known issue 6862950 (CUDA 13.4
Update 2 release notes) gives wrong FP16, BF16 and FP8 results on Hopper under MPS below 100% with algorithm 66
and clusters of 4 or more CTAs only. A test that runs one GEMM, the way DCGM's diagnostic and gpu-burn do, sees
whichever algorithm it happens to get (docs/METHOD.md, head-to-head).

sweep() asks cuBLASLt for every algorithm its heuristic offers for a shape (up to 16), runs each one into an
output filled with NaN, and checks every output element for having been written and a sample of rows against a
float64 reference with a rigorous rounding bound. It talks to the cuBLASLt library PyTorch already loaded,
through ctypes: no compiler, no CUDA toolkit.

    from sdcguard.blas import sweep
    r = sweep(4096, 4096, 2048)                 # out features, in features, tokens (a linear layer's GEMM)
    r["wrong_algorithms"]                       # [] when every algorithm computed correctly

Run it in the same environment as the job (same MPS share, library versions and GPU) to test the kernels the
job can actually get. `sdcguard blas` is the command-line version, and `sdcguard check` runs a small sweep.
Constants come from cublasLt.h of CUDA 13 (cuBLAS 13.1); the calls are the same in CUDA 12.
"""
from __future__ import annotations

import ctypes
import glob
import math
import os
from typing import Optional

import torch

# cublasLt.h / cublas_api.h / library_types.h
_CUBLAS_COMPUTE_32F = 68
_CUDA_R_32F, _CUDA_R_16F, _CUDA_R_16BF = 0, 2, 14
_DESC_TRANSA, _DESC_TRANSB = 3, 4
_OP_N, _OP_T = 0, 1
_PREF_MAX_WORKSPACE_BYTES = 1
_CFG_ID, _CFG_TILE_ID, _CFG_SPLITK_NUM, _CFG_STAGES_ID, _CFG_CLUSTER_SHAPE_ID = 0, 1, 2, 6, 8
_CLUSTER_1x1x1 = 2
# CTAs per cluster for cublasLtClusterShape_t ids (x * y; 0 = auto or unknown)
_CLUSTER_X = [0, 0, 1, 2, 4, 1, 2, 4, 1, 2, 4, 8, 1, 8, 2, 16, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
              3, 5, 6, 7, 1, 2, 3, 4, 5, 3, 1, 2, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
_CLUSTER_Y = [0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 1, 8, 2, 8, 1, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 10, 11, 12, 13, 14, 15]
_DTYPES = {torch.float16: _CUDA_R_16F, torch.bfloat16: _CUDA_R_16BF}
_UNIT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}     # unit roundoff of the output format
_U32 = 2.0 ** -24


def cluster_ctas(shape_id: int) -> int:
    if shape_id < 2 or shape_id >= len(_CLUSTER_X):
        return 0
    return _CLUSTER_X[shape_id] * _CLUSTER_Y[shape_id]


class _Algo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint64 * 8)]


class _Heuristic(ctypes.Structure):
    _fields_ = [("algo", _Algo), ("workspaceSize", ctypes.c_size_t), ("state", ctypes.c_int),
                ("wavesCount", ctypes.c_float), ("reserved", ctypes.c_int * 4)]


_LIB = {"lib": None, "why": None}


def _library():
    """cuBLASLt as PyTorch loaded it (found by its soname), or from the NVIDIA wheel's lib folder."""
    if _LIB["lib"] is not None or _LIB["why"] is not None:
        return _LIB["lib"]
    names = ["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]
    try:
        import nvidia
        for base in getattr(nvidia, "__path__", []):
            names += sorted(glob.glob(os.path.join(base, "*", "lib", "libcublasLt.so*")))
    except Exception:  # noqa: BLE001
        pass
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            lib.cublasLtGetVersion.restype = ctypes.c_size_t
            _LIB["lib"] = lib
            return lib
        except OSError:
            continue
    _LIB["why"] = "cuBLASLt library not found"
    return None


def available() -> tuple:
    """(True, "") when a sweep can run here, else (False, why)."""
    if not torch.cuda.is_available():
        return False, "no CUDA GPU"
    if _library() is None:
        return False, _LIB["why"] or "cuBLASLt library not found"
    return True, ""


class _Call:
    def __init__(self, lib):
        self.lib = lib

    def __call__(self, name, *args):
        st = getattr(self.lib, name)(*args)
        if st != 0:
            raise RuntimeError(f"{name} returned cuBLAS status {st}")


def _reference_rows(W64: torch.Tensor, X64: torch.Tensor, rows: list) -> tuple:
    """float64 outputs for the sampled tokens, and each one's rigorous fp32-accumulation error scale."""
    Xs = X64[rows]                                   # (r, k)
    ref = Xs @ W64.t()                               # (r, m)
    scale = Xs.abs() @ W64.abs().t()                 # sum_k |x||w|
    return ref, scale


def sweep(m: int, k: int, n: int, dtype: torch.dtype = torch.bfloat16, device=None, max_algos: int = 16,
          seed: int = 0, sample_rows: int = 32, workspace_mib: int = 32) -> dict:
    """Run every cuBLASLt algorithm offered for Y = X @ W.T (W: m x k, X: n x k) and judge each.

    An algorithm is wrong if it leaves any output element unwritten (still NaN) or non-finite, or a sampled
    output differs from float64 by more than the rounding bound (one unit of the output format plus
    k * 2^-24 * sum |x||w| for fp32 accumulation, a worst case). One whose call returns an error status didn't
    run (correct is None): it wasn't tested. Also runs every algorithm 66 with a cluster of 4 or more CTAs again
    with a 1x1x1 cluster where it accepts one (the documented workaround for issue 6862950); those runs keep the
    index of the algorithm they came from."""
    if dtype not in _DTYPES:
        raise ValueError("dtype must be torch.bfloat16 or torch.float16")
    ok, why = available()
    if not ok:
        return summarize({"error": why, "shape": [m, k, n], "dtype": str(dtype).replace("torch.", "")})
    lib = _library()
    call = _Call(lib)
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = ((torch.rand(m, k, generator=g) - 0.5)).to(dtype)
    X = ((torch.rand(n, k, generator=g) - 0.5)).to(dtype)
    rows = sorted({round(i * (n - 1) / max(1, sample_rows - 1)) for i in range(min(sample_rows, n))})
    ref, scale = _reference_rows(W.double(), X.double(), rows)
    tol = _UNIT[dtype] * ref.abs() + k * _U32 * scale + 1e-30
    Wd, Xd = W.to(dev).contiguous(), X.to(dev).contiguous()
    Y = torch.empty(n, m, dtype=dtype, device=dev)
    ws = torch.empty(workspace_mib << 20, dtype=torch.uint8, device=dev)
    t = _DTYPES[dtype]
    vp = ctypes.c_void_p
    handle, desc, la, lb, ld, pref = vp(), vp(), vp(), vp(), vp(), vp()
    out = {"shape": [m, k, n], "dtype": str(dtype).replace("torch.", ""), "gpu": torch.cuda.get_device_name(dev),
           "cublaslt_version": int(lib.cublasLtGetVersion()), "algorithms": [],
           "mps_active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}
    try:
        with torch.cuda.device(dev):
            call("cublasLtCreate", ctypes.byref(handle))
            call("cublasLtMatmulDescCreate", ctypes.byref(desc), _CUBLAS_COMPUTE_32F, _CUDA_R_32F)
            ta, tb = ctypes.c_int32(_OP_T), ctypes.c_int32(_OP_N)
            call("cublasLtMatmulDescSetAttribute", desc, _DESC_TRANSA, ctypes.byref(ta), ctypes.sizeof(ta))
            call("cublasLtMatmulDescSetAttribute", desc, _DESC_TRANSB, ctypes.byref(tb), ctypes.sizeof(tb))
            # column-major view of PyTorch's row-major tensors: W (m x k) is a k x m matrix, used transposed;
            # X (n x k) is k x n; Y (n x m) is m x n
            call("cublasLtMatrixLayoutCreate", ctypes.byref(la), t, ctypes.c_uint64(k), ctypes.c_uint64(m),
                 ctypes.c_int64(k))
            call("cublasLtMatrixLayoutCreate", ctypes.byref(lb), t, ctypes.c_uint64(k), ctypes.c_uint64(n),
                 ctypes.c_int64(k))
            call("cublasLtMatrixLayoutCreate", ctypes.byref(ld), t, ctypes.c_uint64(m), ctypes.c_uint64(n),
                 ctypes.c_int64(m))
            call("cublasLtMatmulPreferenceCreate", ctypes.byref(pref))
            wsize = ctypes.c_size_t(ws.numel())
            call("cublasLtMatmulPreferenceSetAttribute", pref, _PREF_MAX_WORKSPACE_BYTES, ctypes.byref(wsize),
                 ctypes.sizeof(wsize))
            heur = (_Heuristic * max_algos)()
            found = ctypes.c_int(0)
            call("cublasLtMatmulAlgoGetHeuristic", handle, desc, la, lb, ld, ld, pref, max_algos, heur,
                 ctypes.byref(found))
            stream = vp(torch.cuda.current_stream(dev).cuda_stream)
            alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)

            def attr(algo, which, ctype):
                v, got = ctype(0), ctypes.c_size_t(0)
                lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), which, ctypes.byref(v),
                                                         ctypes.sizeof(v), ctypes.byref(got))
                return int(v.value)

            def run(algo, index, variant=""):
                cluster = attr(algo, _CFG_CLUSTER_SHAPE_ID, ctypes.c_uint16)
                rec = {"index": index, "variant": variant, "algo_id": attr(algo, _CFG_ID, ctypes.c_int32),
                       "tile_id": attr(algo, _CFG_TILE_ID, ctypes.c_uint16),
                       "stages_id": attr(algo, _CFG_STAGES_ID, ctypes.c_uint16),
                       "splitk": attr(algo, _CFG_SPLITK_NUM, ctypes.c_int32),
                       "cluster_shape_id": cluster, "cluster_ctas": cluster_ctas(cluster)}
                Y.fill_(float("nan"))
                torch.cuda.synchronize(dev)
                st = lib.cublasLtMatmul(handle, desc, ctypes.byref(alpha), vp(Wd.data_ptr()), la,
                                        vp(Xd.data_ptr()), lb, ctypes.byref(beta), vp(Y.data_ptr()), ld,
                                        vp(Y.data_ptr()), ld, ctypes.byref(algo), vp(ws.data_ptr()), wsize,
                                        stream)
                torch.cuda.synchronize(dev)
                rec["status"] = int(st)
                if st != 0:          # the call didn't run (a status such as NOT_SUPPORTED): not a wrong result
                    rec.update(correct=None, why=f"cuBLAS status {st}")
                    return rec
                nonfinite = int((~torch.isfinite(Y)).sum())
                got = Y[rows].double().cpu()
                err = (got - ref).abs()
                bad = ~(err <= tol)                          # NaN counts as bad
                rec.update(unwritten_or_nonfinite=nonfinite, sampled=int(ref.numel()), wrong_sampled=int(bad.sum()),
                           worst_error_over_bound=float((err / tol).nan_to_num(posinf=1e30).max()))
                rec["correct"] = nonfinite == 0 and rec["wrong_sampled"] == 0
                return rec

            for i in range(found.value):
                if heur[i].state == 0:
                    out["algorithms"].append(run(heur[i].algo, i))
            # the documented workaround for issue 6862950, on every algorithm the issue concerns (algorithm 66 with
            # clusters of 4 or more CTAs): the same configuration forced to a 1x1x1 cluster
            for i in range(found.value):
                if heur[i].state != 0 or attr(heur[i].algo, _CFG_ID, ctypes.c_int32) != 66 or \
                        cluster_ctas(attr(heur[i].algo, _CFG_CLUSTER_SHAPE_ID, ctypes.c_uint16)) < 4:
                    continue
                a = _Algo()
                ctypes.memmove(ctypes.byref(a), ctypes.byref(heur[i].algo), ctypes.sizeof(_Algo))
                c1 = ctypes.c_uint16(_CLUSTER_1x1x1)
                chk = _Heuristic()
                if (lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), _CFG_CLUSTER_SHAPE_ID,
                                                             ctypes.byref(c1), ctypes.sizeof(c1)) == 0 and
                        lib.cublasLtMatmulAlgoCheck(handle, desc, la, lb, ld, ld, ctypes.byref(a),
                                                    ctypes.byref(chk)) == 0):
                    out["algorithms"].append(run(a, i, "cluster_1x1x1"))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        for fn, h in (("cublasLtMatmulPreferenceDestroy", pref), ("cublasLtMatrixLayoutDestroy", ld),
                      ("cublasLtMatrixLayoutDestroy", lb), ("cublasLtMatrixLayoutDestroy", la),
                      ("cublasLtMatmulDescDestroy", desc), ("cublasLtDestroy", handle)):
            if h.value:
                getattr(lib, fn)(h)
    return summarize(out)


def _ran(a: dict) -> bool:
    """Whether an algorithm's call ran: an error status (or no judgement) means it didn't."""
    return not a.get("status") and a.get("correct") is not None


def summarize(out: dict) -> dict:
    """Adds algorithms_tested, wrong_algorithms, algorithms_not_run and a one-line verdict to a sweep result.
    FAIL only for an algorithm that ran and computed wrong; one that returned an error status wasn't tested, so
    the shape is COULD NOT TEST unless another algorithm computed wrong."""
    algos = out.get("algorithms", [])
    plain = [a for a in algos if not a.get("variant")]
    ran = [a for a in plain if _ran(a)]
    out["algorithms_tested"] = len(ran)
    out["wrong_algorithms"] = [{k: a.get(k) for k in ("index", "algo_id", "cluster_shape_id", "cluster_ctas",
                                                      "splitk", "why", "unwritten_or_nonfinite", "wrong_sampled")}
                               for a in ran if not a.get("correct")]
    out["algorithms_not_run"] = [{k: a.get(k) for k in ("index", "algo_id", "cluster_shape_id", "cluster_ctas",
                                                        "splitk", "status", "why")} for a in plain if not _ran(a)]
    if out.get("error"):
        out["verdict"] = "COULD NOT TEST"
    elif not plain:
        out["verdict"] = "COULD NOT TEST"
        out["error"] = "the heuristic offered no algorithm"
    elif out["wrong_algorithms"]:
        out["verdict"] = "FAIL"
    elif out["algorithms_not_run"]:
        out["verdict"] = "COULD NOT TEST"
    else:
        out["verdict"] = "PASS"
    return out


DEFAULT_SHAPES = [(4096, 4096, 2048), (11008, 4096, 512), (1000, 2048, 16), (2048, 2048, 2048)]


def parse_shape(s: str) -> tuple:
    parts = [int(x) for x in s.lower().replace(",", "x").split("x")]
    if len(parts) != 3 or min(parts) < 1:
        raise ValueError(f"shape {s!r}: use OUTxINxTOKENS, e.g. 4096x4096x2048")
    return tuple(parts)


def main(argv=None) -> int:
    """sdcguard blas: run every cuBLASLt algorithm for some matmul shapes here and say which compute wrong.
    Exit 0 when all are right, 1 when any computed wrong, 2 when the sweep couldn't run."""
    import argparse
    import json
    p = argparse.ArgumentParser(prog="sdcguard blas", description=main.__doc__)
    p.add_argument("--shape", action="append", default=None,
                   help="OUTxINxTOKENS of a linear layer's matmul (repeatable); default: four LLM-like shapes")
    p.add_argument("--dtype", choices=["bf16", "fp16", "both"], default="both")
    p.add_argument("--device", default=None)
    p.add_argument("--json", default=None, help="also write every algorithm's result to this file")
    a = p.parse_args(argv)
    try:
        shapes = [parse_shape(s) for s in a.shape] if a.shape else DEFAULT_SHAPES
    except ValueError as e:              # (exit 2, with the usage: 1 means an algorithm computed wrong)
        p.error(str(e))
    dtypes = {"bf16": [torch.bfloat16], "fp16": [torch.float16], "both": [torch.bfloat16, torch.float16]}[a.dtype]
    ok, why = available()
    if not ok:
        print(f"sdcguard blas: can't run here: {why}")
        return 2
    results, failed, untested = [], False, False
    mps = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
    print(f"{torch.cuda.get_device_name(a.device)}" + (f", MPS share {mps}%" if mps else "") + ":")
    for shape in shapes:
        for dt in dtypes:
            r = sweep(*shape, dtype=dt, device=a.device)
            results.append(r)
            label = f"  {'x'.join(map(str, shape)):>18} {r.get('dtype', str(dt)):>9}"
            if r["verdict"] == "FAIL":
                failed = True
                bad = ", ".join(f"algo {w['algo_id']} ({w['cluster_ctas'] or '?'} CTA cluster)"
                                for w in r["wrong_algorithms"])
                print(f"{label}: FAIL, {len(r['wrong_algorithms'])} of {r['algorithms_tested']} algorithms wrong: {bad}")
            elif r["verdict"] == "PASS":
                print(f"{label}: PASS, all {r['algorithms_tested']} algorithms right")
            else:
                untested = True
                skipped = r.get("algorithms_not_run") or []
                why = r.get("error") or (f"{len(skipped)} of {len(skipped) + r['algorithms_tested']} algorithms "
                                         f"didn't run: {skipped[0].get('why')}")
                print(f"{label}: COULD NOT TEST ({why})")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)
    if failed:
        print("At least one cuBLASLt algorithm computed wrong results here. PyTorch may or may not pick it for your "
              "job: that depends on shape, data type and settings. Rerun under the job's exact settings, and see "
              "NVIDIA's cuBLAS known issues for your CUDA version.")
    return 1 if failed else 2 if untested else 0
