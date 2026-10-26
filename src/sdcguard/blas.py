"""Every cuBLASLt kernel a matmul could run, each checked against float64.

PyTorch runs a matmul with whichever cuBLASLt algorithm the library's heuristic picks for that shape, data type,
GPU and setting (for example the share of the GPU an MPS client gets). A library defect can sit in one of the
other algorithms and show up only when the heuristic switches to it: cuBLAS known issue 6862950 (CUDA 13.4
Update 1 and 2 release notes; since cuBLAS 12.6.3) gives wrong FP16, BF16 and FP8 results on Hopper under MPS below
100% with algorithm 66 and clusters of 4 or more CTAs only. A test that runs one GEMM, the way DCGM's diagnostic and gpu-burn do, sees
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
Constants come from cublasLt.h of CUDA 13 (cuBLAS 13.1; the stage and cluster ids are the same in 13.7); the calls
are the same in CUDA 12.
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
_CFG_REDUCTION_SCHEME = 3
_REDUCTION_SCHEMES = (1, 2, 4)        # cublasLtReductionScheme_t: in place, in the compute type, in the output type
# cublasLtMatmulAlgoConfigAttributes_t value types (cublasLt.h): a smaller buffer is refused with no value read
_CFG_TYPES = {_CFG_ID: ctypes.c_int32, _CFG_TILE_ID: ctypes.c_uint32, _CFG_SPLITK_NUM: ctypes.c_int32,
              _CFG_REDUCTION_SCHEME: ctypes.c_uint32, _CFG_STAGES_ID: ctypes.c_uint32,
              _CFG_CLUSTER_SHAPE_ID: ctypes.c_uint16}
# cublasLtMatmulStages_t: "size and number of stages in which elements are read into shared memory" (SIZExCOUNT);
# the stage size of each id, 0 (undefined) and unknown ids left out
_STAGE_SIZE = {**{i: 16 for i in range(1, 7)}, **{i: 32 for i in range(7, 13)}, **{i: 64 for i in range(13, 19)},
               **{i: 128 for i in range(19, 25)}, 25: 32, 26: 8, 27: 16, 28: 8, 31: 8, 32: 8, 33: 16, 34: 32, 35: 64,
               36: 128, 37: 256, 38: 768}
_STAGE_SIZES = sorted(set(_STAGE_SIZE.values()))
_CLUSTER_1x1x1 = 2
# CTAs per cluster for cublasLtClusterShape_t ids (x * y; 0 = auto or unknown)
_CLUSTER_X = [0, 0, 1, 2, 4, 1, 2, 4, 1, 2, 4, 8, 1, 8, 2, 16, 1, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15,
              3, 5, 6, 7, 1, 2, 3, 4, 5, 3, 1, 2, 3, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
_CLUSTER_Y = [0, 0, 1, 1, 1, 2, 2, 2, 4, 4, 4, 1, 8, 2, 8, 1, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7, 9, 10, 11, 12, 13, 14, 15]
_DTYPES = {torch.float16: _CUDA_R_16F, torch.bfloat16: _CUDA_R_16BF}
_UNIT = {torch.float16: 2.0 ** -11, torch.bfloat16: 2.0 ** -8}     # unit roundoff of the output format
_U32 = 2.0 ** -24


def stage_size(stages_id) -> Optional[int]:
    return _STAGE_SIZE.get(stages_id) if isinstance(stages_id, int) else None


def issue_6580689(k: int, splitk, stages_id) -> Optional[bool]:
    """Whether a split-K configuration meets the condition of cuBLAS issue 6580689 (CUDA 13.4 release notes: wrong
    output from algorithm 66 on compute capability 9.0, 10.x and 11.x "if the number of K splits (SPLITK_NUM) does
    not evenly divide the K dimension and the floor division of K / SPLITK_NUM is an exact multiple of the stage
    size"; since cuBLAS 12.6.3, resolved in CUDA 13.4 Update 1, cuBLAS 13.8.0). None when the stage size isn't
    known and the answer depends on it."""
    if not isinstance(splitk, int) or splitk <= 1 or k % splitk == 0:
        return False
    q = k // splitk
    size = stage_size(stages_id)
    if size is not None:
        return q % size == 0
    hits = {q % sz == 0 for sz in _STAGE_SIZES}
    return hits.pop() if len(hits) == 1 else None


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
          seed: int = 0, sample_rows: int = 32, workspace_mib: int = 32, splitk=(), splitk_algos=(66,),
          splitk_configs: int = 4) -> dict:
    """Run every cuBLASLt algorithm offered for Y = X @ W.T (W: m x k, X: n x k) and judge each.

    An algorithm is wrong if it leaves any output element unwritten (still NaN) or non-finite, or a sampled
    output differs from float64 by more than the rounding bound (one unit of the output format plus
    k * 2^-24 * sum |x||w| for fp32 accumulation, a worst case). One whose call returns an error status didn't
    run (correct is None): it wasn't tested. Also runs every algorithm 66 with a cluster of 4 or more CTAs again
    with a 1x1x1 cluster where it accepts one (the documented workaround for issue 6862950); those runs keep the
    index of the algorithm they came from. splitk: also run the first splitk_configs configurations of the
    algorithms splitk_algos offered, each forced to every split-K count in splitk with every reduction scheme
    cuBLASLt accepts for it (each its own variant, splitk_S_rR): issue 6580689 (see issue_6580689(), which each
    record of algorithm 66 with more than one split carries). A forced configuration records the workspace and waves
    cuBLASLt's check gave, whether its output has the same bits as the same configuration unforced, and from those
    whether the split was applied (split_applied: different bits, or a reduction needing workspace; the count read
    back from the configuration is the one just set, so it shows only that the setting was taken); ones cuBLASLt
    doesn't accept are listed (splitk_refused), not run. The rounding bound doesn't cover reduction schemes 1 and 4,
    which round each split's partial sum to the output type; runs of a fixed cuBLAS and counts outside the issue's
    condition are the check that it doesn't flag correct split-K results."""
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
           "compute_capability": list(torch.cuda.get_device_capability(dev)),
           "cublaslt_version": int(lib.cublasLtGetVersion()), "algorithms": [], "sampled_rows": len(rows),
           "workspace_bytes": workspace_mib << 20,
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

            def attr(algo, which):
                """The configuration attribute's value, None when cuBLASLt doesn't give one."""
                v, got = _CFG_TYPES[which](0), ctypes.c_size_t(0)
                st = lib.cublasLtMatmulAlgoConfigGetAttribute(ctypes.byref(algo), which, ctypes.byref(v),
                                                              ctypes.sizeof(v), ctypes.byref(got))
                return int(v.value) if st == 0 else None

            base = {}                          # heuristic index -> its output, for the forced variants' comparison

            def run(algo, index, variant="", keep=False):
                cluster = attr(algo, _CFG_CLUSTER_SHAPE_ID)
                stages = attr(algo, _CFG_STAGES_ID)
                rec = {"index": index, "variant": variant, "algo_id": attr(algo, _CFG_ID),
                       "tile_id": attr(algo, _CFG_TILE_ID), "stages_id": stages, "stage_size": stage_size(stages),
                       "splitk": attr(algo, _CFG_SPLITK_NUM), "reduction_scheme": attr(algo, _CFG_REDUCTION_SCHEME),
                       "cluster_shape_id": cluster, "cluster_ctas": cluster_ctas(cluster or 0)}
                if rec["algo_id"] == 66 and isinstance(rec["splitk"], int) and rec["splitk"] > 1:
                    rec["issue_6580689_condition"] = issue_6580689(k, rec["splitk"], stages)
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
                if keep:
                    base[index] = Y.clone()
                elif variant.startswith("splitk_") and index in base:
                    rec["same_bits_as_unforced"] = bool(torch.equal(Y, base[index]))
                return rec

            # the configurations whose split-K count is forced (issue 6580689): their own output is kept
            forced = [i for i in range(found.value) if heur[i].state == 0 and
                      attr(heur[i].algo, _CFG_ID) in tuple(splitk_algos)][:max(0, splitk_configs)] if splitk else []
            out["algo66_offered"] = sum(1 for i in range(found.value) if heur[i].state == 0 and
                                        attr(heur[i].algo, _CFG_ID) == 66)
            for i in range(found.value):
                if heur[i].state == 0:
                    out["algorithms"].append(run(heur[i].algo, i, keep=i in forced))
            # the documented workaround for issue 6862950, on every algorithm the issue concerns (algorithm 66 with
            # clusters of 4 or more CTAs): the same configuration forced to a 1x1x1 cluster
            for i in range(found.value):
                if heur[i].state != 0 or attr(heur[i].algo, _CFG_ID) != 66 or \
                        cluster_ctas(attr(heur[i].algo, _CFG_CLUSTER_SHAPE_ID) or 0) < 4:
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
            # forced split-K counts (issue 6580689): each configuration, count and accepted reduction scheme
            out["splitk_refused"] = []
            for i in forced:
                for sk in splitk:
                    for scheme in _REDUCTION_SCHEMES:
                        a = _Algo()
                        ctypes.memmove(ctypes.byref(a), ctypes.byref(heur[i].algo), ctypes.sizeof(_Algo))
                        v, r_, chk = ctypes.c_int32(int(sk)), ctypes.c_uint32(scheme), _Heuristic()
                        why = None
                        if lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), _CFG_SPLITK_NUM, ctypes.byref(v),
                                                                    ctypes.sizeof(v)) != 0:
                            why = "split-K count not accepted"
                        elif lib.cublasLtMatmulAlgoConfigSetAttribute(ctypes.byref(a), _CFG_REDUCTION_SCHEME,
                                                                      ctypes.byref(r_), ctypes.sizeof(r_)) != 0:
                            why = "reduction scheme not accepted"
                        else:
                            st = lib.cublasLtMatmulAlgoCheck(handle, desc, la, lb, ld, ld, ctypes.byref(a),
                                                             ctypes.byref(chk))
                            if st != 0:
                                why = f"cublasLtMatmulAlgoCheck status {st}"
                            elif chk.workspaceSize > ws.numel():
                                why = f"needs {chk.workspaceSize} bytes of workspace, more than {ws.numel()}"
                        if why:
                            out["splitk_refused"].append({"index": i, "splitk": int(sk), "reduction_scheme": scheme,
                                                          "why": why})
                            continue
                        rec = run(a, i, f"splitk_{int(sk)}_r{scheme}")
                        rec.update(requested_splitk=int(sk), forced=rec["splitk"] == int(sk),
                                   check_workspace_bytes=int(chk.workspaceSize),
                                   check_waves=round(float(chk.wavesCount), 4))
                        # The count read back is the one just set, so it can't show the library split K. Evidence
                        # that it did: output bits that differ from the same configuration unforced (another
                        # summation order), or a reduction in the compute or output type that needs workspace
                        same = rec.get("same_bits_as_unforced")
                        rec["split_applied"] = (True if same is False or (scheme in (2, 4) and chk.workspaceSize > 0)
                                                else None if same is None else False)
                        out["algorithms"].append(rec)
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
    # the workaround's forced 1x1x1 configurations aren't algorithms the heuristic offers, but one that ran and
    # computed wrong is a wrong result on this GPU, and says the workaround doesn't hold here
    out["workaround_wrong"] = [{k: a.get(k) for k in ("index", "algo_id", "variant", "why", "unwritten_or_nonfinite",
                                                      "wrong_sampled")}
                               for a in algos if a.get("variant") == "cluster_1x1x1" and _ran(a) and
                               not a.get("correct")]
    # configurations forced to a split-K count (issue 6580689): one that ran and computed wrong is a wrong result
    forced = [a for a in algos if str(a.get("variant") or "").startswith("splitk_")]
    keys = ("index", "algo_id", "variant", "splitk", "requested_splitk", "forced", "split_applied", "reduction_scheme",
            "stages_id", "stage_size", "issue_6580689_condition", "same_bits_as_unforced", "why",
            "unwritten_or_nonfinite", "wrong_sampled")
    out["splitk_tested"] = sum(1 for a in forced if _ran(a))
    # ran, with evidence the split was applied, and in the issue's condition: the runs that test it
    out["splitk_tested_in_condition"] = sum(1 for a in forced if _ran(a) and a.get("split_applied") and
                                            a.get("issue_6580689_condition"))
    out["splitk_wrong"] = [{k: a.get(k) for k in keys} for a in forced if _ran(a) and not a.get("correct")]
    out["splitk_not_run"] = [{k: a.get(k) for k in keys + ("status",)} for a in forced if not _ran(a)]
    # configurations the heuristic offers on its own in (or possibly in) the issue's condition: what a call can get
    out["heuristic_in_condition"] = [{k: a.get(k) for k in ("index", "splitk", "stages_id", "stage_size", "correct",
                                                            "issue_6580689_condition")}
                                     for a in plain if "issue_6580689_condition" in a and
                                     a["issue_6580689_condition"] is not False]
    if out.get("error"):
        out["verdict"] = "COULD NOT TEST"
    elif not plain:
        out["verdict"] = "COULD NOT TEST"
        out["error"] = "the heuristic offered no algorithm"
    elif out["wrong_algorithms"] or out["workaround_wrong"] or out["splitk_wrong"]:
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
    p.add_argument("--splitk", default=None, help="also force algorithm 66's configurations to these split-K counts "
                   "(comma-separated, e.g. 9,11: NVIDIA issue 6580689, whose condition needs K not divisible by the "
                   "count and K // count a multiple of the stage size)")
    p.add_argument("--workspace-mib", type=int, default=32, help="cuBLASLt workspace in MiB (default 32; split-K "
                   "reductions in the compute type need more for large shapes)")
    a = p.parse_args(argv)
    try:
        splitk = tuple(int(x) for x in a.splitk.split(",") if x.strip()) if a.splitk else ()
    except ValueError:
        p.error("--splitk takes comma-separated integers, e.g. 3,5,7")
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
            r = sweep(*shape, dtype=dt, device=a.device, splitk=splitk, workspace_mib=a.workspace_mib)
            results.append(r)
            label = f"  {'x'.join(map(str, shape)):>18} {r.get('dtype', str(dt)):>9}"
            if r["verdict"] == "FAIL":
                failed = True
                bad = ", ".join(f"algo {w['algo_id']} ({w['cluster_ctas'] or '?'} CTA cluster)"
                                for w in r["wrong_algorithms"])
                pad = " " * len(label)
                if r["wrong_algorithms"]:
                    print(f"{label}: FAIL, {len(r['wrong_algorithms'])} of {r['algorithms_tested']} algorithms the "
                          f"heuristic offers wrong: {bad}")
                else:
                    print(f"{label}: FAIL, all {r['algorithms_tested']} algorithms the heuristic offers right, but "
                          "configurations forced as below computed wrong")
                if r.get("splitk_wrong"):
                    print(f"{pad}  with split-K forced ({len(r['splitk_wrong'])} of {r['splitk_tested']} forced runs "
                          "wrong): " + ", ".join(
                              f"algo {w['algo_id']} split-K {w['splitk']} reduction {w['reduction_scheme']}"
                              + (" (in issue 6580689's condition)" if w.get("issue_6580689_condition") else "")
                              for w in r["splitk_wrong"]))
                if r.get("workaround_wrong"):
                    print(f"{pad}  the 1x1x1-cluster workaround computed wrong on "
                          + ", ".join(f"algo {w['algo_id']}" for w in r["workaround_wrong"]))
            elif r["verdict"] == "PASS":
                print(f"{label}: PASS, all {r['algorithms_tested']} algorithms right (every output written, "
                      f"{r.get('sampled_rows', 'a sample of')} rows checked against float64)"
                      + (f"; {r['splitk_tested']} forced split-K runs right ({r['splitk_tested_in_condition']} in "
                         f"issue 6580689's condition), {len(r.get('splitk_refused') or [])} refused by cuBLASLt"
                         if splitk else ""))
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
