"""sdcguard blas (blas.py): the parts that don't need a GPU. The sweep itself runs on GPUs
(scripts/modal_blas.py), including under MPS where cuBLAS known issue 6862950 makes algorithm 66 go wrong."""
import ctypes

import pytest
import torch

from sdcguard import blas


def test_cluster_shape_ids_from_the_header():
    # spot checks of cublasLtClusterShape_t (CUDA 13 cublasLt.h): every id's CTA count was compared with the
    # header when the table was written; these guard against edits
    assert blas.cluster_ctas(0) == 0 and blas.cluster_ctas(1) == 0          # auto, unknown
    assert blas.cluster_ctas(2) == 1                                          # 1x1x1
    assert blas.cluster_ctas(4) == 4 and blas.cluster_ctas(6) == 4           # 4x1x1, 2x2x1
    assert blas.cluster_ctas(5) == 2 and blas.cluster_ctas(10) == 16         # 1x2x1, 4x4x1
    assert blas.cluster_ctas(28) == 6 and blas.cluster_ctas(32) == 3         # 3x2x1, 1x3x1
    assert blas.cluster_ctas(10_000) == 0


def test_struct_layouts_match_cublaslt():
    assert ctypes.sizeof(blas._Algo) == 64 and ctypes.sizeof(blas._Heuristic) == 96


def test_parse_shape():
    assert blas.parse_shape("4096x11008x2048") == (4096, 11008, 2048)
    with pytest.raises(ValueError):
        blas.parse_shape("4096x0x1")
    with pytest.raises(ValueError):
        blas.parse_shape("4096x2")


def test_verdicts():
    good = {"algo_id": 66, "cluster_ctas": 1, "correct": True}
    bad = {"index": 3, "algo_id": 66, "cluster_shape_id": 4, "cluster_ctas": 4, "correct": False,
           "unwritten_or_nonfinite": 4096, "wrong_sampled": 12}
    workaround = dict(good, variant="cluster_1x1x1")
    r = blas.summarize({"algorithms": [good, bad, workaround]})
    assert r["verdict"] == "FAIL" and r["algorithms_tested"] == 2
    assert r["wrong_algorithms"] == [{"index": 3, "algo_id": 66, "cluster_shape_id": 4, "cluster_ctas": 4,
                                      "splitk": None, "why": None, "unwritten_or_nonfinite": 4096,
                                      "wrong_sampled": 12}]
    split = blas.summarize({"algorithms": [dict(bad, splitk=1)]})["wrong_algorithms"][0]
    assert split["splitk"] == 1                  # what rules a split-K bug in or out for that configuration
    assert blas.summarize({"algorithms": [good, workaround]})["verdict"] == "PASS"
    assert blas.summarize({"algorithms": []})["verdict"] == "COULD NOT TEST"
    assert blas.summarize({"error": "x", "algorithms": [good]})["verdict"] == "COULD NOT TEST"
    # a wrong workaround variant fails the shape too, named apart: it isn't an algorithm the heuristic offers,
    # but it computed wrong on this GPU, and says the workaround doesn't hold here (a PASS used to hide it)
    r = blas.summarize({"algorithms": [good, dict(workaround, correct=False)]})
    assert r["verdict"] == "FAIL" and r["wrong_algorithms"] == [] and r["workaround_wrong"][0]["algo_id"] == 66


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU path")
def test_without_a_gpu_it_says_so():
    assert blas.sweep(64, 64, 8)["verdict"] == "COULD NOT TEST"
    assert blas.main([]) == 2
    with pytest.raises(ValueError):
        blas.sweep(64, 64, 8, dtype=torch.float32)


def test_check_reports_wrong_algorithms_from_the_sweep(monkeypatch):
    """sdcguard check's library part: wrong algorithms are listed with their shape; the verdict says FAIL for the
    library (reported in notes, not the GPU's result, until the sweep is measured on GPUs)."""
    import torch
    from sdcguard import blas, check
    monkeypatch.setattr(blas, "available", lambda: (True, ""))

    def fake(m, k, n, dtype, device, max_algos):
        wrong = [{"index": 0, "algo_id": 66, "cluster_shape_id": 13, "cluster_ctas": 4, "why": "unwritten"}] \
            if (m, k, n) == check.LIBRARY_SHAPES[0] else []
        return {"algorithms_tested": 8, "wrong_algorithms": wrong}
    monkeypatch.setattr(blas, "sweep", fake)
    r = check.library_sweep(torch.device("cuda", 0), torch.bfloat16)
    assert r["verdict"] == "FAIL" and r["algorithms_tested"] == 16 and r["wrong"][0]["algo_id"] == 66
    assert r["wrong"][0]["shape"] == "4096x4096x512"
    monkeypatch.setattr(blas, "sweep", lambda *a, **k: {"error": "no algorithm"})
    assert check.library_sweep(torch.device("cuda", 0), torch.bfloat16)["verdict"] == "COULD NOT TEST"


def test_blas_doesnt_call_an_error_status_a_wrong_result(monkeypatch, capsys):
    """sweep() recorded an algorithm whose cublasLtMatmul returned a non-zero status as correct=False, summarize()
    listed it in wrong_algorithms, and `sdcguard blas` printed 'FAIL ... algorithms wrong' and 'At least one
    cuBLASLt algorithm computed wrong results here', exit 1 (sdcguard check's note said the same). A status such as
    CUBLAS_STATUS_NOT_SUPPORTED is a call that didn't run: that shape COULD NOT be TESTED (exit 2), unless another
    algorithm computed wrong."""
    monkeypatch.setattr(blas, "available", lambda: (True, ""))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA H200")
    good = {"index": 0, "algo_id": 66, "cluster_ctas": 1, "correct": True, "status": 0}
    refused = {"index": 1, "algo_id": 66, "cluster_ctas": 4, "status": 15, "correct": None, "why": "cuBLAS status 15"}
    r = blas.summarize({"algorithms": [dict(good), dict(refused)]})
    assert r["verdict"] == "COULD NOT TEST" and r["wrong_algorithms"] == [] and r["algorithms_tested"] == 1
    assert r["algorithms_not_run"][0]["status"] == 15
    wrong = {"index": 2, "algo_id": 66, "cluster_ctas": 4, "status": 0, "correct": False}
    assert blas.summarize({"algorithms": [dict(good), dict(refused), wrong]})["verdict"] == "FAIL"
    monkeypatch.setattr(blas, "sweep", lambda *a, **k: blas.summarize({"dtype": "bfloat16",
                                                                        "algorithms": [dict(good), dict(refused)]}))
    code = blas.main(["--shape", "4096x4096x1", "--dtype", "bf16"])
    out = capsys.readouterr().out
    assert code == 2 and "computed wrong" not in out and "COULD NOT TEST" in out, out
    from sdcguard import check
    lib = check.library_sweep(torch.device("cuda", 0), torch.bfloat16)
    assert lib["verdict"] == "COULD NOT TEST" and lib["wrong"] == [] and lib["problems"]


def test_forced_split_k_runs_are_judged_apart_and_a_wrong_one_fails_the_shape(monkeypatch, capsys):
    """NVIDIA issue 6580689: algorithm 66 forced to a split-K count that doesn't divide K evenly computes wrong in
    cuBLAS 13.7.0. A forced configuration that ran and computed wrong fails the shape, named apart (it isn't one the
    heuristic offered); right ones are counted; the command line passes --splitk through and prints the wrong ones."""
    good = {"index": 0, "algo_id": 66, "cluster_ctas": 1, "correct": True, "status": 0}
    sk_ok = dict(good, variant="splitk_3", splitk=3, reduction_scheme=1)
    sk_bad = dict(good, variant="splitk_5", splitk=5, reduction_scheme=1, correct=False, wrong_sampled=40)
    r = blas.summarize({"algorithms": [dict(good), sk_ok]})
    assert r["verdict"] == "PASS" and r["splitk_tested"] == 1 and r["splitk_wrong"] == []
    assert r["algorithms_tested"] == 1                      # (forced runs aren't counted as the heuristic's)
    r = blas.summarize({"algorithms": [dict(good), sk_ok, sk_bad]})
    assert r["verdict"] == "FAIL" and r["wrong_algorithms"] == [] and r["workaround_wrong"] == []
    assert [(w["algo_id"], w["splitk"]) for w in r["splitk_wrong"]] == [(66, 5)]
    seen = {}
    monkeypatch.setattr(blas, "available", lambda: (True, ""))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA H100 80GB HBM3")

    def fake(*a, **k):
        seen.update(k)
        return blas.summarize({"dtype": "bfloat16", "algorithms": [dict(good), dict(sk_bad)]})
    monkeypatch.setattr(blas, "sweep", fake)
    code = blas.main(["--shape", "4096x4096x512", "--dtype", "bf16", "--splitk", "3,5,7"])
    out = capsys.readouterr().out
    assert code == 1 and seen["splitk"] == (3, 5, 7) and "algo 66 split-K 5" in out, out
    with pytest.raises(SystemExit):
        blas.main(["--splitk", "three"])


def test_issue_6580689_condition_follows_nvidias_words():
    """'the number of K splits does not evenly divide the K dimension and the floor division of K / SPLITK_NUM is an
    exact multiple of the stage size' (CUDA 13.4 release notes), the stage size read from the stages id."""
    assert blas.stage_size(35) == 64 and blas.stage_size(38) == 768 and blas.stage_size(15) == 64
    assert blas.stage_size(0) is None and blas.stage_size(None) is None
    assert blas.issue_6580689(768 * 9 + 8, 9, 35) is True          # 6920 // 9 = 768, a multiple of 64
    assert blas.issue_6580689(768 * 9 + 8, 9, None) is True        # 768: a multiple of every stage size there is
    assert blas.issue_6580689(4096, 8, 35) is False                # divides K evenly
    assert blas.issue_6580689(4096, 3, 35) is False                # 1365: not a multiple of 64
    assert blas.issue_6580689(4096, 1, 35) is False and blas.issue_6580689(4096, None, 35) is False
    assert blas.issue_6580689(4105, 9, None) is None               # 456: a multiple of 8, not of 16: depends


class _FakeLt:
    """cuBLASLt in Python for sweep() on the CPU: three offered configurations (two algorithm 66 ones and, third,
    an algorithm 66 the heuristic itself splits 9 ways), split-K counts it accepts, a reduction scheme whose
    workspace is too large, and wrong output exactly when issue 6580689's condition holds. Attributes are read only
    with their header's size, like the real library."""
    SIZES = {0: 4, 1: 4, 2: 4, 3: 4, 6: 4, 8: 2}
    WORD = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4, 8: 5}

    def __init__(self):
        self.layouts, self.calls = {}, []

    def cublasLtGetVersion(self):
        return 130700

    def _set(self, ref, value):
        ref._obj.value = value
        return 0

    def cublasLtCreate(self, h):
        return self._set(h, 1)

    def cublasLtMatmulDescCreate(self, d, *a):
        return self._set(d, 2)

    def cublasLtMatmulDescSetAttribute(self, *a):
        return 0

    def cublasLtMatrixLayoutCreate(self, lay, t, rows, cols, ld):
        key = 10 + len(self.layouts)
        self.layouts[key] = (rows.value, cols.value)
        return self._set(lay, key)

    def cublasLtMatmulPreferenceCreate(self, p):
        return self._set(p, 3)

    def cublasLtMatmulPreferenceSetAttribute(self, *a):
        return 0

    def cublasLtMatmulAlgoGetHeuristic(self, handle, desc, la, lb, lc, ld, pref, n, heur, found):
        offered = [(66, 7, 1, 0, 35, 2), (66, 9, 1, 0, 35, 4), (66, 9, 9, 1, 35, 2), (39, 3, 1, 0, 0, 0)]
        for i, words in enumerate(offered):
            for w, v in enumerate(words):
                heur[i].algo.data[w] = v
            heur[i].state = 0
        found._obj.value = len(offered)
        return 0

    def cublasLtMatmulAlgoConfigGetAttribute(self, algo, which, out, size, got):
        if size != self.SIZES[which]:
            return 7                                      # CUBLAS_STATUS_INVALID_VALUE: nothing written
        out._obj.value = algo._obj.data[self.WORD[which]]
        return 0

    def cublasLtMatmulAlgoConfigSetAttribute(self, algo, which, val, size):
        if size != self.SIZES[which] or (which == 2 and val._obj.value == 13):
            return 7                                      # (a split count this fake refuses)
        algo._obj.data[self.WORD[which]] = val._obj.value
        return 0

    def cublasLtMatmulAlgoCheck(self, handle, desc, la, lb, lc, ld, algo, chk):
        chk._obj.workspaceSize = (1 << 40) if algo._obj.data[3] == 2 else 4096
        chk._obj.wavesCount = 1.5
        return 0

    def cublasLtMatmul(self, handle, desc, alpha, A, la, B, lb, beta, C, lc, D, ld, algo, ws, wsize, stream):
        k, m = self.layouts[la.value]
        n = self.layouts[lb.value][1]

        def view(ptr, count):
            return torch.frombuffer((ctypes.c_uint16 * count).from_address(ptr.value), dtype=torch.int16).view(
                torch.bfloat16)
        W, X, Y = view(A, m * k).view(m, k), view(B, n * k).view(n, k), view(D, n * m).view(n, m)
        out = (X.double() @ W.double().t()).to(torch.bfloat16)
        d = algo._obj.data
        self.calls.append((d[0], d[2], d[3]))
        if d[0] == 66 and blas.issue_6580689(k, int(d[2]), int(d[4])):
            out[0, 0] += 1000.0
        Y.copy_(out)
        return 0

    def __getattr__(self, name):
        if name.endswith("Destroy"):
            return lambda *a: 0
        raise AttributeError(name)


def test_the_sweep_forces_split_counts_and_judges_them_against_nvidias_condition(monkeypatch):
    """sweep() against _FakeLt: every algorithm 66 configuration is forced to each count with each reduction scheme
    the library accepts; the forced split is read back; a scheme needing too much workspace and a count the
    library refuses are listed with why; outputs are compared with the unforced run; only runs in the issue's
    condition compute wrong, and the heuristic's own split-9 configuration is caught as offered."""
    import contextlib
    from types import SimpleNamespace
    lib = _FakeLt()
    monkeypatch.setattr(blas, "available", lambda: (True, ""))
    monkeypatch.setattr(blas, "_library", lambda: lib)
    monkeypatch.setattr(torch.cuda, "device", lambda d: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    k = 768 * 9 + 8
    r = blas.sweep(64, k, 16, device="cpu", splitk=(9, 10, 13), workspace_mib=1)
    assert "error" not in r, r.get("error")
    assert r["cublaslt_version"] == 130700 and r["compute_capability"] == [9, 0] and r["algo66_offered"] == 3
    plain = [a for a in r["algorithms"] if not a["variant"]]
    assert [a["tile_id"] for a in plain] == [7, 9, 9, 3] and [a["stages_id"] for a in plain] == [35, 35, 35, 0]
    assert [a["stage_size"] for a in plain] == [64, 64, 64, None]       # (uint32 reads: a 2-byte buffer gets none)
    assert [w["index"] for w in r["wrong_algorithms"]] == [2]           # the heuristic's own split-9 configuration
    assert r["heuristic_in_condition"] == [{"index": 2, "splitk": 9, "stages_id": 35, "stage_size": 64,
                                            "correct": False, "issue_6580689_condition": True}]
    forced = [a for a in r["algorithms"] if a["variant"].startswith("splitk_")]
    # 3 configurations x 2 accepted counts (13 refused) x 2 schemes (scheme 2's workspace is too large)
    assert len(forced) == 12 and {a["variant"] for a in forced} == {"splitk_9_r1", "splitk_9_r4", "splitk_10_r1",
                                                                     "splitk_10_r4"}
    assert all(a["forced"] and a["splitk"] == a["requested_splitk"] and a["check_waves"] == 1.5 for a in forced)
    # evidence of a real split: bits unlike the unforced run, or a reduction that needs workspace (scheme 4 here)
    applied = {(a["index"], a["requested_splitk"], a["reduction_scheme"]): a["split_applied"] for a in forced}
    assert applied[(0, 10, 1)] is False and applied[(0, 10, 4)] is True and applied[(0, 9, 1)] is True
    assert applied[(2, 9, 1)] is False         # already split 9 ways and wrong unforced: the same bits, no evidence
    assert r["splitk_tested"] == 12 and r["splitk_tested_in_condition"] == 5 and r["splitk_not_run"] == []
    assert {(w["requested_splitk"], w["issue_6580689_condition"]) for w in r["splitk_wrong"]} == {(9, True)}
    assert len(r["splitk_wrong"]) == 6
    ten = [a for a in forced if a["requested_splitk"] == 10]
    assert all(a["correct"] and a["issue_6580689_condition"] is False for a in ten)
    assert [a["same_bits_as_unforced"] for a in ten if a["index"] in (0, 1)] == [True] * 4
    assert all(a["same_bits_as_unforced"] is False for a in forced if a["requested_splitk"] == 9 and a["index"] < 2)
    why = {(x["splitk"], x["reduction_scheme"]): x["why"] for x in r["splitk_refused"]}
    assert why[(13, 1)] == "split-K count not accepted" and "workspace" in why[(9, 2)]
    assert len(r["splitk_refused"]) == 3 * (3 + 2)                       # per config: 13 x3 schemes, 9 and 10 x r2
    assert r["verdict"] == "FAIL"
    lib2 = _FakeLt()
    monkeypatch.setattr(blas, "_library", lambda: lib2)
    r2 = blas.sweep(64, k, 16, device="cpu")                             # no splitk: nothing forced, nothing kept
    assert not [a for a in r2["algorithms"] if a["variant"].startswith("splitk_")] and r2["splitk_tested"] == 0
