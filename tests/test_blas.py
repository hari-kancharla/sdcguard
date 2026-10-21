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
                                      "why": None, "unwritten_or_nonfinite": 4096, "wrong_sampled": 12}]
    assert blas.summarize({"algorithms": [good, workaround]})["verdict"] == "PASS"
    assert blas.summarize({"algorithms": []})["verdict"] == "COULD NOT TEST"
    assert blas.summarize({"error": "x", "algorithms": [good]})["verdict"] == "COULD NOT TEST"
    # a wrong workaround variant alone doesn't fail the shape: the workaround isn't what PyTorch runs
    assert blas.summarize({"algorithms": [good, dict(workaround, correct=False)]})["verdict"] == "PASS"


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
