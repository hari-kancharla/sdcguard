"""The known-issue report (known_issues.py): NVIDIA's listed cuBLAS wrong-result issues against a machine."""
import json

import pytest

from sdcguard import known_issues as ki


def _env(**kw):
    e = {"cublas": "13.1.1", "cublas_source": "test", "gpu": "NVIDIA H200", "cc": "9.0", "sms": 132,
         "mps_share": None, "mps": None}
    e.update(kw)
    return e


def _status(env, iid):
    return next((m["status"] for m in ki.match(env) if m["id"] == iid), None)


def test_6862950_applies_only_under_mps_below_100():
    assert _status(_env(mps="partial", mps_share=40.0, sms=52), "6862950") == "applies"
    assert _status(_env(mps="full", mps_share=100.0), "6862950") == "not active"
    assert _status(_env(mps="full"), "6862950") == "not active"            # not an MPS client at all
    assert _status(_env(), "6862950") == "may apply"                        # couldn't be told
    assert _status(_env(mps="partial (52 of 132 SMs visible)", sms=52), "6862950") == "applies"
    assert _status(_env(mps="configured (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is 40, but this process sees all 132 SMs)"),
                   "6862950") == "may apply"
    assert _status(_env(cc="10.0", gpu="NVIDIA B200", mps="partial", mps_share=40.0), "6862950") is None
    assert _status(_env(cublas="12.6.2", mps="partial", mps_share=40.0), "6862950") is None   # before it


def test_version_ranges_follow_each_branch():
    """NVIDIA fixes an issue per release branch: 5807900 is fixed in 13.1.1 (the patch for CUDA 13.0 Update 2) but
    still present in 13.2.0 and 13.2.1 (CUDA 13.1 and 13.1 Update 1), fixed again in 13.2.2 and from 13.3.0 on."""
    b200 = dict(gpu="NVIDIA B200", cc="10.0", sms=148)
    assert _status(_env(cublas="13.0.2", **b200), "5807900") == "may apply"
    assert _status(_env(cublas="13.1.1", **b200), "5807900") is None
    assert _status(_env(cublas="13.2.1", **b200), "5807900") == "may apply"
    assert _status(_env(cublas="13.2.2", **b200), "5807900") is None
    assert _status(_env(cublas="13.3.0", **b200), "5807900") is None
    assert _status(_env(cublas="12.6.4", **b200), "5807900") is None            # before 12.8
    assert _status(_env(cublas="13.8.0", **b200), "6798616") == "may apply"     # CUDA 13.4 Update 2's cuBLAS
    assert _status(_env(cublas="13.8.1", **b200), "6798616") is None            # the standalone patch
    assert _status(_env(cublas="13.8.0", **b200), "6580689") is None            # fixed in 13.4 Update 1
    assert _status(_env(cc="12.0", gpu="RTX PRO 6000", cublas="13.6.0"), "CUB-10445") == "may apply"
    assert _status(_env(cc="12.0", gpu="RTX PRO 6000", cublas="13.6.1"), "CUB-10445") is None
    assert _status(_env(cc="9.0", cublas="13.6.0"), "CUB-10445") is None       # another GPU generation


def test_version_strings():
    assert ki._vt("13.1.1.3") == (13, 1, 1) and ki._vt("12.8") == (12, 8, 0) and ki._vt(None) is None


def test_nothing_matches_without_a_version():
    assert ki.match(_env(cublas=None)) == []


def test_cli_json_runs_anywhere(capsys):
    code = ki.main(["--json"])
    out = json.loads(capsys.readouterr().out)
    assert "environment" in out and "matches" in out and code in (0, 1, 2)


def test_mps_state_rests_on_the_sms_this_process_can_use():
    assert ki.mps_state("NVIDIA H200", 52, 40.0)[1].startswith("partial")
    assert ki.mps_state("NVIDIA H200", 52, None)[1].startswith("partial")      # set by the daemon's default
    assert ki.mps_state("NVIDIA H200", 132, 40.0)[1].startswith("configured")  # the variable alone isn't enough
    assert ki.mps_state("NVIDIA H200", 132, 100.0) == (132, "full")
    assert ki.mps_state("NVIDIA H100 80GB HBM3 MIG 3g.40gb", 60, None)[1].startswith("no")
    assert ki.mps_state("NVIDIA H200", 52, None, "MIG-1234")[1].startswith("no")
    assert ki.mps_state(None, None, None) == (None, None)


@pytest.mark.parametrize("iid,env,fixed", [
    # CUDA 13.0 Update 1 (cuBLAS 13.0.2) resolved: cublasLtMatmul with INT8 inputs, INT32 accumulation and output
    # could give incorrect results with A and B in regular order (CUBLASLT_ORDER_COL or ROW)
    ("CUB-8874", dict(cublas="13.0.0", cc="9.0"), "13.0.2"),
    # known in CUDA 13.0, resolved in 13.0 Update 1: some cublasSsyrkx kernels, beta = 0, Blackwell
    ("CUB-8846", dict(cublas="13.0.0", cc="10.0", gpu="NVIDIA B200"), "13.0.2"),
    # resolved in CUDA 13.0, and by the patches 12.8.5, 12.9.2, 13.1.1 and 13.2.2: ztrmm, m = 1, side right, Ada and
    # Blackwell GeForce-class GPUs
    ("5452663", dict(cublas="13.2.1", cc="12.0", gpu="NVIDIA RTX PRO 6000"), "13.2.2"),
    # known in CUDA 13.1 Update 1, resolved in 13.2 (cuBLAS 13.3.0): complex FP64 emulation, 2*m*n + m > UINT_MAX
    ("5720478", dict(cublas="13.2.1", cc="10.0", gpu="NVIDIA B200"), "13.3.0"),
    # resolved in CUDA 13.3 Update 1 (cuBLAS 13.6.0): cublasXt<t>spmm, m > 46340
    ("6155165", dict(cublas="13.5.1"), "13.6.0"),
    # resolved in CUDA 13.3 Update 1: an unsupported type combination ran with A and B read as FP32
    ("CUB-9942", dict(cublas="13.5.1"), "13.6.0"),
    # resolved in CUDA 13.4 (cuBLAS 13.7.0): cublasZgemv, CUBLAS_OP_C, incy > 1, under FP64 emulation
    ("6207926", dict(cublas="13.6.0", cc="10.0", gpu="NVIDIA B200"), "13.7.0"),
    # resolved in CUDA 12.8 Update 1 (cuBLAS 12.8.4): m or n = 1 with CUBLASLT_EPILOGUE_BIAS
    ("5104822", dict(cublas="12.8.3"), "12.8.4"),
    # resolved in CUDA 12.8 (cuBLAS 12.8.3): BGRADA/BGRADB epilogues with a row-major output
    ("4910924", dict(cublas="12.6.4"), "12.8.3"),
    # known in CUDA 12.6 Update 2, resolved in 12.6 Update 3 (cuBLAS 12.6.4): FP8 amax on Hopper, algorithm 66
    ("4941052", dict(cublas="12.6.3"), "12.6.4"),
    ("CUB-7596", dict(cublas="12.6.3"), "12.6.4"),
    # resolved in CUDA 12.6 Update 2 (cuBLAS 12.6.3): INT8 with INT8 output, FP32 scales and split-K
    ("4751576", dict(cublas="12.6.2"), "12.6.3"),
])
def test_nvidias_listed_wrong_result_issues_are_in_the_table(iid, env, fixed):
    """The docstring says the table follows NVIDIA's notes from CUDA 12.6 Update 2 to 13.4 Update 2, but these
    listed wrong-result issues were missing: `sdcguard issues` printed 'No listed cuBLAS wrong-result issue matches
    this machine' on affected versions. Each one matches before its fix, and not from the fixed version on."""
    assert iid in {m["id"] for m in ki.match(_env(**env))}
    assert iid not in {m["id"] for m in ki.match(_env(**{**env, "cublas": fixed}))}


def test_a_fix_on_one_branch_isnt_a_fix_on_another():
    """5452663 is fixed in CUDA 13.0 and in four cuBLAS patches, and listed fixed in no later toolkit."""
    rtx = dict(cc="12.0", gpu="NVIDIA RTX PRO 6000")
    assert _status(_env(cublas="13.0.2", **rtx), "5452663") is None
    assert _status(_env(cublas="13.1.0", **rtx), "5452663") == "may apply"      # CUDA 13.0 Update 2
    assert _status(_env(cublas="12.9.2", **rtx), "5452663") is None
    assert _status(_env(cublas="13.2.1", cc="9.0"), "5452663") is None          # not an Ada or Blackwell GeForce
    assert _status(_env(cublas="13.3.0", **rtx), "5452663") == "may apply"


def test_an_h800_pcie_isnt_taken_for_an_mps_client():
    """FULL_SMS matched by substring, so 'NVIDIA H800 PCIe' got the H800 SXM's 132 SMs. The PCIe card has 114 (the
    H100 PCIe configuration), so every H800 PCIe read as 'partial (114 of 132 SMs visible)' and 6862950 'applied':
    `sdcguard issues` exited 1 and every check card got a 'bad' note. The longest matching name wins now."""
    full, mode = ki.mps_state("NVIDIA H800 PCIe", 114, None)
    assert full == 114 and mode == "full", (full, mode)
    assert ki.mps_state("NVIDIA H800", 132, None) == (132, "full")
    assert ki.mps_state("NVIDIA H200", 132, None) == (132, "full")          # not the H20's 78
    assert ki.mps_state("NVIDIA H20", 78, None) == (78, "full")
