"""experiments/vllm_live_catch.py without a GPU: which inventory calls it flags for issue 6862950, how it reads the
gate's results, and its criteria (docs/METHOD.md) on results made here."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import vllm_live_catch as lc  # noqa: E402

from sdcguard import gate, kernels  # noqa: E402

ALGO66 = "algoId=66 tile=MATMUL_TILE_64x16 stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_4x1x1"


def _line(m=1536, n=16, k=8960, algo=ALGO66, t="R_16BF", epi=None, sms="52"):
    comp = f"computeType=COMPUTE_32F scaleType=R_32F transa=OP_T smCountTarget={sms}" + (f" epilogue={epi}" if epi else "")
    return (f"[cublasLt][9][Trace][cublasLtMatmul] A=0X1 Adesc=[type={t} rows={k} cols={m} ld={k}] B=0X2 "
            f"Bdesc=[type={t} rows={k} cols={n} ld={k}] C=0X3 Cdesc=[type={t} rows={m} cols={n} ld={m}] D=0X3 "
            f"Ddesc=[type={t} rows={m} cols={n} ld={m}] computeDesc=[{comp}] algo=[{algo}] workSpaceSizeInBytes=0")


def test_flagged_lists_the_shapes_on_a_kernel_matching_the_issue():
    inv = kernels.inventory([_line(), _line(n=32, epi="EPILOGUE_BIAS"),
                             _line(algo="algoId=66 tile=MATMUL_TILE_64x16 clusterShape=CLUSTER_SHAPE_1x1x1")])
    fl = lc.flagged(inv)
    assert sorted((f["out"], f["in"], f["rows"], f["bias"]) for f in fl) == [(1536, 8960, 16, False),
                                                                            (1536, 8960, 32, True)]


def _result(wrong, n=20, kernel="algoId=66 tile=64x16 stages=64xAUTO cluster=4x1x1"):
    return {"runs": [{"wrong": 5 if i < wrong else 0, "unwritten": 0} for i in range(n)],
            "kernel_log": {"kernels": {kernel: n}}}


def _res(**over):
    fl = lc.flagged(kernels.inventory([_line()]))
    alert = {"detail": {"captured": {"row": 0, "step": 7}, "recheck": {"verdict": "output_wrong"},
                        "matmul": {"out_features": 1536, "in_features": 8960, "bias": False},
                        "step_rows": {"tokens": 13, "with_padding": 16}}}
    res = {"ref100": {"ready": True, "alerts": [], "identity": ["a", "b"], "inventory": {"flagged_6862950": []}},
           "mps40": {"ready": True, "alerts": [alert], "identity": ["a", "x"], "inventory": {"flagged_6862950": fl}},
           "replays": [{"shape": {"out": 1536, "in": 8960, "rows": 16, "bias": False},
                        "result40": _result(20), "result100": _result(0, kernel="algoId=66 cluster=2x1x1"),
                        "slice_result40": _result(0, kernel="algoId=21 cluster=?")}]}
    res.update(over)
    return res


def test_a_live_catch_with_a_faithful_replay_passes():
    ok, fails, s = lc.judge(_res())
    assert ok and not fails, fails
    assert s["replays"][0]["same_kernel_as_server"] and s["replays"][0]["slice_wrong_40"] == "0/20"
    assert s["completions_differing_40_vs_100"] == 1


def test_each_criterion_can_fail():
    r = _res()
    r["ref100"]["alerts"] = [{"detail": {}}]
    assert not lc.judge(r)[0]                                   # an alert at 100%
    r = _res()
    r["mps40"]["alerts"] = []
    assert not lc.judge(r)[0]                                   # the kernel ran, nothing raised
    r = _res()
    r["replays"][0]["result40"] = _result(12)
    assert not lc.judge(r)[0]                                   # the replay isn't wrong often enough
    r = _res()
    r["replays"][0]["result40"] = _result(20, kernel="algoId=55 tile=64x32 cluster=?")
    assert not lc.judge(r)[0]                                   # wrong, but on another kernel than the server's
    r = _res()
    r["replays"][0]["result100"] = _result(2)
    assert not lc.judge(r)[0]                                   # wrong at 100% too: not this bug


def test_when_the_server_never_reaches_the_kernel_it_is_not_exercised_and_never_a_catch():
    r = _res()
    r["mps40"]["inventory"]["flagged_6862950"] = []
    r["mps40"]["alerts"] = []
    ok, fails, s = lc.judge(r)
    assert not ok and not fails and s["verdict"] == "NOT EXERCISED"
    assert "did not reach the affected kernel" in s["finding"]
    r["ref100"]["alerts"] = [{"detail": {}}]
    ok, fails, s = lc.judge(r)
    assert not ok and s["verdict"] == "NOT EXERCISED" and fails[0].startswith("2:")


def test_verdicts_name_what_happened():
    assert lc.judge(_res())[2]["verdict"] == "CAUGHT"
    r = _res()
    r["mps40"]["alerts"] = []
    assert lc.judge(r)[2]["verdict"] == "NOT CAUGHT"
    r = _res()
    r["mps40"]["ready"] = False
    assert lc.judge(r)[2]["verdict"] == "DIDN'T START"


def test_the_bundle_holds_the_cases_the_pin_and_the_commands(tmp_path):
    out = str(tmp_path)
    pack = os.path.join(out, "replay_live-l3-1536x8960x16")
    os.makedirs(os.path.join(pack, "live-l3-1536x8960x16"))
    open(os.path.join(pack, "live-l3-1536x8960x16", "case.json"), "w").write("{}")
    os.makedirs(os.path.join(out, "pins"))
    open(os.path.join(out, "pins", "merged0.csv"), "w").write("Validator,PT_VERSION,2.14.1\n")
    res = {"model": "m", "summary": {"verdict": "CAUGHT"},
           "replays": [{"pack": pack, "layer": "l3", "shape": {"out": 1536, "in": 8960, "rows": 16, "bias": False},
                        "case": {"conditions": {"mps": 40}}}],
           "pins": {"file": os.path.join(out, "pins", "merged.csv")}}
    d = lc.bundle(out, res)
    sh = open(os.path.join(d, "reproduce.sh")).read()
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40 sdcguard gate run replay_live-l3-1536x8960x16" in sh
    assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100" in sh and "gate workaround" in sh
    assert os.path.exists(os.path.join(d, "pins", "merged0.csv"))
    assert os.path.exists(os.path.join(d, "replay_live-l3-1536x8960x16", "live-l3-1536x8960x16", "case.json"))
    c = json.load(open(os.path.join(d, "conditions.json")))
    assert c["verdict"] == "CAUGHT" and c["cases"][0]["conditions"] == {"mps": 40}


def test_merge_pins_keeps_the_validators_once_and_every_gemm_row():
    a = "Validator,PT_VERSION,2.14.1\nValidator,CUBLASLT_VERSION,130801\nGemmTunableOp_BFloat16_TN,tn_1536_16_8960,x,1\n"
    b = "Validator,PT_VERSION,2.14.1\nValidator,CUBLASLT_VERSION,130801\nGemmTunableOp_BFloat16_TN,tn_1536_32_8960,y,1\n"
    text, rows = gate.merge_pins([a, b, a])
    assert rows == 2 and text.count("Validator,PT_VERSION") == 1 and "tn_1536_32_8960" in text
    import pytest
    with pytest.raises(ValueError):
        gate.merge_pins([a, b.replace("130801", "130500")])


def test_kernel_test_results_are_reported_and_decide_nothing():
    r = _res()
    r["mps40"]["kernel_test"] = {"counts": {"PASSED": 3, "WRONG": 2, "OTHER KERNEL": 0, "NOT RUN": 1},
                             "flagged_calls": 2, "flagged_wrong": 2, "unflagged_wrong": 0}
    ok, fails, s = lc.judge(r)
    assert ok and s["kernel_test_mps40"]["flagged_wrong"] == 2
    r["mps40"]["kernel_test"]["unflagged_wrong"] = 1
    assert lc.judge(r)[0]
