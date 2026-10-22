"""The upgrade gate: cases, runs and the verdicts' rules (a clean rerun is never called a fix)."""
import json
import os

import pytest
import torch

from sdcguard import gate

LOG = """[2026-10-20 06:10:01][cublasLt][118][Api][cublasLtMatmulAlgoGetHeuristic] Adesc=[type=R_16BF rows=2048]
[2026-10-20 06:10:01][cublasLt][118][Trace][cublasLtMatmul] A=0X2B3D2C800000 Adesc=[type=R_16BF rows=2048 cols=1000 ld=2048] computeDesc=[computeType=COMPUTE_32F scaleType=R_32F transa=OP_T] algo=[algoId=66 tile=MATMUL_TILE_64x8 stages=MATMUL_STAGES_64x6 clusterShape=CLUSTER_SHAPE_4x1x1 customOption=0] workSpaceSizeInBytes=0
[2026-10-20 06:10:01][cublasLt][118][Trace][cublasLtMatmul] A=0X2B3D2C800000 Adesc=[type=R_16BF rows=2048 cols=1000 ld=2048] algo=[algoId=66 tile=MATMUL_TILE_64x8 stages=MATMUL_STAGES_64x6 clusterShape=CLUSTER_SHAPE_4x1x1]
[2026-10-20 06:10:02][cublasLt][118][Trace][cublasLtMatmul] A=0X2B3D2C800000 algo=[algoId=66 tile=MATMUL_TILE_64x8 stages=MATMUL_STAGES_64x6 clusterShape=CLUSTER_SHAPE_1x1x1]
"""


def test_the_kernel_log_names_each_call_by_algorithm_tile_stages_and_cluster():
    r = gate.parse_lt_log(LOG)
    assert r["calls"] == 3
    assert r["kernels"] == {"algoId=66 tile=64x8 stages=64x6 cluster=4x1x1": 2,
                            "algoId=66 tile=64x8 stages=64x6 cluster=1x1x1": 1}


def _cond(**kw):
    c = {"gpu_name": "NVIDIA H200", "compute_capability": [9, 0], "sms_visible": 52, "device_type": "cuda",
         "env": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"},
         "matmul": {"allow_tf32": False, "allow_bf16_reduced_precision_reduction": True,
                    "allow_fp16_reduced_precision_reduction": True, "blas_library": "cublas", "tunableop": False},
         "alignment": {"x": 0, "w": 0, "bias": 0}}
    c.update(kw)
    return c


def _run(wrong_runs, n=20, kernels=None, cond=None, stack=None):
    runs = [{"wrong": 7747 if i < wrong_runs else 0, "unwritten": 0, "nonfinite": 0} for i in range(n)]
    return {"case": "c", "runs": runs, "n": n, "conditions": cond or _cond(),
            "stack": stack or {"torch": "2.14.1+cu130", "cublas": "13.1.1.3"},
            "kernel_log": {"log": True, "calls": n, "kernels": kernels if kernels is not None else
                           {"algoId=66 tile=64x8 stages=64x6 cluster=4x1x1": n}}}


def test_right_before_and_after_passes():
    assert gate.verdict(_run(0), _run(0))["verdict"] == "PASSED"


def test_right_before_and_wrong_after_is_a_regression():
    v = gate.verdict(_run(0), _run(3, stack={"torch": "2.15.0", "cublas": "13.8.1.7"}))
    assert v["verdict"] == "REGRESSED" and v["stack_changes"]["cublas"] == ["13.1.1.3", "13.8.1.7"]


def test_wrong_before_and_after_is_unchanged():
    v = gate.verdict(_run(20), _run(20))
    assert v["verdict"] == "UNCHANGED" and v["flavor"] == "still wrong"


def test_right_after_with_another_kernel_is_avoided_not_fixed():
    """The heuristic picking another kernel makes the outputs right without fixing the one that was wrong."""
    v = gate.verdict(_run(20), _run(0, kernels={"algoId=31 tile=128x64 stages=64x4 cluster=1x2x1": 20}))
    assert v["verdict"] == "UNCHANGED" and v["flavor"] == "avoided"
    assert "not shown fixed" in v["why"]


def test_right_after_with_the_same_kernel_is_fixed():
    v = gate.verdict(_run(20), _run(0))
    assert v["verdict"] == "FIXED" and v["p"] < 1e-6
    assert v["candidate"]["rate_95"][1] < 0.2


def test_a_rare_failure_and_a_clean_rerun_is_not_enough_to_call_fixed():
    v = gate.verdict(_run(1), _run(0))
    assert v["verdict"] == "INCONCLUSIVE" and "run more" in v["why"]


def test_a_clean_rerun_under_other_conditions_is_inconclusive_not_fixed():
    """Rerunning at MPS 100% makes every output right, and says nothing about the bug at 40%."""
    full = _cond(sms_visible=132, env={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"})
    v = gate.verdict(_run(20), _run(0, cond=full))
    assert v["verdict"] == "INCONCLUSIVE"
    assert "sms_visible 52 vs 132" in v["why"] and "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE '40' vs '100'" in v["why"]


@pytest.mark.parametrize("key", ["sms_visible", "gpu_name", "env", "matmul"])
def test_an_unrecorded_condition_is_never_a_match(key):
    c = _cond()
    c[key] = None
    assert gate.verdict(_run(20), _run(0, cond=c))["verdict"] == "INCONCLUSIVE"


def test_without_kernel_logs_a_clean_candidate_is_inconclusive():
    v = gate.verdict(_run(20, kernels={}), _run(0, kernels={}))
    assert v["verdict"] == "INCONCLUSIVE" and "log" in v["why"]


def test_too_few_runs_or_a_failed_run_is_inconclusive():
    assert gate.verdict(_run(2, n=3), _run(0, n=3))["verdict"] == "INCONCLUSIVE"
    assert gate.verdict(_run(20), {"error": "boom"})["verdict"] == "INCONCLUSIVE"
    assert gate.verdict(None, _run(0))["verdict"] == "INCONCLUSIVE"


def test_fisher_exact_values():
    assert gate.fisher_one_sided(1, 20, 0, 20) == pytest.approx(0.5)
    assert gate.fisher_one_sided(20, 20, 0, 20) == pytest.approx(1 / 137846528820, rel=1e-6)
    assert gate.fisher_one_sided(5, 20, 0, 20) == pytest.approx(0.02356, abs=1e-4)


def test_a_tunableop_file_is_compared_by_its_contents():
    a = _cond(env={"PYTORCH_TUNABLEOP_FILENAME": "/tmp/a/pin.csv", "PYTORCH_TUNABLEOP_ENABLED": "1"},
              tunableop_file_sha256="abc")
    b = _cond(env={"PYTORCH_TUNABLEOP_FILENAME": "/tmp/b/pin.csv", "PYTORCH_TUNABLEOP_ENABLED": "1"},
              tunableop_file_sha256="abc")
    assert gate.condition_differences(a, b) == []
    b["tunableop_file_sha256"] = "def"
    assert gate.condition_differences(a, b) == ["the TunableOp file's contents differ (or couldn't be read)"]


def test_steering_env_keeps_kernel_switches_and_drops_logging():
    env = {"CUBLASLT_LOG_LEVEL": "5", "CUBLASLT_LOG_FILE": "x", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
           "PYTORCH_TUNABLEOP_ENABLED": "1", "DISABLE_ADDMM_CUDA_LT": "1", "HOME": "/root",
           "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40", "PYTORCH_TUNABLEOP_VERBOSE": "3"}
    assert gate.steering_env(env) == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40",
                                      "DISABLE_ADDMM_CUDA_LT": "1", "PYTORCH_TUNABLEOP_ENABLED": "1"}


# ---------------------------------------------------------------- workaround
def _fake_runs(monkeypatch, control, treatment, file_sha="abc"):
    """file_sha: what the treatment's worker read from the TunableOp file PyTorch looks for (None: not there)."""
    def fake(case_dir, repeats=20, env=None, device=None, timeout=600.0):
        r = treatment if env else control
        c = {**r["conditions"], "env": {**r["conditions"]["env"], **(env or {})},
             "matmul": {**r["conditions"]["matmul"], "tunableop": bool((env or {}).get("PYTORCH_TUNABLEOP_ENABLED"))}}
        if (env or {}).get("PYTORCH_TUNABLEOP_FILENAME"):
            c["tunableop_file"] = gate.tunableop_file(env["PYTORCH_TUNABLEOP_FILENAME"], 0)
            c["tunableop_file_sha256"] = file_sha
        return {**r, "conditions": c}
    monkeypatch.setattr(gate, "run_case", fake)


PIN = {"PYTORCH_TUNABLEOP_ENABLED": "1", "PYTORCH_TUNABLEOP_TUNING": "0", "PYTORCH_TUNABLEOP_FILENAME": "pin.csv"}
ONE = {"algoId=66 tile=64x8 stages=64x6 cluster=1x1x1": 50}


def test_a_workaround_works_only_with_a_failing_control_a_clean_treatment_and_the_pinned_kernel(monkeypatch):
    _fake_runs(monkeypatch, _run(50, n=50), _run(0, n=50, kernels=ONE))
    v = gate.workaround("gates/c", PIN, 50, expect_kernel="algoId=66 cluster=1x1x1")
    assert v["verdict"] == "WORKS", v
    assert "on every call" in v["why"]


def test_a_workaround_whose_kernel_did_not_run_is_inconclusive(monkeypatch):
    """A pin PyTorch rejected (any version change) leaves the default kernel running; right outputs then say
    nothing about the workaround."""
    _fake_runs(monkeypatch, _run(50, n=50), _run(0, n=50, kernels={"algoId=31 cluster=1x2x1": 50}))
    v = gate.workaround("gates/c", PIN, 50, expect_kernel="algoId=66 cluster=1x1x1")
    assert v["verdict"] == "INCONCLUSIVE" and "may not be what ran" in v["why"]


def test_a_workaround_with_a_clean_control_is_inconclusive(monkeypatch):
    _fake_runs(monkeypatch, _run(0, n=50), _run(0, n=50, kernels=ONE))
    assert gate.workaround("gates/c", PIN, 50)["verdict"] == "INCONCLUSIVE"


def test_a_workaround_that_still_fails_doesnt_work(monkeypatch):
    _fake_runs(monkeypatch, _run(50, n=50), _run(4, n=50, kernels=ONE))
    assert gate.workaround("gates/c", PIN, 50)["verdict"] == "DOESN'T WORK"
    v = gate.workaround("gates/c", PIN, 50, expect_kernel="algoId=66 cluster=1x1x1")
    assert v["verdict"] == "DOESN'T WORK" and "on every call" in v["why"], v


def test_a_workaround_still_wrong_because_it_never_ran_is_inconclusive_not_doesnt_work(monkeypatch):
    """The H200 run of October 22: the pin was rejected, the default kernel ran and was wrong as before. That says
    nothing about the workaround, so it isn't DOESN'T WORK."""
    default = {"algoId=66 tile=64x8 stages=64xAUTO cluster=4x2x1": 50}
    _fake_runs(monkeypatch, _run(50, n=50, kernels=default), _run(50, n=50, kernels=default))
    v = gate.workaround("gates/c", PIN, 50, expect_kernel="algoId=66 cluster=1x1x1")
    assert v["verdict"] == "INCONCLUSIVE" and "wrong in 50 of 50 runs" in v["why"] and "may not be what ran" in v["why"]


def test_a_workaround_whose_tunableop_file_isnt_where_pytorch_looks_is_inconclusive(monkeypatch):
    for treatment in (_run(50, n=50), _run(0, n=50, kernels=ONE)):
        _fake_runs(monkeypatch, _run(50, n=50), treatment, file_sha=None)
        v = gate.workaround("gates/c", PIN, 50)
        assert v["verdict"] == "INCONCLUSIVE" and "pin0.csv" in v["why"] and "wasn't applied" in v["why"], v


def test_tunableop_puts_the_gpu_number_in_the_file_name_like_pytorch():
    """TuningContext::SetFilename in PyTorch 2.14: the first %d, else before the last '.', else at the end."""
    assert gate.tunableop_file("pin.csv", 0) == "pin0.csv"
    assert gate.tunableop_file("/tmp/gate/pin.csv", 3) == "/tmp/gate/pin3.csv"
    assert gate.tunableop_file("/tmp/gate/pin%d.csv", 1) == "/tmp/gate/pin1.csv"
    assert gate.tunableop_file("/tmp/gate/pin", "2") == "/tmp/gate/pin2"
    assert gate.tunableop_file("/tmp/v1.2/pin", 0) == "/tmp/v10.2/pin"        # the last '.' of the whole path


def test_a_pin_is_written_where_pytorch_reads_it(tmp_path, monkeypatch):
    import subprocess
    name = "Gemm_Cublaslt_id_66_tile_10_stages_35_splitk_1_red_0_swizzle_0_custom_3_inner_0_cluster_2"
    tuned = ("Validator,PT_VERSION,2.14.1\nValidator,CUDA_DEVICE,9.0:NVIDIA H200\n"
             "GemmAndBiasTunableOp_BFloat16_TN,tn_1000_16_2048_ld_2048_2048_1000,Default,0.0102\n")

    def fake(cmd, env=None, **kw):
        if cmd[3] == "--pin-worker":
            with open(cmd[5], "w") as f:
                json.dump({"name": name, "config": {"id": 66, "cluster": 2}, "bias": True}, f)
        else:                                       # the tuning run: TunableOp writes tuned<GPU>.csv
            with open(gate.tunableop_file(env["PYTORCH_TUNABLEOP_FILENAME"], 1), "w") as f:
                f.write(tuned)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(gate.subprocess, "run", fake)
    res = gate.pin("gates/c", str(tmp_path / "pin.csv"))
    assert res["file"] == str(tmp_path / "pin1.csv") and res["gpu"] == 1, res
    assert res["use"].endswith(f"PYTORCH_TUNABLEOP_FILENAME={tmp_path / 'pin.csv'}")
    assert not (tmp_path / "pin.csv").exists()
    assert f"tn_1000_16_2048_ld_2048_2048_1000,{name},0.0102" in (tmp_path / "pin1.csv").read_text()
    assert gate.pin("gates/c", str(tmp_path / "p%d.csv"))["file"] == str(tmp_path / "p1.csv")


# ---------------------------------------------------------------- real runs, on the CPU
def test_judge_counts_wrong_and_unwritten_outputs():
    ref = torch.ones(2, 3, dtype=torch.float64)
    bound = torch.full_like(ref, 1e-3)
    y = torch.tensor([[1.0, float("nan"), 1.0], [1.5, 1.0, 1.0]], dtype=torch.bfloat16)
    r = gate._judge(y, ref, bound, prefilled=True)
    assert r["wrong"] == 2 and r["unwritten"] == 1 and r["nonfinite"] == 1
    assert gate._judge(y, ref, bound, prefilled=False)["unwritten"] is None


def test_capture_run_and_compare_on_the_cpu(tmp_path):
    pack = str(tmp_path / "gates")
    res = gate.capture((40, 64, 3), "bfloat16", True, pack, seed=1, repeats=5, device="cpu")
    assert res["kind"] == "pin" and res["runs_wrong"] == 0
    case = json.load(open(os.path.join(res["case"], "case.json")))
    assert case["schema"] == gate.CASE_SCHEMA and case["op"] == {"call": "linear", "dtype": "bfloat16", "m": 3,
                                                                  "n": 40, "k": 64, "bias": True}
    assert case["conditions"]["gpu_name"] == "cpu" and case["inputs"]["sha256"]
    a = gate.run_pack(pack, 5, device="cpu")
    b = gate.run_pack(pack, 5, device="cpu")
    (cid, r), = a["cases"].items()
    assert r["n"] == 5 and all(x["wrong"] == 0 for x in r["runs"]) and r["kernel_log"]["calls"] == 0
    assert gate.compare(a, b)[cid]["verdict"] == "PASSED"


def test_run_here_catches_a_wrong_linear(tmp_path, monkeypatch):
    pack = str(tmp_path / "gates")
    res = gate.capture((40, 64, 3), "bfloat16", False, pack, repeats=5, device="cpu")
    real = torch.nn.functional.linear

    def wrong(x, w, b=None):
        y = real(x, w, b)
        y[1, 7] += 5.0
        return y
    monkeypatch.setattr(torch.nn.functional, "linear", wrong)
    r = gate.run_here(res["case"], 4, "cpu")
    assert [x["wrong"] for x in r["runs"]] == [1, 1, 1, 1]


def test_the_command_line_runs_end_to_end(tmp_path, capsys):
    from sdcguard import cli
    pack = str(tmp_path / "gates")
    assert cli.main(["gate", "capture", "--shape", "40,64,3", "--bias", "--pack", pack, "--repeats", "5"]) == 0
    out1, out2 = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    assert cli.main(["gate", "run", pack, "--out", out1, "--repeats", "5", "--device", "cpu"]) == 0
    assert cli.main(["gate", "run", pack, "--out", out2, "--repeats", "5", "--device", "cpu"]) == 0
    assert cli.main(["gate", "compare", out1, out2]) == 0
    assert "PASSED" in capsys.readouterr().out


def test_a_pin_replaces_only_the_solution_of_the_cases_gemm_row():
    text = ("Validator,PT_VERSION,2.14.1\nValidator,CUBLASLT_VERSION,130101\nValidator,CUDA_DEVICE,9.0:NVIDIA H200\n"
            "GemmAndBiasTunableOp_BFloat16_TN,tn_1000_16_2048_ld_2048_2048_1000,Default,0.0123\n")
    name = "Gemm_Cublaslt_id_66_tile_20_stages_8_splitk_1_red_0_swizzle_0_custom_0_inner_1_cluster_2"
    out, rows = gate.rewrite_pin(text, True, name)
    assert rows == 1
    assert out.splitlines()[3] == f"GemmAndBiasTunableOp_BFloat16_TN,tn_1000_16_2048_ld_2048_2048_1000,{name},0.0123"
    assert out.splitlines()[:3] == text.splitlines()[:3]
    assert gate.rewrite_pin(text, False, name)[1] == 0          # no bias: the GemmTunableOp row, which isn't there
