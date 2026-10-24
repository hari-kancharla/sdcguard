"""sdcguard gate investigate with a stand-in for the case runner: which factor a failure follows."""
import json
import os

from sdcguard import investigate as iv

K66 = {"algoId=66 tile=64x16 stages=64xAUTO cluster=4x1x1": 20}
K66_1 = {"algoId=66 tile=64x16 stages=64xAUTO cluster=1x1x1": 20}
COND = {"gpu_name": "NVIDIA H200", "compute_capability": "9.0", "sms_visible": 53, "device_type": "cuda",
        "env": {"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"}, "matmul": {"allow_bf16_reduced_precision_reduction": True}}


def _case(tmp_path, cond=COND):
    d = tmp_path / f"case{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    (d / "case.json").write_text(json.dumps({"conditions": cond}))
    return str(d)


def _res(wrong, n=20, kernels=K66, cond=None, error=None):
    if error:
        return {"error": error}
    runs = [{"wrong": 1 if i < wrong else 0, "unwritten": 0} for i in range(n)]
    return {"n": n, "runs": runs, "conditions": cond or COND, "kernel_log": {"kernels": kernels}}


def _runner(table):
    """table(env, device) -> result."""
    calls = []

    def run(case_dir, repeats, env, device):
        calls.append((dict(env or {}), device))
        return table(env or {}, device)
    run.calls = calls
    return run


def test_a_failure_on_one_gpu_only_follows_the_gpu(tmp_path):
    r = iv.investigate(_case(tmp_path), devices=["cuda:0", "cuda:1"],
                       runner=_runner(lambda env, d: _res(20 if d == "cuda:1" else 0)))
    assert r["verdict"] == "DEVICE" and r["wrong_on"] == ["cuda:1"] and "cuda:1" in r["why"]


def test_a_failure_on_every_gpu_that_clears_at_100_percent_follows_the_condition(tmp_path):
    def t(env, d):
        at40 = env.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == "40"
        return _res(20 if at40 else 0, cond=COND if at40 else dict(COND, env={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE":
                                                                                "100"}, sms_visible=132))
    run = _runner(t)
    r = iv.investigate(_case(tmp_path), devices=["cuda:0", "cuda:1"], runner=run)
    assert r["verdict"] == "CONDITION" and "not a GPU" in r["why"]
    assert {c[0].get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") for c in run.calls} == {"40", "100"}
    one = iv.investigate(_case(tmp_path), devices=["cuda:0"], runner=_runner(t))
    assert one["verdict"] == "CONDITION" and "untested" in one["why"]


def test_wrong_everywhere_and_right_on_the_other_path_follows_the_software(tmp_path):
    cond = dict(COND, env={})
    def t(env, d):
        return _res(0 if env.get("TORCH_BLAS_PREFER_CUBLASLT") == "0" else 20, cond=cond)
    r = iv.investigate(_case(tmp_path, cond), devices=["cuda:0", "cuda:1"], runner=_runner(t))
    assert r["verdict"] == "SOFTWARE PATH"
    r = iv.investigate(_case(tmp_path, cond), devices=["cuda:0"], runner=_runner(t))
    assert r["verdict"] == "INCONCLUSIVE" and "one GPU" in r["why"]
    r = iv.investigate(_case(tmp_path, cond), devices=["cuda:0", "cuda:1"], runner=_runner(lambda e, d: _res(20,
                                                                                                         cond=cond)))
    assert r["verdict"] == "NOT ONE GPU"


def test_right_everywhere_is_not_reproduced_and_mismatched_conditions_are_inconclusive(tmp_path):
    c = _case(tmp_path)
    assert iv.investigate(c, devices=["cuda:0"], runner=_runner(lambda e, d: _res(0)))["verdict"] == "NOT REPRODUCED"
    other = dict(COND, sms_visible=132)     # the MPS share didn't apply (no daemon): not the case's conditions
    r = iv.investigate(c, devices=["cuda:0"], runner=_runner(lambda e, d: _res(0, cond=other)))
    assert r["verdict"] == "INCONCLUSIVE" and "conditions" in r["why"]
    r = iv.investigate(c, devices=["cuda:0", "cuda:1"],
                       runner=_runner(lambda e, d: _res(20 if d == "cuda:1" else 0,
                                                        kernels=K66 if d == "cuda:1" else K66_1)))
    assert r["verdict"] == "INCONCLUSIVE" and "different kernels" in r["why"]
    r = iv.investigate(c, devices=["cuda:0"], runner=_runner(lambda e, d: _res(0, error="boom")))
    assert r["verdict"] == "INCONCLUSIVE"
    assert iv.investigate(c, devices=[], runner=_runner(lambda e, d: _res(0)))["verdict"] == "INCONCLUSIVE"


def test_missing_kernel_logs_never_make_it_the_gpu(tmp_path):
    """Wrong on one GPU and right on another with no kernel logged on either: an unknown kernel isn't the same
    kernel, so the board isn't blamed."""
    r = iv.investigate(_case(tmp_path), devices=["cuda:0", "cuda:1"],
                       runner=_runner(lambda env, d: _res(20 if d == "cuda:1" else 0, kernels={})))
    assert r["verdict"] == "INCONCLUSIVE" and "kernel log" in r["why"]
    r = iv.investigate(_case(tmp_path), devices=["cuda:0", "cuda:1"],
                       runner=_runner(lambda env, d: _res(20 if d == "cuda:1" else 0, kernels=K66 if d == "cuda:1"
                                                          else {})))
    assert r["verdict"] == "INCONCLUSIVE"
