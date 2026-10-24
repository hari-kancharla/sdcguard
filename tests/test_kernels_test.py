"""sdcguard kernels test without a GPU: which calls it can run again, how it maps the worker's runs and the
kernels cuBLASLt logged back to the inventory, and the worker itself on the CPU."""
import json
import os
import subprocess
import sys

from sdcguard import kernels

K66 = "algoId=66 tile=MATMUL_TILE_64x16 stages=MATMUL_STAGES_64xAUTO customOption=1 clusterShape=CLUSTER_SHAPE_4x1x1"
K66_1 = K66.replace("4x1x1", "1x1x1")


def _line(m, n, k, algo=K66, t="R_16BF", epi=None, transa="OP_T"):
    comp = f"computeType=COMPUTE_32F scaleType=R_32F transa={transa}" + (f" epilogue={epi}" if epi else "")
    return (f"[cublasLt][9][Trace][cublasLtMatmul] A=0X1 Adesc=[type={t} rows={k} cols={m} ld={k}] B=0X2 "
            f"Bdesc=[type={t} rows={k} cols={n} ld={k}] C=0X3 Cdesc=[type={t} rows={m} cols={n} ld={m}] D=0X3 "
            f"Ddesc=[type={t} rows={m} cols={n} ld={m}] computeDesc=[{comp}] algo=[{algo}] workSpaceSizeInBytes=0")


def _inv():
    return kernels.inventory([_line(1536, 16, 8960)] * 3 + [_line(256, 16, 1536, epi="EPILOGUE_BIAS")] +
                             [_line(64, 64, 64, t="R_32F")] + [_line(128, 16, 64, transa="OP_N")])


def _runner(wrong_shape=None, logged=None):
    def run(todo, repeats, log_dir, cpu_reference):
        res = []
        for t in todo:
            bad = 1 if (t["out"], t["in"], t["rows"]) == wrong_shape else 0
            res.append({**t, "runs": [{"wrong": bad, "unwritten": 0, "worst_error_over_bound": 3.0 if bad else 0.2}
                                      for _ in range(repeats)]})
        lines = logged if logged is not None else [_line(1536, 16, 8960), _line(256, 16, 1536, epi="EPILOGUE_BIAS")]
        return {"gpu": "fake", "results": res}, lines
    return run


def test_only_half_precision_linear_layers_are_run_again():
    inv = _inv()
    why = {s["key"]: kernels.rerunnable(k, s) for k in inv["kernels"] for s in k["shapes"]}
    assert sum(w is None for w in why.values()) == 2
    assert any(w and "data types" in w for w in why.values())
    assert any(w and "transposes" in w for w in why.values())


def test_right_runs_on_the_same_kernel_pass_and_each_call_keeps_its_known_issues(tmp_path):
    r = kernels.test_calls(_inv(), repeats=3, log_dir=str(tmp_path), runner=_runner())
    assert r["counts"] == {"PASSED": 2, "WRONG": 0, "OTHER KERNEL": 0, "NOT RUN": 2}
    big = next(c for c in r["calls"] if "m=1536" in c["call"])
    assert big["issues"][0]["issue"] == "6862950" and big["calls_in_workload"] == 3 and big["runs"] == 3


def test_a_wrong_run_is_wrong_whatever_kernel_ran(tmp_path):
    r = kernels.test_calls(_inv(), log_dir=str(tmp_path), runner=_runner(wrong_shape=(1536, 8960, 16)))
    assert r["counts"]["WRONG"] == 1 and r["counts"]["PASSED"] == 1


def test_right_runs_on_another_kernel_do_not_pass(tmp_path):
    other = [_line(1536, 16, 8960, algo=K66_1), _line(256, 16, 1536, epi="EPILOGUE_BIAS")]
    r = kernels.test_calls(_inv(), log_dir=str(tmp_path), runner=_runner(logged=other))
    assert r["counts"]["OTHER KERNEL"] == 1 and r["counts"]["PASSED"] == 1
    c = next(c for c in r["calls"] if c["verdict"] == "OTHER KERNEL")
    assert c["kernels_here"][0]["clusterShape"] == "1x1x1" and c["kernel_in_workload"]["clusterShape"] == "4x1x1"
    # no log at all (no cuBLASLt here): never a pass
    r = kernels.test_calls(_inv(), log_dir=str(tmp_path), runner=_runner(logged=[]))
    assert r["counts"]["PASSED"] == 0 and r["counts"]["OTHER KERNEL"] == 2 and "no matmul" in r["warning"]


def test_one_shape_runs_once_for_every_call_that_has_it(tmp_path):
    inv = kernels.inventory([_line(1536, 16, 8960), _line(1536, 16, 8960, algo=K66_1)])
    seen = []

    def run(todo, *a):
        seen.extend(todo)
        return _runner(logged=[_line(1536, 16, 8960)])(todo, *a)
    r = kernels.test_calls(inv, log_dir=str(tmp_path), runner=run)
    assert len(seen) == 1 and r["counts"]["PASSED"] == 1 and r["counts"]["OTHER KERNEL"] == 1


def test_the_worker_on_the_cpu_judges_every_output_against_float64(tmp_path):
    todo = [{"out": 24, "in": 40, "rows": 5, "bias": True, "dtype": "bfloat16"},
            {"out": 16, "in": 32, "rows": 3, "bias": False, "dtype": "float16"}]
    shapes, out = tmp_path / "s.json", tmp_path / "r.json"
    shapes.write_text(json.dumps(todo))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    p = subprocess.run([sys.executable, "-m", "sdcguard.kernels", "--test-worker", str(shapes), str(out), "2",
                        "cpu"], env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-800:]
    res = json.loads(out.read_text())
    assert res["gpu"] == "cpu" and len(res["results"]) == 2
    for r in res["results"]:
        assert len(r["runs"]) == 2 and all(x["wrong"] == 0 for x in r["runs"])
