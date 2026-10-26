"""scripts/modal_nvbug.py without Modal or a GPU: the shapes meet the condition of NVIDIA issue 6580689, and its
expectations N1 to N3 (written in its docstring before the run) on made-up sweep results."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def nvbug(monkeypatch):
    class Builder:
        def __getattr__(self, _):
            return lambda *a, **k: self

    class App:
        def __init__(self, *a, **k):
            pass

        def function(self, **kw):
            def keep(f):
                f._modal = kw
                return f
            return keep

        def local_entrypoint(self):
            return lambda f: f
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Image=SimpleNamespace(debian_slim=lambda **_: Builder()),
                                                              App=App))
    path = Path(__file__).resolve().parents[1] / "scripts/modal_nvbug.py"
    spec = importlib.util.spec_from_file_location("modal_nvbug_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_forced_shape_meets_the_condition_for_its_own_count_only(nvbug):
    from sdcguard.blas import issue_6580689
    for s in nvbug.SPLITS:
        k = nvbug.k_for(s)
        assert k % 8 == 0 and k // s == 768 and k % s != 0
        for stages in (35, 36, 37, 38, 15, None):                  # 64, 128, 256 and 768 sizes, and unknown
            assert issue_6580689(k, s, stages) is True
    controls = [(nvbug.k_for(s), t) for s in nvbug.SPLITS for t in nvbug.SPLITS if t != s]
    assert sum(1 for k, t in controls if issue_6580689(k, t, 35)) < len(controls) / 2   # mostly controls
    assert len(nvbug.FORCED_SHAPES) == 8 and all(int(x.split("x")[1]) in map(nvbug.k_for, nvbug.SPLITS)
                                                 for x in nvbug.FORCED_SHAPES)
    assert nvbug.h100._modal["gpu"] == "H100" and nvbug.h100._modal["timeout"] == nvbug.LIMIT
    assert 3 * (nvbug.ENV_S + nvbug.VERSIONS_S) + 120 < nvbug.LIMIT   # three environments fit the function's limit


def _run(splitk, cond, correct=True, applied=True, shape=(4096, 6920, 256), tile=7, scheme=1):
    return {"variant": f"splitk_{splitk}_r{scheme}", "requested_splitk": splitk, "splitk": splitk, "forced": True,
            "split_applied": applied, "issue_6580689_condition": cond, "correct": correct, "status": 0,
            "tile_id": tile, "stages_id": 35, "cluster_shape_id": 2, "reduction_scheme": scheme}


def _sweep(version, runs, gpu="NVIDIA H100 80GB HBM3", heuristic_wrong=0, cc=(9, 0)):
    plain = [{"variant": "", "correct": True, "status": 0}]
    return {"cublaslt_version": version, "gpu": gpu, "compute_capability": list(cc), "shape": [4096, 6920, 256],
            "dtype": "bfloat16", "algorithms": plain + runs, "algo66_offered": 4, "splitk_refused": [],
            "splitk_not_run": [], "wrong_algorithms": [{"algo_id": 66}] * heuristic_wrong, "workaround_wrong": [],
            "heuristic_in_condition": []}


def _in(n, wrong=0, version=130700):
    runs = [_run(9, True, correct=i >= wrong, tile=i) for i in range(n)] + [_run(11, False, tile=99)]
    return [_sweep(version, runs)]


def _res(aff=None, fix=None, default=None):
    return {"affected": {"sweeps": aff if aff is not None else _in(10, wrong=3)},
            "fixed": {"sweeps": fix if fix is not None else _in(10, version=130801)},
            "default": {"sweeps": default if default is not None else _in(10, wrong=2, version=130101)},
            "nvidia_smi": "NVIDIA H100 80GB HBM3, 580.95.05, 81559 MiB"}


def test_a_reproduced_and_fixed_bug_meets_every_expectation(nvbug):
    met, fails, s = nvbug.judge(_res())
    assert met, fails
    assert s["affected"]["forced_wrong_in_condition"] == 3 and s["fixed"]["forced_in_condition"] == 10
    assert s["pairing"] == {"right_with_13_8_1": 3, "wrong_with_13_8_1": 0, "not_run_with_13_8_1": 0}
    assert s["default_affected"] is True
    met, _, s = nvbug.judge(_res(default=_in(10, version=130101)))
    assert met and s["default_affected"] is False                 # reported, never a criterion
    assert nvbug.judge(_res(default=[]))[2]["default_affected"] is None


def test_each_expectation_fails_on_its_own(nvbug):
    not_forced = [_sweep(130700, [_run(9, True, correct=False, applied=False)] * 3 + [_run(9, True)] * 9)]
    cases = [
        (_res(aff=_in(10)), "N1"),                                               # not reproduced
        (_res(aff=not_forced), "N1"),                                  # wrong only where no split is evident
        (_res(fix=_in(10, wrong=1, version=130801)), "N2"),                      # not fixed
        (_res(fix=[_sweep(130801, [_run(9, True)] * 10 + [_run(11, False, correct=False)])]), "N2"),  # a control
        (_res(fix=[_sweep(130801, [_run(9, True)] * 10, heuristic_wrong=1)]), "N2"),
        (_res(aff=_in(10, wrong=3, version=130801)), "N3"),                      # the wrong cuBLAS loaded
        (_res(fix=_in(7, version=130801)), "N3"),                                # too few runs in the condition
        (_res(fix=[_sweep(130801, [_run(9, True)] * 10, gpu="NVIDIA L4", cc=(8, 9))]), "N3"),
        (_res(fix=[]), "N3"),
    ]
    for res, tag in cases:
        met, fails, _ = nvbug.judge(res)
        assert not met and any(f.startswith(tag) for f in fails), (tag, fails)
    met, fails, _ = nvbug.judge(_res(default=[_sweep(130000, [], gpu="NVIDIA L4")]))
    assert met, fails                                             # the default environment is reported, not judged
