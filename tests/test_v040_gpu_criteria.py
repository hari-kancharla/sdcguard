"""The 0.4.0 GPU run's criteria (docs/METHOD.md, "0.4.0 on GPUs"), as the runners judge them: each gate has to
measure what the criterion says, not something that can pass without it. The runners themselves need GPUs; these
tests feed their judging code the results a run could produce."""
import ast
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import torch

from sdcguard import known_issues
from sdcguard.inject import _rel_change

ROOT = Path(__file__).resolve().parents[1]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launcher(*names, **ns):
    """Functions (and plain assignments) from scripts/modal_v040.py without importing modal."""
    tree = ast.parse((ROOT / "scripts" / "modal_v040.py").read_text())
    body = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))]
    exec(compile(ast.Module(body=body, type_ignores=[]), "modal_v040.py", "exec"), ns)
    return ns


@pytest.mark.parametrize("clean, met", [
    # one checked matmul per rank in 4 steps: not "every matmul checked"
    ({"checked": 1, "checked_dtensor": 1, "skipped_dtensor": 0, "alerts": 0, "unjudged": 0}, False),
    # every one of the 4 x 23 local matmuls checked, nothing skipped (every skip counter recorded, as run() does)
    ({"checked": 92, "checked_dtensor": 92, "skipped_dtensor": 0, "skipped_oom": 0, "skipped_dtype": 0,
      "skipped_dtensor_compiled": 0, "alerts": 0, "unjudged": 0}, True),
    # the right count, but a check skipped for lack of memory
    ({"checked": 92, "checked_dtensor": 92, "skipped_dtensor": 0, "skipped_oom": 1, "skipped_dtype": 0,
      "skipped_dtensor_compiled": 0, "alerts": 0, "unjudged": 0}, False),
    # skip counters not recorded at all: not shown to be 0
    ({"checked": 92, "checked_dtensor": 92, "skipped_dtensor": 0, "alerts": 0, "unjudged": 0}, False),
])
def test_tensor_parallel_gate_needs_every_local_matmul_checked(monkeypatch, tmp_path, clean, met):
    """"every matmul checked as a local shard (none skipped)": on each rank, checked and checked_dtensor equal the
    clean steps times the local matmuls per step (from the model's structure: 23), and the tensor-parallel,
    out-of-memory and dtype skip counters are 0. Asking only for checked_dtensor > 0 let a rank that checked one
    matmul in the whole run pass."""
    import torch.distributed as dist
    import torch.distributed.device_mesh as dm
    tp = _load("experiments/v040_tp.py", "v040_tp_gate")
    assert tp.checks_per_step() == 23

    def fake(rank):
        return {"clean": dict(clean), "grads_identical_to_unguarded_tp": True, "fault_steps": 10,
                "steps_with_alert": 10 if rank == 1 else 0}
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dist, "get_rank", lambda *a, **k: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(dist, "destroy_process_group", lambda *a, **k: None)
    monkeypatch.setattr(dm, "init_device_mesh", lambda *a, **k: None)

    def gather(out, obj):
        out[0] = obj
        out[1] = {"rank": 1, "gpu": "cpu", "torch": torch.__version__, **{m: fake(1) for m in tp.MODES}}
    monkeypatch.setattr(dist, "all_gather_object", gather)
    monkeypatch.setattr(tp, "run", lambda mode, mesh, rank, dev="cuda", width=4096, steps=50: fake(rank))
    assert tp.main(["--cpu", "--out", str(tmp_path)]) == (0 if met else 1)
    summary = json.loads((tmp_path / "v040_tp.json").read_text())
    assert summary["all met"] is met


def _v040_result(grouped):
    return {"fingerprints": {"agree": True, "kernel_used": True},
            "compile": {"eager_checks_per_step": 27, "checks_per_checked_step": [27] * 4,
                        "checks_on_unchecked_steps": 0, "alerts": 0, "unjudged": 0, "fault_trials": 5,
                        "faults_caught": 5, "ms_unchecked": 10.0, "ms_unguarded": 10.0},
            "grouped": grouped, "blas": {"sweeps": [{"verdict": "PASS"}], "wrong_with_listed_issues": []}}


def test_grouped_gate_needs_the_clean_calls_to_have_been_checked():
    """Criterion 4's clean half ("no alarm and nothing unjudged in 200 calls") holds only for calls that were
    checked: every expert with tokens, in the forward and both backward matmuls of every call. With no count, 0
    checks (every random-routing call skipped) met it."""
    v = _load("experiments/v040_check.py", "v040_check_gate")
    key = "4 grouped GEMM: clean, at least 99% of single wrong values caught"
    base = {"experts": 64, "calls": 200, "clean_alarms": 0, "unjudged": 0, "fault_trials": 100, "faults_caught": 100}
    assert v.criteria(_v040_result(dict(base, checks=0)))[key] is not True
    assert v.criteria(_v040_result(dict(base, checks=0, expected_checks=0)))[key] is not True
    assert v.criteria(_v040_result(dict(base, checks=37_000, expected_checks=38_016)))[key] is not True
    assert v.criteria(_v040_result(dict(base, checks=38_016, expected_checks=38_016)))[key] is True


def _grouped_mm_on_cpu() -> bool:
    """As in test_grouped_mm.py: PyTorch 2.8 has torch._grouped_mm, but no CPU kernel for it."""
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        torch._grouped_mm(torch.zeros(4, 16, dtype=torch.bfloat16), torch.zeros(2, 16, 16, dtype=torch.bfloat16),
                          offs=torch.tensor([2, 4], dtype=torch.int32))
        return True
    except (NotImplementedError, RuntimeError):
        return False


def test_grouped_runner_counts_three_checks_per_expert_with_tokens():
    """The expected count is derived per call from the routing (three checks per expert with tokens, and the weight
    gradient of one without), and on the CPU it equals what the guard checked."""
    if not _grouped_mm_on_cpu():
        pytest.skip("this PyTorch can't run torch._grouped_mm on the CPU")
    v = _load("experiments/v040_check.py", "v040_check_grouped")
    r = v.grouped(torch.device("cpu"), calls=3, say=lambda s: None)
    assert r["expected_checks"] > 0 and r["checks"] == r["expected_checks"]
    assert r["clean_alarms"] == 0 and r["faults_caught"] == r["fault_trials"]


def test_fused_criterion3_in_kernel_branch_needs_an_in_kernel_measurement():
    """METHOD 0.4.0: "criterion 3 must hold for both the old scoring and the in-kernel one". Reading the in-kernel
    result with the old scoring's number as its default scored the October 20 H100 file, made before the in-kernel
    scoring existed, as met for both."""
    fc = _load("experiments/fused_check.py", "fused_check_gate")
    d = json.loads((ROOT / "results" / "fused" / "h100_20261020_073624.json").read_text())["fused"]
    assert all("smallest_caught_half_the_time_checked" not in s for s in d["sensitivity"]["shapes"])
    key = "3 same sensitivity as precise mode (within 2x)"
    assert fc.criteria(d)[key] is not True
    for s in d["sensitivity"]["shapes"]:                        # measured, and equal: met
        s["smallest_caught_half_the_time_checked"] = s["smallest_caught_half_the_time_fused"]
    assert fc.criteria(d)[key] is True


def _sweep(shape, dt, algos, wrong=()):
    return {"shape": [int(v) for v in shape.split("x")], "dtype": dt, "algorithms": algos,
            "wrong_algorithms": list(wrong)}


def _mps_runs(part_algos, wrong):
    right = [{"index": 0, "algo_id": 66, "correct": True}]
    full = {"share": 100, "blas_exit": 0, "known_issues": {"matches": [{"id": "6862950", "status": "not active"}]},
            "blas": [_sweep(s, d, right) for s in ("1000x2048x16", "4096x4096x2048") for d in ("bfloat16", "float16")]}
    part = {"share": 40, "blas_exit": 1, "known_issues": {"matches": [{"id": "6862950", "status": "applies"}]},
            "blas": [_sweep("1000x2048x16", "bfloat16", part_algos, wrong)]}
    return {"runs": [full, part]}


def test_mps_workaround_must_be_tried_on_an_affected_algorithm():
    """H200 criterion: "every wrong one is algorithm 66 with a cluster of 4 or more CTAs, and the 1x1x1-cluster
    workaround is right". A workaround run on algorithm 39, which the issue never concerns, says nothing about it;
    it has to be a 1x1x1 rerun of an algorithm-66 configuration with 4 or more CTAs (sdcguard blas reruns each of
    those, under the index of the one it came from)."""
    f = _launcher("_mps_criteria")["_mps_criteria"]
    algo39 = {"index": 0, "algo_id": 39, "cluster_ctas": 2, "correct": True}
    algo66 = {"index": 1, "algo_id": 66, "cluster_ctas": 4, "correct": False}
    wrong = [{"algo_id": 66, "cluster_ctas": 4}]
    on39 = {"index": 0, "variant": "cluster_1x1x1", "algo_id": 39, "cluster_ctas": 1, "correct": True}
    on66 = {"index": 1, "variant": "cluster_1x1x1", "algo_id": 66, "cluster_ctas": 1, "correct": True}
    assert f(_mps_runs([algo39, algo66, on39], wrong))["all met"] is False
    assert f(_mps_runs([algo39, algo66, on66], wrong))["all met"] is True
    assert f(_mps_runs([algo39, algo66, dict(on66, correct=False)], wrong))["all met"] is False


def test_injected_fault_is_the_harm_study_10_percent_size():
    """METHOD 0.4.0: the injected wrong value changes its 128-column block by 10% on the harm study's measure,
    ||change|| / ||block|| (inject._rel_change), the size the default check caught 99% of there. The first version
    added 10% of the block's absolute sum, about 90% on that scale. Every runner injects through add_fault_."""
    v = _load("experiments/v040_check.py", "v040_check_fault")
    g = torch.Generator().manual_seed(0)
    rels = []
    for _ in range(50):
        y = torch.randn(4, 256, generator=g).to(torch.bfloat16)
        col = int(torch.randint(0, 256, (1,), generator=g))
        start = (col // 128) * 128
        old = y[0, start:start + 128].clone()
        v.add_fault_(y[0], col)
        rels.append(_rel_change(old, y[0, start:start + 128], old))
    assert all(0.09 <= r <= 0.11 for r in rels), rels
    for rel in ("experiments/v040_check.py", "experiments/v040_tp.py"):
        src = (ROOT / rel).read_text()
        assert "add_fault_(" in src and "abs().sum())" not in src, rel


def test_wrong_algorithms_are_reported_with_the_listed_cublas_issues_that_match_them():
    """v040 criterion 5 and the H200's 100% criterion fail on any wrong algorithm, including one NVIDIA already
    lists (6580689, a split-K bug in algorithm 66, matches compute capability 9.0 with cuBLAS 13.0). The report
    names the listed issues that apply or may apply here and tie to that algorithm, so such a failure is shown
    as the library's, apart from wrong algorithms nothing listed accounts for."""
    v = _load("experiments/v040_check.py", "v040_check_issues")
    env = {"cublas": "13.0.2", "cublas_source": "test", "gpu": "NVIDIA H200", "cc": "9.0", "sms": 52,
           "mps_share": 40.0, "mps": "partial (52 of 132 SMs visible)"}
    matches = known_issues.match(env)
    ids = v.listed_issues_for({"algo_id": 66, "cluster_ctas": 4}, matches)
    assert "6862950" in ids and "6580689" in ids
    small = v.listed_issues_for({"algo_id": 66, "cluster_ctas": 2}, matches)
    assert "6862950" not in small and "6580689" in small          # 6862950 needs 4 or more CTAs per cluster
    assert v.listed_issues_for({"algo_id": 39, "cluster_ctas": 2}, matches) == []
    full = known_issues.match(dict(env, sms=132, mps_share=None, mps="full"))
    assert "6862950" not in v.listed_issues_for({"algo_id": 66, "cluster_ctas": 4}, full)   # "not active"
    sweeps = [{"shape": [4096, 4096, 2048], "dtype": "bfloat16", "verdict": "FAIL",
               "wrong": [{"algo_id": 66, "cluster_ctas": 2}, {"algo_id": 39, "cluster_ctas": 1}]}]
    wrong = v.wrong_with_listed_issues(sweeps, full)
    r = _v040_result({"experts": 64, "calls": 200, "clean_alarms": 0, "unjudged": 0, "checks": 3,
                      "expected_checks": 3, "fault_trials": 100, "faults_caught": 100})
    r["blas"] = {"sweeps": sweeps, "wrong_with_listed_issues": wrong}
    c = v.criteria(r)
    assert c["5 blas: every algorithm right on every default shape (no MPS)"] is False and c["all met"] is False
    report = c["5 wrong algorithms and the listed cuBLAS issues that match them"]
    assert len(report) == 2 and "6580689" in report[0] and "no listed cuBLAS issue" in report[1]
    lines = _launcher("_library_lines")["_library_lines"]("h100 blas (no MPS)", wrong)
    assert len(lines) == 2 and "accounts for" in lines[0] and "6580689" in lines[0]
    assert "no listed cuBLAS issue" in lines[1] and "algorithm 39" in lines[1]


def test_a_split_k_bug_is_not_listed_for_a_configuration_that_ran_without_split_k():
    """6580689 is a split-K bug in algorithm 66. Every wrong configuration on the H200 at 40% ran with SPLITK_NUM 1
    (October 21, 2026), yet the summary listed 6580689 for each beside 6862950, and printed configurations that
    differ in cluster size as the same line, over and over. A configuration that ran without split-K can't have hit
    a split-K bug; the configurations of one algorithm on one shape are one entry, with their count."""
    v = _load("experiments/v040_check.py", "v040_check_splitk")
    env = {"cublas": "13.1.1", "cublas_source": "test", "gpu": "NVIDIA H200", "cc": "9.0", "sms": 52,
           "mps_share": 40.0, "mps": "partial (52 of 132 SMs visible)"}
    matches = known_issues.match(env)
    assert v.listed_issues_for({"algo_id": 66, "cluster_ctas": 8, "splitk": 1}, matches) == ["6862950"]
    assert "6580689" in v.listed_issues_for({"algo_id": 66, "cluster_ctas": 8, "splitk": 4}, matches)
    assert "6580689" in v.listed_issues_for({"algo_id": 66, "cluster_ctas": 8}, matches)   # split-K not known
    sweeps = [{"shape": [1000, 2048, 16], "dtype": "bfloat16", "verdict": "FAIL",
               "wrong": [{"index": 0, "algo_id": 66, "cluster_ctas": 8, "splitk": 1},
                         {"index": 1, "algo_id": 66, "cluster_ctas": 4, "splitk": 1}]},
              {"shape": [4096, 4096, 2048], "dtype": "float16", "verdict": "FAIL",
               "wrong": [{"index": i, "algo_id": 66, "cluster_ctas": 4 if i % 3 else 8, "splitk": 1}
                         for i in range(8)]}]
    wrong = v.wrong_with_listed_issues(sweeps, matches)
    assert len(wrong) == 10 and all(w["listed_issues"] == ["6862950"] for w in wrong)
    assert [w["index"] for w in wrong[:2]] == [0, 1] and wrong[0]["splitk"] == 1
    lines = _launcher("_library_lines")["_library_lines"]("h200 at 40%", wrong)
    assert lines == [("h200 at 40%: wrong cuBLASLt algorithms that a listed NVIDIA cuBLAS issue accounts for: "
                      "1000x2048x16 bfloat16 algorithm 66, 2 configurations (clusters of 4 and 8 CTAs, no split-K): "
                      "6862950; 4096x4096x2048 float16 algorithm 66, 8 configurations (clusters of 4 and 8 CTAs, "
                      "no split-K): 6862950")]
    r = _v040_result({"experts": 64, "calls": 200, "clean_alarms": 0, "unjudged": 0, "checks": 3,
                      "expected_checks": 3, "fault_trials": 100, "faults_caught": 100})
    r["blas"] = {"sweeps": sweeps, "wrong_with_listed_issues": wrong}
    report = v.criteria(r)["5 wrong algorithms and the listed cuBLAS issues that match them"]
    assert report[0] == "1000x2048x16 bfloat16 algorithm 66 (8-CTA clusters, no split-K): listed cuBLAS issue 6862950"


def test_each_gpu_function_checks_it_has_the_launched_code(tmp_path):
    """Each GPU function fingerprints its own copy of the code (as scripts/modal_check.py does) and stops before
    any GPU work when it isn't what the launcher fingerprinted; the image carries scripts/ because the
    fingerprint covers it."""
    for d, name in (("src/sdcguard", "a.py"), ("tests", "test_a.py"), ("experiments", "e.py"), ("scripts", "s.py")):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / name).write_text("x = 1\n")
    ns = _launcher("_digest", "_on_gpu", Path=Path, R=str(tmp_path))
    launched = ns["_digest"](tmp_path)["sha256"]
    assert ns["_on_gpu"](launched)["matches_launcher"] is True
    (tmp_path / "scripts" / "s.py").write_text("x = 2\n")
    assert ns["_on_gpu"](launched)["matches_launcher"] is False
    src = (ROOT / "scripts" / "modal_v040.py").read_text()
    for fn in ("h100", "h100x2", "h200"):
        body = src.split(f"def {fn}(run_id: str, expected: str) -> dict:\n", 1)[1]
        assert body.lstrip().startswith("src = _on_gpu(expected)\n    if not src[\"matches_launcher\"]:"), fn
    assert 'add_local_dir(LOCAL / "scripts"' in src


def test_launcher_records_git_state_or_that_there_is_none(tmp_path):
    """The launcher records the commit and whether the tree had changes; run from an extracted tree that isn't a
    git checkout, it says so instead of failing (or reporting an enclosing repository's commit)."""
    git = _launcher("_git", Path=Path, subprocess=subprocess)["_git"]
    plain = tmp_path / "extracted"
    plain.mkdir()
    assert git(plain) == {"git": "not a git checkout"}
    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)  # noqa: E731
    run("init", "-q")
    (repo / "f.py").write_text("x = 1\n")
    run("add", "f.py")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "commit", "-q", "-m", "x")
    head = run("rev-parse", "HEAD").stdout.strip()
    assert git(repo) == {"git_head": head, "git_dirty": False}
    (repo / "f.py").write_text("x = 2\n")
    assert git(repo) == {"git_head": head, "git_dirty": True}
    sub = repo / "inner"
    sub.mkdir()
    assert git(sub) == {"git": "not a git checkout"}              # inside another checkout, not its own


def test_cost_estimate_counts_the_pairs_memory():
    """The pair of H100s requests 64 GiB, about $0.70 an hour of CPU and memory, not the one-GPU machines' $0.45;
    at the functions' limits the run still comes to about $8.50, as the docstring and METHOD say."""
    ns = _launcher("RATES", "LIMITS")
    rates, limits = ns["RATES"], ns["LIMITS"]
    assert rates["h100x2"] == pytest.approx(2 * 3.95 + 0.70, abs=0.01)
    assert rates["h100"] == pytest.approx(3.95 + 0.45, abs=0.01) and rates["h200"] == pytest.approx(4.54 + 0.45,
                                                                                                     abs=0.01)
    most = sum(rates[k] * limits[k] / 3600 for k in rates)
    assert 8.3 <= most <= 8.7
    assert "at most about $8.50" in (ROOT / "scripts" / "modal_v040.py").read_text()
    assert "at most about $8.50" in (ROOT / "docs" / "METHOD.md").read_text()


_STRESS = {}


def _cpu_linear_rounds_like_fp32_accumulation(k=256, n=768) -> bool:
    """The check's noise model is that of a matmul accumulated in fp32 and rounded once. PyTorch 2.8's CPU bf16
    linear on arm64 isn't one at 768 inputs (27% of the outputs not correctly rounded, October 21, 2026; at 256
    inputs 0.01%, as fp32 accumulation gives), and the check rightly alarms on its clean outputs there."""
    torch.manual_seed(0)
    m = torch.nn.Linear(k, n).to(torch.bfloat16)
    x = torch.randn(8, k).to(torch.bfloat16)
    with torch.no_grad():
        y = torch.nn.functional.linear(x, m.weight, m.bias)
        ref = (x.double() @ m.weight.double().t() + m.bias.double()).to(torch.bfloat16)
    return float((y != ref).float().mean()) <= 0.01


def _stress_report():
    """experiments/v040_check.capture_stress run once on the CPU (the PyTorch check, no graph)."""
    if not _cpu_linear_rounds_like_fp32_accumulation():
        pytest.skip("this PyTorch's CPU bf16 linear isn't rounded like an fp32 accumulation at the stress's size")
    if not _STRESS:
        v = _load("experiments/v040_check.py", "v040_stress_run")
        _STRESS["report"] = v.capture_stress(torch.device("cpu"), say=lambda s: None)
    return json.loads(json.dumps(_STRESS["report"]))


def _kept_alert(layer, row, blk):
    return {"where": layer, "captured": {"row": row, "col_block": blk, "with_bias": True, "z": 99.0},
            "verdict": "output_wrong", "outside_envelope": 1}


def _met_stress_report():
    """A criterion-8 report from a GPU run that meets it (stated, not computed: it doesn't depend on this
    machine's arithmetic)."""
    full = "the capture buffer was full (2 rows not kept)"
    return {"layers": 6, "batch": 8, "graph": True, "slots": 4, "uses_fast": [True] * 6, "clean_before": 0,
            "clean_between": [0, 0, 0], "stats": {"alerts": 14, "captured": 12, "captures_dropped": 2},
            "a": {"made": 8 * 32, "alerts": [_kept_alert(3, 5, 4)], "captured_was_made_wrong": True},
            "b": {"alerts": [_kept_alert(i, i, i) for i in range(4)]},
            "c": {"alerts": [_kept_alert(i, i, i) for i in range(4)]
                  + [{"where": i, "captured": None, "capture_missing": full} for i in (4, 5)],
                  "then": {"alerts": [_kept_alert(5, 1, 2)]}, "dropped": 2},
            "d": {"alerts_first_read": [], "alerts_second_read": [_kept_alert(0, 2, 3)],
                  "alerts_last_read": [_kept_alert(1, 3, 4)]}}


def _complete_v040_report():
    """A report of the registered run done in full, every criterion met (the numbers are the protocol's)."""
    from sdcguard.blas import DEFAULT_SHAPES
    return {"capture_stress": _met_stress_report(),
            "fingerprints": {"agree": True, "kernel_used": True},
            "compile": {"checks_per_checked_step": [27] * 4, "eager_checks_per_step": 27,
                        "checks_on_unchecked_steps": 0, "alerts": 0, "unjudged": 0, "fault_trials": 5,
                        "faults_caught": 5, "ms_unchecked": 12.0, "ms_unguarded": 12.0},
            "grouped": {"experts": 64, "calls": 200, "clean_alarms": 0, "unjudged": 0, "expected_checks": 38016,
                        "checks": 38016, "fault_trials": 100, "faults_caught": 100},
            "blas": {"sweeps": [{"shape": list(sh), "dtype": dt, "verdict": "PASS", "algorithms_tested": 5,
                                 "wrong": [], "error": None}
                                for sh in DEFAULT_SHAPES for dt in ("bfloat16", "float16")],
                     "wrong_with_listed_issues": []},
            "capture": {"graph": True, "replays": 200, "first_use_comparison": {"bfloat16": True, "float16": True},
                        "uses_fast": [True] * 4, "fault": {"replay": 66, "row": 5, "col": 1000},
                        "checker_fault": {"replay": 133}, "evidence_holds_the_wrong_value": True,
                        "alerts": [{"replay": 66, "where": "1", "captured": {"row": 5, "col_block": 7},
                                    "verdict": "output_wrong", "outside_envelope": 1},
                                   {"replay": 133, "where": "2", "verdict": "output_agrees"}],
                        "check_us": {"with": 73.0, "without": 79.1}}}


def test_a_complete_run_meets_every_v040_criterion():
    assert _load("experiments/v040_check.py", "v040_complete").criteria(_complete_v040_report())["all met"] is True


def test_a_met_stress_report_meets_criterion_8_and_one_without_it_doesnt():
    v = _load("experiments/v040_check.py", "v040_stress_stated")
    assert v.stress_met(_met_stress_report()) is True
    r = _complete_v040_report()
    del r["capture_stress"]
    assert v.criteria(r)["all met"] is False


@pytest.mark.parametrize("gap", ["no capture timing", "no CUDA graph", "kernel use not recorded",
                                 "no compiled fault trials", "no grouped fault trials", "one blas sweep",
                                 "one grouped call", "one capture replay", "a sweep that ran no algorithm",
                                 "one compiled checked step", "one grouped check", "a sweep with a wrong algorithm"])
def test_a_v040_report_with_a_stage_not_run_in_full_meets_nothing(gap):
    """An outside review (October 21, 2026) showed each of these reports met every criterion: the judging asked
    for no more than what the report happened to contain. Each is now a criterion not met."""
    r = _complete_v040_report()
    change = {"no capture timing": lambda: r["capture"].pop("check_us"),
              "no CUDA graph": lambda: r["capture"].update(graph=False),
              "kernel use not recorded": lambda: r["fingerprints"].pop("kernel_used"),
              "no compiled fault trials": lambda: r["compile"].update(fault_trials=0, faults_caught=0),
              "no grouped fault trials": lambda: r["grouped"].update(fault_trials=0, faults_caught=0),
              "one blas sweep": lambda: r["blas"].update(sweeps=r["blas"]["sweeps"][:1]),
              "one grouped call": lambda: r["grouped"].update(calls=1),
              "one capture replay": lambda: r["capture"].update(replays=1),
              "a sweep that ran no algorithm": lambda: r["blas"]["sweeps"][0].update(algorithms_tested=0),
              "one compiled checked step": lambda: r["compile"].update(checks_per_checked_step=[27]),
              "one grouped check": lambda: r["grouped"].update(checks=1, expected_checks=1),
              "a sweep with a wrong algorithm": lambda: r["blas"]["sweeps"][0].update(wrong=[{"algo_id": 66}])}
    change[gap]()
    assert _load("experiments/v040_check.py", "v040_gap").criteria(r)["all met"] is False, gap


def test_capture_under_contention_holds_on_the_cpu():
    """Criterion 8's sequence through the PyTorch check: a layer whose every row alarms keeps one row it made wrong;
    four layers at once keep four rows, each its own; six keep four and the other two say the buffer was full, and
    the next read has room again; two reads in flight keep each alert's own row. On a GPU it must also have run in
    a CUDA graph with the Triton check, which a CPU run hasn't."""
    v = _load("experiments/v040_check.py", "v040_stress")
    st = _stress_report()
    assert v.stress_met(st, gpu=False) is True and v.stress_met(st) is False and v.stress_met({}) is None
    assert len(st["a"]["alerts"]) == 1 and st["a"]["made"] == 8 * 6 and st["c"]["dropped"] == 2
    assert st["stats"]["captures_dropped"] == 2 and st["clean_before"] == 0


@pytest.mark.parametrize("tamper", ["crossed in flight", "every layer kept", "two wrong outputs in a block",
                                    "a row nobody made wrong", "an alert on clean traffic", "an alert nobody made",
                                    "stage missing"])
def test_criterion_8_isnt_met_by_captures_that_dont_hold_up(tamper):
    v = _load("experiments/v040_check.py", "v040_stress_tamper")
    st = _stress_report()
    if tamper == "crossed in flight":
        reads = [r for r in ("alerts_first_read", "alerts_second_read", "alerts_last_read") if st["d"][r]]
        alerts = [a for r in reads for a in st["d"][r]]
        alerts[0]["captured"], alerts[1]["captured"] = alerts[1]["captured"], alerts[0]["captured"]
    elif tamper == "every layer kept":
        for a in st["c"]["alerts"]:
            a["captured"] = a["captured"] or {"row": a["where"] % 8, "col_block": a["where"], "with_bias": True,
                                              "z": 99.0}
    elif tamper == "two wrong outputs in a block":
        st["b"]["alerts"][2]["outside_envelope"] = 2
    elif tamper == "a row nobody made wrong":
        st["a"]["captured_was_made_wrong"] = False
    elif tamper == "an alert on clean traffic":
        st["clean_between"][1] = 1
    elif tamper == "an alert nobody made":
        st["stats"]["alerts"] += 1
    else:
        del st["d"]
    assert v.stress_met(st, gpu=False) is False, tamper

