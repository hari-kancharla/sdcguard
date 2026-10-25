"""experiments/vllm_norm_diag.py without a GPU: the near-tie rule at the first differing token, the comparison of
answers and of compiled norm kernels, and a summary that is complete only when every server ran."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_norm_diag as nd  # noqa: E402


def _ans(tokens, tops):
    return {"text": "".join(tokens), "tokens": tokens, "top": tops}


def test_the_first_divergence_and_the_near_tie_rule():
    a = _ans(["A", "b", "c"], [{"A": -0.1}, {"b": -0.6, "x": -0.7}, {"c": -0.1}])
    b = _ans(["A", "x", "y"], [{"A": -0.1}, {"x": -0.6, "b": -0.65}, {"y": -0.1}])
    d = nd.first_divergence(a, b)
    assert d["index"] == 1 and d["tokens"] == ["b", "x"] and d["gaps"] == [0.1, 0.05] and d["near_tie"] is True
    far = _ans(["A", "x", "y"], [{"A": -0.1}, {"x": -0.1, "b": -2.0}, {"y": -0.1}])
    assert nd.first_divergence(a, far)["near_tie"] is False            # one side preferred its own by 1.9 nat
    gone = _ans(["A", "x", "y"], [{"A": -0.1}, {"x": -0.1, "q": -0.2}, {"y": -0.1}])
    assert nd.first_divergence(a, gone)["gaps"][1] is None and not nd.first_divergence(a, gone)["near_tie"]
    c = nd.compare([a, a, a], [a, b, far])
    assert c["differ"] == 2 and c["all_near_ties"] is False and [x["prompt"] for x in c["diffs"]] == [1, 2]
    assert nd.compare([], [a])["compared"] is False


def test_norm_kernels_are_compared_by_source_without_names_and_configs_as_a_set(tmp_path):
    def write(root, name, body, cfg):
        d = tmp_path / root / "x1"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.py").write_text(f"# comment {name}\n@triton.jit\ndef {name}(in_ptr0):\n    {body}\n")
        (d / f"c{len(root)}{len(name)}.best_config").write_text(json.dumps({**cfg, "time_taken_ms": 3}))
    write("a", "triton_red_fused_add_rsqrt_0", "x = rsqrt(v)", {"R0_BLOCK": 4096, "num_warps": 8})
    write("b", "triton_red_fused_add_rsqrt_7", "x = rsqrt(v)", {"R0_BLOCK": 4096, "num_warps": 8})
    write("c", "triton_red_fused_add_rsqrt_0", "x = rsqrt(v)", {"R0_BLOCK": 2048, "num_warps": 8})
    write("d", "triton_red_fused_add_rsqrt_0", "x = rsqrt(v) * 2", {"R0_BLOCK": 4096, "num_warps": 8})
    (tmp_path / "a" / "x1" / "other.py").write_text("@triton.jit\ndef triton_poi_0(x):\n    y = x\n")
    for root, op in (("a", "pass"), ("b", "torch.ops.sdcguard.norm_check(buf0)")):    # wrappers: left out
        (tmp_path / root / "x1" / "wrapper.py").write_text(
            "async_compile.triton('k', \"\"\"@triton.jit\ndef triton_x(): rsqrt\"\"\")\ndef call(args):\n    " + op + "\n")
    ka, kb = nd.norm_kernels(str(tmp_path / "a"), keep=str(tmp_path / "a" / "kept")), nd.norm_kernels(
        str(tmp_path / "b"))
    kc, kd = nd.norm_kernels(str(tmp_path / "c")), nd.norm_kernels(str(tmp_path / "d"))
    assert len(ka) == 1 and nd.kernel_diff(ka, kb)["same"] is True                 # renamed only: the same
    assert list((tmp_path / "a" / "kept").iterdir())[0].name == f"{next(iter(ka))}.py"
    assert nd.kernel_diff(ka, kc)["same"] is True                                  # the same source...
    ca, cc = nd.tuned_configs(str(tmp_path / "a")), nd.tuned_configs(str(tmp_path / "c"))
    assert nd.config_diff(ca, cc)["same"] is False                                 # ...another tuned config
    assert nd.config_diff(ca, nd.tuned_configs(str(tmp_path / "b")))["same"] is True
    dd = nd.kernel_diff(ka, kd)
    assert dd["only_first"] and dd["only_second"] and not dd["same"]
    assert nd.kernel_diff({}, ka)["compared"] is False and nd.config_diff(None, ca)["compared"] is False


def test_the_summary_is_complete_only_when_every_server_answered():
    a = _ans(["A"], [{"A": -0.1}])
    res = {r: {"ready": True, "answers": [a] * 3, "norm_kernels": {}, "tuned_configs": []} for r in nd.ROLES}
    s = nd.summarize(res)
    assert s["complete"] is True and s["pairs"]["nonorms_vs_guarded"]["answers"]["differ"] == 0
    res["guarded_det"] = {"skipped": "the run's time was nearly up"}
    s = nd.summarize(res)
    assert s["complete"] is False and s["pairs"]["plain_det_vs_guarded_det"]["answers"]["compared"] is False


def test_the_deterministic_servers_set_it_in_vllms_inductor_config(monkeypatch, tmp_path):
    seen = {}

    class P:
        def __init__(self, cmd, env, **k):
            seen.setdefault("cmds", []).append(cmd)
            seen.setdefault("envs", []).append(env)
    monkeypatch.setattr(nd.subprocess, "Popen", P)
    for role in nd.ROLES:
        _, log = nd.server(role, str(tmp_path), 8000, 9400)
        log.close()
    cmds, envs = dict(zip(nd.ROLES, seen["cmds"])), dict(zip(nd.ROLES, seen["envs"]))
    det = '-cc.inductor_compile_config={"deterministic": true}'
    assert det in cmds["plain_det"] and det in cmds["guarded_det"] and det not in cmds["guarded"]
    assert "SDCGUARD" not in envs["plain_a"] and envs["plain_exact"]["SDCGUARD_TEST_EXACT"] == "1"
    assert envs["nonorms"]["SDCGUARD_NORMS"] == "0" and envs["guarded"]["SDCGUARD_NORMS"] == "1"
    assert len({e["VLLM_CACHE_ROOT"] for e in envs.values()}) == len(nd.ROLES)    # fresh caches for each
    assert len({e["TRITON_CACHE_DIR"] for e in envs.values()}) == len(nd.ROLES)
