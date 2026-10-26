"""experiments/vllm_norm_diag.py without a GPU: the near-tie rule at the first differing token, the comparison of
answers and of compiled norm kernels, and a summary that is complete only when every server ran."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_norm_diag as nd  # noqa: E402


def _ans(tokens, tops):
    return {"text": "".join(tokens), "tokens": tokens, "top": tops}


def test_the_first_divergence_and_the_near_tie_rule(monkeypatch):
    monkeypatch.setattr(nd, "IDENTITY", 3)
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
    err = {"error": "HTTP 500: x"}
    c = nd.compare([a, err, a], [a, a, a])                  # a failed request isn't a different answer
    assert c["compared"] is False and c["failed"] == 1 and "prompt 1" in c["why"]


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


def test_the_summary_is_complete_only_when_every_server_answered(monkeypatch):
    monkeypatch.setattr(nd, "IDENTITY", 3)
    a = _ans(["A"], [{"A": -0.1}])

    def full():
        return {r: {"ready": True, "answers": [a] * 3, "norm_kernels": {"h": {"names": ["k"]}}, "tuned_configs": []}
                for r in nd.ROLES}
    s = nd.summarize(full())
    assert s["complete"] is True and s["pairs"]["nonorms_vs_guarded"]["answers"]["differ"] == 0, s
    res = full()
    res["guarded_det"] = {"skipped": "the run's time was nearly up"}
    s = nd.summarize(res)
    assert s["complete"] is False and s["pairs"]["plain_det_vs_guarded_det"]["answers"]["compared"] is False
    # every server up, but every request failed: before, this counted as complete (and as 64 differences)
    res = full()
    for r in nd.ROLES:
        res[r]["answers"] = [{"error": "HTTP 500: boom"}] * 3
    s = nd.summarize(res)
    assert s["complete"] is False and any("0 of 3 usable" in w for w in s["incomplete_because"])
    res = full()
    res["guarded"]["answers"] = [a, a]                    # a missing completion
    assert nd.summarize(res)["complete"] is False
    res = full()
    res["plain_b"]["norm_kernels"] = {}                   # no compiled norm kernel to compare
    s = nd.summarize(res)
    assert s["complete"] is False and any("no compiled norm kernel" in w for w in s["incomplete_because"])


def test_a_completion_without_its_tokens_or_log_probabilities_is_unusable(monkeypatch):
    toks = list(range(nd.TOKENS))
    good = {"choices": [{"text": "x", "logprobs": {"tokens": toks, "top_logprobs": [{"1": -0.1}] * nd.TOKENS}}]}
    monkeypatch.setattr(nd.vn, "_post", lambda *a: (200, good))
    assert "error" not in nd.complete(8000, "p")
    short = {"choices": [{"text": "x", "logprobs": {"tokens": toks[:5], "top_logprobs": [{"1": -0.1}] * 5}}]}
    monkeypatch.setattr(nd.vn, "_post", lambda *a: (200, short))
    assert "unusable" in nd.complete(8000, "p")["error"]
    none = {"choices": [{"text": "x", "logprobs": None}]}
    monkeypatch.setattr(nd.vn, "_post", lambda *a: (200, none))
    assert "unusable" in nd.complete(8000, "p")["error"]
    monkeypatch.setattr(nd.vn, "_post", lambda *a: (500, "down"))
    assert nd.complete(8000, "p")["error"].startswith("HTTP 500")


def test_a_server_waits_only_as_long_as_the_run_has_left(monkeypatch, tmp_path):
    waits, started = [], []
    clock = [0.0]

    class P:
        pass

    def server(role, out, port, metrics_port):
        started.append(role)
        return P(), open(os.path.join(str(tmp_path), f"{role}.log"), "w")

    def wait_ready(port, proc, timeout):
        waits.append(timeout)
        clock[0] += 9 * 60                               # each server's startup takes 9 minutes
        return False
    monkeypatch.setattr(nd, "server", server)
    monkeypatch.setattr(nd.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(nd.vs, "wait_ready", wait_ready)
    monkeypatch.setattr(nd.vs, "stop", lambda proc: {})
    res = nd.run(str(tmp_path), 8000, 9400)
    left, n = [], 0                                       # what the run should do: start while MIN_START_S is left
    while n < len(nd.ROLES) and nd.RUN_S - 9 * 60 * n >= nd.MIN_START_S:
        left.append(nd.RUN_S - 9 * 60 * n)
        n += 1
    assert 0 < n < len(nd.ROLES)                          # (9-minute starts: not all 7 fit, so the skip is tested)
    assert started == list(nd.ROLES[:n]) and all("skipped" in res[r] for r in nd.ROLES[n:])
    assert waits == [max(60.0, min(1200.0, x - nd.ANSWER_S)) for x in left]
    assert all(w + nd.ANSWER_S <= x for w, x in zip(waits, left)) and waits[-1] < 1200.0


def test_tuned_configs_are_paired_with_their_kernels(tmp_path):
    def write(root, name, body, cfg):
        d = tmp_path / root / "ab"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.py"
        path.write_text(f"@triton.jit\ndef {name}(in_ptr0):\n    {body}\n")
        bc = nd.best_config_path(str(path))            # (PyTorch's own naming, in whichever version is installed)
        assert bc and os.path.dirname(bc) == str(d) and bc.endswith(".best_config")
        with open(bc, "w") as f:
            json.dump({**cfg, "time_taken_ms": 1}, f)
    write("a", "triton_red_fused_rsqrt_0", "x = rsqrt(v)", {"R0_BLOCK": 4096})
    write("a", "triton_poi_fused_1", "y = v", {"XBLOCK": 128})
    write("b", "triton_red_fused_rsqrt_5", "x = rsqrt(v)", {"R0_BLOCK": 2048})      # same source, renamed
    write("b", "triton_poi_fused_9", "y = v", {"XBLOCK": 128})
    a, b = nd.configs_by_kernel(str(tmp_path / "a")), nd.configs_by_kernel(str(tmp_path / "b"))
    assert len(a) == 2 and len(b) == 2
    d = nd.tuned_by_kernel_diff(a, b)
    assert d["kernels_in_both"] == 2 and len(d["tuned_differently"]) == 1 and not d["same"]
    assert d["tuned_differently"][0]["first"] == ['{"R0_BLOCK": 4096}']
    assert nd.tuned_by_kernel_diff({}, b)["compared"] is False
    # two kernels of one server differing by name only: both configurations kept, compared as a set
    write("c", "triton_red_fused_rsqrt_0", "x = rsqrt(v)", {"R0_BLOCK": 4096})
    write("c", "triton_red_fused_rsqrt_3", "x = rsqrt(v)", {"R0_BLOCK": 1024})
    write("d", "triton_red_fused_rsqrt_8", "x = rsqrt(v)", {"R0_BLOCK": 1024})
    write("d", "triton_red_fused_rsqrt_9", "x = rsqrt(v)", {"R0_BLOCK": 4096})
    c, dd = nd.configs_by_kernel(str(tmp_path / "c")), nd.configs_by_kernel(str(tmp_path / "d"))
    assert len(next(iter(c.values()))) == 2 and nd.tuned_by_kernel_diff(c, dd)["same"] is True


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
