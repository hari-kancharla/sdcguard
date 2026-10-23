"""experiments/machete_diag.py without vLLM or a GPU: reading Inductor's generated code, comparing runs, the rule that
picks the fix, the variants it installs, and one run end to end with a stand-in vLLM."""
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "experiments" / "machete_diag.py"

TRITON = '''
# kernel path: /tmp/x/ab/cabc.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', \'\'\'
import triton
@triton_heuristics.pointwise(size_hints={'x': 1024}, filename=__file__, triton_meta={},
    inductor_meta={'kernel_name': 'triton_poi_fused_add_0', 'backend_hash': 'ABC123', 'num_load': 2})
@triton.jit
def triton_poi_fused_add_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    # a comment
    x0 = xindex
\'\'\', device_str='cuda')

triton_red_fused_rms_1 = async_compile.triton('triton_red_fused_rms_1', \'\'\'
@triton.jit
def triton_red_fused_rms_1(in_ptr0, out_ptr0, xnumel, r0_numel):
    tmp0 = tl.sum(x, 1)
\'\'\', device_str='cuda')
async_compile.wait(globals())

def call(args):
    arg0_1, arg1_1 = args
    buf0 = torch.ops._C.machete_mm.default(arg0_1, arg1_1, 101, None, None, None, 64, None, None, None)
    buf1 = empty_strided_cuda((s0, 192), (192, 1), torch.bfloat16)
    triton_poi_fused_add_0.run(buf0, buf1, s0, stream=stream0)
    del buf0
    triton_red_fused_rms_1.run(buf1, buf2, s0, 192, stream=stream0)
    return (buf2, )
'''


@pytest.fixture(scope="module")
def diag():
    spec = importlib.util.spec_from_file_location("machete_diag_under_test", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_code_is_read_as_kernels_and_a_call_order(diag):
    r = diag.kernels_and_calls(TRITON)
    assert len(r["kernels"]) == 2
    k = [v for v in r["kernels"].values() if "pointwise" in v][0]
    assert "triton_poi" not in k and "KERNEL" in k and "backend_hash" not in k and "# a comment" not in k
    ids = {("pointwise" in v): kid for kid, v in r["kernels"].items()}
    assert r["calls"] == ["X:MACHETE", f"K:{ids[True]}", f"K:{ids[False]}"]


def test_kernel_ids_ignore_names_but_not_code_and_raw_ids_see_both(diag):
    renamed = TRITON.replace("triton_poi_fused_add_0", "triton_poi_fused_add_7").replace(
        "'backend_hash': 'ABC123'", "'backend_hash': 'FFFF'").replace("# a comment", "# another")
    r, t = diag.kernels_and_calls(renamed), diag.kernels_and_calls(TRITON)
    assert r["kernels"] == t["kernels"] and r["calls"] == t["calls"]
    assert set(r["raw"]) != set(t["raw"]) and sorted(r["raw"].values()) == ["triton_poi_fused_add_7",
                                                                            "triton_red_fused_rms_1"]
    changed = TRITON.replace("x0 = xindex", "x0 = xindex + 1")
    assert set(diag.kernels_and_calls(changed)["kernels"]) != set(t["kernels"])


def test_every_kind_of_opaque_call_is_kept_in_order(diag):
    code = TRITON.replace("    del buf0\n", "    torch.ops.sdcguard.check_static.default(buf0)\n"
                                           "    buf9 = torch.ops.higher_order.auto_functionalized(x)\n"
                                           "    aten.copy_.default(a, b)\n    extern_kernels.mm(a, b, out=c)\n")
    calls = diag.kernels_and_calls(code)["calls"]
    assert [c for c in calls if c.startswith("X:")] == ["X:MACHETE", "X:sdcguard.check_static",
                                                        "X:higher_order.auto_functionalized", "X:aten.copy_",
                                                        "X:extern.mm"]
    for op in ("sdcguard.machete_mm", "sdcguard_diag.machete_checked", "sdcguard_diag.machete_mm", "_C.machete_mm"):
        c = TRITON.replace("torch.ops._C.machete_mm.default", f"torch.ops.{op}.default")
        assert diag.kernels_and_calls(c)["calls"][0] == "X:MACHETE"


def _trace(root, graphs):
    """A debug trace folder as Inductor writes one: torchinductor/<graph>.<n>/output_code.py (None: a cache hit)."""
    for n, code in enumerate(graphs):
        d = root / "torchinductor" / f"model__0_inference_0.{n}"
        d.mkdir(parents=True)
        if code is not None:
            (d / "output_code.py").write_text(code)
    return str(root)


def test_code_summary_and_comparison(diag, tmp_path):
    base = diag.code_summary(_trace(tmp_path / "b", [TRITON, None, None]))
    assert base["graphs"] == 3 and base["with_code"] == 1 and len(base["sequences"]) == 1
    same = diag.code_summary(_trace(tmp_path / "s", [TRITON.replace("torch.ops._C.machete_mm",
                                                                    "torch.ops.sdcguard.machete_mm")] * 3))
    c = diag.compare_code(base, same)
    assert c["same"] and c["raw_same"] and c["captured"] and c["kernels_only_here"] == []
    assert c["sequences_only_here"] == 0 and c["raw_only_here"] == []
    # the same code under other names (round 2: the op named machete_checked): same, but not byte for byte
    renamed = diag.code_summary(_trace(tmp_path / "r", [TRITON.replace("triton_red_fused_rms_1",
                                                                       "triton_red_fused_rms_machete_checked_1")]))
    c = diag.compare_code(base, renamed)
    assert c["same"] and not c["raw_same"]
    assert c["raw_only_here"] == ["triton_red_fused_rms_machete_checked_1"]
    assert c["raw_only_in_base"] == ["triton_red_fused_rms_1"]
    other = diag.code_summary(_trace(tmp_path / "o", [TRITON.replace("x0 = xindex", "x0 = xindex + 1")]))
    c = diag.compare_code(base, other)
    assert not c["same"] and len(c["kernels_only_here"]) == 1 and len(c["kernels_only_in_base"]) == 1
    extra = diag.code_summary(_trace(tmp_path / "e", [TRITON.replace(
        "    del buf0\n", "    torch.ops.sdcguard.check_static.default(buf0)\n")]))
    c = diag.compare_code(base, extra)
    assert not c["same"] and c["kernels_only_here"] == [] and c["sequences_only_here"] == 1
    none = diag.code_summary(str(tmp_path / "missing"))
    assert none["graphs"] == 0 and not diag.compare_code(base, none)["captured"]


def _result(outputs, light, checks=((112, 112),), autotuned=None, cache="shared"):
    """One run's JSON; light: its light-load completions."""
    return {"model": "m", "outputs": outputs, "light_outputs": light, "inductor_cache": cache,
            "checks": [{"layers": a, "layers_with_checks": b} for a, b in checks], "autotuned": autotuned or {}}


def _write(out, name, res, code=TRITON):
    import shutil
    (out / f"{name}.json").write_text(json.dumps(res))
    shutil.rmtree(out / "code" / name, ignore_errors=True)
    if code is not None:
        _trace(out / "code" / name, [code])


RUNS_ALL = ["baseline", "fused", "fused_renamed", "fused_slots", "separate", "baseline_fresh", "fused_fresh"]


def test_the_comparison_and_the_rule_that_picks_the_fix(diag, tmp_path):
    out = tmp_path
    fused_code = TRITON.replace("torch.ops._C.machete_mm", "torch.ops.sdcguard.machete_mm")
    renamed_code = fused_code.replace("triton_red_fused_rms_1", "triton_red_fused_rms_machete_checked_1")
    _write(out, "baseline", {**_result(["a", "b", "c"], ["a"]),
                             "autotuned": {"k1": {"XBLOCK": 64}, "k2": {"XBLOCK": 8}}})
    _write(out, "separate", _result(["a", "x", "c"], ["a"]), TRITON.replace(
        "    del buf0\n", "    torch.ops.sdcguard.check_static.default(buf0)\n"))
    _write(out, "fused", _result(["a", "b", "c"], ["a"]), fused_code)
    _write(out, "fused_renamed", {**_result(["a", "b", "y"], ["a"], autotuned={"k9": {"R0_BLOCK": 16}}),
                                  "autotuned_kernels": {"k9": ["triton_red_fused_rms_machete_checked_1"]}},
           renamed_code)
    _write(out, "fused_slots", _result(["a", "b", "c"], ["a"]), fused_code)
    _write(out, "baseline_fresh", _result(["a", "b", "c"], ["a"], cache="its own",
                                          autotuned={"k1": {"XBLOCK": 64}, "k2": {"XBLOCK": 16}}))
    s = diag.compare(str(out), RUNS_ALL)
    r = s["runs"]
    assert s["prompts"] == 3 and s["baseline_autotuned"] == 2
    assert r["separate"]["same"] == "2/3" and r["separate"]["differing"] == [1] and not r["separate"]["all_same"]
    assert r["separate"]["code"]["sequences_only_here"] == 1 and not r["separate"]["code"]["same"]
    assert r["fused"]["all_same"] and r["fused"]["code"]["same"] and r["fused"]["code"]["raw_same"]
    rn = r["fused_renamed"]
    assert rn["code"]["same"] and not rn["code"]["raw_same"] and rn["differing"] == [2]
    assert rn["code"]["raw_only_here"] == ["triton_red_fused_rms_machete_checked_1"]
    assert rn["autotuned_here"] == 1 and rn["autotuned_not_in_baseline"] == ["k9"]
    assert rn["autotuned_kernels_not_in_baseline"] == ["triton_red_fused_rms_machete_checked_1"]
    fresh = r["baseline_fresh"]
    assert fresh["autotuned_like_baseline"] == 1 and fresh["autotuned_unlike_baseline"] == ["k2"]
    assert r["fused_fresh"] == {"error": "no result"}
    assert s["fix"] == "fused"
    assert json.loads((out / "DIAG.json").read_text()) == s
    assert "graph sequences only in this run" in (out / "code_diff_separate.txt").read_text()
    assert "triton_red_fused_rms_machete_checked_1" in (out / "code_diff_fused_renamed.txt").read_text()
    assert r["fused_slots"]["all_same"]
    # fused: one completion different -> no fix (the other variants are reported, they're not candidates)
    _write(out, "fused", _result(["a", "z", "c"], ["a"]), fused_code)
    assert diag.compare(str(out), RUNS_ALL)["fix"] is None
    # the same completions, but other kernels: not a fix
    _write(out, "fused", _result(["a", "b", "c"], ["a"]), TRITON.replace("x0 = xindex", "x0 = xindex + 1"))
    assert diag.compare(str(out), RUNS_ALL)["fix"] is None
    # the same completions and code, but under other kernel names: not a fix either
    _write(out, "fused", _result(["a", "b", "c"], ["a"]), renamed_code)
    assert diag.compare(str(out), RUNS_ALL)["fix"] is None
    # and in the shared cache, a kernel autotuned anew: not a fix
    _write(out, "fused", _result(["a", "b", "c"], ["a"], autotuned={"k7": {"XBLOCK": 64}}), fused_code)
    assert diag.compare(str(out), RUNS_ALL)["fix"] is None
    _write(out, "fused", _result(["a", "b", "c"], ["a"]), fused_code)
    assert diag.compare(str(out), RUNS_ALL)["fix"] == "fused"


def test_the_first_64_completions_are_counted_too(diag, tmp_path):
    """vllm_check compares the first 64 of 256 requests: the diagnostic counts both."""
    base = [f"t{i}" for i in range(256)]
    other = list(base)
    other[100] = "x"
    _write(tmp_path, "baseline", _result(base, ["a"]))
    _write(tmp_path, "fused", _result(other, ["a"]))
    r = diag.compare(str(tmp_path), ["baseline", "fused"])["runs"]["fused"]
    assert r["same"] == "255/256" and r["same_first_64"] == "64/64" and r["differing"] == [100]


GOOD = {"all_same": True, "code": {"captured": True, "same": True, "raw_same": True}, "inductor_cache": "shared",
        "autotuned_here": 0, "checks": [{"layers": 112, "layers_with_checks": 112}]}


@pytest.mark.parametrize("change,ok", [
    ({}, True),
    ({"code": {"captured": False, "same": False, "raw_same": False}}, True),   # no code captured: the rest decides
    ({"all_same": False}, False),
    ({"code": {"captured": True, "same": False, "raw_same": False}}, False),
    ({"code": {"captured": True, "same": True, "raw_same": False}}, False),     # other kernel names
    ({"autotuned_here": 2}, False),                         # in the shared cache, a kernel autotuned anew
    ({"autotuned_here": 8, "inductor_cache": "its own"}, True),                 # its own cache: all autotuned anew
    ({"checks": [{"layers": 112, "layers_with_checks": 111}]}, False),          # a layer whose check never ran
    ({"checks": [None]}, False),
    ({"checks": None}, False),
])
def test_the_fix_rule(diag, change, ok):
    assert diag._ok({**GOOD, **change}) is ok


def test_autotuned_configs_are_read_from_the_cache_by_time(diag, tmp_path):
    old, new = tmp_path / "ab" / "cold.best_config", tmp_path / "cd" / "cnew.best_config"
    old.parent.mkdir()
    new.parent.mkdir()
    old.write_text(json.dumps({"XBLOCK": 8}))
    new.write_text(json.dumps({"XBLOCK": 64, "num_warps": 4, "num_stages": 1, "configs_hash": "c1",
                               "found_by_coordesc": False, "time_taken_ms": 812, "triton_cache_hash": "t9"}))
    os.utime(old, (1000, 1000))
    (tmp_path / "cd" / "cnew.py").write_text("kernel")
    # the launch config only: what the benchmarking took and the Triton cache hash change between identical picks
    assert diag._autotuned(str(tmp_path), since=2000) == {"cnew": {"XBLOCK": 64, "num_warps": 4, "num_stages": 1}}
    assert set(diag._autotuned(str(tmp_path), since=0)) == {"cold", "cnew"}
    (tmp_path / "cd" / "cnew.py").write_text("import triton\n@triton.jit\ndef triton_red_fused_rms_norm_machete_mm_0("
                                            "in_ptr0):\n    pass\ndef helper():\n    pass\n")
    assert diag._autotuned_names(str(tmp_path), since=2000) == {"cnew": ["triton_red_fused_rms_norm_machete_mm_0"]}


def _fake_vllm(monkeypatch, log):
    """A MacheteLinearKernel whose apply_weights is shaped like the plugin's (vLLM's own in _orig), and a plugin
    whose register() does nothing."""
    def vllm_apply(self, layer, x, bias=None):
        log.append(("vllm", bias is not None))
        return x

    def shipped(self, layer, x, bias=None, _orig=vllm_apply, _buffers=(11, 15, 16)):
        log.append(("sdcguard", bias is not None))
        return x

    K = type("MacheteLinearKernel", (), {"apply_weights": shipped})
    mod = types.ModuleType("machete")
    mod.MacheteLinearKernel = K
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.kernels", "vllm.model_executor.kernels.linear",
                 "vllm.model_executor.kernels.linear.mixed_precision"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.model_executor.kernels.linear.mixed_precision"].machete = mod
    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.linear.mixed_precision.machete", mod)
    from sdcguard import serve, vllm_plugin
    monkeypatch.setattr(vllm_plugin, "register", lambda: None)
    monkeypatch.setitem(vllm_plugin._MACHETE, "op", "the plugin's op")
    monkeypatch.setattr(serve, "_op", lambda: (lambda *a: log.append(("check_op", len(a)))))
    return K, shipped


class Out:
    """A stand-in output: y + bias records the add."""
    def __init__(self, log):
        self.log = log

    def reshape(self, *a):
        return self

    @property
    def shape(self):
        return (2, 3)

    def __add__(self, b):
        self.log.append(("+", None))
        return self


def test_fused_runs_the_plugins_own_patch(diag, monkeypatch):
    log = []
    K, shipped = _fake_vllm(monkeypatch, log)
    diag._install("fused")
    assert K.apply_weights is shipped
    from sdcguard import vllm_plugin
    assert vllm_plugin._MACHETE["op"] == "the plugin's op"


def test_separate_runs_vllms_apply_the_check_op_then_adds_the_bias(diag, monkeypatch):
    log = []
    K, shipped = _fake_vllm(monkeypatch, log)
    diag._install("separate")
    assert K.apply_weights is not shipped
    layer = types.SimpleNamespace(_sdcguard_args=(("a",) * 19, None))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    K.apply_weights(K(), layer, Out(log), "bias")
    assert log == [("vllm", False), ("check_op", 21), ("+", None)]
    log.clear()
    K.apply_weights(K(), types.SimpleNamespace(), Out(log), "bias")      # a layer sdcguard doesn't check
    assert log == [("vllm", True)]


def test_a_missing_machete_op_stops_the_run(diag, monkeypatch):
    log = []
    _fake_vllm(monkeypatch, log)
    from sdcguard import vllm_plugin
    monkeypatch.setitem(vllm_plugin._MACHETE, "op", None)
    with pytest.raises(SystemExit, match="didn't set up its Machete op"):
        diag._install("fused")


def test_the_stand_in_ops_register_with_the_arguments_they_say(diag):
    """fused_renamed's op: the plugin's arguments under round 2's name; fused_slots': named machete_mm, with the
    slot and the check's other arguments (round 2's); both declared as mutating nothing."""
    from sdcguard import vllm_plugin
    shipped = vllm_plugin._machete_op().default._schema
    assert shipped.name == "sdcguard::machete_mm" and shipped.arguments[-1].name == "checksums"
    renamed = diag._diag_op("machete_checked", with_slots=False)
    assert diag._diag_op("machete_checked", with_slots=False) is renamed
    assert renamed.default._schema.name == "sdcguard_diag::machete_checked"
    assert [a.name for a in renamed.default._schema.arguments] == [a.name for a in shipped.arguments]
    slots = diag._diag_op("machete_mm", with_slots=True)
    names = [a.name for a in slots.default._schema.arguments]
    assert slots.default._schema.name == "sdcguard_diag::machete_mm"
    assert names[:6] == [a.name for a in shipped.arguments][:6] and names[6] == "parts" and "slot" in names
    assert "state" not in names and "cap_meta" not in names
    for op in (renamed, slots, vllm_plugin._machete_op()):
        assert not any(a.alias_info is not None and a.alias_info.is_write for a in op.default._schema.arguments)


def test_fused_renamed_swaps_the_op_and_wrap_only_runs_vllms_own(diag, monkeypatch):
    log = []
    K, shipped = _fake_vllm(monkeypatch, log)
    monkeypatch.setattr(diag, "_diag_op", lambda name, with_slots: f"op {name} {with_slots}")
    diag._install("fused_renamed")
    from sdcguard import vllm_plugin
    assert K.apply_weights is shipped and vllm_plugin._MACHETE["op"] == "op machete_checked False"
    log.clear()
    K2, shipped2 = _fake_vllm(monkeypatch, log)
    diag._install("wrap_only")
    layer = types.SimpleNamespace(_sdcguard_args=(("a",) * 19, None))
    K2.apply_weights(K2(), layer, Out(log), "bias")
    assert log == [("vllm", True)]                            # registered layers too: vLLM's own, no check


def test_fused_slots_passes_the_slot_and_the_checks_constants(diag, monkeypatch):
    log, got = [], []
    K, shipped = _fake_vllm(monkeypatch, log)
    monkeypatch.setattr(diag, "_diag_op", lambda name, with_slots: (lambda *a: got.append(a) or Out(log)))
    diag._install("fused_slots")
    assert K.apply_weights is not shipped
    args = tuple(f"t{i}" for i in range(11)) + ("state", 7, 6.0, 1e-20, "cap_meta", "cap_data", 128, 16)
    layer = types.SimpleNamespace(_sdcguard_args=(args, None))
    kern = K()
    kern.config = types.SimpleNamespace(partition_weight_shape=(4, 3), zero_points=False,
                                        weight_type=types.SimpleNamespace(id=101), group_size=64)
    kern._get_weight_params = lambda layer: ("w_q", "w_s", "w_zp")

    class X:
        shape = (2, 4)

        def reshape(self, *a):
            return self
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    K.apply_weights(kern, layer, X(), None)
    assert got[0][1:6] == ("w_q", 101, "w_s", None, 64)
    assert got[0][6:] == tuple(f"t{i}" for i in range(11)) + (7, 6.0, 1e-20, 128, 16)


@pytest.mark.parametrize("codes,exit", [
    ({}, 0), ({"fused_renamed": 1, "baseline_fresh": 124, "fused_fresh": 1}, 0),
    ({"fused": 1}, 1), ({"baseline": 1}, 1),
])
def test_the_diagnostic_fails_only_when_a_run_the_decision_needs_failed(diag, monkeypatch, tmp_path, codes, exit):
    ran, seen = [], []

    def fake_run(cmd, timeout):
        v = cmd[cmd.index("--variant") + 1]
        ran.append(v)
        seen.append((cmd[cmd.index("--model") + 1], cmd[cmd.index("--prompts") + 1]))
        return types.SimpleNamespace(returncode=codes.get(v, 0))
    monkeypatch.setattr(diag.subprocess, "run", fake_run)
    monkeypatch.setattr(diag, "compare", lambda out, runs: {"fix": None, "runs_compared": runs})
    monkeypatch.setattr(diag, "LOCAL", str(tmp_path / "local"))
    monkeypatch.setattr(sys, "argv", ["machete_diag.py", "--out", str(tmp_path), "--model", "m/7b", "--prompts",
                                      "256"])
    assert diag.main() == exit
    assert ran == (["baseline"] if codes.get("baseline") else list(diag.RUNS))
    assert set(seen) == {("m/7b", "256")}
    d = json.loads((tmp_path / "DIAG.json").read_text())
    assert d["exit_codes"] == {v: codes.get(v, 0) for v in ran}
    if not codes.get("baseline"):
        assert d["runs_compared"] == list(diag.RUNS)


@pytest.mark.parametrize("runs", ["fused,baseline", "baseline,separate", "baseline,fused,nonsense", ""])
def test_a_run_list_must_start_with_baseline_and_include_fused(diag, monkeypatch, tmp_path, runs):
    monkeypatch.setattr(sys, "argv", ["machete_diag.py", "--out", str(tmp_path), "--runs", runs])
    monkeypatch.setattr(diag.subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(SystemExit):
        diag.main()


def test_one_run_end_to_end_with_a_stand_in_vllm(diag, monkeypatch, tmp_path):
    """run(): fresh-cache variants get their own Inductor cache, the trace is copied out, the result is written.
    (Every setting it changes is put back after the test.)"""
    import torch._inductor.config as ic
    for k in ("TORCHINDUCTOR_CACHE_DIR", "SDCGUARD", "SDCGUARD_POLL_EVERY", "VLLM_ENABLE_V1_MULTIPROCESSING",
              "VLLM_ALLOW_INSECURE_SERIALIZATION", "VLLM_DISABLE_COMPILE_CACHE", "VLLM_USE_FLASHINFER_SAMPLER"):
        monkeypatch.setenv(k, "unset-by-test")
    for k in ("enabled", "debug_dir", "fx_graph", "fx_graph_transformed", "ir_pre_fusion", "ir_post_fusion",
              "output_code", "graph_diagram", "draw_orig_fx_graph"):
        monkeypatch.setattr(ic.trace, k, getattr(ic.trace, k))
    m = torch.backends.cuda.matmul
    monkeypatch.setattr(m, "allow_bf16_reduced_precision_reduction", m.allow_bf16_reduced_precision_reduction)
    monkeypatch.setattr(m, "allow_fp16_reduced_precision_reduction", m.allow_fp16_reduced_precision_reduction)
    monkeypatch.setattr(diag, "LOCAL", str(tmp_path / "local"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    seen = {}

    class LLM:
        def __init__(self, **kw):
            seen["cache"], seen["trace"] = os.environ["TORCHINDUCTOR_CACHE_DIR"], ic.trace.debug_dir
            seen["kw"] = kw
            d = Path(ic.trace.debug_dir) / "torchinductor" / "model__0_inference_0.0"
            d.mkdir(parents=True)
            (d / "output_code.py").write_text(TRITON)
            c = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "ab"
            c.mkdir(parents=True)
            (c / "ck.best_config").write_text(json.dumps({"XBLOCK": 32}))

        def generate(self, prompts, sp, use_tqdm=False):
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=p[-5:])]) for p in prompts]

    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams, vllm.__version__ = LLM, lambda **kw: kw, "0.31.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    out = tmp_path / "out"
    assert diag.run("baseline_fresh", str(out), "some/model") == 0
    res = json.loads((out / "baseline_fresh.json").read_text())
    assert len(res["outputs"]) == 64 and len(res["light_outputs"]) == 16 and res["inductor_cache"] == "its own"
    assert res["autotuned"] == {"ck": {"XBLOCK": 32}} and "checks" not in res
    assert res["vllm"] == "0.31.0" and res["torch"] == torch.__version__
    assert seen["cache"].endswith("inductor_baseline_fresh") and seen["kw"]["model"] == "some/model"
    assert ic.trace.enabled and ic.trace.ir_post_fusion and ic.trace.output_code and not ic.trace.fx_graph
    assert (out / "code" / "baseline_fresh" / "torchinductor" / "model__0_inference_0.0" / "output_code.py").exists()
    assert os.environ["SDCGUARD"] == "0" and os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert diag.run("baseline", str(out), "some/model") == 0
    assert seen["cache"].endswith("inductor_shared")
