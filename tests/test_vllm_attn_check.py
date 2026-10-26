"""experiments/vllm_attn_check.py without a GPU: its criteria (docs/METHOD.md, "Attention checks in serving") on
made-up results, and its long prompts."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import vllm_attn_check as va  # noqa: E402

NAMES = [f"model.layers.{i}.self_attn.attn" for i in range(28)]


def _stats(impl="FlashAttentionImpl", worst=0.3, steps=500):
    return {"layers": 28, "names": NAMES, "kernel": {"cuda:0 bfloat16": [True, ""]}, "backends": [impl],
            "worst": worst, "steps": steps, "calls_not_checked": {}, "not_checked": {},
            "stats": {"calls_checked": steps, "samples": 4 * steps, "keys": 300000, "writes_checked": 4 * steps,
                      "writes_wrong": 0, "slots_wrong": 0, "bad_blocks": 0, "unjudged": 0}}


def passing():
    ident, long = ["a"] * va.IDENTITY, ["b"] * va.LONG
    speed = lambda v: {"16": {"median": {"output_tokens_per_s": v}, "complete": True}}  # noqa: E731
    hit = {"value": "output_wrong_or_misread", "head": "output_wrong_or_misread", "other": "output_wrong_or_misread",
           "kv": "kv_write_wrong", "short": "output_wrong_or_misread"}
    fp32 = {"bf16": False, "fp16": False}               # cuBLAS partial sums in fp32 on both sides of A2
    return {"roles": list(va.ROLES),
            "plain": {"ready": True, "reductions": fp32, "identity": ident, "long": long, "speed": speed(1000.0)},
            "guarded": {"ready": True, "reductions": fp32, "identity": ident, "long": long,
                        "attn_at_start": _stats(), "attn_clean": _stats(), "clean_alerts": 0, "speed": speed(980.0)},
            "noattn": {"ready": True, "clean_alerts": 0, "speed": speed(1000.0)},
            "eager": {"ready": True, "attn_at_start": _stats(), "attn_clean": _stats(steps=40), "clean_alerts": 0,
                      "faults": [{"mode": m, "layer": NAMES[k], "mark": NAMES[k], "detected": True,
                                  "rounds_to_detect": 1, "findings": [hit[m]],
                                  "verdicts": ["output_wrong"] if m != "kv" else [], "alerts_other_layers": 0}
                                 for k, m in enumerate(va.MODES)]},
            "tplain": {"ready": True, "reductions": fp32, "identity": ident, "long": long},
            "tguarded": {"ready": True, "reductions": fp32, "identity": ident, "long": long,
                         "attn_at_start": _stats("TritonAttentionImpl"),
                         "attn_clean": _stats("TritonAttentionImpl"), "clean_alerts": 0}}


def test_a_run_that_meets_every_criterion_passes():
    ok, fails, s = va.judge(passing())
    assert ok, fails
    assert s["cost"]["16"]["attn_cost"] == 0.02 and s["on_by_default"] is True


def test_an_l4_run_of_plain_and_guarded_is_judged_on_those_alone():
    r = passing()
    r["roles"] = ["plain", "guarded"]
    for k in ("noattn", "eager", "tplain", "tguarded"):
        del r[k]
    ok, fails, _ = va.judge(r)
    assert ok, fails


def test_each_criterion_fails_on_its_own():
    cases = []
    r = passing()
    r["tguarded"]["ready"] = False
    cases.append((r, "A1: tguarded didn't start"))
    r = passing()
    r["guarded"]["attn_at_start"]["layers"] = 27
    cases.append((r, "A1 (guarded): 27 attention layers"))
    r = passing()
    r["eager"]["attn_at_start"]["kernel"] = {"cuda:0 bfloat16": [False, "counts differ"]}
    cases.append((r, "A1 (eager): the Triton kernels"))
    r = passing()
    r["tguarded"]["attn_at_start"]["backends"] = ["FlashAttentionImpl"]
    cases.append((r, "A1 (tguarded): backends"))
    r = passing()
    r["guarded"]["long"] = ["c"] * va.LONG
    cases.append((r, "A2: guarded's long completions differ"))
    r = passing()
    r["tguarded"]["identity"] = ["c"] * va.IDENTITY
    cases.append((r, "A2: tguarded's identity"))
    r = passing()
    r["plain"]["reductions"] = {"bf16": True, "fp16": True}         # plain on PyTorch's default: not comparable
    cases.append((r, "A2: plain didn't keep cuBLAS partial sums in fp32"))
    r = passing()
    r["tplain"]["reductions"] = {"error": "HTTP 500"}
    cases.append((r, "A2: tplain didn't keep"))
    r = passing()
    r["noattn"]["clean_alerts"] = 2
    cases.append((r, "A3 (noattn): 2 alerts"))
    r = passing()
    r["guarded"]["attn_clean"]["worst"] = 1.3
    cases.append((r, "above the bound"))
    r = passing()
    r["tguarded"]["attn_clean"]["stats"]["slots_wrong"] = 1
    cases.append((r, "A3 (tguarded): 1 slots wrong"))
    r = passing()
    r["guarded"]["attn_clean"]["calls_not_checked"] = {"fused output quantization": 3}
    cases.append((r, "calls not checked"))
    r = passing()
    r["guarded"]["attn_clean"]["stats"]["calls_checked"] = 100        # decode steps in CUDA graphs not checked
    cases.append((r, "A6 (guarded): 100 calls checked in 500 model steps"))
    r = passing()
    r["eager"]["faults"][1]["detected"] = False
    cases.append((r, "A4: faults caught in 4 of 5"))
    r = passing()
    r["eager"]["faults"][3]["findings"] = ["output_wrong_or_misread"]       # the KV fault seen only downstream
    cases.append((r, "missed: [('kv'"))
    r = passing()
    r["eager"]["faults"][0]["alerts_other_layers"] = 1
    r["eager"]["faults"][0]["other_layers"] = [NAMES[9]]
    cases.append((r, "A4: alerts on other layers"))
    r = passing()
    r["eager"]["faults"][2]["verdicts"] = ["not_rechecked"]
    cases.append((r, "A5"))
    for res, want in cases:
        ok, fails, _ = va.judge(res)
        assert not ok and any(want in f for f in fails), (want, fails)


def test_a_missed_short_fault_is_reported_not_failed():
    r = passing()
    r["eager"]["faults"][4].update(detected=False, findings=[], verdicts=[])
    ok, fails, s = va.judge(r)
    assert ok, fails
    assert ("short", va.NAMES[4] if hasattr(va, "NAMES") else NAMES[4], False) == s["faults"][4][:3]


def test_the_long_prompts_share_a_long_prefix_and_differ_after_it():
    ps = va.long_prompts()
    assert len(ps) == va.LONG and len(set(ps)) == va.LONG
    common = os.path.commonprefix(ps)
    assert len(common) > 2800 and all(len(p) > len(common) + 400 for p in ps)    # 1,024 tokens shared
    assert max(len(p) for p in ps) < 9000                             # at most 3,295 tokens: under 4,096 with
    #                                                                   the answer (measured with Qwen's tokenizer)


def test_a4_counts_every_alert_naming_another_layer_unless_its_score_isnt_finite(monkeypatch):
    """The trials' stray alerts (A4: "no alert with a finite score names another layer"): from any check, with a
    finite score or none (a wrong KV write has none); one whose score isn't finite is left out (a non-finite value
    the fault made reaches later layers' inputs)."""
    log = []
    target = NAMES[4]
    stray = [{"detector": "serve", "detail": {"where": "model.layers.3.mlp.down_proj", "max_z": 9.5}},
             {"detector": "serve_attn", "detail": {"where": NAMES[7], "finding": "kv_write_wrong"}},
             {"detector": "serve_norm", "detail": {"where": "model.layers.5.input_layernorm", "max_z": float("inf")}},
             {"detector": "serve_attn", "detail": {"where": NAMES[6], "max_z": float("nan")}}]

    def rpc(port, method, args=()):
        if method == "sdcguard_test_corrupt_attn" and args[0] == target:
            log.append({"detector": "serve_attn", "detail": {"where": target, "max_z": 50.0,
                                                             "finding": "output_wrong_or_misread"}})
            log.extend(stray)
        return args[0] if args else True
    monkeypatch.setattr(va.vn, "_try", lambda *a, **k: {"text": ""})
    monkeypatch.setattr(va.vn, "rpc", rpc)
    monkeypatch.setattr(va.vn, "alerts", lambda path: list(log))
    monkeypatch.setattr(va.time, "sleep", lambda s: None)
    rows = va.fault_trials(0, "log", NAMES, ["p1", "p2"])
    first = rows[0]
    assert first["layer"] == target and first["detected"] and first["rounds_to_detect"] == 1
    assert first["alerts_other_layers"] == 2
    assert first["other_layers"] == ["serve:model.layers.3.mlp.down_proj", f"serve_attn:{NAMES[7]}"]
    assert all(not t["detected"] and t["alerts_other_layers"] == 0 for t in rows[1:])


def test_plain_servers_run_with_the_plugins_cublas_setting(monkeypatch, tmp_path):
    """A2 compares a plain server with a guarded one: the plain one keeps bf16/fp16 cuBLAS partial sums in fp32, as
    the plugin does (set in vLLM's worker when it imports the test extension), so only sdcguard's checks differ."""
    import subprocess
    seen = {}

    class Proc:
        def __init__(self, cmd, env=None, **kw):
            seen[env.get("SDCGUARD")] = env
    with monkeypatch.context() as mp:
        mp.setattr(va.subprocess, "Popen", Proc)
        for role in ("plain", "guarded"):
            _, log = va.server(role, str(tmp_path), 8000, 9000)
            log.close()
    assert seen["0"]["SDCGUARD_TEST_EXACT"] == "1" and "SDCGUARD_TEST_EXACT" not in seen["1"]
    code = ("import torch, sdcguard_test_ext as e; m = torch.backends.cuda.matmul; "
            "print(m.allow_bf16_reduced_precision_reduction, m.allow_fp16_reduced_precision_reduction, "
            "e.FaultInjection().sdcguard_test_reductions())")
    exp = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments")
    for flag, want in (("1", "False False {'bf16': False, 'fp16': False}"),
                       ("0", "True True {'bf16': True, 'fp16': True}")):
        env = dict(os.environ, PYTHONPATH=exp, SDCGUARD_TEST_EXACT=flag)
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
        assert out.stdout.strip() == want, out.stdout + out.stderr


def test_an_incomplete_speed_run_decides_nothing():
    r = passing()
    r["guarded"]["speed"]["16"]["complete"] = False      # a round with failed requests
    ok, fails, s = va.judge(r)
    assert s["on_by_default"] is None and s["cost"]["16"]["complete"] is False


def test_a_failed_run_is_never_recommended_on_by_default():
    r = passing()
    r["guarded"]["clean_alerts"] = 3                        # a criterion missed, the cost still small
    ok, fails, s = va.judge(r)
    assert not ok and s["on_by_default"] is False
