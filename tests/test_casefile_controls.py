"""Which replays a casefile may count: replays of this evidence file only, each once, run the way the incident ran.

The case these protect is the one sdcguard has seen for real: cuBLAS known issue 6862950 made an H200 compute
wrong at an MPS share of 40% and right at 100%. A clean replay at another share, on the same GPU or a peer, says
nothing about a fault that needs the original share."""
from __future__ import annotations

import json
import os

import torch

from sdcguard import casefile as cf
from sdcguard.check import gemm_effective, gemm_settings, gemm_settings_applied

from test_casefile import MATMUL, SHA, STACK, VERSIONS, incident, outcome, rep

H200_40 = dict(STACK, gpu_name="NVIDIA H200", environment={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"},
               sms_visible=52, cublas="13.2.0.9")
H200_FULL = dict(H200_40, environment={}, sms_visible=132)


def test_clean_replays_on_the_same_gpu_at_another_mps_share_dont_make_it_transient():
    o, st, a = outcome(incident(stack=H200_40), [rep("GPU-orig", stack=H200_FULL)])
    assert o == cf.INCONCLUSIVE and st["transient"] == cf.NOT_TESTED
    rows = a["observations"]["same-GPU replays with other software or settings (not counted)"]
    assert rows and rows[0]["judged"] == 200 and any("sms_visible 52 vs 132" in d for d in rows[0]["settings_differ"])
    assert any("ran with other software or settings" in n and "MPS share" in n for n in a["next"])
    # the same replays run at the incident's share do count
    o, st, _ = outcome(incident(stack=H200_40), [rep("GPU-orig", stack=H200_40)])
    assert st["transient"] == cf.SUPPORTED


def test_a_wrong_replay_on_the_same_gpu_contradicts_transient_whatever_it_ran_with():
    _, st, a = outcome(incident(stack=H200_40), [rep("GPU-orig", failed=3, stack=H200_FULL)])
    assert st["transient"] == cf.CONTRADICTED
    assert st["recurring"] == cf.NOT_TESTED                  # but it isn't a recurrence under the incident's settings
    assert "under other settings were wrong" in a["explanations"]["recurring"]["because"][0]


def test_settings_missing_on_both_sides_are_unknown_not_a_match():
    legacy = dict(VERSIONS)                                   # recorded versions, no execution settings
    assert cf.same_stack(legacy, dict(legacy)) is None
    o, st, a = outcome(incident(stack=legacy), [rep("GPU-orig", failed=200, stack=legacy),
                                                rep("GPU-peer", stack=legacy)])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    assert any("weren't recorded" in n for n in a["next"])
    no_sms = {k: v for k, v in STACK.items() if k != "sms_visible"}
    assert cf.same_stack(no_sms, dict(no_sms)) is None


def test_recurring_needs_every_setting_to_match():
    o, st, _ = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer")])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["recurring"] and st["recurring"] == cf.SUPPORTED
    tf32 = dict(STACK, matmul=dict(MATMUL, allow_tf32=True))
    o, st, a = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", stack=tf32)])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    assert any("matmul tf32 False vs True" in n for n in a["next"])


def test_tf32_set_through_either_api_matches():
    old_api = dict(MATMUL, fp32_precision="none", allow_tf32=True)
    new_api = dict(MATMUL, fp32_precision="tf32", allow_tf32=True)
    assert gemm_effective(old_api) == gemm_effective(new_api)
    assert cf.same_stack(dict(STACK, matmul=old_api), dict(STACK, matmul=new_api)) is True
    assert gemm_effective(dict(MATMUL, fp32_precision="ieee", allow_tf32=True))["tf32"] is False


def test_replays_of_another_evidence_file_dont_count():
    o, st, a = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", sha="b" * 64)])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    assert a["observations"]["replays of another evidence file, or not saying which (not counted)"] == 1
    assert any("another evidence file" in n for n in a["next"])
    old = rep("GPU-peer")
    del old["evidence_sha256"]                                 # a replay result from before 0.4.0
    o, _, _ = outcome(incident(), [rep("GPU-orig", failed=200), old])
    assert o == cf.INCONCLUSIVE
    o, _, a = outcome(incident(sha=None), [rep("GPU-orig", failed=200), rep("GPU-peer")])
    assert o == cf.INCONCLUSIVE and any("couldn't be read" in n for n in a["next"])


def test_the_same_replay_passed_twice_counts_once():
    few = rep("GPU-orig", n=12)
    o1, _, _ = outcome(incident(), [few])
    o2, _, a = outcome(incident(), [few, dict(few)])
    assert o1 == o2 == cf.INCONCLUSIVE                          # 12 judged is too few, however often it's passed
    assert a["observations"]["the same replay result passed more than once (counted once)"] == 1
    assert a["observations"]["same-GPU replays"]["judged"] == 12
    legacy = {k: v for k, v in few.items() if k != "replay_id"}
    _, _, a = outcome(incident(), [legacy, dict(legacy)])
    assert a["observations"]["same-GPU replays"]["judged"] == 12


def test_a_peer_replayed_by_other_sdcguard_code_is_not_a_matched_control():
    other = dict(STACK, sdcguard="0.4.1")
    o, st, a = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", stack=other)])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    peer = a["observations"]["peer-GPU replays"][0]
    assert peer["same_stack"] is False and any("sdcguard 0.4.1" in d for d in peer["settings_differ"])


def test_the_incident_setup_is_matched_against_nvidia_known_issues():
    leads = cf.known_issue_leads(dict(H200_40, cublaslt_loaded="13.2.0"))
    hit = [m for m in leads if m["id"] == "6862950"]
    assert hit and hit[0]["status"] == "applies"
    assert not [m for m in cf.known_issue_leads(dict(H200_FULL, cublaslt_loaded="13.2.0")) if m["id"] == "6862950"]
    _, _, a = outcome(incident(stack=dict(H200_40, cublaslt_loaded="13.2.0")), [])
    assert any("known issue 6862950" in n for n in a["next"])
    assert "not a signed record to quarantine on" in a["note"]


def _evidence(tmp_path, gemm=None):
    A = torch.randn(8, 256, dtype=torch.float16)
    B = torch.randn(256, 128, dtype=torch.float16)
    ev = {"A": A, "B": B, "C": (A.float() @ B.float()).half(), "bias": None, "alpha": 1.0, "beta": 1.0,
          "op": "aten::mm", "where": "test", "row": None, "col_block": None, "batch": None, "epoch": 0,
          "sign_seed": 1, "gemm_settings": gemm}
    path = str(tmp_path / "evidence_0.pt")
    torch.save(ev, path)
    return path


def test_replay_names_the_exact_file_it_replayed_and_gets_a_new_id(tmp_path):
    import hashlib
    from sdcguard.screen import replay
    path = _evidence(tmp_path)
    r1, r2 = replay(path, 2, "cpu"), replay(path, 2, "cpu")
    assert r1["evidence_sha256"] == hashlib.sha256(open(path, "rb").read()).hexdigest() == r2["evidence_sha256"]
    assert r1["replay_id"] != r2["replay_id"] and r1["replayed_at"]
    assert r1["software"]["sdcguard"] and "matmul" in r1["software"]
    case = cf.build(path, [r1, r2])
    assert case["assessment"]["observations"]["replays of another evidence file, or not saying which (not counted)"] \
        == 0


def test_replay_runs_with_the_recorded_settings_and_records_them(tmp_path):
    from sdcguard.screen import replay
    before = gemm_settings()
    recorded = dict(before, allow_fp16_accumulation=not before.get("allow_fp16_accumulation", False),
                    allow_tf32=not before.get("allow_tf32", False), fp32_precision="none")
    r = replay(_evidence(tmp_path, recorded), 1, "cpu")
    assert r["gemm_settings"]["source"] == "recorded with the evidence"
    ran = gemm_effective(r["software"]["matmul"])
    assert ran["allow_fp16_accumulation"] == recorded["allow_fp16_accumulation"]
    assert ran["tf32"] == recorded["allow_tf32"]
    assert gemm_settings() == before                           # and put back afterwards
    (tmp_path / "older").mkdir()
    r = replay(_evidence(tmp_path / "older", None), 1, "cpu")
    assert r["gemm_settings"]["source"].startswith("this process's")


def test_applied_settings_are_put_back_even_on_an_error():
    before = gemm_settings()
    try:
        with gemm_settings_applied(dict(before, allow_fp16_accumulation=True, allow_tf32=True)):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gemm_settings() == before


def test_evidence_records_the_settings_of_the_checked_step(tmp_path):
    """The read-back (and the evidence it saves) comes after the step; what's recorded is what the step ran with,
    and the alarm line is the guard's own."""
    import torch.nn as nn
    from sdcguard import FaultInjector, SDCGuard
    from sdcguard.inject import fault_model
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
    g = SDCGuard(evidence_dir=str(tmp_path / "ev"), max_evidence=1, z=7.5).protect(m)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=4).arm_on([m[0]])
    mm = torch.backends.cuda.matmul
    saved = mm.allow_fp16_accumulation
    x = torch.randn(4, 16, 256)
    try:
        mm.allow_fp16_accumulation = True
        for s in range(3):
            inj.step = s
            with inj:
                with g.at_step(s):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        m(x).float().pow(2).mean().backward()
                    mm.allow_fp16_accumulation = False          # changed before the step's read-back
            mm.allow_fp16_accumulation = True
    finally:
        mm.allow_fp16_accumulation = saved
        inj.remove_hooks()
        g.unprotect()
    assert g.evidence, "the injected fault should have been saved"
    ev = g.evidence[0]
    info = json.load(open(ev[:-3] + ".json"))
    assert info["software"]["matmul"]["allow_fp16_accumulation"] is True
    assert info["alarm_line"] == 7.5
    assert torch.load(ev, weights_only=True)["gemm_settings"]["allow_fp16_accumulation"] is True
    assert not [f for f in os.listdir(tmp_path / "ev") if f.endswith(".tmp")]


def test_two_writers_never_share_an_evidence_name(tmp_path):
    from sdcguard.screen import _claim
    (tmp_path / "evidence_0.pt").write_bytes(b"")
    a = _claim(str(tmp_path), "evidence", 0)
    b = _claim(str(tmp_path), "evidence", 0)
    assert a.endswith("evidence_1") and b.endswith("evidence_2")


def _corrupting(times=None):
    """A dispatch mode standing in for a GPU that computes one output of every bf16 matmul wrong (or only the
    first `times` of them)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class Corrupt(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.left = times

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default) and out.dtype == torch.bfloat16:
                if self.left is None or self.left > 0:
                    out[0, 0] += 100.0
                    if self.left is not None:
                        self.left -= 1
            return out
    return Corrupt()


def _guarded_matmul(tmp_path, mode, **kw):
    from sdcguard import SDCGuard
    torch.manual_seed(0)
    A, B = torch.randn(8, 256).bfloat16(), torch.randn(256, 128).bfloat16()
    g = SDCGuard(evidence_dir=str(tmp_path / "ev"), max_evidence=1, **kw)
    with mode:
        with g.at_step(0):
            A @ B
    return g


def test_a_failure_that_repeats_right_away_says_so_in_the_alert(tmp_path):
    g = _guarded_matmul(tmp_path, _corrupting())
    c = g.log.events[0].detail["confirmed_in_job"]
    assert c["recomputations"] == 3 and c["recomputations_that_failed_the_check"] == 3
    assert "came back in 3 of 3" in c["reading"]
    case = cf.build(g.evidence[0], [])
    inj = [r for r in case["replays"] if r.get("in_job")]
    assert len(inj) == 1 and inj[0]["evidence_sha256"] == case["incident"]["evidence_sha256"]
    assert inj[0]["software"] == json.load(open(g.evidence[0][:-3] + ".json"))["software"]


def test_a_failure_that_doesnt_repeat_says_that_too(tmp_path):
    g = _guarded_matmul(tmp_path, _corrupting(times=1))
    c = g.log.events[0].detail["confirmed_in_job"]
    assert c["recomputations_that_failed_the_check"] == 0 and "none of 3" in c["reading"]


def test_confirm_zero_recomputes_nothing(tmp_path):
    g = _guarded_matmul(tmp_path, _corrupting(), confirm=0)
    assert "confirmed_in_job" not in g.log.events[0].detail
    assert "in_job_replays" not in json.load(open(g.evidence[0][:-3] + ".json"))


def test_the_confirmation_runs_with_the_steps_settings(tmp_path, monkeypatch):
    from sdcguard import screen
    seen = {}
    real = screen._replay

    def spy(*a, **k):
        seen.update(gemm_settings())
        return real(*a, **k)
    monkeypatch.setattr(screen, "_replay", spy)
    m = torch.backends.cuda.matmul
    saved = m.allow_fp16_accumulation
    from sdcguard import SDCGuard
    torch.manual_seed(0)
    A, B = torch.randn(8, 256).bfloat16(), torch.randn(256, 128).bfloat16()
    g = SDCGuard(evidence_dir=str(tmp_path / "ev"), max_evidence=1)
    try:
        m.allow_fp16_accumulation = True
        with _corrupting():
            with g.at_step(0):
                A @ B
                m.allow_fp16_accumulation = False                    # changed before the read-back
    finally:
        m.allow_fp16_accumulation = saved
    assert seen["allow_fp16_accumulation"] is True and gemm_settings()["allow_fp16_accumulation"] == saved


def test_replays_bound_how_often_the_error_comes_back():
    """Clopper-Pearson, two-sided 95%: the textbook values."""
    assert cf.rate_interval(0, 200) == [0.0, 0.018275]           # 1 - 0.025 ** (1 / 200)
    assert cf.rate_interval(1, 10) == [0.002529, 0.445016]
    assert cf.rate_interval(5, 100) == [0.016432, 0.112835]
    assert cf.rate_interval(3, 3)[1] == 1.0 and cf.rate_interval(0, 0) is None
    _, _, a = outcome(incident(), [rep("GPU-orig", n=200), rep("GPU-peer", n=50, failed=2)])
    assert a["observations"]["same-GPU replays"]["error_rate_95"] == [0.0, 0.018275]
    assert a["observations"]["peer-GPU replays"][0]["error_rate_95"][0] > 0
