"""The incident casefile (casefile.py): evidence-based explanations, INCONCLUSIVE when evidence is missing or mixed."""
from __future__ import annotations

import json
import os

import pytest
import torch

from sdcguard import casefile as cf

# what sdcguard 0.4.0 records: versions, and the settings that change which kernels run
VERSIONS = {"sdcguard": "0.4.0", "torch": "2.14.1", "cuda": "13.0", "cublas": "13.1.1.3", "driver": "610.57.04",
            "gpu_name": "NVIDIA H100"}
MATMUL = {"fp32_precision": "none", "allow_tf32": False, "allow_bf16_reduced_precision_reduction": False,
          "allow_bf16_reduced_precision_reduction_split_k": True, "allow_fp16_reduced_precision_reduction": False,
          "allow_fp16_reduced_precision_reduction_split_k": True, "allow_fp16_accumulation": False,
          "prefer_cublaslt_grouped_gemm": False, "blas_library": "cublas"}
STACK = dict(VERSIONS, environment={}, sms_visible=132, compute_capability=[9, 0], matmul=MATMUL)
SHA = "a" * 64
_IDS = iter(range(10 ** 9))


def incident(cpu="outside envelope", stack=STACK, sha=SHA):
    c = {"outside envelope": {"done": True, "judged": True, "output_wrong": True, "output_mismatch_confirmed": True},
         "statistical": {"done": True, "judged": True, "output_wrong": True, "output_mismatch_confirmed": False},
         "not reproduced": {"done": True, "judged": True, "output_wrong": False, "output_mismatch_confirmed": False},
         "unknown": {"done": False}}[cpu]
    return {"evidence": "evidence_0.pt", "evidence_sha256": sha, "gpu_uuid": "GPU-orig", "stack": dict(stack),
            "cpu": c}


def rep(uuid, n=200, failed=0, stack=STACK, unjudged=0, signs="recorded", sha=SHA):
    return {"replayed_on": {"device": "cuda:0", "gpu_uuid": uuid}, "recomputations": n,
            "recomputations_that_failed_the_check": failed, "recomputations_that_couldnt_be_judged": unjudged,
            "software": dict(stack), "sign_patterns": signs, "evidence_sha256": sha, "replay_id": f"r{next(_IDS)}"}


def outcome(inc, reps):
    a = cf.assess(inc, reps)
    return a["outcome"], {k: v["status"] for k, v in a["explanations"].items()}, a


def test_transient_on_the_original_gpu():
    o, st, a = outcome(incident(), [rep("GPU-orig"), rep("GPU-peer")])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["transient"]
    assert st == {"transient": cf.SUPPORTED, "recurring": cf.CONTRADICTED, "software": cf.NOT_TESTED,
                  "check": cf.CONTRADICTED}
    assert a["explanations"]["transient"]["also_consistent_with"]          # alternatives stay listed


def test_recurring_needs_a_peer_with_the_same_software():
    o, st, _ = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer")])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["recurring"] and st["software"] == cf.CONTRADICTED
    other = dict(STACK, cublas="12.8.3.14")
    o2, st2, a2 = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", stack=other)])
    assert o2 == cf.INCONCLUSIVE and st2["recurring"] == cf.NOT_TESTED and a2["next"]


def test_software_when_a_peer_is_wrong_too():
    o, st, _ = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", failed=200)])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["software"]
    assert st["recurring"] == cf.CONTRADICTED and st["transient"] == cf.CONTRADICTED


def test_the_check_when_the_cpu_finds_the_output_fine():
    o, st, a = outcome(incident("not reproduced"), [rep("GPU-orig")])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["check"] and st["transient"] == cf.CONTRADICTED
    assert "own GPU arithmetic" in " ".join(a["explanations"]["check"]["because"])


@pytest.mark.parametrize("reps,why", [
    ([], "no replay at all"),
    ([rep("GPU-peer")], "no replay on the original GPU"),
    ([rep("GPU-orig", failed=3)], "recurs, but no peer to compare"),
    ([rep("GPU-orig"), rep("GPU-peer", failed=200)], "contradictory: right on the original, wrong on a peer"),
])
def test_inconclusive_when_evidence_is_missing_or_mixed(reps, why):
    o, _, a = outcome(incident(), reps)
    assert o == cf.INCONCLUSIVE, why
    assert a["next"] or why.startswith("contradictory")


def test_unrecorded_software_makes_peers_unmatchable():
    inc = incident(stack={"torch": "2.14.1", "gpu_name": "NVIDIA H100"})        # an older evidence file
    o, st, a = outcome(inc, [rep("GPU-orig", failed=200), rep("GPU-peer")])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    assert any("weren't recorded" in n for n in a["next"])


def test_few_replays_are_marked_weak_and_unjudged_runs_dont_count():
    _, _, a = outcome(incident(), [rep("GPU-orig", n=5), rep("GPU-peer")])
    assert any("only 5" in w for w in a["weak_evidence"])
    o, st, _ = outcome(incident(), [rep("GPU-orig", n=10, unjudged=10)])
    assert st["transient"] == cf.NOT_TESTED and o == cf.INCONCLUSIVE


def test_no_outcome_says_proven():
    for inc, reps in ((incident(), [rep("GPU-orig")]), (incident(), [rep("GPU-orig", failed=9), rep("GPU-p")])):
        a = cf.assess(inc, reps)
        assert "prove" not in json.dumps(a).lower().replace("not proven", "")


def _evidence(tmp_path):
    """A real evidence file: one output of a matmul corrupted far outside rounding, saved the way screen does."""
    from sdcguard import checksum as cs
    from sdcguard.screen import _save_evidence
    torch.manual_seed(0)
    A = torch.randn(64, 256).to(torch.bfloat16)
    B = torch.randn(256, 384).to(torch.bfloat16)
    C = (A.float() @ B.float()).to(torch.bfloat16)
    C[5, 130] += 40.0
    res = cs.check_matmul(A, B, C)
    assert not res.ok
    return _save_evidence(str(tmp_path), 0, res, (A, B, C, None, 1.0, 1.0, "aten::mm", "test layer"),
                          {"device": "cpu", "host": "test", "gpu_uuid": None}, {"torch": torch.__version__})


def test_end_to_end_on_cpu(tmp_path, capsys):
    path = _evidence(tmp_path)
    info = json.loads(open(path + ".json").read())
    assert info["software"]["torch"] == torch.__version__ and "cublas" in info["software"]
    code = cf.main([path + ".pt", "--replay-here", "5", "--device", "cpu", "--out", str(tmp_path / "case")])
    case = json.loads((tmp_path / "case" / "evidence_0_casefile.json").read_text())
    assert case["assessment"]["observations"]["recorded output vs CPU"] == "outside envelope"
    assert case["assessment"]["outcome"] == cf.INCONCLUSIVE and code == 2      # a CPU replay isn't the same GPU
    assert case["replays"][0]["recomputations"] == 5 and case["replays"][0]["software"]["gpu_name"] == "cpu"
    assert case["incident"]["evidence_sha256"] and "Outcome: INCONCLUSIVE" in capsys.readouterr().out
    assert (tmp_path / "case" / "evidence_0_casefile.md").exists()


def test_replay_save_brings_software_back(tmp_path):
    from sdcguard.cli import main
    path = _evidence(tmp_path)
    out = tmp_path / "peer.json"
    main(["replay", path + ".pt", "--times", "3", "--device", "cpu", "--save", str(out)])
    r = json.loads(out.read_text())
    assert r["recomputations"] == 3 and r["software"]["torch"] == torch.__version__
    case = cf.build(path + ".pt", [r])
    assert case["assessment"]["observations"]["replays that couldn't be placed"] == 0
    assert os.path.exists(path + ".pt")


def test_an_in_job_alert_saves_evidence_a_casefile_can_use(tmp_path):
    """SDCGuard(evidence_dir=...): the failed matmul's operands are saved at the alert, and the casefile finds the
    recorded output outside the CPU's rounding envelope."""
    import torch.nn as nn
    from sdcguard import FaultInjector, SDCGuard
    from sdcguard.inject import fault_model
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
    g = SDCGuard(evidence_dir=str(tmp_path / "ev"), max_evidence=2).protect(m)
    inj = FaultInjector(rate=1.0, fault_model=fault_model("nvidia", "warp"), seed=4).arm_on([m[0]])
    x = torch.randn(4, 16, 256)
    try:
        for s in range(3):
            inj.step = s
            with inj:
                with g.at_step(s):
                    with torch.autocast("cpu", dtype=torch.bfloat16):
                        m(x).float().pow(2).mean().backward()
    finally:
        inj.remove_hooks()
        g.unprotect()
    assert g.stats["alerts"] >= 1 and 1 <= len(g.evidence) <= 2
    ev = g.log.events[0].detail["evidence"]
    assert ev == g.evidence[0] and os.path.exists(ev) and os.path.exists(ev[:-3] + ".json")
    info = json.loads(open(ev[:-3] + ".json").read())
    assert info["source"] == "SDCGuard" and info["software"]["torch"] == torch.__version__
    case = cf.build(ev, [])
    assert case["assessment"]["observations"]["recorded output vs CPU"] in ("outside envelope", "statistical")
    assert case["assessment"]["outcome"] == cf.INCONCLUSIVE        # no replays yet: it says which to run
    assert any("--replay-here" in n for n in case["assessment"]["next"])


def test_no_evidence_dir_keeps_nothing():
    import torch.nn as nn
    from sdcguard import SDCGuard
    m = nn.Linear(64, 64)
    g = SDCGuard().protect(m)
    with g.at_step(0):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            m(torch.randn(4, 64)).sum().backward()
    g.unprotect()
    assert g._kept == [] and g.evidence == []


def test_a_statistical_mismatch_is_not_a_wrong_output():
    """Every element within rounding, but the checksum flags it: a small fault or a false alarm, never shown wrong."""
    # the same patterns flag the recomputation on the original GPU and on a peer: the check, not a fault
    o, st, _ = outcome(incident("statistical"), [rep("GPU-orig", failed=200), rep("GPU-peer", failed=200)])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["check"]
    assert st["software"] == cf.NOT_TESTED and st["transient"] == cf.CONTRADICTED
    # the same patterns pass on the original GPU: the recorded output differs from what it computes now
    o, st, a = outcome(incident("statistical"), [rep("GPU-orig")])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["transient"] and st["check"] == cf.CONTRADICTED
    assert "small" in " ".join(a["explanations"]["transient"]["because"])
    # replays judged with another process's patterns can't tell a patterns-bound false alarm from a fault
    o, st, _ = outcome(incident("statistical"), [rep("GPU-orig", signs="this process's")])
    assert o == cf.INCONCLUSIVE and st["transient"] == cf.NOT_TESTED
    # it recurs on the original GPU and nothing else was replayed: inconclusive, with a peer replay to run
    o, st, a = outcome(incident("statistical"), [rep("GPU-orig", failed=200)])
    assert o == cf.INCONCLUSIVE and st["check"] == cf.NOT_TESTED and any("another GPU" in n for n in a["next"])


def test_a_few_clean_replays_dont_decide_it():
    o, st, a = outcome(incident(), [rep("GPU-orig", n=5)])
    assert st["transient"] == cf.SUPPORTED and o == cf.INCONCLUSIVE
    assert any("too few" in n for n in a["next"])
    o, _, _ = outcome(incident(), [rep("GPU-orig", n=20)])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["transient"]
    o, _, _ = outcome(incident(), [rep("GPU-orig", failed=200), rep("GPU-peer", n=4)])
    assert o == cf.INCONCLUSIVE


def test_a_false_alarm_from_the_sign_patterns_comes_back_in_replays(tmp_path):
    """A correct matmul whose columns line up with the next check's +-1 pattern: the alert's own patterns flag
    every recomputation too, so the casefile doesn't call it a transient fault."""
    from sdcguard import SDCGuard, checksum as cs, screen
    torch.manual_seed(0)
    g = SDCGuard(evidence_dir=str(tmp_path), max_evidence=1)
    s = cs._signs(128, "cpu", g._epoch + 1)
    A = torch.randn(256, 512, dtype=torch.bfloat16)
    B = (torch.randn(512, 1) * s.view(1, 128)).to(torch.bfloat16)
    with g.at_step(0):
        C = A @ B
    assert torch.equal(C, (A.double() @ B.double()).to(torch.bfloat16))      # the output is right
    if not g.evidence:
        pytest.skip("the noise model didn't alarm on this data")
    inc = cf.load_incident(g.evidence[0])
    r = screen.replay(g.evidence[0], 10, "cpu")
    assert r["sign_patterns"] == "recorded" and r["recomputations_that_failed_the_check"] == 10
    inc["gpu_uuid"] = "GPU-same"
    r["replayed_on"] = {"device": "cuda:0", "gpu_uuid": "GPU-same"}
    a = cf.assess(inc, [r])
    assert cf.cpu_state(inc["cpu"]) != "outside envelope"
    assert a["explanations"]["transient"]["status"] != cf.SUPPORTED and a["outcome"] == cf.INCONCLUSIVE


def test_a_peer_with_another_mps_share_is_not_a_matched_control():
    """cuBLAS known issue 6862950 corrupts matmuls only below 100% of the SMs under MPS: an H200 wrong at 40% and
    right on a peer at 100% doesn't show a fault in the first GPU, even with identical versions."""
    at40 = dict(STACK, environment={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"}, sms_visible=52)
    at100 = dict(STACK, environment={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "100"}, sms_visible=132)
    assert cf.same_stack(at40, at100) is False and cf.same_stack(at40, dict(at40)) is True
    assert cf.same_stack(at40, VERSIONS) is None               # one side didn't record its settings
    o, st, a = outcome(incident(stack=at40), [rep("GPU-orig", failed=200, stack=at40), rep("GPU-peer", stack=at100)])
    assert o == cf.INCONCLUSIVE and st["recurring"] == cf.NOT_TESTED
    assert any("isn't a matched control" in n and "6862950" in n for n in a["next"])
    o, _, _ = outcome(incident(stack=at40), [rep("GPU-orig", failed=200, stack=at40), rep("GPU-peer", stack=at40)])
    assert o == "MOST CONSISTENT WITH: " + cf.EXPLANATIONS["recurring"]
