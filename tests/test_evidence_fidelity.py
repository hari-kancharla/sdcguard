"""Evidence that says what was checked, judged the way it was checked, and casefiles that don't reach past it:
the guard's own block width and alarm line, fp32 accumulators (precise mode), math-mode files without their
.json, a .json that belongs to another .pt, serving evidence recaptured after the alert, a step that raised
after its checks failed, a screen stopped at its deadline, and peers that didn't run the incident's way."""
from __future__ import annotations

import json
import os
import subprocess

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sdcguard import casefile as cf
from sdcguard import checksum as cs
from sdcguard import screen
from sdcguard.guard import SDCGuard

from test_casefile import STACK as BASE, incident as _incident, rep as _rep

AT40 = dict(BASE, gpu_name="NVIDIA H200", environment={"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "40"}, sms_visible=52)
AT100 = dict(AT40, environment={}, sms_visible=132)
MOST = "MOST CONSISTENT WITH: "


def incident(stack, cpu="outside envelope"):
    return _incident(cpu=cpu, stack=stack)


def rep(uuid, stack, n=200, failed=0):
    return _rep(uuid, n=n, failed=failed, stack=stack)


class _CorruptMM(TorchDispatchMode):
    """Faulty hardware: adds `delta` to one output element of every aten::mm (entered outside the guard)."""

    def __init__(self, row, col, delta=50.0):
        super().__init__()
        self.row, self.col, self.delta = row, col, delta

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is torch.ops.aten.mm.default and out.dtype == torch.bfloat16 and out.shape[0] > self.row \
                and out.shape[1] > self.col:
            out[self.row, self.col] += self.delta
        return out



@pytest.mark.parametrize("peer_stack", [AT40, dict(AT100, gpu_name="NVIDIA A100-SXM4-80GB", driver="550.54.15")])
def test_a_failed_peer_that_isnt_a_matched_control_doesnt_settle_software(peer_stack):
    """The incident ran at 100% of the SMs. A peer replayed under an MPS share (where cuBLAS known issue 6862950
    makes healthy GPUs compute wrong), or a peer of another model and driver, fails: that shows a problem in the
    peer's configuration, not the cause of the original incident, yet the outcome is 'software' with no caveat."""
    a = cf.assess(incident(AT100), [rep("GPU-peer", peer_stack, failed=200)])
    assert a["outcome"] == cf.INCONCLUSIVE, a["outcome"]



@pytest.mark.parametrize("fault", ["transient", "weight memory"])
def test_recaptured_serving_evidence_isnt_blamed_on_the_check(tmp_path, fault):
    """screen saves serving evidence from one more pass over the same tokens. A transient fault isn't in it, and a
    corrupted weight makes the recaptured output consistent with the corrupted weight: the CPU recheck finds the
    saved output fine and, with no replay at all, the casefile says MOST CONSISTENT WITH the check (a false alarm),
    so a real GPU fault is dismissed."""
    from sdcguard import FaultInjector, ServeGuard
    from sdcguard.inject import FaultModel
    torch.manual_seed(0)
    model = screen.ScreenModel(512, 2, 128, 4, 64).to(torch.bfloat16).eval()
    sg = ServeGuard().attach(model)
    x = torch.randint(0, 512, (2, 64))
    try:
        if fault == "transient":
            inj = FaultInjector(rate=1.0, fault_model=FaultModel(p_zero=1.0, p_bitflip=0.0, p_special=0.0),
                                seed=3).arm_on([model.blocks[0].up])
            with inj, torch.no_grad():
                model(x)
            inj.remove_hooks()
        else:
            with torch.no_grad():
                model.blocks[0].up.weight[:64, :] = 0          # a block of weight memory reads back as zeros
                model(x)
        alerts = sg.poll(wait=True)
        assert alerts, "the serving check didn't alarm"
        paths = screen._serving_evidence(model, x, alerts[:1], str(tmp_path), 0, 1,
                                         {"device": "cuda:0", "gpu_uuid": "GPU-orig"}, {"run_id": "t"})
    finally:
        sg.detach()
    assert paths
    a = cf.assess(cf.load_incident(paths[0] + ".pt"), [])
    assert a["outcome"] != MOST + cf.EXPLANATIONS["check"], a["outcome"]



def test_cpu_recheck_judges_the_block_the_guard_flagged(tmp_path):
    """SDCGuard(block=64) flags col_block 3 (columns 192..255). The evidence's CPU recheck (and replay) use 128-wide
    blocks, so col_block 3 becomes columns 384..511, a clean block: 'not reproduced', and the casefile says MOST
    CONSISTENT WITH the check for an output that is wrong by 50 at column 200."""
    torch.manual_seed(0)
    A = torch.randn(16, 256).to(torch.bfloat16)
    B = torch.randn(256, 512).to(torch.bfloat16)
    g = SDCGuard(block=64, evidence_dir=str(tmp_path), max_evidence=1)
    with _CorruptMM(3, 200):
        with g.at_step(0):
            A @ B
    assert g.evidence and g.log.events[0].detail["col_block"] == 3
    inc = cf.load_incident(g.evidence[0])
    assert cf.cpu_state(inc["cpu"]) == "outside envelope", inc["cpu"]



def test_screen_keeps_the_alerts_of_a_training_step_that_raises(tmp_path, monkeypatch):
    """screen.py: 'A failing GPU often computes wrong results and then raises a CUDA error: what was found before
    the error (alerts, saved evidence) still counts'. _Recorder._flush() takes no raise_alerts argument, so
    SDCGuard._flush_after_error's call raises TypeError, the step's checks are dropped, and a GPU that corrupted
    results and then raised is reported INCOMPLETE with 0 corrupted results and no evidence."""
    real = torch.nn.functional.cross_entropy

    def device_error(*a, **k):
        real(*a, **k)
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(screen.F, "cross_entropy", device_error)
    s = screen.screen_one(torch.device("cpu"), 1.0, "tiny", str(tmp_path), simulate_faults=1.0, max_steps=1)
    assert s["error"] and "illegal memory access" in s["error"]
    assert s["corrupted_results_training"] >= 1 and s["result"] == "FAIL", (s["result"], s["corrupted_results_training"])
    assert s["evidence"], "the failed checks of the step that raised weren't saved"



def test_math_mode_evidence_without_its_json_isnt_blamed_on_the_check(tmp_path):
    """load_incident takes math-mode evidence's verdict from the .json's wrong_blocks only. Without the .json
    (copied alone, lost, or not yet written when the run was stopped) it becomes output_wrong False: 'not
    reproduced', and MOST CONSISTENT WITH the check, though the .pt itself records wrong_blocks and holds the
    operands to judge it again."""
    from sdcguard import modes
    gen = torch.Generator().manual_seed(0)
    r = modes.check_once("bf16", torch.device("cpu"), (64, 256, 256), "normal", 1.0, gen, fault=1e4, keep=True)
    assert r["wrong_blocks"]
    fail = {**r.pop("operands"), "wrong_blocks": r["wrong_blocks"], "worst_z": r["worst_z"]}
    path = modes.save_evidence(str(tmp_path), 0, fail, {"device": "cuda:0", "gpu_uuid": "GPU-orig"}, {})
    os.remove(path + ".json")
    a = cf.assess(cf.load_incident(path + ".pt"), [])
    assert a["outcome"] != MOST + cf.EXPLANATIONS["check"], a["outcome"]



def test_precise_mode_evidence_can_be_judged(tmp_path):
    """SDCGuard(precise=True) saves the fp32 accumulator as C (bf16 A and B). cpu_recheck and replay use the
    bf16/fp16-only check, which returns None for an fp32 output: the CPU recheck says 'values that aren't finite
    in the inputs' and every replay is unjudged ('the inputs hold values that aren't finite'), both false: the
    inputs are finite and the accumulator is off by 50."""
    import torch.nn as nn
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=False)

    def corrupt(layer, pass_, C32):
        if pass_ == "forward":
            C32[2, 10] += 50.0

    g = SDCGuard(precise=True, on_accumulator=corrupt, evidence_dir=str(tmp_path), max_evidence=1).protect(lin)
    with g.at_step(0):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            lin(torch.randn(8, 256))
    g.unprotect()
    assert g.evidence, g.log.events
    inc = cf.load_incident(g.evidence[0])
    assert inc["cpu"].get("judged"), inc["cpu"]
    r = screen.replay(g.evidence[0], 3, "cpu")
    assert r["recomputations_that_couldnt_be_judged"] < 3, r["reading"]


def test_a_json_that_describes_another_pt_isnt_read_as_its_incident(tmp_path):
    """The .json records its .pt's SHA-256. Beside a different .pt (a run stopped between writes, a file copied
    over another) it is ignored and everything is judged again from the .pt, which here holds a correct output."""
    torch.manual_seed(0)
    A = torch.randn(4, 256).to(torch.bfloat16)
    B = torch.randn(256, 128).to(torch.bfloat16)
    wrong = (A.float() @ B.float()).to(torch.bfloat16)
    wrong[1, 3] += 40.0
    res = cs.check_matmul(A, B, wrong)
    assert not res.ok
    p = screen._save_evidence(str(tmp_path), 0, res, (A, B, wrong, None, 1.0, 1.0, "aten::mm", "run 1"),
                              {"gpu_uuid": "GPU-orig"}, {"run_id": "run1"})
    assert cf.cpu_state(cf.load_incident(p + ".pt")["cpu"]) == "outside envelope"
    ev = torch.load(p + ".pt", weights_only=True)
    ev["C"] = (A.float() @ B.float()).to(torch.bfloat16)     # another file now sits beside that .json
    torch.save(ev, p + ".pt")
    inc = cf.load_incident(p + ".pt")
    assert inc["info_ignored"] and cf.cpu_state(inc["cpu"]) != "outside envelope"
    assert any("describes another evidence file" in n for n in cf.assess(inc, [])["next"])
    no_tmp = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert not no_tmp


def test_a_screen_stopped_at_its_deadline_doesnt_leave_the_previous_pass(tmp_path, monkeypatch):
    """_all_gpus stops a child that hangs past its deadline (a classic symptom of a bad GPU) and counts it as not
    screened, but writes nothing for it: the GPU's folder keeps the previous run's summary.json, and the fleet
    report built from those folders shows that old PASS for the GPU that just hung."""
    from types import SimpleNamespace
    from sdcguard import fleet
    gdir = tmp_path / "screen_out" / "GPU-hung"
    gdir.mkdir(parents=True)
    (gdir / "summary.json").write_text(json.dumps({"result": "PASS", "gpu": {"gpu_uuid": "GPU-hung"},
                                                   "matmuls_checked": 1000, "run_id": "last week"}))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    class Hung:
        pid = 4242

        def __init__(self, *a, **k):
            pass

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("sdcguard.screen", timeout)
            return -9

        def kill(self):
            pass

    monkeypatch.setattr(screen.subprocess, "Popen", Hung)
    monkeypatch.setattr(screen, "_uuid_of", lambda gid, order: "GPU-hung" if order == "PCI_BUS_ID" else None)
    args = SimpleNamespace(minutes=0.01, size="tiny", out=str(tmp_path / "screen_out"), seed=0, simulate_faults=0.0)
    assert screen._all_gpus(args) == 2
    rows = fleet.collect([str(tmp_path / "screen_out")])
    assert rows[0]["result"] != "PASS", rows[0]["result"]
