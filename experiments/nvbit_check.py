"""Faults injected into a real GPU's tensor-core instructions (NVBit), and whether sdcguard catches them.

The software fault simulator (sdcguard.inject) corrupts matmul outputs after the fact. This uses
instruction-level injection instead: TU Berlin's NVBit fault injector
(github.com/aaltenbernd/llm-sdc-training, rebuilt here against NVBit 1.8, from "Exploring Silent Data Corruption as a
Reliability Challenge in LLM Training", arXiv 2604.00726). While it is switched on, every tensor-core
instruction (HMMA) that runs on one lane of one SM, in every cuBLAS GEMM kernel, gets bits of an input
register flipped. That is a model of a GPU with one bad tensor-core lane: the matmul tiles that land on
that SM come out wrong, the rest are right. It is switched on while the file /tmp/sdcguard_fault_on
exists (scripts/modal_nvbit.py patches the injector to check that file at every kernel launch).

Run it under the injector (scripts/modal_nvbit.py --phase nvbit does this):
    CUDA_INJECTION64_PATH=.../fault_injection.so TARGET_OP=HMMA TARGET_REGISTER=1 TARGET_SMID=0 \
    TARGET_LANEID=0 TARGET_BITMASK=16 TARGET_FUNC=-1 TARGET_FUNC_CONTAINS=gemm python experiments/nvbit_check.py

Positive control: before each phase a plain bf16 matmul on fixed inputs is compared with its result from
before the injector was switched on. If "fault on" changes nothing, the injector isn't working (and a
silent detector would prove nothing): the report then says injector_active = false.

Phases, in one process with one model (training steps checked by SDCGuard fast mode on every step,
plus one serving pass per step checked by ServeGuard on a bf16 copy of the model):
  clean      30 steps with the injector loaded but off: alerts must be 0
  fault on   flag file created, 30 steps: when sdcguard first flags it, which layers, the loss
  fault off  flag file removed, 30 steps: do the alerts stop?
Needs a GPU whose GEMMs use HMMA (Ampere/Ada: A100, A10G, L4, L40S). Hopper's cuBLAS kernels use
HGMMA, which this injector doesn't target.

Note: the injector corrupts every cuBLAS kernel whose name contains "gemm", including the skinny matmul
the serving check itself uses (x @ block sums of W). A wrong result there also makes the check fail: it is
still a wrong result on this GPU, but an alert can't always say which of the two matmuls was hit. The
Triton kernels sdcguard launches are not instrumented.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sdcguard import SDCGuard  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402
from sdcguard.serve import ServeGuard  # noqa: E402

STEPS = int(os.environ.get("NVBIT_STEPS", "30"))
OUT = os.environ.get("NVBIT_OUT", "nvbit.json")
FLAG = "/tmp/sdcguard_fault_on"          # the patched injector injects while this file exists


def _fault(on: bool) -> None:
    if on:
        open(FLAG, "w").close()
    elif os.path.exists(FLAG):
        os.remove(FLAG)


def main():
    _fault(False)
    # (on a machine without a GPU it runs on the CPU, without the injector: a quick test of this script)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    vocab, L, d, h, seq, batch = 8192, 4, 512, 8, 256, 8
    model = ScreenModel(vocab, L, d, h, seq).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    serve = ScreenModel(vocab, L, d, h, seq).to(dev)
    serve.load_state_dict(model.state_dict())
    serve = serve.to(torch.bfloat16).eval()
    sguard = ServeGuard().attach(serve)
    alerts_by_step: dict = {}
    tguard = SDCGuard(on_alert=lambda e: alerts_by_step.setdefault(e.step, []).append(e.detail.get("where")))
    tguard.protect(model)
    gen = torch.Generator().manual_seed(1)
    rows = []
    step = 0
    report = {"gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "bitmask": os.environ.get("TARGET_BITMASK"),
              "smid": os.environ.get("TARGET_SMID"), "laneid": os.environ.get("TARGET_LANEID"),
              "injector_loaded": "fault_injection" in (os.environ.get("CUDA_INJECTION64_PATH", "") +
                                                      os.environ.get("LD_PRELOAD", ""))}
    pa = torch.randn(1024, 1024, generator=torch.Generator().manual_seed(9)).to(dev, torch.bfloat16)
    pb = torch.randn(1024, 1024, generator=torch.Generator().manual_seed(10)).to(dev, torch.bfloat16)
    ref = torch.mm(pa, pb)

    def probe():
        out = torch.mm(pa, pb)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return int((out != ref).sum())
    for phase in ("clean", "fault on", "fault off"):
        if phase == "fault on":
            _fault(True)
        elif phase == "fault off":
            _fault(False)
        report[f"probe_{phase}"] = probe()   # elements of a fixed matmul that differ from the clean result
        t0 = time.time()
        for _ in range(STEPS):
            x = torch.randint(0, vocab, (batch, seq), generator=gen).to(dev)
            y = torch.randint(0, vocab, (batch, seq), generator=gen).to(dev)
            with tguard.at_step(step):
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    loss = F.cross_entropy(model(x).float().view(-1, vocab), y.view(-1))
                loss.backward()
            # a step with a non-finite loss is skipped (as loss-scaling trainers do), so NaN never reaches the
            # weights: every training check then judges finite inputs, and "fault off" means something
            if torch.isfinite(loss):
                opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                serve(x)
            sev = sguard.poll(wait=True)
            rows.append({"step": step, "phase": phase, "loss": loss.item(),
                         "train_alerts": sorted(set(alerts_by_step.get(step, []))),
                         "serve_alerts": sorted({e.detail["where"] for e in sev})})
            step += 1
        print(f"{phase}: {time.time() - t0:.1f} s", flush=True)
    for phase in ("clean", "fault on", "fault off"):
        pr = [r for r in rows if r["phase"] == phase]
        flagged = [r for r in pr if r["train_alerts"] or r["serve_alerts"]]
        first = flagged[0]["step"] - pr[0]["step"] if flagged else None
        layers = sorted({a for r in pr for a in r["train_alerts"] + r["serve_alerts"]})
        report[phase] = {"steps": len(pr), "steps_with_alerts": len(flagged), "first_alert_after_steps": first,
                         "layers_flagged": layers[:40], "loss_first_last": [pr[0]["loss"], pr[-1]["loss"]]}
        print(f"{phase}: {len(flagged)}/{len(pr)} steps with alerts, first after {first} steps, "
              f"{len(layers)} layers flagged; positive control: {report.get('probe_' + phase)} elements of a "
              f"fixed matmul changed", flush=True)
    report["injector_active"] = report.get("probe_fault on", 0) > 0 and report.get("probe_clean", 1) == 0
    if not report["injector_active"]:
        print("THE INJECTOR DID NOT SWITCH ON (the positive control didn't change): this run says nothing "
              "about sdcguard either way", flush=True)
    report["checks"] = tguard.stats
    report["rows"] = rows
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
