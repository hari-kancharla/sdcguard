"""One-minute demo: a device goes bad in the middle of training, and sdcguard catches it.

    python -m sdcguard.demo                 # CPU (or your NVIDIA GPU if PyTorch sees one)
    python -m sdcguard.demo --device mps    # Apple GPU

What happens:
  1. A small network trains on a made-up task. Every 10th step, sdcguard checks every matrix
     multiply (forward and backward).
  2. At step 45 the "device" starts corrupting the output of one layer's matrix multiplies,
     using the fault patterns from NVIDIA's gate-level fault-injection study of a data-center
     GPU (zeroed values, bit flips).
     The corruption is simulated in software: no hardware is harmed.
  3. A later check catches it and names the layer, the row and the columns. Training rolls
     back to the newest checkpoint saved before the last two checks that passed, and the demo
     tells you whether that checkpoint really is from before the first fault.
  4. Training continues without the bad device and finishes with no more alerts.

Nothing is downloaded. Exit code 0 means the fault was caught, the rollback landed before the
first fault, and there were no false alarms. Other --seed values give other fault patterns;
on CPU, 39 of seeds 0-39 pass, and the one that fails (seed 20) says why.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .events import rollback_checkpoint
from .guard import SDCGuard
from .inject import FaultInjector, fault_model


class ToyNet(nn.Module):
    def __init__(self, d: int = 256, depth: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(depth)])
        self.out = nn.Linear(d, d)

    def forward(self, x):
        for layer in self.layers:
            x = x + F.gelu(layer(x))
        return self.out(x)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _pick_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _dtype_for(device: torch.device) -> torch.dtype:
    # sdcguard checks bf16/fp16 matmuls (mixed precision). Apple GPUs, and NVIDIA GPUs without
    # native bf16 (like Colab's T4): fp16.
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.float16
    return torch.bfloat16


def run(steps: int = 150, bad_from: int = 45, check_every: int = 10, ckpt_every: int = 20,
        rate: float = 0.5, bad_layer: int = 2, device: Optional[str] = None, seed: int = 0,
        out=print) -> dict:
    dev = _pick_device(device)
    dtype = _dtype_for(dev)
    torch.manual_seed(seed)
    teacher = ToyNet().to(dev)
    model = ToyNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    data_gen = torch.Generator().manual_seed(seed + 1)

    def batch():
        x = torch.randn(128, 256, generator=data_gen).to(dev)
        with torch.no_grad():
            y = teacher(x)
        return x, y

    def train_step(step, guard, inj):
        x, y = batch()
        if inj is not None:
            inj.step = step
        checked = guard is not None
        with (inj if inj is not None else _Null()):                  # the faulty hardware
            with (guard.at_step(step) if checked else _Null()):      # the detector, on top
                with torch.autocast(dev.type, dtype=dtype):
                    loss = F.mse_loss(model(x).float(), y)
                loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    inj = FaultInjector(rate=rate, fault_model=fault_model("nvidia", "warp"), seed=seed + 7,
                        start_step=bad_from).arm_on([model.layers[bad_layer]])

    # Only so the demo can show the truth next to each alert: tell the simulator which of its
    # faults a check looked at, and whether it was flagged. Real use needs none of this.
    def on_check(_op, token, res):
        if token is not None and inj is not None:
            inj.resolve(token, res.ok, res.max_z)

    guard = SDCGuard(on_check=on_check,
                     on_check_start=lambda _op, C: inj.match(C) if inj is not None else None).watch(model)

    out(f"sdcguard demo on {dev} ({str(dtype).replace('torch.', '')} mixed precision), {steps} steps.")
    out(f"Checks every matrix multiply on every {check_every}th step. Checkpoint every {ckpt_every} steps.")
    out(f"Simulated bad device: from step {bad_from}, it corrupts {rate:.0%} of layer "
        f"'layers.{bad_layer}' outputs (NVIDIA's published fault patterns).\n")

    t0 = time.time()
    ckpts: dict[int, dict] = {}
    losses: dict[int, float] = {}
    alert = None
    false_alarms_before = 0
    step = 0
    while step < steps:
        checked = step % check_every == 0
        loss = train_step(step, guard if checked else None, inj)
        losses[step] = loss
        if checked:
            new = guard.log.events
            if new:
                if step < bad_from:
                    false_alarms_before += len(new)
                ev = new[0]
                d = ev.detail
                cols = f"{d['col_block'] * guard.block}-{(d['col_block'] + 1) * guard.block - 1}"
                prev = [losses[s] for s in range(max(0, step - 10), step) if s in losses]
                prev_avg = sum(prev) / len(prev) if prev else float("nan")
                injected = [e for e in inj.events]
                who = ev.identity
                where_dev = (f"{who.get('gpu_name')} {who.get('gpu_uuid')} PCI {who.get('pci_bus_id')}"
                             if who.get("gpu_uuid") else f"{who.get('device')} on host {who.get('host')}")
                out(f"step {step:4d}  loss {loss:.4f}  sdcguard ALERT")
                out(f"      layer:     {d['where']} ({d['pass']} pass, {d['op']})")
                out(f"      location:  row {d['row']}, columns {cols} of the output")
                z = d["max_z"]
                out(f"      mismatch:  {z:.1f}x the expected rounding noise (alarm line: {guard.z:g})"
                    if z == z and z != float("inf") else "      mismatch:  NaN/Inf in the output")
                out(f"      device:    {where_dev}")
                out(f"      loss:      {loss:.4f} vs {prev_avg:.4f} average over the previous 10 steps")
                out(f"      truth:     the simulator corrupted {len(injected)} outputs since step "
                    f"{injected[0]['step'] if injected else '?'}; checks looked at these:")
                if not any(e.get("checked") for e in injected):
                    out("                 (none: no fault landed on a checked step; this alert is the damage spreading)")
                for e in injected:
                    if e.get("checked"):
                        verdict = "flagged" if e.get("caught") else "too small to flag"
                        out(f"                 step {e['step']}: changed its output block by "
                            f"{e['rel_change']:.2%}, {verdict}" if e["rel_change"] != float("inf")
                            else f"                 step {e['step']}: NaN/Inf, {verdict}")
                alert = {"step": step, "detail": {k: d[k] for k in ("where", "pass", "op", "row", "col_block")},
                         "max_z": z, "injected_before": len(injected),
                         "first_fault": injected[0]["step"] if injected else None}
                # windows=2: step back past the last TWO checks that passed. Faults come and go,
                # so a single passing check doesn't prove the device was healthy then.
                safe = rollback_checkpoint(step, sorted(ckpts), check_every=check_every, windows=2)
                if safe is None:
                    out("\n--> No checkpoint old enough to trust. Stopping.")
                    break
                out(f"\n--> Evicting the bad device and rolling back to the checkpoint from step {safe}: the "
                    f"newest one saved before the last two checks that passed (steps {step - 2 * check_every} "
                    f"and {step - check_every}).")
                first_fault = injected[0]["step"] if injected else None
                clean = first_fault is None or safe < first_fault
                out(f"    truth: the first fault was at step {first_fault}, so this checkpoint is "
                    + ("clean.\n" if clean else "NOT clean: it may hold corrupted weights.\n"))
                model.load_state_dict(ckpts[safe]["model"])
                opt.load_state_dict(ckpts[safe]["opt"])
                alert["rollback_to"] = safe
                alert["rollback_clean"] = clean
                inj.remove_hooks()
                inj = None                                  # training continues on healthy hardware
                guard.log.events.clear()
                step = safe + 1
                continue
            if step % (check_every * 2) == 0:
                out(f"step {step:4d}  loss {loss:.4f}  checked so far: {guard.stats['checked']:,} matmuls, "
                    f"new alerts: {len(guard.log.events)}")
        if step % ckpt_every == 0 and step > 0:
            ckpts[step] = {"model": copy.deepcopy(model.state_dict()), "opt": copy.deepcopy(opt.state_dict())}
        step += 1

    alerts_after = len(guard.log.events)
    caught = alert is not None and alert["step"] >= bad_from
    out(f"\nFinished {steps} steps in {time.time() - t0:.1f}s. Final loss {losses[steps - 1]:.4f}.")
    out(f"Checks: {guard.stats['checked']:,} matmuls. Alerts before the fault started: {false_alarms_before}. "
        f"Alerts after the rollback: {alerts_after}.")
    if caught:
        out(f"Caught at step {alert['step']}, {alert['step'] - bad_from} steps after the device went bad.")
    else:
        out("The fault was NOT caught in this run.")
    ok = caught and alert.get("rollback_clean", False) and false_alarms_before == 0 and alerts_after == 0
    out("RESULT: " + ("PASS" if ok else "FAIL"))
    return {"ok": ok, "alert": alert, "false_alarms_before": false_alarms_before,
            "alerts_after": alerts_after, "checked": guard.stats["checked"], "final_loss": losses[steps - 1],
            "device": str(dev)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cpu, cuda or mps (default: cuda if available, else cpu)")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--bad-from", type=int, default=45, help="step at which the device goes bad")
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--rate", type=float, default=0.5, help="share of the bad layer's matmuls corrupted")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    res = run(steps=a.steps, bad_from=a.bad_from, check_every=a.check_every, rate=a.rate,
              device=a.device, seed=a.seed)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
