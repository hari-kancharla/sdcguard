"""Precise mode (fp32 accumulator checks) on a GPT-style model: false alarms, what the model sees, cost,
and injected faults caught, next to the ordinary bf16 checks.

    python experiments/precise_gpu.py --device cuda --out results/precise
    python experiments/precise_gpu.py                       # CPU, tiny model: a smoke test

Four parts, on the same model (sdcguard.screen.ScreenModel) and data:
  clean      training steps with every linear layer's fp32 accumulators checked: alarms, worst score
  same       one step unguarded vs one step in precise mode: largest difference in logits and gradients
  cost       time per step unguarded, with bf16 checks, with precise checks (every step), and what that
             means at the README's every=20
  faults     the same fault injector (NVIDIA's published fault patterns) on the attention projections,
             corrupting the fp32 accumulator in precise mode and the bf16 result in the ordinary mode:
             how many faults were injected, checked, caught, and how big the missed ones were
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import FaultInjector, SDCGuard  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

CONFIG = {"cuda": dict(vocab=32000, layers=8, width=1024, heads=16, seq=512, batch=8),
          "cpu": dict(vocab=512, layers=2, width=128, heads=4, seq=64, batch=4)}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def run(dev: torch.device, clean_steps: int, fault_steps: int, rate: float, seed: int = 0, say=print) -> dict:
    c = CONFIG["cuda" if dev.type == "cuda" else "cpu"]
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    torch.manual_seed(seed)
    model = ScreenModel(c["vocab"], c["layers"], c["width"], c["heads"], c["seq"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(seed + 1)

    def batch():
        return torch.randint(0, c["vocab"], (c["batch"], c["seq"] + 1), generator=gen).to(dev)

    def step(guard=None, n=0, train=True):
        idx = batch()
        ctx = guard.on_step(n) if guard is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            with torch.autocast(dev.type, dtype=low):
                logits = model(idx[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
            loss.backward()
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        if train:
            opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    out = {"config": c, "dtype": str(low).replace("torch.", ""), "device": str(dev)}

    # clean
    zs = []
    g = SDCGuard(every=1, precise=True, on_result=lambda d, r: zs.append(r.max_z)).protect(model)
    for n in range(clean_steps):
        step(g, n)
    g.unprotect()
    judged = [z for z in zs if z >= 0]
    out["clean"] = {"steps": clean_steps, "checks": g.stats["checked"], "fp32_checks": g.stats["checked_fp32"],
                    "unjudged": g.stats["unjudged"], "alarms": g.stats["alerts"],
                    "worst_score": round(max(judged), 3) if judged else None,
                    "p99_score": round(sorted(judged)[int(0.99 * (len(judged) - 1))], 3) if judged else None}
    say(f"clean: {out['clean']}")

    # same: logits and gradients, unguarded vs precise, same weights and input
    idx = batch()

    def fwd_bwd(guard):
        ctx = guard.on_step(0) if guard is not None else None
        if ctx is not None:
            ctx.__enter__()
        try:
            with torch.autocast(dev.type, dtype=low):
                logits = model(idx[:, :-1])
                loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
            loss.backward()
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        grads = [p.grad.detach().float().clone() for p in model.parameters() if p.grad is not None]
        opt.zero_grad(set_to_none=True)
        return logits.detach().float(), grads

    l0, g0 = fwd_bwd(None)
    gp = SDCGuard(every=1, precise=True).protect(model)
    l1, g1 = fwd_bwd(gp)
    gp.unprotect()
    rel = [float((a - b).abs().max() / a.abs().max().clamp_min(1e-30)) for a, b in zip(g0, g1)]
    out["same"] = {"logits_max_abs_diff": float((l0 - l1).abs().max()),
                   "logits_identical": bool(torch.equal(l0, l1)),
                   "grad_max_rel_diff": max(rel), "grad_median_rel_diff": statistics.median(rel),
                   "grads_identical": all(torch.equal(a, b) for a, b in zip(g0, g1))}
    say(f"same: {out['same']}")

    # cost
    def timed(guard, reps=8):
        for _ in range(2):
            step(guard, 0, train=False)
        ts = []
        for _ in range(reps):
            _sync(dev)
            t = time.perf_counter()
            step(guard, 0, train=False)
            _sync(dev)
            ts.append(time.perf_counter() - t)
        return statistics.median(ts)

    t_plain = timed(None)
    gb = SDCGuard(every=1).protect(model)
    t_bf16 = timed(gb)
    gb.unprotect()
    gq = SDCGuard(every=1, precise=True).protect(model)
    t_prec = timed(gq)
    gq.unprotect()
    out["cost"] = {"seconds_per_step_unguarded": round(t_plain, 4), "seconds_bf16_checked": round(t_bf16, 4),
                   "seconds_precise_checked": round(t_prec, 4),
                   "bf16_overhead_at_every_20_pct": round(100 * (t_bf16 - t_plain) / t_plain / 20, 2),
                   "precise_overhead_at_every_20_pct": round(100 * (t_prec - t_plain) / t_plain / 20, 2)}
    say(f"cost: {out['cost']}")

    # faults
    targets = [b.qkv for b in model.blocks] + [b.proj for b in model.blocks]

    def faults(precise: bool):
        inj = FaultInjector(rate=rate, seed=seed + 7).arm_on(targets)
        caught = {}

        def start(op, C):
            return inj.match(C)

        def done(op, token, res):
            inj.resolve(token, res.ok, res.max_z)

        g = SDCGuard(every=1, precise=precise, on_check_start=start, on_check=done,
                     on_accumulator=inj.corrupt_tensor if precise else None).protect(model)
        if precise:
            for n in range(fault_steps):
                step(g, n, train=False)
        else:
            with inj:
                for n in range(fault_steps):
                    step(g, n, train=False)
        g.unprotect()
        inj.remove_hooks()
        ev = inj.events
        missed = [e["rel_change"] for e in ev if e.get("checked") and not e.get("caught")]
        caught_rel = [e["rel_change"] for e in ev if e.get("caught")]
        return {"injected": len(ev), "checked": sum(bool(e.get("checked")) for e in ev),
                "caught": sum(bool(e.get("caught")) for e in ev), "alarms": g.stats["alerts"],
                "missed_rel_change_max": max(missed) if missed else None,
                "missed_rel_change_median": statistics.median(missed) if missed else None,
                "caught_rel_change_min": min(caught_rel) if caught_rel else None}

    out["faults"] = {"rate": rate, "steps": fault_steps, "bf16": faults(False), "precise": faults(True)}
    say(f"faults: {out['faults']}")
    return out


def markdown(res: dict) -> str:
    c, s, k, f = res["clean"], res["same"], res["cost"], res["faults"]
    lines = [f"# Precise mode on {res.get('gpu', res['device'])} ({res['dtype']})", "",
             f"Model: {res['config']}. torch {res.get('torch', '?')}.", "",
             f"Clean: {c['steps']} steps, {c['fp32_checks']} fp32 checks, {c['alarms']} false alarms, worst score "
             f"{c['worst_score']} (99th percentile {c['p99_score']}; the alarm is at 6).", "",
             f"Same step unguarded vs precise: logits identical {s['logits_identical']} (largest difference "
             f"{s['logits_max_abs_diff']:.3g}), gradients largest relative difference {s['grad_max_rel_diff']:.3g}.", "",
             f"Time per step: unguarded {k['seconds_per_step_unguarded']} s, bf16 checks {k['seconds_bf16_checked']} s, "
             f"precise checks {k['seconds_precise_checked']} s. At every=20: bf16 {k['bf16_overhead_at_every_20_pct']}%, "
             f"precise {k['precise_overhead_at_every_20_pct']}%.", "",
             "| injected faults | injected | checked | caught | largest missed change | smallest caught change |",
             "|---|---|---|---|---|---|"]
    for name in ("bf16", "precise"):
        r = f[name]
        lines.append(f"| {name} | {r['injected']} | {r['checked']} | {r['caught']} | "
                     f"{r['missed_rel_change_max'] if r['missed_rel_change_max'] is not None else '-'} | "
                     f"{r['caught_rel_change_min'] if r['caught_rel_change_min'] is not None else '-'} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", default="cpu")
    p.add_argument("--clean-steps", type=int, default=None)
    p.add_argument("--fault-steps", type=int, default=None)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    dev = torch.device(a.device)
    gpu = dev.type == "cuda"
    res = run(dev, a.clean_steps or (30 if gpu else 3), a.fault_steps or (20 if gpu else 3), a.rate)
    res.update(gpu=torch.cuda.get_device_name(dev) if gpu else "CPU", torch=torch.__version__,
               date=time.strftime("%Y-%m-%d"))
    md = markdown(res)
    print(md)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = res["gpu"].replace("NVIDIA ", "").replace(" ", "_") if gpu else "cpu"
        with open(os.path.join(a.out, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        with open(os.path.join(a.out, f"{name}.md"), "w") as fh:
            fh.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
