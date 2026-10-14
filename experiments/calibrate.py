"""Calibration: kill criterion #1 (false alarms and slowdown) on clean training.

Kill criterion #1: zero false alarms over ~5 clean bf16 GPU-hours, at 3% slowdown or less.

Three modes (paste one line at a time):
  python experiments/calibrate.py --preset ts-gpt2 --overhead
      slowdown for several detector settings ("every:share", e.g. 10:1 = check every matmul on
      every 10th step; 1:0.1 = check 10% of matmuls on every step)
  python experiments/calibrate.py --preset ts-gpt2 --soak-minutes 50 --worker 0
      clean training with the detector checking EVERY matmul, counting false alarms. Several
      workers (GPUs) can run at once; each saves its counts every 2 minutes
  python experiments/calibrate.py --preset ts-gpt2 --summarize
      combine everything into runs/calibration/<preset>/CALIBRATION.md
  python experiments/calibrate.py --preset ts-gpt2 --diag
      where do false alarms come from: per-matmul worst residuals with and without
      reduced-precision sums, plus CPU vs GPU timing (runs/diag/<preset>/DIAG.md)

Old one-shot form still works: --minutes N runs a short soak after the overhead sweep.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import uuid

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harm_study import (GPT, PRESETS, autocast_ctx, code_version, env_info, get_batch,  # noqa: E402
                        load_dataset, lr_at, pick_device, resolve_dtype)
from sdcguard import SDCGuard  # noqa: E402

Z_BINS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, float("inf")]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make(cfg, vocab, device, seed=0):
    torch.manual_seed(seed)
    model = GPT(vocab, cfg["block"], cfg["n_layer"], cfg["n_head"], cfg["n_embd"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    return model, opt


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _scaler(opt, device, dtype):
    """fp16 training on NVIDIA GPUs uses loss scaling (torch.amp.GradScaler), as real fp16 training
    does: it overflows to Inf now and then on purpose and skips those steps. bf16 needs none."""
    if dtype != torch.float16 or device.type != "cuda":
        return None
    if getattr(opt, "_sdc_scaler", None) is None:
        opt._sdc_scaler = torch.amp.GradScaler("cuda")
        opt._sdc_overflows = 0
    return opt._sdc_scaler


def train_steps(model, opt, data, cfg, device, dtype, n, gen, guard=None, every=1, start=0, schedule=False):
    """n realistic training steps (loss read every step, as a training loop that logs would).
    Returns seconds per step."""
    sc = _scaler(opt, device, dtype)
    sync(device)
    t0 = time.perf_counter()
    for i in range(n):
        step = start + i
        if schedule:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
        x, y = get_batch(data, cfg["block"], cfg["batch"], gen, device)
        ctx = guard.at_step(step) if (guard is not None and step % every == 0) else _null()
        with ctx:
            with autocast_ctx(device, dtype):
                loss = model(x, y)
            (sc.scale(loss) if sc is not None else loss).backward()
        if sc is not None:
            sc.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if sc is not None:
            before = sc.get_scale()
            sc.step(opt)
            sc.update()
            if sc.get_scale() < before:          # the step overflowed and was skipped
                opt._sdc_overflows += 1
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        loss.item()
    sync(device)
    return (time.perf_counter() - t0) / max(n, 1)


def parse_sweep(s):
    out = []
    for item in s.split(","):
        every, _, share = item.partition(":")
        out.append((int(every), float(share or 1.0)))
    return out


def _alert_row(e):
    d = e.detail
    return {"pass": d.get("pass"), "where": d.get("where"), "op": d.get("op"), "shape_A": list(d.get("shape_A", ())),
            "shape_B": list(d.get("shape_B", ())), "z": d.get("max_z"), "step": e.step}


def overhead(model, opt, data, cfg, device, dtype, gen, sweep, timing_steps, rounds, fast=False):
    """Slowdown per setting, measured two ways, plus the false alarms seen while measuring.

    1. Per step: alternate plain and checked steps, take the median extra time of a checked step.
       Checking every k-th step then costs extra / (k x plain step): the estimate.
    2. Directly, for "every k steps" settings: plain vs checked windows of whole multiples of k
       steps, aligned so each checked window holds exactly the right number of checked steps.
       (The first version timed 20-step windows, so "every 50" windows held 0 or 1 checked steps.)
    """
    # fast=True: SDCGuard.protect(model), so only the linear layers' matmuls go through Python (and
    # plain steps also pay the one `if` per layer that fast mode adds). One guard for every setting.
    fast_guard = SDCGuard().protect(model) if fast else None
    train_steps(model, opt, data, cfg, device, dtype, 10, gen)                       # warm-up
    shares = sorted({share for _, share in sweep}, reverse=True)
    per_share, step = {}, 0
    base_samples = []
    for share in shares:
        alerts = []
        if fast_guard is not None:
            g = fast_guard
            g.sample_rate, g.on_alert = share, (lambda e, a=alerts: a.append(_alert_row(e)))
        else:
            g = SDCGuard(sample_rate=share, on_alert=lambda e, a=alerts: a.append(_alert_row(e)))
        train_steps(model, opt, data, cfg, device, dtype, 3, gen, g, 1)              # warm up the guard path
        plain, checked = [], []
        for i in range(timing_steps):
            for kind in (("p", "c") if i % 2 == 0 else ("c", "p")):
                if kind == "p":
                    plain.append(train_steps(model, opt, data, cfg, device, dtype, 1, gen))
                else:
                    checked.append(train_steps(model, opt, data, cfg, device, dtype, 1, gen, g, 1, start=step))
                step += 1
        b, c = _median(plain), _median(checked)
        base_samples += plain
        per_share[share] = {"guard": g, "alerts": alerts, "plain_s": b, "checked_s": c, "extra_s": c - b}
        print(f"  {share:.0%} of matmuls checked: plain step {1000 * b:.1f} ms, checked step {1000 * c:.1f} ms "
              f"(+{1000 * (c - b):.1f} ms per checked step)", flush=True)
    base = _median(base_samples)

    res = []
    for every, share in sweep:
        ps = per_share[share]
        est = ps["extra_s"] / (every * ps["plain_s"])
        measured, rounds_pct = None, []
        if share == 1.0 and every > 1:
            g = ps["guard"]
            g.sample_rate = 1.0
            window = every * max(2, math.ceil(20 / every))
            for _ in range(rounds):
                step = math.ceil(step / every) * every                             # align: step % every == 0
                p = train_steps(model, opt, data, cfg, device, dtype, window, gen)
                t = train_steps(model, opt, data, cfg, device, dtype, window, gen, g, every, start=step)
                step += window
                rounds_pct.append(100 * (t - p) / p)
            measured = _median(rounds_pct)
        shown = measured if measured is not None else 100 * est
        res.append({"every": every, "share": share, "coverage": share / every,
                    "estimated_pct": round(100 * est, 2),
                    "measured_pct": None if measured is None else round(measured, 2),
                    "measured_rounds_pct": [round(r, 2) for r in rounds_pct],
                    "slowdown_pct": round(shown, 2)})
        print(f"  check {share:.0%} of matmuls every {every:3d} steps: {100 * est:+.2f}% estimated"
              + (f", {measured:+.2f}% measured (rounds: {', '.join(f'{r:+.1f}%' for r in rounds_pct)})"
                 if measured is not None else ""), flush=True)

    alerts = [a for share in shares for a in per_share[share]["alerts"]]
    guards = list({id(per_share[sh]["guard"]): per_share[sh]["guard"] for sh in shares}.values())
    checks = sum(g.stats["checked"] for g in guards)
    worst = max(g.max_z_seen for g in guards)
    print(f"  false alarms during these clean runs: {len(alerts)} in {checks:,} checks "
          f"(worst clean z {worst:.2f}, alert threshold 6)", flush=True)
    print(f"  {_triton_note()}", flush=True)
    for a in alerts[:10]:
        print(f"    false alarm: {a}", flush=True)
    timing = {"base_ms_per_step": round(1000 * base, 2),
              "per_share": {str(sh): {"plain_ms": round(1000 * v["plain_s"], 2),
                                      "checked_ms": round(1000 * v["checked_s"], 2),
                                      "extra_ms_per_checked_step": round(1000 * v["extra_s"], 2)}
                            for sh, v in per_share.items()},
              "false_alarms": len(alerts), "checks": checks, "worst_clean_z": worst, "alerts": alerts[:50],
              "hook": "fast (protect: linear layers only)" if fast else "operator-level (every op)"}
    return res, base, timing


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _diag_train(model, opt, data, cfg, device, dtype, steps, gen, guard, schedule):
    """Clean training with the guard on every step; returns (losses, seconds per step)."""
    losses = []
    sync(device)
    t0 = time.perf_counter()
    for step in range(steps):
        if schedule:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
        x, y = get_batch(data, cfg["block"], cfg["batch"], gen, device)
        with guard.at_step(step):
            with autocast_ctx(device, dtype):
                loss = model(x, y)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    sync(device)
    return losses, (time.perf_counter() - t0) / max(steps, 1)


DIAG_MODES = (
    ("with LR warmup (like the harm study and the soak)", True, None),
    ("no LR warmup, OLD noise floor 1e-30 (reproduces check run #2's false alarms)", False, 1e-30),
    ("no LR warmup, NEW noise floor 1e-20 (the fix)", False, None),
)


def _cpu_vs_gpu(cfg, ds, device, dtype, seed, vocab, label, profile_kinds, out):
    """Median CPU-queuing and GPU-finish times of plain vs checked steps, plus GPU profiles."""
    lines = []
    model, opt = make(cfg, vocab, device, seed)
    gen = torch.Generator().manual_seed(seed)
    g = SDCGuard()
    g_fast_red = SDCGuard(exact_reductions=False)
    guards = {"plain": None, "checked": g, "checked, reduced-precision sums allowed": g_fast_red}
    rows = {}
    for kind, gk in guards.items():
        cpu, wall = [], []
        for i in range(8):
            x, y = get_batch(ds.train, cfg["block"], cfg["batch"], gen, device)
            sync(device)
            t0 = time.perf_counter()
            ctx = gk.at_step(i) if gk is not None else _null()
            with ctx:
                with autocast_ctx(device, dtype):
                    loss = model(x, y)
                loss.backward()
                t1 = time.perf_counter()
            sync(device)
            t2 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            if i >= 2:
                cpu.append(t1 - t0)
                wall.append(t2 - t0)
        rows[kind] = (1000 * _median(cpu), 1000 * _median(wall))
    lines += [f"## CPU vs GPU, {label} (forward + backward only, median of 6)", "",
              "| step | CPU done queuing work (ms) | GPU done (ms) |", "|---|---|---|"]
    lines += [f"| {k} | {c:.1f} | {w:.1f} |" for k, (c, w) in rows.items()]
    lines += ["", "If the two columns are close, the step is limited by Python/kernel launches (CPU), "
              "not by the GPU's math.", "",
              f"Guard stats over these steps: {g.stats}; with reduced-precision sums allowed: {g_fast_red.stats}",
              "", _triton_note(), ""]
    out["cpu_vs_gpu_ms"][label] = rows
    out["guard_stats"][label] = {"checked": dict(g.stats), "reduced_precision_sums_allowed": dict(g_fast_red.stats)}
    for kind, gk in guards.items():
        if profile_kinds is not None and kind not in profile_kinds:
            continue
        try:
            from torch.profiler import ProfilerActivity, profile
            x, y = get_batch(ds.train, cfg["block"], cfg["batch"], gen, device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                with (gk.at_step(99) if gk is not None else _null()):
                    with autocast_ctx(device, dtype):
                        loss = model(x, y)
                    loss.backward()
                sync(device)
            opt.zero_grad(set_to_none=True)
            ka = prof.key_averages()
            for key in ("self_cuda_time_total", "self_device_time_total"):
                try:
                    table = ka.table(sort_by=key, row_limit=20)
                    break
                except (AttributeError, KeyError, ValueError, RuntimeError):
                    table = None
            if table:
                lines += [f"## Profile of one {kind} step, {label} (forward + backward), top 20 by GPU time", "",
                          "```", table, "```", ""]
        except Exception as e:  # noqa: BLE001
            lines += [f"(profiler skipped for {kind}, {label}: {e})", ""]
    print("\n".join(lines[:12]), flush=True)
    return lines


def diag(cfg, ds, device, dtype, seed, steps, base):
    """Where do false alarms come from? Clean training with every matmul checked on every step,
    with and without LR warmup, and with the old and new noise floor. Lists the worst matmul
    sites, alarm details (with the block's noise scale), losses, plus CPU vs GPU timing."""
    os.makedirs(base, exist_ok=True)
    out = {"env": env_info(device), "preset": cfg.get("name"), "steps": steps, "modes": {}}
    lines = [f"# Diagnosis: {cfg.get('name')} on {out['env'].get('gpu') or device.type}", "",
             f"Each mode: a fresh model (seed {seed}), {steps} clean steps, every matmul checked on every step.", ""]
    for label, schedule, floor in DIAG_MODES:
        model, opt = make(cfg, ds.vocab, device, seed)
        gen = torch.Generator().manual_seed(seed)
        sites, alarms, smallest = {}, [], [float("inf")]

        def on_result(d, r, sites=sites, alarms=alarms, smallest=smallest):
            key = f"{d['pass']:8s} {d['op']:12s} {str(d['where'])[:22]:22s} A{tuple(d['shape_A'])} B{tuple(d['shape_B'])}"
            row = sites.setdefault(key, [0, 0, 0.0])
            row[0] += 1
            row[1] += 0 if r.ok else 1
            row[2] = max(row[2], r.max_z if math.isfinite(r.max_z) else float("inf"))
            if r.noise is not None and r.noise > 0:
                smallest[0] = min(smallest[0], r.noise)
            if not r.ok and len(alarms) < 12:
                alarms.append({"site": key.strip(), "z": round(r.max_z, 2), "noise": r.noise})

        g = SDCGuard(on_result=on_result, abs_floor=floor).watch(model)
        losses, t = _diag_train(model, opt, ds.train, cfg, device, dtype, steps, gen, g, schedule)
        g.unwatch()
        rows = sorted(sites.items(), key=lambda kv: -kv[1][2])
        mode = {"checks": g.stats["checked"], "false_alarms": g.stats["alerts"], "worst_z": g.max_z_seen,
                "ms_per_step": round(1000 * t, 2), "smallest_noise_at_worst_block": smallest[0],
                "loss_every_25": [round(x, 4) for x in losses[::25]] + [round(losses[-1], 4)],
                "alarms": alarms,
                "sites": [{"site": k, "checks": v[0], "alarms": v[1], "worst_z": v[2]} for k, v in rows]}
        out["modes"][label] = mode
        lines += [f"## {label}", "",
                  f"- {mode['checks']:,} checks, **{mode['false_alarms']} false alarms**, worst clean z "
                  f"{mode['worst_z']:.2f} (threshold 6), {mode['ms_per_step']} ms per checked step",
                  f"- training loss every 25 steps: {mode['loss_every_25']}",
                  f"- smallest block noise scale seen (at worst blocks): {mode['smallest_noise_at_worst_block']:.2e}"]
        if alarms:
            lines += ["- first alarms (site, z, block noise scale):"]
            lines += [f"    {a['site']}  z={a['z']}  noise={a['noise']:.2e}" for a in alarms]
        lines += ["", "Worst 10 matmul sites (pass, op, where, shapes): checks, alarms, worst z", "", "```"]
        lines += [f"{k}  {v[0]:6d} {v[1]:6d} {v[2]:10.2f}" for k, v in rows[:10]]
        lines += ["```", ""]
        print("\n".join(lines[-22:]), flush=True)

    # CPU-bound or GPU-bound? Time until the Python code returns vs until the GPU finishes.
    # Also with the vocabulary padded to a multiple of 64 (as nanoGPT does): GPT-2's 50,257-word
    # output layer is an odd shape for the GPU's matmul kernels.
    if device.type == "cuda":
        out["cpu_vs_gpu_ms"], out["guard_stats"] = {}, {}
        padded = -(-ds.vocab // 64) * 64
        for vocab, label, profile_kinds in ((ds.vocab, f"vocab {ds.vocab}", None),
                                            (padded, f"vocab padded to {padded}", ("checked",))):
            if vocab == ds.vocab and label != f"vocab {ds.vocab}":
                continue
            lines += _cpu_vs_gpu(cfg, ds, device, dtype, seed, vocab, label, profile_kinds, out)
    with open(os.path.join(base, "DIAG.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(base, "diag.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("\n".join(lines[-40:]), flush=True)
    return out


def _triton_note() -> str:
    from sdcguard import _group
    if not _group.enabled():
        return "Triton GPU check: off (plain PyTorch check)"
    return (f"Triton GPU check: {_group.USED[0]:,} checks so far, "
            f"{_group.FALLBACKS[0]} compile/launch problems (plain check took over)")


def soak(model, opt, data, cfg, device, dtype, gen, minutes, path, sample_rate=1.0, every=1, save_every_s=300,
         on_save=None):
    """Clean training, detector on every matmul of every step; saves cumulative counts to `path`
    every `save_every_s` seconds (then calls on_save(), e.g. to persist a cloud volume)."""
    hist = [0] * len(Z_BINS)
    alerts = []

    def on_check(_op, _tok, res):
        z = res.max_z if math.isfinite(res.max_z) else float("inf")
        for i, edge in enumerate(Z_BINS):
            if z < edge or i == len(Z_BINS) - 1:
                hist[i] += 1
                break

    guard = SDCGuard(sample_rate=sample_rate, on_check=on_check, on_alert=lambda e: alerts.append(e.detail))
    t_end = time.time() + 60 * minutes
    step, secs, last_save = 0, 0.0, time.time()
    rec = {}
    while True:
        t = train_steps(model, opt, data, cfg, device, dtype, 20, gen, guard, every, start=step, schedule=True)
        secs += 20 * t
        step += 20
        done = time.time() >= t_end
        if done or time.time() - last_save > save_every_s:
            rec = {"env": env_info(device), "code": code_version(("calibrate.py",)), "finished": done,
                   "preset": cfg.get("name"), "steps": step, "dtype": str(dtype).replace("torch.", ""),
                   "loss_scale_overflow_steps": getattr(opt, "_sdc_overflows", None),
                   "gpu_hours": round(secs / 3600, 4), "checks": guard.stats["checked"],
                   "matmuls_seen": guard.stats["matmuls_seen"], "false_alarms": guard.stats["alerts"],
                   "worst_clean_z": guard.max_z_seen, "z_hist_edges": [str(e) for e in Z_BINS],
                   "z_hist": hist, "alerts": alerts[:50], "ms_per_step": round(1000 * secs / step, 2),
                   "updated": time.time()}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + ".part", "w") as f:
                json.dump(rec, f)
            os.replace(path + ".part", path)
            if on_save is not None:
                on_save()
            last_save = time.time()
            print(f"  soak: {step} steps, {rec['gpu_hours']:.3f} GPU-h, {rec['checks']:,} checks, "
                  f"{rec['false_alarms']} false alarms, worst z {rec['worst_clean_z']:.2f}", flush=True)
        if done:
            return rec


def _harm_study_alarms(base, preset):
    """False alarms from the harm study's runs (base = <out>/calibration/<preset>)."""
    raw = os.path.join(os.path.dirname(os.path.dirname(base)), preset, "raw")
    clean_alarms = clean_checks = off_fault = 0
    n_clean = n_faulty = 0
    for p in glob.glob(os.path.join(raw, "*.json")):
        try:
            d = json.load(open(p))
        except (OSError, ValueError):
            continue
        name = os.path.basename(p)
        if name.startswith("clean_"):
            n_clean += 1
            clean_alarms += d.get("guard_alerts", 0)
            clean_checks += d.get("guard_checked", 0)
        elif name.startswith("faulty_"):
            n_faulty += 1
            # alerts on NaN/Inf/overflowing outputs are the injected corruption spreading, not false alarms
            off_fault += d.get("alerts_off_fault", 0) - d.get("off_fault_nonfinite", 0)
    return {"clean_runs": n_clean, "clean_alarms": clean_alarms, "clean_checks": clean_checks,
            "faulty_runs": n_faulty, "off_fault_alerts": off_fault}


def _soak_dir(dtype) -> str:
    """bf16 soaks count toward kill criterion #1; fp16 (with loss scaling) is reported separately."""
    return "soak_fp16" if dtype == torch.float16 else "soak"


def summarize(base, preset):
    soaks = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(base, "soak", "*.json")))]
    soaks16 = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(base, "soak_fp16", "*.json")))]
    ovs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(base, "overhead*.json")))]
    hs = _harm_study_alarms(base, preset)
    L = [f"# Calibration: {preset}", ""]
    for name, group in (("bf16 soak", soaks), ("fp16 soak", soaks16)):
        versions = sorted({str(sk.get("code")) for sk in group})
        if len(versions) > 1:
            L += [f"**WARNING: the {name} files come from {len(versions)} code versions {versions}.**", ""]
    total = None
    if soaks:
        hist = [sum(s["z_hist"][i] for s in soaks) for i in range(len(Z_BINS))]
        total = {"workers": len(soaks), "gpu_hours": sum(s["gpu_hours"] for s in soaks),
                 "checks": sum(s["checks"] for s in soaks), "false_alarms": sum(s["false_alarms"] for s in soaks),
                 "worst_clean_z": max(s["worst_clean_z"] for s in soaks), "z_hist": hist,
                 "gpu": soaks[0]["env"].get("gpu") or soaks[0]["env"].get("device")}
        unfinished = sum(1 for sk in soaks if sk.get("finished") is False)
        L += ["## False alarms on clean training (detector checking every matmul, forward and backward)", "",
              f"- {total['gpu']}: {total['gpu_hours']:.2f} GPU-hours over {total['workers']} runs, "
              f"{total['checks']:,} checks" + (f" ({unfinished} run(s) stopped early; their hours up to the last "
                                               "save are counted)" if unfinished else ""),
              f"- False alarms: **{total['false_alarms']}**",
              f"- Worst clean residual: z = {total['worst_clean_z']:.2f} (alert threshold 6)", "",
              "| z range | checks |", "|---|---|"]
        lo = 0
        for edge, n in zip(Z_BINS, hist):
            L.append(f"| {lo}-{edge} | {n:,} |")
            lo = edge
        L.append("")
    if soaks16:
        L += ["## False alarms on clean fp16 training with loss scaling (torch.amp.GradScaler)", "",
              f"- {soaks16[0]['env'].get('gpu') or soaks16[0]['env'].get('device')}: "
              f"{sum(s['gpu_hours'] for s in soaks16):.2f} GPU-hours over {len(soaks16)} run(s), "
              f"{sum(s['checks'] for s in soaks16):,} checks, every matmul checked",
              f"- Steps where the loss scale overflowed (GradScaler skipped them): "
              f"{sum(s.get('loss_scale_overflow_steps') or 0 for s in soaks16)}",
              f"- False alarms: **{sum(s['false_alarms'] for s in soaks16)}**",
              f"- Worst clean residual: z = {max(s['worst_clean_z'] for s in soaks16):.2f} (alert threshold 6)", ""]
    ov_rows, ov_alarms, ov_checks = [], 0, 0
    for o in ovs:
        L += [f"## Slowdown ({o['env'].get('gpu') or o['env'].get('device')}, base {o['base_ms_per_step']} ms/step)", "",
              "| setting | share of matmuls checked | slowdown (estimated) | slowdown (measured) |",
              "|---|---|---|---|"]
        for r in o["results"]:
            est = r.get("estimated_pct", r["slowdown_pct"])
            meas = r.get("measured_pct")
            L.append(f"| {r['share']:.0%} of matmuls every {r['every']} steps | {r['coverage']:.1%} | "
                     f"{est:+.2f}% | {'' if meas is None else f'{meas:+.2f}%'} |")
            ov_rows.append(r)
        t = o.get("timing", {})
        if t:
            L.append("")
            for sh, v in t.get("per_share", {}).items():
                L.append(f"- {float(sh):.0%} of matmuls checked: plain step {v['plain_ms']} ms, checked step "
                         f"{v['checked_ms']} ms (+{v['extra_ms_per_checked_step']} ms per checked step)")
            ov_alarms += t.get("false_alarms", 0)
            ov_checks += t.get("checks", 0)
            L.append(f"- False alarms during the slowdown runs (clean training): **{t.get('false_alarms', 0)}** "
                     f"in {t.get('checks', 0):,} checks, worst clean z {t.get('worst_clean_z', 0):.2f}")
        L.append("")
    def worst_case(r):        # judge a setting by the higher of its estimated and measured slowdown
        vals = [v for v in (r.get("estimated_pct"), r.get("measured_pct"), r["slowdown_pct"]) if v is not None]
        return max(vals)

    ok = [r for r in ov_rows if worst_case(r) <= 3.0]
    best = max(ok, key=lambda r: r["coverage"]) if ok else None
    if hs["clean_runs"] or hs["faulty_runs"]:
        L += ["## False alarms in the harm study", "",
              f"- Clean runs ({hs['clean_runs']}, incl. the same-seed repeat): **{hs['clean_alarms']}** in "
              f"{hs['clean_checks']:,} checks",
              f"- Faulty runs ({hs['faulty_runs']}): **{hs['off_fault_alerts']}** unexplained alerts on outputs the "
              "injector did not corrupt (alerts on NaN/Inf/overflowing outputs are the corruption spreading and are "
              "not counted)", ""]
    L += ["## Kill criterion #1", ""]
    every_alarm = ov_alarms + hs["clean_alarms"] + hs["off_fault_alerts"]
    if total is not None:
        alarms = total["false_alarms"] + every_alarm
        L.append(f"- Zero false alarms over ~5 clean GPU-hours: "
                 f"{'MET' if alarms == 0 and total['gpu_hours'] >= 4.5 else 'NOT MET'} "
                 f"({total['false_alarms']} in {total['gpu_hours']:.2f} GPU-hours of the soak; "
                 f"{ov_alarms} during the slowdown runs; {hs['clean_alarms']} in the harm study's clean runs; "
                 f"{hs['off_fault_alerts']} unexplained in its faulty runs)")
    elif ov_rows:
        L.append(f"- False alarms so far (no soak yet): {every_alarm} ({ov_alarms} in {ov_checks:,} slowdown-run "
                 f"checks; {hs['clean_alarms']} in harm-study clean runs; {hs['off_fault_alerts']} unexplained in "
                 "faulty runs)")
    if ov_rows:
        L.append("- Slowdown of 3% or less: " + (
            f"MET at {best['share']:.0%} of matmuls every {best['every']} steps "
            f"({best['coverage']:.1%} of all matmuls checked, {worst_case(best):+.2f}%, the higher of the "
            "estimated and measured numbers)"
            if best else "NOT MET at any tested setting"))
    out = os.path.join(base, "CALIBRATION.md")
    with open(out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return {"soak": total, "overhead": ov_rows, "best": best}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="smoke")
    p.add_argument("--overhead", action="store_true", help="measure slowdown for --sweep settings")
    p.add_argument("--sweep", default="1:1,5:1,10:1,20:1,50:1,1:0.1",
                   help="comma-separated every:share settings")
    p.add_argument("--timing-steps", type=int, default=30, help="plain/checked step pairs per share")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--diag", action="store_true", help="find where false alarms come from")
    p.add_argument("--diag-steps", type=int, default=150)
    p.add_argument("--soak-minutes", type=float, default=0.0)
    p.add_argument("--minutes", type=float, default=0.0, help="(old form) short soak after the overhead sweep")
    p.add_argument("--worker", default=None, help="name for this soak worker (default: random)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--exact-reductions", action="store_true")
    p.add_argument("--fast", action="store_true", help="time fast mode (SDCGuard.protect: linear layers only)")
    p.add_argument("--vocab-pad", type=int, default=0,
                   help="round the vocabulary up to a multiple of this (nanoGPT uses 64) for --overhead/--soak")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--out", default="runs")
    p.add_argument("--data-dir", default="data")
    a = p.parse_args()

    base = os.path.join(a.out, "calibration", a.preset)
    if a.summarize:
        summarize(base, a.preset)
        return
    if a.exact_reductions and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    cfg = dict(PRESETS[a.preset], name=a.preset)
    device = pick_device(a.device)
    dtype = resolve_dtype(device, torch.bfloat16 if a.dtype == "bf16" else torch.float16)
    ds = load_dataset(cfg["dataset"], a.data_dir, cfg.get("ts_train_mb", 200))
    gen = torch.Generator().manual_seed(a.seed)
    print(f"device={device} dtype={dtype} preset={a.preset} env={env_info(device)}", flush=True)
    if a.diag:
        diag(cfg, ds, device, dtype, a.seed, a.diag_steps, os.path.join(a.out, "diag", a.preset))
        if not (a.overhead or a.soak_minutes or a.minutes):
            return
    vocab = -(-ds.vocab // a.vocab_pad) * a.vocab_pad if a.vocab_pad else ds.vocab
    if vocab != ds.vocab:
        print(f"vocabulary padded from {ds.vocab:,} to {vocab:,} (the extra outputs are never targets)", flush=True)
        cfg["vocab_padded_to"] = vocab
    model, opt = make(cfg, vocab, device, a.seed)

    if a.overhead or (a.minutes and not a.soak_minutes):
        res, base_t, timing = overhead(model, opt, ds.train, cfg, device, dtype, gen, parse_sweep(a.sweep),
                                       a.timing_steps, a.rounds, fast=a.fast)
        tag = (env_info(device).get("gpu") or device.type).replace(" ", "_") + ("_fast" if a.fast else "")
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, f"overhead_{tag}.json"), "w") as f:
            json.dump({"env": env_info(device), "preset": a.preset, "cfg": cfg,
                       "base_ms_per_step": round(1000 * base_t, 2), "results": res, "timing": timing},
                      f, indent=1, default=str)
    minutes = a.soak_minutes or a.minutes
    if minutes:
        worker = a.worker or uuid.uuid4().hex[:6]
        path = os.path.join(base, _soak_dir(dtype), f"{worker}_{int(time.time())}.json")
        model, opt = make(cfg, vocab, device, a.seed)             # fresh model: cover a whole training run
        soak(model, opt, ds.train, cfg, device, dtype, gen, minutes, path)
    summarize(base, a.preset)


if __name__ == "__main__":
    main()
