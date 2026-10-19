"""The default check and precise mode on the same training matmuls, from one revision: how small an error each
catches, and what each costs per training step.

Trains the calibration model of experiments/precise_models.py in precise mode (SDCGuard(precise=True) with
protect()) and takes every linear-layer matmul (forward, dX, dW) of the first steps as it happens: its bf16
inputs, its fp32 result before rounding, its bias. On each, at a few random outputs, it adds an error of e times
the typical (root-mean-square) output of that output's 128-column block, with a random sign, before rounding,
for each e in SIZES. Then it checks the same corrupted matmul two ways:

  default   the result rounded to bf16 (what a bf16 matmul returns), checked by checksum.matmul_zscores, the
            check SDCGuard runs by default (its Triton version gives the same scores: tests/_fast_compare.py)
  precise   the fp32 result itself, checked by checksum.matmul_zscores_acc, the check precise mode runs

and records whether each flagged it (score above 6), and whether the error changed the bf16 output at all
(an error that rounds away never reaches the model). Clean matmuls give each check's false alarms and worst
score. Last, the seconds per training step unchecked, with the default check and with precise mode
(precise_models.overhead, same model).

Reporting, written before the first GPU run (Oct 19, 2026): every error size, both checks, all trials, per GPU,
with 95% intervals; the size at which each check flags half and 90% of the errors; the same among errors that
changed the bf16 output; clean false alarms and worst scores; step times. No pass line and no re-run with other
sizes. Precise mode stays experimental whatever this shows (it missed its held-out criteria on Oct 19); this
measures what its extra sensitivity buys against its cost.

    python experiments/mode_compare.py                                   # CPU, tiny model: a smoke test
    python experiments/mode_compare.py --device cuda --out results/mode_compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from sdcguard import SDCGuard  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

ALARM = 6.0
SIZES = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)   # as experiments/sensitivity.py
CONFIG = {"cuda": dict(vocab=32000, layers=8, width=1024, heads=16, seq=512, batch=8, seed=11, steps=2, trials=3),
          "cpu": dict(vocab=512, layers=2, width=128, heads=4, seq=64, batch=4, seed=11, steps=1, trials=1)}


def code_digest() -> str:
    """SHA-256 of the package and this script, so every number here is tied to the code that produced it."""
    h = hashlib.sha256()
    pkg = os.path.join(HERE, "..", "src", "sdcguard")
    for f in sorted(os.listdir(pkg)) + ["../../experiments/mode_compare.py"]:
        p = os.path.normpath(os.path.join(pkg, f))
        if p.endswith(".py") and os.path.isfile(p):
            h.update(os.path.basename(p).encode() + b"\0" + open(p, "rb").read() + b"\0")
    return h.hexdigest()


def _wilson(k: int, n: int) -> tuple:
    """95% Wilson interval of a proportion k/n."""
    if n == 0:
        return (None, None)
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - w), 4), round(min(1.0, c + w), 4))


def _crossing(rates: list, level: float):
    """The error size at which a check flags `level` of the trials (log-linear between measured sizes)."""
    for (s0, r0), (s1, r1) in zip(zip(SIZES, rates), zip(SIZES[1:], rates[1:])):
        if r0 is not None and r1 is not None and r0 < level <= r1:
            t = (level - r0) / (r1 - r0)
            return float(f"{math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0))):.2g}")
    return SIZES[0] if rates and rates[0] is not None and rates[0] >= level else None


def _inject_and_check(A, B, C32, bias, nfma, epoch, low, gen, trials):
    """Queue the clean and corrupted checks of one matmul (no host sync per check). Returns the queued stats
    and, per trial and size, whether the bf16 output changed (a GPU bool, read later)."""
    m, n = C32.shape
    base = C32.to(low)
    queued = [("clean", None, cs.matmul_zscores(A, B, base, bias, epoch=epoch),
               cs.matmul_zscores_acc(A, B, C32, bias, nfma=nfma, epoch=epoch), None)]
    c = C32.clone()
    for _ in range(trials):
        i = int(torch.randint(0, m, (1,), generator=gen))
        j = int(torch.randint(0, n, (1,), generator=gen))
        sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
        lo = (j // 128) * 128
        blk = C32[i, lo:min(n, lo + 128)].double()
        typical = float(blk.norm()) / math.sqrt(blk.numel())
        if not (typical > 0 and math.isfinite(typical)):
            continue
        v = C32[i, j].clone()
        for s_i, e in enumerate(SIZES):
            c[i, j] = v + sign * e * typical
            cl = c.to(low)
            changed = cl[i, j] != base[i, j]
            queued.append(("err", s_i, cs.matmul_zscores(A, B, cl, bias, epoch=epoch),
                           cs.matmul_zscores_acc(A, B, c, bias, nfma=nfma, epoch=epoch), changed))
        c[i, j] = v
    return queued


def run(dev: torch.device, cfg: dict, say=print, max_matmuls: int = 0) -> dict:
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    torch.manual_seed(cfg["seed"])
    model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    pos = torch.Generator().manual_seed(cfg["seed"] + 2)
    g = SDCGuard(every=1, precise=True)
    k_sizes = len(SIZES)
    tally = {key: {"flag": [0] * k_sizes, "flag_changed": [0] * k_sizes, "unjudged": [0] * k_sizes}
             for key in ("default", "precise")}
    trials = [0] * k_sizes
    changed = [0] * k_sizes
    clean = {key: {"checks": 0, "false_alarms": 0, "unjudged": 0, "worst_z": 0.0} for key in ("default", "precise")}
    by_pass: dict = {}
    seen = {"matmuls": 0, "skipped": 0}

    def drain(queued, key_pass):
        stats = [s for q in queued for s in (q[2], q[3]) if s is not None]
        res = iter(cs.resolve(stats))
        bits = [q[4] for q in queued if q[4] is not None]
        bits = iter(torch.stack(bits).tolist() if bits else [])                     # one read for the changed bits
        flags = [bool(next(bits)) if q[4] is not None else None for q in queued]
        for q, ch in zip(queued, flags):
            kind, s_i = q[0], q[1]
            zs = {}
            for key, st in (("default", q[2]), ("precise", q[3])):
                zs[key] = float(next(res).max_z) if st is not None else None
            if kind == "clean":
                for key, z in zs.items():
                    if z is None:
                        continue
                    clean[key]["checks"] += 1
                    clean[key]["false_alarms"] += z > ALARM
                    clean[key]["unjudged"] += z < 0
                    clean[key]["worst_z"] = max(clean[key]["worst_z"], z)
                continue
            trials[s_i] += 1
            changed[s_i] += ch
            bp = by_pass.setdefault(key_pass, {"trials": [0] * k_sizes, "default": [0] * k_sizes,
                                               "precise": [0] * k_sizes})
            bp["trials"][s_i] += 1
            for key, z in zs.items():
                hit = z is not None and z > ALARM
                tally[key]["flag"][s_i] += hit
                tally[key]["flag_changed"][s_i] += hit and ch
                tally[key]["unjudged"][s_i] += z is not None and z < 0
                bp[key][s_i] += hit

    def grab(self, name, pass_, A, B, C32, bias):
        if state["active"] and C32.dim() == 2 and A.dim() == 2 and (not max_matmuls or seen["matmuls"] < max_matmuls):
            with torch.no_grad(), torch.autocast(dev.type, enabled=False):
                q = _inject_and_check(A, B, C32, bias, self._tensor_core_steps(C32, A.dtype), self._epoch, low, pos,
                                      cfg["trials"])
                drain(q, f"{name.split('.')[-1]} {pass_}")
            seen["matmuls"] += 1
        elif state["active"]:
            seen["skipped"] += 1
        return orig(self, name, pass_, A, B, C32, bias)

    state = {"active": True}
    orig = SDCGuard._accumulator
    SDCGuard._accumulator = grab
    t0 = time.time()
    try:
        g.protect(model)
        for step in range(cfg["steps"]):
            idx = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), generator=gen).to(dev)
            with g.on_step(step):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            say(f"step {step}: {seen['matmuls']} matmuls, {sum(trials)} corrupted checks of each kind")
    finally:
        state["active"] = False
        g.unprotect()
        SDCGuard._accumulator = orig
    seconds = round(time.time() - t0, 1)

    def rates(counts, denom):
        return [None if d == 0 else c / d for c, d in zip(counts, denom)]
    out = {"sizes": list(SIZES), "trials": trials, "matmuls": seen["matmuls"], "matmuls_not_2d": seen["skipped"],
           "changed_bf16_output": rates(changed, trials), "clean": clean, "dtype": str(low).replace("torch.", ""),
           "config": cfg, "seconds_sensitivity": seconds, "guard_stats": dict(g.stats)}
    for key in ("default", "precise"):
        fr = rates(tally[key]["flag"], trials)
        fc = rates(tally[key]["flag_changed"], changed)
        out[key] = {"flagged": fr, "flagged_ci95": [_wilson(k, n) for k, n in zip(tally[key]["flag"], trials)],
                    "flagged_among_changed": fc,
                    "flagged_among_changed_ci95": [_wilson(k, n) for k, n in zip(tally[key]["flag_changed"], changed)],
                    "unjudged": tally[key]["unjudged"],
                    "half_caught_at": _crossing(fr, 0.5), "most_caught_at": _crossing(fr, 0.9),
                    "half_caught_at_among_changed": _crossing(fc, 0.5),
                    "most_caught_at_among_changed": _crossing(fc, 0.9)}
    out["by_layer_and_pass"] = {k: {kk: (vv if kk == "trials" else rates(vv, v["trials"])) for kk, vv in v.items()}
                                for k, v in sorted(by_pass.items())}
    return out


def markdown(r: dict) -> str:
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.0f}%"

    def size(e):
        return "none measured" if e is None else f"{100 * e:.4g}%"
    lo = r["dtype"]
    out = [f"# Default check vs precise mode, same matmuls ({r.get('gpu', r.get('device'))})", "",
           f"Code {r.get('code_sha256', '?')[:12]}, torch {r.get('torch')}. {r['matmuls']} linear-layer matmuls "
           f"(forward, dX, dW) of {r['config']['steps']} training steps, {r['config']['trials']} random outputs each, "
           f"so {r['trials'][0]} corrupted matmuls per error size. An error is relative to a typical output of "
           f"its 128-column block, added before rounding to {lo}.", "",
           f"| error size | default check flags | precise mode flags | changed the {lo} output |", "|---|---|---|---|"]
    for i, e in enumerate(r["sizes"]):
        out.append(f"| {size(e)} | {pct(r['default']['flagged'][i])} | {pct(r['precise']['flagged'][i])} | "
                   f"{pct(r['changed_bf16_output'][i])} |")
    d, p = r["default"], r["precise"]
    out += ["", f"Half the errors are flagged from about {size(d['half_caught_at'])} (default) and "
                f"{size(p['half_caught_at'])} (precise); 90% from {size(d['most_caught_at'])} and "
                f"{size(p['most_caught_at'])}.",
            f"Among errors that changed the {lo} output (the ones the model sees): half from "
            f"{size(d['half_caught_at_among_changed'])} (default) and {size(p['half_caught_at_among_changed'])} "
            f"(precise); 90% from {size(d['most_caught_at_among_changed'])} and "
            f"{size(p['most_caught_at_among_changed'])}."]
    c = r["clean"]
    out.append(f"Clean matmuls: default {c['default']['false_alarms']} false alarms in {c['default']['checks']} "
               f"(worst score {c['default']['worst_z']:.2f}), precise {c['precise']['false_alarms']} in "
               f"{c['precise']['checks']} (worst {c['precise']['worst_z']:.2f}); the alarm is at 6.")
    oh = r.get("overhead")
    if oh:
        s = oh["seconds_per_step"]
        out += ["", "| per training step | seconds | vs unchecked |", "|---|---|---|"]
        for k, v in s.items():
            out.append(f"| {k} | {v} | {'' if k == 'unchecked' else '+' + str(oh['overhead_pct'][k]) + '%'} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="folder for <device>.json and .md")
    ap.add_argument("--max-matmuls", type=int, default=0, help="stop corrupting after this many matmuls (0: all)")
    ap.add_argument("--no-overhead", action="store_true", help="skip the step-time measurement")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    cfg = dict(CONFIG["cuda" if dev.type == "cuda" else "cpu"])
    r = run(dev, cfg, max_matmuls=a.max_matmuls)
    r.update(device=str(dev), torch=torch.__version__, code_sha256=code_digest(),
             gpu=torch.cuda.get_device_name(dev) if dev.type == "cuda" else (platform.processor() or "CPU"))
    if dev.type == "cuda":
        r["cc"] = list(torch.cuda.get_device_capability(dev))
    if not a.no_overhead:
        import precise_models
        r["overhead"] = precise_models.overhead(dev, cfg)
    md = markdown(r)
    print(md, flush=True)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        name = (r["gpu"] if dev.type == "cuda" else "cpu").replace(" ", "_").replace("/", "_")
        with open(os.path.join(a.out, f"{name}.json"), "w") as f:
            json.dump(r, f, indent=1)
        with open(os.path.join(a.out, f"{name}.md"), "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
