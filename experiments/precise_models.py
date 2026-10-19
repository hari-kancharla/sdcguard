"""Precise mode's noise model against real training matmuls: the current model and two candidates, scored on
every fp32-accumulator check of 30 clean training steps.

On Oct 19, 2026 precise mode raised false alarms on an A100, H100 and B200 (worst score 17.9 to 20.3 in 30
steps). experiments/precise_diag.py traced them to the backward dW matmuls of the attention input (qkv) and
of the output layer, where per-output errors were 3 to 5 times what the model allows; even plain fp32 FMAs
scored up to 2.8 there. The model scales the accumulation's rounding with the final output |C|, but the
rounding happens at every partial sum along k: when the running sum wanders far beyond where it ends up
(tokens whose contributions cancel), |C| understates it.

For every check this computes the block checksums' distance from the float64 answer (as the check does) and
scores it under:
  current     u (|a| |B_block| + (sqrt(k) + k/N + sqrt(128)) |C_block|)
  running     the same with |C| replaced by P = max(|C|, mean over S points along k of |running sum|),
              the running sums measured with S partial matmuls on the GPU (S = 4, 8 and 16)
  sqrt_steps  the current model with the |a| |B| term multiplied by sqrt(k / N)
and reports alarms, worst and 99.9th-percentile scores, and how small an error each model typically catches
(6 x noise / typical output, median and 90th percentile over checks), so a fix that kills the false alarms
also has to show what it costs.

    python experiments/precise_models.py --device cuda --steps 30 --out results/precise_models
    python experiments/precise_models.py                     # CPU, tiny model: a smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sdcguard import SDCGuard  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

CONFIG = {"cuda": dict(vocab=32000, layers=8, width=1024, heads=16, seq=512, batch=8),
          "cpu": dict(vocab=512, layers=2, width=128, heads=4, seq=64, batch=4)}
BLOCK = 128
ALARM = 6.0
SEGMENTS = 16                      # running sums measured at 16 points; 8 and 4 use every 2nd and 4th


def _mm32(a, b):
    if a.is_cuda:
        return torch.ops.aten.mm.dtype(a, b, torch.float32)
    return a.float() @ b.float()


def running_scale(A, B, C):
    """P = max(|C|, mean of |running sum| at the segment ends), for S = 16, 8 and 4."""
    k = A.shape[1]
    edges = [round(k * (i + 1) / SEGMENTS) for i in range(SEGMENTS)]
    Q = torch.zeros_like(C)
    acc = {16: torch.zeros_like(C), 8: torch.zeros_like(C), 4: torch.zeros_like(C)}
    start = 0
    for i, end in enumerate(edges):
        if end > start:
            Q += _mm32(A[:, start:end], B[start:end])
        start = end
        aq = Q.abs()
        acc[16] += aq
        if (i + 1) % 2 == 0:
            acc[8] += aq
        if (i + 1) % 4 == 0:
            acc[4] += aq
    c = C.abs()
    return {s: torch.maximum(c, acc[s] / s) for s in acc}


def score(A, B, C, bias, nfma, epoch, running: bool = True):
    """Per (row, block): the checksums' distance from float64, and each model's noise."""
    u = cs.ACC_UNIT
    m, k = A.shape
    n = C.shape[1]
    nblk = n // BLOCK
    if nblk == 0:
        return None
    w = n - n % BLOCK
    Cb = C[:, :w].reshape(m, nblk, BLOCK)
    Bb = B[:, :w].reshape(k, nblk, BLOCK)
    B2 = cs._pair_sums64(Bb, epoch)                                  # (k, nblk, 2)
    expected = (A.double() @ B2.reshape(k, nblk * 2)).view(m, nblk, 2)
    if bias is not None:
        expected = expected + cs._pair_sums64(bias[:w].reshape(nblk, BLOCK), epoch)
    actual = cs._pair_sums64(Cb, epoch)
    diff = (expected - actual).abs().amax(-1)                        # (m, nblk): worse of plain and signed
    steps = k / nfma if nfma else 0.0
    per_out = math.sqrt(k) + steps + math.sqrt(BLOCK)
    a_norm = A.double().norm(dim=1)[:, None]
    b_frob = Bb.double().pow(2).sum(dim=(0, 2)).sqrt()[None, :]
    prod = u * a_norm * b_frob
    cnorm = Cb.double().norm(dim=-1)
    noise = {"current": prod + u * per_out * cnorm,                              # before Oct 19 (no sqrt(k))
             "sqrt_steps": prod * math.sqrt(max(1.0, steps or k)) + u * per_out * cnorm,
             # what checksum._acc_zscores computes since Oct 19 (frozen; see its comment)
             "shipped": prod * (cs.ACC_C1 * math.sqrt(k)) + u * per_out * cnorm}
    if running:
        for s, P in running_scale(A, B, C).items():
            noise[f"running_S{s}"] = prod + u * per_out * P[:, :w].reshape(m, nblk, BLOCK).double().norm(dim=-1)
    typical = (cnorm / math.sqrt(BLOCK)).clamp_min(1e-30)
    out = {}
    for name, nz in noise.items():
        z = diff / nz.clamp_min(1e-300)
        caught = (ALARM * nz / typical)
        out[name] = (float(z.max()), caught.flatten())
    return out


# Held-out workloads for validating the model frozen on Oct 19, 2026 (checksum.py: C1 sqrt(k) |a| |B|). None of
# them was used to choose it: other shapes, seeds, learning rates, and token streams with a word-like (Zipf)
# frequency, which makes the output layer's gradients far more lopsided than uniform random tokens.
# Pre-registered criteria (METHOD.md): 0 alarms in every check on every GPU, worst score at most 4.
VALIDATE = {
    "cuda": [dict(name="wide, 8,192 tokens, Zipf tokens", vocab=50257, layers=4, width=2048, heads=16, seq=1024,
                  batch=8, seed=7, lr=1e-3, zipf=True, steps=12),
             dict(name="calibration shape, new seed, Zipf tokens", vocab=32000, layers=8, width=1024, heads=16,
                  seq=512, batch=8, seed=11, lr=3e-4, zipf=True, steps=20),
             dict(name="narrow and deep, 8,192 tokens", vocab=8192, layers=12, width=512, heads=8, seq=256,
                  batch=32, seed=23, lr=2e-3, zipf=False, steps=12)],
    "cpu": [dict(name="tiny, Zipf tokens", vocab=512, layers=2, width=128, heads=4, seq=64, batch=4, seed=7,
                 lr=1e-3, zipf=True, steps=2)],
}


def _tokens(cfg: dict, gen: torch.Generator, shape) -> torch.Tensor:
    if not cfg.get("zipf"):
        return torch.randint(0, cfg["vocab"], shape, generator=gen)
    w = 1.0 / torch.arange(1, cfg["vocab"] + 1, dtype=torch.float64)          # Zipf, exponent 1
    return torch.multinomial(w, int(torch.tensor(shape).prod()), replacement=True, generator=gen).view(*shape)


def run(dev: torch.device, steps: int, seed: int = 0, say=print, cfg: dict = None, models: tuple = None) -> dict:
    cfg = dict(cfg or CONFIG["cuda" if dev.type == "cuda" else "cpu"])
    seed = cfg.get("seed", seed)
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    torch.manual_seed(seed)
    model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
    # This fixed workload has trainable inputs/weights at every Linear: forward, dX and dW.
    expected_checks = 3 * steps * sum(isinstance(layer, torch.nn.Linear) for layer in model.modules())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4))
    gen = torch.Generator().manual_seed(seed + 1)
    g = SDCGuard(every=1, precise=True)
    worst: dict = {}
    zs: dict = {}
    caught: dict = {}
    by_pass: dict = {}
    state = {"step": 0}

    def grab(self, name, pass_, A, B, C32, bias):
        nfma = self._tensor_core_steps(C32, A.dtype)
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            r = score(A, B, C32, bias, nfma, self._epoch, running=models is None or any("running" in x for x in models))
            st = cs.matmul_zscores_acc(A, B, C32, bias, nfma=nfma, epoch=self._epoch)
        if r is not None and st is not None:     # the check exactly as precise mode runs it today
            r = {"current_as_shipped": (float(cs.resolve([st])[0].max_z), r["shipped"][1]), **r}
        if r is not None and models is not None:
            r = {k: v for k, v in r.items() if k in models}
        if r is not None:
            key = f"{name.split('.')[-1]} {pass_}"
            for model_name, (z, cau) in r.items():
                zs.setdefault(model_name, []).append(z)
                worst[model_name] = max(worst.get(model_name, 0.0), z)
                c = caught.setdefault(model_name, [])
                if len(c) < 400:
                    c.append(cau[torch.randint(0, cau.numel(), (64,), device=cau.device)].cpu())
                bp = by_pass.setdefault(key, {})
                bp[model_name] = max(bp.get(model_name, 0.0), z)
        return orig(self, name, pass_, A, B, C32, bias)

    orig = SDCGuard._accumulator
    SDCGuard._accumulator = grab
    try:
        g.protect(model)
        for n in range(steps):
            state["step"] = n
            idx = _tokens(cfg, gen, (cfg["batch"], cfg["seq"] + 1)).to(dev)
            with g.on_step(n):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if n % 5 == 4 or n == steps - 1:
                say(f"step {n}: worst {({k: round(v, 2) for k, v in worst.items()})}")
    finally:
        g.unprotect()
        SDCGuard._accumulator = orig

    summary = {}
    for name, z in zs.items():
        zt = torch.tensor(z)
        ct = torch.cat(caught[name]).float()
        summary[name] = {"checks": len(z), "alarms": int((zt > ALARM).sum()),
                         "unjudged": int(((zt < 0) | ~torch.isfinite(zt)).sum()),
                         "worst": round(float(zt.max()), 3),
                         "p99_9": round(float(zt.kthvalue(max(1, int(0.999 * len(z)))).values), 3),
                         "typical_error_caught_median": float(f"{float(ct.median()):.3g}"),
                         "typical_error_caught_p90": float(f"{float(ct.kthvalue(max(1, int(0.9 * ct.numel()))).values):.3g}")}
    out = {"config": cfg, "dtype": str(low).replace("torch.", ""), "steps": steps, "models": summary,
           "guard_stats": dict(g.stats), "expected_checks": expected_checks,
           "lr": cfg.get("lr", 3e-4),
           "worst_by_layer_and_pass": dict(sorted(by_pass.items(), key=lambda kv: -kv[1]["current"]))}
    if dev.type == "cuda":
        out.update(gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)),
                   torch=torch.__version__)
    return out


def overhead(dev: torch.device, cfg: dict, steps: int = 6) -> dict:
    """Seconds per training step of one model: unchecked, with every matmul of every step checked by the
    default check, and by precise mode (both through protect(), the fast mode). Warm-up steps not timed."""
    import time as _t
    low = torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
    out = {}
    for label, make in (("unchecked", lambda: None), ("default check, every step", lambda: SDCGuard(every=1)),
                        ("precise mode, every step", lambda: SDCGuard(every=1, precise=True))):
        torch.manual_seed(0)
        model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        g = make()
        if g is not None:
            g.protect(model)
        gen = torch.Generator().manual_seed(1)
        times = []
        for n in range(steps + 2):
            idx = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), generator=gen).to(dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = _t.perf_counter()
            with (g.on_step(n) if g is not None else torch.autograd.set_grad_enabled(True)):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            if n >= 2:
                times.append(_t.perf_counter() - t0)
        times.sort()
        out[label] = round(times[len(times) // 2], 4)
    base = out["unchecked"]
    return {"seconds_per_step": out, "overhead_pct": {k: round(100 * (v / base - 1), 1) for k, v in out.items()
                                                      if k != "unchecked"}}


def validate(dev: torch.device, say=print) -> dict:
    """The held-out run: the frozen model (as shipped) and the earlier candidates on workloads none of them was
    chosen on, plus what precise mode costs."""
    res, skipped = [], []
    for cfg in VALIDATE["cuda" if dev.type == "cuda" else "cpu"]:
        say(f"validate: {cfg['name']}")
        try:
            r = run(dev, cfg["steps"], say=lambda s, n=cfg["name"]: say(f"{n}: {s}"), cfg=cfg,
                    models=("current_as_shipped", "shipped", "current", "sqrt_steps"))
        except torch.cuda.OutOfMemoryError as e:
            # one workload too big for this GPU (the wide one on a 16 GB T4, Oct 19) shouldn't lose the others
            skipped.append({"name": cfg["name"], "error": f"did not fit in GPU memory: {str(e).splitlines()[0][:160]}"})
            say(f"{cfg['name']}: did not fit in GPU memory, skipped")
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            continue
        r["name"] = cfg["name"]
        res.append(r)
    measured = [r["models"]["current_as_shipped"] for r in res]
    worst = max((r["worst"] for r in measured), default=0.0)
    alarms = sum(r["alarms"] for r in measured)
    unjudged = sum(r["unjudged"] for r in measured)
    # An absent check or an unknown score cannot establish a clean validation run, and neither can a workload
    # that didn't run. Inspect every workload: Python max() can conceal a later NaN.
    complete = bool(measured) and not skipped and all(
        r["checks"] > 0 and r["unjudged"] == 0 and math.isfinite(r["worst"]) and 0 <= r["worst"] <= 4.0
        for r in measured
    )
    # The diagnostic calculation above is independent of the production guard queue.
    # Both must have executed all checks; diagnostic-only scores cannot validate protection.
    complete = complete and all(
        workload.get("expected_checks", 0) == observed["checks"]
        and workload.get("guard_stats", {}).get("checked") == observed["checks"]
        and workload.get("guard_stats", {}).get("checked_fp32") == observed["checks"]
        and all(workload.get("guard_stats", {}).get(key) == 0
                for key in ("unjudged", "skipped_oom", "skipped_dtype", "alerts"))
        for workload, observed in zip(res, measured)
    )
    out = {"workloads": res, "skipped": skipped,
           "as_shipped": {"alarms": alarms, "worst": worst, "complete": not skipped,
                          "checks": sum(r["checks"] for r in measured), "unjudged": unjudged,
                          "criteria_met": complete and alarms == 0}}
    try:
        out["overhead"] = overhead(dev, VALIDATE["cuda" if dev.type == "cuda" else "cpu"][-1 if dev.type != "cuda" else 1])
    except Exception as e:  # noqa: BLE001
        out["overhead"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    if dev.type == "cuda":
        out.update(gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)),
                   torch=torch.__version__)
    return out


def validate_markdown(v: dict) -> str:
    a = v["as_shipped"]
    lines = [f"## {v.get('gpu', 'CPU')}: {a['checks']} held-out checks, {a['alarms']} alarms, "
             f"{a.get('unjudged', 0)} unjudged, worst {a['worst']} "
             f"(criteria {'MET' if a['criteria_met'] else 'NOT MET'}: 0 alarms, worst at most 4"
             f"{'' if a.get('complete', True) else ', and every workload run'})", ""]
    for sk in v.get("skipped", []):
        lines += [f"### {sk['name']}: not run ({sk['error']})", ""]
    for r in v["workloads"]:
        lines.append(markdown(r).replace("## ", "### " + r["name"] + ": ", 1))
    if v.get("overhead"):
        lines.append(f"Overhead (median seconds per step, same model): {json.dumps(v['overhead'])}\n")
    return "\n".join(lines)


def markdown(r: dict) -> str:
    lines = [f"## {r.get('gpu', 'CPU')} ({r['dtype']}), {r['steps']} clean training steps", "",
             "| model | checks | alarms | worst | 99.9th pct | typical error caught (median, 90th pct) |",
             "|---|---|---|---|---|---|"]
    for name, s in r["models"].items():
        lines.append(f"| {name} | {s['checks']} | {s['alarms']} | {s['worst']} | {s['p99_9']} | "
                     f"{100 * s['typical_error_caught_median']:.3g}%, {100 * s['typical_error_caught_p90']:.3g}% |")
    lines += ["", "Worst score by layer and pass:", "",
              "| layer, pass | " + " | ".join(r["models"]) + " |", "|---|" + "---|" * len(r["models"])]
    for key, d in r["worst_by_layer_and_pass"].items():
        lines.append(f"| {key} | " + " | ".join(f"{d.get(m, 0):.2f}" for m in r["models"]) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true", help="the held-out run of the frozen model, and its cost")
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.validate:
        v = validate(torch.device(a.device))
        print(validate_markdown(v))
        return 0 if v["as_shipped"]["criteria_met"] else 1
    res = run(torch.device(a.device), a.steps)
    res["seconds"] = round(time.time() - t0, 1)
    print(markdown(res))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{(res.get('gpu') or 'cpu').replace(' ', '_')}.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
