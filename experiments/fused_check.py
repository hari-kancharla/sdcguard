"""The fused precise check (src/sdcguard/fused.py) on a GPU: is it right, is it the same check as precise mode,
and what does it cost? Criteria are in docs/METHOD.md, written before the first run.

1. correctness: tests/_fused_compare.py on the GPU (outputs, block statistics, clean scores, located errors,
   an error below bf16's resolution, the guard end to end).
2. same matmuls: clean training steps with SDCGuard(precise="fused"); every fused check is also scored the
   way precise=True scores it (a cuBLAS fp32-output matmul of the same inputs), so both checks see the same
   matmuls. Alarms, worst scores, and how far the two scores are apart.
3. sensitivity: errors of a given size (a share of a typical output) put into one output of real-sized
   matmuls, before rounding: how often the fused check, precise mode, and the default check of the rounded
   bf16 output flag them.
4. timing: one matmul at real sizes: cuBLAS bf16, the fused kernel (a few tile shapes), the check from its
   statistics, and precise mode's fp32-output matmul plus check.
5. overhead: seconds per training step, unchecked and with every step checked by the default check,
   precise=True and precise="fused".
6. gradients: one training step of the calibration-shape model with precise=True and with precise="fused", same
   weights and data: how far apart each parameter's gradient is (the fused kernel sums in another order).

What this checks is the Triton kernel's own matmuls: on checked steps precise="fused" computes the protected
layers itself, so cuBLAS is not exercised on those calls and a cuBLAS defect (like known issue 6862950) can't
show there. The runs record how many matmuls went through the kernel.

    python experiments/fused_check.py --device cuda --out results/fused
    TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1 python experiments/fused_check.py --device cpu --quick
                                                                # Triton's CPU interpreter: a smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from sdcguard import SDCGuard, fused  # noqa: E402
from sdcguard import checksum as cs  # noqa: E402
from sdcguard.guard import _acc_mm  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

ALARM = 6.0
WORKLOADS = {
    "cuda": [dict(name="calibration shape, seed 41, Zipf tokens", vocab=32000, layers=8, width=1024, heads=16,
                  seq=512, batch=8, seed=41, lr=3e-4, zipf=True, steps=10),
             dict(name="wide, 8,192 tokens, seed 43, Zipf tokens", vocab=50257, layers=4, width=2048, heads=16,
                  seq=1024, batch=8, seed=43, lr=1e-3, zipf=True, steps=6)],
    "cpu": [dict(name="tiny", vocab=512, layers=1, width=128, heads=4, seq=32, batch=2, seed=41, lr=1e-3,
                 zipf=False, steps=1)],
}
SHAPES = {"cuda": [(4096, 1024, 3072), (4096, 1024, 4096), (4096, 4096, 1024), (3072, 4096, 1024),
                   (4096, 1024, 32000), (8192, 2048, 8192)],
          "cpu": [(64, 96, 200)]}
TILES = [(64, 32, 4), (64, 32, 8), (128, 32, 8), (64, 64, 4), (128, 64, 8)]       # (bm, bk, warps), bn = 128
FRACS = [3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1]


def _low(dev):
    return torch.bfloat16 if dev.type != "cuda" or torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16


def _tokens(cfg, gen, shape):
    if not cfg.get("zipf"):
        return torch.randint(0, cfg["vocab"], shape, generator=gen)
    w = 1.0 / torch.arange(1, cfg["vocab"] + 1, dtype=torch.float64)
    return torch.multinomial(w, int(torch.tensor(shape).prod()), replacement=True, generator=gen).view(*shape)


def correctness(dev) -> dict:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(HERE, "..", "src"),
                                                       os.environ.get("PYTHONPATH", "")]))
    if dev.type == "cpu":
        env.update(TRITON_INTERPRET="1", SDCGUARD_FAST_CPU_TEST="1")
    p = subprocess.run([sys.executable, os.path.join(HERE, "..", "tests", "_fused_compare.py"), dev.type],
                       env=env, capture_output=True, text=True, timeout=1800)
    return {"passed": p.returncode == 0 and "ALL AGREE" in p.stdout, "exit": p.returncode,
            "output": p.stdout + p.stderr}


def same_matmuls(dev, cfg, say=print) -> dict:
    low = _low(dev)
    torch.manual_seed(cfg["seed"])
    model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    g = SDCGuard(every=1, precise="fused")
    rows = []
    orig = SDCGuard._fused_mm

    def grab(self, name, pass_, A, B, bias):
        n0 = len(self._pending)
        C = orig(self, name, pass_, A, B, bias)
        if C is None or len(self._pending) == n0:
            return C
        zf = cs.resolve([self._pending[-1][1]])[0].max_z
        with torch.no_grad(), torch.autocast(dev.type, enabled=False):
            Bc = B.to(A.dtype)
            C32 = _acc_mm(A, Bc)
            if bias is not None:
                C32 = C32.add_(bias.float())
            nfma = self._tensor_core_steps(C32, A.dtype)
            zp = cs.resolve([cs.matmul_zscores_acc(A, Bc, C32, bias, nfma=nfma, epoch=self._epoch)])[0].max_z
            differ = float((C != C32.to(C.dtype)).float().mean())
        rows.append({"where": f"{name.split('.')[-1]} {pass_}", "fused": zf, "precise": zp, "bits_differ": differ})
        return C

    SDCGuard._fused_mm = grab
    try:
        g.protect(model)
        for n in range(cfg["steps"]):
            idx = _tokens(cfg, gen, (cfg["batch"], cfg["seq"] + 1)).to(dev)
            with g.on_step(n):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        say(f"{cfg['name']}: {len(rows)} fused checks, guard {g.stats}")
    finally:
        g.unprotect()
        SDCGuard._fused_mm = orig
    zf = torch.tensor([r["fused"] for r in rows], dtype=torch.float64)
    zp = torch.tensor([r["precise"] for r in rows], dtype=torch.float64)
    ok = torch.isfinite(zf) & torch.isfinite(zp) & (zf >= 0) & (zp >= 0)
    worst_by = {}
    for r in rows:
        w = worst_by.setdefault(r["where"], {"fused": 0.0, "precise": 0.0})
        w["fused"], w["precise"] = max(w["fused"], r["fused"]), max(w["precise"], r["precise"])
    return {"name": cfg["name"], "checks": len(rows), "guard_stats": dict(g.stats),
            "fused_alarms": int((zf > ALARM).sum()), "precise_alarms": int((zp > ALARM).sum()),
            "fused_unjudged": int((~torch.isfinite(zf) | (zf < 0)).sum()),
            "fused_worst": round(float(zf.max()), 3) if len(rows) else None,
            "precise_worst": round(float(zp.max()), 3) if len(rows) else None,
            "largest_score_gap": round(float((zf[ok] - zp[ok]).abs().max()), 3) if bool(ok.any()) else None,
            "median_share_of_outputs_with_other_bits": float(torch.tensor([r["bits_differ"] for r in rows]).median())
            if rows else None,
            "worst_by_layer_and_pass": dict(sorted(worst_by.items(), key=lambda kv: -kv[1]["fused"])[:12])}


def sensitivity(dev, shapes, trials, say=print) -> dict:
    low = _low(dev)
    gen = torch.Generator().manual_seed(5)
    out = []
    for (m, k, n) in shapes:
        A = torch.randn(m, k, generator=gen).to(low).to(dev)
        w = (torch.randn(n, k, generator=gen) / k ** 0.5).to(low).to(dev)
        B = w.t()
        C, _ = fused.fused_mm(A, B)
        typical = float(C.float().abs().median())
        C32 = _acc_mm(A, B)
        nfma = None
        if dev.type == "cuda":
            from sdcguard.modes import n_fma
            nfma = n_fma("bf16" if low == torch.bfloat16 else "fp16", torch.cuda.get_device_capability(dev))
        res = {"shape": [m, k, n], "typical_output": typical, "fractions": {}}
        for frac in FRACS:
            hit = {"fused": 0, "checked": 0, "precise": 0, "default": 0}
            for _ in range(trials):
                r = int(torch.randint(0, m, (1,), generator=gen))
                c = int(torch.randint(0, n, (1,), generator=gen))
                delta = frac * typical * (1 if torch.rand(1, generator=gen).item() < 0.5 else -1)
                _, st = fused.fused_mm(A, B, inject=(r, c, delta))
                hit["fused"] += not cs.resolve([fused.zscores_from_stats(A, B, st, nfma=nfma)])[0].ok
                # 0.4.0: the same kernel with the check scored inside it (no float64)
                _, st2 = fused.fused_mm_checked(A, B, inject=(r, c, delta), nfma=nfma)
                hit["checked"] += not cs.resolve([st2])[0].ok
                Cc = C32.clone()
                Cc[r, c] += delta
                hit["precise"] += not cs.check_matmul_acc(A, B, Cc, nfma=nfma).ok
                hit["default"] += not cs.check_matmul(A, B, Cc.to(low)).ok
            res["fractions"][str(frac)] = {kk: v / trials for kk, v in hit.items()}
        for kk in ("fused", "checked", "precise", "default"):
            res[f"smallest_caught_half_the_time_{kk}"] = next(
                (f for f in FRACS if res["fractions"][str(f)][kk] >= 0.5), None)
        say(f"sensitivity {m}x{k}x{n}: " + ", ".join(f"{kk} {res[f'smallest_caught_half_the_time_{kk}']}"
                                                     for kk in ("fused", "checked", "precise", "default")))
        out.append(res)
    return {"trials": trials, "fractions": FRACS, "shapes": out}


def _bench(dev, fn, reps=20):
    for _ in range(3):
        fn()
    if dev.type != "cuda":
        t0 = time.perf_counter()
        fn()
        return round((time.perf_counter() - t0) * 1e3, 3)
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return round(ts[len(ts) // 2], 4)


def timing(dev, shapes, say=print) -> list:
    low = _low(dev)
    gen = torch.Generator().manual_seed(9)
    out = []
    for (m, k, n) in shapes:
        A = torch.randn(m, k, generator=gen).to(low).to(dev)
        w = (torch.randn(n, k, generator=gen) / k ** 0.5).to(low).to(dev)
        B = w.t()
        row = {"shape": [m, k, n], "cublas_bf16_ms": _bench(dev, lambda: A @ B), "fused_gemm_ms": {}}
        best = None
        for bm, bk, nw in (TILES if dev.type == "cuda" else TILES[:1]):
            try:
                t = _bench(dev, lambda: fused.fused_mm(A, B, bm=bm, bk=bk, num_warps=nw))
            except Exception as e:  # noqa: BLE001  (a tile that doesn't fit this GPU's shared memory)
                row["fused_gemm_ms"][f"{bm}x128x{bk} w{nw}"] = f"{type(e).__name__}: {str(e)[:120]}"
                continue
            row["fused_gemm_ms"][f"{bm}x128x{bk} w{nw}"] = t
            if best is None or t < best[0]:
                best = (t, bm, bk, nw)
        if best is None:
            row["error"] = "no tile shape ran"
            out.append(row)
            continue
        _, bm, bk, nw = best
        _, st = fused.fused_mm(A, B, bm=bm, bk=bk, num_warps=nw)
        row["fused_best_tile"] = f"{bm}x128x{bk} w{nw}"
        row["fused_gemm_best_ms"] = best[0]
        row["check_from_stats_ms"] = _bench(dev, lambda: fused.zscores_from_stats(A, B, st))
        row["fused_total_ms"] = _bench(dev, lambda: fused.zscores_from_stats(
            A, B, fused.fused_mm(A, B, bm=bm, bk=bk, num_warps=nw)[1]))
        row["fused_checked_total_ms"] = _bench(dev, lambda: fused.fused_mm_checked(A, B, bm=bm, bk=bk,
                                                                                   num_warps=nw))
        row["precise_fp32_gemm_ms"] = _bench(dev, lambda: _acc_mm(A, B))
        row["precise_total_ms"] = _bench(dev, lambda: cs.matmul_zscores_acc(A, B, _acc_mm(A, B)))
        say(f"timing {m}x{k}x{n}: cuBLAS {row['cublas_bf16_ms']} ms, fused GEMM {best[0]} ms, check "
            f"{row['check_from_stats_ms']} ms, fused total {row['fused_total_ms']} ms, precise total "
            f"{row['precise_total_ms']} ms")
        out.append(row)
    return out


def overhead(dev, cfg, steps=6, say=print) -> dict:
    low = _low(dev)

    def measure(make):
        torch.manual_seed(0)
        model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        g = make()
        times = []
        try:
            if g is not None:
                g.protect(model)
            gen = torch.Generator().manual_seed(1)
            for n in range(steps + 2):
                idx = torch.randint(0, cfg["vocab"], (cfg["batch"], cfg["seq"] + 1), generator=gen).to(dev)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
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
                    times.append(time.perf_counter() - t0)
        finally:
            if g is not None:
                g.unprotect()
        times.sort()
        return round(times[len(times) // 2], 4), (dict(g.stats) if g is not None else {})

    secs, stats = {}, {}
    for label, make in (("unchecked", lambda: None), ("default check", lambda: SDCGuard(every=1)),
                        ("precise=True", lambda: SDCGuard(every=1, precise=True)),
                        ("precise=fused", lambda: SDCGuard(every=1, precise="fused"))):
        secs[label], stats[label] = measure(make)
        say(f"overhead {label}: {secs[label]} s/step")
    base = secs["unchecked"]
    return {"workload": cfg["name"], "seconds_per_step": secs,
            "overhead_pct": {k: round(100 * (v / base - 1), 1) for k, v in secs.items() if k != "unchecked"},
            "fused_checks_in_timed_run": stats["precise=fused"].get("checked_fused", 0),
            "fused_fallbacks": stats["precise=fused"].get("fused_fallbacks", 0)}


def gradients(dev, cfg, say=print) -> dict:
    """One step, same weights and data, precise=True vs precise="fused": per-parameter gradient differences
    relative to the parameter's largest gradient."""
    low = _low(dev)
    out = {}
    grads = {}
    for mode in (True, "fused"):
        torch.manual_seed(cfg["seed"])
        model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
        g = SDCGuard(every=1, precise=mode).protect(model)
        try:
            idx = _tokens(cfg, torch.Generator().manual_seed(cfg["seed"] + 5), (cfg["batch"], cfg["seq"] + 1)).to(dev)
            with g.on_step(0):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            grads[str(mode)] = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
                                if p.grad is not None}
            out[f"stats_{mode}"] = {k: g.stats.get(k) for k in ("checked", "checked_fp32", "checked_fused", "alerts",
                                                                "fused_fallbacks")}
            out[f"loss_{mode}"] = float(loss)
        finally:
            g.unprotect()
        del model
    a, b = grads["True"], grads["fused"]
    rel = {n: float((a[n] - b[n]).abs().max() / a[n].abs().max().clamp_min(1e-30)) for n in a if n in b}
    out["max_relative_difference"] = max(rel.values()) if rel else None
    out["worst_parameters"] = dict(sorted(rel.items(), key=lambda kv: -kv[1])[:8])
    out["parameters_compared"] = len(rel)
    out["matmuls_in_the_triton_kernel"] = (out.get("stats_fused") or {}).get("checked_fused")
    say(f"gradients: max relative difference {out['max_relative_difference']}, losses {out.get('loss_True')} vs "
        f"{out.get('loss_fused')}")
    return out


def criteria(r: dict) -> dict:
    """The pre-registered criteria (docs/METHOD.md)."""
    sm_all = r.get("same_matmuls", [])
    sm_all = sm_all if isinstance(sm_all, list) else [sm_all]
    sm = [w for w in sm_all if isinstance(w, dict) and "error" not in w]
    c1 = bool((r.get("correctness") or {}).get("passed"))
    c2 = bool(sm) and len(sm) == len(sm_all) and all(
        w["checks"] > 0 and w["fused_alarms"] == 0 and w["fused_unjudged"] == 0 and w["fused_worst"] is not None
        and 0 <= w["fused_worst"] <= 4.0 and w["guard_stats"].get("fused_fallbacks", 0) == 0
        and w["guard_stats"].get("alerts", 0) == 0 for w in sm)
    c3 = None
    shapes = (r.get("sensitivity") or {}).get("shapes", []) if isinstance(r.get("sensitivity"), dict) else []
    if shapes:
        ratios = []
        for s in shapes:
            p = s.get("smallest_caught_half_the_time_precise")
            # 0.4.0: the in-kernel scoring ("checked") must keep the sensitivity too; a result without its own
            # measurement of it (one made before it existed) doesn't meet the criterion for it
            for key in ("fused", "checked"):
                f = s.get(f"smallest_caught_half_the_time_{key}")
                ratios.append(f is not None and p is not None and 0.5 <= f / p <= 2.0)
        c3 = all(ratios)
    ovr = r.get("overhead") if isinstance(r.get("overhead"), dict) else {}
    ov = ovr.get("overhead_pct", {})
    c4 = None
    if "precise=fused" in ov and "precise=True" in ov and ovr.get("fused_checks_in_timed_run", 0) > 0:
        c4 = ov["precise=fused"] <= 0.5 * ov["precise=True"]
    gr = r.get("gradients") if isinstance(r.get("gradients"), dict) else {}
    c5 = None
    if gr.get("max_relative_difference") is not None and gr.get("matmuls_in_the_triton_kernel"):
        c5 = gr["max_relative_difference"] <= 2.0 ** -6
    return {"1 correctness": c1, "2 clean, worst score at most 4": c2,
            "3 same sensitivity as precise mode (within 2x)": c3,
            "4 overhead at most half of precise=True's": c4,
            "5 gradients within 2^-6 of precise=True's (relative to each parameter's largest)": c5,
            "all met": bool(c1 and c2 and c3 and c4 and c5)}


def markdown(r: dict) -> str:
    lines = [f"## {r.get('label', '')} {r.get('gpu', '')}", ""]
    for k, v in r.get("criteria", {}).items():
        lines.append(f"- {k}: {'met' if v is True else 'not met' if v is False else 'not measured'}")
    lines.append("")
    sm_md = r.get("same_matmuls", [])
    for w in (sm_md if isinstance(sm_md, list) else [sm_md]):
        if not isinstance(w, dict) or "error" in w:
            lines.append(f"- {w.get('name', 'same matmuls') if isinstance(w, dict) else 'same matmuls'}: "
                         f"{w.get('error') if isinstance(w, dict) else w}")
            continue
        lines.append(f"- {w['name']}: {w['checks']} checks, fused alarms {w['fused_alarms']} (worst "
                     f"{w['fused_worst']}), precise alarms {w['precise_alarms']} (worst {w['precise_worst']}), "
                     f"largest gap {w['largest_score_gap']}")
    for s in ((r.get("sensitivity") or {}).get("shapes", []) if isinstance(r.get("sensitivity"), dict) else []):
        lines.append(f"- {tuple(s['shape'])}: smallest error caught half the time (share of a typical output): "
                     f"fused {s['smallest_caught_half_the_time_fused']}, in-kernel scoring "
                     f"{s.get('smallest_caught_half_the_time_checked')}, precise "
                     f"{s['smallest_caught_half_the_time_precise']}, default "
                     f"{s['smallest_caught_half_the_time_default']}")
    for t in (r.get("timing") if isinstance(r.get("timing"), list) else []):
        if "error" in t:
            continue
        lines.append(f"- {tuple(t['shape'])} ms: cuBLAS bf16 {t['cublas_bf16_ms']}, fused GEMM "
                     f"{t['fused_gemm_best_ms']} ({t['fused_best_tile']}), check from stats "
                     f"{t['check_from_stats_ms']}, fused total {t['fused_total_ms']}, in-kernel scoring total "
                     f"{t.get('fused_checked_total_ms')}, precise total {t['precise_total_ms']}")
    gr = r.get("gradients") if isinstance(r.get("gradients"), dict) else {}
    if gr.get("max_relative_difference") is not None:
        lines.append(f"- gradients vs precise=True: max relative difference {gr['max_relative_difference']:.3g}; "
                     f"{gr.get('matmuls_in_the_triton_kernel')} matmuls ran in the Triton kernel, not cuBLAS")
    ov = r.get("overhead") if isinstance(r.get("overhead"), dict) else {}
    if ov.get("overhead_pct"):
        lines.append(f"- step overhead ({ov['workload']}): " +
                     ", ".join(f"{k} +{v}%" for k, v in ov["overhead_pct"].items()))
    return "\n".join(lines) + "\n"


def run_all(dev, quick=False, say=print, deadline=None, reserve=180.0) -> dict:
    """deadline: time.monotonic() value after which no part starts (it's recorded as not run, never as passed)."""
    kind = "cuda" if dev.type == "cuda" else "cpu"
    r = {}
    if dev.type == "cuda":
        r.update(gpu=torch.cuda.get_device_name(dev), cc=list(torch.cuda.get_device_capability(dev)),
                 torch=torch.__version__)
        try:
            import triton
            r["triton"] = triton.__version__
        except Exception:  # noqa: BLE001
            pass
    parts = [("correctness", lambda: correctness(dev)),
             ("same_matmuls", lambda: [_guarded(lambda c=c: same_matmuls(dev, c, say), c["name"], dev)
                                       for c in WORKLOADS[kind]]),
             ("sensitivity", lambda: sensitivity(dev, SHAPES[kind][:3], 2 if quick else 16, say)),
             ("timing", lambda: timing(dev, SHAPES[kind], say)),
             ("overhead", lambda: overhead(dev, WORKLOADS[kind][0], 2 if quick else 6, say)),
             ("gradients", lambda: gradients(dev, WORKLOADS[kind][0], say))]
    for name, fn in parts:
        t0 = time.time()
        if deadline is not None and time.monotonic() > deadline - reserve:
            r[name] = {"error": "not run: no time left before the function's limit"}
            say(f"{name}: not run, no time left")
            continue
        try:
            r[name] = fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            r[name] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "trace": traceback.format_exc()[-1500:]}
        say(f"{name}: done in {time.time() - t0:.0f} s")
    r["criteria"] = criteria(r)
    return r


def _guarded(fn, name, dev):
    try:
        return fn()
    except torch.cuda.OutOfMemoryError as e:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return {"name": name, "error": f"did not fit in GPU memory: {str(e).splitlines()[0][:160]}"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    r = run_all(torch.device(a.device), a.quick)
    print(markdown(r))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"fused_{a.device}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
    return 0 if r["criteria"]["all met"] else 1


if __name__ == "__main__":
    sys.exit(main())
