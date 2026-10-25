"""The FP8 matmul check's tolerance, fitted to the calibration run's data (docs/METHOD.md, "Calibrating an FP8 matmul
check", written before the run). Per GPU, recipe and kernel: alpha and beta of

    |O - E| <= z * sqrt(S^2 + alpha^2 A^2 + beta^2 B^2)

fitted on 52 of the 68 prompts and judged on the other 16 (the largest clean score on them at most 3, half the alarm
line of 6), with how many times the bf16 noise S the fitted tolerance allows on each layer, so what errors it could
see. The pre-registration names the 52/16 split and the goal, nothing more; these were chosen after the run: the 16
held-out prompts, drawn with a fixed seed independent of the data (HELD_OUT), and "as small as the clean data allow"
read as: for each beta on a grid, the smallest alpha that keeps every training score at most 3 (found by bisection:
the largest score only falls as alpha grows), and of those pairs the one whose median allowance
sqrt(S^2 + alpha^2 A^2 + beta^2 B^2) / S over the training blocks is smallest. About 90% of the tokens, so of the
sampled blocks, are the 4 long documents (one paragraph's words shuffled and repeated 40 times; LONG_DOCS), two on
each side of the split, so the worst scores are also reported for the short prompts and the long documents apart.
Blocks with a value that isn't finite (S, A and B are kept in float16) are left out and counted. Needs only numpy.

    python experiments/fp8mm_fit.py runs/fp8mm_cal_h100 runs/fp8mm_cal_l4 --out runs/fp8mm_fit.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

N_PROMPTS = 68                   # the calibration's 64 prompts and 4 long documents (fp8mm_calibration.py)
HELD_OUT = tuple(int(i) for i in sorted(np.random.default_rng(0).choice(N_PROMPTS, 16, replace=False)))
LONG_DOCS = (64, 65, 66, 67)     # the 4 long documents follow the 64 prompts (fp8mm_calibration.long_docs)
TARGET = 3.0                     # the largest clean score allowed: half the alarm line of 6
LINE = 6.0
BETAS = (0.0,) + tuple(float(b) for b in np.geomspace(1e-5, 1.0, 41))
KEYS = ("per_token_cutlass", "per_tensor_cutlass", "per_token_torch_promoted", "per_tensor_torch_promoted")


def load(run_dir: str, key: str) -> dict:
    """Every layer's blocks for one recipe and kernel: r, S, A, B as float64 [blocks], the prompt of each block's row,
    and the layer of each block."""
    out = {"r": [], "S": [], "A": [], "B": [], "prompt": [], "layer": []}
    names = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.npz"))):
        z = np.load(path)
        if f"{key}_r" not in z.files:
            continue
        r = z[f"{key}_r"].astype(np.float64)
        rows, nb = r.shape
        out["r"].append(r.reshape(-1))
        for nm in ("S", "A", "B"):
            out[nm].append(z[f"{key}_{nm}"].astype(np.float64).reshape(-1))
        out["prompt"].append(np.repeat(z["prompt"].astype(np.int64), nb))
        out["layer"].append(np.full(rows * nb, len(names), dtype=np.int64))
        names.append(os.path.basename(path)[:-4])
    if not names:
        return {}
    d = {k: np.concatenate(v) for k, v in out.items()}
    ok = np.isfinite(d["r"]) & np.isfinite(d["S"]) & np.isfinite(d["A"]) & np.isfinite(d["B"])
    d["blocks_total"], d["blocks_not_finite"] = int(ok.size), int((~ok).sum())
    if not ok.all():
        d = {k: v[ok] for k, v in d.items() if isinstance(v, np.ndarray)} | \
            {"blocks_total": d["blocks_total"], "blocks_not_finite": d["blocks_not_finite"]}
    d["blocks_zero_noise"] = int((d["S"] == 0).sum())
    d["names"] = names
    return d


def scores(d: dict, alpha: float, beta: float, sel=None) -> np.ndarray:
    S, A, B, r = (d[k] if sel is None else d[k][sel] for k in ("S", "A", "B", "r"))
    den = np.sqrt(S * S + (alpha * A) ** 2 + (beta * B) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(r) / den
    return np.where(den > 0, z, np.where(r == 0, 0.0, np.inf))


def allowance(d: dict, alpha: float, beta: float, sel=None) -> np.ndarray:
    """How many times the bf16 noise S the tolerance allows, per block (blocks with S = 0 left out)."""
    S, A, B = (d[k] if sel is None else d[k][sel] for k in ("S", "A", "B"))
    ok = S > 0
    return np.sqrt(S[ok] ** 2 + (alpha * A[ok]) ** 2 + (beta * B[ok]) ** 2) / S[ok]


def smallest_alpha(d: dict, beta: float, sel, target: float = TARGET, hi: float = 1e4, iters: int = 60):
    """The smallest alpha keeping every selected score at most target with this beta, or None if none does."""
    if np.max(scores(d, hi, beta, sel)) > target:
        return None
    if np.max(scores(d, 0.0, beta, sel)) <= target:
        return 0.0
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if np.max(scores(d, mid, beta, sel)) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def fit(d: dict, held_out=HELD_OUT) -> dict:
    """alpha and beta as small as the training prompts allow, and the held-out verdict."""
    test = np.isin(d["prompt"], np.asarray(held_out))
    train = ~test
    best = None
    for beta in BETAS:
        a = smallest_alpha(d, beta, train)
        if a is None:
            continue
        med = float(np.median(allowance(d, a, beta, train)))
        if best is None or med < best[2]:
            best = (a, beta, med)
    if best is None:
        return {"fitted": False, "why": "no alpha and beta on the grid keep every training score at most 3"}
    a, beta, _ = best
    zt, zh = scores(d, a, beta, train), scores(d, a, beta, test)
    per_layer = {}
    for i, name in enumerate(d["names"]):
        lay = d["layer"] == i
        al = allowance(d, a, beta, lay)
        per_layer[name] = {"allowance_median": float(np.median(al)) if al.size else None,
                           "allowance_p99": float(np.quantile(al, 0.99)) if al.size else None,
                           "held_out_worst": float(np.max(zh[lay[test]])) if np.any(lay & test) else None}
    al = allowance(d, a, beta)
    longd = np.isin(d["prompt"], np.asarray(LONG_DOCS))
    z = scores(d, a, beta)

    def worst(sel):
        return float(np.max(z[sel])) if np.any(sel) else None

    def share(sel):
        return float(np.mean(longd[sel])) if np.any(sel) else None
    apart = {"long_documents_share_train": share(train), "long_documents_share_held_out": share(test),
             "train_worst_short_prompts": worst(train & ~longd), "train_worst_long_documents": worst(train & longd),
             "held_out_worst_short_prompts": worst(test & ~longd),
             "held_out_worst_long_documents": worst(test & longd)}
    return {"fitted": True, "alpha": a, "beta": beta, "blocks_total": d.get("blocks_total"),
            "blocks_not_finite": d.get("blocks_not_finite"), "blocks_zero_noise": d.get("blocks_zero_noise"),
            "blocks_train": int(train.sum()), "blocks_held_out": int(test.sum()),
            "train_worst": float(np.max(zt)), "held_out_worst": float(np.max(zh)),
            "held_out_passed": bool(np.max(zh) <= TARGET), **apart,
            "allowance_median": float(np.median(al)), "allowance_p99": float(np.quantile(al, 0.99)),
            # the alarm line, and the error caught whatever the sign of a clean part scoring up to TARGET (an error
            # between them may or may not be), in a median block's bf16 noise
            "line_in_bf16_noise_units_median": float(LINE * np.median(al)),
            "caught_for_sure_in_bf16_noise_units_median": float((LINE + TARGET) * np.median(al)),
            "layers": per_layer}


def run(dirs: list) -> dict:
    out = {"held_out_prompts": list(HELD_OUT), "target": TARGET, "line": LINE, "runs": {}}
    for run_dir in dirs:
        summ = {}
        try:
            with open(os.path.join(run_dir, "SUMMARY.json")) as f:
                summ = json.load(f)
        except OSError:
            pass
        res = {"gpu": summ.get("gpu"), "torch": summ.get("torch"), "fits": {}}
        for key in KEYS:
            d = load(run_dir, key)
            if d:
                res["fits"][key] = fit(d)
        out["runs"][os.path.basename(os.path.normpath(run_dir))] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(a.dirs)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for name, r in res["runs"].items():
        for key, ft in r["fits"].items():
            brief = {k: ft.get(k) for k in ("alpha", "beta", "train_worst", "held_out_worst", "held_out_passed",
                                            "allowance_median", "allowance_p99")} if ft.get("fitted") else ft
            print(name, r.get("gpu"), key, json.dumps(brief))
    return 0


if __name__ == "__main__":
    sys.exit(main())
