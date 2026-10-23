"""The fused precise check (SDCGuard(precise="fused")) on clean training, for a long time: false alarms and the
worst clean score, the way the default check's 24.6 million clean checks were counted on October 14-15. The
October 21 run had 990 clean fused checks (worst 1.53 on the H100); the October 20 run's worst on an L4 was 4.11.

Clean bf16 training of the calibration-shape model of experiments/fused_check.py (8 layers, 1024 wide, Zipf
tokens), every linear layer's forward, dX and dW checked on every step by the fused check, for --minutes; counts
saved every 2 minutes. Criteria in docs/METHOD.md ("The fused precise check, a long clean run"), written before
the run.

    python experiments/fused_soak.py --device cuda --minutes 40 --out results/fused_soak
    TRITON_INTERPRET=1 SDCGUARD_FAST_CPU_TEST=1 python experiments/fused_soak.py --device cpu --minutes 0.05 \\
        --out /tmp/soak                                   # Triton's CPU interpreter: a smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from fused_check import WORKLOADS, _low, _tokens  # noqa: E402
from sdcguard import SDCGuard  # noqa: E402
from sdcguard.screen import ScreenModel  # noqa: E402

Z_EDGES = (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, math.inf)


def histogram(zs) -> list:
    """Counts of scores below each edge of Z_EDGES (the last bin is 6 and up)."""
    out = [0] * len(Z_EDGES)
    for z in zs:
        for i, e in enumerate(Z_EDGES):
            if z < e or i == len(Z_EDGES) - 1:
                out[i] += 1
                break
    return out


def soak(dev, minutes: float, out: str, cfg=None, save_every_s: float = 120.0, say=print) -> dict:
    cfg = cfg or WORKLOADS["cuda" if dev.type == "cuda" else "cpu"][0]
    low = _low(dev)
    torch.manual_seed(cfg["seed"])
    model = ScreenModel(cfg["vocab"], cfg["layers"], cfg["width"], cfg["heads"], cfg["seq"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    scores: list = []
    g = SDCGuard(every=1, precise="fused", on_result=lambda detail, res: scores.append(res.max_z) if res.ok and
                 res.judged and math.isfinite(res.max_z) else None)
    g.protect(model)
    t_end = time.time() + 60 * minutes
    t0 = time.time()
    step, last, hist = 0, time.time(), [0] * len(Z_EDGES)
    rec: dict = {}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"soak_{dev.type}.json")
    try:
        while True:
            idx = _tokens(cfg, gen, (cfg["batch"], cfg["seq"] + 1)).to(dev)
            with g.on_step(step):
                with torch.autocast(dev.type, dtype=low):
                    logits = model(idx[:, :-1])
                    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), idx[:, 1:].reshape(-1))
                loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if scores:
                h = histogram(scores)
                hist = [a + b for a, b in zip(hist, h)]
                scores.clear()
            done = time.time() >= t_end
            if done or time.time() - last > save_every_s:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                rec = {"device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                       "torch": torch.__version__, "workload": cfg["name"], "steps": step, "finished": done,
                       "minutes": round((time.time() - t0) / 60, 2), "loss": float(loss),
                       "fused_checks": g.stats.get("checked_fp32", 0), "checks": g.stats.get("checked", 0),
                       "unjudged": g.stats.get("unjudged", 0), "false_alarms": g.stats.get("alerts", 0),
                       "worst_clean_z": g.max_z_seen, "z_hist_edges": [str(e) for e in Z_EDGES], "z_hist": hist,
                       "scored": sum(hist), "alerts": [e.detail for e in g.log.events[:20]], "stats": dict(g.stats)}
                with open(path + ".part", "w") as f:
                    json.dump(rec, f, default=str)
                os.replace(path + ".part", path)
                last = time.time()
                say(f"soak: {step} steps, {rec['fused_checks']:,} fused checks, {rec['false_alarms']} alarms, "
                    f"worst {rec['worst_clean_z']:.2f}")
            if done:
                return rec
    finally:
        g.unprotect()


def judge(rec: dict) -> list:
    """Failures of the criteria (empty: all met)."""
    fails = []
    if not rec.get("finished"):
        fails.append("the run didn't finish")
    if rec.get("false_alarms"):
        fails.append(f"{rec['false_alarms']} alarms on clean training")
    if not rec.get("fused_checks") or rec.get("scored", 0) < 0.9 * rec["fused_checks"]:
        fails.append(f"scores read for {rec.get('scored')} of {rec.get('fused_checks')} fused checks")
    if rec.get("unjudged"):
        fails.append(f"{rec['unjudged']} checks couldn't be judged")
    st = rec.get("stats") or {}
    if st.get("fused_fallbacks") or st.get("checked_fused", rec.get("fused_checks")) != rec.get("fused_checks"):
        fails.append(f"not every check ran in the fused kernel: {st.get('fused_fallbacks')} fallbacks, "
                     f"{st.get('checked_fused')} of {rec.get('fused_checks')} fused")
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--minutes", type=float, default=40.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rec = soak(torch.device(a.device), a.minutes, a.out)
    rec["failures"] = judge(rec)
    with open(os.path.join(a.out, f"soak_{torch.device(a.device).type}.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("alerts", "stats")}, indent=1, default=str))
    return 0 if not rec["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
