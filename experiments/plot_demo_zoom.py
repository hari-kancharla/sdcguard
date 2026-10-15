"""Zoomed chart of the multi-GPU demo around the moment the GPU goes bad (for the README).

    python experiments/plot_demo_zoom.py runs/gpu/demo/demo.json docs/img

Writes demo_zoom.png (light) and demo_zoom_dark.png (dark) into the output folder.
"""
from __future__ import annotations

import json
import math
import os
import sys

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  s1="#2a78d6", s2="#eb6834", band="#ece9df"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 s1="#3987e5", s2="#d95926", band="#2a2925"),
}


def _xy(att, lo, hi):
    pts = [(s, v) for s, v in att["losses"] if lo <= s <= hi and math.isfinite(v)]
    return [p[0] for p in pts], [p[1] for p in pts]


def plot(demo: dict, path: str, theme: str, lo: int = 150, hi: int = 260) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEMES[theme]
    a = demo["args"]
    bad_from = a["bad_from"]
    base, det = demo["stories"]["without_sdcguard"], demo["stories"]["with_sdcguard"]
    crash = base[0]["last_step"]
    alert = demo["summary"]["first_alert"]
    rollback = det[0]["rollback_to"]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": t["grid"], "axes.labelcolor": t["ink2"],
                         "xtick.color": t["muted"], "ytick.color": t["muted"], "text.color": t["ink"]})
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, sharey=True, facecolor=t["surface"])
    ymin = min(v for st in base + det for s, v in st["losses"] if lo <= s <= hi and math.isfinite(v))
    ymax = max(v for st in base + det for s, v in st["losses"] if lo <= s <= hi and math.isfinite(v))
    pad = (ymax - ymin) * 0.25
    for i, ax in enumerate(axes):
        ax.set_facecolor(t["surface"])
        ax.grid(axis="y", color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        # with sdcguard, the bad GPU stops being used at the alert
        ax.axvspan(bad_from, hi if i == 0 else alert, color=t["band"], linewidth=0, zorder=0)
        ax.set_ylim(ymin - pad, ymax + pad * 1.6)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("training loss")
    top = ymax + pad * 1.35

    # Without sdcguard: trains normally, then NaN. Restarts from the same checkpoint crash again.
    ax = axes[0]
    xs, ys = _xy(base[0], lo, hi)
    ax.plot(xs, ys, color=t["s1"], linewidth=2, solid_capstyle="round", label="training on GPUs 0-3")
    ax.plot([xs[-1]], [ys[-1]], "o", color=t["s1"], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
    ax.axvline(crash, color=t["ink2"], linewidth=1)
    restarts = sum(s["status"] == "crashed" for s in base) - 1
    ax.text(crash - 1.5, top, f"step {crash}: loss is NaN, training crashed", va="top", ha="right", fontsize=9,
            color=t["ink"])
    ax.text(crash - 1.5, top - pad * 0.55,
            f"restarted from the step {base[1]['start'] - 1} checkpoint {restarts} times: same crash",
            va="top", ha="right", fontsize=9, color=t["ink2"])
    ax.set_title("Without sdcguard", loc="left", fontsize=11, color=t["ink"], fontweight="bold")
    from matplotlib.patches import Patch
    handles = [ax.lines[0], Patch(color=t["band"], label=f"GPU {a['bad_gpu']} corrupting results (from step {bad_from})")]
    leg = ax.legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    # With sdcguard: alert at the first check after the fault, roll back, finish on 3 GPUs.
    ax = axes[1]
    xs, ys = _xy(det[0], lo, hi)
    ax.plot(xs, ys, color=t["s1"], linewidth=2, solid_capstyle="round", label="training on GPUs 0-3")
    ax.plot([xs[-1]], [ys[-1]], "o", color=t["s1"], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
    xs2, ys2 = _xy(det[1], lo, hi)
    ax.plot(xs2, ys2, color=t["s2"], linewidth=2, solid_capstyle="round",
            label=f"after rollback, GPUs {', '.join(str(g) for g in det[1]['gpus'])}")
    ax.axvline(alert, color=t["ink2"], linewidth=1)
    ident = demo["summary"]["identity"]
    ax.text(alert + 1.5, top, f"step {alert}: sdcguard names GPU {ident.get('gpu_index')}", va="top",
            fontsize=9, color=t["ink"])
    ax.text(alert + 1.5, top - pad * 0.55, f"UUID {ident.get('gpu_uuid', '')[:12]}..., PCI "
            f"{ident.get('pci_bus_id')}", va="top", fontsize=9, color=t["ink2"])
    ax.text(alert + 1.5, top - pad * 1.1, f"evicts it, rolls back to step {rollback}, finishes all "
            f"{a['steps']:,} steps", va="top", fontsize=9, color=t["ink2"])
    ax.set_title("With sdcguard", loc="left", fontsize=11, color=t["ink"], fontweight="bold")
    ax.set_xlabel("training step")
    leg = ax.legend(loc="lower left", fontsize=8.5, frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=t["surface"])
    plt.close(fig)


def main():
    src, out = sys.argv[1], sys.argv[2]
    with open(src) as f:
        demo = json.load(f)
    os.makedirs(out, exist_ok=True)
    plot(demo, os.path.join(out, "demo_zoom.png"), "light")
    plot(demo, os.path.join(out, "demo_zoom_dark.png"), "dark")


if __name__ == "__main__":
    main()
