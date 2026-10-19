"""Chart of the 4-GPU demo for the README: the moment GPU 2 goes bad, and the whole run.

    python experiments/plot_demo_zoom.py results/demo_4gpu/demo.json docs/img

Writes demo_zoom.png (light) and demo_zoom_dark.png (dark) into the output folder.
"""
from __future__ import annotations

import json
import math
import os
import sys

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", hero="#2a78d6", base="#898781", crash="#d03b3b"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", hero="#3987e5", base="#898781", crash="#d03b3b"),
}
SMOOTH = 15        # moving-average window for the full-run panel


def _finite(points, lo, hi):
    return [(s, v) for s, v in points if lo <= s <= hi and math.isfinite(v)]


def _moving_average(points, w):
    out = []
    for i in range(len(points)):
        win = points[max(0, i - w + 1):i + 1]
        out.append((points[i][0], sum(v for _, v in win) / len(win)))
    return out


def _runs(demo):
    """The run without sdcguard (until it crashes) and the run with it as it actually continued: steps up
    to the rollback point, then the 3-GPU run after the rollback."""
    base, det = demo["stories"]["without_sdcguard"], demo["stories"]["with_sdcguard"]
    rollback = det[0]["rollback_to"]
    without = [(s, v) for s, v in base[0]["losses"]]
    with_ = [(s, v) for s, v in det[0]["losses"] if s <= rollback] + [(s, v) for s, v in det[1]["losses"]]
    return without, with_


def plot(demo: dict, path: str, theme: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    t = THEMES[theme]
    family = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.edgecolor": t["axis"],
                         "axes.labelcolor": t["muted"], "xtick.color": t["muted"], "ytick.color": t["muted"],
                         "text.color": t["ink"], "axes.linewidth": 1})

    a, s = demo["args"], demo["summary"]
    bad_gpu, bad_from, alert = a["bad_gpu"], a["bad_from"], s["first_alert"]
    det = demo["stories"]["with_sdcguard"]
    survivors = det[1]["gpus"]
    crash = demo["stories"]["without_sdcguard"][0]["last_step"]
    restarts = sum(r["status"] == "crashed" for r in demo["stories"]["without_sdcguard"]) - 1
    without, with_ = _runs(demo)

    fig = plt.figure(figsize=(10, 4.6), facecolor=t["surface"])
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], left=0.065, right=0.975, top=0.70, bottom=0.13, wspace=0.16)
    zoom, full = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    def frame(ax, title):
        ax.set_facecolor(t["surface"])
        ax.grid(axis="y", color=t["grid"], linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, pad=6)
        ax.set_title(title, loc="left", fontsize=10.5, color=t["ink2"], pad=10)
        ax.set_xlabel("training step", fontsize=9.5, labelpad=6)

    # ---- left: steps around the fault
    lo, hi = 190, 226
    frame(zoom, f"Steps {lo} to {hi - 1}")
    zw, zs = _finite(without, lo, hi), _finite(with_, lo, hi)
    zoom.plot(*zip(*zw), color=t["base"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    zoom.plot(*zip(*zs), color=t["hero"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    yv = [v for _, v in zw + zs]
    y0, y1 = min(yv), max(yv)
    span = y1 - y0
    zoom.set_ylim(y0 - span * 0.35, y1 + span * 0.95)
    zoom.set_xlim(lo, hi - 1)
    zoom.set_xticks(range(lo, hi, 10))
    ytop = y1 + span * 0.85
    for x in (bad_from, alert):
        zoom.axvline(x, color=t["axis"], linewidth=1, zorder=1)
    zoom.text(bad_from - 0.6, ytop, f"step {bad_from}\nGPU {bad_gpu} goes bad", ha="right", va="top", fontsize=9,
              color=t["ink2"], linespacing=1.35)
    zoom.text(alert + 0.6, ytop, f"step {alert}\nsdcguard flags GPU {bad_gpu}", ha="left", va="top", fontsize=9,
              color=t["ink"], linespacing=1.35)
    lx, lv = zw[-1]
    zoom.plot([crash], [lv], marker="X", markersize=10, color=t["crash"], markeredgecolor=t["surface"],
              markeredgewidth=1.5, zorder=6, linestyle="none", clip_on=False)
    zoom.text(crash - 0.7, lv - span * 0.05, "crash\n(NaN)", ha="right", va="top", fontsize=9,
              color=t["ink2"], linespacing=1.25)

    # ---- right: the whole run, smoothed
    lo2, hi2 = 100, 1000
    frame(full, f"Whole run, steps {lo2} to {hi2:,} ({SMOOTH}-step moving average)")
    fw = _moving_average(_finite(without, 0, hi2), SMOOTH)
    fs = _moving_average(_finite(with_, 0, hi2), SMOOTH)
    fw, fs = [p for p in fw if p[0] >= lo2], [p for p in fs if p[0] >= lo2]
    # the two runs are the same run until the fault, so the run without sdcguard sits under the other one
    full.plot(*zip(*fw), color=t["base"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=4)
    full.plot(*zip(*fs), color=t["hero"], linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=5)
    full.set_xlim(lo2, hi2)
    vals = [v for _, v in fw + fs]
    fy0, fy1 = min(vals), max(vals)
    fspan = fy1 - fy0
    full.set_ylim(fy0 - fspan * 0.12, fy1 + fspan * 0.12)
    full.set_xticks(range(200, hi2 + 1, 200))
    full.set_xticklabels([f"{x:,}" for x in range(200, hi2 + 1, 200)])
    cx, cy = fw[-1]
    full.plot([cx], [cy], marker="X", markersize=10, color=t["crash"], markeredgecolor=t["surface"],
              markeredgewidth=1.5, zorder=6, linestyle="none", clip_on=False)
    times = "twice" if restarts == 2 else f"{restarts} times"
    full.annotate(f"Without sdcguard: crashed at step {crash},\nthen again after restarting {times}",
                  xy=(cx, cy), xytext=(cx + 40, cy + fspan * 0.2), fontsize=9, color=t["ink2"], va="bottom",
                  linespacing=1.35,
                  arrowprops=dict(arrowstyle="-", color=t["axis"], linewidth=1, shrinkA=0, shrinkB=6))
    ex, ey = fs[-1]
    full.plot([ex], [ey], marker="o", markersize=8, color=t["hero"], markeredgecolor=t["surface"],
              markeredgewidth=2, zorder=6, linestyle="none", clip_on=False)
    gpus = ", ".join(str(g) for g in survivors)
    full.text(ex - 10, ey + fspan * 0.12, f"With sdcguard: finished all {a['steps']:,} steps\non GPUs {gpus}",
              ha="right", va="bottom", fontsize=9, color=t["ink"], linespacing=1.35)
    zoom.set_ylabel("training loss", fontsize=9.5, labelpad=6)

    # ---- title, subtitle, legend
    fig.text(0.065, 0.955, "One GPU starts returning wrong results in the middle of training",
             fontsize=14, fontweight="semibold", color=t["ink"], va="top")
    fig.text(0.065, 0.885, f"4 H100s training a GPT-style model. From step {bad_from}, GPU {bad_gpu} corrupts some "
             f"matmul outputs (simulated with NVIDIA's published fault patterns).", fontsize=9.5, color=t["ink2"],
             va="top")
    for i, (color, label) in enumerate(((t["hero"], "with sdcguard"), (t["base"], "without sdcguard"))):
        x = 0.065 + i * 0.15
        fig.lines.append(matplotlib.lines.Line2D([x, x + 0.022], [0.815, 0.815], transform=fig.transFigure,
                                                 color=color, linewidth=2, solid_capstyle="round"))
        fig.text(x + 0.029, 0.815, label, fontsize=9.5, color=t["ink2"], va="center")

    fig.savefig(path, dpi=200, facecolor=t["surface"])
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
