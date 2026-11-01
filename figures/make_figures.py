"""Portfolio figures, drawn only from the saved files in results/ (no model runs, no test-set access).

Run from the repository root:  python figures/make_figures.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
N_GOOD = 32
FLOOR = json.loads((ROOT / "config" / "acceptance.json").read_text())["robustness_floor_auroc"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def read_csv(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


def figure_attacks():
    rows = read_csv("fgsm_sweep.csv")
    fg = [r for r in rows if r["mode"] == "worst_case"]
    pg = [r for r in rows if r["mode"].startswith("pgd")]
    x_fg = [float(r["epsilon_255"]) for r in fg]
    x_pg = [float(r["epsilon_255"]) for r in pg]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    # Left: ranking quality (what the pre-set criterion measured).
    a.axhline(FLOOR, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    a.text(0.1, FLOOR - 0.03, f"pass floor {FLOOR}", ha="left", va="top", color=INK2, fontsize=9)
    a.plot(x_fg, [float(r["auroc_FP32"]) for r in fg], "-o", color=BLUE, lw=2, ms=7,
           mec="white", mew=1.5, zorder=3)
    a.plot(x_pg, [float(r["auroc_FP32"]) for r in pg], "-s", color=ORANGE, lw=2, ms=7,
           mec="white", mew=1.5, zorder=3)
    a.text(7.9, float(fg[-1]["auroc_FP32"]) + 0.035, "FGSM, 1 step: 0.923 at 7.65/255", ha="right", va="bottom", color=INK, fontsize=9)
    a.text(2.75, 0.04, "PGD, 10 steps: 0.0008 at 2.55/255", ha="left", va="bottom", color=INK, fontsize=9)
    a.set_ylim(-0.03, 1.05)
    a.set_ylabel("AUROC (ranking quality)")
    a.set_title("A. Ranking quality (the pre-set criterion)", loc="left", fontsize=11, color=INK)

    # Right: what the production line would actually do at the frozen threshold.
    b.axhline(N_GOOD, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    b.text(0.05, N_GOOD + 0.8, "all 32 good parts flagged (same as rejecting every part)",
           ha="left", va="bottom", color=INK2, fontsize=9)
    b.plot(x_fg, [int(r["false_alarms_FP32"]) for r in fg], "-o", color=BLUE, lw=2, ms=7,
           mec="white", mew=1.5, zorder=3, label="FGSM, 1 step")
    b.plot(x_pg, [int(r["false_alarms_FP32"]) for r in pg], "s", color=ORANGE, ms=9,
           mec="white", mew=1.5, zorder=4, label="PGD, 10 steps")
    for r in fg[1:3]:
        b.annotate(r["false_alarms_FP32"], (float(r["epsilon_255"]), int(r["false_alarms_FP32"])),
                   textcoords="offset points", xytext=(8, -4), color=INK, fontsize=9)
    b.set_ylim(-1, 37)
    b.set_ylabel(f"False alarms (good parts flagged, of {N_GOOD})")
    b.set_title("B. False alarms at the frozen threshold (what the line sees)", loc="left", fontsize=11, color=INK)
    b.legend(loc="center right", frameon=False)

    for ax in (a, b):
        ax.set_xlabel("Attack strength: largest change to any pixel, out of 255")
        ax.set_xticks([0, 0.51, 1.28, 2.55, 5.1, 7.65])
        ax.set_xticklabels(["0", ".51", "1.28", "2.55", "5.1", "7.65"])
        ax.set_xlim(-0.2, 8.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_attacks_auroc_vs_false_alarms.png", dpi=170)
    plt.close(fig)


def figure_natural_shift():
    tm = json.loads((RESULTS / "test_metrics.json").read_text())
    shift = read_csv("natural_shift.csv")
    names = ["clean", "darker\n(x0.7)", "blur\n(radius 2)", "noise\n(sd 0.04)", "JPEG\n(quality 30)"]
    auroc = {"FP32": [tm["clean"]["FP32"]["auroc"]], "INT8": [tm["clean"]["INT8"]["auroc"]]}
    fa = {"FP32": [tm["clean"]["FP32"]["fp"]], "INT8": [tm["clean"]["INT8"]["fp"]]}
    for r in shift:
        for m in ("FP32", "INT8"):
            auroc[m].append(float(r[f"auroc_{m}"]))
            fa[m].append(int(r[f"false_alarms_{m}"]))

    xs = range(len(names))
    w = 0.38
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, data, title, ylabel in (
        (a, auroc, f"A. Ranking quality: every condition clears the {FLOOR} floor (dashed)", "AUROC"),
        (b, fa, "B. False alarms at the frozen threshold", f"False alarms (good parts flagged, of {N_GOOD})"),
    ):
        for k, (m, color) in enumerate((("FP32", BLUE), ("INT8", ORANGE))):
            pos = [x + (k - 0.5) * (w + 0.02) for x in xs]
            bars = ax.bar(pos, data[m], width=w, color=color, label=m, zorder=3)
            if ax is b:
                for p, v in zip(pos, data[m]):
                    ax.text(p, v + 0.6, str(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylabel(ylabel)
    a.axhline(FLOOR, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=4)
    a.set_ylim(0, 1.08)
    b.set_ylim(0, 37)
    b.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_natural_shift_auroc_vs_false_alarms.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    figure_attacks()
    figure_natural_shift()
    print("Saved figures to", OUT)
