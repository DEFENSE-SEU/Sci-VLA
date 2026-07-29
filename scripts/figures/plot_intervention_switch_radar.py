"""Render the completion-model metrics in tab:intervention-switch-eval."""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Figures" / "intervention_switch_metrics.csv"
OUTPUT_STEM = ROOT / "Figures" / "intervention_switch_radar"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
})


def load_rows():
    with DATA_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = load_rows()
    labels = [
        "Frame\naccuracy", "Precision", "Recall",
        "FP-free\nframes", "FN-free\nframes",
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.r_[angles, angles[0]]
    colors = [
        "#1f4e79", "#4e79a7", "#76b7b2", "#59a14f",
        "#edc948", "#e17c05", "#b07aa1",
    ]

    fig, ax = plt.subplots(
        figsize=(3.65, 3.35), subplot_kw={"projection": "polar"}
    )
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], color="#6b7280", fontsize=6)
    ax.set_rlabel_position(108)
    ax.grid(color="#cbd5e1", linewidth=0.6)
    ax.spines["polar"].set_color("#64748b")
    ax.spines["polar"].set_linewidth(0.7)

    for row, color in zip(rows, colors):
        frames = float(row["Frames"])
        values = [
            100 - float(row["Error"]),
            float(row["Precision"]),
            float(row["Recall"]),
            100 * (1 - float(row["FP"]) / frames),
            100 * (1 - float(row["FN"]) / frames),
        ]
        values = np.r_[values, values[0]]
        ax.plot(angles, values, color=color, linewidth=1.25, label=row["Task"])
        ax.fill(angles, values, color=color, alpha=0.045)
        ax.scatter(angles[:-1], values[:-1], color=color, s=10, zorder=3)

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        fontsize=6.4, columnspacing=1.15, handlelength=1.8, frameon=False,
    )
    fig.subplots_adjust(top=0.92, bottom=0.23, left=0.05, right=0.95)

    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
    }.items():
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
