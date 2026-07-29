"""Render completion-model metrics from tab:intervention-switch-eval as bars."""

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Figures" / "intervention_switch_metrics.csv"
OUTPUT_STEM = ROOT / "Figures" / "intervention_switch_bars"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.7,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def load_rows():
    with DATA_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = load_rows()
    task_labels = []
    for row in rows:
        task = row["Task"].replace(" ", "\n", 1)
        task_labels.append(task)
    x = np.arange(len(rows))

    fig, (ax_metrics, ax_errors) = plt.subplots(
        1, 2, figsize=(7.15, 2.8), gridspec_kw={"width_ratios": [1.25, 1.2]}
    )

    metric_specs = [
        ("Frame accuracy", [100 - float(row["Error"]) for row in rows], "#4e79a7"),
        ("Precision", [float(row["Precision"]) for row in rows], "#76b7b2"),
        ("Recall", [float(row["Recall"]) for row in rows], "#f28e2b"),
    ]
    width = 0.23
    for index, (label, values, color) in enumerate(metric_specs):
        ax_metrics.bar(
            x + (index - 1) * width, values, width=width, label=label,
            color=color, edgecolor="white", linewidth=0.35,
        )
    ax_metrics.set_ylim(0, 105)
    ax_metrics.set_yticks([0, 25, 50, 75, 100])
    ax_metrics.set_ylabel("Rate (%)")
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(task_labels, fontsize=6)
    ax_metrics.grid(axis="y", color="#d9e2ec", linewidth=0.6)
    ax_metrics.set_axisbelow(True)
    ax_metrics.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3,
        fontsize=6.3, columnspacing=1.0, handlelength=1.2,
    )

    fp = [int(row["FP"]) for row in rows]
    fn = [int(row["FN"]) for row in rows]
    ax_errors.bar(x, fp, label="False-positive frames", color="#b07aa1",
                  edgecolor="white", linewidth=0.35)
    ax_errors.bar(x, fn, bottom=fp, label="False-negative frames", color="#e15759",
                  edgecolor="white", linewidth=0.35)
    ax_errors.set_ylim(0, 125)
    ax_errors.set_yticks([0, 25, 50, 75, 100, 125])
    ax_errors.set_ylabel("Misclassified frames")
    ax_errors.set_xticks(x)
    ax_errors.set_xticklabels(task_labels, fontsize=5.5)
    ax_errors.grid(axis="y", color="#d9e2ec", linewidth=0.6)
    ax_errors.set_axisbelow(True)
    ax_errors.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2,
        fontsize=6.3, columnspacing=1.0, handlelength=1.2,
    )

    for label, axis in zip(["a", "b"], [ax_metrics, ax_errors]):
        axis.text(-0.15, 1.08, label, transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="bottom")
        axis.tick_params(axis="x", length=0, pad=3)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.80, wspace=0.32)
    for suffix, kwargs in {
        ".pdf": {},
        ".svg": {},
        ".png": {"dpi": 600},
    }.items():
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
