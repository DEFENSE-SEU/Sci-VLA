"""Plot temporal label and error distributions from completion-switch predictions."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "eval_results" / "completion_switch_thermalcycler_stride10" / "predictions.csv"
OUTPUT_DIR = ROOT / "Figures"
OUTPUT_STEM = OUTPUT_DIR / "completion_switch_temporal_distributions"
SUMMARY_CSV = OUTPUT_DIR / "completion_switch_temporal_distribution_summary.csv"
N_BINS = 20

TASK_ORDER = [
    "close the lid of the thermal cycler",
    "open the lid of the thermal cycler",
    "place pcrPlate into the thermal cycler",
    "press the button of the thermal cycler",
    "screw loosen the knob of the thermal cycler",
    "screw tighten the knob of the thermal cycler",
    "take pcrPlate from the thermal cycler",
]
TASK_LABELS = [
    "Close lid", "Open lid", "Place PCR plate", "Press button",
    "Loosen knob", "Tighten knob", "Take PCR plate",
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def load_temporal_rates():
    counts = defaultdict(lambda: {"total": 0, "positive": 0, "fp": 0, "fn": 0})
    with PREDICTIONS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if task not in TASK_ORDER:
                continue
            position = float(row["position_fraction"])
            bin_index = min(N_BINS - 1, int(position * N_BINS))
            values = counts[(task, bin_index)]
            truth = int(row["groundtruth"])
            prediction = int(row["prediction"])
            values["total"] += 1
            values["positive"] += truth
            values["fp"] += int(prediction == 1 and truth == 0)
            values["fn"] += int(prediction == 0 and truth == 1)

    matrices = {name: np.zeros((len(TASK_ORDER), N_BINS)) for name in ["positive", "fp", "fn"]}
    summary_rows = []
    for task_index, task in enumerate(TASK_ORDER):
        for bin_index in range(N_BINS):
            values = counts[(task, bin_index)]
            total = values["total"]
            for name in matrices:
                matrices[name][task_index, bin_index] = values[name] / total if total else np.nan
            summary_rows.append({
                "task": task,
                "time_bin": bin_index,
                "time_start": bin_index / N_BINS,
                "time_end": (bin_index + 1) / N_BINS,
                "frames": total,
                "positive_fraction": matrices["positive"][task_index, bin_index],
                "false_positive_fraction": matrices["fp"][task_index, bin_index],
                "false_negative_fraction": matrices["fn"][task_index, bin_index],
            })
    return matrices, summary_rows


def main():
    matrices, summary_rows = load_temporal_rates()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    panels = [
        ("Ground-truth positive frames", matrices["positive"], "Blues"),
        ("False-positive frames", matrices["fp"], "Reds"),
        ("False-negative frames", matrices["fn"], "Purples"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.8), sharey=True)
    for panel_index, (title, matrix, cmap) in enumerate(panels):
        ax = axes[panel_index]
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=7.3, pad=8)
        ax.set_xticks([0, 4, 8, 12, 16, 19])
        ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
        ax.set_xlabel("Normalized episode time (%)")
        ax.set_yticks(range(len(TASK_LABELS)))
        if panel_index == 0:
            ax.set_yticklabels(TASK_LABELS, fontsize=6.3)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="both", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        colorbar.set_ticks([0, 0.5, 1.0])
        colorbar.set_label("Frame fraction", fontsize=6.2, labelpad=2)
        colorbar.ax.tick_params(labelsize=5.8, length=2)
        ax.text(-0.12, 1.08, chr(ord("a") + panel_index), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")

    fig.subplots_adjust(left=0.14, right=0.99, bottom=0.23, top=0.83, wspace=0.38)
    for suffix, kwargs in {".pdf": {}, ".svg": {}, ".png": {"dpi": 600}}.items():
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
