"""Combine task-level completion metrics with temporal label/error heatmaps."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "eval_results" / "completion_switch_thermalcycler_stride10" / "predictions.csv"
OUTPUT_DIR = ROOT / "Figures"
OUTPUT_STEM = OUTPUT_DIR / "completion_switch_metrics_and_temporal_distributions"
TASK_METRICS_CSV = OUTPUT_DIR / "completion_switch_task_metrics_10_episodes.csv"
N_BINS = 20

# Soft pastel colours keep the figure light while preserving the distinction
# between ground-truth, false-positive, and false-negative temporal patterns.
PASTEL_CMAPS = {
    "blue": LinearSegmentedColormap.from_list(
        "pastel_blue", ["#f8fbff", "#dbeafa", "#9fc4e4", "#6e9fca"]
    ),
    "green": LinearSegmentedColormap.from_list(
        "pastel_green", ["#f7fbf8", "#dceee3", "#a9d2ba", "#79ae91"]
    ),
    "orange": LinearSegmentedColormap.from_list(
        "pastel_orange", ["#fffaf6", "#fbe4cf", "#f3bc91", "#df9365"]
    ),
}

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
    "Close\nlid", "Open\nlid", "Place\nPCR plate", "Press\nbutton",
    "Loosen\nknob", "Tighten\nknob", "Take\nPCR plate",
]

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


def load_data():
    temporal = defaultdict(lambda: {"total": 0, "positive": 0, "fp": 0, "fn": 0})
    episodes = defaultdict(lambda: {"total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0})
    with PREDICTIONS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task = row["task"]
            if task not in TASK_ORDER:
                continue
            truth = int(row["groundtruth"])
            prediction = int(row["prediction"])
            episode = episodes[(task, row["episode"])]
            episode["total"] += 1
            if prediction and truth:
                episode["tp"] += 1
            elif prediction and not truth:
                episode["fp"] += 1
            elif not prediction and truth:
                episode["fn"] += 1
            else:
                episode["tn"] += 1

            position = float(row["position_fraction"])
            bin_index = min(N_BINS - 1, int(position * N_BINS))
            values = temporal[(task, bin_index)]
            values["total"] += 1
            values["positive"] += truth
            values["fp"] += int(prediction and not truth)
            values["fn"] += int(not prediction and truth)
    return temporal, episodes


def compute_metrics(episodes):
    values = {name: [[] for _ in TASK_ORDER] for name in ["Accuracy", "Precision", "Recall"]}
    rows = []
    for task_index, task in enumerate(TASK_ORDER):
        task_episodes = sorted((key, counts) for key, counts in episodes.items() if key[0] == task)
        if len(task_episodes) != 10:
            raise ValueError(f"Expected 10 episodes for {task!r}, found {len(task_episodes)}")
        for (_, episode_path), counts in task_episodes:
            accuracy = (counts["tp"] + counts["tn"]) / counts["total"]
            precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
            recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
            for name, value in [("Accuracy", accuracy), ("Precision", precision), ("Recall", recall)]:
                values[name][task_index].append(value)
            rows.append({
                "task": task,
                "episode": episode_path,
                "frames": counts["total"],
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            })
    means = {name: np.array([np.mean(task_values) for task_values in values[name]]) for name in values}
    stds = {name: np.array([np.std(task_values, ddof=1) for task_values in values[name]]) for name in values}
    return means, stds, rows


def temporal_matrices(temporal):
    matrices = {name: np.zeros((len(TASK_ORDER), N_BINS)) for name in ["positive", "fp", "fn"]}
    for task_index, task in enumerate(TASK_ORDER):
        for bin_index in range(N_BINS):
            values = temporal[(task, bin_index)]
            for name in matrices:
                matrices[name][task_index, bin_index] = values[name] / values["total"]
    return matrices


def main():
    temporal, episodes = load_data()
    means, stds, metric_rows = compute_metrics(episodes)
    matrices = temporal_matrices(temporal)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TASK_METRICS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    fig = plt.figure(figsize=(7.15, 5.05))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.38, wspace=0.38)
    ax_bar = fig.add_subplot(grid[0, :])
    x = np.arange(len(TASK_ORDER))
    width = 0.23
    metric_specs = [
        ("Accuracy", "#8fb6d8"),
        ("Precision", "#a7cdbd"),
        ("Recall", "#f2bfa4"),
    ]
    for index, (name, color) in enumerate(metric_specs):
        ax_bar.bar(
            x + (index - 1) * width, 100 * means[name], width,
            yerr=100 * stds[name], capsize=2, color=color,
            edgecolor="white", linewidth=0.35, label=name,
        )
    ax_bar.set_ylim(0, 105)
    ax_bar.set_yticks([0, 25, 50, 75, 100])
    ax_bar.set_ylabel("Rate (%)")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(TASK_LABELS, fontsize=6.2)
    ax_bar.grid(axis="y", color="#d9e2ec", linewidth=0.6)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.20), ncol=3,
                  fontsize=6.5, columnspacing=1.1, handlelength=1.3)
    ax_bar.text(-0.04, 1.08, "a", transform=ax_bar.transAxes, fontsize=9, fontweight="bold")
    ax_bar.text(1.0, 1.10, "Mean $\\pm$ s.d. across 10 episodes", transform=ax_bar.transAxes,
                fontsize=6.2, ha="right", va="bottom", color="#475569")
    ax_bar.tick_params(axis="x", length=0, pad=3)

    panels = [
        ("Ground-truth positive frames", matrices["positive"], PASTEL_CMAPS["blue"]),
        ("False-positive frames", matrices["fp"], PASTEL_CMAPS["green"]),
        ("False-negative frames", matrices["fn"], PASTEL_CMAPS["orange"]),
    ]
    for panel_index, (title, matrix, cmap) in enumerate(panels):
        ax = fig.add_subplot(grid[1, panel_index])
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=7.1, pad=7)
        ax.set_xticks([0, 4, 8, 12, 16, 19])
        ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
        ax.set_xlabel("Normalized time (%)")
        ax.set_yticks(range(len(TASK_LABELS)))
        if panel_index == 0:
            ax.set_yticklabels([label.replace("\n", " ") for label in TASK_LABELS], fontsize=5.8)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="both", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
        colorbar.set_ticks([0, 0.5, 1.0])
        colorbar.ax.tick_params(labelsize=5.6, length=2)
        ax.text(-0.13, 1.08, chr(ord("b") + panel_index), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.94)
    for suffix, kwargs in {".pdf": {}, ".svg": {}, ".png": {"dpi": 600}}.items():
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), bbox_inches="tight", **kwargs)


if __name__ == "__main__":
    main()
