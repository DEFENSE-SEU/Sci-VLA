#!/usr/bin/env python
"""Evaluate a completion switch on rendered raw MuJoCo logs.

The script expects episode folders produced by scripts/autobio_scripts/*.py and
rendered by scripts/autobio_scripts/render_all.bash. Each episode must contain:

- info.json with frame-level info[*].task_is_complete labels;
- downsample.json with rendered camera video paths and source state indices.

It reports raw per-frame binary error rate. Debouncing is intentionally not
applied, because the metric is frame-level classification against ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor

from model import load_checkpoint


@dataclass
class Counts:
    total: int = 0
    errors: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, *, truth: bool, prediction: bool) -> None:
        self.total += 1
        if prediction != truth:
            self.errors += 1
        if prediction and truth:
            self.tp += 1
        elif prediction and not truth:
            self.fp += 1
        elif not prediction and truth:
            self.fn += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, float | int]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        return {
            **asdict(self),
            "error_rate": self.errors / max(1, self.total),
            "accuracy": (self.tp + self.tn) / max(1, self.total),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


@dataclass
class PositionBins:
    bin_00: int = 0
    bin_01: int = 0
    bin_02: int = 0
    bin_03: int = 0
    bin_04: int = 0
    bin_05: int = 0
    bin_06: int = 0
    bin_07: int = 0
    bin_08: int = 0
    bin_09: int = 0

    def update(self, fraction: float) -> None:
        index = min(9, max(0, int(fraction * 10.0)))
        field = f"bin_{index:02d}"
        setattr(self, field, getattr(self, field) + 1)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_episode_dirs(paths: list[Path]) -> list[Path]:
    episodes: list[Path] = []
    for path in paths:
        if (path / "info.json").exists() and (path / "downsample.json").exists():
            episodes.append(path)
            continue
        episodes.extend(
            sorted(
                item
                for item in path.rglob("*")
                if item.is_dir()
                and (item / "info.json").exists()
                and (item / "downsample.json").exists()
            )
        )
    return sorted(set(episodes))


def load_episode_metadata(log_dir: Path, camera_key: str) -> tuple[str, list[int], list[bool], Path]:
    with (log_dir / "info.json").open() as file:
        info = json.load(file)
    with (log_dir / "downsample.json").open() as file:
        downsample = json.load(file)

    prompt = str(info["task"]["prefix"])
    frame_infos = info.get("info")
    indices = [int(item) for item in downsample["indices"]]
    camera_mapping = info["task"]["camera_mapping"]
    camera_name = camera_mapping.get(camera_key, camera_key)
    camera_files = downsample["cameras"]
    if camera_name not in camera_files:
        raise KeyError(
            f"{log_dir}: camera {camera_key!r} resolved to {camera_name!r}, "
            f"but available cameras are {sorted(camera_files)}"
        )
    if not isinstance(frame_infos, list):
        raise ValueError(f"{log_dir}: info.json has no frame-level info list")

    labels: list[bool] = []
    for state_index in indices:
        if state_index >= len(frame_infos):
            raise IndexError(
                f"{log_dir}: downsample index {state_index} exceeds "
                f"frame info length {len(frame_infos)}"
            )
        item = frame_infos[state_index]
        if "task_is_complete" not in item:
            raise ValueError(f"{log_dir}: missing task_is_complete at state index {state_index}")
        labels.append(bool(item["task_is_complete"]))

    return prompt, indices, labels, log_dir / camera_files[camera_name]


def preprocess_frames(processor: Any, rgb_frames: list[Any], device: torch.device) -> torch.Tensor:
    images = [Image.fromarray(frame) for frame in rgb_frames]
    return processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(device)


@torch.inference_mode()
def evaluate_episode(
    *,
    model: Any,
    processor: Any,
    device: torch.device,
    log_dir: Path,
    camera_key: str,
    threshold: float,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt, state_indices, labels, video_path = load_episode_metadata(log_dir, camera_key)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, first_bgr = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    text_inputs = processor.tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
    initial_pixels = preprocess_frames(processor, [first_rgb], device)
    initial_feature = model.encode_image(initial_pixels)
    text_feature = model.encode_text(text_inputs["input_ids"], text_inputs["attention_mask"])

    pending_frames: list[Any] = [first_rgb]
    frame_rows: list[dict[str, Any]] = []
    counts = Counts()
    video_frame_index = 0

    def flush() -> None:
        nonlocal pending_frames, video_frame_index
        if not pending_frames:
            return
        current_pixels = preprocess_frames(processor, pending_frames, device)
        current_features = model.encode_image(current_pixels)
        initial_features = initial_feature.expand(current_features.shape[0], -1)
        text_features = text_feature.expand(current_features.shape[0], -1)
        logits = model.classify_features(current_features, initial_features, text_features)
        probabilities = torch.sigmoid(logits).detach().cpu().tolist()

        start = video_frame_index - len(pending_frames) + 1
        for offset, probability in enumerate(probabilities):
            row_index = start + offset
            if row_index >= len(labels):
                raise ValueError(
                    f"{log_dir}: video has more frames than labels "
                    f"({row_index + 1} > {len(labels)})"
                )
            truth = labels[row_index]
            prediction = float(probability) >= threshold
            position_fraction = row_index / max(1, len(labels) - 1)
            position_percent = 100.0 * position_fraction
            position_bin = min(9, max(0, int(position_fraction * 10.0)))
            counts.update(truth=truth, prediction=prediction)
            frame_rows.append(
                {
                    "episode": str(log_dir),
                    "task": prompt,
                    "video_frame_index": row_index,
                    "state_index": state_indices[row_index],
                    "position_fraction": position_fraction,
                    "position_percent": position_percent,
                    "position_bin": position_bin,
                    "groundtruth": int(truth),
                    "probability": float(probability),
                    "prediction": int(prediction),
                    "error": int(prediction != truth),
                }
            )
        pending_frames = []

    while True:
        if len(pending_frames) >= batch_size:
            flush()
        ok, bgr = capture.read()
        if not ok:
            break
        video_frame_index += 1
        pending_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    capture.release()
    flush()

    if len(frame_rows) != len(labels):
        raise ValueError(
            f"{log_dir}: video/label length mismatch: "
            f"video_frames={len(frame_rows)} labels={len(labels)}"
        )

    episode_summary = {
        "episode": str(log_dir),
        "task": prompt,
        "video": str(video_path),
        **counts.metrics(),
    }
    return episode_summary, frame_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_roots",
        nargs="+",
        type=Path,
        help="Episode folders or roots containing rendered raw episode folders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/completion_switch_v1_stride10/best.pt"),
    )
    parser.add_argument("--camera-key", default="image")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results/completion_switch_raw_logs"),
    )
    parser.add_argument(
        "--no-frame-csv",
        action="store_true",
        help="Skip writing per-frame predictions.csv.",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    episode_dirs = discover_episode_dirs(args.log_roots)
    if not episode_dirs:
        raise FileNotFoundError(f"No rendered episodes found under: {args.log_roots}")

    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)
    processor = AutoProcessor.from_pretrained(checkpoint["model_name"])
    threshold = float(checkpoint["threshold"] if args.threshold is None else args.threshold)

    overall = Counts()
    by_task: dict[str, Counts] = defaultdict(Counts)
    overall_error_positions = PositionBins()
    by_task_error_positions: dict[str, PositionBins] = defaultdict(PositionBins)
    episode_summaries: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for log_dir in episode_dirs:
        episode_summary, rows = evaluate_episode(
            model=model,
            processor=processor,
            device=device,
            log_dir=log_dir,
            camera_key=args.camera_key,
            threshold=threshold,
            batch_size=args.batch_size,
        )
        episode_summaries.append(episode_summary)
        frame_rows.extend(rows)
        episode_error_positions = PositionBins()
        for row in rows:
            truth = bool(row["groundtruth"])
            prediction = bool(row["prediction"])
            overall.update(truth=truth, prediction=prediction)
            by_task[str(row["task"])].update(truth=truth, prediction=prediction)
            if bool(row["error"]):
                position_fraction = float(row["position_fraction"])
                overall_error_positions.update(position_fraction)
                by_task_error_positions[str(row["task"])].update(position_fraction)
                episode_error_positions.update(position_fraction)
        episode_summary["error_position_bins"] = asdict(episode_error_positions)

    summary = {
        "checkpoint": str(args.checkpoint),
        "threshold": threshold,
        "camera_key": args.camera_key,
        "episodes": len(episode_summaries),
        "overall": overall.metrics(),
        "overall_error_position_bins": asdict(overall_error_positions),
        "by_task": {task: counts.metrics() for task, counts in sorted(by_task.items())},
        "by_task_error_position_bins": {
            task: asdict(bins)
            for task, bins in sorted(by_task_error_positions.items())
        },
        "by_episode": episode_summaries,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    csv_path = None
    if not args.no_frame_csv:
        csv_path = args.output_dir / "predictions.csv"
        with csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode",
                    "task",
                    "video_frame_index",
                    "state_index",
                    "position_fraction",
                    "position_percent",
                    "position_bin",
                    "groundtruth",
                    "probability",
                    "prediction",
                    "error",
                ],
            )
            writer.writeheader()
            writer.writerows(frame_rows)

    print(json.dumps(summary["overall"], indent=2))
    print(f"summary: {summary_path}")
    if csv_path is not None:
        print(f"predictions: {csv_path}")


if __name__ == "__main__":
    main()
