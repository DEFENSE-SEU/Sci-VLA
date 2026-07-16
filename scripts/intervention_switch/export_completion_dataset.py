#!/usr/bin/env python
"""Export labeled LeRobot frames into a trajectory-split completion dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar, got shape={array.shape}")
    return array.reshape(-1)[0].item()


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(array).convert("RGB")


def _sample_value(sample: dict[str, Any], requested: str, fallbacks: tuple[str, ...] = ()):
    candidates = (requested, *fallbacks)
    for key in candidates:
        if key in sample:
            return sample[key]
    raise KeyError(f"None of {candidates!r} are present. Available keys: {sorted(sample)}")


def _resolve_task(sample: dict[str, Any], tasks: dict[int, str]) -> str:
    task = sample.get("task")
    if isinstance(task, str) and task.strip():
        return task
    task_index = int(_scalar(sample["task_index"]))
    if task_index not in tasks:
        raise KeyError(f"Unknown task_index={task_index}; known tasks={sorted(tasks)}")
    return tasks[task_index]


def _assign_split(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    if fraction < train_ratio:
        return "train"
    if fraction < train_ratio + val_ratio:
        return "val"
    return "test"


def export_repo(
    repo_id: str,
    output_dir: Path,
    *,
    image_key: str,
    label_key: str,
    stride: int,
    jpeg_quality: int,
    train_ratio: float,
    val_ratio: float,
    text_config: dict[str, dict[str, list[str]]],
) -> list[dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)
    tasks = {int(key): str(value) for key, value in metadata.tasks.items()}
    source = repo_id.replace("/", "__")
    records: list[dict[str, Any]] = []
    first_image_by_episode: dict[str, str] = {}

    for dataset_index in tqdm(range(len(dataset)), desc=f"export {repo_id}"):
        sample = dataset[dataset_index]
        episode_index = int(_scalar(sample["episode_index"]))
        frame_index = int(_scalar(sample["frame_index"]))
        if frame_index % stride != 0:
            continue

        episode_key = f"{source}:{episode_index}"
        relative_path = Path("images") / source / f"episode_{episode_index:06d}" / f"{frame_index:06d}.jpg"
        destination = output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        image_value = _sample_value(
            sample,
            image_key,
            ("image", "observation/image", "observation.images.image"),
        )
        _to_image(image_value).save(destination, quality=jpeg_quality)
        first_image_by_episode.setdefault(episode_key, relative_path.as_posix())

        label = bool(round(float(_scalar(sample[label_key]))))
        canonical_text = _resolve_task(sample, tasks)
        text_pairs = [
            (canonical_text, label, "observed"),
            *[
                (text, label, "paraphrase")
                for text in text_config.get("paraphrases", {}).get(canonical_text, [])
            ],
        ]
        if label:
            for contradictory_text in text_config.get("contradictions", {}).get(canonical_text, []):
                text_pairs.append((contradictory_text, False, "contradiction"))
                text_pairs.extend(
                    (text, False, "contradiction_paraphrase")
                    for text in text_config.get("paraphrases", {}).get(contradictory_text, [])
                )

        seen_pairs: set[tuple[str, bool]] = set()
        for task_description, pair_label, pair_kind in text_pairs:
            pair_key = (task_description, pair_label)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            records.append({
                "source_repo": repo_id,
                "episode_index": episode_index,
                "frame_index": frame_index,
                "image": relative_path.as_posix(),
                "task_description": task_description,
                "task_is_complete": pair_label,
                "pair_kind": pair_kind,
                "episode_key": episode_key,
                "split": _assign_split(episode_key, train_ratio, val_ratio),
            })

    for record in records:
        record["initial_image"] = first_image_by_episode[record["episode_key"]]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", nargs="+", required=True, help="One or more local/Hugging Face LeRobot repo ids.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--label-key", default="task_is_complete")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--text-config",
        type=Path,
        default=Path(__file__).with_name("task_text_config.json"),
        help="JSON containing paraphrases and physically valid contradictory goals.",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"{args.output_dir} is not empty. Choose a new output directory to avoid mixing datasets."
        )
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be less than 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.text_config.open() as file:
        text_config = json.load(file)
    records: list[dict[str, Any]] = []
    for repo_id in args.repo_id:
        records.extend(
            export_repo(
                repo_id,
                args.output_dir,
                image_key=args.image_key,
                label_key=args.label_key,
                stride=args.stride,
                jpeg_quality=args.jpeg_quality,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                text_config=text_config,
            )
        )

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    split_counts = Counter(record["split"] for record in records)
    label_counts = Counter(record["task_is_complete"] for record in records)
    summary = {
        "repos": args.repo_id,
        "samples": len(records),
        "unique_frames": len({(record["source_repo"], record["episode_index"], record["frame_index"]) for record in records}),
        "episodes": len({record["episode_key"] for record in records}),
        "splits": dict(split_counts),
        "labels": {"false": label_counts[False], "true": label_counts[True]},
        "image_key": args.image_key,
        "label_key": args.label_key,
        "stride": args.stride,
        "text_config": str(args.text_config),
    }
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
