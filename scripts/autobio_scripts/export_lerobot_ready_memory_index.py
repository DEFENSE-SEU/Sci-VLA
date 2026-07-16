import argparse
import json
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from ready_memory_retrieval_agent import (
    _as_uint8_image,
    _extract_front_image,
    _extract_prompt,
    _extract_state,
    _get_episode_bounds,
    _scalar_to_int,
)


def _safe_path_part(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    return safe[:80] or "task"


def _write_image_relative(
    image: Any,
    *,
    output_path: Path,
    image_output_dir: Path,
    task_prompt: str,
    episode_index: int | None,
    frame_index: int,
) -> str:
    task_dir = image_output_dir / _safe_path_part(task_prompt)
    task_dir.mkdir(parents=True, exist_ok=True)
    episode_part = "unknown" if episode_index is None else str(episode_index)
    image_path = task_dir / f"episode_{episode_part}_frame_{frame_index:06d}.jpg"
    imageio.imwrite(image_path, _as_uint8_image(image))
    try:
        return str(image_path.relative_to(output_path.parent))
    except ValueError:
        return str(image_path)


def _select_local_indices(
    length: int,
    *,
    frame_stride: int,
    max_frames: int,
) -> list[int]:
    if length <= 0:
        return []
    stride = max(1, int(frame_stride))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)

    if max_frames and max_frames > 0 and len(indices) > max_frames:
        selected = np.linspace(0, len(indices) - 1, int(max_frames))
        indices = [indices[int(round(i))] for i in selected]
        indices = sorted(set(indices))
        if indices[0] != 0:
            indices.insert(0, 0)
        if indices[-1] != length - 1:
            indices.append(length - 1)
    return indices


def _parse_task_filter(tasks: str | None) -> set[str] | None:
    if not tasks:
        return None
    return {item.strip() for item in tasks.split(",") if item.strip()}


def export_ready_memory_index(
    repo_id: str,
    output_path: Path,
    *,
    image_output_dir: Path | None = None,
    front_image_key: str = "observation/image",
    samples_per_task: int = 1,
    selection: str = "first",
    seed: int = 0,
    frame_stride: int = 1,
    max_frames: int = 0,
    tasks_filter: set[str] | None = None,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)
    tasks_map = {int(k): str(v) for k, v in dataset_meta.tasks.items()}
    image_output_dir = image_output_dir or (output_path.parent / "ready_memory_frames")

    episode_bounds = _get_episode_bounds(dataset)
    task_to_episodes: dict[str, list[dict[str, Any]]] = {}
    for start, end in episode_bounds:
        sample0 = dataset[start]
        try:
            task_prompt = _extract_prompt(sample0, tasks_map)
        except Exception as exc:
            print(f"[ReadyMemoryExport] Skip episode at dataset index {start}: cannot resolve task ({exc})")
            continue
        if tasks_filter is not None and task_prompt not in tasks_filter:
            continue
        episode_index = _scalar_to_int(sample0.get("episode_index"))
        task_to_episodes.setdefault(task_prompt, []).append(
            {
                "start": int(start),
                "end": int(end),
                "episode_index": episode_index,
                "length": int(end - start),
            }
        )

    rng = np.random.default_rng(seed)
    memories: list[dict[str, Any]] = []
    for task_prompt, episodes in sorted(task_to_episodes.items(), key=lambda item: item[0]):
        if not episodes:
            continue
        if selection == "random":
            order = rng.permutation(len(episodes)).tolist()
            selected_episodes = [episodes[i] for i in order[: max(1, int(samples_per_task))]]
        elif selection == "longest":
            selected_episodes = sorted(episodes, key=lambda item: item["length"], reverse=True)[
                : max(1, int(samples_per_task))
            ]
        else:
            selected_episodes = episodes[: max(1, int(samples_per_task))]

        for sample_id, episode in enumerate(selected_episodes):
            start = int(episode["start"])
            end = int(episode["end"])
            length = int(episode["length"])
            episode_index = episode["episode_index"]
            local_indices = _select_local_indices(
                length,
                frame_stride=frame_stride,
                max_frames=max_frames,
            )
            frames = []
            for local_index in local_indices:
                dataset_index = start + int(local_index)
                sample = dataset[dataset_index]
                frame_index = _scalar_to_int(sample.get("frame_index"))
                if frame_index is None:
                    frame_index = int(local_index)
                image = _extract_front_image(sample, front_image_key)
                if image is None:
                    raise ValueError(
                        f"Episode {episode_index} frame {frame_index} has no image "
                        f"under {front_image_key!r}, observation/image, or image"
                    )
                image_path = _write_image_relative(
                    image,
                    output_path=output_path,
                    image_output_dir=image_output_dir,
                    task_prompt=task_prompt,
                    episode_index=episode_index,
                    frame_index=frame_index,
                )
                frames.append(
                    {
                        "frame_index": int(frame_index),
                        "dataset_index": int(dataset_index),
                        "image_path": image_path,
                        "state": _extract_state(sample),
                    }
                )

            memory_id = f"{_safe_path_part(task_prompt)}__episode_{episode_index}__sample_{sample_id}"
            memories.append(
                {
                    "task": task_prompt,
                    "memory_id": memory_id,
                    "episode_index": episode_index,
                    "episode_start_index": start,
                    "episode_end_index": end,
                    "episode_length": length,
                    "frame_stride": max(1, int(frame_stride)),
                    "max_frames": int(max_frames),
                    "frames": frames,
                }
            )

    payload = {
        "schema": "ready_memory_index/v1",
        "repo_id": repo_id,
        "front_image_key": front_image_key,
        "selection": selection,
        "samples_per_task": int(samples_per_task),
        "frame_stride": max(1, int(frame_stride)),
        "max_frames": int(max_frames),
        "memory_count": len(memories),
        "memories": memories,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Exported ready memory index to: {output_path}")
    print(f"Repo ID: {repo_id}")
    print(f"Tasks: {len(task_to_episodes)}")
    print(f"Memories: {len(memories)}")
    print(f"Frame images: {image_output_dir}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export one or more visual trajectory memories per atomic task.")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ready_memory_index.json"),
        help="Output ready memory index JSON",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=None,
        help="Directory for exported frame images. Defaults to <output parent>/ready_memory_frames",
    )
    parser.add_argument("--front-image-key", type=str, default="observation/image")
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--selection", choices=["first", "random", "longest"], default="first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Uniformly cap frames per trajectory; 0 disables")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated exact task prompts to export")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_ready_memory_index(
        args.repo_id,
        args.output,
        image_output_dir=args.image_output_dir,
        front_image_key=args.front_image_key,
        samples_per_task=args.samples_per_task,
        selection=args.selection,
        seed=args.seed,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        tasks_filter=_parse_task_filter(args.tasks),
    )
