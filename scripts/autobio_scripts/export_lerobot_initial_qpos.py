import argparse
import json
import re
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def _scalar_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return int(value[0])
    return int(value)


def _extract_prompt(sample: dict[str, Any], tasks_map: dict[int, str]) -> str:
    for key in ("prompt", "task"):
        if key in sample and sample[key] is not None:
            return str(sample[key])

    if "task_index" in sample and sample["task_index"] is not None:
        task_index = _scalar_to_int(sample["task_index"])
        if task_index is not None and task_index in tasks_map:
            return tasks_map[task_index]

    raise ValueError("Cannot resolve task prompt from sample; expected prompt/task/task_index")


def _extract_state(sample: dict[str, Any]) -> list[float]:
    if "state" not in sample or sample["state"] is None:
        raise ValueError('Sample does not contain "state" field')
    return np.asarray(sample["state"], dtype=np.float64).reshape(-1).tolist()


def _extract_front_image(sample: dict[str, Any], front_image_key: str) -> Any | None:
    for key in (front_image_key, "observation/image", "image"):
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _as_uint8_image(image: Any) -> np.ndarray:
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported front image shape: {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _safe_path_part(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("_")
    return safe[:80] or "task"


def _write_front_image(
    image: Any | None,
    *,
    output_path: Path,
    image_output_dir: Path,
    prompt: str,
    episode_index: int | None,
    sample_index: int,
) -> str | None:
    if image is None:
        return None

    image_dir = image_output_dir / _safe_path_part(prompt)
    image_dir.mkdir(parents=True, exist_ok=True)
    episode_part = episode_index if episode_index is not None else sample_index
    image_path = image_dir / f"episode_{episode_part}_front.png"
    imageio.imwrite(image_path, _as_uint8_image(image))
    try:
        return str(image_path.relative_to(output_path.parent))
    except ValueError:
        return str(image_path)


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return np.asarray(value).reshape(-1).astype(np.int64).tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        # torch.Tensor-like
        return value.detach().cpu().numpy().reshape(-1).astype(np.int64).tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _get_episode_start_indices(dataset: LeRobotDataset) -> list[int] | None:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is None:
        return None

    # Typical LeRobot format: {"from": tensor([...]), "to": tensor([...])}
    if isinstance(episode_data_index, dict):
        for key in ("from", "start", "starts"):
            if key in episode_data_index:
                starts = _to_int_list(episode_data_index[key])
                return sorted(set(starts))

    # Fallback formats: list/tuple/array of starts
    starts = _to_int_list(episode_data_index)
    if starts:
        return sorted(set(starts))
    return None


def export_initial_qpos(
    repo_id: str,
    output_path: Path,
    *,
    image_output_dir: Path | None = None,
    front_image_key: str = "observation/image",
):
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)
    image_output_dir = image_output_dir or (output_path.parent / "lerobot_initial_images")

    tasks_map = {int(k): str(v) for k, v in dataset_meta.tasks.items()}

    task_to_entries: dict[str, list[dict[str, Any]]] = {}
    seen_episode: set[int] = set()

    episode_start_indices = _get_episode_start_indices(dataset)
    if episode_start_indices is None:
        # Fallback for datasets without episode index metadata.
        sample_indices = range(len(dataset))
        print("episode_data_index not found, fallback to frame-wise scan.")
    else:
        sample_indices = episode_start_indices
        print(f"Using episode-wise scan with {len(episode_start_indices)} trajectories.")

    for i in sample_indices:
        sample = dataset[i]

        episode_index = _scalar_to_int(sample.get("episode_index"))
        frame_index = _scalar_to_int(sample.get("frame_index"))

        if frame_index is not None and frame_index != 0:
            continue

        if frame_index is None and episode_index is not None:
            if episode_index in seen_episode:
                continue

        prompt = _extract_prompt(sample, tasks_map)
        qpos = _extract_state(sample)
        front_image_path = _write_front_image(
            _extract_front_image(sample, front_image_key),
            output_path=output_path,
            image_output_dir=image_output_dir,
            prompt=prompt,
            episode_index=episode_index,
            sample_index=int(i),
        )

        task_to_entries.setdefault(prompt, []).append(
            {
                "episode_index": episode_index,
                "initial_qpos": qpos,
                "initial_front_image_path": front_image_path,
            }
        )

        if episode_index is not None:
            seen_episode.add(episode_index)

    tasks = []
    for prompt, entries in sorted(task_to_entries.items(), key=lambda x: x[0]):
        stacked_qpos = [entry["initial_qpos"] for entry in entries]
        stacked_front_image_paths = [entry.get("initial_front_image_path") for entry in entries]
        tasks.append(
            {
                "task": prompt,
                "initial_qpos": stacked_qpos,
                "initial_front_image_paths": stacked_front_image_paths,
            }
        )

    total_trajectories = int(sum(len(t["initial_qpos"]) for t in tasks))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"Exported initial qpos JSON to: {output_path}")
    print(f"Repo ID: {repo_id}")
    print(f"Total tasks: {len(tasks)}")
    print(f"Total trajectories: {total_trajectories}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export initial qpos per task from a LeRobot dataset")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/lerobot_initial_qpos.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=None,
        help="Directory for exported initial front-view PNGs. Defaults to <output parent>/lerobot_initial_images",
    )
    parser.add_argument(
        "--front-image-key",
        type=str,
        default="observation/image",
        help="Dataset sample key containing the front-view image",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_initial_qpos(
        args.repo_id,
        args.output,
        image_output_dir=args.image_output_dir,
        front_image_key=args.front_image_key,
    )
