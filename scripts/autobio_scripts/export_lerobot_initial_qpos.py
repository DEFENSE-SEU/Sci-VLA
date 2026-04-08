import argparse
import json
from pathlib import Path
from typing import Any

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


def export_initial_qpos(repo_id: str, output_path: Path):
    dataset_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)

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

        task_to_entries.setdefault(prompt, []).append(
            {
                "episode_index": episode_index,
                "initial_qpos": qpos,
            }
        )

        if episode_index is not None:
            seen_episode.add(episode_index)

    tasks = []
    for prompt, entries in sorted(task_to_entries.items(), key=lambda x: x[0]):
        stacked_qpos = [entry["initial_qpos"] for entry in entries]
        tasks.append(
            {
                "task": prompt,
                "initial_qpos": stacked_qpos,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_initial_qpos(args.repo_id, args.output)
