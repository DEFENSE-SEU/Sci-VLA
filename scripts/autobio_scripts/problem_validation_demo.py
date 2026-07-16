from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ProblemValidationDemoConfig:
    task_name: str = "thermal_cycler_long_task_1"
    prompts: tuple[str, str] = (
        "open the lid of the thermal cycler",
        "place pcrPlate into the thermal cycler",
    )
    post_success_seconds: float = 5.0
    dataset_repo_id: str = "mani_thermalcycler"
    dataset_root: Path | None = None
    prefix_fraction: float = 0.30
    restore_steps_per_segment: int = 250
    video_filename_prefix: str = "problem_validation_open_lid_place_pcr_plate"


@dataclass(frozen=True, eq=False)
class SampledRobotState:
    state: np.ndarray
    episode_index: int
    episode_length: int
    prefix_frame_count: int
    frame_index: int
    frame_ratio: float
    task: str
    dataset_root: Path

    def __eq__(self, other):
        return (
            isinstance(other, SampledRobotState)
            and np.array_equal(self.state, other.state)
            and self.episode_index == other.episode_index
            and self.episode_length == other.episode_length
            and self.prefix_frame_count == other.prefix_frame_count
            and self.frame_index == other.frame_index
            and self.task == other.task
            and self.dataset_root == other.dataset_root
        )


@dataclass
class PromptRunController:
    start_time: float
    time_limit: float
    post_success_seconds: float = 0.0
    success_time: float | None = None

    def should_continue(self, current_time: float) -> bool:
        if self.success_time is None:
            return current_time - self.start_time < self.time_limit
        return False

    def observe(self, current_time: float, *, success: bool | None) -> bool:
        if success is True and self.success_time is None:
            self.success_time = float(current_time)
        return self.success_time is not None and not self.should_continue(current_time)

    @property
    def succeeded(self) -> bool:
        return self.success_time is not None


@dataclass(frozen=True)
class ProblemValidationDemoResult:
    success: bool
    first_prompt_healthy: bool
    first_prompt_success: bool
    second_prompt_healthy: bool | None
    second_prompt_success: bool | None


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse dataset metadata file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Dataset metadata file {path} must contain a JSON object")
    return value


def _load_episodes(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode metadata file: {path}")

    episodes = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read episode metadata file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            episode = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
        if not isinstance(episode, dict):
            raise ValueError(f"Episode metadata in {path} at line {line_number} must be an object")
        episodes.append(episode)
    if not episodes:
        raise ValueError(f"Episode metadata file is empty: {path}")
    return episodes


def sample_problem_validation_state(
    config: ProblemValidationDemoConfig,
    rng: np.random.Generator,
) -> SampledRobotState:
    dataset_root = config.dataset_root or (
        Path.home() / ".cache/huggingface/lerobot" / config.dataset_repo_id
    )
    dataset_root = Path(dataset_root)
    if not 0.0 < config.prefix_fraction <= 1.0:
        raise ValueError(f"prefix_fraction must be in (0, 1], got {config.prefix_fraction}")

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = _load_episodes(episodes_path)
    target_task = config.prompts[1]
    matching_episodes = [
        episode
        for episode in episodes
        if isinstance(episode.get("tasks"), list) and target_task in episode["tasks"]
    ]
    if not matching_episodes:
        raise ValueError(f"No episode in {episodes_path} has task {target_task!r}")

    episode = matching_episodes[int(rng.integers(len(matching_episodes)))]
    try:
        episode_index = int(episode["episode_index"])
        episode_length = int(episode["length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid episode metadata for task {target_task!r}: {episode!r}") from exc
    if episode_length <= 0:
        raise ValueError(f"Episode {episode_index} has invalid length {episode_length}")

    info_path = dataset_root / "meta" / "info.json"
    info = _load_json(info_path)
    try:
        chunks_size = int(info["chunks_size"])
        data_path_template = info["data_path"]
        if chunks_size <= 0 or not isinstance(data_path_template, str):
            raise ValueError
        data_path = data_path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid data layout metadata in {info_path}: {info!r}") from exc

    episode_path = dataset_root / data_path
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing Parquet data for episode {episode_index}: {episode_path}")

    import pyarrow.parquet as pq

    try:
        table = pq.read_table(episode_path, columns=["state", "frame_index"])
    except Exception as exc:
        raise ValueError(f"Could not read state and frame_index from {episode_path}: {exc}") from exc
    if table.num_rows == 0:
        raise ValueError(f"Episode {episode_index} has empty state data in {episode_path}")

    prefix_frame_count = math.ceil(config.prefix_fraction * episode_length)
    try:
        states = np.asarray(table.column("state").to_pylist(), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Episode {episode_index} must contain finite 7-dimensional state data"
        ) from exc
    if states.ndim != 2 or states.shape[1] != 7 or not np.isfinite(states).all():
        raise ValueError(f"Episode {episode_index} must contain finite 7-dimensional state data")

    frame_indices = table.column("frame_index").to_pylist()
    expected_frame_indices = set(range(episode_length))
    if (
        len(frame_indices) != len(states)
        or len(frame_indices) != episode_length
        or any(
            not isinstance(frame_index, int) or isinstance(frame_index, bool)
            for frame_index in frame_indices
        )
        or len(set(frame_indices)) != len(frame_indices)
        or set(frame_indices) != expected_frame_indices
    ):
        raise ValueError(
            f"Episode {episode_index} has invalid frame_index data in {episode_path}; "
            f"expected unique integers covering 0..{episode_length - 1}"
        )

    frame_to_row = {frame_index: row_index for row_index, frame_index in enumerate(frame_indices)}
    frame_index = int(rng.integers(prefix_frame_count))
    row_index = frame_to_row[frame_index]

    return SampledRobotState(
        state=states[row_index],
        episode_index=episode_index,
        episode_length=episode_length,
        prefix_frame_count=prefix_frame_count,
        frame_index=frame_index,
        frame_ratio=frame_index / episode_length,
        task=target_task,
        dataset_root=dataset_root,
    )


def execute_problem_validation_sequence(
    *,
    config: ProblemValidationDemoConfig,
    rng: np.random.Generator,
    run_prompt: Callable[[str, float], tuple[bool, bool]],
    sample_state: Callable[
        [ProblemValidationDemoConfig, np.random.Generator], SampledRobotState
    ],
    restore_state: Callable[[SampledRobotState, int], None],
) -> ProblemValidationDemoResult:
    first_healthy, first_success = run_prompt(config.prompts[0], config.post_success_seconds)
    if not first_healthy or not first_success:
        return ProblemValidationDemoResult(
            success=False,
            first_prompt_healthy=first_healthy,
            first_prompt_success=first_success,
            second_prompt_healthy=None,
            second_prompt_success=None,
        )

    sampled_state = sample_state(config, rng)
    restore_state(sampled_state, config.restore_steps_per_segment)
    second_healthy, second_success = run_prompt(config.prompts[1], 0.0)
    return ProblemValidationDemoResult(
        success=second_healthy and second_success,
        first_prompt_healthy=first_healthy,
        first_prompt_success=first_success,
        second_prompt_healthy=second_healthy,
        second_prompt_success=second_success,
    )
