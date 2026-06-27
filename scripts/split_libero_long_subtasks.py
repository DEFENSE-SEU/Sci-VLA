#!/usr/bin/env python3
"""Infer LIBERO long-task subtask boundaries from atomic goal predicates.

This script replays raw LIBERO hdf5 simulator states, evaluates each atomic
BDDL goal predicate at every frame, and writes JSON metadata for contiguous
subtask segments. Adjacent segments share the boundary frame so their end/start
states remain identical when sliced later.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence


def find_first_stable_true(values: Sequence[bool], stable_window: int = 1) -> int | None:
    """Return the first index where values stay true for stable_window frames."""
    if stable_window < 1:
        raise ValueError("stable_window must be >= 1")
    if len(values) < stable_window:
        return None

    run_length = 0
    for index, value in enumerate(values):
        run_length = run_length + 1 if value else 0
        if run_length >= stable_window:
            return index - stable_window + 1
    return None


def goal_state_to_list(goal_state: Sequence[str]) -> list[str]:
    return [str(item) for item in goal_state]


def goal_state_to_prompt(goal_state: Sequence[str]) -> str:
    return " ".join(goal_state_to_list(goal_state))


def build_segments(
    *,
    goal_states: Sequence[Sequence[str]],
    status_by_goal: Sequence[Sequence[bool]],
    total_frames: int,
    stable_window: int = 3,
    min_segment_frames: int = 2,
    include_terminal_segment: bool = True,
) -> list[dict]:
    """Build subtask segments from per-goal predicate truth values.

    Each atomic goal contributes one completion event at its first stable true
    frame. Events that happen on the same frame are grouped into one segment.
    The next segment starts at the previous segment's end frame for continuity.
    """
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if len(goal_states) != len(status_by_goal):
        raise ValueError("goal_states and status_by_goal must have the same length")
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be >= 1")

    events_by_frame: dict[int, list[int]] = {}
    for goal_index, values in enumerate(status_by_goal):
        if len(values) != total_frames:
            raise ValueError(
                f"goal {goal_index} has {len(values)} status values, expected {total_frames}"
            )
        frame = find_first_stable_true(values, stable_window=stable_window)
        if frame is not None:
            events_by_frame.setdefault(frame, []).append(goal_index)

    segments: list[dict] = []
    start_frame = 0
    for end_frame in sorted(events_by_frame):
        goal_indices = sorted(events_by_frame[end_frame])
        if end_frame - start_frame + 1 < min_segment_frames:
            if segments:
                segments[-1]["goal_indices"].extend(goal_indices)
                segments[-1]["goal_indices"] = sorted(set(segments[-1]["goal_indices"]))
                segments[-1]["goal_states"] = [
                    goal_state_to_list(goal_states[i]) for i in segments[-1]["goal_indices"]
                ]
                segments[-1]["prompt"] = "; ".join(
                    goal_state_to_prompt(goal_states[i]) for i in segments[-1]["goal_indices"]
                )
            continue

        segment_goal_states = [goal_state_to_list(goal_states[i]) for i in goal_indices]
        segments.append(
            {
                "subtask_index": len(segments),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "goal_indices": goal_indices,
                "goal_states": segment_goal_states,
                "prompt": "; ".join(goal_state_to_prompt(goal_states[i]) for i in goal_indices),
            }
        )
        start_frame = end_frame

    terminal_end = total_frames - 1
    if (
        include_terminal_segment
        and start_frame < terminal_end
        and terminal_end - start_frame + 1 >= min_segment_frames
    ):
        segments.append(
            {
                "subtask_index": len(segments),
                "start_frame": start_frame,
                "end_frame": terminal_end,
                "goal_indices": [],
                "goal_states": [],
                "prompt": "terminal stabilization",
            }
        )

    return segments


def natural_demo_key(name: str) -> tuple[int, str]:
    match = re.search(r"demo_(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def iter_demo_states(hdf5_path: Path) -> Iterable[tuple[str, object]]:
    import h5py

    with h5py.File(hdf5_path, "r") as h5:
        data = h5["data"]
        for demo_name in sorted(data.keys(), key=natural_demo_key):
            if "states" not in data[demo_name]:
                continue
            yield demo_name, data[demo_name]["states"][()]


def evaluate_goal_statuses(env, states, goal_states: Sequence[Sequence[str]]) -> list[list[bool]]:
    status_by_goal = [[] for _ in goal_states]
    for state in states:
        env.set_init_state(state)
        for goal_index, goal_state in enumerate(goal_states):
            status_by_goal[goal_index].append(bool(env.env._eval_predicate(goal_state)))
    return status_by_goal


def make_env(bddl_file: Path, *, render_gpu_device_id: int = -1):
    # Robosuite's numba cache can fail in editable / conda installs where
    # transform_utils.py has no cache locator.
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    from libero.libero.envs.env_wrapper import ControlEnv

    return ControlEnv(
        bddl_file_name=str(bddl_file),
        use_camera_obs=False,
        has_renderer=False,
        has_offscreen_renderer=False,
        camera_heights=1,
        camera_widths=1,
        render_gpu_device_id=render_gpu_device_id,
    )


def default_output_record(
    *,
    benchmark_name: str,
    task_id: int,
    task,
    bddl_file: Path,
    demo_file: Path,
    goal_states: Sequence[Sequence[str]],
    demos: list[dict],
) -> dict:
    return {
        "benchmark_name": benchmark_name,
        "task_id": task_id,
        "task_name": task.name,
        "language": task.language,
        "bddl_file": str(bddl_file),
        "demo_file": str(demo_file),
        "goal_states": [goal_state_to_list(goal_state) for goal_state in goal_states],
        "demos": demos,
    }


def split_task(
    *,
    benchmark_name: str,
    task_id: int,
    demo_root: Path | None,
    stable_window: int,
    min_segment_frames: int,
    include_terminal_segment: bool,
    max_demos: int | None,
    render_gpu_device_id: int,
) -> dict:
    from libero.libero import benchmark, get_libero_path

    benchmark_instance = benchmark.get_benchmark_dict()[benchmark_name]()
    task = benchmark_instance.get_task(task_id)
    bddl_file = Path(benchmark_instance.get_task_bddl_file_path(task_id))
    datasets_root = Path(get_libero_path("datasets")) if demo_root is None else demo_root
    demo_file = datasets_root / benchmark_instance.get_task_demonstration(task_id)
    if not demo_file.exists():
        raise FileNotFoundError(
            f"Demo hdf5 not found: {demo_file}. Pass --demo-root or download LIBERO demos first."
        )

    env = make_env(bddl_file, render_gpu_device_id=render_gpu_device_id)
    try:
        env.reset()
        goal_states = [tuple(goal_state) for goal_state in env.env.parsed_problem["goal_state"]]
        demo_records: list[dict] = []
        for demo_index, (demo_name, states) in enumerate(iter_demo_states(demo_file)):
            if max_demos is not None and demo_index >= max_demos:
                break
            status_by_goal = evaluate_goal_statuses(env, states, goal_states)
            segments = build_segments(
                goal_states=goal_states,
                status_by_goal=status_by_goal,
                total_frames=len(states),
                stable_window=stable_window,
                min_segment_frames=min_segment_frames,
                include_terminal_segment=include_terminal_segment,
            )
            demo_records.append(
                {
                    "demo_name": demo_name,
                    "num_frames": int(len(states)),
                    "segments": segments,
                    "goal_completion_frames": [
                        find_first_stable_true(values, stable_window=stable_window)
                        for values in status_by_goal
                    ],
                }
            )
    finally:
        env.close()

    return default_output_record(
        benchmark_name=benchmark_name,
        task_id=task_id,
        task=task,
        bddl_file=bddl_file,
        demo_file=demo_file,
        goal_states=goal_states,
        demos=demo_records,
    )


def parse_task_ids(raw_task_ids: Sequence[int] | None, num_tasks: int) -> list[int]:
    if raw_task_ids:
        return list(raw_task_ids)
    return list(range(num_tasks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer subtask boundaries for LIBERO long-horizon demos from BDDL atomic goals."
    )
    parser.add_argument("--benchmark-name", default="libero_10")
    parser.add_argument("--task-id", type=int, action="append", help="Task id to process. Repeatable.")
    parser.add_argument("--demo-root", type=Path, default=None, help="Root containing LIBERO dataset folders.")
    parser.add_argument("--output", type=Path, default=Path("libero_10_subtask_segments.json"))
    parser.add_argument("--stable-window", type=int, default=3)
    parser.add_argument("--min-segment-frames", type=int, default=2)
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument(
        "--no-terminal-segment",
        action="store_true",
        help="Do not add a final stabilization segment after the last newly achieved goal.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from libero.libero import benchmark

    benchmark_instance = benchmark.get_benchmark_dict()[args.benchmark_name]()
    task_ids = parse_task_ids(args.task_id, benchmark_instance.n_tasks)
    records = [
        split_task(
            benchmark_name=args.benchmark_name,
            task_id=task_id,
            demo_root=args.demo_root,
            stable_window=args.stable_window,
            min_segment_frames=args.min_segment_frames,
            include_terminal_segment=not args.no_terminal_segment,
            max_demos=args.max_demos,
            render_gpu_device_id=args.render_gpu_device_id,
        )
        for task_id in task_ids
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {args.output} with {len(records)} task records")


if __name__ == "__main__":
    main()
