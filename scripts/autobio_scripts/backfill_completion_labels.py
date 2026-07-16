#!/usr/bin/env python
"""Backfill frame-level completion labels into existing raw MuJoCo logs.

New demonstrations receive these labels online. This utility is for raw logs
collected before that annotation was added; it replays the recorded MuJoCo
states through the same task check predicates used during collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco

from serialize import STATE_SPEC, load_log


def _task_for_prompt(prompt: str):
    normalized = " ".join(prompt.replace("_", " ").lower().split())
    try:
        if "thermal cycler" in normalized:
            from thermal_cycler_tasks import ThermalCyclerManipulate

            task_cls = ThermalCyclerManipulate
        elif "centrifuge" in normalized:
            from centrifuge5910_tasks import Centrifuge5910Manipulate

            task_cls = Centrifuge5910Manipulate
        else:
            raise ValueError(f"Cannot infer task family from prompt: {prompt!r}")
    except mujoco.FatalError as error:
        raise RuntimeError(
            "Could not load the task's MuJoCo plugin. Run this backfill utility "
            "in the same MuJoCo 3.3.0 environment used for data collection."
        ) from error
    task = task_cls(task_cls.load())
    task.task = "__unrecognized__"
    resolved = task._success_task_from_prompt(prompt)
    if resolved == "__unrecognized__":
        raise ValueError(f"Unsupported task prompt: {prompt!r}")
    task.task = resolved
    return task


def backfill_episode(log_dir: Path, *, overwrite: bool = False) -> int:
    _, states, payload = load_log(log_dir)
    frame_infos = payload.get("info")
    if not isinstance(frame_infos, list) or len(frame_infos) != len(states):
        raise ValueError(
            f"Invalid info/state alignment in {log_dir}: "
            f"{len(frame_infos) if isinstance(frame_infos, list) else None} != {len(states)}"
        )
    if not overwrite and all("task_is_complete" in item for item in frame_infos):
        return 0

    prompt = str(payload["task"]["prefix"])
    task = _task_for_prompt(prompt)
    expected_size = mujoco.mj_stateSize(task.model, STATE_SPEC)
    if states.shape[1] != expected_size:
        raise ValueError(
            f"Recorded model state size {states.shape[1]} does not match the current "
            f"task model size {expected_size} for {log_dir}. Use the original model/code "
            "or re-collect this episode."
        )

    if hasattr(task, "_centrifuge5910_button_touched"):
        task._centrifuge5910_button_touched = False
    if hasattr(task, "_thermal_cycler_button_touched"):
        task._thermal_cycler_button_touched = False
    task._atomic_start_conditions = {}

    labels: list[bool] = []
    for index, state in enumerate(states):
        mujoco.mj_setState(task.model, task.data, state, STATE_SPEC)
        mujoco.mj_forward(task.model, task.data)
        mujoco.mj_rnePostConstraint(task.model, task.data)
        if index == 0:
            task.record_atomic_start(prompt)
        labels.append(bool(task.check(prompt)))

    for item, label in zip(frame_infos, labels, strict=True):
        item["task_is_complete"] = label

    info_path = log_dir / "info.json"
    with info_path.open("w") as file:
        json.dump(payload, file, indent=2)
    return len(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_roots",
        type=Path,
        nargs="+",
        help="Raw log roots containing episode folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute labels already present.",
    )
    args = parser.parse_args()

    episodes = 0
    frames = 0
    for root in args.log_roots:
        for log_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if not (log_dir / "info.json").exists():
                continue
            count = backfill_episode(log_dir, overwrite=args.overwrite)
            if count:
                episodes += 1
                frames += count
                print(f"labeled {count:6d} states: {log_dir}")
    print(f"completed: episodes={episodes} states={frames}")


if __name__ == "__main__":
    main()
